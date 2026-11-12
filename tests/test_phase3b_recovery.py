"""Recovery equivalence: tiny fixtures plus authentic dense 2021/2024 QA scopes."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import ast
import json
import math
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from agniscope import activity_verification as v
import verify_phase3b_activity as original
from agniscope.provenance import sha256


@pytest.mark.parametrize("resolution", [500, 1000, 2000])
def test_independent_index_equivalence_at_edges(resolution):
    coords = [0., -0.00001, float(resolution), math.nextafter(float(resolution), -math.inf),
              math.nextafter(float(resolution), math.inf), 170982.9264, 3481335.9299]
    for x in coords:
        for y in coords: assert v.independent_cell(x, y, resolution) == original.independent_id(x, y, resolution)


def test_temporal_ties_use_utc_then_raw_and_numeric_record():
    values = {v.ID: ["a"*64+":2", "a"*64+":1", "b"*64+":1"], v.X: [5., 5., 5.], v.Y: [5., 5., 5.],
              v.UTC: [v.DAY_NS, v.DAY_NS, 2*v.DAY_NS], "agniscope_raw_sha256": ["a"*64, "a"*64, "b"*64], "agniscope_source_record": [2, 1, 1]}
    assert v.sampling_indices(values) == {1, 2}
    values[v.UTC][0] -= 1
    assert v.sampling_indices(values) == {0, 1, 2}  # first instant moved to the previous UTC date


def test_arrow_primary_utc_confidence_and_tree_boundaries(tmp_path):
    days = ["2019-02-14T23:59:00+00:00", "2019-02-15T00:00:00+00:00", "2019-06-15T23:59:00+00:00", "2019-06-16T00:00:00+00:00"]
    ids = ["a"*64+":"+str(i+1) for i in range(4)]
    table = pa.table({v.ID: ids, v.UTC: pa.array([datetime.fromisoformat(t) for t in days], type=pa.timestamp("ns", tz="UTC")),
                      v.X: [10.]*4, v.Y: [10.]*4, "agniscope_confidence": ["nominal", "low", "nominal", "nominal"],
                      "agniscope_worldcover_2021_class": pa.array([10, None, 10, 10], type=pa.int64())})
    path = tmp_path/"tiny.parquet"; pq.write_table(table, path)
    all_values, profile = v.load_population(path, {"year": 2019, "subset": "all"})
    assert all_values[v.ID] == ids[1:3] and profile["full_grid_geometries_read"] == 0
    for name in ("nominal", "tree"):
        actual, _ = v.load_population(path, {"year": 2019, "subset": name})
        assert actual[v.ID] == [ids[2]]


def test_compensated_strength_and_negative_membership_semantics():
    ids = ["a"*64+":"+str(i+1) for i in range(5)]
    data = {v.ID: ids, v.X: [0.]*5, v.Y: [0.]*5, v.UTC: [v.DAY_NS]*5}
    labels = dict(zip(ids, [0, -1, -2, -3, 0]))
    strengths = dict(zip(ids, [.1, 0., None, None, .2]))
    expected, n = v.expectation(data, 1000, labels, strengths)
    row = next(iter(expected.values()))
    assert n == row["detections"] == row["strength_available"] == 2
    assert row["distinct_UTC_dates"] == 1 and row["strength_sum"] == math.fsum([.1, .2])
    assert row["strength_min"] == .1 and row["strength_max"] == .2


def original_aggregation(source_rows, method, strengths, resolution):
    # Execute the exact sealed original accumulator statements, not an imitation
    # of the replacement. The original CLI's seven-year driver is never run.
    path = ROOT/"scripts/verify_phase3b_activity.py"
    main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "resolution")
    prefix = []
    for statement in loop.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "actual" for t in statement.targets): break
        prefix.append(statement)
    env = {"defaultdict": defaultdict, "source_rows": source_rows, "method": method,
           "strengths": strengths, "resolution": resolution, "independent_id": original.independent_id}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), env)
    return {cell: {"detections": count, "distinct_UTC_dates": len(env["dates"][cell]),
                   "strength_available": len(env["values"][cell]), "strength_sum": math.fsum(env["values"][cell]),
                   "strength_min": min(env["values"][cell]) if env["values"][cell] else None,
                   "strength_max": max(env["values"][cell]) if env["values"][cell] else None}
            for cell, count in env["counts"].items()}


@pytest.mark.parametrize("year,subset,method,resolution", [
    (2021, "all", "hdbscan", 500), (2024, "all", "detection", 1000),
    (2024, "grid_day", "dbscan", 500), (2024, "nominal", "hdbscan", 2000), (2024, "tree", "hdbscan", 500)])
def test_authentic_dense_scope_equivalence(year, subset, method, resolution):
    import pandas as pd
    master = ROOT/"data/processed/firms_2019_2025/validated_observations_2019_2025.parquet"
    columns = [v.ID, v.UTC, v.X, v.Y, "agniscope_raw_sha256", "agniscope_source_record", "agniscope_confidence", "agniscope_worldcover_2021_class"]
    legacy = pd.concat([t.select(columns).to_pandas() for t in original.iter_season(master, "core", year)], ignore_index=True)
    legacy["gx"] = [math.floor(x/1000) for x in legacy[v.X]]
    legacy["gy"] = [math.floor(y/1000) for y in legacy[v.Y]]
    legacy["day"] = legacy[v.UTC].dt.strftime("%Y-%m-%d")
    if subset == "grid_day": legacy = legacy.sort_values([v.UTC, "agniscope_raw_sha256", "agniscope_source_record"], kind="stable").drop_duplicates(["gx", "gy", "day"])
    if subset == "nominal": legacy = legacy.loc[legacy.agniscope_confidence.eq("nominal")]
    if subset == "tree": legacy = legacy.loc[legacy.agniscope_worldcover_2021_class.eq(10)]
    new, profile = v.load_population(master, {"year": year, "subset": subset})
    assert set(new[v.ID]) == set(legacy[v.ID]) and len(new[v.ID]) == len(legacy)
    labels = strengths = None
    old_strengths = {}
    if method != "detection":
        references = json.loads((ROOT/"outputs/tables/phase3b/resume_20261023/reference_index.json").read_text())
        spec = next(s for s in references if s["season"] == f"CORE_{year}" and s["subset"] == subset and s["algorithm"] == method)
        assert sha256(ROOT/spec["assignment_path"]) == spec["assignment_sha256"]
        assignment = pd.read_parquet(ROOT/spec["assignment_path"], columns=[v.ID, "cluster_label", "membership_strength"])
        labels = dict(zip(assignment[v.ID], assignment.cluster_label))
        legacy = legacy.loc[legacy[v.ID].isin(assignment.loc[assignment.cluster_label.ge(0), v.ID])]
        if method == "hdbscan": strengths = old_strengths = dict(zip(assignment[v.ID], assignment.membership_strength))
    source_rows = list(zip(legacy[v.ID], legacy[v.X], legacy[v.Y], legacy.day))
    reference = original_aggregation(source_rows, method, old_strengths, resolution)
    actual, supporting = v.expectation(new, resolution, labels, strengths)
    assert set(actual) == set(reference) and supporting == len(legacy)
    for cell, expected in reference.items():
        for key in ("detections", "distinct_UTC_dates", "strength_available", "strength_min", "strength_max"):
            assert actual[cell][key] == expected[key]
        assert math.isclose(actual[cell]["strength_sum"], expected["strength_sum"], rel_tol=1e-12, abs_tol=1e-12)
    assert len(profile["requested_columns"]) <= 6 and profile["full_grid_geometries_read"] == 0
