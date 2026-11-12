"""Independent master-ID to cell/date/count reconstruction, without recurrence helpers."""
from pathlib import Path
from collections import defaultdict
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from agniscope.seasonal_inputs import iter_season
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json
from phase3b_grid import TABLE, SPATIAL, verify_receipt

ID = "agniscope_observation_id"
UTC = "agniscope_acquired_utc"
MASTER = ROOT/"data/processed/firms_2019_2025/validated_observations_2019_2025.parquet"


def independent_id(x, y, resolution):
    ix, iy = math.floor(float(x)/resolution), math.floor(float(y)/resolution)
    code = {500: 1, 1000: 2, 2000: 3}[resolution]
    return code*4294967296+(iy+32768)*65536+ix+32768


def main():
    assert not (TABLE/"gate_c.json").exists() and not (TABLE/"gate_d.json").exists()
    a = json.loads((TABLE/"gate_a.json").read_text())
    assert sha256(MASTER) == a["master_sha256"]
    references = json.loads((TABLE/"reference_index.json").read_text())
    checks = []
    hashes = {}
    columns = [ID, UTC, "agniscope_x_m", "agniscope_y_m", "agniscope_raw_sha256", "agniscope_source_record",
               "agniscope_confidence", "agniscope_worldcover_2021_class"]
    for year in range(2019, 2026):
        original = pd.concat([t.select(columns).to_pandas() for t in iter_season(MASTER, "core", year)], ignore_index=True)
        assert original[ID].is_unique
        original["gx"] = [math.floor(float(x)/1000) for x in original.agniscope_x_m]
        original["gy"] = [math.floor(float(y)/1000) for y in original.agniscope_y_m]
        original["day"] = original[UTC].dt.strftime("%Y-%m-%d")
        first = original.sort_values([UTC, "agniscope_raw_sha256", "agniscope_source_record"], kind="stable").drop_duplicates(["gx", "gy", "day"])
        subsets = {"all": original, "grid_day": first,
                   "nominal": original.loc[original.agniscope_confidence.eq("nominal")],
                   "tree": original.loc[original.agniscope_worldcover_2021_class.eq(10)]}
        directory = TABLE/"activity"/f"CORE_{year}"
        verify_receipt(directory)
        hashes[str(year)] = sha256(directory/"receipt.json")
        observed = pd.read_parquet(directory/"seasonal_activity.parquet")
        assert len(observed) == len(observed.drop_duplicates(["resolution_m", "subset", "method", "cell_id"]))
        for population, points in subsets.items():
            for method in ("detection", "dbscan", "hdbscan"):
                selected = points
                strengths = {}
                if method != "detection":
                    spec = next(v for v in references if v["season"] == f"CORE_{year}" and v["subset"] == population and v["algorithm"] == method)
                    assignment = ROOT/spec["assignment_path"]
                    assert sha256(assignment) == spec["assignment_sha256"]
                    data = pd.read_parquet(assignment, columns=[ID, "cluster_label", "membership_strength"])
                    assert data[ID].is_unique and set(data[ID]) == set(points[ID])
                    member_ids = set(data.loc[data.cluster_label.ge(0), ID])
                    selected = points.loc[points[ID].isin(member_ids)]
                    if method == "hdbscan": strengths = dict(zip(data[ID], data.membership_strength))
                source_rows = list(zip(selected[ID], selected.agniscope_x_m, selected.agniscope_y_m, selected.day))
                for resolution in (500, 1000, 2000):
                    counts = defaultdict(int)
                    dates = defaultdict(set)
                    values = defaultdict(list)
                    for identity, x, y, day in source_rows:
                        cell = independent_id(x, y, resolution)
                        counts[cell] += 1
                        dates[cell].add(day)
                        if method == "hdbscan": values[cell].append(float(strengths[identity]))
                    actual = observed.loc[observed.subset.eq(population) & observed.method.eq(method) & observed.resolution_m.eq(resolution)]
                    assert set(actual.cell_id) == set(counts)
                    for row in actual.itertuples():
                        assert row.year == year and row.activity == 1
                        assert row.detections == counts[row.cell_id] and row.distinct_UTC_dates == len(dates[row.cell_id])
                        if method == "hdbscan":
                            vector = values[row.cell_id]
                            assert row.strength_available == len(vector)
                            assert math.isclose(row.strength_sum, math.fsum(vector), rel_tol=1e-12, abs_tol=1e-12)
                            assert row.strength_min == min(vector) and row.strength_max == max(vector)
                        else:
                            assert row.strength_available == 0 and pd.isna(row.strength_min) and pd.isna(row.strength_max)
                    assert int(actual.detections.sum()) == len(selected)
                    checks.append({"year": year, "subset": population, "method": method, "resolution_m": resolution,
                                   "source_observations": len(points), "supporting_observations": len(selected),
                                   "active_cells": len(actual), "status": "passed"})
        print(json.dumps({"independent_master_source_year": year, "primary_rows": len(original), "views_verified": 36}), flush=True)
    assert len(checks) == 252
    for gate, methods in (("C", {"detection"}), ("D", {"dbscan", "hdbscan"})):
        result = {"status": "passed", "gate": gate, "verified_utc": utc_now(), "master_sha256": sha256(MASTER),
                  "independent_direct_master_source_reconstruction": True, "independent_math_floor_assignment": True,
                  "primary_seasons": 7, "resolutions_m": [500, 1000, 2000], "checks": [v for v in checks if v["method"] in methods],
                  "activity_receipts": hashes, "reference_index_sha256": sha256(TABLE/"reference_index.json"),
                  "clustering_refits": 0, "credential_reads": 0, "observation_HTTP_calls": 0}
        atomic_json(TABLE/("gate_"+gate.lower()+".json"), result)
    print(json.dumps({"status": "passed", "gates": ["C", "D"], "independent_views_verified": 252}))


if __name__ == "__main__": main()
