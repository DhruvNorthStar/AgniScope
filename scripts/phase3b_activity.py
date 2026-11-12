"""One primary season at a time: sparse authentic point-supported grid activity."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256
from agniscope.acquisition_store import atomic_json, StoreConflict
from phase3b_grid import TABLE, SPATIAL, verify_receipt, seal


def build(year):
    target = TABLE/"activity"/f"CORE_{year}"
    if target.exists():
        return {"reused": True, "receipt": verify_receipt(target)}
    from phase3b_certificates import require_certificate
    require_certificate(TABLE/"pytest_grid_certified.xml", TABLE/"pytest_grid_certified.certificate.json")
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from agniscope import recurrence as r
    assert year in r.YEARS
    assert json.loads((TABLE/"gate_a.json").read_text())["status"] == "passed"
    eligible = {}
    grid_receipts = {}
    for resolution in r.RESOLUTIONS:
        directory = SPATIAL/f"grid_{resolution}"
        verify_receipt(directory)
        grid_receipts[str(resolution)] = sha256(directory/"receipt.json")
        eligible[resolution] = pq.read_table(directory/"grid.parquet", columns=["cell_id"])["cell_id"].to_numpy()
    references = json.loads((TABLE/"reference_index.json").read_text())
    assert len(references) == 56
    staging = target.with_name(target.name+".staged")
    if staging.exists(): raise StoreConflict("Preserve uncommitted activity staging.")
    staging.mkdir(parents=True)
    activities = []
    source_rows = []
    support_sets = {}
    source_hashes = {}
    for population in r.POPULATIONS:
        specs = [v for v in references if v["season"] == f"CORE_{year}" and v["subset"] == population]
        assert len(specs) == 2 and len({v["input_sha256"] for v in specs}) == 1
        path = ROOT/specs[0]["input_path"]
        assert sha256(path) == specs[0]["input_sha256"]
        source_hashes[specs[0]["input_path"]] = sha256(path)
        points = pd.read_parquet(path)
        r.validate_source_points(points, year)
        assert len(points) == specs[0]["n"]
        populations = {"detection": points}
        for spec in specs:
            algorithm = spec["algorithm"]
            expected = ({"algorithm": "dbscan", "eps_km": 1., "min_samples": 3} if algorithm == "dbscan" else
                        {"algorithm": "hdbscan", "min_cluster_size": 5, "min_samples": 5})
            assert spec["parameters"] == expected
            assignment = ROOT/spec["assignment_path"]
            assert sha256(assignment) == spec["assignment_sha256"]
            source_hashes[spec["assignment_path"]] = sha256(assignment)
            assigned = pd.read_parquet(assignment, columns=[r.ID, *r.XY, r.UTC, "cluster_label", "membership_strength"])
            populations[algorithm] = r.join_memberships(points, assigned, algorithm)
        for method in r.METHODS:
            selected = populations[method]
            source_rows.append({"year": year, "subset": population, "method": method, "source_observations": len(points),
                                "supporting_observations": len(selected), "nonmember_observations": len(points)-len(selected),
                                "assignment_join_exact": True, "subset_refit": population != "all" and method != "detection"})
            for resolution in r.RESOLUTIONS:
                activity = r.seasonal_activity(selected, resolution, year)
                assert np.isin(activity.cell_id.to_numpy(), eligible[resolution]).all(), "Retained point owns an ineligible cell; never snap or discard."
                activity["resolution_m"] = resolution
                activity["subset"] = population
                activity["method"] = method
                activities.append(activity)
                support_sets[population, method, resolution] = set(activity.cell_id)
                if method != "detection":
                    assert support_sets[population, method, resolution] <= support_sets[population, "detection", resolution]
        del populations
    for resolution in (1000, 2000):
        assert support_sets["all", "detection", resolution] == support_sets["grid_day", "detection", resolution]
    combined = pd.concat(activities, ignore_index=True).sort_values(["resolution_m", "subset", "method", "cell_id"], kind="stable").reset_index(drop=True)
    assert not combined.duplicated(["resolution_m", "subset", "method", "year", "cell_id"]).any()
    table = pa.Table.from_pandas(combined, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, staging/"seasonal_activity.parquet", compression="zstd", row_group_size=4096)
    pd.DataFrame(source_rows).to_csv(staging/"source_membership_accounting.csv", index=False, lineterminator="\n")
    atomic_json(staging/"summary.json", {"year": year, "primary_season": f"{year}-02-15/{year}-06-15 UTC inclusive",
        "sparse_activity_rows": len(combined), "scopes": 36, "source_membership_accounting": source_rows,
        "ineligible_owned_cells": 0, "cluster_support_subset_invariants": True,
        "temporal_sample_preserves_1km_and_2km_detection_activity": True,
        "activity_zeros": "Implicit only after complete season receipt and eligible-grid left join; observation opportunity unverified.",
        "source_hashes": source_hashes, "grid_receipt_hashes": grid_receipts,
        "HDBSCAN_strength": "Preserved count/sum/min/max algorithmic diagnostics, not fire probabilities."})
    seal(staging, target, {"kind": "seasonal_cell_activity", "year": year,
        "kernel_sha256": sha256(ROOT/"src/agniscope/recurrence.py"), "worker_sha256": sha256(Path(__file__)),
        "reference_index_sha256": sha256(TABLE/"reference_index.json"),
        "methods_sha256": sha256(ROOT/"configs/phase3b_methods.toml"), "source_hashes": source_hashes})
    return {"year": year, "status": "passed", "activity_rows": len(combined), "source_membership_accounting": source_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=range(2019, 2026))
    args = parser.parse_args()
    print(json.dumps(build(args.year), indent=2))
