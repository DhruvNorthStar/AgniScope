"""Prompt 17 reviewed recovery: preserve the old stop and reconstruct scope lineage."""
from pathlib import Path
import hashlib
import json
import sys
import tomllib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json
from agniscope.resources import resource_snapshot
from phase3b_grid import TABLE, SPATIAL, verify_receipt

REC = ROOT/"outputs/tables/phase3b_recovery"
OLD_SEAL = TABLE/"partial_20261023/file_integrity_manifest.json"
OLD_SHA = "627d966a3dc9d0baa57c98bc372c81ea6f6cd0e5848ebcaa08060c54dca118fc"
MASTER = ROOT/"data/processed/firms_2019_2025/validated_observations_2019_2025.parquet"


def main():
    assert not (REC/"preflight.json").exists(), "Preserve existing recovery preflight."
    assert sha256(OLD_SEAL) == OLD_SHA
    prior = json.loads(OLD_SEAL.read_text())
    assert len(prior["files"]) == 124
    for item in prior["files"]: assert sha256(ROOT/item["path"]) == item["sha256"]
    a = json.loads((TABLE/"gate_a.json").read_text())
    assert sha256(MASTER) == a["master_sha256"]
    assert len(a["completed_fits_independently_verified"]) == 209 and a["primary_reference_count"] == 56
    attempts = sorted((ROOT/"outputs/tables/phase3b/preflight_attempts").glob("*/preflight.json"))
    startup = json.loads(attempts[-1].read_text())
    assert startup["status"] == "READY_FOR_REMAINING_PREFLIGHT"
    assert startup["startup_resource_gate"]["minimum_available_bytes"] >= 896*1024**2
    assert startup["hash_preflight"]["scientific_receipts_verified"] == 201
    stop = TABLE/"STOP.json"
    interrupted = json.loads(stop.read_text())
    log = ROOT/interrupted["log_path"]
    assert sha256(log) == interrupted["log_sha256"]
    completed = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"independent_master_source_year"')]
    assert [v["independent_master_source_year"] for v in completed] == [2019, 2020, 2021]
    assert all(v["views_verified"] == 36 for v in completed)
    assert not (TABLE/"gate_c.json").exists() and not (TABLE/"gate_d.json").exists()
    methods_path = ROOT/"configs/phase3b_methods.toml"
    methods = tomllib.loads(methods_path.read_text())["design"]
    assert methods["origin_m"] == [0, 0] and methods["resolutions_m"] == [500, 1000, 2000]
    assert methods["thresholds"] == [2, 3, 4, 5] and methods["eligibility"] == "strictly_positive_2021_district_union_intersection_area"
    references = json.loads((TABLE/"reference_index.json").read_text())
    plan = []
    packages = []
    for year in range(2019, 2026):
        directory = TABLE/"activity"/f"CORE_{year}"
        verify_receipt(directory)
        summary = json.loads((directory/"summary.json").read_text())
        for path, digest in summary["source_hashes"].items(): assert sha256(ROOT/path) == digest
        packages.append({"year": year, "receipt_sha256": sha256(directory/"receipt.json"), "path": directory.relative_to(ROOT).as_posix()})
        for subset in ("all", "grid_day", "nominal", "tree"):
            refs = [r for r in references if r["season"] == f"CORE_{year}" and r["subset"] == subset]
            assert len(refs) == 2 and refs[0]["input_sha256"] == refs[1]["input_sha256"]
            for method in ("detection", "dbscan", "hdbscan"):
                ref = next((r for r in refs if r["algorithm"] == method), None)
                for resolution in (500, 1000, 2000):
                    grid = SPATIAL/f"grid_{resolution}"
                    verify_receipt(grid)
                    body = {"year": year, "subset": subset, "method": method, "resolution_m": resolution,
                        "master_sha256": sha256(MASTER), "activity_path": (directory/"seasonal_activity.parquet").relative_to(ROOT).as_posix(),
                        "activity_sha256": sha256(directory/"seasonal_activity.parquet"), "activity_receipt_sha256": sha256(directory/"receipt.json"),
                        "population_path": refs[0]["input_path"], "population_sha256": refs[0]["input_sha256"],
                        "assignment_path": ref["assignment_path"] if ref else None,
                        "assignment_sha256": ref["assignment_sha256"] if ref else None,
                        "grid_receipt_sha256": sha256(grid/"receipt.json"), "methods_sha256": sha256(methods_path)}
                    identity = "scope_"+hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    plan.append({"scope_id": identity, **body, "prior_verified": year <= 2021})
    assert len(plan) == len({p["scope_id"] for p in plan}) == 252
    assert len({(p["year"], p["subset"], p["method"], p["resolution_m"]) for p in plan}) == 252
    assert sum(p["prior_verified"] for p in plan) == 108 and sum(not p["prior_verified"] for p in plan) == 144
    assert len({p["scope_id"][:26] for p in plan}) == 252
    REC.mkdir(parents=True, exist_ok=True)
    for p in plan:
        if p["prior_verified"]:
            atomic_json(REC/"scopes"/(p["scope_id"][:26]+".json"), {
                "status": "passed", "specification": p, "reused_historical_verification": True,
                "proof_kind": "sealed_verifier_year_success_after_all_36_scopes",
                "original_verifier_sha256": sha256(ROOT/"scripts/verify_phase3b_activity.py"),
                "original_verifier_log_sha256": sha256(log), "old_phase3b_inventory_sha256": OLD_SHA,
                "new_scientific_reconstruction": False, "adopted_utc": utc_now()})
    atomic_json(REC/"scope_plan.json", plan)
    orphaned = [p.relative_to(ROOT).as_posix() for base in (TABLE, SPATIAL) for p in base.rglob("*.staged")]
    assert not orphaned, "Preserve/review uncommitted staging first."
    record = {"status": "passed", "verified_utc": utc_now(), "authorization": "User Prompt 17 reviewed low-RAM recovery; old stop remains immutable.",
        "old_phase3b_seal_sha256": OLD_SHA, "protected_phase3b_files": prior["files"], "old_stop_sha256": sha256(stop),
        "master_sha256": sha256(MASTER), "scope_plan_sha256": sha256(REC/"scope_plan.json"),
        "startup_record": attempts[-1].relative_to(ROOT).as_posix(), "startup_sha256": sha256(attempts[-1]),
        "scope_count": 252, "reused_scope_count": 108, "pending_scope_count": 144,
        "historical_verified_years": [2019, 2020, 2021], "prior_per_scope_receipts_existed": False,
        "receipt_adoption_policy": "Exact 36-scope/year successful sealed log plus original independent source-code and source-file hashes; no new verification claimed.",
        "uncommitted_staging": orphaned, "activity_packages": packages,
        "resources": resource_snapshot(), "observation_HTTP_calls": 0, "credential_reads": 0, "clustering_refits": 0}
    atomic_json(REC/"preflight.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("protected_phase3b_files", "activity_packages")}, indent=2))


if __name__ == "__main__": main()
