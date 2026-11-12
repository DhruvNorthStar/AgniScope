"""Resume precisely the 144 pending scope identities, one guarded worker at a time."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256, utc_now
from agniscope.acquisition_store import atomic_json
from preflight_phase3b_recovery import REC, TABLE
from phase3b_recovery import run_job
from certify_phase3b_recovery_tests import check


def certify_all(plan):
    receipts = []
    for spec in plan:
        path = REC/"scopes"/(spec["scope_id"][:26]+".json")
        record = json.loads(path.read_text())
        assert record["status"] == "passed" and record["specification"] == spec
        assert record["reused_historical_verification"] == spec["prior_verified"]
        receipts.append({"scope_id": spec["scope_id"], "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                         "historical_reused": spec["prior_verified"]})
    assert len(receipts) == 252
    for gate, methods in (("C", {"detection"}), ("D", {"dbscan", "hdbscan"})):
        target = TABLE/("gate_"+gate.lower()+".json")
        body = {"status": "passed", "gate": gate, "verified_utc": utc_now(),
            "activity_verification_receipts": [r for r, s in zip(receipts, plan) if s["method"] in methods],
            "total_scopes_verified": 252, "historical_scopes_reused": 108, "new_scopes_verified": 144,
            "independent_direct_master_source_reconstruction": True, "one_scope_per_worker": True,
            "test_gate_sha256": sha256(REC/"verification_test_gate.json"), "scope_plan_sha256": sha256(REC/"scope_plan.json"),
            "clustering_refits": 0, "credential_reads": 0, "observation_HTTP_calls": 0}
        if target.exists(): assert json.loads(target.read_text())["scope_plan_sha256"] == body["scope_plan_sha256"]
        else: atomic_json(target, body)
    atomic_json(REC/"activity_verification_complete.json", {"status": "passed", "verified_utc": utc_now(),
        "scopes": 252, "reused": 108, "newly_verified": 144, "receipts": receipts,
        "gate_C_sha256": sha256(TABLE/"gate_c.json"), "gate_D_sha256": sha256(TABLE/"gate_d.json")})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    check()
    before = json.loads((REC/"preflight.json").read_text())
    assert sha256(REC/"scope_plan.json") == before["scope_plan_sha256"]
    plan = json.loads((REC/"scope_plan.json").read_text())
    remaining = [p for p in plan if not p["prior_verified"]]
    assert len(plan) == 252 and len(remaining) == 144
    for i, spec in enumerate(remaining, 1):
        receipt = REC/"scopes"/(spec["scope_id"][:26]+".json")
        if receipt.exists():
            record = json.loads(receipt.read_text()); assert record["status"] == "passed" and record["specification"] == spec
            continue
        job = "verify_"+spec["scope_id"][6:26]
        result = run_job(job, ["scripts/phase3b_scope_verify.py", spec["scope_id"]], args.execute)
        print(json.dumps({"pending_progress": i, "pending_total": 144, "year": spec["year"], "subset": spec["subset"],
                          "method": spec["method"], "resolution_m": spec["resolution_m"], "status": result["status"],
                          "peak_worker_bytes": result.get("peak_worker_bytes"), "minimum_available_bytes": result.get("minimum_available_bytes")}), flush=True)
        if result["status"] not in ("passed", "dry_run"): return 2
    if args.execute: certify_all(plan)
    return 0


if __name__ == "__main__": raise SystemExit(main())
