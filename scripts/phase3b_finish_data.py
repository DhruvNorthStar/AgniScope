"""Sequential continuation only after 252 source-bound activity checks pass."""
import argparse
import json
from preflight_phase3b_recovery import REC
from phase3b_recovery import run_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    proof = json.loads((REC/"activity_verification_complete.json").read_text())
    assert proof["status"] == "passed" and proof["scopes"] == 252
    for stage in ("recurrence", "components"):
        for resolution in (500, 1000, 2000):
            job = f"{stage}_{resolution}"
            result = run_job(job, [f"scripts/phase3b_{stage}_recovery.py", str(resolution)], args.execute)
            print(json.dumps({k: result[k] for k in ("job", "status", "peak_worker_bytes", "minimum_available_bytes", "elapsed_seconds") if k in result}), flush=True)
            if result["status"] not in ("passed", "dry_run"): raise SystemExit(2)
