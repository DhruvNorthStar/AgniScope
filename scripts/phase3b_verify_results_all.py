"""Independent geographic gates run serially, before sensitivity interpretation."""
import argparse
import json
from phase3b_recovery import run_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    for resolution in (500, 1000, 2000):
        result = run_job(f"independent_{resolution}", ["scripts/verify_phase3b_results.py", str(resolution)], args.execute)
        print(json.dumps({k: result[k] for k in ("job", "status", "peak_worker_bytes", "minimum_available_bytes", "elapsed_seconds") if k in result}), flush=True)
        if result["status"] not in ("passed", "dry_run"): raise SystemExit(2)
