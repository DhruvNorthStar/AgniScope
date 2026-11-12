"""Prompt 17 adapter: reuse the unchanged serial guard in a reviewed output area."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256
from agniscope.acquisition_store import StoreConflict
import phase3b_guard as guard
from preflight_phase3b_recovery import REC, TABLE


def run_job(job, arguments, execute=False):
    if execute:
        preflight = json.loads((REC/"preflight.json").read_text())
        assert preflight["status"] == "passed"
        assert sha256(TABLE/"STOP.json") == preflight["old_stop_sha256"]
        assert sha256(ROOT/preflight["startup_record"]) == preflight["startup_sha256"]
    # Only the destination for the new, explicitly authorized recovery changes.
    # Guard code, RAM/disk/cap thresholds and old stop/evidence remain unchanged.
    guard.OUT = REC
    return guard.run(job, arguments, execute=execute)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = run_job(args.job, args.arguments, args.execute)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("passed", "dry_run") else 2


if __name__ == "__main__": raise SystemExit(main())
