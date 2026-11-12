"""Explicit, serial offline Phase 3B stage execution. Default is dry run."""
import argparse
import json
from phase3b_guard import run


def jobs(stage):
    if stage == "grids":
        return [("grid_1000", ["scripts/phase3b_grid.py", "1000"]),
                ("grid_2000", ["scripts/phase3b_grid.py", "2000"])]
    if stage == "activity":
        return [("gate_b", ["scripts/certify_phase3b_grids.py"])] + [(f"activity_{year}", ["scripts/phase3b_activity.py", str(year)]) for year in range(2019, 2026)]
    if stage == "recurrence":
        return [("verify_activity", ["scripts/verify_phase3b_activity.py"])] + [(f"recurrence_{size}", ["scripts/phase3b_recurrence.py", str(size)]) for size in (500, 1000, 2000)]
    raise ValueError("Unimplemented or unauthorized stage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("grids", "activity", "recurrence"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for job, arguments in jobs(args.stage):
        result = run(job, arguments, execute=args.execute)
        print(json.dumps({k: result[k] for k in ("job", "status", "reused", "elapsed_seconds", "peak_worker_bytes", "minimum_available_bytes") if k in result}), flush=True)
        if result["status"] not in ("passed", "dry_run"):
            raise SystemExit(2)
