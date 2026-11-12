"""Small serial job monitor for offline Phase 3B; default is a dry run."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agniscope.acquisition_store import atomic_json, store_lock, StoreConflict
from agniscope.provenance import sha256, utc_now
from agniscope.resources import resource_snapshot
from phase3b_preflight import settings, runtime_safe
from phase3a import worker_memory

OUT = ROOT / "outputs/tables/phase3b/resume_20261023"


def run(job, arguments, *, execute=False, timeout=1800):
    limits = settings()["resources"]
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "STOP.json").exists():
        raise StoreConflict("A previous critical stop requires a separate reviewed resume.")
    if not job.replace("_", "").isalnum():
        raise ValueError("Use a simple job identifier.")
    commands = [sys.executable, "-B", *arguments]
    if not execute:
        return {"status": "dry_run", "job": job, "worker_count": 0}
    record_path = OUT / "jobs" / (job + ".json")
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["status"] != "passed" or previous["arguments"] != arguments:
            raise StoreConflict("Conflicting or incomplete job record; preserve evidence.")
        if sha256(ROOT / previous["log_path"]) != previous["log_sha256"]:
            raise StoreConflict("Job log checksum changed.")
        return previous | {"reused": True}
    attempts = sorted((ROOT / "outputs/tables/phase3b/preflight_attempts").glob("*/preflight.json"))
    startup = json.loads(attempts[-1].read_text())
    if startup["status"] != "READY_FOR_REMAINING_PREFLIGHT":
        raise StoreConflict("A successful stable startup observation is required.")
    before = resource_snapshot()
    if not runtime_safe(before, limits) or before["available_physical_bytes"] < 768 * 1024**2:
        raise StoreConflict("Unsafe RAM/disk before isolated job.")
    cap = min(limits["maximum_worker_working_set_bytes"], before["available_physical_bytes"] // 2)
    log_path = OUT / "jobs" / (job + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise StoreConflict("Uncommitted log exists; do not overwrite.")
    record = {"job": job, "arguments": arguments, "started_utc": utc_now(),
              "resources_before": before, "cap_bytes": cap, "timeout_seconds": timeout,
              "startup_record": attempts[-1].relative_to(ROOT).as_posix(),
              "startup_record_sha256": sha256(attempts[-1]),
              "log_path": log_path.relative_to(ROOT).as_posix()}
    peak = 0
    minimum = before["available_physical_bytes"]
    disk_minimum = before["disk_free_bytes"]
    started = time.monotonic()
    failure = None
    with store_lock(OUT / "worker.lock"):
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(commands, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while process.poll() is None:
                current = resource_snapshot()
                memory = worker_memory(process.pid)
                minimum = min(minimum, current["available_physical_bytes"])
                disk_minimum = min(disk_minimum, current["disk_free_bytes"])
                if memory is not None:
                    peak = max(peak, memory["peak"])
                if not runtime_safe(current, limits, peak, before["available_physical_bytes"]):
                    failure = "runtime_RAM_disk_or_worker_cap"
                if time.monotonic() - started > timeout:
                    failure = "bounded_job_timeout"
                if failure:
                    process.terminate()
                    break
                time.sleep(limits["monitor_interval_seconds"])
            code = process.wait(timeout=30)
    record.update(status="passed" if code == 0 and failure is None else "BLOCKED",
                  returncode=code, stop_reason=failure, finished_utc=utc_now(),
                  elapsed_seconds=time.monotonic() - started, peak_worker_bytes=peak,
                  minimum_available_bytes=minimum, minimum_disk_free_bytes=disk_minimum,
                  resources_after=resource_snapshot(), log_sha256=sha256(log_path), reused=False)
    atomic_json(record_path, record)
    if record["status"] != "passed":
        atomic_json(OUT / "STOP.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        parser.error("An offline Python script or module is required.")
    result = run(args.job, arguments, execute=args.execute, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("passed", "dry_run") else 2


if __name__ == "__main__":
    raise SystemExit(main())
