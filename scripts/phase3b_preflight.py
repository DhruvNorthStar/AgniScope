"""Offline Phase 3B prerequisite checks. Never fit models or read credentials.

Each invocation creates an immutable attempt directory. A failed resource gate
stops before numerical/GIS imports, regression execution or recurrence work.
Passing this lightweight check is not, by itself, completion of scientific Gate A.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agniscope.acquisition_store import atomic_json, StoreConflict
from agniscope.provenance import sha256, utc_now
from agniscope.resources import resource_snapshot

CONFIG = ROOT / "configs/phase3b.toml"
OUT = ROOT / "outputs/tables/phase3b"
SEAL = ROOT / "outputs/tables/phase3a_low_ram/file_integrity_manifest.json"
SEAL_SHA = "84bf5292cd83bf3f19dc96dc10bb5590c8e3acb707beea3e8029883ee1113669"


def settings():
    result = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    r = result["resources"]
    expected = {
        "minimum_startup_available_ram_bytes": 896 * 1024**2,
        "minimum_available_ram_bytes": 512 * 1024**2,
        "minimum_disk_free_bytes": 15 * 1024**3,
        "maximum_worker_working_set_bytes": 512 * 1024**2,
        "startup_samples": 7,
        "startup_interval_seconds": 5,
        "monitor_interval_seconds": .1,
        "concurrency": 1,
    }
    if r != expected:
        raise StoreConflict("Phase 3B resource safeguards differ from the approved contract.")
    previous = tomllib.loads((ROOT / "configs/phase3a.toml").read_text())["resources"]
    for key in ("minimum_startup_available_ram_bytes", "minimum_available_ram_bytes",
                "minimum_disk_free_bytes", "maximum_worker_working_set_bytes",
                "monitor_interval_seconds", "concurrency"):
        if r[key] != previous[key]:
            raise StoreConflict("Phase 3A resource protection changed.")
    return result


def runtime_safe(sample, limits, worker_bytes=0, launch_available_bytes=None):
    cap = limits["maximum_worker_working_set_bytes"]
    if launch_available_bytes is not None:
        cap = min(cap, launch_available_bytes // 2)
    return (sample["available_physical_bytes"] >= limits["minimum_available_ram_bytes"]
            and sample["disk_free_bytes"] >= limits["minimum_disk_free_bytes"]
            and worker_bytes <= cap)


def stable_headroom(limits, *, snapshot=resource_snapshot, sleeper=time.sleep):
    samples = []
    started = time.monotonic()
    for index in range(limits["startup_samples"]):
        samples.append({"sample": index, "utc": utc_now(), **snapshot()})
        if index + 1 < limits["startup_samples"]:
            sleeper(limits["startup_interval_seconds"])
    falling = all(b["available_physical_bytes"] < a["available_physical_bytes"]
                  for a, b in zip(samples, samples[1:]))
    passed = (not falling and all(
        row["available_physical_bytes"] >= limits["minimum_startup_available_ram_bytes"]
        and runtime_safe(row, limits) for row in samples))
    return {
        "status": "passed" if passed else "BLOCKED",
        "samples": samples,
        "observation_seconds": (limits["startup_samples"] - 1) * limits["startup_interval_seconds"],
        "elapsed_seconds": time.monotonic() - started,
        "continuously_dropping": falling,
        "minimum_available_bytes": min(r["available_physical_bytes"] for r in samples),
        "maximum_available_bytes": max(r["available_physical_bytes"] for r in samples),
        "minimum_disk_free_bytes": min(r["disk_free_bytes"] for r in samples),
        "limits": limits,
        "numerical_modules_loaded": any(k in sys.modules for k in ("numpy", "pandas", "sklearn", "geopandas")),
    }


def hash_preflight(config):
    assert sha256(SEAL) == SEAL_SHA, "Current Phase 3A seal changed."
    inventory = json.loads(SEAL.read_text())
    assert len(inventory["files"]) == 2257
    for index, row in enumerate(inventory["files"], 1):
        path = (ROOT / row["path"]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file()
        assert sha256(path) == row["sha256"], "Sealed Phase 3A artifact changed."
        if index % 100 == 0:
            assert runtime_safe(resource_snapshot(), config["resources"]), "Unsafe resources during hash verification."
    # Reuse read-only archive-aware checks for the original GIS, raw observations,
    # previous scientific evidence and prior historical seals.
    from phase3a_recovery import preserved, load_plan
    protection = preserved()
    plan = load_plan()
    assert len(plan) == 209
    science = diagnostic = references = 0
    for spec in plan:
        directory = ROOT / "outputs/tables/phase3a/runs" / spec["run_id"][:20]
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["spec"] == spec
        assert sha256(ROOT / spec["input_path"]) == spec["input_sha256"]
        for row in receipt["files"]:
            assert sha256(ROOT / row["path"]) == row["sha256"]
        statistics = json.loads((directory / "statistics.json").read_text())
        assert statistics["spec"] == spec
        metrics = statistics["metrics"]
        assert metrics["n_observations"] == spec["n_observations"]
        assert metrics["n_clustered"] + metrics["n_nonmember"] == spec["n_observations"]
        is_probe = spec["role"] in ("high_volume_probe", "extended_probe")
        diagnostic += is_probe
        science += not is_probe
        if spec["reference"] and spec["season_id"].startswith("CORE_") and not is_probe:
            assert (directory / "assignments.parquet").is_file()
            references += 1
    assert (science, diagnostic, references) == (201, 8, 56)
    master = ROOT / config["study"]["master"]
    assert sha256(master) == config["study"]["master_sha256"]
    ledger = json.loads((ROOT / "outputs/tables/phase3a_low_ram/final_experiment_ledger.json").read_text())
    assert len(ledger) == 209 and all(row["classification"] == "verified_complete" for row in ledger)
    installed = [json.loads(p.read_text()) for p in (Path(sys.prefix) / "conda-meta").glob("*.json")]
    lock = {line for line in (ROOT / "environment.lock.win-64.txt").read_text().splitlines()
            if line and not line.startswith(("#", "@"))}
    assert lock == {p["url"] + "#" + p["sha256"] for p in installed}
    return {"status": "passed", "current_seal_sha256": SEAL_SHA,
            "current_seal_files_verified": len(inventory["files"]),
            "historical_preservation": protection,
            "master_sha256": sha256(master), "scientific_receipts_verified": science,
            "diagnostic_receipts_verified": diagnostic,
            "primary_full_and_subset_reference_files_hashed": references,
            "all_receipt_payload_hashes_verified": True,
            "prior_ledger_verified": True, "installed_conda_artifacts": len(installed),
            "environment_lock_verified": True,
            "independent_Parquet_and_geometry_reconstruction": "pending_resource_gate"}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    config = settings()
    stamp = utc_now().replace(":", "").replace(".", "").replace("+", "_")
    attempt = OUT / "preflight_attempts" / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"started_utc": utc_now(), "configuration_sha256": sha256(CONFIG),
              "resources_before": resource_snapshot(), "observation_HTTP_calls": 0,
              "credential_reads": 0, "clustering_refits": 0,
              "new_recurrence_calculations": 0, "new_workers_launched": 0}
    try:
        record["hash_preflight"] = hash_preflight(config)
        atomic_json(attempt / "hash_verification.json", record["hash_preflight"])
        print(json.dumps({"hash_preflight": record["hash_preflight"], "sampling_seconds": 30}), flush=True)
        gate = stable_headroom(config["resources"])
        atomic_json(attempt / "startup_resource_gate.json", gate)
        record["startup_resource_gate"] = gate
        record["status"] = "READY_FOR_REMAINING_PREFLIGHT" if gate["status"] == "passed" else "BLOCKED"
        record["reason"] = ("Independent scientific reconstruction and current regression suite remain required."
                            if gate["status"] == "passed" else
                            "The configured 30-second available-RAM/disk/trend gate failed; stop before numerical processing.")
    except (AssertionError, StoreConflict, OSError) as error:
        record["status"] = "BLOCKED"
        # Only our fixed validation messages are exposed, never URLs/environment values.
        record["reason"] = str(error) if isinstance(error, AssertionError) else type(error).__name__
    record["finished_utc"] = utc_now()
    record["resources_after"] = resource_snapshot()
    atomic_json(attempt / "preflight.json", record)
    print(json.dumps({"status": record["status"], "reason": record["reason"],
                      "attempt": attempt.relative_to(ROOT).as_posix(),
                      "gate": record.get("startup_resource_gate")}), flush=True)
    return 0 if record["status"] == "READY_FOR_REMAINING_PREFLIGHT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
