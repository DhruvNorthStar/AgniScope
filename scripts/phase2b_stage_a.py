"""Stage A only. Default: offline dry run. No automatic next-stage execution."""

import argparse
from collections import Counter
import csv
from datetime import date
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agniscope import acquisition as a
from agniscope.acquisition_store import StoreConflict, atomic_json
from agniscope.config import load_config, workspace_path
from agniscope.historical_qc import build_stage_a_qc
from agniscope.planning import build_request_plan, dates
from agniscope.provenance import sha256, utc_now, verify_sources
from agniscope.resources import resource_snapshot


OUTPUT = workspace_path("outputs/tables/phase2b_stage_a", writable=True)
TESTED_FILES = ["src/agniscope/acquisition.py", "src/agniscope/acquisition_store.py", "src/agniscope/historical_qc.py",
                "scripts/phase2b_stage_a.py", "scripts/verify_phase2b_stage_a.py", "configs/phase2b_stage_a.toml",
                "tests/test_historical_acquisition.py", "tests/test_historical_qc.py", "tests/test_stage_a_cli.py"]


def verify_preflight():
    preflight = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
    changed = [r["path"] for r in preflight["protected_files"] if sha256(workspace_path(r["path"])) != r["sha256"]]
    if set(changed) - {"src/agniscope/acquisition.py"}:
        raise StoreConflict("Protected Phase 0-2A evidence changed.")
    sources = verify_sources(load_config())
    if not sources["all_unchanged"] or len(sources["files"]) != 14:
        raise StoreConflict("Copied original GIS integrity failed.")
    plan = build_request_plan(load_config(workspace_path("configs/historical_protocol.toml")))
    with workspace_path("outputs/tables/phase2a/request_plan.csv").open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    if [{k: "" if v is None else str(v) for k, v in row.items()} for row in plan] != saved:
        raise StoreConflict("The Phase 2A plan does not reproduce exactly.")
    expanded = [day for r in plan for day in dates(date.fromisoformat(r["start_utc_date"]), date.fromisoformat(r["end_utc_date"]))]
    stages = dict(Counter(r["proposed_stage"] for r in plan if r["status"] == "planned_not_requested"))
    if expanded != list(dates(date(2019, 1, 1), date(2025, 12, 31))) or len(plan) != 513 or stages != {"A": 1, "B": 73, "C": 438}:
        raise StoreConflict("Study calendar, interval or stage counts fail reconciliation.")
    a_rows = [r for r in plan if r["proposed_stage"] == "A"]
    if len(a_rows) != 1 or (a_rows[0]["start_utc_date"], a_rows[0]["end_utc_date"]) != ("2023-04-01", "2023-04-05"):
        raise StoreConflict("Stage A no longer identifies the approved interval.")
    report = {"checked_utc": utc_now(), "plan_reproduced": True, "study_days": len(expanded), "logical_intervals": len(plan),
              "stage_requests_before_A": stages, "protected_existing_files": len(preflight["protected_files"]),
              "protected_data_and_prior_outputs_unchanged": True, "approved_existing_code_extensions": changed,
              "copied_original_gis_files_unchanged": 14, "current_acquisition_module_sha256": sha256(workspace_path("src/agniscope/acquisition.py"))}
    atomic_json(OUTPUT / "preflight_recheck.json", report)
    return report


def verify_test_gate():
    gate = json.loads((OUTPUT / "executor_test_gate.json").read_text(encoding="utf-8"))
    if gate["failures"] or gate["errors"] or gate["tests"] < 159:
        raise StoreConflict("A successful full regression test gate is required before execution.")
    for item in gate["tested_files"]:
        if sha256(workspace_path(item["path"])) != item["sha256"]:
            raise StoreConflict("Executor/config/tests changed after the passing test gate.")
    xml_name = gate.get("xml_file", "pytest_before_download.xml")
    if xml_name not in {"pytest_before_download.xml", "pytest_regression.xml"} or sha256(OUTPUT / xml_name) != gate["xml_sha256"]:
        raise StoreConflict("Passing regression evidence changed.")


def certify_tests():
    """Offline receipt for the latest complete regression; no acquisition authorization."""
    xml_path = OUTPUT / "pytest_regression.xml"
    suite = ET.parse(xml_path).getroot().find("testsuite")
    if suite is None or int(suite.attrib["tests"]) < 159 or any(int(suite.attrib[k]) for k in ("failures", "errors", "skipped")):
        raise StoreConflict("A passing full regression XML is required to certify the current code.")
    gate_path = OUTPUT / "executor_test_gate.json"
    archive = OUTPUT / "executor_test_gate_at_download.json"
    if gate_path.exists() and not archive.exists():
        with archive.open("xb") as handle:
            handle.write(gate_path.read_bytes())
    gate = {"checked_utc": utc_now(), "tests": int(suite.attrib["tests"]), "failures": 0, "errors": 0, "skipped": 0,
            "xml_file": xml_path.name, "xml_sha256": sha256(xml_path),
            "tested_files": [{"path": name, "sha256": sha256(workspace_path(name))} for name in TESTED_FILES]}
    atomic_json(gate_path, gate)
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-stage-a", action="store_true", help="Only the approved April 1-5, 2023 interval; reuse verified existing acquisition")
    mode.add_argument("--audit-only", action="store_true", help="Local verification/QC of an already committed interval; never acquire")
    mode.add_argument("--certify-tests", action="store_true", help="Offline receipt for the latest passing regression XML; no acquisition")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    settings = load_config(workspace_path("configs/phase2b_stage_a.toml"))
    try:
        preflight = verify_preflight()
        if args.certify_tests:
            gate = certify_tests()
            print(json.dumps({"status": "current_code_test_gate_certified", "tests": gate["tests"], "observation_requests": 0}))
            return 0
        if not args.execute_stage_a and not args.audit_only:
            result = a.acquire_historical(settings)
            result["full_plan_dry_run"] = {"expected_dates": preflight["study_days"], "logical_intervals": preflight["logical_intervals"],
                                           "future_requests_before_A": 512, "stage_requests_before_A": preflight["stage_requests_before_A"],
                                           "stage_b_c_execution_permitted": False}
            print(json.dumps(result, indent=2))
            return 0
        verify_test_gate()
        manifest_path = OUTPUT / "acquisition_manifest.json"
        before = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if args.audit_only and before.get("acquisition_status") not in {"acquired_positive", "acquired_zero", "retrieved_qc_failed"}:
            raise StoreConflict("Audit-only requires a committed local response; no request attempted.")
        resources_before = resource_snapshot()
        if resources_before["disk_free_bytes"] < 15 * 1024**3 or resources_before.get("available_physical_bytes", 1024**3) < 512 * 1024**2:
            raise StoreConflict("Disk/RAM reserve gate failed; no acquisition attempted.")
        manifest = a.acquire_historical(settings, execute_stage_a=True)
        new_attempts = len(manifest["attempts"]) - len(before.get("attempts", []))
        if manifest["acquisition_status"] not in {"acquired_positive", "acquired_zero"}:
            print(json.dumps({"status": manifest["acquisition_status"], "failure_category": manifest["failure_category"], "observation_attempts": len(manifest["attempts"])}))
            return 2
        if new_attempts:
            try:
                after_quota = a._transaction_status(a.local_map_key())
                atomic_json(OUTPUT / "post_acquisition_quota.json", {"snapshot": after_quota,
                            "observed_counter_difference": after_quota["current_transactions"] - manifest["quota_snapshots"][-1]["current_transactions"],
                            "not_exclusive_request_cost": True, "note": "Status calls, window resets and other account activity can affect this difference."})
            except a.AcquisitionError:
                atomic_json(OUTPUT / "post_acquisition_quota.json", {"status": "metadata_check_failed", "response_remains_committed": True})
        summary, reused_qc = build_stage_a_qc(settings, manifest)
        if manifest["qc_status"] != "passed":
            manifest = a.record_historical_qc(settings, workspace_path(settings["paths"]["derived"]) / "qc_summary.json", passed=summary["qc_contract"]["all_input_rows_accounted_for"])
        history_path = OUTPUT / "run_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"runs": []}
        history["runs"].append({"finished_utc": utc_now(), "mode": "audit_only" if args.audit_only else "execute_stage_a",
                                "elapsed_seconds": round(time.perf_counter() - started, 3), "new_observation_attempts": new_attempts,
                                "raw_sha256": manifest["sha256"], "raw_rows": summary["raw_observation_count"],
                                "retained_rows": summary["retained_observation_count"], "qc_bundle_reused": reused_qc,
                                "resources_before": resources_before, "resources_after": resource_snapshot()})
        atomic_json(history_path, history)
        print(json.dumps({"status": "stage_a_acquired_and_audited", "raw_rows": summary["raw_observation_count"],
                          "retained_rows": summary["retained_observation_count"], "new_observation_attempts_this_run": new_attempts,
                          "total_observation_attempts": len(manifest["attempts"]), "qc_bundle_reused": reused_qc,
                          "satellite_completeness": "unverified", "stage_b_authorized": False}, indent=2))
        return 0
    except (a.AcquisitionError, StoreConflict, OSError, ValueError, KeyError):
        # No arbitrary exception, response or credential-bearing URL reaches console.
        print("Stage A stopped: a scope, integrity, resource, metadata or validation gate failed. Existing evidence is preserved; inspect the sanitized manifest/checks.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
