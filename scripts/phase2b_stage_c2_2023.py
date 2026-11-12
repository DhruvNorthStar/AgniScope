"""Explicit 2023-only Stage C2; one sequential batch per invocation, offline by default."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agniscope import acquisition as a
from agniscope.acquisition_store import StoreConflict, atomic_json
from agniscope.config import load_config, workspace_path
from agniscope.provenance import sha256, utc_now
from agniscope.resources import resource_snapshot
from agniscope.stage_b import plan_2023, protected_integrity, run_batch, load_catalog
from agniscope.stage_b_qc import coverage_rows, integrate_year, write_table

CONFIG = "configs/phase2b_stage_c2_2023.toml"
OUTPUT = workspace_path("outputs/tables/phase2b_stage_c2_2023", writable=True)
TRACKED = ["src/agniscope/acquisition.py", "src/agniscope/acquisition_store.py", "src/agniscope/stage_b.py", "src/agniscope/stage_b_qc.py",
           "scripts/phase2b_stage_c2_2023.py", "scripts/verify_phase2b_stage_b.py", "scripts/verify_phase2b_stage_c2_2023.py",
           CONFIG, "tests/test_stage_c2.py"]


def test_gate(certify=False):
    xml = OUTPUT / "pytest_regression.xml"
    path = OUTPUT / "executor_test_gate.json"
    if certify:
        suite = ET.parse(xml).getroot().find("testsuite")
        if suite is None or int(suite.attrib["tests"]) < 273 or any(int(suite.attrib[k]) for k in ("failures", "errors", "skipped")):
            raise StoreConflict("Full passing regression evidence is required.")
        atomic_json(path, {"certified_utc": utc_now(), "tests": int(suite.attrib["tests"]), "xml_sha256": sha256(xml),
                           "files": [{"path": name, "sha256": sha256(workspace_path(name))} for name in TRACKED]})
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence["tests"] < 273 or sha256(xml) != evidence["xml_sha256"] or any(sha256(workspace_path(r["path"])) != r["sha256"] for r in evidence["files"]):
        raise StoreConflict("The current implementation differs from passing regression evidence.")
    return evidence


def metadata_check(settings):
    # Only source availability and public missing-data metadata; never observation data.
    key = a.local_map_key()
    if not key:
        raise StoreConflict("A configured local MAP_KEY is required.")
    http = {}
    body = a._bounded_get(f"{a.API_BASE}/data_availability/csv/{key}/VIIRS_SNPP_SP", 65536, response_metadata=http)
    available = a._availability(body, "VIIRS_SNPP_SP")
    if available["min_date"] > "2023-01-01" or available["max_date"] < "2023-12-31":
        raise StoreConflict("Official availability does not include all approved 2023 dates.")
    evidence = {"checked_utc": utc_now(), "source": "VIIRS_SNPP_SP", "availability": available, "http": http,
                "availability_response_sha256": hashlib.sha256(body).hexdigest(), "observation_requests": 0,
                "upstream_issue_register_sha256": sha256(workspace_path("configs/historical_upstream_issues.csv"))}
    public_http = {}
    try:
        public = a._bounded_get("https://firms.modaps.eosdis.nasa.gov/api/missing_data/", 524288, response_metadata=public_http)
        evidence["global_missing_metadata"] = {"public_url": "https://firms.modaps.eosdis.nasa.gov/api/missing_data/", "http": public_http,
             "size_bytes": len(public), "sha256": hashlib.sha256(public).hexdigest(), "failed_to_load_message_present": b"Failed to load data" in public,
             "daily_missing_calendar_verified": False}
    except a.AcquisitionError:
        evidence["global_missing_metadata"] = {"status": "unavailable", "daily_missing_calendar_verified": False}
    atomic_json(OUTPUT / "upstream_metadata_evidence.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for flag in ("execute-stage-c2", "certify-tests", "check-metadata", "integrate", "audit-only"):
        mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    settings = load_config(workspace_path(CONFIG))
    try:
        plan = plan_2023(settings)
        protected_integrity(settings)
        if args.certify_tests:
            print(json.dumps({"tests": test_gate(True)["tests"], "observation_requests": 0})); return 0
        if args.check_metadata:
            print(json.dumps(metadata_check(settings), indent=2)); return 0
        if not any((args.execute_stage_c2, args.integrate, args.audit_only)):
            print(json.dumps(run_batch(settings, batch_size=args.batch_size), indent=2)); return 0
        test_gate()
        if args.execute_stage_c2:
            metadata = json.loads((OUTPUT / "upstream_metadata_evidence.json").read_text(encoding="utf-8"))
            if metadata["source"] != "VIIRS_SNPP_SP" or metadata["availability"]["max_date"] < "2023-12-31":
                raise StoreConflict("Reviewed 2023 metadata preflight is required.")
            result = run_batch(settings, execute=True, batch_size=args.batch_size)
        else:
            result = {"status": "local_audit", "observation_requests": 0}
        catalog = load_catalog(settings)
        if catalog is None:
            raise StoreConflict("No C2 manifest exists; local audit cannot initiate acquisition.")
        write_table(OUTPUT / "daily_coverage_2023.csv", coverage_rows(settings, catalog, plan))
        if args.integrate:
            before = resource_snapshot()
            summary, reused = integrate_year(settings, catalog, plan)
            atomic_json(OUTPUT / "integration_resources.json", {"before": before, "after": resource_snapshot()})
            result = {"status": summary["stage_c2_status"], "counts": summary["counts"], "annual_dataset_reused": reused, "observation_requests": 0}
        print(json.dumps(result, indent=2))
        return 2 if catalog["halt_reason"] else 0
    except (a.AcquisitionError, StoreConflict, OSError, ValueError, KeyError):
        print("Stage C2 stopped by a scope, test, integrity, resource, metadata or QC gate. Inspect the safe evidence; sensitive details withheld.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
