"""Default offline plan; explicit execution processes one bounded 2024-only batch."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agniscope import acquisition as a
from agniscope.acquisition_store import StoreConflict, atomic_json
from agniscope.config import load_config, workspace_path
from agniscope.provenance import sha256, utc_now
from agniscope.stage_b import plan_2024, protected_integrity, run_batch, load_catalog
from agniscope.stage_b_qc import coverage_rows, integrate_year, write_table


OUTPUT = workspace_path("outputs/tables/phase2b_stage_b", writable=True)
TRACKED = ["src/agniscope/acquisition.py", "src/agniscope/acquisition_store.py", "src/agniscope/stage_b.py", "src/agniscope/stage_b_qc.py",
           "scripts/phase2b_stage_b.py", "configs/phase2b_stage_b.toml", "tests/test_stage_b.py", "tests/test_stage_b_qc.py"]


def certify_tests():
    path = OUTPUT / "pytest_regression.xml"
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None or int(suite.attrib["tests"]) < 211 or any(int(suite.attrib[k]) for k in ("failures", "errors", "skipped")):
        raise StoreConflict("A passing full regression suite is required.")
    result = {"certified_utc": utc_now(), "tests": int(suite.attrib["tests"]), "failures": 0, "xml_sha256": sha256(path),
              "files": [{"path": name, "sha256": sha256(workspace_path(name))} for name in TRACKED]}
    atomic_json(OUTPUT / "executor_test_gate.json", result)
    return result


def verify_test_gate():
    gate = json.loads((OUTPUT / "executor_test_gate.json").read_text(encoding="utf-8"))
    if gate["tests"] < 211 or gate["failures"] or sha256(OUTPUT / "pytest_regression.xml") != gate["xml_sha256"]:
        raise StoreConflict("Passing regression evidence changed.")
    for item in gate["files"]:
        if sha256(workspace_path(item["path"])) != item["sha256"]:
            raise StoreConflict("Current executor/config/tests differ from the tested version.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute-stage-b", action="store_true")
    mode.add_argument("--certify-tests", action="store_true")
    mode.add_argument("--integrate", action="store_true", help="Local annual integration after all intervals pass")
    mode.add_argument("--audit-only", action="store_true", help="Local manifests/coverage only; no acquisition")
    mode.add_argument("--check-upstream", action="store_true", help="Public metadata only; no observation request")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(argv)
    settings = load_config(workspace_path("configs/phase2b_stage_b.toml"))
    try:
        plan = plan_2024(settings)
        protected_integrity(settings)
        if args.certify_tests:
            print(json.dumps({"status": "test_gate_certified", "tests": certify_tests()["tests"], "observation_requests": 0}))
            return 0
        if args.check_upstream:
            http = {}
            body = a._bounded_get("https://firms.modaps.eosdis.nasa.gov/api/missing_data/", 524288, response_metadata=http)
            evidence = {"checked_utc": utc_now(), "public_url": "https://firms.modaps.eosdis.nasa.gov/api/missing_data/", "http": http,
                        "size_bytes": len(body), "response_sha256": __import__("hashlib").sha256(body).hexdigest(),
                        "failed_to_load_message_present": b"Failed to load data" in body,
                        "daily_missing_calendar_verified": False,
                        "official_gps_notice": "https://landweb.modaps.eosdis.nasa.gov/displayissue?id=605",
                        "official_cdp_notice": "https://landweb.modaps.eosdis.nasa.gov/displayissue?id=716",
                        "upstream_issue_register_sha256": sha256(workspace_path("configs/historical_upstream_issues.csv"))}
            atomic_json(OUTPUT / "upstream_metadata_evidence.json", evidence)
            print(json.dumps(evidence, indent=2))
            return 0
        if not any((args.execute_stage_b, args.integrate, args.audit_only)):
            print(json.dumps(run_batch(settings, batch_size=args.batch_size), indent=2))
            return 0
        verify_test_gate()
        if args.execute_stage_b:
            result = run_batch(settings, execute=True, batch_size=args.batch_size)
        else:
            result = {"status": "local_audit", "observation_requests": 0}
        catalog = load_catalog(settings)
        if catalog is None:
            raise StoreConflict("No Stage B manifest exists; audit mode never initiates acquisition.")
        write_table(OUTPUT / "daily_coverage_2024.csv", coverage_rows(settings, catalog, plan))
        if args.integrate:
            summary, reused = integrate_year(settings, catalog, plan)
            result = {"status": summary["stage_b_status"], "counts": summary["counts"], "annual_dataset_reused": reused, "observation_requests": 0}
        print(json.dumps(result, indent=2))
        return 2 if catalog["halt_reason"] else 0
    except (a.AcquisitionError, StoreConflict, OSError, ValueError, KeyError):
        print("Stage B stopped by a scope, test, integrity, resource, metadata or QC gate. Sensitive details withheld; inspect the safe manifests.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
