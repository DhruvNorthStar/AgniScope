"""Run Phase 3B fixture or final regression checks in the guarded worker."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from agniscope.provenance import sha256
from phase3b_certificates import SOURCES, certify

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("grid", "final"))
    args = parser.parse_args()
    cases = (["tests/test_recurrence.py", "scripts/qa/phase3b_resource_cases.py"] if args.stage == "grid" else
             ["tests", "scripts/qa/prior_context_cases.py", "scripts/qa/integration_summary_cases.py",
              "scripts/qa/phase3a_cases.py", "scripts/qa/phase3a_partial_cases.py", "scripts/qa/phase3b_resource_cases.py"])
    suffix = "grid_certified" if args.stage == "grid" else "final"
    xml = ROOT / ("outputs/tables/phase3b/resume_20261023/pytest_"+suffix+".xml")
    assert not xml.exists(), "Preserve previous XML records."
    sources = {name: sha256(ROOT/name) for name in SOURCES}
    cases += ["-q", "--basetemp=.cache/pytest/phase3b_"+suffix, "--junitxml="+str(xml)]
    result = pytest.main(cases)
    if result == 0 and args.stage == "grid":
        certify(xml, xml.with_suffix(".certificate.json"), sources)
    raise SystemExit(result)
