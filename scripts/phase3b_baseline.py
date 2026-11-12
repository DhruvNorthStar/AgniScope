"""Run the unchanged 547-case pre-existing suite into a new evidence area."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([
        "tests", "scripts/qa/prior_context_cases.py", "scripts/qa/integration_summary_cases.py",
        "scripts/qa/phase3a_cases.py", "scripts/qa/phase3a_partial_cases.py", "-q",
        "--basetemp=.cache/pytest/phase3b_baseline_retry",
        "--junitxml=outputs/tables/phase3b/resume_20261023/pytest_baseline.xml",
    ]))
