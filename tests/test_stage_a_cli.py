"""The command's test receipt must track exactly the code and XML it certifies."""

import importlib.util
import json

import pytest

from agniscope.acquisition_store import StoreConflict
from agniscope.config import workspace_path
from agniscope.provenance import sha256


@pytest.mark.parametrize("mutate", [False, True])
def test_code_gate_accepts_matching_files_and_rejects_changed_code(tmp_path, monkeypatch, mutate):
    spec = importlib.util.spec_from_file_location("synthetic_cli_import", workspace_path("scripts/phase2b_stage_a.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    source = tmp_path / "synthetic_executor.py"
    source.write_text("# synthetic fixture\n", encoding="utf-8")
    xml = tmp_path / "pytest_regression.xml"
    xml.write_text('<testsuites><testsuite tests="159" failures="0" errors="0" skipped="0" /></testsuites>', encoding="utf-8")
    gate = {"tests": 159, "failures": 0, "errors": 0, "xml_file": xml.name, "xml_sha256": sha256(xml),
            "tested_files": [{"path": str(source), "sha256": sha256(source)}]}
    (tmp_path / "executor_test_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    if mutate:
        source.write_text("# changed fixture\n", encoding="utf-8")
        with pytest.raises(StoreConflict):
            module.verify_test_gate()
    else:
        module.verify_test_gate()
