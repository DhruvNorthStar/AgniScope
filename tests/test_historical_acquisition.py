"""All HTTP responses below are synthetic local fixtures; no NASA requests."""

from copy import deepcopy
import hashlib
import json

import pytest
import requests

import agniscope.acquisition as a
import agniscope.acquisition_store as store
from agniscope.config import load_config, workspace_path


HEADER = b"latitude,longitude,acq_date,acq_time,satellite,confidence,version,instrument,frp,daynight\n"
BODY = HEADER + b"30,79,2023-04-01,2330,N,n,2,VIIRS,0,N\n"
SECRET = "synthetic_stage_a_credential_123456"


class Response:
    def __init__(self, body=BODY, status=200, content_type="text/plain", extra=None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": content_type, **(extra or {})}

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    def iter_content(self, chunk_size):
        for index in range(0, len(self.body), 7):
            yield self.body[index:index+7]


class Session:
    def __init__(self, observations=(), quota=1):
        self.observations = iter(observations)
        self.observation_calls = 0
        self.metadata_calls = 0
        self.targets = []
        self.quota = quota

    def get(self, url, **kwargs):
        assert kwargs == {"stream": True, "timeout": (15, 90), "allow_redirects": False}
        if "/data_availability/" in url:
            self.metadata_calls += 1
            return Response(b"data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-06-30\n")
        if "mapkey_status" in url:
            self.metadata_calls += 1
            return Response(json.dumps({"transaction_limit": 5000, "current_transactions": self.quota, "transaction_interval": "10 minutes"}).encode(), content_type="application/json")
        self.observation_calls += 1
        self.targets.append(url)
        item = next(self.observations)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def setup_stage(tmp_path, monkeypatch):
    settings = load_config(workspace_path("configs/phase2b_stage_a.toml"))
    raw = tmp_path / "synthetic.csv"
    directory = tmp_path / "synthetic_store"
    original = a.workspace_path

    def resolver(path, **kwargs):
        if str(path) == settings["paths"]["raw"]:
            return raw
        if str(path) == settings["paths"]["store"]:
            return directory
        return original(path, **kwargs)

    monkeypatch.setattr(a, "workspace_path", resolver)
    monkeypatch.setattr(a, "local_map_key", lambda: SECRET)
    return settings, raw, directory


def execute(settings, session, sleeper=lambda seconds: None):
    return a.acquire_historical(settings, execute_stage_a=True, session=session, sleeper=sleeper)


def test_default_is_dry_run_without_network_or_files(setup_stage):
    settings, raw, directory = setup_stage
    session = Session()
    result = a.acquire_historical(settings, session=session)
    assert result["mode"] == "dry_run" and result["observation_requests"] == 0
    assert not raw.exists() and not directory.exists() and session.metadata_calls == 0


@pytest.mark.parametrize("start,end", [("2023-04-01", "2023-04-06"), ("2023-04-02", "2023-04-01")])
def test_five_day_hard_limit(setup_stage, start, end):
    settings, _, _ = setup_stage
    settings["request"].update(start=start, end=end)
    with pytest.raises(a.AcquisitionError, match="five days"):
        execute(settings, Session())


@pytest.mark.parametrize("stage,start,end", [("B", "2024-04-01", "2024-04-05"), ("C", "2025-04-01", "2025-04-05"), ("A", "2023-04-02", "2023-04-05")])
def test_execution_rejects_every_other_interval(setup_stage, stage, start, end):
    settings, _, _ = setup_stage
    settings["request"].update(stage=stage, start=start, end=end)
    with pytest.raises(a.AcquisitionError, match="allowlist"):
        execute(settings, Session())


@pytest.mark.parametrize("field,value", [("source", "VIIRS_SNPP_NRT"), ("bbox", [77, 28, 82, 32]), ("allowed_collection_values", ["1"]), ("maximum_response_bytes", 10485761)])
def test_scope_and_collection_fences(setup_stage, field, value):
    settings, _, _ = setup_stage
    settings["request"][field] = value
    with pytest.raises(a.AcquisitionError):
        execute(settings, Session())


def test_stage_flags_do_not_authorize_other_work(setup_stage):
    settings, _, _ = setup_stage
    settings["scope"]["stage_b_authorized"] = True
    with pytest.raises(a.AcquisitionError):
        execute(settings, Session())


def test_dry_run_leap_and_year_boundary_dates_are_pure():
    settings = load_config(workspace_path("configs/phase2b_stage_a.toml"))
    for start, end in [("2024-02-27", "2024-03-02"), ("2023-12-29", "2024-01-02")]:
        settings["request"].update(start=start, end=end)
        assert a.acquire_historical(settings)["request"]["days"] == 5


def test_positive_commit_preserves_bytes_versions_and_is_network_free_on_rerun(setup_stage):
    settings, raw, directory = setup_stage
    first = execute(settings, Session([Response()]))
    before = (raw.stat().st_mtime_ns, raw.with_suffix(".provenance.json").stat().st_mtime_ns)
    second_session = Session()
    second = execute(settings, second_session)
    assert raw.read_bytes() == BODY and first == second
    assert second["sha256"] == hashlib.sha256(BODY).hexdigest()
    assert second["collection_values"] == ["2"]
    assert second["acquisition_status"] == "acquired_positive" and second["satellite_completeness"] == "unverified"
    assert before == (raw.stat().st_mtime_ns, raw.with_suffix(".provenance.json").stat().st_mtime_ns)
    assert second_session.metadata_calls == second_session.observation_calls == 0
    assert SECRET not in (directory / "acquisition_manifest.json").read_text()
    assert SECRET not in raw.with_suffix(".provenance.json").read_text()


def test_header_only_is_successful_zero_without_complete_coverage(setup_stage):
    settings, raw, _ = setup_stage
    result = execute(settings, Session([Response(HEADER)]))
    assert result["acquisition_status"] == "acquired_zero" and result["raw_observation_count"] == 0
    assert result["collection_values"] == [] and result["satellite_completeness"] == "unverified"
    assert raw.read_bytes() == HEADER


@pytest.mark.parametrize("body", [b"", b"API failed", b"<html>failure</html>", b"latitude,longitude\n30,79\n"])
def test_invalid_csv_rejected_permanently_without_raw_commit(setup_stage, body):
    settings, raw, _ = setup_stage
    result = execute(settings, Session([Response(body)]))
    assert result["acquisition_status"] == "attempted_failed" and result["permanent_failure"]
    assert result["failure_category"] == "response_format" and not raw.exists()


@pytest.mark.parametrize("replacement", [b"2023-04-06", b"2024-04-01"])
def test_unexpected_dates_preserved_but_never_accepted(setup_stage, replacement):
    settings, raw, _ = setup_stage
    body = BODY.replace(b"2023-04-01", replacement)
    result = execute(settings, Session([Response(body)]))
    assert raw.read_bytes() == body
    assert result["acquisition_status"] == "retrieved_qc_failed"
    assert result["response_validation"]["failed_checks"]["acquisition_date"] == 1


@pytest.mark.parametrize("body", [BODY.replace(b",2,VIIRS", b",1,VIIRS"), BODY.replace(b",2,VIIRS", b",2NRT,VIIRS"), BODY.replace(b"30,79", b"30,90"), HEADER+b"30\n"])
def test_incompatible_parseable_source_is_preserved_and_flagged(setup_stage, body):
    settings, raw, _ = setup_stage
    result = execute(settings, Session([Response(body)]))
    assert result["acquisition_status"] == "retrieved_qc_failed" and raw.read_bytes() == body


def test_transient_attempts_are_bounded_same_target_and_durable(setup_stage):
    settings, raw, _ = setup_stage
    session = Session([Response(status=500), requests.ConnectionError("sensitive/"+SECRET), Response()])
    sleeps = []
    result = execute(settings, session, sleeps.append)
    assert session.observation_calls == len(result["attempts"]) == 3
    assert len(set(session.targets)) == 1 and sleeps == [2, 5, 20]
    assert result["acquisition_status"] == "acquired_positive" and raw.exists()
    assert SECRET not in json.dumps(result)


def test_retry_budget_is_not_reset_by_reinvocation(setup_stage):
    settings, _, _ = setup_stage
    failed = execute(settings, Session([Response(status=503)] * 3))
    session = Session()
    assert len(failed["attempts"]) == 3
    assert execute(settings, session) == failed
    assert session.observation_calls == session.metadata_calls == 0


def test_permanent_http_error_stops_without_retry_or_url_disclosure(setup_stage):
    settings, raw, _ = setup_stage
    session = Session([Response(status=403, body=SECRET.encode())])
    result = execute(settings, session)
    assert session.observation_calls == 1 and result["failure_category"] == "http_403"
    assert SECRET not in json.dumps(result) and not raw.exists()


def test_retry_after_longer_than_policy_stops(setup_stage):
    settings, _, _ = setup_stage
    session = Session([Response(status=429, extra={"Retry-After": "700"})])
    result = execute(settings, session)
    assert session.observation_calls == 1 and result["permanent_failure"]


def test_no_area_request_when_quota_reserve_is_low(setup_stage):
    settings, _, _ = setup_stage
    session = Session(quota=4999)
    with pytest.raises(a.AcquisitionError, match="reserve"):
        execute(settings, session)
    assert session.observation_calls == 0


@pytest.mark.parametrize("extra", [{}, {"Content-Length": "100000"}])
def test_cap_checks_actual_bytes_and_declared_size(setup_stage, extra):
    settings, raw, _ = setup_stage
    settings["request"]["maximum_response_bytes"] = 10
    result = execute(settings, Session([Response(BODY, extra=extra)]))
    assert result["failure_category"] == "response_size_limit" and not raw.exists()


def test_content_type_and_credential_echo_are_rejected(setup_stage):
    settings, raw, _ = setup_stage
    result = execute(settings, Session([Response(BODY+SECRET.encode(), content_type="text/"+SECRET)]))
    assert result["failure_category"] == "credential_echo" and not raw.exists()
    assert SECRET not in json.dumps(result)


def test_preexisting_untracked_raw_causes_conflict_without_network(setup_stage):
    settings, raw, _ = setup_stage
    raw.write_bytes(b"original fixture evidence")
    session = Session()
    with pytest.raises(store.StoreConflict):
        execute(settings, session)
    assert raw.read_bytes() == b"original fixture evidence" and session.metadata_calls == 0


@pytest.mark.parametrize("which", ["raw", "provenance", "manifest_count"])
def test_hash_and_manifest_conflicts_never_trigger_redownload(setup_stage, which):
    settings, raw, directory = setup_stage
    execute(settings, Session([Response()]))
    if which == "manifest_count":
        path = directory / "acquisition_manifest.json"
        value = json.loads(path.read_text())
        value["raw_observation_count"] = 99
        store.atomic_json(path, value)
    else:
        target = raw if which == "raw" else raw.with_suffix(".provenance.json")
        target.write_bytes(b"conflicting fixture")
    session = Session()
    with pytest.raises(store.StoreConflict):
        execute(settings, session)
    assert session.metadata_calls == session.observation_calls == 0


@pytest.mark.parametrize("field,value", [("http", {"http_status": 500, "content_type": "text/html"}), ("provenance_file", "unexpected_file.json")])
def test_manifest_metadata_and_paths_must_match_immutable_provenance(setup_stage, field, value):
    settings, _, directory = setup_stage
    execute(settings, Session([Response()]))
    path = directory / "acquisition_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    store.atomic_json(path, manifest)
    session = Session()
    with pytest.raises(store.StoreConflict):
        execute(settings, session)
    assert session.observation_calls == session.metadata_calls == 0


def test_recovery_after_raw_promotion_never_reissues_http(setup_stage, monkeypatch):
    settings, raw, _ = setup_stage
    original = store.promote_immutable
    count = 0

    def crash_after_first(*args):
        nonlocal count
        original(*args)
        count += 1
        if count == 1:
            raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(store, "promote_immutable", crash_after_first)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute(settings, Session([Response()]))
    assert raw.read_bytes() == BODY
    monkeypatch.setattr(store, "promote_immutable", original)
    session = Session()
    result = execute(settings, session)
    assert result["acquisition_status"] == "acquired_positive" and session.metadata_calls == 0


def test_orphan_staging_blocks_network_before_retry(setup_stage):
    settings, raw, _ = setup_stage
    raw.with_name(raw.name + ".stage_a_staged").write_bytes(BODY)
    session = Session()
    with pytest.raises(store.StoreConflict, match="staging"):
        execute(settings, session)
    assert session.observation_calls == session.metadata_calls == 0


def test_atomic_snapshot_failure_preserves_prior_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_manifest.json"
    store.atomic_json(path, {"state": "old"})

    def fail(*args):
        raise OSError("synthetic disk interruption")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(OSError):
        store.atomic_json(path, {"state": "new"})
    assert json.loads(path.read_text()) == {"state": "old"}
    assert not list(tmp_path.glob("*.tmp_*"))


def test_immutable_promotion_cannot_clobber_existing_file(tmp_path):
    staged, final = tmp_path / "synthetic_staged", tmp_path / "synthetic_final"
    staged.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(store.StoreConflict):
        store.promote_immutable(staged, final)
    assert final.read_bytes() == b"old" and staged.read_bytes() == b"new"


@pytest.mark.parametrize("body,expected_rows", [(HEADER, 0), (BODY, 1)])
def test_complete_qc_bundle_and_idempotent_local_integration(setup_stage, tmp_path, monkeypatch, body, expected_rows):
    import agniscope.historical_qc as qc
    settings, raw, _ = setup_stage
    manifest = execute(settings, Session([Response(body)]))
    output = tmp_path / "derived_synthetic_qc"
    original = qc.workspace_path

    def resolver(path, **kwargs):
        if str(path) == settings["paths"]["raw"]:
            return raw
        if str(path) == settings["paths"]["derived"]:
            return output
        return original(path, **kwargs)

    monkeypatch.setattr(qc, "workspace_path", resolver)
    summary, reused = qc.build_stage_a_qc(settings, manifest)
    assert summary["raw_observation_count"] == expected_rows
    assert summary["qc_contract"]["all_input_rows_accounted_for"] and not reused
    assert summary["integration"]["stable_ids_unique"] and summary["integration"]["utc_ist_instants_equal"]
    assert summary["integration"]["cross_interval_exact_payload_duplicate_groups"] == 0
    again, reused = qc.build_stage_a_qc(settings, manifest)
    assert again == summary and reused
    a.record_historical_qc(settings, output / "qc_summary.json", passed=True)
    session = Session()
    assert execute(settings, session)["qc_status"] == "passed"
    assert session.observation_calls == session.metadata_calls == 0
