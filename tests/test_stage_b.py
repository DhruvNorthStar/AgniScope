"""Stage B HTTP is mocked. Synthetic artifacts stay in workspace-local test cache."""

from copy import deepcopy
import json

import pytest

from agniscope import acquisition as a
from agniscope import acquisition_store as storage
from agniscope import stage_b as b
from agniscope import stage_b_qc as q
from agniscope.config import load_config, workspace_path


HEADER = b"latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type\n"
SYNTHETIC_KEY = "synthetic_stage_b_fixture_key_123456"


class Response:
    def __init__(self, body, status=200, mime="text/plain", extra=None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Type": mime, **(extra or {})}

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def iter_content(self, chunk_size): yield self.body


class Session:
    def __init__(self, failures=(), zero_dates=(), row_date=None):
        self.failures = list(failures)
        self.zero_dates = set(zero_dates)
        self.row_date = row_date
        self.observation_dates = []
        self.metadata_calls = 0

    def get(self, url, **kwargs):
        assert kwargs == {"stream": True, "timeout": (15, 90), "allow_redirects": False}
        if "/data_availability/" in url:
            self.metadata_calls += 1
            return Response(b"data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-06-30\n")
        if "mapkey_status" in url:
            self.metadata_calls += 1
            return Response(b'{"transaction_limit":5000,"current_transactions":1,"transaction_interval":"10 minutes"}', mime="application/json")
        day = url.rsplit("/", 1)[1]
        self.observation_dates.append(day)
        status = self.failures.pop(0) if self.failures else 200
        body = HEADER if day in self.zero_dates else HEADER + f"30,79,320,0.38,0.59,{self.row_date or day},2330,N,VIIRS,n,2,290,0,N,0\n".encode()
        return Response(body, status)


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    settings = load_config(workspace_path("configs/phase2b_stage_b.toml"))
    original = workspace_path
    def resolver(path, **kwargs):
        text = str(path).replace("\\", "/")
        for prefix, target in [(settings["paths"]["output"], tmp_path / "output"), (settings["paths"]["derived"], tmp_path / "derived"), (settings["paths"]["annual"], tmp_path / "annual")]:
            if text == prefix or text.startswith(prefix + "/"):
                return target / text[len(prefix):].lstrip("/")
        if text.startswith("data/raw/firms/VIIRS_SNPP_SP_2024-") and "2024-04-01_5days" not in text:
            return tmp_path / "raw" / text.rsplit("/", 1)[1]
        return original(path, **kwargs)
    monkeypatch.setattr(b, "workspace_path", resolver)
    monkeypatch.setattr(q, "workspace_path", resolver)
    resolver.protected_checker = b.protected_integrity
    monkeypatch.setattr(a, "local_map_key", lambda: SYNTHETIC_KEY)
    monkeypatch.setattr(b, "protected_integrity", lambda config: {"synthetic_fixture": True})
    monkeypatch.setattr(b, "resource_snapshot", lambda: {"disk_free_bytes": 32*1024**3, "available_physical_bytes": 2*1024**3})
    def qc_stub(config, request, entry):
        path = tmp_path / f"synthetic_qc_{request['request_id']}.json"
        summary = {"qc_passed": True, "retained_rows": entry["raw_observation_count"], "qc_summary_file": path.relative_to(original(".")).as_posix()}
        storage.atomic_json(path, summary)
        return summary
    return settings, resolver, qc_stub


def batch(stage_env, size=1, session=None):
    settings, _, stub = stage_env
    return b.run_batch(settings, execute=True, batch_size=size, session=session or Session(), sleeper=lambda seconds: None, qc_function=stub)


def test_independent_leap_calendar_and_pilot_reuse(stage_env):
    settings, _, _ = stage_env
    plan = b.plan_2024(settings)
    assert len(plan) == 74 and sum(r["days"] for r in plan) == 366
    assert sum(r["proposed_stage"] == "B" for r in plan) == 73
    assert [(r["start_utc_date"], r["end_utc_date"]) for r in plan if r["status"] == "reuse_verified_pilot"] == [("2024-04-01", "2024-04-05")]
    assert all(1 <= r["days"] <= 5 and r["start_utc_date"].startswith("2024-") and r["end_utc_date"].startswith("2024-") for r in plan)
    assert any(r["start_utc_date"] <= "2024-02-29" <= r["end_utc_date"] for r in plan)


@pytest.mark.parametrize("section,key,value", [("study", "stage", "C"), ("study", "start", "2023-01-01"), ("study", "end", "2025-12-31"), ("study", "source", "VIIRS_NOAA20_SP"), ("study", "maximum_new_intervals", 74), ("batch", "maximum_new_intervals", 11), ("batch", "concurrency", 2), ("scope", "stage_c_authorized", True)])
def test_stage_b_rejects_scope_expansion_before_network(stage_env, section, key, value):
    settings, _, _ = stage_env
    settings[section][key] = value
    with pytest.raises(a.AcquisitionError): b.run_batch(settings)


def test_default_dry_run_makes_no_calls_or_manifest(stage_env):
    settings, resolver, _ = stage_env
    session = Session()
    result = b.run_batch(settings, session=session)
    assert result["mode"] == "dry_run" and result["observation_requests"] == 0
    assert session.metadata_calls == 0 and not resolver(settings["paths"]["output"]).exists()


@pytest.mark.parametrize("size", [0, 11, 73])
def test_rejects_uncontrolled_batch_sizes(stage_env, size):
    with pytest.raises(a.AcquisitionError, match="ten"): batch(stage_env, size)


def test_canonical_interval_cannot_be_mutated(stage_env):
    settings, _, _ = stage_env
    request = deepcopy(b.plan_2024(settings)[0])
    request["end_utc_date"] = "2024-01-06"
    with pytest.raises(a.AcquisitionError): b.acquire_stage_b_interval(settings, request, execute=True, session=Session())


def test_pilot_is_never_a_network_target(stage_env):
    settings, _, _ = stage_env
    request = next(r for r in b.plan_2024(settings) if r["proposed_stage"] == "existing_pilot")
    session = Session()
    with pytest.raises(a.AcquisitionError, match="reused"): b.acquire_stage_b_interval(settings, request, execute=True, session=session)
    assert session.metadata_calls == 0 and not session.observation_dates


def test_small_sequential_batches_resume_at_next_interval(stage_env):
    settings, _, _ = stage_env
    session = Session()
    first = batch(stage_env, 3, session)
    second = batch(stage_env, 2, session)
    assert first["unique_new_intervals_attempted"] == 3 and second["unique_new_intervals_attempted"] == 5
    assert session.observation_dates == ["2024-01-01", "2024-01-06", "2024-01-11", "2024-01-16", "2024-01-21"]
    catalog = b.load_catalog(settings)
    assert len(catalog["intervals"]) == 74 and sum(r["reused_existing_pilot"] for r in catalog["intervals"].values()) == 1
    assert SYNTHETIC_KEY not in json.dumps(catalog)


def test_completed_interval_is_reused_without_metadata_or_observation_calls(stage_env):
    settings, _, _ = stage_env
    batch(stage_env)
    session = Session()
    result = b.acquire_stage_b_interval(settings, b.plan_2024(settings)[0], execute=True, session=session)
    assert result["qc_status"] == "passed" and session.metadata_calls == 0 and not session.observation_dates


def test_transient_retry_counts_attempts_separately_from_unique_intervals(stage_env):
    session = Session(failures=[500, 200])
    result = batch(stage_env, session=session)
    assert result["unique_new_intervals_attempted"] == 1 and result["total_observation_attempt_slots"] == 2
    assert session.observation_dates == ["2024-01-01", "2024-01-01"]


@pytest.mark.parametrize("statuses,expected", [([403], 1), ([429, 429, 429], 3), ([503, 503, 503], 3)])
def test_authentication_rate_limit_and_repeated_server_failure_stop_batch(stage_env, statuses, expected):
    session = Session(failures=statuses)
    result = batch(stage_env, size=10, session=session)
    assert result["status"] == "stopped" and len(session.observation_dates) == expected
    assert len(set(session.observation_dates)) == 1
    with pytest.raises(storage.StoreConflict, match="previous critical"): batch(stage_env)


def test_actual_and_declared_response_limits_cannot_be_bypassed(stage_env):
    settings, _, _ = stage_env
    settings["study"]["maximum_response_bytes"] = 10
    result = batch(stage_env)
    assert result["status"] == "stopped" and result["halt_reason"] == "response_size_limit"


@pytest.mark.parametrize("target", ["raw", "annual_metadata"])
def test_corrupted_raw_or_manifest_blocks_next_request(stage_env, target):
    settings, resolver, _ = stage_env
    batch(stage_env)
    catalog = b.load_catalog(settings)
    first = catalog["intervals"][b.plan_2024(settings)[0]["request_id"]]
    if target == "raw":
        workspace_path(first["raw_file"]).write_bytes(b"synthetic corruption")
    else:
        first["http"] = {"http_status": 500, "content_type": "text/html"}
        storage.atomic_json(resolver(settings["paths"]["output"]) / "acquisition_manifest.json", catalog)
    session = Session()
    with pytest.raises(storage.StoreConflict): batch(stage_env, session=session)
    assert session.metadata_calls == 0 and not session.observation_dates


def test_stage_b_prepared_pair_recovers_locally_after_interruption(stage_env, monkeypatch):
    settings, _, _ = stage_env
    original = storage.promote_immutable
    def interrupted(*args):
        original(*args)
        raise RuntimeError("synthetic interruption after raw placement")
    monkeypatch.setattr(storage, "promote_immutable", interrupted)
    with pytest.raises(RuntimeError): batch(stage_env)
    monkeypatch.setattr(storage, "promote_immutable", original)
    catalog = b.load_catalog(settings)
    entry = catalog["intervals"][b.plan_2024(settings)[0]["request_id"]]
    assert entry["acquisition_status"] == "acquired_positive"
    session = Session()
    b.acquire_stage_b_interval(settings, b.plan_2024(settings)[0], execute=True, session=session)
    assert session.metadata_calls == 0 and not session.observation_dates


def test_protected_asset_content_change_is_detected_before_network(stage_env, tmp_path, monkeypatch):
    settings, resolver, _ = stage_env
    asset = tmp_path / "synthetic_protected_asset"
    asset.write_bytes(b"original")
    from agniscope.provenance import sha256
    protected = {"protected_files": [{"path": str(asset), "sha256": sha256(asset)}]}
    storage.atomic_json(resolver(settings["paths"]["preflight"]), protected)
    asset.write_bytes(b"changed")
    with pytest.raises(storage.StoreConflict, match="Protected"):
        resolver.protected_checker(settings)


def test_global_attempt_ledger_cannot_include_another_year_or_pilot(stage_env):
    settings, resolver, _ = stage_env
    catalog = b.load_catalog(settings, create=True)
    pilot = next(r for r in b.plan_2024(settings) if r["proposed_stage"] == "existing_pilot")
    catalog["attempt_reservations"].append({"request_id": pilot["request_id"], "number": 1})
    storage.atomic_json(resolver(settings["paths"]["output"]) / "acquisition_manifest.json", catalog)
    with pytest.raises(storage.StoreConflict, match="attempt identities"):
        b.load_catalog(settings)


def test_http_date_retry_after_is_bounded_without_logging_header_content():
    from datetime import datetime, timezone, timedelta
    from email.utils import format_datetime
    class DatedLimit:
        def get(self, *args, **kwargs):
            return Response(b"withheld", 429, extra={"Retry-After": format_datetime(datetime.now(timezone.utc)+timedelta(seconds=120))})
    metadata = {}
    with pytest.raises(a.AcquisitionError):
        a._bounded_get("https://synthetic.invalid", 100, session=DatedLimit(), response_metadata=metadata)
    assert 115 <= metadata["retry_after_seconds"] <= 120
