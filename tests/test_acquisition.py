import hashlib
import json
from datetime import date

import pytest
import requests

import agniscope.acquisition as acquisition


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status_code, self.headers = body, status, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        yield self.body


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        assert kwargs["allow_redirects"] is False
        return next(self.responses)


@pytest.fixture
def fake_key(monkeypatch):
    key = "synthetic_key_for_tests_only_123456"
    monkeypatch.setattr(acquisition, "local_map_key", lambda: key)
    return key


def test_missing_credential_prevents_any_network_request(monkeypatch, config):
    monkeypatch.setattr(acquisition, "local_map_key", lambda: None)
    session = FakeSession([])
    result = acquisition.acquire_pilot(config, download=True, session=session)
    assert result["status"] == "authentication_required"
    assert not result["available_dates_verified"]
    assert session.calls == 0


def test_availability_csv_schema_and_endpoint_range(config, fake_key):
    session = FakeSession([FakeResponse(b"data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-01-01\n")])
    result = acquisition.acquire_pilot(config, session=session)
    assert result["available_dates_verified"]
    assert result["proposed_study_endpoints_within_available_range"]
    assert not result["study_period_completeness_verified"]
    assert fake_key not in json.dumps(result)


@pytest.mark.parametrize("days", [0, 6, 31])
def test_rejects_unapproved_request_window(days, config, fake_key):
    config["firms"]["pilot_days"] = days
    with pytest.raises(acquisition.AcquisitionError, match="five days"):
        acquisition.acquire_pilot(config, download=True, session=FakeSession([]))


def test_network_exception_cannot_leak_credential(fake_key):
    class FailedSession:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("secret-url/" + fake_key)
    with pytest.raises(acquisition.AcquisitionError) as caught:
        acquisition._bounded_get("https://example.test/" + fake_key, 100, session=FailedSession())
    assert fake_key not in str(caught.value)
    assert caught.value.__suppress_context__


def test_enforced_response_size_limit():
    with pytest.raises(acquisition.AcquisitionError, match="byte limit"):
        acquisition._bounded_get("https://example.test", 10, session=FakeSession([FakeResponse(b"x" * 11)]))


@pytest.mark.parametrize("version", ["2", "2.0"])
def test_mock_pilot_provenance_and_no_overwrite(tmp_path, monkeypatch, config, fake_key, version):
    # Fictional responses ONLY. All files go to pytest's workspace-local fixture cache.
    original_resolver = acquisition.workspace_path
    def resolver(path, **kwargs):
        if str(path).startswith("data/raw/firms/"):
            return tmp_path / "synthetic_mock_pilot.csv"
        return original_resolver(path, **kwargs)
    monkeypatch.setattr(acquisition, "workspace_path", resolver)
    available = b"data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-01-01\n"
    body = f"latitude,longitude,acq_date,acq_time,satellite,confidence,version\n30,79,2024-04-01,0003,N,n,{version}\n".encode()
    session = FakeSession([FakeResponse(available), FakeResponse(body)])
    result = acquisition.acquire_pilot(config, download=True, session=session)
    assert result["authentic_sample_acquired"]  # simulated branch, never a project empirical result
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / "synthetic_mock_pilot.csv").read_bytes() == body
    assert result["response_validation"]["failed_checks"]["processing_version"] == 0
    metadata_path = tmp_path / "synthetic_mock_pilot.provenance.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["synthetic_test_fixture"] = True
    metadata["origin"] = "SYNTHETIC MOCK OF NASA RESPONSE - TEST ONLY"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert fake_key not in metadata_path.read_text(encoding="utf-8")
    with pytest.raises(acquisition.AcquisitionError, match="already exists"):
        acquisition.acquire_pilot(config, download=True, session=FakeSession([FakeResponse(available)]))


def test_rejects_nrt_in_pilot_response(tmp_path, monkeypatch, config, fake_key):
    original_resolver = acquisition.workspace_path
    monkeypatch.setattr(acquisition, "workspace_path", lambda path, **kwargs: tmp_path / "rejected_synthetic.csv" if str(path).startswith("data/raw/firms/") else original_resolver(path, **kwargs))
    responses = [FakeResponse(b"data_id,min_date,max_date\nVIIRS_SNPP_SP,2012-01-20,2026-01-01\n"), FakeResponse(b"latitude,longitude,acq_date,acq_time,satellite,confidence,version\n30,79,2024-04-01,0003,N,n,2.0NRT\n")]
    with pytest.raises(acquisition.AcquisitionError, match="Standard Processing"):
        acquisition.acquire_pilot(config, download=True, session=FakeSession(responses))
    assert not (tmp_path / "rejected_synthetic.csv").exists()


@pytest.mark.parametrize("date_value,satellite,version,instrument,lon,failed_rule", [
    ("2024-04-06", "N", "2", "VIIRS", "79", "acquisition_date"),
    ("2024-04-01", "N20", "2", "VIIRS", "79", "satellite_source"),
    ("2024-04-01", "N", "2", "MODIS", "79", "instrument"),
    ("2024-04-01", "N", "2NRT", "VIIRS", "79", "processing_version"),
    ("2024-04-01", "N", "2RT", "VIIRS", "79", "processing_version"),
    ("2024-04-01", "N", "2URT", "VIIRS", "79", "processing_version"),
    ("2024-04-01", "N", "", "VIIRS", "79", "processing_version"),
    ("2024-04-01", "N", "unknown", "VIIRS", "79", "processing_version"),
    ("2024-04-01", "N", "2", "VIIRS", "82", "coordinates"),
])
def test_rejection_names_the_independent_failed_check(date_value, satellite, version, instrument, lon, failed_rule):
    body = f"latitude,longitude,acq_date,acq_time,satellite,confidence,version,instrument\n30,{lon},{date_value},0003,{satellite},n,{version},{instrument}\n".encode()
    with pytest.raises(acquisition.AcquisitionError, match=failed_rule + "=1"):
        acquisition._validate_pilot_csv(body, date(2024, 4, 1), 5, [77.5, 28.6, 81.2, 31.6])


@pytest.mark.parametrize("body", [b"", b"Error in processing: No data.", b"<html>API error</html>", b"latitude,longitude\n30,79\n"])
def test_api_errors_or_missing_schema_are_not_empty_observation_datasets(body):
    with pytest.raises(acquisition.AcquisitionError, match="schema"):
        acquisition._validate_pilot_csv(body, date(2024, 4, 1), 5, [77.5, 28.6, 81.2, 31.6])


def test_header_only_result_is_explicitly_zero_rows():
    body = b"latitude,longitude,acq_date,acq_time,satellite,confidence,version\n"
    fields, rows, checks = acquisition._validate_pilot_csv(body, date(2024, 4, 1), 5, [77.5, 28.6, 81.2, 31.6])
    assert not rows
    assert checks["row_count"] == 0
    assert not any(checks["failed_checks"].values())


def test_logged_http_metadata_cannot_echo_credentials(fake_key):
    metadata = {}
    session = FakeSession([FakeResponse(b"ok", headers={"Content-Type": "text/" + fake_key})])
    assert acquisition._bounded_get("https://example.test/" + fake_key, 10, session=session, response_metadata=metadata) == b"ok"
    assert metadata == {"http_status": 200, "content_type": "other"}
    assert fake_key not in json.dumps(metadata)
