"""Credential-safe, size-limited NASA FIRMS pilot acquisition."""

import csv
from datetime import date, timedelta
import hashlib
import io
import json
import os
import re
import time

import requests

from .config import workspace_path
from .firms import REQUIRED_COLUMNS, SNPP_LABELS, classify_processing_version
from .provenance import utc_now, write_json

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"


class AcquisitionError(RuntimeError):
    def __init__(self, message, *, code="acquisition_error", retryable=False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def local_map_key() -> str | None:
    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("NASA_FIRMS_MAP_KEY") or os.environ.get("MAP_KEY")
    secret_file = workspace_path(".env")
    if not key and secret_file.is_file():
        for line in secret_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("FIRMS_MAP_KEY="):
                key = line.strip().split("=", 1)[1].strip().strip("\"'")
                break
    if not key or key == "REPLACE_LOCALLY":
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", key):
        raise AcquisitionError("Local FIRMS credential format is invalid; its value has been withheld.")
    return key


def _bounded_get(url: str, maximum_bytes: int, session=None, *, response_metadata: dict | None = None) -> bytes:
    client = session or requests
    try:
        with client.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as response:
            if response_metadata is not None:
                mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                # Only fixed MIME labels are logged; never arbitrary headers or URLs.
                response_metadata.update(http_status=response.status_code, content_type=mime if mime in {"text/plain", "text/csv", "application/csv", "application/octet-stream", "application/json", "text/html"} else "other")
            if response.status_code != 200:
                retry_after = response.headers.get("Retry-After", "")
                if response_metadata is not None and retry_after:
                    if retry_after.isdigit():
                        seconds = int(retry_after)
                    else:
                        from datetime import datetime, timezone
                        from email.utils import parsedate_to_datetime
                        import math
                        try:
                            seconds = max(0, math.ceil((parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()))
                        except (ValueError, TypeError, OverflowError):
                            seconds = 601  # Unrecognized server instruction: stop rather than guess.
                    response_metadata["retry_after_seconds"] = min(seconds, 601)
                raise AcquisitionError(f"NASA request returned HTTP {response.status_code}; response details withheld.", code=f"http_{response.status_code}", retryable=response.status_code == 429 or 500 <= response.status_code < 600)
            try:
                declared_size = int(response.headers.get("Content-Length", "0"))
            except ValueError:
                declared_size = 0
            if declared_size > maximum_bytes:
                raise AcquisitionError("Response exceeds the approved byte limit; transfer stopped.", code="response_size_limit")
            data = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if len(data) + len(chunk) > maximum_bytes:
                    raise AcquisitionError("Response exceeds the approved byte limit; transfer stopped.", code="response_size_limit")
                data.extend(chunk)
            return bytes(data)
    except AcquisitionError:
        raise
    except requests.RequestException:
        # Requests exceptions may contain the credential-bearing URL. Never chain them.
        raise AcquisitionError("NASA network request failed; sensitive URL and exception details withheld.", code="network_failure", retryable=True) from None


def _availability(body: bytes, source: str) -> dict:
    try:
        rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
        row = next(item for item in rows if item.get("data_id", "").upper() == source)
        start = date.fromisoformat(row["min_date"])
        end = date.fromisoformat(row["max_date"])
    except (UnicodeError, KeyError, ValueError, StopIteration):
        raise AcquisitionError("NASA availability response did not match its documented CSV schema.") from None
    return {"source": source, "min_date": start.isoformat(), "max_date": end.isoformat(), "coverage_is_continuous": "not established by range endpoints"}


def _validate_pilot_csv(body: bytes, start: date, days: int, bbox: list, *, require_compatible=True) -> tuple[list, list, dict]:
    """Validate the response and report independent failure counts, not a compound guess."""
    try:
        reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")), strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(REQUIRED_COLUMNS) - set(fields):
            raise AcquisitionError("Pilot response is not the documented FIRMS CSV schema; nothing saved.")
        rows = list(reader)
        allowed_dates = {str(start + timedelta(days=i)) for i in range(days)}
        failures = {"malformed_field_count": 0, "acquisition_date": 0, "satellite_source": 0, "instrument": 0, "processing_version": 0, "coordinates": 0}
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                failures["malformed_field_count"] += 1
                continue
            failures["acquisition_date"] += row["acq_date"] not in allowed_dates
            failures["satellite_source"] += row["satellite"].strip().upper() not in SNPP_LABELS
            failures["instrument"] += "instrument" in row and row["instrument"].strip().upper() != "VIIRS"
            failures["processing_version"] += classify_processing_version(row["version"]) != "standard_processing"
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                coordinate_ok = bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]
            except (TypeError, ValueError):
                coordinate_ok = False
            failures["coordinates"] += not coordinate_ok
        if any(failures.values()) and require_compatible:
            counts = ", ".join(f"{name}={count}" for name, count in failures.items())
            raise AcquisitionError("Pilot response rejected (Standard Processing required): " + counts + "; nothing saved.")
        return fields, rows, {"row_count": len(rows), "failed_checks": failures, "processing_evidence": "authenticated VIIRS_SNPP_SP request and per-row collection-only version; raw version values preserved"}
    except (UnicodeError, csv.Error):
        raise AcquisitionError("Pilot response could not be safely parsed as UTF-8 CSV; nothing saved.") from None


def acquire_pilot(config: dict, *, download: bool = False, session=None) -> dict:
    firms = config["firms"]
    if firms["api_base"].rstrip("/") != API_BASE or firms["source"] != "VIIRS_SNPP_SP":
        raise AcquisitionError("Only the approved NASA VIIRS_SNPP_SP source is allowed.")
    days = firms["pilot_days"]
    start = date.fromisoformat(firms["pilot_start"])
    bbox = firms["bbox_west_south_east_north"]
    if not isinstance(days, int) or not 1 <= days <= min(5, firms["maximum_days_per_request"]):
        raise AcquisitionError("Pilot must contain between one and five days.")
    if start.year != 2024 or start.month != 4 or (start + timedelta(days=days - 1)).month != 4:
        raise AcquisitionError("The authorized pilot is limited to April 2024.")
    if bbox != [77.5, 28.6, 81.2, 31.6]:
        raise AcquisitionError("Acquisition extent differs from the approved bounded pilot.")
    maximum_bytes = firms["maximum_response_bytes"]
    if not 1 <= maximum_bytes <= 10 * 1024**2:
        raise AcquisitionError("Pilot byte cap must not exceed 10 MiB.")
    report = {"checked_utc": utc_now(), "source": firms["source"], "pilot_start": start.isoformat(), "pilot_end": (start + timedelta(days=days - 1)).isoformat(), "days": days, "bbox_west_south_east_north": bbox, "maximum_response_bytes": maximum_bytes, "authentic_sample_acquired": False, "study_period_completeness_verified": False}
    key = local_map_key()
    if key is None:
        report.update(status="authentication_required", available_dates_verified=False, reason="No local FIRMS MAP_KEY configured. No authenticated request or data download was attempted.")
        return report
    source = firms["source"]
    availability_http = {}
    availability_bytes = _bounded_get(f"{API_BASE}/data_availability/csv/{key}/{source}", 65536, session=session, response_metadata=availability_http)
    available = _availability(availability_bytes, source)
    report.update(status="availability_verified", available_dates_verified=True, availability=available, availability_http=availability_http)
    period_start, period_end = date.fromisoformat(available["min_date"]), date.fromisoformat(available["max_date"])
    report["proposed_study_endpoints_within_available_range"] = period_start <= date.fromisoformat(config["project"]["study_start"]) and period_end >= date.fromisoformat(config["project"]["study_end"])
    if not download:
        return report
    if not period_start <= start <= start + timedelta(days=days - 1) <= period_end:
        raise AcquisitionError("The pilot falls outside the currently available Standard Processing range.")
    destination = workspace_path(f"data/raw/firms/VIIRS_SNPP_SP_{start.isoformat()}_{days}days_uttarakhand_bbox.csv", writable=True)
    metadata = destination.with_suffix(".provenance.json")
    if destination.exists() or metadata.exists():
        raise AcquisitionError("Pilot destination already exists; refusing to overwrite or reacquire it.")
    coordinates = ",".join(str(value) for value in bbox)
    pilot_http = {}
    body = _bounded_get(f"{API_BASE}/area/csv/{key}/{source}/{coordinates}/{days}/{start.isoformat()}", maximum_bytes, session=session, response_metadata=pilot_http)
    fields, rows, checks = _validate_pilot_csv(body, start, days, bbox)
    provenance = {"origin": "NASA FIRMS API", "source": source, "retrieved_utc": utc_now(), "request": {"start_date": start.isoformat(), "days": days, "bbox": bbox, "credential_included": False}, "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body), "returned_rows": len(rows), "columns": fields, "availability_at_download": available, "http": pilot_http, "response_validation": checks}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(body)
    write_json(metadata, provenance)
    report.update(status="pilot_acquired", authentic_sample_acquired=True, raw_file=destination.relative_to(workspace_path(".")).as_posix(), provenance_file=metadata.relative_to(workspace_path(".")).as_posix(), returned_rows=len(rows), sha256=provenance["sha256"], size_bytes=len(body), pilot_http=pilot_http, response_validation=checks)
    return report


# Historical execution is a separate branch; acquire_pilot's original fence remains.
def historical_spec(settings: dict, *, execute=False) -> dict:
    """Validate configured limits before resolving credentials or making requests."""
    request = settings["request"]
    try:
        start, end = date.fromisoformat(request["start"]), date.fromisoformat(request["end"])
    except (ValueError, TypeError):
        raise AcquisitionError("Invalid configured UTC date interval.", code="scope_rejected") from None
    days = (end - start).days + 1
    if not 1 <= days <= 5 or request["maximum_days"] != 5:
        raise AcquisitionError("Historical requests must contain one to five days.", code="scope_rejected")
    if type(request["maximum_response_bytes"]) is not int or not 0 < request["maximum_response_bytes"] <= 10 * 1024**2:
        raise AcquisitionError("Historical response cap must not exceed 10 MiB.", code="scope_rejected")
    if request["source"] != "VIIRS_SNPP_SP" or request["bbox"] != [77.5, 28.6, 81.2, 31.6]:
        raise AcquisitionError("Historical source or acquisition extent is not approved.", code="scope_rejected")
    if request["allowed_collection_values"] != ["2", "2.0"]:
        raise AcquisitionError("Collection policy changed; scientific review required.", code="scope_rejected")
    retry = settings["retry"]
    if (type(retry["maximum_attempts"]) is not int or not 1 <= retry["maximum_attempts"] <= 3
            or retry["backoff_seconds"] != [5, 20] or retry["minimum_spacing_seconds"] != 2
            or retry["transaction_reserve_fraction"] != 0.20):
        raise AcquisitionError("Only the declared bounded Stage A retry policy is approved.", code="scope_rejected")
    if execute and (request["stage"], start, end) != ("A", date(2023, 4, 1), date(2023, 4, 5)):
        raise AcquisitionError("Execution allowlist permits only Stage A April 1-5, 2023.", code="scope_rejected")
    if any(settings["scope"].values()):
        raise AcquisitionError("Stage B/C or model execution is not authorized.", code="scope_rejected")
    return {"source": request["source"], "start": str(start), "end": str(end), "days": days,
            "bbox": request["bbox"], "allowed_collection_values": request["allowed_collection_values"]}


def _historical_paths(settings):
    expected_raw = "data/raw/firms/VIIRS_SNPP_SP_2023-04-01_5days_uttarakhand_bbox.csv"
    if settings["paths"]["raw"] != expected_raw or settings["paths"]["store"] != "outputs/tables/phase2b_stage_a":
        raise AcquisitionError("Stage A artifact destinations changed.", code="scope_rejected")
    raw = workspace_path(expected_raw, writable=True)
    store = workspace_path(settings["paths"]["store"], writable=True)
    return raw, raw.with_suffix(".provenance.json"), store / "acquisition_manifest.json"


def _read_manifest(path):
    from .acquisition_store import StoreConflict
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, OSError):
        raise StoreConflict("Acquisition manifest cannot be read; evidence preserved.") from None


def _transaction_status(key, session=None):
    http = {}
    body = _bounded_get("https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=" + key,
                        65536, session=session, response_metadata=http)
    try:
        parsed = json.loads(body)
        limit, current = parsed["transaction_limit"], parsed["current_transactions"]
        if type(limit) is not int or type(current) is not int or not 0 <= current <= limit or limit <= 0:
            raise ValueError
        if parsed["transaction_interval"] != "10 minutes":
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise AcquisitionError("Transaction status schema is unrecognized.", code="quota_unknown") from None
    return {"checked_utc": utc_now(), "http": http, "transaction_limit": limit,
            "current_transactions": current, "transaction_interval": "10 minutes"}


def inspect_historical_csv(body, spec):
    """Reuse pilot CSV parsing/checks; retain parseable incompatible rows as QC failed."""
    try:
        fields, rows, checks = _validate_pilot_csv(body, date.fromisoformat(spec["start"]), spec["days"], spec["bbox"], require_compatible=False)
    except AcquisitionError:
        raise AcquisitionError("Response is not a valid FIRMS observation CSV; body withheld.", code="response_format") from None
    checks["failed_checks"]["collection"] = sum(str(row.get("version") or "").strip() not in spec["allowed_collection_values"] for row in rows)
    checks["failed_checks"]["reserved_source_columns"] = sum(name.startswith("agniscope_") for name in fields)
    checks["collection_values"] = sorted({str(row.get("version") or "") for row in rows})
    checks["compatible"] = not any(checks["failed_checks"].values())
    return fields, rows, checks


def _verified_historical_pair(manifest, spec, raw, provenance):
    from .acquisition_store import StoreConflict
    from .provenance import sha256
    if manifest.get("request") != spec:
        raise StoreConflict("Manifest request differs from the approved interval.")
    root = workspace_path(".")
    if manifest.get("raw_file") != raw.relative_to(root).as_posix() or manifest.get("provenance_file") != provenance.relative_to(root).as_posix():
        raise StoreConflict("Manifest artifact paths differ from the approved destinations.")
    if not 0 < manifest.get("maximum_response_bytes", 0) <= 10 * 1024**2 or manifest.get("size_bytes", 0) > manifest["maximum_response_bytes"]:
        raise StoreConflict("Manifest size evidence violates the response cap.")
    for path, expected_hash, expected_size in ((raw, manifest.get("sha256"), manifest.get("size_bytes")),
                                               (provenance, manifest.get("provenance_sha256"), manifest.get("provenance_size_bytes"))):
        if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_hash:
            raise StoreConflict("Raw/provenance pair is incomplete or conflicts with recorded hashes.")
    metadata = _read_manifest(provenance)
    if metadata.get("sha256") != manifest["sha256"] or metadata.get("request") != spec:
        raise StoreConflict("Provenance does not identify the approved raw response.")
    for field in ("request_id", "size_bytes", "http", "retrieved_utc", "response_validation"):
        if metadata.get(field) != manifest.get(field):
            raise StoreConflict("Manifest metadata conflicts with immutable provenance.")
    if metadata.get("returned_rows") != manifest.get("raw_observation_count") or metadata.get("source") != spec["source"]:
        raise StoreConflict("Manifest source/count evidence conflicts with immutable provenance.")
    _, rows, checks = inspect_historical_csv(raw.read_bytes(), spec)
    if len(rows) != manifest.get("raw_observation_count") or checks != manifest.get("response_validation"):
        raise StoreConflict("Manifest count/collection/QC evidence conflicts with raw CSV.")
    if manifest.get("qc_summary_file") and sha256(workspace_path(manifest["qc_summary_file"])) != manifest.get("qc_summary_sha256"):
        raise StoreConflict("Scientific QC summary conflicts with the recorded hash.")


def _finish_prepared(manifest, manifest_path, raw, provenance, spec, *, stage="A"):
    from .acquisition_store import StoreConflict, atomic_json, promote_immutable
    from .provenance import sha256
    for final, field, digest in ((raw, "raw_staged", "sha256"), (provenance, "provenance_staged", "provenance_sha256")):
        staged = workspace_path(manifest[field], writable=True)
        # Recovery can touch only the fixed sibling staging names, never arbitrary paths.
        if staged != final.with_name(final.name + f".stage_{stage.lower()}_staged"):
            raise StoreConflict("Manifest staging destination conflicts with the authorized stage.")
        if final.exists():
            if sha256(final) != manifest[digest]:
                raise StoreConflict("Existing immutable artifact conflicts with prepared hash.")
        else:
            if not staged.is_file() or sha256(staged) != manifest[digest]:
                raise StoreConflict("Prepared artifact is missing or has a conflicting hash.")
            promote_immutable(staged, final)
    _verified_historical_pair(manifest, spec, raw, provenance)
    compatible = manifest["response_validation"]["compatible"]
    manifest["acquisition_status"] = ("acquired_positive" if manifest["raw_observation_count"] else "acquired_zero") if compatible else "retrieved_qc_failed"
    manifest["qc_status"] = "response_validated_pending_scientific_qc" if compatible else "response_compatibility_failed"
    manifest["failure_category"] = None if compatible else "response_compatibility_failed"
    manifest["events"].append({"event": "pair_committed", "at_utc": utc_now()})
    atomic_json(manifest_path, manifest)
    for field in ("raw_staged", "provenance_staged"):
        workspace_path(manifest[field], writable=True).unlink(missing_ok=True)
    return manifest


def acquire_historical(settings: dict, *, execute_stage_a=False, session=None, sleeper=time.sleep) -> dict:
    """Exactly one allowlisted logical interval, at most three total persisted attempts.

    Completed reruns verify local evidence and make no network call, including metadata.
    """
    from .acquisition_store import StoreConflict, atomic_json, staged_bytes, json_bytes, store_lock
    from .planning import request_id
    from .provenance import sha256
    spec = historical_spec(settings, execute=execute_stage_a)
    if not execute_stage_a:
        return {"acquisition_status": "not_attempted", "mode": "dry_run", "request": spec,
                "observation_requests": 0, "satellite_completeness": "unverified"}
    raw, provenance, manifest_path = _historical_paths(settings)
    return _acquire_interval(settings, spec, raw, provenance, manifest_path, stage="A", session=session, sleeper=sleeper)


def _acquire_interval(settings, spec, raw, provenance, manifest_path, *, stage, session=None, sleeper=time.sleep, before_attempt=None):
    """Shared durable interval engine. Public stage wrappers enforce their allowlists."""
    from .acquisition_store import StoreConflict, atomic_json, staged_bytes, json_bytes, store_lock
    from .planning import request_id
    from .provenance import sha256
    root = workspace_path(".")
    relative = lambda path: path.relative_to(root).as_posix()
    expected_id = request_id(spec["source"], spec["bbox"], date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"]))
    with store_lock(manifest_path.parent):
        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
            if (manifest.get("request"), manifest.get("request_id"), manifest.get("raw_file")) != (spec, expected_id, relative(raw)):
                raise StoreConflict("Manifest scope or artifact identity conflicts; no request attempted.")
            if manifest["acquisition_status"] == "prepared":
                return _finish_prepared(manifest, manifest_path, raw, provenance, spec, stage=stage)
            if manifest["acquisition_status"] in {"acquired_positive", "acquired_zero", "retrieved_qc_failed"}:
                _verified_historical_pair(manifest, spec, raw, provenance)
                return manifest
        else:
            if raw.exists() or provenance.exists():
                raise StoreConflict("An untracked interval artifact exists; refusing overwrite or reacquisition.")
            manifest = {"schema_version": 1, "stage": stage, "request_id": expected_id, "request": spec,
                        "raw_file": relative(raw), "provenance_file": relative(provenance),
                        "sha256": None, "size_bytes": None, "raw_observation_count": None,
                        "acquisition_status": "not_attempted", "qc_status": "not_run", "failure_category": None,
                        "satellite_completeness": "unverified", "attempts": [], "events": [],
                        "maximum_response_bytes": settings["request"]["maximum_response_bytes"],
                        "created_utc": utc_now()}
            atomic_json(manifest_path, manifest)
        if raw.exists() or provenance.exists():
            raise StoreConflict("Artifacts exist without a committed/prepared state; no network call attempted.")
        if any(p.with_name(p.name + f".stage_{stage.lower()}_staged").exists() for p in (raw, provenance)):
            raise StoreConflict("Untracked staging bytes exist; reconcile them before any network request.")
        if manifest["acquisition_status"] == "attempting":
            manifest["acquisition_status"] = "attempted_failed"
            manifest["failure_category"] = "interrupted_attempt_outcome_unknown"
            manifest["events"].append({"event": "interrupted_attempt_detected", "at_utc": utc_now()})
            atomic_json(manifest_path, manifest)
        limit = settings["retry"]["maximum_attempts"]
        if len(manifest["attempts"]) >= limit or manifest.get("permanent_failure"):
            return manifest
        key = local_map_key()
        if key is None:
            raise AcquisitionError("A local NASA credential is required; no observation request attempted.", code="authentication_required")
        availability_http = {}
        try:
            available = _availability(_bounded_get(f"{API_BASE}/data_availability/csv/{key}/{spec['source']}", 65536, session=session, response_metadata=availability_http), spec["source"])
            if not available["min_date"] <= spec["start"] <= spec["end"] <= available["max_date"]:
                raise AcquisitionError("Stage A is outside the source availability range.", code="source_unavailable")
            manifest["availability_at_download"] = available
            manifest["availability_http"] = availability_http
        except AcquisitionError:
            manifest["failure_category"] = "availability_check_failed"
            atomic_json(manifest_path, manifest)
            raise AcquisitionError("Availability check failed; no observation request attempted.", code="availability_check_failed") from None
        while len(manifest["attempts"]) < limit:
            try:
                quota = _transaction_status(key, session)
            except AcquisitionError:
                manifest["failure_category"] = "quota_check_failed"
                atomic_json(manifest_path, manifest)
                raise AcquisitionError("Quota check failed; no next observation request attempted.", code="quota_check_failed") from None
            manifest.setdefault("quota_snapshots", []).append(quota)
            if quota["transaction_limit"] - quota["current_transactions"] < max(1000, quota["transaction_limit"] * 0.20):
                manifest["failure_category"] = "transaction_reserve"
                atomic_json(manifest_path, manifest)
                raise AcquisitionError("Transaction reserve is insufficient; no next observation attempt.", code="transaction_reserve")
            number = len(manifest["attempts"]) + 1
            delay = 2 if number == 1 else settings["retry"]["backoff_seconds"][number - 2]
            delay = max(delay, manifest.get("retry_after_seconds", 0))
            sleeper(delay)
            attempt = {"number": number, "started_utc": utc_now()}
            manifest["attempts"].append(attempt)
            manifest["acquisition_status"] = "attempting"
            manifest["events"].append({"event": "attempt_started", **attempt})
            atomic_json(manifest_path, manifest)  # Consume the slot BEFORE the HTTP call.
            if before_attempt is not None:
                before_attempt(manifest)
            http = {}
            try:
                coordinates = ",".join(map(str, spec["bbox"]))
                body = _bounded_get(f"{API_BASE}/area/csv/{key}/{spec['source']}/{coordinates}/{spec['days']}/{spec['start']}",
                                    settings["request"]["maximum_response_bytes"], session=session, response_metadata=http)
                if key.encode() in body:
                    raise AcquisitionError("Response contains credential material; body withheld.", code="credential_echo")
                if http.get("content_type") not in {"text/plain", "text/csv", "application/csv", "application/octet-stream"}:
                    raise AcquisitionError("Response content type is incompatible with observation CSV.", code="response_content_type")
                fields, rows, checks = inspect_historical_csv(body, spec)
            except AcquisitionError as error:
                manifest["acquisition_status"] = "attempted_failed"
                manifest["failure_category"] = error.code
                manifest["http"] = http
                manifest["permanent_failure"] = not error.retryable or http.get("retry_after_seconds", 0) > 600
                manifest["retry_after_seconds"] = http.get("retry_after_seconds", 0)
                manifest["events"].append({"event": "attempt_failed", "number": number, "at_utc": utc_now(), "failure_category": error.code, "http": http})
                atomic_json(manifest_path, manifest)
                if manifest["permanent_failure"]:
                    break
                continue
            digest = hashlib.sha256(body).hexdigest()
            retrieved = utc_now()
            metadata = {"origin": "NASA FIRMS API", "source": spec["source"], "request": spec,
                        "request_id": expected_id, "retrieved_utc": retrieved, "sha256": digest,
                        "size_bytes": len(body), "returned_rows": len(rows), "columns": fields,
                        "http": http, "availability_at_download": available, "response_validation": checks,
                        "satellite_completeness": "unverified", "credential_included": False}
            raw_staged = raw.with_name(raw.name + f".stage_{stage.lower()}_staged")
            provenance_staged = provenance.with_name(provenance.name + f".stage_{stage.lower()}_staged")
            # Unexpected orphan stages are evidence, never overwritten or reused blindly.
            if raw_staged.exists() or provenance_staged.exists():
                raise StoreConflict("Untracked staging files require reconciliation; no evidence overwritten.")
            staged_bytes(raw_staged, body)
            staged_bytes(provenance_staged, json_bytes(metadata))
            manifest.update(acquisition_status="prepared", sha256=digest, size_bytes=len(body),
                            raw_observation_count=len(rows), retrieved_utc=retrieved, http=http,
                            collection_values=checks["collection_values"], response_validation=checks,
                            provenance_sha256=sha256(provenance_staged), provenance_size_bytes=provenance_staged.stat().st_size,
                            raw_staged=relative(raw_staged), provenance_staged=relative(provenance_staged))
            manifest["events"].append({"event": "response_prepared", "number": number, "at_utc": utc_now()})
            atomic_json(manifest_path, manifest)
            return _finish_prepared(manifest, manifest_path, raw, provenance, spec, stage=stage)
        return manifest


def record_historical_qc(settings, qc_summary_path, *, passed):
    """Update only the mutable manifest; immutable source provenance stays untouched."""
    from .acquisition_store import StoreConflict, atomic_json, store_lock
    from .provenance import sha256
    spec = historical_spec(settings, execute=True)
    raw, provenance, path = _historical_paths(settings)
    return _record_interval_qc(spec, raw, provenance, path, qc_summary_path, passed=passed)


def _record_interval_qc(spec, raw, provenance, path, qc_summary_path, *, passed):
    from .acquisition_store import StoreConflict, atomic_json, store_lock
    from .provenance import sha256
    with store_lock(path.parent):
        manifest = _read_manifest(path)
        _verified_historical_pair(manifest, spec, raw, provenance)
        if not manifest["response_validation"]["compatible"]:
            raise StoreConflict("Incompatible response cannot be marked as passing scientific QC.")
        target = workspace_path(qc_summary_path)
        manifest["qc_status"] = "passed" if passed else "failed"
        manifest["qc_summary_file"] = target.relative_to(workspace_path(".")).as_posix()
        manifest["qc_summary_sha256"] = sha256(target)
        if not passed:
            manifest["acquisition_status"] = "retrieved_qc_failed"
            manifest["failure_category"] = "scientific_qc_failed"
        manifest["events"].append({"event": "scientific_qc_recorded", "passed": passed, "at_utc": utc_now()})
        atomic_json(path, manifest)
        return manifest
