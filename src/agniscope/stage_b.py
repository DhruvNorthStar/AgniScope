"""Explicit annual allowlists and bounded sequential batches around the shared engine."""

from copy import deepcopy
from datetime import date, timedelta
from functools import lru_cache
import json
import time

from . import acquisition as a
from .acquisition_store import StoreConflict, atomic_json, store_lock
from .config import load_config, workspace_path
from .planning import build_request_plan, dates
from .provenance import sha256, utc_now, verify_sources
from .resources import resource_snapshot


SUCCESS = {"acquired_positive", "acquired_zero"}
SYNC_FIELDS = ("acquisition_status", "raw_file", "provenance_file", "sha256", "size_bytes", "raw_observation_count",
               "http", "collection_values", "qc_status", "failure_category", "retrieved_utc", "qc_summary_file", "qc_summary_sha256")


@lru_cache(maxsize=7)
def _canonical_plan_json(year=2024):
    # The unchanged protocol is hash-verified at every batch boundary.
    return json.dumps([r for r in build_request_plan(load_config(workspace_path("configs/historical_protocol.toml"))) if r["year"] == year])


def stage_profile(settings):
    stage = settings["study"]["stage"]
    if stage == "B":
        return {"stage": "B", "year": 2024, "days": 366, "intervals": 74, "pilot": True, "reuse_stage": "existing_pilot", "new_intervals": 73, "planned_stage": "B", "directory": "phase2b_stage_b"}
    if stage == "C1":
        return {"stage": "C1", "year": 2025, "days": 365, "intervals": 73, "pilot": False, "reuse_stage": None, "new_intervals": 73, "planned_stage": "C", "directory": "phase2b_stage_c1_2025"}
    if stage == "C2":
        return {"stage": "C2", "year": 2023, "days": 365, "intervals": 73, "pilot": True, "reuse_stage": "A", "new_intervals": 72, "planned_stage": "C", "directory": "phase2b_stage_c2_2023"}
    if stage == "C3":
        return {"stage": "C3", "year": 2022, "days": 365, "intervals": 73, "pilot": False, "reuse_stage": None, "new_intervals": 73, "planned_stage": "C", "directory": "phase2b_stage_c3_2022"}
    if stage in ("C4", "C5"):
        year, days, intervals = (2021, 365, 73) if stage == "C4" else (2020, 366, 74)
        return {"stage": stage, "year": year, "days": days, "intervals": intervals, "pilot": False, "reuse_stage": None,
                "new_intervals": intervals, "planned_stage": "C", "directory": f"phase2b_stage_{stage.lower()}_{year}"}
    if stage == "C6":
        return {"stage": "C6", "year": 2019, "days": 365, "intervals": 73, "pilot": False, "reuse_stage": None, "new_intervals": 73, "planned_stage": "C", "directory": "phase2b_stage_c6_2019"}
    raise a.AcquisitionError("Only explicitly configured historical annual profiles are supported.", code="scope_rejected")


def reused_interval(settings, request):
    return request["proposed_stage"] == stage_profile(settings)["reuse_stage"]


def plan_2024(settings):
    if settings["study"]["stage"] != "B":
        raise a.AcquisitionError("Stage B accepts only 2024.", code="scope_rejected")
    return approved_plan(settings)


def plan_2025(settings):
    if settings["study"]["stage"] != "C1":
        raise a.AcquisitionError("Stage C1 accepts only 2025.", code="scope_rejected")
    return approved_plan(settings)


def plan_2023(settings):
    if settings["study"]["stage"] != "C2":
        raise a.AcquisitionError("Stage C2 accepts only 2023.", code="scope_rejected")
    return approved_plan(settings)


def plan_2022(settings):
    if settings["study"]["stage"] != "C3":
        raise a.AcquisitionError("Stage C3 accepts only 2022.", code="scope_rejected")
    return approved_plan(settings)


def plan_c4_c5(settings):
    if settings["study"]["stage"] not in ("C4", "C5"):
        raise a.AcquisitionError("Joint mode permits only C4/2021 and C5/2020.", code="scope_rejected")
    return approved_plan(settings)


def plan_2019(settings):
    if settings["study"]["stage"] != "C6":
        raise a.AcquisitionError("Stage C6 accepts only 2019.", code="scope_rejected")
    return approved_plan(settings)


def approved_plan(settings):
    profile = stage_profile(settings)
    year, stage = profile["year"], profile["stage"]
    study, batch = settings["study"], settings["batch"]
    if (study["stage"], study["source"], study["start"], study["end"], study["bbox"]) != (
            stage, "VIIRS_SNPP_SP", f"{year}-01-01", f"{year}-12-31", [77.5, 28.6, 81.2, 31.6]):
        raise a.AcquisitionError("Dates differ from the explicitly approved annual profile.", code="scope_rejected")
    if (study["maximum_new_intervals"], study["maximum_logical_intervals"], batch["maximum_new_intervals"], batch["concurrency"]) != (profile["new_intervals"], profile["intervals"], 10, 1):
        raise a.AcquisitionError("Stage B interval or sequential batch limits changed.", code="scope_rejected")
    if any(settings["scope"].values()) or batch["maximum_total_attempts"] != 3 * profile["new_intervals"]:
        raise a.AcquisitionError("Stage C, another year or unrestricted attempts are not authorized.", code="scope_rejected")
    if study["maximum_days"] != 5 or not 0 < study["maximum_response_bytes"] <= 10485760 or study["allowed_collection_values"] != ["2", "2.0"]:
        raise a.AcquisitionError("Request size/duration or collection policy changed.", code="scope_rejected")
    directory = profile["directory"]
    expected_paths = {"output": f"outputs/tables/{directory}", "derived": f"data/interim/{directory}",
                      "annual": f"data/processed/firms_{year}", "preflight": f"outputs/tables/{directory}/preflight.json"}
    if settings["paths"] != expected_paths:
        raise a.AcquisitionError("Stage B output destinations changed.", code="scope_rejected")
    plan = json.loads(_canonical_plan_json(year))
    independent = []
    segments = [(date(2024, 1, 1), date(2024, 3, 31)), (date(2024, 4, 1), date(2024, 4, 5)), (date(2024, 4, 6), date(2024, 12, 31))] if year == 2024 else [(date(year, 1, 1), date(year, 12, 31))]
    for first, last in segments:
        cursor = first
        while cursor <= last:
            end = min(cursor + timedelta(days=4), last)
            independent.append((str(cursor), str(end)))
            cursor = end + timedelta(days=1)
    if [(r["start_utc_date"], r["end_utc_date"]) for r in plan] != independent:
        raise StoreConflict("Independent 2024 plan disagrees with the approved historical protocol.")
    expanded = [day for r in plan for day in dates(date.fromisoformat(r["start_utc_date"]), date.fromisoformat(r["end_utc_date"]))]
    if len(plan) != profile["intervals"] or expanded != list(dates(date(year, 1, 1), date(year, 12, 31))) or sum(r["proposed_stage"] == profile["planned_stage"] for r in plan) != profile["new_intervals"]:
        raise StoreConflict("2024 dates/intervals do not reconcile; no scope adjustment allowed.")
    return plan


def interval_spec(settings, request):
    plan = {r["request_id"]: r for r in approved_plan(settings)}
    if request != plan.get(request.get("request_id")):
        raise a.AcquisitionError("Interval is outside the exact canonical Stage B plan.", code="scope_rejected")
    temporary = {"request": {**settings["study"], "start": request["start_utc_date"], "end": request["end_utc_date"]},
                 "retry": settings["retry"], "scope": settings["scope"]}
    return a.historical_spec(temporary, execute=False)


def interval_paths(settings, request):
    raw = workspace_path(f"data/raw/firms/VIIRS_SNPP_SP_{request['start_utc_date']}_{request['days']}days_uttarakhand_bbox.csv", writable=True)
    output = workspace_path(settings["paths"]["output"], writable=True)
    # Full identities remain in the manifest; short directory names avoid Windows path limits.
    return raw, raw.with_suffix(".provenance.json"), output / "intervals" / ("r_" + request["request_id"][6:22]) / "acquisition_manifest.json"


def acquire_stage_b_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_2024(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_stage_c1_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_2025(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_stage_c2_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_2023(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_stage_c3_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_2022(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_c4_c5_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_c4_c5(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_stage_c6_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    plan_2019(settings)
    return acquire_approved_interval(settings, request, execute=execute, session=session, sleeper=sleeper, before_attempt=before_attempt)


def acquire_approved_interval(settings, request, *, execute=False, session=None, sleeper=time.sleep, before_attempt=None):
    profile = stage_profile(settings)
    spec = interval_spec(settings, request)
    if request["proposed_stage"] != profile["planned_stage"]:
        raise a.AcquisitionError("The April pilot must be reused locally, never downloaded.", code="pilot_reuse_required")
    if not execute:
        return {"mode": "dry_run", "request": spec, "observation_requests": 0}
    if profile["stage"] in ("C4", "C5"):
        from .paired_acquisition import require_authorization
        require_authorization(settings)
    raw, provenance, path = interval_paths(settings, request)
    temporary = {"request": settings["study"], "retry": settings["retry"]}
    return a._acquire_interval(temporary, spec, raw, provenance, path, stage=profile["stage"], session=session, sleeper=sleeper, before_attempt=before_attempt)


def protected_integrity(settings):
    snapshot = json.loads(workspace_path(settings["paths"]["preflight"]).read_text(encoding="utf-8"))
    changed = [r["path"] for r in snapshot["protected_files"] if not workspace_path(r["path"]).is_file() or sha256(workspace_path(r["path"])) != r["sha256"]]
    allowed = {"src/agniscope/acquisition.py"} if stage_profile(settings)["stage"] == "B" else {"src/agniscope/stage_b.py", "src/agniscope/stage_b_qc.py", "scripts/verify_phase2b_stage_b.py"}
    if set(changed) - allowed:
        raise StoreConflict("Protected original data or prior scientific outputs changed.")
    sources = verify_sources(load_config())
    if not sources["all_unchanged"] or len(sources["files"]) != 14:
        raise StoreConflict("Original GIS source inventory failed verification.")
    return {"protected_existing_files": len(snapshot["protected_files"]), "allowed_code_changes": changed, "original_gis_unchanged": 14}


def pilot_entry(settings, request):
    if stage_profile(settings)["stage"] == "C2":
        return stage_a_entry(settings, request)
    spec = interval_spec(settings, request)
    raw, provenance, _ = interval_paths(settings, request)
    original = a._read_manifest(provenance)
    recorded = original.get("request", {})
    if (original.get("source"), recorded.get("start_date"), recorded.get("days"), recorded.get("bbox")) != ("VIIRS_SNPP_SP", "2024-04-01", 5, spec["bbox"]):
        raise StoreConflict("Preserved pilot provenance scope mismatch.")
    expected = load_config()["dbscan_pilot"]["raw_sha256"]
    if sha256(raw) != expected or original.get("sha256") != expected or raw.stat().st_size != original.get("size_bytes"):
        raise StoreConflict("Preserved 2024 pilot hash/size mismatch.")
    _, rows, checks = a.inspect_historical_csv(raw.read_bytes(), spec)
    if len(rows) != 613 or not checks["compatible"]:
        raise StoreConflict("Preserved pilot row/processing evidence changed.")
    return {"request_id": request["request_id"], "reused_existing_pilot": True, "request": spec,
            "raw_file": raw.relative_to(workspace_path(".")).as_posix(), "provenance_file": provenance.relative_to(workspace_path(".")).as_posix(),
            "sha256": expected, "size_bytes": raw.stat().st_size, "provenance_sha256": sha256(provenance),
            "raw_observation_count": len(rows), "http": original["http"], "collection_values": checks["collection_values"],
            "retrieved_utc": original["retrieved_utc"], "acquisition_status": "acquired_positive", "qc_status": "existing_pilot_verified",
            "failure_category": None, "satellite_completeness": "unverified", "attempt_count_stage_b": 0}


def stage_a_entry(settings, request):
    """Reference the original Stage A evidence without changing its manifest or source."""
    if not reused_interval(settings, request):
        raise StoreConflict("Only the preserved Stage A interval may be referenced.")
    spec = interval_spec(settings, request)
    raw, provenance, _ = interval_paths(settings, request)
    original_path = workspace_path("outputs/tables/phase2b_stage_a/acquisition_manifest.json")
    original = a._read_manifest(original_path)
    if original.get("stage") != "A" or original.get("request_id") != request["request_id"] or original.get("qc_status") != "passed" or original.get("acquisition_status") != "acquired_positive":
        raise StoreConflict("Original Stage A acquisition/QC evidence is incompatible.")
    a._verified_historical_pair(original, spec, raw, provenance)
    if original["sha256"] != "91323877473a523406cf0a8e2bea32847d1dd63db30846b87c7333b27c9d1cdc" or original["raw_observation_count"] != 35:
        raise StoreConflict("Preserved Stage A hash/count mismatch.")
    return {"request_id": request["request_id"], "reused_existing_pilot": True, "reused_existing_stage_a": True, "request": spec,
            **{key: original[key] for key in ("raw_file", "provenance_file", "sha256", "size_bytes", "provenance_sha256", "raw_observation_count", "http", "collection_values", "retrieved_utc", "acquisition_status")},
            "qc_status": "existing_stage_a_verified", "failure_category": None, "satellite_completeness": "unverified", "attempt_count_stage_b": 0,
            "source_manifest_file": original_path.relative_to(workspace_path(".")).as_posix(), "source_manifest_sha256": sha256(original_path)}


def _synchronize(settings, catalog, plan):
    profile = stage_profile(settings)
    if catalog.get("stage") != profile["stage"] or set(catalog["intervals"]) != {r["request_id"] for r in plan}:
        raise StoreConflict("Annual manifest interval identities conflict with the approved plan.")
    reservations = {(r["request_id"], r["number"]) for r in catalog["attempt_reservations"]}
    authorized = {r["request_id"] for r in plan if r["proposed_stage"] == profile["planned_stage"]}
    if len(reservations) != len(catalog["attempt_reservations"]) or any(request_id not in authorized or type(number) is not int or not 1 <= number <= 3 for request_id, number in reservations):
        raise StoreConflict("Global attempt identities are duplicated or outside the authorized plan.")
    for request in plan:
        entry = catalog["intervals"][request["request_id"]]
        if entry.get("request") != interval_spec(settings, request):
            raise StoreConflict("Annual manifest interval scope changed.")
        if reused_interval(settings, request):
            reference = pilot_entry(settings, request)
            keys = ("sha256", "size_bytes", "provenance_sha256", "raw_file", "provenance_file", "raw_observation_count")
            if profile["stage"] == "C2":
                keys += ("source_manifest_file", "source_manifest_sha256", "http", "collection_values", "retrieved_utc", "acquisition_status", "reused_existing_stage_a", "reused_existing_pilot", "attempt_count_stage_b")
            for key in keys:
                if entry.get(key) != reference[key]:
                    raise StoreConflict("Annual manifest pilot reference conflicts with original evidence.")
            if entry.get("qc_summary_file") and sha256(workspace_path(entry["qc_summary_file"])) != entry.get("qc_summary_sha256"):
                raise StoreConflict("Pilot's new Stage B QC summary hash changed.")
            continue
        raw, provenance, path = interval_paths(settings, request)
        if not path.exists():
            if raw.exists() or provenance.exists() or entry.get("sha256") or entry.get("attempt_count_stage_b", 0) or entry.get("manifest_file"):
                raise StoreConflict("Missing interval manifest or untracked raw file; no reacquisition allowed.")
            continue
        manifest = a._read_manifest(path)
        spec = interval_spec(settings, request)
        if manifest.get("request") != spec or manifest.get("request_id") != request["request_id"] or manifest.get("stage") != profile["stage"]:
            raise StoreConflict("Per-interval manifest scope mismatch.")
        if manifest["acquisition_status"] == "prepared":
            manifest = a._finish_prepared(manifest, path, raw, provenance, spec, stage=profile["stage"])
        if manifest["acquisition_status"] in SUCCESS | {"retrieved_qc_failed"}:
            a._verified_historical_pair(manifest, spec, raw, provenance)
            if entry.get("sha256") and entry["sha256"] != manifest["sha256"]:
                raise StoreConflict("Annual manifest raw hash conflicts with immutable interval evidence.")
            for key in ("size_bytes", "raw_observation_count", "http", "collection_values", "retrieved_utc", "raw_file", "provenance_file"):
                if entry.get("sha256") and key in entry and entry[key] != manifest.get(key):
                    raise StoreConflict("Annual manifest metadata conflicts with immutable interval evidence.")
        for key in SYNC_FIELDS:
            if key in manifest:
                entry[key] = manifest[key]
        entry["attempt_count_stage_b"] = len(manifest["attempts"])
        entry["manifest_file"] = path.relative_to(workspace_path(".")).as_posix()
        entry["manifest_sha256"] = sha256(path)
        for attempt in manifest["attempts"]:
            identity = (request["request_id"], attempt["number"])
            if identity not in reservations:
                catalog["attempt_reservations"].append({"request_id": identity[0], **attempt, "reconciled_from_interval": True})
                reservations.add(identity)
        if len(manifest["attempts"]) > 3:
            raise StoreConflict("Per-interval attempt budget exceeded.")
    if len(reservations) > 3 * profile["new_intervals"] or len({r[0] for r in reservations}) > profile["new_intervals"]:
        raise StoreConflict("Stage B unique interval or HTTP-attempt budget exceeded.")
    sizes = [r["size_bytes"] for r in catalog["intervals"].values() if r.get("size_bytes") is not None]
    if any(n > 10485760 for n in sizes) or sum(r.get("size_bytes") or 0 for r in catalog["intervals"].values() if not r["reused_existing_pilot"]) > profile["new_intervals"] * 10485760:
        raise StoreConflict("Recorded response sizes exceed authorized limits.")
    return catalog


def load_catalog(settings, *, create=False):
    plan = approved_plan(settings)
    output = workspace_path(settings["paths"]["output"], writable=True)
    path = output / "acquisition_manifest.json"
    if path.exists():
        catalog = a._read_manifest(path)
    elif create:
        entries = {}
        for request in plan:
            entries[request["request_id"]] = pilot_entry(settings, request) if reused_interval(settings, request) else {
                "request_id": request["request_id"], "request": interval_spec(settings, request), "reused_existing_pilot": False,
                "acquisition_status": "not_attempted", "qc_status": "not_run", "failure_category": None,
                "satellite_completeness": "unverified", "attempt_count_stage_b": 0}
        catalog = {"schema_version": 1, "stage": stage_profile(settings)["stage"], "created_utc": utc_now(), "intervals": entries,
                   "attempt_reservations": [], "batches": [], "halt_reason": None, "satellite_completeness": "unverified"}
        atomic_json(path, catalog)
    else:
        return None
    return _synchronize(settings, catalog, plan)


def failure_counts(settings, catalog):
    outcomes = []
    for entry in catalog["intervals"].values():
        if entry.get("manifest_file"):
            events = a._read_manifest(workspace_path(entry["manifest_file"]))["events"]
            outcomes.extend((event.get("at_utc", ""), event["event"] == "attempt_failed") for event in events if event["event"] in {"attempt_failed", "response_prepared"})
    recent = sorted(outcomes)[-settings["batch"]["recent_attempt_window"]:]
    return {"failed_attempts": sum(failed for _, failed in outcomes), "recent_failed_attempts": sum(failed for _, failed in recent)}


def run_batch(settings, *, execute=False, batch_size=10, session=None, sleeper=time.sleep, qc_function=None):
    plan = approved_plan(settings)
    profile = stage_profile(settings)
    if type(batch_size) is not int or not 1 <= batch_size <= 10:
        raise a.AcquisitionError("A sequential batch may contain one to ten new intervals.", code="scope_rejected")
    if not execute:
        return {"mode": "dry_run", "calendar_days": profile["days"], "logical_intervals": profile["intervals"], "new_intervals": profile["new_intervals"],
                "pilot_reused": profile["pilot"], "batch_limit": batch_size, "observation_requests": 0}
    if profile["stage"] in ("C4", "C5"):
        from .paired_acquisition import require_authorization
        require_authorization(settings)
    from .stage_b_qc import ensure_interval_qc
    qc_function = qc_function or ensure_interval_qc
    output = workspace_path(settings["paths"]["output"], writable=True)
    path = output / "acquisition_manifest.json"
    with store_lock(output):
        before_integrity = protected_integrity(settings)
        catalog = load_catalog(settings, create=True)
        if catalog["halt_reason"]:
            raise StoreConflict("A previous critical batch stop requires review; automatic continuation is blocked.")
        resources = resource_snapshot()
        if resources["disk_free_bytes"] < settings["batch"]["minimum_disk_free_bytes"] or resources.get("available_physical_bytes", 0) < settings["batch"]["minimum_available_ram_bytes"]:
            raise StoreConflict("Batch disk/RAM reserve is insufficient.")
        if failure_counts(settings, catalog)["recent_failed_attempts"] >= settings["batch"]["maximum_recent_failed_attempts"]:
            raise StoreConflict("Recent server/transport failure rate requires review.")
        batch = {"number": len(catalog["batches"]) + 1, "started_utc": utc_now(), "requested_new_interval_limit": batch_size,
                 "resources_before": resources, "integrity_before": before_integrity, "new_intervals": [], "status": "running"}
        catalog["batches"].append(batch)
        atomic_json(path, catalog)
        def before_attempt(manifest):
            identity = (manifest["request_id"], manifest["attempts"][-1]["number"])
            if identity in {(r["request_id"], r["number"]) for r in catalog["attempt_reservations"]}:
                raise StoreConflict("This HTTP attempt identity was already consumed.")
            if len(catalog["attempt_reservations"]) >= 3 * profile["new_intervals"]:
                raise StoreConflict("Total Stage B HTTP-attempt budget exhausted.")
            catalog["attempt_reservations"].append({"request_id": identity[0], **manifest["attempts"][-1]})
            atomic_json(path, catalog)
        try:
            for request in plan:
                entry = catalog["intervals"][request["request_id"]]
                if entry["qc_status"] == "passed" and entry["acquisition_status"] in SUCCESS:
                    continue
                if entry["reused_existing_pilot"]:
                    result = entry
                else:
                    is_new = entry["attempt_count_stage_b"] == 0
                    if is_new and len(batch["new_intervals"]) >= batch_size:
                        break
                    acquire = {"B": acquire_stage_b_interval, "C1": acquire_stage_c1_interval, "C2": acquire_stage_c2_interval, "C3": acquire_stage_c3_interval,
                               "C4": acquire_c4_c5_interval, "C5": acquire_c4_c5_interval, "C6": acquire_stage_c6_interval}[profile["stage"]]
                    result = acquire(settings, request, execute=True, session=session, sleeper=sleeper, before_attempt=before_attempt)
                    _synchronize(settings, catalog, plan)
                    if is_new and result["attempts"]:
                        batch["new_intervals"].append(request["request_id"])
                    if result["acquisition_status"] not in SUCCESS:
                        catalog["halt_reason"] = result.get("failure_category") or "interval_acquisition_failed"
                        break
                summary = qc_function(settings, request, result)
                if not summary["qc_passed"]:
                    catalog["halt_reason"] = "scientific_qc_failed"
                    break
                if not entry["reused_existing_pilot"]:
                    raw, provenance, interval_manifest = interval_paths(settings, request)
                    a._record_interval_qc(interval_spec(settings, request), raw, provenance, interval_manifest,
                                          workspace_path(summary["qc_summary_file"]), passed=True)
                    _synchronize(settings, catalog, plan)
                entry.update(qc_status="passed", qc_summary_file=summary["qc_summary_file"], qc_summary_sha256=sha256(workspace_path(summary["qc_summary_file"])),
                             retained_observation_count=summary["retained_rows"])
                atomic_json(path, catalog)
                if failure_counts(settings, catalog)["recent_failed_attempts"] >= settings["batch"]["maximum_recent_failed_attempts"]:
                    catalog["halt_reason"] = "recent_server_failure_rate"
                    break
        except (a.AcquisitionError, StoreConflict, OSError, ValueError) as error:
            catalog["halt_reason"] = error.code if isinstance(error, a.AcquisitionError) else "integrity_conflict" if isinstance(error, StoreConflict) else "storage_or_qc_failure"
        _synchronize(settings, catalog, plan)
        batch["finished_utc"] = utc_now()
        batch["resources_after"] = resource_snapshot()
        batch["integrity_after"] = protected_integrity(settings)
        batch["status"] = "stopped" if catalog["halt_reason"] else "completed"
        batch["failure_counts"] = failure_counts(settings, catalog)
        atomic_json(path, catalog)
        return {"batch": batch["number"], "status": batch["status"], "halt_reason": catalog["halt_reason"],
                "new_intervals_this_batch": len(batch["new_intervals"]),
                "acquired_intervals_including_pilot": sum(e["acquisition_status"] in SUCCESS for e in catalog["intervals"].values()),
                "qc_passed_intervals": sum(e["qc_status"] == "passed" for e in catalog["intervals"].values()),
                "unique_new_intervals_attempted": len({r["request_id"] for r in catalog["attempt_reservations"]}),
                "total_observation_attempt_slots": len(catalog["attempt_reservations"])}
