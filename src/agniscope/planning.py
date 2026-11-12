"""Offline calendar planning only. No network access or acquisition executor."""

from datetime import date, timedelta
import hashlib
import json
import math


def dates(start: date, end: date):
    if start > end:
        raise ValueError("Date interval is reversed.")
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def validate_protocol(settings: dict) -> None:
    study, execution, requests = (settings[k] for k in ("study", "execution", "requests"))
    if any(execution.values()):
        raise ValueError("Phase 2A permits planning only; execution flags must remain false.")
    if (study["start"], study["end"], study["source"], study["bbox"]) != (
        "2019-01-01", "2025-12-31", "VIIRS_SNPP_SP", [77.5, 28.6, 81.2, 31.6]
    ):
        raise ValueError("The approved historical planning scope changed.")
    if requests["maximum_days"] != 5 or not 0 < requests["maximum_response_bytes"] <= 10 * 1024**2:
        raise ValueError("Planning must preserve the five-day and 10 MiB limits.")
    if study["time_basis"] != "UTC" or study["analysis_crs"] != "EPSG:32644":
        raise ValueError("Approved time and metric CRS are required.")
    reuse = settings["reuse"]
    if (reuse["start"], reuse["end"]) != ("2024-04-01", "2024-04-05"):
        raise ValueError("Only the authentic existing pilot can be reused.")
    if settings["recurrence_design"]["implemented"]:
        raise ValueError("Recurrence remains a design only.")


def request_id(source: str, bbox: list, start: date, end: date) -> str:
    payload = {"source": source, "bbox": bbox, "start": str(start), "end": str(end)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return "firms_" + digest


def build_request_plan(settings: dict) -> list[dict]:
    """Partition every UTC date once, pinning the already acquired pilot."""
    validate_protocol(settings)
    study, reuse = settings["study"], settings["reuse"]
    first, last = date.fromisoformat(study["start"]), date.fromisoformat(study["end"])
    pinned_start, pinned_end = date.fromisoformat(reuse["start"]), date.fromisoformat(reuse["end"])
    plans = []
    start = first
    while start <= last:
        end = min(start + timedelta(days=4), date(start.year, 12, 31), last)
        if start < pinned_start <= end:
            end = pinned_start - timedelta(days=1)
        reused = start == pinned_start
        if reused:
            end = pinned_end
        a = settings["stages"]["a"]
        stage = "existing_pilot" if reused else "B" if start.year == 2024 else "A" if (str(start), str(end)) == (a["start"], a["end"]) else "C"
        plans.append({
            "request_id": request_id(study["source"], study["bbox"], start, end),
            "year": start.year, "start_utc_date": str(start), "end_utc_date": str(end),
            "days": (end - start).days + 1, "planned_source": study["source"],
            "bbox_west_south_east_north": ";".join(map(str, study["bbox"])),
            "proposed_stage": stage, "authorized": False,
            "status": "reuse_verified_pilot" if reused else "planned_not_requested",
            "maximum_response_bytes": settings["requests"]["maximum_response_bytes"],
            "raw_file": reuse["raw_csv"] if reused else "",
            "sha256": reuse["raw_sha256"] if reused else "",
            "attempt_count": None if reused else 0,
        })
        start = end + timedelta(days=1)
    return plans


def coverage_state(*, requested: bool, validated: bool = False, raw_rows: int | None = None) -> str:
    """Zero retrieved detections is never evidence of complete local coverage."""
    if not requested:
        if raw_rows is not None or validated:
            raise ValueError("Unrequested dates cannot have acquisition evidence.")
        return "not_requested"
    if not validated:
        return "failed_or_unvalidated"
    if raw_rows is None or isinstance(raw_rows, bool) or raw_rows < 0:
        raise ValueError("A validated response needs a nonnegative count.")
    return "queried_zero_coverage_unverified" if raw_rows == 0 else "acquired_detections_coverage_unverified"


def build_season_plan(settings: dict) -> list[dict]:
    validate_protocol(settings)
    lower, upper = (date.fromisoformat(settings["study"][key]) for key in ("start", "end"))
    result = []
    for year in range(lower.year, upper.year + 1):
        for name, definition in settings["seasons"].items():
            start_year = year + definition["start_year_offset"]
            start = date.fromisoformat(f"{start_year}-{definition['start_month_day']}")
            end = date.fromisoformat(f"{year}-{definition['end_month_day']}")
            intersection_start, intersection_end = max(start, lower), min(end, upper)
            expected = (end - start).days + 1
            available = max(0, (intersection_end - intersection_start).days + 1)
            result.append({"season_id": definition["identifier"].format(start_year=start_year, end_year=year),
                           "definition": name, "start_utc_date": str(start), "end_utc_date": str(end),
                           "expected_calendar_days": expected, "days_within_study_plan": available,
                           "calendar_window_fully_within_study": expected == available,
                           "actual_satellite_completeness": "unverified"})
    return result


def resource_estimates(settings: dict, plan: list[dict], pilot_bytes: int) -> dict:
    future = sum(item["status"] == "planned_not_requested" for item in plan)
    days = sum(item["days"] for item in plan)
    cap = settings["requests"]["maximum_response_bytes"]
    linear = pilot_bytes / 5 * days
    return {"logical_intervals": len(plan), "study_days": days, "future_observation_requests": future,
            "pilot_linear_bytes_estimate_not_forecast": math.ceil(linear),
            "five_times_pilot_estimate_bytes": math.ceil(5 * linear),
            "twenty_times_pilot_estimate_bytes": math.ceil(20 * linear),
            "maximum_future_persisted_response_bytes_at_caps": future * cap,
            "maximum_attempt_transport_bytes_at_caps": future * cap * settings["requests"]["maximum_attempts"],
            "transaction_cost_measured": False, "duration_estimated": False}
