"""Scientific calendar and unknown-versus-zero planning regressions."""

from copy import deepcopy
from datetime import date

import pytest

from agniscope.config import load_config, workspace_path
from agniscope.planning import build_request_plan, build_season_plan, coverage_state, dates, resource_estimates


@pytest.fixture
def settings():
    return load_config(workspace_path("configs/historical_protocol.toml"))


def test_every_study_day_once_with_leap_days_and_no_overlap(settings):
    plan = build_request_plan(settings)
    expanded = [day for r in plan for day in dates(date.fromisoformat(r["start_utc_date"]), date.fromisoformat(r["end_utc_date"]))]
    assert expanded == list(dates(date(2019, 1, 1), date(2025, 12, 31)))
    assert len(expanded) == 2557
    assert date(2020, 2, 29) in expanded and date(2024, 2, 29) in expanded
    assert all(1 <= r["days"] <= 5 for r in plan)
    assert all(r["start_utc_date"][:4] == r["end_utc_date"][:4] for r in plan)


def test_pilot_reused_once_and_stage_requests_reconcile(settings):
    plan = build_request_plan(settings)
    pilot = [r for r in plan if r["status"] == "reuse_verified_pilot"]
    assert len(plan) == 513 and len(pilot) == 1
    assert (pilot[0]["start_utc_date"], pilot[0]["end_utc_date"]) == ("2024-04-01", "2024-04-05")
    assert {stage: sum(r["proposed_stage"] == stage for r in plan) for stage in "ABC"} == {"A": 1, "B": 73, "C": 438}
    assert all(not r["authorized"] for r in plan)


def test_request_ids_stable_and_unique_without_credentials(settings):
    first = build_request_plan(settings)
    assert first == build_request_plan(settings)
    assert len({r["request_id"] for r in first}) == len(first)
    assert all("http" not in str(r) and "MAP_KEY" not in str(r) for r in first)


@pytest.mark.parametrize("section,key,value", [
    ("execution", "enabled", True), ("execution", "stage_a_authorized", True),
    ("execution", "stage_b_authorized", True), ("execution", "stage_c_authorized", True),
    ("study", "source", "VIIRS_SNPP_NRT"), ("study", "end", "2026-12-31"),
    ("study", "bbox", [77, 28, 82, 32]), ("requests", "maximum_days", 6),
    ("requests", "maximum_response_bytes", 10485761),
    ("reuse", "start", "2023-04-01"), ("recurrence_design", "implemented", True),
])
def test_no_silent_scope_or_execution_changes(settings, section, key, value):
    changed = deepcopy(settings)
    changed[section][key] = value
    with pytest.raises(ValueError):
        build_request_plan(changed)


def test_cross_year_season_is_left_censored_without_2018_extension(settings):
    seasons = build_season_plan(settings)
    assert len(seasons) == 14
    first = next(r for r in seasons if r["season_id"] == "EXT_2018_2019")
    assert first["start_utc_date"] == "2018-11-01"
    assert not first["calendar_window_fully_within_study"]
    assert first["expected_calendar_days"] - first["days_within_study_plan"] == 61
    assert sum(r["calendar_window_fully_within_study"] for r in seasons if r["definition"] == "extended") == 6
    assert all(r["actual_satellite_completeness"] == "unverified" for r in seasons)


def test_core_season_leap_calendar(settings):
    cores = {r["season_id"]: r for r in build_season_plan(settings) if r["definition"] == "core"}
    assert cores["CORE_2019"]["expected_calendar_days"] == 121
    assert cores["CORE_2020"]["expected_calendar_days"] == 122
    assert cores["CORE_2024"]["expected_calendar_days"] == 122
    assert all(r["calendar_window_fully_within_study"] for r in cores.values())


def test_unknown_failed_zero_and_positive_are_distinct():
    states = [coverage_state(requested=False), coverage_state(requested=True),
              coverage_state(requested=True, validated=True, raw_rows=0),
              coverage_state(requested=True, validated=True, raw_rows=3)]
    assert len(set(states)) == 4
    assert states[2] == "queried_zero_coverage_unverified"
    assert states[3] == "acquired_detections_coverage_unverified"


@pytest.mark.parametrize("arguments", [
    {"requested": False, "raw_rows": 0}, {"requested": False, "validated": True},
    {"requested": True, "validated": True}, {"requested": True, "validated": True, "raw_rows": -1},
])
def test_impossible_coverage_evidence_rejected(arguments):
    with pytest.raises(ValueError):
        coverage_state(**arguments)


def test_estimates_separate_request_counts_cost_and_hard_caps(settings):
    estimate = resource_estimates(settings, build_request_plan(settings), 47686)
    assert estimate["study_days"] == 2557 and estimate["future_observation_requests"] == 512
    assert estimate["maximum_future_persisted_response_bytes_at_caps"] == 5 * 1024**3
    assert estimate["maximum_attempt_transport_bytes_at_caps"] == 15 * 1024**3
    assert not estimate["transaction_cost_measured"]


def test_reversed_date_range_is_rejected():
    with pytest.raises(ValueError):
        list(dates(date(2025, 1, 2), date(2025, 1, 1)))
