"""Synthetic source/QC/annual integration, using existing bounded GIS sampling."""

import json
import sqlite3

import pandas as pd
import pytest

from agniscope import stage_b as b
from agniscope import stage_b_qc as q
from test_stage_b import stage_env, Session


@pytest.mark.parametrize("empty", [True, False])
def test_interval_qc_preserves_zero_and_positive_source_identity(stage_env, empty):
    settings, _, _ = stage_env
    request = b.plan_2024(settings)[0]
    session = Session(zero_dates={"2024-01-01"} if empty else ())
    entry = b.acquire_stage_b_interval(settings, request, execute=True, session=session, sleeper=lambda seconds: None)
    summary = q.ensure_interval_qc(settings, request, entry)
    assert summary["raw_rows"] == (0 if empty else 1) and summary["qc_passed"]
    accepted = pd.read_parquet(q.bundle_path(settings, request) / "retained.parquet")
    if len(accepted):
        assert accepted.version.tolist() == ["2"] and accepted.acq_time.tolist() == ["2330"]
        assert accepted.agniscope_observation_id.iloc[0] == entry["sha256"] + ":1"
        assert accepted.frp.iloc[0] == "0" and summary["retained_population"]["frp"]["zero_count"] == 1
    assert q.ensure_interval_qc(settings, request, entry) == summary


def test_utc_calendar_and_ist_year_boundary_remain_distinct(stage_env):
    settings, _, _ = stage_env
    request = b.plan_2024(settings)[-1]
    entry = b.acquire_stage_b_interval(settings, request, execute=True, session=Session(row_date="2024-12-31"), sleeper=lambda seconds: None)
    q.ensure_interval_qc(settings, request, entry)
    accepted = pd.read_parquet(q.bundle_path(settings, request) / "retained.parquet")
    assert accepted.agniscope_utc_date.tolist() == ["2024-12-31"]
    assert accepted.agniscope_ist_date.tolist() == ["2025-01-01"]


def test_core_and_two_partial_extended_seasons_use_exact_boundaries():
    membership = q.season_membership(["2024-02-14", "2024-02-15", "2024-02-29", "2024-06-15", "2024-06-16", "2024-11-01"])
    assert membership.agniscope_core_2024.tolist() == [False, True, True, True, False, False]
    assert membership.agniscope_extended_season_id.iloc[0] == "EXT_2023_2024"
    assert membership.agniscope_extended_season_id.iloc[-1] == "EXT_2024_2025"
    assert membership.agniscope_extended_window_partial.all()


def test_duplicate_identity_index_records_canonical_occurrence():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE fingerprints(payload_sha256 TEXT PRIMARY KEY, observation_id TEXT, request_id TEXT)")
    assert q.remember_identity(connection, "synthetic_payload", "file_a:1", "interval_a") is None
    assert q.remember_identity(connection, "synthetic_payload", "file_b:1", "interval_b") == ("file_a:1", "interval_a")
    assert connection.execute("SELECT count(*) FROM fingerprints").fetchone()[0] == 1
    connection.close()


def test_unrequested_dates_have_unknown_counts_and_coverage(stage_env):
    settings, _, _ = stage_env
    catalog = b.load_catalog(settings, create=True)
    daily = q.coverage_rows(settings, catalog, b.plan_2024(settings))
    assert len(daily) == 366
    assert daily[0]["raw_observations"] is None and daily[0]["retained_observations"] is None
    assert all(r["satellite_coverage"] == "unknown" for r in daily)


def test_complete_mock_year_streaming_accounting_and_idempotent_integration(stage_env):
    settings, resolver, _ = stage_env
    session = Session(zero_dates={"2024-01-01"})
    for _ in range(8):
        result = b.run_batch(settings, execute=True, batch_size=10, session=session, sleeper=lambda seconds: None)
        assert result["status"] == "completed"
    assert len(session.observation_dates) == 73 and "2024-04-01" not in session.observation_dates
    catalog = b.load_catalog(settings)
    summary, reused = q.integrate_year(settings, catalog, b.plan_2024(settings))
    assert not reused and summary["counts"]["raw_rows"] == 613 + 72
    assert summary["counts"]["raw_rows"] == summary["counts"]["retained_rows"] + summary["counts"]["geographic_exclusions"]
    assert summary["counts"]["cross_interval_duplicate_occurrences"] == 0
    assert summary["coverage"]["validated_dates"] == 366
    annual = resolver(settings["paths"]["annual"])
    frame = pd.read_parquet(annual / "validated_observations_2024.parquet")
    assert len(frame) == summary["counts"]["retained_rows"] and frame.agniscope_observation_id.is_unique
    assert frame.acq_date.str.startswith("2024-").all()
    again, reused = q.integrate_year(settings, catalog, b.plan_2024(settings))
    assert reused and again == summary
