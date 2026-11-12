"""Identity, source preservation and scientific row accounting on synthetic rows."""

import pandas as pd

from agniscope.firms import validate_frame
from agniscope.historical_qc import observation_index, describe_population


def test_stable_identity_retains_logical_record_order_and_payload():
    frame = pd.DataFrame([{"acq_time": "0003", "version": "2"}, {"acq_time": "0003", "version": "2"}])
    first = observation_index(frame, "a" * 64)
    assert first.equals(observation_index(frame, "a" * 64))
    assert first.agniscope_observation_id.tolist() == ["a"*64+":1", "a"*64+":2"]
    assert first.agniscope_source_payload_sha256.nunique() == 1
    assert frame.acq_time.tolist() == ["0003", "0003"] and frame.version.tolist() == ["2", "2"]


def test_duplicate_fingerprints_ignore_column_order_but_preserve_every_source_value():
    a = pd.DataFrame([{"latitude": "30", "version": "2", "type": "0"}])
    b = a[["type", "latitude", "version"]].copy()
    assert observation_index(a, "a").agniscope_source_payload_sha256.iloc[0] == observation_index(b, "b").agniscope_source_payload_sha256.iloc[0]
    b.loc[0, "type"] = "1"
    assert observation_index(a, "a").agniscope_source_payload_sha256.iloc[0] != observation_index(b, "b").agniscope_source_payload_sha256.iloc[0]


def test_absent_source_column_is_not_an_invented_blank_value():
    a = pd.DataFrame([{"a": "x"}]).reindex(columns=["a", "extra"])
    b = pd.DataFrame([{"a": "x", "extra": ""}])
    assert observation_index(a, "a").agniscope_source_payload_sha256.iloc[0] != observation_index(b, "b").agniscope_source_payload_sha256.iloc[0]


def test_utc_ist_cross_midnight_and_leap_date_preserve_source_strings(config, districts, valid_record):
    frame = pd.DataFrame([{**valid_record, "acq_date": "2024-02-29", "acq_time": "2330", "version": "2"}])
    result = validate_frame(frame, districts, config)
    row = result.accepted.iloc[0]
    assert str(row.agniscope_acquired_utc) == "2024-02-29 23:30:00+00:00"
    assert str(row.agniscope_acquired_ist) == "2024-03-01 05:00:00+05:30"
    assert row.acq_time == "2330" and row.version == "2"


def test_qc_accounts_for_invalid_duplicate_and_outside_rows(config, districts, valid_record):
    frame = pd.DataFrame([valid_record, valid_record, {**valid_record, "latitude": "bad"},
                          {**valid_record, "latitude": "0", "longitude": "0"}])
    result = validate_frame(frame, districts, config)
    assert result.summary["input_rows"] == 4 and result.summary["retained_rows"] == 1
    assert result.summary["invalid_rows"] == 1 and result.summary["duplicate_rows_removed"] == 1
    assert result.summary["scientific_selection_exclusions"] == 1
    assert result.summary["all_input_rows_accounted_for"]
    assert result.records.latitude.tolist() == frame.latitude.tolist()


def test_frp_zero_is_usable_missing_and_negative_are_reported():
    frame = pd.DataFrame({"acq_date": ["2023-04-01"]*4, "frp": ["0", "", "-1", "2"], "confidence": ["n"]*4})
    values = describe_population(frame)["frp"]
    assert values["usable_count"] == 2 and values["zero_count"] == 1
    assert values["missing_count"] == 1 and values["unusable_nonmissing_count"] == 1
    assert values["statistics"]["mean"] == 1
