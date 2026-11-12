from pathlib import Path

import pandas as pd
import pytest

from agniscope.firms import REQUIRED_COLUMNS, SchemaError, load_firms_csv, save_validation, validate_frame


def validate(rows, districts, config):
    return validate_frame(pd.DataFrame(rows), districts, config, dataset_kind="synthetic_test_fixture")


@pytest.mark.parametrize("field,value", [("latitude", "91"), ("latitude", "nan"), ("latitude", ""), ("longitude", "-181"), ("longitude", "inf"), ("longitude", "not-a-number")])
def test_invalid_coordinates_are_accounted_for(field, value, valid_record, districts, config):
    valid_record[field] = value
    result = validate([valid_record], districts, config)
    assert result.summary["invalid_rows"] == 1
    assert result.records.iloc[0]["agniscope_primary_reason"] == "invalid_coordinates"
    assert result.summary["all_input_rows_accounted_for"]


@pytest.mark.parametrize("field,value", [("acq_date", "2024-02-30"), ("acq_date", "2024-4-01"), ("acq_date", ""), ("acq_time", "2400"), ("acq_time", "1260"), ("acq_time", "3.0"), ("acq_time", "")])
def test_invalid_timestamps(field, value, valid_record, districts, config):
    valid_record[field] = value
    result = validate([valid_record], districts, config)
    assert result.records.iloc[0]["agniscope_primary_reason"] == "invalid_timestamp"


def test_utc_zero_padding_and_original_fields(valid_record, districts, config):
    result = validate([valid_record], districts, config)
    record = result.accepted.iloc[0]
    assert record["acq_time"] == "3"
    assert record["agniscope_acquired_utc"] == pd.Timestamp("2024-04-01T00:03:00Z")
    assert record["agniscope_acquired_ist"].hour == 5
    assert record["agniscope_acquired_ist"].minute == 33
    assert record["custom_source_field"] == "preserve me"


def test_leap_date_and_ist_rollover(valid_record, districts, config):
    valid_record.update(acq_date="2024-02-29", acq_time="2330")
    result = validate([valid_record], districts, config)
    assert str(result.accepted.iloc[0]["agniscope_acquired_ist"].date()) == "2024-03-01"


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_missing_required_columns(missing, valid_record, districts, config):
    del valid_record[missing]
    with pytest.raises(SchemaError, match="Missing required"):
        validate([valid_record], districts, config)


def test_header_only_dataset_is_supported(districts, config):
    result = validate_frame(pd.DataFrame(columns=REQUIRED_COLUMNS), districts, config, dataset_kind="synthetic_test_fixture")
    assert result.summary["input_rows"] == 0
    assert result.accepted.empty
    assert result.summary["all_input_rows_accounted_for"]
    assert result.summary["district_buckets_sum_to_statewide_retained"]


def test_exact_duplicates_removed_but_distinct_overpasses_retained(valid_record, districts, config):
    later = dict(valid_record, acq_time="1300")
    result = validate([valid_record, valid_record.copy(), later], districts, config)
    assert result.summary["input_rows"] == 3
    assert result.summary["duplicate_rows_removed"] == 1
    assert result.summary["retained_rows"] == 2
    assert result.summary["all_input_rows_accounted_for"]


@pytest.mark.parametrize("value,normalized", [("l", "low"), ("low", "low"), ("nominal", "nominal"), ("H", "high")])
def test_viirs_confidence_retains_low_for_context(value, normalized, valid_record, districts, config):
    valid_record["confidence"] = value
    result = validate([valid_record], districts, config)
    assert result.summary["retained_rows"] == 1
    assert result.accepted.iloc[0]["agniscope_confidence"] == normalized


def test_modis_numeric_confidence_is_not_accepted(valid_record, districts, config):
    valid_record["confidence"] = "95"
    assert validate([valid_record], districts, config).summary["invalid_rows"] == 1


@pytest.mark.parametrize("change,reason", [({"satellite": "N20"}, "different_satellite"), ({"instrument": "MODIS"}, "different_instrument"), ({"version": "2.0NRT"}, "nonstandard_processing"), ({"acq_date": "2026-01-01"}, "outside_study_period"), ({"latitude": "0", "longitude": "0"}, "outside_footprint")])
def test_scientific_exclusion_is_distinct_from_invalid_data(change, reason, valid_record, districts, config):
    valid_record.update(change)
    result = validate([valid_record], districts, config)
    assert result.summary["invalid_rows"] == 0
    assert result.summary["scientific_selection_exclusions"] == 1
    assert result.records.iloc[0]["agniscope_primary_reason"] == reason


def test_all_failure_flags_and_first_failure_funnel(valid_record, districts, config):
    valid_record.update(latitude="999", acq_time="bad", confidence="unknown")
    result = validate([valid_record], districts, config)
    assert result.summary["invalid_rows"] == 1
    assert "invalid_timestamp" in result.records.iloc[0]["agniscope_all_flags"]
    assert sum(x["removed_first_failure"] for x in result.summary["funnel"]) == 1


def test_csv_loader_preserves_zeros_and_blank_record(tmp_path, valid_record, districts, config):
    source = tmp_path / "synthetic.csv"
    pd.DataFrame([dict(valid_record, acq_time="0003")]).to_csv(source, index=False)
    loaded = load_firms_csv(source)
    assert loaded.iloc[0]["acq_time"] == "0003"
    result = validate_frame(loaded, districts, config, dataset_kind="synthetic_test_fixture")
    target = save_validation(result, tmp_path / "synthetic_qc")
    assert pd.read_parquet(target / "accepted_records.parquet").iloc[0]["acq_time"] == "0003"
    assert (target / "all_records_with_qc.csv").is_file()
    assert (target / "qc_summary.json").is_file()


def test_empty_file_and_duplicate_headers_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SchemaError, match="no header"):
        load_firms_csv(empty)
    duplicated = tmp_path / "duplicate_headers.csv"
    duplicated.write_text("latitude,latitude\n1,2\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate column"):
        load_firms_csv(duplicated)


def test_reserved_annotation_prefix_rejected(valid_record, districts, config):
    valid_record["agniscope_disposition"] = "spoofed"
    with pytest.raises(SchemaError, match="reserved"):
        validate([valid_record], districts, config)


def test_blank_csv_records_are_not_silently_skipped(tmp_path, valid_record, districts, config):
    source = tmp_path / "synthetic_with_blank.csv"
    pd.DataFrame([valid_record]).to_csv(source, index=False)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    loaded = load_firms_csv(source)
    result = validate_frame(loaded, districts, config, dataset_kind="synthetic_test_fixture")
    assert result.summary["input_rows"] == 2
    assert result.summary["retained_rows"] == 1
    assert result.summary["invalid_rows"] == 1


def test_extra_csv_fields_cannot_shift_coordinates(tmp_path):
    source = tmp_path / "synthetic_misaligned.csv"
    source.write_text("latitude,longitude\n30,79,unexpected\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="more fields"):
        load_firms_csv(source)


def test_historical_integer_collection_is_retained_without_relabelling(valid_record, districts, config):
    valid_record["version"] = "2"
    result = validate([valid_record], districts, config)
    assert result.summary["retained_rows"] == 1
    assert result.accepted.iloc[0]["version"] == "2"
    assert result.summary["retained_processing_versions"] == ["2"]


@pytest.mark.parametrize("version,reason", [("2NRT", "nonstandard_processing"), ("2RT", "nonstandard_processing"), ("2URT", "nonstandard_processing"), ("", "invalid_metadata"), ("unknown", "invalid_metadata"), ("2SP", "invalid_metadata")])
def test_integer_support_does_not_accept_missing_unknown_or_nonstandard_versions(version, reason, valid_record, districts, config):
    valid_record["version"] = version
    result = validate([valid_record], districts, config)
    assert result.summary["retained_rows"] == 0
    assert result.records.iloc[0]["agniscope_primary_reason"] == reason
