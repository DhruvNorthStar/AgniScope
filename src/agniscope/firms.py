"""Transparent row-level FIRMS CSV quality control, without clustering."""

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import workspace_path
from .provenance import sha256, utc_now, write_json
from .spatial import assign_points

REQUIRED_COLUMNS = ("latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "version")
EXPECTED_OPTIONAL_COLUMNS = ("bright_ti4", "scan", "track", "bright_ti5", "frp", "daynight")
CONFIDENCE = {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal", "h": "high", "high": "high"}
SNPP_LABELS = {"N", "SNPP", "S-NPP", "SUOMI-NPP", "SUOMI NPP"}


def classify_processing_version(value: str) -> str:
    """Read NASA's collection-only SP convention without rewriting source values.

    The historical SP endpoint returns integer strings such as "2", while other
    exports use "2.0". A decimal point and an explicit SP suffix are not required.
    Documented near/real-time suffixes remain distinct; blanks/unknowns fail.
    """
    match = re.fullmatch(r"\d+(?:\.\d+)*(?P<suffix>NRT|URT|RT)?", str(value).strip().upper())
    if match is None:
        return "invalid"
    return match.group("suffix") or "standard_processing"


class SchemaError(ValueError):
    """The input cannot support the documented validation contract."""


@dataclass
class ValidationResult:
    records: pd.DataFrame
    summary: dict

    @property
    def accepted(self) -> pd.DataFrame:
        return self.records.loc[self.records["agniscope_disposition"].eq("retained")].copy()


def load_firms_csv(path: str | Path) -> pd.DataFrame:
    source = workspace_path(path)
    try:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                raise SchemaError("CSV has no header; an empty file is different from a zero-row dataset.")
            if len(header) != len(set(header)):
                raise SchemaError("CSV contains duplicate column names.")
            records = []
            for row in reader:
                if len(row) > len(header):
                    raise SchemaError("CSV record has more fields than its header; refusing ambiguous column alignment.")
                # Retain blank/short records explicitly; missing fields fail row-level QA.
                records.append(row + [""] * (len(header) - len(row)))
        return pd.DataFrame(records, columns=header, dtype=str)
    except (csv.Error, UnicodeError) as exc:
        raise SchemaError("CSV parsing failed; the source has not been changed.") from exc


def validate_frame(frame: pd.DataFrame, districts: gpd.GeoDataFrame, config: dict, *, dataset_kind: str = "unverified_input") -> ValidationResult:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise SchemaError("Missing required columns: " + ", ".join(missing))
    if any(str(column).startswith("agniscope_") for column in frame.columns):
        raise SchemaError("Input uses the reserved agniscope_ annotation prefix.")
    raw = frame.copy().reset_index(drop=True)
    for column in raw:
        raw[column] = raw[column].astype("string").fillna("")
    result = raw.copy()
    n = len(raw)
    result["agniscope_source_record"] = np.arange(1, n + 1)
    lat = pd.to_numeric(raw["latitude"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    lon = pd.to_numeric(raw["longitude"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid_coordinates = np.isfinite(lat) & np.isfinite(lon) & (lat >= -90) & (lat <= 90) & (lon >= -180) & (lon <= 180)
    result["agniscope_latitude"] = lat
    result["agniscope_longitude"] = lon

    dates = raw["acq_date"].str.strip()
    times = raw["acq_time"].str.strip()
    formats_ok = dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}") & times.str.fullmatch(r"\d{1,4}")
    padded = times.str.zfill(4)
    clock_ok = pd.to_numeric(padded.str[:2], errors="coerce").between(0, 23) & pd.to_numeric(padded.str[2:], errors="coerce").between(0, 59)
    acquired = pd.to_datetime((dates + " " + padded).where(formats_ok & clock_ok), format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    result["agniscope_acquired_utc"] = acquired
    result["agniscope_acquired_ist"] = acquired.dt.tz_convert("Asia/Kolkata")
    confidence = raw["confidence"].str.strip().str.lower().map(CONFIDENCE)
    result["agniscope_confidence"] = confidence
    satellite = raw["satellite"].str.strip().str.upper()
    version = raw["version"].str.strip().str.upper()
    processing = version.map(classify_processing_version)
    version_format_ok = processing.ne("invalid").to_numpy(dtype=bool)
    standard_processing = processing.eq("standard_processing").to_numpy(dtype=bool)

    coordinate_positions = np.flatnonzero(valid_coordinates)
    geo = gpd.GeoDataFrame(index=coordinate_positions, geometry=gpd.points_from_xy(lon[valid_coordinates], lat[valid_coordinates]), crs=config["spatial"]["observation_crs"])
    annotations = assign_points(geo, districts, near_distance_m=config["spatial"]["near_boundary_distance_m"])
    result = result.join(annotations)
    inside = result["agniscope_in_footprint"].fillna(False).to_numpy(dtype=bool)
    start = pd.Timestamp(config["project"]["study_start"], tz="UTC")
    end_exclusive = pd.Timestamp(config["project"]["study_end"], tz="UTC") + pd.Timedelta(days=1)
    in_period = acquired.ge(start) & acquired.lt(end_exclusive)
    duplicates = raw.duplicated(keep="first").to_numpy()
    rules = [
        ("invalid_coordinates", "invalid_data", ~valid_coordinates),
        ("invalid_timestamp", "invalid_data", acquired.isna().to_numpy()),
        ("invalid_confidence", "invalid_data", confidence.isna().to_numpy()),
        ("invalid_metadata", "invalid_data", ~version_format_ok | satellite.eq("").to_numpy(dtype=bool)),
        ("exact_duplicate", "duplicate_data", duplicates),
        ("different_satellite", "scientific_selection", ~satellite.isin(SNPP_LABELS).to_numpy()),
        ("different_instrument", "scientific_selection", raw["instrument"].str.strip().str.upper().ne("VIIRS").to_numpy(dtype=bool) if "instrument" in raw else np.zeros(n, dtype=bool)),
        ("nonstandard_processing", "scientific_selection", ~standard_processing),
        ("outside_study_period", "scientific_selection", ~in_period.to_numpy()),
        ("outside_footprint", "scientific_selection", valid_coordinates & ~inside),
    ]
    active = np.ones(n, dtype=bool)
    reasons = np.full(n, "retained", dtype=object)
    dispositions = np.full(n, "retained", dtype=object)
    flags: list[list[str]] = [[] for _ in range(n)]
    funnel = []
    for rule, category, fails in rules:
        for position in np.flatnonzero(fails):
            flags[int(position)].append(rule)
        removed = active & fails
        reasons[removed] = rule
        dispositions[removed] = category
        active[removed] = False
        funnel.append({"rule": rule, "category": category, "removed_first_failure": int(removed.sum()), "remaining": int(active.sum()), "flagged_including_other_failures": int(fails.sum())})
    result["agniscope_primary_reason"] = reasons
    result["agniscope_disposition"] = dispositions
    result["agniscope_all_flags"] = [";".join(items) for items in flags]
    accepted = result.loc[active]
    bucket = accepted["agniscope_district_original"].astype("string").fillna("UNASSIGNED_OR_AMBIGUOUS")
    district_counts = {str(key): int(value) for key, value in bucket.value_counts().items()}
    versions = sorted(version.loc[active].unique().tolist())
    warnings = []
    if len(versions) > 1:
        warnings.append("Multiple Standard Processing collection/version values remain; harmonize before analysis.")
    if "type" not in raw:
        warnings.append("Hotspot type is unavailable; vegetation/static-source classification has not been verified.")
    if not n:
        warnings.append("Header-only dataset: no observations; this is not evidence of no fires.")
    key = pd.DataFrame({"lat": lat, "lon": lon, "utc": acquired, "satellite": satellite})
    potential = key.loc[valid_coordinates & acquired.notna().to_numpy()].duplicated(keep=False)
    summary = {
        "generated_utc": utc_now(), "dataset_kind": dataset_kind,
        "input_rows": n, "retained_rows": int(active.sum()),
        "invalid_rows": int((dispositions == "invalid_data").sum()),
        "duplicate_rows_removed": int((dispositions == "duplicate_data").sum()),
        "scientific_selection_exclusions": int((dispositions == "scientific_selection").sum()),
        "all_input_rows_accounted_for": int(sum(step["removed_first_failure"] for step in funnel) + active.sum()) == n,
        "funnel": funnel,
        "confidence_input_normalized": {str(k): int(v) for k, v in confidence.fillna("unknown").value_counts().items()},
        "confidence_retained": {str(k): int(v) for k, v in accepted["agniscope_confidence"].value_counts().items()},
        "retained_processing_versions": versions,
        "potential_same_position_time_satellite_rows": int(potential.sum()),
        "district_counts_including_unassigned_bucket": district_counts,
        "district_buckets_sum_to_statewide_retained": sum(district_counts.values()) == int(active.sum()),
        "district_assignment_counts": {str(k): int(v) for k, v in accepted["agniscope_district_assignment"].value_counts().items()},
        "missing_optional_columns": sorted(set(EXPECTED_OPTIONAL_COLUMNS) - set(raw.columns)),
        "warnings": warnings,
        "low_confidence_policy": "retained initially; contextual/sensitivity decisions are separate",
        "tree_cover_filter_applied": False,
        "timestamp_timezone": "UTC primary; Asia/Kolkata derived separately",
        "source_fields_preserved": list(raw.columns),
    }
    return ValidationResult(result, summary)


def validate_csv(path: str | Path, districts: gpd.GeoDataFrame, config: dict, *, dataset_kind: str = "unverified_input") -> ValidationResult:
    source = workspace_path(path)
    result = validate_frame(load_firms_csv(source), districts, config, dataset_kind=dataset_kind)
    result.summary["source"] = {"path": source.relative_to(workspace_path(".")).as_posix(), "sha256": sha256(source), "size_bytes": source.stat().st_size}
    return result


def save_validation(result: ValidationResult, output_dir: str | Path) -> Path:
    target = workspace_path(output_dir, writable=True)
    target.mkdir(parents=True, exist_ok=False)
    result.records.to_csv(target / "all_records_with_qc.csv", index=False)
    result.accepted.to_csv(target / "accepted_records.csv", index=False)
    result.accepted.to_parquet(target / "accepted_records.parquet", index=False)
    write_json(target / "qc_summary.json", result.summary)
    return target
