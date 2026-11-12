"""Bounded two-interval schema integration and Stage A QC; no clustering."""

from collections import Counter
from datetime import date
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import load_config, workspace_path
from .firms import REQUIRED_COLUMNS, load_firms_csv, validate_csv
from .landcover import sample_landcover
from .planning import dates
from .provenance import sha256, utc_now
from .spatial import load_boundaries
from .acquisition_store import StoreConflict, atomic_json


def observation_index(raw: pd.DataFrame, raw_hash: str) -> pd.DataFrame:
    """Payload fingerprints span a union schema; absent fields are distinct from blanks."""
    if len(set(raw.columns)) != len(raw.columns):
        raise ValueError("Source columns must be unique.")
    ids, payloads = [], []
    for position, row in enumerate(raw.to_dict("records"), 1):
        ids.append(f"{raw_hash}:{position}")
        normalized = {key: (None if pd.isna(value) else str(value)) for key, value in row.items()}
        payloads.append(hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
    return pd.DataFrame({"agniscope_observation_id": ids, "agniscope_source_payload_sha256": payloads}, index=raw.index)


def _counts(series):
    values = series.astype("string").fillna("MISSING")
    values = values.mask(values.str.strip().eq(""), "MISSING")
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


def _optional(frame, column):
    return frame[column] if column in frame else pd.Series([""] * len(frame), index=frame.index, dtype="string")


def describe_population(frame):
    frp_source = _optional(frame, "frp").astype("string").fillna("")
    frp = pd.to_numeric(frp_source, errors="coerce").astype(float)
    usable = np.isfinite(frp) & frp.ge(0)
    usable_values = frp.loc[usable]
    confidence = _optional(frame, "agniscope_confidence") if "agniscope_confidence" in frame else _optional(frame, "confidence")
    return {"rows": len(frame), "confidence": _counts(confidence),
            "daynight": _counts(_optional(frame, "daynight")),
            "source_dates": _counts(frame["acq_date"]),
            "frp": {"units": "MW", "usable_count": int(usable.sum()), "missing_count": int(frp_source.str.strip().eq("").sum()),
                    "unusable_nonmissing_count": int((~usable & frp_source.str.strip().ne("")).sum()),
                    "zero_count": int((usable & frp.eq(0)).sum()),
                    "statistics": {key: float(value) for key, value in usable_values.describe().items() if pd.notna(value)}}}


def build_stage_a_qc(settings, manifest):
    """Write a separate, hash-indexed QC bundle once; reuse it on a matching rerun."""
    config = load_config()
    raw_path = workspace_path(settings["paths"]["raw"])
    output = workspace_path(settings["paths"]["derived"], writable=True)
    if output != workspace_path("data/interim/phase2b_stage_a"):
        raise StoreConflict("Derived Stage A output destination changed.")
    receipt = output / "qc_artifact_manifest.json"
    if output.exists():
        if not receipt.is_file():
            raise StoreConflict("Partial derived QC bundle exists; preserve and reconcile before rerun.")
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["raw_sha256"] != manifest["sha256"]:
            raise StoreConflict("Derived QC belongs to another raw response.")
        for item in saved["files"]:
            if sha256(workspace_path(item["path"])) != item["sha256"]:
                raise StoreConflict("Derived QC artifact conflicts with its receipt.")
        return json.loads((output / "qc_summary.json").read_text(encoding="utf-8")), True
    if sha256(raw_path) != manifest["sha256"]:
        raise StoreConflict("Stage A raw hash mismatch.")
    _, districts = load_boundaries(config)
    footprint = gpd.read_file(workspace_path(config["paths"]["analysis_footprint"]), layer="analysis_footprint", engine="pyogrio")
    if len(footprint) != 1 or footprint.crs != districts.crs or not footprint.geometry.iloc[0].equals(districts.geometry.union_all()):
        raise StoreConflict("The saved analytical footprint differs from the approved district union.")
    raw = load_firms_csv(raw_path)
    validation = validate_csv(raw_path, districts, config, dataset_kind="authentic_nasa_stage_a_2023")
    records = validation.records.join(observation_index(raw, manifest["sha256"]))
    accepted = records.loc[records.agniscope_disposition.eq("retained")].copy()
    geographic = gpd.GeoDataFrame(accepted, geometry=gpd.points_from_xy(accepted.agniscope_longitude, accepted.agniscope_latitude), crs="EPSG:4326")
    cover = sample_landcover(geographic, config)
    accepted = accepted.join(cover)
    missing_required = {column: int(raw[column].astype("string").str.strip().eq("").sum()) for column in REQUIRED_COLUMNS}
    flags = {row["rule"]: row["flagged_including_other_failures"] for row in validation.summary["funnel"]}
    district_table = districts[["district_original", "agniscope_display_name"]].copy()
    district_table["retained_observations"] = district_table.district_original.map(accepted.agniscope_district_original.value_counts()).fillna(0).astype(int)
    district_table["agniscope_display_name"] = district_table.agniscope_display_name.astype(str)
    landcover_table = pd.DataFrame([{"class_code": str(code), "observations": int(n)} for code, n in accepted.agniscope_worldcover_2021_class.value_counts(dropna=False).sort_index().items()], columns=["class_code", "observations"])
    daily = []
    issues = pd.read_csv(workspace_path("configs/historical_upstream_issues.csv"), keep_default_na=False)
    for day in dates(date(2023, 4, 1), date(2023, 4, 5)):
        day_text = str(day)
        raw_count, accepted_count = int(raw.acq_date.eq(day_text).sum()), int(accepted.acq_date.eq(day_text).sum())
        concern_ids = [r.issue_id for r in issues.itertuples() if r.start_utc[:10] <= day_text <= (r.end_utc or r.start_utc)[:10]]
        daily.append({"utc_date": day_text, "request_id": manifest["request_id"], "requested": True,
                      "http_status": manifest["http"]["http_status"], "retrieval_status": "validated_csv",
                      "raw_observations": raw_count, "retained_observations": accepted_count,
                      "returned_detection_state": "positive" if raw_count else "zero_returned",
                      "retrieval_error": False, "actual_source": manifest["request"]["source"],
                      "actual_collection_values": ";".join(sorted(raw.loc[raw.acq_date.eq(day_text), "version"].unique())),
                      "qc_status": "passed", "upstream_concern_ids": ";".join(concern_ids),
                      "upstream_calendar_status": "nonexhaustive_unverified",
                      "satellite_observation_completeness": "unknown", "raw_sha256": manifest["sha256"]})
    # This is a ten-day index, not a complete annual/multiyear dataset.
    pilot_path = workspace_path(config["dbscan_pilot"]["raw_csv"])
    pilot_hash = sha256(pilot_path)
    if pilot_hash != config["dbscan_pilot"]["raw_sha256"]:
        raise StoreConflict("Preserved 2024 pilot changed.")
    pilot_raw = load_firms_csv(pilot_path)
    pilot_validation = validate_csv(pilot_path, districts, config, dataset_kind="preserved_pilot_integration_recheck")
    union_columns = sorted(set(raw.columns) | set(pilot_raw.columns))
    # Reindexing retains schema absence as null, never as an invented source value.
    index_a = observation_index(raw.reindex(columns=union_columns), manifest["sha256"])
    index_b = observation_index(pilot_raw.reindex(columns=union_columns), pilot_hash)
    fingerprints_a, fingerprints_b = set(index_a.agniscope_source_payload_sha256), set(index_b.agniscope_source_payload_sha256)
    cross_duplicates = len(fingerprints_a & fingerprints_b)
    integration_index = []
    for label, frame, validation_result, index, digest in (("stage_a_2023", raw, validation, index_a, manifest["sha256"]), ("pilot_2024", pilot_raw, pilot_validation, index_b, pilot_hash)):
        for pos in frame.index:
            checked = validation_result.records.loc[pos]
            integration_index.append({"interval": label, "raw_sha256": digest, "source_record": pos + 1,
                                      "observation_id": index.loc[pos, "agniscope_observation_id"],
                                      "source_payload_sha256": index.loc[pos, "agniscope_source_payload_sha256"],
                                      "acquired_utc": checked.agniscope_acquired_utc, "acquired_ist": checked.agniscope_acquired_ist,
                                      "source_version_literal": frame.loc[pos, "version"], "satellite_literal": frame.loc[pos, "satellite"],
                                      "district_original": checked.agniscope_district_original,
                                      "district_display": checked.agniscope_district_display,
                                      "disposition": checked.agniscope_disposition})
    integration = {"scope": "two_disjoint_five_day_intervals_only", "calendar_years_complete": False,
                   "source_schema_equal": raw.columns.tolist() == pilot_raw.columns.tolist(),
                   "union_source_columns": union_columns, "cross_interval_exact_payload_duplicate_groups": cross_duplicates,
                   "stable_ids_unique": len({r["observation_id"] for r in integration_index}) == len(integration_index),
                   "stable_ids_match_raw_hash_and_logical_record": all(r["observation_id"] == f"{r['raw_sha256']}:{r['source_record']}" for r in integration_index),
                   "source_versions_preserved": {"2023": sorted(raw.version.unique()), "2024": sorted(pilot_raw.version.unique())},
                   "pilot_raw_sha256": pilot_hash, "pilot_raw_rows": len(pilot_raw), "pilot_retained_rows": len(pilot_validation.accepted),
                   "district_annotation_schema_equal": [c for c in validation.records if "district" in c] == [c for c in pilot_validation.records if "district" in c],
                   "utc_ist_instants_equal": all(pd.Timestamp(r["acquired_utc"]) == pd.Timestamp(r["acquired_ist"]) for r in integration_index),
                   "type_lineage_resolved": False, "satellite_completeness_verified": False}
    summary = {"generated_utc": utc_now(), "source": manifest["request"]["source"], "raw_sha256": manifest["sha256"] ,
               "source_interval": {"start": "2023-04-01", "end": "2023-04-05", "timezone": "UTC"},
               "raw_observation_count": len(raw), "retained_observation_count": len(accepted),
               "geographic_exclusion_count": flags["outside_footprint"], "invalid_coordinates": flags["invalid_coordinates"],
               "invalid_rows": validation.summary["invalid_rows"], "exact_duplicates": flags["exact_duplicate"],
               "rows_with_missing_required_fields": int(pd.concat([raw[c].astype("string").str.strip().eq("") for c in REQUIRED_COLUMNS], axis=1).any(axis=1).sum()),
               "missing_required_fields": missing_required,
               "source_date_distribution": _counts(raw.acq_date), "raw_population": describe_population(records),
               "retained_population": describe_population(accepted),
               "district_counts": dict(zip(district_table.district_original, map(int, district_table.retained_observations))),
               "unassigned_retained": int(accepted.agniscope_district_assignment.eq("unassigned_inside").sum()),
               "ambiguous_retained": int(accepted.agniscope_district_assignment.eq("ambiguous_shared_edge").sum()),
               "boundary_cases": {"on_outer_edge": int(accepted.agniscope_on_outer_edge.sum()),
                                  "within_375m_outer_edge": int(accepted.agniscope_near_outer_edge.sum()),
                                  "single_district_edge": int(accepted.agniscope_district_assignment.eq("assigned_single_edge").sum())},
               "landcover": {"reference": "ESA WorldCover 2021 v200", "valid_samples": int(accepted.agniscope_landcover_status.eq("sampled").sum()),
                             "tree_cover": int(accepted.agniscope_worldcover_2021_class.eq(10).sum()),
                             "classes": _counts(accepted.agniscope_worldcover_2021_class), "sample_status": _counts(accepted.agniscope_landcover_status),
                             "all_classes_retained": True},
               "qc_contract": validation.summary, "integration": integration, "satellite_completeness": "unverified"}
    assert len(records) == sum(r["removed_first_failure"] for r in validation.summary["funnel"]) + len(accepted)
    assert int(district_table.retained_observations.sum()) + summary["unassigned_retained"] + summary["ambiguous_retained"] == len(accepted)
    assert sum(r["raw_observations"] for r in daily) == len(raw)
    output.mkdir(parents=True, exist_ok=False)
    records.to_csv(output / "all_records_with_qc.csv", index=False)
    accepted.to_csv(output / "retained_with_landcover.csv", index=False)
    accepted.to_parquet(output / "retained_with_landcover.parquet", index=False)
    district_table.to_csv(output / "district_counts.csv", index=False)
    landcover_table.to_csv(output / "landcover_counts.csv", index=False)
    pd.DataFrame(daily).to_csv(output / "daily_coverage.csv", index=False)
    pd.DataFrame(integration_index).to_csv(output / "two_interval_source_index.csv", index=False)
    atomic_json(output / "integration_checks.json", integration)
    atomic_json(output / "qc_summary.json", summary)
    files=[{"path":p.relative_to(workspace_path(".")).as_posix(), "size_bytes":p.stat().st_size, "sha256":sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]
    atomic_json(receipt, {"raw_sha256": manifest["sha256"], "self_excluded": True, "files": files})
    return summary, False
