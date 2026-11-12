"""Per-interval scientific QC and streaming annual integration, without model fitting."""

from array import array
from collections import Counter
from datetime import date
from functools import lru_cache
import json
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .acquisition_store import StoreConflict, atomic_json
from .config import load_config, workspace_path
from .firms import REQUIRED_COLUMNS, load_firms_csv, validate_csv
from .historical_qc import observation_index, describe_population, _counts
from .landcover import sample_landcover
from .planning import dates
from .provenance import sha256, utc_now
from .spatial import load_boundaries
from .stage_b import approved_plan, stage_profile


@lru_cache(maxsize=1)
def geography():
    config = load_config()
    _, districts = load_boundaries(config)
    footprint = gpd.read_file(workspace_path(config["paths"]["analysis_footprint"]), layer="analysis_footprint", engine="pyogrio")
    if len(footprint) != 1 or footprint.crs != districts.crs or not footprint.geometry.iloc[0].equals(districts.geometry.union_all()):
        raise StoreConflict("The approved footprint does not equal the unchanged district union.")
    return config, districts


def issue_register(settings=None):
    issues = pd.read_csv(workspace_path("configs/historical_upstream_issues.csv"), keep_default_na=False).to_dict("records")
    if settings is not None and stage_profile(settings)["stage"] == "C3":
        issues += pd.read_csv(workspace_path("configs/phase2b_stage_c3_upstream_issues.csv"), keep_default_na=False).to_dict("records")
    if settings is not None and stage_profile(settings)["stage"] in ("C4", "C5"):
        issues += pd.read_csv(workspace_path("configs/phase2b_c4_c5_upstream_issues.csv"), keep_default_na=False).to_dict("records")
    if settings is not None and stage_profile(settings)["stage"] == "C6":
        issues += pd.read_csv(workspace_path("configs/phase2b_stage_c6_upstream_issues.csv"), keep_default_na=False).to_dict("records")
    return issues


def issue_ids(timestamp, issues):
    moment = pd.Timestamp(timestamp)
    if pd.isna(moment):
        return ""
    result = []
    for issue in issues:
        if issue["end_utc"]:
            applies = pd.Timestamp(issue["start_utc"]) <= moment <= pd.Timestamp(issue["end_utc"])
        else:
            # Unknown recovery is not silently inferred from an onset-only notice.
            applies = str(moment.date()) == issue["start_utc"][:10]
        if applies:
            result.append(issue["issue_id"])
    return ";".join(result)


def season_membership(utc_dates, year=2024):
    if year not in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
        raise StoreConflict("Unapproved annual QC year.")
    values = pd.Series(utc_dates, dtype="string")
    return pd.DataFrame({"agniscope_calendar_year": pd.Series([year] * len(values), dtype="int64"),
                         f"agniscope_core_{year}": values.between(f"{year}-02-15", f"{year}-06-15"),
                         "agniscope_extended_season_id": values.map(lambda day: f"EXT_{year-1}_{year}" if day <= f"{year}-06-30" else f"EXT_{year}_{year+1}" if day >= f"{year}-11-01" else "OUTSIDE_EXTENDED_WINDOWS"),
                         "agniscope_extended_window_partial": values.map(lambda day: day <= f"{year}-06-30" or day >= f"{year}-11-01")})


def standardize(frame, source_columns):
    frame = frame.copy()
    integer = {"agniscope_source_record", "agniscope_worldcover_2021_class", "agniscope_calendar_year"}
    floating = {"agniscope_latitude", "agniscope_longitude", "agniscope_x_m", "agniscope_y_m"}
    boolean = {"agniscope_in_footprint", "agniscope_on_outer_edge", "agniscope_near_outer_edge", "agniscope_core_2019", "agniscope_core_2020", "agniscope_core_2021", "agniscope_core_2022", "agniscope_core_2023", "agniscope_core_2024", "agniscope_core_2025", "agniscope_extended_window_partial"}
    timestamps = {"agniscope_acquired_utc": "UTC", "agniscope_acquired_ist": "Asia/Kolkata"}
    for column in frame:
        if column in integer:
            frame[column] = pd.array(frame[column], dtype="Int64")
        elif column in floating:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        elif column in boolean:
            frame[column] = pd.array(frame[column], dtype="boolean")
        elif column in timestamps:
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(timestamps[column]).astype(f"datetime64[ns, {timestamps[column]}]")
        else:
            frame[column] = frame[column].astype("string")
    order = [*source_columns, *sorted(set(frame.columns) - set(source_columns))]
    return frame[order]


def bundle_path(settings, request):
    return workspace_path(settings["paths"]["derived"], writable=True) / ("r_" + request["request_id"][6:22])


def verify_bundle(settings, request, entry):
    output = bundle_path(settings, request)
    receipt = output / "receipt.json"
    if not receipt.is_file():
        raise StoreConflict("QC bundle is incomplete; preserve its staging evidence before recovery.")
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    if evidence["raw_sha256"] != entry["sha256"] or evidence["request_id"] != request["request_id"]:
        raise StoreConflict("QC receipt belongs to another interval/source file.")
    for item in evidence["files"]:
        if sha256(workspace_path(item["path"])) != item["sha256"]:
            raise StoreConflict("Per-interval QC artifact hash mismatch.")
    return json.loads((output / "qc_summary.json").read_text(encoding="utf-8"))


def ensure_interval_qc(settings, request, entry):
    year = stage_profile(settings)["year"]
    if request not in approved_plan(settings):
        raise StoreConflict("QC interval is outside the approved year.")
    output = bundle_path(settings, request)
    if output.exists():
        return verify_bundle(settings, request, entry)
    config, districts = geography()
    raw_path = workspace_path(entry["raw_file"])
    if sha256(raw_path) != entry["sha256"]:
        raise StoreConflict("Raw hash changed before scientific QC.")
    raw = load_firms_csv(raw_path)
    pilot_columns = load_firms_csv(workspace_path(config["dbscan_pilot"]["raw_csv"])).columns.tolist()
    if set(raw.columns) != set(pilot_columns):
        raise StoreConflict("Historical response source schema differs from the reviewed pilot.")
    validation = validate_csv(raw_path, districts, config, dataset_kind=f"authentic_snpp_sp_{settings['study']['stage']}_{year}")
    records = validation.records.join(observation_index(raw, entry["sha256"]))
    records["agniscope_request_id"] = request["request_id"]
    records["agniscope_raw_sha256"] = entry["sha256"]
    records["agniscope_raw_file"] = entry["raw_file"]
    records["agniscope_utc_date"] = records.agniscope_acquired_utc.dt.strftime("%Y-%m-%d")
    records["agniscope_ist_date"] = records.agniscope_acquired_ist.dt.strftime("%Y-%m-%d")
    records = records.join(season_membership(records.agniscope_utc_date, year))
    issues = issue_register(settings)
    records["agniscope_upstream_issue_ids"] = records.agniscope_acquired_utc.map(lambda value: issue_ids(value, issues))
    records["agniscope_satellite_opportunity"] = "unverified"
    accepted = records.loc[records.agniscope_disposition.eq("retained")].copy()
    points = gpd.GeoDataFrame(accepted, geometry=gpd.points_from_xy(accepted.agniscope_longitude, accepted.agniscope_latitude), crs="EPSG:4326")
    accepted = accepted.join(sample_landcover(points, config))
    accepted = standardize(accepted, pilot_columns)
    flags = {r["rule"]: r["flagged_including_other_failures"] for r in validation.summary["funnel"]}
    missing = {column: int(raw[column].astype("string").str.strip().eq("").sum()) for column in raw}
    daily = []
    for day in dates(date.fromisoformat(request["start_utc_date"]), date.fromisoformat(request["end_utc_date"])):
        text = str(day)
        count = int(raw.acq_date.eq(text).sum())
        concerns = [r["issue_id"] for r in issues if r["start_utc"][:10] <= text <= (r["end_utc"] or r["start_utc"])[:10]]
        daily.append({"utc_date": text, "request_id": request["request_id"], "raw_observations": count,
                      "retained_observations": int(accepted.acq_date.eq(text).sum()),
                      "retrieval_state": "positive_detections" if count else "zero_returned_detections",
                      "source": "VIIRS_SNPP_SP", "collection_values": ";".join(sorted(raw.loc[raw.acq_date.eq(text), "version"].unique())),
                      "upstream_concern_ids": ";".join(concerns), "satellite_coverage": "unknown"})
    passed = bool(validation.summary["all_input_rows_accounted_for"] and validation.summary["invalid_rows"] == 0 and all(missing[c] == 0 for c in REQUIRED_COLUMNS))
    summary = {"request_id": request["request_id"], "raw_sha256": entry["sha256"], "raw_rows": len(raw), "retained_rows": len(accepted),
               "rows_with_missing_required": int(np.column_stack([raw[c].astype("string").str.strip().eq("").to_numpy(dtype=bool) for c in REQUIRED_COLUMNS]).any(axis=1).sum()),
               "qc_passed": passed, "qc_summary_file": (output / "qc_summary.json").relative_to(workspace_path(".")).as_posix(),
               "source_columns": raw.columns.tolist(), "missing_source_values": missing, "flags": flags,
               "qc_contract": validation.summary, "raw_population": describe_population(records), "retained_population": describe_population(accepted),
               "district_counts": _counts(accepted.agniscope_district_original), "district_assignment": _counts(accepted.agniscope_district_assignment),
               "landcover_classes": _counts(accepted.agniscope_worldcover_2021_class), "landcover_status": _counts(accepted.agniscope_landcover_status),
               "daily": daily, "version_values": _counts(raw.version), "type_lineage_resolved": False,
               "satellite_completeness": "unverified", "generated_utc": utc_now()}
    if len(records) != len(accepted) + sum(r["removed_first_failure"] for r in validation.summary["funnel"]):
        raise StoreConflict("Per-interval row accounting failed.")
    staging = output.with_name(output.name + ".qc_staged")
    if staging.exists():
        raise StoreConflict("An uncommitted QC staging bundle exists; no overwrite permitted.")
    staging.mkdir(parents=True, exist_ok=False)
    # The raw CSV preserves all source values. This compact ledger indexes all rows.
    ledger_columns = [c for c in records if c.startswith("agniscope_")]
    records[ledger_columns].to_parquet(staging / "row_ledger.parquet", index=False)
    accepted.to_parquet(staging / "retained.parquet", index=False)
    atomic_json(staging / "qc_summary.json", summary)
    receipt = {"raw_sha256": entry["sha256"], "request_id": request["request_id"], "self_excluded": True,
               "files": [{"path": (output / p.name).relative_to(workspace_path(".")).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(staging.iterdir())]}
    atomic_json(staging / "receipt.json", receipt)
    if output.exists():
        raise StoreConflict("QC destination appeared during staging; existing evidence preserved.")
    staging.rename(output)
    return summary


def write_table(path, rows, columns=None):
    import csv
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_rows(settings, catalog, plan):
    rows = []
    for request in plan:
        entry = catalog["intervals"][request["request_id"]]
        completed = entry["acquisition_status"] in {"acquired_positive", "acquired_zero"} and entry["qc_status"] == "passed"
        summary = verify_bundle(settings, request, entry) if completed else None
        mapped = {r["utc_date"]: r for r in summary["daily"]} if summary else {}
        for day in dates(date.fromisoformat(request["start_utc_date"]), date.fromisoformat(request["end_utc_date"])):
            text = str(day)
            row = mapped.get(text, {"utc_date": text, "request_id": request["request_id"], "raw_observations": None,
                                    "retained_observations": None, "retrieval_state": entry["acquisition_status"],
                                    "source": "VIIRS_SNPP_SP", "collection_values": None,
                                    "upstream_concern_ids": "", "satellite_coverage": "unknown"})
            rows.append({**row, "interval_start": request["start_utc_date"], "interval_end": request["end_utc_date"],
                         "http_status": entry.get("http", {}).get("http_status"), "qc_status": entry["qc_status"],
                         "provenance_status": "hash_verified" if completed else "not_verified",
                         "failure_category": entry.get("failure_category"), "raw_sha256": entry.get("sha256"),
                         "upstream_calendar_status": "nonexhaustive_unverified", "local_clear_sky_opportunity": "unknown"})
    expected = stage_profile(settings)["days"]
    if len(rows) != expected or len({r["utc_date"] for r in rows}) != expected:
        raise StoreConflict("Daily coverage table does not contain the approved unique UTC dates.")
    return rows


def frp_statistics(values):
    values = np.asarray(values, dtype="float64")
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [.25, .5, .75])
    return {"count": len(values), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "q25": float(quantiles[0]), "median": float(quantiles[1]), "q75": float(quantiles[2])}


def remember_identity(connection, fingerprint, observation_id, request_id):
    """Disk-backed first-occurrence index; duplicate occurrences remain in the ledger."""
    previous = connection.execute("SELECT observation_id, request_id FROM fingerprints WHERE payload_sha256=?", (fingerprint,)).fetchone()
    if previous is None:
        connection.execute("INSERT INTO fingerprints VALUES(?,?,?)", (fingerprint, observation_id, request_id))
    return previous


def integrate_year(settings, catalog, plan):
    profile = stage_profile(settings)
    year, expected = profile["year"], profile["days"]
    if plan != approved_plan(settings):
        raise StoreConflict("Annual integration plan is not canonical.")
    output = workspace_path(settings["paths"]["annual"], writable=True)
    if output.exists():
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        sources = {r["request_id"]: catalog["intervals"][r["request_id"]]["sha256"] for r in plan}
        if receipt["source_hashes"] != sources:
            raise StoreConflict("Existing annual dataset references different source hashes.")
        for item in receipt["files"]:
            if sha256(workspace_path(item["path"])) != item["sha256"]:
                raise StoreConflict("Annual dataset file hash mismatch.")
        return json.loads((output / "annual_qc_summary.json").read_text(encoding="utf-8")), True
    if len(plan) != profile["intervals"] or any(e["acquisition_status"] not in {"acquired_positive", "acquired_zero"} or e["qc_status"] != "passed" for e in catalog["intervals"].values()):
        raise StoreConflict("All approved intervals must pass QC before complete annual integration.")
    staging = output.with_name(output.name + ".integration_staged")
    if staging.exists():
        raise StoreConflict("Uncommitted annual integration exists; preserve staging for reconciliation.")
    staging.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(staging / "source_identity_index.sqlite")
    db.execute("CREATE TABLE fingerprints(payload_sha256 TEXT PRIMARY KEY, observation_id TEXT, request_id TEXT)")
    db.execute("CREATE TABLE observations(observation_id TEXT PRIMARY KEY, request_id TEXT, source_record INTEGER, payload_sha256 TEXT, cross_interval_duplicate INTEGER)")
    counts = Counter()
    raw_conf, raw_dn, accepted_conf, accepted_dn, district, landcover, cover_status, versions = (Counter() for _ in range(8))
    raw_frp, accepted_frp = array("d"), array("d")
    monthly = {f"{year}-{month:02d}": {"month": f"{year}-{month:02d}", "raw_observations": 0, "retained_observations": 0} for month in range(1, 13)}
    seasons = {name: {"raw_observations": 0, "retained_observations": 0} for name in (f"CORE_{year}", f"OUTSIDE_CORE_{year}", f"EXT_{year-1}_{year}_JAN_JUN_PARTIAL", f"EXT_{year}_{year+1}_NOV_DEC_PARTIAL")}
    duplicates, collection_audit, interval_accounting = [], [], []
    cross_removed_daily = Counter()
    writer = None
    schema = None
    try:
        for request in plan:
            entry = catalog["intervals"][request["request_id"]]
            summary = verify_bundle(settings, request, entry)
            raw = load_firms_csv(workspace_path(entry["raw_file"]))
            ledger = pd.read_parquet(bundle_path(settings, request) / "row_ledger.parquet")
            accepted = pd.read_parquet(bundle_path(settings, request) / "retained.parquet")
            cross_ids = set()
            for row in ledger.itertuples(index=False):
                previous = remember_identity(db, row.agniscope_source_payload_sha256, row.agniscope_observation_id, request["request_id"])
                cross = previous is not None and previous[1] != request["request_id"]
                db.execute("INSERT INTO observations VALUES(?,?,?,?,?)", (row.agniscope_observation_id, request["request_id"], int(row.agniscope_source_record), row.agniscope_source_payload_sha256, int(cross)))
                if cross:
                    cross_ids.add(row.agniscope_observation_id)
                    duplicates.append({"duplicate_observation_id": row.agniscope_observation_id, "canonical_observation_id": previous[0], "request_id": request["request_id"], "payload_sha256": row.agniscope_source_payload_sha256})
            before_cross = len(accepted)
            cross_removed_daily.update(accepted.loc[accepted.agniscope_observation_id.isin(cross_ids), "acq_date"])
            accepted = accepted.loc[~accepted.agniscope_observation_id.isin(cross_ids)].copy()
            db.commit()
            table = pa.Table.from_pandas(accepted, preserve_index=False).replace_schema_metadata(None)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(staging / f"validated_observations_{year}.parquet", schema, compression="zstd")
            writer.write_table(table.cast(schema))
            counts.update(raw_rows=len(raw), retained_rows=len(accepted), interval_retained_before_cross=before_cross,
                          cross_interval_duplicate_occurrences=len(cross_ids), cross_interval_duplicate_retained_removed=before_cross-len(accepted),
                          invalid_rows=summary["qc_contract"]["invalid_rows"], within_interval_duplicates=summary["qc_contract"]["duplicate_rows_removed"],
                          scientific_selection_exclusions=summary["qc_contract"]["scientific_selection_exclusions"],
                          geographic_exclusions=next(r["removed_first_failure"] for r in summary["qc_contract"]["funnel"] if r["rule"] == "outside_footprint"),
                          missing_required_rows=summary["rows_with_missing_required"],
                          unassigned_retained=int(accepted.agniscope_district_assignment.eq("unassigned_inside").sum()),
                          ambiguous_retained=int(accepted.agniscope_district_assignment.eq("ambiguous_shared_edge").sum()))
            for key in ("missing_count", "unusable_nonmissing_count", "zero_count"):
                counts["raw_frp_" + key] += summary["raw_population"]["frp"][key]
            for key, value in describe_population(accepted)["frp"].items():
                if key in {"missing_count", "unusable_nonmissing_count", "zero_count"}:
                    counts["retained_frp_" + key] += value
            raw_conf.update(summary["raw_population"]["confidence"]); raw_dn.update(summary["raw_population"]["daynight"])
            accepted_conf.update(_counts(accepted.agniscope_confidence)); accepted_dn.update(_counts(accepted.daynight))
            district.update(_counts(accepted.agniscope_district_original)); landcover.update(_counts(accepted.agniscope_worldcover_2021_class))
            cover_status.update(_counts(accepted.agniscope_landcover_status)); versions.update(_counts(raw.version))
            for population, values in ((raw, raw_frp), (accepted, accepted_frp)):
                numeric = pd.to_numeric(population.frp, errors="coerce").to_numpy(dtype=float)
                values.extend(numeric[np.isfinite(numeric) & (numeric >= 0)])
            for frame, column in ((raw, "raw_observations"), (accepted, "retained_observations")):
                for month, n in frame.acq_date.str[:7].value_counts().items():
                    monthly[str(month)][column] += int(n)
                core = frame.acq_date.between(f"{year}-02-15", f"{year}-06-15")
                seasons[f"CORE_{year}"][column] += int(core.sum())
                seasons[f"OUTSIDE_CORE_{year}"][column] += int((~core).sum())
                seasons[f"EXT_{year-1}_{year}_JAN_JUN_PARTIAL"][column] += int(frame.acq_date.le(f"{year}-06-30").sum())
                seasons[f"EXT_{year}_{year+1}_NOV_DEC_PARTIAL"][column] += int(frame.acq_date.ge(f"{year}-11-01").sum())
            collection_audit.append({"request_id": request["request_id"], "start": request["start_utc_date"], "end": request["end_utc_date"],
                                     "source": "VIIRS_SNPP_SP", "literal_version_counts": json.dumps(_counts(raw.version), sort_keys=True),
                                     "raw_rows": len(raw), "collection_compatible": True, "upstream_revision_lineage": "not_independently_verified"})
            interval_accounting.append({"request_id": request["request_id"], "raw_sha256": entry["sha256"], "raw_rows": len(raw),
                                        "retained_before_cross": before_cross, "retained_final": len(accepted),
                                        "cross_duplicate_retained_removed": before_cross-len(accepted),
                                        "local_qc_exclusions": len(raw)-before_cross, "reused_pilot": entry["reused_existing_pilot"]})
        writer.close(); writer = None
        db.close()
        daily = coverage_rows(settings, catalog, plan)
        for row in daily:
            row["cross_interval_duplicate_retained_removed"] = cross_removed_daily[row["utc_date"]]
            row["retained_observations"] -= cross_removed_daily[row["utc_date"]]
        if counts["raw_rows"] != counts["retained_rows"] + counts["invalid_rows"] + counts["within_interval_duplicates"] + counts["scientific_selection_exclusions"] + counts["cross_interval_duplicate_retained_removed"]:
            raise StoreConflict("Annual raw-to-retained exclusion accounting failed.")
        config, districts = geography()
        district_rows = [{"district_original": row.district_original, "display_name": row.agniscope_display_name,
                          "retained_observations": district[row.district_original]} for row in districts.itertuples()]
        if sum(r["retained_observations"] for r in district_rows) + counts["unassigned_retained"] + counts["ambiguous_retained"] != counts["retained_rows"]:
            raise StoreConflict("Annual district totals do not reconcile.")
        summary = {"generated_utc": utc_now(), f"stage_{profile['stage'].lower()}_status": "COMPLETE", "meaning": "Complete approved API retrieval and declared row QC; satellite coverage remains unverified",
                   "source": "VIIRS_SNPP_SP", "utc_start": f"{year}-01-01", "utc_end": f"{year}-12-31", "counts": dict(counts),
                   "logical_intervals": profile["intervals"], "reused_pilot_intervals": int(profile["pilot"]), "unique_new_intervals": len({r["request_id"] for r in catalog["attempt_reservations"]}),
                   "http_observation_attempt_slots": len(catalog["attempt_reservations"]), "raw_confidence": dict(raw_conf), "retained_confidence": dict(accepted_conf),
                   "raw_daynight": dict(raw_dn), "retained_daynight": dict(accepted_dn), "district_counts": dict(district),
                   "raw_frp": frp_statistics(raw_frp), "retained_frp": frp_statistics(accepted_frp),
                   "landcover_classes": dict(landcover), "landcover_status": dict(cover_status), "literal_version_counts": dict(versions),
                   "season_membership": seasons, "extended_seasons_complete": False,
                   "coverage": {"expected_dates": expected, "validated_dates": len(daily), "raw_positive_dates": sum(r["raw_observations"] > 0 for r in daily),
                                "raw_zero_dates": sum(r["raw_observations"] == 0 for r in daily), "retained_positive_dates": sum(r["retained_observations"] > 0 for r in daily),
                                "failed_dates": 0, "satellite_opportunity_unknown_dates": expected, "upstream_flagged_dates": sum(bool(r["upstream_concern_ids"]) for r in daily)},
                   "observation_crs": "EPSG:4326", "analysis_crs": "EPSG:32644", "landcover_reference": "ESA WorldCover 2021 v200",
                   "type_lineage_resolved": False, "clustering_executed": False, "satellite_completeness": "unverified"}
        write_table(staging / f"daily_coverage_{year}.csv", daily)
        write_table(staging / "monthly_counts.csv", list(monthly.values()))
        write_table(staging / "district_summary.csv", district_rows)
        write_table(staging / "source_collection_audit.csv", collection_audit)
        write_table(staging / "interval_row_accounting.csv", interval_accounting)
        write_table(staging / "cross_interval_duplicates.csv", duplicates, ["duplicate_observation_id", "canonical_observation_id", "request_id", "payload_sha256"])
        atomic_json(staging / "annual_qc_summary.json", summary)
        receipt = {"self_excluded": True, "source_hashes": {r["request_id"]: catalog["intervals"][r["request_id"]]["sha256"] for r in plan},
                   "files": [{"path": (output / p.name).relative_to(workspace_path(".")).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(staging.iterdir())]}
        atomic_json(staging / "receipt.json", receipt)
        staging.rename(output)
        return summary, False
    finally:
        if writer is not None:
            writer.close()
        try:
            db.close()
        except Exception:
            pass
