"""One resolution at a time: seven-season recurrence and matched-support summaries."""
from pathlib import Path
import argparse
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from agniscope.provenance import sha256
from agniscope.acquisition_store import atomic_json, StoreConflict
from phase3b_grid import TABLE, SPATIAL, verify_receipt, seal, geo_metadata


def write_frame(frame, path, *, geo=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    if geo:
        from pyproj import CRS
        table = table.replace_schema_metadata(geo_metadata(CRS.from_epsg(32644)))
    pq.write_table(table, path, compression="zstd", row_group_size=4096)


def build(resolution):
    target = TABLE/"recurrence"/f"r{resolution}"
    if target.exists(): return {"reused": True, "receipt": verify_receipt(target)}
    from phase3b_certificates import require_certificate
    require_certificate(TABLE/"pytest_grid_certified.xml", TABLE/"pytest_grid_certified.certificate.json")
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from agniscope import recurrence as r
    assert resolution in r.RESOLUTIONS and json.loads((TABLE/"gate_b.json").read_text())["status"] == "passed"
    assert all(json.loads((TABLE/("gate_"+gate+".json")).read_text())["status"] == "passed" for gate in ("c", "d"))
    directory = SPATIAL/f"grid_{resolution}"
    verify_receipt(directory)
    grid = pd.read_parquet(directory/"grid.parquet", columns=["cell_id", "grid_x", "grid_y", "clipped_area_m2"])
    parts = pd.read_parquet(directory/"district_intersections.parquet")
    assert not parts.duplicated(["cell_id", "district_original"]).any()
    areas = dict(zip(grid.cell_id, grid.clipped_area_m2))
    total_area = math.fsum(areas.values())
    inputs = {}
    frames = []
    for year in r.YEARS:
        season = TABLE/"activity"/f"CORE_{year}"
        verify_receipt(season)
        inputs[str(year)] = sha256(season/"receipt.json")
        frames.append(pd.read_parquet(season/"seasonal_activity.parquet", filters=[("resolution_m", "=", resolution)]))
    activity = pd.concat(frames, ignore_index=True)
    del frames
    assert set(activity.year) == set(r.YEARS)
    assert not activity.duplicated(["subset", "method", "year", "cell_id"]).any()
    staging = target.with_name(target.name+".staged")
    if staging.exists(): raise StoreConflict("Preserve uncommitted recurrence staging.")
    staging.mkdir(parents=True)
    wide = grid[["cell_id"]].copy()
    wide_index = pd.Index(wide.cell_id)
    recurrences = {}
    sparse_frames = []
    distributions = []
    thresholds = []
    district_rows = []
    district_agreement = []
    method_comparison = []
    agreement_categories = []
    district_base = parts.groupby(["district_original", "district_display"], sort=True).agg(
        eligible_cells=("cell_id", "nunique"), eligible_area_m2=("intersection_area_m2", "sum")).reset_index()
    district_names = district_base.district_original.tolist()
    for population in r.POPULATIONS:
        for method in r.METHODS:
            selected = activity.loc[activity.subset.eq(population) & activity.method.eq(method)]
            rec = r.recurrent_cells(selected)
            assert set(rec.cell_id) <= set(areas)
            recurrences[population, method] = rec
            prefix = population+"_"+method
            positions = wide_index.get_indexer(rec.cell_id)
            assert (positions >= 0).all()
            for column, dtype in [("active_seasons", "uint8"), ("season_mask", "uint8"),
                                  ("detections", "int32"), ("distinct_UTC_dates", "int16")]:
                values = np.zeros(len(wide), dtype=dtype)
                values[positions] = rec[column].to_numpy(dtype)
                wide[prefix+"_"+column] = values
            sparse_frames.append(rec.assign(subset=population, method=method, resolution_m=resolution))
            distributions += [dict(resolution_m=resolution, subset=population, method=method, **v)
                              for v in r.frequency_distribution(rec, areas)]
            part_counts = parts.cell_id.map(rec.set_index("cell_id").active_seasons).fillna(0).to_numpy("uint8")
            active_by_district = parts.loc[part_counts >= 1].groupby("district_original").cell_id.nunique().to_dict()
            for k in (1, 2, 3, 4, 5):
                ids = r.threshold_ids(rec, k)
                clipped_area = math.fsum(areas[c] for c in ids)
                if k >= 2:
                    thresholds.append({"resolution_m": resolution, "subset": population, "method": method, "k": k,
                        "recurrent_cells": len(ids), "eligible_cells": len(grid), "eligible_cell_share": len(ids)/len(grid),
                        "recurrent_area_km2": clipped_area/1e6, "eligible_area_km2": total_area/1e6,
                        "eligible_area_percent": clipped_area/total_area*100})
                selected_parts = parts.loc[part_counts >= k]
                counts = selected_parts.groupby("district_original").cell_id.nunique().to_dict()
                sums = selected_parts.groupby("district_original").intersection_area_m2.sum().to_dict()
                for row in district_base.itertuples():
                    area = float(sums.get(row.district_original, 0.))
                    district_rows.append({"resolution_m": resolution, "subset": population, "method": method, "k": k,
                        "district_original": row.district_original, "district_display": row.district_display,
                        "eligible_cells": int(row.eligible_cells), "eligible_area_km2": row.eligible_area_m2/1e6,
                        "active_at_least_one_season_cells": int(active_by_district.get(row.district_original, 0)),
                        "qualifying_cells": int(counts.get(row.district_original, 0)), "qualifying_area_km2": area/1e6,
                        "district_area_percent": 100*area/row.eligible_area_m2})
        for method in ("dbscan", "hdbscan"):
            r.assert_supported(recurrences[population, "detection"], recurrences[population, method])
        for k in (2, 3, 4, 5):
            sets = {method: r.threshold_ids(recurrences[population, method], k) for method in r.METHODS}
            assert sets["dbscan"] <= sets["detection"] and sets["hdbscan"] <= sets["detection"]
            for left, right in (("detection", "dbscan"), ("detection", "hdbscan"), ("dbscan", "hdbscan")):
                method_comparison.append(dict(resolution_m=resolution, subset=population, k=k, left=left, right=right,
                                              **r.set_comparison(sets[left], sets[right], areas)))
            codes = np.zeros(len(grid), dtype="uint8")
            for bit, method in enumerate(r.METHODS):
                codes[np.isin(grid.cell_id.to_numpy(), list(sets[method]))] += (1 << bit)
            assert not np.isin(codes, [2, 4, 6]).any()
            for code in range(8):
                selection = codes == code
                agreement_categories.append({"resolution_m": resolution, "subset": population, "k": k,
                    "support_code": code, "detection": bool(code&1), "dbscan": bool(code&2), "hdbscan": bool(code&4),
                    "cells": int(selection.sum()), "clipped_area_km2": math.fsum(grid.clipped_area_m2.to_numpy()[selection])/1e6})
            part_codes = parts.cell_id.map(pd.Series(codes, index=grid.cell_id)).to_numpy("uint8")
            all_three = parts.loc[part_codes == 7].groupby("district_original").intersection_area_m2.sum().to_dict()
            detection_only = parts.loc[part_codes == 1].groupby("district_original").intersection_area_m2.sum().to_dict()
            db_only = parts.loc[part_codes == 3].groupby("district_original").intersection_area_m2.sum().to_dict()
            hdb_only = parts.loc[part_codes == 5].groupby("district_original").intersection_area_m2.sum().to_dict()
            for name in district_names:
                district_agreement.append({"resolution_m": resolution, "subset": population, "k": k, "district_original": name,
                    "all_three_area_km2": all_three.get(name, 0.)/1e6, "detection_only_area_km2": detection_only.get(name, 0.)/1e6,
                    "dbscan_without_hdbscan_area_km2": db_only.get(name, 0.)/1e6,
                    "hdbscan_without_dbscan_area_km2": hdb_only.get(name, 0.)/1e6})
    for population in ("grid_day", "nominal", "tree"):
        r.assert_supported(recurrences["all", "detection"], recurrences[population, "detection"])
    if resolution in (1000, 2000):
        for field in ("cell_id", "season_mask", "active_seasons"):
            assert recurrences["all", "detection"][field].tolist() == recurrences["grid_day", "detection"][field].tolist()
    sparse = pd.concat(sparse_frames, ignore_index=True).sort_values(["subset", "method", "cell_id"], kind="stable").reset_index(drop=True)
    write_frame(sparse, staging/"cell_recurrence.parquet")
    write_frame(wide, staging/"eligible_cell_activity.parquet")
    for name, values in (("frequency_distribution", distributions), ("threshold_comparison", thresholds),
                         ("district_summary", district_rows), ("district_agreement", district_agreement),
                         ("method_overlap", method_comparison), ("three_method_agreement", agreement_categories)):
        pd.DataFrame(values).to_csv(staging/(name+".csv"), index=False, lineterminator="\n")
    # Geometries are stored only for active baseline cells; all zero cells remain
    # explicit in eligible_cell_activity and in the immutable full grid reference.
    active_ids = recurrences["all", "detection"].cell_id.to_numpy()
    spatial = pd.read_parquet(directory/"grid.parquet", filters=[("cell_id", "in", active_ids.tolist())])
    baseline_columns = [name for name in wide if name == "cell_id" or name.startswith("all_")]
    spatial = spatial.merge(wide[baseline_columns], on="cell_id", validate="one_to_one")
    for k in (2, 3, 4, 5):
        spatial[f"agreement_k{k}"] = sum((spatial[f"all_{method}_active_seasons"] >= k).astype("uint8")*(1 << bit)
                                         for bit, method in enumerate(r.METHODS))
    write_frame(spatial, staging/"recurrence_active.geoparquet", geo=True)
    metadata = {"resolution_m": resolution, "eligible_cells": len(grid), "eligible_area_km2": total_area/1e6,
                "sparse_activity_rows": len(activity), "recurrence_rows": len(sparse), "views": 12,
                "all_seven_seasons_verified": True, "primary_season_denominator": 7,
                "cluster_superset_violations": 0, "source_hashes": inputs,
                "grid_receipt_sha256": sha256(directory/"receipt.json"), "satellite_observation_opportunity": "unverified",
                "zero_policy": "All eligible cells explicitly carry zero masks/counts when no supporting rows returned; not proven fire absence.",
                "spatial_active_layer_policy": "Baseline active cells only; join immutable full grid or explicit eligible_cell_activity for zero cells.",
                "district_cell_counts_additive": False, "district_areas_additive": True,
                "subset_clustering": "Saved matching subset-refit assignments, not filtered full-population labels."}
    atomic_json(staging/"summary.json", metadata)
    seal(staging, target, {"kind": "seven_primary_season_recurrence", "resolution_m": resolution,
        "worker_sha256": sha256(Path(__file__)), "kernel_sha256": sha256(ROOT/"src/agniscope/recurrence.py"),
        "methods_sha256": sha256(ROOT/"configs/phase3b_methods.toml"), "activity_receipts": inputs})
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("resolution", type=int, choices=(500, 1000, 2000))
    args = parser.parse_args()
    print(json.dumps(build(args.resolution), indent=2))
