"""Run only the authentic five-day Phase 1A experiment; no network operations."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import get_parts
from threadpoolctl import threadpool_limits

from agniscope.config import load_config, workspace_path
from agniscope.dbscan import (ID, UTC, XY, DISTRICT, prepare_points, sensitivity_subsets,
    run_grid, repeat_diagnostics, distance_crosscheck, membership_comparison)
from agniscope.firms import load_firms_csv, validate_csv
from agniscope.provenance import sha256, verify_sources, write_json, utc_now
from agniscope.resources import resource_snapshot
from agniscope.spatial import load_boundaries


def check_protected(preflight, config):
    changed = [item["path"] for item in preflight["protected_files"]
               if not workspace_path(item["path"]).is_file()
               or workspace_path(item["path"]).stat().st_size != item["size_bytes"]
               or sha256(workspace_path(item["path"])) != item["sha256"]]
    sources = verify_sources(config)
    if changed or not sources["all_unchanged"]:
        raise ValueError(f"Protected evidence changed; stop and investigate: {changed}")
    return {"all_protected_phase0_files_unchanged": True,
            "protected_phase0_file_count": len(preflight["protected_files"]),
            "original_gis_14_unchanged": len(sources["files"]) == 14 and sources["all_unchanged"]}


def load_pilot(config):
    settings = config["dbscan_pilot"]
    path = workspace_path(settings["raw_csv"])
    if sha256(path) != settings["raw_sha256"]:
        raise ValueError("Authentic raw hash mismatch; stop and investigate.")
    _, districts = load_boundaries(config)
    qc = validate_csv(path, districts, config, dataset_kind="authentic_nasa_pilot_phase1a_recheck")
    accepted = qc.accepted.sort_values(ID).reset_index(drop=True)
    enriched = pd.read_parquet(workspace_path(settings["enriched_parquet"])).sort_values(ID).reset_index(drop=True)
    raw_columns = load_firms_csv(path).columns.tolist()
    if len(qc.records) != settings["expected_raw_rows"] or len(accepted) != settings["expected_retained_rows"] or len(enriched) != len(accepted):
        raise ValueError("613/346 input contract mismatch; stop and investigate.")
    for column in [*raw_columns, ID, DISTRICT, "agniscope_confidence", "agniscope_disposition", *XY]:
        if accepted[column].astype(str).tolist() != enriched[column].astype(str).tolist():
            raise ValueError(f"Saved accepted data differs from fresh QC: {column}")
    points = prepare_points(enriched, settings["analysis_crs"])
    if not points.agniscope_landcover_status.eq("sampled").all() or points.agniscope_worldcover_2021_class.isna().any():
        raise ValueError("Existing land-cover context is incomplete; no new raster extraction authorized.")
    footprint = gpd.read_file(workspace_path(config["paths"]["analysis_footprint"]), layer="analysis_footprint", engine="pyogrio")
    if len(footprint) != 1 or footprint.crs != districts.crs or not footprint.geometry.iloc[0].equals(districts.geometry.union_all()):
        raise ValueError("Saved analysis footprint is not the approved district union.")
    if not footprint.geometry.iloc[0].covers(gpd.points_from_xy(points[XY[0]], points[XY[1]])).all():
        raise ValueError("Experiment coordinates fall outside the approved footprint.")
    return points, districts, footprint, qc.summary, raw_columns


def main():
    start = time.perf_counter()
    resources_start = resource_snapshot()
    config = load_config()
    settings = config["dbscan_pilot"]
    if (settings["algorithm"], settings["metric"], settings["n_jobs"]) != ("kd_tree", "euclidean", 1):
        raise ValueError("Implementation requires the declared single-worker Euclidean kd_tree settings.")
    preflight = json.loads(workspace_path(settings["preflight"]).read_text(encoding="utf-8"))
    integrity_start = check_protected(preflight, config)
    points, districts, footprint, qc, raw_columns = load_pilot(config)
    subsets, sample_ledger = sensitivity_subsets(points, settings["temporal_grid_m"], settings["temporal_grid_origin_xy_m"])
    assert len(subsets["nominal_only"]) == settings["expected_nominal_rows"]
    assert len(subsets["tree_cover_only"]) == settings["expected_tree_rows"]
    dates = pd.date_range(config["firms"]["pilot_start"], periods=config["firms"]["pilot_days"]).strftime("%Y-%m-%d").tolist()
    district_names = districts[config["spatial"]["district_original_column"]].sort_values().tolist()
    outputs = []
    with threadpool_limits(limits=1):
        for name, subset in subsets.items():
            outputs.append(run_grid(subset, settings["eps_km"], settings["min_samples"], name, district_names, dates))
        check_distances, selected_distances = distance_crosscheck(points, [x * 1000 for x in settings["eps_km"]], settings["earth_radius_m_for_haversine_check"])
    combined = {key: pd.concat([output[key] for output in outputs], ignore_index=True) for key in outputs[0]}
    parameters = combined["parameters"]
    assignments = combined["assignments"]
    baseline = parameters.loc[parameters.dataset.eq("baseline")]
    if len(baseline) != 9 or len(parameters) != 36:
        raise ValueError("Declared nine-setting grid was not reproduced for all four populations.")
    stability = []
    for row in parameters.itertuples(index=False):
        query = assignments.eps_km.eq(row.eps_km) & assignments.min_samples.eq(row.min_samples)
        reference = assignments.loc[query & assignments.dataset.eq("baseline")]
        alternative = assignments.loc[query & assignments.dataset.eq(row.dataset)]
        stability.append({"dataset": row.dataset, "eps_km": row.eps_km, "min_samples": row.min_samples,
                          **membership_comparison(reference, alternative)})
    sensitivity = parameters.merge(pd.DataFrame(stability), on=["dataset", "eps_km", "min_samples"], validate="one_to_one")
    parameter_stability = []
    from itertools import combinations
    for a, b in combinations(list(baseline.itertuples(index=False)), 2):
        left = assignments.loc[assignments.dataset.eq("baseline") & assignments.eps_km.eq(a.eps_km) & assignments.min_samples.eq(a.min_samples)]
        right = assignments.loc[assignments.dataset.eq("baseline") & assignments.eps_km.eq(b.eps_km) & assignments.min_samples.eq(b.min_samples)]
        parameter_stability.append({"eps_a_km": a.eps_km, "min_samples_a": a.min_samples, "eps_b_km": b.eps_km,
                                    "min_samples_b": b.min_samples, **membership_comparison(left, right)})
    pairs, repeat_counts, repeat_summary = repeat_diagnostics(points, settings["close_pair_thresholds_m"])
    target = workspace_path(settings["output_dir"], writable=True)
    target.mkdir(parents=True, exist_ok=True)
    tables = {
        "parameter_comparison": baseline, "sensitivity_comparison": sensitivity,
        "point_assignments": assignments, "cluster_summary": combined["clusters"],
        "temporal_sampling_ledger": sample_ledger, "close_observation_pairs": pairs,
        "close_pair_counts": repeat_counts, "selected_distance_checks": selected_distances,
        "parameter_membership_stability": pd.DataFrame(parameter_stability),
        **{f"diagnostic_{name}": combined[name] for name in ["district", "daily", "daynight", "cluster_daily", "cluster_daynight", "frp"]},
    }
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False)
    # JSON geometry allows an existing plotting-only installation to render figures.
    # Fits, projections and all scientific measurements stay in the isolated environment.
    line_coordinates = lambda geometry: [np.asarray(part.coords).tolist() for part in get_parts(geometry.boundary)]
    plot_data = {
        "crs": settings["analysis_crs"],
        "footprint_boundary_xy_m": line_coordinates(footprint.geometry.iloc[0]),
        "district_boundary_xy_m": [line for geom in districts.geometry for line in line_coordinates(geom)],
        "district_labels": [{"name": row.agniscope_display_name, "x_m": row.geometry.representative_point().x,
                             "y_m": row.geometry.representative_point().y} for row in districts.itertuples()],
    }
    write_json(target / "plot_geometry.json", plot_data)
    nearest = cKDTree(points[XY].to_numpy()).query(points[XY].to_numpy(), k=2)[0][:, 1]
    import sklearn, scipy, pyproj, shapely, pyarrow
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "pyproj": pyproj.__version__,
                "geopandas": gpd.__version__, "shapely": shapely.__version__, "pyarrow": pyarrow.__version__}
    integrity_end = check_protected(preflight, config)
    result = {
        "generated_utc": utc_now(), "settings": settings, "versions": versions,
        "python_executable": sys.executable, "fits_executed": len(parameters),
        "input_contract": {"raw_rows": len(points) + qc["scientific_selection_exclusions"], "retained_rows": len(points),
                           "raw_sha256": settings["raw_sha256"], "raw_source_columns_preserved": raw_columns,
                           "input_enriched_sha256": sha256(workspace_path(settings["enriched_parquet"])),
                           "population_sizes": {name: len(subset) for name, subset in subsets.items()}},
        "qc_summary": qc, "integrity_start": integrity_start, "integrity_end": integrity_end,
        "repeated_coordinates": repeat_summary, "distance_crosscheck": check_distances,
        "nearest_other_detection_m": {name: float(np.quantile(nearest, q)) for name, q in [("minimum", 0), ("q25", .25), ("median", .5), ("q75", .75), ("maximum", 1)]},
        "subset_utc_counts": {name: {date: int(subset.utc_date.eq(date).sum()) for date in dates} for name, subset in subsets.items()},
        "source_code_sha256": {name: sha256(workspace_path(name)) for name in ["src/agniscope/dbscan.py", "scripts/phase1a.py", "configs/project.toml"]},
        "wall_seconds": time.perf_counter() - start, "resources_start": resources_start, "resources_end": resource_snapshot(),
        "interpretation": "Five-day coordinate-only thermal detection concentrations; no fire incidents or long-term hotspots inferred.",
        "csv_missing_result_convention": "Empty numeric cells mean unavailable; silhouette_status explains why.",
    }
    write_json(target / "experiment.json", result)
    print(json.dumps({"retained_rows": len(points), "population_sizes": result["input_contract"]["population_sizes"],
                      "fits": len(parameters), "all_phase0_evidence_preserved": integrity_end, "wall_seconds": result["wall_seconds"]}, indent=2))
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
