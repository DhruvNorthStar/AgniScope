"""HDBSCAN experiment using immutable pilot data and saved DBSCAN labels only."""

import json
from pathlib import Path
import sys
import time
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from agniscope.config import load_config, workspace_path
from agniscope.dbscan import ID, XY, DISTRICT, sensitivity_subsets, diagnostic_tables
from agniscope import hdbscan as h
from agniscope.provenance import sha256, write_json, verify_sources, utc_now
from agniscope.resources import resource_snapshot
from phase1a import load_pilot


KEYS = ["dataset", "min_cluster_size", "min_samples"]


def load_settings():
    return load_config(workspace_path("configs/phase1b.toml"))


def check_integrity(settings, project):
    preflight = json.loads(workspace_path(settings["experiment"]["preflight"]).read_text(encoding="utf-8"))
    changed = [item["path"] for item in preflight["protected_files"]
               if not workspace_path(item["path"]).is_file()
               or workspace_path(item["path"]).stat().st_size != item["size_bytes"]
               or sha256(workspace_path(item["path"])) != item["sha256"]]
    sources = verify_sources(project)
    if changed or not sources["all_unchanged"] or len(sources["files"]) != 14:
        raise ValueError(f"Protected Phase 0/1A evidence changed; stop and investigate: {changed}")
    return {"all_unchanged": True, "protected_existing_files": len(preflight["protected_files"]), "original_gis_files": 14}


def load_inputs(settings, project):
    points, districts, footprint, qc, columns = load_pilot(project)
    if settings["experiment"]["analysis_crs"] != project["spatial"]["analysis_crs"]:
        raise ValueError("Phase 1B must retain the approved metric CRS.")
    subsets, ledger = sensitivity_subsets(points, project["dbscan_pilot"]["temporal_grid_m"], project["dbscan_pilot"]["temporal_grid_origin_xy_m"])
    expected = {"baseline": 346, "nominal_only": 320, "tree_cover_only": 297, "grid_day_sample": 266}
    if {name: len(group) for name, group in subsets.items()} != expected:
        raise ValueError("Phase 1A population contract failed.")
    old_ledger = pd.read_csv(workspace_path("outputs/tables/phase1a/temporal_sampling_ledger.csv"))
    pd.testing.assert_frame_equal(ledger, old_ledger)
    path = workspace_path(settings["experiment"]["saved_dbscan_assignments"])
    saved = pd.read_csv(path, dtype={column: str for column in columns}, keep_default_na=False)
    selected = saved.loc[saved.eps_km.eq(settings["experiment"]["saved_dbscan_eps_km"]) & saved.min_samples.eq(settings["experiment"]["saved_dbscan_min_samples"])]
    references = {}
    preflight = json.loads(workspace_path(settings["experiment"]["preflight"]).read_text(encoding="utf-8"))
    for name, subset in subsets.items():
        reference = selected.loc[selected.dataset.eq(name)].sort_values(ID).reset_index(drop=True)
        if reference[ID].tolist() != subset[ID].tolist() or not np.allclose(reference[XY], subset[XY], rtol=0, atol=1e-6):
            raise ValueError("Saved DBSCAN identities/coordinates differ from approved input.")
        for column in columns:
            if reference[column].astype(str).tolist() != subset[column].astype(str).tolist():
                raise ValueError(f"Original source field differs in saved reference: {column}")
        reference["utc_date"] = pd.to_datetime(reference.agniscope_acquired_utc, utc=True).dt.strftime("%Y-%m-%d")
        counts = {"n": len(reference), "clusters": int(reference.loc[reference.cluster_id.ge(0), "cluster_id"].nunique()),
                  "clustered": int(reference.cluster_id.ge(0).sum()), "noise": int(reference.cluster_id.lt(0).sum())}
        if counts != preflight["saved_dbscan_references"][name]:
            raise ValueError("Saved DBSCAN reference changed from preflight.")
        references[name] = reference
    return points, subsets, references, districts, qc, columns


def select(frame, dataset, size, minimum):
    return frame.loc[frame.dataset.eq(dataset) & frame.min_cluster_size.eq(size) & frame.min_samples.eq(minimum)].copy()


def run_experiment():
    started = time.perf_counter()
    settings, project = load_settings(), load_config()
    resources_start = resource_snapshot()
    integrity_before = check_integrity(settings, project)
    points, subsets, references, districts, qc, source_columns = load_inputs(settings, project)
    exp = settings["experiment"]
    grid = h.parameter_grid(exp["min_cluster_size"], exp["min_samples"])
    if len(grid) != 12:
        raise ValueError("The authorized main grid must contain exactly 12 settings.")
    dates = pd.date_range(project["firms"]["pilot_start"], periods=project["firms"]["pilot_days"]).strftime("%Y-%m-%d").tolist()
    district_names = sorted(districts[project["spatial"]["district_original_column"]])
    grids, params = [], []
    with threadpool_limits(limits=1):
        for name, subset in subsets.items():
            output, actual = h.run_grid(subset, exp["min_cluster_size"], exp["min_samples"], settings["estimator"], name, district_names, dates)
            grids.append(output)
            params.append(actual)
    combined = {key: pd.concat([output[key] for output in grids], ignore_index=True) for key in grids[0]}
    if not all(value == params[0] for value in params):
        raise ValueError("Estimator options differ between populations.")
    assignments, parameters = combined["assignments"], combined["parameters"]
    comparisons, point_comparisons, overlaps, sensitivities = [], [], [], []
    for name in subsets:
        for size, minimum in grid:
            tag = {"dataset": name, "min_cluster_size": size, "min_samples": minimum}
            alternative = select(assignments, name, size, minimum)
            result, transitions, edges = h.compare_assignments(references[name], alternative, scope="identical")
            comparisons.append({**tag, "dbscan_eps_km": exp["saved_dbscan_eps_km"], "dbscan_min_samples": exp["saved_dbscan_min_samples"], **result})
            point_comparisons.append(transitions.assign(**tag))
            overlaps.append(edges.assign(**tag))
            result, _, _ = h.compare_assignments(select(assignments, "baseline", size, minimum), alternative,
                                                 scope="identical" if name == "baseline" else "common")
            sensitivities.append({**tag, **result})
    pair_stability = []
    for (size_a, min_a), (size_b, min_b) in combinations(grid, 2):
        result, _, _ = h.compare_assignments(select(assignments, "baseline", size_a, min_a),
                                           select(assignments, "baseline", size_b, min_b))
        pair_stability.append({"cluster_size_a": size_a, "min_samples_a": min_a, "cluster_size_b": size_b, "min_samples_b": min_b, **result})
    stability, stability_assignments, stability_statistics, sampling_ledger = [], [], [], []
    sub = settings["subsampling"]
    if sub["count_rule"] != "floor_fraction_times_346" or sub["replacement"] or sub["rng"] != "numpy.default_rng_PCG64":
        raise ValueError("Subsampling semantics must remain the declared fixed-size sampling without replacement.")
    with threadpool_limits(limits=1):
        for seed in sub["seeds"]:
            sample = h.deterministic_subsample(points, sub["fraction"], seed)
            ledger = points[[ID]].copy()
            ledger["selected"] = ledger[ID].isin(sample[ID])
            sampling_ledger.append(ledger.assign(seed=seed))
            for size, minimum in grid:
                tag = {"seed": seed, "min_cluster_size": size, "min_samples": minimum}
                fitted, _ = h.fit_configuration(sample, size, minimum, settings["estimator"])
                result, _, _ = h.compare_assignments(select(assignments, "baseline", size, minimum), fitted, scope="common")
                stability.append({**tag, **result})
                stability_assignments.append(fitted.assign(**tag))
                stability_statistics.append({**tag, **h.configuration_statistics(fitted)})
    target = workspace_path(exp["output_dir"], writable=True)
    target.mkdir(parents=True, exist_ok=True)
    matching_comparisons = parameters.merge(pd.DataFrame(comparisons), on=KEYS, validate="one_to_one")
    population_sensitivity = parameters.merge(pd.DataFrame(sensitivities), on=KEYS, validate="one_to_one")
    reference_diagnostics = {key: [] for key in ["district", "daily", "daynight", "frp"]}
    for name, reference in references.items():
        for key, table in diagnostic_tables(reference, district_names, dates).items():
            if key in reference_diagnostics:
                reference_diagnostics[key].append(table.assign(dataset=name))
    reference_diagnostics = {name: pd.concat(tables, ignore_index=True) for name, tables in reference_diagnostics.items()}
    composition = []
    for name, column in [("district", DISTRICT), ("daily", "utc_date"), ("daynight", "daynight")]:
        db = reference_diagnostics[name][["dataset", column, "membership", "n_observations"]].rename(columns={"n_observations": "dbscan_count"})
        both = combined[name].merge(db, on=["dataset", column, "membership"], validate="many_to_one")
        both["hdbscan_minus_dbscan_count"] = both.n_observations - both.dbscan_count
        both = both.rename(columns={column: "category", "n_observations": "hdbscan_count"})
        composition.append(both.assign(dimension=name))
    tables = {
        "hdbscan_parameter_comparison": parameters.loc[parameters.dataset.eq("baseline")],
        "all_population_parameters": parameters, "hdbscan_assignments": assignments,
        "hdbscan_cluster_summary": combined["clusters"], "dbscan_hdbscan_comparison": matching_comparisons,
        "point_membership_comparison": pd.concat(point_comparisons, ignore_index=True),
        "cluster_overlap": pd.concat(overlaps, ignore_index=True),
        "population_sensitivity": population_sensitivity, "parameter_stability": pd.DataFrame(pair_stability),
        "subsample_stability": pd.DataFrame(stability), "subsample_statistics": pd.DataFrame(stability_statistics),
        "subsample_assignments": pd.concat(stability_assignments, ignore_index=True),
        "subsampling_ledger": pd.concat(sampling_ledger, ignore_index=True),
        "composition_comparison": pd.concat(composition, ignore_index=True),
        **{f"diagnostic_{key}": combined[key] for key in ["district", "daily", "daynight", "cluster_daily", "cluster_daynight", "frp"]},
        **{f"dbscan_reference_{key}": value for key, value in reference_diagnostics.items()},
    }
    for name, table in tables.items():
        table.to_csv(target / f"{name}.csv", index=False)
    geometry_source = workspace_path("outputs/tables/phase1a/plot_geometry.json")
    (target / "plot_geometry.json").write_bytes(geometry_source.read_bytes())
    integrity_after = check_integrity(settings, project)
    result = {"generated_utc": utc_now(), "settings": settings, "python_executable": sys.executable,
              "versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
              "actual_estimator_parameters": params[0], "main_grid_fits": len(parameters), "subsample_fits": len(stability),
              "population_sizes": {name: len(group) for name, group in subsets.items()}, "qc_summary": qc,
              "dbscan_reference_refitted": False, "dbscan_source_sha256": sha256(workspace_path(exp["saved_dbscan_assignments"])),
              "source_fields_preserved": source_columns, "raw_sha256": project["dbscan_pilot"]["raw_sha256"],
              "integrity_before": integrity_before, "integrity_after": integrity_after,
              "source_code_sha256": {name: sha256(workspace_path(name)) for name in ["configs/phase1b.toml", "src/agniscope/hdbscan.py", "scripts/phase1b.py"]},
              "public_diagnostics": {"labels_": True, "probabilities_": True, "n_features_in_": 2, "cluster_persistence_": False, "outlier_scores_": False,
                                     "unavailable_policy": "No contrib or private hierarchy attributes substituted; no recurrence inferred."},
              "resources_start": resources_start, "resources_end": resource_snapshot(), "wall_seconds": time.perf_counter() - started,
              "metric_conventions": {"clustered": "label>=0", "noise_outlier": "all labels<0; documented -1/-2/-3 also counted separately",
                  "silhouette": "nonnegative labels only, requires 2<=k<n_clustered", "ARI": "negative labels normalized to one nonmembership label; both all-common and clustered-union reported",
                  "clustered_union": "common source records clustered by at least one method", "largest_cluster_shares": "both all-input and clustered-input denominators",
                  "membership_probability": "strength in assigned spatial cluster, not probability of real fire", "missing_numeric_csv": "blank means unavailable, never invented zero"}}
    write_json(target / "experiment.json", result)
    print(json.dumps({"populations": result["population_sizes"], "main_fits": len(parameters), "subsample_fits": len(stability),
                      "saved_dbscan_refitted": False, "integrity": integrity_after, "wall_seconds": result["wall_seconds"]}, indent=2))
    print(tables["hdbscan_parameter_comparison"][["min_cluster_size", "min_samples", "n_clusters", "n_clustered", "n_noise_outliers", "noise_outlier_pct_of_input", "size_min", "size_median", "size_max", "largest_share_pct_of_input", "silhouette_nonnoise", "prob_clustered_median"]].to_string(index=False))


if __name__ == "__main__":
    run_experiment()
