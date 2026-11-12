"""Bounded sklearn HDBSCAN and identity-aligned descriptive comparisons.

All features are prepared EPSG:32644 XY metres. No acquisition or incident labels.
Phase 1A utilities are imported without changing or rerunning its experiments.
"""

from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .dbscan import ID, XY, DISTRICT, cluster_summaries, diagnostic_tables


OUTLIERS = {-1: "noise", -2: "infinite_input", -3: "missing_input"}


def parameter_grid(cluster_sizes, minimum_samples):
    sizes, samples = list(cluster_sizes), list(minimum_samples)
    if not sizes or not samples or len(set(sizes)) != len(sizes) or len(set(samples)) != len(samples):
        raise ValueError("Parameter grid must be nonempty and unique.")
    for values, lower in [(sizes, 2), (samples, 1)]:
        if any(isinstance(v, bool) or not np.isfinite(v) or int(v) != v or v < lower for v in values):
            raise ValueError("HDBSCAN sizes must be integers: cluster size >=2, min_samples >=1.")
    return [(int(size), int(minimum)) for size, minimum in product(sizes, samples)]


def estimator_parameters(min_cluster_size, min_samples, fixed):
    parameter_grid([min_cluster_size], [min_samples])
    if fixed.get("metric") != "euclidean" or fixed.get("cluster_selection_method") != "eom" or fixed.get("cluster_selection_epsilon") != 0:
        raise ValueError("Main experiment requires Euclidean EOM and selection epsilon zero metres.")
    if fixed.get("n_jobs") != 1 or fixed.get("copy") is not True:
        raise ValueError("Single-worker execution and input protection must remain explicit.")
    return {"max_cluster_size": None, "metric_params": None, "store_centers": None,
            **fixed, "min_cluster_size": min_cluster_size, "min_samples": min_samples}


def validate_membership(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("One label and membership probability are required per observation.")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Cluster labels must be finite integers.")
    if any(int(label) not in OUTLIERS for label in labels[labels < 0]):
        raise ValueError("Undocumented negative label; stop rather than inventing a cluster.")
    missing = labels == -3
    finite = ~missing
    if not np.isnan(probabilities[missing]).all() or not np.isfinite(probabilities[finite]).all():
        raise ValueError("Missing-input probability must be NaN; other probabilities must be finite.")
    if ((probabilities[finite] < 0) | (probabilities[finite] > 1)).any():
        raise ValueError("Membership strength must lie in [0,1].")
    if not np.equal(probabilities[(labels == -1) | (labels == -2)], 0).all():
        raise ValueError("Noise and infinite-input observations must have zero membership strength.")


def fit_configuration(points, min_cluster_size, min_samples, fixed):
    if points.attrs.get("analysis_crs") != "EPSG:32644":
        raise ValueError("Use prepare_points to establish EPSG:32644 metre coordinates first.")
    if points[ID].isna().any() or points[ID].duplicated().any():
        raise ValueError("Unique source identities are required.")
    result = points.sort_values(ID, kind="stable").reset_index(drop=True).copy(deep=True)
    coordinates = result[XY].to_numpy(dtype=float, copy=True)
    if not np.isfinite(coordinates).all():
        raise ValueError("QC-accepted experiment coordinates must remain finite; stop and investigate.")
    params = estimator_parameters(min_cluster_size, min_samples, fixed)
    if len(result) <= min_samples:
        raise ValueError("Input must contain more observations than min_samples.")
    model = HDBSCAN(**params).fit(coordinates)
    labels, probabilities = model.labels_, model.probabilities_
    validate_membership(labels, probabilities)
    result["cluster_id"] = labels
    result["membership_probability"] = probabilities
    result["membership_status"] = [OUTLIERS.get(int(label), "clustered") for label in labels]
    return result, model.get_params(deep=False)


def probability_statistics(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {f"{prefix}_available": len(finite), f"{prefix}_unavailable": len(values) - len(finite),
            **{f"{prefix}_{name}": float(np.quantile(finite, quantile)) if len(finite) else None
               for name, quantile in [("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1)]},
            f"{prefix}_mean": float(np.mean(finite)) if len(finite) else None}


def configuration_statistics(assignments):
    validate_membership(assignments.cluster_id, assignments.membership_probability)
    clustered = assignments.loc[assignments.cluster_id.ge(0)]
    sizes = clustered.groupby("cluster_id").size()
    n, k, nc = len(assignments), len(sizes), len(clustered)
    score, status = None, "unavailable_no_clusters"
    if k == 1:
        status = "unavailable_one_cluster"
    elif k >= 2:
        if k < nc:
            score = float(silhouette_score(clustered[XY].to_numpy(), clustered.cluster_id, metric="euclidean"))
            status = "available_all_negative_labels_excluded"
        else:
            status = "unavailable_one_point_per_cluster"
    return {"n_observations": n, "n_clusters": k, "n_clustered": nc,
            "n_noise_outliers": n - nc,
            "n_noise_minus1": int(assignments.cluster_id.eq(-1).sum()),
            "n_infinite_minus2": int(assignments.cluster_id.eq(-2).sum()),
            "n_missing_minus3": int(assignments.cluster_id.eq(-3).sum()),
            "noise_outlier_pct_of_input": 100 * (n - nc) / n if n else None,
            "size_min": int(sizes.min()) if k else None, "size_median": float(sizes.median()) if k else None,
            "size_max": int(sizes.max()) if k else None,
            "largest_share_pct_of_input": 100 * int(sizes.max()) / n if k else None,
            "largest_share_pct_of_clustered": 100 * int(sizes.max()) / nc if k else None,
            "clustered_districts": int(clustered[DISTRICT].nunique()),
            "silhouette_nonnoise": score, "silhouette_status": status,
            **probability_statistics(assignments.membership_probability, "prob_all"),
            **probability_statistics(clustered.membership_probability, "prob_clustered")}


def summarize_clusters(assignments):
    result = cluster_summaries(assignments)
    if result.empty:
        return pd.DataFrame(columns=["cluster_id", "n_observations"])
    probabilities = pd.DataFrame([{"cluster_id": int(label),
        "centroid_x_m": float(group[XY[0]].mean()), "centroid_y_m": float(group[XY[1]].mean()),
        **probability_statistics(group.membership_probability, "probability")}
        for label, group in assignments.loc[assignments.cluster_id.ge(0)].groupby("cluster_id")])
    return result.merge(probabilities, on="cluster_id", validate="one_to_one")


def diagnostics(assignments, district_names, dates):
    # Phase 1A treats only -1 as noise. Normalize a temporary reporting copy,
    # preserving original HDBSCAN labels/probabilities in assignments and counts.
    normalized = assignments.copy(deep=True)
    normalized.loc[normalized.cluster_id.lt(0), "cluster_id"] = -1
    return diagnostic_tables(normalized, district_names, dates)


def run_grid(points, sizes, samples, fixed, dataset, district_names, dates):
    statistics, assignments, summaries = [], [], []
    diagnostic_results = {name: [] for name in ["district", "daily", "daynight", "cluster_daily", "cluster_daynight", "frp"]}
    parameters = []
    for size, minimum in parameter_grid(sizes, samples):
        tag = {"dataset": dataset, "min_cluster_size": size, "min_samples": minimum}
        result, params = fit_configuration(points, size, minimum, fixed)
        parameters.append(params)
        statistics.append({**tag, **configuration_statistics(result)})
        assignments.append(result.assign(**tag))
        summaries.append(summarize_clusters(result).assign(**tag))
        for name, table in diagnostics(result, district_names, dates).items():
            diagnostic_results[name].append(table.assign(**tag))
    return {"parameters": pd.DataFrame(statistics), "assignments": pd.concat(assignments, ignore_index=True),
            "clusters": pd.concat(summaries, ignore_index=True),
            **{name: pd.concat(tables, ignore_index=True) for name, tables in diagnostic_results.items()}}, parameters


def align_assignments(reference, alternative, scope="identical"):
    if scope not in {"identical", "common"}:
        raise ValueError("Comparison scope must explicitly be identical or common.")
    for frame in [reference, alternative]:
        if frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError("Comparison requires unique nonmissing source identities.")
        labels = frame.cluster_id.to_numpy()
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("Comparison labels must be finite integers.")
    if scope == "identical" and set(reference[ID]) != set(alternative[ID]):
        raise ValueError("Identical-population comparison requires identical source identities.")
    required = [ID, "cluster_id", *XY]
    common = reference[required].merge(alternative[required], on=ID, suffixes=("_reference", "_alternative"), validate="one_to_one").sort_values(ID).reset_index(drop=True)
    if len(common) == 0:
        raise ValueError("Comparison has no common observations.")
    left = common[[f"{col}_reference" for col in XY]].to_numpy()
    right = common[[f"{col}_alternative" for col in XY]].to_numpy()
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right, rtol=0, atol=1e-6):
        raise ValueError("Common source identities must have identical metric coordinates.")
    return common


def compare_assignments(reference, alternative, scope="identical"):
    common = align_assignments(reference, alternative, scope)
    left = common.cluster_id_reference.to_numpy()
    right = common.cluster_id_alternative.to_numpy()
    # All negative labels are one nonmembership category for ARI, never clusters.
    left, right = np.where(left < 0, -1, left), np.where(right < 0, -1, right)
    lc, rc = left >= 0, right >= 0
    union, both = lc | rc, lc & rc
    def ari(mask):
        return float(adjusted_rand_score(left[mask], right[mask])) if np.count_nonzero(mask) >= 2 else None
    def fraction(numerator, denominator):
        return float(numerator / denominator) if denominator else None
    n = len(common)
    stats = {"population_scope": scope, "reference_input_n": len(reference), "alternative_input_n": len(alternative),
        "n_common": n, "reference_rows_not_common": len(reference) - n, "alternative_rows_not_common": len(alternative) - n,
        "reference_clustered_on_common": int(lc.sum()), "alternative_clustered_on_common": int(rc.sum()),
        "clustered_both": int(both.sum()), "reference_only_clustered": int((lc & ~rc).sum()),
        "alternative_only_clustered": int((~lc & rc).sum()), "nonclustered_both": int((~union).sum()),
        "clustered_union_n": int(union.sum()), "binary_membership_agreement": float(np.mean(lc == rc)),
        "clustered_population_jaccard": fraction(both.sum(), union.sum()),
        "shared_fraction_of_reference_clustered": fraction(both.sum(), lc.sum()),
        "shared_fraction_of_alternative_clustered": fraction(both.sum(), rc.sum()),
        "ari_all_common_noise_shared": ari(np.ones(n, dtype=bool)),
        "ari_clustered_union": ari(union), "ari_clustered_intersection": ari(both),
        "reference_clusters_on_common": len(set(left[lc])), "alternative_clusters_on_common": len(set(right[rc]))}
    # Compare co-clustered pairs, excluding same-noise pairs entirely.
    i, j = np.triu_indices(n, k=1)
    lp = (left[i] == left[j]) & lc[i] & lc[j]
    rp = (right[i] == right[j]) & rc[i] & rc[j]
    stats.update(reference_coclustered_pairs=int(lp.sum()), alternative_coclustered_pairs=int(rp.sum()),
                 shared_coclustered_pairs=int((lp & rp).sum()),
                 coclustered_pair_jaccard=fraction((lp & rp).sum(), (lp | rp).sum()))
    overlaps = []
    for l, r in sorted(set(zip(left[both], right[both]))):
        overlap = int(((left == l) & (right == r)).sum())
        nl, nr = int((left == l).sum()), int((right == r).sum())
        overlaps.append({"reference_cluster_id": int(l), "alternative_cluster_id": int(r), "shared_points": overlap,
                         "reference_cluster_n_common": nl, "alternative_cluster_n_common": nr,
                         "point_set_jaccard": overlap / (nl + nr - overlap)})
    edges = pd.DataFrame(overlaps, columns=["reference_cluster_id", "alternative_cluster_id", "shared_points", "reference_cluster_n_common", "alternative_cluster_n_common", "point_set_jaccard"])
    stats["reference_clusters_split_across_alternative"] = int(edges.groupby("reference_cluster_id").size().gt(1).sum())
    stats["alternative_clusters_merge_reference"] = int(edges.groupby("alternative_cluster_id").size().gt(1).sum())
    stats["reference_clusters_with_no_clustered_overlap"] = len(set(left[lc]) - set(edges.reference_cluster_id))
    stats["alternative_clusters_with_no_clustered_overlap"] = len(set(right[rc]) - set(edges.alternative_cluster_id))
    coordinates = common[[f"{col}_reference" for col in XY]].to_numpy()
    for name, queries, targets in [("reference_only_to_alternative_clustered", lc & ~rc, rc),
                                   ("alternative_only_to_reference_clustered", ~lc & rc, lc)]:
        distances = cKDTree(coordinates[targets]).query(coordinates[queries], k=1)[0] if queries.any() and targets.any() else np.array([])
        stats[f"{name}_n"] = len(distances)
        stats[f"{name}_median_m"] = float(np.median(distances)) if len(distances) else None
        stats[f"{name}_p95_m"] = float(np.quantile(distances, .95)) if len(distances) else None
    transitions = common[[ID]].copy()
    transitions["reference_cluster_id"] = common.cluster_id_reference
    transitions["alternative_cluster_id"] = common.cluster_id_alternative
    transitions["membership_comparison"] = np.select([both, lc & ~rc, ~lc & rc],
        ["clustered_both", "reference_only_clustered", "alternative_only_clustered"], default="nonclustered_both")
    return stats, transitions, edges


def deterministic_subsample(points, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Subsampling fraction must be strictly between zero and one.")
    ordered = points.sort_values(ID).reset_index(drop=True)
    n = int(np.floor(len(ordered) * fraction))
    indices = np.sort(np.random.default_rng(seed).choice(len(ordered), size=n, replace=False))
    return ordered.iloc[indices].copy(deep=True).reset_index(drop=True)
