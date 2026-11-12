"""Small, deterministic coordinate-only experiments on QC-accepted detections.

No acquisition, raster reads, incident deduplication or parameter optimization.
Cluster labels have meaning only within (dataset, eps_km, min_samples).
"""

from itertools import product

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree, distance
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score


ID = "agniscope_source_record"
UTC = "agniscope_acquired_utc"
XY = ["agniscope_x_m", "agniscope_y_m"]
DISTRICT = "agniscope_district_original"
TREE = "agniscope_worldcover_2021_class"


def eps_metres(eps_km: float) -> float:
    value = float(eps_km)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Epsilon must be finite and positive in kilometres.")
    return value * 1000.0


def parameter_grid(eps_values, min_samples_values) -> list[tuple[float, int]]:
    eps_values, min_samples_values = list(eps_values), list(min_samples_values)
    if not eps_values or not min_samples_values:
        raise ValueError("Parameter grid must not be empty.")
    if len(set(eps_values)) != len(eps_values) or len(set(min_samples_values)) != len(min_samples_values):
        raise ValueError("Parameter grid must not contain duplicates.")
    for eps in eps_values:
        eps_metres(eps)
    for minimum in min_samples_values:
        if isinstance(minimum, bool) or int(minimum) != minimum or minimum < 1:
            raise ValueError("min_samples must be a positive integer, including the point itself.")
    return [(float(eps), int(minimum)) for eps, minimum in product(eps_values, min_samples_values)]


def prepare_points(frame: pd.DataFrame, crs: str = "EPSG:32644") -> pd.DataFrame:
    """Reproject original lon/lat; never trust a relabelled coordinate array."""
    if CRS.from_user_input(crs) != CRS.from_epsg(32644):
        raise ValueError("This bounded experiment requires EPSG:32644 metre coordinates.")
    required = {ID, "latitude", "longitude", UTC, DISTRICT, TREE,
                "agniscope_confidence", "daynight", "frp"}
    if missing := required - set(frame):
        raise ValueError(f"Missing experiment fields: {sorted(missing)}")
    result = frame.copy(deep=True)
    identifiers = pd.to_numeric(result[ID], errors="raise")
    if identifiers.isna().any() or identifiers.duplicated().any() or not np.isfinite(identifiers).all() or (identifiers < 1).any() or (identifiers % 1 != 0).any():
        raise ValueError("Source record identities must be unique positive integers.")
    result[ID] = identifiers.astype("int64")
    result = result.sort_values(ID, kind="stable").reset_index(drop=True)
    lon = pd.to_numeric(result.longitude, errors="raise").to_numpy(dtype=float)
    lat = pd.to_numeric(result.latitude, errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all() and ((lon >= -180) & (lon <= 180)).all() and ((lat >= -90) & (lat <= 90)).all()):
        raise ValueError("Finite geographic coordinates are required.")
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    coordinates = np.column_stack([x, y])
    if not np.isfinite(coordinates).all():
        raise ValueError("Projection produced invalid coordinates.")
    if all(col in result for col in XY) and not np.allclose(result[XY].to_numpy(dtype=float), coordinates, rtol=0, atol=1e-6):
        raise ValueError("Saved projected coordinates disagree with original lon/lat.")
    result[XY] = coordinates
    result[UTC] = pd.to_datetime(result[UTC], utc=True, errors="raise")
    if result[UTC].isna().any():
        raise ValueError("Missing acquisition timestamps are not accepted.")
    result["utc_date"] = result[UTC].dt.strftime("%Y-%m-%d")
    if "agniscope_disposition" in result and not result.agniscope_disposition.eq("retained").all():
        raise ValueError("Experiment input must already be retained by scientific QC.")
    result.attrs["analysis_crs"] = "EPSG:32644"
    return result


def daily_grid_sample(points: pd.DataFrame, cell_m: float, origin=(0.0, 0.0)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One representative per half-open projected cell per UTC day.

    Earliest timestamp then lowest original record. Grid origin is explicit.
    This is sampling sensitivity, not fire-incident identification.
    """
    if not np.isfinite(cell_m) or cell_m <= 0 or not np.isfinite(origin).all() or len(origin) != 2:
        raise ValueError("Grid size must be positive and origin a finite XY pair.")
    work = points.copy(deep=True)
    work[["grid_x", "grid_y"]] = np.floor((work[XY].to_numpy() - np.asarray(origin)) / cell_m).astype("int64")
    keys = ["grid_x", "grid_y", "utc_date"]
    ordered = work.sort_values([UTC, ID], kind="stable")
    reps = ordered.drop_duplicates(keys, keep="first")
    mapping = work[[ID, *keys]].merge(reps[[ID, *keys]].rename(columns={ID: "representative_source_record"}), on=keys, validate="many_to_one")
    mapping["selected"] = mapping[ID].eq(mapping.representative_source_record)
    return (reps.drop(columns=["grid_x", "grid_y"]).sort_values(ID).reset_index(drop=True),
            mapping.sort_values(ID).reset_index(drop=True))


def sensitivity_subsets(points: pd.DataFrame, cell_m=1000.0, origin=(0.0, 0.0)):
    sampled, ledger = daily_grid_sample(points, cell_m, origin)
    return {
        "baseline": points.copy(deep=True),
        "nominal_only": points.loc[points.agniscope_confidence.eq("nominal")].copy(),
        "tree_cover_only": points.loc[points[TREE].eq(10)].copy(),
        "grid_day_sample": sampled,
    }, ledger


def fit_configuration(points: pd.DataFrame, eps_km: float, min_samples: int) -> pd.DataFrame:
    """Fixed ordering makes ambiguous border-point assignment reproducible."""
    parameter_grid([eps_km], [min_samples])
    if points.attrs.get("analysis_crs") != "EPSG:32644":
        raise ValueError("Run prepare_points first to establish EPSG:32644 metre coordinates.")
    result = points.sort_values(ID, kind="stable").copy(deep=True).reset_index(drop=True)
    if len(result):
        model = DBSCAN(eps=eps_metres(eps_km), min_samples=int(min_samples),
                       metric="euclidean", algorithm="kd_tree", n_jobs=1)
        labels = model.fit_predict(result[XY].to_numpy(dtype=float))
        core = np.zeros(len(result), dtype=bool)
        core[model.core_sample_indices_] = True
    else:
        labels, core = np.array([], dtype=int), np.array([], dtype=bool)
    result["cluster_id"] = labels
    result["point_role"] = np.where(labels == -1, "noise", np.where(core, "core", "border"))
    return result


def configuration_statistics(assignments: pd.DataFrame) -> dict:
    clustered = assignments.loc[assignments.cluster_id.ge(0)]
    sizes = clustered.groupby("cluster_id").size()
    n, k, n_clustered = len(assignments), len(sizes), len(clustered)
    n_noise = n - n_clustered
    score, score_status = None, "unavailable_no_clusters"
    if k == 1:
        score_status = "unavailable_one_cluster"
    elif k >= 2:
        if k < n_clustered:
            score = float(silhouette_score(clustered[XY].to_numpy(), clustered.cluster_id, metric="euclidean"))
            score_status = "available_noise_excluded"
        else:
            score_status = "unavailable_one_point_per_cluster"
    return {
        "n_observations": n, "n_clusters": k, "n_noise": n_noise,
        "noise_pct_of_input": 100 * n_noise / n if n else None,
        "n_clustered": n_clustered,
        "size_min": int(sizes.min()) if k else None,
        "size_median": float(sizes.median()) if k else None,
        "size_max": int(sizes.max()) if k else None,
        "largest_share_pct_of_input": 100 * int(sizes.max()) / n if k else None,
        "largest_share_pct_of_clustered": 100 * int(sizes.max()) / n_clustered if k else None,
        "clustered_districts": int(clustered[DISTRICT].nunique()),
        "n_core": int(assignments.point_role.eq("core").sum()),
        "n_border": int(assignments.point_role.eq("border").sum()),
        "silhouette_nonnoise": score, "silhouette_status": score_status,
    }


def frp_statistics(group: pd.DataFrame) -> dict:
    values = pd.to_numeric(group.frp, errors="coerce")
    values = values.loc[np.isfinite(values)]
    return {
        "frp_available": len(values), "frp_missing_or_nonfinite": len(group) - len(values),
        "frp_min_mw": float(values.min()) if len(values) else None,
        "frp_q25_mw": float(values.quantile(.25)) if len(values) else None,
        "frp_median_mw": float(values.median()) if len(values) else None,
        "frp_mean_mw": float(values.mean()) if len(values) else None,
        "frp_q75_mw": float(values.quantile(.75)) if len(values) else None,
        "frp_max_mw": float(values.max()) if len(values) else None,
        "frp_sample_std_mw": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def cluster_summaries(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in assignments.loc[assignments.cluster_id.ge(0)].groupby("cluster_id", sort=True):
        pair_distances = distance.pdist(group[XY].to_numpy())
        rows.append({
            "cluster_id": int(label), "n_observations": len(group),
            "source_records": ";".join(group[ID].astype(str)),
            "districts_original": ";".join(sorted(group[DISTRICT].dropna().unique())),
            "n_districts": int(group[DISTRICT].nunique()),
            "west_lon": float(pd.to_numeric(group.longitude).min()),
            "east_lon": float(pd.to_numeric(group.longitude).max()),
            "south_lat": float(pd.to_numeric(group.latitude).min()),
            "north_lat": float(pd.to_numeric(group.latitude).max()),
            "bbox_width_m": float(group[XY[0]].max() - group[XY[0]].min()),
            "bbox_height_m": float(group[XY[1]].max() - group[XY[1]].min()),
            "max_pair_distance_m": float(pair_distances.max()) if len(pair_distances) else 0.0,
            "first_utc": group[UTC].min().isoformat(), "last_utc": group[UTC].max().isoformat(),
            "span_hours": (group[UTC].max() - group[UTC].min()).total_seconds() / 3600,
            "distinct_acquisition_timestamps": int(group[UTC].nunique()),
            "distinct_utc_days": int(group.utc_date.nunique()),
            "n_april_4_5": int(group.utc_date.isin(["2024-04-04", "2024-04-05"]).sum()),
            "april_4_5_pct": 100 * float(group.utc_date.isin(["2024-04-04", "2024-04-05"]).mean()),
            "n_day": int(group.daynight.eq("D").sum()), "n_night": int(group.daynight.eq("N").sum()),
            **frp_statistics(group),
        })
    return pd.DataFrame(rows)


def diagnostic_tables(assignments: pd.DataFrame, district_names, utc_dates) -> dict[str, pd.DataFrame]:
    points = assignments.copy()
    points["membership"] = np.where(points.cluster_id.eq(-1), "noise", "clustered")
    tables = {}
    for name, column, levels in [
        ("district", DISTRICT, list(district_names) + ["UNASSIGNED_OR_AMBIGUOUS"]),
        ("daily", "utc_date", list(utc_dates)),
        ("daynight", "daynight", ["D", "N", "UNKNOWN"]),
    ]:
        points[column] = points[column].fillna("UNASSIGNED_OR_AMBIGUOUS" if name == "district" else "UNKNOWN")
        index = pd.MultiIndex.from_product([levels, ["clustered", "noise"]], names=[column, "membership"])
        counts = points.groupby([column, "membership"]).size().reindex(index, fill_value=0).rename("n_observations").reset_index()
        counts["input_n"] = len(points)
        tables[name] = counts
    tables["cluster_daily"] = points.groupby(["cluster_id", "utc_date"]).size().rename("n_observations").reset_index()
    tables["cluster_daynight"] = points.groupby(["cluster_id", "daynight"]).size().rename("n_observations").reset_index()
    tables["frp"] = pd.DataFrame([
        {"membership": status, "n_observations": len(group), **frp_statistics(group)}
        for status in ["clustered", "noise"]
        for group in [points.loc[points.membership.eq(status)]]
    ])
    return tables


def run_grid(points, eps_values, min_samples_values, dataset="baseline", district_names=(), utc_dates=()):
    comparisons, assignments, summaries = [], [], []
    diagnostics = {key: [] for key in ["district", "daily", "daynight", "cluster_daily", "cluster_daynight", "frp"]}
    for eps, minimum in parameter_grid(eps_values, min_samples_values):
        tag = {"dataset": dataset, "eps_km": eps, "min_samples": minimum}
        result = fit_configuration(points, eps, minimum)
        comparisons.append({**tag, **configuration_statistics(result)})
        assignments.append(result.assign(**tag))
        summary = cluster_summaries(result)
        if len(summary):
            summaries.append(summary.assign(**tag))
        for name, table in diagnostic_tables(result, district_names, utc_dates).items():
            diagnostics[name].append(table.assign(**tag))
    return {
        "parameters": pd.DataFrame(comparisons), "assignments": pd.concat(assignments, ignore_index=True),
        "clusters": pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=["dataset", "eps_km", "min_samples", "cluster_id"]),
        **{key: pd.concat(tables, ignore_index=True) for key, tables in diagnostics.items()},
    }


def repeat_diagnostics(points, thresholds=(375.0, 500.0, 1000.0)):
    """Undirected unordered pairs; distinct timestamps do not prove a repeat fire."""
    ordered = points.sort_values(ID).reset_index(drop=True)
    rows = []
    coords = ordered[XY].to_numpy()
    for i, j in sorted(cKDTree(coords).query_pairs(max(thresholds))):
        a, b = ordered.iloc[i], ordered.iloc[j]
        rows.append({
            "source_record_a": int(a[ID]), "source_record_b": int(b[ID]),
            "distance_m": float(np.linalg.norm(coords[i] - coords[j])),
            "timestamp_a": a[UTC].isoformat(), "timestamp_b": b[UTC].isoformat(),
            "different_timestamp": bool(a[UTC] != b[UTC]),
            "different_utc_day": bool(a.utc_date != b.utc_date),
            "elapsed_hours": abs((a[UTC] - b[UTC]).total_seconds()) / 3600,
        })
    pairs = pd.DataFrame(rows, columns=["source_record_a", "source_record_b", "distance_m", "timestamp_a", "timestamp_b", "different_timestamp", "different_utc_day", "elapsed_hours"])
    summary = []
    for threshold in thresholds:
        selected = pairs.loc[pairs.distance_m.le(threshold)]
        distinct = selected.loc[selected.different_timestamp.eq(True)]
        summary.append({"threshold_m": threshold, "n_pairs": len(selected),
                        "different_timestamp_pairs": len(distinct),
                        "different_utc_day_pairs": int(selected.different_utc_day.sum()),
                        "points_in_different_timestamp_pairs": len(set(distinct.source_record_a) | set(distinct.source_record_b))})
    coordinate_counts = ordered[["latitude", "longitude"]].apply(pd.to_numeric).value_counts()
    return pairs, pd.DataFrame(summary), {
        "unique_coordinate_pairs": len(coordinate_counts),
        "repeated_coordinate_groups": int(coordinate_counts.gt(1).sum()),
        "observations_in_repeated_coordinate_groups": int(coordinate_counts.loc[coordinate_counts.gt(1)].sum()),
    }


def distance_crosscheck(points, thresholds=(500.0, 1000.0, 2000.0), radius_m=6371008.8):
    """All pilot pairs versus spherical haversine, with ellipsoidal reference.

    Haversine spherical model differences are reported, not called projection error.
    """
    from pyproj import Geod
    from sklearn.metrics.pairwise import haversine_distances

    coords = points[XY].to_numpy()
    i, j = np.triu_indices(len(points), k=1)
    planar = np.linalg.norm(coords[i] - coords[j], axis=1)
    latlon = np.radians(points[["latitude", "longitude"]].to_numpy(dtype=float))
    spherical = haversine_distances(latlon)[i, j] * radius_m
    lon, lat = points.longitude.to_numpy(dtype=float), points.latitude.to_numpy(dtype=float)
    _, _, geodesic = Geod(ellps="WGS84").inv(lon[i], lat[i], lon[j], lat[j])
    valid = geodesic > 0
    summary = {
        "n_pairs_checked": len(i), "haversine_earth_radius_m": radius_m,
        "max_abs_relative_utm_vs_geodesic_pct": float(np.max(np.abs(planar[valid] / geodesic[valid] - 1)) * 100),
        "max_abs_relative_utm_vs_haversine_pct": float(np.max(np.abs(planar[valid] / spherical[valid] - 1)) * 100),
        "neighbour_pair_disagreements": [{"eps_m": eps,
            "utm_vs_haversine": int(np.count_nonzero((planar <= eps) != (spherical <= eps))),
            "utm_vs_geodesic": int(np.count_nonzero((planar <= eps) != (geodesic <= eps)))} for eps in thresholds],
    }
    # Closest pair, representative median-scale pair, farthest, closest to each eps.
    selected = sorted(set([int(np.argmin(planar)), int(np.argmax(planar)),
                           int(np.argmin(abs(planar - np.median(planar)))),
                           *[int(np.argmin(abs(planar - eps))) for eps in thresholds]]))
    rows = [{"source_record_a": int(points.iloc[i[p]][ID]), "source_record_b": int(points.iloc[j[p]][ID]),
             "utm_m": float(planar[p]), "haversine_m": float(spherical[p]), "geodesic_m": float(geodesic[p])} for p in selected]
    return summary, pd.DataFrame(rows)


def membership_comparison(reference, alternative) -> dict:
    common = reference[[ID, "cluster_id"]].merge(alternative[[ID, "cluster_id"]], on=ID, suffixes=("_baseline", "_sensitivity"), validate="one_to_one")
    base = common.cluster_id_baseline.ge(0)
    other = common.cluster_id_sensitivity.ge(0)
    both = base & other
    return {"n_common_source_records": len(common),
            "clustered_in_both": int(both.sum()), "baseline_clustered_to_noise": int((base & ~other).sum()),
            "baseline_noise_to_clustered": int((~base & other).sum()), "noise_in_both": int((~base & ~other).sum()),
            "ari_on_common_including_noise": float(adjusted_rand_score(common.cluster_id_baseline, common.cluster_id_sensitivity)) if len(common) > 1 else None,
            "ari_on_common_clustered_in_both": float(adjusted_rand_score(common.loc[both, "cluster_id_baseline"], common.loc[both, "cluster_id_sensitivity"])) if both.sum() > 1 else None}
