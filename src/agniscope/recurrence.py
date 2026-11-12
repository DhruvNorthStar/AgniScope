"""Fixed-grid summaries of observed thermal activity; no estimator or network code."""
from collections import deque
import math
import numpy as np
import pandas as pd
import shapely

RESOLUTIONS = (500, 1000, 2000)
YEARS = tuple(range(2019, 2026))
METHODS = ("detection", "dbscan", "hdbscan")
POPULATIONS = ("all", "grid_day", "nominal", "tree")
ID = "agniscope_observation_id"
UTC = "agniscope_acquired_utc"
XY = ["agniscope_x_m", "agniscope_y_m"]


def grid_indices(xy, resolution, origin=(0, 0)):
    if resolution not in RESOLUTIONS:
        raise ValueError("Resolution must be 500, 1000 or 2000 metres.")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all() or (origin % 2000 != 0).any():
        raise ValueError("Common origin must be finite and aligned to 2000 metres.")
    values = np.asarray(xy, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Finite two-dimensional projected metre coordinates required.")
    return np.floor((values-origin)/resolution).astype("int64")


def cell_ids(ix, iy, resolution):
    if resolution not in RESOLUTIONS:
        raise ValueError("Unapproved metre resolution.")
    x, y = np.broadcast_arrays(np.asarray(ix), np.asarray(iy))
    if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or (x != np.floor(x)).any() or (y != np.floor(y)).any()
            or (x < -32768).any() or (x > 32767).any()
            or (y < -32768).any() or (y > 32767).any()):
        raise ValueError("Grid indices must be signed 16-bit integers.")
    code = RESOLUTIONS.index(resolution)+1
    return code*2**32 + (y.astype("int64")+32768)*2**16 + x.astype("int64")+32768


def point_cell_ids(xy, resolution, origin=(0, 0)):
    indices = grid_indices(xy, resolution, origin)
    return cell_ids(indices[:, 0], indices[:, 1], resolution)


def validate_source_points(frame, year):
    if year not in YEARS:
        raise ValueError("Only seven primary years are permitted.")
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError("Missing or duplicate original source identities.")
    if not frame[ID].str.fullmatch(r"[0-9a-f]{64}:[1-9][0-9]*").all():
        raise ValueError("Original raw-SHA256:record identities required.")
    if getattr(frame[UTC].dtype, "tz", None) is None or str(frame[UTC].dt.tz) != "UTC":
        raise ValueError("Explicit timezone-aware UTC acquisition times required.")
    dates = frame[UTC].dt.strftime("%Y-%m-%d")
    if frame[UTC].isna().any() or not dates.between(f"{year}-02-15", f"{year}-06-15").all():
        raise ValueError("Observation lies outside the declared UTC primary season.")
    if not np.isfinite(frame[XY].to_numpy(float)).all():
        raise ValueError("Invalid projected point.")


def join_memberships(points, assignments, algorithm):
    if algorithm not in ("dbscan", "hdbscan"):
        raise ValueError("An explicit saved reference algorithm is required.")
    if (not points[ID].is_unique or not assignments[ID].is_unique
            or points[ID].isna().any() or assignments[ID].isna().any()
            or set(points[ID]) != set(assignments[ID])):
        raise ValueError("Reference join must be exact and one-to-one.")
    ordered = assignments.set_index(ID).loc[points[ID]]
    if not np.array_equal(ordered[XY].to_numpy(float), points[XY].to_numpy(float)):
        raise ValueError("Saved member coordinates disagree with authentic source points.")
    if not np.array_equal(ordered[UTC].to_numpy(), points[UTC].to_numpy()):
        raise ValueError("Saved member times disagree with authentic source points.")
    labels = ordered.cluster_label.to_numpy()
    if not np.isfinite(labels).all() or (labels != np.floor(labels)).any() or set(labels[labels < 0]) - {-1, -2, -3}:
        raise ValueError("Invalid or undocumented cluster label.")
    strengths = ordered.membership_strength.to_numpy(float)
    members = labels >= 0
    if algorithm == "hdbscan" and (not np.isfinite(strengths[members]).all()
                                  or ((strengths[members] < 0) | (strengths[members] > 1)).any()):
        raise ValueError("Invalid HDBSCAN membership-strength diagnostic.")
    result = points.copy()
    result["cluster_label"] = labels.astype("int64")
    result["membership_strength"] = strengths
    return result.loc[members].reset_index(drop=True)


def seasonal_activity(points, resolution, year):
    validate_source_points(points, year)
    work = pd.DataFrame({"cell_id": point_cell_ids(points[XY].to_numpy(float), resolution),
                         "utc_date": points[UTC].dt.strftime("%Y-%m-%d").to_numpy()})
    if "membership_strength" in points:
        work["strength"] = points.membership_strength.to_numpy(float)
    else:
        work["strength"] = np.nan
    result = work.groupby("cell_id", sort=True).agg(
        detections=("utc_date", "size"), distinct_UTC_dates=("utc_date", "nunique"),
        strength_available=("strength", "count"), strength_sum=("strength", "sum"),
        strength_min=("strength", "min"), strength_max=("strength", "max")).reset_index()
    result["year"] = year
    result["activity"] = np.int8(1)
    assert int(result.detections.sum()) == len(points)
    assert (result.distinct_UTC_dates <= result.detections).all()
    return result


def recurrent_cells(activity):
    required = ["cell_id", "year", "detections", "distinct_UTC_dates"]
    if activity.duplicated(["cell_id", "year"]).any():
        raise ValueError("A cell has more than one aggregate per primary year.")
    if not set(activity.year) <= set(YEARS) or (activity.detections <= 0).any():
        raise ValueError("Sparse activity must contain positive primary-season support only.")
    if (activity.distinct_UTC_dates <= 0).any() or (activity.distinct_UTC_dates > activity.detections).any():
        raise ValueError("Invalid distinct-date accounting.")
    work = activity[required].copy()
    work["season_bit"] = np.left_shift(1, work.year.to_numpy("int64")-2019)
    result = work.groupby("cell_id", sort=True).agg(
        active_seasons=("year", "nunique"), season_mask=("season_bit", "sum"),
        detections=("detections", "sum"), distinct_UTC_dates=("distinct_UTC_dates", "sum"),
        first_active_year=("year", "min"), last_active_year=("year", "max")).reset_index()
    result["active_years"] = result.season_mask.map(lambda mask: ";".join(str(y) for y in YEARS if int(mask)&(1 << (y-2019))))
    result["activity_span_years"] = result.last_active_year-result.first_active_year
    result["observed_active_season_fraction"] = result.active_seasons/7
    assert result.active_seasons.between(1, 7).all()
    assert all(int(mask).bit_count() == n for mask, n in zip(result.season_mask, result.active_seasons))
    return result


def assert_supported(detection, clustered):
    left = detection.set_index("cell_id").season_mask
    right = clustered.set_index("cell_id").season_mask
    if not set(right.index) <= set(left.index):
        raise ValueError("Cluster-supported cell lacks detection support.")
    if ((right.to_numpy("int64") & ~left.loc[right.index].to_numpy("int64")) != 0).any():
        raise ValueError("Cluster-supported season lacks detection support.")


def threshold_ids(frame, k):
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 7:
        raise ValueError("Threshold is an integer number of seasons, 1..7.")
    return set(frame.loc[frame.active_seasons >= k, "cell_id"].astype(int))


def frequency_distribution(frame, areas):
    if frame.cell_id.duplicated().any() or not set(frame.cell_id) <= set(areas):
        raise ValueError("Recurrence rows must be unique eligible cells.")
    if not frame.active_seasons.between(1, 7).all():
        raise ValueError("Sparse recurrence must contain counts 1..7.")
    active = set(frame.cell_id)
    rows = []
    for count in range(8):
        selected = (set(areas)-active if count == 0 else
                    set(frame.loc[frame.active_seasons == count, "cell_id"]))
        rows.append({"active_seasons": count, "cells": len(selected),
                     "clipped_area_km2": math.fsum(areas[c] for c in selected)/1e6})
    assert sum(r["cells"] for r in rows) == len(areas)
    return rows


def set_comparison(left, right, areas):
    a, b = set(left), set(right)
    if not (a | b) <= set(areas):
        raise ValueError("Compared cells must share a verified eligible grid.")
    intersection, union = a & b, a | b
    area = lambda ids: math.fsum(float(areas[c]) for c in ids)
    uarea = area(union)
    return {"left_cells": len(a), "right_cells": len(b), "intersection_cells": len(intersection),
            "union_cells": len(union), "left_only_cells": len(a-b), "right_only_cells": len(b-a),
            "cell_jaccard": len(intersection)/len(union) if union else None,
            "intersection_area_km2": area(intersection)/1e6, "union_area_km2": uarea/1e6,
            "area_jaccard": area(intersection)/uarea if uarea else None}


def connected_cells(cells, adjacency=4):
    """Deterministic grid-index connectivity; each component's ID is its min cell ID.

    Clipped geometry can be multipart. Connectivity describes qualifying fixed
    cells, not continuity of a burn perimeter or a single fire event.
    """
    if adjacency not in (4, 8):
        raise ValueError("Use four- or eight-neighbour adjacency.")
    if cells.cell_id.duplicated().any() or cells.duplicated(["grid_x", "grid_y"]).any():
        raise ValueError("Repeated grid cells in component input.")
    remaining = {(int(r.grid_x), int(r.grid_y)): int(r.cell_id) for r in cells.itertuples()}
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if adjacency == 8:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    result = []
    for start in sorted(remaining, key=remaining.get):
        if start not in remaining:
            continue
        queue = deque([start])
        member_ids = [remaining.pop(start)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in offsets:
                target = (x+dx, y+dy)
                if target in remaining:
                    member_ids.append(remaining.pop(target))
                    queue.append(target)
        result.append(sorted(member_ids))
    return result


def clip_cells(ix, iy, resolution, mask, origin=(0, 0)):
    x, y = np.broadcast_arrays(np.asarray(ix, dtype="int64"), np.asarray(iy, dtype="int64"))
    cell_ids(x, y, resolution)  # validate both indexing and metre resolution
    boxes = shapely.box(origin[0]+x*resolution, origin[1]+y*resolution,
                        origin[0]+(x+1)*resolution, origin[1]+(y+1)*resolution)
    clipped = shapely.intersection(boxes, mask)
    areas = shapely.area(clipped)
    return clipped, areas


def district_parts(geometry, districts):
    """Exact positive-area intersections, never centroid attribution."""
    parts = []
    for name, polygon in districts.items():
        if geometry.intersects(polygon):
            area = geometry.intersection(polygon).area
            if area > 0:
                parts.append((name, area))
    if not math.isclose(math.fsum(v for _, v in parts), geometry.area, rel_tol=1e-11, abs_tol=.001):
        raise ValueError("District intersections do not reconcile with clipped cell area.")
    return parts


def common_support_ids(fine_cells, coarse_ids, resolution):
    if resolution not in RESOLUTIONS:
        raise ValueError("Unapproved comparison resolution.")
    parents = cell_ids(fine_cells.grid_x.to_numpy("int64") // (resolution//500),
                       fine_cells.grid_y.to_numpy("int64") // (resolution//500), resolution)
    return set(fine_cells.loc[np.isin(parents, list(coarse_ids)), "cell_id"].astype(int))
