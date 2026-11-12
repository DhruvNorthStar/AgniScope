"""A district-complete analytical footprint without editing source boundaries."""

from itertools import combinations
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely

from .config import workspace_path


def _validate_polygons(frame: gpd.GeoDataFrame, label: str) -> None:
    if frame.crs is None or frame.empty:
        raise ValueError(f"{label}: missing CRS or empty layer.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError(f"{label}: null, empty or invalid geometry; no automatic repair is permitted.")
    if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{label}: expected polygon geometries.")


def load_boundaries(config: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    paths = config["paths"]
    state = gpd.read_file(workspace_path(paths["state_boundary"]), engine="pyogrio")
    districts = gpd.read_file(workspace_path(paths["district_boundaries"]), engine="pyogrio")
    _validate_polygons(state, "state")
    _validate_polygons(districts, "districts")
    if len(state) != 1 or len(districts) != 13:
        raise ValueError("Unexpected source feature count; investigate before proceeding.")
    original = config["spatial"]["district_original_column"]
    district_column = config["spatial"]["district_column"]
    for field in (original, district_column):
        if field not in districts or districts[field].isna().any() or districts[field].duplicated().any():
            raise ValueError(f"Missing, null or duplicate district label: {field}")
    aliases = pd.read_csv(workspace_path(paths["district_aliases"]), dtype=str, keep_default_na=False)
    if aliases["district_original"].duplicated().any() or not set(aliases["district_original"]).issubset(set(districts[original])):
        raise ValueError("District display aliases do not match unique source labels.")
    mapping = dict(zip(aliases["district_original"], aliases["display_name"]))
    districts["agniscope_display_name"] = districts[original].map(mapping).fillna(districts[original])
    crs = config["spatial"]["analysis_crs"]
    parsed_crs = CRS.from_user_input(crs)
    if not parsed_crs.is_projected or not all(axis.unit_name.lower() in {"metre", "meter"} for axis in parsed_crs.axis_info):
        raise ValueError("Analysis CRS must be projected in metres.")
    return state.to_crs(crs), districts.to_crs(crs)


def boundary_report(state: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> dict:
    outline = state.geometry.union_all()
    footprint = districts.geometry.union_all()
    overlap = sum(a.intersection(b).area for a, b in combinations(districts.geometry, 2))
    if overlap > 1e-6:
        raise ValueError("Positive-area district overlaps need investigation.")
    if not footprint.is_valid:
        raise ValueError("District-union geometry is invalid.")
    geographic = gpd.GeoSeries([footprint], crs=districts.crs).to_crs(4326)
    return {
        "analysis_crs": str(districts.crs),
        "measurement": "planar areas in projected CRS, not official administrative area figures",
        "primary_footprint": "district_union_2021",
        "source_state_features": len(state),
        "source_district_features": len(districts),
        "all_geometries_valid": bool(state.geometry.is_valid.all() and districts.geometry.is_valid.all()),
        "state_area_km2": outline.area / 1e6,
        "district_union_area_km2": footprint.area / 1e6,
        "district_pairwise_overlap_km2": overlap / 1e6,
        "state_outside_district_union_km2": outline.difference(footprint).area / 1e6,
        "district_union_outside_state_km2": footprint.difference(outline).area / 1e6,
        "symmetric_difference_km2": outline.symmetric_difference(footprint).area / 1e6,
        "district_union_wgs84_bounds": geographic.total_bounds.tolist(),
        "districts": districts[["district", "district_original", "agniscope_display_name"]].to_dict("records"),
        "licence": "ODbL-1.0; derived from geoBoundaries IND ADM2, representing 2021",
    }


def save_footprint(districts: gpd.GeoDataFrame, config: dict):
    target = workspace_path(config["paths"]["analysis_footprint"], writable=True)
    footprint = districts.geometry.union_all()
    if target.exists():
        existing = gpd.read_file(target, layer="analysis_footprint", engine="pyogrio")
        if existing.crs != districts.crs or len(existing) != 1 or not existing.geometry.iloc[0].equals(footprint):
            raise ValueError("Existing derived footprint differs; refusing to overwrite it.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    result = gpd.GeoDataFrame(
        {"footprint": ["district_union_2021"], "represented_year": [2021], "licence": ["ODbL-1.0"]},
        geometry=[footprint], crs=districts.crs,
    )
    result.to_file(target, layer="analysis_footprint", driver="GPKG", engine="pyogrio")
    return target


def assign_points(points: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, *, near_distance_m: float = 375.0) -> pd.DataFrame:
    """Include outer edges; preserve shared-edge ambiguity instead of forcing a district."""
    if points.crs is None or districts.crs is None:
        raise ValueError("Point and district CRS must be explicit.")
    if points.geometry.isna().any() or points.geometry.is_empty.any() or not points.geom_type.eq("Point").all():
        raise ValueError("Expected nonempty point geometries.")
    if not CRS(districts.crs).is_projected:
        raise ValueError("District CRS must be projected for distance diagnostics.")
    projected = points.to_crs(districts.crs)
    footprint = districts.geometry.union_all()
    candidates: list[list[int]] = [[] for _ in range(len(projected))]
    if len(projected):
        point_positions, district_positions = districts.sindex.query(projected.geometry, predicate="intersects")
        for point_pos, district_pos in zip(point_positions, district_positions):
            candidates[int(point_pos)].append(int(district_pos))
    rows = []
    for position, point in enumerate(projected.geometry):
        matches = candidates[position]
        inside = bool(footprint.covers(point))
        originals = sorted(str(districts.iloc[p]["district_original"]) for p in matches)
        row = {
            "agniscope_in_footprint": inside,
            "agniscope_district_original": None,
            "agniscope_district_display": None,
            "agniscope_district_candidates": json.dumps(originals, ensure_ascii=False),
            "agniscope_on_outer_edge": bool(inside and footprint.boundary.intersects(point)),
            "agniscope_near_outer_edge": bool(point.distance(footprint.boundary) <= near_distance_m),
            "agniscope_x_m": float(point.x), "agniscope_y_m": float(point.y),
        }
        if not inside:
            row["agniscope_district_assignment"] = "outside_footprint"
        elif len(matches) == 1:
            district = districts.iloc[matches[0]]
            row["agniscope_district_original"] = str(district["district_original"])
            row["agniscope_district_display"] = str(district.get("agniscope_display_name", district["district_original"]))
            row["agniscope_district_assignment"] = "assigned_interior" if district.geometry.contains(point) else "assigned_single_edge"
        elif len(matches) > 1:
            row["agniscope_district_assignment"] = "ambiguous_shared_edge"
        else:
            row["agniscope_district_assignment"] = "unassigned_inside"
        rows.append(row)
    columns = ["agniscope_in_footprint", "agniscope_district_original", "agniscope_district_display", "agniscope_district_candidates", "agniscope_on_outer_edge", "agniscope_near_outer_edge", "agniscope_x_m", "agniscope_y_m", "agniscope_district_assignment"]
    return pd.DataFrame(rows, columns=columns, index=points.index)
