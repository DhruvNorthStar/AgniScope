import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from agniscope.config import workspace_path
from agniscope.spatial import assign_points, boundary_report, load_boundaries


def test_interior_outer_edge_shared_edge_and_outside(districts):
    points = gpd.GeoDataFrame(geometry=[Point(500050, 3300050), Point(500000, 3300050), Point(500100, 3300050), Point(499999, 3300050)], crs=32644)
    assigned = assign_points(points, districts)
    assert assigned["agniscope_district_assignment"].tolist() == ["assigned_interior", "assigned_single_edge", "ambiguous_shared_edge", "outside_footprint"]
    assert assigned["agniscope_in_footprint"].tolist() == [True, True, True, False]
    assert pd.isna(assigned.iloc[2]["agniscope_district_original"])
    assert assigned.iloc[2]["agniscope_district_candidates"] == '["A", "B"]'


def test_crs_transform_preserves_assignment_and_xy(districts):
    projected = gpd.GeoDataFrame(geometry=[Point(500050, 3300050)], crs=32644)
    one = assign_points(projected, districts)
    two = assign_points(projected.to_crs(4326), districts)
    assert two.iloc[0]["agniscope_district_original"] == "A"
    assert np.allclose(one[["agniscope_x_m", "agniscope_y_m"]], two[["agniscope_x_m", "agniscope_y_m"]], atol=1e-6, rtol=0)


def test_missing_crs_rejected(districts):
    with pytest.raises(ValueError, match="CRS"):
        assign_points(gpd.GeoDataFrame(geometry=[Point(0, 0)]), districts)


def test_workspace_confinement_and_source_write_protection():
    with pytest.raises(ValueError, match="inside"):
        workspace_path("../outside.csv")
    with pytest.raises(ValueError, match="immutable"):
        workspace_path("data/external/should_not_exist.txt", writable=True)


def test_actual_local_boundaries(config):
    state, districts = load_boundaries(config)
    report = boundary_report(state, districts)
    assert report["all_geometries_valid"]
    assert abs(report["symmetric_difference_km2"] - 623.1405717982587) < 0.01
    assert report["district_pairwise_overlap_km2"] < 1e-8
    assert set(districts["district_original"]) >= {"Garhwal", "Hardwar"}
    assert districts.set_index("district_original").loc["Garhwal", "agniscope_display_name"] == "Pauri Garhwal"
    points = gpd.GeoDataFrame(geometry=districts.geometry.representative_point(), crs=districts.crs)
    assigned = assign_points(points, districts)
    assert assigned["agniscope_district_assignment"].eq("assigned_interior").all()
    assert assigned["agniscope_district_original"].tolist() == districts["district_original"].tolist()
