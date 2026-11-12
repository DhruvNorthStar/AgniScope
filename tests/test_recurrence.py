"""Synthetic fixtures only: grid, membership, recurrence and clipped geography."""
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
import pytest
import shapely
from agniscope import recurrence as r


def points(n=4):
    return pd.DataFrame({r.ID: ["a"*64+":"+str(i+1) for i in range(n)],
                         r.UTC: pd.to_datetime(["2019-02-20T12:00:00Z"]*n),
                         r.XY[0]: np.arange(n)*100., r.XY[1]: np.zeros(n)})


@pytest.mark.parametrize("resolution", [500, 1000, 2000])
def test_half_open_edges_negative_indices_and_numeric_ids(resolution):
    xy = np.array([[0., 0.], [resolution, resolution], [-.001, -.001], [resolution-.001, 0.]])
    actual = r.grid_indices(xy, resolution)
    independent = np.array([[math.floor(x/resolution), math.floor(y/resolution)] for x, y in xy])
    assert np.array_equal(actual, independent)
    assert actual.tolist() == [[0, 0], [1, 1], [-1, -1], [0, 0]]
    ids = r.point_cell_ids(xy, resolution)
    assert ids.dtype == np.int64 and ids[0] == ids[-1] and len(set(ids)) == 3
    assert len({int(r.cell_ids(0, 0, size)) for size in r.RESOLUTIONS}) == 3


@pytest.mark.parametrize("bad", [.5, 1, 2, 375, 999, "1000", None])
def test_resolution_units_are_metres_and_allowlisted(bad):
    with pytest.raises(ValueError): r.grid_indices([[1, 1]], bad)


def test_origin_alignment_nesting_and_invalid_coordinates():
    xy = np.array([[-1000.01, 3499999.9], [170982.9264, 3176225.313], [506000, 3482000]])
    fine = r.grid_indices(xy, 500)
    assert np.array_equal(fine//2, r.grid_indices(xy, 1000))
    assert np.array_equal(fine//4, r.grid_indices(xy, 2000))
    for invalid in ([500, 0], [np.nan, 0]):
        with pytest.raises(ValueError): r.grid_indices(xy, 500, invalid)
    with pytest.raises(ValueError): r.grid_indices([[np.inf, 1]], 500)
    with pytest.raises(ValueError): r.cell_ids(0.5, 0, 500)
    with pytest.raises(ValueError): r.cell_ids(32768, 0, 500)


def test_repeated_detections_give_only_one_season_vote():
    data = points(100)
    data[r.XY[0]] = 100.
    activity = r.seasonal_activity(data, 1000, 2019)
    later = data.iloc[:1].copy()
    later[r.UTC] = pd.to_datetime(["2025-06-15T23:59:59Z"])
    combined = pd.concat([activity, r.seasonal_activity(later, 1000, 2025)])
    recurrent = r.recurrent_cells(combined)
    assert recurrent.active_seasons.tolist() == [2] and recurrent.detections.tolist() == [101]
    assert recurrent.distinct_UTC_dates.tolist() == [2]
    assert recurrent.active_years.tolist() == ["2019;2025"] and recurrent.activity_span_years.tolist() == [6]
    assert recurrent.observed_active_season_fraction.iloc[0] == 2/7
    pd.testing.assert_frame_equal(recurrent, r.recurrent_cells(combined.sample(frac=1, random_state=7)))
    with pytest.raises(ValueError): r.recurrent_cells(pd.concat([activity, activity]))


@pytest.mark.parametrize("day,accepted", [("2019-02-14T23:59:59Z", False), ("2019-02-15T00:00:00Z", True),
                                         ("2019-06-15T23:59:59Z", True), ("2019-06-16T00:00:00Z", False),
                                         ("2020-02-29T00:00:00Z", False)])
def test_primary_membership_uses_utc_and_fixed_year(day, accepted):
    frame = points(1)
    frame[r.UTC] = pd.to_datetime([day])
    if accepted: r.validate_source_points(frame, 2019)
    else:
        with pytest.raises(ValueError): r.validate_source_points(frame, 2019)


def test_source_identity_and_timezone_contract():
    frame = points()
    frame.loc[1, r.ID] = frame.loc[0, r.ID]
    with pytest.raises(ValueError): r.validate_source_points(frame, 2019)
    frame = points()
    frame[r.UTC] = frame[r.UTC].dt.tz_localize(None)
    with pytest.raises(ValueError): r.validate_source_points(frame, 2019)
    with pytest.raises(ValueError): r.validate_source_points(points(), 2018)


def test_exact_membership_joins_and_all_negative_hdbscan_labels():
    frame = points(4)
    assigned = frame.copy()
    assigned["cluster_label"] = [4, -1, -2, -3]
    assigned["membership_strength"] = [0.7, 0., 0., np.nan]
    result = r.join_memberships(frame, assigned.iloc[::-1], "hdbscan")
    assert result[r.ID].tolist() == frame[r.ID].iloc[:1].tolist()
    assert result.membership_strength.tolist() == [.7]
    for broken in (assigned.iloc[:3], pd.concat([assigned, assigned.iloc[:1]])):
        with pytest.raises(ValueError): r.join_memberships(frame, broken, "hdbscan")
    broken = assigned.copy()
    broken.loc[0, r.XY[0]] += .01
    with pytest.raises(ValueError): r.join_memberships(frame, broken, "hdbscan")


def test_detection_superset_is_season_specific():
    d = pd.DataFrame({"cell_id": [1, 2], "season_mask": [3, 4]})
    r.assert_supported(d, pd.DataFrame({"cell_id": [1], "season_mask": [1]}))
    with pytest.raises(ValueError): r.assert_supported(d, pd.DataFrame({"cell_id": [1], "season_mask": [4]}))
    with pytest.raises(ValueError): r.assert_supported(d, pd.DataFrame({"cell_id": [3], "season_mask": [1]}))


def test_empty_eligible_cells_and_all_thresholds():
    frame = pd.DataFrame({"cell_id": [1, 2, 3, 4, 5, 6, 7], "active_seasons": list(range(1, 8))})
    areas = {i: 100. for i in range(9)}
    distribution = r.frequency_distribution(frame, areas)
    assert [row["cells"] for row in distribution] == [2, 1, 1, 1, 1, 1, 1, 1]
    for k in [2, 3, 4, 5]:
        assert r.threshold_ids(frame, k) == set(range(k, 8))
    for k in [0, 8, 3.5, True]:
        with pytest.raises(ValueError): r.threshold_ids(frame, k)


def test_four_and_eight_neighbours_component_identity_and_shuffle():
    cells = pd.DataFrame({"cell_id": [100, 200, 300, 400], "grid_x": [0, 1, 5, 5], "grid_y": [0, 1, 5, 6]})
    assert r.connected_cells(cells, 4) == [[100], [200], [300, 400]]
    assert r.connected_cells(cells, 8) == [[100, 200], [300, 400]]
    assert r.connected_cells(cells.sample(frac=1, random_state=3), 8) == r.connected_cells(cells, 8)
    with pytest.raises(ValueError): r.connected_cells(pd.concat([cells, cells.iloc[:1]]))


def test_clipped_partial_area_not_nominal_cell_area_and_district_fractions():
    mask = shapely.box(0, 0, 750, 1000)
    clipped, area = r.clip_cells([0, 1], [0, 0], 1000, mask)
    assert area.tolist() == [750000., 0.]
    districts = {"original_A": shapely.box(0, 0, 500, 1000), "original_B": shapely.box(500, 0, 750, 1000)}
    parts = dict(r.district_parts(clipped[0], districts))
    assert parts == {"original_A": 500000., "original_B": 250000.}
    assert math.fsum(parts.values()) == area[0]
    with pytest.raises(ValueError): r.district_parts(clipped[0], {"A": districts["original_A"]})


def test_common_resolution_support_and_jaccard_area_denominators():
    fine = pd.DataFrame({"grid_x": [0, 1, 2, 3], "grid_y": [0, 0, 0, 0]})
    fine["cell_id"] = r.cell_ids(fine.grid_x, fine.grid_y, 500)
    support = r.common_support_ids(fine, {int(r.cell_ids(0, 0, 1000))}, 1000)
    assert support == set(fine.cell_id.iloc[:2])
    areas = {1: 100., 2: 50., 3: 25.}
    comparison = r.set_comparison({1, 2}, {2, 3}, areas)
    assert comparison["cell_jaccard"] == 1/3 and comparison["area_jaccard"] == 50/175
    assert r.set_comparison(set(), set(), areas)["cell_jaccard"] is None


def test_empty_season_is_not_missing_execution():
    empty = r.seasonal_activity(points().iloc[:0], 1000, 2019)
    assert len(empty) == 0 and len(r.recurrent_cells(empty)) == 0
