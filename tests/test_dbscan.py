"""Synthetic scientific-method tests; authentic pilot results are separate evidence."""

import hashlib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Geod, Transformer
import pytest

from agniscope.dbscan import (ID, UTC, XY, DISTRICT, TREE, eps_metres, parameter_grid,
    prepare_points, daily_grid_sample, sensitivity_subsets, fit_configuration,
    configuration_statistics, cluster_summaries, diagnostic_tables, run_grid,
    repeat_diagnostics, distance_crosscheck, membership_comparison)


def points_at(xy):
    xy = np.asarray(xy, dtype=float) + [500000, 3300000]
    lon, lat = Transformer.from_crs(32644, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
    n = len(xy)
    return prepare_points(pd.DataFrame({
        ID: np.arange(1, n + 1), "longitude": [f"{x:.15f}" for x in lon],
        "latitude": [f"{x:.15f}" for x in lat], UTC: ["2024-04-01T00:00:00Z"] * n,
        DISTRICT: ["A"] * n, TREE: [10] * n, "agniscope_confidence": ["nominal"] * n,
        "daynight": ["N"] * n, "frp": ["2.5"] * n,
        "source_extra": ["keep original"] * n,
    }))


@pytest.mark.parametrize("km,metres", [(.5, 500), (1, 1000), (2, 2000)])
def test_epsilon_units(km, metres):
    assert eps_metres(km) == metres


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_epsilon(value):
    with pytest.raises(ValueError):
        eps_metres(value)


def test_nine_explicit_settings_and_rejected_duplicates():
    grid = parameter_grid([.5, 1, 2], [3, 5, 10])
    assert len(grid) == len(set(grid)) == 9
    assert grid == [(.5, 3), (.5, 5), (.5, 10), (1, 3), (1, 5), (1, 10), (2, 3), (2, 5), (2, 10)]
    for eps, minimum in [([1, 1], [3]), ([1], [2.5]), ([], [3]), ([1], [True])]:
        with pytest.raises(ValueError):
            parameter_grid(eps, minimum)


def test_projection_distance_and_geographic_crs_rejection():
    points = points_at([[0, 0], [1000, 0], [0, 2000]])
    assert np.linalg.norm(points[XY].iloc[0] - points[XY].iloc[1]) == pytest.approx(1000, abs=1e-6)
    a, b = points.iloc[0], points.iloc[1]
    _, _, geodesic = Geod(ellps="WGS84").inv(float(a.longitude), float(a.latitude), float(b.longitude), float(b.latitude))
    assert geodesic == pytest.approx(1000, rel=.001)
    with pytest.raises(ValueError, match="32644"):
        prepare_points(points, "EPSG:4326")
    mislabelled = points.copy()
    mislabelled[XY] = mislabelled[["longitude", "latitude"]].astype(float).to_numpy()
    with pytest.raises(ValueError, match="disagree"):
        prepare_points(mislabelled)


def test_fit_requires_established_projected_units():
    points = points_at([[0, 0], [100, 0], [200, 0]])
    points.attrs.clear()
    with pytest.raises(ValueError, match="prepare_points"):
        fit_configuration(points, 1, 3)


def test_original_identity_coordinates_and_in_memory_input_preserved():
    points = points_at([[0, 0], [20, 0], [40, 0], [5000, 0]])
    points[ID] = [613, 37, 350, 80]
    original = points.copy(deep=True)
    result = fit_configuration(points, .5, 3)
    assert_frame_equal(points, original)
    assert result[ID].tolist() == [37, 80, 350, 613]
    for col in ["latitude", "longitude", "source_extra"]:
        assert result.set_index(ID)[col].to_dict() == original.set_index(ID)[col].to_dict()
    assert result.set_index(ID).loc[80, "cluster_id"] == -1


@pytest.mark.parametrize("identities", [[1, 1], [1, 1.5], [0, 2]])
def test_invalid_source_identities(identities):
    points = points_at([[0, 0], [50, 0]])
    points[ID] = identities
    with pytest.raises(ValueError, match="identities"):
        prepare_points(points)


def test_zero_cluster_and_empty_input():
    points = points_at([[0, 0], [3000, 0], [6000, 0]])
    result = fit_configuration(points, .5, 3)
    summary = configuration_statistics(result)
    assert result.cluster_id.tolist() == [-1, -1, -1]
    assert summary["n_clusters"] == 0 and summary["n_noise"] == 3
    assert summary["noise_pct_of_input"] == 100
    assert summary["size_min"] is None and summary["silhouette_nonnoise"] is None
    assert summary["silhouette_status"] == "unavailable_no_clusters"
    assert cluster_summaries(result).empty
    empty = configuration_statistics(fit_configuration(points.iloc[:0], .5, 3))
    assert empty["n_observations"] == 0 and empty["noise_pct_of_input"] is None


def test_one_cluster_core_includes_self_and_undefined_silhouette():
    result = fit_configuration(points_at([[0, 0], [100, 0], [200, 0]]), .11, 3)
    summary = configuration_statistics(result)
    assert summary["n_clusters"] == 1
    assert summary["n_core"] == 1 and summary["n_border"] == 2
    assert summary["silhouette_nonnoise"] is None
    assert summary["silhouette_status"] == "unavailable_one_cluster"
    # Epsilon constrains links, not the entire connected cluster's diameter.
    assert cluster_summaries(result).max_pair_distance_m.iloc[0] == pytest.approx(200, abs=1e-6)


def test_silhouette_invalid_singleton_clusters():
    result = fit_configuration(points_at([[0, 0], [3000, 0], [6000, 0]]), .5, 1)
    stats = configuration_statistics(result)
    assert stats["n_clusters"] == 3
    assert stats["silhouette_nonnoise"] is None
    assert stats["silhouette_status"] == "unavailable_one_point_per_cluster"


def test_known_cluster_statistics_and_district_spanning():
    points = points_at([[0, 0], [10, 0], [20, 0], [5000, 0], [5010, 0], [5020, 0], [5030, 0], [10000, 0]])
    points.loc[1, DISTRICT] = "B"
    result = fit_configuration(points, .5, 3)
    stats = configuration_statistics(result)
    assert (stats["n_clusters"], stats["n_noise"], stats["n_clustered"]) == (2, 1, 7)
    assert (stats["size_min"], stats["size_median"], stats["size_max"]) == (3, 3.5, 4)
    assert stats["largest_share_pct_of_input"] == 50
    assert stats["largest_share_pct_of_clustered"] == pytest.approx(400 / 7)
    assert .9 < stats["silhouette_nonnoise"] <= 1
    clusters = cluster_summaries(result)
    assert clusters.n_districts.tolist() == [2, 1]
    assert clusters.iloc[0].districts_original == "A;B"
    assert clusters.iloc[0].frp_mean_mw == 2.5


def test_subsets_are_separate_and_do_not_relabel_confidence():
    points = points_at([[0, 0], [10, 0], [20, 0], [30, 0]])
    points.agniscope_confidence = ["nominal", "low", "nominal", "high"]
    points[TREE] = [10, 10, 30, 10]
    subsets, _ = sensitivity_subsets(points)
    assert len(subsets["baseline"]) == 4
    assert subsets["nominal_only"][ID].tolist() == [1, 3]
    assert subsets["tree_cover_only"][ID].tolist() == [1, 2, 4]
    assert points.agniscope_confidence.tolist() == ["nominal", "low", "nominal", "high"]


def test_temporal_sample_utc_tie_breaking_and_ledger():
    points = points_at([[100, 100], [200, 100], [300, 100], [400, 100], [1200, 100]])
    points[UTC] = pd.to_datetime(["2024-04-01T01:00Z", "2024-04-01T00:00Z", "2024-04-01T00:00Z", "2024-04-02T00:00Z", "2024-04-01T00:00Z"], utc=True)
    points.utc_date = points[UTC].dt.strftime("%Y-%m-%d")
    sampled, ledger = daily_grid_sample(points.sample(frac=1, random_state=8), 1000)
    assert sampled[ID].tolist() == [2, 4, 5]
    assert ledger.set_index(ID).representative_source_record.to_dict() == {1: 2, 2: 2, 3: 2, 4: 4, 5: 5}
    assert ledger.selected.sum() == 3
    assert len(points) == 5


def test_grid_half_open_cells_and_explicit_origin():
    points = points_at([[0, 0], [10, 0], [20, 0]])
    points[XY] = [[-.1, 0], [0, 0], [.1, 0]]
    sampled, ledger = daily_grid_sample(points, 1000)
    assert len(sampled) == 2
    assert ledger.grid_x.tolist() == [-1, 0, 0]
    assert len(daily_grid_sample(points, 1000, (-500, 0))[0]) == 1


def test_repeat_pairs_same_vs_different_timestamp():
    points = points_at([[0, 0], [100, 0], [200, 0], [200, 0]])
    points.loc[2:, UTC] = pd.Timestamp("2024-04-02T00:00Z")
    points.loc[2:, "utc_date"] = "2024-04-02"
    pairs, counts, repeated = repeat_diagnostics(points, [375])
    assert len(pairs) == 6
    assert counts.iloc[0].different_timestamp_pairs == 4
    assert counts.iloc[0].different_utc_day_pairs == 4
    assert repeated["repeated_coordinate_groups"] == 1
    assert repeated["observations_in_repeated_coordinate_groups"] == 2


def test_coordinate_repeats_use_numeric_coordinates_preserving_raw_spelling():
    points = points_at([[0, 0], [0, 0]])
    points.loc[1, "latitude"] = points.loc[0, "latitude"] + "0"
    spelling = points.latitude.tolist()
    _, _, repeated = repeat_diagnostics(points)
    assert repeated["repeated_coordinate_groups"] == 1
    assert points.latitude.tolist() == spelling


def test_district_temporal_daynight_frp_accounting():
    points = points_at([[0, 0], [10, 0], [20, 0], [5000, 0]])
    points.loc[3, "daynight"] = "D"
    points.loc[3, "frp"] = "8"
    tables = diagnostic_tables(fit_configuration(points, .5, 3), ["A", "B"], ["2024-04-01", "2024-04-02"])
    for key in ["district", "daily", "daynight", "cluster_daily", "cluster_daynight", "frp"]:
        assert tables[key].n_observations.sum() == 4
    assert tables["district"].loc[tables["district"][DISTRICT].eq("B"), "n_observations"].sum() == 0
    assert tables["frp"].set_index("membership").loc["noise", "frp_mean_mw"] == 8
    assert pd.isna(tables["frp"].set_index("membership").loc["noise", "frp_sample_std_mw"])


def test_grid_reproducibility_and_raw_file_preservation(tmp_path):
    points = points_at([[0, 0], [100, 0], [200, 0], [5000, 0], [5100, 0], [5200, 0], [15000, 0]])
    source = tmp_path / "synthetic_input.csv"
    points.to_csv(source, index=False)
    digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
    first = run_grid(points, [.5, 1, 2], [3, 5, 10], district_names=["A"], utc_dates=["2024-04-01"])
    second = run_grid(points.sample(frac=1, random_state=7), [.5, 1, 2], [3, 5, 10], district_names=["A"], utc_dates=["2024-04-01"])
    assert len(first["parameters"]) == 9
    assert len(first["assignments"]) == 9 * len(points)
    for key in first:
        assert_frame_equal(first[key], second[key])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest_before


def test_alternate_distance_check_and_label_invariant_comparison():
    points = points_at([[0, 0], [100, 0], [200, 0], [6000, 0]])
    check, selected = distance_crosscheck(points)
    assert check["n_pairs_checked"] == 6
    assert check["max_abs_relative_utm_vs_geodesic_pct"] < .1
    assert len(selected) >= 2
    left = fit_configuration(points, .5, 3)
    right = left.copy()
    right.cluster_id = right.cluster_id.replace({0: 7})
    comparison = membership_comparison(left, right)
    assert comparison["ari_on_common_including_noise"] == 1
    assert comparison["clustered_in_both"] == 3
    assert comparison["noise_in_both"] == 1
