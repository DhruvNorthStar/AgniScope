"""Focused HDBSCAN semantics and fair comparison tests, using synthetic data."""

import hashlib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Transformer
import pytest

from agniscope.config import load_config, workspace_path
from agniscope.dbscan import ID, XY, DISTRICT, TREE, UTC, prepare_points, sensitivity_subsets
from agniscope import hdbscan as h


@pytest.fixture
def fixed():
    return load_config(workspace_path("configs/phase1b.toml"))["estimator"]


def synthetic_points(n=20):
    xy = np.array([[500000 + (i % 10) * 10 + (i // 10) * 5000, 3300000 + (i % 3) * 10] for i in range(n)])
    lon, lat = Transformer.from_crs(32644, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
    return prepare_points(pd.DataFrame({ID: np.arange(1, n + 1), "latitude": lat.astype(str), "longitude": lon.astype(str),
        UTC: ["2024-04-01T01:00Z"] * n, DISTRICT: ["A"] * n, TREE: [10] * n,
        "agniscope_confidence": ["nominal"] * n, "daynight": ["N"] * n, "frp": ["2.5"] * n,
        "original_extra": ["preserve"] * n}))


def labelled(labels, probabilities=None):
    points = synthetic_points(len(labels))
    points["cluster_id"] = labels
    if probabilities is not None:
        points["membership_probability"] = probabilities
    return points


def test_declared_twelve_settings_and_sklearn_semantics(fixed):
    settings = load_config(workspace_path("configs/phase1b.toml"))
    grid = h.parameter_grid(settings["experiment"]["min_cluster_size"], settings["experiment"]["min_samples"])
    assert len(grid) == len(set(grid)) == 12
    assert grid == [(s, m) for s in [3, 5, 10, 15] for m in [3, 5, 10]]
    assert h.HDBSCAN.__module__.startswith("sklearn.cluster")
    params = h.estimator_parameters(5, 3, fixed)
    assert params["min_samples"] == 3
    assert params["cluster_selection_method"] == "eom"
    assert params["cluster_selection_epsilon"] == 0
    assert params["allow_single_cluster"] is False
    assert params["copy"] is True and params["n_jobs"] == 1
    assert params["max_cluster_size"] is params["metric_params"] is params["store_centers"] is None


@pytest.mark.parametrize("sizes,samples", [([], [3]), ([3, 3], [3]), ([1], [3]), ([3.5], [3]), ([3], [0]), ([3], [True])])
def test_invalid_grid(sizes, samples):
    with pytest.raises(ValueError):
        h.parameter_grid(sizes, samples)


@pytest.mark.parametrize("field,value", [("metric", "haversine"), ("cluster_selection_method", "leaf"), ("cluster_selection_epsilon", 1000), ("n_jobs", -1), ("copy", False)])
def test_no_hidden_selection_epsilon_or_distance_change(fixed, field, value):
    with pytest.raises(ValueError):
        h.estimator_parameters(5, 3, {**fixed, field: value})


def test_distance_units_are_projected_metres_and_input_order_is_stable(fixed, monkeypatch):
    points = synthetic_points()
    captured = {}
    original_class = h.HDBSCAN
    def capture(**params):
        estimator = original_class(**params)
        actual_fit = estimator.fit
        def fit(coordinates):
            captured["coordinates"] = coordinates.copy()
            return actual_fit(coordinates)
        estimator.fit = fit
        return estimator
    monkeypatch.setattr(h, "HDBSCAN", capture)
    result, _ = h.fit_configuration(points.sample(frac=1, random_state=3), 3, 3, fixed)
    np.testing.assert_allclose(captured["coordinates"], points[XY].to_numpy(), atol=1e-8)
    assert result[ID].tolist() == points[ID].tolist()
    assert captured["coordinates"][0, 0] == pytest.approx(500000, abs=1e-6)
    without_crs = points.copy(); without_crs.attrs.clear()
    with pytest.raises(ValueError, match="metre"):
        h.fit_configuration(without_crs, 3, 3, fixed)
    invalid = points.copy(); invalid.loc[0, XY[0]] = np.inf
    with pytest.raises(ValueError, match="finite"):
        h.fit_configuration(invalid, 3, 3, fixed)


def test_fit_preserves_original_fields_and_has_valid_probabilities(fixed):
    points = synthetic_points()
    original = points.copy(deep=True)
    result, params = h.fit_configuration(points, 3, 3, fixed)
    assert_frame_equal(points, original)
    for column in original:
        assert result[column].tolist() == original[column].tolist()
    assert result.membership_probability.between(0, 1).all()
    assert result.cluster_id.nunique() >= 2
    assert params["metric"] == "euclidean"


@pytest.mark.parametrize("label,probability", [(-1, .1), (-2, .1), (-3, 0), (0, np.nan), (0, 1.1), (0, -.1), (-4, 0)])
def test_invalid_membership_probabilities_and_labels(label, probability):
    with pytest.raises(ValueError):
        h.validate_membership([label], [probability])


def test_all_documented_negative_labels_are_nonmembership_and_remain_preserved():
    result = labelled([0, 0, 1, -1, -2, -3], [.2, 1, .8, 0, 0, np.nan])
    before = result.copy(deep=True)
    statistics = h.configuration_statistics(result)
    assert (statistics["n_observations"], statistics["n_clusters"], statistics["n_clustered"], statistics["n_noise_outliers"]) == (6, 2, 3, 3)
    assert statistics["n_noise_minus1"] == statistics["n_infinite_minus2"] == statistics["n_missing_minus3"] == 1
    assert statistics["prob_all_available"] == 5 and statistics["prob_all_unavailable"] == 1
    assert statistics["prob_clustered_median"] == .8
    assert statistics["largest_share_pct_of_input"] == pytest.approx(100 / 3)
    tables = h.diagnostics(result, ["A", "B"], ["2024-04-01"])
    for table in tables.values():
        assert table.n_observations.sum() == 6
    assert tables["daily"].set_index("membership").loc["noise", "n_observations"] == 3
    assert_frame_equal(result, before)


def test_zero_cluster_single_cluster_and_singleton_silhouette(fixed):
    fitted, _ = h.fit_configuration(synthetic_points(), 50, 3, fixed)
    zero = h.configuration_statistics(fitted)
    assert zero["n_clusters"] == 0 and zero["n_noise_outliers"] == 20
    assert zero["size_min"] is None and zero["silhouette_nonnoise"] is None
    assert h.summarize_clusters(fitted).empty
    one = h.configuration_statistics(labelled([0, 0, 0, -1], [1, .8, .6, 0]))
    assert one["n_clusters"] == 1 and one["silhouette_status"] == "unavailable_one_cluster"
    assert one["silhouette_nonnoise"] is None
    singleton = h.configuration_statistics(labelled([0, 1, 2], [1, 1, 1]))
    assert singleton["silhouette_status"] == "unavailable_one_point_per_cluster"


def test_cluster_summary_size_probability_and_cross_district_accounting():
    result = labelled([0, 0, 0, 1, 1, -1], [.2, .5, .8, 1, 1, 0])
    result.loc[1, DISTRICT] = "B"
    summary = h.summarize_clusters(result)
    assert summary.n_observations.tolist() == [3, 2]
    assert summary.n_districts.tolist() == [2, 1]
    assert summary.probability_median.tolist() == [.5, 1]
    assert summary.n_observations.sum() == 5


def test_comparison_aligns_identities_and_ignores_arbitrary_numeric_labels():
    reference = labelled([0, 0, 1, 1, -1, -1])
    alternative = reference.copy()
    alternative.cluster_id = [7, 7, 3, 3, -2, -3]
    result, transitions, edges = h.compare_assignments(reference, alternative.sample(frac=1, random_state=3))
    assert result["ari_all_common_noise_shared"] == result["ari_clustered_union"] == 1
    assert result["binary_membership_agreement"] == result["clustered_population_jaccard"] == 1
    assert result["coclustered_pair_jaccard"] == 1
    assert result["nonclustered_both"] == 2
    assert transitions[ID].tolist() == reference[ID].tolist()
    assert edges.shared_points.sum() == 4


def test_union_diagnostic_excludes_common_noise_and_denominators_are_explicit():
    reference = labelled([0, 0, 1, 1, -1, -1])
    alternative = labelled([9, 9, -1, 8, 8, -1])
    result, _, _ = h.compare_assignments(reference, alternative)
    assert (result["clustered_both"], result["reference_only_clustered"], result["alternative_only_clustered"], result["nonclustered_both"]) == (3, 1, 1, 1)
    assert result["clustered_union_n"] == 5
    assert result["clustered_population_jaccard"] == .6
    assert result["binary_membership_agreement"] == pytest.approx(4 / 6)
    # Adding common noise changes all-population ARI, not the clustered-union ARI.
    extra = synthetic_points(30).iloc[6:].copy(); extra["cluster_id"] = -1
    augmented, _, _ = h.compare_assignments(pd.concat([reference, extra]), pd.concat([alternative, extra]))
    assert augmented["ari_clustered_union"] == result["ari_clustered_union"]
    assert augmented["ari_all_common_noise_shared"] != result["ari_all_common_noise_shared"]


def test_population_mismatch_requires_explicit_common_scope():
    reference = labelled([0, 0, 1, 1, -1, -1])
    alternative = reference.iloc[:3].copy()
    with pytest.raises(ValueError, match="identical source"):
        h.compare_assignments(reference, alternative)
    result, _, _ = h.compare_assignments(reference, alternative, scope="common")
    assert result["n_common"] == 3 and result["reference_rows_not_common"] == 3
    assert result["alternative_rows_not_common"] == 0
    assert result["ari_all_common_noise_shared"] == 1
    duplicate = reference.copy(); duplicate.loc[1, ID] = duplicate.loc[0, ID]
    with pytest.raises(ValueError, match="unique"):
        h.compare_assignments(reference, duplicate)
    altered = reference.copy(); altered.loc[0, XY[0]] += 10
    with pytest.raises(ValueError, match="coordinates"):
        h.compare_assignments(reference, altered)


def test_fragmentation_and_merge_diagnostics_do_not_match_numeric_ids():
    fine = labelled([0, 0, 0, 1, 1, 1])
    coarse = labelled([7, 7, 7, 7, 7, 7])
    result, _, _ = h.compare_assignments(fine, coarse)
    assert result["alternative_clusters_merge_reference"] == 1
    assert result["reference_clusters_split_across_alternative"] == 0
    assert result["clustered_population_jaccard"] == 1
    assert result["coclustered_pair_jaccard"] == pytest.approx(6 / 15)
    reverse, _, _ = h.compare_assignments(coarse, fine)
    assert reverse["reference_clusters_split_across_alternative"] == 1


def test_all_noise_comparison_has_no_fabricated_union_metrics():
    result, _, edges = h.compare_assignments(labelled([-1, -2, -3]), labelled([-1, -1, -1]))
    assert result["clustered_union_n"] == 0
    assert result["ari_clustered_union"] is None
    assert result["clustered_population_jaccard"] is None
    assert result["coclustered_pair_jaccard"] is None
    assert result["binary_membership_agreement"] == 1
    assert edges.empty


def test_reused_subsets_and_seeded_subsampling_are_deterministic():
    points = synthetic_points()
    points.loc[:2, "agniscope_confidence"] = "low"
    points.loc[:3, TREE] = 30
    first, ledger = sensitivity_subsets(points)
    second, ledger2 = sensitivity_subsets(points.sample(frac=1, random_state=3))
    assert len(first["nominal_only"]) == 17 and len(first["tree_cover_only"]) == 16
    assert first["grid_day_sample"][ID].tolist() == second["grid_day_sample"][ID].tolist()
    assert_frame_equal(ledger, ledger2)
    sample = h.deterministic_subsample(points, .8, 1729)
    shuffled = h.deterministic_subsample(points.sample(frac=1, random_state=4), .8, 1729)
    assert_frame_equal(sample, shuffled)
    assert len(sample) == 16 and sample[ID].is_unique
    assert sample.attrs["analysis_crs"] == "EPSG:32644"
    assert sample[ID].tolist() != h.deterministic_subsample(points, .8, 1730)[ID].tolist()


def test_full_grid_accounting_repeatability_and_raw_preservation(fixed, tmp_path):
    points = synthetic_points()
    source = tmp_path / "synthetic.csv"
    points.to_csv(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result, params = h.run_grid(points, [3, 5, 10, 15], [3, 5, 10], fixed, "synthetic", ["A"], ["2024-04-01"])
    repeated, params2 = h.run_grid(points.sample(frac=1, random_state=5), [3, 5, 10, 15], [3, 5, 10], fixed, "synthetic", ["A"], ["2024-04-01"])
    assert len(result["parameters"]) == len(params) == 12
    assert len(result["assignments"]) == 20 * 12
    assert params == params2
    for key in result:
        assert_frame_equal(result[key], repeated[key])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
