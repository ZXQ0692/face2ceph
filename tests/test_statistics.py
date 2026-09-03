from math import sqrt

import numpy as np
import pytest

from face2ceph.statistics import (
    balanced_accuracy,
    conformal_quantile,
    icc_1_1,
    regression_metrics,
    reliability_ceiling,
    single_tracing_error,
    stratum_offset,
)


def test_reliability_statistics_match_their_definitions() -> None:
    first = np.array([1.0, 2.0, 4.0, 8.0])
    second = np.array([1.2, 1.8, 4.4, 7.7])
    means = (first + second) / 2.0
    grand = np.concatenate((first, second)).mean()
    between = 2.0 * np.square(means - grand).sum() / 3.0
    within = np.square(first - second).sum() / 8.0

    assert icc_1_1(first, second) == pytest.approx((between - within) / (between + within))
    assert single_tracing_error(first, second) == pytest.approx(
        np.std(first - second, ddof=1) / sqrt(2.0)
    )


def test_stratum_offset_uses_both_pair_directions() -> None:
    first = ["lower", "upper", "lower", "upper"]
    second = ["upper", "lower", "upper", "lower"]
    differences = [2.0, -4.0, 6.0, -8.0]
    assert stratum_offset(first, second, differences, "lower") == pytest.approx(5.0)


def test_reliability_ceiling_is_the_mean_gaussian_class_probability() -> None:
    assert reliability_ceiling([0.0, 1.0], 1.0) == pytest.approx(0.6706723730342714)


def test_prediction_metrics_use_case_level_definitions() -> None:
    regression = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert regression == pytest.approx({"mae": 1.0 / 3.0, "r2": 0.5})

    reference = [0, 0, 1, 1, 2, 2]
    prediction = [0, 1, 1, 1, 0, 2]
    assert balanced_accuracy(reference, prediction, [0, 1, 2]) == pytest.approx(2.0 / 3.0)


def test_conformal_quantile_uses_the_finite_sample_correction() -> None:
    assert conformal_quantile([0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1], 0.25) == 4.0
