"""Statistical estimators used by the publication workflow."""

from __future__ import annotations

from math import erf, sqrt
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _pair(first: ArrayLike, second: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = _vector(first, "first")
    y = _vector(second, "second")
    if x.shape != y.shape or x.size < 2:
        raise ValueError("paired arrays must have the same length and at least two values")
    return x, y


def icc_1_1(first: ArrayLike, second: ArrayLike) -> float:
    x, y = _pair(first, second)
    case_means = (x + y) / 2.0
    grand_mean = np.concatenate((x, y)).mean()
    between = 2.0 * np.square(case_means - grand_mean).sum() / (x.size - 1)
    within = np.square(x - y).sum() / (2.0 * x.size)
    denominator = between + within
    return float((between - within) / denominator) if denominator > 0 else float("nan")


def single_tracing_error(first: ArrayLike, second: ArrayLike) -> float:
    x, y = _pair(first, second)
    return float(np.std(x - y, ddof=1) / sqrt(2.0))


def stratum_offset(
    first_strata: Sequence[str],
    second_strata: Sequence[str],
    differences: ArrayLike,
    lower_stratum: str,
) -> float:
    values = _vector(differences, "differences")
    first = np.asarray(first_strata, dtype=str)
    second = np.asarray(second_strata, dtype=str)
    if first.shape != values.shape or second.shape != values.shape:
        raise ValueError("strata and differences must have the same shape")
    forward = values[(first == lower_stratum) & (second != lower_stratum)]
    reverse = values[(first != lower_stratum) & (second == lower_stratum)]
    if not forward.size or not reverse.size:
        raise ValueError("both cross-stratum directions are required")
    return float((forward.mean() - reverse.mean()) / 2.0)


def reliability_ceiling(distances: ArrayLike, sigma: float) -> float:
    """Return a Gaussian same-radiograph relabeling approximation, not a strict upper bound."""
    distance = _vector(distances, "distances")
    if not distance.size or np.any(distance < 0) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("distances must be non-negative and sigma must be positive")
    scale = sigma * sqrt(2.0)
    cdf = np.fromiter((0.5 * (1.0 + erf(value / scale)) for value in distance), float)
    return float(cdf.mean())


def regression_metrics(reference: ArrayLike, prediction: ArrayLike) -> dict[str, float]:
    y, estimate = _pair(reference, prediction)
    residual = estimate - y
    total = np.square(y - y.mean()).sum()
    return {
        "mae": float(np.abs(residual).mean()),
        "r2": float(1.0 - np.square(residual).sum() / total) if total > 0 else float("nan"),
    }


def balanced_accuracy(
    reference: ArrayLike,
    prediction: ArrayLike,
    labels: Iterable[int] | None = None,
) -> float:
    y, estimate = _pair(reference, prediction)
    classes = np.asarray(tuple(labels) if labels is not None else np.unique(y))
    if not classes.size:
        raise ValueError("at least one class is required")
    recalls = []
    for label in classes:
        selected = y == label
        if not selected.any():
            raise ValueError(f"reference contains no cases for class {label}")
        recalls.append(np.mean(estimate[selected] == label))
    return float(np.mean(recalls))


def conformal_quantile(
    reference: ArrayLike,
    prediction: ArrayLike,
    sigma: ArrayLike,
    alpha: float,
) -> float:
    y, estimate = _pair(reference, prediction)
    uncertainty = _vector(sigma, "sigma")
    if uncertainty.shape != y.shape or not 0 < alpha < 1:
        raise ValueError("sigma must match the paired arrays and alpha must be in (0, 1)")
    scores = np.abs(estimate - y) / np.maximum(uncertainty, 1e-9)
    level = min((1.0 - alpha) * (1.0 + 1.0 / scores.size), 1.0)
    return float(np.quantile(scores, level, method="higher"))
