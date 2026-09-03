"""Split-conformal calibration and interval summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    targets: tuple[str, ...]
    quantiles: np.ndarray
    calibration_size: int

    def __post_init__(self) -> None:
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie between zero and one")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("Conformal targets must be non-empty and unique")
        if self.calibration_size < 1:
            raise ValueError("calibration_size must be positive")
        if (
            quantiles.shape != (len(self.targets),)
            or not np.isfinite(quantiles).all()
            or (quantiles < 0).any()
        ):
            raise ValueError("One finite nonnegative conformal quantile is required per target")
        object.__setattr__(self, "quantiles", quantiles)

    def intervals(self, mean: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_array, sigma_array = _prediction_arrays(mean, sigma, len(self.targets))
        half_width = sigma_array * self.quantiles
        return mean_array - half_width, mean_array + half_width

    def coverage(self, truth: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        truth_array = np.asarray(truth, dtype=np.float64)
        lower, upper = self.intervals(mean, sigma)
        if truth_array.shape != lower.shape or not np.isfinite(truth_array).all():
            raise ValueError("Truth and predictions must be finite and have matching shapes")
        return ((truth_array >= lower) & (truth_array <= upper)).mean(axis=0)

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "targets": list(self.targets),
            "q_hat": self.quantiles.tolist(),
            "n_calibration": self.calibration_size,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "ConformalCalibration":
        return cls(
            float(values["alpha"]),
            tuple(str(value) for value in values["targets"]),
            np.asarray(values["q_hat"], dtype=np.float64),
            int(values.get("n_calibration", values.get("n_calib", 0))),
        )


def _prediction_arrays(mean: np.ndarray, sigma: np.ndarray, targets: int) -> tuple[np.ndarray, np.ndarray]:
    mean_array = np.asarray(mean, dtype=np.float64)
    sigma_array = np.asarray(sigma, dtype=np.float64)
    if mean_array.ndim != 2 or mean_array.shape != sigma_array.shape or mean_array.shape[1] != targets:
        raise ValueError("mean and sigma must be matching case-by-target arrays")
    if not np.isfinite(mean_array).all() or not np.isfinite(sigma_array).all() or (sigma_array < 0).any():
        raise ValueError("Predictions must be finite and sigma cannot be negative")
    return mean_array, sigma_array


def split_conformal_quantile(residual: np.ndarray, sigma: np.ndarray, alpha: float = 0.10) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    residual_array, sigma_array = np.broadcast_arrays(
        np.asarray(residual, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    )
    if (sigma_array < 0).any():
        raise ValueError("sigma cannot be negative")
    score = np.abs(residual_array) / np.maximum(sigma_array, 1e-9)
    score = score[np.isfinite(score)]
    if not len(score):
        raise ValueError("No finite calibration scores")
    level = min((1.0 - alpha) * (1.0 + 1.0 / len(score)), 1.0)
    return float(np.quantile(score, level, method="higher"))


def fit_split_conformal(
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    *,
    alpha: float = 0.10,
    targets: Sequence[str] | None = None,
) -> ConformalCalibration:
    truth_array = np.asarray(truth, dtype=np.float64)
    if truth_array.ndim != 2:
        raise ValueError("truth must be a case-by-target array")
    mean_array, sigma_array = _prediction_arrays(mean, sigma, truth_array.shape[1])
    if truth_array.shape != mean_array.shape or not np.isfinite(truth_array).all():
        raise ValueError("Truth and predictions must be finite and have matching shapes")
    names = tuple(targets or (f"target_{index}" for index in range(truth_array.shape[1])))
    if len(names) != truth_array.shape[1]:
        raise ValueError("Target names do not match the prediction columns")
    quantiles = np.array(
        [
            split_conformal_quantile(truth_array[:, index] - mean_array[:, index], sigma_array[:, index], alpha)
            for index in range(truth_array.shape[1])
        ],
        dtype=np.float64,
    )
    return ConformalCalibration(alpha, names, quantiles, len(truth_array))


def interval_score(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = 0.10,
) -> np.ndarray:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    truth_array, lower_array, upper_array = np.broadcast_arrays(
        np.asarray(truth, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    if not np.isfinite((truth_array, lower_array, upper_array)).all() or (lower_array > upper_array).any():
        raise ValueError("Intervals must be finite and ordered")
    score = upper_array - lower_array
    score += 2.0 / alpha * (lower_array - truth_array) * (truth_array < lower_array)
    score += 2.0 / alpha * (truth_array - upper_array) * (truth_array > upper_array)
    return score
