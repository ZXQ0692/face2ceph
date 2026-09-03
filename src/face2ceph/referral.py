"""Calibration-only selective-referral scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .preprocessing import TARGETS, TARGET_DIRECTIONS, VERTICAL_TARGETS, age_band, normalize_sex


DEFAULT_REFERRAL_RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


@dataclass(frozen=True)
class ZScoreReference:
    targets: tuple[str, ...]
    means: Mapping[tuple[str, str], np.ndarray]
    standard_deviations: Mapping[tuple[str, str], np.ndarray]

    def transform(self, predictions: np.ndarray, sex: Sequence[str], age: Sequence[float]) -> np.ndarray:
        values = np.asarray(predictions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.targets) or not np.isfinite(values).all():
            raise ValueError("Predictions must be a finite case-by-target array")
        if len(sex) != len(values) or len(age) != len(values):
            raise ValueError("Metadata and predictions must contain the same cases")
        directions = np.array([TARGET_DIRECTIONS[TARGETS.index(target)] for target in self.targets])
        standardized = np.empty_like(values)
        for index, (sex_value, age_value) in enumerate(zip(sex, age)):
            key = (normalize_sex(sex_value), age_band(float(age_value)))
            if key not in self.means or key not in self.standard_deviations:
                raise KeyError(f"No training reference for stratum {key}")
            mean = np.asarray(self.means[key], dtype=np.float64)
            standard_deviation = np.asarray(self.standard_deviations[key], dtype=np.float64)
            if mean.shape != (len(self.targets),) or standard_deviation.shape != mean.shape:
                raise ValueError(f"Invalid reference statistics for stratum {key}")
            if not np.isfinite((mean, standard_deviation)).all() or (standard_deviation <= 0).any():
                raise ValueError(f"Invalid reference statistics for stratum {key}")
            standardized[index] = directions * (values[index] - mean) / standard_deviation
        return standardized


@dataclass(frozen=True)
class ReferralAxisCalibration:
    calibration_confidence: np.ndarray
    calibration_discordance: np.ndarray
    operating_points: Mapping[float, float]

    def __post_init__(self) -> None:
        confidence = np.asarray(self.calibration_confidence, dtype=np.float64)
        discordance = np.asarray(self.calibration_discordance, dtype=np.float64)
        if confidence.ndim != 1 or confidence.shape != discordance.shape or not len(confidence):
            raise ValueError("Calibration signals must be non-empty one-dimensional arrays")
        if (
            not np.isfinite((confidence, discordance)).all()
            or (confidence < 0).any()
            or (confidence > 1).any()
            or (discordance < 0).any()
        ):
            raise ValueError("Calibration signals are invalid")
        points = {float(rate): float(value) for rate, value in self.operating_points.items()}
        if (
            not points
            or len(points) != len(self.operating_points)
            or any(not 0 < rate < 1 for rate in points)
            or any(not np.isfinite(value) or not 0 <= value <= 1 for value in points.values())
        ):
            raise ValueError("Referral operating points are invalid")
        object.__setattr__(self, "calibration_confidence", confidence)
        object.__setattr__(self, "calibration_discordance", discordance)
        object.__setattr__(self, "operating_points", points)

    def scores(self, probabilities: np.ndarray, discordance: np.ndarray) -> np.ndarray:
        return referral_scores(
            probabilities,
            discordance,
            self.calibration_confidence,
            self.calibration_discordance,
        )

    def refer(self, probabilities: np.ndarray, discordance: np.ndarray, rate: float = 0.20) -> np.ndarray:
        selected = min(self.operating_points, key=lambda value: abs(float(value) - rate))
        if abs(float(selected) - rate) > 1e-12:
            raise KeyError(f"No operating point for referral rate {rate}")
        return self.scores(probabilities, discordance) > float(self.operating_points[selected])

    def public_dict(self) -> dict[str, object]:
        return {"operating_points": {str(rate): float(value) for rate, value in self.operating_points.items()}}


def fit_zscore_reference(
    values: np.ndarray,
    sex: Sequence[str],
    age: Sequence[float],
    *,
    targets: Sequence[str] = TARGETS,
) -> ZScoreReference:
    measurements = np.asarray(values, dtype=np.float64)
    names = tuple(targets)
    if measurements.ndim != 2 or measurements.shape[1] != len(names) or not np.isfinite(measurements).all():
        raise ValueError("Training values must be a finite case-by-target array")
    if len(sex) != len(measurements) or len(age) != len(measurements):
        raise ValueError("Metadata and training values must contain the same cases")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (sex_value, age_value) in enumerate(zip(sex, age)):
        groups.setdefault((normalize_sex(sex_value), age_band(float(age_value))), []).append(index)
    means: dict[tuple[str, str], np.ndarray] = {}
    standard_deviations: dict[tuple[str, str], np.ndarray] = {}
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"Stratum {key} needs at least two training cases")
        group = measurements[indices]
        standard_deviation = group.std(axis=0, ddof=1)
        if (standard_deviation <= 0).any():
            raise ValueError(f"Stratum {key} contains a constant target")
        means[key] = group.mean(axis=0)
        standard_deviations[key] = standard_deviation
    return ZScoreReference(names, means, standard_deviations)


def measurement_discordance(
    aligned_z_scores: np.ndarray,
    *,
    targets: Sequence[str] = TARGETS,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(aligned_z_scores, dtype=np.float64)
    names = tuple(targets)
    if scores.ndim != 2 or scores.shape[1] != len(names) or not np.isfinite(scores).all():
        raise ValueError("Z scores must be a finite case-by-target array")
    index = {target: position for position, target in enumerate(names)}
    required = {"ANB", "Wits", *VERTICAL_TARGETS}
    missing = sorted(required - set(index))
    if missing:
        raise KeyError(f"Missing discordance targets: {missing}")
    sagittal = np.abs(scores[:, index["ANB"]] - scores[:, index["Wits"]])
    vertical = scores[:, [index[target] for target in VERTICAL_TARGETS]].std(axis=1, ddof=1)
    return sagittal, vertical


def percentile_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if not np.isfinite(values_array).all() or reference_array.ndim != 1 or not len(reference_array):
        raise ValueError("Values and the reference distribution must be finite")
    if not np.isfinite(reference_array).all():
        raise ValueError("Values and the reference distribution must be finite")
    return np.searchsorted(np.sort(reference_array), values_array, side="right") / len(reference_array)


def classification_confidence(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be a finite case-by-three array")
    if (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Rows must be probability distributions")
    return values.max(axis=1)


def referral_scores(
    probabilities: np.ndarray,
    discordance: np.ndarray,
    reference_confidence: np.ndarray,
    reference_discordance: np.ndarray,
) -> np.ndarray:
    confidence = classification_confidence(probabilities)
    discordance_array = np.asarray(discordance, dtype=np.float64)
    if discordance_array.shape != confidence.shape or not np.isfinite(discordance_array).all():
        raise ValueError("Discordance must provide one finite value per case")
    uncertainty_rank = percentile_rank(-confidence, -np.asarray(reference_confidence, dtype=np.float64))
    discordance_rank = percentile_rank(discordance_array, np.asarray(reference_discordance, dtype=np.float64))
    return np.maximum(uncertainty_rank, discordance_rank)


def fit_referral_axis(
    probabilities: np.ndarray,
    discordance: np.ndarray,
    *,
    rates: Sequence[float] = DEFAULT_REFERRAL_RATES,
) -> ReferralAxisCalibration:
    confidence = classification_confidence(probabilities)
    discordance_array = np.asarray(discordance, dtype=np.float64)
    if discordance_array.shape != confidence.shape or not np.isfinite(discordance_array).all():
        raise ValueError("Discordance must provide one finite value per calibration case")
    score = referral_scores(probabilities, discordance_array, confidence, discordance_array)
    selected_rates = tuple(float(rate) for rate in rates)
    if not selected_rates or len(set(selected_rates)) != len(selected_rates):
        raise ValueError("Referral rates must be non-empty and unique")
    points: dict[float, float] = {}
    for rate in selected_rates:
        if not 0 < rate < 1:
            raise ValueError("Referral rates must lie between zero and one")
        points[float(rate)] = float(np.quantile(score, 1.0 - rate))
    return ReferralAxisCalibration(confidence, discordance_array, points)


def selective_accuracy(correct: np.ndarray, referred: np.ndarray) -> dict[str, float | int]:
    correct_array = np.asarray(correct, dtype=bool)
    referred_array = np.asarray(referred, dtype=bool)
    if correct_array.ndim != 1 or correct_array.shape != referred_array.shape or not len(correct_array):
        raise ValueError("Correctness and referral flags must be matching one-dimensional arrays")
    retained = ~referred_array
    return {
        "actual_rate": float(referred_array.mean()),
        "n_referred": int(referred_array.sum()),
        "n_retained": int(retained.sum()),
        "accuracy_all": float(correct_array.mean()),
        "accuracy_retained": float(correct_array[retained].mean()) if retained.any() else float("nan"),
        "accuracy_referred": float(correct_array[referred_array].mean()) if referred_array.any() else float("nan"),
    }
