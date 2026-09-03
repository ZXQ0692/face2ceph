"""Regression, classification, conformal, referral, and subgroup evaluation."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Sequence

import numpy as np

from .calibration import ConformalCalibration
from .preprocessing import SAGITTAL_CLASSES, TARGETS, VERTICAL_CLASSES, age_stratum, normalize_sex
from .referral import selective_accuracy


def bootstrap_interval(
    statistic: Callable[..., float],
    *arrays: np.ndarray,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = [np.asarray(array) for array in arrays]
    if not values or not len(values[0]) or any(len(array) != len(values[0]) for array in values):
        raise ValueError("Bootstrap arrays must be non-empty and aligned")
    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.integers(0, len(values[0]), len(values[0]))
        estimates[index] = statistic(*(array[selected] for array in values))
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, (alpha / 2.0, 1.0 - alpha / 2.0)))


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_array, prediction_array = np.broadcast_arrays(
        np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    )
    if truth_array.ndim != 1 or not np.isfinite((truth_array, prediction_array)).all() or not len(truth_array):
        raise ValueError("Regression inputs must be finite one-dimensional arrays")
    error = prediction_array - truth_array
    total = float(np.sum((truth_array - truth_array.mean()) ** 2))
    residual = float(np.sum(error**2))
    correlation = (
        float(np.corrcoef(truth_array, prediction_array)[0, 1])
        if truth_array.std() > 1e-12 and prediction_array.std() > 1e-12
        else float("nan")
    )
    bias = float(error.mean())
    difference_sd = float(error.std(ddof=0))
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "r": correlation,
        "bias": bias,
        "loa_lower": bias - 1.96 * difference_sd,
        "loa_upper": bias + 1.96 * difference_sd,
    }


def regression_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    targets: Sequence[str] = TARGETS,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float | list[float]]]:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    names = tuple(targets)
    if (
        truth_array.ndim != 2
        or truth_array.shape != prediction_array.shape
        or truth_array.shape[1] != len(names)
        or not np.isfinite((truth_array, prediction_array)).all()
    ):
        raise ValueError("Regression matrices and target names must align")
    report: dict[str, dict[str, float | list[float]]] = {}
    for index, name in enumerate(names):
        target_truth, target_prediction = truth_array[:, index], prediction_array[:, index]
        metrics: dict[str, float | list[float]] = regression_metrics(target_truth, target_prediction)
        metrics["MAE_ci"] = list(
            bootstrap_interval(
                lambda a, b: float(np.abs(a - b).mean()),
                target_truth,
                target_prediction,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        )
        metrics["R2_ci"] = list(
            bootstrap_interval(
                lambda a, b: float(
                    1.0 - np.sum((a - b) ** 2) / max(float(np.sum((a - a.mean()) ** 2)), 1e-12)
                ),
                target_truth,
                target_prediction,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        )
        report[name] = metrics
    return report


def encode_labels(values: Sequence[object] | np.ndarray, classes: Sequence[str]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    if np.issubdtype(array.dtype, np.integer):
        encoded = array.astype(np.int64)
    else:
        index = {name: position for position, name in enumerate(classes)}
        try:
            encoded = np.array([index[str(value)] for value in array], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Unsupported class label: {error.args[0]}") from error
    if ((encoded < 0) | (encoded >= len(classes))).any():
        raise ValueError("Class indices are out of range")
    return encoded


def confusion_matrix(truth: np.ndarray, prediction: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return matrix


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or labels_array.shape != scores_array.shape or not np.isfinite(scores_array).all():
        raise ValueError("AUC inputs must be aligned and finite")
    positives, negatives = int(labels_array.sum()), int((~labels_array).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(len(scores_array), dtype=np.float64)
    start = 0
    while start < len(scores_array):
        stop = start + 1
        while stop < len(scores_array) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[labels_array].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def expected_calibration_error(
    truth: Sequence[object] | np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
    bins: int = 10,
    minimum_bin_size: int = 20,
) -> float:
    encoded = encode_labels(truth, classes)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(encoded), len(classes)) or not np.isfinite(probability).all():
        raise ValueError("Probabilities and labels must align")
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == encoded
    edges = np.linspace(1.0 / len(classes), 1.0, bins + 1)
    weighted_error = 0.0
    included = 0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)
        count = int(selected.sum())
        if count < minimum_bin_size:
            continue
        weighted_error += count * abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
        included += count
    return weighted_error / included if included else float("nan")


def classification_metrics(
    truth: Sequence[object] | np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
    bootstrap_resamples: int = 2000,
    seed: int = 0,
    ece_minimum_bin_size: int = 20,
) -> dict[str, object]:
    names = tuple(classes)
    encoded = encode_labels(truth, names)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(encoded), len(names)) or not np.isfinite(probability).all():
        raise ValueError("Probabilities and labels must align")
    if (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Rows must be probability distributions")
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(encoded, prediction, len(names))
    per_class: dict[str, dict[str, float | int | list[float]]] = {}
    recalls, f1_scores = [], []
    for class_index, name in enumerate(names):
        positive = encoded == class_index
        predicted_positive = prediction == class_index
        true_positive = int((positive & predicted_positive).sum())
        false_positive = int((~positive & predicted_positive).sum())
        false_negative = int((positive & ~predicted_positive).sum())
        true_negative = int((~positive & ~predicted_positive).sum())
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        negative_predictive_value = true_negative / max(true_negative + false_negative, 1)
        recalls.append(sensitivity)
        f1_scores.append(2.0 * sensitivity * precision / max(sensitivity + precision, 1e-12))
        sensitivity_ci = bootstrap_interval(
            lambda a, b: float((b[a == class_index] == class_index).mean()) if (a == class_index).any() else np.nan,
            encoded,
            prediction,
            resamples=bootstrap_resamples,
            seed=seed + 1,
        )
        per_class[name] = {
            "n": int(positive.sum()),
            "prevalence": float(positive.mean()),
            "sensitivity": sensitivity,
            "sensitivity_ci": list(sensitivity_ci),
            "specificity": specificity,
            "PPV": precision,
            "NPV": negative_predictive_value,
            "AUC": binary_auc(positive, probability[:, class_index]),
        }
    accuracy = float((encoded == prediction).mean())
    accuracy_ci = bootstrap_interval(
        lambda a, b: float((a == b).mean()),
        encoded,
        prediction,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    balanced_accuracy_ci = bootstrap_interval(
        lambda a, b: float(np.mean([(b[a == value] == value).mean() for value in np.unique(a)])),
        encoded,
        prediction,
        resamples=bootstrap_resamples,
        seed=seed + 2,
    )
    return {
        "n": len(encoded),
        "accuracy": accuracy,
        "accuracy_ci": list(accuracy_ci),
        "balanced_accuracy": float(np.mean(recalls)),
        "balanced_accuracy_ci": list(balanced_accuracy_ci),
        "macro_f1": float(np.mean(f1_scores)),
        "expected_calibration_error": expected_calibration_error(
            encoded,
            probability,
            classes=names,
            minimum_bin_size=ece_minimum_bin_size,
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def conformal_report(
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    calibration: ConformalCalibration,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float | list[float]]]:
    truth_array = np.asarray(truth, dtype=np.float64)
    lower, upper = calibration.intervals(mean, sigma)
    if truth_array.shape != lower.shape or not np.isfinite(truth_array).all():
        raise ValueError("Truth and conformal intervals must align")
    report: dict[str, dict[str, float | list[float]]] = {}
    for index, target in enumerate(calibration.targets):
        inside = (truth_array[:, index] >= lower[:, index]) & (truth_array[:, index] <= upper[:, index])
        report[target] = {
            "coverage": float(inside.mean()),
            "coverage_ci": list(
                bootstrap_interval(
                    lambda values: float(values.mean()),
                    inside,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            ),
            "mean_half_width": float(((upper[:, index] - lower[:, index]) * 0.5).mean()),
        }
    return report


def referral_report(
    truth: Sequence[object] | np.ndarray,
    probabilities: np.ndarray,
    referred: np.ndarray,
    *,
    classes: Sequence[str],
) -> dict[str, float | int]:
    encoded = encode_labels(truth, classes)
    prediction = np.asarray(probabilities, dtype=np.float64).argmax(axis=1)
    return selective_accuracy(prediction == encoded, referred)


def stratified_report(
    truth_measurements: np.ndarray | None,
    predicted_measurements: np.ndarray | None,
    sagittal_truth: Sequence[object] | np.ndarray,
    sagittal_probabilities: np.ndarray,
    vertical_truth: Sequence[object] | np.ndarray,
    vertical_probabilities: np.ndarray,
    sex: Sequence[object] | np.ndarray,
    age: Sequence[float] | np.ndarray,
    *,
    minimum_group_size: int = 30,
) -> dict[str, dict[str, dict[str, float | int]]]:
    sagittal = encode_labels(sagittal_truth, SAGITTAL_CLASSES)
    vertical = encode_labels(vertical_truth, VERTICAL_CLASSES)
    sagittal_prediction = np.asarray(sagittal_probabilities, dtype=np.float64).argmax(axis=1)
    vertical_prediction = np.asarray(vertical_probabilities, dtype=np.float64).argmax(axis=1)
    count = len(sagittal)
    if (
        len(vertical) != count
        or sagittal_prediction.shape != (count,)
        or vertical_prediction.shape != (count,)
        or minimum_group_size < 1
    ):
        raise ValueError("Stratified evaluation inputs must align")
    sex_values = np.asarray([normalize_sex(value) for value in sex], dtype=str)
    age_values = np.asarray(age, dtype=np.float64)
    if sex_values.shape != (count,) or age_values.shape != (count,) or not np.isfinite(age_values).all():
        raise ValueError("Sex and age must provide one valid value per case")
    truth_array = None if truth_measurements is None else np.asarray(truth_measurements, dtype=np.float64)
    prediction_array = (
        None if predicted_measurements is None else np.asarray(predicted_measurements, dtype=np.float64)
    )
    if (truth_array is None) != (prediction_array is None):
        raise ValueError("Regression truth and predictions must be provided together")
    if truth_array is not None and (
        truth_array.shape != prediction_array.shape
        or truth_array.ndim != 2
        or truth_array.shape[0] != count
        or truth_array.shape[1] < 1
        or not np.isfinite((truth_array, prediction_array)).all()
    ):
        raise ValueError("Regression matrices must align with the stratification metadata")

    def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
        classes = np.unique(truth)
        return float(np.mean([(prediction[truth == value] == value).mean() for value in classes]))

    reports: dict[str, dict[str, dict[str, float | int]]] = {}
    groups = {
        "sex": sex_values,
        "age_stratum": np.asarray([age_stratum(value) for value in age_values], dtype=str),
    }
    for group_name, values in groups.items():
        group_report: dict[str, dict[str, float | int]] = {}
        for value in sorted(set(values.tolist())):
            selected = values == value
            if int(selected.sum()) < minimum_group_size:
                continue
            metrics: dict[str, float | int] = {
                "n": int(selected.sum()),
                "acc_sag": float((sagittal_prediction[selected] == sagittal[selected]).mean()),
                "acc_vert": float((vertical_prediction[selected] == vertical[selected]).mean()),
                "bal_sag": balanced_accuracy(sagittal[selected], sagittal_prediction[selected]),
                "bal_vert": balanced_accuracy(vertical[selected], vertical_prediction[selected]),
            }
            if truth_array is not None:
                metrics["MAE_ANB"] = float(np.abs(prediction_array[selected, 0] - truth_array[selected, 0]).mean())
            group_report[value] = metrics
        reports[group_name] = group_report
    return reports


def publication_evaluation_result(
    report: dict[str, object],
    *,
    split: str,
    stratified: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, object]:
    result = deepcopy(report)
    result["split"] = split
    result["stratified"] = stratified
    regression = result.get("regression")
    if isinstance(regression, dict):
        for metrics in regression.values():
            if isinstance(metrics, dict):
                metrics["loa_lo"] = metrics.pop("loa_lower")
                metrics["loa_hi"] = metrics.pop("loa_upper")
    classification = result.get("classification")
    confusion: dict[str, object] = {}
    if isinstance(classification, dict):
        for axis, metrics in classification.items():
            if not isinstance(metrics, dict):
                continue
            confusion[str(axis)] = metrics.pop("confusion_matrix")
            metrics.pop("expected_calibration_error", None)
            per_class = metrics.get("per_class")
            if isinstance(per_class, dict):
                for class_metrics in per_class.values():
                    if isinstance(class_metrics, dict):
                        class_metrics["sens_ci"] = class_metrics.pop("sensitivity_ci")
    result["confusion"] = confusion
    conformal = result.pop("conformal", None)
    if isinstance(conformal, dict):
        for metrics in conformal.values():
            if isinstance(metrics, dict):
                metrics["mean_halfwidth"] = metrics.pop("mean_half_width")
        result["conformal_coverage"] = conformal
    referral = result.get("referral")
    if isinstance(referral, dict):
        converted: dict[str, object] = {}
        for axis, operating_points in referral.items():
            if not isinstance(operating_points, dict):
                continue
            rows = []
            baseline = None
            for rate, metrics in operating_points.items():
                if not isinstance(metrics, dict):
                    continue
                if baseline is None:
                    baseline = float(metrics["accuracy_all"])
                rows.append(
                    {
                        "target_rate": float(rate),
                        "actual_rate": float(metrics["actual_rate"]),
                        "n_kept": int(metrics["n_retained"]),
                        "accuracy_kept": float(metrics["accuracy_retained"]),
                        "accuracy_referred": float(metrics["accuracy_referred"]),
                    }
                )
            converted[str(axis)] = {"baseline_accuracy": baseline, "operating_points": rows}
        result["referral"] = converted
    return result


def evaluate_predictions(
    truth_measurements: np.ndarray,
    predicted_measurements: np.ndarray,
    predicted_sigma: np.ndarray,
    sagittal_truth: Sequence[object] | np.ndarray,
    sagittal_probabilities: np.ndarray,
    vertical_truth: Sequence[object] | np.ndarray,
    vertical_probabilities: np.ndarray,
    *,
    conformal: ConformalCalibration | None = None,
    targets: Sequence[str] = TARGETS,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "n": int(len(np.asarray(truth_measurements))),
        "regression": regression_report(
            truth_measurements,
            predicted_measurements,
            targets=targets,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "classification": {
            "sagittal": classification_metrics(
                sagittal_truth,
                sagittal_probabilities,
                classes=SAGITTAL_CLASSES,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            ),
            "vertical": classification_metrics(
                vertical_truth,
                vertical_probabilities,
                classes=VERTICAL_CLASSES,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            ),
        },
    }
    if conformal is not None:
        result["conformal"] = conformal_report(
            truth_measurements,
            predicted_measurements,
            predicted_sigma,
            conformal,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
    return result
