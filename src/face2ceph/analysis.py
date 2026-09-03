"""Aggregate analyses computed from frozen predictions and authorized measurements."""

from __future__ import annotations

import json
from math import ceil, erf, exp, lgamma, log, log1p, sqrt
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import fit_split_conformal
from .preprocessing import (
    SAGITTAL_CLASSES,
    THRESHOLD_SCHEMES,
    VERTICAL_CLASSES,
    age_stratum,
    apply_thresholds,
)
from .targets import TARGETS
from .statistics import icc_1_1, single_tracing_error
from .workspace import create_directory, write_json


ADDITIONAL_INPUT_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "compare_arms_summary.json": (
        "per-fold validation histories including every epoch for every model arm",
        "arm descriptions and architecture flags from the public registry",
    ),
    "learning_curve.json": (
        "per-fold validation histories for every declared training fraction",
        "measured fold-level training counts",
    ),
    "learning_curve_fit.json": (
        "per-fold validation histories for every declared training fraction",
        "measured fold-level training counts",
    ),
    "perturbation_summary.json": (
        "the declared perturbation specification",
        "per-case prediction archives for every perturbation condition",
    ),
    "perturbation_condition_metrics.csv": (
        "the declared perturbation specification",
        "per-case prediction archives for every perturbation condition",
    ),
    "confound_probe.json": (
        "held-out pooled feature vectors",
        "authorized acquisition-batch labels",
    ),
    "boundary_full_source_sd.json": (
        "pre-eligibility measurements for rows excluded from the controlled release bundle",
    ),
    "historical_bootstrap_order.json": (
        "the non-public pre-pseudonymization row order used by historical Monte Carlo resampling",
    ),
}

GENERATOR_NOT_IN_RELEASE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "c0a_geometry_summary.json": (
        "authorized normalized frontal landmarks and profile contours",
        "the matching targets and frozen outer-fold assignments",
    ),
    "reliability_summary.json": (
        "authorized first, second, and third tracing measurements",
        "authorized tracer assignments and operator-experience mapping",
    ),
    "learning_curve_test.json": (
        "aligned frozen internal-test predictions for every learning-curve arm",
        "the full-training main-arm internal-test predictions",
    ),
    "boundary_learning_curve_far.json": (
        "aligned per-case internal-test predictions for every learning-curve arm",
        "the declared training fractions and public arm registry",
    ),
}


def _regression_arrays(
    truth: np.ndarray,
    prediction: np.ndarray,
    targets: Sequence[str],
    *,
    minimum_cases: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    y = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    names = tuple(str(value) for value in targets)
    if (
        y.ndim != 2
        or y.shape != estimate.shape
        or y.shape[1] != len(names)
        or y.shape[0] < minimum_cases
        or not np.isfinite((y, estimate)).all()
    ):
        raise ValueError("truth, predictions, and target names must be finite and aligned")
    if len(set(names)) != len(names):
        raise ValueError("target names must be unique")
    return y, estimate, names


def _sigma_array(sigma: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(sigma, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError("sigma must be positive, finite, and aligned with predictions")
    return value


def _probabilities(probabilities: np.ndarray, cases: int, name: str) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if (
        value.shape != (cases, 3)
        or not np.isfinite(value).all()
        or (value < 0).any()
        or not np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"{name} must be an aligned N x 3 probability matrix")
    return value


def _regularized_beta(value: float, first: float, second: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0

    def fraction(a: float, b: float, x: float) -> float:
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        d = 1e-300 if abs(d) < 1e-300 else d
        d = 1.0 / d
        result = d
        for iteration in range(1, 201):
            twice = 2 * iteration
            coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
            d = 1.0 + coefficient * d
            d = 1e-300 if abs(d) < 1e-300 else d
            c = 1.0 + coefficient / c
            c = 1e-300 if abs(c) < 1e-300 else c
            d = 1.0 / d
            result *= d * c
            coefficient = -(a + iteration) * (qab + iteration) * x / (
                (a + twice) * (qap + twice)
            )
            d = 1.0 + coefficient * d
            d = 1e-300 if abs(d) < 1e-300 else d
            c = 1.0 + coefficient / c
            c = 1e-300 if abs(c) < 1e-300 else c
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= 3e-14:
                return result
        raise ArithmeticError("incomplete beta evaluation did not converge")

    factor = exp(
        lgamma(first + second)
        - lgamma(first)
        - lgamma(second)
        + first * log(value)
        + second * log1p(-value)
    )
    if value < (first + 1.0) / (first + second + 2.0):
        return factor * fraction(first, second, value) / first
    return 1.0 - factor * fraction(second, first, 1.0 - value) / second


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or len(first) < 3:
        raise ValueError("linear regression requires at least three aligned values")
    centered_x = first - first.mean()
    centered_y = second - second.mean()
    sum_xx = float(centered_x @ centered_x)
    sum_yy = float(centered_y @ centered_y)
    if sum_xx <= 0:
        raise ValueError("linear regression requires variation in its predictor")
    sum_xy = float(centered_x @ centered_y)
    slope = sum_xy / sum_xx
    if sum_yy <= 0 or sum_xy == 0:
        return float(slope), 1.0
    correlation = max(-1.0, min(1.0, sum_xy / sqrt(sum_xx * sum_yy)))
    if abs(correlation) >= 1.0:
        return float(slope), 0.0
    degrees = len(first) - 2
    statistic_squared = correlation * correlation * degrees / (1.0 - correlation * correlation)
    p_value = _regularized_beta(degrees / (degrees + statistic_squared), degrees / 2.0, 0.5)
    return float(slope), float(max(0.0, min(1.0, p_value)))


def bland_altman_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    targets: Sequence[str] = TARGETS,
) -> list[dict[str, object]]:
    """Return agreement limits, proportional-bias tests, and shrinkage slopes."""
    y, estimate, names = _regression_arrays(truth, prediction, targets, minimum_cases=3)
    rows: list[dict[str, object]] = []
    for index, target in enumerate(names):
        reference = y[:, index]
        predicted = estimate[:, index]
        difference = predicted - reference
        pair_mean = (reference + predicted) / 2.0
        bias = float(difference.mean())
        sd = float(difference.std(ddof=1))
        slope_mean, p_mean = _linear_regression(pair_mean, difference)
        slope_reference, p_reference = _linear_regression(reference, difference)
        shrinkage, _ = _linear_regression(reference, predicted)
        rows.append(
            {
                "target": target,
                "n": len(reference),
                "bias": bias,
                "bias_ci": [bias - 1.96 * sd / sqrt(len(reference)), bias + 1.96 * sd / sqrt(len(reference))],
                "loa_lower": bias - 1.96 * sd,
                "loa_upper": bias + 1.96 * sd,
                "loa_se": sd * sqrt(3.0 / len(reference)),
                "sd_diff": sd,
                "slope_vs_mean": slope_mean,
                "p_vs_mean": p_mean,
                "slope_vs_reference": slope_reference,
                "p_vs_reference": p_reference,
                "shrinkage_coefficient": shrinkage,
            }
        )
    return rows


def _encoded_labels(values: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    names = tuple(classes)
    index = {name: position for position, name in enumerate(names)}
    try:
        return np.asarray([index[str(value)] for value in values], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unsupported class label: {exc.args[0]}") from exc


def _classification_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    classes: Sequence[str],
) -> dict[str, object]:
    y = np.asarray(truth, dtype=str)
    estimate = np.asarray(prediction, dtype=str)
    if y.ndim != 1 or y.shape != estimate.shape or not len(y):
        raise ValueError("classification labels must be non-empty and aligned")
    correct = y == estimate
    present = [name for name in classes if np.any(y == name)]
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean([correct[y == name].mean() for name in present])),
        "majority_baseline": float(max(np.mean(y == name) for name in classes)),
        "class_prevalence": {name: float(np.mean(y == name)) for name in classes},
        "class_recall": {
            name: float(correct[y == name].mean()) if np.any(y == name) else None for name in classes
        },
    }


def age_strata_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    sagittal_probabilities: np.ndarray,
    vertical_probabilities: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    *,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
    minimum_cases: int = 30,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Report regression and classification performance within norm-age strata."""
    y, estimate, names = _regression_arrays(truth, prediction, targets)
    cases = len(y)
    sagittal_probability = _probabilities(sagittal_probabilities, cases, "sagittal probabilities")
    vertical_probability = _probabilities(vertical_probabilities, cases, "vertical probabilities")
    ages = np.asarray(age, dtype=np.float64)
    sexes = np.asarray(sex, dtype=object)
    if (
        ages.shape != (cases,)
        or sexes.shape != (cases,)
        or not np.isfinite(ages).all()
        or (ages < 7).any()
        or minimum_cases < 2
        or bootstrap_resamples < 1
    ):
        raise ValueError("age, sex, and bootstrap settings are invalid")
    target_index = {name: index for index, name in enumerate(names)}
    if "ANB" not in target_index or "SN_MP" not in target_index:
        raise ValueError("ANB and SN_MP are required for classification")
    sagittal_truth, vertical_truth = apply_thresholds(
        y[:, target_index["ANB"]], y[:, target_index["SN_MP"]], ages, sexes
    )
    sagittal_code = _encoded_labels(sagittal_truth, SAGITTAL_CLASSES)
    vertical_code = _encoded_labels(vertical_truth, VERTICAL_CLASSES)
    sagittal_prediction = sagittal_probability.argmax(axis=1)
    vertical_prediction = vertical_probability.argmax(axis=1)
    strata = np.asarray([age_stratum(value) for value in ages], dtype=str)
    rng = np.random.default_rng(seed)
    rows: dict[str, object] = {}
    for stratum in ("7-10", "11-30", ">30"):
        selected = strata == stratum
        count = int(selected.sum())
        extrapolated = bool(count and not np.all((ages[selected] >= 11) & (ages[selected] <= 30)))
        if count < minimum_cases:
            rows[stratum] = {
                "n": count,
                "insufficient": True,
                "norm_extrapolated": extrapolated,
            }
            continue
        indices = np.flatnonzero(selected)
        bootstrap = [rng.choice(indices, len(indices), replace=True) for _ in range(bootstrap_resamples)]
        regression: dict[str, object] = {}
        for index, target in enumerate(names):
            target_truth, target_prediction = y[:, index], estimate[:, index]
            total = float(np.square(target_truth[selected] - target_truth[selected].mean()).sum())
            residual = float(np.square(target_truth[selected] - target_prediction[selected]).sum())
            mae_samples = np.asarray(
                [np.abs(target_truth[sample] - target_prediction[sample]).mean() for sample in bootstrap]
            )
            regression[target] = {
                "MAE": float(np.abs(target_truth[selected] - target_prediction[selected]).mean()),
                "MAE_ci": [float(value) for value in np.percentile(mae_samples, (2.5, 97.5))],
                "R2": float(1.0 - residual / total) if total > 0 else None,
            }
        classification: dict[str, object] = {}
        for axis, true_code, predicted_code, classes in (
            ("sagittal", sagittal_code, sagittal_prediction, SAGITTAL_CLASSES),
            ("vertical", vertical_code, vertical_prediction, VERTICAL_CLASSES),
        ):
            correct = true_code == predicted_code
            present = [value for value in range(3) if np.any(true_code[selected] == value)]
            balanced = float(np.mean([correct[selected][true_code[selected] == value].mean() for value in present]))
            bootstrap_balanced = []
            for sample in bootstrap:
                sampled_classes = [value for value in range(3) if np.any(true_code[sample] == value)]
                bootstrap_balanced.append(
                    np.mean([correct[sample][true_code[sample] == value].mean() for value in sampled_classes])
                )
            classification[axis] = {
                "balanced_accuracy": balanced,
                "balanced_accuracy_ci": [
                    float(value) for value in np.percentile(bootstrap_balanced, (2.5, 97.5))
                ],
                "accuracy": float(correct[selected].mean()),
                "recall": {
                    classes[value]: (
                        float(correct[selected][true_code[selected] == value].mean())
                        if np.any(true_code[selected] == value)
                        else None
                    )
                    for value in range(3)
                },
                "prevalence": {
                    classes[value]: float(np.mean(true_code[selected] == value)) for value in range(3)
                },
            }
        rows[stratum] = {
            "n": count,
            "insufficient": False,
            "norm_extrapolated": extrapolated,
            "regression": regression,
            "classification": classification,
        }

    available = {name: value for name, value in rows.items() if not value["insufficient"]}
    differences: list[dict[str, str]] = []
    available_names = list(available)
    for target in names:
        for left in range(len(available_names)):
            for right in range(left + 1, len(available_names)):
                first, second = available_names[left], available_names[right]
                first_ci = available[first]["regression"][target]["MAE_ci"]
                second_ci = available[second]["regression"][target]["MAE_ci"]
                if first_ci[1] < second_ci[0] or second_ci[1] < first_ci[0]:
                    better, worse = (first, second) if first_ci[1] < second_ci[0] else (second, first)
                    differences.append({"metric": f"regression, {target}", "better": better, "worse": worse})
    for axis in ("sagittal", "vertical"):
        for left in range(len(available_names)):
            for right in range(left + 1, len(available_names)):
                first, second = available_names[left], available_names[right]
                first_ci = available[first]["classification"][axis]["balanced_accuracy_ci"]
                second_ci = available[second]["classification"][axis]["balanced_accuracy_ci"]
                if first_ci[1] < second_ci[0] or second_ci[1] < first_ci[0]:
                    better, worse = (second, first) if first_ci[1] < second_ci[0] else (first, second)
                    differences.append({"metric": f"classification, {axis}", "better": better, "worse": worse})
    return {
        "config": config,
        "strata": ["7-10", "11-30", ">30"],
        "norm_range": [11, 30],
        "splits": {"internal_test": rows},
        "stratum_differences": differences,
    }


def shrinkage_report(
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    *,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
) -> dict[str, object]:
    """Fit variance-restoration coefficients on calibration data and evaluate once on test data."""
    calibration_y, calibration_estimate, names = _regression_arrays(
        calibration_truth, calibration_prediction, targets, minimum_cases=3
    )
    test_y, test_estimate, _ = _regression_arrays(test_truth, test_prediction, names, minimum_cases=3)
    coefficients: dict[str, object] = {}
    rows: dict[str, object] = {}
    for index, target in enumerate(names):
        slope, _ = _linear_regression(calibration_y[:, index], calibration_estimate[:, index])
        if abs(slope) <= 1e-12:
            raise ValueError(f"calibration shrinkage slope is zero for {target}")
        mean_prediction = float(calibration_estimate[:, index].mean())
        coefficients[target] = {
            "b": slope,
            "mean_pred": mean_prediction,
            "sd_ref": float(calibration_y[:, index].std()),
            "sd_pred": float(calibration_estimate[:, index].std()),
        }
        corrected = mean_prediction + (test_estimate[:, index] - mean_prediction) / slope
        slope_raw, _ = _linear_regression(test_y[:, index], test_estimate[:, index])
        slope_corrected, _ = _linear_regression(test_y[:, index], corrected)
        raw_error = test_estimate[:, index] - test_y[:, index]
        corrected_error = corrected - test_y[:, index]
        rows[target] = {
            "mae_raw": float(np.abs(raw_error).mean()),
            "mae_corrected": float(np.abs(corrected_error).mean()),
            "slope_raw": slope_raw,
            "slope_corrected": slope_corrected,
            "loa_width_raw": float(2.0 * 1.96 * raw_error.std(ddof=1)),
            "loa_width_corrected": float(2.0 * 1.96 * corrected_error.std(ddof=1)),
        }
    mae_raw = float(np.mean([value["mae_raw"] for value in rows.values()]))
    mae_corrected = float(np.mean([value["mae_corrected"] for value in rows.values()]))
    slope_raw = float(np.mean([abs(1.0 - value["slope_raw"]) for value in rows.values()]))
    slope_corrected = float(np.mean([abs(1.0 - value["slope_corrected"]) for value in rows.values()]))
    return {
        "config": config,
        "analysis_status": "post_hoc sensitivity analysis",
        "coefficients": coefficients,
        "splits": {
            "internal_test": {
                "targets": rows,
                "mae_raw": mae_raw,
                "mae_corrected": mae_corrected,
                "slope_dev_raw": slope_raw,
                "slope_dev_corrected": slope_corrected,
            }
        },
    }


def _quintile_codes(values: np.ndarray) -> np.ndarray:
    codes = np.asarray(pd.qcut(values, 5, labels=False, duplicates="drop"), dtype=np.int64)
    if set(codes.tolist()) != set(range(5)):
        raise ValueError("five non-empty uncertainty quantiles are required")
    return codes


def conformal_adaptivity_report(
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_sigma: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    test_sigma: np.ndarray,
    *,
    alpha: float = 0.10,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
) -> dict[str, object]:
    """Compare adaptive split-conformal intervals with a fixed-width baseline."""
    calibration_y, calibration_estimate, names = _regression_arrays(
        calibration_truth, calibration_prediction, targets
    )
    test_y, test_estimate, _ = _regression_arrays(test_truth, test_prediction, names)
    calibration_uncertainty = _sigma_array(calibration_sigma, calibration_y.shape)
    test_uncertainty = _sigma_array(test_sigma, test_y.shape)
    calibration = fit_split_conformal(
        calibration_y,
        calibration_estimate,
        calibration_uncertainty,
        alpha=alpha,
        targets=names,
    )
    nominal = 1.0 - alpha
    calibration_error = np.abs(calibration_estimate - calibration_y)
    test_error = np.abs(test_estimate - test_y)
    fixed_index = min(int(ceil((len(calibration_y) + 1) * nominal)) - 1, len(calibration_y) - 1)
    rows: dict[str, object] = {}
    for index, target in enumerate(names):
        adaptive_width = calibration.quantiles[index] * test_uncertainty[:, index]
        fixed_width = float(np.sort(calibration_error[:, index])[fixed_index])
        adaptive_coverage = float(np.mean(test_error[:, index] <= adaptive_width))
        fixed_coverage = float(np.mean(test_error[:, index] <= fixed_width))

        def interval_score(width: np.ndarray | float) -> float:
            lower = test_estimate[:, index] - width
            upper = test_estimate[:, index] + width
            penalty = (2.0 / alpha) * (
                np.maximum(lower - test_y[:, index], 0) + np.maximum(test_y[:, index] - upper, 0)
            )
            return float(np.mean(2.0 * width + penalty))

        adaptive_score = interval_score(adaptive_width)
        fixed_score = interval_score(fixed_width)
        quantiles = _quintile_codes(test_uncertainty[:, index])
        conditional = {
            f"Q{group + 1}": float(np.mean(test_error[quantiles == group, index] <= adaptive_width[quantiles == group]))
            for group in range(5)
        }
        rows[target] = {
            "sigma_cv": float(test_uncertainty[:, index].std(ddof=1) / test_uncertainty[:, index].mean()),
            "coverage_adaptive": adaptive_coverage,
            "coverage_fixed": fixed_coverage,
            "halfwidth_adaptive_mean": float(adaptive_width.mean()),
            "halfwidth_fixed": fixed_width,
            "interval_score_adaptive": adaptive_score,
            "interval_score_fixed": fixed_score,
            "interval_score_ratio": adaptive_score / fixed_score,
            "conditional_coverage_by_sigma_quintile": conditional,
        }
    adaptive = [value["coverage_adaptive"] for value in rows.values()]
    fixed = [value["coverage_fixed"] for value in rows.values()]
    ratios = [value["interval_score_ratio"] for value in rows.values()]
    first = [value["conditional_coverage_by_sigma_quintile"]["Q1"] for value in rows.values()]
    fifth = [value["conditional_coverage_by_sigma_quintile"]["Q5"] for value in rows.values()]
    return {
        "config": config,
        "alpha": alpha,
        "nominal": nominal,
        "n_calib": len(calibration_y),
        "n_test": len(test_y),
        "_status": "descriptive post-hoc analysis computed from frozen predictions; it runs no new inference",
        "per_target": rows,
        "summary": {
            "coverage_adaptive_range": [min(adaptive), max(adaptive)],
            "coverage_fixed_range": [min(fixed), max(fixed)],
            "interval_score_ratio_range": [min(ratios), max(ratios)],
            "n_targets_adaptive_better": int(sum(value < 1.0 for value in ratios)),
            "conditional_coverage_Q1_range": [min(first), max(first)],
            "conditional_coverage_Q5_range": [min(fifth), max(fifth)],
        },
    }


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = sqrt(float(centered_x @ centered_x) * float(centered_y @ centered_y))
    if denominator <= 0:
        raise ValueError("correlation requires variation in both inputs")
    return float((centered_x @ centered_y) / denominator)


def patient_sigma_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
    *,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
) -> dict[str, object]:
    """Describe per-case uncertainty ranking without making inferential claims."""
    y, estimate, names = _regression_arrays(truth, prediction, targets, minimum_cases=5)
    uncertainty = _sigma_array(sigma, y.shape)
    error = np.abs(estimate - y)
    keep = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)
    rows: dict[str, object] = {}
    spearman: list[float] = []
    for index, target in enumerate(names):
        target_error = error[:, index]
        target_sigma = uncertainty[:, index]
        rho = _pearson(
            pd.Series(target_sigma).rank(method="average").to_numpy(dtype=np.float64),
            pd.Series(target_error).rank(method="average").to_numpy(dtype=np.float64),
        )
        pearson = _pearson(target_sigma, target_error)
        order = np.argsort(target_sigma, kind="mergesort")
        risk = {
            f"{100 * fraction:.0f}%": float(target_error[order[: int(round(fraction * len(y)))]].mean())
            for fraction in keep
        }
        spearman.append(rho)
        rows[target] = {
            "spearman_rho": rho,
            "pearson_r": pearson,
            "risk_coverage_mae": risk,
            "mae_drop_pct_at_80pct_kept": 100.0 * (risk["100%"] - risk["80%"]) / risk["100%"],
        }
    anb_index = names.index("ANB") if "ANB" in names else 0
    codes = _quintile_codes(uncertainty[:, anb_index])
    labels = ("Q1 most confident", "Q2", "Q3", "Q4", "Q5 least confident")
    quintiles = {
        label: {
            "n": int(np.sum(codes == group)),
            "MAE": float(error[codes == group, anb_index].mean()),
            "sigma_mean": float(uncertainty[codes == group, anb_index].mean()),
        }
        for group, label in enumerate(labels)
    }
    ratio = quintiles[labels[-1]]["MAE"] / quintiles[labels[0]]["MAE"]
    return {
        "config": config,
        "split": "internal_test",
        "n": len(y),
        "_status": "descriptive post-hoc analysis; no significance claim is made",
        "per_target": rows,
        "spearman_median": float(np.median(spearman)),
        "spearman_range": [min(spearman), max(spearman)],
        "anb_sigma_quintiles": quintiles,
        "anb_q5_over_q1_mae_ratio": ratio,
    }


def threshold_sensitivity_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    sagittal_probabilities: np.ndarray,
    vertical_probabilities: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    *,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Apply every declared threshold scheme to the same continuous predictions and references."""
    y, estimate, names = _regression_arrays(truth, prediction, targets)
    cases = len(y)
    sagittal_probability = _probabilities(sagittal_probabilities, cases, "sagittal probabilities")
    vertical_probability = _probabilities(vertical_probabilities, cases, "vertical probabilities")
    ages = np.asarray(age, dtype=np.float64)
    sexes = np.asarray(sex, dtype=object)
    if ages.shape != (cases,) or sexes.shape != (cases,) or not np.isfinite(ages).all():
        raise ValueError("age and sex must align with predictions")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    indices = {name: index for index, name in enumerate(names)}
    if "ANB" not in indices or "SN_MP" not in indices:
        raise ValueError("ANB and SN_MP are required for threshold sensitivity")
    rng = np.random.default_rng(seed)
    schemes: dict[str, object] = {}
    all_robust: list[bool] = []
    primary_truth: tuple[np.ndarray, np.ndarray] | None = None
    for scheme_name in THRESHOLD_SCHEMES:
        true_labels = apply_thresholds(
            y[:, indices["ANB"]], y[:, indices["SN_MP"]], ages, sexes, scheme_name
        )
        predicted_labels = apply_thresholds(
            estimate[:, indices["ANB"]], estimate[:, indices["SN_MP"]], ages, sexes, scheme_name
        )
        if scheme_name == "wu2021_1.5sd":
            primary_truth = true_labels
        entry: dict[str, object] = {}
        for axis, true_label, predicted_label, classes in (
            ("sagittal", true_labels[0], predicted_labels[0], SAGITTAL_CLASSES),
            ("vertical", true_labels[1], predicted_labels[1], VERTICAL_CLASSES),
        ):
            metrics = _classification_summary(true_label, predicted_label, classes)
            encoded = _encoded_labels(true_label, classes)
            correct = (np.asarray(true_label) == np.asarray(predicted_label)).astype(np.float64)
            gains = np.empty(bootstrap_resamples, dtype=np.float64)
            for sample_index in range(bootstrap_resamples):
                sample = rng.integers(0, cases, cases)
                counts = np.bincount(encoded[sample], minlength=len(classes)).astype(np.float64)
                hits = np.bincount(encoded[sample], weights=correct[sample], minlength=len(classes))
                present = counts > 0
                gains[sample_index] = float(np.mean(hits[present] / counts[present]) - 1.0 / len(classes))
            lower, upper = (float(value) for value in np.percentile(gains, (2.5, 97.5)))
            metrics["gain_over_chance"] = float(metrics["balanced_accuracy"] - 1.0 / len(classes))
            metrics["gain_ci"] = [lower, upper]
            metrics["beats_chance"] = lower > 0
            entry[axis] = metrics
            all_robust.append(bool(lower > 0 and metrics["balanced_accuracy"] > 1.0 / 3.0 + 0.05))
        schemes[scheme_name] = entry
    assert primary_truth is not None
    schemes["_classification_head_reference"] = {
        "sagittal": _classification_summary(
            primary_truth[0], np.asarray(SAGITTAL_CLASSES)[sagittal_probability.argmax(axis=1)], SAGITTAL_CLASSES
        ),
        "vertical": _classification_summary(
            primary_truth[1], np.asarray(VERTICAL_CLASSES)[vertical_probability.argmax(axis=1)], VERTICAL_CLASSES
        ),
    }
    return {
        "config": config,
        "primary": "wu2021_1.5sd",
        "splits": {"internal_test": schemes},
        "robust_across_schemes": bool(all(all_robust)),
    }


def _primary_labels(
    truth: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(targets)
    indices = {name: index for index, name in enumerate(names)}
    if "ANB" not in indices or "SN_MP" not in indices:
        raise ValueError("ANB and SN_MP are required for classification")
    return apply_thresholds(truth[:, indices["ANB"]], truth[:, indices["SN_MP"]], age, sex)


def _balanced_accuracy_codes(truth: np.ndarray, prediction: np.ndarray) -> float:
    classes = np.unique(truth)
    return float(np.mean([(prediction[truth == value] == value).mean() for value in classes]))


def fit_class_bias(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Fit multiplicative class weights by balanced accuracy on calibration data only."""
    probability = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    if probability.ndim != 2 or labels.shape != (len(probability),):
        raise ValueError("calibration probabilities and labels must align")
    values = np.geomspace(0.25, 6.0, 41) if grid is None else np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("class-bias grid values must be positive and finite")
    weights = np.ones(probability.shape[1], dtype=np.float64)
    best = _balanced_accuracy_codes(labels, (probability * weights).argmax(axis=1))
    for _ in range(4):
        for class_index in range(probability.shape[1]):
            selected = weights[class_index]
            for candidate in values:
                weights[class_index] = candidate
                score = _balanced_accuracy_codes(labels, (probability * weights).argmax(axis=1))
                if score > best:
                    best, selected = score, candidate
            weights[class_index] = selected
    return weights / weights.max()


def _paired_route_comparison(
    truth: np.ndarray,
    direct: np.ndarray,
    regression_route: np.ndarray,
    bias_corrected: np.ndarray,
    bootstrap: Sequence[np.ndarray],
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "_definition": f"paired case-level bootstrap, B = {len(bootstrap)}, seed = {seed}"
    }
    for name, prediction in (
        ("regression_route", regression_route),
        ("bias_corrected", bias_corrected),
    ):
        balanced_delta = _balanced_accuracy_codes(truth, prediction) - _balanced_accuracy_codes(truth, direct)
        balanced_samples = np.asarray(
            [
                _balanced_accuracy_codes(truth[sample], prediction[sample])
                - _balanced_accuracy_codes(truth[sample], direct[sample])
                for sample in bootstrap
            ]
        )
        accuracy_delta = float(np.mean(truth == prediction) - np.mean(truth == direct))
        accuracy_samples = np.asarray(
            [np.mean(truth[sample] == prediction[sample]) - np.mean(truth[sample] == direct[sample]) for sample in bootstrap]
        )
        balanced_interval = [float(value) for value in np.percentile(balanced_samples, (2.5, 97.5))]
        accuracy_interval = [float(value) for value in np.percentile(accuracy_samples, (2.5, 97.5))]
        result[name] = {
            "bal_delta": float(balanced_delta),
            "bal_delta_ci95": balanced_interval,
            "bal_distinguishable": bool(balanced_interval[0] > 0 or balanced_interval[1] < 0),
            "acc_delta": accuracy_delta,
            "acc_delta_ci95": accuracy_interval,
            "acc_distinguishable": bool(accuracy_interval[0] > 0 or accuracy_interval[1] < 0),
        }
    result["_interpretation"] = (
        "An interval containing zero means only that no difference was detected; equivalence was not tested."
    )
    return result


def paired_reliability(
    first_measurement: np.ndarray,
    second_measurement: np.ndarray,
    *,
    targets: Sequence[str] = TARGETS,
) -> dict[str, dict[str, float | int]]:
    """Compute ICC(1,1) and single-tracing error from available paired measurements."""
    first = np.asarray(first_measurement, dtype=np.float64)
    second = np.asarray(second_measurement, dtype=np.float64)
    names = tuple(targets)
    if first.ndim != 2 or first.shape != second.shape or first.shape[1] != len(names):
        raise ValueError("paired measurement matrices and targets must align")
    result: dict[str, dict[str, float | int]] = {}
    for index, target in enumerate(names):
        selected = np.isfinite(first[:, index]) & np.isfinite(second[:, index])
        if int(selected.sum()) < 3:
            raise ValueError(f"at least three paired measurements are required for {target}")
        x, y = first[selected, index], second[selected, index]
        result[target] = {
            "n": int(selected.sum()),
            "icc_1_1": icc_1_1(x, y),
            "single_tracing_error": single_tracing_error(x, y),
        }
    return result


def posthoc_route_report(
    calibration_truth: np.ndarray,
    calibration_sagittal_probabilities: np.ndarray,
    calibration_vertical_probabilities: np.ndarray,
    calibration_age: np.ndarray,
    calibration_sex: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    test_sagittal_probabilities: np.ndarray,
    test_vertical_probabilities: np.ndarray,
    test_age: np.ndarray,
    test_sex: np.ndarray,
    *,
    repeat_first: np.ndarray | None = None,
    repeat_second: np.ndarray | None = None,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
    bootstrap_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Compare classification routes with all fitted decision weights restricted to calibration."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    calibration_y = np.asarray(calibration_truth, dtype=np.float64)
    test_y, test_estimate, names = _regression_arrays(test_truth, test_prediction, targets)
    if calibration_y.ndim != 2 or calibration_y.shape[1] != len(names) or not np.isfinite(calibration_y).all():
        raise ValueError("calibration truth must be a finite case-by-target matrix")
    calibration_age_array = np.asarray(calibration_age, dtype=np.float64)
    calibration_sex_array = np.asarray(calibration_sex, dtype=object)
    test_age_array = np.asarray(test_age, dtype=np.float64)
    test_sex_array = np.asarray(test_sex, dtype=object)
    if calibration_age_array.shape != (len(calibration_y),) or calibration_sex_array.shape != (len(calibration_y),):
        raise ValueError("calibration demographics must align with calibration truth")
    if test_age_array.shape != (len(test_y),) or test_sex_array.shape != (len(test_y),):
        raise ValueError("test demographics must align with test truth")
    calibration_probabilities = (
        _probabilities(calibration_sagittal_probabilities, len(calibration_y), "calibration sagittal probabilities"),
        _probabilities(calibration_vertical_probabilities, len(calibration_y), "calibration vertical probabilities"),
    )
    test_probabilities = (
        _probabilities(test_sagittal_probabilities, len(test_y), "test sagittal probabilities"),
        _probabilities(test_vertical_probabilities, len(test_y), "test vertical probabilities"),
    )
    calibration_labels = _primary_labels(calibration_y, calibration_age_array, calibration_sex_array, names)
    test_labels = _primary_labels(test_y, test_age_array, test_sex_array, names)
    predicted_route = _primary_labels(test_estimate, test_age_array, test_sex_array, names)
    classes_by_axis = (SAGITTAL_CLASSES, VERTICAL_CLASSES)
    weights: list[np.ndarray] = []
    for labels, probabilities, classes in zip(calibration_labels, calibration_probabilities, classes_by_axis):
        weights.append(fit_class_bias(probabilities, _encoded_labels(labels, classes)))
    rng = np.random.default_rng(seed)
    routes: dict[str, object] = {}
    for axis, labels, regression_labels, probabilities, class_names, class_weights in zip(
        ("sagittal", "vertical"),
        test_labels,
        predicted_route,
        test_probabilities,
        classes_by_axis,
        weights,
    ):
        truth_code = _encoded_labels(labels, class_names)
        regression_code = _encoded_labels(regression_labels, class_names)
        direct = probabilities.argmax(axis=1)
        biased = (probabilities * class_weights).argmax(axis=1)
        bootstrap = [rng.integers(0, len(test_y), len(test_y)) for _ in range(bootstrap_resamples)]

        def route_metrics(prediction: np.ndarray) -> dict[str, float]:
            return {
                "bal": _balanced_accuracy_codes(truth_code, prediction),
                "acc": float(np.mean(truth_code == prediction)),
            }

        routes[axis] = {
            "direct": route_metrics(direct),
            "regression_route": route_metrics(regression_code),
            "bias_corrected": route_metrics(biased),
            "per_class_recall_direct": {
                name: float(np.mean(direct[truth_code == index] == index))
                for index, name in enumerate(class_names)
                if np.any(truth_code == index)
            },
            "per_class_recall_biased": {
                name: float(np.mean(biased[truth_code == index] == index))
                for index, name in enumerate(class_names)
                if np.any(truth_code == index)
            },
            "per_class_recall_regression": {
                name: float(np.mean(regression_code[truth_code == index] == index))
                for index, name in enumerate(class_names)
                if np.any(truth_code == index)
            },
            "paired_vs_direct": _paired_route_comparison(
                truth_code, direct, regression_code, biased, bootstrap, seed
            ),
        }
    ceiling: dict[str, object]
    if repeat_first is None or repeat_second is None:
        ceiling = {
            "status": "unavailable_without_paired_measurements",
            "required_inputs": ["first and independent second measurements for each target"],
        }
    else:
        reliability = paired_reliability(repeat_first, repeat_second, targets=names)
        ceiling = {}
        for index, target in enumerate(names):
            total = float(np.square(test_y[:, index] - test_y[:, index].mean()).sum())
            r2 = float(1.0 - np.square(test_y[:, index] - test_estimate[:, index]).sum() / total)
            icc = round(float(reliability[target]["icc_1_1"]), 4)
            ceiling[target] = {
                "R2": r2,
                "ICC": icc,
                "ratio": r2 / icc,
                "icc_is_measured": True,
                "icc_source": "paired measurements from the authorized cohort; ICC(1,1)",
            }
    return {
        "config": config,
        "analysis_status": "post_hoc",
        "T0_1_and_T0_2": {"internal_test": routes},
        "class_bias": {
            "sagittal": dict(zip(SAGITTAL_CLASSES, weights[0].tolist())),
            "vertical": dict(zip(VERTICAL_CLASSES, weights[1].tolist())),
        },
        "T0_3_icc_ceiling": ceiling,
        "protocol": "Class weights were fitted on calibration data and applied unchanged to test data.",
        "limitation": "R2 divided by ICC is descriptive; ICC is not a strict upper bound on predictive performance.",
    }


def _decision_metrics(truth: np.ndarray, prediction: np.ndarray, classes: int) -> dict[str, object]:
    if any(not np.any(truth == value) for value in range(classes)):
        raise ValueError("decision metrics require every declared class")
    recall = [float(np.mean(prediction[truth == value] == value)) for value in range(classes)]
    specificity = [float(np.mean(prediction[truth != value] != value)) for value in range(classes)]
    f1 = []
    for value in range(classes):
        true_positive = float(np.sum((prediction == value) & (truth == value)))
        f1.append(2.0 * true_positive / max(float(np.sum(prediction == value) + np.sum(truth == value)), 1e-9))
    return {
        "acc": float(np.mean(truth == prediction)),
        "bal": float(np.mean(recall)),
        "recall": recall,
        "spec": specificity,
        "macro_f1": float(np.mean(f1)),
    }


def _prior_adjust(probabilities: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    return (probabilities / np.power(prior, tau)[None, :]).argmax(axis=1)


def cost_sensitive_report(
    training_truth: np.ndarray,
    training_age: np.ndarray,
    training_sex: np.ndarray,
    calibration_truth: np.ndarray,
    calibration_sagittal_probabilities: np.ndarray,
    calibration_vertical_probabilities: np.ndarray,
    calibration_age: np.ndarray,
    calibration_sex: np.ndarray,
    test_truth: np.ndarray,
    test_sagittal_probabilities: np.ndarray,
    test_vertical_probabilities: np.ndarray,
    test_age: np.ndarray,
    test_sex: np.ndarray,
    *,
    config: str = "main",
    targets: Sequence[str] = TARGETS,
    minimum_minority_recall: float = 0.70,
) -> dict[str, object]:
    """Select prior-adjustment strength on calibration and apply it unchanged to test."""
    if not 0 < minimum_minority_recall <= 1:
        raise ValueError("minimum_minority_recall must lie in (0, 1]")
    names = tuple(targets)
    training_y = np.asarray(training_truth, dtype=np.float64)
    calibration_y = np.asarray(calibration_truth, dtype=np.float64)
    test_y = np.asarray(test_truth, dtype=np.float64)
    matrices = ((training_y, training_age, training_sex), (calibration_y, calibration_age, calibration_sex), (test_y, test_age, test_sex))
    labels = []
    for truth, age, sex in matrices:
        ages = np.asarray(age, dtype=np.float64)
        sexes = np.asarray(sex, dtype=object)
        if truth.ndim != 2 or truth.shape[1] != len(names) or ages.shape != (len(truth),) or sexes.shape != (len(truth),):
            raise ValueError("training, calibration, and test measurements must align with demographics")
        labels.append(_primary_labels(truth, ages, sexes, names))
    calibration_probability = (
        _probabilities(calibration_sagittal_probabilities, len(calibration_y), "calibration sagittal probabilities"),
        _probabilities(calibration_vertical_probabilities, len(calibration_y), "calibration vertical probabilities"),
    )
    test_probability = (
        _probabilities(test_sagittal_probabilities, len(test_y), "test sagittal probabilities"),
        _probabilities(test_vertical_probabilities, len(test_y), "test vertical probabilities"),
    )
    taus = np.round(np.arange(0.0, 1.55, 0.05), 2)
    output: dict[str, object] = {
        "config": config,
        "analysis_status": "post_hoc",
        "selection_rule": (
            "smallest tau reaching the minority-class recall target on calibration; otherwise maximum calibration balanced accuracy"
        ),
        "min_minor_recall": minimum_minority_recall,
        "protocol": "Tau is selected using calibration data alone and applied unchanged to test data.",
    }
    for axis_index, (axis, class_names) in enumerate(
        (("sagittal", SAGITTAL_CLASSES), ("vertical", VERTICAL_CLASSES))
    ):
        training_code = _encoded_labels(labels[0][axis_index], class_names)
        calibration_code = _encoded_labels(labels[1][axis_index], class_names)
        test_code = _encoded_labels(labels[2][axis_index], class_names)
        prior = np.asarray([np.mean(training_code == value) for value in range(3)], dtype=np.float64)
        if (prior <= 0).any():
            raise ValueError(f"every {axis} class must be represented in training data")
        calibration_rows = []
        for tau in taus:
            metrics = _decision_metrics(
                calibration_code, _prior_adjust(calibration_probability[axis_index], prior, float(tau)), 3
            )
            calibration_rows.append(
                {
                    "tau": float(tau),
                    "cal_acc": metrics["acc"],
                    "cal_bal": metrics["bal"],
                    "cal_recall": metrics["recall"],
                }
            )
        minority = int(np.argmin(prior))
        eligible = [row for row in calibration_rows if row["cal_recall"][minority] >= minimum_minority_recall]
        if eligible:
            tau_star = float(eligible[0]["tau"])
        else:
            tau_star = float(max(calibration_rows, key=lambda row: row["cal_bal"])["tau"])
        baseline = _decision_metrics(test_code, _prior_adjust(test_probability[axis_index], prior, 0.0), 3)
        selected = _decision_metrics(test_code, _prior_adjust(test_probability[axis_index], prior, tau_star), 3)
        test_curve = []
        for tau in taus:
            metrics = _decision_metrics(test_code, _prior_adjust(test_probability[axis_index], prior, float(tau)), 3)
            test_curve.append(
                {
                    "tau": float(tau),
                    "acc": metrics["acc"],
                    "bal": metrics["bal"],
                    "recall": metrics["recall"],
                    "spec": metrics["spec"],
                }
            )
        output[axis] = {
            "prior": dict(zip(class_names, prior.tolist())),
            "classes": list(class_names),
            "tau_star": tau_star,
            "test_tau0": baseline,
            "test_tau_star": selected,
            "test_curve": test_curve,
            "cal_curve": calibration_rows,
        }
    return output


def boundary_analysis_report(
    test_truth: np.ndarray,
    test_sagittal_probabilities: np.ndarray,
    test_vertical_probabilities: np.ndarray,
    test_age: np.ndarray,
    test_sex: np.ndarray,
    *,
    repeat_first: np.ndarray | None = None,
    repeat_second: np.ndarray | None = None,
    cohort_truth: np.ndarray | None = None,
    targets: Sequence[str] = TARGETS,
) -> dict[str, object]:
    """Relate direct-head errors to boundary distance and quantify a measurement-noise ceiling."""
    truth = np.asarray(test_truth, dtype=np.float64)
    names = tuple(targets)
    ages = np.asarray(test_age, dtype=np.float64)
    sexes = np.asarray(test_sex, dtype=object)
    if truth.ndim != 2 or truth.shape[1] != len(names) or not np.isfinite(truth).all():
        raise ValueError("test truth and target names must align")
    if ages.shape != (len(truth),) or sexes.shape != (len(truth),):
        raise ValueError("test demographics must align")
    probabilities = (
        _probabilities(test_sagittal_probabilities, len(truth), "test sagittal probabilities"),
        _probabilities(test_vertical_probabilities, len(truth), "test vertical probabilities"),
    )
    labels = _primary_labels(truth, ages, sexes, names)
    reliability = (
        paired_reliability(repeat_first, repeat_second, targets=names)
        if repeat_first is not None and repeat_second is not None
        else None
    )
    population = truth if cohort_truth is None else np.asarray(cohort_truth, dtype=np.float64)
    if population.ndim != 2 or population.shape[1] != len(names) or not np.isfinite(population).all():
        raise ValueError("cohort truth must be finite and match the target order")
    target_index = {name: index for index, name in enumerate(names)}
    scheme = THRESHOLD_SCHEMES["wu2021_1.5sd"]
    axes: dict[str, object] = {}
    for axis_index, (axis, metric, classes) in enumerate(
        (("sagittal", "ANB", SAGITTAL_CLASSES), ("vertical", "SN_MP", VERTICAL_CLASSES))
    ):
        measurement = truth[:, target_index[metric]]
        edges = np.empty((len(truth), 2), dtype=np.float64)
        for row in range(len(truth)):
            band = scheme.sagittal if axis == "sagittal" else scheme.vertical_band(ages[row], str(sexes[row]))
            edges[row] = (band.lower, band.upper)
        distance = np.minimum(np.abs(measurement - edges[:, 0]), np.abs(measurement - edges[:, 1]))
        truth_code = _encoded_labels(labels[axis_index], classes)
        correct = probabilities[axis_index].argmax(axis=1) == truth_code
        errors = int(np.sum(~correct))
        bins = []
        for lower, upper in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf)):
            selected = (distance >= lower) & (distance < upper)
            if not selected.any():
                continue
            bins.append(
                {
                    "lo": lower,
                    "hi": None if np.isinf(upper) else upper,
                    "n": int(selected.sum()),
                    "frac": float(selected.mean()),
                    "accuracy": float(correct[selected].mean()),
                    "share_of_errors": float(np.sum(~correct[selected]) / max(errors, 1)),
                }
            )
        row: dict[str, object] = {
            "metric": metric,
            "bins": bins,
            "observed_accuracy": float(correct.mean()),
        }
        if reliability is None:
            row["label_noise_ceiling"] = {
                "status": "unavailable_without_paired_measurements",
                "required_inputs": [f"paired first and second {metric} measurements"],
            }
        else:
            sem = round(float(reliability[metric]["single_tracing_error"]), 4)
            measured_icc = round(float(reliability[metric]["icc_1_1"]), 4)
            flip_probability = np.asarray(
                [0.5 * (1.0 - erf(value / (sem * sqrt(2.0)))) for value in distance], dtype=np.float64
            )
            row.update(
                {
                    "icc_used": measured_icc,
                    "icc_source": "paired measurements from the authorized cohort",
                    "icc_is_measured": True,
                    "icc_form": "ICC(1,1)",
                    "sd_measurement_error_source": "single-tracing error measured from paired cases",
                    "sd_eligible_cohort": float(population[:, target_index[metric]].std(ddof=1)),
                    "sd_population": "eligible cohort supplied in the controlled release bundle",
                    "sd_measurement_error": sem,
                    "mean_flip_prob": float(flip_probability.mean()),
                    "label_noise_accuracy_ceiling": float(1.0 - flip_probability.mean()),
                    "label_noise_ceiling_assumption": "one-sided Gaussian crossing probability with constant measured error SD",
                }
            )
        axes[axis] = row
    unavailable = {
        "status": "generator_not_in_release",
        "required_inputs": list(
            GENERATOR_NOT_IN_RELEASE_REQUIREMENTS["boundary_learning_curve_far.json"]
        ),
        "input_availability": (
            "The minimum controlled bundle does not include these prediction archives. "
            "Separately approved archives, if available to an authorized researcher, are not accepted by this analyzer."
        ),
        "interpretation": "No conclusion about the boundary hypothesis is inferred from an absent learning-curve comparison.",
    }
    full_source_sd = {
        "status": "unavailable_without_pre_eligibility_rows",
        "required_inputs": list(ADDITIONAL_INPUT_REQUIREMENTS["boundary_full_source_sd.json"]),
        "interpretation": (
            "The frozen artifact used a full-source SD that included pre-eligibility rows absent from the controlled bundle; "
            "the eligible-cohort SD reported here is not presented as the same estimand."
        ),
    }
    return {
        "split": "internal_test",
        "analysis_status": "post_hoc",
        "axes": axes,
        "learning_curve_far": unavailable,
        "last_segment_gain": unavailable,
        "frozen_full_source_sd": full_source_sd,
        "limitation": "The label-noise ceiling is an approximation under its stated measurement-error model.",
    }


def analysis_status(generated_files: Sequence[str]) -> dict[str, object]:
    """Describe generated outputs, unavailable inputs, and omitted generators."""
    inputs = {
        "bland_altman.json": ("internal-test measurements", "internal-test regression predictions"),
        "age_strata": (
            "internal-test measurements",
            "internal-test predictions",
            "authorized age and sex fields",
        ),
        "shrinkage": ("calibration measurements and predictions", "internal-test measurements and predictions"),
        "conformal_adaptivity": (
            "calibration measurements, predictions, and uncertainty",
            "internal-test measurements, predictions, and uncertainty",
        ),
        "sigma_patient_level": ("internal-test measurements, predictions, and uncertainty",),
        "threshold_sensitivity": (
            "declared threshold schemes",
            "internal-test continuous and classification predictions",
        ),
        "posthoc": (
            "calibration and internal-test measurements and predictions",
            "paired measurements for the reliability comparison",
        ),
        "cost_sensitive": (
            "training class prevalences",
            "calibration and internal-test classification predictions",
        ),
        "boundary_analysis.json": (
            "internal-test measurements and classification predictions",
            "paired measurements for the label-noise ceiling",
            "eligible-cohort measurements for the release-bundle SD",
        ),
    }
    generated: dict[str, object] = {}
    for filename in generated_files:
        if filename in {"bland_altman.json", "boundary_analysis.json"}:
            key = filename
        else:
            matches = [name for name in inputs if filename.startswith(f"{name}_")]
            if len(matches) != 1:
                raise ValueError(f"unsupported aggregate report name: {filename}")
            key = matches[0]
        generated[filename] = {
            "status": "generated",
            "required_inputs": list(inputs[key]),
        }
    missing = {
        name: {
            "status": "unavailable_without_intermediate_data",
            "required_inputs": list(requirements),
        }
        for name, requirements in ADDITIONAL_INPUT_REQUIREMENTS.items()
    }
    omitted_generators = {
        name: {
            "status": "generator_not_in_release",
            "required_inputs": list(requirements),
        }
        for name, requirements in GENERATOR_NOT_IN_RELEASE_REQUIREMENTS.items()
    }
    return {
        "schema_version": 2,
        "generated": generated,
        "requires_additional_input": missing,
        "generator_not_in_release": omitted_generators,
    }


def aggregate_analysis_reports(
    training_truth: np.ndarray,
    training_age: np.ndarray,
    training_sex: np.ndarray,
    calibration_truth: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_sigma: np.ndarray,
    calibration_sagittal_probabilities: np.ndarray,
    calibration_vertical_probabilities: np.ndarray,
    calibration_age: np.ndarray,
    calibration_sex: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    test_sigma: np.ndarray,
    sagittal_probabilities: np.ndarray,
    vertical_probabilities: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    repeat_first: np.ndarray,
    repeat_second: np.ndarray,
    cohort_truth: np.ndarray,
    *,
    alpha: float = 0.10,
    config: str = "main",
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Compute every aggregate analysis supported by the controlled prediction bundle."""
    if not config or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in config):
        raise ValueError("config must be a safe alphanumeric name")
    reports: dict[str, object] = {
        "bland_altman.json": bland_altman_report(test_truth, test_prediction),
        f"age_strata_{config}.json": age_strata_report(
            test_truth,
            test_prediction,
            sagittal_probabilities,
            vertical_probabilities,
            age,
            sex,
            config=config,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        f"shrinkage_{config}.json": shrinkage_report(
            calibration_truth,
            calibration_prediction,
            test_truth,
            test_prediction,
            config=config,
        ),
        f"conformal_adaptivity_{config}.json": conformal_adaptivity_report(
            calibration_truth,
            calibration_prediction,
            calibration_sigma,
            test_truth,
            test_prediction,
            test_sigma,
            alpha=alpha,
            config=config,
        ),
        f"sigma_patient_level_{config}.json": patient_sigma_report(
            test_truth, test_prediction, test_sigma, config=config
        ),
        f"threshold_sensitivity_{config}.json": threshold_sensitivity_report(
            test_truth,
            test_prediction,
            sagittal_probabilities,
            vertical_probabilities,
            age,
            sex,
            config=config,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        f"posthoc_{config}.json": posthoc_route_report(
            calibration_truth,
            calibration_sagittal_probabilities,
            calibration_vertical_probabilities,
            calibration_age,
            calibration_sex,
            test_truth,
            test_prediction,
            sagittal_probabilities,
            vertical_probabilities,
            age,
            sex,
            repeat_first=repeat_first,
            repeat_second=repeat_second,
            config=config,
            bootstrap_resamples=bootstrap_resamples,
            seed=42,
        ),
        f"cost_sensitive_{config}.json": cost_sensitive_report(
            training_truth,
            training_age,
            training_sex,
            calibration_truth,
            calibration_sagittal_probabilities,
            calibration_vertical_probabilities,
            calibration_age,
            calibration_sex,
            test_truth,
            sagittal_probabilities,
            vertical_probabilities,
            age,
            sex,
            config=config,
        ),
        "boundary_analysis.json": boundary_analysis_report(
            test_truth,
            sagittal_probabilities,
            vertical_probabilities,
            age,
            sex,
            repeat_first=repeat_first,
            repeat_second=repeat_second,
            cohort_truth=cohort_truth,
        ),
    }
    reports["analysis_status.json"] = analysis_status(tuple(reports))
    return reports


def write_aggregate_reports(output_directory: str | Path, reports: Mapping[str, object]) -> Path:
    """Write a complete report set to a new directory under the generated tree."""
    selected = sorted(reports.items())
    for filename, payload in selected:
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".json":
            raise ValueError("aggregate report names must be plain JSON filenames")
        json.dumps(payload, allow_nan=False)
    destination = create_directory(output_directory)
    for filename, payload in selected:
        write_json(destination / filename, payload)
    return destination
