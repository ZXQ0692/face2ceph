"""Validation-history comparisons and publication learning-curve summaries."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


METRICS = ("mae_mean", "balanced_accuracy_sagittal", "balanced_accuracy_vertical")
REPORTED_METRICS = (*METRICS, "balanced_accuracy_mean")
PUBLICATION_ARM_LABELS: Mapping[str, str] = MappingProxyType({
    "classification_rgb": "c1",
    "classification_shape": "c2",
    "multitask": "c3",
    "main": "c4b",
    "learning_10": "learning_curve_10pct",
    "learning_25": "learning_curve_25pct",
    "learning_50": "learning_curve_50pct",
    "learning_75": "learning_curve_75pct",
})


def _number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _selected_folds(
    arm_name: str,
    arm: Mapping[str, object],
    *,
    selection_criterion: str | None = None,
) -> list[dict[str, object]]:
    criterion = str(selection_criterion or arm.get("selection_criterion", "mae_mean"))
    if criterion not in {"mae_mean", "balanced_accuracy_mean"}:
        raise ValueError(f"{arm_name}.selection_criterion is unsupported")
    folds = arm.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or len(folds) < 2:
        raise ValueError(f"{arm_name}.folds must contain at least two folds")
    selected: list[dict[str, object]] = []
    fold_ids: set[int] = set()
    for position, raw_fold in enumerate(folds):
        if not isinstance(raw_fold, Mapping):
            raise ValueError(f"{arm_name}.folds[{position}] must be an object")
        fold_id = int(raw_fold.get("fold", position))
        if fold_id in fold_ids:
            raise ValueError(f"{arm_name} contains duplicate fold {fold_id}")
        fold_ids.add(fold_id)
        n_train = int(raw_fold.get("n_train", 0))
        if n_train < 2:
            raise ValueError(f"{arm_name}.folds[{position}].n_train must be at least two")
        epochs = raw_fold.get("epochs")
        if not isinstance(epochs, Sequence) or isinstance(epochs, (str, bytes)) or not epochs:
            raise ValueError(f"{arm_name}.folds[{position}].epochs must be non-empty")
        candidates: list[dict[str, object]] = []
        epoch_ids: set[int] = set()
        for epoch_position, raw_epoch in enumerate(epochs):
            if not isinstance(raw_epoch, Mapping):
                raise ValueError(f"{arm_name} epoch records must be objects")
            epoch = int(raw_epoch.get("epoch", epoch_position))
            if epoch in epoch_ids:
                raise ValueError(f"{arm_name} fold {fold_id} contains duplicate epoch {epoch}")
            epoch_ids.add(epoch)
            sagittal = _number(raw_epoch.get("balanced_accuracy_sagittal"), f"{arm_name}.balanced_accuracy_sagittal")
            vertical = _number(raw_epoch.get("balanced_accuracy_vertical"), f"{arm_name}.balanced_accuracy_vertical")
            if not 0.0 <= sagittal <= 1.0 or not 0.0 <= vertical <= 1.0:
                raise ValueError(f"{arm_name} balanced accuracies must be in [0, 1]")
            raw_mae = raw_epoch.get("mae_mean")
            mae = None if raw_mae is None else _number(raw_mae, f"{arm_name}.mae_mean")
            candidates.append({
                "fold": fold_id,
                "epoch": epoch,
                "n_train": n_train,
                "mae_mean": mae,
                "balanced_accuracy_sagittal": sagittal,
                "balanced_accuracy_vertical": vertical,
                "balanced_accuracy_mean": 0.5 * (sagittal + vertical),
            })
        mae_available = [row["mae_mean"] is not None for row in candidates]
        if any(mae_available) and not all(mae_available):
            raise ValueError(f"{arm_name} fold {fold_id} mixes epochs with and without regression metrics")
        if criterion == "mae_mean":
            if any(row["mae_mean"] is None for row in candidates):
                raise ValueError(f"{arm_name} requires finite mae_mean values for its selection criterion")
            chosen = min(candidates, key=lambda row: (float(row["mae_mean"]), int(row["epoch"])))
        else:
            chosen = max(candidates, key=lambda row: (float(row["balanced_accuracy_mean"]), -int(row["epoch"])))
        selected.append(chosen)
    return sorted(selected, key=lambda row: int(row["fold"]))


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "population_sd": float(values.std(ddof=0)),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compare_arm_histories(histories: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Compare arms at each arm's declared fold-level checkpoint criterion."""
    if not histories:
        raise ValueError("histories must contain at least one arm")
    arms: dict[str, object] = {}
    for name in sorted(histories):
        folds = _selected_folds(name, histories[name])
        mae_available = [row["mae_mean"] is not None for row in folds]
        if any(mae_available) and not all(mae_available):
            raise ValueError(f"{name} mixes folds with and without regression metrics")
        criterion = str(histories[name].get("selection_criterion", "mae_mean"))
        metrics: dict[str, object] = {}
        for metric in REPORTED_METRICS:
            values = [row[metric] for row in folds]
            metrics[metric] = (
                None
                if all(value is None for value in values)
                else _metric_summary(np.asarray(values, dtype=float))
            )
        arms[name] = {
            "selection_criterion": criterion,
            "n_folds": len(folds),
            "n_train_mean": float(np.mean([row["n_train"] for row in folds])),
            "selected_epoch_mean": float(np.mean([row["epoch"] for row in folds])),
            "metrics": metrics,
            "fold_values": folds,
        }
    return {
        "selection_rule": "declared per arm; earliest epoch breaks exact ties",
        "classification_reporting": "classification metrics are evaluated at the declared checkpoint epoch",
        "fold_dispersion": {
            "population_sd": "ddof=0; convention used in the frozen arm-comparison artifact",
            "sample_sd": "ddof=1; sample standard deviation across validation folds",
        },
        "arms": arms,
    }


def _publication_fold_summary(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def pair(metric: str) -> list[float | None]:
        values = [row[metric] for row in folds]
        if all(value is None for value in values):
            return [None, None]
        if any(value is None for value in values):
            raise ValueError(f"{metric} is unavailable in only some folds")
        array = np.asarray(values, dtype=float)
        return [float(array.mean()), float(array.std(ddof=0))]

    return {
        "bal_sag": pair("balanced_accuracy_sagittal"),
        "bal_vert": pair("balanced_accuracy_vertical"),
        "mae": pair("mae_mean"),
        "epoch": float(np.mean([row["epoch"] for row in folds])),
        "n_folds": len(folds),
    }


def _publication_arm_names(
    names: Sequence[str],
    arm_labels: Mapping[str, str] | None,
) -> dict[str, str]:
    labels = dict(PUBLICATION_ARM_LABELS)
    if arm_labels is not None:
        labels.update({str(name): str(label) for name, label in arm_labels.items()})
    result = {name: labels.get(name, name).strip() for name in names}
    if any(not label for label in result.values()):
        raise ValueError("publication arm labels must be non-empty")
    if len(set(result.values())) != len(result):
        raise ValueError("publication arm labels must be unique")
    return result


def publication_compare_arms_summary(
    histories: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    arm_labels: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the frozen compare_arms_summary.json structure from validation histories."""
    if not histories:
        raise ValueError("histories must contain at least one arm")
    if set(metadata) != set(histories):
        raise ValueError("metadata must contain exactly the declared arms")
    publication_names = _publication_arm_names(list(histories), arm_labels)
    rows: list[dict[str, object]] = []
    for name, arm in histories.items():
        details = metadata[name]
        if not isinstance(details, Mapping):
            raise ValueError(f"{name} metadata must be an object")
        architecture = details.get("arch")
        if not isinstance(architecture, Mapping):
            raise ValueError(f"{name}.arch must be an object")
        primary = _publication_fold_summary(_selected_folds(name, arm))
        oracle = _publication_fold_summary(
            _selected_folds(name, arm, selection_criterion="balanced_accuracy_mean")
        )
        criterion = str(arm.get("selection_criterion", "mae_mean"))
        rows.append({
            "arm": publication_names[name],
            "desc": str(details.get("desc", "")),
            "arch": dict(architecture),
            "criterion": "bal_mean" if criterion == "balanced_accuracy_mean" else criterion,
            "cls": dict(primary),
            "reg": dict(primary),
            "cls_oracle": oracle,
        })
    return {"arms": [publication_names[name] for name in histories], "rows": rows}


def publication_learning_curve_summary(
    histories: Mapping[str, Mapping[str, object]],
    *,
    arm_labels: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the frozen learning_curve.json structure using its ddof=0 convention."""
    if len(histories) < 4:
        raise ValueError("a publication learning curve requires at least four trained sample-size points")
    publication_names = _publication_arm_names(list(histories), arm_labels)
    points: list[tuple[float, dict[str, object]]] = []
    for name, arm in histories.items():
        fraction = _number(arm.get("fraction"), f"{name}.fraction")
        if not 0 < fraction <= 1:
            raise ValueError(f"{name}.fraction must be in (0, 1]")
        if str(arm.get("selection_criterion", "mae_mean")) != "mae_mean":
            raise ValueError(f"{name} is not a regression-MAE-selected learning-curve arm")
        folds = _selected_folds(name, arm)

        def values(metric: str) -> np.ndarray:
            return np.asarray([row[metric] for row in folds], dtype=float)

        mae = values("mae_mean")
        sagittal = values("balanced_accuracy_sagittal")
        vertical = values("balanced_accuracy_vertical")
        points.append((fraction, {
            "frac": fraction,
            "arm": publication_names[name],
            "n_train": int(round(float(np.mean([row["n_train"] for row in folds])))),
            "mae": float(mae.mean()),
            "mae_sd": float(mae.std(ddof=0)),
            "bal_sag": float(sagittal.mean()),
            "bal_sag_sd": float(sagittal.std(ddof=0)),
            "bal_vert": float(vertical.mean()),
            "bal_vert_sd": float(vertical.std(ddof=0)),
        }))
    points.sort(key=lambda item: item[0])
    fractions = np.asarray([item[0] for item in points])
    if np.unique(fractions).size != fractions.size:
        raise ValueError("arm fractions must be unique")
    return {"points": [item[1] for item in points]}


def fit_power_curve(
    n_train: Sequence[float],
    observed: Sequence[float],
    *,
    decreasing: bool,
    exponent_grid: Sequence[float] | None = None,
) -> dict[str, float] | None:
    """Fit y = c + s*a*n^(-b), where a and b are positive and s encodes direction."""
    n = np.asarray(n_train, dtype=float)
    y = np.asarray(observed, dtype=float)
    if n.ndim != 1 or y.shape != n.shape or n.size < 3 or not np.isfinite(n).all() or not np.isfinite(y).all():
        raise ValueError("n_train and observed must be finite paired vectors with at least three values")
    if np.any(n <= 0) or np.unique(n).size != n.size:
        raise ValueError("n_train must contain distinct positive values")
    exponents = np.asarray(
        tuple(exponent_grid) if exponent_grid is not None else np.linspace(0.05, 2.0, 400),
        dtype=float,
    )
    if exponents.ndim != 1 or not exponents.size or not np.isfinite(exponents).all() or np.any(exponents <= 0):
        raise ValueError("exponent_grid must contain positive finite values")
    sign = 1.0 if decreasing else -1.0
    best: tuple[float, float, float, float] | None = None
    for exponent in exponents:
        design = np.column_stack((np.ones_like(n), sign * np.power(n, -exponent)))
        intercept, scale = np.linalg.lstsq(design, y, rcond=None)[0]
        if scale <= 0:
            continue
        residual_sum_squares = float(np.square(design @ np.array((intercept, scale)) - y).sum())
        if best is None or residual_sum_squares < best[0]:
            best = (residual_sum_squares, float(intercept), float(scale), float(exponent))
    if best is None:
        return None
    return {"c": best[1], "a": best[2], "b": best[3], "residual_sum_squares": best[0]}


def _predict_power(fit: Mapping[str, float], n: float, decreasing: bool) -> float:
    sign = 1.0 if decreasing else -1.0
    return float(fit["c"] + sign * fit["a"] * n ** (-fit["b"]))


def aggregate_learning_curve(
    histories: Mapping[str, Mapping[str, object]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    extrapolation_factors: Sequence[float] = (2.0, 4.0, 10.0),
) -> dict[str, object]:
    """Aggregate four or more arm histories using measured fold-level training sizes."""
    if len(histories) < 4:
        raise ValueError("a publication learning curve requires at least four trained sample-size points")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    factors = np.asarray(extrapolation_factors, dtype=float)
    if factors.ndim != 1 or not factors.size or not np.isfinite(factors).all() or np.any(factors <= 1):
        raise ValueError("extrapolation_factors must be finite and greater than one")

    points = []
    for name, arm in histories.items():
        fraction = _number(arm.get("fraction"), f"{name}.fraction")
        if not 0 < fraction <= 1:
            raise ValueError(f"{name}.fraction must be in (0, 1]")
        folds = _selected_folds(name, arm)
        if str(arm.get("selection_criterion", "mae_mean")) != "mae_mean":
            raise ValueError(f"{name} is not a regression-MAE-selected learning-curve arm")
        points.append((fraction, name, folds))
    points.sort(key=lambda point: point[0])
    fractions = np.asarray([point[0] for point in points])
    if np.unique(fractions).size != fractions.size:
        raise ValueError("arm fractions must be unique")
    n_train = np.asarray([np.mean([row["n_train"] for row in point[2]]) for point in points])
    if np.any(np.diff(n_train) <= 0):
        raise ValueError("measured mean training sizes must increase with fraction")

    rng = np.random.default_rng(seed)
    result: dict[str, object] = {
        "selection_rule": "minimum validation regression MAE in each fold",
        "fit_scope": "exploratory power-law fit; extrapolations are not observed performance",
        "arms": [point[1] for point in points],
        "fractions": fractions.tolist(),
        "n_train": n_train.tolist(),
        "metrics": {},
    }
    directions = {
        "mae_mean": True,
        "balanced_accuracy_sagittal": False,
        "balanced_accuracy_vertical": False,
    }
    for metric, decreasing in directions.items():
        fold_values = [np.asarray([row[metric] for row in point[2]], dtype=float) for point in points]
        observed = np.asarray([values.mean() for values in fold_values])
        observed_sd = np.asarray([values.std(ddof=1) for values in fold_values])
        fit = fit_power_curve(n_train, observed, decreasing=decreasing)
        if fit is None:
            raise ValueError(f"{metric} does not admit a monotone positive-scale power fit")

        c_samples: list[float] = []
        predictions = {float(factor): [] for factor in factors}
        for _ in range(n_boot):
            resampled = np.asarray([
                rng.choice(values, size=values.size, replace=True).mean() for values in fold_values
            ])
            boot_fit = fit_power_curve(n_train, resampled, decreasing=decreasing)
            if boot_fit is None:
                continue
            c_samples.append(boot_fit["c"])
            for factor in factors:
                predictions[float(factor)].append(
                    _predict_power(boot_fit, float(n_train[-1] * factor), decreasing)
                )
        if not c_samples:
            raise ValueError(f"no valid bootstrap power fits for {metric}")

        delta_last = float(observed[-1] - observed[-2])
        pooled_sd = float(np.sqrt((observed_sd[-1] ** 2 + observed_sd[-2] ** 2) / 2.0))
        current = float(observed[-1])
        metric_result: dict[str, object] = {
            "observed": observed.tolist(),
            "observed_sample_sd": observed_sd.tolist(),
            "power_fit": fit,
            "asymptote_bootstrap_ci": np.percentile(c_samples, (2.5, 97.5)).tolist(),
            "n_boot_requested": int(n_boot),
            "n_boot_valid": len(c_samples),
            "last_segment_delta": delta_last,
            "last_segment_pooled_sd": pooled_sd,
            "last_segment_standardized": float(abs(delta_last) / max(pooled_sd, 1e-12)),
            "extrapolation": {},
        }
        for factor, values in predictions.items():
            estimates = np.asarray(values)
            better = estimates < current if decreasing else estimates > current
            metric_result["extrapolation"][f"{factor:g}x"] = {
                "n_train": float(n_train[-1] * factor),
                "median": float(np.median(estimates)),
                "ci": np.percentile(estimates, (2.5, 97.5)).tolist(),
                "probability_better_than_current": float(better.mean()),
            }
        result["metrics"][metric] = metric_result
    return result


def publication_learning_curve_fit(
    histories: Mapping[str, Mapping[str, object]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    extrapolation_factors: Sequence[float] = (2.0, 4.0, 10.0),
) -> dict[str, object]:
    """Build the frozen learning_curve_fit.json structure using its ddof=1 convention."""
    aggregate = aggregate_learning_curve(
        histories,
        n_boot=n_boot,
        seed=seed,
        extrapolation_factors=extrapolation_factors,
    )
    metric_names = {
        "mae_mean": "mae",
        "balanced_accuracy_sagittal": "bal_sag",
        "balanced_accuracy_vertical": "bal_vert",
    }
    metrics: dict[str, object] = {}
    for source_name, output_name in metric_names.items():
        source = aggregate["metrics"][source_name]
        fit = source["power_fit"]
        extrapolation = {
            key.removesuffix("x"): {
                "median": value["median"],
                "ci": value["ci"],
                "p_better_than_current": value["probability_better_than_current"],
            }
            for key, value in source["extrapolation"].items()
        }
        metrics[output_name] = {
            "observed": source["observed"],
            "observed_sd": source["observed_sample_sd"],
            "c": fit["c"],
            "a": fit["a"],
            "b": fit["b"],
            "c_ci": source["asymptote_bootstrap_ci"],
            "last_segment_delta": source["last_segment_delta"],
            "last_segment_pooled_sd": source["last_segment_pooled_sd"],
            "last_segment_n_sd": source["last_segment_standardized"],
            "extrapolation": extrapolation,
        }
    return {"n": aggregate["n_train"], "metrics": metrics}
