"""Independent Extra Trees baseline for authorized geometry features."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _matrix(values: object, name: str, *, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    invalid = np.isinf(array).any() if allow_nan else not np.isfinite(array).all()
    if array.ndim != 2 or array.shape[0] < 4 or array.shape[1] < 1 or invalid:
        finite = "without infinite values" if allow_nan else "finite"
        raise ValueError(f"{name} must be a {finite} two-dimensional array with at least four rows")
    if allow_nan and np.isnan(array).all(axis=0).any():
        raise ValueError(f"{name} contains an entirely missing feature")
    return array


def _labels(values: object, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must be a one-dimensional array matching the features")
    if any(value is None or (isinstance(value, float) and not np.isfinite(value)) for value in array.tolist()):
        raise ValueError(f"{name} contains missing values")
    return array.astype(str)


def evaluate_geometry_baseline(
    features: np.ndarray,
    regression_targets: np.ndarray,
    sagittal_labels: Sequence[object],
    vertical_labels: Sequence[object],
    fold_ids: Sequence[object],
    *,
    target_names: Sequence[str] | None = None,
    n_estimators: int = 500,
    min_samples_leaf: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate an Extra Trees geometry baseline with preassigned outer folds."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        r2_score,
    )

    x = _matrix(features, "features", allow_nan=True)
    y = _matrix(regression_targets, "regression_targets")
    if y.shape[0] != x.shape[0]:
        raise ValueError("regression_targets must match features")
    if n_estimators < 1 or min_samples_leaf < 1:
        raise ValueError("n_estimators and min_samples_leaf must be positive")
    names = tuple(target_names) if target_names is not None else tuple(f"target_{index}" for index in range(y.shape[1]))
    if len(names) != y.shape[1] or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("target_names must be unique and match regression_targets")

    sag = _labels(sagittal_labels, x.shape[0], "sagittal_labels")
    vert = _labels(vertical_labels, x.shape[0], "vertical_labels")
    folds = _labels(fold_ids, x.shape[0], "fold_ids")
    unique_folds = np.unique(folds)
    if unique_folds.size < 2:
        raise ValueError("fold_ids must contain at least two folds")
    sag_classes = np.unique(sag)
    vert_classes = np.unique(vert)
    if sag_classes.size < 2 or vert_classes.size < 2:
        raise ValueError("both classification outcomes must contain at least two classes")

    pred_reg = np.empty_like(y)
    pred_sag = np.empty_like(sag)
    pred_vert = np.empty_like(vert)
    for fold_index, fold in enumerate(unique_folds):
        test = folds == fold
        train = ~test
        if test.sum() < 1 or train.sum() < 2:
            raise ValueError(f"fold {fold} does not define a valid train/test partition")
        if set(np.unique(sag[train])) != set(sag_classes) or set(np.unique(vert[train])) != set(vert_classes):
            raise ValueError(f"fold {fold} leaves a classification class absent from training")
        imputer = SimpleImputer(strategy="median").fit(x[train])
        train_x = imputer.transform(x[train])
        test_x = imputer.transform(x[test])
        common = {
            "n_estimators": int(n_estimators),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": 1.0,
            "random_state": int(seed + fold_index),
            "n_jobs": 1,
        }
        regressor = ExtraTreesRegressor(**common).fit(train_x, y[train])
        sagittal = ExtraTreesClassifier(class_weight="balanced", **common).fit(train_x, sag[train])
        vertical = ExtraTreesClassifier(class_weight="balanced", **common).fit(train_x, vert[train])
        pred_reg[test] = regressor.predict(test_x)
        pred_sag[test] = sagittal.predict(test_x)
        pred_vert[test] = vertical.predict(test_x)

    regression = {
        name: {
            "mae": float(mean_absolute_error(y[:, index], pred_reg[:, index])),
            "r2": float(r2_score(y[:, index], pred_reg[:, index])),
        }
        for index, name in enumerate(names)
    }

    def classification(reference: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> dict[str, object]:
        return {
            "classes": classes.tolist(),
            "accuracy": float(accuracy_score(reference, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(reference, prediction)),
            "macro_f1": float(f1_score(reference, prediction, labels=classes, average="macro")),
            "confusion": confusion_matrix(reference, prediction, labels=classes).astype(int).tolist(),
            "support": [int(np.sum(reference == value)) for value in classes],
        }

    return {
        "method": {
            "estimator": "scikit-learn ExtraTreesRegressor and ExtraTreesClassifier",
            "evaluation": "out-of-fold predictions from preassigned outer folds",
            "feature_imputation": "training-fold median",
            "n_estimators": int(n_estimators),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": 1.0,
            "seed": int(seed),
        },
        "n_cases": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_folds": int(unique_folds.size),
        "regression": regression,
        "regression_mae_mean": float(np.mean([entry["mae"] for entry in regression.values()])),
        "classification": {
            "sagittal": classification(sag, pred_sag, sag_classes),
            "vertical": classification(vert, pred_vert, vert_classes),
        },
    }
