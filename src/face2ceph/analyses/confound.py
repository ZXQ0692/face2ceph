"""Descriptive probes of acquisition information in frozen representations."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _features(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array with at least two rows")
    return array


def _labels(values: object, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must be a one-dimensional array matching the features")
    if any(value is None or (isinstance(value, float) and not np.isfinite(value)) for value in array.tolist()):
        raise ValueError(f"{name} contains missing values")
    return array.astype(str)


def _not_applicable(reason: str) -> dict[str, object]:
    return {
        "status": "not_applicable",
        "reason": reason,
        "interpretation": "Not applicable does not mean that the groups are indistinguishable.",
    }


def _probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes, encoded = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        return _not_applicable("Only one group is present.")
    counts = np.bincount(encoded)
    folds = min(int(n_splits), int(counts.min()))
    if folds < 2:
        return _not_applicable("At least one group has fewer than two observations.")

    predicted = np.full(encoded.shape, -1, dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, encoded):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"),
        )
        model.fit(features[train], encoded[train])
        predicted[test] = model.predict(features[test])
    return {
        "status": "estimated",
        "estimator": "standardized multinomial logistic regression",
        "n": int(encoded.size),
        "n_groups": int(classes.size),
        "n_splits": folds,
        "accuracy": float(accuracy_score(encoded, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(encoded, predicted)),
        "balanced_chance": float(1.0 / classes.size),
        "majority_accuracy": float(counts.max() / encoded.size),
    }


def _shuffled_control(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    shuffled = np.random.default_rng(seed).permutation(labels)
    return _probe(features, shuffled, n_splits=n_splits, seed=seed)


def _silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    max_cases: int,
) -> dict[str, object]:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    classes, encoded = np.unique(labels, return_inverse=True)
    if classes.size < 2:
        return _not_applicable("Only one group is present.")
    if classes.size >= encoded.size or np.bincount(encoded).min() < 2:
        return _not_applicable("Silhouette estimation requires at least two observations per group.")
    sample_size = min(int(max_cases), encoded.size)
    score = silhouette_score(
        StandardScaler().fit_transform(features),
        encoded,
        metric="euclidean",
        sample_size=sample_size if sample_size < encoded.size else None,
        random_state=seed,
    )
    return {"status": "estimated", "n_sampled": sample_size, "silhouette": float(score)}


def analyze_confound_probes(
    train_features: np.ndarray,
    train_acquisition_labels: Sequence[object],
    inference_features: np.ndarray,
    inference_domain_labels: Sequence[object],
    *,
    train_demographics: np.ndarray | None = None,
    inference_phenotype_labels: Sequence[object] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    silhouette_max_cases: int = 4000,
) -> dict[str, object]:
    """Estimate aggregate frozen-feature probes without using them for model selection."""
    train_x = _features(train_features, "train_features")
    inference_x = _features(inference_features, "inference_features")
    train_groups = _labels(train_acquisition_labels, train_x.shape[0], "train_acquisition_labels")
    domains = _labels(inference_domain_labels, inference_x.shape[0], "inference_domain_labels")
    if n_splits < 2 or silhouette_max_cases < 2:
        raise ValueError("n_splits and silhouette_max_cases must be at least two")

    acquisition_probe = _probe(train_x, train_groups, n_splits=n_splits, seed=seed)
    shuffled_probe = _shuffled_control(train_x, train_groups, n_splits=n_splits, seed=seed + 1)
    demographics_probe: dict[str, object] | None = None
    if train_demographics is not None:
        demographics = _features(train_demographics, "train_demographics")
        if demographics.shape[0] != train_x.shape[0]:
            raise ValueError("train_demographics must match train_features")
        demographics_probe = _probe(demographics, train_groups, n_splits=n_splits, seed=seed)

    geometry: dict[str, object] = {
        "domain": _silhouette(inference_x, domains, seed=seed, max_cases=silhouette_max_cases)
    }
    if inference_phenotype_labels is not None:
        phenotypes = _labels(inference_phenotype_labels, inference_x.shape[0], "inference_phenotype_labels")
        geometry["phenotype"] = _silhouette(
            inference_x, phenotypes, seed=seed, max_cases=silhouette_max_cases
        )

    return {
        "method": {
            "features": "frozen",
            "role": "descriptive probe only; no probe result was used to train or select the prediction model",
            "domain_adversarial_training": False,
            "domain_adversarial_statement": "No gradient-reversal or domain-adversarial branch was trained.",
        },
        "training_acquisition_probe": acquisition_probe,
        "training_acquisition_shuffled_control": shuffled_probe,
        "training_demographic_control": demographics_probe,
        "inference_domain_probe": _probe(inference_x, domains, n_splits=n_splits, seed=seed),
        "representation_geometry": geometry,
    }
