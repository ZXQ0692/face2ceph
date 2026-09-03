from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import face2ceph.dataset as dataset_module
from face2ceph.analyses.confound import analyze_confound_probes
from face2ceph.analyses.geometry import evaluate_geometry_baseline
from face2ceph.analyses.learning import (
    PUBLICATION_ARM_LABELS,
    aggregate_learning_curve,
    compare_arm_histories,
    fit_power_curve,
    publication_compare_arms_summary,
    publication_learning_curve_fit,
    publication_learning_curve_summary,
)
from face2ceph.analyses.perturbation import (
    DEFAULT_PERTURBATIONS,
    PerturbationTransform,
    apply_perturbation,
    score_perturbation_grid,
)
from face2ceph.dataset import ClinicalPhotoDataset


def _prediction(y: np.ndarray, offset: float = 0.0, sigma: float = 1.0) -> dict[str, np.ndarray]:
    labels = np.arange(y.shape[0]) % 3
    probabilities = np.eye(3)[labels]
    return {
        "mu": y + offset,
        "prob_sag": probabilities,
        "prob_vert": probabilities,
        "sigma": np.full_like(y, sigma),
    }


def test_perturbations_are_deterministic_and_grid_is_complete() -> None:
    assert len(DEFAULT_PERTURBATIONS) == 23
    images = np.arange(2 * 12 * 10 * 3, dtype=np.uint8).reshape(2, 12, 10, 3)
    masks = images[..., :1]
    original = images.copy()
    noise = next(spec for spec in DEFAULT_PERTURBATIONS if spec.kind == "noise")
    first = apply_perturbation(images, images, masks, noise, seed=7)
    second = apply_perturbation(images, images, masks, noise, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(images, original)
    assert np.array_equal(first[2], masks)
    transform = PerturbationTransform(noise, seed=7)
    transformed_case = transform(images[0], images[0], masks[0, ..., 0])
    assert np.array_equal(transformed_case[0], first[0][0])
    assert np.array_equal(transformed_case[2], first[2][0, ..., 0])
    assert pickle.loads(pickle.dumps(transform)) == transform
    for spec in DEFAULT_PERTURBATIONS:
        transformed = PerturbationTransform(spec)(images[0], images[0], masks[0, ..., 0])
        assert transformed[0].shape == images[0].shape and transformed[0].dtype == np.uint8
        assert transformed[1].shape == images[0].shape and transformed[1].dtype == np.uint8
        assert transformed[2].shape == masks[0, ..., 0].shape and transformed[2].dtype == np.uint8

    y = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.arange(6) % 3
    baseline = _prediction(y, 0.1, 1.0)
    predictions = {
        spec.tag: _prediction(y, 0.1 + index / 100.0, 1.0 + index / 100.0)
        for index, spec in enumerate(DEFAULT_PERTURBATIONS)
    }
    table, summary = score_perturbation_grid(baseline, predictions, y, labels, labels)
    assert len(table) == 24
    assert summary["n_conditions_expected"] == summary["n_conditions_completed"] == 23
    incomplete = dict(predictions)
    incomplete.pop(DEFAULT_PERTURBATIONS[-1].tag)
    with pytest.raises(ValueError, match="incomplete perturbation grid"):
        score_perturbation_grid(baseline, incomplete, y, labels, labels)
    with pytest.raises(ValueError, match="integer-valued"):
        score_perturbation_grid(baseline, predictions, y, labels + 0.25, labels)


def test_perturbation_transform_matches_dataset_image_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    color = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    silhouette = np.arange(12 * 12, dtype=np.uint8).reshape(12, 12)

    def fake_read(_: object, grayscale: bool) -> np.ndarray:
        return silhouette.copy() if grayscale else color.copy()

    monkeypatch.setattr(dataset_module, "_read_image", fake_read)
    rotate = next(spec for spec in DEFAULT_PERTURBATIONS if spec.kind == "rotate")
    dataset = ClinicalPhotoDataset(
        pd.DataFrame({"case_id": ["case"], "age": [18], "sex": ["F"]}),
        ".",
        image_size=12,
        input_transform=PerturbationTransform(rotate),
    )
    sample = dataset[0]
    assert tuple(sample["frontal"].shape) == (3, 12, 12)
    assert tuple(sample["profile"].shape) == (4, 12, 12)


def test_confound_analysis_marks_single_domain_not_applicable() -> None:
    rng = np.random.default_rng(3)
    groups = np.repeat(("batch_a", "batch_b"), 30)
    train = rng.normal(size=(60, 4))
    train[:, 0] += (groups == "batch_b") * 3.0
    inference = rng.normal(size=(30, 4))
    result = analyze_confound_probes(
        train,
        groups,
        inference,
        np.repeat("single_domain", 30),
        train_demographics=train[:, :2],
        inference_phenotype_labels=np.tile(("I", "II", "III"), 10),
        n_splits=3,
        seed=11,
    )
    assert result["training_acquisition_probe"]["status"] == "estimated"
    assert result["inference_domain_probe"]["status"] == "not_applicable"
    assert "does not mean" in result["inference_domain_probe"]["interpretation"]
    assert result["method"]["domain_adversarial_training"] is False
    assert result["method"]["domain_adversarial_statement"].startswith("No gradient-reversal")


def _histories() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1):
        n_train = 100 * index
        folds = []
        for fold in range(3):
            base_mae = 0.8 + 5.0 * n_train**-0.5 + fold * 0.003
            base_bal = 0.82 - 2.0 * n_train**-0.5 - fold * 0.002
            folds.append({
                "fold": fold,
                "n_train": n_train + fold,
                "epochs": [
                    {
                        "epoch": epoch,
                        "mae_mean": base_mae + abs(epoch - 1) * 0.02,
                        "balanced_accuracy_sagittal": base_bal - abs(epoch - 2) * 0.005,
                        "balanced_accuracy_vertical": base_bal - abs(epoch - 2) * 0.006,
                    }
                    for epoch in range(3)
                ],
            })
        result[f"arm_{index}"] = {
            "fraction": fraction,
            "selection_criterion": "mae_mean",
            "folds": folds,
        }
    return result


def test_arm_comparison_and_learning_curve_use_regression_selected_epoch() -> None:
    histories = _histories()
    comparison = compare_arm_histories(histories)
    assert comparison["arms"]["arm_1"]["selected_epoch_mean"] == 1.0
    selected_bal = comparison["arms"]["arm_1"]["metrics"]["balanced_accuracy_sagittal"]["mean"]
    best_bal = max(
        epoch["balanced_accuracy_sagittal"]
        for epoch in histories["arm_1"]["folds"][0]["epochs"]
    )
    assert selected_bal < best_bal

    classification = {
        "classification": {
            "fraction": 1.0,
            "selection_criterion": "balanced_accuracy_mean",
            "folds": [
                {
                    "fold": fold,
                    "n_train": 100,
                    "epochs": [
                        {
                            "epoch": epoch,
                            "mae_mean": None,
                            "balanced_accuracy_sagittal": 0.5 + 0.1 * epoch,
                            "balanced_accuracy_vertical": 0.5 + 0.1 * epoch,
                        }
                        for epoch in range(3)
                    ],
                }
                for fold in range(3)
            ],
        }
    }
    classification_result = compare_arm_histories(classification)["arms"]["classification"]
    assert classification_result["selected_epoch_mean"] == 2.0
    assert classification_result["metrics"]["mae_mean"] is None
    assert classification_result["metrics"]["balanced_accuracy_mean"]["mean"] == pytest.approx(0.7)
    classification["classification"]["folds"][0]["epochs"][0]["mae_mean"] = 1.0
    with pytest.raises(ValueError, match="mixes epochs"):
        compare_arm_histories(classification)

    curve = aggregate_learning_curve(histories, n_boot=30, seed=5, extrapolation_factors=(2.0,))
    assert curve["arms"] == ["arm_1", "arm_2", "arm_3", "arm_4"]
    assert curve["metrics"]["mae_mean"]["n_boot_valid"] > 0
    assert "2x" in curve["metrics"]["mae_mean"]["extrapolation"]
    fit = fit_power_curve((100, 200, 300, 400), (1.3, 1.15, 1.08, 1.05), decreasing=True)
    assert fit is not None and fit["a"] > 0 and fit["b"] > 0


def test_publication_learning_artifacts_preserve_both_sd_conventions() -> None:
    histories = _histories()
    comparison = compare_arm_histories(histories)
    selected_mae = np.asarray([
        fold["epochs"][1]["mae_mean"] for fold in histories["arm_1"]["folds"]
    ])
    metric = comparison["arms"]["arm_1"]["metrics"]["mae_mean"]
    assert metric["population_sd"] == pytest.approx(selected_mae.std(ddof=0))
    assert metric["sample_sd"] == pytest.approx(selected_mae.std(ddof=1))

    summary = publication_learning_curve_summary(histories)
    first = summary["points"][0]
    assert list(first) == [
        "frac",
        "arm",
        "n_train",
        "mae",
        "mae_sd",
        "bal_sag",
        "bal_sag_sd",
        "bal_vert",
        "bal_vert_sd",
    ]
    assert first["n_train"] == 101
    assert first["mae_sd"] == pytest.approx(selected_mae.std(ddof=0))

    fitted = publication_learning_curve_fit(
        histories,
        n_boot=30,
        seed=5,
        extrapolation_factors=(2.0,),
    )
    exploratory = aggregate_learning_curve(
        histories,
        n_boot=30,
        seed=5,
        extrapolation_factors=(2.0,),
    )
    assert fitted["n"] == exploratory["n_train"]
    assert exploratory["metrics"]["mae_mean"]["observed_sample_sd"][0] == pytest.approx(
        selected_mae.std(ddof=1)
    )
    assert fitted["metrics"]["mae"]["observed_sd"] == (
        exploratory["metrics"]["mae_mean"]["observed_sample_sd"]
    )
    assert set(fitted["metrics"]["mae"]["extrapolation"]) == {"2"}


def test_publication_compare_arms_artifact_uses_population_sd() -> None:
    histories = _histories()
    metadata = {
        name: {"desc": f"Description {name}", "arch": {"regression": True}}
        for name in histories
    }
    artifact = publication_compare_arms_summary(histories, metadata)
    assert artifact["arms"] == list(histories)
    first = artifact["rows"][0]
    selected_mae = np.asarray([
        fold["epochs"][1]["mae_mean"] for fold in histories["arm_1"]["folds"]
    ])
    assert first["criterion"] == "mae_mean"
    assert first["cls"]["mae"] == pytest.approx(
        [selected_mae.mean(), selected_mae.std(ddof=0)]
    )
    assert first["cls"]["epoch"] == 1.0
    assert first["cls_oracle"]["epoch"] == 2.0


def test_publication_artifacts_apply_the_declared_arm_crosswalk() -> None:
    assert PUBLICATION_ARM_LABELS == {
        "classification_rgb": "c1",
        "classification_shape": "c2",
        "multitask": "c3",
        "main": "c4b",
        "learning_10": "learning_curve_10pct",
        "learning_25": "learning_curve_25pct",
        "learning_50": "learning_curve_50pct",
        "learning_75": "learning_curve_75pct",
    }
    histories = _histories()
    histories = {
        "learning_10": histories["arm_1"],
        "learning_25": histories["arm_2"],
        "learning_50": histories["arm_3"],
        "main": histories["arm_4"],
    }
    metadata = {
        name: {"desc": "", "arch": {"regression": True}}
        for name in histories
    }

    curve = publication_learning_curve_summary(histories)
    comparison = publication_compare_arms_summary(histories, metadata)

    expected = ["learning_curve_10pct", "learning_curve_25pct", "learning_curve_50pct", "c4b"]
    assert [point["arm"] for point in curve["points"]] == expected
    assert comparison["arms"] == expected
    assert [row["arm"] for row in comparison["rows"]] == expected

    with pytest.raises(ValueError, match="unique"):
        publication_learning_curve_summary(histories, arm_labels={"main": "learning_curve_10pct"})


def test_geometry_baseline_returns_only_aggregate_metrics() -> None:
    rng = np.random.default_rng(9)
    n = 72
    features = rng.normal(size=(n, 6))
    features[0, 0] = np.nan
    targets = np.column_stack((features[:, 1] + rng.normal(0, 0.1, n), features[:, 2] * 2.0))
    sagittal = np.asarray([f"s{index % 3}" for index in range(n)])
    vertical = np.asarray([f"v{(index // 3) % 3}" for index in range(n)])
    folds = np.asarray([f"fold_{(index // 9) % 3}" for index in range(n)])
    result = evaluate_geometry_baseline(
        features,
        targets,
        sagittal,
        vertical,
        folds,
        target_names=("first", "second"),
        n_estimators=16,
        min_samples_leaf=2,
        seed=4,
    )
    assert result["n_cases"] == n
    assert result["method"]["estimator"].startswith("scikit-learn ExtraTrees")
    assert set(result["regression"]) == {"first", "second"}
    assert "predictions" not in result
    assert np.asarray(result["classification"]["sagittal"]["confusion"]).sum() == n
