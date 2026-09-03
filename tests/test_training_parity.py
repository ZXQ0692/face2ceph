from pathlib import Path

import numpy as np
import pandas as pd
import torch

from face2ceph.dataset import AugmentationConfig, ClinicalPhotoDataset
from face2ceph.training import (
    EpochValidationMetrics,
    FoldTrainingResult,
    _stratified_subset,
    _subset_seed,
    arm_training_history,
    focal_cross_entropy,
)


def _records() -> pd.DataFrame:
    return pd.DataFrame({"case_id": ["case"], "age": [18.0], "sex": ["F"]})


def test_augmentation_rng_matches_the_declared_fold_stream(tmp_path) -> None:
    first = ClinicalPhotoDataset(
        _records(), tmp_path, augment=AugmentationConfig(), augmentation_seed=47
    )
    second = ClinicalPhotoDataset(
        _records(), tmp_path, augment=AugmentationConfig(), augmentation_seed=47
    )
    np.testing.assert_array_equal(
        first._random_generator().random(8), second._random_generator().random(8)
    )


def test_focal_weights_keep_reference_precision() -> None:
    logits = torch.tensor([[0.5, -0.5, 0.0]], dtype=torch.float16)
    target = torch.tensor([0])
    weights = torch.tensor([1.0003, 2.0, 3.0], dtype=torch.float32)
    loss = focal_cross_entropy(logits, target, weights)
    log_probabilities = torch.log_softmax(logits, dim=1)
    selected = log_probabilities[:, 0]
    expected = (-(1.0 - selected.exp()).pow(2.0) * selected * weights[target]).mean()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)


def test_learning_subset_uses_the_historical_fold_seed() -> None:
    records = pd.DataFrame(
        {
            "case_id": [f"case-{index}" for index in range(20)],
            "sex": ["F"] * 20,
            "sagittal": ["I"] * 20,
            "vertical": ["Normo"] * 20,
        }
    )
    seed = _subset_seed(42, 3)
    observed = _stratified_subset(records, 0.5, seed)
    expected = records.sample(n=10, replace=False, random_state=45).reset_index(drop=True)
    assert seed == 45
    assert observed["case_id"].tolist() == expected["case_id"].tolist()


def test_learning_subset_preserves_the_historical_stratum_order() -> None:
    groups = (
        ("III", "Hypo", "F"),
        ("I", "Normo", "M"),
        ("II", "Hyper", "F"),
        ("I", "Hypo", "M"),
    )
    records = pd.DataFrame(
        [
            {"case_id": f"g{group}-{index}", "sagittal": sagittal, "vertical": vertical, "sex": sex}
            for group, (sagittal, vertical, sex) in enumerate(groups)
            for index in range(5)
        ]
    )
    observed = _stratified_subset(records, 0.4, 45)
    assert observed["case_id"].tolist() == [
        "g0-1",
        "g0-4",
        "g2-1",
        "g2-4",
        "g3-1",
        "g3-4",
        "g1-1",
        "g1-4",
    ]


def _fold_result(fold: int, selection_metric: str = "mae") -> FoldTrainingResult:
    return FoldTrainingResult(
        fold=fold,
        best_epoch=1,
        selection_metric=selection_metric,
        validation_score=1.5,
        validation_mae=1.5 if selection_metric == "mae" else None,
        validation_balanced_accuracy=0.75,
        training_cases=100 + fold,
        validation_cases=20,
        checkpoint_path=Path(f"fold_{fold}.pt"),
        epoch_history=(
            EpochValidationMetrics(fold, 1.7, 0.7, 0.8),
            EpochValidationMetrics(fold + 1, 1.5, 0.72, 0.81),
        ),
    )


def test_training_history_contains_only_fold_level_validation_aggregates() -> None:
    payload = arm_training_history("learning_25", 0.25, [_fold_result(1), _fold_result(0)])
    arm = payload["learning_25"]
    assert set(arm) == {"fraction", "selection_criterion", "folds"}
    assert arm["fraction"] == 0.25
    assert arm["selection_criterion"] == "mae_mean"
    assert [fold["fold"] for fold in arm["folds"]] == [0, 1]
    assert set(arm["folds"][0]) == {"fold", "n_train", "epochs"}
    assert set(arm["folds"][0]["epochs"][0]) == {
        "epoch",
        "mae_mean",
        "balanced_accuracy_sagittal",
        "balanced_accuracy_vertical",
    }
