"""Five-fold model training with declared checkpoint selection."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .dataset import (
    AugmentationConfig,
    ClinicalPhotoDataset,
    SAGITTAL_TO_INDEX,
    VERTICAL_TO_INDEX,
    TargetScaler,
    inverse_frequency_weights,
    load_manifest,
    select_fold,
)
from .model import FaceToCephalometryModel, ModelConfig, ModelOutput
from .workspace import GENERATED_ROOT, output_path as guarded_output_path


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    fold_count: int = 5
    image_size: int = 384
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_workers: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 30
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 5.0
    focal_gamma: float = 2.0
    mixed_precision: bool = True
    channels_last: bool = True
    subset_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.fold_count < 2 or self.image_size < 1 or self.batch_size < 1:
            raise ValueError("Training dimensions and fold count must be positive")
        if self.gradient_accumulation_steps < 1 or self.num_workers < 0:
            raise ValueError("Invalid accumulation or worker count")
        if self.epochs < 1 or self.warmup_epochs < 0 or self.early_stopping_patience < 1:
            raise ValueError("Invalid epoch configuration")
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs cannot exceed epochs")
        if (
            not np.isfinite((self.learning_rate, self.weight_decay, self.gradient_clip_norm, self.focal_gamma)).all()
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.gradient_clip_norm <= 0
            or self.focal_gamma < 0
        ):
            raise ValueError("Invalid optimization configuration")
        if not 0 < self.subset_fraction <= 1:
            raise ValueError("subset_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class FoldTrainingResult:
    fold: int
    best_epoch: int
    selection_metric: str
    validation_score: float
    validation_mae: float | None
    validation_balanced_accuracy: float
    training_cases: int
    validation_cases: int
    checkpoint_path: Path
    epoch_history: tuple["EpochValidationMetrics", ...] = ()


@dataclass(frozen=True)
class EpochValidationMetrics:
    epoch: int
    mae_mean: float | None
    balanced_accuracy_sagittal: float
    balanced_accuracy_vertical: float


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    log_probabilities = functional.log_softmax(logits, dim=1)
    selected = log_probabilities.gather(1, target[:, None]).squeeze(1)
    weights = class_weights.to(device=logits.device)[target]
    return (-(1.0 - selected.exp()).pow(gamma) * selected * weights).mean()


def gaussian_negative_log_likelihood(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))).mean()


def multitask_loss(
    output: ModelOutput,
    regression_target: torch.Tensor,
    sagittal_target: torch.Tensor,
    vertical_target: torch.Tensor,
    sagittal_weights: torch.Tensor,
    vertical_weights: torch.Tensor,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    sagittal = focal_cross_entropy(output.sagittal_logits, sagittal_target, sagittal_weights, focal_gamma)
    vertical = focal_cross_entropy(output.vertical_logits, vertical_target, vertical_weights, focal_gamma)
    loss = sagittal + vertical
    if output.regression_mean is not None:
        regression = (
            gaussian_negative_log_likelihood(
                output.regression_mean,
                output.regression_log_variance,
                regression_target,
            )
            if output.regression_log_variance is not None
            else functional.mse_loss(output.regression_mean, regression_target)
        )
        loss = loss + regression
    return loss


def learning_rate_at(step: int, total_steps: int, warmup_steps: int, peak: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _output_path(output_root: str | Path, fold: int) -> Path:
    root = Path(output_root)
    root = (GENERATED_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
    path = (root / "checkpoints" / f"fold_{fold}.pt").resolve()
    if path != root and root not in path.parents:
        raise ValueError("Checkpoint path must remain inside output_root")
    return guarded_output_path(path)


def _stratified_subset(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame.reset_index(drop=True)
    columns = ("sagittal", "vertical", "sex")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Subset selection requires columns: {missing}")
    strata = (
        frame["sagittal"].astype(str)
        + "|"
        + frame["vertical"].astype(str)
        + "|"
        + frame["sex"].astype(str)
    )
    samples = []
    for _, group in frame.groupby(strata, sort=True, observed=True):
        count = max(int(round(len(group) * fraction)), 1)
        samples.append(group.sample(n=count, replace=False, random_state=seed))
    return pd.concat(samples, ignore_index=True)


def _subset_seed(seed: int, fold: int) -> int:
    return seed + fold


def arm_training_history(
    arm_name: str,
    fraction: float,
    results: Sequence[FoldTrainingResult],
) -> dict[str, object]:
    if not arm_name:
        raise ValueError("arm_name must be non-empty")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if len(results) < 2:
        raise ValueError("at least two fold results are required")
    fold_ids = [result.fold for result in results]
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("fold results must have unique fold identifiers")
    metrics = {result.selection_metric for result in results}
    if len(metrics) != 1:
        raise ValueError("fold results must use one selection metric")
    selection_metric = metrics.pop()
    criteria = {"mae": "mae_mean", "balanced_accuracy": "balanced_accuracy_mean"}
    if selection_metric not in criteria:
        raise ValueError(f"unsupported selection metric: {selection_metric}")
    folds = []
    for result in sorted(results, key=lambda value: value.fold):
        if result.training_cases < 2 or not result.epoch_history:
            raise ValueError(f"fold {result.fold} has incomplete training history")
        folds.append(
            {
                "fold": result.fold,
                "n_train": result.training_cases,
                "epochs": [asdict(record) for record in result.epoch_history],
            }
        )
    return {
        arm_name: {
            "fraction": float(fraction),
            "selection_criterion": criteria[selection_metric],
            "folds": folds,
        }
    }


def _make_loader(
    dataset: ClinicalPhotoDataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=config.num_workers > 0,
    )


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frontal = batch["frontal"].to(device, non_blocking=True)
    profile = batch["profile"].to(device, non_blocking=True)
    if channels_last:
        frontal = frontal.contiguous(memory_format=torch.channels_last)
        profile = profile.contiguous(memory_format=torch.channels_last)
    return (
        frontal,
        profile,
        batch["metadata"].to(device, non_blocking=True),
        batch["regression_target"].to(device, non_blocking=True),
        batch["sagittal_target"].to(device, non_blocking=True),
        batch["vertical_target"].to(device, non_blocking=True),
    )


@torch.inference_mode()
def validation_metrics(
    model: FaceToCephalometryModel,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    *,
    mixed_precision: bool,
    channels_last: bool,
) -> dict[str, float | None]:
    model.eval()
    absolute_error = np.zeros(len(scaler.mean), dtype=np.float64)
    case_count = 0
    sagittal_predictions: list[np.ndarray] = []
    vertical_predictions: list[np.ndarray] = []
    sagittal_targets: list[np.ndarray] = []
    vertical_targets: list[np.ndarray] = []
    has_regression = False
    for batch in loader:
        frontal, profile, metadata, _, _, _ = _move_batch(batch, device, channels_last)
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            output = model(frontal, profile, metadata)
        sagittal_predictions.append(output.sagittal_logits.float().argmax(dim=1).cpu().numpy())
        vertical_predictions.append(output.vertical_logits.float().argmax(dim=1).cpu().numpy())
        sagittal_targets.append(batch["sagittal_target"].numpy())
        vertical_targets.append(batch["vertical_target"].numpy())
        if output.regression_mean is not None:
            has_regression = True
            prediction = output.regression_mean.float().cpu().numpy()
            raw_prediction = scaler.inverse_transform(prediction)
            raw_target = batch["regression_target_raw"].numpy().astype(np.float64)
            absolute_error += np.abs(raw_prediction - raw_target).sum(axis=0)
        case_count += len(batch["sagittal_target"])
    if case_count == 0:
        raise ValueError("Validation loader is empty")

    def balanced(predictions: list[np.ndarray], targets: list[np.ndarray]) -> float:
        predicted = np.concatenate(predictions)
        true = np.concatenate(targets)
        recalls = [
            float(np.mean(predicted[true == index] == index)) if np.any(true == index) else 0.0
            for index in range(3)
        ]
        return float(np.mean(recalls))

    sagittal_balanced = balanced(sagittal_predictions, sagittal_targets)
    vertical_balanced = balanced(vertical_predictions, vertical_targets)
    return {
        "mae": float((absolute_error / case_count).mean()) if has_regression else None,
        "sagittal_balanced_accuracy": sagittal_balanced,
        "vertical_balanced_accuracy": vertical_balanced,
        "balanced_accuracy": 0.5 * (sagittal_balanced + vertical_balanced),
    }


def validation_mae(
    model: FaceToCephalometryModel,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    *,
    mixed_precision: bool,
    channels_last: bool,
) -> float:
    value = validation_metrics(
        model,
        loader,
        scaler,
        device,
        mixed_precision=mixed_precision,
        channels_last=channels_last,
    )["mae"]
    if value is None:
        raise ValueError("The model has no regression output")
    return value


def train_fold(
    clinical_manifest_path: str | Path,
    split_manifest_path: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    fold: int,
    *,
    training_config: TrainingConfig = TrainingConfig(),
    model_config: ModelConfig = ModelConfig(),
    augmentation_config: AugmentationConfig = AugmentationConfig(),
    device: str | torch.device | None = None,
) -> FoldTrainingResult:
    if not 0 <= fold < training_config.fold_count:
        raise ValueError(f"fold must be between 0 and {training_config.fold_count - 1}")
    checkpoint_path = _output_path(output_root, fold)

    fold_seed = training_config.seed + 1000 * fold
    set_random_seed(fold_seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = training_config.mixed_precision and selected_device.type == "cuda"
    use_channels_last = training_config.channels_last and selected_device.type == "cuda"

    manifest = load_manifest(
        clinical_manifest_path,
        split_manifest_path,
        require_targets=True,
        availability_column="analyzed" if model_config.use_profile_sdf else "usable",
    )
    train_records, validation_records = select_fold(manifest, fold)
    train_records = _stratified_subset(
        train_records,
        training_config.subset_fraction,
        _subset_seed(training_config.seed, fold),
    )
    scaler = TargetScaler.fit(train_records)
    train_dataset = ClinicalPhotoDataset(
        train_records,
        image_root,
        target_scaler=scaler,
        augment=augmentation_config,
        image_size=training_config.image_size,
        use_profile_sdf=model_config.use_profile_sdf,
        input_mode=model_config.input_mode,
        augmentation_seed=training_config.seed + fold,
    )
    validation_dataset = ClinicalPhotoDataset(
        validation_records,
        image_root,
        target_scaler=scaler,
        image_size=training_config.image_size,
        use_profile_sdf=model_config.use_profile_sdf,
        input_mode=model_config.input_mode,
    )
    train_loader = _make_loader(train_dataset, training_config, shuffle=True)
    validation_loader = _make_loader(validation_dataset, training_config, shuffle=False)

    sagittal_labels = train_records["sagittal"].map(SAGITTAL_TO_INDEX)
    vertical_labels = train_records["vertical"].map(VERTICAL_TO_INDEX)
    if sagittal_labels.isna().any() or vertical_labels.isna().any():
        raise ValueError("Training data contain an unknown class label")
    sagittal_weights = inverse_frequency_weights(sagittal_labels.astype(int).to_numpy())
    vertical_weights = inverse_frequency_weights(vertical_labels.astype(int).to_numpy())

    model = FaceToCephalometryModel(model_config).to(selected_device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(len(train_loader) // training_config.gradient_accumulation_steps, 1)
    total_steps = steps_per_epoch * training_config.epochs
    warmup_steps = steps_per_epoch * training_config.warmup_epochs
    optimization_step = 0
    selection_metric = "balanced_accuracy" if model_config.regression_mode == "none" else "mae"
    best_score = -math.inf if selection_metric == "balanced_accuracy" else math.inf
    best_metrics: dict[str, float | None] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epoch_history: list[EpochValidationMetrics] = []

    for epoch in range(training_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            frontal, profile, metadata, regression_target, sagittal_target, vertical_target = _move_batch(
                batch, selected_device, use_channels_last
            )
            with torch.amp.autocast(device_type=selected_device.type, enabled=use_amp):
                output = model(frontal, profile, metadata)
                loss = multitask_loss(
                    output,
                    regression_target,
                    sagittal_target,
                    vertical_target,
                    sagittal_weights,
                    vertical_weights,
                    training_config.focal_gamma,
                ) / training_config.gradient_accumulation_steps
            gradient_scaler.scale(loss).backward()
            should_step = (batch_index + 1) % training_config.gradient_accumulation_steps == 0
            if should_step:
                rate = learning_rate_at(
                    optimization_step,
                    total_steps,
                    warmup_steps,
                    training_config.learning_rate,
                )
                for group in optimizer.param_groups:
                    group["lr"] = rate
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimization_step += 1

        metrics = validation_metrics(
            model,
            validation_loader,
            scaler,
            selected_device,
            mixed_precision=use_amp,
            channels_last=use_channels_last,
        )
        epoch_history.append(
            EpochValidationMetrics(
                epoch=epoch,
                mae_mean=None if metrics["mae"] is None else float(metrics["mae"]),
                balanced_accuracy_sagittal=float(metrics["sagittal_balanced_accuracy"]),
                balanced_accuracy_vertical=float(metrics["vertical_balanced_accuracy"]),
            )
        )
        raw_score = metrics[selection_metric]
        if raw_score is None:
            raise RuntimeError(f"Validation metric {selection_metric} is unavailable")
        score = float(raw_score)
        improved = score > best_score if selection_metric == "balanced_accuracy" else score < best_score
        if improved:
            best_score = score
            best_metrics = metrics
            best_epoch = epoch
            stale_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.early_stopping_patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_model_config = asdict(model_config)
    serialized_model_config["pretrained_weights"] = None
    payload = {
        "model": best_state,
        "model_config": serialized_model_config,
        "training_config": asdict(training_config),
        "target_scaler": scaler.to_dict(),
        "fold": fold,
        "seed": fold_seed,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "validation_score": best_score,
        "validation_mae": best_metrics["mae"],
        "validation_balanced_accuracy": best_metrics["balanced_accuracy"],
    }
    with checkpoint_path.open("xb") as stream:
        torch.save(payload, stream)
    return FoldTrainingResult(
        fold=fold,
        best_epoch=best_epoch,
        selection_metric=selection_metric,
        validation_score=best_score,
        validation_mae=best_metrics["mae"],
        validation_balanced_accuracy=float(best_metrics["balanced_accuracy"]),
        training_cases=len(train_dataset),
        validation_cases=len(validation_dataset),
        checkpoint_path=checkpoint_path,
        epoch_history=tuple(epoch_history),
    )


def train_five_fold_ensemble(
    clinical_manifest_path: str | Path,
    split_manifest_path: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    *,
    training_config: TrainingConfig = TrainingConfig(),
    model_config: ModelConfig = ModelConfig(),
    augmentation_config: AugmentationConfig = AugmentationConfig(),
    device: str | torch.device | None = None,
) -> list[FoldTrainingResult]:
    for fold in range(training_config.fold_count):
        _output_path(output_root, fold)
    return [
        train_fold(
            clinical_manifest_path,
            split_manifest_path,
            image_root,
            output_root,
            fold,
            training_config=training_config,
            model_config=model_config,
            augmentation_config=augmentation_config,
            device=device,
        )
        for fold in range(training_config.fold_count)
    ]
