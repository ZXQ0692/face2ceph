"""Ensemble inference and controlled prediction archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ClinicalPhotoDataset, ImageTransform, MEASUREMENT_NAMES, TargetScaler, load_manifest
from .model import FaceToCephalometryModel, ModelConfig
from .workspace import GENERATED_ROOT, output_path as guarded_output_path


@dataclass(frozen=True)
class InferenceConfig:
    fold_count: int = 5
    image_size: int = 384
    batch_size: int = 16
    num_workers: int = 8
    mixed_precision: bool = True
    channels_last: bool = True

    def __post_init__(self) -> None:
        if self.fold_count < 1 or self.image_size < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid inference configuration")


@dataclass(frozen=True)
class PredictionSet:
    case_id: np.ndarray
    mu: np.ndarray | None
    sigma: np.ndarray | None
    prob_sag: np.ndarray
    prob_vert: np.ndarray
    variance_aleatoric: np.ndarray | None
    variance_epistemic: np.ndarray | None
    y_raw: np.ndarray | None = None
    features: np.ndarray | None = None

    @property
    def sagittal_class_index(self) -> np.ndarray:
        return self.prob_sag.argmax(axis=1)

    @property
    def vertical_class_index(self) -> np.ndarray:
        return self.prob_vert.argmax(axis=1)


@dataclass(frozen=True)
class _CheckpointMember:
    envelope: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    target_scaler: TargetScaler
    fold: int
    seed: int
    model_state: dict[str, torch.Tensor]


_HISTORICAL_PREFIXES = (
    ("backbone_f.", "frontal_backbone."),
    ("backbone_p.", "profile_backbone."),
    ("film.net.", "metadata_modulation.network."),
    ("head_reg.mu.", "regression_head.mean."),
    ("head_reg.logvar.", "regression_head.log_variance."),
    ("head_sag.", "sagittal_head."),
    ("head_vert.", "vertical_head."),
)

_INFERENCE_RUNTIME_TRAINING_FIELDS = frozenset(
    {"batch_size", "num_workers", "mixed_precision", "channels_last"}
)


def fold_checkpoint_paths(checkpoint_directory: str | Path, fold_count: int = 5) -> list[Path]:
    directory = Path(checkpoint_directory)
    native = [directory / f"fold_{fold}.pt" for fold in range(fold_count)]
    historical = [directory / f"fold{fold}.pt" for fold in range(fold_count)]
    native_present = [path.is_file() for path in native]
    historical_present = [path.is_file() for path in historical]
    if any(native_present) and any(historical_present):
        raise ValueError("Checkpoint directory mixes native and historical fold naming")
    if all(native_present):
        return native
    if all(historical_present):
        if fold_count != 5:
            raise ValueError("Historical c4b checkpoints require exactly five folds")
        return historical
    selected = native if any(native_present) else historical
    missing = [path.name for path in selected if not path.is_file()]
    if not any(native_present) and not any(historical_present):
        missing = [f"fold_{fold}.pt or fold{fold}.pt" for fold in range(fold_count)]
    raise FileNotFoundError(f"The ensemble requires all fold checkpoints; missing: {missing}")


def _checkpoint_convention(paths: Sequence[Path], fold_count: int) -> str:
    names = {path.name for path in paths}
    native = {f"fold_{fold}.pt" for fold in range(fold_count)}
    historical = {f"fold{fold}.pt" for fold in range(fold_count)}
    if names == native:
        return "native"
    if names == historical:
        if fold_count != 5:
            raise ValueError("Historical c4b checkpoints require exactly five folds")
        return "historical"
    raise ValueError("Checkpoint paths must use one complete native or historical fold convention")


def _checkpoint_envelope(payload: Mapping[str, Any]) -> str:
    native_markers = {"model_config", "training_config", "target_scaler"}
    historical_markers = {"config", "metric", "scaler"}
    native_seen = native_markers.intersection(payload)
    historical_seen = historical_markers.intersection(payload)
    if native_seen and historical_seen:
        raise ValueError("Checkpoint envelope mixes native and historical fields")
    if native_seen:
        missing = native_markers.difference(payload)
        if missing:
            raise ValueError(f"Native checkpoint envelope is incomplete: {sorted(missing)}")
        return "native"
    if historical_seen:
        missing = historical_markers.difference(payload)
        if missing:
            raise ValueError(f"Historical checkpoint envelope is incomplete: {sorted(missing)}")
        return "historical"
    raise ValueError("Checkpoint envelope is neither native nor supported historical c4b")


def _validated_model_signature(values: Mapping[str, Any]) -> dict[str, Any]:
    signature = dict(values)
    expected_fields = set(asdict(ModelConfig()))
    if set(signature) != expected_fields:
        raise ValueError("Checkpoint model configuration fields do not match the release contract")
    return asdict(ModelConfig(**signature))


def _validated_training_signature(values: Mapping[str, Any]) -> dict[str, Any]:
    from .training import TrainingConfig

    signature = dict(values)
    expected_fields = set(asdict(TrainingConfig()))
    if set(signature) != expected_fields:
        raise ValueError("Checkpoint training configuration fields do not match the release contract")
    return asdict(TrainingConfig(**signature))


def _training_compatibility_signature(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in values.items()
        if name not in _INFERENCE_RUNTIME_TRAINING_FIELDS
    }


def _canonical_c4b_signatures() -> tuple[dict[str, Any], dict[str, Any]]:
    from .training import TrainingConfig

    return asdict(ModelConfig()), asdict(TrainingConfig())


def _validated_scaler(values: Mapping[str, Any]) -> TargetScaler:
    if set(values) != {"mean", "std"}:
        raise ValueError("Checkpoint target scaler must contain only mean and std")
    return TargetScaler.from_dict(values)


def _validated_state(values: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = dict(values)
    if not state or any(not isinstance(name, str) for name in state):
        raise ValueError("Checkpoint model state must be a nonempty string-keyed mapping")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint model state values must be tensors")
    return state


def _translate_historical_state_dict(values: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    source = _validated_state(values)
    translated: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        target = name if name.startswith("neck.") else None
        if target is None:
            for historical, native in _HISTORICAL_PREFIXES:
                if name.startswith(historical):
                    target = native + name[len(historical) :]
                    break
        if target is None:
            raise ValueError(f"Unsupported historical model-state key: {name}")
        if target in translated:
            raise ValueError(f"Historical model-state translation collides at {target}")
        translated[target] = value
    return translated


def _state_signature(state: Mapping[str, torch.Tensor]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(sorted((name, tuple(value.shape), str(value.dtype)) for name, value in state.items()))


def _integer_field(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Checkpoint {field} must be an integer")
    return int(value)


def _normalize_checkpoint(
    payload: Mapping[str, Any],
    *,
    convention: str,
    expected_fold: int,
    expected_model_config: Mapping[str, Any] | ModelConfig | None,
    expected_training_config: Mapping[str, Any] | None,
    expected_arm: str | None,
) -> _CheckpointMember:
    envelope = _checkpoint_envelope(payload)
    if envelope != convention:
        raise ValueError("Checkpoint filename convention and envelope format disagree")

    declared_model = (
        _validated_model_signature(
            asdict(expected_model_config)
            if isinstance(expected_model_config, ModelConfig)
            else expected_model_config
        )
        if expected_model_config is not None
        else None
    )
    declared_training = (
        _validated_training_signature(expected_training_config)
        if expected_training_config is not None
        else None
    )
    if envelope == "native":
        model_config = _validated_model_signature(payload["model_config"])
        training_config = _validated_training_signature(payload["training_config"])
        if declared_model is not None and model_config != declared_model:
            raise ValueError("Native checkpoint model configuration differs from the declared release arm")
        if (
            declared_training is not None
            and _training_compatibility_signature(training_config)
            != _training_compatibility_signature(declared_training)
        ):
            raise ValueError("Native checkpoint training configuration differs from the declared release arm")
        scaler = _validated_scaler(payload["target_scaler"])
        state = _validated_state(payload.get("model", {}))
    else:
        if payload.get("config") != "c4b":
            raise ValueError("Only the historical c4b checkpoint envelope is supported")
        if expected_arm not in {None, "main", "c4b"}:
            raise ValueError("Historical c4b checkpoints require the declared main release arm")
        canonical_model, canonical_training = _canonical_c4b_signatures()
        if declared_model is not None and declared_model != canonical_model:
            raise ValueError("Declared model configuration is not compatible with historical c4b")
        if (
            declared_training is not None
            and _training_compatibility_signature(declared_training)
            != _training_compatibility_signature(canonical_training)
        ):
            raise ValueError("Declared training configuration is not compatible with historical c4b")
        if payload.get("metric") != "mae_mean":
            raise ValueError("Historical c4b checkpoint selection metric must be mae_mean")
        epoch = _integer_field(payload, "epoch")
        value = payload.get("value")
        if not 0 <= epoch < int(canonical_training["epochs"]):
            raise ValueError("Historical c4b checkpoint epoch is outside the declared training range")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise ValueError("Historical c4b checkpoint validation value must be finite")
        model_config = canonical_model
        training_config = canonical_training
        scaler = _validated_scaler(payload["scaler"])
        state = _translate_historical_state_dict(payload.get("model", {}))

    fold = _integer_field(payload, "fold")
    seed = _integer_field(payload, "seed")
    if fold != expected_fold:
        raise ValueError("Checkpoint fold field does not match its filename")
    fold_count = int(training_config["fold_count"])
    if envelope == "historical" and fold_count != 5:
        raise ValueError("Historical c4b checkpoints require exactly five folds")
    if fold not in range(fold_count):
        raise ValueError("Checkpoint fold is outside the declared fold range")
    expected_seed = int(training_config["seed"]) + 1000 * fold
    if seed != expected_seed:
        raise ValueError("Checkpoint seed does not match the declared fold seed")
    return _CheckpointMember(envelope, model_config, training_config, scaler, fold, seed, state)


def combine_member_predictions(
    member_means: np.ndarray,
    member_variances: np.ndarray,
    member_sagittal_probabilities: np.ndarray,
    member_vertical_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(member_means, dtype=np.float64)
    variances = np.asarray(member_variances, dtype=np.float64)
    sagittal = np.asarray(member_sagittal_probabilities, dtype=np.float64)
    vertical = np.asarray(member_vertical_probabilities, dtype=np.float64)
    if means.ndim != 3 or means.shape != variances.shape or means.shape[2] != len(MEASUREMENT_NAMES):
        raise ValueError("Member regression arrays must have shape K x N x 8")
    if sagittal.shape != (means.shape[0], means.shape[1], 3) or vertical.shape != sagittal.shape:
        raise ValueError("Member probability arrays must have shape K x N x 3")
    ensemble_mean = means.mean(axis=0)
    aleatoric = variances.mean(axis=0)
    epistemic = means.var(axis=0)
    sigma = np.sqrt(aleatoric + epistemic)
    return ensemble_mean, sigma, sagittal.mean(axis=0), vertical.mean(axis=0), aleatoric, epistemic


@torch.inference_mode()
def _predict_member(
    model: FaceToCephalometryModel,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool,
    channels_last: bool,
    return_features: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    means: list[torch.Tensor] = []
    log_variances: list[torch.Tensor] = []
    sagittal: list[torch.Tensor] = []
    vertical: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        frontal = batch["frontal"].to(device, non_blocking=True)
        profile = batch["profile"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        if channels_last:
            frontal = frontal.contiguous(memory_format=torch.channels_last)
            profile = profile.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            output = model(frontal, profile, metadata)
        if output.regression_mean is not None:
            means.append(output.regression_mean.float().cpu())
        if output.regression_log_variance is not None:
            log_variances.append(output.regression_log_variance.float().cpu())
        sagittal.append(output.sagittal_logits.float().softmax(dim=1).cpu())
        vertical.append(output.vertical_logits.float().softmax(dim=1).cpu())
        if return_features:
            features.append(output.features.float().cpu())
    if not sagittal:
        raise ValueError("Inference dataset is empty")
    feature_array = torch.cat(features).numpy() if features else None
    return (
        torch.cat(means).numpy() if means else None,
        torch.cat(log_variances).numpy() if log_variances else None,
        torch.cat(sagittal).numpy(),
        torch.cat(vertical).numpy(),
        feature_array,
    )


def predict_ensemble(
    clinical_manifest_path: str | Path,
    image_root: str | Path,
    checkpoint_paths: Sequence[str | Path],
    *,
    split_manifest_path: str | Path | None = None,
    split: str | None = None,
    inference_config: InferenceConfig = InferenceConfig(),
    device: str | torch.device | None = None,
    return_features: bool = False,
    input_transform: ImageTransform | None = None,
    expected_model_config: Mapping[str, Any] | ModelConfig | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
    expected_arm: str | None = None,
) -> PredictionSet:
    paths = [Path(path) for path in checkpoint_paths]
    if len(paths) != inference_config.fold_count or len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError(f"Exactly {inference_config.fold_count} distinct checkpoints are required")
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    convention = _checkpoint_convention(paths, inference_config.fold_count)

    first_payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    first_fold = int(paths[0].stem.removeprefix("fold_").removeprefix("fold"))
    first_checkpoint = _normalize_checkpoint(
        first_payload,
        convention=convention,
        expected_fold=first_fold,
        expected_model_config=expected_model_config,
        expected_training_config=expected_training_config,
        expected_arm=expected_arm,
    )
    model_signature = first_checkpoint.model_config
    training_signature = first_checkpoint.training_config
    state_signature = _state_signature(first_checkpoint.model_state)
    del first_checkpoint, first_payload
    if int(training_signature["fold_count"]) != inference_config.fold_count:
        raise ValueError("Inference fold_count does not match the checkpoints")
    if int(training_signature["image_size"]) != inference_config.image_size:
        raise ValueError("Inference image_size does not match the checkpoints")
    data_model_config = ModelConfig(**model_signature)
    manifest = load_manifest(
        clinical_manifest_path,
        split_manifest_path,
        require_targets=False,
        availability_column="analyzed" if data_model_config.use_profile_sdf else "usable",
    )
    if split is not None:
        if "split" not in manifest.columns:
            raise ValueError("A split selector requires a split manifest")
        manifest = manifest.loc[manifest["split"].eq(split)].reset_index(drop=True)
    dataset = ClinicalPhotoDataset(
        manifest,
        image_root,
        image_size=inference_config.image_size,
        use_profile_sdf=data_model_config.use_profile_sdf,
        input_mode=data_model_config.input_mode,
        input_transform=input_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=inference_config.batch_size,
        shuffle=False,
        num_workers=inference_config.num_workers,
        pin_memory=True,
        persistent_workers=inference_config.num_workers > 0,
    )
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = inference_config.mixed_precision and selected_device.type == "cuda"
    use_channels_last = inference_config.channels_last and selected_device.type == "cuda"

    member_means: list[np.ndarray] = []
    member_variances: list[np.ndarray] = []
    member_sagittal: list[np.ndarray] = []
    member_vertical: list[np.ndarray] = []
    member_features: list[np.ndarray] = []
    seen_folds: set[int] = set()

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        filename_fold = int(path.stem.removeprefix("fold_").removeprefix("fold"))
        checkpoint = _normalize_checkpoint(
            payload,
            convention=convention,
            expected_fold=filename_fold,
            expected_model_config=expected_model_config,
            expected_training_config=expected_training_config,
            expected_arm=expected_arm,
        )
        signature = checkpoint.model_config
        member_training_signature = checkpoint.training_config
        fold = checkpoint.fold
        if signature != model_signature:
            raise ValueError("All ensemble members must use the same model configuration")
        if member_training_signature != training_signature:
            raise ValueError("All ensemble members must use the same training configuration")
        if _state_signature(checkpoint.model_state) != state_signature:
            raise ValueError("All ensemble members must use the same model-state signature")
        if fold in seen_folds or not 0 <= fold < inference_config.fold_count:
            raise ValueError("Checkpoint fold identifiers must be unique and complete")
        seen_folds.add(fold)

        scaler = checkpoint.target_scaler
        model_config = replace(
            ModelConfig(**signature),
            pretrained=False,
            pretrained_weights=None,
        )
        model = FaceToCephalometryModel(model_config).to(selected_device)
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint.model_state, strict=True)
        mean_z, log_variance_z, sagittal, vertical, features = _predict_member(
            model,
            loader,
            selected_device,
            mixed_precision=use_amp,
            channels_last=use_channels_last,
            return_features=return_features,
        )
        if mean_z is not None:
            member_means.append(scaler.inverse_transform(mean_z))
        if log_variance_z is not None:
            member_variances.append(np.exp(log_variance_z) * np.square(scaler.std))
        member_sagittal.append(sagittal)
        member_vertical.append(vertical)
        if features is not None:
            member_features.append(features)
        del checkpoint, model, payload
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    if seen_folds != set(range(inference_config.fold_count)):
        raise ValueError("Checkpoint fold identifiers must cover every ensemble fold")
    prob_sag = np.stack(member_sagittal).mean(axis=0)
    prob_vert = np.stack(member_vertical).mean(axis=0)
    regression_mode = data_model_config.regression_mode
    if regression_mode == "heteroscedastic":
        if len(member_means) != inference_config.fold_count or len(member_variances) != inference_config.fold_count:
            raise ValueError("Heteroscedastic checkpoints must provide means and variances")
        mu, sigma, prob_sag, prob_vert, aleatoric, epistemic = combine_member_predictions(
            np.stack(member_means),
            np.stack(member_variances),
            np.stack(member_sagittal),
            np.stack(member_vertical),
        )
    elif regression_mode == "homoscedastic":
        if len(member_means) != inference_config.fold_count or member_variances:
            raise ValueError("Homoscedastic checkpoints must provide means without variances")
        stacked_means = np.stack(member_means)
        mu = stacked_means.mean(axis=0)
        epistemic = stacked_means.var(axis=0)
        sigma = aleatoric = None
    else:
        if member_means or member_variances:
            raise ValueError("Classification-only checkpoints cannot provide regression outputs")
        mu = sigma = aleatoric = epistemic = None
    y_raw = (
        manifest.loc[:, list(MEASUREMENT_NAMES)].to_numpy(dtype=np.float64)
        if set(MEASUREMENT_NAMES).issubset(manifest.columns)
        else None
    )
    averaged_features = np.stack(member_features).mean(axis=0) if member_features else None
    return PredictionSet(
        case_id=manifest["case_id"].astype(str).to_numpy(),
        mu=mu,
        sigma=sigma,
        prob_sag=prob_sag,
        prob_vert=prob_vert,
        variance_aleatoric=aleatoric,
        variance_epistemic=epistemic,
        y_raw=y_raw,
        features=averaged_features,
    )


def save_prediction_archive(
    predictions: PredictionSet,
    output_path: str | Path,
    output_root: str | Path,
) -> Path:
    root = Path(output_root)
    root = (GENERATED_ROOT / root).resolve() if not root.is_absolute() else root.resolve()
    path = Path(output_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path != root and root not in path.parents:
        raise ValueError("Prediction archive must remain inside output_root")
    path = guarded_output_path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Prediction archive must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "case_id": predictions.case_id.astype(str),
        "prob_sag": predictions.prob_sag,
        "prob_vert": predictions.prob_vert,
    }
    if predictions.mu is not None:
        arrays["mu"] = predictions.mu
    if predictions.sigma is not None:
        arrays["sigma"] = predictions.sigma
    if predictions.variance_aleatoric is not None:
        arrays["var_alea"] = predictions.variance_aleatoric
    if predictions.variance_epistemic is not None:
        arrays["var_epi"] = predictions.variance_epistemic
    if predictions.y_raw is not None:
        arrays["y_raw"] = predictions.y_raw
    if predictions.features is not None:
        features = np.asarray(predictions.features)
        if features.ndim != 2 or features.shape[0] != len(predictions.case_id) or not np.isfinite(features).all():
            raise ValueError("Features must be a finite case-by-feature array")
        arrays["features"] = features
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path
