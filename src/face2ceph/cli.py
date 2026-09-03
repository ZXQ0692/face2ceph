"""Command-line entry points for the publication workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .workspace import GENERATED_ROOT, RELEASE_ROOT, create_directory, input_path, output_path, write_json


CONFIG_ROOT = RELEASE_ROOT / "configs"
ARM_NAMES = tuple(path.stem for path in sorted((CONFIG_ROOT / "arms").glob("*.yaml")))
SAFE_CASE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _yaml(path: str | Path) -> dict[str, Any]:
    source = input_path(path, "file")
    with source.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {source.name}")
    return value


def _pipeline(path: str | Path, arm: str | None = None) -> dict[str, Any]:
    from .configuration import load

    source = input_path(path, "file")
    if arm is None:
        values = _yaml(source)
    else:
        values = load(CONFIG_ROOT / "arms" / f"{arm}.yaml", source)
    _validate_pipeline(values, arm)
    return values


def _keys(value: object, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{name} keys differ: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")


def _validate_pipeline(values: dict[str, Any], arm: str | None) -> None:
    top = {"seed", "split", "preprocessing", "model", "training", "augmentation", "conformal", "referral"}
    if arm is not None:
        top.add("name")
    _keys(values, top, "pipeline")
    if arm is not None and values["name"] != arm:
        raise ValueError("arm name does not match its filename")
    _keys(
        values["split"],
        {"test_ratio", "calibration_ratio", "folds", "strata", "minimum_stratum_size", "degradation_order"},
        "split",
    )
    preprocessing = values["preprocessing"]
    _keys(preprocessing, {"image_size", "jpeg_quality", "frontal", "profile", "quality", "segmentation"}, "preprocessing")
    _keys(preprocessing["frontal"], {"interpupillary_ratio", "eye_center_y"}, "preprocessing.frontal")
    _keys(preprocessing["profile"], {"face_height_ratio", "face_center_x"}, "preprocessing.profile")
    _keys(
        preprocessing["quality"],
        {"minimum_source_side", "maximum_roll_degrees", "minimum_detection_score"},
        "preprocessing.quality",
    )
    _keys(
        preprocessing["segmentation"],
        {"minimum_foreground_fraction", "maximum_foreground_fraction", "maximum_face_components"},
        "preprocessing.segmentation",
    )
    model_keys = {"image_backbone", "use_profile_sdf", "metadata_conditioning", "regression", "dropout"}
    if "input" in values["model"]:
        model_keys.add("input")
    _keys(values["model"], model_keys, "model")
    training_keys = {
        "batch_size",
        "gradient_accumulation",
        "workers",
        "channels_last",
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "maximum_epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "mixed_precision",
        "focal_gamma",
    }
    if "subset_fraction" in values["training"]:
        training_keys.add("subset_fraction")
    _keys(values["training"], training_keys, "training")
    _keys(
        values["augmentation"],
        {"rotation_degrees", "translation_fraction", "brightness_fraction", "contrast_fraction", "clahe_probability"},
        "augmentation",
    )
    _keys(values["conformal"], {"alpha"}, "conformal")
    _keys(values["referral"], {"rates"}, "referral")


def _asset_specs() -> dict[str, dict[str, str]]:
    values = _yaml(CONFIG_ROOT / "assets.yaml")
    required = {"filename", "url", "sha256"}
    for name, value in values.items():
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(f"invalid asset definition: {name}")
    return values


def _verified_asset(name: str, supplied: str | None = None) -> Path:
    from .assets import require, sha256

    specification = _asset_specs()[name]
    expected = specification["sha256"].lower()
    if supplied is None:
        return require(specification["filename"], expected)
    path = input_path(supplied, "file")
    if sha256(path) != expected:
        raise ValueError(f"checksum mismatch: {path.name}")
    return path


def _validate_case_ids(values: pd.Series | Sequence[object] | np.ndarray) -> np.ndarray:
    identifiers = np.asarray(values, dtype=object)
    if identifiers.ndim != 1 or not len(identifiers):
        raise ValueError("case_id must be a non-empty one-dimensional field")
    result: list[str] = []
    for value in identifiers:
        if value is None or pd.isna(value):
            raise ValueError("case_id must be complete")
        identifier = str(value)
        stem = identifier.split(".", 1)[0].upper()
        if (
            SAFE_CASE_ID.fullmatch(identifier) is None
            or stem in WINDOWS_RESERVED
            or not any(character.isalpha() for character in identifier)
        ):
            raise ValueError("case_id must be a stable opaque ASCII code containing a letter")
        result.append(identifier)
    if len(set(result)) != len(result):
        raise ValueError("case_id must be unique")
    return np.asarray(result, dtype=str)


def _read_cohort(path: str | Path, *, paths: bool = False) -> pd.DataFrame:
    from .preprocessing import SAGITTAL_CLASSES, TARGETS, VERTICAL_CLASSES, apply_thresholds, normalize_sex

    source = input_path(path, "file")
    frame = pd.read_csv(source, dtype={"case_id": "string"})
    required = {"case_id", "age", "sex", *TARGETS}
    if paths:
        required.update(("frontal_path", "profile_path"))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cohort is missing columns: {missing}")
    frame = frame.copy()
    frame["case_id"] = _validate_case_ids(frame["case_id"])
    frame["age"] = pd.to_numeric(frame["age"].astype(str).str.strip().str.removesuffix("+"), errors="raise")
    for column in TARGETS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = ["age", *TARGETS]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all() or (frame["age"] < 7).any():
        raise ValueError("ages and measurements must be finite and age must be at least seven")
    frame["sex"] = [normalize_sex(value) for value in frame["sex"]]
    sagittal, vertical = apply_thresholds(frame["ANB"], frame["SN_MP"], frame["age"], frame["sex"])
    if "sagittal" in frame and not np.array_equal(frame["sagittal"].astype(str).to_numpy(), sagittal):
        raise ValueError("sagittal labels do not match the declared threshold scheme")
    if "vertical" in frame and not np.array_equal(frame["vertical"].astype(str).to_numpy(), vertical):
        raise ValueError("vertical labels do not match the declared threshold scheme")
    frame["sagittal"] = sagittal
    frame["vertical"] = vertical
    if not set(frame["sagittal"]) <= set(SAGITTAL_CLASSES) or not set(frame["vertical"]) <= set(VERTICAL_CLASSES):
        raise ValueError("unsupported class label")
    return frame


def _boolean(values: pd.Series, name: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{name} must be complete")
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{name} must contain only boolean values")
    return parsed.to_numpy(dtype=bool)


def _partition_config(values: Mapping[str, Any]):
    from .partition import PartitionConfig

    split = values["split"]
    return PartitionConfig(
        test_fraction=float(split["test_ratio"]),
        calibration_fraction=float(split["calibration_ratio"]),
        folds=int(split["folds"]),
        seed=int(values["seed"]),
        strata=tuple(split["strata"]),
        minimum_stratum_size=int(split["minimum_stratum_size"]),
        degrade_order=tuple(split["degradation_order"]),
    )


def _read_partition(path: str | Path, cohort: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    from .partition import validate_partition

    frame = pd.read_csv(input_path(path, "file"), dtype={"case_id": "string"})
    if "case_id" not in frame:
        raise ValueError("partition is missing case_id")
    frame["case_id"] = _validate_case_ids(frame["case_id"])
    partition_config = _partition_config(config)
    validate_partition(frame, cohort["case_id"], partition_config)
    return frame


def _select_analysis_frame(
    cohort: pd.DataFrame,
    partition: pd.DataFrame,
    config: Mapping[str, Any],
    split: str,
) -> pd.DataFrame:
    availability = "analyzed" if bool(config["model"]["use_profile_sdf"]) else "usable"
    if availability not in cohort:
        raise ValueError(f"cohort is missing the canonical {availability} field")
    merged = cohort.drop(columns=[name for name in ("split", "fold") if name in cohort]).merge(
        partition[["case_id", "split", "fold"]], on="case_id", how="left", validate="one_to_one"
    )
    selected = _boolean(merged[availability], availability) & merged["split"].eq(split).to_numpy()
    result = merged.loc[selected].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"no analyzed cases are assigned to {split}")
    return result


def _analysis_frame(
    cohort_path: str | Path,
    partition_path: str | Path,
    config: Mapping[str, Any],
    split: str,
) -> pd.DataFrame:
    cohort = _read_cohort(cohort_path)
    partition = _read_partition(partition_path, cohort, config)
    return _select_analysis_frame(cohort, partition, config, split)


def _write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = output_path(path)
    if destination.suffix.lower() != ".csv":
        raise ValueError("CSV output must use the .csv extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n")
    return destination


def _write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = output_path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("array output must use the .npz extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def _write_image(path: Path, image: np.ndarray, extension: str, quality: int | None = None) -> None:
    import cv2

    parameters = [] if quality is None else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise ValueError(f"image encoding failed for {path.name}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def _load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    source = input_path(path, "file")
    with np.load(source, allow_pickle=False) as archive:
        required = {"case_id", "prob_sag", "prob_vert"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"prediction archive is missing arrays: {missing}")
        optional = {"mu", "sigma", "y_raw", "var_alea", "var_epi", "features"}
        selected = required | (optional & set(archive.files))
        arrays = {name: np.asarray(archive[name]).copy() for name in selected}
    case_id = _validate_case_ids(arrays["case_id"])
    arrays["case_id"] = case_id
    count = len(case_id)
    for name in ("prob_sag", "prob_vert"):
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != (count, 3) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must be a finite N x 3 probability array")
        if not np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError(f"{name} rows must sum to one")
        arrays[name] = value
    for name in ("mu", "sigma", "y_raw", "var_alea", "var_epi"):
        if name in arrays:
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.shape != (count, 8) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite N x 8 array")
            arrays[name] = value
    for name in ("sigma", "var_alea", "var_epi"):
        if name in arrays and (arrays[name] < 0).any():
            raise ValueError(f"{name} cannot be negative")
    if "sigma" in arrays and "mu" not in arrays:
        raise ValueError("sigma requires regression means")
    if ("var_alea" in arrays) != ("var_epi" in arrays):
        raise ValueError("aleatoric and epistemic variances must be provided together")
    if "var_alea" in arrays:
        if "sigma" not in arrays or not np.allclose(
            np.square(arrays["sigma"]), arrays["var_alea"] + arrays["var_epi"], atol=1e-5, rtol=1e-5
        ):
            raise ValueError("variance components must sum to sigma squared")
    if "features" in arrays:
        features = np.asarray(arrays["features"], dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != count or not np.isfinite(features).all():
            raise ValueError("features must be a finite N x D array")
        arrays["features"] = features
    return arrays


def _require_prediction_arrays(arrays: Mapping[str, np.ndarray], names: Sequence[str]) -> None:
    missing = sorted(set(names).difference(arrays))
    if missing:
        raise ValueError(f"prediction archive is missing arrays: {missing}")


def _aligned_predictions(
    predictions_path: str | Path,
    cohort_path: str | Path,
    partition_path: str | Path,
    config: Mapping[str, Any],
    split: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    arrays = _load_predictions(predictions_path)
    expected = _analysis_frame(cohort_path, partition_path, config, split)
    return _align_prediction_arrays(arrays, expected, split)


def _align_prediction_arrays(
    arrays: dict[str, np.ndarray],
    expected: pd.DataFrame,
    split: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    from .targets import TARGETS

    identifiers = arrays["case_id"].tolist()
    if len(identifiers) != len(expected) or set(identifiers) != set(expected["case_id"]):
        raise ValueError(f"predictions do not match analyzed cases in {split}")
    aligned = expected.set_index("case_id", drop=False).loc[identifiers].reset_index(drop=True)
    truth = aligned.loc[:, list(TARGETS)].to_numpy(dtype=np.float64)
    if "y_raw" in arrays and not np.allclose(arrays["y_raw"], truth, rtol=0, atol=1e-5):
        raise ValueError("prediction truth does not match the authorized cohort")
    return arrays, aligned


def _cmd_assets(args: argparse.Namespace) -> int:
    from .assets import fetch

    specifications = _asset_specs()
    selected = args.names or list(specifications)
    for name in selected:
        value = specifications[name]
        fetch(value["url"], value["filename"], value["sha256"])
    print(f"{len(selected)} assets ready.")
    return 0


def _source_photo(value: object, root: Path) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def _decode_photo(path: Path | None) -> np.ndarray | None:
    import cv2

    if path is None or not path.is_file():
        return None
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocessing import (
        TARGETS,
        create_face_landmarker,
        create_profile_segmenter,
        preprocess_frontal,
        preprocess_profile,
    )

    config = _pipeline(args.config)
    cohort_source = input_path(args.cohort, "file")
    cohort = _read_cohort(cohort_source, paths=True)
    landmarker_path = _verified_asset("face_landmarker", args.landmarker_asset)
    segmenter_path = _verified_asset("profile_segmenter", args.segmenter_asset)
    preprocessing = config["preprocessing"]
    frontal_config = preprocessing["frontal"]
    profile_config = preprocessing["profile"]
    quality = preprocessing["quality"]
    segmentation = preprocessing["segmentation"]
    landmarker = create_face_landmarker(
        landmarker_path,
        minimum_detection_score=float(quality["minimum_detection_score"]),
    )
    segmenter = None
    try:
        segmenter = create_profile_segmenter(segmenter_path)
        root = create_directory(args.output_dir)
        folders = {name: root / name for name in ("frontal", "profile", "profile_sdf")}
        for folder in folders.values():
            folder.mkdir()
        status: list[dict[str, object]] = []
        normalized_paths: list[tuple[str, str, str]] = []
        for _, row in cohort.iterrows():
            identifier = str(row["case_id"])
            frontal_source = _source_photo(row["frontal_path"], cohort_source.parent)
            profile_source = _source_photo(row["profile_path"], cohort_source.parent)
            frontal_image = _decode_photo(frontal_source)
            profile_image = _decode_photo(profile_source)
            reasons: list[str] = []
            frontal = None
            profile = None
            if frontal_image is None:
                reasons.append("frontal_unreadable")
            else:
                frontal = preprocess_frontal(
                    frontal_image,
                    landmarker,
                    output_size=int(preprocessing["image_size"]),
                    ipd_ratio=float(frontal_config["interpupillary_ratio"]),
                    eye_y=float(frontal_config["eye_center_y"]),
                    min_source_pixels=int(quality["minimum_source_side"]),
                    max_roll_deg=float(quality["maximum_roll_degrees"]),
                )
                reasons.extend(f"frontal_{reason}" for reason in frontal.qc.reasons)
            if profile_image is None:
                reasons.append("profile_unreadable")
            else:
                profile = preprocess_profile(
                    profile_image,
                    segmenter,
                    output_size=int(preprocessing["image_size"]),
                    face_height_ratio=float(profile_config["face_height_ratio"]),
                    face_x=float(profile_config["face_center_x"]),
                    min_source_pixels=int(quality["minimum_source_side"]),
                    jpeg_quality=int(preprocessing["jpeg_quality"]),
                    minimum_foreground=float(segmentation["minimum_foreground_fraction"]),
                    maximum_foreground=float(segmentation["maximum_foreground_fraction"]),
                    maximum_face_components=int(segmentation["maximum_face_components"]),
                )
                reasons.extend(f"profile_{reason}" for reason in profile.qc.reasons)
            usable = bool(
                frontal is not None
                and frontal.image is not None
                and profile is not None
                and profile.image is not None
            )
            analyzed = bool(usable and profile is not None and profile.sdf is not None)
            if usable:
                frontal_relative = f"frontal/{identifier}.jpg"
                profile_relative = f"profile/{identifier}.jpg"
                _write_image(root / frontal_relative, frontal.image, ".jpg", int(preprocessing["jpeg_quality"]))
                _write_image(root / profile_relative, profile.image, ".jpg", int(preprocessing["jpeg_quality"]))
            else:
                frontal_relative = profile_relative = ""
            if analyzed:
                sdf_relative = f"profile_sdf/{identifier}.png"
                _write_image(root / sdf_relative, profile.sdf, ".png")
            else:
                sdf_relative = ""
            normalized_paths.append((frontal_relative, profile_relative, sdf_relative))
            status.append(
                {"case_id": identifier, "usable": usable, "analyzed": analyzed, "reasons": ";".join(reasons)}
            )
        columns = ["case_id", "age", "sex", *TARGETS, "sagittal", "vertical"]
        sanitized = cohort.loc[:, columns].copy()
        sanitized["usable"] = [value["usable"] for value in status]
        sanitized["analyzed"] = [value["analyzed"] for value in status]
        sanitized[["frontal_path", "profile_path", "profile_sdf_path"]] = pd.DataFrame(
            normalized_paths, index=sanitized.index
        )
        with (root / "cohort.csv").open("x", encoding="utf-8", newline="") as stream:
            sanitized.to_csv(stream, index=False, lineterminator="\n")
        with (root / "preprocessing_status.csv").open("x", encoding="utf-8", newline="") as stream:
            pd.DataFrame(status).to_csv(stream, index=False, lineterminator="\n")
    finally:
        landmarker.close()
        if segmenter is not None:
            segmenter.close()
    usable_count = int(sum(bool(value["usable"]) for value in status))
    analyzed_count = int(sum(bool(value["analyzed"]) for value in status))
    print(f"{usable_count} usable; {analyzed_count}/{len(status)} analyzed.")
    return 0


def _cmd_partition(args: argparse.Namespace) -> int:
    from .partition import make_partition
    from .targets import age_band

    config = _pipeline(args.config)
    cohort = _read_cohort(args.cohort)
    cohort["age_band"] = [age_band(float(value)) for value in cohort["age"]]
    result = make_partition(cohort, _partition_config(config))
    _write_csv(args.output, result.assignments)
    print(f"{len(result.assignments)} cases partitioned.")
    return 0


def _model_config(values: Mapping[str, Any], weights: Path | None):
    from .model import ModelConfig

    model = values["model"]
    return ModelConfig(
        backbone_name=str(model["image_backbone"]),
        pretrained=True,
        pretrained_weights=str(weights) if weights is not None else None,
        use_profile_sdf=bool(model["use_profile_sdf"]),
        metadata_conditioning=str(model["metadata_conditioning"]),
        regression_mode=str(model["regression"]),
        input_mode=str(model.get("input", "both")),
        dropout=float(model["dropout"]),
    )


def _training_config(values: Mapping[str, Any]):
    from .training import TrainingConfig

    training = values["training"]
    return TrainingConfig(
        seed=int(values["seed"]),
        fold_count=int(values["split"]["folds"]),
        image_size=int(values["preprocessing"]["image_size"]),
        batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation"]),
        num_workers=int(training["workers"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        epochs=int(training["maximum_epochs"]),
        warmup_epochs=int(training["warmup_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        focal_gamma=float(training["focal_gamma"]),
        mixed_precision=bool(training["mixed_precision"]),
        channels_last=bool(training["channels_last"]),
        subset_fraction=float(training.get("subset_fraction", 1.0)),
    )


def _augmentation_config(values: Mapping[str, Any]):
    from .dataset import AugmentationConfig

    augmentation = values["augmentation"]
    return AugmentationConfig(
        rotation_degrees=float(augmentation["rotation_degrees"]),
        translation_fraction=float(augmentation["translation_fraction"]),
        brightness_fraction=float(augmentation["brightness_fraction"]),
        contrast_fraction=float(augmentation["contrast_fraction"]),
        clahe_probability=float(augmentation["clahe_probability"]),
    )


def _cmd_train(args: argparse.Namespace) -> int:
    from .training import arm_training_history, train_five_fold_ensemble

    config = _pipeline(args.config, args.arm)
    cohort = _read_cohort(args.cohort)
    for availability in ("usable", "analyzed"):
        if availability not in cohort:
            raise ValueError(f"cohort is missing the canonical {availability} field")
        _boolean(cohort[availability], availability)
    _read_partition(args.partition, cohort, config)
    image_root = input_path(args.image_root, "dir")
    backbone = str(config["model"]["image_backbone"])
    candidates = [name for name, value in _asset_specs().items() if value.get("name") == backbone]
    if len(candidates) != 1:
        raise ValueError(f"no unique declared weights for backbone {backbone}")
    weights = _verified_asset(candidates[0], args.backbone_weights)
    destination = output_path(args.output_dir)
    results = train_five_fold_ensemble(
        input_path(args.cohort, "file"),
        input_path(args.partition, "file"),
        image_root,
        destination,
        training_config=_training_config(config),
        model_config=_model_config(config, weights),
        augmentation_config=_augmentation_config(config),
        device=args.device,
    )
    history = arm_training_history(
        str(args.arm),
        float(config["training"].get("subset_fraction", 1.0)),
        results,
    )
    write_json(destination / "validation_history.json", history)
    print(f"{len(results)} folds trained.")
    return 0


def _inference_config(values: Mapping[str, Any]):
    from .inference import InferenceConfig

    training = values["training"]
    return InferenceConfig(
        fold_count=int(values["split"]["folds"]),
        image_size=int(values["preprocessing"]["image_size"]),
        batch_size=int(training["batch_size"]),
        num_workers=int(training["workers"]),
        mixed_precision=bool(training["mixed_precision"]),
        channels_last=bool(training["channels_last"]),
    )


def _cmd_predict(args: argparse.Namespace) -> int:
    from .inference import fold_checkpoint_paths, predict_ensemble, save_prediction_archive

    input_transform = None
    perturbation = getattr(args, "perturbation", None)
    if perturbation:
        from .analyses import DEFAULT_PERTURBATIONS, PerturbationTransform

        registered = {spec.tag: spec for spec in DEFAULT_PERTURBATIONS}
        if perturbation not in registered:
            raise ValueError("unknown perturbation; expected one of " + ", ".join(sorted(registered)))
        input_transform = PerturbationTransform(registered[perturbation])

    config = _pipeline(args.config, args.arm)
    cohort = _read_cohort(args.cohort)
    for availability in ("usable", "analyzed"):
        if availability not in cohort:
            raise ValueError(f"cohort is missing the canonical {availability} field")
        _boolean(cohort[availability], availability)
    _read_partition(args.partition, cohort, config)
    checkpoints = fold_checkpoint_paths(
        input_path(args.checkpoints, "dir"), int(config["split"]["folds"])
    )
    destination = output_path(args.output)
    predictions = predict_ensemble(
        input_path(args.cohort, "file"),
        input_path(args.image_root, "dir"),
        checkpoints,
        split_manifest_path=input_path(args.partition, "file"),
        split=args.split,
        inference_config=_inference_config(config),
        device=args.device,
        return_features=bool(args.include_features),
        input_transform=input_transform,
        expected_model_config=_model_config(config, None),
        expected_training_config=asdict(_training_config(config)),
        expected_arm=str(config["name"]),
    )
    save_prediction_archive(predictions, destination, GENERATED_ROOT)
    print(f"{len(predictions.case_id)} cases predicted.")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import fit_split_conformal
    from .targets import TARGETS

    config = _pipeline(args.config)
    arrays, calibration_frame = _aligned_predictions(
        args.predictions, args.cohort, args.partition, config, "calibration"
    )
    _require_prediction_arrays(arrays, ("mu", "sigma"))
    calibration = fit_split_conformal(
        calibration_frame.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        arrays["mu"],
        arrays["sigma"],
        alpha=float(config["conformal"]["alpha"] if args.alpha is None else args.alpha),
        targets=TARGETS,
    )
    write_json(args.output, calibration.as_dict())
    print(f"{calibration.calibration_size} cases calibrated.")
    return 0


def _training_reference(
    cohort_path: str | Path,
    partition_path: str | Path,
    config: Mapping[str, Any],
):
    from .referral import fit_zscore_reference
    from .targets import TARGETS

    cohort = _read_cohort(cohort_path)
    partition = _read_partition(partition_path, cohort, config)
    training = cohort.drop(columns=[name for name in ("split", "fold") if name in cohort]).merge(
        partition[["case_id", "split", "fold"]], on="case_id", how="left", validate="one_to_one"
    )
    training = training.loc[training["split"].eq("train_cv")].reset_index(drop=True)
    if training.empty:
        raise ValueError("the full eligible partition has no train_cv cases")
    return fit_zscore_reference(
        training.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        training["sex"].astype(str),
        training["age"].to_numpy(dtype=np.float64),
        targets=TARGETS,
    )


def _cmd_fit_referral(args: argparse.Namespace) -> int:
    from .referral import fit_referral_axis, measurement_discordance

    config = _pipeline(args.config)
    arrays, calibration_frame = _aligned_predictions(
        args.predictions, args.cohort, args.partition, config, "calibration"
    )
    _require_prediction_arrays(arrays, ("mu",))
    reference = _training_reference(args.cohort, args.partition, config)
    z_scores = reference.transform(
        arrays["mu"], calibration_frame["sex"].astype(str), calibration_frame["age"].to_numpy(dtype=float)
    )
    sagittal_discordance, vertical_discordance = measurement_discordance(z_scores)
    rates = tuple(float(value) for value in config["referral"]["rates"])
    sagittal = fit_referral_axis(arrays["prob_sag"], sagittal_discordance, rates=rates)
    vertical = fit_referral_axis(arrays["prob_vert"], vertical_discordance, rates=rates)
    _write_npz(
        args.output,
        {
            "rates": np.asarray(rates, dtype=np.float64),
            "sagittal_confidence": sagittal.calibration_confidence,
            "sagittal_discordance": sagittal.calibration_discordance,
            "sagittal_thresholds": np.asarray([sagittal.operating_points[rate] for rate in rates]),
            "vertical_confidence": vertical.calibration_confidence,
            "vertical_discordance": vertical.calibration_discordance,
            "vertical_thresholds": np.asarray([vertical.operating_points[rate] for rate in rates]),
        },
    )
    print(f"{len(calibration_frame)} cases fitted.")
    return 0


def _load_referral_state(path: str | Path):
    from .referral import ReferralAxisCalibration

    source = input_path(path, "file")
    names = {
        "rates",
        "sagittal_confidence",
        "sagittal_discordance",
        "sagittal_thresholds",
        "vertical_confidence",
        "vertical_discordance",
        "vertical_thresholds",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(names.difference(archive.files))
        if missing:
            raise ValueError(f"referral state is missing arrays: {missing}")
        values = {name: np.asarray(archive[name], dtype=np.float64).copy() for name in names}
    rates = values["rates"]
    if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all() or ((rates <= 0) | (rates >= 1)).any():
        raise ValueError("referral rates must be finite and lie in (0, 1)")
    if len(set(rates.tolist())) != len(rates):
        raise ValueError("referral rates must be unique")
    axes = []
    for prefix in ("sagittal", "vertical"):
        thresholds = values[f"{prefix}_thresholds"]
        if thresholds.shape != rates.shape or not np.isfinite(thresholds).all():
            raise ValueError(f"invalid {prefix} referral thresholds")
        axes.append(
            ReferralAxisCalibration(
                values[f"{prefix}_confidence"],
                values[f"{prefix}_discordance"],
                {float(rate): float(value) for rate, value in zip(rates, thresholds)},
            )
        )
    return rates, axes[0], axes[1]


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .calibration import ConformalCalibration
    from .evaluation import (
        classification_metrics,
        evaluate_predictions,
        publication_evaluation_result,
        referral_report,
        regression_report,
        stratified_report,
    )
    from .referral import measurement_discordance
    from .targets import CLASS_NAMES, TARGETS

    config = _pipeline(args.config, args.arm)
    arrays, frame = _aligned_predictions(
        args.predictions, args.cohort, args.partition, config, args.split
    )
    conformal = None
    if args.conformal:
        with input_path(args.conformal, "file").open(encoding="utf-8") as stream:
            conformal = ConformalCalibration.from_dict(json.load(stream))
        if conformal.targets != tuple(TARGETS):
            raise ValueError("conformal target order does not match the model")
    truth = frame.loc[:, list(TARGETS)].to_numpy(dtype=np.float64)
    if conformal is not None:
        _require_prediction_arrays(arrays, ("mu", "sigma"))
    if "mu" in arrays and "sigma" in arrays:
        result = evaluate_predictions(
            truth,
            arrays["mu"],
            arrays["sigma"],
            frame["sagittal"].astype(str),
            arrays["prob_sag"],
            frame["vertical"].astype(str),
            arrays["prob_vert"],
            conformal=conformal,
            targets=TARGETS,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=0,
        )
    else:
        result = {
            "n": len(frame),
            "classification": {
                "sagittal": classification_metrics(
                    frame["sagittal"].astype(str),
                    arrays["prob_sag"],
                    classes=CLASS_NAMES["sagittal"],
                    bootstrap_resamples=args.bootstrap_resamples,
                    seed=0,
                ),
                "vertical": classification_metrics(
                    frame["vertical"].astype(str),
                    arrays["prob_vert"],
                    classes=CLASS_NAMES["vertical"],
                    bootstrap_resamples=args.bootstrap_resamples,
                    seed=0,
                ),
            },
        }
        if "mu" in arrays:
            result["regression"] = regression_report(
                truth,
                arrays["mu"],
                targets=TARGETS,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=0,
            )
    if args.referral:
        _require_prediction_arrays(arrays, ("mu",))
        rates, sagittal_axis, vertical_axis = _load_referral_state(args.referral)
        reference = _training_reference(args.cohort, args.partition, config)
        z_scores = reference.transform(
            arrays["mu"], frame["sex"].astype(str), frame["age"].to_numpy(dtype=float)
        )
        sagittal_discordance, vertical_discordance = measurement_discordance(z_scores)
        result["referral"] = {"sagittal": {}, "vertical": {}}
        for rate in rates:
            key = f"{float(rate):g}"
            result["referral"]["sagittal"][key] = referral_report(
                frame["sagittal"].astype(str),
                arrays["prob_sag"],
                sagittal_axis.refer(arrays["prob_sag"], sagittal_discordance, float(rate)),
                classes=CLASS_NAMES["sagittal"],
            )
            result["referral"]["vertical"][key] = referral_report(
                frame["vertical"].astype(str),
                arrays["prob_vert"],
                vertical_axis.refer(arrays["prob_vert"], vertical_discordance, float(rate)),
                classes=CLASS_NAMES["vertical"],
            )
    stratified = stratified_report(
        truth if "mu" in arrays else None,
        arrays.get("mu"),
        frame["sagittal"].astype(str),
        arrays["prob_sag"],
        frame["vertical"].astype(str),
        arrays["prob_vert"],
        frame["sex"].astype(str),
        frame["age"].to_numpy(dtype=np.float64),
    )
    published = publication_evaluation_result(result, split=args.split, stratified=stratified)
    envelope = {
        "config": str(config["name"]),
        "alpha": float(conformal.alpha if conformal is not None else config["conformal"]["alpha"]),
        "n_boot": int(args.bootstrap_resamples),
        "results": [published],
        "protocol": "thresholds and operating points are frozen on the calibration split; nothing is tuned here",
    }
    write_json(args.output, envelope)
    print(f"{len(frame)} cases evaluated.")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import aggregate_analysis_reports, write_aggregate_reports
    from .partition import validate_partition
    from .targets import TARGETS

    config = _pipeline(args.config, args.arm)
    cohort = _read_cohort(args.cohort)
    if args.partition:
        partition = _read_partition(args.partition, cohort, config)
    else:
        if not {"split", "fold"}.issubset(cohort.columns):
            raise ValueError("provide --partition or include split and fold in the controlled cohort")
        partition = cohort[["case_id", "split", "fold"]].copy()
        validate_partition(partition, cohort["case_id"], _partition_config(config))
    calibration = _load_predictions(args.calibration_predictions)
    calibration_frame = _select_analysis_frame(cohort, partition, config, "calibration")
    calibration, calibration_frame = _align_prediction_arrays(calibration, calibration_frame, "calibration")
    test = _load_predictions(args.test_predictions)
    test_frame = _select_analysis_frame(cohort, partition, config, "internal_test")
    test, test_frame = _align_prediction_arrays(test, test_frame, "internal_test")
    _require_prediction_arrays(calibration, ("mu", "sigma"))
    _require_prediction_arrays(test, ("mu", "sigma"))

    merged = cohort.drop(columns=[name for name in ("split", "fold") if name in cohort]).merge(
        partition[["case_id", "split", "fold"]], on="case_id", how="left", validate="one_to_one"
    )
    training = merged.loc[merged["split"].eq("train_cv")]
    paired = merged.loc[merged["split"].eq("internal_test")]
    second_columns = [f"{target}_t2" for target in TARGETS]
    missing = sorted(set(second_columns).difference(paired.columns))
    if missing:
        raise ValueError(f"cohort is missing paired measurement columns: {missing}")
    repeat_second = paired.loc[:, second_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    tracer_columns = {"tracer_1", "tracer_2"}
    if not tracer_columns.issubset(paired.columns):
        raise ValueError("cohort is missing paired tracer fields")
    observed = np.isfinite(repeat_second).any(axis=1)
    first_tracer = paired.loc[observed, "tracer_1"].astype("string").str.strip()
    second_tracer = paired.loc[observed, "tracer_2"].astype("string").str.strip()
    if (
        first_tracer.isna().any()
        or second_tracer.isna().any()
        or first_tracer.eq("").any()
        or second_tracer.eq("").any()
        or (first_tracer == second_tracer).any()
    ):
        raise ValueError("paired measurements require two identified, distinct tracers")
    reports = aggregate_analysis_reports(
        training_truth=training.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        training_age=training["age"].to_numpy(dtype=np.float64),
        training_sex=training["sex"].astype(str).to_numpy(),
        calibration_truth=calibration_frame.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        calibration_prediction=calibration["mu"],
        calibration_sigma=calibration["sigma"],
        calibration_sagittal_probabilities=calibration["prob_sag"],
        calibration_vertical_probabilities=calibration["prob_vert"],
        calibration_age=calibration_frame["age"].to_numpy(dtype=np.float64),
        calibration_sex=calibration_frame["sex"].astype(str).to_numpy(),
        test_truth=test_frame.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        test_prediction=test["mu"],
        test_sigma=test["sigma"],
        sagittal_probabilities=test["prob_sag"],
        vertical_probabilities=test["prob_vert"],
        age=test_frame["age"].to_numpy(dtype=np.float64),
        sex=test_frame["sex"].astype(str).to_numpy(),
        repeat_first=paired.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        repeat_second=repeat_second,
        cohort_truth=cohort.loc[:, list(TARGETS)].to_numpy(dtype=np.float64),
        alpha=float(config["conformal"]["alpha"] if args.alpha is None else args.alpha),
        config=args.arm,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    write_aggregate_reports(args.output_dir, reports)
    print(f"{len(reports) - 1} aggregate analyses generated.")
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from .reproduction import main as reproduce_main

    forwarded = ["--data-dir", args.data_dir, "--reference-dir", args.reference_dir, "--atol", str(args.atol)]
    if args.operator_map:
        forwarded.extend(("--operator-map", args.operator_map))
    if args.output:
        forwarded.extend(("--output", args.output))
    if args.show_values:
        forwarded.append("--show-values")
    return reproduce_main(forwarded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face2ceph", description="Publication pipeline for facial cephalometry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assets_parser = subparsers.add_parser("assets", help="download and verify declared model assets")
    assets_parser.add_argument("names", nargs="*", choices=tuple(_asset_specs()))
    assets_parser.set_defaults(handler=_cmd_assets)

    preprocess_parser = subparsers.add_parser("preprocess", help="normalize authorized photographs")
    preprocess_parser.add_argument("--cohort", required=True)
    preprocess_parser.add_argument("--output-dir", required=True)
    preprocess_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    preprocess_parser.add_argument("--landmarker-asset")
    preprocess_parser.add_argument("--segmenter-asset")
    preprocess_parser.set_defaults(handler=_cmd_preprocess)

    partition_parser = subparsers.add_parser("partition", help="create a frozen eligible-cohort partition")
    partition_parser.add_argument("--cohort", required=True)
    partition_parser.add_argument("--output", required=True)
    partition_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    partition_parser.set_defaults(handler=_cmd_partition)

    train_parser = subparsers.add_parser("train", help="train a five-fold ensemble")
    train_parser.add_argument("--cohort", required=True)
    train_parser.add_argument("--partition", required=True)
    train_parser.add_argument("--image-root", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--arm", choices=ARM_NAMES, default="main")
    train_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    train_parser.add_argument("--backbone-weights")
    train_parser.add_argument("--device")
    train_parser.set_defaults(handler=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="run fold-ensemble inference")
    predict_parser.add_argument("--cohort", required=True)
    predict_parser.add_argument("--partition", required=True)
    predict_parser.add_argument("--image-root", required=True)
    predict_parser.add_argument("--checkpoints", required=True)
    predict_parser.add_argument("--split", choices=("train_cv", "calibration", "internal_test"), required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--arm", choices=ARM_NAMES, default="main")
    predict_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    predict_parser.add_argument("--device")
    predict_parser.add_argument("--include-features", action="store_true")
    predict_parser.add_argument("--perturbation")
    predict_parser.set_defaults(handler=_cmd_predict)

    calibrate_parser = subparsers.add_parser("calibrate", help="fit split-conformal quantiles")
    calibrate_parser.add_argument("--cohort", required=True)
    calibrate_parser.add_argument("--partition", required=True)
    calibrate_parser.add_argument("--predictions", required=True)
    calibrate_parser.add_argument("--output", required=True)
    calibrate_parser.add_argument("--alpha", type=float)
    calibrate_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    calibrate_parser.set_defaults(handler=_cmd_calibrate)

    referral_parser = subparsers.add_parser("fit-referral", help="fit calibration-only referral state")
    referral_parser.add_argument("--cohort", required=True)
    referral_parser.add_argument("--partition", required=True)
    referral_parser.add_argument("--predictions", required=True)
    referral_parser.add_argument("--output", required=True)
    referral_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    referral_parser.set_defaults(handler=_cmd_fit_referral)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a frozen prediction archive")
    evaluate_parser.add_argument("--cohort", required=True)
    evaluate_parser.add_argument("--partition", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--split", choices=("calibration", "internal_test"), default="internal_test")
    evaluate_parser.add_argument("--conformal")
    evaluate_parser.add_argument("--referral")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--arm", choices=ARM_NAMES, default="main")
    evaluate_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    evaluate_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

    analyze_parser = subparsers.add_parser("analyze", help="regenerate aggregate analyses from frozen predictions")
    analyze_parser.add_argument("--cohort", required=True)
    analyze_parser.add_argument("--partition")
    analyze_parser.add_argument("--calibration-predictions", required=True)
    analyze_parser.add_argument("--test-predictions", required=True)
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--arm", choices=ARM_NAMES, default="main")
    analyze_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    analyze_parser.add_argument("--alpha", type=float)
    analyze_parser.add_argument("--config", default=str(CONFIG_ROOT / "pipeline.yaml"))
    analyze_parser.set_defaults(handler=_cmd_analyze)

    reproduce_parser = subparsers.add_parser("reproduce", help="recompute and verify the 46 reported quantities")
    reproduce_parser.add_argument("--data-dir", required=True)
    reproduce_parser.add_argument("--reference-dir", default=str(RELEASE_ROOT / "reference"))
    reproduce_parser.add_argument("--operator-map")
    reproduce_parser.add_argument("--atol", type=float, default=5e-6)
    reproduce_parser.add_argument("--output")
    reproduce_parser.add_argument("--show-values", action="store_true")
    reproduce_parser.set_defaults(handler=_cmd_reproduce)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileExistsError, FileNotFoundError, KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
