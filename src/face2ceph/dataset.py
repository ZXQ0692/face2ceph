"""Dataset contracts, deterministic augmentation, and target scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .targets import AGE_MIN as COHORT_AGE_MIN
from .targets import CLASS_NAMES, TARGETS

MEASUREMENT_NAMES = TARGETS
SAGITTAL_CLASSES = CLASS_NAMES["sagittal"]
VERTICAL_CLASSES = CLASS_NAMES["vertical"]
SAGITTAL_TO_INDEX = {name: index for index, name in enumerate(SAGITTAL_CLASSES)}
VERTICAL_TO_INDEX = {name: index for index, name in enumerate(VERTICAL_CLASSES)}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
AGE_MIN = float(COHORT_AGE_MIN)
AGE_MAX = 60.0
ImageTransform = Callable[
    [np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
]


@dataclass(frozen=True)
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (len(MEASUREMENT_NAMES),) or std.shape != mean.shape:
            raise ValueError("Target statistics must contain eight values")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Target statistics must be finite with positive standard deviations")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def fit(cls, values: pd.DataFrame | np.ndarray) -> "TargetScaler":
        array = (
            values.loc[:, list(MEASUREMENT_NAMES)].to_numpy(dtype=np.float64)
            if isinstance(values, pd.DataFrame)
            else np.asarray(values, dtype=np.float64)
        )
        if array.ndim != 2 or array.shape[1] != len(MEASUREMENT_NAMES) or not np.isfinite(array).all():
            raise ValueError("Regression targets must be a finite N x 8 array")
        std = array.std(axis=0)
        return cls(array.mean(axis=0), np.where(std > 1e-12, std, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "TargetScaler":
        return cls(np.asarray(state["mean"]), np.asarray(state["std"]))


@dataclass(frozen=True)
class AugmentationConfig:
    rotation_degrees: float = 5.0
    translation_fraction: float = 0.10
    brightness_fraction: float = 0.15
    contrast_fraction: float = 0.15
    clahe_probability: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.rotation_degrees,
            self.translation_fraction,
            self.brightness_fraction,
            self.contrast_fraction,
            self.clahe_probability,
        )
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Augmentation values must be finite and nonnegative")
        if self.translation_fraction >= 1 or self.brightness_fraction > 1 or self.contrast_fraction > 1:
            raise ValueError("Augmentation fractions are out of range")
        if self.clahe_probability > 1:
            raise ValueError("clahe_probability must lie in [0, 1]")


def load_manifest(
    clinical_manifest_path: str | Path,
    split_manifest_path: str | Path | None = None,
    *,
    require_targets: bool = True,
    filter_analyzed: bool = True,
    availability_column: str | None = None,
) -> pd.DataFrame:
    clinical = pd.read_csv(Path(clinical_manifest_path))
    required = {"case_id", "age", "sex"}
    if require_targets:
        required.update(MEASUREMENT_NAMES)
        required.update(("sagittal", "vertical"))
    missing = sorted(required.difference(clinical.columns))
    if missing:
        raise ValueError(f"Clinical manifest is missing columns: {missing}")
    if clinical["case_id"].isna().any() or clinical["case_id"].duplicated().any():
        raise ValueError("case_id must be present and unique in the clinical manifest")

    if split_manifest_path is not None:
        split = pd.read_csv(Path(split_manifest_path))
        missing = sorted({"case_id", "split", "fold"}.difference(split.columns))
        if missing:
            raise ValueError(f"Split manifest is missing columns: {missing}")
        if split["case_id"].isna().any() or split["case_id"].duplicated().any():
            raise ValueError("case_id must be present and unique in the split manifest")
        clinical_ids = set(clinical["case_id"])
        split_ids = set(split["case_id"])
        if clinical_ids != split_ids:
            raise ValueError("Clinical and split manifests must contain the same case identifiers")
        overlap = [name for name in ("split", "fold") if name in clinical.columns]
        clinical = clinical.drop(columns=overlap).merge(
            split.loc[:, ["case_id", "split", "fold"]],
            on="case_id",
            how="inner",
            validate="one_to_one",
        )
    if filter_analyzed:
        if availability_column is not None:
            if availability_column not in {"analyzed", "usable"}:
                raise ValueError("availability_column must be analyzed or usable")
            if availability_column not in clinical.columns:
                raise ValueError(f"Clinical manifest is missing {availability_column}")
            status_column = availability_column
        elif "analyzed" in clinical.columns:
            status_column = "analyzed"
        elif "usable" in clinical.columns:
            status_column = "usable"
        else:
            status_column = None
        if status_column is not None:
            clinical = clinical.loc[_boolean_mask(clinical[status_column], status_column)]
    return clinical.reset_index(drop=True)


def _boolean_mask(values: pd.Series, column: str) -> np.ndarray:
    if values.isna().any():
        raise ValueError(f"{column} must contain complete boolean values")
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{column} must contain only boolean values")
    return parsed.to_numpy(dtype=bool)


def select_fold(frame: pd.DataFrame, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"split", "fold"}.issubset(frame.columns):
        raise ValueError("Fold selection requires split and fold columns")
    training_pool = frame.loc[frame["split"].eq("train_cv")].copy()
    numeric_folds = pd.to_numeric(training_pool["fold"], errors="raise")
    if not np.equal(numeric_folds, np.floor(numeric_folds)).all():
        raise ValueError("Training fold identifiers must be integers")
    fold_values = numeric_folds.astype(int)
    train = training_pool.loc[fold_values.ne(validation_fold)].reset_index(drop=True)
    validation = training_pool.loc[fold_values.eq(validation_fold)].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError(f"Fold {validation_fold} does not define non-empty training and validation sets")
    return train, validation


def inverse_frequency_weights(labels: Sequence[int], class_count: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=class_count)
    if counts.shape[0] != class_count or (counts == 0).any():
        raise ValueError("Every class must occur in the training fold")
    return torch.as_tensor(counts.sum() / (class_count * counts), dtype=torch.float32)


def _read_image(path: Path, grayscale: bool) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    except OSError as error:
        raise FileNotFoundError(f"Unable to read an input image: {path.name}") from error
    if image is None:
        raise ValueError(f"Unable to decode an input image: {path.name}")
    return image


def _warp(images: Sequence[np.ndarray], rng: np.random.Generator, config: AugmentationConfig) -> list[np.ndarray]:
    height, width = images[0].shape[:2]
    angle = float(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    shift_x = float(rng.uniform(-config.translation_fraction, config.translation_fraction)) * width
    shift_y = float(rng.uniform(-config.translation_fraction, config.translation_fraction)) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[:, 2] += (shift_x, shift_y)
    return [
        cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for image in images
    ]


def _adjust_rgb(image: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> np.ndarray:
    alpha = 1.0 + float(rng.uniform(-config.contrast_fraction, config.contrast_fraction))
    beta = 255.0 * float(rng.uniform(-config.brightness_fraction, config.brightness_fraction))
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < config.clahe_probability:
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _sex_value(value: Any) -> float:
    normalized = str(value).strip().lower()
    if normalized in {"m", "male", "1", "1.0"}:
        return 1.0
    if normalized in {"f", "female", "0", "0.0"}:
        return 0.0
    raise ValueError("sex must encode female or male")


class ClinicalPhotoDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: pd.DataFrame,
        image_root: str | Path,
        *,
        target_scaler: TargetScaler | None = None,
        augment: AugmentationConfig | None = None,
        image_size: int = 384,
        use_profile_sdf: bool = True,
        input_mode: str = "both",
        augmentation_seed: int | None = None,
        input_transform: ImageTransform | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.target_scaler = target_scaler
        self.augment = augment
        self.image_size = int(image_size)
        self.use_profile_sdf = bool(use_profile_sdf)
        self.input_mode = str(input_mode)
        self.augmentation_seed = None if augmentation_seed is None else int(augmentation_seed)
        self.input_transform = input_transform
        self._rng: np.random.Generator | None = None

        if self.input_mode not in {"both", "frontal", "profile", "silhouette"}:
            raise ValueError("input_mode must be both, frontal, profile, or silhouette")
        if self.input_mode == "silhouette" and not self.use_profile_sdf:
            raise ValueError("Silhouette input requires the profile signed-distance channel")

        required = {"case_id", "age", "sex"}
        missing = sorted(required.difference(self.records.columns))
        if missing:
            raise ValueError(f"Dataset records are missing columns: {missing}")
        self.has_targets = set(MEASUREMENT_NAMES).issubset(self.records.columns)
        self.has_classes = {"sagittal", "vertical"}.issubset(self.records.columns)
        if target_scaler is not None and not self.has_targets:
            raise ValueError("A target scaler requires regression targets")

    def __len__(self) -> int:
        return len(self.records)

    def _random_generator(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.augmentation_seed)
        return self._rng

    def _path(self, row: pd.Series, column: str, folder: str, suffix: str) -> Path:
        value = row.get(column)
        path = (
            Path(str(value))
            if value is not None and not pd.isna(value) and str(value).strip()
            else Path(folder) / f"{row['case_id']}{suffix}"
        )
        return path if path.is_absolute() else self.image_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records.iloc[index]
        frontal = _read_image(self._path(row, "frontal_path", "frontal", ".jpg"), False)
        profile = _read_image(self._path(row, "profile_path", "profile", ".jpg"), False)
        sdf = (
            _read_image(self._path(row, "profile_sdf_path", "profile_sdf", ".png"), True)
            if self.use_profile_sdf
            else None
        )
        expected = (self.image_size, self.image_size)
        shapes = (frontal.shape[:2], profile.shape[:2]) + (() if sdf is None else (sdf.shape[:2],))
        if any(shape != expected for shape in shapes):
            raise ValueError(
                f"Preprocessed inputs for case {row['case_id']} must be "
                f"{self.image_size} x {self.image_size}"
            )

        if self.input_transform is not None:
            frontal, profile, sdf = self.input_transform(frontal, profile, sdf)
            rgb_valid = all(
                isinstance(image, np.ndarray)
                and image.shape == (*expected, 3)
                and image.dtype == np.uint8
                for image in (frontal, profile)
            )
            sdf_valid = (
                isinstance(sdf, np.ndarray) and sdf.shape == expected and sdf.dtype == np.uint8
                if self.use_profile_sdf
                else sdf is None
            )
            if not rgb_valid or not sdf_valid:
                raise ValueError("Input transform returned invalid image arrays")

        if self.augment is not None:
            rng = self._random_generator()
            frontal = _adjust_rgb(_warp((frontal,), rng, self.augment)[0], rng, self.augment)
            if sdf is None:
                profile = _warp((profile,), rng, self.augment)[0]
            else:
                profile, sdf = _warp((profile, sdf), rng, self.augment)
            profile = _adjust_rgb(profile, rng, self.augment)

        frontal_array = _normalize_rgb(frontal)
        profile_array = _normalize_rgb(profile)
        if sdf is not None:
            sdf_array = sdf.astype(np.float32) / 127.5 - 1.0
            if self.input_mode == "silhouette":
                frontal_array = np.zeros_like(frontal_array)
                profile_array = np.repeat(sdf_array[None], 3, axis=0)
            profile_array = np.concatenate((profile_array, sdf_array[None]), axis=0)
        if self.input_mode == "frontal":
            profile_array = np.zeros_like(profile_array)
        elif self.input_mode == "profile":
            frontal_array = np.zeros_like(frontal_array)
        metadata = np.asarray(
            ((float(row["age"]) - AGE_MIN) / (AGE_MAX - AGE_MIN), _sex_value(row["sex"])),
            dtype=np.float32,
        )
        sample: dict[str, Any] = {
            "case_id": str(row["case_id"]),
            "frontal": torch.from_numpy(np.ascontiguousarray(frontal_array)),
            "profile": torch.from_numpy(np.ascontiguousarray(profile_array)),
            "metadata": torch.from_numpy(metadata),
        }
        if self.has_targets:
            raw = row.loc[list(MEASUREMENT_NAMES)].to_numpy(dtype=np.float32)
            sample["regression_target_raw"] = torch.from_numpy(raw)
            if self.target_scaler is not None:
                standardized = self.target_scaler.transform(raw).astype(np.float32)
                sample["regression_target"] = torch.from_numpy(standardized)
        if self.has_classes:
            try:
                sample["sagittal_target"] = torch.tensor(SAGITTAL_TO_INDEX[str(row["sagittal"])], dtype=torch.long)
                sample["vertical_target"] = torch.tensor(VERTICAL_TO_INDEX[str(row["vertical"])], dtype=torch.long)
            except KeyError as error:
                raise ValueError(f"Unknown class label for case {row['case_id']}") from error
        return sample
