"""Deterministic input perturbations and aggregate robustness scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationSpec:
    kind: str
    level: float

    @property
    def tag(self) -> str:
        value = f"{self.level:g}".replace("-", "m").replace(".", "p")
        return f"{self.kind}_{value}"

    @property
    def label(self) -> str:
        units = {"rotate": "degrees", "blur": "pixels", "jpeg": "quality", "noise": "gray levels"}
        unit = units.get(self.kind, "factor")
        return f"{self.kind} {self.level:g} {unit}"


DEFAULT_PERTURBATIONS = tuple(
    PerturbationSpec(kind, level)
    for kind, levels in (
        ("brightness", (0.85, 1.15, 0.70, 1.30)),
        ("contrast", (0.85, 1.15, 0.70, 1.30)),
        ("gamma", (0.80, 1.25)),
        ("colortemp", (0.90, 1.10)),
        ("rotate", (2.0, 5.0, -5.0)),
        ("blur", (1.0, 2.0)),
        ("jpeg", (60.0, 35.0)),
        ("downup", (0.50, 0.35)),
        ("noise", (5.0, 12.0)),
    )
    for level in levels
)


def _image_batch(values: np.ndarray, name: str, allow_grayscale: bool = False) -> np.ndarray:
    array = np.asarray(values)
    valid_shape = array.ndim == 4 and array.shape[-1] in ({1, 3} if allow_grayscale else {3})
    if not valid_shape or array.dtype != np.uint8:
        channels = "one or three" if allow_grayscale else "three"
        raise ValueError(f"{name} must be a uint8 [case, height, width, channel] array with {channels} channels")
    return array


def _photo(image: np.ndarray, spec: PerturbationSpec, seed: int) -> np.ndarray:
    kind, level = spec.kind, spec.level
    values = image.astype(np.float32)
    if kind == "brightness":
        values *= level
    elif kind == "contrast":
        values = (values - 128.0) * level + 128.0
    elif kind == "gamma":
        values = 255.0 * np.power(values / 255.0, level)
    elif kind == "colortemp":
        values[..., 0] *= level
        values[..., 2] *= 2.0 - level
    elif kind == "noise":
        values += np.random.default_rng(seed).normal(0.0, level, size=values.shape)
    elif kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=level)
    elif kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(level)])
        if not ok:
            raise ValueError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("JPEG decoding failed")
        return decoded
    elif kind == "downup":
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(8, int(width * level)), max(8, int(height * level))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"unsupported photometric perturbation: {kind}")
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated[..., None] if rotated.ndim == 2 and image.shape[-1] == 1 else rotated


def apply_perturbation(
    frontal: np.ndarray,
    profile: np.ndarray,
    silhouette: np.ndarray | None,
    spec: PerturbationSpec,
    *,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply one condition to aligned BGR image batches without changing the inputs."""
    front = _image_batch(frontal, "frontal")
    side = _image_batch(profile, "profile")
    mask = None if silhouette is None else _image_batch(silhouette, "silhouette", allow_grayscale=True)
    if front.shape[0] != side.shape[0] or (mask is not None and mask.shape[0] != front.shape[0]):
        raise ValueError("all image batches must contain the same cases")
    if spec not in DEFAULT_PERTURBATIONS:
        raise ValueError(f"condition is not in the registered perturbation grid: {spec.tag}")

    if spec.kind == "rotate":
        front_out = np.stack([_rotate(image, spec.level) for image in front])
        side_out = np.stack([_rotate(image, spec.level) for image in side])
        mask_out = None if mask is None else np.stack([_rotate(image, spec.level) for image in mask])
        return front_out, side_out, mask_out

    front_out = np.stack([_photo(image, spec, seed) for image in front])
    side_out = np.stack([_photo(image, spec, seed) for image in side])
    return front_out, side_out, None if mask is None else mask.copy()


@dataclass(frozen=True)
class PerturbationTransform:
    spec: PerturbationSpec
    seed: int = 12345

    def __call__(
        self,
        frontal: np.ndarray,
        profile: np.ndarray,
        silhouette: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if frontal.ndim != 3 or profile.ndim != 3 or (silhouette is not None and silhouette.ndim != 2):
            raise ValueError("PerturbationTransform expects two H x W x 3 images and an optional H x W silhouette")
        mask = None if silhouette is None else silhouette[..., None]
        front, side, transformed_mask = apply_perturbation(
            frontal[None],
            profile[None],
            None if mask is None else mask[None],
            self.spec,
            seed=self.seed,
        )
        return front[0], side[0], None if transformed_mask is None else transformed_mask[0, ..., 0]


def _finite(values: object, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional numeric array")
    return array


def _balanced_accuracy(reference: np.ndarray, prediction: np.ndarray) -> float:
    classes = np.unique(reference)
    return float(np.mean([np.mean(prediction[reference == value] == value) for value in classes]))


def _score(
    prediction: Mapping[str, object],
    y_regression: np.ndarray,
    y_sagittal: np.ndarray,
    y_vertical: np.ndarray,
) -> dict[str, float]:
    mu = _finite(prediction.get("mu"), "mu", 2)
    prob_sag = _finite(prediction.get("prob_sag"), "prob_sag", 2)
    prob_vert = _finite(prediction.get("prob_vert"), "prob_vert", 2)
    n = y_regression.shape[0]
    if mu.shape != y_regression.shape or prob_sag.shape != (n, 3) or prob_vert.shape != (n, 3):
        raise ValueError("prediction arrays do not match the reference arrays")
    for probabilities, name in ((prob_sag, "prob_sag"), (prob_vert, "prob_vert")):
        if np.any((probabilities < 0.0) | (probabilities > 1.0)) or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(f"{name} must contain row-normalized probabilities")
    pred_sag = np.argmax(prob_sag, axis=1)
    pred_vert = np.argmax(prob_vert, axis=1)
    low_angle = y_vertical == 0
    sigma_mean = float("nan")
    if prediction.get("sigma") is not None:
        sigma = _finite(prediction["sigma"], "sigma", 2)
        if sigma.shape != y_regression.shape or np.any(sigma < 0):
            raise ValueError("sigma must be non-negative and match mu")
        sigma_mean = float(sigma.mean())
    return {
        "mae_mean": float(np.abs(mu - y_regression).mean()),
        "balanced_accuracy_sagittal": _balanced_accuracy(y_sagittal, pred_sag),
        "balanced_accuracy_vertical": _balanced_accuracy(y_vertical, pred_vert),
        "recall_low_angle": float(np.mean(pred_vert[low_angle] == 0)) if low_angle.any() else float("nan"),
        "sigma_mean": sigma_mean,
    }


def score_perturbation_grid(
    baseline: Mapping[str, object],
    perturbed: Mapping[str, Mapping[str, object]],
    y_regression: np.ndarray,
    y_sagittal: np.ndarray,
    y_vertical: np.ndarray,
    *,
    grid: Sequence[PerturbationSpec] = DEFAULT_PERTURBATIONS,
    mild_mae_increase: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score a complete grid and reject partial or unregistered condition sets."""
    specs = tuple(grid)
    if not specs:
        raise ValueError("perturbation grid must not be empty")
    if not np.isfinite(mild_mae_increase) or mild_mae_increase < 0:
        raise ValueError("mild_mae_increase must be finite and non-negative")
    tags = tuple(spec.tag for spec in specs)
    if len(tags) != len(set(tags)):
        raise ValueError("perturbation tags must be unique")
    missing = sorted(set(tags) - set(perturbed))
    extra = sorted(set(perturbed) - set(tags))
    if missing or extra:
        raise ValueError(f"incomplete perturbation grid; missing={missing}, extra={extra}")

    y_reg = _finite(y_regression, "y_regression", 2)
    y_sag_raw = _finite(y_sagittal, "y_sagittal", 1)
    y_vert_raw = _finite(y_vertical, "y_vertical", 1)
    if not np.equal(y_sag_raw, np.rint(y_sag_raw)).all() or not np.equal(y_vert_raw, np.rint(y_vert_raw)).all():
        raise ValueError("classification references must be integer-valued")
    y_sag = y_sag_raw.astype(int)
    y_vert = y_vert_raw.astype(int)
    if y_sag.shape != (y_reg.shape[0],) or y_vert.shape != (y_reg.shape[0],):
        raise ValueError("classification references must match y_regression")
    if np.any((y_sag < 0) | (y_sag > 2)) or np.any((y_vert < 0) | (y_vert > 2)):
        raise ValueError("classification references must use integer labels 0, 1, and 2")

    base = _score(baseline, y_reg, y_sag, y_vert)
    rows: list[dict[str, object]] = [{
        "condition": "baseline",
        "kind": "baseline",
        "level": 0.0,
        "label": "unperturbed",
        **base,
        "delta_mae": 0.0,
        "delta_balanced_accuracy_sagittal": 0.0,
        "delta_balanced_accuracy_vertical": 0.0,
        "delta_sigma": 0.0 if np.isfinite(base["sigma_mean"]) else float("nan"),
    }]
    for spec in specs:
        score = _score(perturbed[spec.tag], y_reg, y_sag, y_vert)
        rows.append({
            "condition": spec.tag,
            "kind": spec.kind,
            "level": spec.level,
            "label": spec.label,
            **score,
            "delta_mae": score["mae_mean"] - base["mae_mean"],
            "delta_balanced_accuracy_sagittal": score["balanced_accuracy_sagittal"] - base["balanced_accuracy_sagittal"],
            "delta_balanced_accuracy_vertical": score["balanced_accuracy_vertical"] - base["balanced_accuracy_vertical"],
            "delta_sigma": score["sigma_mean"] - base["sigma_mean"],
        })

    table = pd.DataFrame(rows)
    conditions = table.iloc[1:]
    correlation = float("nan")
    if (
        conditions["delta_sigma"].notna().all()
        and conditions["delta_sigma"].std() > 0
        and conditions["delta_mae"].std() > 0
    ):
        correlation = float(np.corrcoef(conditions["delta_mae"], conditions["delta_sigma"])[0, 1])
    worst = conditions.loc[conditions["delta_mae"].idxmax()]
    summary: dict[str, object] = {
        "n_conditions_expected": len(specs),
        "n_conditions_completed": len(conditions),
        "n_mae_increase_within_threshold": int((conditions["delta_mae"] <= mild_mae_increase).sum()),
        "mae_increase_threshold": float(mild_mae_increase),
        "worst_condition": str(worst["condition"]),
        "worst_delta_mae": float(worst["delta_mae"]),
        "error_uncertainty_correlation": correlation,
        "scope": "Controlled transformations quantify sensitivity; they do not establish performance at another site.",
    }
    return table, summary
