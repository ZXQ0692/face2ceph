"""Eligibility, thresholding, and fixed geometric image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


TARGETS = (
    "ANB",
    "Wits",
    "SN_MP",
    "FMA",
    "PP_MP",
    "Jarabak",
    "Y_axis",
    "LAFH_TAFH",
)
SAGITTAL_TARGETS = ("ANB", "Wits")
VERTICAL_TARGETS = ("SN_MP", "FMA", "PP_MP", "Jarabak", "Y_axis", "LAFH_TAFH")
TARGET_DIRECTIONS = np.array((1, 1, 1, 1, 1, -1, 1, 1), dtype=np.float64)
SAGITTAL_CLASSES = ("III", "I", "II")
VERTICAL_CLASSES = ("Hypo", "Normo", "Hyper")

LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)
HEAD_CLASSES = (HAIR, FACE_SKIN)
PERSON_CLASSES = (HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER)
DETECTION_MAX_SIDE = 640


@dataclass(frozen=True)
class Band:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.lower, self.upper)).all() or self.lower >= self.upper:
            raise ValueError("A band requires finite, increasing bounds")


@dataclass(frozen=True)
class ThresholdScheme:
    name: str
    sagittal: Band
    vertical_adult: Band
    vertical_male_minor: Band
    vertical_female_minor: Band

    def vertical_band(self, age: float, sex: str) -> Band:
        if age >= 18:
            return self.vertical_adult
        code = normalize_sex(sex)
        return self.vertical_male_minor if code == "M" else self.vertical_female_minor


THRESHOLD_SCHEMES: Mapping[str, ThresholdScheme] = {
    "wu2021_1.5sd": ThresholdScheme(
        "wu2021_1.5sd", Band(0.5, 5.0), Band(24.0, 38.0), Band(25.5, 38.0), Band(28.5, 41.0)
    ),
    "wu2021_1.0sd": ThresholdScheme(
        "wu2021_1.0sd", Band(1.0, 4.5), Band(26.5, 36.0), Band(27.5, 36.0), Band(30.5, 39.0)
    ),
    "wu2021_2.0sd": ThresholdScheme(
        "wu2021_2.0sd", Band(-0.5, 6.0), Band(22.0, 40.5), Band(23.5, 40.0), Band(26.0, 43.5)
    ),
    "abo": ThresholdScheme(
        "abo", Band(-2.0, 6.0), Band(26.0, 38.0), Band(26.0, 38.0), Band(26.0, 38.0)
    ),
}
PRIMARY_THRESHOLDS = THRESHOLD_SCHEMES["wu2021_1.5sd"]


def normalize_sex(value: str) -> str:
    code = str(value).strip().upper()
    if code in {"M", "MALE"}:
        return "M"
    if code in {"F", "FEMALE"}:
        return "F"
    raise ValueError(f"Unsupported sex value: {value!r}")


def age_band(age: float) -> str:
    if not np.isfinite(age):
        raise ValueError("Age must be finite")
    if age < 7:
        return "<7"
    if age <= 9:
        return "7-9"
    if age <= 12:
        return "10-12"
    if age <= 15:
        return "13-15"
    if age <= 17:
        return "16-17"
    return ">=18"


def age_stratum(age: float) -> str:
    if age < 7:
        return "<7"
    if age < 11:
        return "7-10"
    if age <= 30:
        return "11-30"
    return ">30"


def reference_is_eligible(age: float, sn_mp: float, bjork_sum: float | None = None) -> bool:
    if not np.isfinite((age, sn_mp)).all() or age < 7:
        return False
    return bjork_sum is None or (
        np.isfinite(bjork_sum) and abs(float(bjork_sum) - float(sn_mp) - 360.0) <= 0.1 + 1e-12
    )


def _classify(value: float, band: Band, classes: Sequence[str]) -> str:
    if not np.isfinite(value):
        raise ValueError("Classification values must be finite")
    if value < band.lower:
        return classes[0]
    if value > band.upper:
        return classes[2]
    return classes[1]


def label_case(
    anb: float,
    sn_mp: float,
    age: float,
    sex: str,
    scheme: str | ThresholdScheme = PRIMARY_THRESHOLDS,
) -> tuple[str, str]:
    selected = THRESHOLD_SCHEMES[scheme] if isinstance(scheme, str) else scheme
    return (
        _classify(float(anb), selected.sagittal, SAGITTAL_CLASSES),
        _classify(float(sn_mp), selected.vertical_band(float(age), sex), VERTICAL_CLASSES),
    )


def apply_thresholds(
    anb: Sequence[float] | np.ndarray,
    sn_mp: Sequence[float] | np.ndarray,
    age: Sequence[float] | np.ndarray,
    sex: Sequence[str] | np.ndarray,
    scheme: str | ThresholdScheme = PRIMARY_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.broadcast_arrays(
        np.asarray(anb, dtype=np.float64),
        np.asarray(sn_mp, dtype=np.float64),
        np.asarray(age, dtype=np.float64),
        np.asarray(sex, dtype=object),
    )
    sagittal = np.empty(arrays[0].shape, dtype=object)
    vertical = np.empty(arrays[0].shape, dtype=object)
    for index in np.ndindex(arrays[0].shape):
        sagittal[index], vertical[index] = label_case(
            arrays[0][index], arrays[1][index], arrays[2][index], arrays[3][index], scheme
        )
    return sagittal, vertical


@dataclass
class QualityControl:
    reasons: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def reject(self, reason: str, **details: object) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.details.update(details)

    def merge(self, other: "QualityControl", prefix: str = "") -> None:
        self.reasons.extend(f"{prefix}{reason}" for reason in other.reasons)
        self.details.update({f"{prefix}{key}": value for key, value in other.details.items()})


@dataclass(frozen=True)
class FrontalGeometry:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    eye_center: tuple[float, float]
    ipd: float
    roll_deg: float


@dataclass(frozen=True)
class ProfileGeometry:
    center: tuple[float, float]
    face_height: float
    facing: int
    nose_tip: tuple[float, float]


@dataclass
class NormalizedImage:
    image: np.ndarray | None
    transform: np.ndarray | None
    qc: QualityControl


@dataclass
class ProfileInput:
    image: np.ndarray | None
    silhouette: np.ndarray | None
    sdf: np.ndarray | None
    transform: np.ndarray | None
    qc: QualityControl


def downscale_for_detection(
    image: np.ndarray, max_side: int = DETECTION_MAX_SIDE
) -> tuple[np.ndarray, float]:
    import cv2

    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image, 1.0
    nominal_scale = max_side / max(height, width)
    resized = cv2.resize(
        image,
        (max(int(round(width * nominal_scale)), 1), max(int(round(height * nominal_scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, resized.shape[1] / width


def frontal_geometry(
    landmarks: np.ndarray,
    image_shape: Sequence[int],
    *,
    normalized: bool = True,
    max_roll_deg: float = 12.0,
) -> tuple[FrontalGeometry | None, QualityControl]:
    qc = QualityControl()
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] <= max(RIGHT_EYE_CORNERS):
        qc.reject("invalid_landmarks")
        return None, qc
    points = points[:, :2].copy()
    height, width = int(image_shape[0]), int(image_shape[1])
    if normalized:
        points *= np.array((width, height), dtype=np.float64)
    if points.shape[0] > max(LEFT_IRIS + RIGHT_IRIS):
        left = points[list(LEFT_IRIS)].mean(axis=0)
        right = points[list(RIGHT_IRIS)].mean(axis=0)
    else:
        left = points[list(LEFT_EYE_CORNERS)].mean(axis=0)
        right = points[list(RIGHT_EYE_CORNERS)].mean(axis=0)
    ipd = float(np.linalg.norm(left - right))
    if not np.isfinite(ipd) or ipd < 1e-6:
        qc.reject("invalid_ipd")
        return None, qc
    delta = left - right
    roll = float(np.degrees(np.arctan2(delta[1], delta[0])))
    roll = (roll + 180.0) % 180.0
    if roll > 90.0:
        roll -= 180.0
    ratio = ipd / width
    if abs(roll) > max_roll_deg:
        qc.reject("roll_out_of_range")
    if not 0.05 < ratio < 0.60:
        qc.reject("ipd_out_of_range")
    qc.details.update(ipd=ipd, roll_deg=roll, ipd_ratio=ratio)
    center = (left + right) / 2.0
    return FrontalGeometry(tuple(left), tuple(right), tuple(center), ipd, roll), qc


def similarity_transform(
    center: tuple[float, float],
    scale: float,
    output_size: int,
    *,
    target_x: float,
    target_y: float,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    cx, cy = center
    angle = np.radians(rotation_deg)
    a, b = scale * np.cos(angle), scale * np.sin(angle)
    return np.array(
        (
            (a, b, target_x * output_size - a * cx - b * cy),
            (-b, a, target_y * output_size + b * cx - a * cy),
        ),
        dtype=np.float64,
    )


def warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    output_size: int,
    *,
    categorical: bool = False,
) -> np.ndarray:
    import cv2

    if categorical:
        return cv2.warpAffine(
            image,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    scale = float(np.hypot(transform[0, 0], transform[0, 1]))
    return cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_frontal(
    image: np.ndarray,
    landmarks: np.ndarray,
    *,
    output_size: int = 384,
    ipd_ratio: float = 0.32,
    eye_y: float = 0.40,
    min_source_pixels: int = 400,
    max_roll_deg: float = 12.0,
    normalized_landmarks: bool = True,
) -> NormalizedImage:
    qc = QualityControl()
    if image.ndim != 3 or image.shape[2] != 3:
        qc.reject("invalid_image")
        return NormalizedImage(None, None, qc)
    if min(image.shape[:2]) < min_source_pixels:
        qc.reject("source_too_small")
        return NormalizedImage(None, None, qc)
    geometry, detected = frontal_geometry(
        landmarks, image.shape, normalized=normalized_landmarks, max_roll_deg=max_roll_deg
    )
    qc.merge(detected)
    if geometry is None or not qc.ok:
        return NormalizedImage(None, None, qc)
    scale = ipd_ratio * output_size / geometry.ipd
    transform = similarity_transform(
        geometry.eye_center,
        scale,
        output_size,
        target_x=0.5,
        target_y=eye_y,
        rotation_deg=-geometry.roll_deg,
    )
    qc.details["scale"] = scale
    return NormalizedImage(warp_image(image, transform, output_size), transform, qc)


def preprocess_frontal(
    image: np.ndarray,
    landmarker,
    *,
    output_size: int = 384,
    ipd_ratio: float = 0.32,
    eye_y: float = 0.40,
    min_source_pixels: int = 400,
    max_roll_deg: float = 12.0,
) -> NormalizedImage:
    qc = QualityControl()
    if image.ndim != 3 or image.shape[2] != 3:
        qc.reject("invalid_image")
        return NormalizedImage(None, None, qc)
    if min(image.shape[:2]) < min_source_pixels:
        qc.reject("source_too_small")
        return NormalizedImage(None, None, qc)
    detection_image, detection_scale = downscale_for_detection(image)
    try:
        landmarks = (
            mediapipe_landmarks(detection_image, landmarker)
            if hasattr(landmarker, "detect")
            else np.asarray(landmarker(detection_image), dtype=np.float64)
        )
    except ValueError:
        qc.reject("no_face_detected")
        return NormalizedImage(None, None, qc)
    result = normalize_frontal(
        image,
        landmarks,
        output_size=output_size,
        ipd_ratio=ipd_ratio,
        eye_y=eye_y,
        min_source_pixels=min_source_pixels,
        max_roll_deg=max_roll_deg,
    )
    result.qc.details["detection_scale"] = detection_scale
    return result


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = int(np.argmax(areas)) + 1
    return (labels == selected).astype(np.uint8), float(areas.max() / max(binary.sum(), 1))


def head_and_face_masks(
    category_mask: np.ndarray,
    *,
    minimum_component_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    import cv2

    categories = np.asarray(category_mask)
    if categories.ndim == 3 and categories.shape[-1] == 1:
        categories = categories[..., 0]
    if categories.ndim != 2:
        raise ValueError("The category mask must be two-dimensional")
    head, _ = _largest_component(np.isin(categories, HEAD_CLASSES))
    face = ((categories == FACE_SKIN) & (head > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
    if count <= 1:
        return head, np.zeros_like(face), 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= minimum_component_fraction * areas.max())[0] + 1
    merged = np.isin(labels, keep).astype(np.uint8)
    kept_fraction = float(areas[keep - 1].sum() / max(areas.sum(), 1))
    return head, merged, kept_fraction, int(len(keep))


def profile_geometry(
    category_mask: np.ndarray,
    image_shape: Sequence[int],
    *,
    minimum_foreground: float = 0.10,
    maximum_foreground: float = 0.85,
    maximum_face_components: int = 4,
    detection_scale: float | None = None,
) -> tuple[ProfileGeometry | None, QualityControl]:
    import cv2

    if not 0 <= minimum_foreground < maximum_foreground <= 1:
        raise ValueError("Foreground limits must be ordered within [0, 1]")
    if maximum_face_components < 1:
        raise ValueError("maximum_face_components must be positive")
    qc = QualityControl()
    categories = np.asarray(category_mask)
    if categories.ndim == 3 and categories.shape[-1] == 1:
        categories = categories[..., 0]
    if categories.ndim != 2:
        qc.reject("invalid_segmentation")
        return None, qc
    height, width = categories.shape
    foreground = float(np.isin(categories, PERSON_CLASSES).mean())
    if not minimum_foreground <= foreground <= maximum_foreground:
        qc.reject("foreground_out_of_range")
    head, face, kept, components = head_and_face_masks(categories)
    if face.sum() < 200:
        qc.reject("face_mask_too_small")
        return None, qc
    if components > maximum_face_components:
        qc.reject("face_mask_fragmented")
    y, x = np.nonzero(face)
    x0, x1, y0, y1 = int(x.min()), int(x.max()), int(y.min()), int(y.max())
    box_height, box_width = y1 - y0 + 1, x1 - x0 + 1
    aspect = box_width / box_height
    fill = float(face.sum() / (box_width * box_height))
    if not 0.35 <= aspect <= 1.15:
        qc.reject("face_aspect_out_of_range")
    if not 0.30 <= fill <= 0.90:
        qc.reject("face_fill_out_of_range")
    if not 0.08 < box_height / height < 0.95:
        qc.reject("face_height_out_of_range")
    if y0 <= 1 or y1 >= height - 2:
        qc.reject("face_touches_vertical_edge")
    hx = np.nonzero(head)[1]
    if not len(hx):
        qc.reject("head_mask_too_small")
        return None, qc
    facing = 1 if float(x.mean()) >= float(hx.mean()) else -1
    rows = np.arange(y0, y1 + 1)
    edge = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        columns = np.nonzero(face[row])[0]
        if columns.size:
            edge[index] = columns.max() if facing > 0 else columns.min()
    valid = np.isfinite(edge)
    if valid.sum() < 20:
        qc.reject("profile_edge_too_short")
        return None, qc
    kernel_size = max(3, int(round(valid.sum() * 0.02)) | 1)
    smoothed = cv2.GaussianBlur(edge[valid].astype(np.float32).reshape(-1, 1), (1, kernel_size), 0).ravel()
    tip = int(np.argmax(smoothed * facing))
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if detection_scale is None:
        sx, sy = image_width / width, image_height / height
    else:
        if not np.isfinite(detection_scale) or detection_scale <= 0:
            raise ValueError("detection_scale must be positive and finite")
        sx = sy = 1.0 / detection_scale
    qc.details.update(
        foreground_fraction=foreground,
        face_component_fraction=kept,
        face_components=components,
        aspect=aspect,
        fill=fill,
    )
    return (
        ProfileGeometry(
            ((x0 + x1) * 0.5 * sx, (y0 + y1) * 0.5 * sy),
            box_height * sy,
            facing,
            (float(smoothed[tip]) * sx, float(rows[valid][tip]) * sy),
        ),
        qc,
    )


def normalize_profile(
    image: np.ndarray,
    category_mask: np.ndarray,
    *,
    output_size: int = 384,
    face_height_ratio: float = 0.72,
    face_x: float = 0.44,
    min_source_pixels: int = 400,
    detection_scale: float | None = None,
    minimum_foreground: float = 0.10,
    maximum_foreground: float = 0.85,
    maximum_face_components: int = 4,
) -> NormalizedImage:
    qc = QualityControl()
    if image.ndim != 3 or image.shape[2] != 3:
        qc.reject("invalid_image")
        return NormalizedImage(None, None, qc)
    if min(image.shape[:2]) < min_source_pixels:
        qc.reject("source_too_small")
        return NormalizedImage(None, None, qc)
    geometry, detected = profile_geometry(
        category_mask,
        image.shape,
        minimum_foreground=minimum_foreground,
        maximum_foreground=maximum_foreground,
        maximum_face_components=maximum_face_components,
        detection_scale=detection_scale,
    )
    qc.merge(detected)
    if geometry is None or not qc.ok:
        return NormalizedImage(None, None, qc)
    scale = face_height_ratio * output_size / geometry.face_height
    target_x = face_x if geometry.facing > 0 else 1.0 - face_x
    transform = similarity_transform(
        geometry.center, scale, output_size, target_x=target_x, target_y=0.5
    )
    qc.details.update(scale=scale, facing=geometry.facing)
    return NormalizedImage(warp_image(image, transform, output_size), transform, qc)


def profile_silhouette(
    category_mask: np.ndarray,
    *,
    minimum_foreground: float = 0.10,
    maximum_foreground: float = 0.85,
    maximum_face_components: int = 4,
) -> tuple[np.ndarray | None, QualityControl]:
    import cv2

    if not 0 <= minimum_foreground < maximum_foreground <= 1:
        raise ValueError("Foreground limits must be ordered within [0, 1]")
    if maximum_face_components < 1:
        raise ValueError("maximum_face_components must be positive")
    qc = QualityControl()
    categories = np.asarray(category_mask)
    if categories.ndim == 3 and categories.shape[-1] == 1:
        categories = categories[..., 0]
    if categories.ndim != 2:
        qc.reject("invalid_segmentation")
        return None, qc
    foreground = float(np.isin(categories, PERSON_CLASSES).mean())
    if not minimum_foreground <= foreground <= maximum_foreground:
        qc.reject("foreground_out_of_range")
    head, face, kept, components = head_and_face_masks(categories)
    if head.sum() < 500:
        qc.reject("head_mask_too_small")
    if face.sum() < 200:
        qc.reject("face_mask_too_small")
    if components > maximum_face_components:
        qc.reject("face_mask_fragmented")
    if face.size and face[0].any():
        qc.reject("face_touches_top_edge")
    if face.size and face[-1].any():
        qc.reject("face_touches_bottom_edge")
    if face.size and face[:, 0].any() and face[:, -1].any():
        qc.reject("face_spans_side_edges")
    contours, _ = cv2.findContours(head.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        qc.reject("missing_head_contour")
        return None, qc
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(head, dtype=np.uint8)
    cv2.drawContours(filled, (contour,), -1, 255, thickness=cv2.FILLED)
    qc.details.update(
        foreground_fraction=foreground,
        face_component_fraction=kept,
        face_components=components,
        contour_points=int(len(contour)),
    )
    return (filled if qc.ok else None), qc


def signed_distance_field(mask: np.ndarray) -> np.ndarray:
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.min() == binary.max():
        raise ValueError("A signed distance field requires a non-constant two-dimensional mask")
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    signed = inside - outside
    limit = max(float(np.abs(signed).max()), 1.0)
    return np.clip(127.5 + 127.5 * signed / limit, 0, 255).astype(np.uint8)


def jpeg_round_trip(image: np.ndarray, quality: int = 95) -> np.ndarray:
    import cv2

    if not 0 <= quality <= 100:
        raise ValueError("JPEG quality must lie between zero and 100")
    encoded, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not encoded:
        raise ValueError("JPEG encoding failed")
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def preprocess_profile(
    image: np.ndarray,
    segmenter,
    *,
    output_size: int = 384,
    face_height_ratio: float = 0.72,
    face_x: float = 0.44,
    min_source_pixels: int = 400,
    jpeg_quality: int = 95,
    minimum_foreground: float = 0.10,
    maximum_foreground: float = 0.85,
    maximum_face_components: int = 4,
) -> ProfileInput:
    qc = QualityControl()
    if image.ndim != 3 or image.shape[2] != 3:
        qc.reject("invalid_image")
        return ProfileInput(None, None, None, None, qc)
    if min(image.shape[:2]) < min_source_pixels:
        qc.reject("source_too_small")
        return ProfileInput(None, None, None, None, qc)
    detection_image, detection_scale = downscale_for_detection(image)
    initial_mask = (
        mediapipe_categories(detection_image, segmenter)
        if hasattr(segmenter, "segment")
        else np.asarray(segmenter(detection_image))
    )
    normalized = normalize_profile(
        image,
        initial_mask,
        output_size=output_size,
        face_height_ratio=face_height_ratio,
        face_x=face_x,
        min_source_pixels=min_source_pixels,
        detection_scale=detection_scale,
        minimum_foreground=minimum_foreground,
        maximum_foreground=maximum_foreground,
        maximum_face_components=maximum_face_components,
    )
    normalized.qc.details["detection_scale"] = detection_scale
    if normalized.image is None:
        return ProfileInput(None, None, None, normalized.transform, normalized.qc)
    segmentation_image = jpeg_round_trip(normalized.image, jpeg_quality)
    crop_mask = (
        mediapipe_categories(segmentation_image, segmenter)
        if hasattr(segmenter, "segment")
        else np.asarray(segmenter(segmentation_image))
    )
    silhouette, mask_qc = profile_silhouette(
        crop_mask,
        minimum_foreground=minimum_foreground,
        maximum_foreground=maximum_foreground,
        maximum_face_components=maximum_face_components,
    )
    normalized.qc.merge(mask_qc, "silhouette_")
    if silhouette is None or not normalized.qc.ok:
        return ProfileInput(normalized.image, None, None, normalized.transform, normalized.qc)
    return ProfileInput(
        normalized.image,
        silhouette,
        signed_distance_field(silhouette),
        normalized.transform,
        normalized.qc,
    )


def mediapipe_bgr_image(bgr: np.ndarray):
    import mediapipe as mp
    import cv2

    rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def mediapipe_landmarks(bgr: np.ndarray, landmarker) -> np.ndarray:
    result = landmarker.detect(mediapipe_bgr_image(bgr))
    if not result.face_landmarks:
        raise ValueError("No frontal face was detected")
    return np.array([(point.x, point.y) for point in result.face_landmarks[0]], dtype=np.float64)


def mediapipe_categories(bgr: np.ndarray, segmenter) -> np.ndarray:
    result = segmenter.segment(mediapipe_bgr_image(bgr))
    mask = np.asarray(result.category_mask.numpy_view())
    return mask[..., 0] if mask.ndim == 3 else mask


def create_face_landmarker(
    model_path: str | Path,
    *,
    minimum_detection_score: float = 0.3,
):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not 0 <= minimum_detection_score <= 1:
        raise ValueError("minimum_detection_score must lie between zero and one")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(Path(model_path))),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=minimum_detection_score,
        min_face_presence_confidence=minimum_detection_score,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def create_profile_segmenter(model_path: str | Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(Path(model_path))),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)
