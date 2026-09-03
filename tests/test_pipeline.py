from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from face2ceph.calibration import (
    ConformalCalibration,
    fit_split_conformal,
    interval_score,
    split_conformal_quantile,
)
from face2ceph.evaluation import (
    classification_metrics,
    evaluate_predictions,
    publication_evaluation_result,
    regression_metrics,
    stratified_report,
)
from face2ceph.partition import PartitionConfig, attach_frozen_partition, make_partition, validate_partition
from face2ceph.preprocessing import (
    TARGETS,
    THRESHOLD_SCHEMES,
    apply_thresholds,
    downscale_for_detection,
    label_case,
    normalize_frontal,
    normalize_profile,
    preprocess_frontal,
    preprocess_profile,
    profile_geometry,
    profile_silhouette,
    reference_is_eligible,
    signed_distance_field,
    warp_image,
)
from face2ceph.referral import (
    fit_referral_axis,
    fit_zscore_reference,
    measurement_discordance,
    selective_accuracy,
)


def test_thresholds_use_closed_bands_and_age_sex_strata() -> None:
    assert label_case(0.5, 28.5, 17, "F") == ("I", "Normo")
    assert label_case(5.0, 41.0, 17, "F") == ("I", "Normo")
    assert label_case(0.49, 28.49, 17, "F") == ("III", "Hypo")
    assert label_case(5.01, 41.01, 17, "F") == ("II", "Hyper")
    assert label_case(2.0, 39.0, 20, "F")[1] == "Hyper"
    sagittal, vertical = apply_thresholds(
        [0.0, 3.0, 6.0], [20.0, 30.0, 45.0], [20, 20, 20], ["M", "F", "F"]
    )
    assert sagittal.tolist() == ["III", "I", "II"]
    assert vertical.tolist() == ["Hypo", "Normo", "Hyper"]
    assert reference_is_eligible(7, 30, 390.1)
    assert not reference_is_eligible(6, 30, 390)
    assert not reference_is_eligible(20, 30, 390.11)


def test_threshold_schemes_match_reference_yaml() -> None:
    reference = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "reference" / "thresholds.yaml").read_text(encoding="utf-8")
    )
    assert reference["primary"] == "wu2021_1.5sd"
    assert set(THRESHOLD_SCHEMES) == set(reference["schemes"])
    for name, scheme in THRESHOLD_SCHEMES.items():
        source = reference["schemes"][name]
        sagittal = source["sagittal"]["bands"]["all"]
        assert (scheme.sagittal.lower, scheme.sagittal.upper) == (
            sagittal["lower"],
            sagittal["upper"],
        )
        vertical = source["vertical"]["bands"]
        if source["vertical"]["strata"]:
            bands = (vertical["adult"], vertical["minor_M"], vertical["minor_F"])
        else:
            bands = (vertical["all"],) * 3
        observed = (scheme.vertical_adult, scheme.vertical_male_minor, scheme.vertical_female_minor)
        assert [(band.lower, band.upper) for band in observed] == [
            (band["lower"], band["upper"]) for band in bands
        ]


def test_frontal_similarity_normalization() -> None:
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    landmarks = np.zeros((478, 2), dtype=np.float64)
    landmarks[[474, 475, 476, 477]] = (0.60, 0.40)
    landmarks[[469, 470, 471, 472]] = (0.40, 0.40)
    result = normalize_frontal(image, landmarks)
    assert result.qc.ok
    assert result.image is not None and result.image.shape == (384, 384, 3)
    eye_center = np.array((300.0, 320.0, 1.0))
    mapped = result.transform @ eye_center
    assert mapped == pytest.approx((192.0, 153.6))
    mapped_left = result.transform @ np.array((360.0, 320.0, 1.0))
    mapped_right = result.transform @ np.array((240.0, 320.0, 1.0))
    assert abs(mapped_left[0] - mapped_right[0]) == pytest.approx(0.32 * 384)


def test_direct_frontal_preprocessing_uses_detection_thumbnail() -> None:
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    detected_shapes = []

    def landmarker(detection_image: np.ndarray) -> np.ndarray:
        detected_shapes.append(detection_image.shape)
        landmarks = np.zeros((478, 2), dtype=np.float64)
        landmarks[[474, 475, 476, 477]] = (0.60, 0.40)
        landmarks[[469, 470, 471, 472]] = (0.40, 0.40)
        return landmarks

    result = preprocess_frontal(image, landmarker)
    assert result.qc.ok and result.image is not None
    assert max(detected_shapes[0][:2]) == 640
    assert result.qc.details["detection_scale"] == pytest.approx(detected_shapes[0][1] / 800)


def _profile_categories(height: int = 600, width: int = 400) -> np.ndarray:
    categories = np.zeros((height, width), dtype=np.uint8)
    categories[100:500, 110:300] = 1
    cv2.ellipse(categories, (250, 300), (45, 120), 0, 0, 360, 3, thickness=-1)
    return categories


def test_profile_normalization_silhouette_and_sdf() -> None:
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    categories = _profile_categories()
    normalized = normalize_profile(image, categories)
    assert normalized.qc.ok
    assert normalized.image is not None and normalized.image.shape == (384, 384, 3)
    crop_categories = warp_image(categories, normalized.transform, 384, categorical=True)
    silhouette, qc = profile_silhouette(crop_categories)
    assert qc.ok and silhouette is not None
    sdf = signed_distance_field(silhouette)
    assert sdf.shape == (384, 384) and sdf.dtype == np.uint8
    inside = np.argwhere(silhouette > 0)[len(np.argwhere(silhouette > 0)) // 2]
    outside = np.argwhere(silhouette == 0)[0]
    assert sdf[tuple(inside)] > 127
    assert sdf[tuple(outside)] < 127


def test_direct_profile_preprocessing_uses_uniform_restoration_and_two_segmentations() -> None:
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    segmented_shapes = []

    def segmenter(input_image: np.ndarray) -> np.ndarray:
        height, width = input_image.shape[:2]
        segmented_shapes.append((height, width))
        categories = np.zeros((height, width), dtype=np.uint8)
        categories[int(0.12 * height) : int(0.88 * height), int(0.22 * width) : int(0.78 * width)] = 1
        cv2.ellipse(
            categories,
            (int(0.65 * width), int(0.50 * height)),
            (max(int(0.12 * width), 2), max(int(0.22 * height), 2)),
            0,
            0,
            360,
            3,
            thickness=-1,
        )
        return categories

    result = preprocess_profile(image, segmenter)
    assert result.qc.ok and result.image is not None and result.sdf is not None
    assert segmented_shapes[0] == (640, 427)
    assert segmented_shapes[1] == (384, 384)
    expected_scale = 427 / 800
    assert result.qc.details["detection_scale"] == pytest.approx(expected_scale)
    thumbnail, scale = downscale_for_detection(image)
    geometry, qc = profile_geometry(segmenter(thumbnail), image.shape, detection_scale=scale)
    assert qc.ok and geometry is not None
    categories = segmenter(thumbnail)
    face_y = np.nonzero(categories == 3)[0]
    expected_height = (face_y.max() - face_y.min() + 1) / scale
    assert geometry.face_height == pytest.approx(expected_height)


def test_profile_rgb_is_retained_when_silhouette_qc_fails() -> None:
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    calls = 0

    def segmenter(input_image: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        height, width = input_image.shape[:2]
        if calls == 2:
            return np.zeros((height, width), dtype=np.uint8)
        categories = np.zeros((height, width), dtype=np.uint8)
        categories[int(0.12 * height) : int(0.88 * height), int(0.22 * width) : int(0.78 * width)] = 1
        cv2.ellipse(
            categories,
            (int(0.65 * width), int(0.50 * height)),
            (max(int(0.12 * width), 2), max(int(0.22 * height), 2)),
            0,
            0,
            360,
            3,
            thickness=-1,
        )
        return categories

    result = preprocess_profile(image, segmenter)
    assert result.image is not None
    assert result.silhouette is None and result.sdf is None
    assert not result.qc.ok


def test_partition_matches_frozen_design_and_is_deterministic() -> None:
    rows = []
    age_bands = ("7-9", "10-12", "13-15", "16-17", ">=18")
    case = 0
    for sex in ("F", "M"):
        for sagittal in ("III", "I", "II"):
            for vertical in ("Hypo", "Normo", "Hyper"):
                for band in age_bands:
                    for _ in range(10):
                        rows.append(
                            {
                                "case_id": f"case_{case:04d}",
                                "sex": sex,
                                "age_band": band,
                                "sagittal": sagittal,
                                "vertical": vertical,
                            }
                        )
                        case += 1
    frame = pd.DataFrame(rows)
    config = PartitionConfig()
    first = make_partition(frame, config)
    second = make_partition(frame, config)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert first.used_strata == ("sex", "sagittal", "vertical")
    assert first.dropped_strata == ("age_band",)
    assert first.assignments["split"].value_counts().to_dict() == {
        "train_cv": 648,
        "internal_test": 180,
        "calibration": 72,
    }
    training_folds = first.assignments.loc[first.assignments["split"] == "train_cv", "fold"]
    assert set(training_folds.astype(int)) == set(range(5))
    attached = attach_frozen_partition(frame, first.assignments)
    assert len(attached) == len(frame) and attached["split"].notna().all()


def test_partition_rejects_fractional_fold_identifiers() -> None:
    assignments = pd.DataFrame(
        {
            "case_id": [f"case_{index}" for index in range(7)],
            "split": ["train_cv"] * 5 + ["calibration", "internal_test"],
            "fold": [0, 1, 2, 3, 4.5, pd.NA, pd.NA],
        }
    )
    with pytest.raises(ValueError, match="valid fold"):
        validate_partition(assignments)


def test_split_conformal_uses_finite_sample_quantile() -> None:
    residual = np.array([1.0, 2.0, 3.0, 4.0])
    assert split_conformal_quantile(residual, np.ones(4), 0.10) == 4.0
    truth = np.column_stack((residual, residual * 2))
    mean = np.zeros_like(truth)
    sigma = np.ones_like(truth)
    fitted = fit_split_conformal(truth, mean, sigma, targets=("a", "b"))
    assert fitted.quantiles.tolist() == [4.0, 8.0]
    assert fitted.coverage(truth, mean, sigma).tolist() == [1.0, 1.0]
    lower, upper = fitted.intervals(mean, sigma)
    assert np.all(interval_score(truth, lower, upper) >= upper - lower)


def test_conformal_state_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ConformalCalibration(0.1, ("target",), np.array([-1.0]), 4)
    with pytest.raises(ValueError, match="alpha"):
        interval_score(np.ones(1), np.zeros(1), np.ones(1), alpha=0)


def test_referral_uses_confidence_and_direction_aligned_discordance() -> None:
    base = np.arange(8, dtype=np.float64)
    training = np.stack([base + multiplier * np.arange(1, 9) for multiplier in range(6)])
    sex = ["F"] * len(training)
    age = [20] * len(training)
    reference = fit_zscore_reference(training, sex, age)
    z_scores = reference.transform(training, sex, age)
    raw_jarabak = (training[:, 5] - training[:, 5].mean()) / training[:, 5].std(ddof=1)
    assert z_scores[:, 5] == pytest.approx(-raw_jarabak)
    sagittal, vertical = measurement_discordance(z_scores)
    assert np.isfinite(sagittal).all() and np.isfinite(vertical).all()
    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.75, 0.15, 0.10],
            [0.60, 0.25, 0.15],
            [0.45, 0.35, 0.20],
            [0.34, 0.33, 0.33],
            [0.80, 0.10, 0.10],
        ]
    )
    fitted = fit_referral_axis(probabilities, sagittal, rates=(0.20,))
    referred = fitted.refer(probabilities, sagittal, 0.20)
    assert referred.shape == (6,)
    summary = selective_accuracy(np.array([1, 1, 1, 0, 0, 1], dtype=bool), referred)
    assert summary["n_referred"] == int(referred.sum())
    assert summary["accuracy_all"] == pytest.approx(4 / 6)
    with pytest.raises(ValueError, match="non-empty and unique"):
        fit_referral_axis(probabilities, sagittal, rates=())


def test_primary_evaluation_metrics() -> None:
    truth = np.arange(48, dtype=np.float64).reshape(6, 8)
    prediction = truth + 1.0
    sigma = np.ones_like(truth)
    calibration = fit_split_conformal(truth[:4], prediction[:4], sigma[:4], targets=TARGETS)
    sagittal_truth = np.array(("III", "I", "II", "III", "I", "II"))
    vertical_truth = np.array(("Hypo", "Normo", "Hyper", "Hypo", "Normo", "Hyper"))
    sagittal_probability = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.6, 0.2, 0.2],
        ]
    )
    vertical_probability = sagittal_probability.copy()
    metrics = regression_metrics(truth[:, 0], prediction[:, 0])
    assert metrics["MAE"] == 1.0 and metrics["RMSE"] == 1.0
    classification = classification_metrics(
        sagittal_truth,
        sagittal_probability,
        classes=("III", "I", "II"),
        bootstrap_resamples=40,
        ece_minimum_bin_size=1,
    )
    assert classification["accuracy"] == pytest.approx(5 / 6)
    assert classification["balanced_accuracy"] == pytest.approx(5 / 6)
    report = evaluate_predictions(
        truth,
        prediction,
        sigma,
        sagittal_truth,
        sagittal_probability,
        vertical_truth,
        vertical_probability,
        conformal=calibration,
        bootstrap_resamples=40,
    )
    assert report["n"] == 6
    assert report["regression"]["ANB"]["MAE"] == 1.0
    assert report["conformal"]["ANB"]["coverage"] == 1.0


def test_publication_evaluation_contract_preserves_lower_level_report() -> None:
    count = 36
    truth = np.arange(count * 8, dtype=np.float64).reshape(count, 8)
    prediction = truth + 1.0
    sigma = np.ones_like(truth)
    sagittal_truth = np.resize(np.array(("III", "I", "II")), count)
    vertical_truth = np.resize(np.array(("Hypo", "Normo", "Hyper")), count)
    probabilities = np.eye(3, dtype=np.float64)[np.arange(count) % 3]
    calibration = fit_split_conformal(truth, prediction, sigma, targets=TARGETS)
    lower_level = evaluate_predictions(
        truth,
        prediction,
        sigma,
        sagittal_truth,
        probabilities,
        vertical_truth,
        probabilities,
        conformal=calibration,
        bootstrap_resamples=10,
    )
    lower_level["referral"] = {
        "sagittal": {
            "0.2": {
                "actual_rate": 0.25,
                "n_referred": 9,
                "n_retained": 27,
                "accuracy_all": 1.0,
                "accuracy_retained": 1.0,
                "accuracy_referred": 1.0,
            }
        }
    }
    stratified = stratified_report(
        truth,
        prediction,
        sagittal_truth,
        probabilities,
        vertical_truth,
        probabilities,
        np.repeat("F", count),
        np.repeat(20.0, count),
    )
    published = publication_evaluation_result(lower_level, split="internal_test", stratified=stratified)

    assert "conformal" in lower_level
    assert "confusion_matrix" in lower_level["classification"]["sagittal"]
    assert published["split"] == "internal_test"
    assert "conformal_coverage" in published and "conformal" not in published
    assert published["conformal_coverage"]["ANB"]["mean_halfwidth"] == 1.0
    assert published["confusion"]["sagittal"] == np.diag([12, 12, 12]).tolist()
    assert "sens_ci" in published["classification"]["sagittal"]["per_class"]["III"]
    assert published["stratified"]["sex"]["F"]["MAE_ANB"] == 1.0
    assert published["referral"]["sagittal"] == {
        "baseline_accuracy": 1.0,
        "operating_points": [
            {
                "target_rate": 0.2,
                "actual_rate": 0.25,
                "n_kept": 27,
                "accuracy_kept": 1.0,
                "accuracy_referred": 1.0,
            }
        ],
    }
