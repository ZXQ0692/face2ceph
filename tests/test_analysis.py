from __future__ import annotations

import numpy as np

from face2ceph.analysis import (
    ADDITIONAL_INPUT_REQUIREMENTS,
    GENERATOR_NOT_IN_RELEASE_REQUIREMENTS,
    aggregate_analysis_reports,
    analysis_status,
    bland_altman_report,
    boundary_analysis_report,
    cost_sensitive_report,
    shrinkage_report,
)
from face2ceph.preprocessing import SAGITTAL_CLASSES, VERTICAL_CLASSES, apply_thresholds
from face2ceph.targets import TARGETS


def _analysis_inputs(seed: int = 7):
    rng = np.random.default_rng(seed)
    target_mean = np.array((3.0, 0.0, 34.0, 25.0, 23.0, 66.0, 61.0, 54.5))
    target_sd = np.array((3.0, 4.0, 6.0, 6.0, 5.0, 5.0, 3.5, 2.0))

    def regression(cases: int):
        truth = target_mean + rng.normal(size=(cases, len(TARGETS))) * target_sd
        prediction = target_mean + 0.72 * (truth - target_mean) + rng.normal(
            scale=target_sd * 0.08, size=truth.shape
        )
        sigma = 0.3 + rng.random(truth.shape) * (0.2 + target_sd * 0.08)
        return truth, prediction, sigma

    training_truth, _, _ = regression(120)
    calibration_truth, calibration_prediction, calibration_sigma = regression(60)
    test_truth, test_prediction, test_sigma = regression(90)
    training_age = np.linspace(7, 50, len(training_truth))
    training_sex = np.where(np.arange(len(training_truth)) % 2, "F", "M")
    calibration_age = np.linspace(7, 50, len(calibration_truth))
    calibration_sex = np.where(np.arange(len(calibration_truth)) % 2, "F", "M")
    age = np.concatenate((np.linspace(7, 10, 30), np.linspace(11, 30, 30), np.linspace(31, 50, 30)))
    sex = np.where(np.arange(len(age)) % 2, "F", "M")
    calibration_sagittal, calibration_vertical = apply_thresholds(
        calibration_truth[:, 0], calibration_truth[:, 2], calibration_age, calibration_sex
    )
    sagittal, vertical = apply_thresholds(test_truth[:, 0], test_truth[:, 2], age, sex)

    def probabilities(labels: np.ndarray, classes: tuple[str, ...]):
        result = np.full((len(labels), len(classes)), 0.075)
        index = {name: position for position, name in enumerate(classes)}
        for row, label in enumerate(labels):
            result[row, index[str(label)]] = 0.85
        return result

    return {
        "training_truth": training_truth,
        "training_age": training_age,
        "training_sex": training_sex,
        "calibration_truth": calibration_truth,
        "calibration_prediction": calibration_prediction,
        "calibration_sigma": calibration_sigma,
        "calibration_sagittal_probabilities": probabilities(calibration_sagittal, SAGITTAL_CLASSES),
        "calibration_vertical_probabilities": probabilities(calibration_vertical, VERTICAL_CLASSES),
        "calibration_age": calibration_age,
        "calibration_sex": calibration_sex,
        "test_truth": test_truth,
        "test_prediction": test_prediction,
        "test_sigma": test_sigma,
        "sagittal_probabilities": probabilities(sagittal, SAGITTAL_CLASSES),
        "vertical_probabilities": probabilities(vertical, VERTICAL_CLASSES),
        "age": age,
        "sex": sex,
        "repeat_first": test_truth,
        "repeat_second": test_truth + rng.normal(scale=0.25, size=test_truth.shape),
        "cohort_truth": np.vstack((training_truth, calibration_truth, test_truth)),
    }


def test_bland_altman_slope_identity_and_limits():
    values = _analysis_inputs()
    report = bland_altman_report(values["test_truth"], values["test_prediction"])
    assert [row["target"] for row in report] == list(TARGETS)
    for row in report:
        assert np.isclose(row["shrinkage_coefficient"], 1.0 + row["slope_vs_reference"])
        assert row["loa_lower"] < row["bias"] < row["loa_upper"]
        assert 0 <= row["p_vs_reference"] <= 1


def test_shrinkage_coefficients_are_fit_only_on_calibration_data():
    calibration_truth = np.tile(np.arange(7, dtype=float)[:, None], (1, len(TARGETS)))
    calibration_prediction = 1.0 + 0.5 * calibration_truth
    test_truth = calibration_truth + 10.0
    test_prediction = 1.0 + 0.5 * test_truth
    report = shrinkage_report(
        calibration_truth,
        calibration_prediction,
        test_truth,
        test_prediction,
    )
    for target in TARGETS:
        assert np.isclose(report["coefficients"][target]["b"], 0.5)
        assert np.isclose(report["coefficients"][target]["mean_pred"], 2.5)
        assert np.isclose(report["splits"]["internal_test"]["targets"][target]["slope_raw"], 0.5)


def test_aggregate_reports_cover_every_supported_analysis():
    reports = aggregate_analysis_reports(
        **_analysis_inputs(),
        config="main",
        bootstrap_resamples=20,
    )
    assert set(reports) == {
        "bland_altman.json",
        "age_strata_main.json",
        "shrinkage_main.json",
        "conformal_adaptivity_main.json",
        "sigma_patient_level_main.json",
        "threshold_sensitivity_main.json",
        "posthoc_main.json",
        "cost_sensitive_main.json",
        "boundary_analysis.json",
        "analysis_status.json",
    }
    assert reports["age_strata_main.json"]["splits"]["internal_test"]["7-10"]["n"] == 30
    assert len(reports["conformal_adaptivity_main.json"]["per_target"]) == len(TARGETS)
    assert reports["sigma_patient_level_main.json"]["anb_sigma_quintiles"]["Q1 most confident"]["n"] == 18
    assert set(reports["threshold_sensitivity_main.json"]["splits"]["internal_test"]) == {
        *{"wu2021_1.5sd", "wu2021_1.0sd", "wu2021_2.0sd", "abo"},
        "_classification_head_reference",
    }
    assert reports["posthoc_main.json"]["analysis_status"] == "post_hoc"
    assert reports["cost_sensitive_main.json"]["protocol"].startswith("Tau is selected using calibration")
    assert reports["boundary_analysis.json"]["learning_curve_far"]["status"] == (
        "generator_not_in_release"
    )
    assert "minimum controlled bundle does not include" in (
        reports["boundary_analysis.json"]["learning_curve_far"]["input_availability"]
    )


def test_status_names_every_unavailable_intermediate():
    status = analysis_status(("bland_altman.json", "age_strata_stronger_backbone.json"))
    assert status["schema_version"] == 2
    assert set(status["requires_additional_input"]) == set(ADDITIONAL_INPUT_REQUIREMENTS)
    assert all(
        item["status"] == "unavailable_without_intermediate_data"
        for item in status["requires_additional_input"].values()
    )
    assert set(status["generator_not_in_release"]) == set(
        GENERATOR_NOT_IN_RELEASE_REQUIREMENTS
    )
    assert "reliability_summary.json" in status["generator_not_in_release"]
    assert "c0a_geometry_summary.json" in status["generator_not_in_release"]
    assert "boundary_learning_curve_far.json" in status["generator_not_in_release"]
    assert "learning_curve_fit.json" in status["requires_additional_input"]
    assert "learning_curve_fit.json" not in status["generator_not_in_release"]
    assert all(
        item["status"] == "generator_not_in_release"
        for item in status["generator_not_in_release"].values()
    )


def test_cost_sensitive_tau_is_independent_of_test_probabilities():
    values = _analysis_inputs()
    arguments = {
        key: values[key]
        for key in (
            "training_truth",
            "training_age",
            "training_sex",
            "calibration_truth",
            "calibration_sagittal_probabilities",
            "calibration_vertical_probabilities",
            "calibration_age",
            "calibration_sex",
            "test_truth",
            "sagittal_probabilities",
            "vertical_probabilities",
            "age",
            "sex",
        )
    }
    baseline = cost_sensitive_report(
        arguments.pop("training_truth"),
        arguments.pop("training_age"),
        arguments.pop("training_sex"),
        arguments.pop("calibration_truth"),
        arguments.pop("calibration_sagittal_probabilities"),
        arguments.pop("calibration_vertical_probabilities"),
        arguments.pop("calibration_age"),
        arguments.pop("calibration_sex"),
        arguments.pop("test_truth"),
        arguments.pop("sagittal_probabilities"),
        arguments.pop("vertical_probabilities"),
        arguments.pop("age"),
        arguments.pop("sex"),
    )
    altered = _analysis_inputs()
    altered["sagittal_probabilities"] = altered["sagittal_probabilities"][:, ::-1]
    altered["vertical_probabilities"] = altered["vertical_probabilities"][:, ::-1]
    changed = cost_sensitive_report(
        altered["training_truth"],
        altered["training_age"],
        altered["training_sex"],
        altered["calibration_truth"],
        altered["calibration_sagittal_probabilities"],
        altered["calibration_vertical_probabilities"],
        altered["calibration_age"],
        altered["calibration_sex"],
        altered["test_truth"],
        altered["sagittal_probabilities"],
        altered["vertical_probabilities"],
        altered["age"],
        altered["sex"],
    )
    assert baseline["sagittal"]["tau_star"] == changed["sagittal"]["tau_star"]
    assert baseline["vertical"]["tau_star"] == changed["vertical"]["tau_star"]


def test_boundary_ceiling_uses_paired_measurement_error():
    values = _analysis_inputs()
    report = boundary_analysis_report(
        values["test_truth"],
        values["sagittal_probabilities"],
        values["vertical_probabilities"],
        values["age"],
        values["sex"],
        repeat_first=values["repeat_first"],
        repeat_second=values["repeat_second"],
        cohort_truth=values["cohort_truth"],
    )
    expected = round(
        np.std(values["repeat_first"][:, 0] - values["repeat_second"][:, 0], ddof=1) / np.sqrt(2),
        4,
    )
    assert np.isclose(report["axes"]["sagittal"]["sd_measurement_error"], expected)
    assert report["axes"]["sagittal"]["label_noise_accuracy_ceiling"] < 1
    assert "sd_cohort" not in report["axes"]["sagittal"]
    assert np.isclose(
        report["axes"]["sagittal"]["sd_eligible_cohort"],
        np.std(values["cohort_truth"][:, 0], ddof=1),
    )
    assert report["frozen_full_source_sd"]["status"] == "unavailable_without_pre_eligibility_rows"
