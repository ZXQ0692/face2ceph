from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import face2ceph.analysis as analysis_module
import face2ceph.cli as cli_module
import face2ceph.inference as inference_module
import face2ceph.training as training_module
from face2ceph.cli import ARM_NAMES, CONFIG_ROOT, _augmentation_config, _model_config, _pipeline, _training_config
from face2ceph.targets import TARGETS
from face2ceph.training import EpochValidationMetrics, FoldTrainingResult


def test_all_declared_arms_map_to_runtime_configuration() -> None:
    expected = {
        "classification_rgb",
        "classification_shape",
        "frontal_only",
        "learning_10",
        "learning_25",
        "learning_50",
        "learning_75",
        "main",
        "multitask",
        "profile_only",
        "silhouette_only",
        "stronger_backbone",
    }
    assert set(ARM_NAMES) == expected
    for arm in ARM_NAMES:
        values = _pipeline(CONFIG_ROOT / "pipeline.yaml", arm)
        model = _model_config(values, Path("weights.safetensors"))
        training = _training_config(values)
        augmentation = _augmentation_config(values)
        assert model.input_mode in {"both", "frontal", "profile", "silhouette"}
        assert training.fold_count == 5
        assert augmentation.rotation_degrees == 5.0


def test_arm_semantics_are_explicit() -> None:
    rgb = _pipeline(CONFIG_ROOT / "pipeline.yaml", "classification_rgb")
    shape = _pipeline(CONFIG_ROOT / "pipeline.yaml", "classification_shape")
    main = _pipeline(CONFIG_ROOT / "pipeline.yaml", "main")
    assert rgb["model"]["use_profile_sdf"] is False and rgb["model"]["regression"] == "none"
    assert shape["model"]["use_profile_sdf"] is True and shape["model"]["regression"] == "none"
    assert main["model"]["regression"] == "heteroscedastic"


def test_train_writes_schema_ready_epoch_history(monkeypatch, tmp_path: Path) -> None:
    config = _pipeline(CONFIG_ROOT / "pipeline.yaml", "learning_25")
    cohort = pd.DataFrame({"usable": [True], "analyzed": [True]})
    results = [
        FoldTrainingResult(
            fold=fold,
            best_epoch=0,
            selection_metric="mae",
            validation_score=1.5,
            validation_mae=1.5,
            validation_balanced_accuracy=0.75,
            training_cases=100 + fold,
            validation_cases=20,
            checkpoint_path=tmp_path / f"fold_{fold}.pt",
            epoch_history=(EpochValidationMetrics(0, 1.5, 0.7, 0.8),),
        )
        for fold in range(5)
    ]
    destination = tmp_path / "run"
    captured = {}
    monkeypatch.setattr(cli_module, "_pipeline", lambda *args: config)
    monkeypatch.setattr(cli_module, "_read_cohort", lambda *args: cohort)
    monkeypatch.setattr(cli_module, "_read_partition", lambda *args: None)
    monkeypatch.setattr(cli_module, "input_path", lambda value, kind=None: Path(value))
    monkeypatch.setattr(
        cli_module,
        "_asset_specs",
        lambda: {"backbone": {"name": config["model"]["image_backbone"]}},
    )
    monkeypatch.setattr(cli_module, "_verified_asset", lambda *args: tmp_path / "backbone.pt")
    monkeypatch.setattr(cli_module, "output_path", lambda value: destination)
    monkeypatch.setattr(training_module, "train_five_fold_ensemble", lambda *args, **kwargs: results)
    monkeypatch.setattr(
        cli_module,
        "write_json",
        lambda path, payload: captured.update({"path": path, "payload": payload}),
    )

    status = cli_module._cmd_train(
        Namespace(
            config="pipeline.yaml",
            arm="learning_25",
            cohort="cohort.csv",
            partition="partition.csv",
            image_root="images",
            output_dir="run",
            backbone_weights=None,
            device=None,
        )
    )

    assert status == 0
    assert captured["path"] == destination / "validation_history.json"
    history = captured["payload"]["learning_25"]
    assert history["fraction"] == 0.25
    assert history["selection_criterion"] == "mae_mean"
    assert len(history["folds"]) == 5


def test_evaluate_writes_reference_compatible_envelope(monkeypatch) -> None:
    count = 60
    truth = np.arange(count, dtype=np.float64)[:, None] + np.arange(8, dtype=np.float64)[None, :]
    labels = np.arange(count) % 3
    probabilities = np.eye(3, dtype=np.float64)[labels]
    frame = pd.DataFrame({target: truth[:, index] for index, target in enumerate(TARGETS)})
    frame["sagittal"] = np.asarray(("III", "I", "II"), dtype=object)[labels]
    frame["vertical"] = np.asarray(("Hypo", "Normo", "Hyper"), dtype=object)[labels]
    frame["sex"] = np.repeat(("F", "M"), count // 2)
    frame["age"] = np.repeat((20.0, 35.0), count // 2)
    arrays = {
        "mu": truth + 0.5,
        "sigma": np.ones_like(truth),
        "prob_sag": probabilities,
        "prob_vert": probabilities,
    }
    config = {"name": "main", "conformal": {"alpha": 0.1}}
    captured = {}
    monkeypatch.setattr(cli_module, "_pipeline", lambda *args: config)
    monkeypatch.setattr(cli_module, "_aligned_predictions", lambda *args: (arrays, frame))
    monkeypatch.setattr(cli_module, "write_json", lambda path, payload: captured.update(payload))

    status = cli_module._cmd_evaluate(
        Namespace(
            config="pipeline.yaml",
            arm="main",
            predictions="predictions.npz",
            cohort="cohort.csv",
            partition="partition.csv",
            split="internal_test",
            conformal=None,
            referral=None,
            bootstrap_resamples=4,
            output="evaluation.json",
        )
    )

    assert status == 0
    assert set(captured) == {"config", "alpha", "n_boot", "results", "protocol"}
    assert captured["config"] == "main" and captured["alpha"] == 0.1 and captured["n_boot"] == 4
    result = captured["results"][0]
    assert result["split"] == "internal_test" and result["n"] == count
    assert set(result["stratified"]["sex"]) == {"F", "M"}
    assert set(result["stratified"]["age_stratum"]) == {"11-30", ">30"}
    assert result["confusion"]["sagittal"] == np.diag([20, 20, 20]).tolist()


def test_analyze_passes_separate_training_calibration_and_test_inputs(monkeypatch) -> None:
    target_values = np.arange(8, dtype=np.float64)

    def frame(prefix: str, count: int) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "case_id": [f"{prefix}{index:03d}" for index in range(count)],
                "age": np.linspace(12, 35, count),
                "sex": np.where(np.arange(count) % 2, "F", "M"),
            }
        )
        for index, target in enumerate(TARGETS):
            result[target] = index + np.arange(count, dtype=np.float64)
        return result

    calibration_frame = frame("C", 12)
    test_frame = frame("T", 15)
    training_frame = frame("R", 18)
    cohort = pd.concat((training_frame, calibration_frame, test_frame), ignore_index=True)
    cohort["analyzed"] = True
    cohort.loc[0, "analyzed"] = False
    cohort["tracer_1"] = "A"
    cohort["tracer_2"] = "B"
    for target in TARGETS:
        cohort[f"{target}_t2"] = cohort[target] + 0.1
    partition = pd.DataFrame(
        {
            "case_id": cohort["case_id"],
            "split": (["train_cv"] * len(training_frame) + ["calibration"] * len(calibration_frame) + ["internal_test"] * len(test_frame)),
            "fold": ([0] * len(training_frame) + [-1] * (len(calibration_frame) + len(test_frame))),
        }
    )

    def arrays(count: int) -> dict[str, np.ndarray]:
        probabilities = np.tile(np.array((0.2, 0.6, 0.2)), (count, 1))
        return {
            "mu": np.tile(target_values, (count, 1)),
            "sigma": np.ones((count, len(TARGETS))),
            "prob_sag": probabilities,
            "prob_vert": probabilities,
        }

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "_pipeline",
        lambda *args: {"name": "main", "conformal": {"alpha": 0.1}},
    )
    monkeypatch.setattr(cli_module, "_read_cohort", lambda *args: cohort)
    monkeypatch.setattr(cli_module, "_read_partition", lambda *args: partition)
    monkeypatch.setattr(
        cli_module,
        "_load_predictions",
        lambda path: arrays(len(calibration_frame)) if "calibration" in path else arrays(len(test_frame)),
    )
    monkeypatch.setattr(
        cli_module,
        "_select_analysis_frame",
        lambda cohort, partition, config, split: (
            training_frame
            if split == "train_cv"
            else calibration_frame
            if split == "calibration"
            else test_frame
        ),
    )
    monkeypatch.setattr(cli_module, "_align_prediction_arrays", lambda arrays, frame, split: (arrays, frame))
    monkeypatch.setattr(
        analysis_module,
        "aggregate_analysis_reports",
        lambda **kwargs: captured.update(kwargs) or {"bland_altman.json": {}, "analysis_status.json": {}},
    )
    monkeypatch.setattr(
        analysis_module,
        "write_aggregate_reports",
        lambda output, reports: captured.update({"output": output, "reports": reports}),
    )
    status = cli_module._cmd_analyze(
        Namespace(
            config="pipeline.yaml",
            arm="main",
            cohort="cohort.csv",
            partition="partition.csv",
            calibration_predictions="calibration.npz",
            test_predictions="test.npz",
            output_dir="aggregate",
            alpha=None,
            bootstrap_resamples=8,
        )
    )
    assert status == 0
    assert captured["training_truth"].shape == (18, len(TARGETS))
    assert captured["training_truth"][0, 0] == training_frame.loc[0, TARGETS[0]]
    assert captured["calibration_truth"].shape == (12, len(TARGETS))
    assert captured["test_truth"].shape == (15, len(TARGETS))
    assert captured["repeat_second"].shape == (15, len(TARGETS))
    assert captured["cohort_truth"].shape == (45, len(TARGETS))
    assert captured["output"] == "aggregate"


def test_analyze_parser_keeps_prediction_and_reproduction_options() -> None:
    parser = cli_module._parser()
    analyze = parser.parse_args(
        [
            "analyze",
            "--cohort",
            "cohort.csv",
            "--calibration-predictions",
            "calibration.npz",
            "--test-predictions",
            "test.npz",
            "--output-dir",
            "analysis",
        ]
    )
    predict = parser.parse_args(
        [
            "predict",
            "--cohort",
            "cohort.csv",
            "--partition",
            "partition.csv",
            "--image-root",
            "images",
            "--checkpoints",
            "weights",
            "--split",
            "internal_test",
            "--output",
            "predictions.npz",
            "--arm",
            "profile_only",
            "--include-features",
            "--perturbation",
            "jpeg_60",
        ]
    )
    reproduce = parser.parse_args(
        ["reproduce", "--data-dir", "controlled", "--show-values"]
    )
    assert analyze.handler is cli_module._cmd_analyze
    assert analyze.partition is None
    assert predict.arm == "profile_only"
    assert predict.include_features and predict.perturbation == "jpeg_60"
    assert reproduce.show_values


def test_predict_uses_the_selected_native_arm(monkeypatch, tmp_path: Path) -> None:
    config = _pipeline(CONFIG_ROOT / "pipeline.yaml", "profile_only")
    cohort = pd.DataFrame({"usable": [True], "analyzed": [True]})
    captured = {}

    def selected_pipeline(path, arm):
        captured["arm"] = arm
        return config

    def fake_predict(*args, **kwargs):
        captured.update(kwargs)
        return Namespace(case_id=np.asarray(["case"]))

    monkeypatch.setattr(cli_module, "_pipeline", selected_pipeline)
    monkeypatch.setattr(cli_module, "_read_cohort", lambda *args: cohort)
    monkeypatch.setattr(cli_module, "_read_partition", lambda *args: None)
    monkeypatch.setattr(cli_module, "input_path", lambda value, kind=None: Path(value))
    monkeypatch.setattr(cli_module, "output_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        inference_module,
        "fold_checkpoint_paths",
        lambda *args: [tmp_path / f"fold_{fold}.pt" for fold in range(5)],
    )
    monkeypatch.setattr(inference_module, "predict_ensemble", fake_predict)
    monkeypatch.setattr(inference_module, "save_prediction_archive", lambda *args: None)

    result = cli_module._cmd_predict(
        Namespace(
            config="pipeline.yaml",
            arm="profile_only",
            cohort="cohort.csv",
            partition="partition.csv",
            image_root="images",
            checkpoints="checkpoints",
            split="internal_test",
            output="predictions.npz",
            device=None,
            include_features=False,
            perturbation=None,
        )
    )

    assert result == 0
    assert captured["arm"] == "profile_only"
    assert captured["expected_arm"] == "profile_only"
    assert captured["expected_model_config"].input_mode == "profile"


def test_analysis_frame_filters_the_arm_availability_population() -> None:
    cohort = pd.DataFrame(
        {
            "case_id": ["A1", "A2", "A3", "A4"],
            "analyzed": [True, False, True, False],
            "usable": [True, True, True, True],
        }
    )
    partition = pd.DataFrame(
        {
            "case_id": cohort["case_id"],
            "split": ["train_cv", "train_cv", "internal_test", "internal_test"],
            "fold": [0, 1, pd.NA, pd.NA],
        }
    )
    config = {"model": {"use_profile_sdf": True}}
    training = cli_module._select_analysis_frame(cohort, partition, config, "train_cv")
    test = cli_module._select_analysis_frame(cohort, partition, config, "internal_test")
    assert training["case_id"].tolist() == ["A1"]
    assert test["case_id"].tolist() == ["A3"]
