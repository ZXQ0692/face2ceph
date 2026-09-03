from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import torch

import face2ceph.inference as inference_module
from face2ceph.dataset import TargetScaler
from face2ceph.inference import InferenceConfig, fold_checkpoint_paths, predict_ensemble
from face2ceph.model import ModelConfig
from face2ceph.training import TrainingConfig


def test_fold_checkpoint_paths_supports_each_complete_convention(tmp_path) -> None:
    native = tmp_path / "native"
    historical = tmp_path / "historical"
    native.mkdir()
    historical.mkdir()
    for fold in range(5):
        (native / f"fold_{fold}.pt").write_bytes(b"checkpoint")
        (historical / f"fold{fold}.pt").write_bytes(b"checkpoint")

    assert [path.name for path in fold_checkpoint_paths(native)] == [f"fold_{fold}.pt" for fold in range(5)]
    assert [path.name for path in fold_checkpoint_paths(historical)] == [f"fold{fold}.pt" for fold in range(5)]


def test_fold_checkpoint_paths_rejects_mixed_conventions(tmp_path) -> None:
    for fold in range(5):
        (tmp_path / f"fold_{fold}.pt").write_bytes(b"checkpoint")
    (tmp_path / "fold0.pt").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="mixes"):
        fold_checkpoint_paths(tmp_path)


def test_native_checkpoint_paths_honor_declared_fold_count(tmp_path) -> None:
    for fold in range(3):
        (tmp_path / f"fold_{fold}.pt").write_bytes(b"checkpoint")

    assert [path.name for path in fold_checkpoint_paths(tmp_path, 3)] == [
        "fold_0.pt",
        "fold_1.pt",
        "fold_2.pt",
    ]


def test_historical_checkpoint_paths_require_five_folds(tmp_path) -> None:
    for fold in range(5):
        (tmp_path / f"fold{fold}.pt").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="exactly five"):
        fold_checkpoint_paths(tmp_path, 3)


def test_historical_translation_has_exact_declared_prefix_set() -> None:
    historical = {
        source + "weight": torch.full((1,), index, dtype=torch.float32)
        for index, (source, _) in enumerate(inference_module._HISTORICAL_PREFIXES)
    }
    historical["neck.0.weight"] = torch.ones(1)

    translated = inference_module._translate_historical_state_dict(historical)

    assert len(inference_module._HISTORICAL_PREFIXES) == 7
    assert set(translated) == {
        target + "weight" for _, target in inference_module._HISTORICAL_PREFIXES
    } | {"neck.0.weight"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("config", "c3", "historical c4b"),
        ("seed", 43, "fold seed"),
        ("metric", "balanced_accuracy", "mae_mean"),
    ),
)
def test_historical_checkpoint_rejects_inconsistent_fields(field, value, message) -> None:
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "epoch": 3,
        "fold": 0,
        "seed": 42,
        "config": "c4b",
        "metric": "mae_mean",
        "value": 1.0,
        "scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
    }
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        inference_module._normalize_checkpoint(
            payload,
            convention="historical",
            expected_fold=0,
            expected_model_config=ModelConfig(),
            expected_training_config=asdict(TrainingConfig()),
            expected_arm="main",
        )


def test_historical_checkpoint_accepts_runtime_inference_overrides() -> None:
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "epoch": 3,
        "fold": 0,
        "seed": 42,
        "config": "c4b",
        "metric": "mae_mean",
        "value": 1.0,
        "scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
    }
    declared_config = replace(
        TrainingConfig(),
        batch_size=2,
        num_workers=0,
        mixed_precision=False,
        channels_last=False,
    )

    member = inference_module._normalize_checkpoint(
        payload,
        convention="historical",
        expected_fold=0,
        expected_model_config=ModelConfig(),
        expected_training_config=asdict(declared_config),
        expected_arm="main",
    )

    assert member.envelope == "historical"


def test_native_checkpoint_accepts_a_matching_non_main_arm() -> None:
    model_config = ModelConfig(input_mode="profile")
    training_config = TrainingConfig()
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 0,
        "seed": 42,
    }

    member = inference_module._normalize_checkpoint(
        payload,
        convention="native",
        expected_fold=0,
        expected_model_config=model_config,
        expected_training_config=asdict(training_config),
        expected_arm="profile_only",
    )

    assert member.envelope == "native"
    assert member.model_config["input_mode"] == "profile"


def test_native_checkpoint_rejects_declared_arm_model_mismatch() -> None:
    training_config = TrainingConfig()
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "model_config": asdict(ModelConfig()),
        "training_config": asdict(training_config),
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 0,
        "seed": training_config.seed,
    }

    with pytest.raises(ValueError, match="declared release arm"):
        inference_module._normalize_checkpoint(
            payload,
            convention="native",
            expected_fold=0,
            expected_model_config=ModelConfig(input_mode="profile"),
            expected_training_config=asdict(training_config),
            expected_arm="profile_only",
        )


def test_native_checkpoint_accepts_runtime_inference_overrides() -> None:
    checkpoint_config = TrainingConfig()
    declared_config = replace(
        checkpoint_config,
        batch_size=2,
        num_workers=0,
        mixed_precision=False,
        channels_last=False,
    )
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "model_config": asdict(ModelConfig()),
        "training_config": asdict(checkpoint_config),
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 0,
        "seed": checkpoint_config.seed,
    }

    member = inference_module._normalize_checkpoint(
        payload,
        convention="native",
        expected_fold=0,
        expected_model_config=ModelConfig(),
        expected_training_config=asdict(declared_config),
        expected_arm="main",
    )

    assert member.training_config == asdict(checkpoint_config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 7),
        ("fold_count", 3),
        ("image_size", 256),
        ("subset_fraction", 0.5),
        ("learning_rate", 2e-4),
    ),
)
def test_native_checkpoint_rejects_training_contract_mismatch(field, value) -> None:
    checkpoint_config = TrainingConfig()
    declared_config = replace(checkpoint_config, **{field: value})
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "model_config": asdict(ModelConfig()),
        "training_config": asdict(checkpoint_config),
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 0,
        "seed": checkpoint_config.seed,
    }

    with pytest.raises(ValueError, match="training configuration"):
        inference_module._normalize_checkpoint(
            payload,
            convention="native",
            expected_fold=0,
            expected_model_config=ModelConfig(),
            expected_training_config=asdict(declared_config),
            expected_arm="main",
        )


def test_native_checkpoint_accepts_non_five_fold_contract() -> None:
    training_config = TrainingConfig(fold_count=3)
    payload = {
        "model": {"neck.0.weight": torch.ones(1)},
        "model_config": asdict(ModelConfig()),
        "training_config": asdict(training_config),
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 2,
        "seed": training_config.seed + 2000,
    }

    member = inference_module._normalize_checkpoint(
        payload,
        convention="native",
        expected_fold=2,
        expected_model_config=ModelConfig(),
        expected_training_config=asdict(training_config),
        expected_arm="main",
    )

    assert member.fold == 2
    assert member.training_config["fold_count"] == 3


def test_historical_c4b_ensemble_is_loaded_strictly_in_memory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strict_calls: list[bool] = []

    class FakeModel:
        def to(self, *_: object, **__: object) -> "FakeModel":
            return self

        def load_state_dict(self, state, *, strict: bool) -> None:
            strict_calls.append(strict)
            assert "frontal_backbone.weight" in state
            assert "regression_head.log_variance.bias" in state

    checkpoints = [tmp_path / f"fold{fold}.pt" for fold in range(5)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    scaler = TargetScaler(np.arange(8.0), np.arange(1.0, 9.0))
    historical_state = {
        "backbone_f.weight": torch.ones(1),
        "backbone_p.weight": torch.ones(1),
        "film.net.0.weight": torch.ones(1),
        "head_reg.mu.weight": torch.ones(1),
        "head_reg.logvar.bias": torch.ones(1),
        "head_sag.weight": torch.ones(1),
        "head_vert.weight": torch.ones(1),
        "neck.0.weight": torch.ones(1),
    }

    def fake_load(path, **_: object):
        fold = int(path.stem.removeprefix("fold"))
        return {
            "model": historical_state,
            "epoch": 3,
            "fold": fold,
            "seed": 42 + 1000 * fold,
            "config": "c4b",
            "metric": "mae_mean",
            "value": 1.0,
            "scaler": scaler.to_dict(),
        }

    manifest = pd.DataFrame({"case_id": ["a", "b"], "analyzed": [True, True]})
    member = (
        np.zeros((2, 8)),
        np.zeros((2, 8)),
        np.full((2, 3), 1.0 / 3.0),
        np.full((2, 3), 1.0 / 3.0),
        None,
    )
    monkeypatch.setattr(inference_module.torch, "load", fake_load)
    monkeypatch.setattr(inference_module, "load_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(inference_module, "ClinicalPhotoDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(inference_module, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(inference_module, "FaceToCephalometryModel", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(inference_module, "_predict_member", lambda *args, **kwargs: member)

    predictions = predict_ensemble(
        "cohort.csv",
        ".",
        checkpoints,
        inference_config=InferenceConfig(num_workers=0, mixed_precision=False, channels_last=False),
        device="cpu",
        expected_model_config=ModelConfig(),
        expected_training_config=asdict(TrainingConfig()),
        expected_arm="main",
    )

    assert strict_calls == [True] * 5
    np.testing.assert_allclose(predictions.mu, np.broadcast_to(scaler.mean, (2, 8)))
    np.testing.assert_allclose(predictions.sigma, np.broadcast_to(scaler.std, (2, 8)))
