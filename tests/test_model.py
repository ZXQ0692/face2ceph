from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import face2ceph.dataset as dataset_module
import face2ceph.inference as inference_module
import face2ceph.workspace as workspace_module
from face2ceph.dataset import ClinicalPhotoDataset, TargetScaler, load_manifest
from face2ceph.inference import (
    InferenceConfig,
    PredictionSet,
    combine_member_predictions,
    predict_ensemble,
    save_prediction_archive,
)
from face2ceph.model import FaceToCephalometryModel, FeatureModulation, ModelConfig, ModelOutput
from face2ceph.training import (
    TrainingConfig,
    _output_path as checkpoint_output_path,
    _stratified_subset,
    gaussian_negative_log_likelihood,
    multitask_loss,
    validation_metrics,
)


class TinyBackbone(nn.Module):
    def __init__(self, input_channels: int, output_dim: int = 12) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, output_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


def tiny_backbone(_: str, input_channels: int, __: bool, ___: str | None) -> tuple[nn.Module, int]:
    return TinyBackbone(input_channels), 12


def test_published_model_contract() -> None:
    config = ModelConfig(pretrained=False, neck_dim=16, dropout=0.0)
    model = FaceToCephalometryModel(config, backbone_factory=tiny_backbone)
    output = model(
        torch.randn(2, 3, 24, 24),
        torch.randn(2, 4, 24, 24),
        torch.tensor(((0.2, 0.0), (0.7, 1.0))),
    )

    assert ModelConfig().backbone_name == "convnext_tiny.in12k_ft_in1k"
    assert model.frontal_backbone is not model.profile_backbone
    assert output.regression_mean.shape == (2, 8)
    assert output.regression_log_variance.shape == (2, 8)
    assert output.sagittal_logits.shape == (2, 3)
    assert output.vertical_logits.shape == (2, 3)
    assert output.features.shape == (2, 16)
    assert torch.equal(output.regression_log_variance, torch.zeros_like(output.regression_log_variance))


def test_main_backbone_parameter_contract_without_download() -> None:
    model = FaceToCephalometryModel(ModelConfig(pretrained=False))
    assert sum(parameter.numel() for parameter in model.parameters()) == 56_642_966
    assert model.frontal_backbone.stem[0].weight.shape[1] == 3
    assert model.profile_backbone.stem[0].weight.shape[1] == 4


def test_feature_modulation_starts_as_identity() -> None:
    layer = FeatureModulation(metadata_dim=2, feature_dim=7, hidden_dim=4)
    features = torch.randn(3, 7)
    metadata = torch.randn(3, 2)
    assert torch.equal(layer(features, metadata), features)


def test_multitask_loss_is_finite_and_differentiable() -> None:
    model = FaceToCephalometryModel(
        ModelConfig(pretrained=False, neck_dim=16, dropout=0.0),
        backbone_factory=tiny_backbone,
    )
    output = model(torch.randn(4, 3, 16, 16), torch.randn(4, 4, 16, 16), torch.randn(4, 2))
    loss = multitask_loss(
        output,
        torch.randn(4, 8),
        torch.tensor((0, 1, 2, 1)),
        torch.tensor((2, 1, 0, 1)),
        torch.ones(3),
        torch.ones(3),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.regression_head.mean.weight.grad is not None


def test_gaussian_loss_matches_unit_variance_squared_error() -> None:
    mean = torch.tensor(((1.0, -1.0),))
    target = torch.zeros_like(mean)
    log_variance = torch.zeros_like(mean)
    assert torch.equal(gaussian_negative_log_likelihood(mean, log_variance, target), torch.tensor(0.5))


def test_ensemble_combines_aleatoric_and_epistemic_variance() -> None:
    means = np.stack((np.zeros((2, 8)), np.full((2, 8), 2.0)))
    variances = np.ones_like(means)
    sagittal = np.full((2, 2, 3), 1.0 / 3.0)
    vertical = sagittal.copy()
    mu, sigma, prob_sag, prob_vert, aleatoric, epistemic = combine_member_predictions(
        means, variances, sagittal, vertical
    )
    np.testing.assert_allclose(mu, 1.0)
    np.testing.assert_allclose(aleatoric, 1.0)
    np.testing.assert_allclose(epistemic, 1.0)
    np.testing.assert_allclose(sigma, np.sqrt(2.0))
    np.testing.assert_allclose(prob_sag, 1.0 / 3.0)
    np.testing.assert_allclose(prob_vert, 1.0 / 3.0)


def test_inference_rejects_checkpoint_image_size_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = [tmp_path / f"fold_{fold}.pt" for fold in range(5)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    payload = {
        "model": {"neck.weight": torch.ones(1)},
        "model_config": ModelConfig(pretrained=False).__dict__,
        "training_config": TrainingConfig(image_size=384).__dict__,
        "target_scaler": TargetScaler(np.zeros(8), np.ones(8)).to_dict(),
        "fold": 0,
        "seed": 42,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match="image_size"):
        predict_ensemble(
            "cohort.csv",
            ".",
            checkpoints,
            inference_config=InferenceConfig(image_size=256),
        )


def test_heteroscedastic_predict_ensemble_rescales_variance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeModel:
        def to(self, *_: object, **__: object) -> "FakeModel":
            return self

        def load_state_dict(self, *_: object, **__: object) -> None:
            return None

    checkpoints = [tmp_path / f"fold_{fold}.pt" for fold in range(5)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"checkpoint")
    model_config = ModelConfig(pretrained=False).__dict__
    training_config = TrainingConfig(image_size=384).__dict__
    scaler = TargetScaler(np.arange(8.0), np.arange(1.0, 9.0))

    def fake_load(path: Path, **_: object) -> dict[str, object]:
        fold = int(Path(path).stem.rsplit("_", 1)[1])
        return {
            "model_config": model_config,
            "training_config": training_config,
            "target_scaler": scaler.to_dict(),
            "fold": fold,
            "seed": 42 + 1000 * fold,
            "model": {"neck.weight": torch.ones(1)},
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
        inference_config=InferenceConfig(
            fold_count=5,
            image_size=384,
            batch_size=2,
            num_workers=0,
            mixed_precision=False,
            channels_last=False,
        ),
        device="cpu",
    )

    np.testing.assert_allclose(predictions.mu, np.broadcast_to(scaler.mean, (2, 8)))
    np.testing.assert_allclose(predictions.variance_aleatoric, np.broadcast_to(scaler.std**2, (2, 8)))
    np.testing.assert_allclose(predictions.variance_epistemic, 0.0)
    np.testing.assert_allclose(predictions.sigma, np.broadcast_to(scaler.std, (2, 8)))


def test_prediction_archive_preserves_variance_components(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_module, "GENERATED_ROOT", tmp_path)
    monkeypatch.setattr(inference_module, "GENERATED_ROOT", tmp_path)
    predictions = PredictionSet(
        case_id=np.array(("a", "b")),
        mu=np.zeros((2, 8)),
        sigma=np.ones((2, 8)),
        prob_sag=np.full((2, 3), 1.0 / 3.0),
        prob_vert=np.full((2, 3), 1.0 / 3.0),
        variance_aleatoric=np.full((2, 8), 0.75),
        variance_epistemic=np.full((2, 8), 0.25),
        y_raw=np.ones((2, 8)),
        features=np.full((2, 4), 2.0),
    )

    path = save_prediction_archive(predictions, "predictions/test.npz", tmp_path)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "case_id",
            "mu",
            "sigma",
            "var_alea",
            "var_epi",
            "features",
            "y_raw",
            "prob_sag",
            "prob_vert",
        }
        np.testing.assert_allclose(archive["var_alea"], 0.75)
        np.testing.assert_allclose(archive["var_epi"], 0.25)
        np.testing.assert_allclose(archive["features"], 2.0)


@pytest.mark.parametrize(
    ("regression_mode", "has_mean", "has_variance"),
    (("none", False, False), ("homoscedastic", True, False), ("heteroscedastic", True, True)),
)
def test_regression_modes(regression_mode: str, has_mean: bool, has_variance: bool) -> None:
    config = ModelConfig(
        pretrained=False,
        use_profile_sdf=False,
        metadata_conditioning="concatenate",
        regression_mode=regression_mode,
        neck_dim=16,
        dropout=0.0,
    )
    model = FaceToCephalometryModel(config, backbone_factory=tiny_backbone)
    output = model(torch.randn(2, 3, 12, 12), torch.randn(2, 3, 12, 12), torch.randn(2, 2))
    assert (output.regression_mean is not None) is has_mean
    assert (output.regression_log_variance is not None) is has_variance
    assert model.metadata_modulation is None


def test_classification_only_loss_excludes_regression() -> None:
    model = FaceToCephalometryModel(
        ModelConfig(pretrained=False, regression_mode="none", neck_dim=16, dropout=0.0),
        backbone_factory=tiny_backbone,
    )
    output = model(torch.randn(3, 3, 12, 12), torch.randn(3, 4, 12, 12), torch.randn(3, 2))
    loss = multitask_loss(
        output,
        torch.randn(3, 8),
        torch.tensor((0, 1, 2)),
        torch.tensor((2, 1, 0)),
        torch.ones(3),
        torch.ones(3),
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_classification_only_validation_uses_balanced_accuracy() -> None:
    class PerfectClassifier(nn.Module):
        def forward(self, frontal: torch.Tensor, profile: torch.Tensor, metadata: torch.Tensor) -> ModelOutput:
            logits = torch.eye(3, device=frontal.device) * 10.0
            return ModelOutput(None, None, logits, logits, torch.zeros(3, 4, device=frontal.device))

    batch = {
        "frontal": torch.zeros(3, 3, 4, 4),
        "profile": torch.zeros(3, 4, 4, 4),
        "metadata": torch.zeros(3, 2),
        "regression_target": torch.zeros(3, 8),
        "regression_target_raw": torch.zeros(3, 8),
        "sagittal_target": torch.arange(3),
        "vertical_target": torch.arange(3),
    }
    metrics = validation_metrics(
        PerfectClassifier(),
        [batch],
        TargetScaler(np.zeros(8), np.ones(8)),
        torch.device("cpu"),
        mixed_precision=False,
        channels_last=False,
    )
    assert metrics["mae"] is None
    assert metrics["balanced_accuracy"] == 1.0


@pytest.mark.parametrize("input_mode", ("frontal", "profile", "silhouette"))
def test_dataset_input_modes(monkeypatch: pytest.MonkeyPatch, input_mode: str) -> None:
    def fake_read(_: object, grayscale: bool) -> np.ndarray:
        if grayscale:
            return np.full((4, 4), 255, dtype=np.uint8)
        return np.full((4, 4, 3), 64, dtype=np.uint8)

    monkeypatch.setattr(dataset_module, "_read_image", fake_read)
    records = pd.DataFrame({"case_id": ["case"], "age": [18], "sex": ["F"]})
    sample = ClinicalPhotoDataset(records, ".", image_size=4, input_mode=input_mode)[0]
    if input_mode == "frontal":
        assert torch.count_nonzero(sample["profile"]) == 0
        assert torch.count_nonzero(sample["frontal"]) > 0
    elif input_mode == "profile":
        assert torch.count_nonzero(sample["frontal"]) == 0
        assert torch.count_nonzero(sample["profile"]) > 0
    else:
        assert torch.count_nonzero(sample["frontal"]) == 0
        assert torch.equal(sample["profile"], torch.ones_like(sample["profile"]))


def test_rgb_profile_omits_sdf(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_read(_: object, grayscale: bool) -> np.ndarray:
        calls.append(grayscale)
        return np.full((4, 4), 127, dtype=np.uint8) if grayscale else np.full((4, 4, 3), 64, dtype=np.uint8)

    monkeypatch.setattr(dataset_module, "_read_image", fake_read)
    records = pd.DataFrame({"case_id": ["case"], "age": [18], "sex": ["M"]})
    sample = ClinicalPhotoDataset(records, ".", image_size=4, use_profile_sdf=False)[0]
    assert sample["profile"].shape == (3, 4, 4)
    assert calls == [False, False]


def test_training_subset_preserves_each_stratum() -> None:
    frame = pd.DataFrame(
        [
            {"sex": sex, "sagittal": sagittal, "vertical": "Normo", "value": index}
            for sex in ("F", "M")
            for sagittal in ("I", "II")
            for index in range(10)
        ]
    )
    subset = _stratified_subset(frame, 0.5, 42)
    counts = subset.groupby(["sex", "sagittal", "vertical"], observed=True).size()
    assert len(subset) == 20
    assert set(counts) == {5}


def test_manifest_filters_analyzed_without_changing_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "age": [18, 19, 20],
            "sex": ["F", "M", "F"],
            "analyzed": [True, False, True],
            "usable": [False, True, True],
        }
    )
    split = pd.DataFrame(
        {"case_id": ["a", "b", "c"], "split": ["train_cv"] * 3, "fold": [0, 1, 2]}
    )
    frames = iter((clinical, split))
    monkeypatch.setattr(pd, "read_csv", lambda _: next(frames).copy())
    selected = load_manifest("clinical.csv", "split.csv", require_targets=False)
    assert selected["case_id"].tolist() == ["a", "c"]
    assert split["case_id"].tolist() == ["a", "b", "c"]


def test_manifest_selects_rgb_usable_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "age": [18, 19],
            "sex": ["F", "M"],
            "usable": [True, True],
            "analyzed": [True, False],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda _: clinical.copy())
    selected = load_manifest(
        "clinical.csv",
        require_targets=False,
        availability_column="usable",
    )
    assert selected["case_id"].tolist() == ["a", "b"]


def test_manifest_uses_usable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = pd.DataFrame(
        {"case_id": ["a", "b"], "age": [18, 19], "sex": ["F", "M"], "usable": [1, 0]}
    )
    split = pd.DataFrame(
        {"case_id": ["a", "b"], "split": ["train_cv", "train_cv"], "fold": [0, 1]}
    )
    frames = iter((clinical, split))
    monkeypatch.setattr(pd, "read_csv", lambda _: next(frames).copy())
    selected = load_manifest("clinical.csv", "split.csv", require_targets=False)
    assert selected["case_id"].tolist() == ["a"]


def test_manifest_rejects_partial_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = pd.DataFrame({"case_id": ["a", "b"], "age": [18, 19], "sex": ["F", "M"]})
    split = pd.DataFrame({"case_id": ["a"], "split": ["train_cv"], "fold": [0]})
    frames = iter((clinical, split))
    monkeypatch.setattr(pd, "read_csv", lambda _: next(frames).copy())
    with pytest.raises(ValueError, match="same case identifiers"):
        load_manifest("clinical.csv", "split.csv", require_targets=False)


def test_manifest_requires_strict_analysis_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    clinical = pd.DataFrame(
        {"case_id": ["a"], "age": [18], "sex": ["F"], "analyzed": ["yes"]}
    )
    monkeypatch.setattr(pd, "read_csv", lambda _: clinical.copy())
    with pytest.raises(ValueError, match="only boolean values"):
        load_manifest("clinical.csv", require_targets=False)


def test_implicit_pretrained_download_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit local file"):
        FaceToCephalometryModel(ModelConfig())


def test_checkpoint_output_cannot_escape_generated_root() -> None:
    outside = Path(__file__).resolve().parents[1] / "outside-checkpoint-test"
    with pytest.raises(ValueError, match="outputs must be inside"):
        checkpoint_output_path(outside, 0)
