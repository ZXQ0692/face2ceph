"""Neural architecture for two-view cephalometric prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .dataset import MEASUREMENT_NAMES


@dataclass(frozen=True)
class ModelConfig:
    backbone_name: str = "convnext_tiny.in12k_ft_in1k"
    pretrained: bool = True
    pretrained_weights: str | None = None
    use_profile_sdf: bool = True
    metadata_conditioning: str = "film"
    regression_mode: str = "heteroscedastic"
    input_mode: str = "both"
    metadata_dim: int = 2
    film_hidden_dim: int = 64
    neck_dim: int = 512
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if not self.backbone_name.strip() or min(self.metadata_dim, self.film_hidden_dim, self.neck_dim) < 1:
            raise ValueError("Model dimensions and backbone name must be valid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.metadata_conditioning not in {"film", "concatenate"}:
            raise ValueError("metadata_conditioning must be film or concatenate")
        if self.regression_mode not in {"none", "homoscedastic", "heteroscedastic"}:
            raise ValueError("regression_mode must be none, homoscedastic, or heteroscedastic")
        if self.input_mode not in {"both", "frontal", "profile", "silhouette"}:
            raise ValueError("input_mode must be both, frontal, profile, or silhouette")
        if self.input_mode == "silhouette" and not self.use_profile_sdf:
            raise ValueError("Silhouette input requires the profile signed-distance channel")


@dataclass
class ModelOutput:
    regression_mean: torch.Tensor | None
    regression_log_variance: torch.Tensor | None
    sagittal_logits: torch.Tensor
    vertical_logits: torch.Tensor
    features: torch.Tensor


class FeatureModulation(nn.Module):
    def __init__(self, metadata_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        scale, offset = self.network(metadata).chunk(2, dim=1)
        return (1.0 + scale) * features + offset


class HeteroscedasticRegressionHead(nn.Module):
    def __init__(self, input_dim: int, target_count: int) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, target_count)
        self.log_variance = nn.Linear(input_dim, target_count)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.zeros_(self.log_variance.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean(features), self.log_variance(features).clamp(-8.0, 8.0)


def _timm_backbone(
    name: str,
    input_channels: int,
    pretrained: bool,
    pretrained_weights: str | None,
) -> tuple[nn.Module, int]:
    import timm

    arguments = {}
    if pretrained:
        if pretrained_weights is None:
            raise ValueError("pretrained_weights must identify an explicit local file")
        weight_path = Path(pretrained_weights).expanduser().resolve(strict=True)
        if not weight_path.is_file():
            raise ValueError("pretrained_weights must identify a local file")
        arguments["pretrained_cfg_overlay"] = {"file": str(weight_path)}
    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=input_channels,
        num_classes=0,
        global_pool="avg",
        **arguments,
    )
    return backbone, int(backbone.num_features)


class FaceToCephalometryModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig = ModelConfig(),
        *,
        backbone_factory: Callable[[str, int, bool, str | None], tuple[nn.Module, int]] = _timm_backbone,
    ) -> None:
        super().__init__()
        self.config = config
        self.frontal_backbone, frontal_dim = backbone_factory(
            config.backbone_name, 3, config.pretrained, config.pretrained_weights
        )
        profile_channels = 4 if config.use_profile_sdf else 3
        self.profile_backbone, profile_dim = backbone_factory(
            config.backbone_name, profile_channels, config.pretrained, config.pretrained_weights
        )
        fused_dim = frontal_dim + profile_dim
        if config.metadata_conditioning == "film":
            self.metadata_modulation = FeatureModulation(config.metadata_dim, fused_dim, config.film_hidden_dim)
            neck_input_dim = fused_dim
        else:
            self.metadata_modulation = None
            neck_input_dim = fused_dim + config.metadata_dim
        self.neck = nn.Sequential(
            nn.LayerNorm(neck_input_dim),
            nn.Dropout(config.dropout),
            nn.Linear(neck_input_dim, config.neck_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if config.regression_mode == "heteroscedastic":
            self.regression_head: nn.Module | None = HeteroscedasticRegressionHead(
                config.neck_dim, len(MEASUREMENT_NAMES)
            )
        elif config.regression_mode == "homoscedastic":
            self.regression_head = nn.Linear(config.neck_dim, len(MEASUREMENT_NAMES))
        else:
            self.regression_head = None
        self.sagittal_head = nn.Linear(config.neck_dim, 3)
        self.vertical_head = nn.Linear(config.neck_dim, 3)

    def extract_features(
        self,
        frontal: torch.Tensor,
        profile: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if frontal.ndim != 4 or frontal.shape[1] != 3:
            raise ValueError("frontal must have shape N x 3 x H x W")
        profile_channels = 4 if self.config.use_profile_sdf else 3
        if profile.ndim != 4 or profile.shape[1] != profile_channels:
            raise ValueError(f"profile must have shape N x {profile_channels} x H x W")
        if metadata.ndim != 2 or metadata.shape[1] != self.config.metadata_dim:
            raise ValueError(f"metadata must have shape N x {self.config.metadata_dim}")
        fused = torch.cat((self.frontal_backbone(frontal), self.profile_backbone(profile)), dim=1)
        conditioned = (
            self.metadata_modulation(fused, metadata)
            if self.metadata_modulation is not None
            else torch.cat((fused, metadata), dim=1)
        )
        return self.neck(conditioned)

    def forward(
        self,
        frontal: torch.Tensor,
        profile: torch.Tensor,
        metadata: torch.Tensor,
    ) -> ModelOutput:
        features = self.extract_features(frontal, profile, metadata)
        if isinstance(self.regression_head, HeteroscedasticRegressionHead):
            mean, log_variance = self.regression_head(features)
        elif self.regression_head is not None:
            mean, log_variance = self.regression_head(features), None
        else:
            mean = log_variance = None
        return ModelOutput(
            regression_mean=mean,
            regression_log_variance=log_variance,
            sagittal_logits=self.sagittal_head(features),
            vertical_logits=self.vertical_head(features),
            features=features,
        )


def build_model(config: ModelConfig = ModelConfig()) -> FaceToCephalometryModel:
    return FaceToCephalometryModel(config)
