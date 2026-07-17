"""Focused five-class and retained legacy waveform models."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TYPE_COUNT = 5
DISTANCE_EXPERT_COUNT = 4
DISTANCE_BIN_COUNT = 30
LOCAL_LENGTH = 8000
GLOBAL_LENGTH = 2000


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Legacy-compatible one-dimensional residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 9, stride=stride, padding=4
        )
        self.gn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 9, padding=4)
        self.gn2 = _group_norm(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                _group_norm(out_channels),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(values) if self.downsample else values
        encoded = F.relu(self.gn1(self.conv1(values)))
        encoded = self.gn2(self.conv2(encoded))
        return F.relu(encoded + residual)


class LegacyMultiTaskResNet(nn.Module):
    """Architecture retained solely for legacy five-class checkpoint loading."""

    def __init__(
        self,
        base: int = 64,
        num_types: int = TYPE_COUNT,
        num_dists: int = DISTANCE_BIN_COUNT,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=2, padding=7),
            _group_norm(base),
            nn.ReLU(),
        )
        self.layer1 = self._layer(base, base, 3, 1)
        self.layer2 = self._layer(base, base * 2, 3, 2)
        self.layer3 = self._layer(base * 2, base * 2, 3, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        feature_dim = base * 2
        self.type_head = nn.Linear(feature_dim, num_types)
        self.d_heads = nn.ModuleList(
            nn.Linear(feature_dim, num_dists)
            for _ in range(DISTANCE_EXPERT_COUNT)
        )

    @staticmethod
    def _layer(
        in_channels: int, out_channels: int, count: int, stride: int
    ) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride)]
        blocks.extend(
            ResidualBlock(out_channels, out_channels)
            for _ in range(1, count)
        )
        return nn.Sequential(*blocks)

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        return self.gap(values).squeeze(-1)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features = self._encode(values)
        return (
            self.type_head(features),
            tuple(head(features) for head in self.d_heads),
        )


class MultiScaleResidualBlock(nn.Module):
    """Fuse short, medium, and long temporal filters."""

    def __init__(
        self, channels: int, kernels: tuple[int, ...] = (7, 31, 127)
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                    bias=False,
                ),
                _group_norm(channels),
                nn.GELU(),
            )
            for kernel_size in kernels
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(kernels), channels, 1, bias=False),
            _group_norm(channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixed = self.fuse(
            torch.cat([branch(values) for branch in self.branches], dim=1)
        )
        return F.gelu(values + mixed)


class WaveformBranch(nn.Module):
    """Encode one signed local or global waveform view."""

    def __init__(self, base: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=4, padding=7, bias=False),
            _group_norm(base),
            nn.GELU(),
            MultiScaleResidualBlock(base),
            nn.Conv1d(base, base * 2, 7, stride=2, padding=3, bias=False),
            _group_norm(base * 2),
            nn.GELU(),
            MultiScaleResidualBlock(base * 2),
            nn.Conv1d(base * 2, base * 4, 7, stride=2, padding=3, bias=False),
            _group_norm(base * 4),
            nn.GELU(),
            MultiScaleResidualBlock(base * 4),
        )
        self.average = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)
        self.output_dim = base * 8

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values)
        return torch.cat(
            [
                self.average(encoded).squeeze(-1),
                self.maximum(encoded).squeeze(-1),
            ],
            dim=1,
        )


@dataclass(frozen=True)
class ModelOutput:
    """Immutable predictions and shared representation from ``FiveClassNet``."""

    type_logits: torch.Tensor
    distance_logits: tuple[torch.Tensor, ...]
    features: torch.Tensor


class FiveClassNet(nn.Module):
    """Dual-view five-class network with four non-IC distance experts."""

    def __init__(self, base_channels: int = 64):
        super().__init__()
        if type(base_channels) is not int or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")
        self.base_channels = base_channels
        self.local_branch = WaveformBranch(base_channels)
        self.global_branch = WaveformBranch(base_channels)
        feature_dim = base_channels * 4
        fusion_input_dim = (
            self.local_branch.output_dim + self.global_branch.output_dim + 1
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, feature_dim),
            nn.GELU(),
        )
        self.feature_dim = feature_dim
        self.type_head = nn.Linear(feature_dim, TYPE_COUNT)
        self.distance_heads = nn.ModuleList(
            nn.Linear(feature_dim, DISTANCE_BIN_COUNT)
            for _ in range(DISTANCE_EXPERT_COUNT)
        )

    @staticmethod
    def _validate_inputs(
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> None:
        if local.ndim != 3 or tuple(local.shape[1:]) != (1, LOCAL_LENGTH):
            raise ValueError("local must have shape [batch, 1, 8000]")
        if global_view.ndim != 3 or tuple(global_view.shape[1:]) != (
            1,
            GLOBAL_LENGTH,
        ):
            raise ValueError("global_view must have shape [batch, 1, 2000]")
        if daylight.ndim != 2 or daylight.shape[1] != 1:
            raise ValueError("daylight must have shape [batch, 1]")
        if not (len(local) == len(global_view) == len(daylight)):
            raise ValueError("model inputs must have the same batch size")

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> ModelOutput:
        self._validate_inputs(local, global_view, daylight)
        local_features = self.local_branch(local)
        global_features = self.global_branch(global_view)
        features = self.fusion(
            torch.cat([local_features, global_features, daylight], dim=1)
        )
        return ModelOutput(
            type_logits=self.type_head(features),
            distance_logits=tuple(
                head(features) for head in self.distance_heads
            ),
            features=features,
        )


def create_five_class_model(base_channels: int = 64) -> FiveClassNet:
    """Create a randomly initialized five-class model."""

    return FiveClassNet(base_channels=base_channels)
