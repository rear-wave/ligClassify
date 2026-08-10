"""Two-stage waveform cascade and retained legacy inference model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
TYPE_COUNT = len(TYPE_NAMES)
DISTANCE_EXPERT_COUNT = len(DISTANCE_NAMES)
DISTANCE_BIN_COUNT = 30
LOCAL_LENGTH = 8000
GLOBAL_LENGTH = 2000
CASCADE_ARCHITECTURE = "five_class_v1"
MODEL_VARIANT = "type_then_prototype_distance_v1"
HIERARCHICAL_TYPE_ARCHITECTURE = "hierarchical_five_class_v2"
HIERARCHICAL_TYPE_VARIANT = "ic_gate_known_prototypes_v2"


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
    """Exact architecture retained for ``weights/old/model.pt`` only."""

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

    def extract_type_features(self, values: torch.Tensor) -> torch.Tensor:
        return self._encode(values)

    def forward_type(self, values: torch.Tensor) -> torch.Tensor:
        return self.type_head(self.extract_type_features(values))

    def forward_with_features(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        features = self._encode(values)
        return (
            features,
            self.type_head(features),
            tuple(head(features) for head in self.d_heads),
        )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _, type_logits, distance_logits = self.forward_with_features(values)
        return type_logits, distance_logits


class WaveformBranch(nn.Module):
    """Memory-bounded encoder for one signed waveform view."""

    def __init__(self, base: int):
        super().__init__()
        self.output_dim = base * 8
        encoded_dim = self.output_dim // 2
        width = max(8, min(32, encoded_dim // 4))
        self.network = nn.Sequential(
            nn.Conv1d(
                1,
                width,
                kernel_size=17,
                stride=8,
                padding=8,
                bias=False,
            ),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(
                width,
                width * 2,
                kernel_size=9,
                stride=4,
                padding=4,
                bias=False,
            ),
            nn.GroupNorm(1, width * 2),
            nn.GELU(),
            nn.Conv1d(
                width * 2,
                encoded_dim,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(1, encoded_dim),
            nn.GELU(),
        )
        self.average = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim not in (2, 3):
            raise ValueError(
                "waveform input must have shape [batch, length] or "
                "[batch, channels, length]"
            )
        encoded = self.network(
            values.unsqueeze(1) if values.ndim == 2 else values
        )
        return torch.cat(
            (
                self.average(encoded).squeeze(-1),
                self.maximum(encoded).squeeze(-1),
            ),
            dim=1,
        )


class DualViewEncoder(nn.Module):
    """Encode local/global views and daylight into one task-specific vector."""

    def __init__(self, base_channels: int):
        super().__init__()
        self.local_branch = WaveformBranch(base_channels)
        self.global_branch = WaveformBranch(base_channels)
        fusion_input = (
            self.local_branch.output_dim + self.global_branch.output_dim + 1
        )
        self.output_dim = base_channels * 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, self.output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            torch.cat(
                (
                    self.local_branch(local),
                    self.global_branch(global_view),
                    daylight,
                ),
                dim=1,
            )
        )


class GatedDualViewEncoder(nn.Module):
    """Fuse local/global waveform evidence with learned per-piece weights."""

    def __init__(self, base_channels: int):
        super().__init__()
        self.local_branch = WaveformBranch(base_channels)
        self.global_branch = WaveformBranch(base_channels)
        self.output_dim = base_channels * 4
        self.local_projection = nn.Sequential(
            nn.LayerNorm(self.local_branch.output_dim),
            nn.Linear(self.local_branch.output_dim, self.output_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(self.global_branch.output_dim),
            nn.Linear(self.global_branch.output_dim, self.output_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Linear(self.output_dim * 2 + 1, 2)
        self.daylight_projection = nn.Linear(1, self.output_dim, bias=False)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_features = self.local_projection(self.local_branch(local))
        global_features = self.global_projection(
            self.global_branch(global_view)
        )
        weights = self.fusion_gate(
            torch.cat((local_features, global_features, daylight), dim=1)
        ).softmax(dim=1)
        fused = (
            weights[:, :1] * local_features
            + weights[:, 1:] * global_features
            + self.daylight_projection(daylight)
        )
        return local_features, global_features, self.fusion(fused)


class KnownPrototypeMatcher(nn.Module):
    """Match known-class embeddings against multiple learned prototypes."""

    def __init__(
        self,
        embedding_dim: int,
        prototypes_per_class: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.prototypes_per_class = prototypes_per_class
        self.prototypes = nn.Parameter(
            torch.empty(
                DISTANCE_EXPERT_COUNT,
                prototypes_per_class,
                embedding_dim,
            )
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def normalized_prototypes(self) -> torch.Tensor:
        """Return unit prototypes used by cosine matching."""
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def forward(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_embedding = F.normalize(embedding, dim=-1, eps=1e-6)
        cosine = torch.einsum(
            "bd,ckd->bck",
            normalized_embedding,
            self.normalized_prototypes(),
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = cosine * scale
        aggregated = torch.logsumexp(logits, dim=2) - math.log(
            self.prototypes_per_class
        )
        return logits, aggregated, cosine.max(dim=2).values


class PrototypeDistanceMatcher(nn.Module):
    """Match distance features to ordered, class-specific 100-km prototypes."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.empty(
                DISTANCE_EXPERT_COUNT,
                DISTANCE_BIN_COUNT,
                feature_dim,
            )
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        with torch.no_grad():
            distance_axis = torch.linspace(
                -1.0, 1.0, DISTANCE_BIN_COUNT
            )[None, :, None]
            self.prototypes[:, :, :1].add_(distance_axis)

    def _normalized(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def match_type(
        self, features: torch.Tensor, expert_index: int
    ) -> torch.Tensor:
        if not 0 <= int(expert_index) < DISTANCE_EXPERT_COUNT:
            raise ValueError("expert_index must be in 0..3")
        normalized_features = F.normalize(features, dim=-1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * normalized_features @ self._normalized()[
            int(expert_index)
        ].transpose(0, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            self.match_type(features, expert_index)
            for expert_index in range(DISTANCE_EXPERT_COUNT)
        )


@dataclass(frozen=True)
class ModelOutput:
    """Training output for type classification and distance matching."""

    type_logits: torch.Tensor
    distance_logits: tuple[torch.Tensor, ...]
    features: torch.Tensor


@dataclass(frozen=True)
class HierarchicalTypeOutput:
    """Complete evidence produced by the hierarchical type classifier."""

    type_logits: torch.Tensor
    gate_logits: torch.Tensor
    known_logits: torch.Tensor
    prototype_logits: torch.Tensor
    prototype_scores: torch.Tensor
    local_known_logits: torch.Tensor
    global_known_logits: torch.Tensor
    embedding: torch.Tensor


class FiveClassNet(nn.Module):
    """Two-stage type classifier and class-conditional distance matcher."""

    architecture = CASCADE_ARCHITECTURE
    model_variant = MODEL_VARIANT

    def __init__(self, base_channels: int = 64):
        super().__init__()
        if type(base_channels) is not int or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")
        self.base_channels = base_channels
        self.type_encoder = DualViewEncoder(base_channels)
        self.distance_encoder = DualViewEncoder(base_channels)
        self.type_head = nn.Linear(self.type_encoder.output_dim, TYPE_COUNT)
        self.distance_matcher = PrototypeDistanceMatcher(
            self.distance_encoder.output_dim
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

    def forward_type(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(local, global_view, daylight)
        features = self.type_encoder(local, global_view, daylight)
        return self.type_head(features), features

    def forward_distance(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        self._validate_inputs(local, global_view, daylight)
        features = self.distance_encoder(local, global_view, daylight)
        return self.distance_matcher(features), features

    def forward_distance_type(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
        *,
        expert_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one class-specific distance matcher."""
        self._validate_inputs(local, global_view, daylight)
        features = self.distance_encoder(local, global_view, daylight)
        return self.distance_matcher.match_type(features, expert_index), features

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> ModelOutput:
        type_logits, type_features = self.forward_type(
            local, global_view, daylight
        )
        distance_logits, _ = self.forward_distance(
            local, global_view, daylight
        )
        return ModelOutput(type_logits, distance_logits, type_features)

    def predict_cascade(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> ModelOutput:
        """Classify first and evaluate only the selected non-IC matchers."""
        type_logits, type_features = self.forward_type(
            local, global_view, daylight
        )
        predicted_types = type_logits.argmax(dim=1)
        routed = tuple(
            type_logits.new_zeros((len(type_logits), DISTANCE_BIN_COUNT))
            for _ in range(DISTANCE_EXPERT_COUNT)
        )
        known_positions = torch.nonzero(
            predicted_types != 0, as_tuple=False
        ).flatten()
        if len(known_positions):
            distance_features = self.distance_encoder(
                local[known_positions],
                global_view[known_positions],
                daylight[known_positions],
            )
            known_types = predicted_types[known_positions]
            for expert_index in range(DISTANCE_EXPERT_COUNT):
                selected = known_types == expert_index + 1
                if torch.any(selected):
                    rows = known_positions[selected]
                    logits = self.distance_matcher.match_type(
                        distance_features[selected], expert_index
                    )
                    routed[expert_index].index_copy_(0, rows, logits)
        return ModelOutput(type_logits, routed, type_features)

    def set_training_stage(
        self, stage: Literal["type", "distance", "joint"]
    ) -> None:
        """Freeze the other stage so sequential training cannot corrupt it."""
        if stage not in {"type", "distance", "joint"}:
            raise ValueError("stage must be type, distance, or joint")
        type_enabled = stage in {"type", "joint"}
        distance_enabled = stage in {"distance", "joint"}
        for parameter in self.type_encoder.parameters():
            parameter.requires_grad_(type_enabled)
        for parameter in self.type_head.parameters():
            parameter.requires_grad_(type_enabled)
        for parameter in self.distance_encoder.parameters():
            parameter.requires_grad_(distance_enabled)
        for parameter in self.distance_matcher.parameters():
            parameter.requires_grad_(distance_enabled)

    def set_training_role(self, role: str) -> int | None:
        """Enable only the network used by one independently saved role."""
        if role == "type":
            self.set_training_stage("type")
            return None
        if role not in DISTANCE_NAMES:
            raise ValueError(
                f"role must be one of {('type', *DISTANCE_NAMES)}"
            )
        self.set_training_stage("distance")
        return DISTANCE_NAMES.index(role)


class HierarchicalTypeNet(nn.Module):
    """Five-class type model with an IC gate and known-class evidence."""

    architecture = HIERARCHICAL_TYPE_ARCHITECTURE
    model_variant = HIERARCHICAL_TYPE_VARIANT

    def __init__(
        self,
        base_channels: int = 64,
        embedding_dim: int = 128,
        prototypes_per_class: int = 4,
        prototype_logit_weight: float = 0.25,
    ):
        super().__init__()
        if type(base_channels) is not int or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if (
            type(prototypes_per_class) is not int
            or prototypes_per_class <= 0
        ):
            raise ValueError(
                "prototypes_per_class must be a positive integer"
            )
        if (
            isinstance(prototype_logit_weight, bool)
            or not math.isfinite(float(prototype_logit_weight))
            or float(prototype_logit_weight) < 0.0
        ):
            raise ValueError(
                "prototype_logit_weight must be finite and non-negative"
            )
        self.base_channels = base_channels
        self.embedding_dim = embedding_dim
        self.prototypes_per_class = prototypes_per_class
        self.prototype_logit_weight = float(prototype_logit_weight)
        self.encoder = GatedDualViewEncoder(base_channels)
        feature_dim = self.encoder.output_dim
        self.gate_head = nn.Linear(feature_dim, 2)
        self.known_head = nn.Linear(feature_dim, DISTANCE_EXPERT_COUNT)
        self.local_known_head = nn.Linear(
            feature_dim, DISTANCE_EXPERT_COUNT
        )
        self.global_known_head = nn.Linear(
            feature_dim, DISTANCE_EXPERT_COUNT
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.prototype_matcher = KnownPrototypeMatcher(
            embedding_dim,
            prototypes_per_class,
        )

    @staticmethod
    def _validate_inputs(
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> None:
        FiveClassNet._validate_inputs(local, global_view, daylight)

    def forward_hierarchical(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> HierarchicalTypeOutput:
        """Return gate, known-head, prototype, and joint type evidence."""
        self._validate_inputs(local, global_view, daylight)
        local_features, global_features, fused = self.encoder(
            local, global_view, daylight
        )
        gate_logits = self.gate_head(fused)
        embedding = F.normalize(
            self.embedding_head(fused), dim=1, eps=1e-6
        )
        prototype_logits, prototype_evidence, prototype_scores = (
            self.prototype_matcher(embedding)
        )
        known_logits = (
            self.known_head(fused)
            + self.prototype_logit_weight * prototype_evidence
        )
        gate_log_probabilities = F.log_softmax(gate_logits, dim=1)
        known_log_probabilities = F.log_softmax(known_logits, dim=1)
        type_logits = torch.cat(
            (
                gate_log_probabilities[:, :1],
                gate_log_probabilities[:, 1:]
                + known_log_probabilities,
            ),
            dim=1,
        )
        return HierarchicalTypeOutput(
            type_logits=type_logits,
            gate_logits=gate_logits,
            known_logits=known_logits,
            prototype_logits=prototype_logits,
            prototype_scores=prototype_scores,
            local_known_logits=self.local_known_head(local_features),
            global_known_logits=self.global_known_head(global_features),
            embedding=embedding,
        )

    def forward_type(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint five-class log probabilities and metric features."""
        output = self.forward_hierarchical(local, global_view, daylight)
        return output.type_logits, output.embedding

    def forward(
        self,
        local: torch.Tensor,
        global_view: torch.Tensor,
        daylight: torch.Tensor,
    ) -> HierarchicalTypeOutput:
        return self.forward_hierarchical(local, global_view, daylight)


def create_five_class_model(base_channels: int = 64) -> FiveClassNet:
    """Create a randomly initialized two-stage cascade."""

    return FiveClassNet(base_channels=base_channels)


def create_hierarchical_type_model(
    base_channels: int = 64,
    embedding_dim: int = 128,
    prototypes_per_class: int = 4,
    prototype_logit_weight: float = 0.25,
) -> HierarchicalTypeNet:
    """Create a randomly initialized hierarchical five-class type model."""
    return HierarchicalTypeNet(
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        prototypes_per_class=prototypes_per_class,
        prototype_logit_weight=prototype_logit_weight,
    )
