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
ANCHOR_TYPE_VARIANT = "anchor_patch_moe_v1"
MULTISCALE_TYPE_VARIANT = "multiscale_temporal_v1"


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Legacy-compatible one-dimensional residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 9, stride=stride, padding=4)
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
        self.d_heads = nn.ModuleList([
            nn.Linear(feature_dim, num_dists) for _ in range(DISTANCE_EXPERT_COUNT)])

    @staticmethod
    def _layer(in_channels: int, out_channels: int, count: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride)]
        blocks.extend(ResidualBlock(out_channels, out_channels) for _ in range(1, count))
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
            nn.Conv1d(1, width, kernel_size=17, stride=8, padding=8, bias=False),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv1d(width, width * 2, kernel_size=9, stride=4, padding=4, bias=False),
            nn.GroupNorm(1, width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, encoded_dim, kernel_size=7, stride=2, padding=3, bias=False),
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
        self.prototypes = nn.Parameter(torch.empty(
            DISTANCE_EXPERT_COUNT, prototypes_per_class, embedding_dim))
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
        self.prototypes = nn.Parameter(torch.empty(
            DISTANCE_EXPERT_COUNT, DISTANCE_BIN_COUNT, feature_dim))
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
    anchor_orth_loss: torch.Tensor | None = None
    reliability_loss: torch.Tensor | None = None


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
        if global_view.ndim != 3 or tuple(global_view.shape[1:]) != (1, GLOBAL_LENGTH):
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
        self.local_known_head = nn.Linear(feature_dim, DISTANCE_EXPERT_COUNT)
        self.global_known_head = nn.Linear(feature_dim, DISTANCE_EXPERT_COUNT)
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


class MultiScaleTemporalBlock(nn.Module):
    """Mix nearby and distant pulse structure without quadratic attention."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.scales = nn.ModuleList([
            nn.Conv1d(width, width, 7, padding=3 * dilation,
                      dilation=dilation, groups=width, bias=False)
            for dilation in (1, 4, 16)
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(3 * width, width, 1), nn.GELU(), nn.Dropout(0.1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        return values + self.mix(torch.cat([
            scale(normalized) for scale in self.scales
        ], dim=1))


class MultiScaleWaveformBranch(nn.Module):
    """Preserve pulse ordering with dilated convolution and attention pooling."""

    def __init__(self, base: int):
        super().__init__()
        self.output_dim = base * 8
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 17, stride=8, padding=8, bias=False),
            nn.GroupNorm(1, base), nn.GELU(),
            nn.Conv1d(base, base, 9, stride=2, padding=4, bias=False),
            nn.GroupNorm(1, base), nn.GELU(),
            MultiScaleTemporalBlock(base), MultiScaleTemporalBlock(base),
        )
        self.attention = nn.Conv1d(base, 1, 1)
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * base), nn.Linear(2 * base, self.output_dim), nn.GELU()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.stem(values if values.ndim == 3 else values.unsqueeze(1))
        weights = self.attention(features).softmax(dim=-1)
        pooled = torch.cat(((features * weights).sum(dim=-1),
                            features.amax(dim=-1)), dim=1)
        return self.projection(pooled)


class MultiScaleHierarchicalTypeNet(HierarchicalTypeNet):
    """Five-class gate and prototypes with a multi-scale temporal backbone."""

    model_variant = MULTISCALE_TYPE_VARIANT

    def __init__(self, **config):
        super().__init__(**config)
        self.encoder.local_branch = MultiScaleWaveformBranch(self.base_channels)
        self.encoder.global_branch = MultiScaleWaveformBranch(self.base_channels)


class AnchorPatchEncoder(nn.Module):
    """Encode temporal patches through diverse reliability-gated experts."""

    def __init__(
        self,
        base_channels: int,
        patch_length: int,
        patch_stride: int,
        expert_count: int,
        expert_topk: int,
        frequency_bands: int,
        use_reliability: bool,
    ):
        super().__init__()
        self.output_dim = base_channels * 2
        self.patch_length, self.patch_stride = patch_length, patch_stride
        self.expert_topk, self.frequency_bands = expert_topk, frequency_bands
        self.use_reliability = use_reliability
        dimension = self.output_dim
        self.patch_embedding = nn.Conv1d(
            1, dimension, patch_length, stride=patch_stride, bias=False
        )
        self.temporal_refine = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.spectral_projection = nn.Sequential(
            nn.LayerNorm(frequency_bands), nn.Linear(frequency_bands, dimension)
        )
        self.context_projection = nn.Sequential(nn.Linear(1, dimension), nn.GELU())
        self.global_branch = WaveformBranch(max(8, base_channels // 2))
        self.global_projection = nn.Linear(self.global_branch.output_dim, dimension)
        self.evidence_norm = nn.LayerNorm(dimension)
        self.router = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, expert_count))
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dimension), nn.Linear(dimension, 2 * dimension),
                nn.GELU(), nn.Linear(2 * dimension, dimension),
            )
            for _ in range(expert_count)
        )
        self.confidence_heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
            for _ in range(expert_count)
        )
        self.daylight_projection = nn.Linear(1, dimension, bias=False)
        self.fusion = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU())
        self.temporal_gate, self.spectral_gate, self.context_gate = (nn.Parameter(torch.tensor(value)) for value in (-2.0, 0.0, -2.0))

    def _frequency_view(self, local: torch.Tensor) -> torch.Tensor:
        windows = local.unfold(2, self.patch_length, self.patch_stride).squeeze(1)
        power = torch.fft.rfft(windows.float(), dim=-1).abs().square()
        bands = torch.stack(
            [torch.log1p(chunk.mean(dim=-1)) for chunk in torch.tensor_split(
                power, self.frequency_bands, dim=-1
            )],
            dim=-1,
        )
        return self.spectral_projection(bands)

    @staticmethod
    def _anchor_orthogonality(
        evidence: torch.Tensor, probabilities: torch.Tensor
    ) -> torch.Tensor:
        mass = probabilities.sum(dim=1).clamp_min(1e-6)
        anchors = torch.einsum("bpk,bpd->bkd", probabilities, evidence) / mass.unsqueeze(-1)
        normalized = F.normalize(anchors, dim=-1, eps=1e-6)
        similarity = normalized @ normalized.transpose(1, 2)
        count = similarity.shape[1]
        mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
        return similarity[:, mask].square().mean()

    def forward(self, local: torch.Tensor, global_view: torch.Tensor,
                daylight: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.patch_embedding(local).transpose(1, 2)
        centroid = tokens.mean(dim=1, keepdim=True)
        context = (tokens * centroid).sum(dim=-1, keepdim=True) / math.sqrt(
            tokens.shape[-1]
        )
        global_features = self.global_projection(self.global_branch(global_view))
        evidence = self.evidence_norm(
            tokens
            + self.temporal_gate.sigmoid() * self.temporal_refine(tokens)
            + self.spectral_gate.sigmoid() * self._frequency_view(local)
            + self.context_gate.sigmoid() * self.context_projection(context)
            + global_features[:, None, :]
        )
        dense_routing = self.router(evidence).softmax(dim=-1)
        values, indices = dense_routing.topk(self.expert_topk, dim=-1)
        routing = torch.zeros_like(dense_routing).scatter(
            -1, indices, values / values.sum(dim=-1, keepdim=True)
        )
        expert_features = torch.stack(
            [evidence + expert(evidence) for expert in self.experts], dim=2
        )
        confidence = torch.stack(
            [head(expert_features[:, :, index]).squeeze(-1) for index, head in enumerate(self.confidence_heads)],
            dim=2,
        )
        effective = confidence.sigmoid() if self.use_reliability else torch.ones_like(confidence)
        patch_strength = (routing * effective).sum(dim=2).clamp_min(1e-6)
        patch_importance = patch_strength / patch_strength.sum(dim=1, keepdim=True)
        mixed = (routing * effective).unsqueeze(-1) * expert_features
        local_features = evidence.mean(dim=1)
        fused = self.fusion(
            (patch_importance.unsqueeze(-1) * mixed.sum(dim=2)).sum(dim=1)
            + self.daylight_projection(daylight)
        )
        return {
            "local": local_features, "global": global_features, "fused": fused,
            "experts": expert_features, "routing": routing, "confidence": confidence.sigmoid(), "confidence_logits": confidence,
            "importance": patch_importance,
            "anchor_loss": self._anchor_orthogonality(evidence, dense_routing),
        }


class AnchorHierarchicalTypeNet(nn.Module):
    """Hierarchical five-class classifier with AnchorMoE patch evidence."""

    architecture = HIERARCHICAL_TYPE_ARCHITECTURE
    model_variant = ANCHOR_TYPE_VARIANT

    def __init__(
        self,
        base_channels: int = 64,
        embedding_dim: int = 128,
        prototypes_per_class: int = 4,
        prototype_logit_weight: float = 0.25,
        known_fusion_weight: float = 0.0,
        patch_length: int = 256,
        patch_stride: int = 128,
        expert_count: int = 4,
        expert_topk: int = 2,
        frequency_bands: int = 4,
        use_reliability: bool = True,
    ):
        super().__init__()
        integer_values = (base_channels, embedding_dim, prototypes_per_class, patch_length, patch_stride, expert_count, expert_topk, frequency_bands)
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("AnchorMoE dimensions must be positive integers")
        if expert_topk > expert_count or patch_length > LOCAL_LENGTH:
            raise ValueError("AnchorMoE patch or top-k configuration is invalid")
        self.base_channels, self.embedding_dim = base_channels, embedding_dim
        self.prototypes_per_class = prototypes_per_class
        self.prototype_logit_weight, self.known_fusion_weight = float(prototype_logit_weight), float(known_fusion_weight)
        self.patch_length, self.patch_stride = patch_length, patch_stride
        self.expert_count, self.expert_topk = expert_count, expert_topk
        self.frequency_bands = frequency_bands
        self.use_reliability = bool(use_reliability)
        self.encoder = AnchorPatchEncoder(
            base_channels, patch_length, patch_stride, expert_count, expert_topk,
            frequency_bands, self.use_reliability,
        )
        dimension = self.encoder.output_dim
        self.gate_head = nn.Linear(dimension, 2)
        self.known_head = nn.Linear(dimension, DISTANCE_EXPERT_COUNT)
        self.local_known_head = nn.Linear(dimension, DISTANCE_EXPERT_COUNT)
        self.global_known_head = nn.Linear(dimension, DISTANCE_EXPERT_COUNT)
        self.embedding_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, embedding_dim))
        self.prototype_matcher = KnownPrototypeMatcher(embedding_dim, prototypes_per_class)

    def forward_hierarchical(self, local: torch.Tensor, global_view: torch.Tensor,
                             daylight: torch.Tensor) -> HierarchicalTypeOutput:
        FiveClassNet._validate_inputs(local, global_view, daylight)
        encoded = self.encoder(local, global_view, daylight)
        expert_logits = self.known_head(encoded["experts"])
        effective = encoded["confidence"] if self.use_reliability else torch.ones_like(
            encoded["confidence"]
        )
        patch_logits = (
            encoded["routing"] * effective
        ).unsqueeze(-1) * expert_logits
        additive_known = (
            encoded["importance"].unsqueeze(-1) * patch_logits.sum(dim=2)
        ).sum(dim=1)
        embedding = F.normalize(self.embedding_head(encoded["fused"]), dim=1, eps=1e-6)
        prototype_logits, prototype_evidence, prototype_scores = self.prototype_matcher(embedding)
        branch_known = self.known_head(encoded["fused"]) + 0.5 * (self.local_known_head(encoded["local"]) + self.global_known_head(encoded["global"]))
        known_logits = additive_known + self.known_fusion_weight * branch_known + self.prototype_logit_weight * prototype_evidence
        gate_logits = self.gate_head(encoded["fused"])
        gate_log = F.log_softmax(gate_logits, dim=1)
        type_logits = torch.cat((gate_log[:, :1], gate_log[:, 1:] + F.log_softmax(
            known_logits, dim=1
        )), dim=1)
        routing_only = (encoded["routing"].unsqueeze(-1) * expert_logits).sum(dim=2)
        predicted = additive_known.detach().argmax(dim=1)
        support = routing_only.gather(
            2, predicted[:, None, None].expand(-1, routing_only.shape[1], 1)
        ).squeeze(-1).abs()
        target = (support / support.max(dim=1, keepdim=True).values.clamp_min(1e-6)).detach()
        reliability = F.binary_cross_entropy_with_logits(
            (encoded["routing"] * encoded["confidence_logits"]).sum(dim=2), target
        )
        return HierarchicalTypeOutput(
            type_logits, gate_logits, known_logits, prototype_logits,
            prototype_scores, self.local_known_head(encoded["local"]),
            self.global_known_head(encoded["global"]), embedding,
            encoded["anchor_loss"], reliability,
        )

    def forward_type(self, local: torch.Tensor, global_view: torch.Tensor,
                     daylight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_hierarchical(local, global_view, daylight)
        return output.type_logits, output.embedding

    def forward(self, local: torch.Tensor, global_view: torch.Tensor,
                daylight: torch.Tensor) -> HierarchicalTypeOutput:
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


def create_anchor_hierarchical_type_model(**config) -> AnchorHierarchicalTypeNet:
    """Create a randomly initialized AnchorMoE hierarchical type model."""

    return AnchorHierarchicalTypeNet(**config)
