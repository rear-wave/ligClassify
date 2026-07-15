"""
ligClassify — ResNet1D with GroupNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 9, stride=stride, padding=4)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 9, padding=4)
        self.gn2 = _gn(out_ch)
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride), _gn(out_ch))

    def forward(self, x):
        r = self.downsample(x) if self.downsample else x
        z = F.relu(self.gn1(self.conv1(x)))
        z = self.gn2(self.conv2(z))
        return F.relu(z + r)


class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 15, stride=2, padding=7),
            _gn(base), nn.ReLU())
        self.layer1 = self._layer(base, base, 3, 1)
        self.layer2 = self._layer(base, base * 2, 3, 2)
        self.layer3 = self._layer(base * 2, base * 2, 3, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base * 2, num_classes)

    def _layer(self, in_ch, out_ch, n, stride):
        blocks = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            blocks.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def create_model(num_classes=4, base_channels=64):
    model = ResNet1D(num_classes=num_classes, base=base_channels)
    n = sum(p.numel() for p in model.parameters())
    print(f"  resnet: {n:,} params")
    return model


class MultiTaskResNet(nn.Module):
    """Shared ResNet1D encoder + type head + 4 per-class distance heads (30 bins each)."""

    def __init__(self, base=64, num_types=5, num_dists=30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=2, padding=7),
            _gn(base), nn.ReLU())
        self.layer1 = self._layer(base, base, 3, 1)
        self.layer2 = self._layer(base, base * 2, 3, 2)
        self.layer3 = self._layer(base * 2, base * 2, 3, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        feat_dim = base * 2
        self.type_head = nn.Linear(feat_dim, num_types)
        # d_heads: [NCG, NNBE, PCG, PNBE]  (type indices 1..4)
        self.d_heads = nn.ModuleList([
            nn.Linear(feat_dim, num_dists) for _ in range(4)
        ])

    def _layer(self, in_ch, out_ch, n, stride):
        blocks = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            blocks.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def _encode(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.gap(x).squeeze(-1)

    def forward_type(self, x):
        return self.type_head(self.extract_type_features(x))

    def extract_type_features(self, x):
        """Return pooled encoder features used by the type head."""
        return self._encode(x)

    def forward(self, x):
        _, type_logits, dist_logits = self.forward_with_features(x)
        return type_logits, dist_logits

    def forward_with_features(self, x):
        """Return type features and both prediction outputs in one pass."""
        features = self._encode(x)
        type_logits = self.type_head(features)
        dist_logits = [h(features) for h in self.d_heads]
        return features, type_logits, dist_logits


class MultiTaskOrdinalResNet(nn.Module):
    """Shared type encoder with a distance-specific pooled projection."""

    def __init__(
        self,
        base=64,
        num_types=5,
        num_dists=30,
        dist_mlp_dim=128,
        dist_dropout=0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=2, padding=7),
            _gn(base),
            nn.ReLU(),
        )
        self.layer1 = self._layer(base, base, 3, 1)
        self.layer2 = self._layer(base, base * 2, 3, 2)
        self.layer3 = self._layer(base * 2, base * 2, 3, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        feat_dim = base * 2
        self.type_head = nn.Linear(feat_dim, num_types)
        self.distance_projection = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, dist_mlp_dim),
            nn.GELU(),
            nn.Dropout(dist_dropout),
        )
        self.d_heads = nn.ModuleList([
            nn.Linear(dist_mlp_dim, num_dists) for _ in range(4)
        ])

    def _layer(self, in_ch, out_ch, n, stride):
        blocks = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            blocks.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def _encode_map(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward_type(self, x):
        return self.type_head(self.extract_type_features(x))

    def extract_type_features(self, x):
        """Return pooled encoder features used by the type head."""
        encoded = self._encode_map(x)
        return self.gap(encoded).squeeze(-1)

    def forward(self, x):
        _, type_logits, distance_logits = self.forward_with_features(x)
        return type_logits, distance_logits

    def forward_with_features(self, x):
        """Return type features and both prediction outputs in one pass."""
        x = self._encode_map(x)
        average = self.gap(x).squeeze(-1)
        maximum = self.gmp(x).squeeze(-1)
        type_logits = self.type_head(average)
        distance_features = self.distance_projection(
            torch.cat([average, maximum], dim=1)
        )
        distance_logits = [head(distance_features) for head in self.d_heads]
        return average, type_logits, distance_logits


class MultiScaleResidualBlock(nn.Module):
    """Fuse short, medium, and long depthwise temporal filters."""

    def __init__(self, channels, kernels=(7, 31, 127)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                    bias=False,
                ),
                _gn(channels),
                nn.GELU(),
            )
            for kernel_size in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(kernels), channels, 1, bias=False),
            _gn(channels),
        )

    def forward(self, x):
        mixed = self.fuse(torch.cat([branch(x) for branch in self.branches], 1))
        return F.gelu(x + mixed)


class WaveformBranch(nn.Module):
    """Encode one local or full-piece waveform view."""

    def __init__(self, base):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, base, 15, stride=4, padding=7, bias=False),
            _gn(base),
            nn.GELU(),
            MultiScaleResidualBlock(base),
            nn.Conv1d(base, base * 2, 7, stride=2, padding=3, bias=False),
            _gn(base * 2),
            nn.GELU(),
            MultiScaleResidualBlock(base * 2),
            nn.Conv1d(base * 2, base * 4, 7, stride=2, padding=3, bias=False),
            _gn(base * 4),
            nn.GELU(),
            MultiScaleResidualBlock(base * 4),
        )
        self.average = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)
        self.output_dim = base * 8

    def forward(self, x):
        encoded = self.network(x)
        return torch.cat([
            self.average(encoded).squeeze(-1),
            self.maximum(encoded).squeeze(-1),
        ], dim=1)


class ConditionalExpertNet(nn.Module):
    """Four-class type encoder with context-conditioned distance experts."""

    def __init__(
        self,
        base=64,
        num_types=4,
        num_dists=30,
        num_coarse=6,
        context_dim=3,
        dist_mlp_dim=128,
        dist_dropout=0.2,
    ):
        super().__init__()
        self.local_branch = WaveformBranch(base)
        self.global_branch = WaveformBranch(base)
        feature_dim = base * 4
        fusion_dim = self.local_branch.output_dim + self.global_branch.output_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, feature_dim),
            nn.GELU(),
        )
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.type_head = nn.Linear(feature_dim, num_types)
        expert_input_dim = feature_dim + context_dim
        self.distance_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(expert_input_dim),
                nn.Linear(expert_input_dim, dist_mlp_dim),
                nn.GELU(),
                nn.Dropout(dist_dropout),
            )
            for _ in range(num_types)
        ])
        self.d_heads = nn.ModuleList([
            nn.Linear(dist_mlp_dim, num_dists) for _ in range(num_types)
        ])
        self.coarse_heads = nn.ModuleList([
            nn.Linear(dist_mlp_dim, num_coarse) for _ in range(num_types)
        ])

    def extract_type_features(self, local, global_view):
        local_features = self.local_branch(local)
        global_features = self.global_branch(global_view)
        return self.fusion(torch.cat([local_features, global_features], dim=1))

    def forward_type(self, local, global_view):
        return self.type_head(self.extract_type_features(local, global_view))

    def forward_with_features(self, local, global_view, context):
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, {self.context_dim}]"
            )
        features = self.extract_type_features(local, global_view)
        if len(features) != len(context):
            raise ValueError("waveform views and context must have the same batch size")
        type_logits = self.type_head(features)
        conditioned = torch.cat([features, context], dim=1)
        expert_features = [
            projection(conditioned) for projection in self.distance_projections
        ]
        distance_logits = [
            head(expert_features[index])
            for index, head in enumerate(self.d_heads)
        ]
        coarse_logits = [
            head(expert_features[index])
            for index, head in enumerate(self.coarse_heads)
        ]
        return features, type_logits, distance_logits, coarse_logits

    def forward(self, local, global_view, context):
        _, type_logits, distance_logits, coarse_logits = (
            self.forward_with_features(local, global_view, context)
        )
        return type_logits, distance_logits, coarse_logits


def create_mtl_model(
    base_channels=64,
    architecture="mtl_resnet",
    num_types=5,
    dist_mlp_dim=128,
    dist_dropout=0.2,
):
    if architecture == "mtl_resnet":
        model = MultiTaskResNet(base=base_channels, num_types=num_types)
    elif architecture == "ordinal_v2":
        model = MultiTaskOrdinalResNet(
            base=base_channels,
            num_types=num_types,
            dist_mlp_dim=dist_mlp_dim,
            dist_dropout=dist_dropout,
        )
    elif architecture == "conditional_expert_v1":
        model = ConditionalExpertNet(
            base=base_channels,
            num_types=num_types,
            dist_mlp_dim=dist_mlp_dim,
            dist_dropout=dist_dropout,
        )
    else:
        raise ValueError(f"Unknown MTL architecture: {architecture}")
    n = sum(p.numel() for p in model.parameters())
    print(f"  MTL-ResNet ({architecture}): {n:,} params")
    return model
