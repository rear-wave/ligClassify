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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        type_logits = self.type_head(x)
        dist_logits = [h(x) for h in self.d_heads]  # list of 4 tensors [B, 30]
        return type_logits, dist_logits


def create_mtl_model(base_channels=64):
    model = MultiTaskResNet(base=base_channels)
    n = sum(p.numel() for p in model.parameters())
    print(f"  MTL-ResNet: {n:,} params")
    return model
