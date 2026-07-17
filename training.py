"""Joint loss, epoch training, and early stopping for five-class models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models import DISTANCE_BIN_COUNT, DISTANCE_EXPERT_COUNT, ModelOutput, TYPE_COUNT


@dataclass(frozen=True)
class JointLoss:
    """Differentiable total loss with detached logging components."""

    total: torch.Tensor
    type_loss: float
    distance_loss: float
    distance_count: int


def _validate_joint_shapes(
    output: ModelOutput,
    type_labels: torch.Tensor,
    distance_bins: torch.Tensor,
) -> None:
    batch_size = len(type_labels)
    if type_labels.ndim != 1 or distance_bins.ndim != 1:
        raise ValueError("type_label and distance_bin must be one-dimensional")
    if len(distance_bins) != batch_size:
        raise ValueError("type_label and distance_bin batch sizes differ")
    if tuple(output.type_logits.shape) != (batch_size, TYPE_COUNT):
        raise ValueError("type_logits must have shape [batch, 5]")
    if len(output.distance_logits) != DISTANCE_EXPERT_COUNT:
        raise ValueError("exactly four distance experts are required")
    if any(
        tuple(logits.shape) != (batch_size, DISTANCE_BIN_COUNT)
        for logits in output.distance_logits
    ):
        raise ValueError("distance logits must have shape [batch, 30]")


def compute_joint_loss(
    output: ModelOutput,
    batch: Mapping[str, object],
    distance_weight: float = 0.5,
) -> JointLoss:
    """Compute five-class CE and true-type-routed ordered distance loss."""
    if not math.isfinite(float(distance_weight)) or float(distance_weight) < 0:
        raise ValueError("distance_weight must be finite and non-negative")
    if not isinstance(batch.get("type_label"), torch.Tensor) or not isinstance(
        batch.get("distance_bin"), torch.Tensor
    ):
        raise TypeError("batch labels must be tensors")

    device = output.type_logits.device
    type_labels = batch["type_label"].to(device=device, dtype=torch.long)
    distance_bins = batch["distance_bin"].to(device=device, dtype=torch.long)
    _validate_joint_shapes(output, type_labels, distance_bins)

    if torch.any((type_labels < 0) | (type_labels >= TYPE_COUNT)):
        raise ValueError("type_label must be in the five-class range 0..4")
    ic_mask = type_labels == 0
    if torch.any(distance_bins[ic_mask] != -1):
        raise ValueError("IC labels must use distance_bin -1")
    non_ic_mask = ~ic_mask
    if torch.any(
        (distance_bins[non_ic_mask] < 0)
        | (distance_bins[non_ic_mask] >= DISTANCE_BIN_COUNT)
    ):
        raise ValueError("non-IC labels must use a distance bin in 0..29")

    type_loss_tensor = F.cross_entropy(output.type_logits, type_labels)
    distance_count = int(non_ic_mask.sum().item())
    if distance_count:
        routed_chunks = []
        target_chunks = []
        for type_index in range(1, TYPE_COUNT):
            selected = type_labels == type_index
            if torch.any(selected):
                routed_chunks.append(output.distance_logits[type_index - 1][selected])
                target_chunks.append(distance_bins[selected])
        routed_logits = torch.cat(routed_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        categorical = F.cross_entropy(routed_logits, targets)
        probabilities = routed_logits.softmax(dim=1)
        predicted_cdf = probabilities.cumsum(dim=1)
        target_cdf = (
            torch.arange(DISTANCE_BIN_COUNT, device=device)[None, :]
            >= targets[:, None]
        ).to(routed_logits.dtype)
        ordered = torch.abs(predicted_cdf - target_cdf).mean()
        distance_loss_tensor = categorical + 0.2 * ordered
    else:
        distance_loss_tensor = output.type_logits.sum() * 0.0

    total = type_loss_tensor + float(distance_weight) * distance_loss_tensor
    return JointLoss(
        total=total,
        type_loss=float(type_loss_tensor.detach().item()),
        distance_loss=float(distance_loss_tensor.detach().item()),
        distance_count=distance_count,
    )


def _move_training_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    moved = dict(batch)
    for key in ("local", "global", "daylight", "type_label", "distance_bin"):
        value = moved.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {key!r} must be a tensor")
        moved[key] = value.to(device)
    return moved


def train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    distance_weight: float = 0.5,
) -> dict[str, float | int]:
    """Train for one epoch using exactly one model forward per batch."""
    target_device = torch.device(device)
    amp_enabled = bool(amp and target_device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler(target_device.type, enabled=amp_enabled)
    model.train()

    sample_count = 0
    distance_count = 0
    type_sum = 0.0
    distance_sum = 0.0
    for raw_batch in loader:
        batch = _move_training_batch(raw_batch, target_device)
        batch_size = int(batch["type_label"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=target_device.type,
            enabled=amp_enabled,
        ):
            output = model(batch["local"], batch["global"], batch["daylight"])
            losses = compute_joint_loss(
                output, batch, distance_weight=distance_weight
            )
        scaler.scale(losses.total).backward()
        scaler.step(optimizer)
        scaler.update()

        sample_count += batch_size
        distance_count += losses.distance_count
        type_sum += losses.type_loss * batch_size
        distance_sum += losses.distance_loss * losses.distance_count

    if sample_count == 0:
        raise ValueError("training loader produced no samples")
    mean_type = type_sum / sample_count
    mean_distance = distance_sum / distance_count if distance_count else 0.0
    return {
        "loss": mean_type + float(distance_weight) * mean_distance,
        "type_loss": mean_type,
        "distance_loss": mean_distance,
        "sample_count": sample_count,
        "distance_count": distance_count,
    }


@dataclass
class EarlyStoppingState:
    """Track a CPU-cloned best model using a strict improvement tolerance."""

    best_score: float = float("-inf")
    best_epoch: int = -1
    wait: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def update(self, score: float, epoch: int, model: nn.Module) -> bool:
        """Record the model only when ``score`` improves by more than 1e-6."""
        numeric_score = float(score)
        if numeric_score > self.best_score + 1e-6:
            self.best_score = numeric_score
            self.best_epoch = int(epoch)
            self.wait = 0
            self.best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            return True
        self.wait += 1
        return False
