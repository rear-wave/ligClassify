"""Training losses and steps for the conditional expert model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from distance_ordinal import interval_distance_loss


COARSE_DISTANCE_EDGES_KM = (0, 300, 600, 1200, 1700, 2400, 3000)


def coarse_interval_loss(logits, low_km, high_km):
    """Reward probability assigned to every coarse band overlapping a label."""
    if logits.ndim != 2 or logits.shape[1] != 6:
        raise ValueError("coarse logits must have shape [batch, 6]")
    low_km = torch.as_tensor(low_km, device=logits.device, dtype=logits.dtype)
    high_km = torch.as_tensor(high_km, device=logits.device, dtype=logits.dtype)
    if low_km.shape != high_km.shape or low_km.ndim != 1 or len(low_km) != len(logits):
        raise ValueError("coarse intervals must be aligned one-dimensional tensors")
    if torch.any((low_km < 0) | (high_km > 3000) | (high_km <= low_km)):
        raise ValueError("coarse intervals must satisfy 0 <= low < high <= 3000")

    edges = torch.tensor(
        COARSE_DISTANCE_EDGES_KM,
        device=logits.device,
        dtype=logits.dtype,
    )
    band_low = edges[:-1].unsqueeze(0)
    band_high = edges[1:].unsqueeze(0)
    compatible = (
        (band_low < high_km.unsqueeze(1))
        & (band_high > low_km.unsqueeze(1))
    )
    probabilities = F.softmax(logits, dim=1)
    mass = (probabilities * compatible).sum(dim=1).clamp_min(1e-8)
    return -mass.log().mean()


def compute_conditional_distance_loss(
    distance_logits,
    coarse_logits,
    type_labels,
    low_km,
    high_km,
    coarse_weight=0.5,
    ordered_weight=0.2,
):
    """Macro-average interval losses across represented true types."""
    if len(distance_logits) != 4 or len(coarse_logits) != 4:
        raise ValueError("conditional distance training requires four expert heads")
    valid = (low_km >= 0) & (high_km > low_km)
    interval_losses = []
    coarse_losses = []
    for lightning_type in range(4):
        mask = valid & (type_labels == lightning_type)
        if not mask.any():
            continue
        interval_loss, _ = interval_distance_loss(
            distance_logits[lightning_type][mask],
            low_km[mask],
            high_km[mask],
            ordered_weight=ordered_weight,
        )
        interval_losses.append(interval_loss)
        coarse_losses.append(coarse_interval_loss(
            coarse_logits[lightning_type][mask],
            low_km[mask],
            high_km[mask],
        ))

    if not interval_losses:
        zero = distance_logits[0].sum() * 0.0
        return zero, {"interval": zero, "coarse": zero}
    interval = torch.stack(interval_losses).mean()
    coarse = torch.stack(coarse_losses).mean()
    return interval + coarse_weight * coarse, {
        "interval": interval,
        "coarse": coarse,
    }


def conditional_train_step(
    model,
    batch,
    stream,
    optimizer,
    type_criterion,
    distance_loss_weight=1.0,
    distance_batch_type_weight=0.1,
    coarse_weight=0.5,
    ordered_weight=0.2,
    scaler=None,
    amp=False,
):
    """Optimize one type or condition-balanced interval-distance batch."""
    if stream not in {"type", "distance"}:
        raise ValueError(f"Unknown training stream: {stream}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    local = batch["local"]
    global_view = batch["global_view"]
    context = batch["context"]
    labels = batch["type_label"]
    device_type = local.device.type
    amp_enabled = bool(amp and device_type == "cuda")

    with torch.autocast(device_type=device_type, enabled=amp_enabled):
        if stream == "type":
            type_logits = model.forward_type(local, global_view)
            type_loss = type_criterion(type_logits, labels)
            distance_loss = type_loss.new_zeros(())
            total_loss = type_loss
            components = {}
            distance_count = 0
        else:
            type_logits, distance_logits, coarse_logits = model(
                local, global_view, context
            )
            type_loss = type_criterion(type_logits, labels)
            distance_loss, components = compute_conditional_distance_loss(
                distance_logits,
                coarse_logits,
                labels,
                batch["distance_low_km"],
                batch["distance_high_km"],
                coarse_weight=coarse_weight,
                ordered_weight=ordered_weight,
            )
            total_loss = (
                distance_loss_weight * distance_loss
                + distance_batch_type_weight * type_loss
            )
            distance_count = int((
                (batch["distance_low_km"] >= 0)
                & (batch["distance_high_km"] > batch["distance_low_km"])
            ).sum().item())

    if scaler is not None and amp_enabled:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
    return {
        "type_loss": float(type_loss.detach().item()),
        "distance_loss": float(distance_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "type_correct": int((type_logits.argmax(1) == labels).sum().item()),
        "type_count": int(len(labels)),
        "distance_count": distance_count,
        "components": {
            name: float(value.detach().item())
            for name, value in components.items()
        },
    }
