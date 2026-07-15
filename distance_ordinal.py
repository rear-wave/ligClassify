"""Ordinal objectives and reliability helpers for distance estimation."""

import torch
import torch.nn.functional as F


def aggregate_coarse_probabilities(probabilities, group_size=3):
    """Sum consecutive fine-bin probabilities into consistent coarse bins."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, bins]")
    if group_size <= 0 or probabilities.shape[1] % group_size:
        raise ValueError("fine-bin count must be divisible by group_size")
    return probabilities.reshape(
        probabilities.shape[0], -1, group_size
    ).sum(dim=2)


def _soft_targets(targets, num_bins, tau, dtype):
    if tau <= 0:
        raise ValueError("tau must be positive")
    bins = torch.arange(num_bins, device=targets.device, dtype=dtype)
    distances = (bins.unsqueeze(0) - targets.unsqueeze(1).to(dtype)).abs()
    weights = torch.exp(-distances / tau)
    return weights / weights.sum(dim=1, keepdim=True)


def ordinal_distance_loss(
    logits,
    targets,
    tau=1.0,
    lambda_emd=1.0,
    lambda_reg=0.5,
    lambda_coarse=0.5,
    coarse_group_size=3,
):
    """Combine soft CE, CDF distance, expected-bin Huber, and coarse NLL."""
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits and targets must have shapes [B, C] and [B]")
    if len(targets) == 0:
        raise ValueError("distance loss requires at least one target")
    num_bins = logits.shape[1]
    if torch.any((targets < 0) | (targets >= num_bins)):
        raise ValueError("targets must be valid distance-bin indices")

    log_probabilities = F.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    soft_targets = _soft_targets(targets, num_bins, tau, logits.dtype)

    soft_ce = -(soft_targets * log_probabilities).sum(dim=1).mean()
    cdf = (
        probabilities.cumsum(dim=1) - soft_targets.cumsum(dim=1)
    ).abs().mean()

    bins = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    expected_bins = (probabilities * bins.unsqueeze(0)).sum(dim=1)
    huber = F.smooth_l1_loss(expected_bins, targets.to(logits.dtype))

    coarse = aggregate_coarse_probabilities(
        probabilities, group_size=coarse_group_size
    )
    coarse_targets = torch.div(
        targets, coarse_group_size, rounding_mode="floor"
    )
    coarse_target_probability = coarse.gather(
        1, coarse_targets.unsqueeze(1)
    ).squeeze(1)
    coarse_loss = -coarse_target_probability.clamp_min(1e-12).log().mean()

    components = {
        "soft_ce": soft_ce,
        "cdf": cdf,
        "huber": huber,
        "coarse": coarse_loss,
    }
    total = (
        soft_ce
        + lambda_emd * cdf
        + lambda_reg * huber
        + lambda_coarse * coarse_loss
    )
    return total, components


def interval_distance_loss(
    logits,
    low_km,
    high_km,
    ordered_weight=0.2,
):
    """Train an ordered distribution from exact or interval distance labels."""
    low_km = torch.as_tensor(low_km, device=logits.device)
    high_km = torch.as_tensor(high_km, device=logits.device)
    if (
        logits.ndim != 2
        or low_km.ndim != 1
        or high_km.ndim != 1
        or len(logits) != len(low_km)
        or len(logits) != len(high_km)
    ):
        raise ValueError(
            "logits, low_km, and high_km must have shapes [B, C], [B], [B]"
        )
    if len(logits) == 0:
        raise ValueError("distance loss requires at least one interval")
    max_distance_km = logits.shape[1] * 100
    if torch.any(
        (low_km < 0)
        | (high_km > max_distance_km)
        | (high_km <= low_km)
    ):
        raise ValueError("distance intervals must satisfy 0 <= low < high <= max")
    if ordered_weight < 0:
        raise ValueError("ordered_weight must be non-negative")

    probabilities = F.softmax(logits, dim=1)
    centers = torch.arange(
        50,
        max_distance_km,
        100,
        device=logits.device,
        dtype=logits.dtype,
    )
    low = low_km.to(logits.dtype).unsqueeze(1)
    high = high_km.to(logits.dtype).unsqueeze(1)
    inside = (centers.unsqueeze(0) >= low) & (centers.unsqueeze(0) < high)
    interval_mass = (probabilities * inside).sum(dim=1).clamp_min(1e-8)
    interval_nll = -interval_mass.log().mean()

    below = (low - centers.unsqueeze(0)).clamp_min(0)
    above = (centers.unsqueeze(0) - high).clamp_min(0)
    outside_bins = torch.maximum(below, above) / 100.0
    ordered = (probabilities * outside_bins).sum(dim=1).mean()
    total = interval_nll + ordered_weight * ordered
    return total, {"interval_nll": interval_nll, "ordered": ordered}


def decode_distance_logits(logits, temperature=1.0):
    """Decode an ordered distribution into point, interval, and confidence."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = F.softmax(logits / temperature, dim=1)
    num_bins = logits.shape[1]
    bins = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    expected_bins = (probabilities * bins.unsqueeze(0)).sum(dim=1)
    predicted_bins = expected_bins.round().long().clamp(0, num_bins - 1)

    cdf = probabilities.cumsum(dim=1)
    low_bins = (cdf >= 0.10).to(torch.int64).argmax(dim=1)
    high_bins = (cdf >= 0.90).to(torch.int64).argmax(dim=1)

    integer_bins = torch.arange(num_bins, device=logits.device).unsqueeze(0)
    within_two = (integer_bins - predicted_bins.unsqueeze(1)).abs() <= 2
    confidence = (probabilities * within_two).sum(dim=1)

    return {
        "probabilities": probabilities,
        "expected_bin": expected_bins,
        "bin": predicted_bins,
        "distance_km": 100.0 * (expected_bins + 0.5),
        "low_km": 100.0 * low_bins.to(logits.dtype),
        "high_km": 100.0 * (high_bins.to(logits.dtype) + 1.0),
        "confidence": confidence,
    }


def decode_distance_distribution(logits, temperature=1.0):
    """Decode expected distance, modal bin, quantiles, and local confidence."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, bins]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = F.softmax(logits / temperature, dim=1)
    num_bins = logits.shape[1]
    centers = torch.arange(
        50,
        num_bins * 100,
        100,
        device=logits.device,
        dtype=logits.dtype,
    )
    expected_km = (probabilities * centers.unsqueeze(0)).sum(dim=1)
    bin_index = probabilities.argmax(dim=1)
    cdf = probabilities.cumsum(dim=1)
    low_bins = (cdf >= 0.10).to(torch.int64).argmax(dim=1)
    high_bins = (cdf >= 0.90).to(torch.int64).argmax(dim=1)
    integer_bins = torch.arange(num_bins, device=logits.device).unsqueeze(0)
    within_two = (integer_bins - bin_index.unsqueeze(1)).abs() <= 2
    confidence = (probabilities * within_two).sum(dim=1)
    return {
        "probabilities": probabilities,
        "expected_km": expected_km,
        "bin_index": bin_index,
        "low_km": low_bins.to(logits.dtype) * 100.0,
        "high_km": (high_bins.to(logits.dtype) + 1.0) * 100.0,
        "confidence": confidence,
    }


def fit_temperature_grid(logits, targets):
    """Choose a validation-only scalar temperature by categorical NLL."""
    logits = logits.detach()
    targets = targets.detach()
    best_temperature = 1.0
    best_loss = float(F.cross_entropy(logits, targets).item())
    for temperature in torch.arange(0.50, 5.01, 0.05).tolist():
        loss = float(F.cross_entropy(logits / temperature, targets).item())
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def select_confidence_threshold(confidence, error_bins, target_w2=0.8):
    """Maximize retained coverage subject to a validation w2 requirement."""
    confidence = torch.as_tensor(confidence, dtype=torch.float32)
    error_bins = torch.as_tensor(error_bins, dtype=torch.float32)
    if confidence.ndim != 1 or error_bins.shape != confidence.shape:
        raise ValueError("confidence and error_bins must be aligned vectors")
    if not 0 <= target_w2 <= 1:
        raise ValueError("target_w2 must be in [0, 1]")
    if len(confidence) == 0:
        return {"threshold": 1.0, "coverage": 0.0, "w2": 0.0}

    for threshold in torch.unique(confidence).sort().values:
        kept = confidence >= threshold
        w2 = float((error_bins[kept] <= 2).float().mean().item())
        if w2 >= target_w2:
            return {
                "threshold": float(threshold.item()),
                "coverage": float(kept.float().mean().item()),
                "w2": w2,
            }
    return {"threshold": 1.0, "coverage": 0.0, "w2": 0.0}


def make_selection_key(metrics, guardrail=0.70, min_type_f1=0.85):
    """Return the validation ordering used by early stopping."""
    per_type = [float(value) for value in metrics["per_type_w2"]]
    first = float(metrics["w2"]) if min(per_type) >= guardrail else min(per_type)
    return (
        float(float(metrics["type_f1"]) >= min_type_f1),
        first,
        float(metrics["macro_w2"]),
        -float(metrics["macro_mae_km"]),
        float(metrics["type_f1"]),
    )


def is_meaningful_improvement(
    candidate,
    best,
    tolerances=None,
):
    """Compare early-stopping keys while ignoring numerical noise."""
    if best is None:
        return True
    if tolerances is None:
        tolerances = (
            (0.0, 0.002, 0.002, 5.0, 0.001)
            if len(candidate) == 5
            else (0.002, 0.002, 5.0, 0.001)
        )
    for candidate_value, best_value, tolerance in zip(
        candidate, best, tolerances
    ):
        if candidate_value > best_value + tolerance:
            return True
        if candidate_value < best_value - tolerance:
            return False
    return False
