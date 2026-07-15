"""Validation-fitted rejection for the four researched lightning types."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _as_float_tensor(values):
    return torch.as_tensor(values, dtype=torch.float32).detach().cpu()


def fit_feature_reference(features, labels, num_types=4):
    """Fit per-type diagonal feature distributions without negative samples."""
    features = _as_float_tensor(features)
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels must be aligned matrices/vectors")

    centroids = []
    scales = []
    for type_idx in range(int(num_types)):
        selected = features[labels == type_idx]
        if not len(selected):
            raise ValueError(f"type {type_idx} has no feature samples")
        centroids.append(selected.mean(dim=0))
        scales.append(selected.std(dim=0, correction=0).clamp_min(1e-6))
    return {
        "centroids": torch.stack(centroids).tolist(),
        "scales": torch.stack(scales).tolist(),
    }


def _predicted_feature_distance(features, predicted, reference):
    centroids = _as_float_tensor(reference["centroids"])
    scales = _as_float_tensor(reference["scales"]).clamp_min(1e-6)
    centered = (features - centroids[predicted]) / scales[predicted]
    return centered.square().mean(dim=1).sqrt()


def _type_signals(logits, features, temperature, reference):
    probabilities = torch.softmax(logits / float(temperature), dim=1)
    top = probabilities.topk(k=2, dim=1)
    predicted = top.indices[:, 0]
    confidence = top.values[:, 0]
    margin = top.values[:, 0] - top.values[:, 1]
    distance = _predicted_feature_distance(features, predicted, reference)
    return predicted, confidence, margin, distance


def _fit_temperature(logits, labels):
    candidates = torch.arange(0.50, 5.01, 0.05)
    losses = torch.stack([
        F.cross_entropy(logits / float(value), labels)
        for value in candidates
    ])
    return float(candidates[int(losses.argmin())].item())


def _threshold_values(values, lower_is_stricter):
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.unique(np.quantile(array, np.linspace(0.0, 1.0, 11)))
    if lower_is_stricter:
        return sorted(quantiles.tolist(), reverse=True)
    return sorted(quantiles.tolist())


def waveform_quality_score(quality):
    """Compress waveform diagnostics to a bounded higher-is-better score."""
    quality = _as_float_tensor(quality)
    if quality.ndim == 1:
        return quality.clamp(0.0, 1.0)
    if quality.ndim != 2 or quality.shape[1] not in {1, 3}:
        raise ValueError("quality must have shape [batch], [batch, 1], or [batch, 3]")
    if quality.shape[1] == 1:
        return quality[:, 0].clamp(0.0, 1.0)
    snr = (quality[:, 0] / 20.0).clamp(0.0, 1.0)
    unclipped = (1.0 - quality[:, 1]).clamp(0.0, 1.0)
    stable = torch.exp(-quality[:, 2].clamp_min(0.0) / 5.0)
    return (snr * unclipped * stable).clamp(0.0, 1.0)


def _fit_joint_thresholds(
    confidence,
    margin,
    distance,
    quality,
    correct,
    true_count,
    target_precision,
    recall_floor,
):
    """Search nested trust cutoffs, then coordinate-refine for coverage."""
    quantiles = torch.linspace(0.0, 1.0, 101)
    probability_values = torch.quantile(confidence, quantiles)
    margin_values = torch.quantile(margin, quantiles)
    distance_values = torch.quantile(distance, 1.0 - quantiles)
    quality_values = torch.quantile(quality, quantiles)
    permissive = (
        float(confidence.min()),
        float(margin.min()),
        float(distance.max()),
        float(quality.min()),
    )
    best = None

    def consider(thresholds):
        nonlocal best
        probability_threshold, margin_threshold, distance_threshold, quality_threshold = thresholds
        accepted = (
            (confidence >= probability_threshold)
            & (margin >= margin_threshold)
            & (distance <= distance_threshold)
            & (quality >= quality_threshold)
        )
        accepted_count = int(accepted.sum().item())
        if not accepted_count:
            return
        correct_count = int((accepted & correct).sum().item())
        precision = correct_count / accepted_count
        recall = correct_count / true_count
        if precision < target_precision:
            return
        if recall_floor is not None and recall < recall_floor:
            return
        score = (accepted_count, correct_count, -float(distance_threshold))
        if best is None or score > best[0]:
            best = (score, tuple(float(value) for value in thresholds), precision, recall, accepted)

    for index in range(len(quantiles)):
        consider((
            probability_values[index],
            margin_values[index],
            distance_values[index],
            quality_values[index],
        ))
        for dimension, values in enumerate((
            probability_values, margin_values, distance_values, quality_values
        )):
            thresholds = list(permissive)
            thresholds[dimension] = values[index]
            consider(thresholds)

    if best is None:
        return None
    for _ in range(2):
        for dimension, values in enumerate((
            probability_values, margin_values, distance_values, quality_values
        )):
            fixed = list(best[1])
            for value in values:
                thresholds = fixed.copy()
                thresholds[dimension] = value
                consider(thresholds)
    return best


def fit_rejection_policy(
    logits,
    features,
    labels,
    reference,
    precision_floor=None,
    recall_floor=None,
    quality=None,
    target_precision=0.95,
    min_coverage=0.80,
    calibration_split_hash="",
):
    """Maximize validation coverage under per-type precision constraints."""
    logits = _as_float_tensor(logits)
    features = _as_float_tensor(features)
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if logits.ndim != 2 or features.ndim != 2 or labels.ndim != 1:
        raise ValueError("logits, features, and labels have invalid dimensions")
    if len(logits) != len(features) or len(logits) != len(labels):
        raise ValueError("logits, features, and labels must be aligned")
    if precision_floor is not None:
        target_precision = float(precision_floor)
    if not 0 <= target_precision <= 1:
        raise ValueError("target_precision must be in [0, 1]")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    if quality is None:
        quality_score = torch.ones(len(labels), dtype=torch.float32)
    else:
        quality_score = waveform_quality_score(quality)
        if len(quality_score) != len(labels):
            raise ValueError("quality must align with logits and labels")
    num_types = logits.shape[1]
    temperature = _fit_temperature(logits, labels)
    predicted, confidence, margin, distance = _type_signals(
        logits, features, temperature, reference
    )
    for type_idx in range(num_types):
        if not ((predicted == type_idx) & (labels == type_idx)).any():
            raise ValueError(f"type {type_idx} has no correct validation prediction")

    probability_thresholds = []
    margin_thresholds = []
    distance_thresholds = []
    quality_thresholds = []
    validation_precision = []
    validation_recall = []
    accepted_all = torch.zeros(len(labels), dtype=torch.bool)

    for type_idx in range(num_types):
        predicted_mask = predicted == type_idx
        true_count = int((labels == type_idx).sum().item())
        correct_mask = predicted_mask & (labels == type_idx)
        if true_count == 0 or not correct_mask.any():
            raise ValueError(f"type {type_idx} has no correct validation prediction")

        class_positions = torch.nonzero(predicted_mask, as_tuple=False).flatten()
        fitted = _fit_joint_thresholds(
            confidence[class_positions],
            margin[class_positions],
            distance[class_positions],
            quality_score[class_positions],
            labels[class_positions] == type_idx,
            true_count,
            target_precision,
            recall_floor,
        )
        if fitted is None:
            raise ValueError(
                f"type {type_idx} cannot meet precision={target_precision:.2f}"
            )
        (
            _, thresholds, precision, recall, class_accepted,
        ) = fitted
        (
            probability_threshold,
            margin_threshold,
            distance_threshold,
            quality_threshold,
        ) = thresholds
        accepted = torch.zeros(len(labels), dtype=torch.bool)
        accepted[class_positions] = class_accepted
        probability_thresholds.append(float(probability_threshold))
        margin_thresholds.append(float(margin_threshold))
        distance_thresholds.append(float(distance_threshold))
        quality_thresholds.append(float(quality_threshold))
        validation_precision.append(float(precision))
        validation_recall.append(float(recall))
        accepted_all |= accepted

    validation_coverage = float(accepted_all.float().mean().item())
    if validation_coverage < min_coverage:
        raise ValueError(
            f"calibrated coverage={validation_coverage:.4f} below "
            f"minimum={min_coverage:.4f}"
        )

    return {
        "version": 2,
        "temperature": temperature,
        "centroids": reference["centroids"],
        "scales": reference["scales"],
        "probability_thresholds": probability_thresholds,
        "margin_thresholds": margin_thresholds,
        "distance_thresholds": distance_thresholds,
        "quality_thresholds": quality_thresholds,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "validation_coverage": validation_coverage,
        "target_precision": float(target_precision),
        "minimum_coverage": float(min_coverage),
        "calibration_split_hash": str(calibration_split_hash),
        "quality_score": "snr_unclipped_stable_v1",
    }


def decode_with_rejection(logits, features, policy, quality=None):
    """Decode researched types and explain every rejected prediction."""
    logits = _as_float_tensor(logits)
    features = _as_float_tensor(features)
    predicted, confidence, margin, distance = _type_signals(
        logits, features, policy["temperature"], policy
    )
    probability_thresholds = _as_float_tensor(policy["probability_thresholds"])
    margin_thresholds = _as_float_tensor(policy["margin_thresholds"])
    distance_thresholds = _as_float_tensor(policy["distance_thresholds"])
    quality_thresholds = _as_float_tensor(
        policy.get("quality_thresholds", [0.0] * len(probability_thresholds))
    )
    quality_score = (
        torch.ones(len(logits), dtype=torch.float32)
        if quality is None
        else waveform_quality_score(quality)
    )
    if len(quality_score) != len(logits):
        raise ValueError("quality must align with logits")

    accepted = torch.ones(len(logits), dtype=torch.bool)
    reasons = []
    for row, type_idx in enumerate(predicted.tolist()):
        if quality_score[row] < quality_thresholds[type_idx]:
            accepted[row] = False
            reasons.append("low_quality")
        elif confidence[row] < probability_thresholds[type_idx]:
            accepted[row] = False
            reasons.append("low_probability")
        elif margin[row] < margin_thresholds[type_idx]:
            accepted[row] = False
            reasons.append("low_margin")
        elif distance[row] > distance_thresholds[type_idx]:
            accepted[row] = False
            reasons.append("feature_distance")
        else:
            reasons.append("accepted")
    return {
        "predicted": predicted,
        "confidence": confidence,
        "margin": margin,
        "feature_distance": distance,
        "quality_score": quality_score,
        "accepted": accepted,
        "final_type": torch.where(accepted, predicted, -torch.ones_like(predicted)),
        "reason": reasons,
    }
