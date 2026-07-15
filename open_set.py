"""Validation-fitted rejection for the four researched lightning types."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from data.split_artifacts import stable_json_hash
from evaluation import evaluate_predictions, file_equal_piece_weights, round_metrics


TRANSFERABLE_POLICY_FIELDS = (
    "version", "temperature", "fold_temperatures",
    "probability_thresholds", "margin_thresholds",
    "normalized_distance_thresholds", "quality_thresholds",
    "target_precision", "minimum_coverage", "oof_metrics",
    "fold_hashes", "calibration_hash",
)


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


def attach_final_feature_reference(template, features, labels):
    """Rebind a transferable OOF policy to final-model feature coordinates."""
    if int(template.get("version", 0)) != 3:
        raise ValueError("final feature rebinding requires policy version 3")
    rebound = {
        key: copy.deepcopy(template[key])
        for key in TRANSFERABLE_POLICY_FIELDS
        if key in template
    }
    rebound.update(fit_feature_reference(features, labels, num_types=4))
    return rebound


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


def rejection_signals(logits, features, reference, temperature, quality=None):
    """Return aligned, model-normalized signals used by rejection fitting."""
    logits = _as_float_tensor(logits)
    features = _as_float_tensor(features)
    if logits.ndim != 2 or features.ndim != 2 or len(logits) != len(features):
        raise ValueError("logits and features must be aligned matrices")
    predicted, confidence, margin, normalized_distance = _type_signals(
        logits, features, temperature, reference
    )
    quality_score = (
        waveform_quality_score(quality)
        if quality is not None
        else torch.ones(len(logits), dtype=torch.float32)
    )
    if quality_score.ndim != 1 or len(quality_score) != len(logits):
        raise ValueError("quality must align with logits")
    return {
        "predicted": predicted,
        "confidence": confidence,
        "margin": margin,
        "normalized_distance": normalized_distance,
        "quality_score": quality_score,
    }


def _fit_joint_thresholds(
    confidence,
    margin,
    distance,
    quality,
    correct,
    weights,
    true_count,
    target_precision,
    recall_floor,
):
    """Search nested trust cutoffs, then coordinate-refine for coverage."""
    quantiles = torch.linspace(0.0, 1.0, 101, dtype=confidence.dtype)
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
        accepted_weight = weights[accepted].sum().clamp_min(1e-8)
        precision = float(
            (weights[accepted] * correct[accepted].to(weights.dtype)).sum().item()
            / accepted_weight.item()
        )
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


def _numpy_vector(values, name, length=None, dtype=None):
    """Convert one OOF signal to a finite aligned one-dimensional array."""
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must align with true_type")
    if dtype is not None and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fit_oof_rejection_policy(
    signals,
    target_precision=0.96,
    min_coverage=0.80,
    fold_temperatures=None,
    fold_hashes=None,
):
    """Fit transferable thresholds and return (policy, decoded_decisions)."""
    if not 0 <= float(target_precision) <= 1:
        raise ValueError("target_precision must be in [0, 1]")
    if not 0 <= float(min_coverage) <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    required = (
        "true_type", "predicted", "file_id", "confidence", "margin",
        "normalized_distance", "quality_score",
    )
    missing = [name for name in required if name not in signals]
    if missing:
        raise ValueError(f"OOF signals missing fields: {', '.join(missing)}")

    true_type = _numpy_vector(signals["true_type"], "true_type")
    if not len(true_type):
        raise ValueError("OOF signals must not be empty")
    length = len(true_type)
    predicted = _numpy_vector(signals["predicted"], "predicted", length=length)
    file_id = _numpy_vector(signals["file_id"], "file_id", length=length)
    confidence = _numpy_vector(
        signals["confidence"], "confidence", length=length, dtype=np.float64
    )
    margin = _numpy_vector(
        signals["margin"], "margin", length=length, dtype=np.float64
    )
    normalized_distance = _numpy_vector(
        signals["normalized_distance"],
        "normalized_distance",
        length=length,
        dtype=np.float64,
    )
    quality_score = _numpy_vector(
        signals["quality_score"], "quality_score", length=length, dtype=np.float64
    )
    if not np.equal(true_type, true_type.astype(np.int64)).all():
        raise ValueError("true_type must contain integer class indices")
    if not np.equal(predicted, predicted.astype(np.int64)).all():
        raise ValueError("predicted must contain integer class indices")
    true_type = true_type.astype(np.int64)
    predicted = predicted.astype(np.int64)
    if np.any((true_type < 0) | (true_type >= 4)):
        raise ValueError("true_type must index one of the four researched types")
    if np.any((predicted < 0) | (predicted >= 4)):
        raise ValueError("predicted must index one of the four researched types")

    hard_target_precision = max(float(target_precision), 0.96)
    hard_min_coverage = max(float(min_coverage), 0.80)
    weights = file_equal_piece_weights(file_id)
    accepted = np.zeros(length, dtype=bool)
    probability_thresholds = []
    margin_thresholds = []
    normalized_distance_thresholds = []
    quality_thresholds = []
    for type_index in range(4):
        positions = predicted == type_index
        if not positions.any() or not np.any(positions & (true_type == type_index)):
            raise ValueError(f"type {type_index} has no correct OOF prediction")
        fitted = _fit_joint_thresholds(
            torch.as_tensor(confidence[positions], dtype=torch.float64),
            torch.as_tensor(margin[positions], dtype=torch.float64),
            torch.as_tensor(normalized_distance[positions], dtype=torch.float64),
            torch.as_tensor(quality_score[positions], dtype=torch.float64),
            torch.as_tensor(true_type[positions] == type_index),
            torch.as_tensor(weights[positions], dtype=torch.float64),
            int(np.sum(true_type == type_index)),
            hard_target_precision,
            None,
        )
        if fitted is None:
            raise ValueError(
                f"type {type_index} cannot meet file-equal precision="
                f"{hard_target_precision:.2f}"
            )
        _, thresholds, _, _, class_accepted = fitted
        (
            probability_threshold,
            margin_threshold,
            normalized_distance_threshold,
            quality_threshold,
        ) = thresholds
        accepted[positions] = class_accepted.numpy()
        probability_thresholds.append(float(probability_threshold))
        margin_thresholds.append(float(margin_threshold))
        normalized_distance_thresholds.append(float(normalized_distance_threshold))
        quality_thresholds.append(float(quality_threshold))

    decisions = {
        "predicted": torch.as_tensor(predicted, dtype=torch.long),
        "confidence": torch.as_tensor(confidence, dtype=torch.float32),
        "margin": torch.as_tensor(margin, dtype=torch.float32),
        "normalized_distance": torch.as_tensor(
            normalized_distance, dtype=torch.float32
        ),
        "quality_score": torch.as_tensor(quality_score, dtype=torch.float32),
        "accepted": torch.as_tensor(accepted, dtype=torch.bool),
        "final_type": torch.as_tensor(
            np.where(accepted, predicted, -1), dtype=torch.long
        ),
        "reason": ["accepted" if value else "rejected" for value in accepted],
    }
    records = [{
        "file_id": str(file_id[index]),
        "true_type": int(true_type[index]),
        "predicted_type": int(predicted[index]),
        "accepted": bool(accepted[index]),
    } for index in range(length)]
    oof_metrics = evaluate_predictions(records)
    failing_precision = [
        (index, float(value))
        for index, value in enumerate(oof_metrics["type_file_equal_precision"])
        if float(value) + 1e-12 < hard_target_precision
    ]
    if failing_precision:
        detail = ", ".join(
            f"type {index}={value:.4f}" for index, value in failing_precision
        )
        raise ValueError(f"OOF post-fit precision below contract: {detail}")
    if float(oof_metrics["type_coverage"]) + 1e-12 < hard_min_coverage:
        raise ValueError(
            f"OOF post-fit coverage={float(oof_metrics['type_coverage']):.4f} "
            f"below contract={hard_min_coverage:.4f}"
        )

    if fold_temperatures is None:
        fold_temperatures = {0: 1.0}
    if not hasattr(fold_temperatures, "items") or not fold_temperatures:
        raise ValueError("fold_temperatures must be a non-empty mapping")
    temperature_items = sorted(
        ((str(index), float(value)) for index, value in fold_temperatures.items()),
        key=lambda item: item[0],
    )
    if any(not np.isfinite(value) or value <= 0 for _, value in temperature_items):
        raise ValueError("fold temperatures must be positive and finite")
    fold_temperatures = dict(temperature_items)
    fold_hashes = {} if fold_hashes is None else dict(fold_hashes)
    piece_keys = list(signals.get(
        "piece_key", [str(index) for index in range(length)]
    ))
    if len(piece_keys) != length:
        raise ValueError("piece_key must align with true_type")
    calibration_hash = stable_json_hash({
        "fold_hashes": dict(fold_hashes),
        "piece_keys": piece_keys,
        "accepted": [bool(value) for value in decisions["accepted"]],
        "probability_thresholds": [
            round(float(value), 12) for value in probability_thresholds
        ],
        "margin_thresholds": [
            round(float(value), 12) for value in margin_thresholds
        ],
        "normalized_distance_thresholds": [
            round(float(value), 12) for value in normalized_distance_thresholds
        ],
        "quality_thresholds": [
            round(float(value), 12) for value in quality_thresholds
        ],
    })
    policy = {
        "version": 3,
        "temperature": float(np.median(list(fold_temperatures.values()))),
        "fold_temperatures": fold_temperatures,
        "probability_thresholds": probability_thresholds,
        "margin_thresholds": margin_thresholds,
        "normalized_distance_thresholds": normalized_distance_thresholds,
        "quality_thresholds": quality_thresholds,
        "target_precision": float(target_precision),
        "minimum_coverage": float(min_coverage),
        "oof_metrics": round_metrics(oof_metrics),
        "fold_hashes": dict(fold_hashes),
        "calibration_hash": calibration_hash,
    }
    return policy, decisions


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
    groups=None,
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
    groups_are_files = groups is not None
    if groups is None:
        groups = torch.arange(len(labels), dtype=torch.long)
    else:
        groups = torch.as_tensor(groups, dtype=torch.long).detach().cpu()
        if groups.ndim != 1 or len(groups) != len(labels):
            raise ValueError("groups must align with logits and labels")
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
        _, inverse, group_counts = torch.unique(
            groups[class_positions], return_inverse=True, return_counts=True
        )
        file_equal_weights = 1.0 / group_counts[inverse].to(torch.float32)
        fitted = _fit_joint_thresholds(
            confidence[class_positions],
            margin[class_positions],
            distance[class_positions],
            quality_score[class_positions],
            labels[class_positions] == type_idx,
            file_equal_weights,
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
        "precision_weighting": (
            "equal_source_file_v1" if groups_are_files else "piece_v1"
        ),
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
    policy_version = int(policy.get("version", 1))
    if policy_version >= 3:
        if "normalized_distance_thresholds" not in policy:
            raise ValueError(
                f"policy version {policy_version} requires "
                "normalized_distance_thresholds"
            )
        distance_thresholds = _as_float_tensor(
            policy["normalized_distance_thresholds"]
        )
    else:
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
