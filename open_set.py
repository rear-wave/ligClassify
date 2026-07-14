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


def fit_rejection_policy(
    logits,
    features,
    labels,
    reference,
    precision_floor=0.85,
    recall_floor=0.70,
):
    """Fit per-predicted-type thresholds using labelled validation samples."""
    logits = _as_float_tensor(logits)
    features = _as_float_tensor(features)
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if logits.ndim != 2 or features.ndim != 2 or labels.ndim != 1:
        raise ValueError("logits, features, and labels have invalid dimensions")
    if len(logits) != len(features) or len(logits) != len(labels):
        raise ValueError("logits, features, and labels must be aligned")
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
        candidates = None
        for probability_threshold in _threshold_values(
            confidence[class_positions], lower_is_stricter=False
        ):
            for margin_threshold in _threshold_values(
                margin[class_positions], lower_is_stricter=False
            ):
                for distance_threshold in _threshold_values(
                    distance[class_positions], lower_is_stricter=True
                ):
                    accepted = predicted_mask
                    accepted = accepted & (confidence >= probability_threshold)
                    accepted = accepted & (margin >= margin_threshold)
                    accepted = accepted & (distance <= distance_threshold)
                    accepted_count = int(accepted.sum().item())
                    if not accepted_count:
                        continue
                    correct = int((accepted & (labels == type_idx)).sum().item())
                    precision = correct / accepted_count
                    recall = correct / true_count
                    if precision < precision_floor or recall < recall_floor:
                        continue
                    score = (accepted_count, correct, -distance_threshold)
                    if candidates is None or score > candidates[0]:
                        candidates = (
                            score,
                            probability_threshold,
                            margin_threshold,
                            distance_threshold,
                            precision,
                            recall,
                            accepted,
                        )
        if candidates is None:
            raise ValueError(
                f"type {type_idx} cannot meet precision={precision_floor:.2f} "
                f"and recall={recall_floor:.2f}"
            )
        (
            _, probability_threshold, margin_threshold, distance_threshold,
            precision, recall, accepted,
        ) = candidates
        probability_thresholds.append(float(probability_threshold))
        margin_thresholds.append(float(margin_threshold))
        distance_thresholds.append(float(distance_threshold))
        validation_precision.append(float(precision))
        validation_recall.append(float(recall))
        accepted_all |= accepted

    return {
        "version": 1,
        "temperature": temperature,
        "centroids": reference["centroids"],
        "scales": reference["scales"],
        "probability_thresholds": probability_thresholds,
        "margin_thresholds": margin_thresholds,
        "distance_thresholds": distance_thresholds,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "validation_coverage": float(accepted_all.float().mean().item()),
    }


def decode_with_rejection(logits, features, policy):
    """Decode researched types and explain every rejected prediction."""
    logits = _as_float_tensor(logits)
    features = _as_float_tensor(features)
    predicted, confidence, margin, distance = _type_signals(
        logits, features, policy["temperature"], policy
    )
    probability_thresholds = _as_float_tensor(policy["probability_thresholds"])
    margin_thresholds = _as_float_tensor(policy["margin_thresholds"])
    distance_thresholds = _as_float_tensor(policy["distance_thresholds"])

    accepted = torch.ones(len(logits), dtype=torch.bool)
    reasons = []
    for row, type_idx in enumerate(predicted.tolist()):
        if confidence[row] < probability_thresholds[type_idx]:
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
        "accepted": accepted,
        "reason": reasons,
    }
