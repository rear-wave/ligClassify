"""Distance evaluation metrics that do not depend on model code."""

import numpy as np


def interval_distance_errors(predictions, low_km, high_km):
    """Return distance to the nearest valid interval boundary in kilometres."""
    predictions = np.asarray(predictions, dtype=np.float64)
    low_km = np.asarray(low_km, dtype=np.float64)
    high_km = np.asarray(high_km, dtype=np.float64)
    if not (
        predictions.ndim == low_km.ndim == high_km.ndim == 1
        and predictions.shape == low_km.shape == high_km.shape
    ):
        raise ValueError("predictions and interval bounds must be aligned vectors")
    if np.any((low_km < 0) | (high_km <= low_km)):
        raise ValueError("distance intervals must be valid")
    return np.maximum.reduce([
        low_km - predictions,
        predictions - high_km,
        np.zeros_like(predictions),
    ])


def summarize_equal_bin_distance_predictions(predictions, targets):
    """Average metrics equally over populated true-distance bins."""
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if predictions.shape != targets.shape or not len(targets):
        raise ValueError("non-empty aligned predictions and targets are required")

    summaries = []
    for distance_bin in sorted(np.unique(targets)):
        mask = targets == distance_bin
        error = np.abs(predictions[mask] - targets[mask])
        summaries.append({
            "acc": float(np.mean(error == 0)),
            "mae_km": float(np.mean(error) * 100),
            "w1": float(np.mean(error <= 1)),
            "w2": float(np.mean(error <= 2)),
        })

    result = {
        key: float(np.mean([summary[key] for summary in summaries]))
        for key in ("acc", "mae_km", "w1", "w2")
    }
    result["bin_count"] = len(summaries)
    return result
