"""Record-based evaluation and release gates for lightning classifiers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from data.cross_validation import MINIMUM_SUPPORTED_PIECES
from distance_metrics import interval_distance_errors


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
COARSE_DISTANCE_EDGES_KM = (0, 300, 600, 1200, 1700, 2400, 3000)
RELEASE_GATES = {
    "min_per_type_precision": 0.95,
    "min_coverage": 0.80,
    "min_file_equal_macro_recall": 0.90,
    "min_100km_interval_within_200": 0.85,
    "min_per_type_100km_interval_within_200": 0.75,
    "min_supported_condition_within_200": 0.55,
    "minimum_supported_pieces": MINIMUM_SUPPORTED_PIECES,
}


def distance_calibration_is_safe(before, after, tolerance=1e-12):
    """Accept calibration only when every reported point metric is preserved."""
    return (
        float(after["distance_file_macro_within_200"]) + tolerance
        >= float(before["distance_file_macro_within_200"])
        and float(after["distance_100km_interval_within_200"]) + tolerance
        >= float(before["distance_100km_interval_within_200"])
        and float(after["distance_interval_mae_km"])
        <= float(before["distance_interval_mae_km"]) + tolerance
    )


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def record_file_identity(record):
    """Return the stable source identity carried by a prediction record."""
    if record.get("source_path"):
        return str(record["source_path"])
    if "file_id" in record:
        return str(record["file_id"])
    raise ValueError("prediction record has no stable source identity")


def file_equal_piece_weights(file_ids):
    """Give every source file total weight one across all of its pieces."""
    file_ids = np.asarray(file_ids)
    _, inverse, counts = np.unique(file_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def weighted_type_metrics(
    true_types,
    predicted_types,
    accepted,
    weights,
    num_types=4,
):
    """Compute per-type precision and recall using caller-supplied weights."""
    precision, recall = [], []
    for type_index in range(num_types):
        predicted = accepted & (predicted_types == type_index)
        actual = true_types == type_index
        correct = predicted & actual
        precision.append(_safe_ratio(weights[correct].sum(), weights[predicted].sum()))
        recall.append(_safe_ratio(weights[correct].sum(), weights[actual].sum()))
    return precision, recall


def round_metrics(value):
    """Recursively round metrics deterministically and reject non-finite values."""
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [round_metrics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("metrics contain a non-finite value")
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _coarse_band(low_km, high_km):
    midpoint = (float(low_km) + float(high_km)) / 2.0
    index = int(np.searchsorted(COARSE_DISTANCE_EDGES_KM, midpoint, side="right") - 1)
    index = min(max(index, 0), len(COARSE_DISTANCE_EDGES_KM) - 2)
    return COARSE_DISTANCE_EDGES_KM[index], COARSE_DISTANCE_EDGES_KM[index + 1]


def _distance_summary(records, prediction_key="predicted_distance_km", exact_only=False):
    selected = []
    for record in records:
        low = record.get("distance_low_km")
        high = record.get("distance_high_km")
        prediction = record.get(prediction_key)
        if low is None or high is None:
            continue
        if not (0 <= float(low) < float(high) <= 3000):
            continue
        if exact_only and float(high) - float(low) != 100:
            continue
        if prediction_key == "predicted_distance_km" and not record.get("accepted", True):
            prediction = None
        selected.append((
            np.nan if prediction is None else float(prediction),
            float(low),
            float(high),
        ))
    if not selected:
        return {
            "count": 0,
            "prediction_count": 0,
            "coverage": 0.0,
            "mae_km": 0.0,
            "within_100": 0.0,
            "within_200": 0.0,
        }
    values = np.asarray(selected, dtype=np.float64)
    covered = np.isfinite(values[:, 0])
    errors = np.full(len(values), np.inf, dtype=np.float64)
    if covered.any():
        errors[covered] = interval_distance_errors(
            values[covered, 0], values[covered, 1], values[covered, 2]
        )
    return {
        "count": int(len(errors)),
        "prediction_count": int(covered.sum()),
        "coverage": float(covered.mean()),
        "mae_km": float(errors[covered].mean()) if covered.any() else 0.0,
        "within_100": float(np.mean(errors <= 100)),
        "within_200": float(np.mean(errors <= 200)),
    }


def distance_condition_metrics(records):
    """Report file-macro quality for each type/light/100-km condition."""
    conditions = defaultdict(list)
    for record in records:
        low = record.get("distance_low_km")
        high = record.get("distance_high_km")
        if low is None or high is None or float(high) - float(low) != 100:
            continue
        type_name = TYPE_NAMES[int(record["true_type"])]
        light = "day" if bool(record.get("daylight", False)) else "night"
        name = f"{type_name}/{light}/{int(low)}-{int(high)}km"
        conditions[name].append(record)
    result = {}
    for name, selected in sorted(conditions.items()):
        by_file = defaultdict(list)
        for record in selected:
            by_file[record_file_identity(record)].append(record)
        file_scores = [
            _distance_summary(rows, exact_only=True)["within_200"]
            for rows in by_file.values()
        ]
        result[name] = {
            "piece_count": len(selected),
            "file_count": len(by_file),
            "file_macro_within_200": float(np.mean(file_scores)),
        }
    return result


def evaluate_predictions(records):
    """Summarize raw, rejected, distance, file-macro, and subgroup quality."""
    records = [dict(record) for record in records]
    if not records:
        raise ValueError("at least one prediction record is required")
    true_types = np.asarray([int(record["true_type"]) for record in records])
    predicted_types = np.asarray([int(record["predicted_type"]) for record in records])
    accepted = np.asarray([
        bool(record.get("accepted", int(record["predicted_type"]) >= 0))
        for record in records
    ])
    if np.any((true_types < 0) | (true_types >= len(TYPE_NAMES))):
        raise ValueError("true_type must index one of the four researched types")

    file_ids = np.asarray([record_file_identity(record) for record in records])
    piece_weights = np.ones(len(records), dtype=np.float64)
    equal_file_weights = file_equal_piece_weights(file_ids)
    raw_correct = predicted_types == true_types
    precision, recall = weighted_type_metrics(
        true_types,
        predicted_types,
        accepted,
        piece_weights,
    )
    raw_precision, raw_recall = weighted_type_metrics(
        true_types,
        predicted_types,
        np.ones(len(records), dtype=bool),
        piece_weights,
    )
    file_equal_precision, file_equal_recall = weighted_type_metrics(
        true_types,
        predicted_types,
        accepted,
        equal_file_weights,
    )
    raw_file_equal_precision, raw_file_equal_recall = weighted_type_metrics(
        true_types,
        predicted_types,
        np.ones(len(records), dtype=bool),
        equal_file_weights,
    )

    by_file = defaultdict(list)
    for index, file_id in enumerate(file_ids):
        by_file[str(file_id)].append(index)
    file_type_accuracy = [
        float(raw_correct[positions].mean()) for positions in by_file.values()
    ]

    interval_summary = _distance_summary(records)
    interval_100km_summary = _distance_summary(records, exact_only=True)
    oracle_summary = _distance_summary(records, prediction_key="oracle_distance_km")
    routed_summary = _distance_summary(records)

    per_type_100km = []
    for lightning_type in range(len(TYPE_NAMES)):
        typed = [record for record in records if int(record["true_type"]) == lightning_type]
        per_type_100km.append(
            _distance_summary(typed, exact_only=True)["within_200"]
        )

    subgroup_records = defaultdict(list)
    for record in records:
        low, high = record.get("distance_low_km"), record.get("distance_high_km")
        if low is None or high is None or not 0 <= float(low) < float(high) <= 3000:
            continue
        band_low, band_high = _coarse_band(low, high)
        type_name = TYPE_NAMES[int(record["true_type"])]
        light = "day" if bool(record.get("daylight", False)) else "night"
        subgroup_records[f"{type_name}/{light}/{band_low}-{band_high}km"].append(record)
    distance_subgroups = {}
    for name, selected in sorted(subgroup_records.items()):
        interval = _distance_summary(selected)
        interval_100km = _distance_summary(selected, exact_only=True)
        distance_subgroups[name] = {
            "count": interval["count"],
            "interval_mae_km": interval["mae_km"],
            "interval_100km_count": interval_100km["count"],
            "within_200": (
                interval_100km["within_200"]
                if interval_100km["count"]
                else None
            ),
        }

    interval_100km_file_scores = []
    for positions in by_file.values():
        selected = [records[index] for index in positions]
        summary = _distance_summary(selected, exact_only=True)
        if summary["count"]:
            interval_100km_file_scores.append(summary["within_200"])

    split_hashes = {
        str(record["split_hash"]) for record in records if record.get("split_hash")
    }
    if len(split_hashes) > 1:
        raise ValueError("prediction records contain multiple split hashes")
    return {
        "split_hash": next(iter(split_hashes), ""),
        "piece_count": len(records),
        "file_count": len(by_file),
        "type_piece_accuracy": float(raw_correct.mean()),
        "type_file_macro_accuracy": float(np.mean(file_type_accuracy)),
        "type_coverage": float(accepted.mean()),
        "type_precision": precision,
        "type_recall": recall,
        "type_macro_recall": float(np.mean(recall)),
        "type_file_equal_precision": file_equal_precision,
        "type_file_equal_recall": file_equal_recall,
        "type_file_equal_recall_mean": float(np.mean(file_equal_recall)),
        "raw_type_precision": raw_precision,
        "raw_type_recall": raw_recall,
        "raw_type_macro_recall": float(np.mean(raw_recall)),
        "raw_type_file_equal_precision": raw_file_equal_precision,
        "raw_type_file_equal_recall": raw_file_equal_recall,
        "raw_type_file_equal_recall_mean": float(np.mean(raw_file_equal_recall)),
        "distance_interval_count": interval_summary["count"],
        "distance_coverage": interval_summary["coverage"],
        "distance_interval_mae_km": interval_summary["mae_km"],
        "distance_oracle_mae_km": oracle_summary["mae_km"],
        "distance_routed_mae_km": routed_summary["mae_km"],
        "distance_100km_interval_count": interval_100km_summary["count"],
        "distance_100km_interval_within_100": interval_100km_summary["within_100"],
        "distance_100km_interval_within_200": interval_100km_summary["within_200"],
        "distance_file_macro_within_200": (
            float(np.mean(interval_100km_file_scores))
            if interval_100km_file_scores
            else 0.0
        ),
        "distance_per_type_100km_interval_within_200": per_type_100km,
        "distance_subgroups": distance_subgroups,
        "distance_conditions_100km": distance_condition_metrics(records),
    }


def file_bootstrap_metrics(records, iterations=1000, seed=0):
    """Return confidence intervals from resampling acquisition files."""
    records = [dict(record) for record in records]
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    by_file = defaultdict(list)
    for record in records:
        by_file[record_file_identity(record)].append(record)
    file_ids = sorted(by_file)
    if not file_ids:
        raise ValueError("at least one source file is required")
    rng = np.random.default_rng(seed)
    type_values, distance_values = [], []
    for _ in range(iterations):
        sampled = rng.choice(file_ids, size=len(file_ids), replace=True)
        bootstrap_records = []
        for draw, file_id in enumerate(sampled):
            for original in by_file[str(file_id)]:
                copied = dict(original)
                copied["file_id"] = f"{draw}:{file_id}"
                if "source_path" in copied:
                    copied["source_path"] = f"{draw}:{file_id}"
                bootstrap_records.append(copied)
        metrics = evaluate_predictions(bootstrap_records)
        type_values.append(metrics["type_file_macro_accuracy"])
        distance_values.append(metrics["distance_100km_interval_within_200"])
    return {
        "type_file_macro_accuracy_ci95": [
            float(value) for value in np.percentile(type_values, [2.5, 97.5])
        ],
        "distance_100km_interval_within_200_ci95": [
            float(value) for value in np.percentile(distance_values, [2.5, 97.5])
        ],
    }


def absolute_gate_failures(candidate):
    """Return failures of the fixed absolute release contract."""
    reasons = []
    precision = candidate.get("type_file_equal_precision", [])
    if len(precision) != 4:
        reasons.append("four file-equal per-type precision values are required")
    for index, value in enumerate(precision):
        if float(value) < RELEASE_GATES["min_per_type_precision"]:
            reasons.append(
                f"type_file_equal_precision[{index}]={float(value):.4f} below 0.95"
            )
    scalar_gates = (
        ("type_coverage", "min_coverage"),
        ("type_file_equal_recall_mean", "min_file_equal_macro_recall"),
        ("distance_100km_interval_within_200", "min_100km_interval_within_200"),
    )
    for metric, gate in scalar_gates:
        value = float(candidate.get(metric, 0.0))
        if value < RELEASE_GATES[gate]:
            reasons.append(
                f"{metric}={value:.4f} below {RELEASE_GATES[gate]:.2f}"
            )
    per_type = candidate.get("distance_per_type_100km_interval_within_200", [])
    if len(per_type) != 4:
        reasons.append("four per-type 100-km interval values are required")
    for index, value in enumerate(per_type):
        if float(value) < RELEASE_GATES[
            "min_per_type_100km_interval_within_200"
        ]:
            reasons.append(
                f"distance_within_200[{index}]={float(value):.4f} below 0.75"
            )
    return reasons


def evaluate_release(candidate):
    """Apply absolute gates, hard-gating only sufficiently supported conditions."""
    reasons = absolute_gate_failures(candidate)
    for name, group in candidate["distance_conditions_100km"].items():
        if (
            group["piece_count"] >= RELEASE_GATES["minimum_supported_pieces"]
            and group["file_macro_within_200"]
            < RELEASE_GATES["min_supported_condition_within_200"]
        ):
            reasons.append(
                f"supported subgroup {name}="
                f"{group['file_macro_within_200']:.4f} below 0.70"
            )
    return not reasons, reasons


def checkpoint_selection_key(metrics):
    """Order fold checkpoints after a non-release type-readiness floor."""
    supported = [
        group["file_macro_within_200"]
        for group in metrics["distance_conditions_100km"].values()
        if group["piece_count"] >= MINIMUM_SUPPORTED_PIECES
    ]
    readiness = (
        metrics["type_file_equal_recall_mean"] >= 0.85
        and min(metrics["type_file_equal_recall"]) >= 0.75
    )
    return (
        int(readiness),
        metrics["type_file_equal_recall_mean"],
        min(metrics["type_file_equal_recall"]),
        min(supported, default=0.0),
        metrics["distance_file_macro_within_200"],
        min(metrics["distance_per_type_100km_interval_within_200"]),
        -metrics["distance_interval_mae_km"],
    )
