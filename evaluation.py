"""Record-based evaluation and release gates for lightning classifiers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from distance_metrics import interval_distance_errors


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
COARSE_DISTANCE_EDGES_KM = (0, 300, 600, 1200, 1700, 2400, 3000)
RELEASE_GATES = {
    "min_per_type_precision": 0.95,
    "min_coverage": 0.80,
    "min_macro_recall": 0.90,
    "min_exact_within_200": 0.85,
    "min_per_type_exact_within_200": 0.75,
}


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


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

    raw_correct = predicted_types == true_types
    precision, recall = [], []
    for lightning_type in range(len(TYPE_NAMES)):
        predicted = accepted & (predicted_types == lightning_type)
        actual = true_types == lightning_type
        correct = predicted & actual
        precision.append(_safe_ratio(correct.sum(), predicted.sum()))
        recall.append(_safe_ratio(correct.sum(), actual.sum()))

    by_file = defaultdict(list)
    for index, record in enumerate(records):
        by_file[str(record["file_id"])].append(index)
    file_type_accuracy = [
        float(raw_correct[positions].mean()) for positions in by_file.values()
    ]
    file_macro_precision, file_macro_recall = [], []
    for lightning_type in range(len(TYPE_NAMES)):
        precision_values, recall_values = [], []
        for positions in by_file.values():
            positions = np.asarray(positions, dtype=np.int64)
            predicted = accepted[positions] & (
                predicted_types[positions] == lightning_type
            )
            actual = true_types[positions] == lightning_type
            if predicted.any():
                precision_values.append(float(np.mean(
                    true_types[positions][predicted] == lightning_type
                )))
            if actual.any():
                recall_values.append(float(np.mean(
                    accepted[positions][actual]
                    & (predicted_types[positions][actual] == lightning_type)
                )))
        file_macro_precision.append(
            float(np.mean(precision_values)) if precision_values else 0.0
        )
        file_macro_recall.append(
            float(np.mean(recall_values)) if recall_values else 0.0
        )

    interval_summary = _distance_summary(records)
    exact_summary = _distance_summary(records, exact_only=True)
    oracle_summary = _distance_summary(records, prediction_key="oracle_distance_km")
    routed_summary = _distance_summary(records)

    per_type_exact = []
    for lightning_type in range(len(TYPE_NAMES)):
        typed = [record for record in records if int(record["true_type"]) == lightning_type]
        per_type_exact.append(_distance_summary(typed, exact_only=True)["within_200"])

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
        exact = _distance_summary(selected, exact_only=True)
        distance_subgroups[name] = {
            "count": interval["count"],
            "interval_mae_km": interval["mae_km"],
            "exact_count": exact["count"],
            "within_200": exact["within_200"] if exact["count"] else None,
        }

    exact_file_scores = []
    for positions in by_file.values():
        selected = [records[index] for index in positions]
        summary = _distance_summary(selected, exact_only=True)
        if summary["count"]:
            exact_file_scores.append(summary["within_200"])

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
        "type_file_macro_precision": file_macro_precision,
        "type_file_macro_recall": file_macro_recall,
        "type_file_macro_recall_mean": float(np.mean(file_macro_recall)),
        "distance_interval_count": interval_summary["count"],
        "distance_coverage": interval_summary["coverage"],
        "distance_interval_mae_km": interval_summary["mae_km"],
        "distance_oracle_mae_km": oracle_summary["mae_km"],
        "distance_routed_mae_km": routed_summary["mae_km"],
        "distance_exact_count": exact_summary["count"],
        "distance_exact_within_100": exact_summary["within_100"],
        "distance_exact_within_200": exact_summary["within_200"],
        "distance_file_macro_within_200": (
            float(np.mean(exact_file_scores)) if exact_file_scores else 0.0
        ),
        "distance_per_type_exact_within_200": per_type_exact,
        "distance_subgroups": distance_subgroups,
    }


def file_bootstrap_metrics(records, iterations=1000, seed=0):
    """Return confidence intervals from resampling acquisition files."""
    records = [dict(record) for record in records]
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    by_file = defaultdict(list)
    for record in records:
        by_file[str(record["file_id"])].append(record)
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
                bootstrap_records.append(copied)
        metrics = evaluate_predictions(bootstrap_records)
        type_values.append(metrics["type_file_macro_accuracy"])
        distance_values.append(metrics["distance_exact_within_200"])
    return {
        "type_file_macro_accuracy_ci95": [
            float(value) for value in np.percentile(type_values, [2.5, 97.5])
        ],
        "distance_exact_within_200_ci95": [
            float(value) for value in np.percentile(distance_values, [2.5, 97.5])
        ],
    }


def evaluate_release(candidate, baseline=None, subgroup_tolerance=0.02):
    """Apply fixed reliability gates and same-split baseline comparison."""
    if baseline is not None and candidate.get("split_hash") != baseline.get("split_hash"):
        raise ValueError("candidate and baseline split hash must match")
    reasons = []
    for index, value in enumerate(candidate.get("type_precision", [])):
        if float(value) < RELEASE_GATES["min_per_type_precision"]:
            reasons.append(
                f"type_precision[{index}]={float(value):.4f} below "
                f"{RELEASE_GATES['min_per_type_precision']:.2f}"
            )
    if len(candidate.get("type_precision", [])) != 4:
        reasons.append("four per-type precision values are required")
    file_precision = candidate.get("type_file_macro_precision")
    if file_precision is not None:
        for index, value in enumerate(file_precision):
            if float(value) < RELEASE_GATES["min_per_type_precision"]:
                reasons.append(
                    f"type_file_macro_precision[{index}]={float(value):.4f} below "
                    f"{RELEASE_GATES['min_per_type_precision']:.2f}"
                )
        if len(file_precision) != 4:
            reasons.append("four file-macro precision values are required")
    for key, gate_key in (
        ("type_coverage", "min_coverage"),
        (
            "type_file_macro_recall_mean"
            if "type_file_macro_recall_mean" in candidate
            else "type_macro_recall",
            "min_macro_recall",
        ),
        ("distance_exact_within_200", "min_exact_within_200"),
    ):
        value = float(candidate.get(key, 0.0))
        threshold = RELEASE_GATES[gate_key]
        if value < threshold:
            reasons.append(f"{key}={value:.4f} below {threshold:.2f}")
    per_type_distance = candidate.get("distance_per_type_exact_within_200", [])
    for index, value in enumerate(per_type_distance):
        if float(value) < RELEASE_GATES["min_per_type_exact_within_200"]:
            reasons.append(
                f"distance_within_200[{index}]={float(value):.4f} below "
                f"{RELEASE_GATES['min_per_type_exact_within_200']:.2f}"
            )
    if len(per_type_distance) != 4:
        reasons.append("four per-type exact distance values are required")

    if baseline is not None:
        baseline_groups = baseline.get("distance_subgroups", {})
        for name, candidate_group in candidate.get("distance_subgroups", {}).items():
            baseline_group = baseline_groups.get(name)
            if not baseline_group:
                continue
            candidate_value = candidate_group.get("within_200")
            baseline_value = baseline_group.get("within_200")
            if candidate_value is None or baseline_value is None:
                continue
            if float(candidate_value) + subgroup_tolerance < float(baseline_value):
                reasons.append(
                    f"subgroup regression {name}: {float(candidate_value):.4f} "
                    f"below baseline {float(baseline_value):.4f}"
                )
    return not reasons, reasons
