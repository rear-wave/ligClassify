"""Deployment-routed piece metrics for the five-class classifier."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from torch import nn

from models import DISTANCE_BIN_COUNT, ModelOutput, TYPE_COUNT


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _type_metrics(confusion: np.ndarray) -> dict[str, object]:
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for type_index in range(TYPE_COUNT):
        true_positive = int(confusion[type_index, type_index])
        type_precision = _ratio(true_positive, int(confusion[:, type_index].sum()))
        type_recall = _ratio(true_positive, int(confusion[type_index, :].sum()))
        precision.append(type_precision)
        recall.append(type_recall)
        f1.append(
            _ratio(
                2.0 * type_precision * type_recall,
                type_precision + type_recall,
            )
        )
    total = int(confusion.sum())
    return {
        "type_confusion": confusion.tolist(),
        "type_accuracy": _ratio(int(np.trace(confusion)), total),
        "type_precision": precision,
        "type_recall": recall,
        "type_f1": f1,
        "type_macro_precision": float(np.mean(precision)),
        "type_macro_recall": float(np.mean(recall)),
        "type_macro_f1": float(np.mean(f1)),
    }


def _validate_output(output: ModelOutput, batch_size: int) -> None:
    if tuple(output.type_logits.shape) != (batch_size, TYPE_COUNT):
        raise ValueError("type_logits must have shape [batch, 5]")
    if len(output.distance_logits) != 4 or any(
        tuple(head.shape) != (batch_size, DISTANCE_BIN_COUNT)
        for head in output.distance_logits
    ):
        raise ValueError("four distance experts with shape [batch, 30] are required")


def _evaluation_tensors(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key in ("local", "global", "daylight", "type_label", "distance_bin"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {key!r} must be a tensor")
        tensors[key] = value.to(device)
    return tensors


def evaluate_type_role(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device | str,
) -> dict[str, object]:
    """Evaluate the independent five-class role."""
    target_device = torch.device(device)
    confusion = np.zeros((TYPE_COUNT, TYPE_COUNT), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _evaluation_tensors(batch, target_device)
            targets = tensors["type_label"].long()
            if targets.ndim != 1 or torch.any(
                (targets < 0) | (targets >= TYPE_COUNT)
            ):
                raise ValueError("type_label must be one-dimensional in 0..4")
            logits, _ = model.forward_type(
                tensors["local"], tensors["global"], tensors["daylight"]
            )
            if tuple(logits.shape) != (len(targets), TYPE_COUNT):
                raise ValueError("type logits must have shape [batch, 5]")
            predictions = logits.argmax(dim=1)
            for truth, prediction in zip(
                targets.cpu().tolist(), predictions.cpu().tolist()
            ):
                confusion[int(truth), int(prediction)] += 1
    if not int(confusion.sum()):
        raise ValueError("evaluation loader produced no samples")
    metrics = _type_metrics(confusion)
    metrics["piece_count"] = int(confusion.sum())
    return metrics


def evaluate_distance_role(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device | str,
    *,
    type_index: int,
) -> dict[str, object]:
    """Evaluate one independently trained non-IC distance role."""
    if type(type_index) is not int or not 1 <= type_index < TYPE_COUNT:
        raise ValueError("type_index must be in 1..4")
    target_device = torch.device(device)
    errors: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _evaluation_tensors(batch, target_device)
            type_targets = tensors["type_label"].long()
            targets = tensors["distance_bin"].long()
            if torch.any(type_targets != type_index):
                raise ValueError(
                    "distance evaluation contains a different type"
                )
            if targets.ndim != 1 or torch.any(
                (targets < 0) | (targets >= DISTANCE_BIN_COUNT)
            ):
                raise ValueError(
                    "distance targets must be one-dimensional in 0..29"
                )
            logits, _ = model.forward_distance_type(
                tensors["local"],
                tensors["global"],
                tensors["daylight"],
                expert_index=type_index - 1,
            )
            if tuple(logits.shape) != (len(targets), DISTANCE_BIN_COUNT):
                raise ValueError(
                    "distance logits must have shape [batch, 30]"
                )
            predictions = logits.argmax(dim=1)
            errors.extend(
                torch.abs(predictions - targets).cpu().tolist()
            )
    if not errors:
        raise ValueError("evaluation loader produced no samples")
    values = np.asarray(errors, dtype=np.float64)
    return {
        "piece_count": len(errors),
        "exact_accuracy": float(np.mean(values == 0)),
        "within_100": float(np.mean(values <= 1)),
        "within_200": float(np.mean(values <= 2)),
        "mae_bins": float(values.mean()),
        "mae_km": float(values.mean() * 100.0),
    }


def evaluate_loader(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device | str,
    *,
    include_distance: bool = True,
) -> dict[str, object]:
    """Evaluate type and optionally predicted-type-routed distance."""
    target_device = torch.device(device)
    confusion = np.zeros((TYPE_COUNT, TYPE_COUNT), dtype=np.int64)
    true_non_ic: list[int] = []
    covered: list[bool] = []
    errors_km: list[float] = []
    exact: list[bool] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            tensors = _evaluation_tensors(batch, target_device)
            type_labels = tensors["type_label"].to(dtype=torch.long)
            distance_bins = tensors["distance_bin"].to(dtype=torch.long)
            if type_labels.ndim != 1 or distance_bins.ndim != 1:
                raise ValueError("evaluation labels must be one-dimensional")
            if len(type_labels) != len(distance_bins):
                raise ValueError("evaluation label batch sizes differ")
            if torch.any((type_labels < 0) | (type_labels >= TYPE_COUNT)):
                raise ValueError("type_label must be in the five-class range 0..4")
            ic_mask = type_labels == 0
            if include_distance:
                if torch.any(distance_bins[ic_mask] != -1):
                    raise ValueError("IC labels must use distance_bin -1")
                if torch.any(
                    (distance_bins[~ic_mask] < 0)
                    | (distance_bins[~ic_mask] >= DISTANCE_BIN_COUNT)
                ):
                    raise ValueError(
                        "non-IC labels must use a distance bin in 0..29"
                    )

            output = model(
                tensors["local"], tensors["global"], tensors["daylight"]
            )
            _validate_output(output, len(type_labels))
            predicted_types = output.type_logits.argmax(dim=1)
            for truth, prediction in zip(
                type_labels.detach().cpu().tolist(),
                predicted_types.detach().cpu().tolist(),
            ):
                confusion[int(truth), int(prediction)] += 1

            if not include_distance:
                continue

            predicted_bins = torch.full_like(distance_bins, -1)
            for predicted_type in range(1, TYPE_COUNT):
                selected = (~ic_mask) & (predicted_types == predicted_type)
                predicted_bins[selected] = output.distance_logits[
                    predicted_type - 1
                ][selected].argmax(dim=1)
            cpu_types = type_labels.detach().cpu().tolist()
            cpu_predictions = predicted_types.detach().cpu().tolist()
            cpu_true_bins = distance_bins.detach().cpu().tolist()
            cpu_predicted_bins = predicted_bins.detach().cpu().tolist()
            for row in torch.nonzero(~ic_mask, as_tuple=False).flatten().tolist():
                true_type = int(cpu_types[row])
                predicted_type = int(cpu_predictions[row])
                true_bin = int(cpu_true_bins[row])
                true_non_ic.append(true_type)
                is_covered = predicted_type != 0
                covered.append(is_covered)
                if not is_covered:
                    errors_km.append(float("inf"))
                    exact.append(False)
                    continue
                predicted_bin = int(cpu_predicted_bins[row])
                errors_km.append(float(abs(predicted_bin - true_bin) * 100))
                exact.append(predicted_bin == true_bin)

    if not int(confusion.sum()):
        raise ValueError("evaluation loader produced no samples")
    metrics = _type_metrics(confusion)
    metrics["piece_count"] = int(confusion.sum())
    if not include_distance:
        return metrics

    present_non_ic = set(true_non_ic)
    if present_non_ic != {1, 2, 3, 4}:
        raise ValueError("validation must contain all four non-IC types")

    errors = np.asarray(errors_km, dtype=np.float64)
    coverage = np.asarray(covered, dtype=bool)
    exact_array = np.asarray(exact, dtype=bool)
    per_type_within_200 = [
        float(np.mean(errors[np.asarray(true_non_ic) == type_index] <= 200.0))
        for type_index in range(1, TYPE_COUNT)
    ]
    metrics.update(
        {
            "distance_count": len(true_non_ic),
            "distance_prediction_count": int(coverage.sum()),
            "distance_coverage": float(coverage.mean()),
            "distance_exact_accuracy": float(exact_array.mean()),
            "distance_mae_km": (
                float(errors[coverage].mean()) if coverage.any() else 0.0
            ),
            "distance_within_100": float(np.mean(errors <= 100.0)),
            "distance_within_200": float(np.mean(errors <= 200.0)),
            "per_type_within_200": per_type_within_200,
            "mean_non_ic_within_200": float(np.mean(per_type_within_200)),
        }
    )
    return metrics


def selection_score(metrics: Mapping[str, object]) -> float:
    """Return the exact 50/50 validation checkpoint selection score."""
    return 0.5 * float(metrics["type_macro_f1"]) + 0.5 * float(
        metrics["mean_non_ic_within_200"]
    )
