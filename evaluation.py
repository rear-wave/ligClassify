"""Deployment-routed piece metrics for the five-class classifier."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models import (
    DISTANCE_BIN_COUNT,
    HIERARCHICAL_TYPE_ARCHITECTURE,
    HierarchicalTypeOutput,
    ModelOutput,
    TYPE_COUNT,
)


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


@dataclass(frozen=True)
class HierarchicalDecisionConfig:
    """Calibrated acceptance limits for stable known-class evidence."""

    known_probability_thresholds: tuple[float, float, float, float]
    prototype_similarity_thresholds: tuple[float, float, float, float]
    max_js_divergence: float
    min_branch_votes: int
    max_ic_gate_probabilities: tuple[float, float, float, float] = (
        0.50,
        0.50,
        0.50,
        0.50,
    )


def conservative_hierarchical_config() -> HierarchicalDecisionConfig:
    """Return a reject-by-default configuration for uncalibrated models."""
    values: dict[str, Any] = {}
    for field in fields(HierarchicalDecisionConfig):
        name = field.name
        if "gate" in name:
            values[name] = (0.0, 0.0, 0.0, 0.0)
        elif "probability" in name or "prototype" in name:
            values[name] = (1.0, 1.0, 1.0, 1.0)
        elif "divergence" in name or "js" in name:
            values[name] = 0.0
        elif "vote" in name:
            values[name] = 4
        else:
            raise ValueError(
                f"unsupported hierarchical decision field: {name}"
            )
    return HierarchicalDecisionConfig(**values)


@dataclass(frozen=True)
class HierarchicalDecision:
    """Batch decisions and diagnostics shared by evaluation and inference."""

    final_type: torch.Tensor
    candidate_known_type: torch.Tensor
    ic_gate_probability: torch.Tensor
    known_type_probability: torch.Tensor
    prototype_similarity: torch.Tensor
    local_prediction: torch.Tensor
    global_prediction: torch.Tensor
    consistency_score: torch.Tensor
    decision_reason: tuple[str, ...]


def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_log = F.log_softmax(first, dim=1)
    second_log = F.log_softmax(second, dim=1)
    midpoint = torch.logaddexp(first_log, second_log) - np.log(2.0)
    return 0.5 * (
        (first_log.exp() * (first_log - midpoint)).sum(dim=1)
        + (second_log.exp() * (second_log - midpoint)).sum(dim=1)
    )


def decide_hierarchical_types(
    primary: HierarchicalTypeOutput,
    alternate: HierarchicalTypeOutput,
    config: HierarchicalDecisionConfig,
) -> HierarchicalDecision:
    """Accept stable known evidence; use IC only when no match is stable."""
    first_probability = primary.known_logits.softmax(dim=1)
    second_probability = alternate.known_logits.softmax(dim=1)
    mean_probability = 0.5 * (first_probability + second_probability)
    candidate_zero = mean_probability.argmax(dim=1)
    rows = torch.arange(len(candidate_zero), device=candidate_zero.device)
    candidate = candidate_zero + 1
    candidate_probability = mean_probability[rows, candidate_zero]
    prototype = 0.5 * (
        primary.prototype_scores[rows, candidate_zero]
        + alternate.prototype_scores[rows, candidate_zero]
    )
    ic_gate_probability = 0.5 * (
        primary.gate_logits.softmax(dim=1)[:, 0]
        + alternate.gate_logits.softmax(dim=1)[:, 0]
    )
    first_candidate = first_probability.argmax(dim=1)
    second_candidate = second_probability.argmax(dim=1)
    local = primary.local_known_logits.argmax(dim=1) + 1
    global_prediction = primary.global_known_logits.argmax(dim=1) + 1
    votes = torch.stack(
        (
            local,
            global_prediction,
            alternate.local_known_logits.argmax(dim=1) + 1,
            alternate.global_known_logits.argmax(dim=1) + 1,
        )
    ).eq(candidate).sum(dim=0)
    divergence = _js_divergence(
        primary.known_logits, alternate.known_logits
    )
    probability_thresholds = primary.known_logits.new_tensor(
        config.known_probability_thresholds
    )[candidate_zero]
    prototype_thresholds = primary.known_logits.new_tensor(
        config.prototype_similarity_thresholds
    )[candidate_zero]
    gate_thresholds = primary.known_logits.new_tensor(
        config.max_ic_gate_probabilities
    )[candidate_zero]
    stable = (
        first_candidate.eq(second_candidate)
        & candidate_probability.ge(probability_thresholds)
        & prototype.ge(prototype_thresholds)
        & ic_gate_probability.le(gate_thresholds)
        & divergence.le(float(config.max_js_divergence))
        & votes.ge(int(config.min_branch_votes))
    )
    final_type = torch.where(stable, candidate, torch.zeros_like(candidate))
    reasons = tuple(
        "stable_known_override"
        if bool(value)
        else "insufficient_known_evidence"
        for value in stable.detach().cpu().tolist()
    )
    return HierarchicalDecision(
        final_type=final_type,
        candidate_known_type=candidate,
        ic_gate_probability=ic_gate_probability,
        known_type_probability=candidate_probability,
        prototype_similarity=prototype,
        local_prediction=local,
        global_prediction=global_prediction,
        consistency_score=1.0 - divergence,
        decision_reason=reasons,
    )


def _quantiles(values: torch.Tensor) -> list[float]:
    if not len(values):
        return []
    levels = torch.linspace(0.0, 1.0, 11, device=values.device)
    return sorted(set(float(value) for value in torch.quantile(values, levels)))


def _shift_tensor_view(values: torch.Tensor, amount: int) -> torch.Tensor:
    if amount <= 0 or amount >= values.shape[-1]:
        raise ValueError("shift amount must be inside the waveform length")
    shifted = torch.zeros_like(values)
    shifted[..., amount:] = values[..., :-amount]
    return shifted


def infer_hierarchical_types(
    model: Any,
    decision_config: dict[str, Any],
    local: torch.Tensor,
    global_view: torch.Tensor,
    daylight: torch.Tensor,
    alternate_daylight: torch.Tensor | None,
    missing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired-view evidence and conservative unknown-time routing."""
    config = HierarchicalDecisionConfig(**dict(decision_config))
    shifted_local = _shift_tensor_view(local, 16)
    shifted_global = _shift_tensor_view(global_view, 4)
    primary = model(local, global_view, daylight)
    alternate = model(shifted_local, shifted_global, daylight)
    decision = decide_hierarchical_types(primary, alternate, config)
    logits = 0.5 * (primary.type_logits + alternate.type_logits)
    final_type = decision.final_type
    if alternate_daylight is None:
        return logits, final_type

    second_primary = model(local, global_view, alternate_daylight)
    second_alternate = model(
        shifted_local, shifted_global, alternate_daylight
    )
    second_decision = decide_hierarchical_types(
        second_primary, second_alternate, config
    )
    second_logits = 0.5 * (
        second_primary.type_logits + second_alternate.type_logits
    )
    mask = missing.unsqueeze(1)
    logits = torch.where(mask, 0.5 * (logits + second_logits), logits)
    agreed = final_type.eq(second_decision.final_type)
    unknown_final = torch.where(
        agreed, final_type, torch.zeros_like(final_type)
    )
    return logits, torch.where(missing, unknown_final, final_type)


def calibrate_hierarchical_decision(
    primary: HierarchicalTypeOutput,
    alternate: HierarchicalTypeOutput,
    targets: torch.Tensor,
    *,
    minimum_precision: float = 0.90,
    target_ic_fraction: float = 0.80,
    maximum_ic_false_positive_rate: float = 0.01,
) -> HierarchicalDecisionConfig:
    """Select deterministic validation thresholds with known-recall priority."""
    if targets.ndim != 1 or len(targets) != len(primary.known_logits):
        raise ValueError("calibration targets must align with outputs")
    if set(targets.detach().cpu().tolist()) & {1, 2, 3, 4} != {1, 2, 3, 4}:
        raise ValueError("calibration requires all four known classes")
    if not 0.0 < target_ic_fraction < 1.0:
        raise ValueError("target_ic_fraction must be between zero and one")
    if not 0.0 <= maximum_ic_false_positive_rate < 1.0:
        raise ValueError(
            "maximum_ic_false_positive_rate must be in [0, 1)"
        )
    class_counts = torch.bincount(targets, minlength=TYPE_COUNT).float()
    if torch.any(class_counts == 0):
        raise ValueError("calibration requires all five classes")
    target_priors = targets.new_tensor(
        [
            target_ic_fraction,
            *([(1.0 - target_ic_fraction) / 4.0] * 4),
        ],
        dtype=torch.float32,
    )
    sample_weights = target_priors[targets] / class_counts[targets]
    first = primary.known_logits.softmax(dim=1)
    second = alternate.known_logits.softmax(dim=1)
    mean = 0.5 * (first + second)
    candidate_zero = mean.argmax(dim=1)
    rows = torch.arange(len(targets), device=targets.device)
    probability = mean[rows, candidate_zero]
    prototype = 0.5 * (
        primary.prototype_scores[rows, candidate_zero]
        + alternate.prototype_scores[rows, candidate_zero]
    )
    gate_probability = 0.5 * (
        primary.gate_logits.softmax(dim=1)[:, 0]
        + alternate.gate_logits.softmax(dim=1)[:, 0]
    )
    divergence = _js_divergence(primary.known_logits, alternate.known_logits)
    agreed = first.argmax(dim=1).eq(second.argmax(dim=1))
    candidate_type = candidate_zero + 1
    votes = torch.stack(
        (
            primary.local_known_logits.argmax(dim=1) + 1,
            primary.global_known_logits.argmax(dim=1) + 1,
            alternate.local_known_logits.argmax(dim=1) + 1,
            alternate.global_known_logits.argmax(dim=1) + 1,
        )
    ).eq(candidate_type).sum(dim=0)
    true_known = targets > 0
    if not torch.any(agreed & votes.ge(2) & true_known):
        raise ValueError("calibration has no stable known candidates")
    max_js = float(
        torch.quantile(divergence[agreed & votes.ge(2) & true_known], 0.95)
    )
    probability_limits: list[float] = []
    prototype_limits: list[float] = []
    gate_limits: list[float] = []
    for type_index in range(1, TYPE_COUNT):
        candidate_rows = (
            agreed & votes.ge(2) & candidate_zero.eq(type_index - 1)
        )
        truth_rows = targets.eq(type_index)
        best: tuple[float, ...] | None = None
        for probability_limit in _quantiles(probability[candidate_rows]):
            for prototype_limit in _quantiles(prototype[candidate_rows]):
                for gate_limit in _quantiles(
                    gate_probability[candidate_rows]
                ):
                    accepted = (
                        candidate_rows
                        & probability.ge(probability_limit)
                        & prototype.ge(prototype_limit)
                        & gate_probability.le(gate_limit)
                        & divergence.le(max_js)
                    )
                    accepted_weight = float(sample_weights[accepted].sum())
                    if accepted_weight <= 0.0:
                        continue
                    correct = accepted & truth_rows
                    weighted_precision = float(
                        sample_weights[correct].sum()
                    ) / accepted_weight
                    recall = int(correct.sum()) / int(truth_rows.sum())
                    ic_false_positive_rate = float(
                        (accepted & targets.eq(0)).float().sum()
                        / targets.eq(0).float().sum()
                    )
                    if weighted_precision + 1e-12 < minimum_precision:
                        continue
                    if (
                        ic_false_positive_rate
                        > maximum_ic_false_positive_rate + 1e-12
                    ):
                        continue
                    candidate_score = (
                        recall,
                        weighted_precision,
                        -ic_false_positive_rate,
                        -probability_limit,
                        -prototype_limit,
                        -gate_limit,
                    )
                    if best is None or candidate_score > best:
                        best = candidate_score
        if best is None:
            probability_limits.append(1.0)
            prototype_limits.append(1.0)
            gate_limits.append(0.0)
            continue
        probability_limits.append(-best[3])
        prototype_limits.append(-best[4])
        gate_limits.append(-best[5])
    return HierarchicalDecisionConfig(
        known_probability_thresholds=tuple(probability_limits),
        prototype_similarity_thresholds=tuple(prototype_limits),
        max_js_divergence=max(max_js, 1e-8),
        min_branch_votes=2,
        max_ic_gate_probabilities=tuple(gate_limits),
    )


def decision_config_dict(
    config: HierarchicalDecisionConfig,
) -> dict[str, object]:
    """Return a JSON-safe calibrated decision configuration."""
    payload = asdict(config)
    payload["known_probability_thresholds"] = list(
        config.known_probability_thresholds
    )
    payload["prototype_similarity_thresholds"] = list(
        config.prototype_similarity_thresholds
    )
    return payload


def _shift_view(values: torch.Tensor, amount: int) -> torch.Tensor:
    shifted = torch.zeros_like(values)
    shifted[..., amount:] = values[..., :-amount]
    return shifted


def _hierarchical_outputs(
    model: nn.Module, tensors: Mapping[str, torch.Tensor]
) -> tuple[HierarchicalTypeOutput, HierarchicalTypeOutput]:
    primary = model.forward_hierarchical(
        tensors["local"], tensors["global"], tensors["daylight"]
    )
    alternate = model.forward_hierarchical(
        _shift_view(tensors["local"], 16),
        _shift_view(tensors["global"], 4),
        tensors["daylight"],
    )
    return primary, alternate


def evaluate_hierarchical_type_role(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device | str,
    *,
    decision_config: HierarchicalDecisionConfig | None = None,
) -> tuple[dict[str, object], HierarchicalDecisionConfig]:
    """Evaluate stable known decisions and return the used calibration."""
    target_device = torch.device(device)
    primary_batches: list[HierarchicalTypeOutput] = []
    alternate_batches: list[HierarchicalTypeOutput] = []
    targets: list[torch.Tensor] = []
    source_paths: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _evaluation_tensors(batch, target_device)
            primary, alternate = _hierarchical_outputs(model, tensors)
            primary_batches.append(primary)
            alternate_batches.append(alternate)
            targets.append(tensors["type_label"].long())
            paths = batch.get("source_path")
            source_paths.extend(
                [str(path) for path in paths]
                if isinstance(paths, list)
                else [f"piece-{len(source_paths) + row}" for row in range(len(targets[-1]))]
            )
    if not targets:
        raise ValueError("evaluation loader produced no samples")

    def combine(items: list[HierarchicalTypeOutput]) -> HierarchicalTypeOutput:
        fields = HierarchicalTypeOutput.__dataclass_fields__
        return HierarchicalTypeOutput(
            **{
                name: torch.cat([getattr(item, name) for item in items])
                for name in fields
            }
        )

    primary = combine(primary_batches)
    alternate = combine(alternate_batches)
    labels = torch.cat(targets)
    config = decision_config or calibrate_hierarchical_decision(
        primary, alternate, labels
    )
    decision = decide_hierarchical_types(primary, alternate, config)
    confusion = np.zeros((TYPE_COUNT, TYPE_COUNT), dtype=np.int64)
    for truth, prediction in zip(
        labels.cpu().tolist(), decision.final_type.cpu().tolist()
    ):
        confusion[int(truth), int(prediction)] += 1
    metrics = _type_metrics(confusion)
    false_to_ic = [
        _ratio(confusion[index, 0], confusion[index].sum())
        for index in range(1, TYPE_COUNT)
    ]
    known_recall = list(metrics["type_recall"])[1:]
    known = labels > 0
    consistency = primary.known_logits.argmax(dim=1).eq(
        alternate.known_logits.argmax(dim=1)
    )
    source_predictions: dict[str, list[int]] = {}
    for path, prediction in zip(
        source_paths, decision.final_type.cpu().tolist()
    ):
        source_predictions.setdefault(path, []).append(int(prediction))
    metrics.update(
        {
            "piece_count": int(len(labels)),
            "source_count": len(source_predictions),
            "known_macro_recall": float(np.mean(known_recall)),
            "known_recall": known_recall,
            "known_false_to_ic": false_to_ic,
            "max_known_false_to_ic": max(false_to_ic),
            "known_consistency": float(consistency[known].float().mean()),
            "nbe_confusion_rate": _ratio(
                confusion[2, 4] + confusion[4, 2],
                confusion[2].sum() + confusion[4].sum(),
            ),
            "cg_confusion_rate": _ratio(
                confusion[1, 3] + confusion[3, 1],
                confusion[1].sum() + confusion[3].sum(),
            ),
            "decision_config": decision_config_dict(config),
        }
    )
    return metrics, config


def hierarchical_selection_score(metrics: Mapping[str, object]) -> float:
    """Prioritize known recall while penalizing false IC rejection."""
    return (
        float(metrics["known_macro_recall"])
        - 0.50 * float(metrics["max_known_false_to_ic"])
        + 0.10 * float(metrics["type_macro_f1"])
    )


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
    if (
        getattr(model, "architecture", None)
        == HIERARCHICAL_TYPE_ARCHITECTURE
    ):
        metrics, _ = evaluate_hierarchical_type_role(model, loader, device)
        return metrics
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
    expected_errors: list[float] = []
    expected_biases: list[float] = []
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
            probabilities = logits.softmax(dim=1)
            bin_indices = torch.arange(
                DISTANCE_BIN_COUNT,
                device=logits.device,
                dtype=logits.dtype,
            )
            expected_bins = probabilities @ bin_indices
            errors.extend(
                torch.abs(predictions - targets).cpu().tolist()
            )
            biases = expected_bins - targets.to(logits.dtype)
            expected_biases.extend(biases.cpu().tolist())
            expected_errors.extend(torch.abs(biases).cpu().tolist())
    if not errors:
        raise ValueError("evaluation loader produced no samples")
    values = np.asarray(errors, dtype=np.float64)
    expected_values = np.asarray(expected_errors, dtype=np.float64)
    expected_bias = np.asarray(expected_biases, dtype=np.float64)
    return {
        "piece_count": len(errors),
        "exact_accuracy": float(np.mean(values == 0)),
        "within_100": float(np.mean(values <= 1)),
        "within_200": float(np.mean(values <= 2)),
        "mae_bins": float(values.mean()),
        "mae_km": float(values.mean() * 100.0),
        "expected_within_100": float(np.mean(expected_values <= 1.0 + 1e-6)),
        "expected_within_200": float(np.mean(expected_values <= 2.0 + 1e-6)),
        "expected_mae_bins": float(expected_values.mean()),
        "expected_mae_km": float(expected_values.mean() * 100.0),
        "expected_rmse_km": float(
            np.sqrt(np.mean(np.square(expected_values))) * 100.0
        ),
        "expected_bias_km": float(expected_bias.mean() * 100.0),
    }


def evaluate_model_bundle(bundle: object, loader: Iterable[Mapping[str, object]],
                          device: torch.device | str) -> dict[str, object]:
    """Evaluate the five checkpoints with deployment-equivalent routing."""
    from checkpoints import HIERARCHICAL_FIVE_CLASS_SCHEMA, forward_model_bundle

    target_device = torch.device(device)
    confusion = np.zeros((TYPE_COUNT, TYPE_COUNT), dtype=np.int64)
    true_types: list[int] = []
    covered: list[bool] = []
    errors: list[float] = []
    exact: list[bool] = []
    type_checkpoint = bundle.type_checkpoint
    models = (type_checkpoint.model,) + tuple(
        item.model for item in bundle.distance_checkpoints)
    for model in models:
        model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = _evaluation_tensors(batch, target_device)
            labels = tensors["type_label"].long()
            distance_bins = tensors["distance_bin"].long()
            if type_checkpoint.schema == HIERARCHICAL_FIVE_CLASS_SCHEMA:
                config = type_checkpoint.metadata.get("decision_config")
                if not isinstance(config, dict):
                    raise ValueError("hierarchical bundle has no decision_config")
                type_logits, predictions = infer_hierarchical_types(
                    type_checkpoint.model, config, tensors["local"],
                    tensors["global"], tensors["daylight"], None,
                    torch.zeros(len(labels), dtype=torch.bool, device=target_device),
                )
                heads = [
                    type_logits.new_zeros((len(labels), DISTANCE_BIN_COUNT))
                    for _ in range(TYPE_COUNT - 1)]
                for type_index, checkpoint in enumerate(
                    bundle.distance_checkpoints, start=1):
                    selected = predictions == type_index
                    if torch.any(selected):
                        heads[type_index - 1][selected] = checkpoint.model.forward_distance_type(
                                tensors["local"][selected],
                                tensors["global"][selected],
                                tensors["daylight"][selected],
                                expert_index=type_index - 1,
                            )[0]
            else:
                type_logits, raw_heads = forward_model_bundle(
                    bundle, tensors["local"], tensors["global"],
                    tensors["daylight"])
                predictions = type_logits.argmax(dim=1)
                heads = list(raw_heads)
            pairs = zip(labels.cpu().tolist(), predictions.cpu().tolist())
            for row, (truth, prediction) in enumerate(pairs):
                confusion[int(truth), int(prediction)] += 1
                if truth == 0:
                    continue
                true_types.append(int(truth))
                is_covered = prediction != 0
                covered.append(is_covered)
                if not is_covered:
                    errors.append(float("inf"))
                    exact.append(False)
                    continue
                probabilities = heads[prediction - 1][row].softmax(dim=0)
                indices = torch.arange(DISTANCE_BIN_COUNT, device=target_device,
                                       dtype=probabilities.dtype)
                expected_bin = float((probabilities @ indices).item())
                true_bin = int(distance_bins[row].item())
                errors.append(abs(expected_bin - true_bin) * 100.0)
                exact.append(int(probabilities.argmax().item()) == true_bin)
    if not int(confusion.sum()):
        raise ValueError("evaluation loader produced no samples")
    values = np.asarray(errors, dtype=np.float64)
    coverage = np.asarray(covered, dtype=bool)
    metrics = _type_metrics(confusion)
    metrics.update(
        {
            "piece_count": int(confusion.sum()),
            "distance_count": len(values),
            "distance_prediction_count": int(coverage.sum()),
            "distance_coverage": float(coverage.mean()),
            "distance_exact_accuracy": float(np.mean(exact)),
            "distance_expected_mae_km": (
                float(values[coverage].mean()) if coverage.any() else 0.0
            ),
            "distance_expected_within_100": float(np.mean(values <= 100.0)),
            "distance_expected_within_200": float(np.mean(values <= 200.0)),
            "known_macro_recall": float(np.mean(metrics["type_recall"][1:])),
            "max_known_false_to_ic": max(
                _ratio(confusion[index, 0], confusion[index].sum())
                for index in range(1, TYPE_COUNT)
            ),
        }
    )
    return metrics


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
