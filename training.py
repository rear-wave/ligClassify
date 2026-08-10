"""Joint loss, epoch training, and early stopping for five-class models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from tqdm import tqdm


from torch import nn
from torch.nn import functional as F

from checkpoints import TRAINING_STATE_SCHEMA
from models import DISTANCE_BIN_COUNT, DISTANCE_NAMES, HIERARCHICAL_TYPE_ARCHITECTURE, HierarchicalTypeOutput


@dataclass(frozen=True)
class DistanceLossWeights:
    """Weights for categorical, ordered-CDF, and expected-bin losses."""

    categorical: float = 1.0
    ordered: float = 0.20
    expected: float = 0.05


def _move_training_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    moved = dict(batch)
    for key in ("local", "global", "daylight", "type_label", "distance_bin"):
        value = moved.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {key!r} must be a tensor")
        moved[key] = value.to(device)
    paired = ("local_alt" in moved, "global_alt" in moved)
    if any(paired) and not all(paired):
        raise ValueError("training batch has incomplete paired views")
    for key in ("local_alt", "global_alt"):
        if key in moved:
            value = moved[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"batch field {key!r} must be a tensor")
            moved[key] = value.to(device)
    return moved


def distance_expert_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: DistanceLossWeights = DistanceLossWeights(),
) -> torch.Tensor:
    """Combine categorical, ordered, and expected-bin distance losses."""
    if logits.ndim != 2 or logits.shape[1] != DISTANCE_BIN_COUNT:
        raise ValueError("distance logits must have shape [batch, 30]")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("distance targets must have shape [batch]")
    if torch.any((targets < 0) | (targets >= DISTANCE_BIN_COUNT)):
        raise ValueError("distance targets must be in 0..29")
    categorical = F.cross_entropy(logits, targets)
    probabilities = logits.softmax(dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)
    target_cdf = (
        torch.arange(DISTANCE_BIN_COUNT, device=logits.device)[None, :]
        >= targets[:, None]
    ).to(logits.dtype)
    ordered = torch.abs(predicted_cdf - target_cdf).mean()
    bin_indices = torch.arange(
        DISTANCE_BIN_COUNT, device=logits.device, dtype=logits.dtype
    )
    expected_bins = probabilities @ bin_indices
    expected = F.smooth_l1_loss(expected_bins, targets.to(logits.dtype))
    return (
        weights.categorical * categorical
        + weights.ordered * ordered
        + weights.expected * expected
    )


@dataclass(frozen=True)
class HierarchicalLossWeights:
    """Weights for gate, known-class, metric, and consistency objectives."""

    gate: float = 1.0
    known: float = 1.0
    five_class: float = 0.25
    prototype: float = 0.25
    branch: float = 0.10
    contrastive: float = 0.10
    consistency: float = 0.10
    gate_consistency: float = 0.10
    prototype_consistency: float = 0.10
    ic_margin: float = 0.25
    prototype_diversity: float = 0.02
    label_smoothing: float = 0.05
    temperature: float = 0.10
    ic_similarity_margin: float = 0.25
    prototype_diversity_margin: float = 0.20


def _paired_average(function: Any, primary: torch.Tensor, alternate: torch.Tensor) -> torch.Tensor:
    return 0.5 * (function(primary) + function(alternate))


def _supervised_contrastive_loss(
    primary: torch.Tensor, alternate: torch.Tensor, labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    known = labels > 0
    if not torch.any(known):
        return primary.sum() * 0.0
    features = torch.cat((primary[known], alternate[known]), dim=0)
    features = F.normalize(features, dim=1, eps=1e-6)
    known_labels = torch.cat((labels[known], labels[known]), dim=0)
    similarity = features @ features.transpose(0, 1) / temperature
    valid = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(
        similarity.masked_fill(~valid, float("-inf")), dim=1
    )
    positive = (known_labels[:, None] == known_labels[None, :]) & valid
    log_probability = similarity - log_denominator[:, None]
    positive_log = log_probability.masked_fill(~positive, 0.0).sum(dim=1)
    return -(positive_log / positive.sum(dim=1).clamp_min(1)).mean()


def _known_consistency(primary: torch.Tensor, alternate: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
    known = labels > 0
    if not torch.any(known):
        return primary.sum() * 0.0
    first = primary[known].log_softmax(dim=1)
    second = alternate[known].log_softmax(dim=1)
    return 0.5 * (
        F.kl_div(first, second.exp(), reduction="batchmean")
        + F.kl_div(second, first.exp(), reduction="batchmean")
    )


def _symmetric_consistency(
    primary: torch.Tensor, alternate: torch.Tensor
) -> torch.Tensor:
    first = primary.log_softmax(dim=1)
    second = alternate.log_softmax(dim=1)
    return 0.5 * (
        F.kl_div(first, second.exp(), reduction="batchmean")
        + F.kl_div(second, first.exp(), reduction="batchmean")
    )


def _prototype_class_logits(output: HierarchicalTypeOutput) -> torch.Tensor:
    return torch.logsumexp(output.prototype_logits, dim=2) - math.log(
        output.prototype_logits.shape[2]
    )


def _ic_prototype_margin(
    output: HierarchicalTypeOutput,
    targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    ic = targets == 0
    if not torch.any(ic):
        return output.prototype_scores.sum() * 0.0
    maximum_similarity = output.prototype_scores[ic].max(dim=1).values
    return F.relu(maximum_similarity - margin).mean()


def _prototype_diversity(
    prototypes: torch.Tensor | None,
    reference: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if prototypes is None or prototypes.shape[1] < 2:
        return reference.sum() * 0.0
    normalized = F.normalize(prototypes, dim=-1, eps=1e-6)
    similarity = normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(
        similarity.shape[1], dtype=torch.bool, device=similarity.device
    )[None, :, :]
    return F.relu(similarity[mask.expand_as(similarity)] - margin).mean()


def hierarchical_type_loss(
    primary: HierarchicalTypeOutput, alternate: HierarchicalTypeOutput,
    targets: torch.Tensor, weights: HierarchicalLossWeights = HierarchicalLossWeights(),
    prototype_vectors: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute two-view hierarchical losses without compacting IC."""
    if targets.ndim != 1 or len(targets) != len(primary.type_logits):
        raise ValueError("type targets must have shape [batch]")
    gate_targets = (targets > 0).long()
    gate = _paired_average(
        lambda logits: F.cross_entropy(
            logits,
            gate_targets,
            label_smoothing=weights.label_smoothing,
        ),
        primary.gate_logits,
        alternate.gate_logits,
    )
    five_class = _paired_average(
        lambda logits: F.cross_entropy(
            logits,
            targets,
            label_smoothing=weights.label_smoothing,
        ),
        primary.type_logits,
        alternate.type_logits,
    )
    known_mask = targets > 0
    zero = primary.known_logits.sum() * 0.0
    primary_prototypes = _prototype_class_logits(primary)
    alternate_prototypes = _prototype_class_logits(alternate)
    if torch.any(known_mask):
        known_targets = targets[known_mask] - 1
        def average_ce(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            first_loss = F.cross_entropy(first[known_mask], known_targets)
            return 0.5 * (
                first_loss + F.cross_entropy(second[known_mask], known_targets)
            )
        known = average_ce(primary.known_logits, alternate.known_logits)
        prototype = average_ce(primary_prototypes, alternate_prototypes)
        local = average_ce(
            primary.local_known_logits, alternate.local_known_logits
        )
        branch = 0.5 * (local + average_ce(
            primary.global_known_logits, alternate.global_known_logits
        ))
    else:
        known = prototype = branch = zero
    contrastive = _supervised_contrastive_loss(
        primary.embedding,
        alternate.embedding,
        targets,
        weights.temperature,
    )
    consistency = _known_consistency(
        primary.known_logits, alternate.known_logits, targets
    )
    gate_consistency = _symmetric_consistency(
        primary.gate_logits, alternate.gate_logits
    )
    prototype_consistency = _known_consistency(
        primary_prototypes, alternate_prototypes, targets
    )
    ic_margin = 0.5 * (
        _ic_prototype_margin(
            primary, targets, weights.ic_similarity_margin
        )
        + _ic_prototype_margin(
            alternate, targets, weights.ic_similarity_margin
        )
    )
    prototype_diversity = _prototype_diversity(
        prototype_vectors,
        primary.prototype_logits,
        weights.prototype_diversity_margin,
    )
    components = {
        "gate": gate,
        "known": known,
        "five_class": five_class,
        "prototype": prototype,
        "branch": branch,
        "contrastive": contrastive,
        "consistency": consistency,
        "gate_consistency": gate_consistency,
        "prototype_consistency": prototype_consistency,
        "ic_margin": ic_margin,
        "prototype_diversity": prototype_diversity,
    }
    total = sum(
        components[name] * float(getattr(weights, name))
        for name in components
    )
    return {"total": total, **components}


def train_role_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    role: str,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
) -> dict[str, float | int | str]:
    """Train exactly one independent type or distance role for one epoch."""
    if role != "type" and role not in DISTANCE_NAMES:
        raise ValueError(f"unknown training role: {role}")
    target_device = torch.device(device)
    amp_enabled = bool(amp and target_device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler(target_device.type, enabled=amp_enabled)
    model.train()
    sample_count = correct = 0
    loss_sum = 0.0
    known_count = known_correct = known_to_ic = 0
    progress = tqdm(
        loader,
        desc=f"{role} Epoch",
        dynamic_ncols=True,
        unit="batch",
        leave=True,
    )
    for raw_batch in progress:
        batch = _move_training_batch(raw_batch, target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=target_device.type,
            enabled=amp_enabled,
        ):
            if role == "type":
                targets = batch["type_label"].long()
                if (
                    getattr(model, "architecture", None)
                    == HIERARCHICAL_TYPE_ARCHITECTURE
                ):
                    if "local_alt" not in batch or "global_alt" not in batch:
                        raise ValueError(
                            "hierarchical type training requires paired views"
                        )
                    primary = model.forward_hierarchical(
                        batch["local"], batch["global"], batch["daylight"]
                    )
                    alternate = model.forward_hierarchical(
                        batch["local_alt"],
                        batch["global_alt"],
                        batch["daylight"],
                    )
                    loss_parts = hierarchical_type_loss(
                        primary,
                        alternate,
                        targets,
                        prototype_vectors=(
                            getattr(
                                getattr(model, "prototype_matcher", None),
                                "prototypes",
                                None,
                            )
                        ),
                    )
                    loss = loss_parts["total"]
                    logits = primary.type_logits
                else:
                    logits, _ = model.forward_type(
                        batch["local"], batch["global"], batch["daylight"]
                    )
                    loss = F.cross_entropy(logits, targets)
            else:
                expert_index = DISTANCE_NAMES.index(role)
                expected_type = expert_index + 1
                type_targets = batch["type_label"].long()
                if torch.any(type_targets != expected_type):
                    raise ValueError(
                        f"{role} distance batches must contain only "
                        f"type index {expected_type}"
                    )
                targets = batch["distance_bin"].long()
                logits, _ = model.forward_distance_type(
                    batch["local"],
                    batch["global"],
                    batch["daylight"],
                    expert_index=expert_index,
                )
                loss = distance_expert_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = len(targets)
        sample_count += count
        loss_sum += float(loss.detach().item()) * count
        correct += int(logits.detach().argmax(dim=1).eq(targets).sum().item())
        if role == "type":
            known = targets > 0
            predictions = logits.detach().argmax(dim=1)
            known_count += int(known.sum().item())
            known_correct += int(
                predictions[known].eq(targets[known]).sum().item()
            )
            known_to_ic += int(predictions[known].eq(0).sum().item())
        progress.set_postfix(
            loss=f"{loss_sum / sample_count:.4f}",
            acc=f"{correct / sample_count:.4f}",
            refresh=False,
        )
    if sample_count == 0:
        raise ValueError("training loader produced no samples")
    metrics: dict[str, float | int | str] = {
        "role": role,
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
    }
    if role == "type":
        metrics["known_accuracy"] = (
            known_correct / known_count if known_count else 0.0
        )
        metrics["known_to_ic_rate"] = (
            known_to_ic / known_count if known_count else 0.0
        )
    return metrics


@dataclass
class EarlyStoppingState:
    """Track a CPU-cloned best model using a strict improvement tolerance."""

    best_score: float = float("-inf")
    best_epoch: int = -1
    wait: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def update(self, score: float, epoch: int, model: nn.Module) -> bool:
        """Record the model only when ``score`` improves by more than 1e-6."""
        numeric_score = float(score)
        if numeric_score > self.best_score + 1e-6:
            self.best_score = numeric_score
            self.best_epoch = int(epoch)
            self.wait = 0
            self.best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            return True
        self.wait += 1
        return False


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_last_state(
    path: str | os.PathLike[str],
    *,
    epoch: int,
    sampler_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    early_stopping: EarlyStoppingState,
    split_hash: str,
    config_hash: str,
) -> None:
    """Atomically save an exact same-run training continuation state."""
    completed_epoch = int(epoch)
    current_sampler_epoch = int(sampler_epoch)
    if completed_epoch < 0 or current_sampler_epoch < 0:
        raise ValueError("resume epochs must be non-negative")
    if not split_hash or not config_hash:
        raise ValueError("resume hashes must be non-empty")
    if early_stopping.best_state is None:
        raise ValueError("resume state requires an in-memory best model")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    optimizer_state = optimizer.state_dict()
    optimizer_state_membership = sorted(optimizer_state["state"])
    payload: dict[str, Any] = {
        "schema": TRAINING_STATE_SCHEMA,
        "epoch": completed_epoch,
        "sampler_epoch": current_sampler_epoch,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer_state,
        "optimizer_state_membership": optimizer_state_membership,
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "early_stopping": {
            "best_score": float(early_stopping.best_score),
            "best_epoch": int(early_stopping.best_epoch),
            "wait": int(early_stopping.wait),
            "best_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in early_stopping.best_state.items()
            },
        },
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "split_hash": str(split_hash),
        "config_hash": str(config_hash),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_training_payload(
    path: str | os.PathLike[str], device: torch.device | str
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("resume checkpoint payload must be a mapping")
    return dict(payload)


def _validated_tensor_state(
    value: object,
    expected: Mapping[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"resume configuration mismatch: {label} keys")
    validated: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        saved_tensor = value[name]
        if (
            not isinstance(saved_tensor, torch.Tensor)
            or saved_tensor.shape != expected_tensor.shape
            or saved_tensor.dtype != expected_tensor.dtype
        ):
            raise ValueError(
                f"resume configuration mismatch: {label} tensor {name!r}"
            )
        validated[name] = saved_tensor
    return validated


def _validate_scheduler_state(
    value: object,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("resume configuration mismatch: scheduler state")
    saved = dict(value)
    current = scheduler.state_dict()
    for field in ("T_max", "eta_min", "base_lrs"):
        if field in current and saved.get(field) != current[field]:
            raise ValueError(
                f"resume configuration mismatch: scheduler {field}"
            )
    base_lrs = saved.get("base_lrs")
    if not isinstance(base_lrs, (list, tuple)) or len(base_lrs) != len(
        optimizer.param_groups
    ):
        raise ValueError("resume configuration mismatch: scheduler base_lrs")
    last_epoch = saved.get("last_epoch")
    if type(last_epoch) is not int or last_epoch != epoch:
        raise ValueError("resume configuration mismatch: scheduler epoch")
    if ("_step_count" in saved) != ("_step_count" in current):
        raise ValueError(
            "resume configuration mismatch: scheduler step-count field"
        )
    if "_step_count" in current:
        step_count = saved["_step_count"]
        if type(step_count) is not int or step_count != epoch + 1:
            raise ValueError(
                "resume configuration mismatch: scheduler step count"
            )
    if "_last_lr" not in saved:
        raise ValueError("resume configuration mismatch: scheduler last lr")
    last_lrs = saved["_last_lr"]
    if (
        not isinstance(last_lrs, (list, tuple))
        or len(last_lrs) != len(optimizer.param_groups)
    ):
        raise ValueError("resume configuration mismatch: scheduler last lr")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in last_lrs
    ):
        raise ValueError("resume configuration mismatch: scheduler last lr")
    return saved


def load_last_state(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    early_stopping: EarlyStoppingState,
    expected_split_hash: str,
    expected_config_hash: str,
    device: torch.device | str = "cpu",
) -> dict[str, int | str]:
    """Validate and restore an exact resumable training checkpoint."""
    # Stage on CPU for stable AdamW validation; state_dict restoration moves
    # model and optimizer tensors to their supported runtime devices.
    payload = _load_training_payload(path, "cpu")
    required = {
        "schema",
        "epoch",
        "sampler_epoch",
        "model_state",
        "optimizer_state",
        "optimizer_state_membership",
        "scheduler_state",
        "scaler_state",
        "early_stopping",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "split_hash",
        "config_hash",
    }
    if set(payload) != required:
        raise ValueError("resume configuration mismatch: checkpoint fields")
    if payload["schema"] != TRAINING_STATE_SCHEMA:
        raise ValueError("resume configuration mismatch: checkpoint schema")
    if (
        payload["split_hash"] != expected_split_hash
        or payload["config_hash"] != expected_config_hash
    ):
        raise ValueError("resume configuration mismatch")

    epoch = payload["epoch"]
    sampler_epoch = payload["sampler_epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("resume configuration mismatch: epoch")
    if type(sampler_epoch) is not int or sampler_epoch != epoch:
        raise ValueError("resume configuration mismatch: sampler epoch")
    early_payload = payload["early_stopping"]
    if not isinstance(early_payload, Mapping):
        raise ValueError("resume configuration mismatch: early stopping")
    if set(early_payload) != {
        "best_score",
        "best_epoch",
        "wait",
        "best_state",
    }:
        raise ValueError("resume configuration mismatch: early stopping")

    expected_state = model.state_dict()
    model_state = _validated_tensor_state(
        payload["model_state"], expected_state, "model state"
    )
    best_state = _validated_tensor_state(
        early_payload["best_state"], expected_state, "best state"
    )
    scheduler_state = _validate_scheduler_state(
        payload["scheduler_state"], scheduler, optimizer, epoch
    )
    optimizer_state = payload["optimizer_state"]
    membership = payload["optimizer_state_membership"]
    if (
        not isinstance(optimizer_state, Mapping)
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state["state"], Mapping)
        or not isinstance(optimizer_state["param_groups"], list)
        or not isinstance(membership, list)
        or membership != sorted(optimizer_state["state"])
    ):
        raise ValueError("resume configuration mismatch: optimizer state")
    scaler_state = payload["scaler_state"]
    if not isinstance(scaler_state, Mapping):
        raise ValueError("resume configuration mismatch: scaler state")
    try:
        best_score = float(early_payload["best_score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume configuration mismatch: best score") from exc
    best_epoch = early_payload["best_epoch"]
    wait = early_payload["wait"]
    if not math.isfinite(best_score):
        raise ValueError("resume configuration mismatch: best score")
    if type(best_epoch) is not int or not 1 <= best_epoch <= epoch:
        raise ValueError("resume configuration mismatch: best epoch")
    if (
        type(wait) is not int
        or wait < 0
        or wait > epoch
        or wait != epoch - best_epoch
    ):
        raise ValueError("resume configuration mismatch: early-stop wait")

    try:
        random.Random().setstate(payload["python_rng_state"])
        np.random.RandomState().set_state(payload["numpy_rng_state"])
        torch.Generator(device="cpu").set_state(
            payload["torch_rng_state"].cpu()
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resume configuration mismatch: RNG state is invalid: {exc}"
        ) from exc

    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        scaler.load_state_dict(dict(scaler_state))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resume configuration mismatch: state restoration failed: {exc}"
        ) from exc

    early_stopping.best_score = best_score
    early_stopping.best_epoch = best_epoch
    early_stopping.wait = wait
    early_stopping.best_state = {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in best_state.items()
    }

    try:
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload["cuda_rng_state"]
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA RNG availability differs")
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in cuda_state]
            )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resume configuration mismatch: RNG restoration failed: {exc}"
        ) from exc

    return {
        "epoch": epoch,
        "sampler_epoch": sampler_epoch,
        "split_hash": str(payload["split_hash"]),
        "config_hash": str(payload["config_hash"]),
    }
