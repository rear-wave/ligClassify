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
from torch import nn
from torch.nn import functional as F

from checkpoints import TRAINING_STATE_SCHEMA
from models import DISTANCE_BIN_COUNT, DISTANCE_EXPERT_COUNT, ModelOutput, TYPE_COUNT


@dataclass(frozen=True)
class JointLoss:
    """Differentiable total loss with detached logging components."""

    total: torch.Tensor
    type_loss: float
    distance_loss: float
    distance_count: int


def _validate_joint_shapes(
    output: ModelOutput,
    type_labels: torch.Tensor,
    distance_bins: torch.Tensor,
) -> None:
    batch_size = len(type_labels)
    if type_labels.ndim != 1 or distance_bins.ndim != 1:
        raise ValueError("type_label and distance_bin must be one-dimensional")
    if len(distance_bins) != batch_size:
        raise ValueError("type_label and distance_bin batch sizes differ")
    if tuple(output.type_logits.shape) != (batch_size, TYPE_COUNT):
        raise ValueError("type_logits must have shape [batch, 5]")
    if len(output.distance_logits) != DISTANCE_EXPERT_COUNT:
        raise ValueError("exactly four distance experts are required")
    if any(
        tuple(logits.shape) != (batch_size, DISTANCE_BIN_COUNT)
        for logits in output.distance_logits
    ):
        raise ValueError("distance logits must have shape [batch, 30]")


def compute_joint_loss(
    output: ModelOutput,
    batch: Mapping[str, object],
    distance_weight: float = 0.5,
) -> JointLoss:
    """Compute five-class CE and true-type-routed ordered distance loss."""
    if not math.isfinite(float(distance_weight)) or float(distance_weight) < 0:
        raise ValueError("distance_weight must be finite and non-negative")
    if not isinstance(batch.get("type_label"), torch.Tensor) or not isinstance(
        batch.get("distance_bin"), torch.Tensor
    ):
        raise TypeError("batch labels must be tensors")

    device = output.type_logits.device
    type_labels = batch["type_label"].to(device=device, dtype=torch.long)
    distance_bins = batch["distance_bin"].to(device=device, dtype=torch.long)
    _validate_joint_shapes(output, type_labels, distance_bins)

    if torch.any((type_labels < 0) | (type_labels >= TYPE_COUNT)):
        raise ValueError("type_label must be in the five-class range 0..4")
    ic_mask = type_labels == 0
    if torch.any(distance_bins[ic_mask] != -1):
        raise ValueError("IC labels must use distance_bin -1")
    non_ic_mask = ~ic_mask
    if torch.any(
        (distance_bins[non_ic_mask] < 0)
        | (distance_bins[non_ic_mask] >= DISTANCE_BIN_COUNT)
    ):
        raise ValueError("non-IC labels must use a distance bin in 0..29")

    type_loss_tensor = F.cross_entropy(output.type_logits, type_labels)
    distance_count = int(non_ic_mask.sum().item())
    if distance_count:
        routed_chunks = []
        target_chunks = []
        for type_index in range(1, TYPE_COUNT):
            selected = type_labels == type_index
            if torch.any(selected):
                routed_chunks.append(output.distance_logits[type_index - 1][selected])
                target_chunks.append(distance_bins[selected])
        routed_logits = torch.cat(routed_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        categorical = F.cross_entropy(routed_logits, targets)
        probabilities = routed_logits.softmax(dim=1)
        predicted_cdf = probabilities.cumsum(dim=1)
        target_cdf = (
            torch.arange(DISTANCE_BIN_COUNT, device=device)[None, :]
            >= targets[:, None]
        ).to(routed_logits.dtype)
        ordered = torch.abs(predicted_cdf - target_cdf).mean()
        distance_loss_tensor = categorical + 0.2 * ordered
    else:
        distance_loss_tensor = output.type_logits.sum() * 0.0

    total = type_loss_tensor + float(distance_weight) * distance_loss_tensor
    return JointLoss(
        total=total,
        type_loss=float(type_loss_tensor.detach().item()),
        distance_loss=float(distance_loss_tensor.detach().item()),
        distance_count=distance_count,
    )


def _move_training_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, object]:
    moved = dict(batch)
    for key in ("local", "global", "daylight", "type_label", "distance_bin"):
        value = moved.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch field {key!r} must be a tensor")
        moved[key] = value.to(device)
    return moved


def train_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    distance_weight: float = 0.5,
) -> dict[str, float | int]:
    """Train for one epoch using exactly one model forward per batch."""
    target_device = torch.device(device)
    amp_enabled = bool(amp and target_device.type == "cuda")
    if scaler is None:
        scaler = torch.amp.GradScaler(target_device.type, enabled=amp_enabled)
    model.train()

    sample_count = 0
    distance_count = 0
    type_sum = 0.0
    distance_sum = 0.0
    for raw_batch in loader:
        batch = _move_training_batch(raw_batch, target_device)
        batch_size = int(batch["type_label"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=target_device.type,
            enabled=amp_enabled,
        ):
            output = model(batch["local"], batch["global"], batch["daylight"])
            losses = compute_joint_loss(
                output, batch, distance_weight=distance_weight
            )
        scaler.scale(losses.total).backward()
        scaler.step(optimizer)
        scaler.update()

        sample_count += batch_size
        distance_count += losses.distance_count
        type_sum += losses.type_loss * batch_size
        distance_sum += losses.distance_loss * losses.distance_count

    if sample_count == 0:
        raise ValueError("training loader produced no samples")
    mean_type = type_sum / sample_count
    mean_distance = distance_sum / distance_count if distance_count else 0.0
    return {
        "loss": mean_type + float(distance_weight) * mean_distance,
        "type_loss": mean_type,
        "distance_loss": mean_distance,
        "sample_count": sample_count,
        "distance_count": distance_count,
    }


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


def _same_number(saved: object, current: object) -> bool:
    return (
        not isinstance(saved, bool)
        and isinstance(saved, Real)
        and not isinstance(current, bool)
        and isinstance(current, Real)
        and math.isfinite(float(saved))
        and math.isclose(
            float(saved), float(current), rel_tol=1e-12, abs_tol=1e-15
        )
    )


def _validate_optimizer_state(
    value: object,
    membership: object,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scheduler_state: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "state",
        "param_groups",
    }:
        raise ValueError("resume configuration mismatch: optimizer state")
    saved_state = value["state"]
    saved_groups = value["param_groups"]
    current_state = optimizer.state_dict()
    current_groups = current_state["param_groups"]
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, list):
        raise ValueError("resume configuration mismatch: optimizer structure")
    if len(saved_groups) != len(current_groups):
        raise ValueError("resume configuration mismatch: optimizer groups")

    scheduler_base_lrs = scheduler_state["base_lrs"]
    scheduler_last_lrs = scheduler_state["_last_lr"]
    all_parameter_ids: list[int] = []
    parameter_by_id: dict[int, torch.Tensor] = {}
    amsgrad_by_id: dict[int, bool] = {}
    step_dtype_by_id: dict[int, torch.dtype] = {}
    step_device_by_id: dict[int, torch.device] = {}
    immutable_fields = (
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    )
    for index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups)
    ):
        if not isinstance(saved_group, Mapping):
            raise ValueError(
                "resume configuration mismatch: optimizer group structure"
            )
        if set(saved_group) != set(current_group):
            raise ValueError("resume configuration mismatch: optimizer group keys")
        saved_parameters = saved_group.get("params")
        current_parameters = current_group["params"]
        if (
            not isinstance(saved_parameters, list)
            or saved_parameters != current_parameters
            or any(type(parameter_id) is not int for parameter_id in saved_parameters)
        ):
            raise ValueError(
                "resume configuration mismatch: optimizer parameter membership"
            )
        all_parameter_ids.extend(saved_parameters)
        live_parameters = optimizer.param_groups[index]["params"]
        if len(live_parameters) != len(saved_parameters):
            raise ValueError(
                "resume configuration mismatch: optimizer parameter count"
            )
        parameter_by_id.update(zip(saved_parameters, live_parameters))
        amsgrad_by_id.update(
            (parameter_id, bool(current_group["amsgrad"]))
            for parameter_id in saved_parameters
        )
        fused = bool(current_group.get("fused", False))
        step_dtype = (
            torch.float64
            if torch.get_default_dtype() == torch.float64 and not fused
            else torch.float32
        )
        for parameter_id in saved_parameters:
            step_dtype_by_id[parameter_id] = step_dtype
            step_device_by_id[parameter_id] = torch.device("cpu")

        for field in immutable_fields:
            if field not in current_group:
                continue
            saved_option = saved_group[field]
            current_option = current_group[field]
            if isinstance(current_option, bool):
                same_option = (
                    type(saved_option) is bool and saved_option == current_option
                )
            elif current_option is None:
                same_option = saved_option is None
            else:
                same_option = saved_option == current_option
            if not same_option:
                raise ValueError(
                    f"resume configuration mismatch: optimizer {field}"
                )
        betas = saved_group["betas"]
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or not all(
                _same_number(saved, current)
                for saved, current in zip(betas, current_group["betas"])
            )
        ):
            raise ValueError("resume configuration mismatch: optimizer betas")
        for field in ("weight_decay", "eps"):
            if not _same_number(saved_group[field], current_group[field]):
                raise ValueError(
                    f"resume configuration mismatch: optimizer {field}"
                )

        if "initial_lr" in current_group:
            if (
                "initial_lr" not in saved_group
                or not _same_number(
                    saved_group["initial_lr"], current_group["initial_lr"]
                )
                or not _same_number(
                    saved_group["initial_lr"], scheduler_base_lrs[index]
                )
            ):
                raise ValueError(
                    "resume configuration mismatch: optimizer initial lr"
                )
        saved_lr = saved_group.get("lr")
        if (
            not _same_number(saved_lr, scheduler_last_lrs[index])
            or float(saved_lr) < 0.0
        ):
            raise ValueError("resume configuration mismatch: optimizer current lr")

        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            base_lr = float(scheduler_base_lrs[index])
            eta_min = float(scheduler_state["eta_min"])
            t_max = int(scheduler_state["T_max"])
            expected_lr = eta_min + (base_lr - eta_min) * (
                1.0 + math.cos(math.pi * epoch / t_max)
            ) / 2.0
            if not math.isclose(
                float(saved_lr), expected_lr, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(
                    "resume configuration mismatch: cosine learning rate"
                )

    if len(set(all_parameter_ids)) != len(all_parameter_ids):
        raise ValueError("resume configuration mismatch: optimizer parameters")
    parameter_id_set = set(all_parameter_ids)
    if (
        not isinstance(membership, list)
        or any(type(parameter_id) is not int for parameter_id in membership)
        or len(set(membership)) != len(membership)
        or not set(membership).issubset(parameter_id_set)
    ):
        raise ValueError(
            "resume configuration mismatch: optimizer state membership manifest"
        )
    if set(saved_state) != set(membership):
        raise ValueError(
            "resume configuration mismatch: optimizer state membership"
        )
    for parameter_id, state in saved_state.items():
        if type(parameter_id) is not int or parameter_id not in parameter_id_set:
            raise ValueError(
                "resume configuration mismatch: optimizer state membership"
            )
        if not isinstance(state, Mapping):
            raise ValueError(
                "resume configuration mismatch: optimizer parameter state"
            )
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if amsgrad_by_id[parameter_id]:
            expected_keys.add("max_exp_avg_sq")
        if set(state) != expected_keys:
            raise ValueError(
                "resume configuration mismatch: optimizer parameter state keys"
            )
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != torch.Size([])
            or step.dtype != step_dtype_by_id[parameter_id]
            or step.device != step_device_by_id[parameter_id]
            or not bool(torch.isfinite(step).item())
            or float(step.item()) < 1.0
        ):
            raise ValueError(
                "resume configuration mismatch: optimizer step state"
            )
        parameter = parameter_by_id[parameter_id]
        moment_names = {"exp_avg", "exp_avg_sq"}
        if amsgrad_by_id[parameter_id]:
            moment_names.add("max_exp_avg_sq")
        for name in moment_names:
            moment = state[name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
                or not bool(torch.isfinite(moment).all().item())
            ):
                raise ValueError(
                    f"resume configuration mismatch: optimizer {name} state"
                )
    return {"state": dict(saved_state), "param_groups": list(saved_groups)}


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
    optimizer_state = _validate_optimizer_state(
        payload["optimizer_state"],
        payload["optimizer_state_membership"],
        optimizer,
        scheduler,
        scheduler_state,
        epoch,
    )
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
