"""Strict inference checkpoint adapters for the two supported schemas."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
import pickle
from typing import Any

import torch
from torch import nn

from models import LegacyMultiTaskResNet, create_five_class_model


FIVE_CLASS_SCHEMA = "five_class_v1"
TRAINING_STATE_SCHEMA = "five_class_training_state_v1"
LEGACY_FIVE_CLASS_SCHEMA = "legacy_five_class"
TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS_KM = tuple(range(0, 3000, 100))

_NEW_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "model_config",
        "model_state",
        "type_names",
        "distance_names",
        "distance_bins_km",
        "preprocess_config",
        "split_hash",
        "training_config",
    }
)
_LEGACY_REQUIRED_FIELDS = frozenset(
    {
        "model_name",
        "base_channels",
        "type_names",
        "dist_names",
        "dist_bin_starts",
        "preprocessing",
        "model_state_dict",
    }
)
_STATE_FIELDS = frozenset({"model_state", "model_state_dict"})
_NEW_PREPROCESS_KEYS = frozenset(
    {
        "local_length",
        "global_length",
        "use_filter",
        "cutoff_hz",
        "sample_rate_hz",
    }
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated inference model and schema-normalized metadata."""

    model: nn.Module
    schema: str
    type_names: tuple[str, ...]
    distance_names: tuple[str, ...]
    distance_bins_km: tuple[int, ...]
    preprocess_config: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _load_payload(path: os.PathLike[str] | str, device: torch.device | str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("checkpoint payload cannot be safely loaded") from exc
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=device)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return dict(value)


def _required_fields(checkpoint: Mapping[str, Any], required: frozenset[str]):
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")


def _ordered_names(value: object, expected: tuple[str, ...], label: str):
    if not isinstance(value, (list, tuple)) or tuple(value) != expected:
        raise ValueError(f"checkpoint {label} order is invalid")
    return expected


def _distance_bins(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or tuple(value) != DISTANCE_BINS_KM:
        raise ValueError("checkpoint distance-bin configuration is invalid")
    return DISTANCE_BINS_KM


def _base_channels(config: Mapping[str, Any]) -> int:
    if set(config) != {"base_channels"}:
        raise ValueError("checkpoint model_config is invalid")
    value = config["base_channels"]
    if type(value) is not int or value <= 0:
        raise ValueError("checkpoint base_channels must be a positive integer")
    return value


def _state_mapping(value: object) -> dict[str, torch.Tensor]:
    state = _mapping(value, "model state")
    if not state or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in state.items()
    ):
        raise ValueError("checkpoint model state contains malformed tensors")
    return state


def _new_preprocess_config(value: object) -> dict[str, Any]:
    config = _mapping(value, "preprocess_config")
    unknown = sorted(set(config) - _NEW_PREPROCESS_KEYS)
    if unknown:
        raise ValueError(
            f"checkpoint preprocess_config has unknown fields: {unknown}"
        )
    if type(config.get("local_length")) is not int or config[
        "local_length"
    ] != 8000:
        raise ValueError("checkpoint preprocess_config local_length is invalid")
    if type(config.get("global_length")) is not int or config[
        "global_length"
    ] != 2000:
        raise ValueError("checkpoint preprocess_config global_length is invalid")
    if "use_filter" in config and type(config["use_filter"]) is not bool:
        raise ValueError("checkpoint preprocess_config use_filter is invalid")
    cutoff_hz = config.get("cutoff_hz", 120_000.0)
    sample_rate_hz = config.get("sample_rate_hz", 5_000_000.0)
    for name, number in (
        ("cutoff_hz", cutoff_hz),
        ("sample_rate_hz", sample_rate_hz),
    ):
        if (
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise ValueError(
                f"checkpoint preprocess_config {name} is invalid"
            )
    if float(cutoff_hz) >= float(sample_rate_hz) / 2.0:
        raise ValueError(
            "checkpoint preprocess_config cutoff_hz must be below Nyquist"
        )
    return config


def _legacy_preprocess_config(value: object) -> dict[str, Any]:
    config = _mapping(value, "preprocessing")
    if config.get("normalize_mode") != "minmax":
        raise ValueError("checkpoint legacy preprocessing mode is invalid")
    if config.get("target_length") != 8000:
        raise ValueError("checkpoint legacy preprocessing length is invalid")
    return config


def _reject_optimizer_metadata(value: object, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key)
            nested_path = f"{path}.{key_name}"
            if isinstance(key, str) and "optimizer" in key.casefold():
                raise ValueError(
                    f"checkpoint optimizer metadata is forbidden: {nested_path}"
                )
            _reject_optimizer_metadata(nested, nested_path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_optimizer_metadata(nested, f"{path}[{index}]")


def _inference_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value for key, value in checkpoint.items() if key not in _STATE_FIELDS
    }
    _reject_optimizer_metadata(metadata)
    return {
        key: deepcopy(value)
        for key, value in metadata.items()
    }


def _strict_load_model(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> nn.Module:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint model state is invalid: {exc}") from exc
    return model.to(device)


def _load_new(
    checkpoint: Mapping[str, Any], device: torch.device | str
) -> LoadedCheckpoint:
    _required_fields(checkpoint, _NEW_REQUIRED_FIELDS)
    type_names = _ordered_names(checkpoint["type_names"], TYPE_NAMES, "type")
    distance_names = _ordered_names(
        checkpoint["distance_names"], DISTANCE_NAMES, "distance-name"
    )
    distance_bins = _distance_bins(checkpoint["distance_bins_km"])
    model_config = _mapping(checkpoint["model_config"], "model_config")
    base_channels = _base_channels(model_config)
    preprocess_config = _new_preprocess_config(
        checkpoint["preprocess_config"]
    )
    if not isinstance(checkpoint["split_hash"], str) or not checkpoint[
        "split_hash"
    ]:
        raise ValueError("checkpoint split_hash is invalid")
    _mapping(checkpoint["training_config"], "training_config")
    state = _state_mapping(checkpoint["model_state"])
    model = _strict_load_model(
        create_five_class_model(base_channels=base_channels), state, device
    )
    return LoadedCheckpoint(
        model=model,
        schema=FIVE_CLASS_SCHEMA,
        type_names=type_names,
        distance_names=distance_names,
        distance_bins_km=distance_bins,
        preprocess_config=preprocess_config,
        metadata=_inference_metadata(checkpoint),
    )


def _load_legacy(
    checkpoint: Mapping[str, Any], device: torch.device | str
) -> LoadedCheckpoint:
    _required_fields(checkpoint, _LEGACY_REQUIRED_FIELDS)
    if checkpoint["model_name"] != "mtl_resnet":
        raise ValueError("checkpoint legacy model name is invalid")
    base = checkpoint["base_channels"]
    if type(base) is not int or base <= 0:
        raise ValueError("checkpoint legacy base_channels is invalid")
    type_names = _ordered_names(checkpoint["type_names"], TYPE_NAMES, "type")
    distance_names = _ordered_names(
        checkpoint["dist_names"], DISTANCE_NAMES, "distance-name"
    )
    distance_bins = _distance_bins(checkpoint["dist_bin_starts"])
    preprocess_config = _legacy_preprocess_config(checkpoint["preprocessing"])
    state = _state_mapping(checkpoint["model_state_dict"])
    model = _strict_load_model(LegacyMultiTaskResNet(base=base), state, device)
    return LoadedCheckpoint(
        model=model,
        schema=LEGACY_FIVE_CLASS_SCHEMA,
        type_names=type_names,
        distance_names=distance_names,
        distance_bins_km=distance_bins,
        preprocess_config=preprocess_config,
        metadata=_inference_metadata(checkpoint),
    )


def load_model_checkpoint(
    path: os.PathLike[str] | str, device: torch.device | str = "cpu"
) -> LoadedCheckpoint:
    """Load and strictly validate one supported inference checkpoint."""

    payload = _load_payload(path, device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    checkpoint = dict(payload)
    _reject_optimizer_metadata(
        {
            key: value
            for key, value in checkpoint.items()
            if key not in _STATE_FIELDS
        }
    )
    schema = checkpoint.get("schema")
    if schema == FIVE_CLASS_SCHEMA:
        return _load_new(checkpoint, device)
    if schema == LEGACY_FIVE_CLASS_SCHEMA or (
        schema is None and checkpoint.get("model_name") == "mtl_resnet"
    ):
        return _load_legacy(checkpoint, device)
    if schema is None:
        raise ValueError("checkpoint format is not a supported five-class model")
    raise ValueError(f"checkpoint schema is unsupported: {schema!r}")


def save_model_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    *,
    model_config: Mapping[str, Any],
    preprocess_config: Mapping[str, Any],
    split_hash: str,
    training_config: Mapping[str, Any],
) -> None:
    """Atomically save a validated ``five_class_v1`` inference checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    checkpoint = {
        "schema": FIVE_CLASS_SCHEMA,
        "model_config": dict(model_config),
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "type_names": list(TYPE_NAMES),
        "distance_names": list(DISTANCE_NAMES),
        "distance_bins_km": list(DISTANCE_BINS_KM),
        "preprocess_config": dict(preprocess_config),
        "split_hash": split_hash,
        "training_config": dict(training_config),
    }
    try:
        torch.save(checkpoint, temporary)
        load_model_checkpoint(temporary, "cpu")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def validate_optimizer_resume_state(
    value: object,
    membership: object,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scheduler_state: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Validate optimizer state before exact same-run restoration."""
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


def model_sha256(path: os.PathLike[str] | str) -> str:
    """Return a streaming SHA-256 digest of a checkpoint file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
