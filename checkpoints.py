"""Strict inference checkpoint adapters for the two supported schemas."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models import LegacyMultiTaskResNet, create_five_class_model


FIVE_CLASS_SCHEMA = "five_class_v1"
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
    if config.get("local_length") != 8000:
        raise ValueError("checkpoint preprocess_config local_length is invalid")
    if config.get("global_length") != 2000:
        raise ValueError("checkpoint preprocess_config global_length is invalid")
    return config


def _legacy_preprocess_config(value: object) -> dict[str, Any]:
    config = _mapping(value, "preprocessing")
    if config.get("normalize_mode") != "minmax":
        raise ValueError("checkpoint legacy preprocessing mode is invalid")
    if config.get("target_length") != 8000:
        raise ValueError("checkpoint legacy preprocessing length is invalid")
    return config


def _inference_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in checkpoint.items()
        if key not in _STATE_FIELDS and "optimizer" not in key.lower()
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


def model_sha256(path: os.PathLike[str] | str) -> str:
    """Return a streaming SHA-256 digest of a checkpoint file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
