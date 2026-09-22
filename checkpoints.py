"""Strict inference checkpoint adapters for the two supported schemas."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import pickle
from typing import Any

import torch
from torch import nn

from data.preprocess import PreprocessConfig
from models import (
    ANCHOR_TYPE_VARIANT, HIERARCHICAL_TYPE_ARCHITECTURE,
    LegacyMultiTaskResNet,
    create_anchor_hierarchical_type_model, create_five_class_model,
    create_hierarchical_type_model,
)


FIVE_CLASS_SCHEMA = "five_class_v1"
HIERARCHICAL_FIVE_CLASS_SCHEMA = "hierarchical_five_class_v2"
TRAINING_STATE_SCHEMA = "five_class_training_state_v1"
LEGACY_FIVE_CLASS_SCHEMA = "legacy_five_class"
TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS_KM = tuple(range(0, 3000, 100))
MODEL_BUNDLE_SCHEMA = "five_class_model_bundle_v1"
HIERARCHICAL_MODEL_BUNDLE_SCHEMA = "hierarchical_model_bundle_v2"
BUNDLE_ROLES = ("type", *DISTANCE_NAMES)
DECISION_CONFIG_SCHEMA_V1 = "hierarchical_decision_config_v1"
DECISION_CONFIG_SCHEMA = "hierarchical_decision_config_v2"

_NEW_REQUIRED_FIELDS = frozenset(
    {"schema", "model_config", "model_state", "type_names", "distance_names",
     "distance_bins_km", "preprocess_config", "split_hash", "training_config"}
)
_HIERARCHICAL_REQUIRED_FIELDS = _NEW_REQUIRED_FIELDS | {"decision_config"}
_LEGACY_REQUIRED_FIELDS = frozenset(
    {"model_name", "base_channels", "type_names", "dist_names",
     "dist_bin_starts", "preprocessing", "model_state_dict"}
)
_STATE_FIELDS = frozenset({"model_state", "model_state_dict"})
_NEW_PREPROCESS_KEYS = frozenset(
    {"local_length", "global_length", "use_filter", "cutoff_hz",
     "sample_rate_hz", "local_center_mode", "local_energy_window"}
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


@dataclass(frozen=True)
class LoadedModelBundle:
    """Five independently initialized role checkpoints."""

    type_checkpoint: LoadedCheckpoint
    distance_checkpoints: tuple[LoadedCheckpoint, ...]
    hashes: dict[str, str]


@dataclass(frozen=True)
class GuardedTypeVerifier:
    """Optional checkpoint-bound confirmation of rejected known candidates."""

    primary: LoadedCheckpoint
    checkpoint: LoadedCheckpoint
    limits: tuple[tuple[float, ...], ...]
    signature: str
    sha256: str
    decision_hashes: tuple[str, str]

    def infer(self, primary, local, global_view, daylight, alternate_daylight, missing):
        from evaluation import infer_hierarchical_types

        current = tuple(decision_config_sha256(cp.metadata["decision_config"])
                        for cp in (primary, self.checkpoint))
        if primary is not self.primary or current != self.decision_hashes:
            raise ValueError("type verifier primary or decision configuration changed")
        return infer_hierarchical_types(primary.model, primary.metadata["decision_config"],
            local, global_view, daylight, alternate_daylight, missing, verifier=self)


def load_type_verifier(path, primary: LoadedCheckpoint, primary_sha256: str,
                       device: torch.device | str = "cpu") -> GuardedTypeVerifier | None:
    """Load an explicit, hash-bound verification recipe; never change defaults."""
    if path is None:
        return None
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("type verifier JSON is invalid") from exc
    required = {"schema", "primary_sha256", "primary_decision_sha256", "verifier_path", "verifier_sha256", "limits"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != "guarded_type_verifier_v1":
        raise ValueError("type verifier schema or fields are invalid")
    for key in ("primary_sha256", "primary_decision_sha256", "verifier_sha256"):
        value = payload[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"type verifier {key} is invalid")
    if primary.schema != HIERARCHICAL_FIVE_CLASS_SCHEMA or payload["primary_sha256"] != primary_sha256:
        raise ValueError("type verifier primary checkpoint mismatch")
    decision_hash = decision_config_sha256(primary.metadata["decision_config"])
    if (decision_hash != payload["primary_decision_sha256"] or
            primary.metadata["decision_config"]["min_branch_votes"] != 2):
        raise ValueError("type verifier primary decision configuration mismatch")
    raw_limits = payload["limits"]
    if not isinstance(raw_limits, dict) or set(raw_limits) != set(DISTANCE_NAMES):
        raise ValueError("type verifier limits must cover all four known classes")
    limits = []
    for name in DISTANCE_NAMES:
        row = raw_limits[name]
        if (not isinstance(row, list) or len(row) != 4 or any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v <= 1 for v in row)):
            raise ValueError(f"type verifier limits for {name} must be four finite probabilities")
        limits.append(tuple(float(v) for v in row))
    relative = payload["verifier_path"]
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("type verifier path is invalid")
    verifier_path = (config_path.parent / relative).resolve()
    if not verifier_path.is_file() or model_sha256(verifier_path) != payload["verifier_sha256"]:
        raise ValueError("type verifier checkpoint missing or hash mismatch")
    checkpoint = load_model_checkpoint(verifier_path, device)
    if (checkpoint.schema != HIERARCHICAL_FIVE_CLASS_SCHEMA or
            checkpoint.preprocess_config != primary.preprocess_config or
            checkpoint.metadata.get("split_hash") != primary.metadata.get("split_hash")):
        raise ValueError("type verifier checkpoint is incompatible")
    primary.model.eval()
    checkpoint.model.eval()
    identity = {key: value for key, value in payload.items() if key != "verifier_path"}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return GuardedTypeVerifier(primary, checkpoint, tuple(limits), signature, payload["verifier_sha256"],
        (decision_hash, decision_config_sha256(checkpoint.metadata["decision_config"])))


def forward_model_bundle(
    bundle: LoadedModelBundle,
    local: torch.Tensor,
    global_view: torch.Tensor,
    daylight: torch.Tensor,
    *,
    type_only: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run type first and route rows to independent distance checkpoints."""
    models = (
        bundle.type_checkpoint.model,
        *(item.model for item in bundle.distance_checkpoints),
    )
    for model in models:
        model.eval()
    type_logits, _ = models[0].forward_type(local, global_view, daylight)
    if tuple(type_logits.shape) != (len(local), len(TYPE_NAMES)):
        raise ValueError("type model returned malformed logits")
    predicted = type_logits.argmax(dim=1)
    heads = [
        type_logits.new_zeros((len(local), len(DISTANCE_BINS_KM)))
        for _ in DISTANCE_NAMES
    ]
    if type_only:
        return type_logits, tuple(heads)
    for type_index, checkpoint in enumerate(
        bundle.distance_checkpoints, start=1
    ):
        selected = predicted == type_index
        if not torch.any(selected):
            continue
        distance_logits, _ = checkpoint.model.forward_distance_type(
            local[selected],
            global_view[selected],
            daylight[selected],
            expert_index=type_index - 1,
        )
        expected = (int(selected.sum().item()), len(DISTANCE_BINS_KM))
        if tuple(distance_logits.shape) != expected:
            raise ValueError(
                f"{TYPE_NAMES[type_index]} model returned malformed logits"
            )
        heads[type_index - 1][selected] = distance_logits
    return type_logits, tuple(heads)


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


def _hierarchical_model_config(value: object) -> dict[str, Any]:
    config = _mapping(value, "model_config")
    common = {"base_channels", "embedding_dim", "prototypes_per_class", "prototype_logit_weight"}
    anchor = {"type_backbone", "patch_length", "patch_stride", "expert_count", "expert_topk", "frequency_bands", "use_reliability"}
    from models import MULTISCALE_TYPE_VARIANT
    multiscale = config.get("type_backbone") == MULTISCALE_TYPE_VARIANT
    allowed = ({frozenset(common | {"type_backbone"})} if multiscale else
               {frozenset(common), frozenset(common | anchor), frozenset(common | anchor | {"known_fusion_weight"})})
    if set(config) not in allowed:
        raise ValueError("checkpoint hierarchical model_config is invalid")
    integers = ("base_channels", "embedding_dim", "prototypes_per_class")
    if "type_backbone" in config and not multiscale:
        if config["type_backbone"] != ANCHOR_TYPE_VARIANT or type(config["use_reliability"]) is not bool:
            raise ValueError("checkpoint AnchorMoE model_config is invalid")
        integers += ("patch_length", "patch_stride", "expert_count", "expert_topk", "frequency_bands")
    for name in integers:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"checkpoint {name} must be a positive integer")
    for name in ("prototype_logit_weight", "known_fusion_weight"):
        if name not in config:
            continue
        weight = config[name]
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(float(weight)) or float(weight) < 0.0):
            raise ValueError(f"checkpoint {name} is invalid")
    return config


def _decision_config(value: object) -> dict[str, Any]:
    config = _mapping(value, "decision_config")
    common = {
        "known_probability_thresholds",
        "prototype_similarity_thresholds",
        "min_branch_votes",
    }
    optional = {"max_ic_gate_probabilities"}
    has_v1_js = "max_js_divergence" in config
    has_v2_js = "max_js_divergences" in config
    if has_v1_js == has_v2_js:
        raise ValueError(
            "checkpoint decision_config must contain exactly one JS field"
        )
    required = common | {
        "max_js_divergence" if has_v1_js else "max_js_divergences"
    }
    if not required.issubset(config) or set(config) - required - optional:
        raise ValueError("checkpoint decision_config fields are invalid")
    probability = config["known_probability_thresholds"]
    similarity = config["prototype_similarity_thresholds"]
    gate_probability = config.get(
        "max_ic_gate_probabilities", [0.5, 0.5, 0.5, 0.5]
    )
    if (
        not isinstance(probability, (list, tuple))
        or len(probability) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not 0.0 <= float(item) <= 1.0
            for item in probability
        )
    ):
        raise ValueError("checkpoint known probability thresholds are invalid")
    if (
        not isinstance(similarity, (list, tuple))
        or len(similarity) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not -1.0 <= float(item) <= 1.0
            for item in similarity
        )
    ):
        raise ValueError("checkpoint prototype thresholds are invalid")
    if (
        not isinstance(gate_probability, (list, tuple))
        or len(gate_probability) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not 0.0 <= float(item) <= 1.0
            for item in gate_probability
        )
    ):
        raise ValueError("checkpoint IC gate thresholds are invalid")
    raw_divergence = config[
        "max_js_divergence" if has_v1_js else "max_js_divergences"
    ]
    divergences = [raw_divergence] * 4 if has_v1_js else raw_divergence
    votes = config["min_branch_votes"]
    if (
        not isinstance(divergences, (list, tuple))
        or len(divergences) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= math.log(2.0) + 1e-6
            for item in divergences
        )
    ):
        raise ValueError("checkpoint max_js_divergences are invalid")
    if type(votes) is not int or not 1 <= votes <= 4:
        raise ValueError("checkpoint min_branch_votes is invalid")
    return {
        "known_probability_thresholds": [float(item) for item in probability],
        "prototype_similarity_thresholds": [float(item) for item in similarity],
        "max_ic_gate_probabilities": [float(item) for item in gate_probability],
        "max_js_divergences": [float(item) for item in divergences],
        "min_branch_votes": votes,
    }


def validate_decision_config(value: object) -> dict[str, Any]:
    """Validate and normalize hierarchical inference thresholds."""
    return _decision_config(value)


def load_decision_config(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Load one versioned external hierarchical decision configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"decision_config is not a file: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision_config JSON is invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "decision_config"
    }:
        raise ValueError("decision_config must use schema and decision_config fields")
    schema = payload["schema"]
    if schema not in {DECISION_CONFIG_SCHEMA_V1, DECISION_CONFIG_SCHEMA}:
        raise ValueError(
            "decision_config must use schema "
            f"{DECISION_CONFIG_SCHEMA_V1!r} or {DECISION_CONFIG_SCHEMA!r}"
        )
    raw_config = _mapping(payload["decision_config"], "decision_config")
    expected_js = (
        "max_js_divergence"
        if schema == DECISION_CONFIG_SCHEMA_V1
        else "max_js_divergences"
    )
    unexpected_js = (
        "max_js_divergences"
        if schema == DECISION_CONFIG_SCHEMA_V1
        else "max_js_divergence"
    )
    if expected_js not in raw_config or unexpected_js in raw_config:
        raise ValueError(
            f"decision_config schema {schema!r} requires {expected_js}"
        )
    return validate_decision_config(raw_config)


def override_decision_config(
    checkpoint: LoadedCheckpoint,
    decision_config: Mapping[str, Any] | None,
) -> LoadedCheckpoint:
    """Return a checkpoint with an in-memory hierarchical decision override."""
    if decision_config is None:
        return checkpoint
    if checkpoint.schema != HIERARCHICAL_FIVE_CLASS_SCHEMA:
        raise ValueError(
            "--decision_config requires a hierarchical_five_class_v2 "
            "type checkpoint"
        )
    metadata = dict(checkpoint.metadata)
    metadata["decision_config"] = validate_decision_config(decision_config)
    return replace(checkpoint, metadata=metadata)


def decision_config_sha256(
    decision_config: Mapping[str, Any] | None,
) -> str | None:
    """Hash a normalized external decision configuration."""
    if decision_config is None:
        return None
    canonical = json.dumps(
        {
            "schema": DECISION_CONFIG_SCHEMA,
            "decision_config": validate_decision_config(decision_config),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def checkpoint_preprocess_config(
    checkpoint: LoadedCheckpoint,
) -> PreprocessConfig:
    """Build the preprocessing adapter stored by a loaded checkpoint."""
    config = checkpoint.preprocess_config
    return PreprocessConfig(
        local_length=int(config["local_length"]),
        global_length=int(config["global_length"]),
        use_filter=bool(config.get("use_filter", True)),
        cutoff_hz=float(config.get("cutoff_hz", 120_000.0)),
        sample_rate_hz=float(config.get("sample_rate_hz", 5_000_000.0)),
        local_center_mode=str(config.get("local_center_mode", "peak_abs_v1")),
        local_energy_window=int(config.get("local_energy_window", 128)),
    )


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
    center_mode = config.get("local_center_mode", "peak_abs_v1")
    energy_window = config.get("local_energy_window", 128)
    if center_mode not in {"peak_abs_v1", "energy_envelope_v2"}:
        raise ValueError("checkpoint local_center_mode is invalid")
    if type(energy_window) is not int or energy_window <= 0:
        raise ValueError("checkpoint local_energy_window is invalid")
    if "local_center_mode" in config:
        config["local_center_mode"] = center_mode
    if "local_energy_window" in config:
        config["local_energy_window"] = energy_window
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
    distance_names = _ordered_names(checkpoint["distance_names"], DISTANCE_NAMES, "distance-name")
    distance_bins = _distance_bins(checkpoint["distance_bins_km"])
    model_config = _mapping(checkpoint["model_config"], "model_config")
    base_channels = _base_channels(model_config)
    preprocess_config = _new_preprocess_config(checkpoint["preprocess_config"])
    if not isinstance(checkpoint["split_hash"], str) or not checkpoint["split_hash"]:
        raise ValueError("checkpoint split_hash is invalid")
    _mapping(checkpoint["training_config"], "training_config")
    state = _state_mapping(checkpoint["model_state"])
    model = _strict_load_model(
        create_five_class_model(base_channels=base_channels), state, device
    )
    return LoadedCheckpoint(model=model, schema=FIVE_CLASS_SCHEMA, type_names=type_names,
        distance_names=distance_names, distance_bins_km=distance_bins,
        preprocess_config=preprocess_config, metadata=_inference_metadata(checkpoint))


def _load_hierarchical(
    checkpoint: Mapping[str, Any], device: torch.device | str
) -> LoadedCheckpoint:
    _required_fields(checkpoint, _HIERARCHICAL_REQUIRED_FIELDS)
    if set(checkpoint) != set(_HIERARCHICAL_REQUIRED_FIELDS):
        raise ValueError("hierarchical checkpoint fields are invalid")
    type_names = _ordered_names(checkpoint["type_names"], TYPE_NAMES, "type")
    distance_names = _ordered_names(checkpoint["distance_names"], DISTANCE_NAMES, "distance-name")
    distance_bins = _distance_bins(checkpoint["distance_bins_km"])
    model_config = _hierarchical_model_config(checkpoint["model_config"])
    preprocess_config = _new_preprocess_config(checkpoint["preprocess_config"])
    decision_config = _decision_config(checkpoint["decision_config"])
    training = _mapping(checkpoint["training_config"], "training_config")
    if training.get("role") != "type":
        raise ValueError("hierarchical checkpoint role must be type")
    if not isinstance(checkpoint["split_hash"], str) or not checkpoint["split_hash"]:
        raise ValueError("checkpoint split_hash is invalid")
    backbone = model_config.pop("type_backbone", None)
    from models import MULTISCALE_TYPE_VARIANT, MultiScaleHierarchicalTypeNet
    factory = (MultiScaleHierarchicalTypeNet if backbone == MULTISCALE_TYPE_VARIANT else
               create_anchor_hierarchical_type_model if backbone == ANCHOR_TYPE_VARIANT
               else create_hierarchical_type_model)
    model = factory(**model_config)
    model = _strict_load_model(
        model, _state_mapping(checkpoint["model_state"]), device
    )
    metadata = _inference_metadata(checkpoint)
    metadata["decision_config"] = decision_config
    return LoadedCheckpoint(model=model, schema=HIERARCHICAL_FIVE_CLASS_SCHEMA,
        type_names=type_names, distance_names=distance_names, distance_bins_km=distance_bins,
        preprocess_config=preprocess_config, metadata=metadata)


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
    return LoadedCheckpoint(model=model, schema=LEGACY_FIVE_CLASS_SCHEMA, type_names=type_names,
        distance_names=distance_names, distance_bins_km=distance_bins,
        preprocess_config=preprocess_config, metadata=_inference_metadata(checkpoint))


def load_model_checkpoint(
    path: os.PathLike[str] | str, device: torch.device | str = "cpu"
) -> LoadedCheckpoint:
    """Load and strictly validate one supported inference checkpoint."""

    payload = _load_payload(path, device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    checkpoint = dict(payload)
    _reject_optimizer_metadata({key: value for key, value in checkpoint.items() if key not in _STATE_FIELDS})
    schema = checkpoint.get("schema")
    if schema == FIVE_CLASS_SCHEMA:
        return _load_new(checkpoint, device)
    if schema == HIERARCHICAL_FIVE_CLASS_SCHEMA:
        return _load_hierarchical(checkpoint, device)
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
    decision_config: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save one strictly validated inference checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    hierarchical = getattr(model, "architecture", None) == HIERARCHICAL_TYPE_ARCHITECTURE
    schema = HIERARCHICAL_FIVE_CLASS_SCHEMA if hierarchical else FIVE_CLASS_SCHEMA
    checkpoint = {
        "schema": schema,
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
    if hierarchical:
        if decision_config is None:
            from evaluation import conservative_hierarchical_config, decision_config_dict

            decision_config = decision_config_dict(conservative_hierarchical_config())
        checkpoint["decision_config"] = _decision_config(decision_config)
    elif decision_config is not None:
        raise ValueError("flat checkpoint cannot store decision_config")
    try:
        torch.save(checkpoint, temporary)
        load_model_checkpoint(temporary, "cpu")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bundle_role_paths(
    root: Path, role_paths: Mapping[str, os.PathLike[str] | str]
) -> dict[str, Path]:
    if set(role_paths) != set(BUNDLE_ROLES):
        raise ValueError(
            f"bundle roles must be exactly {BUNDLE_ROLES}; "
            "missing or extra roles were provided"
        )
    resolved_root = root.resolve()
    normalized: dict[str, Path] = {}
    for role in BUNDLE_ROLES:
        path = Path(role_paths[role]).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("bundle model paths must stay below model_dir") from exc
        if not path.is_file():
            raise ValueError(f"bundle role {role} model is missing: {path}")
        normalized[role] = path
    return normalized


def save_model_bundle(
    model_dir: os.PathLike[str] | str,
    role_paths: Mapping[str, os.PathLike[str] | str],
    *,
    preprocess_config: Mapping[str, Any],
) -> Path:
    """Validate five role checkpoints and atomically write ``bundle.json``."""
    root = Path(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = _bundle_role_paths(root, role_paths)
    preprocess = _new_preprocess_config(preprocess_config)
    roles: dict[str, dict[str, str]] = {}
    split_hashes: set[str] = set()
    bundle_schema = MODEL_BUNDLE_SCHEMA
    for role, path in paths.items():
        checkpoint = load_model_checkpoint(path, "cpu")
        training = checkpoint.metadata.get("training_config")
        split_hash = checkpoint.metadata.get("split_hash")
        expected_schema = (
            HIERARCHICAL_FIVE_CLASS_SCHEMA
            if role == "type"
            and checkpoint.schema == HIERARCHICAL_FIVE_CLASS_SCHEMA
            else FIVE_CLASS_SCHEMA
        )
        if (
            checkpoint.schema != expected_schema
            or checkpoint.preprocess_config != preprocess
            or not isinstance(training, Mapping)
            or training.get("role") != role
            or not isinstance(split_hash, str)
            or not split_hash
        ):
            raise ValueError(f"bundle role {role} checkpoint is incompatible")
        if role != "type" and checkpoint.schema != FIVE_CLASS_SCHEMA:
            raise ValueError("distance bundle roles must use five_class_v1")
        if role == "type" and checkpoint.schema == HIERARCHICAL_FIVE_CLASS_SCHEMA:
            bundle_schema = HIERARCHICAL_MODEL_BUNDLE_SCHEMA
        split_hashes.add(split_hash)
        roles[role] = {
            "path": path.relative_to(root.resolve()).as_posix(),
            "sha256": model_sha256(path),
        }
    if len(split_hashes) != 1:
        raise ValueError("bundle role checkpoints use different split hashes")
    payload = {
        "schema": bundle_schema,
        "roles": roles,
        "type_names": list(TYPE_NAMES),
        "distance_names": list(DISTANCE_NAMES),
        "distance_bins_km": list(DISTANCE_BINS_KM),
        "preprocess_config": dict(preprocess),
        "split_hash": split_hashes.pop(),
    }
    destination = root / "bundle.json"
    temporary = Path(f"{destination}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_model_bundle(
    model_dir: os.PathLike[str] | str,
    device: torch.device | str = "cpu",
) -> LoadedModelBundle:
    """Load and cross-check a five-role model bundle."""
    root = Path(model_dir).resolve()
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise ValueError(f"model bundle manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("model bundle manifest is invalid") from exc
    required = {"schema", "roles", "type_names", "distance_names",
                "distance_bins_km", "preprocess_config", "split_hash"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("model bundle manifest fields are invalid")
    if payload["schema"] not in {
        MODEL_BUNDLE_SCHEMA,
        HIERARCHICAL_MODEL_BUNDLE_SCHEMA,
    }:
        raise ValueError("model bundle schema is invalid")
    _ordered_names(payload["type_names"], TYPE_NAMES, "bundle type")
    _ordered_names(
        payload["distance_names"], DISTANCE_NAMES, "bundle distance-name"
    )
    _distance_bins(payload["distance_bins_km"])
    preprocess = _new_preprocess_config(payload["preprocess_config"])
    split_hash = payload["split_hash"]
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError("model bundle split hash is invalid")
    roles = _mapping(payload["roles"], "bundle roles")
    if set(roles) != set(BUNDLE_ROLES):
        raise ValueError("model bundle roles are invalid")
    loaded: dict[str, LoadedCheckpoint] = {}
    hashes: dict[str, str] = {}
    for role in BUNDLE_ROLES:
        entry = _mapping(roles[role], f"bundle role {role}")
        if set(entry) != {"path", "sha256"}:
            raise ValueError(f"model bundle role {role} fields are invalid")
        relative = entry["path"]
        expected_hash = entry["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"model bundle role {role} path is invalid")
        path = (root / Path(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("model bundle path escapes model_dir") from exc
        if not path.is_file():
            raise ValueError(f"model bundle role {role} model is missing")
        actual_hash = model_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"model bundle role {role} hash mismatch")
        checkpoint = load_model_checkpoint(path, device)
        training = checkpoint.metadata.get("training_config")
        expected_schema = (
            HIERARCHICAL_FIVE_CLASS_SCHEMA
            if payload["schema"] == HIERARCHICAL_MODEL_BUNDLE_SCHEMA
            and role == "type"
            else FIVE_CLASS_SCHEMA
        )
        if (
            checkpoint.schema != expected_schema
            or checkpoint.preprocess_config != preprocess
            or not isinstance(training, Mapping)
            or training.get("role") != role
            or checkpoint.metadata.get("split_hash") != split_hash
        ):
            raise ValueError(f"model bundle role {role} is incompatible")
        loaded[role] = checkpoint
        hashes[role] = actual_hash
    return LoadedModelBundle(type_checkpoint=loaded["type"],
        distance_checkpoints=tuple(loaded[name] for name in DISTANCE_NAMES), hashes=hashes)


def model_sha256(path: os.PathLike[str] | str) -> str:
    """Return a streaming SHA-256 digest of a checkpoint file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
