"""Bounded, byte-preserving inference for both five-class schemas."""

from __future__ import annotations

import argparse
import csv, json
import os
import warnings
from threading import RLock
from time import perf_counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoints import (
    DECISION_CONFIG_SCHEMA,
    FIVE_CLASS_SCHEMA,
    LEGACY_FIVE_CLASS_SCHEMA,
    LoadedCheckpoint,
    LoadedModelBundle,
    GuardedTypeVerifier, load_type_verifier,
    MODEL_BUNDLE_SCHEMA,
    checkpoint_preprocess_config as _new_preprocess_config,
    decision_config_sha256 as _decision_config_sha256,
    load_decision_config,
    load_model_bundle,
    load_model_checkpoint,
    model_sha256,
    override_decision_config as _with_decision_config,
)
from data.lig import (LigInferenceJournal, LigOutputRegrouper, iter_inference_lig_batches)
from data.manifest import discover_date_inputs, discover_lig_files, readable_lig_files
from data.preprocess import (
    StreamContextBank, TemporalContextConfig, TemporalContextState,
    average_unknown_logits as _average_unknown_logits,
    daylight_inputs as _daylight_inputs, legacy_preprocess_batch, preprocess_views,
)


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS_KM = tuple(range(0, 3000, 100))
SUPPORTED_SCHEMAS = frozenset({FIVE_CLASS_SCHEMA, LEGACY_FIVE_CLASS_SCHEMA})

PREDICTION_FIELDS = tuple(
    "source_path piece_index piece_key final_type prob_IC prob_NCG "
    "prob_NNBE prob_PCG prob_PNBE type_confidence distance_bin "
    "distance_low_km distance_high_km expected_distance_km "
    "distance_confidence checkpoint_schema model_sha256 model_hashes "
    "decision_config_sha256 output_file".split()
)


@dataclass(frozen=True)
class Prediction:
    """Schema-independent prediction for one waveform piece."""

    final_type: str
    output_class: str
    type_probabilities: tuple[float, float, float, float, float]
    type_confidence: float
    distance_bin: int | None
    expected_distance_km: float | None
    distance_confidence: float | None


def _decode_prediction(
    type_logits: torch.Tensor,
    distance_logits: Sequence[torch.Tensor],
    *,
    type_only: bool = False,
    forced_type_index: int | None = None,
) -> Prediction:
    if type_logits.ndim != 1 or type_logits.numel() != len(TYPE_NAMES):
        raise ValueError("type_logits must contain exactly five values")
    if len(distance_logits) != len(DISTANCE_NAMES):
        raise ValueError("distance_logits must contain exactly four experts")

    type_probabilities_tensor = torch.softmax(type_logits.detach().float(), dim=0)
    type_index = (
        int(type_probabilities_tensor.argmax().item())
        if forced_type_index is None
        else int(forced_type_index)
    )
    if not 0 <= type_index < len(TYPE_NAMES):
        raise ValueError("forced_type_index is outside the type label range")
    type_probabilities = tuple(
        float(value) for value in type_probabilities_tensor.cpu().tolist()
    )
    type_confidence = type_probabilities[type_index]
    final_type = TYPE_NAMES[type_index]
    if type_only or type_index == 0:
        return Prediction(
            final_type=final_type, output_class=final_type,
            type_probabilities=type_probabilities, type_confidence=type_confidence,
            distance_bin=None, expected_distance_km=None, distance_confidence=None,
        )

    expert_logits = distance_logits[type_index - 1]
    if expert_logits.ndim != 1 or expert_logits.numel() != len(DISTANCE_BINS_KM):
        raise ValueError("each distance expert must contain exactly 30 values")
    distance_probabilities = torch.softmax(expert_logits.detach().float(), dim=0)
    distance_bin = int(distance_probabilities.argmax().item())
    centers = torch.arange(
        50.0, 3050.0, 100.0, dtype=distance_probabilities.dtype,
        device=distance_probabilities.device,
    )
    expected_distance_km = float(torch.sum(distance_probabilities * centers).item())
    distance_confidence = float(distance_probabilities[distance_bin].item())
    distance_low_km = DISTANCE_BINS_KM[distance_bin]
    return Prediction(
        final_type=final_type,
        output_class=f"{final_type}/{distance_low_km:04d}-{distance_low_km + 100:04d}km",
        type_probabilities=type_probabilities,
        type_confidence=type_confidence,
        distance_bin=distance_bin,
        expected_distance_km=expected_distance_km,
        distance_confidence=distance_confidence,
    )


decode_new_prediction = decode_legacy_prediction = _decode_prediction


def _decode_batch(decoder: Any, type_logits: torch.Tensor, heads: Sequence[torch.Tensor], *, type_only: bool,
                  forced_types: torch.Tensor | None) -> list[Prediction]:
    return [decoder(type_logits[row], [head[row] for head in heads],
                    type_only=type_only, forced_type_index=None if forced_types is None
                    else int(forced_types[row].item())) for row in range(len(type_logits))]


def _hierarchical_type_inference(
    checkpoint: LoadedCheckpoint,
    local: torch.Tensor,
    global_view: torch.Tensor,
    daylight: torch.Tensor,
    alternate_daylight: torch.Tensor | None,
    missing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt a validated checkpoint to shared hierarchical inference."""
    from evaluation import infer_hierarchical_types

    raw_config = checkpoint.metadata.get("decision_config")
    if not isinstance(raw_config, dict):
        raise ValueError("hierarchical checkpoint has no decision_config")
    return infer_hierarchical_types(checkpoint.model, raw_config, local, global_view,
                                    daylight, alternate_daylight, missing)


def predict_batch(
    checkpoint: LoadedCheckpoint,
    waveforms: np.ndarray,
    timestamps: Sequence[datetime | None],
    *,
    device: torch.device,
    type_only: bool,
    direct_type: bool = False,
    temporal_context: TemporalContextState | None = None,
    verifier: GuardedTypeVerifier | None = None,
) -> list[Prediction]:
    """Run one bounded batch through its schema-specific input adapter."""
    hierarchical_schema = "hierarchical_five_class_v2"
    if verifier is not None and (direct_type or temporal_context is not None or checkpoint.schema != hierarchical_schema):
        raise ValueError("type verifier requires hierarchical inference without direct_type or temporal context")
    type_inference = _hierarchical_type_inference if verifier is None else verifier.infer
    if (
        checkpoint.schema not in SUPPORTED_SCHEMAS
        and checkpoint.schema != hierarchical_schema
    ):
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.schema!r}")
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    if len(values) != len(timestamps):
        raise ValueError("waveforms and timestamps must be aligned")

    checkpoint.model.eval()
    forced_type_indices: torch.Tensor | None = None
    with torch.inference_mode():
        if checkpoint.schema == hierarchical_schema:
            if not type_only:
                raise ValueError(
                    "a hierarchical type checkpoint requires --type_only; "
                    "use --model_dir for type and distance inference"
                )
            local, global_view = preprocess_views(
                values, _new_preprocess_config(checkpoint)
            )
            local_tensor = torch.from_numpy(local).unsqueeze(1).to(device)
            global_tensor = (
                torch.from_numpy(global_view).unsqueeze(1).to(device)
            )
            daylight_tensor, alternate_daylight, missing = _daylight_inputs(
                timestamps,
                device,
            )
            type_logits, forced_type_indices = type_inference(
                checkpoint,
                local_tensor,
                global_tensor,
                daylight_tensor,
                alternate_daylight,
                missing,
            )
            if direct_type:
                forced_type_indices = None
            distance_logits = tuple(
                type_logits.new_zeros(
                    (len(values), len(DISTANCE_BINS_KM))
                )
                for _ in DISTANCE_NAMES
            )
            decoder = decode_new_prediction
        elif checkpoint.schema == FIVE_CLASS_SCHEMA:
            local, global_view = preprocess_views(
                values, _new_preprocess_config(checkpoint)
            )
            local_tensor = torch.from_numpy(local).unsqueeze(1).to(device)
            global_tensor = torch.from_numpy(global_view).unsqueeze(1).to(device)
            daylight_tensor, alternate_daylight, missing = _daylight_inputs(
                timestamps,
                device,
            )
            if type_only:
                type_logits, _ = checkpoint.model.forward_type(
                    local_tensor, global_tensor, daylight_tensor
                )
                if alternate_daylight is not None:
                    alternate_logits, _ = checkpoint.model.forward_type(
                        local_tensor,
                        global_tensor,
                        alternate_daylight,
                    )
                    type_logits = _average_unknown_logits(
                        type_logits,
                        alternate_logits,
                        missing,
                    )
                distance_logits = tuple(
                    type_logits.new_zeros((len(values), len(DISTANCE_BINS_KM)))
                    for _ in DISTANCE_NAMES
                )
            else:
                cascade = getattr(checkpoint.model, "predict_cascade", None)
                run_model = checkpoint.model if cascade is None or temporal_context is not None else cascade
                output = run_model(
                    local_tensor,
                    global_tensor,
                    daylight_tensor,
                )
                type_logits = output.type_logits
                distance_logits = output.distance_logits
                if alternate_daylight is not None:
                    alternate_output = run_model(
                        local_tensor,
                        global_tensor,
                        alternate_daylight,
                    )
                    type_logits = _average_unknown_logits(
                        type_logits,
                        alternate_output.type_logits,
                        missing,
                    )
                    distance_logits = tuple(
                        _average_unknown_logits(
                            primary_head,
                            alternate_head,
                            missing,
                        )
                        for primary_head, alternate_head in zip(
                            distance_logits,
                            alternate_output.distance_logits,
                        )
                    )
            decoder = decode_new_prediction
        else:
            legacy_values = legacy_preprocess_batch(values)
            legacy_tensor = torch.from_numpy(legacy_values).unsqueeze(1).to(device)
            type_logits, distance_logits = checkpoint.model(legacy_tensor)
            decoder = decode_legacy_prediction

    if type_logits.ndim != 2 or len(type_logits) != len(values):
        raise ValueError("model returned malformed type logits")
    if len(distance_logits) != 4 or any(head.ndim != 2 or len(head) != len(values)
                                        for head in distance_logits):
        raise ValueError("model returned malformed distance logits")
    if temporal_context is not None:
        base_types = type_logits.argmax(dim=1) if forced_type_indices is None else forced_type_indices
        forced_type_indices = temporal_context.adjust(type_logits, base_types)
    return _decode_batch(decoder, type_logits, distance_logits,
                         type_only=type_only, forced_types=forced_type_indices)


def _route_distance_heads(
    bundle: LoadedModelBundle,
    local: torch.Tensor,
    global_view: torch.Tensor,
    daylight: torch.Tensor,
    alternate_daylight: torch.Tensor | None,
    missing: torch.Tensor,
    final_type: torch.Tensor,
    *,
    type_only: bool,
) -> tuple[torch.Tensor, ...]:
    """Run only the distance model selected by the stable type decision."""
    heads = [
        local.new_zeros((len(local), len(DISTANCE_BINS_KM)))
        for _ in DISTANCE_NAMES
    ]
    if type_only:
        return tuple(heads)
    for type_index, checkpoint in enumerate(
        bundle.distance_checkpoints, start=1
    ):
        selected = final_type.eq(type_index)
        if not torch.any(selected):
            continue
        checkpoint.model.eval()
        logits, _ = checkpoint.model.forward_distance_type(
            local[selected],
            global_view[selected],
            daylight[selected],
            expert_index=type_index - 1,
        )
        if alternate_daylight is not None:
            alternate_logits, _ = checkpoint.model.forward_distance_type(
                local[selected],
                global_view[selected],
                alternate_daylight[selected],
                expert_index=type_index - 1,
            )
            logits = _average_unknown_logits(
                logits,
                alternate_logits,
                missing[selected],
            )
        expected = (int(selected.sum().item()), len(DISTANCE_BINS_KM))
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"{TYPE_NAMES[type_index]} model returned malformed logits"
            )
        heads[type_index - 1][selected] = logits
    return tuple(heads)


def predict_bundle_batch(
    bundle: LoadedModelBundle,
    waveforms: np.ndarray,
    timestamps: Sequence[datetime | None],
    *,
    device: torch.device,
    type_only: bool,
    temporal_context: TemporalContextState | None = None,
    verifier: GuardedTypeVerifier | None = None,
) -> list[Prediction]:
    """Run type inference first, then only the selected role checkpoints."""
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    if len(values) != len(timestamps):
        raise ValueError("waveforms and timestamps must be aligned")
    checkpoint = bundle.type_checkpoint
    if verifier is not None and (temporal_context is not None or checkpoint.schema != "hierarchical_five_class_v2"):
        raise ValueError("type verifier requires hierarchical inference without temporal context")
    type_inference = _hierarchical_type_inference if verifier is None else verifier.infer
    checkpoint.model.eval()
    local, global_view = preprocess_views(
        values, _new_preprocess_config(checkpoint)
    )
    local_tensor = torch.from_numpy(local).unsqueeze(1).to(device)
    global_tensor = torch.from_numpy(global_view).unsqueeze(1).to(device)
    daylight, alternate_daylight, missing = _daylight_inputs(
        timestamps,
        device,
    )
    forced_type_indices: torch.Tensor | None = None
    with torch.inference_mode():
        if checkpoint.schema == "hierarchical_five_class_v2":
            type_logits, forced_type_indices = type_inference(
                checkpoint,
                local_tensor,
                global_tensor,
                daylight,
                alternate_daylight,
                missing,
            )
        else:
            type_logits, _ = checkpoint.model.forward_type(
                local_tensor, global_tensor, daylight,
            )
            if alternate_daylight is not None:
                alternate_logits, _ = checkpoint.model.forward_type(
                    local_tensor, global_tensor, alternate_daylight,
                )
                type_logits = _average_unknown_logits(
                    type_logits, alternate_logits, missing,
                )
            forced_type_indices = type_logits.argmax(dim=1)
        if type_logits.shape != (len(values), len(TYPE_NAMES)):
            raise ValueError("model returned malformed type logits")
        if temporal_context is not None:
            forced_type_indices = temporal_context.adjust(type_logits, forced_type_indices)
        heads = _route_distance_heads(bundle, local_tensor, global_tensor, daylight,
                                      alternate_daylight, missing, forced_type_indices,
                                      type_only=type_only)
    return _decode_batch(decode_new_prediction, type_logits, heads,
                         type_only=type_only, forced_types=forced_type_indices)


@dataclass(frozen=True)
class StreamPrediction:
    """One completed waveform result plus causal-context and latency evidence."""
    prediction: Prediction
    stream_id: str
    timestamp: datetime | None
    context_status: str
    temporal_promoted: bool
    elapsed_ms: float


class StreamingClassifier:
    """Load a bundle once and classify complete first-channel waveforms on arrival."""

    def __init__(self, model_dir: str | os.PathLike[str], *, device: str = "cpu",
                 temporal_config: TemporalContextConfig | None = None,
                 max_gap_seconds: float = 60.0, max_streams: int = 64,
                 type_only: bool = False, type_verifier_config: str | os.PathLike[str] | None = None):
        self.context = StreamContextBank(temporal_config, max_gap_seconds=max_gap_seconds,
                                         max_streams=max_streams)
        self.device = _resolve_device(device)
        self.bundle = load_model_bundle(model_dir, self.device)
        self.verifier = _configured_verifier(type_verifier_config, self.bundle.type_checkpoint,
            self.bundle.hashes["type"], self.device, temporal_config=temporal_config)
        self._model_hashes = dict(self.bundle.hashes)
        if self.verifier is not None:
            self._model_hashes["type_verifier_config"] = self.verifier.signature
        self.type_only, self._lock = type_only, RLock()

    def predict_piece(self, waveform: np.ndarray, *, stream_id: str,
                      timestamp: datetime | None) -> StreamPrediction:
        """Predict one complete 16,000-sample waveform; naive timestamps are UTC."""
        started = perf_counter()
        values = np.array(waveform, dtype=np.float32, copy=True)
        if values.shape != (16000,) or not np.isfinite(values).all():
            raise ValueError("stream waveform must contain 16000 finite first-channel samples")
        timestamp = self.context.utc(timestamp)
        with self._lock:
            state, reason = self.context.prepare(stream_id, timestamp)
            before = 0 if state is None else sum(state.promoted_counts)
            result = predict_bundle_batch(self.bundle, values[None], [timestamp],
                device=self.device, type_only=self.type_only, temporal_context=state, verifier=self.verifier)[0]
            promoted = state is not None and sum(state.promoted_counts) > before
            self.context.commit(stream_id, timestamp, state, reason)
            return StreamPrediction(result, stream_id, timestamp, reason, promoted,
                                    (perf_counter() - started) * 1000)

    def snapshot(self) -> dict:
        """Return model-bound state; the caller chooses durable storage."""
        with self._lock:
            return {"model_hashes": dict(self._model_hashes), "type_only": self.type_only,
                    "context": self.context.snapshot()}

    def restore(self, payload: dict) -> None:
        """Restore only state produced by the same models and context settings."""
        with self._lock:
            if (not isinstance(payload, dict) or set(payload) != {"model_hashes", "type_only", "context"}
                    or payload["model_hashes"] != self._model_hashes
                    or payload["type_only"] != self.type_only):
                raise ValueError("stream model configuration mismatch")
            self.context.restore(payload["context"])


def _configured_verifier(path, checkpoint, checkpoint_hash, device, *,
                         decision_config=None, direct_type=False, temporal_config=None):
    if path is not None and (decision_config is not None or direct_type or temporal_config is not None):
        raise ValueError("type_verifier_config cannot combine with decision_config, direct_type or temporal context")
    return load_type_verifier(path, checkpoint, checkpoint_hash, device)


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath([parent, child]) == os.fspath(parent)
    except ValueError:
        return False


def _validated_paths(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    model_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    source_root = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(model_path).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"input_dir is not a directory: {source_root}")
    if not checkpoint_path.is_file():
        raise ValueError(f"model is not a file: {checkpoint_path}")
    if destination_root.exists() and not destination_root.is_dir():
        raise ValueError(f"output_dir is not a directory: {destination_root}")
    if _path_contains(destination_root, source_root):
        raise ValueError("output_dir must not contain input_dir")
    if _path_contains(source_root, destination_root):
        raise ValueError("input_dir must not contain output_dir")
    return source_root, destination_root, checkpoint_path


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device).casefold() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return resolved


def _csv_row(
    relative_source: str,
    piece_index: int,
    prediction: Prediction,
    *,
    checkpoint_schema: str,
    checkpoint_sha256: str | None,
    model_hashes: str | None,
    decision_config_sha256: str | None,
    output_file: str,
) -> dict[str, object]:
    distance_low_km = (
        None if prediction.distance_bin is None else prediction.distance_bin * 100
    )
    distance_high_km = (
        None if distance_low_km is None else distance_low_km + 100
    )
    row: dict[str, object] = {
        "source_path": relative_source,
        "piece_index": piece_index,
        "piece_key": f"{relative_source}#{piece_index}",
        "final_type": prediction.final_type,
        "type_confidence": prediction.type_confidence,
        "distance_bin": prediction.distance_bin,
        "distance_low_km": distance_low_km,
        "distance_high_km": distance_high_km,
        "expected_distance_km": prediction.expected_distance_km,
        "distance_confidence": prediction.distance_confidence,
        "checkpoint_schema": checkpoint_schema,
        "model_sha256": checkpoint_sha256,
        "model_hashes": model_hashes,
        "decision_config_sha256": decision_config_sha256,
        "output_file": output_file,
    }
    row.update({
        f"prob_{type_name}": probability
        for type_name, probability in zip(
            TYPE_NAMES, prediction.type_probabilities
        )
    })
    return row


def _classify_files(
    files: Sequence[tuple[str, Path]],
    destination_root: Path,
    *,
    batch_size: int,
    predict: Any,
    checkpoint_schema: str,
    checkpoint_hash: str | None,
    model_hashes: str | None = None,
    decision_config_hash: str | None = None,
    output_types: Sequence[str] | None = None,
    run_mode: str = "inference",
    resume: bool = False,
    skip_io_errors: bool = False,
    temporal_context: TemporalContextState | None = None,
) -> Path:
    if not files:
        raise FileNotFoundError("no .lig files found in requested inputs")
    selected = None if output_types is None else frozenset(output_types)
    if selected is not None and (not selected or not selected <= frozenset(TYPE_NAMES)):
        raise ValueError(f"output_types must contain only {TYPE_NAMES}")
    destination_root.mkdir(parents=True, exist_ok=True)
    csv_path = destination_root / "predictions.csv"
    regrouper = LigOutputRegrouper(destination_root)
    invalid_timestamp_count = 0
    constraints = dict(
        checkpoint_schema=checkpoint_schema, model_sha256=checkpoint_hash or "",
        model_hashes=model_hashes or "", decision_config_sha256=decision_config_hash or "",
    )
    readable_files, input_sizes = readable_lig_files(
        files, skip_io_errors=skip_io_errors)
    if not readable_files:
        raise FileNotFoundError("no readable .lig files found in requested inputs")

    signature = {
        **constraints,
        "inputs": input_sizes,
        "output_types": None if selected is None else sorted(selected),
        "run_mode": run_mode,
    }
    if temporal_context is not None: signature["temporal_context"] = temporal_context.config.signature()
    if resume and temporal_context is not None and csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            temporal_context.seed(
                (TYPE_NAMES.index(row["final_type"]), float(row["type_confidence"]))
                for row in csv.DictReader(handle)
            )
    try:
        with LigInferenceJournal(
            destination_root, PREDICTION_FIELDS, signature, constraints,
            selected, resume=resume,
        ) as writer:
            for relative_source, source_path in readable_files:
                if writer.source_is_complete(relative_source):
                    continue
                writer.begin_source(relative_source)
                for source_header, start, waveforms, timestamps, raw_pieces in iter_inference_lig_batches(
                    source_path,
                    batch_size,
                    allow_invalid_timestamps=True,
                    skip_io_errors=skip_io_errors,
                ):
                    pending = [offset for offset in range(len(waveforms)) if
                               f"{relative_source}#{start + offset}" not in writer.completed_keys]
                    invalid_timestamp_count += sum(
                        timestamps[offset] is None for offset in pending
                    )
                    if not pending:
                        continue
                    predictions = predict(waveforms[pending], [timestamps[offset] for offset in pending])
                    for offset, prediction in zip(pending, predictions):
                        piece_index = start + offset
                        raw_piece, timestamp = raw_pieces[offset], timestamps[offset]
                        output_file = ""
                        if not selected or prediction.final_type in selected:
                            output_file = regrouper.add(
                                prediction.output_class,
                                source_header,
                                raw_piece,
                                timestamp,
                            )
                        writer.write(_csv_row(
                            relative_source,
                            piece_index,
                            prediction,
                            checkpoint_schema=checkpoint_schema,
                            checkpoint_sha256=checkpoint_hash,
                            model_hashes=model_hashes,
                            decision_config_sha256=decision_config_hash,
                            output_file=output_file,
                        ))
                writer.complete_source(relative_source)
            regrouper.flush_all()
            writer.finish()
    finally:
        regrouper.flush_all()
    if invalid_timestamp_count:
        warnings.warn(
            f"{invalid_timestamp_count} waveform records have invalid "
            "timestamps; predictions averaged day/night probabilities and "
            "unknown output groups use GZ_unknown names",
            RuntimeWarning,
            stacklevel=2,
        )
    return csv_path


def classify_directory(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    model: str | os.PathLike[str],
    *,
    batch_size: int = 256,
    type_only: bool = False,
    direct_type: bool = False,
    decision_config: str | os.PathLike[str] | None = None,
    output_types: Sequence[str] | None = None,
    resume: bool = False,
    skip_io_errors: bool = False,
    device: str | torch.device = "auto",
    temporal_config: TemporalContextConfig | None = None,
    type_verifier_config: str | os.PathLike[str] | None = None,
) -> Path:
    """Classify a directory recursively with bounded byte-exact regrouping."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    source_root, destination_root, checkpoint_path = _validated_paths(
        input_dir, output_dir, model
    )
    files = discover_lig_files(source_root, skip_io_errors=skip_io_errors)
    if not files:
        raise FileNotFoundError(f"no .lig files found below {source_root}")

    runtime_device = _resolve_device(device)
    checkpoint = load_model_checkpoint(checkpoint_path, runtime_device)
    if direct_type and decision_config is not None:
        raise ValueError(
            "--decision_config cannot be combined with --direct_type"
        )
    override = (
        None if decision_config is None else load_decision_config(decision_config)
    )
    override_hash = _decision_config_sha256(override)
    checkpoint = _with_decision_config(checkpoint, override)
    if (
        checkpoint.schema not in SUPPORTED_SCHEMAS
        and checkpoint.schema != "hierarchical_five_class_v2"
    ):
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.schema!r}")
    checkpoint.model.eval()
    checkpoint_hash = model_sha256(checkpoint_path)
    verifier = _configured_verifier(type_verifier_config, checkpoint, checkpoint_hash, runtime_device,
        decision_config=decision_config, direct_type=direct_type, temporal_config=temporal_config)
    temporal_context = None if temporal_config is None else TemporalContextState(temporal_config)
    return _classify_files(
        files, destination_root, batch_size=batch_size,
        predict=lambda waveforms, timestamps: predict_batch(
            checkpoint, waveforms, timestamps, device=runtime_device,
            type_only=type_only, direct_type=direct_type,
            temporal_context=temporal_context, verifier=verifier,
        ),
        checkpoint_schema=checkpoint.schema, checkpoint_hash=checkpoint_hash,
        model_hashes=None if verifier is None else json.dumps({"type_verifier": verifier.sha256}, sort_keys=True),
        decision_config_hash=override_hash if verifier is None else verifier.signature,
        output_types=output_types, run_mode=f"single:{type_only}:{direct_type}",
        resume=resume, skip_io_errors=skip_io_errors, temporal_context=temporal_context,
    )


def classify_date_range(
    input_root: str | os.PathLike[str],
    start_date: str,
    end_date: str,
    output_dir: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    *,
    batch_size: int = 256,
    type_only: bool = False,
    decision_config: str | os.PathLike[str] | None = None,
    output_types: Sequence[str] | None = None,
    resume: bool = False,
    skip_io_errors: bool = False,
    device: str | torch.device = "auto",
    prefix: str = "GZ_",
    temporal_config: TemporalContextConfig | None = None,
    type_verifier_config: str | os.PathLike[str] | None = None,
) -> Path:
    """Classify exact inclusive date directories with a five-role bundle."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    root = Path(input_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    bundle_root = Path(model_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"output_dir is not a directory: {destination}")
    if _path_contains(destination, root) or _path_contains(root, destination):
        raise ValueError("input_root and output_dir must not contain each other")
    date_paths = discover_date_inputs(root, start_date, end_date, prefix=prefix)
    files = discover_lig_files(
        root, search_roots=date_paths, skip_io_errors=skip_io_errors)
    runtime_device = _resolve_device(device)
    bundle = load_model_bundle(bundle_root, runtime_device)
    override = (
        None if decision_config is None else load_decision_config(decision_config)
    )
    override_hash = _decision_config_sha256(override)
    bundle = replace(bundle, type_checkpoint=_with_decision_config(bundle.type_checkpoint, override))
    verifier = _configured_verifier(type_verifier_config, bundle.type_checkpoint, bundle.hashes["type"],
        runtime_device, decision_config=decision_config, temporal_config=temporal_config)
    hashes = dict(bundle.hashes)
    if verifier is not None:
        hashes["type_verifier"] = verifier.sha256
    combined_hash = json.dumps(hashes, sort_keys=True)
    temporal_context = None if temporal_config is None else TemporalContextState(temporal_config)
    return _classify_files(
        files, destination, batch_size=batch_size,
        predict=lambda waveforms, timestamps: predict_bundle_batch(
            bundle, waveforms, timestamps, device=runtime_device, type_only=type_only,
            temporal_context=temporal_context, verifier=verifier,
        ),
        checkpoint_schema=MODEL_BUNDLE_SCHEMA, checkpoint_hash=None, model_hashes=combined_hash,
        decision_config_hash=override_hash if verifier is None else verifier.signature,
        output_types=output_types, run_mode=f"bundle:{type_only}",
        resume=resume, skip_io_errors=skip_io_errors, temporal_context=temporal_context,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build single-directory and date-range inference modes."""
    parser = argparse.ArgumentParser(description="Classify LIG pieces.")
    parser.add_argument("--input_dir")
    parser.add_argument("--input_root")
    parser.add_argument("--start_date")
    parser.add_argument("--end_date")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--model_dir")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--type_only", action="store_true")
    parser.add_argument("--direct_type", action="store_true")
    parser.add_argument("--decision_config")
    parser.add_argument("--type_verifier_config", help="Opt-in checkpoint-bound guarded verification JSON.")
    parser.add_argument("--output_type", action="append", type=str.upper, choices=TYPE_NAMES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_io_errors", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefix", default="GZ_")
    parser.add_argument("--temporal_context", action="store_true")
    parser.add_argument("--temporal_history", type=int, default=256)
    parser.add_argument("--temporal_trigger", type=int, default=4)
    parser.add_argument("--temporal_anchor_confidence", type=float, default=0.98)
    parser.add_argument("--temporal_min_probability", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the inference CLI and return the generated CSV path."""
    args = build_arg_parser().parse_args(argv)
    temporal_config = TemporalContextConfig(
        args.temporal_history, args.temporal_trigger,
        args.temporal_anchor_confidence, args.temporal_min_probability,
    ) if args.temporal_context else None
    options = dict(batch_size=args.batch_size, type_only=args.type_only,
                   decision_config=args.decision_config, output_types=args.output_type,
                   resume=args.resume, skip_io_errors=args.skip_io_errors,
                   device=args.device, temporal_config=temporal_config,
                   type_verifier_config=args.type_verifier_config)
    single = args.input_dir is not None or args.model is not None
    range_values = (args.input_root, args.start_date, args.end_date, args.model_dir)
    ranged = any(value is not None for value in range_values)
    if single == ranged:
        raise ValueError(
            "choose exactly one mode: --input_dir/--model or "
            "--input_root/--start_date/--end_date/--model_dir"
        )
    if ranged:
        if any(value is None for value in range_values):
            raise ValueError("date-range mode requires all four range options")
        return classify_date_range(
            args.input_root, args.start_date, args.end_date, args.output_dir,
            args.model_dir, prefix=args.prefix, **options,
        )
    if args.input_dir is None or args.model is None:
        raise ValueError("single mode requires --input_dir and --model")
    return classify_directory(
        args.input_dir, args.output_dir, args.model,
        direct_type=args.direct_type, **options,
    )


if __name__ == "__main__":
    main()
