"""Bounded, byte-preserving inference for both five-class schemas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoints import (
    FIVE_CLASS_SCHEMA,
    LEGACY_FIVE_CLASS_SCHEMA,
    LoadedCheckpoint,
    LoadedModelBundle,
    MODEL_BUNDLE_SCHEMA,
    forward_model_bundle,
    load_model_bundle,
    load_model_checkpoint,
    model_sha256,
)
from data.lig import LigOutputRegrouper, iter_lig_batches, read_file_header
from data.manifest import discover_date_inputs
from data.preprocess import (
    PreprocessConfig,
    legacy_preprocess_batch,
    preprocess_views,
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
    "output_file".split())


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


class PredictionCsvWriter:
    """Write the fixed inference audit schema incrementally."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._handle: Any = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "PredictionCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=PREDICTION_FIELDS,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self._handle.flush()
        return self

    def write(self, row: Mapping[str, object]) -> None:
        """Append one complete row and make it visible immediately."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("PredictionCsvWriter is not open")
        self._writer.writerow(
            {name: "" if row.get(name) is None else row.get(name, "")
             for name in PREDICTION_FIELDS}
        )
        self._handle.flush()

    def __exit__(self, *_args: object) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None


def _decode_prediction(
    type_logits: torch.Tensor,
    distance_logits: Sequence[torch.Tensor],
    *,
    type_only: bool = False,
) -> Prediction:
    if type_logits.ndim != 1 or type_logits.numel() != len(TYPE_NAMES):
        raise ValueError("type_logits must contain exactly five values")
    if len(distance_logits) != len(DISTANCE_NAMES):
        raise ValueError("distance_logits must contain exactly four experts")

    type_probabilities_tensor = torch.softmax(type_logits.detach().float(), dim=0)
    type_index = int(type_probabilities_tensor.argmax().item())
    type_probabilities = tuple(
        float(value) for value in type_probabilities_tensor.cpu().tolist()
    )
    type_confidence = type_probabilities[type_index]
    final_type = TYPE_NAMES[type_index]
    if type_only or type_index == 0:
        return Prediction(
            final_type=final_type,
            output_class=final_type,
            type_probabilities=type_probabilities,
            type_confidence=type_confidence,
            distance_bin=None,
            expected_distance_km=None,
            distance_confidence=None,
        )

    expert_logits = distance_logits[type_index - 1]
    if expert_logits.ndim != 1 or expert_logits.numel() != len(DISTANCE_BINS_KM):
        raise ValueError("each distance expert must contain exactly 30 values")
    distance_probabilities = torch.softmax(expert_logits.detach().float(), dim=0)
    distance_bin = int(distance_probabilities.argmax().item())
    centers = torch.arange(
        50.0,
        3050.0,
        100.0,
        dtype=distance_probabilities.dtype,
        device=distance_probabilities.device,
    )
    expected_distance_km = float(
        torch.sum(distance_probabilities * centers).item()
    )
    distance_confidence = float(distance_probabilities[distance_bin].item())
    distance_low_km = DISTANCE_BINS_KM[distance_bin]
    return Prediction(
        final_type=final_type,
        output_class=(
            f"{final_type}/"
            f"{distance_low_km:04d}-{distance_low_km + 100:04d}km"
        ),
        type_probabilities=type_probabilities,
        type_confidence=type_confidence,
        distance_bin=distance_bin,
        expected_distance_km=expected_distance_km,
        distance_confidence=distance_confidence,
    )


decode_new_prediction = _decode_prediction
decode_legacy_prediction = _decode_prediction


def _is_daylight(timestamp: datetime) -> bool:
    """Match training's local UTC+8 daylight interval for each piece."""
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


def _daylight_inputs(
    timestamps: Sequence[datetime | None],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Build known daylight values and two alternatives for missing times."""
    missing = torch.tensor(
        [timestamp is None for timestamp in timestamps],
        dtype=torch.bool,
        device=device,
    )
    primary = torch.tensor(
        [
            [0.0 if timestamp is None else float(_is_daylight(timestamp))]
            for timestamp in timestamps
        ],
        dtype=torch.float32,
        device=device,
    )
    if not bool(missing.any()):
        return primary, None, missing
    alternate = primary.clone()
    alternate[missing] = 1.0
    return primary, alternate, missing


def _average_unknown_logits(
    primary: torch.Tensor,
    alternate: torch.Tensor,
    missing: torch.Tensor,
) -> torch.Tensor:
    """Average day/night probabilities only for missing timestamps."""
    averaged = torch.logaddexp(
        primary.log_softmax(dim=-1),
        alternate.log_softmax(dim=-1),
    ) - math.log(2.0)
    return torch.where(missing.unsqueeze(1), averaged, primary)


def _new_preprocess_config(checkpoint: LoadedCheckpoint) -> PreprocessConfig:
    config = checkpoint.preprocess_config
    return PreprocessConfig(
        local_length=int(config["local_length"]),
        global_length=int(config["global_length"]),
        use_filter=bool(config.get("use_filter", True)),
        cutoff_hz=float(config.get("cutoff_hz", 120_000.0)),
        sample_rate_hz=float(config.get("sample_rate_hz", 5_000_000.0)),
    )


def predict_batch(
    checkpoint: LoadedCheckpoint,
    waveforms: np.ndarray,
    timestamps: Sequence[datetime | None],
    *,
    device: torch.device,
    type_only: bool,
) -> list[Prediction]:
    """Run one bounded batch through its schema-specific input adapter."""
    if checkpoint.schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.schema!r}")
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    if len(values) != len(timestamps):
        raise ValueError("waveforms and timestamps must be aligned")

    checkpoint.model.eval()
    with torch.inference_mode():
        if checkpoint.schema == FIVE_CLASS_SCHEMA:
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
                run_model = checkpoint.model if cascade is None else cascade
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
    if len(distance_logits) != 4 or any(
        head.ndim != 2 or len(head) != len(values) for head in distance_logits
    ):
        raise ValueError("model returned malformed distance logits")
    return [
        decoder(
            type_logits[row_index],
            [head[row_index] for head in distance_logits],
            type_only=type_only,
        )
        for row_index in range(len(values))
    ]


def predict_bundle_batch(
    bundle: LoadedModelBundle,
    waveforms: np.ndarray,
    timestamps: Sequence[datetime | None],
    *,
    device: torch.device,
    type_only: bool,
) -> list[Prediction]:
    """Run type inference first, then only the selected role checkpoints."""
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    if len(values) != len(timestamps):
        raise ValueError("waveforms and timestamps must be aligned")
    checkpoint = bundle.type_checkpoint
    local, global_view = preprocess_views(
        values, _new_preprocess_config(checkpoint)
    )
    local_tensor = torch.from_numpy(local).unsqueeze(1).to(device)
    global_tensor = torch.from_numpy(global_view).unsqueeze(1).to(device)
    daylight, alternate_daylight, missing = _daylight_inputs(
        timestamps,
        device,
    )
    with torch.inference_mode():
        type_logits, heads = forward_model_bundle(
            bundle,
            local_tensor,
            global_tensor,
            daylight,
            type_only=type_only,
        )
        if alternate_daylight is not None:
            alternate_type_logits, alternate_heads = forward_model_bundle(
                bundle,
                local_tensor,
                global_tensor,
                alternate_daylight,
                type_only=type_only,
            )
            type_logits = _average_unknown_logits(
                type_logits,
                alternate_type_logits,
                missing,
            )
            heads = tuple(
                _average_unknown_logits(
                    primary_head,
                    alternate_head,
                    missing,
                )
                for primary_head, alternate_head in zip(
                    heads,
                    alternate_heads,
                )
            )
    return [
        decode_new_prediction(
            type_logits[row],
            [head[row] for head in heads],
            type_only=type_only,
        )
        for row in range(len(values))
    ]


def _relative_lig_files(input_dir: Path) -> list[tuple[str, Path]]:
    discovered = [
        (path.relative_to(input_dir).as_posix(), path)
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".lig"
    ]
    return sorted(discovered, key=lambda item: item[0])


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
) -> Path:
    if not files:
        raise FileNotFoundError("no .lig files found in requested inputs")
    destination_root.mkdir(parents=True, exist_ok=True)
    csv_path = destination_root / "predictions.csv"
    regrouper = LigOutputRegrouper(destination_root)
    invalid_timestamp_count = 0
    try:
        with PredictionCsvWriter(csv_path) as writer:
            for relative_source, source_path in files:
                source_header = read_file_header(source_path)
                for start, waveforms, timestamps, raw_pieces in iter_lig_batches(
                    source_path,
                    batch_size,
                    allow_invalid_timestamps=True,
                ):
                    invalid_timestamp_count += sum(
                        timestamp is None for timestamp in timestamps
                    )
                    predictions = predict(waveforms, timestamps)
                    for offset, (raw_piece, prediction, timestamp) in enumerate(
                        zip(raw_pieces, predictions, timestamps)
                    ):
                        output_file = regrouper.add(
                            prediction.output_class,
                            source_header,
                            raw_piece,
                            timestamp,
                        )
                        writer.write(_csv_row(
                            relative_source,
                            start + offset,
                            prediction,
                            checkpoint_schema=checkpoint_schema,
                            checkpoint_sha256=checkpoint_hash,
                            model_hashes=model_hashes,
                            output_file=output_file,
                        ))
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
    device: str | torch.device = "auto",
) -> Path:
    """Classify a directory recursively with bounded byte-exact regrouping."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    source_root, destination_root, checkpoint_path = _validated_paths(
        input_dir, output_dir, model
    )
    files = _relative_lig_files(source_root)
    if not files:
        raise FileNotFoundError(f"no .lig files found below {source_root}")

    runtime_device = _resolve_device(device)
    checkpoint = load_model_checkpoint(checkpoint_path, runtime_device)
    if checkpoint.schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported checkpoint schema: {checkpoint.schema!r}")
    checkpoint.model.eval()
    checkpoint_hash = model_sha256(checkpoint_path)
    return _classify_files(
        files,
        destination_root,
        batch_size=batch_size,
        predict=lambda waveforms, timestamps: predict_batch(
            checkpoint,
            waveforms,
            timestamps,
            device=runtime_device,
            type_only=type_only,
        ),
        checkpoint_schema=checkpoint.schema,
        checkpoint_hash=checkpoint_hash,
        model_hashes=None,
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
    device: str | torch.device = "auto",
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
    date_paths = discover_date_inputs(root, start_date, end_date)
    files = sorted(
        (
            source.relative_to(root).as_posix(),
            source,
        )
        for date_path in date_paths
        for source in date_path.rglob("*")
        if source.is_file() and source.suffix.casefold() == ".lig"
    )
    runtime_device = _resolve_device(device)
    bundle = load_model_bundle(bundle_root, runtime_device)
    combined_hash = json.dumps(bundle.hashes, sort_keys=True)
    return _classify_files(
        files,
        destination,
        batch_size=batch_size,
        predict=lambda waveforms, timestamps: predict_bundle_batch(
            bundle,
            waveforms,
            timestamps,
            device=runtime_device,
            type_only=type_only,
        ),
        checkpoint_schema=MODEL_BUNDLE_SCHEMA,
        checkpoint_hash=None,
        model_hashes=combined_hash,
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
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the inference CLI and return the generated CSV path."""
    args = build_arg_parser().parse_args(argv)
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
            args.input_root,
            args.start_date,
            args.end_date,
            args.output_dir,
            args.model_dir,
            batch_size=args.batch_size,
            type_only=args.type_only,
            device=args.device,
        )
    if args.input_dir is None or args.model is None:
        raise ValueError("single mode requires --input_dir and --model")
    return classify_directory(
        args.input_dir,
        args.output_dir,
        args.model,
        batch_size=args.batch_size,
        type_only=args.type_only,
        device=args.device,
    )


if __name__ == "__main__":
    main()
