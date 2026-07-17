"""Bounded, byte-preserving inference for both five-class schemas."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from checkpoints import (
    FIVE_CLASS_SCHEMA,
    LEGACY_FIVE_CLASS_SCHEMA,
    LoadedCheckpoint,
    load_model_checkpoint,
    model_sha256,
)
from data.lig import (
    FILE_HEADER_BYTES,
    MAX_PIECES_PER_FILE,
    PIECE_BYTES,
    LigFileIndex,
    LigFormatError,
    read_file_header,
    write_lig_file,
)
from data.preprocess import PreprocessConfig, preprocess_views


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS_KM = tuple(range(0, 3000, 100))
SUPPORTED_SCHEMAS = frozenset({FIVE_CLASS_SCHEMA, LEGACY_FIVE_CLASS_SCHEMA})

PREDICTION_FIELDS = (
    "source_path",
    "piece_index",
    "piece_key",
    "final_type",
    "prob_IC",
    "prob_NCG",
    "prob_NNBE",
    "prob_PCG",
    "prob_PNBE",
    "type_confidence",
    "distance_bin",
    "distance_low_km",
    "distance_high_km",
    "expected_distance_km",
    "distance_confidence",
    "checkpoint_schema",
    "model_sha256",
    "output_file",
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
    type_only: bool,
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
    final_type = TYPE_NAMES[type_index]
    type_confidence = type_probabilities[type_index]
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
            f"{final_type}_{distance_low_km}-{distance_low_km + 100}km"
        ),
        type_probabilities=type_probabilities,
        type_confidence=type_confidence,
        distance_bin=distance_bin,
        expected_distance_km=expected_distance_km,
        distance_confidence=distance_confidence,
    )


def decode_new_prediction(
    type_logits: torch.Tensor,
    distance_logits: Sequence[torch.Tensor],
    *,
    type_only: bool = False,
) -> Prediction:
    """Decode direct five-class argmax and its routed new-model expert."""
    return _decode_prediction(
        type_logits, distance_logits, type_only=type_only
    )


def decode_legacy_prediction(
    type_logits: torch.Tensor,
    distance_logits: Sequence[torch.Tensor],
    *,
    type_only: bool = False,
) -> Prediction:
    """Decode the retained legacy model into the normalized contract."""
    return _decode_prediction(
        type_logits, distance_logits, type_only=type_only
    )


def _legacy_preprocess_batch(waveforms: np.ndarray) -> np.ndarray:
    """Reproduce the retained model's filter/crop/min-max inference view."""
    pieces = np.asarray(waveforms, dtype=np.float32)
    if pieces.ndim != 2 or pieces.shape[1] == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(
            2,
            120_000.0 / (5_000_000.0 / 2.0),
            btype="low",
            output="sos",
        )
        pieces = sosfiltfilt(sos, pieces, axis=-1).astype(np.float32)
    except ImportError:  # pragma: no cover - SciPy is a runtime dependency
        pass

    target_length = 8000
    before = 2000
    source_length = pieces.shape[1]
    peaks = np.argmax(pieces, axis=1).astype(np.int64)
    begins = peaks - before
    ends = peaks + 6000
    before_source = begins < 0
    begins[before_source] = 0
    ends[before_source] = target_length
    after_source = ends > source_length
    ends[after_source] = source_length
    begins[after_source] = ends[after_source] - target_length

    cropped = np.empty((len(pieces), target_length), dtype=np.float32)
    for row_index in range(len(pieces)):
        segment = pieces[row_index, begins[row_index]:ends[row_index]]
        length = len(segment)
        cropped[row_index, :length] = segment[:target_length]
        if length < target_length:
            cropped[row_index, length:] = 0.0

    minimum = cropped.min(axis=1, keepdims=True)
    maximum = cropped.max(axis=1, keepdims=True)
    mean = cropped.mean(axis=1, keepdims=True)
    denominator = maximum - minimum
    denominator[denominator < 1e-8] = 1.0
    return ((cropped - mean) / denominator).astype(np.float32)


def _is_daylight(timestamp: datetime) -> bool:
    """Match training's local UTC+8 daylight interval for each piece."""
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


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
    timestamps: Sequence[datetime],
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
            daylight_tensor = torch.tensor(
                [[float(_is_daylight(timestamp))] for timestamp in timestamps],
                dtype=torch.float32,
                device=device,
            )
            output = checkpoint.model(
                local_tensor, global_tensor, daylight_tensor
            )
            type_logits = output.type_logits
            distance_logits = output.distance_logits
            decoder = decode_new_prediction
        else:
            legacy_values = _legacy_preprocess_batch(values)
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


@dataclass
class _OutputBuffer:
    """One output file; its first piece deterministically owns the header."""

    relative_path: str
    source_header: bytes
    raw_pieces: list[bytes] = field(default_factory=list)


class _OutputRegrouper:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._buffers: dict[str, _OutputBuffer] = {}
        self._next_file = defaultdict(int)

    def _new_buffer(self, output_class: str, source_header: bytes) -> _OutputBuffer:
        self._next_file[output_class] += 1
        filename = f"{output_class}_{self._next_file[output_class]:06d}.lig"
        relative_path = (Path(output_class) / filename).as_posix()
        buffer = _OutputBuffer(
            relative_path=relative_path,
            source_header=source_header,
        )
        self._buffers[output_class] = buffer
        return buffer

    def add(self, output_class: str, source_header: bytes, raw_piece: bytes) -> str:
        """Buffer one exact piece and return its already assigned output path."""
        if len(raw_piece) != PIECE_BYTES:
            raise LigFormatError("refusing to buffer a reconstructed piece")
        buffer = self._buffers.get(output_class)
        if buffer is None:
            buffer = self._new_buffer(output_class, source_header)
        buffer.raw_pieces.append(raw_piece)
        relative_path = buffer.relative_path
        if len(buffer.raw_pieces) == MAX_PIECES_PER_FILE:
            self._flush(output_class)
        return relative_path

    def _flush(self, output_class: str) -> None:
        buffer = self._buffers.pop(output_class, None)
        if buffer is None or not buffer.raw_pieces:
            return
        destination = self.output_dir / Path(buffer.relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_lig_file(destination, buffer.source_header, buffer.raw_pieces)

    def flush_all(self) -> None:
        """Flush partial files in deterministic class-name order."""
        for output_class in sorted(tuple(self._buffers)):
            self._flush(output_class)


def _source_batches(
    source_path: Path, batch_size: int
):
    """Yield bounded waveforms, timestamps, and complete raw pieces."""
    with LigFileIndex([source_path], validate=True) as index, source_path.open(
        "rb"
    ) as raw_handle:
        for start in range(0, len(index), batch_size):
            stop = min(start + batch_size, len(index))
            positions = range(start, stop)
            waveforms = np.stack(index.read_pieces_batch(positions), axis=0)
            timestamps = index.read_timestamps_batch(positions)
            raw_handle.seek(FILE_HEADER_BYTES + start * PIECE_BYTES)
            raw_pieces = [raw_handle.read(PIECE_BYTES) for _ in positions]
            if any(len(raw) != PIECE_BYTES for raw in raw_pieces):
                raise LigFormatError(f"short raw piece batch: {source_path}")
            yield start, waveforms, timestamps, raw_pieces


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
    checkpoint_sha256: str,
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
        "output_file": output_file,
    }
    row.update({
        f"prob_{type_name}": probability
        for type_name, probability in zip(
            TYPE_NAMES, prediction.type_probabilities
        )
    })
    return row


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
    destination_root.mkdir(parents=True, exist_ok=True)
    csv_path = destination_root / "predictions.csv"
    regrouper = _OutputRegrouper(destination_root)

    try:
        with PredictionCsvWriter(csv_path) as writer:
            for relative_source, source_path in files:
                source_header = read_file_header(source_path)
                for start, waveforms, timestamps, raw_pieces in _source_batches(
                    source_path, batch_size
                ):
                    predictions = predict_batch(
                        checkpoint,
                        waveforms,
                        timestamps,
                        device=runtime_device,
                        type_only=type_only,
                    )
                    for offset, (raw_piece, prediction) in enumerate(
                        zip(raw_pieces, predictions)
                    ):
                        piece_index = start + offset
                        output_file = regrouper.add(
                            prediction.output_class, source_header, raw_piece
                        )
                        writer.write(_csv_row(
                            relative_source,
                            piece_index,
                            prediction,
                            checkpoint_schema=checkpoint.schema,
                            checkpoint_sha256=checkpoint_hash,
                            output_file=output_file,
                        ))
    finally:
        regrouper.flush_all()
    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the compact two-schema inference command line."""
    parser = argparse.ArgumentParser(
        description="Classify LIG pieces with a five-class checkpoint."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--type_only", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the inference CLI and return the generated CSV path."""
    args = build_arg_parser().parse_args(argv)
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
