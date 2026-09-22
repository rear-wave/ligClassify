"""Bounded, byte-preserving access to fixed-size LIG waveform pieces."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import struct
import time
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Sequence

import numpy as np


FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208
PIECE_HEADER_BYTES = 208
WAVEFORM_SAMPLES = 16000
WAVEFORM_BYTES = WAVEFORM_SAMPLES * 2
EXTENDED_PIECE_HEADER_BYTES = PIECE_HEADER_BYTES + 256
THREE_CHANNEL_PIECE_BYTES = EXTENDED_PIECE_HEADER_BYTES + 3 * WAVEFORM_BYTES
SUPPORTED_PIECE_BYTES = (
    PIECE_BYTES,
    EXTENDED_PIECE_HEADER_BYTES + WAVEFORM_BYTES,
    THREE_CHANNEL_PIECE_BYTES,
)
MAX_PIECES_PER_FILE = 512
MAX_WAVEFORM_SAMPLES = 10_000_000

_PIECE_COUNT_OFFSET = 4
_SAMPLE_COUNT_OFFSET = 20
_CHANNEL_COUNT_OFFSET = 24
_TIMESTAMP_OFFSET = 108
_TIMESTAMP_BYTES = 36
_MAX_OPEN_FILES = 64
_FILE_OPERATION_RETRY_SECONDS = 8.0


def _retry_transient_file_operation(operation: Callable[[], object]) -> None:
    """Retry a file replacement briefly when Windows reports a sharing lock."""
    deadline = time.monotonic() + _FILE_OPERATION_RETRY_SECONDS
    delay = 0.025
    while True:
        try:
            operation()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.5)


class LigFormatError(ValueError):
    """Raised when a LIG container violates the fixed binary layout."""


class LigInferenceJournal:
    """Append predictions and checkpoint completed source files safely."""

    _SCHEMA = "lig_inference_resume_v1"

    def __init__(
        self,
        output_dir: Path,
        fieldnames: Sequence[str],
        signature: Mapping[str, object],
        row_constraints: Mapping[str, str],
        allowed_types: Iterable[str] | None,
        *,
        resume: bool,
    ) -> None:
        self.output_dir = output_dir
        self.csv_path = output_dir / "predictions.csv"
        self.state_path = output_dir / ".classification_resume.json"
        self.fieldnames = tuple(fieldnames)
        encoded = json.dumps(
            signature, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.signature_sha256 = hashlib.sha256(encoded).hexdigest()
        self.row_constraints = dict(row_constraints)
        self.allowed_types = None if allowed_types is None else frozenset(allowed_types)
        self.resume = bool(resume)
        self.completed_keys: set[str] = set()
        self.completed_source = ""
        self.legacy_frontier = ""
        self.active_source = ""
        self._handle: Any = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "LigInferenceJournal":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_dir.iterdir())
        if existing and not self.resume:
            raise ValueError("output_dir is not empty; use --resume to continue it")
        if self.resume:
            self._load_predictions()
            self._load_state()
        mode = "a" if self.resume and self.csv_path.exists() else "w"
        self._handle = self.csv_path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.fieldnames, extrasaction="raise"
        )
        if mode == "w":
            self._writer.writeheader()
            self._handle.flush()
        return self

    def _load_predictions(self) -> None:
        if not self.csv_path.exists():
            unexpected = [path for path in self.output_dir.iterdir()
                          if path != self.state_path]
            if unexpected:
                raise ValueError("cannot resume outputs without predictions.csv")
            return
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != self.fieldnames:
                raise ValueError("cannot resume predictions.csv with another schema")
            first = next(reader, None)
        if first is None:
            return
        self._validate_prediction_row(first)
        last_source = ""
        for line in self._reverse_lines(self.csv_path):
            values = next(csv.reader([line]))
            if tuple(values) == self.fieldnames:
                break
            if len(values) != len(self.fieldnames):
                raise ValueError("cannot resume a truncated predictions.csv")
            row = dict(zip(self.fieldnames, values))
            source = row["source_path"]
            if last_source and source != last_source:
                break
            last_source = source
            self._validate_prediction_row(row)
            key = row["piece_key"]
            if not key or key in self.completed_keys:
                raise ValueError("resume predictions contain an invalid duplicate key")
            self.completed_keys.add(key)
        self.active_source = last_source
        self.legacy_frontier = last_source

    @staticmethod
    def _reverse_lines(path: Path, block_size: int = 1 << 20):
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            remainder = b""
            while position:
                size = min(block_size, position)
                position -= size
                handle.seek(position)
                parts = (handle.read(size) + remainder).split(b"\n")
                remainder = parts[0]
                for raw in reversed(parts[1:]):
                    if raw:
                        yield raw.rstrip(b"\r").decode("utf-8")
            if remainder:
                yield remainder.rstrip(b"\r").decode("utf-8")

    def _validate_prediction_row(self, row: Mapping[str, str]) -> None:
        if any(row.get(key, "") != value
               for key, value in self.row_constraints.items()):
            raise ValueError("resume model or decision configuration mismatch")
        selected = (
            self.allowed_types is None
            or row["final_type"] in self.allowed_types
        )
        output_name = row["output_file"]
        if not selected:
            if output_name:
                raise ValueError("resume output_type selection mismatch")
            return
        if not output_name:
            raise ValueError("resume selected prediction has no output file")
        output = (self.output_dir / output_name).resolve()
        if os.path.commonpath([self.output_dir.resolve(), output]) != os.fspath(
            self.output_dir.resolve()
        ) or not output.is_file():
            raise ValueError("resume prediction references a missing output file")

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("resume state is unreadable") from exc
        if (
            payload.get("schema") != self._SCHEMA
            or payload.get("signature_sha256") != self.signature_sha256
            or not isinstance(payload.get("completed_source"), str)
        ):
            raise ValueError("resume input or inference configuration mismatch")
        self.completed_source = payload["completed_source"]
        self.legacy_frontier = ""

    def source_is_complete(self, relative_source: str) -> bool:
        """Return whether a source precedes the durable resume cursor."""
        return bool(
            (self.completed_source and relative_source <= self.completed_source)
            or (self.legacy_frontier and relative_source < self.legacy_frontier)
        )

    def begin_source(self, relative_source: str) -> None:
        """Retain duplicate keys only for the source currently being resumed."""
        if relative_source != self.active_source:
            self.completed_keys.clear()
            self.active_source = relative_source

    def write(self, row: Mapping[str, object]) -> None:
        """Append and flush one prediction unless its piece already exists."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("LigInferenceJournal is not open")
        key = str(row["piece_key"])
        if key in self.completed_keys:
            return
        self._writer.writerow({name: "" if row.get(name) is None else row.get(name, "")
                               for name in self.fieldnames})
        self._handle.flush()
        self.completed_keys.add(key)

    def complete_source(self, relative_source: str) -> None:
        """Atomically persist the last fully flushed source file."""
        payload = {
            "schema": self._SCHEMA,
            "signature_sha256": self.signature_sha256,
            "completed_source": relative_source,
        }
        temporary = self.state_path.with_name(
            f"{self.state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            _retry_transient_file_operation(
                lambda: os.replace(temporary, self.state_path)
            )
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
        self.completed_source = relative_source
        self.legacy_frontier = ""

    def finish(self) -> None:
        """Remove the transient cursor after every requested source completes."""
        _retry_transient_file_operation(
            lambda: self.state_path.unlink(missing_ok=True)
        )

    def __exit__(self, *_args: object) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None


def read_file_header(path: str | os.PathLike[str]) -> bytes:
    """Read one complete LIG file header."""
    with open(path, "rb") as handle:
        header = handle.read(FILE_HEADER_BYTES)
    if len(header) != FILE_HEADER_BYTES:
        raise LigFormatError(f"short LIG header: {path}")
    return header


def _declared_piece_count(path: str | os.PathLike[str]) -> int:
    header = read_file_header(path)
    piece_count = struct.unpack_from("<i", header, _PIECE_COUNT_OFFSET)[0]
    if piece_count < 0:
        raise LigFormatError(f"negative piece count {piece_count}: {path}")
    return piece_count


def _extended_layout(
    piece_header: bytes,
    piece_bytes: int,
) -> tuple[int, int] | None:
    if len(piece_header) < _CHANNEL_COUNT_OFFSET + 4:
        return None
    sample_count, channel_count = struct.unpack_from(
        "<2i", piece_header, _SAMPLE_COUNT_OFFSET
    )
    if (
        sample_count < WAVEFORM_SAMPLES
        or sample_count > MAX_WAVEFORM_SAMPLES
        or channel_count not in (1, 3)
    ):
        return None
    expected = EXTENDED_PIECE_HEADER_BYTES + 2 * sample_count * channel_count
    return (sample_count, channel_count) if expected == piece_bytes else None


def _dynamic_source_layout(
    path: str | os.PathLike[str],
    piece_count: int,
    piece_bytes: int,
) -> tuple[int, int] | None:
    layout = None
    with open(path, "rb") as handle:
        for piece_index in range(piece_count):
            handle.seek(FILE_HEADER_BYTES + piece_index * piece_bytes)
            candidate = _extended_layout(
                handle.read(_CHANNEL_COUNT_OFFSET + 4), piece_bytes
            )
            if candidate is None or layout not in (None, candidate):
                return None
            layout = candidate
    return layout


def _validate_source(path: str | os.PathLike[str], piece_count: int) -> int:
    if piece_count > MAX_PIECES_PER_FILE:
        raise LigFormatError(
            f"piece count exceeds {MAX_PIECES_PER_FILE}: {path}: {piece_count}"
        )
    actual_size = os.path.getsize(path)
    payload_size = actual_size - FILE_HEADER_BYTES
    if piece_count == 0:
        if payload_size == 0:
            return PIECE_BYTES
    elif payload_size >= 0 and payload_size % piece_count == 0:
        piece_bytes = payload_size // piece_count
        if piece_bytes in SUPPORTED_PIECE_BYTES:
            # Some files have a size that divides evenly as 32208 but are
            # actually laid out with the 32464 variant (256-byte extra piece
            # header). Disambiguate by checking whether piece #1's timestamp
            # is valid under each layout; the correct layout yields a valid
            # timestamp, the wrong one reads waveform bytes as a timestamp.
            if piece_bytes == PIECE_BYTES and piece_count >= 2:
                for candidate in (PIECE_BYTES + 256, PIECE_BYTES):
                    if _piece1_timestamp_valid(path, candidate):
                        return candidate
            return piece_bytes
        if _dynamic_source_layout(path, piece_count, piece_bytes) is not None:
            return piece_bytes
    expected = ", ".join(
        str(FILE_HEADER_BYTES + piece_count * size)
        for size in SUPPORTED_PIECE_BYTES
    )
    raise LigFormatError(
        f"LIG size mismatch: {path}: expected one of [{expected}], "
        f"got {actual_size}"
    )


def _piece1_timestamp_valid(path: str | os.PathLike[str], piece_bytes: int) -> bool:
    """Return True if piece #1's timestamp decodes as a plausible datetime."""
    try:
        with open(path, "rb") as handle:
            handle.seek(FILE_HEADER_BYTES + piece_bytes + _TIMESTAMP_OFFSET)
            raw = handle.read(_TIMESTAMP_BYTES)
        year, month, day = struct.unpack_from("<3i", raw, 0)
        if year < 100:
            year += 2000
        return 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
    except (OSError, struct.error):
        return False


def _source_piece_bytes(path: str | os.PathLike[str]) -> tuple[int, int]:
    piece_count = _declared_piece_count(path)
    return piece_count, _validate_source(path, piece_count)


def read_raw_piece(path: str | os.PathLike[str], piece_index: int) -> bytes:
    """Read a complete piece without decoding or reconstructing its bytes."""
    piece_index = int(piece_index)
    piece_count, piece_bytes = _source_piece_bytes(path)
    if piece_index < 0 or piece_index >= piece_count:
        raise IndexError(
            f"piece_index={piece_index} outside file with {piece_count} pieces"
        )
    offset = FILE_HEADER_BYTES + piece_index * piece_bytes
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(piece_bytes)
    if len(raw) != piece_bytes:
        raise LigFormatError(
            f"short piece {piece_index}: {path}"
        )
    return raw


def write_lig_file(
    path: str | os.PathLike[str],
    source_header: bytes,
    raw_pieces: Sequence[bytes],
) -> None:
    """Write byte-exact pieces under a copied header with an updated count."""
    if len(source_header) != FILE_HEADER_BYTES:
        raise LigFormatError("refusing to write an invalid LIG header")
    if len(raw_pieces) > MAX_PIECES_PER_FILE:
        raise LigFormatError(
            f"refusing to write more than {MAX_PIECES_PER_FILE} pieces"
        )
    piece_sizes = {len(raw) for raw in raw_pieces}
    valid_pieces = all(
        len(raw) in SUPPORTED_PIECE_BYTES
        or _extended_layout(raw, len(raw)) is not None
        for raw in raw_pieces
    )
    if len(piece_sizes) > 1 or not valid_pieces:
        raise LigFormatError("refusing to write a reconstructed piece")

    header = bytearray(source_header)
    struct.pack_into("<i", header, _PIECE_COUNT_OFFSET, len(raw_pieces))
    with open(path, "wb") as handle:
        handle.write(header)
        for raw in raw_pieces:
            handle.write(raw)


def _decode_timestamp(
    raw: bytes,
    path: str | os.PathLike[str],
    piece_index: int,
) -> datetime:
    if len(raw) != _TIMESTAMP_BYTES:
        raise LigFormatError(f"short timestamp {piece_index}: {path}")
    try:
        year, month, day, hour, minute, second = struct.unpack_from(
            "<6i4x", raw, 0
        )
        sec_frac = struct.unpack_from("<d", raw, 28)[0]
        if year < 100:
            year += 2000
        if not 2000 <= year <= 2100:
            raise ValueError(f"invalid year: {year}")
        if not 0 <= hour <= 24:
            raise ValueError(f"invalid hour: {hour}")
        if not 0 <= minute <= 59:
            raise ValueError(f"invalid minute: {minute}")
        if not 0 <= second <= 60:
            raise ValueError(f"invalid second: {second}")
        if not math.isfinite(sec_frac) or not 0.0 <= sec_frac < 60.0:
            raise ValueError(f"invalid fractional second: {sec_frac}")
        return datetime(year, month, day) + timedelta(
            hours=hour,
            minutes=minute,
            seconds=second + sec_frac,
        )
    except (OverflowError, struct.error, ValueError) as exc:
        raise LigFormatError(
            f"invalid timestamp: {path}: piece_index={piece_index}: {exc}"
        ) from exc


def read_lig_timestamp(
    path: str | os.PathLike[str], piece_index: int = 0
) -> datetime:
    """Read one piece timestamp without loading its waveform payload."""
    piece_count, piece_bytes = _source_piece_bytes(path)
    piece_index = int(piece_index)
    if piece_index < 0 or piece_index >= piece_count:
        raise IndexError(
            f"piece_index={piece_index} outside file with {piece_count} pieces"
        )
    offset = FILE_HEADER_BYTES + piece_index * piece_bytes + _TIMESTAMP_OFFSET
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(_TIMESTAMP_BYTES)
    return _decode_timestamp(raw, path, piece_index)


class LigFileIndex:
    """Index fixed-size pieces across files while loading payloads lazily."""

    def __init__(
        self,
        filepaths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
        validate: bool = True,
    ) -> None:
        if isinstance(filepaths, (str, os.PathLike)):
            filepaths = [filepaths]
        self.filepaths: list[str] = []
        self.num_pieces_per_file: list[int] = []
        self.piece_bytes_per_file: list[int] = []
        self.waveform_samples_per_file: list[int] = []
        self._file_handles: OrderedDict[int, BinaryIO] = OrderedDict()
        for filepath in filepaths:
            path = str(Path(filepath))
            piece_count = _declared_piece_count(path)
            if validate:
                piece_bytes = _validate_source(path, piece_count)
            else:
                piece_bytes = PIECE_BYTES
            # When the file uses the larger piece variant, the declared count
            # may overstate how many pieces actually fit; clamp to the number
            # of complete pieces the payload can hold.
            if piece_bytes > PIECE_BYTES:
                payload = os.path.getsize(path) - FILE_HEADER_BYTES
                piece_count = min(piece_count, payload // piece_bytes)
            self.filepaths.append(path)
            self.num_pieces_per_file.append(piece_count)
            self.piece_bytes_per_file.append(piece_bytes)
            layout = (
                _dynamic_source_layout(path, piece_count, piece_bytes)
                if piece_bytes not in SUPPORTED_PIECE_BYTES
                else None
            )
            self.waveform_samples_per_file.append(
                WAVEFORM_SAMPLES if layout is None else layout[0]
            )

        self._cumsum = np.cumsum(
            np.asarray([0, *self.num_pieces_per_file], dtype=np.int64)
        )
        self.total_pieces = int(self._cumsum[-1])

    def __len__(self) -> int:
        return self.total_pieces

    def __enter__(self) -> "LigFileIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _locate(self, global_index: int) -> tuple[int, int]:
        global_index = int(global_index)
        if global_index < 0 or global_index >= self.total_pieces:
            raise IndexError(
                f"global piece index {global_index} outside [0, {self.total_pieces})"
            )
        file_index = int(
            np.searchsorted(self._cumsum, global_index, side="right") - 1
        )
        return file_index, global_index - int(self._cumsum[file_index])

    def _get_file_handle(self, file_index: int) -> BinaryIO:
        handle = self._file_handles.pop(file_index, None)
        if handle is None:
            if len(self._file_handles) >= _MAX_OPEN_FILES:
                _, oldest = self._file_handles.popitem(last=False)
                oldest.close()
            handle = open(self.filepaths[file_index], "rb")
        self._file_handles[file_index] = handle
        return handle

    def _waveform_offset(self, file_index: int, piece_index: int) -> int:
        piece_bytes = self.piece_bytes_per_file[file_index]
        waveform_header_bytes = (
            PIECE_HEADER_BYTES
            if piece_bytes == PIECE_BYTES
            else EXTENDED_PIECE_HEADER_BYTES
        )
        return (
            FILE_HEADER_BYTES
            + piece_index * piece_bytes
            + waveform_header_bytes
        )

    def _timestamp_offset(self, file_index: int, piece_index: int) -> int:
        return (
            FILE_HEADER_BYTES
            + piece_index * self.piece_bytes_per_file[file_index]
            + _TIMESTAMP_OFFSET
        )

    def read_piece(self, global_index: int) -> np.ndarray:
        """Read one waveform as float32 without caching its payload."""
        file_index, piece_index = self._locate(global_index)
        return self._read_waveform(file_index, piece_index)

    def _read_waveform(self, file_index: int, piece_index: int) -> np.ndarray:
        handle = self._get_file_handle(file_index)
        sample_count = self.waveform_samples_per_file[file_index]
        handle.seek(self._waveform_offset(file_index, piece_index))
        raw = handle.read(sample_count * 2)
        if len(raw) != sample_count * 2:
            raise LigFormatError(
                f"short waveform {piece_index}: {self.filepaths[file_index]}"
            )
        values = np.frombuffer(raw, dtype="<u2")
        if sample_count > WAVEFORM_SAMPLES:
            stride = max(1, sample_count // 200_000)
            baseline = int(round(float(np.median(values[::stride]))))
            peak = 0
            peak_deviation = -1
            for start in range(0, sample_count, 1_000_000):
                chunk = values[start:start + 1_000_000].astype(np.int32)
                deviations = np.abs(chunk - baseline)
                local_peak = int(np.argmax(deviations))
                local_deviation = int(deviations[local_peak])
                if local_deviation > peak_deviation:
                    peak = start + local_peak
                    peak_deviation = local_deviation
            window_start = min(
                max(peak - WAVEFORM_SAMPLES // 4, 0),
                sample_count - WAVEFORM_SAMPLES,
            )
            values = values[window_start:window_start + WAVEFORM_SAMPLES]
        return values.astype(np.float32)

    def read_pieces_batch(self, global_indices: Iterable[int]) -> list[np.ndarray]:
        """Read waveforms grouped by source file and preserve input order."""
        indices = [int(index) for index in global_indices]
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for order, global_index in enumerate(indices):
            file_index, piece_index = self._locate(global_index)
            groups[file_index].append((order, piece_index))

        results: list[np.ndarray | None] = [None] * len(indices)
        for file_index, positions in groups.items():
            for order, piece_index in sorted(positions, key=lambda item: item[1]):
                results[order] = self._read_waveform(file_index, piece_index)
        return [result for result in results if result is not None]

    def read_timestamps_batch(
        self,
        global_indices: Iterable[int],
        *,
        allow_invalid: bool = False,
    ) -> list[datetime | None]:
        """Read timestamps, optionally retaining invalid entries as ``None``."""
        indices = [int(index) for index in global_indices]
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for order, global_index in enumerate(indices):
            file_index, piece_index = self._locate(global_index)
            groups[file_index].append((order, piece_index))

        results: list[datetime | None] = [None] * len(indices)
        for file_index, positions in groups.items():
            handle = self._get_file_handle(file_index)
            for order, piece_index in sorted(positions, key=lambda item: item[1]):
                handle.seek(self._timestamp_offset(file_index, piece_index))
                raw = handle.read(_TIMESTAMP_BYTES)
                try:
                    results[order] = _decode_timestamp(
                        raw, self.filepaths[file_index], piece_index
                    )
                except LigFormatError:
                    if not allow_invalid:
                        raise
                    results[order] = None
        return results

    def close(self) -> None:
        """Close every cached file handle."""
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:
        self.close()


def iter_lig_batches(
    path: Path,
    batch_size: int,
    *,
    allow_invalid_timestamps: bool = False,
    start_piece: int = 0,
):
    """Yield bounded waveforms, timestamps, and byte-exact raw pieces."""
    with LigFileIndex([path], validate=True) as index, path.open("rb") as raw:
        if not 0 <= start_piece <= len(index):
            raise ValueError("start_piece is outside the LIG piece range")
        piece_bytes = index.piece_bytes_per_file[0]
        for start in range(start_piece, len(index), batch_size):
            stop = min(start + batch_size, len(index))
            positions = range(start, stop)
            waveforms = np.stack(index.read_pieces_batch(positions), axis=0)
            timestamps = index.read_timestamps_batch(
                positions,
                allow_invalid=allow_invalid_timestamps,
            )
            raw.seek(FILE_HEADER_BYTES + start * piece_bytes)
            pieces = [raw.read(piece_bytes) for _ in positions]
            if any(len(piece) != piece_bytes for piece in pieces):
                raise LigFormatError(f"short raw piece batch: {path}")
            yield start, waveforms, timestamps, pieces


def iter_inference_lig_batches(
    path: Path,
    batch_size: int,
    *,
    allow_invalid_timestamps: bool = False,
    skip_io_errors: bool = False,
):
    """Yield inference batches with their header, skipping invalid containers."""
    next_piece = 0
    for attempt in range(3):
        try:
            header = read_file_header(path)
            for batch in iter_lig_batches(
                path,
                batch_size,
                allow_invalid_timestamps=allow_invalid_timestamps,
                start_piece=next_piece,
            ):
                yield header, *batch
                next_piece = batch[0] + len(batch[3])
            return
        except LigFormatError as exc:
            if next_piece and not skip_io_errors:
                raise
            scope = "remainder of invalid LIG file" if next_piece else "invalid LIG file"
            warnings.warn(
                f"skipping {scope} from piece {next_piece}: {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        except OSError as exc:
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
            if not skip_io_errors:
                raise OSError(
                    exc.errno,
                    f"failed reading LIG file at piece {next_piece}: {path}: {exc}",
                ) from exc
            warnings.warn(
                f"skipping unreadable LIG file after 3 attempts: "
                f"{path} from piece {next_piece}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return


@dataclass
class _OutputBuffer:
    relative_path: str
    source_header: bytes
    raw_pieces: list[bytes] = field(default_factory=list)


class LigOutputRegrouper:
    """Write byte-exact 512-piece LIG groups with timestamp names."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._buffers: dict[str, _OutputBuffer] = {}
        self._reserved: set[str] = set()

    def _new_buffer(
        self,
        output_class: str,
        header: bytes,
        timestamp: datetime | None,
    ) -> _OutputBuffer:
        stem = (
            "GZ_unknown"
            if timestamp is None
            else timestamp.strftime("GZ_%Y%m%d%H%M%S")
        )
        suffix = 1
        while True:
            name = f"{stem}.lig" if suffix == 1 else f"{stem}_{suffix:03d}.lig"
            relative = (Path(output_class) / name).as_posix()
            if (
                relative not in self._reserved
                and not (self.output_dir / relative).exists()
            ):
                break
            suffix += 1
        self._reserved.add(relative)
        buffer = _OutputBuffer(relative, header)
        self._buffers[output_class] = buffer
        return buffer

    def add(
        self,
        output_class: str,
        source_header: bytes,
        raw_piece: bytes,
        timestamp: datetime | None,
    ) -> str:
        """Buffer one complete piece and return its assigned relative path."""
        if (
            len(raw_piece) not in SUPPORTED_PIECE_BYTES
            and _extended_layout(raw_piece, len(raw_piece)) is None
        ):
            raise LigFormatError("refusing to buffer a reconstructed piece")
        if timestamp is not None and not isinstance(timestamp, datetime):
            raise LigFormatError("output piece timestamp is invalid")
        buffer = self._buffers.get(output_class)
        if (
            buffer is not None
            and buffer.raw_pieces
            and len(buffer.raw_pieces[0]) != len(raw_piece)
        ):
            self._flush(output_class)
            buffer = None
        if buffer is None:
            buffer = self._new_buffer(
                output_class, source_header, timestamp
            )
        buffer.raw_pieces.append(raw_piece)
        relative = buffer.relative_path
        if (
            len(buffer.raw_pieces) == MAX_PIECES_PER_FILE
            or len(raw_piece) > THREE_CHANNEL_PIECE_BYTES
        ):
            self._flush(output_class)
        return relative

    def _flush(self, output_class: str) -> None:
        buffer = self._buffers.pop(output_class, None)
        if buffer is None or not buffer.raw_pieces:
            return
        destination = self.output_dir / buffer.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_lig_file(
            destination, buffer.source_header, buffer.raw_pieces
        )

    def flush_all(self) -> None:
        """Flush partial groups in deterministic leaf order."""
        for output_class in sorted(tuple(self._buffers)):
            self._flush(output_class)
