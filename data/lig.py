"""Bounded, byte-preserving access to fixed-size LIG waveform pieces."""

from __future__ import annotations

import math
import os
import struct
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np


FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208
SUPPORTED_PIECE_BYTES = (PIECE_BYTES, PIECE_BYTES + 256)
PIECE_HEADER_BYTES = 208
WAVEFORM_SAMPLES = 16000
WAVEFORM_BYTES = WAVEFORM_SAMPLES * 2
MAX_PIECES_PER_FILE = 512

_PIECE_COUNT_OFFSET = 4
_TIMESTAMP_OFFSET = 108
_TIMESTAMP_BYTES = 36
_MAX_OPEN_FILES = 64


class LigFormatError(ValueError):
    """Raised when a LIG container violates the fixed binary layout."""


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
            return piece_bytes
    expected = ", ".join(
        str(FILE_HEADER_BYTES + piece_count * size)
        for size in SUPPORTED_PIECE_BYTES
    )
    raise LigFormatError(
        f"LIG size mismatch: {path}: expected one of [{expected}], "
        f"got {actual_size}"
    )


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
    if len(piece_sizes) > 1 or not piece_sizes.issubset(SUPPORTED_PIECE_BYTES):
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
    piece_count = _declared_piece_count(path)
    piece_index = int(piece_index)
    if piece_index < 0 or piece_index >= piece_count:
        raise IndexError(
            f"piece_index={piece_index} outside file with {piece_count} pieces"
        )
    offset = FILE_HEADER_BYTES + piece_index * PIECE_BYTES + _TIMESTAMP_OFFSET
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
        self._file_handles: OrderedDict[int, BinaryIO] = OrderedDict()
        for filepath in filepaths:
            path = str(Path(filepath))
            piece_count = _declared_piece_count(path)
            if validate:
                piece_bytes = _validate_source(path, piece_count)
            else:
                piece_bytes = PIECE_BYTES
            self.filepaths.append(path)
            self.num_pieces_per_file.append(piece_count)
            self.piece_bytes_per_file.append(piece_bytes)

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
            PIECE_HEADER_BYTES + piece_bytes - PIECE_BYTES
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
        handle = self._get_file_handle(file_index)
        handle.seek(self._waveform_offset(file_index, piece_index))
        raw = handle.read(WAVEFORM_BYTES)
        if len(raw) != WAVEFORM_BYTES:
            raise LigFormatError(
                f"short waveform {piece_index}: {self.filepaths[file_index]}"
            )
        return np.frombuffer(raw, dtype="<u2").astype(np.float32)

    def read_pieces_batch(self, global_indices: Iterable[int]) -> list[np.ndarray]:
        """Read waveforms grouped by source file and preserve input order."""
        indices = [int(index) for index in global_indices]
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for order, global_index in enumerate(indices):
            file_index, piece_index = self._locate(global_index)
            groups[file_index].append((order, piece_index))

        results: list[np.ndarray | None] = [None] * len(indices)
        for file_index, positions in groups.items():
            handle = self._get_file_handle(file_index)
            for order, piece_index in sorted(positions, key=lambda item: item[1]):
                handle.seek(self._waveform_offset(file_index, piece_index))
                raw = handle.read(WAVEFORM_BYTES)
                if len(raw) != WAVEFORM_BYTES:
                    raise LigFormatError(
                        f"short waveform {piece_index}: {self.filepaths[file_index]}"
                    )
                results[order] = np.frombuffer(raw, dtype="<u2").astype(np.float32)
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
):
    """Yield bounded waveforms, timestamps, and byte-exact raw pieces."""
    with LigFileIndex([path], validate=True) as index, path.open("rb") as raw:
        piece_bytes = index.piece_bytes_per_file[0]
        for start in range(0, len(index), batch_size):
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
        if len(raw_piece) not in SUPPORTED_PIECE_BYTES:
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
        if len(buffer.raw_pieces) == MAX_PIECES_PER_FILE:
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
