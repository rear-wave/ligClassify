"""Bounded, byte-preserving access to fixed-size LIG waveform pieces."""

from __future__ import annotations

import math
import os
import struct
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np


FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208
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


def _validate_source(path: str | os.PathLike[str], piece_count: int) -> None:
    if piece_count > MAX_PIECES_PER_FILE:
        raise LigFormatError(
            f"piece count exceeds {MAX_PIECES_PER_FILE}: {path}: {piece_count}"
        )
    expected_size = FILE_HEADER_BYTES + piece_count * PIECE_BYTES
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise LigFormatError(
            f"LIG size mismatch: {path}: expected {expected_size}, got {actual_size}"
        )


def read_raw_piece(path: str | os.PathLike[str], piece_index: int) -> bytes:
    """Read a complete piece without decoding or reconstructing its bytes."""
    piece_index = int(piece_index)
    if piece_index < 0:
        raise IndexError(f"piece_index must be non-negative: {piece_index}")
    offset = FILE_HEADER_BYTES + piece_index * PIECE_BYTES
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(PIECE_BYTES)
    if len(raw) != PIECE_BYTES:
        raise LigFormatError(f"short piece {piece_index}: {path}")
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
    for raw in raw_pieces:
        if len(raw) != PIECE_BYTES:
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
        if not math.isfinite(sec_frac) or not 0.0 <= sec_frac < 1.0:
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
        self._file_handles: OrderedDict[int, BinaryIO] = OrderedDict()
        for filepath in filepaths:
            path = str(Path(filepath))
            piece_count = _declared_piece_count(path)
            if validate:
                _validate_source(path, piece_count)
            self.filepaths.append(path)
            self.num_pieces_per_file.append(piece_count)

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

    @staticmethod
    def _waveform_offset(piece_index: int) -> int:
        return FILE_HEADER_BYTES + piece_index * PIECE_BYTES + PIECE_HEADER_BYTES

    @staticmethod
    def _timestamp_offset(piece_index: int) -> int:
        return FILE_HEADER_BYTES + piece_index * PIECE_BYTES + _TIMESTAMP_OFFSET

    def read_piece(self, global_index: int) -> np.ndarray:
        """Read one waveform as float32 without caching its payload."""
        file_index, piece_index = self._locate(global_index)
        handle = self._get_file_handle(file_index)
        handle.seek(self._waveform_offset(piece_index))
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
                handle.seek(self._waveform_offset(piece_index))
                raw = handle.read(WAVEFORM_BYTES)
                if len(raw) != WAVEFORM_BYTES:
                    raise LigFormatError(
                        f"short waveform {piece_index}: {self.filepaths[file_index]}"
                    )
                results[order] = np.frombuffer(raw, dtype="<u2").astype(np.float32)
        return [result for result in results if result is not None]

    def read_timestamps_batch(self, global_indices: Iterable[int]) -> list[datetime]:
        """Read piece timestamps in a batch without loading waveform payloads."""
        indices = [int(index) for index in global_indices]
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for order, global_index in enumerate(indices):
            file_index, piece_index = self._locate(global_index)
            groups[file_index].append((order, piece_index))

        results: list[datetime | None] = [None] * len(indices)
        for file_index, positions in groups.items():
            handle = self._get_file_handle(file_index)
            for order, piece_index in sorted(positions, key=lambda item: item[1]):
                handle.seek(self._timestamp_offset(piece_index))
                raw = handle.read(_TIMESTAMP_BYTES)
                results[order] = _decode_timestamp(
                    raw, self.filepaths[file_index], piece_index
                )
        return [result for result in results if result is not None]

    def close(self) -> None:
        """Close every cached file handle."""
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def __del__(self) -> None:
        self.close()
