"""Memory-compact, piece-level labels for the five-class training corpus."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .lig import LigFileIndex


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")

_DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,4})[-_](\d{1,4})km", re.IGNORECASE)
_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class SourceRecord:
    """Labels and storage metadata shared by every piece in one file."""

    path: str
    relative_path: str
    type_index: int
    distance_bin: int
    piece_count: int


@dataclass
class PieceTable:
    """Aligned compact arrays describing positions in source LIG files."""

    root: str
    sources: tuple[SourceRecord, ...]
    source_index: np.ndarray
    piece_index: np.ndarray
    type_index: np.ndarray
    distance_bin: np.ndarray
    daylight: np.ndarray
    timestamp_seconds: np.ndarray

    def __len__(self) -> int:
        return len(self.source_index)

    def piece_key(self, position: int) -> str:
        source = self.sources[int(self.source_index[position])]
        return piece_key(source.relative_path, self.piece_index[position])


def piece_key(relative_path: str | os.PathLike[str], piece_index: int) -> str:
    """Return the platform-independent stable identity for one piece."""
    normalized = str(relative_path).replace("\\", "/")
    return f"{normalized}#{int(piece_index)}"


def _parse_distance_bin(relative_path: str, type_name: str) -> int:
    if type_name == "IC":
        return -1
    intervals = [
        (int(match.group(1)), int(match.group(2)))
        for match in _DISTANCE_RE.finditer(relative_path)
    ]
    exact = [
        (low, high)
        for low, high in intervals
        if 0 <= low < high <= 3000
        and low % 100 == 0
        and high % 100 == 0
        and high - low == 100
    ]
    if exact:
        return exact[-1][0] // 100
    if any(low < 0 or high > 3000 for low, high in intervals):
        raise ValueError(
            f"distance interval must lie within 0-3000 km: {relative_path}"
        )
    raise ValueError(
        f"non-IC source requires an exact 100-km interval: {relative_path}"
    )


def _is_daylight(timestamp: datetime) -> bool:
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


def _timestamp_seconds(timestamp: datetime) -> int:
    return int((timestamp - _EPOCH).total_seconds())


def _empty_piece_table(root: Path) -> PieceTable:
    return PieceTable(
        root=str(root),
        sources=(),
        source_index=np.empty(0, dtype=np.int32),
        piece_index=np.empty(0, dtype=np.int32),
        type_index=np.empty(0, dtype=np.int8),
        distance_bin=np.empty(0, dtype=np.int8),
        daylight=np.empty(0, dtype=bool),
        timestamp_seconds=np.empty(0, dtype=np.int64),
    )


def build_piece_table(
    root: str | os.PathLike[str],
) -> tuple[PieceTable, dict[str, int]]:
    """Scan the five trusted type directories and expand files into pieces."""
    root_path = Path(root).resolve()
    discovered: list[tuple[int, str, Path]] = []
    seen_paths: set[str] = set()
    for type_index, type_name in enumerate(TYPE_NAMES):
        type_dir = root_path / type_name
        if not type_dir.is_dir():
            continue
        for path in type_dir.rglob("*.lig"):
            relative_path = path.relative_to(root_path).as_posix()
            resolved_key = os.path.normcase(str(path.resolve()))
            if resolved_key in seen_paths:
                raise ValueError(f"duplicate source path: {relative_path}")
            seen_paths.add(resolved_key)
            discovered.append((type_index, relative_path, path))
    discovered.sort(key=lambda item: (item[0], item[1]))

    if not discovered:
        return _empty_piece_table(root_path), {"files": 0, "pieces": 0}

    sources: list[SourceRecord] = []
    source_arrays: list[np.ndarray] = []
    piece_arrays: list[np.ndarray] = []
    type_arrays: list[np.ndarray] = []
    distance_arrays: list[np.ndarray] = []
    daylight_arrays: list[np.ndarray] = []
    timestamp_arrays: list[np.ndarray] = []
    identities: set[str] = set()

    for type_index, relative_path, path in discovered:
        type_name = TYPE_NAMES[type_index]
        distance_bin = _parse_distance_bin(relative_path, type_name)
        with LigFileIndex([path], validate=True) as index:
            piece_count = len(index)
            timestamps = index.read_timestamps_batch(range(piece_count))

        source_position = len(sources)
        sources.append(
            SourceRecord(
                path=str(path),
                relative_path=relative_path,
                type_index=type_index,
                distance_bin=distance_bin,
                piece_count=piece_count,
            )
        )
        for piece_index in range(piece_count):
            identity = piece_key(relative_path, piece_index)
            if identity in identities:
                raise ValueError(f"duplicate piece identity: {identity}")
            identities.add(identity)

        source_arrays.append(np.full(piece_count, source_position, dtype=np.int32))
        piece_arrays.append(np.arange(piece_count, dtype=np.int32))
        type_arrays.append(np.full(piece_count, type_index, dtype=np.int8))
        distance_arrays.append(np.full(piece_count, distance_bin, dtype=np.int8))
        daylight_arrays.append(
            np.fromiter((_is_daylight(item) for item in timestamps), dtype=bool)
        )
        timestamp_arrays.append(
            np.fromiter(
                (_timestamp_seconds(item) for item in timestamps), dtype=np.int64
            )
        )

    table = PieceTable(
        root=str(root_path),
        sources=tuple(sources),
        source_index=np.concatenate(source_arrays),
        piece_index=np.concatenate(piece_arrays),
        type_index=np.concatenate(type_arrays),
        distance_bin=np.concatenate(distance_arrays),
        daylight=np.concatenate(daylight_arrays),
        timestamp_seconds=np.concatenate(timestamp_arrays),
    )
    return table, {"files": len(sources), "pieces": len(table)}
