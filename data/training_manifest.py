"""Trusted-type file discovery and piece metadata expansion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from data.lig_parser import (
    LigFormatError,
    read_lig_timestamp,
    read_lig_timestamps,
    validate_lig_file,
)


_DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,4})[-_](\d{1,4})km", re.IGNORECASE)
_FILENAME_TIME_RE = re.compile(r"_(\d{12})(?:\.(\d+))?", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestEntry:
    filepath: str
    type_idx: int
    dist_bin: int
    timestamp: datetime
    n_pieces: int
    distance_low_km: int | None = None
    distance_high_km: int | None = None
    is_daytime: bool | None = None


@dataclass(frozen=True)
class PieceManifestEntry:
    """One lazily readable waveform piece with inherited file labels."""

    filepath: str
    piece_index: int
    type_idx: int
    dist_bin: int
    timestamp: datetime
    distance_low_km: int | None = None
    distance_high_km: int | None = None
    is_daytime: bool | None = None

    @property
    def identity(self):
        return os.path.normcase(os.path.abspath(self.filepath)), self.piece_index


def parse_distance_interval(path: str) -> tuple[int, int] | None:
    """Return the last valid 100-km-aligned interval in a path."""
    valid = []
    for match in _DISTANCE_RE.finditer(path):
        low, high = int(match.group(1)), int(match.group(2))
        if low % 100 == 0 and high % 100 == 0 and 0 <= low < high <= 3000:
            valid.append((low, high))
    return valid[-1] if valid else None


def parse_distance_bin(path: str) -> int:
    """Return an exact 100-km bin for legacy metadata compatibility."""
    interval = parse_distance_interval(path)
    if interval is None or interval[1] - interval[0] != 100:
        return -1
    return interval[0] // 100


def infer_daytime(path: str, timestamp: datetime) -> bool:
    """Prefer explicit day/night folders, then infer local UTC+8 daylight."""
    parts = {
        part.lower()
        for part in re.split(r"[\\/]", os.path.normpath(path))
        if part
    }
    if "day" in parts:
        return True
    if "night" in parts:
        return False
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


def parse_filename_timestamp(path: str) -> datetime:
    """Parse ``YYMMDDhhmmss.fraction`` from a LIG filename."""
    match = _FILENAME_TIME_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"filename contains no timestamp: {path}")
    stamp = match.group(1)
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    base = datetime(
        2000 + int(stamp[0:2]),
        int(stamp[2:4]),
        int(stamp[4:6]),
    )
    return base + timedelta(
        hours=int(stamp[6:8]),
        minutes=int(stamp[8:10]),
        seconds=int(stamp[10:12]),
        microseconds=int(fraction or 0),
    )


def build_manifest(data_dir: str, type_names) -> tuple[list[ManifestEntry], dict]:
    """Discover validated LIG files and attach type, interval, and context."""
    entries = []
    diagnostics = {
        "discovered_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "timestamp_errors": 0,
        "filename_timestamp_fallbacks": 0,
        "distance_labeled_files": 0,
        "type_only_files": 0,
        "skipped_files": [],
    }
    for type_idx, type_name in enumerate(type_names):
        class_dir = Path(data_dir) / type_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*.lig")):
            diagnostics["discovered_files"] += 1
            filepath = str(path)
            try:
                result = validate_lig_file(filepath)
            except LigFormatError as exc:
                result = {"valid": False, "n_pieces": 0, "errors": [str(exc)]}
            if not result["valid"] or result["n_pieces"] <= 0:
                diagnostics["invalid_files"] += 1
                diagnostics["skipped_files"].append(filepath)
                continue
            try:
                timestamp = read_lig_timestamp(filepath)
            except (LigFormatError, ValueError):
                try:
                    timestamp = parse_filename_timestamp(filepath)
                    diagnostics["filename_timestamp_fallbacks"] += 1
                except ValueError:
                    diagnostics["timestamp_errors"] += 1
                    diagnostics["skipped_files"].append(filepath)
                    continue

            interval = None if type_name.upper() == "IC" else parse_distance_interval(filepath)
            if interval is None:
                low = high = None
                dist_bin = -1
                diagnostics["type_only_files"] += 1
            else:
                low, high = interval
                dist_bin = low // 100 if high - low == 100 else -1
                diagnostics["distance_labeled_files"] += 1
            entries.append(ManifestEntry(
                filepath=filepath,
                type_idx=type_idx,
                dist_bin=dist_bin,
                timestamp=timestamp,
                n_pieces=int(result["n_pieces"]),
                distance_low_km=low,
                distance_high_km=high,
                is_daytime=infer_daytime(filepath, timestamp),
            ))
            diagnostics["valid_files"] += 1
    entries.sort(key=lambda item: (item.type_idx, item.timestamp, item.filepath))
    return entries, diagnostics


def build_piece_manifest(file_entries):
    """Expand already-split source files into timestamped piece records."""
    pieces = []
    for file_entry in file_entries:
        timestamps = read_lig_timestamps(file_entry.filepath)
        if len(timestamps) != file_entry.n_pieces:
            raise LigFormatError(
                f"piece count changed: {file_entry.filepath}: "
                f"manifest={file_entry.n_pieces}, timestamps={len(timestamps)}"
            )
        pieces.extend(
            PieceManifestEntry(
                filepath=file_entry.filepath,
                piece_index=piece_index,
                type_idx=file_entry.type_idx,
                dist_bin=file_entry.dist_bin,
                timestamp=timestamp,
                distance_low_km=file_entry.distance_low_km,
                distance_high_km=file_entry.distance_high_km,
                is_daytime=file_entry.is_daytime,
            )
            for piece_index, timestamp in enumerate(timestamps)
        )
    return pieces
