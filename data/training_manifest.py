"""Training-file discovery, labels, and chronological split helpers."""

from __future__ import annotations

import os
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

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

    @property
    def acquisition_date(self):
        return self.timestamp.date()


@dataclass(frozen=True)
class PieceManifestEntry:
    """One independently splittable waveform piece and its labels."""

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
        """Return a stable piece identity independent of path casing."""
        return (
            os.path.normcase(os.path.abspath(self.filepath)),
            self.piece_index,
        )


def parse_distance_bin(path: str) -> int:
    """Return the last valid exact 100-km range in a path, or ``-1``."""
    valid = []
    for match in _DISTANCE_RE.finditer(path):
        low, high = int(match.group(1)), int(match.group(2))
        if (
            high - low == 100
            and low % 100 == 0
            and high % 100 == 0
            and 0 <= low < high <= 3000
        ):
            valid.append(low // 100)
    return valid[-1] if valid else -1


def parse_distance_interval(path: str) -> tuple[int, int] | None:
    """Return the last valid 100-km-aligned distance interval in a path."""
    valid = []
    for match in _DISTANCE_RE.finditer(path):
        low, high = int(match.group(1)), int(match.group(2))
        if (
            low % 100 == 0
            and high % 100 == 0
            and 0 <= low < high <= 3000
        ):
            valid.append((low, high))
    return valid[-1] if valid else None


def infer_daytime(path: str, timestamp: datetime) -> bool:
    """Read an explicit day/night folder or infer it from UTC+8 local time."""
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
    year = 2000 + int(stamp[0:2])
    month, day = int(stamp[2:4]), int(stamp[4:6])
    hour, minute, second = int(stamp[6:8]), int(stamp[8:10]), int(stamp[10:12])
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    base = datetime(year, month, day)
    return base + timedelta(
        hours=hour,
        minutes=minute,
        seconds=second,
        microseconds=int(fraction or 0),
    )


def build_manifest(data_dir: str, type_names) -> tuple[list[ManifestEntry], dict]:
    """Discover validated LIG files and attach type, distance, and time labels."""
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

            distance_interval = (
                None
                if type_name.upper() == "IC"
                else parse_distance_interval(filepath)
            )
            if distance_interval is None:
                distance_low_km = None
                distance_high_km = None
                dist_bin = -1
            else:
                distance_low_km, distance_high_km = distance_interval
                dist_bin = (
                    distance_low_km // 100
                    if distance_high_km - distance_low_km == 100
                    else -1
                )
            if distance_interval is not None:
                diagnostics["distance_labeled_files"] += 1
            else:
                diagnostics["type_only_files"] += 1
            entries.append(
                ManifestEntry(
                    filepath=filepath,
                    type_idx=type_idx,
                    dist_bin=dist_bin,
                    timestamp=timestamp,
                    n_pieces=int(result["n_pieces"]),
                    distance_low_km=distance_low_km,
                    distance_high_km=distance_high_km,
                    is_daytime=infer_daytime(filepath, timestamp),
                )
            )
            diagnostics["valid_files"] += 1

    entries.sort(key=lambda item: (item.type_idx, item.timestamp, item.filepath))
    return entries, diagnostics


def build_piece_manifest(file_entries):
    """Expand validated files into timestamped, lazily readable pieces."""
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


def _constrained_split_counts(size, val_fraction, test_fraction):
    """Round split counts while reserving at least one piece per split."""
    fractions = np.asarray(
        [1.0 - val_fraction - test_fraction, val_fraction, test_fraction],
        dtype=np.float64,
    )
    if size < 3:
        raise ValueError(f"at least 3 pieces are required, got {size}")
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train, validation, and test fractions must be positive")

    raw = fractions * size
    counts = np.maximum(1, np.floor(raw).astype(np.int64))
    while int(counts.sum()) > size:
        candidates = np.flatnonzero(counts > 1)
        index = int(candidates[np.argmax((counts - raw)[candidates])])
        counts[index] -= 1
    while int(counts.sum()) < size:
        index = int(np.argmax(raw - counts))
        counts[index] += 1
    return tuple(int(value) for value in counts)


def piece_time_split_manifest(entries, val_fraction=0.15, test_fraction=0.15):
    """Split pieces chronologically inside every type/distance group."""
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry.type_idx, entry.dist_bin)].append(entry)

    splits = {"train": [], "val": [], "test": []}
    for (type_idx, dist_bin), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: (
                item.timestamp,
                os.path.normcase(item.filepath),
                item.piece_index,
            ),
        )
        try:
            train_count, val_count, _ = _constrained_split_counts(
                len(ordered),
                val_fraction,
                test_fraction,
            )
        except ValueError as exc:
            raise ValueError(
                f"type={type_idx} bin={dist_bin} has {len(ordered)} pieces: {exc}"
            ) from exc
        val_end = train_count + val_count
        splits["train"].extend(ordered[:train_count])
        splits["val"].extend(ordered[train_count:val_end])
        splits["test"].extend(ordered[val_end:])
    return splits


def validate_piece_split_isolation(splits):
    """Reject a piece identity assigned to more than one split."""
    owners = {}
    for split_name, entries in splits.items():
        for entry in entries:
            previous = owners.setdefault(entry.identity, split_name)
            if previous != split_name:
                raise ValueError(
                    f"piece identity appears in {previous} and {split_name}: "
                    f"{entry.identity}"
                )


def validate_piece_split_coverage(splits, type_names, min_eval_pieces=500):
    """Require validation and test coverage for every available distance bin."""
    source = splits["train"] + splits["val"] + splits["test"]
    failures = []
    for type_idx, type_name in enumerate(type_names):
        source_bins = {
            entry.dist_bin for entry in source if entry.type_idx == type_idx
        }
        for split_name in ("val", "test"):
            selected = [
                entry
                for entry in splits[split_name]
                if entry.type_idx == type_idx
            ]
            selected_bins = {entry.dist_bin for entry in selected}
            missing = sorted(source_bins - selected_bins)
            if missing:
                failures.append(f"{type_name} {split_name}: missing bins {missing}")
            if len(selected) < min_eval_pieces:
                failures.append(
                    f"{type_name} {split_name}: pieces={len(selected)} "
                    f"below required {min_eval_pieces}"
                )
    if failures:
        raise ValueError("Invalid piece split coverage: " + "; ".join(failures))


def _stable_file_order(entries, seed):
    return sorted(
        entries,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}|{os.path.abspath(item.filepath)}".encode("utf-8")
            ).hexdigest(),
            item.filepath,
        ),
    )


def coverage_temporal_split_manifest(
    entries,
    val_fraction=0.15,
    test_fraction=0.15,
    seed=42,
):
    """Build a file-level coverage validation and a latest-date locked test."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation and test fractions must be in [0, 1)")

    by_type_date = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        by_type_date[entry.type_idx][entry.acquisition_date].append(entry)

    development = []
    test = []
    for type_idx in sorted(by_type_date):
        dates = sorted(by_type_date[type_idx])
        if test_fraction == 0 or len(dates) < 2:
            development.extend(
                entry for date in dates for entry in by_type_date[type_idx][date]
            )
            continue
        test_date_count = max(1, int(round(len(dates) * test_fraction)))
        test_date_count = min(test_date_count, len(dates) - 1)
        selected = set(dates[-test_date_count:])
        for date in dates:
            destination = test if date in selected else development
            destination.extend(by_type_date[type_idx][date])

    validation_groups = defaultdict(list)
    for entry in development:
        group = (entry.type_idx, entry.dist_bin)
        validation_groups[group].append(entry)

    validation_paths = set()
    if val_fraction > 0:
        for group_entries in validation_groups.values():
            if len(group_entries) < 2:
                continue
            count = max(1, int(round(len(group_entries) * val_fraction)))
            count = min(count, len(group_entries) - 1)
            validation_paths.update(
                item.filepath
                for item in _stable_file_order(group_entries, seed)[:count]
            )

    splits = {
        "train": [x for x in development if x.filepath not in validation_paths],
        "val": [x for x in development if x.filepath in validation_paths],
        "test": test,
    }
    for split_entries in splits.values():
        split_entries.sort(
            key=lambda item: (item.type_idx, item.timestamp, item.filepath)
        )
    return splits


def validate_split_coverage(splits, type_names, min_bins=12, min_pieces=500):
    """Fail before training when distance validation cannot support selection."""
    validation = splits.get("val", [])
    failures = []
    for type_idx, type_name in enumerate(type_names):
        if type_name.upper() == "IC":
            continue
        typed = [entry for entry in validation if entry.type_idx == type_idx]
        bins = {entry.dist_bin for entry in typed if entry.dist_bin >= 0}
        pieces = sum(entry.n_pieces for entry in typed if entry.dist_bin >= 0)
        if len(bins) < min_bins:
            failures.append(
                f"{type_name}: bins={len(bins)} below required {min_bins}"
            )
        if pieces < min_pieces:
            failures.append(
                f"{type_name}: pieces={pieces} below required {min_pieces}"
            )
    if failures:
        raise ValueError("Invalid validation coverage: " + "; ".join(failures))
