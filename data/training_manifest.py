"""Training-file discovery, labels, and chronological split helpers."""

from __future__ import annotations

import os
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from data.lig_parser import LigFormatError, read_lig_timestamp, validate_lig_file


_DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,4})[-_](\d{1,4})km", re.IGNORECASE)
_FILENAME_TIME_RE = re.compile(r"_(\d{12})(?:\.(\d+))?", re.IGNORECASE)


@dataclass(frozen=True)
class ManifestEntry:
    filepath: str
    type_idx: int
    dist_bin: int
    timestamp: datetime
    n_pieces: int

    @property
    def acquisition_date(self):
        return self.timestamp.date()


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

            dist_bin = (
                -1 if type_name.upper() == "IC" else parse_distance_bin(filepath)
            )
            if dist_bin >= 0:
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
                )
            )
            diagnostics["valid_files"] += 1

    entries.sort(key=lambda item: (item.type_idx, item.timestamp, item.filepath))
    return entries, diagnostics


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
