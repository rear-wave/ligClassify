"""Deterministic exact-interval cross-validation fold assignment."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from data.training_manifest import ManifestEntry


FOLD_COUNT = 3


def exact_condition_key(entry: ManifestEntry) -> tuple[int, bool, int, int]:
    """Return the validated type, daylight, and exact 100-km interval key."""
    low = entry.distance_low_km
    high = entry.distance_high_km
    if low is None or high is None or high - low != 100:
        raise ValueError("cross-validation requires 100-km interval labels")
    if low % 100 or not 0 <= low < high <= 3000:
        raise ValueError("distance intervals must be aligned inside 0-3000 km")
    if entry.is_daytime is None:
        raise ValueError("cross-validation requires a daylight label")
    return int(entry.type_idx), bool(entry.is_daytime), int(low), int(high)


def fold_train_holdout(
    folds: Mapping[int, Sequence[ManifestEntry]], held_out: int
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Return the two-fold training rows and one selected holdout fold."""
    holdout = list(folds[int(held_out)])
    train = [
        entry
        for fold_index, entries in sorted(folds.items())
        if fold_index != int(held_out)
        for entry in entries
    ]
    return train, holdout


def _entry_signature(
    entry: ManifestEntry,
) -> tuple[int, int, bool, int, int, int, str]:
    return (
        int(entry.type_idx),
        int(entry.dist_bin),
        bool(entry.is_daytime),
        int(entry.distance_low_km),
        int(entry.distance_high_km),
        int(entry.n_pieces),
        entry.timestamp.isoformat(),
    )


def _relative_identities(entries: Sequence[ManifestEntry]) -> dict[str, str]:
    absolute = [os.path.abspath(entry.filepath) for entry in entries]
    root = os.path.commonpath(absolute)
    if len(absolute) == 1 or os.path.isfile(root):
        root = os.path.dirname(root)
    return {
        os.path.normcase(path): Path(os.path.relpath(path, root)).as_posix()
        for path in absolute
    }


def assign_exact_folds(
    entries: Iterable[ManifestEntry],
    n_folds: int = FOLD_COUNT,
    seed: int = 42,
) -> dict[int, list[ManifestEntry]]:
    """Assign every source file to one of exactly three balanced folds."""
    entries = list(entries)
    if int(n_folds) != FOLD_COUNT:
        raise ValueError("this release contract requires exactly three folds")
    if not entries:
        raise ValueError("cross-validation requires at least one source file")
    identities = _relative_identities(entries)
    groups = defaultdict(list)
    seen = set()
    for entry in entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in seen:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        seen.add(path)
        groups[exact_condition_key(entry)].append(entry)
    folds = {index: [] for index in range(FOLD_COUNT)}
    global_pieces = [0] * FOLD_COUNT
    for condition in sorted(groups):
        condition_files = [0] * FOLD_COUNT
        condition_pieces = [0] * FOLD_COUNT
        ordered = sorted(
            groups[condition],
            key=lambda entry: (
                -int(entry.n_pieces),
                hashlib.sha256(
                    f"{int(seed)}|"
                    f"{identities[os.path.normcase(os.path.abspath(entry.filepath))]}"
                    .encode("utf-8")
                ).hexdigest(),
            ),
        )
        for entry in ordered:
            fold = min(
                range(FOLD_COUNT),
                key=lambda index: (
                    condition_files[index],
                    condition_pieces[index],
                    global_pieces[index],
                    index,
                ),
            )
            folds[fold].append(entry)
            condition_files[fold] += 1
            condition_pieces[fold] += int(entry.n_pieces)
            global_pieces[fold] += int(entry.n_pieces)
    return folds


def validate_fold_assignment(
    folds: Mapping[int, Sequence[ManifestEntry]],
    expected_entries: Iterable[ManifestEntry],
    n_folds: int = FOLD_COUNT,
) -> None:
    """Validate contiguous ownership, completeness, and immutable labels."""
    if set(folds) != set(range(int(n_folds))):
        raise ValueError("fold keys must be contiguous from zero")
    expected = {}
    for entry in expected_entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in expected:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        expected[path] = _entry_signature(entry)
    observed = set()
    for entries in folds.values():
        for entry in entries:
            path = os.path.normcase(os.path.abspath(entry.filepath))
            if path in observed:
                raise ValueError(f"duplicate fold owner: {entry.filepath}")
            if path not in expected:
                raise ValueError(f"unknown fold file: {entry.filepath}")
            if _entry_signature(entry) != expected[path]:
                raise ValueError(f"fold label mutation: {entry.filepath}")
            observed.add(path)
    missing = sorted(set(expected) - observed)
    if missing:
        raise ValueError(f"missing fold files: {len(missing)}")


def build_support_map(
    entries: Iterable[ManifestEntry],
    type_names: Sequence[str],
    minimum_files: int = 3,
) -> dict[str, dict[str, int | bool | str]]:
    """Aggregate exact condition support and flag cells below the file floor."""
    support = defaultdict(lambda: {"file_count": 0, "piece_count": 0})
    seen = set()
    for entry in entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in seen:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        seen.add(path)
        type_index, daylight, low_km, high_km = exact_condition_key(entry)
        type_name = type_names[type_index]
        name = (
            f"{type_name}/{'day' if daylight else 'night'}/"
            f"{low_km}-{high_km}km"
        )
        row = support[name]
        row.update({
            "type_index": type_index,
            "daylight": daylight,
            "low_km": low_km,
            "high_km": high_km,
        })
        row["file_count"] += 1
        row["piece_count"] += int(entry.n_pieces)
    result = {}
    for name, row in sorted(support.items()):
        row["status"] = (
            "supported"
            if row["file_count"] >= int(minimum_files)
            else "insufficient_support"
        )
        result[name] = dict(row)
    return result
