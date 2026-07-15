"""Deterministic source-file-isolated dataset splitting."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict

import numpy as np


SPLIT_NAMES = ("train", "val", "test")
COARSE_DISTANCE_EDGES_KM = (0, 300, 600, 1200, 1700, 2400, 3001)


def coarse_distance_band(entry) -> int:
    """Return a stable coarse-distance stratum for a manifest entry."""
    if entry.distance_low_km is None or entry.distance_high_km is None:
        return -1
    midpoint = (entry.distance_low_km + entry.distance_high_km) / 2.0
    return int(np.searchsorted(COARSE_DISTANCE_EDGES_KM, midpoint, side="right") - 1)


def _identity(filepath: str) -> str:
    return os.path.normcase(os.path.abspath(filepath))


def _stable_key(entry, seed: int):
    identity = _identity(entry.filepath)
    digest = hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()
    return digest, identity


def _split_counts(size: int, val_fraction: float, test_fraction: float):
    fractions = np.asarray(
        [1.0 - val_fraction - test_fraction, val_fraction, test_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train, validation, and test fractions must be non-negative")
    raw = fractions * size
    counts = np.floor(raw).astype(np.int64)
    remaining = size - int(counts.sum())
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(int(value) for value in counts)


def _balanced_group_assignment(group, val_fraction, test_fraction, seed):
    fractions = np.asarray(
        [1.0 - val_fraction - test_fraction, val_fraction, test_fraction],
        dtype=np.float64,
    )
    capacities = _split_counts(len(group), val_fraction, test_fraction)
    target_pieces = fractions * sum(entry.n_pieces for entry in group)
    piece_totals = np.zeros(3, dtype=np.float64)
    assigned = [[] for _ in SPLIT_NAMES]
    ordered = sorted(
        group,
        key=lambda entry: (-entry.n_pieces, _stable_key(entry, seed)),
    )
    for entry in ordered:
        candidates = [
            index
            for index, capacity in enumerate(capacities)
            if len(assigned[index]) < capacity
        ]
        scored = []
        for index in candidates:
            proposed = piece_totals.copy()
            proposed[index] += entry.n_pieces
            cost = float(np.square(proposed - target_pieces).sum())
            scored.append((cost, index))
        _, selected = min(scored)
        assigned[selected].append(entry)
        piece_totals[selected] += entry.n_pieces
    return assigned


def group_stratified_split(
    entries,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """Split whole source files within type/daylight/coarse-distance strata."""
    entries = list(entries)
    identities = [_identity(entry.filepath) for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source file in manifest")

    strata = defaultdict(list)
    for entry in entries:
        key = (entry.type_idx, entry.is_daytime, coarse_distance_band(entry))
        strata[key].append(entry)

    splits = {name: [] for name in SPLIT_NAMES}
    for key in sorted(strata, key=repr):
        assigned = _balanced_group_assignment(
            strata[key], val_fraction, test_fraction, seed
        )
        for split_name, selected in zip(SPLIT_NAMES, assigned):
            splits[split_name].extend(selected)

    for selected in splits.values():
        selected.sort(key=lambda entry: (entry.type_idx, entry.timestamp, entry.filepath))
    validate_group_split(splits)
    return splits


def validate_group_split(splits) -> None:
    """Reject missing split names or any source file assigned more than once."""
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"split names must be {SPLIT_NAMES}")
    owners = {}
    for split_name in SPLIT_NAMES:
        for entry in splits[split_name]:
            identity = _identity(entry.filepath)
            previous = owners.setdefault(identity, split_name)
            if previous != split_name:
                raise ValueError(
                    f"source file appears in {previous} and {split_name}: "
                    f"{entry.filepath}"
                )
