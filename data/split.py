"""Deterministic piece-level train, validation, and test ownership."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifest import PieceTable, TYPE_NAMES


TRAIN = 0
VALIDATION = 1
TEST = 2
PARTITION_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SCHEMA = "piece_stratified_split_v2"


@dataclass(frozen=True)
class SplitAssignment:
    """Partition indices aligned with a piece table and their ranking seed."""

    partition: np.ndarray
    seed: int

    def positions(self, name: str) -> np.ndarray:
        """Return table positions owned by one named partition."""
        index = PARTITION_NAMES.index(str(name))
        return np.flatnonzero(self.partition == index)


def _rank(seed: int, key: str) -> bytes:
    payload = f"{int(seed)}|{key}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _stratum_key(
    table: PieceTable, position: int
) -> tuple[int, bool, int]:
    type_index = int(table.type_index[position])
    daylight = bool(table.daylight[position])
    distance_bin = -1 if type_index == 0 else int(table.distance_bin[position])
    return type_index, daylight, distance_bin


def _piece_keys(table: PieceTable) -> list[str]:
    keys = [table.piece_key(position) for position in range(len(table))]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate piece identity")
    return keys


def _piece_strata(
    table: PieceTable,
) -> dict[tuple[int, bool, int], list[int]]:
    strata: dict[tuple[int, bool, int], list[int]] = {}
    for position in range(len(table)):
        strata.setdefault(_stratum_key(table, position), []).append(position)
    return strata


def _partition_counts(piece_count: int) -> tuple[int, int, int]:
    if piece_count < len(PARTITION_NAMES):
        return piece_count, 0, 0
    target = np.asarray(DEFAULT_RATIOS) * piece_count
    counts = np.maximum(np.floor(target).astype(np.int64), 1)
    while int(counts.sum()) < piece_count:
        deficits = target - counts
        counts[int(np.argmax(deficits))] += 1
    while int(counts.sum()) > piece_count:
        candidates = np.flatnonzero(counts > 1)
        excess = counts[candidates] - target[candidates]
        counts[int(candidates[int(np.argmax(excess))])] -= 1
    return tuple(int(value) for value in counts)


def _expected_partition(table: PieceTable, seed: int) -> np.ndarray:
    keys = _piece_keys(table)
    partition = np.full(len(table), TRAIN, dtype=np.uint8)
    strata = _piece_strata(table)
    for stratum in sorted(strata):
        positions = strata[stratum]
        ordered = sorted(
            positions,
            key=lambda position: (
                _rank(seed, keys[position]),
                keys[position],
            ),
        )
        counts = _partition_counts(len(ordered))
        offset = 0
        for owner, count in enumerate(counts):
            selected = ordered[offset : offset + count]
            partition[selected] = owner
            offset += count
    return partition


def assign_piece_splits(table: PieceTable, seed: int) -> SplitAssignment:
    """Assign pieces by stable ranking within label/daylight/distance strata."""
    normalized_seed = int(seed)
    return SplitAssignment(
        partition=_expected_partition(table, normalized_seed),
        seed=normalized_seed,
    )


def validate_piece_split(table: PieceTable, assignment: SplitAssignment) -> None:
    """Validate exhaustive piece ownership and deterministic reconstruction."""
    partition = np.asarray(assignment.partition)
    if partition.ndim != 1 or len(partition) != len(table):
        raise ValueError("missing ownership for one or more pieces")
    if not np.all(np.isin(partition, (TRAIN, VALIDATION, TEST))):
        raise ValueError("invalid partition ownership")
    _piece_keys(table)
    for positions in _piece_strata(table).values():
        owners = set(partition[np.asarray(positions)].tolist())
        expected = (
            {TRAIN}
            if len(positions) < len(PARTITION_NAMES)
            else {TRAIN, VALIDATION, TEST}
        )
        if owners != expected:
            raise ValueError("piece stratum does not populate expected partitions")
    expected_partition = _expected_partition(table, int(assignment.seed))
    if not np.array_equal(partition, expected_partition):
        raise ValueError("piece ownership does not match seed")


def _hash_lines(lines: list[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stratum_name(key: tuple[int, bool, int]) -> str:
    type_index, daylight, distance_bin = key
    if not 0 <= type_index < len(TYPE_NAMES):
        raise ValueError(f"invalid type index: {type_index}")
    light_name = "day" if daylight else "night"
    if type_index == 0 or distance_bin < 0:
        return f"{TYPE_NAMES[type_index]}|{light_name}"
    low = distance_bin * 100
    return f"{TYPE_NAMES[type_index]}|{light_name}|{low}-{low + 100}km"


def split_artifact(
    table: PieceTable, assignment: SplitAssignment
) -> dict[str, object]:
    """Return compact hashes and piece-level ownership summaries."""
    validate_piece_split(table, assignment)
    keys = _piece_keys(table)
    partition = np.asarray(assignment.partition)
    partition_hashes: dict[str, str] = {}
    piece_counts: dict[str, int] = {}
    represented_source_counts: dict[str, int] = {}
    for partition_index, name in enumerate(PARTITION_NAMES):
        selected = np.flatnonzero(partition == partition_index)
        partition_hashes[name] = _hash_lines(
            [f"{keys[position]}\t{name}" for position in selected]
        )
        piece_counts[name] = int(len(selected))
        represented_source_counts[name] = int(
            len(np.unique(table.source_index[selected]))
        )

    strata: dict[str, dict[str, int]] = {}
    for key, raw_positions in sorted(_piece_strata(table).items()):
        positions = np.asarray(raw_positions, dtype=np.int64)
        owned = partition[positions]
        counts = {
            name: int(np.count_nonzero(owned == partition_index))
            for partition_index, name in enumerate(PARTITION_NAMES)
        }
        counts["piece_support"] = int(len(positions))
        counts["insufficient_pieces"] = (
            int(len(positions))
            if len(positions) < len(PARTITION_NAMES)
            else 0
        )
        strata[_stratum_name(key)] = counts

    return {
        "schema": SPLIT_SCHEMA,
        "seed": int(assignment.seed),
        "ratios": list(DEFAULT_RATIOS),
        "manifest_hash": _hash_lines(keys),
        "partition_hashes": partition_hashes,
        "counts": piece_counts,
        "piece_counts": piece_counts,
        "represented_source_counts": represented_source_counts,
        "strata": strata,
    }


def write_split_json(
    path: str | os.PathLike[str],
    artifact: Mapping[str, object],
) -> None:
    """Write a compact deterministic split artifact as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
