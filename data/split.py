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


def _partition_counts(size: int) -> tuple[int, int, int]:
    if size < 3:
        return size, 0, 0
    validation = max(1, int(size * DEFAULT_RATIOS[1]))
    test = max(1, int(size * DEFAULT_RATIOS[2]))
    train = size - validation - test
    if train < 1:
        raise ValueError("stratum cannot populate training")
    return train, validation, test


def _stratum_key(table: PieceTable, position: int) -> tuple[int, bool, int]:
    type_index = int(table.type_index[position])
    daylight = bool(table.daylight[position])
    distance_bin = -1 if type_index == 0 else int(table.distance_bin[position])
    return type_index, daylight, distance_bin


def _piece_keys(table: PieceTable) -> list[str]:
    keys = [table.piece_key(position) for position in range(len(table))]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate piece identity")
    return keys


def _group_positions(
    table: PieceTable,
) -> dict[tuple[int, bool, int], list[int]]:
    strata: dict[tuple[int, bool, int], list[int]] = {}
    for position in range(len(table)):
        strata.setdefault(_stratum_key(table, position), []).append(position)
    return strata


def _expected_partition(
    table: PieceTable, seed: int, keys: list[str]
) -> np.ndarray:
    partition = np.empty(len(table), dtype=np.uint8)
    for positions in _group_positions(table).values():
        ranked = sorted(
            positions,
            key=lambda position: (_rank(seed, keys[position]), keys[position]),
        )
        train_count, validation_count, _ = _partition_counts(len(ranked))
        validation_end = train_count + validation_count
        partition[ranked[:train_count]] = TRAIN
        partition[ranked[train_count:validation_end]] = VALIDATION
        partition[ranked[validation_end:]] = TEST
    return partition


def assign_piece_splits(table: PieceTable, seed: int) -> SplitAssignment:
    """Assign pieces within each label stratum by stable seeded SHA-256 rank."""
    normalized_seed = int(seed)
    keys = _piece_keys(table)
    return SplitAssignment(
        partition=_expected_partition(table, normalized_seed, keys),
        seed=normalized_seed,
    )


def validate_piece_split(table: PieceTable, assignment: SplitAssignment) -> None:
    """Validate complete ownership and its exact deterministic reconstruction."""
    partition = np.asarray(assignment.partition)
    if partition.ndim != 1 or len(partition) != len(table):
        raise ValueError("missing ownership for one or more pieces")
    if not np.all(np.isin(partition, (TRAIN, VALIDATION, TEST))):
        raise ValueError("invalid partition ownership")

    keys = _piece_keys(table)
    for positions in _group_positions(table).values():
        owned = partition[np.asarray(positions, dtype=np.int64)]
        if len(positions) >= 3 and set(owned.tolist()) != {
            TRAIN,
            VALIDATION,
            TEST,
        }:
            raise ValueError(
                "evaluation stratum must populate every partition"
            )
        if len(positions) < 3 and np.any(owned != TRAIN):
            raise ValueError("small stratum must remain train-only")

    expected = _expected_partition(table, int(assignment.seed), keys)
    if not np.array_equal(partition, expected):
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
    """Return compact JSON-safe hashes and per-stratum ownership counts."""
    validate_piece_split(table, assignment)
    keys = _piece_keys(table)
    partition = np.asarray(assignment.partition)

    partition_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for partition_index, name in enumerate(PARTITION_NAMES):
        selected = np.flatnonzero(partition == partition_index)
        partition_hashes[name] = _hash_lines(
            [f"{keys[position]}\t{name}" for position in selected]
        )
        counts[name] = int(len(selected))

    strata: dict[str, dict[str, int]] = {}
    grouped = _group_positions(table)
    for key in sorted(grouped):
        positions = np.asarray(grouped[key], dtype=np.int64)
        owned = partition[positions]
        strata[_stratum_name(key)] = {
            name: int(np.count_nonzero(owned == partition_index))
            for partition_index, name in enumerate(PARTITION_NAMES)
        }
        strata[_stratum_name(key)]["insufficient_for_evaluation"] = (
            len(positions) if len(positions) < 3 else 0
        )

    return {
        "schema": "piece_stratified_split_v1",
        "seed": int(assignment.seed),
        "ratios": list(DEFAULT_RATIOS),
        "manifest_hash": _hash_lines(keys),
        "partition_hashes": partition_hashes,
        "counts": counts,
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
