"""Deterministic source-grouped train, validation, and test ownership."""

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
SPLIT_SCHEMA = "source_grouped_stratified_split_v2"


@dataclass(frozen=True)
class SplitAssignment:
    """Partition indices aligned with a piece table and their ranking seed."""

    partition: np.ndarray
    seed: int

    def positions(self, name: str) -> np.ndarray:
        """Return table positions owned by one named partition."""
        index = PARTITION_NAMES.index(str(name))
        return np.flatnonzero(self.partition == index)


@dataclass(frozen=True)
class _SourceGroup:
    source_index: int
    relative_path: str
    positions: np.ndarray
    stratum_counts: np.ndarray

    @property
    def piece_count(self) -> int:
        return int(len(self.positions))


def _rank(seed: int, key: str) -> bytes:
    payload = f"{int(seed)}|{key}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _piece_stratum_key(
    table: PieceTable, position: int
) -> tuple[int, bool, int]:
    type_index = int(table.type_index[position])
    daylight = bool(table.daylight[position])
    distance_bin = -1 if type_index == 0 else int(table.distance_bin[position])
    return type_index, daylight, distance_bin


def _source_stratum_key(
    table: PieceTable, positions: np.ndarray
) -> tuple[int, int]:
    type_values = np.unique(table.type_index[positions])
    if len(type_values) != 1:
        raise ValueError("source file contains multiple type labels")
    type_index = int(type_values[0])
    if type_index == 0:
        return type_index, -1
    distance_values = np.unique(table.distance_bin[positions])
    if len(distance_values) != 1:
        raise ValueError("source file contains multiple distance labels")
    return type_index, int(distance_values[0])


def _piece_keys(table: PieceTable) -> list[str]:
    keys = [table.piece_key(position) for position in range(len(table))]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate piece identity")
    return keys


def _source_positions(table: PieceTable) -> dict[int, np.ndarray]:
    if len(table) == 0:
        return {}
    source_index = np.asarray(table.source_index)
    if source_index.ndim != 1 or len(source_index) != len(table):
        raise ValueError("source ownership must align with the piece table")
    if np.any(source_index < 0) or np.any(source_index >= len(table.sources)):
        raise ValueError("piece references an invalid source file")
    return {
        int(index): np.flatnonzero(source_index == index)
        for index in np.unique(source_index)
    }


def _piece_strata(
    table: PieceTable,
) -> dict[tuple[int, bool, int], list[int]]:
    strata: dict[tuple[int, bool, int], list[int]] = {}
    for position in range(len(table)):
        strata.setdefault(
            _piece_stratum_key(table, position), []
        ).append(position)
    return strata


def _source_groups(
    table: PieceTable,
) -> dict[tuple[int, int], list[_SourceGroup]]:
    observed_strata = sorted(_piece_strata(table))
    stratum_to_index = {
        key: index for index, key in enumerate(observed_strata)
    }
    grouped: dict[tuple[int, int], list[_SourceGroup]] = {}
    for source_index, positions in _source_positions(table).items():
        counts = np.zeros(len(observed_strata), dtype=np.int64)
        for position in positions:
            counts[stratum_to_index[_piece_stratum_key(table, int(position))]] += 1
        source = table.sources[source_index]
        group = _SourceGroup(
            source_index=source_index,
            relative_path=str(source.relative_path),
            positions=positions,
            stratum_counts=counts,
        )
        grouped.setdefault(
            _source_stratum_key(table, positions), []
        ).append(group)
    return grouped


def _assignment_cost(
    piece_counts: np.ndarray,
    stratum_counts: np.ndarray,
    target_pieces: np.ndarray,
    target_strata: np.ndarray,
) -> float:
    piece_scale = np.maximum(target_pieces, 1.0)
    stratum_scale = np.maximum(target_strata, 1.0)
    piece_error = np.square(
        (piece_counts - target_pieces) / piece_scale
    ).sum()
    stratum_error = np.square(
        (stratum_counts - target_strata) / stratum_scale
    ).sum()
    return float(piece_error + stratum_error)


def _assign_source_group(
    groups: list[_SourceGroup], seed: int
) -> dict[int, int]:
    if len(groups) < len(PARTITION_NAMES):
        return {group.source_index: TRAIN for group in groups}

    total_pieces = sum(group.piece_count for group in groups)
    total_strata = np.sum(
        [group.stratum_counts for group in groups], axis=0
    )
    ratios = np.asarray(DEFAULT_RATIOS, dtype=np.float64)
    target_pieces = ratios * float(total_pieces)
    target_strata = ratios[:, None] * total_strata[None, :]
    piece_counts = np.zeros(len(PARTITION_NAMES), dtype=np.float64)
    stratum_counts = np.zeros_like(target_strata)
    owners: dict[int, int] = {}

    ordered = sorted(
        groups,
        key=lambda group: (
            -group.piece_count,
            _rank(seed, group.relative_path),
            group.relative_path,
        ),
    )
    for order_index, group in enumerate(ordered):
        unfilled = [
            partition
            for partition in range(len(PARTITION_NAMES))
            if partition not in owners.values()
        ]
        remaining = len(ordered) - order_index
        candidates = (
            unfilled
            if unfilled and remaining == len(unfilled)
            else list(range(len(PARTITION_NAMES)))
        )
        scored: list[tuple[float, bytes, int]] = []
        for partition in candidates:
            candidate_pieces = piece_counts.copy()
            candidate_strata = stratum_counts.copy()
            candidate_pieces[partition] += group.piece_count
            candidate_strata[partition] += group.stratum_counts
            scored.append(
                (
                    _assignment_cost(
                        candidate_pieces,
                        candidate_strata,
                        target_pieces,
                        target_strata,
                    ),
                    _rank(
                        seed,
                        f"{group.relative_path}|{PARTITION_NAMES[partition]}",
                    ),
                    partition,
                )
            )
        _, _, selected = min(scored)
        owners[group.source_index] = selected
        piece_counts[selected] += group.piece_count
        stratum_counts[selected] += group.stratum_counts

    if set(owners.values()) != {TRAIN, VALIDATION, TEST}:
        raise RuntimeError("source-group assignment failed partition coverage")
    return owners


def _expected_partition(table: PieceTable, seed: int) -> np.ndarray:
    partition = np.full(len(table), TRAIN, dtype=np.uint8)
    grouped_sources = _source_groups(table)
    for stratum in sorted(grouped_sources):
        groups = grouped_sources[stratum]
        owners = _assign_source_group(groups, seed)
        for group in groups:
            partition[group.positions] = owners[group.source_index]
    return partition


def assign_piece_splits(table: PieceTable, seed: int) -> SplitAssignment:
    """Assign each complete source file by deterministic stratified ranking."""
    normalized_seed = int(seed)
    _piece_keys(table)
    return SplitAssignment(
        partition=_expected_partition(table, normalized_seed),
        seed=normalized_seed,
    )


def validate_piece_split(table: PieceTable, assignment: SplitAssignment) -> None:
    """Validate complete source ownership and deterministic reconstruction."""
    partition = np.asarray(assignment.partition)
    if partition.ndim != 1 or len(partition) != len(table):
        raise ValueError("missing ownership for one or more pieces")
    if not np.all(np.isin(partition, (TRAIN, VALIDATION, TEST))):
        raise ValueError("invalid partition ownership")

    _piece_keys(table)
    grouped_sources = _source_groups(table)
    for groups in grouped_sources.values():
        stratum_owners: set[int] = set()
        for group in groups:
            owners = set(partition[group.positions].tolist())
            if len(owners) != 1:
                raise ValueError("source file crosses partitions")
            stratum_owners.update(owners)
        if len(groups) < len(PARTITION_NAMES):
            if stratum_owners != {TRAIN}:
                raise ValueError(
                    "evaluation-limited source stratum must remain train-only"
                )
        elif stratum_owners != {TRAIN, VALIDATION, TEST}:
            raise ValueError(
                "evaluable source stratum must populate every partition"
            )

    expected = _expected_partition(table, int(assignment.seed))
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
    """Return compact hashes and source/piece ownership summaries."""
    validate_piece_split(table, assignment)
    keys = _piece_keys(table)
    partition = np.asarray(assignment.partition)

    partition_hashes: dict[str, str] = {}
    piece_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    source_positions = _source_positions(table)
    for partition_index, name in enumerate(PARTITION_NAMES):
        selected = np.flatnonzero(partition == partition_index)
        partition_hashes[name] = _hash_lines(
            [f"{keys[position]}\t{name}" for position in selected]
        )
        piece_counts[name] = int(len(selected))
        source_counts[name] = sum(
            bool(
                len(positions)
                and int(partition[int(positions[0])]) == partition_index
            )
            for positions in source_positions.values()
        )

    strata: dict[str, dict[str, int]] = {}
    grouped = _piece_strata(table)
    for key in sorted(grouped):
        positions = np.asarray(grouped[key], dtype=np.int64)
        owned = partition[positions]
        supporting_sources = np.unique(table.source_index[positions])
        source_support = int(len(supporting_sources))
        strata[_stratum_name(key)] = {
            name: int(np.count_nonzero(owned == partition_index))
            for partition_index, name in enumerate(PARTITION_NAMES)
        }
        strata[_stratum_name(key)]["source_support"] = source_support
        strata[_stratum_name(key)]["insufficient_source_groups"] = (
            source_support if source_support < len(PARTITION_NAMES) else 0
        )

    return {
        "schema": SPLIT_SCHEMA,
        "seed": int(assignment.seed),
        "ratios": list(DEFAULT_RATIOS),
        "manifest_hash": _hash_lines(keys),
        "partition_hashes": partition_hashes,
        "counts": piece_counts,
        "piece_counts": piece_counts,
        "source_counts": source_counts,
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
