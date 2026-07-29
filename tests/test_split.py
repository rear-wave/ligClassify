import json

import numpy as np
import pytest

from data.manifest import PieceTable
from data.split import (
    TEST,
    TRAIN,
    VALIDATION,
    SplitAssignment,
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
    write_split_json,
)


def _reorder_table(table, positions):
    positions = np.asarray(positions, dtype=np.int64)
    return PieceTable(
        root=table.root,
        sources=table.sources,
        source_index=table.source_index[positions],
        piece_index=table.piece_index[positions],
        type_index=table.type_index[positions],
        distance_bin=table.distance_bin[positions],
        daylight=table.daylight[positions],
        timestamp_seconds=table.timestamp_seconds[positions],
    )


def _ownership(table, assignment):
    return {
        table.piece_key(position): int(assignment.partition[position])
        for position in range(len(table))
    }


def test_piece_split_is_deterministic_balanced_and_crosses_files(
    single_stratum_table,
):
    first = assign_piece_splits(single_stratum_table, seed=42)
    second = assign_piece_splits(single_stratum_table, seed=42)

    assert np.array_equal(first.partition, second.partition)
    validate_piece_split(single_stratum_table, first)
    counts = np.bincount(first.partition, minlength=3)
    assert counts.tolist() == [14, 3, 3]
    assert np.array_equal(
        first.positions("validation"),
        np.flatnonzero(first.partition == VALIDATION),
    )

    positions_from_source_zero = np.flatnonzero(
        single_stratum_table.source_index == 0
    )
    assert set(first.partition[positions_from_source_zero]) == {
        TRAIN,
        VALIDATION,
        TEST,
    }


def test_split_artifact_is_compact(single_stratum_table):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    artifact = split_artifact(single_stratum_table, assignment)

    assert "piece_rows" not in artifact
    assert artifact["schema"] == "piece_stratified_split_v1"
    assert artifact["ratios"] == [0.70, 0.15, 0.15]
    assert set(artifact["partition_hashes"]) == {
        "train",
        "validation",
        "test",
    }
    assert artifact["counts"] == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    encoded = json.dumps(artifact)
    assert str(single_stratum_table.root) not in encoded
    assert "one.lig#" not in encoded


def test_input_order_and_timestamps_do_not_change_piece_ownership(piece_table):
    baseline = assign_piece_splits(piece_table, seed=42)
    permutation = np.arange(len(piece_table) - 1, -1, -1)
    reordered = _reorder_table(piece_table, permutation)
    reordered.timestamp_seconds[:] = np.arange(len(reordered)) + 4_000_000_000

    changed_order = assign_piece_splits(reordered, seed=42)

    assert _ownership(piece_table, baseline) == _ownership(
        reordered, changed_order
    )
    baseline_artifact = split_artifact(piece_table, baseline)
    reordered_artifact = split_artifact(reordered, changed_order)
    assert baseline_artifact["manifest_hash"] == reordered_artifact[
        "manifest_hash"
    ]
    assert baseline_artifact["partition_hashes"] == reordered_artifact[
        "partition_hashes"
    ]


def test_different_seeds_change_piece_ownership(single_stratum_table):
    first = assign_piece_splits(single_stratum_table, seed=1)
    second = assign_piece_splits(single_stratum_table, seed=2)

    assert _ownership(single_stratum_table, first) != _ownership(
        single_stratum_table, second
    )


def test_ic_ignores_distance_and_non_ic_uses_exact_distance_strata(piece_table):
    changed_distance = _reorder_table(piece_table, np.arange(len(piece_table)))
    ic = changed_distance.type_index == 0
    changed_distance.distance_bin[ic] = 29
    assignment = assign_piece_splits(changed_distance, seed=42)

    validate_piece_split(changed_distance, assignment)
    artifact = split_artifact(changed_distance, assignment)
    assert len(artifact["strata"]) == 10


def test_classification_only_split_strata_do_not_include_distance(piece_table):
    classification_table = _reorder_table(
        piece_table, np.arange(len(piece_table))
    )
    classification_table.distance_bin[:] = -1

    assignment = assign_piece_splits(classification_table, seed=42)
    validate_piece_split(classification_table, assignment)
    artifact = split_artifact(classification_table, assignment)

    assert all("km" not in name for name in artifact["strata"])
    assert set(artifact["strata"]) == {
        f"{type_name}|{daylight}"
        for type_name in ("IC", "NCG", "NNBE", "PCG", "PNBE")
        for daylight in ("day", "night")
    }


def test_two_piece_stratum_is_train_only_and_reported(single_stratum_table):
    tiny = _reorder_table(single_stratum_table, [0, 1])

    assignment = assign_piece_splits(tiny, seed=42)
    artifact = split_artifact(tiny, assignment)

    assert assignment.partition.tolist() == [TRAIN, TRAIN]
    only_stratum = next(iter(artifact["strata"].values()))
    assert only_stratum == {
        "train": 2,
        "validation": 0,
        "test": 0,
        "insufficient_for_evaluation": 2,
    }


def test_three_piece_stratum_populates_every_partition(single_stratum_table):
    smallest_evaluable = _reorder_table(single_stratum_table, [0, 1, 2])

    assignment = assign_piece_splits(smallest_evaluable, seed=42)

    assert sorted(assignment.partition.tolist()) == [TRAIN, VALIDATION, TEST]
    validate_piece_split(smallest_evaluable, assignment)


def test_validation_rejects_missing_ownership(single_stratum_table):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    missing = SplitAssignment(assignment.partition[:-1], assignment.seed)

    with pytest.raises(ValueError, match="missing ownership"):
        validate_piece_split(single_stratum_table, missing)


def test_validation_rejects_invalid_ownership(single_stratum_table):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    invalid_partition = assignment.partition.copy()
    invalid_partition[0] = 3

    with pytest.raises(ValueError, match="invalid partition"):
        validate_piece_split(
            single_stratum_table,
            SplitAssignment(invalid_partition, assignment.seed),
        )


def test_validation_rejects_duplicate_piece_ownership(single_stratum_table):
    duplicate = _reorder_table(single_stratum_table, np.arange(20))
    duplicate.piece_index[1] = duplicate.piece_index[0]
    assignment = SplitAssignment(
        np.full(len(duplicate), TRAIN, dtype=np.uint8), seed=42
    )

    with pytest.raises(ValueError, match="duplicate piece identity"):
        validate_piece_split(duplicate, assignment)


def test_validation_recomputes_expected_seeded_ownership(single_stratum_table):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    changed = assignment.partition.copy()
    train_position = int(np.flatnonzero(changed == TRAIN)[0])
    validation_position = int(np.flatnonzero(changed == VALIDATION)[0])
    changed[train_position], changed[validation_position] = (
        changed[validation_position],
        changed[train_position],
    )

    with pytest.raises(ValueError, match="does not match seed"):
        validate_piece_split(
            single_stratum_table,
            SplitAssignment(changed, assignment.seed),
        )


def test_write_split_json_writes_the_compact_artifact(
    tmp_path, single_stratum_table
):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    artifact = split_artifact(single_stratum_table, assignment)
    output = tmp_path / "run" / "split.json"

    write_split_json(output, artifact)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
