from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from data.cross_validation import (
    assign_exact_folds,
    build_support_map,
    exact_condition_key,
    fold_train_holdout,
    validate_fold_assignment,
)
from data.split_artifacts import make_fold_manifest
from data.training_manifest import ManifestEntry


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def make_entries(lows, files_per_condition, pieces=(50, 100, 150)):
    entries = []
    for low in lows:
        for file_index in range(files_per_condition):
            entries.append(ManifestEntry(
                filepath=f"NCG/day/{low}-{low + 100}km/file-{file_index}.lig",
                type_idx=0,
                dist_bin=low // 100,
                timestamp=datetime(2020, 1, 1) + timedelta(seconds=len(entries)),
                n_pieces=pieces[file_index % len(pieces)],
                distance_low_km=low,
                distance_high_km=low + 100,
                is_daytime=True,
            ))
    return entries


def fold_paths(folds):
    return {
        fold: sorted(entry.filepath for entry in entries)
        for fold, entries in sorted(folds.items())
    }


def test_exact_bins_are_distinct_and_three_file_cells_cover_every_fold():
    entries = make_entries(
        lows=(300, 400), files_per_condition=3, pieces=(50, 100, 150)
    )
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    for low in (300, 400):
        assert [
            sum(entry.distance_low_km == low for entry in folds[fold])
            for fold in range(3)
        ] == [1, 1, 1]


def test_sparse_condition_is_reported_without_splitting_a_file():
    entries = make_entries(lows=(300,), files_per_condition=2)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    support = build_support_map(entries, TYPE_NAMES, minimum_files=3)
    assert sum(len(rows) for rows in folds.values()) == 2
    assert support["NCG/day/300-400km"]["file_count"] == 2
    assert support["NCG/day/300-400km"]["status"] == "insufficient_support"


def test_fold_assignment_is_order_independent_and_complete():
    entries = make_entries(lows=(0, 100, 200), files_per_condition=5)
    first = assign_exact_folds(entries, n_folds=3, seed=11)
    second = assign_exact_folds(reversed(entries), n_folds=3, seed=11)
    assert fold_paths(first) == fold_paths(second)
    validate_fold_assignment(first, entries, n_folds=3)


@pytest.mark.parametrize(
    "changes",
    [
        {"distance_low_km": None, "distance_high_km": None},
        {"distance_low_km": 0, "distance_high_km": 300},
        {"distance_low_km": 50, "distance_high_km": 150},
    ],
)
def test_exact_condition_rejects_non_exact_or_misaligned_intervals(changes):
    entry = replace(make_entries((0,), 1)[0], **changes)

    with pytest.raises(ValueError):
        exact_condition_key(entry)


def test_exact_condition_rejects_missing_daylight_label():
    entry = replace(make_entries((0,), 1)[0], is_daytime=None)

    with pytest.raises(ValueError, match="daylight"):
        exact_condition_key(entry)


def test_assign_exact_folds_rejects_empty_population_clearly():
    with pytest.raises(ValueError, match="at least one"):
        assign_exact_folds([])


def test_validation_rejects_dist_bin_label_mutation():
    entries = make_entries((0,), 3)
    folds = assign_exact_folds(entries, seed=3)
    mutated = {index: list(rows) for index, rows in folds.items()}
    mutated[0][0] = replace(mutated[0][0], dist_bin=99)

    with pytest.raises(ValueError, match="fold label mutation"):
        validate_fold_assignment(mutated, entries)


def test_fold_train_holdout_keeps_source_owners_disjoint_and_complete():
    entries = make_entries((0, 100), 3)
    folds = assign_exact_folds(entries, seed=5)

    train, holdout = fold_train_holdout(folds, held_out=1)

    train_paths = {entry.filepath for entry in train}
    holdout_paths = {entry.filepath for entry in holdout}
    assert train_paths.isdisjoint(holdout_paths)
    assert train_paths | holdout_paths == {entry.filepath for entry in entries}


def test_fold_manifest_serializes_canonical_path_order():
    entries = make_entries((0, 100), 5)
    folds = assign_exact_folds(entries, seed=13)
    scrambled = {
        index: list(reversed(rows)) for index, rows in folds.items()
    }

    first = make_fold_manifest(folds, root=".", seed=13)
    second = make_fold_manifest(scrambled, root=".", seed=13)

    assert first == second
    assert first["schema"] == "file_isolated_exact_interval_cv_v1"
    assert first["fold_count"] == 3
    assert set(first["holdout_hashes"]) == {"0", "1", "2"}
    assert set(first["train_hashes"]) == {"0", "1", "2"}
    for rows in first["folds"].values():
        paths = [row["path"] for row in rows]
        assert paths == sorted(paths)
