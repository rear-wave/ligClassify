import importlib
from datetime import datetime, timedelta

import pytest


def modules():
    manifest = importlib.import_module("data.training_manifest")
    try:
        group_split = importlib.import_module("data.group_split")
    except ModuleNotFoundError:
        pytest.fail("data.group_split is not implemented")
    return manifest, group_split


def make_entries(files_per_stratum=5):
    manifest, _ = modules()
    entries = []
    counter = 0
    for type_idx in range(2):
        for is_daytime in (False, True):
            for low_km in (0, 600, 1800):
                for file_index in range(files_per_stratum):
                    entries.append(manifest.ManifestEntry(
                        filepath=f"type-{type_idx}-day-{is_daytime}-"
                        f"range-{low_km}-file-{file_index}.lig",
                        type_idx=type_idx,
                        dist_bin=low_km // 100,
                        timestamp=datetime(2020, 1, 1) + timedelta(seconds=counter),
                        n_pieces=10 + file_index,
                        distance_low_km=low_km,
                        distance_high_km=low_km + 100,
                        is_daytime=is_daytime,
                    ))
                    counter += 1
    return entries


def stratum(entry):
    if entry.distance_low_km < 600:
        band = 0
    elif entry.distance_low_km < 1800:
        band = 1
    else:
        band = 2
    return entry.type_idx, entry.is_daytime, band


def test_group_split_never_shares_a_source_file():
    _, group_split = modules()

    splits = group_split.group_stratified_split(
        make_entries(), val_fraction=0.2, test_fraction=0.2, seed=7
    )

    group_split.validate_group_split(splits)
    owners = {}
    for split_name, selected in splits.items():
        for entry in selected:
            assert owners.setdefault(entry.filepath, split_name) == split_name


def test_group_split_is_deterministic_and_preserves_strata():
    _, group_split = modules()
    entries = make_entries()

    first = group_split.group_stratified_split(entries, 0.2, 0.2, seed=11)
    second = group_split.group_stratified_split(
        list(reversed(entries)), 0.2, 0.2, seed=11
    )

    expected = {stratum(entry) for entry in entries}
    for split_name in ("train", "val", "test"):
        assert {entry.filepath for entry in first[split_name]} == {
            entry.filepath for entry in second[split_name]
        }
        assert {stratum(entry) for entry in first[split_name]} == expected


def test_group_split_rejects_duplicate_file_entries():
    _, group_split = modules()
    entry = make_entries()[0]

    with pytest.raises(ValueError, match="duplicate source file"):
        group_split.group_stratified_split([entry, entry], 0.2, 0.2, seed=1)


def test_group_split_balances_piece_counts_inside_a_stratum():
    manifest, group_split = modules()
    sizes = [1000, 800, 600, 400, 200, 100, 100, 100, 100, 100]
    entries = [
        manifest.ManifestEntry(
            filepath=f"file-{index}.lig",
            type_idx=0,
            dist_bin=0,
            timestamp=datetime(2020, 1, 1) + timedelta(seconds=index),
            n_pieces=size,
            distance_low_km=0,
            distance_high_km=100,
            is_daytime=True,
        )
        for index, size in enumerate(sizes)
    ]

    splits = group_split.group_stratified_split(entries, 0.2, 0.2, seed=3)

    total = sum(sizes)
    targets = {"train": total * 0.6, "val": total * 0.2, "test": total * 0.2}
    errors = {
        name: abs(sum(entry.n_pieces for entry in selected) - targets[name])
        for name, selected in splits.items()
    }
    assert sum(errors.values()) <= 500, errors
