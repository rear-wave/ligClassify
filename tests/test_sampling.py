from collections import Counter

import numpy as np
import pytest

from data.manifest import PieceTable, SourceRecord
from data.sampling import FiveClassSampler


def test_sampler_uses_exact_60_10_10_10_10_prior(piece_table):
    sampler = FiveClassSampler(
        piece_table,
        positions=range(len(piece_table)),
        num_samples=1000,
        seed=9,
    )

    requests = list(sampler)
    counts = Counter(piece_table.type_index[item.position] for item in requests)

    assert counts == {0: 600, 1: 100, 2: 100, 3: 100, 4: 100}
    assert len({item.augmentation_seed for item in requests}) == 1000


def test_sampler_balances_daylight_before_distance_bins(piece_table):
    distance = piece_table.distance_bin.copy()
    type_one = np.flatnonzero(piece_table.type_index == 1)
    day = type_one[piece_table.daylight[type_one]]
    night = type_one[~piece_table.daylight[type_one]]
    distance[day] = np.arange(len(day), dtype=np.int8) % 10
    distance[night] = 29
    piece_table.distance_bin = distance
    sampler = FiveClassSampler(
        piece_table,
        positions=range(len(piece_table)),
        num_samples=1000,
        seed=11,
    )

    requests = [
        item
        for item in sampler
        if int(piece_table.type_index[item.position]) == 1
    ]
    daylight_counts = Counter(
        bool(piece_table.daylight[item.position]) for item in requests
    )
    day_bin_counts = Counter(
        int(piece_table.distance_bin[item.position])
        for item in requests
        if piece_table.daylight[item.position]
    )

    assert abs(daylight_counts[False] - daylight_counts[True]) <= 1
    assert max(day_bin_counts.values()) - min(day_bin_counts.values()) <= 1


def test_sampler_allows_replacement_and_is_deterministic_by_seed_and_epoch(
    piece_table,
):
    positions = [
        int(np.flatnonzero(piece_table.type_index == index)[0])
        for index in range(5)
    ]
    first = FiveClassSampler(piece_table, positions, num_samples=20, seed=3)
    second = FiveClassSampler(piece_table, positions, num_samples=20, seed=3)

    epoch_one = list(first)

    assert epoch_one == list(second)
    assert len({item.position for item in epoch_one}) == 5
    first.set_epoch(2)
    second.set_epoch(2)
    assert list(first) == list(second)
    assert list(first) != epoch_one


def test_sampler_rejects_missing_types_invalid_arguments_and_distance_bins(
    piece_table,
):
    without_ic = np.flatnonzero(piece_table.type_index != 0)
    with pytest.raises(ValueError, match="missing required type"):
        FiveClassSampler(piece_table, without_ic, num_samples=10, seed=1)
    with pytest.raises(ValueError, match="num_samples"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=0, seed=1
        )
    with pytest.raises(ValueError, match="num_samples"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=10.5, seed=1
        )
    with pytest.raises(ValueError, match="seed"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=10, seed=1.5
        )
    with pytest.raises(ValueError, match="positions"):
        FiveClassSampler(
            piece_table,
            [0.5, *range(1, len(piece_table))],
            num_samples=10,
            seed=1,
        )

    sampler = FiveClassSampler(
        piece_table, range(len(piece_table)), num_samples=10, seed=1
    )
    with pytest.raises(ValueError, match="epoch"):
        sampler.set_epoch(0)

    invalid_distance = piece_table.distance_bin.copy()
    invalid_distance[np.flatnonzero(piece_table.type_index == 2)[0]] = 30
    piece_table.distance_bin = invalid_distance
    with pytest.raises(ValueError, match="distance_bin"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=10, seed=1
        )
