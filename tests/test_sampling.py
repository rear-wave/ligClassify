from collections import Counter

import numpy as np
import pytest

from data.manifest import PieceTable, SourceRecord
from data.sampling import DistanceExpertSampler, FiveClassSampler


def test_sampler_uses_equal_prior_without_replacement(piece_table):
    sampler = FiveClassSampler(
        piece_table,
        positions=range(len(piece_table)),
        num_samples=100,
        seed=9,
    )

    requests = list(sampler)
    counts = Counter(piece_table.type_index[item.position] for item in requests)

    assert counts == {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}
    assert len({item.position for item in requests}) == 100
    assert all(len(item.augmentation_seeds) == 2 for item in requests)
    assert all(
        item.augmentation_seeds[0] != item.augmentation_seeds[1]
        for item in requests
    )
    assert len(
        {
            seed
            for item in requests
            for seed in item.augmentation_seeds
        }
    ) == 200
    for start in range(0, len(requests), 5):
        assert {
            int(piece_table.type_index[item.position])
            for item in requests[start : start + 5]
        } == {0, 1, 2, 3, 4}


def test_classification_only_sampler_accepts_missing_distance_bins(piece_table):
    piece_table.distance_bin[:] = -1
    sampler = FiveClassSampler(
        piece_table,
        positions=range(len(piece_table)),
        num_samples=100,
        seed=9,
        balance_distance=False,
    )

    requests = list(sampler)
    counts = Counter(piece_table.type_index[item.position] for item in requests)

    assert counts == {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}


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
        num_samples=200,
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


def test_sampler_is_no_replacement_and_deterministic_by_seed_and_epoch(
    piece_table,
):
    positions = range(len(piece_table))
    first = FiveClassSampler(piece_table, positions, num_samples=100, seed=3)
    second = FiveClassSampler(piece_table, positions, num_samples=100, seed=3)

    epoch_one = list(first)

    assert epoch_one == list(second)
    assert len({item.position for item in epoch_one}) == len(epoch_one)
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
    with pytest.raises(ValueError, match="divisible by five"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=11, seed=1
        )
    with pytest.raises(ValueError, match="without replacement"):
        FiveClassSampler(
            piece_table, range(len(piece_table)), num_samples=205, seed=1
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


def test_distance_expert_sampler_draws_only_requested_type(piece_table):
    sampler = DistanceExpertSampler(
        piece_table,
        range(len(piece_table)),
        type_index=1,
        num_samples=120,
        seed=7,
    )

    draws = list(sampler)

    assert len(draws) == 120
    assert {
        int(piece_table.type_index[item.position]) for item in draws
    } == {1}
    assert len(
        {
            seed
            for item in draws
            for seed in item.augmentation_seeds
        }
    ) == 240


def test_distance_expert_sampler_balances_observed_cells(piece_table):
    type_two = np.flatnonzero(piece_table.type_index == 2)
    piece_table.distance_bin[type_two] = (
        np.arange(len(type_two), dtype=np.int8) % 3
    )
    sampler = DistanceExpertSampler(
        piece_table,
        range(len(piece_table)),
        type_index=2,
        num_samples=120,
        seed=7,
    )

    cells = Counter(
        (
            bool(piece_table.daylight[item.position]),
            int(piece_table.distance_bin[item.position]),
        )
        for item in sampler
    )

    assert len(cells) == 6
    assert max(cells.values()) - min(cells.values()) <= 1


@pytest.mark.parametrize("type_index", [0, 5, True])
def test_distance_expert_sampler_rejects_invalid_type(
    piece_table, type_index
):
    with pytest.raises(ValueError, match="type_index"):
        DistanceExpertSampler(
            piece_table,
            range(len(piece_table)),
            type_index=type_index,
            num_samples=10,
            seed=1,
        )
