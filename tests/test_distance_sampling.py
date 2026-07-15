import numpy as np
import pytest

from data import distance_sampling
from data.distance_sampling import ConditionBalancedSampler, HierarchicalDistanceSampler


def make_sampler(seed=7):
    return HierarchicalDistanceSampler(
        type_labels=np.array([0, 1, 1, 1, 2, 2, 2, 2]),
        dist_labels=np.array([-1, 0, 0, 2, 0, 1, 1, 1]),
        date_ids=np.array([1, 1, 2, 2, 1, 1, 2, 2]),
        file_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        num_samples=6,
        max_samples_per_file=1,
        seed=seed,
    )


def test_sampler_uses_only_labelled_non_ic_and_caps_files():
    sampler = make_sampler()

    indices = list(sampler)

    assert len(indices) == 6
    assert all(indices.count(index) <= 1 for index in set(indices))
    assert all(index != 0 for index in indices)


def test_sampler_is_reproducible_and_changes_by_epoch():
    first = make_sampler()
    second = make_sampler()

    assert list(first) == list(second)
    second.set_epoch(1)
    assert list(first) != list(second)


def test_sampler_ignores_missing_bins_without_error():
    sampler = make_sampler()

    assert set(list(sampler)).issubset(set(range(1, 8)))


@pytest.mark.parametrize("field", ["num_samples", "max_samples_per_file"])
def test_sampler_rejects_non_positive_limits(field):
    kwargs = {
        "type_labels": np.array([1]),
        "dist_labels": np.array([0]),
        "date_ids": np.array([1]),
        "file_ids": np.array([1]),
        "num_samples": 1,
        "max_samples_per_file": 1,
    }
    kwargs[field] = 0

    with pytest.raises(ValueError, match=field):
        HierarchicalDistanceSampler(**kwargs)


def test_sampler_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="same length"):
        HierarchicalDistanceSampler(
            type_labels=np.array([1, 2]),
            dist_labels=np.array([0]),
            date_ids=np.array([1, 2]),
            file_ids=np.array([1, 2]),
            num_samples=1,
            max_samples_per_file=1,
        )


def test_replacement_sampler_balances_types_and_bins_exactly():
    type_labels = np.array([1] * 20 + [1] * 2 + [2] * 5 + [2] * 30)
    dist_labels = np.array([0] * 20 + [1] * 2 + [0] * 5 + [1] * 30)
    count = len(type_labels)
    sampler = HierarchicalDistanceSampler(
        type_labels=type_labels,
        dist_labels=dist_labels,
        date_ids=np.arange(count) % 3,
        file_ids=np.arange(count),
        num_samples=120,
        max_samples_per_file=1,
        seed=19,
        replacement=True,
    )

    selected = np.asarray(list(sampler))
    assert len(selected) == 120
    type_counts = [np.count_nonzero(type_labels[selected] == value) for value in (1, 2)]
    assert max(type_counts) - min(type_counts) <= 1
    for value in (1, 2):
        mask = type_labels[selected] == value
        bin_counts = [np.count_nonzero(dist_labels[selected][mask] == b) for b in (0, 1)]
        assert max(bin_counts) - min(bin_counts) <= 1
    assert len(set(selected.tolist())) < len(selected)

    again = list(sampler)
    assert selected.tolist() == again
    sampler.set_epoch(1)
    assert selected.tolist() != list(sampler)


def test_balanced_type_sampler_allocates_equal_counts_without_replacement():
    labels = np.repeat(np.arange(4), [50, 40, 30, 20])
    sampler = distance_sampling.BalancedTypeSampler(
        labels, num_samples=80, seed=7
    )

    selected_indices = list(sampler)
    selected = labels[selected_indices]

    assert len(selected_indices) == len(set(selected_indices)) == 80
    assert np.bincount(selected, minlength=4).tolist() == [20, 20, 20, 20]


def test_distance_sampler_includes_zero_index_research_type():
    sampler = HierarchicalDistanceSampler(
        type_labels=np.array([0, 1, 2, 3]),
        dist_labels=np.array([5, 6, 7, 8]),
        date_ids=np.ones(4, dtype=int),
        file_ids=np.arange(4),
        num_samples=4,
        max_samples_per_file=1,
    )

    assert set(sampler) == {0, 1, 2, 3}


def make_condition_sampler(seed=23, distance_only=True):
    return ConditionBalancedSampler(
        type_labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        daylight=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        distance_low_km=np.array([0, 100, 600, 700, 1200, 1300, -1, 1700]),
        distance_high_km=np.array([300, 200, 700, 800, 1300, 1400, -1, 2400]),
        file_ids=np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        num_samples=8,
        max_samples_per_file=2,
        seed=seed,
        distance_only=distance_only,
    )


def test_condition_sampler_includes_broad_intervals_and_excludes_missing_labels():
    sampler = make_condition_sampler()

    selected = list(sampler)

    assert 0 in selected
    assert 7 in selected
    assert 6 not in selected
    assert all(selected.count(index) == 1 for index in selected)
    assert all(
        sum(1 for index in selected if sampler.file_ids[index] == file_id) <= 2
        for file_id in np.unique(sampler.file_ids)
    )


def test_condition_sampler_balances_available_conditions_and_is_deterministic():
    first = make_condition_sampler()
    second = make_condition_sampler()

    first_selected = list(first)
    second_selected = list(second)
    assert first_selected == second_selected
    keys = [first.condition_key(index) for index in first_selected]
    counts = [keys.count(key) for key in set(keys)]
    assert max(counts) - min(counts) <= 1

    second.set_epoch(1)
    assert first_selected != list(second)


def test_condition_sampler_type_stream_keeps_unlabelled_examples():
    sampler = make_condition_sampler(distance_only=False)

    assert 6 in list(sampler)
