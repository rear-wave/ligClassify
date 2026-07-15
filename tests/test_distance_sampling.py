import numpy as np
import pytest

from data.distance_sampling import ConditionBalancedSampler


def make_sampler(seed=23, distance_only=True):
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


def test_sampler_includes_broad_intervals_excludes_missing_and_caps_files():
    sampler = make_sampler()

    selected = list(sampler)

    assert 0 in selected and 7 in selected
    assert 6 not in selected
    assert len(selected) == len(set(selected))
    assert all(
        sum(1 for index in selected if sampler.file_ids[index] == file_id) <= 2
        for file_id in np.unique(sampler.file_ids)
    )


def test_sampler_balances_conditions_and_changes_deterministically_by_epoch():
    first = make_sampler()
    second = make_sampler()
    selected = list(first)

    assert selected == list(second)
    keys = [first.condition_key(index) for index in selected]
    counts = [keys.count(key) for key in set(keys)]
    assert max(counts) - min(counts) <= 1

    second.set_epoch(1)
    assert selected != list(second)


def test_type_stream_keeps_unlabelled_examples():
    assert 6 in list(make_sampler(distance_only=False))


def test_sampler_does_not_recompute_target_length_per_sample(monkeypatch):
    sampler = make_sampler()
    original_len = ConditionBalancedSampler.__len__
    calls = 0

    def counted_len(instance):
        nonlocal calls
        calls += 1
        return original_len(instance)

    monkeypatch.setattr(ConditionBalancedSampler, "__len__", counted_len)

    list(sampler)

    # list() may request one length hint in addition to the sampler's own call.
    assert calls <= 2


def test_sampler_rejects_misaligned_or_empty_inputs():
    with pytest.raises(ValueError, match="same length"):
        ConditionBalancedSampler([0], [1, 0], [0], [100], [0], 1)
    with pytest.raises(ValueError, match="No eligible"):
        ConditionBalancedSampler([0], [1], [-1], [-1], [0], 1)
