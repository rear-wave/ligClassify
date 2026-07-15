from collections import Counter

import pytest

from data.distance_sampling import JointConditionSampler, exact_interval_key


def make_joint_sampler():
    return JointConditionSampler(
        type_labels=[0, 0, 0, 1, 1, 1],
        daylight=[0, 0, 1, 0, 1, 1],
        distance_low_km=[300, 300, 400, 300, 400, 400],
        distance_high_km=[400, 400, 500, 400, 500, 500],
        file_ids=[0, 0, 1, 2, 3, 3],
        num_samples=12,
        max_samples_per_file=4,
        seed=5,
    )


def test_joint_sampler_balances_exact_cells_with_replacement_and_caps_files():
    sampler = make_joint_sampler()
    requests = list(sampler)
    assert len(requests) == 12
    assert len({request.augmentation_seed for request in requests}) == 12
    assert max(Counter(sampler.file_ids[r.position] for r in requests).values()) <= 4
    cells = Counter(
        (
            sampler.type_labels[request.position],
            sampler.daylight[request.position],
            sampler.distance_low_km[request.position],
        )
        for request in requests
    )
    assert max(cells.values()) - min(cells.values()) <= 1


def test_joint_sampler_rejects_impossible_file_cap():
    with pytest.raises(ValueError, match="file cap"):
        JointConditionSampler(
            type_labels=[0, 0],
            daylight=[1, 1],
            distance_low_km=[0, 0],
            distance_high_km=[100, 100],
            file_ids=[0, 1],
            num_samples=9,
            max_samples_per_file=4,
            seed=1,
        )


@pytest.mark.parametrize(
    ("low_km", "high_km"),
    [(0, 200), (50, 150), (-100, 0), (3000, 3100)],
)
def test_exact_interval_key_rejects_noncanonical_intervals(low_km, high_km):
    with pytest.raises(ValueError, match="aligned 100-km intervals"):
        exact_interval_key(low_km, high_km)


def test_joint_sampler_does_not_recompute_target_length_per_sample(monkeypatch):
    sampler = make_joint_sampler()
    original_len = JointConditionSampler.__len__
    calls = 0

    def counted_len(instance):
        nonlocal calls
        calls += 1
        return original_len(instance)

    monkeypatch.setattr(JointConditionSampler, "__len__", counted_len)

    list(sampler)

    # list() may request one length hint in addition to the sampler's own call.
    assert calls <= 2
