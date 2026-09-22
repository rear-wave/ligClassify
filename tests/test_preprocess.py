import numpy as np
import pytest
import torch

from data.preprocess import (
    AugmentationConfig,
    PreprocessConfig,
    TemporalContextConfig,
    TemporalContextState,
    augment_batch,
    preprocess_views,
)


def test_temporal_context_promotes_only_after_high_confidence_anchors():
    state = TemporalContextState(TemporalContextConfig(
        history_size=4, trigger_count=2, anchor_confidence=0.9,
        min_candidate_probability=0.2,
    ))
    logits = torch.tensor([
        [0.0, -5.0, 6.0, -5.0, -5.0],
        [0.0, -5.0, 6.0, -5.0, -5.0],
        [1.2, -3.0, 1.0, -3.0, -3.0],
    ])

    adjusted = state.adjust(logits, torch.tensor([2, 2, 0]))

    assert adjusted.tolist() == [2, 2, 2]
    assert state.promoted_counts == [0, 0, 1, 0, 0]


def test_temporal_context_seed_and_configuration_validation():
    state = TemporalContextState(TemporalContextConfig(
        history_size=3, trigger_count=2, anchor_confidence=0.9,
        min_candidate_probability=0.2,
    ))
    state.seed([(2, 0.95), (2, 0.80), (2, 0.99)])
    result = state.adjust(
        torch.tensor([[1.0, -3.0, 0.9, -3.0, -3.0]]),
        torch.tensor([0]),
    )

    assert result.item() == 2
    with pytest.raises(ValueError, match="trigger_count"):
        TemporalContextConfig(history_size=2, trigger_count=3)


def test_stream_context_isolates_stations_late_events_gaps_and_missing_times():
    from datetime import datetime, timedelta, timezone
    from data.preprocess import StreamContextBank

    config = TemporalContextConfig(history_size=4, trigger_count=2)
    bank = StreamContextBank(config, max_gap_seconds=10, max_streams=2)
    time = datetime(2024, 1, 1)
    state, reason = bank.prepare("GZ", time)
    state.seed([(2, 1.0), (2, 1.0)])
    bank.commit("GZ", time, state, reason)
    working, reason = bank.prepare("GZ", time + timedelta(seconds=1))
    working.reset()
    assert list(bank.streams["GZ"][1].history) == [2, 2]  # uncommitted change
    other, _ = bank.prepare("ZH", time)
    assert not other.history
    late, reason = bank.prepare("GZ", time - timedelta(seconds=1))
    assert late is None and reason == "late_event"
    bank.commit("GZ", time - timedelta(seconds=1), late, reason)
    assert bank.streams["GZ"][0] == time
    fresh, reason = bank.prepare("GZ", time + timedelta(seconds=11))
    assert reason == "gap_reset" and not fresh.history
    assert bank.utc(time.replace(tzinfo=timezone(timedelta(hours=8)))) == time - timedelta(hours=8)
    empty, reason = bank.prepare("GZ", None)
    bank.commit("GZ", None, empty, reason)
    assert "GZ" not in bank.streams
    for name in ("a", "b", "c"):
        state, reason = bank.prepare(name, time)
        bank.commit(name, time, state, reason)
    assert list(bank.streams) == ["b", "c"]


def test_stream_snapshot_restores_causal_decisions_atomically():
    import copy
    import json
    from datetime import datetime
    from data.preprocess import StreamContextBank

    config = TemporalContextConfig(history_size=4, trigger_count=2)
    first, second = StreamContextBank(config), StreamContextBank(config)
    state, reason = first.prepare("GZ", datetime(2024, 1, 1))
    state.seed([(2, 1.0), (2, 1.0)])
    first.commit("GZ", datetime(2024, 1, 1), state, reason)
    payload = json.loads(json.dumps(first.snapshot()))
    second.restore(payload)
    a, _ = first.prepare("GZ", datetime(2024, 1, 1, 0, 0, 1))
    b, _ = second.prepare("GZ", datetime(2024, 1, 1, 0, 0, 1))
    logits = torch.tensor([[1.2, -3.0, 1.0, -3.0, -3.0]])
    assert a.adjust(logits, torch.tensor([0])).item() == b.adjust(logits, torch.tensor([0])).item() == 2
    invalid = copy.deepcopy(payload)
    invalid["streams"][0]["history"] = [2, 7]
    with pytest.raises(ValueError, match="history"):
        second.restore(invalid)
    assert second.snapshot() == payload
    invalid = copy.deepcopy(payload)
    invalid["streams"].append(invalid["streams"][0])
    with pytest.raises(ValueError, match="duplicate"):
        second.restore(invalid)
    with pytest.raises(ValueError, match="configuration"):
        StreamContextBank(config, max_gap_seconds=20).restore(payload)


def test_augmentation_is_deterministic_polarity_safe_and_non_wrapping():
    values = np.zeros((1, 16000), dtype=np.float32)
    values[0, 100] = 10.0
    config = AugmentationConfig(
        max_shift=32,
        gain_min=0.9,
        gain_max=1.1,
        drift_fraction=0.0,
        noise_fraction=0.0,
    )

    first = augment_batch(values, [17], config)
    second = augment_batch(values, [17], config)

    assert np.array_equal(first, second)
    assert first.max() > 0
    assert first.min() >= 0
    assert first[0, -64:].max() == 0


def test_bandwidth_augmentation_is_reproducible_and_preserves_signed_symmetry():
    values = np.random.default_rng(17).normal(size=(2, 16000)).astype(np.float32)
    config = AugmentationConfig(max_shift=0, gain_min=1, gain_max=1,
                                drift_fraction=0, noise_fraction=0,
                                bandwidth_probability=1)
    first = augment_batch(values, [11, 12], config)
    assert np.array_equal(first, augment_batch(values, [11, 12], config))
    assert np.allclose(first, -augment_batch(-values, [11, 12], config))
    assert first.std() < values.std()
    with pytest.raises(ValueError, match="bandwidth_probability"):
        augment_batch(values, [11, 12], AugmentationConfig(bandwidth_probability=float("nan")))


def test_local_and_global_views_come_from_one_signed_waveform():
    values = np.linspace(-4.0, 8.0, 16000, dtype=np.float32)[None, :]

    local, global_view = preprocess_views(
        values, PreprocessConfig(use_filter=False)
    )

    assert local.shape == (1, 8000)
    assert global_view.shape == (1, 2000)
    assert local.min() < 0 < local.max()
    assert global_view.min() < 0 < global_view.max()


def test_energy_center_prefers_wide_pulse_over_isolated_spike():
    values = np.zeros((1, 16000), dtype=np.float32)
    values[0, 1000] = 100.0
    values[0, 11000:11200] = -20.0

    local, global_view = preprocess_views(
        values,
        PreprocessConfig(
            use_filter=False,
            local_center_mode="energy_envelope_v2",
            local_energy_window=128,
        ),
    )

    assert local.shape == (1, 8000)
    assert global_view.shape == (1, 2000)
    assert local.min() < -0.9
    assert local.max() < 1.0


def test_preprocess_rejects_unknown_center_mode_and_invalid_window():
    values = np.zeros((1, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="local_center_mode"):
        preprocess_views(
            values,
            PreprocessConfig(use_filter=False, local_center_mode="unknown"),
        )
    with pytest.raises(ValueError, match="local_energy_window"):
        preprocess_views(
            values,
            PreprocessConfig(
                use_filter=False,
                local_center_mode="energy_envelope_v2",
                local_energy_window=0,
            ),
        )


def test_augmentation_rejects_non_positive_gain_and_misaligned_seeds():
    values = np.zeros((2, 16000), dtype=np.float32)

    with pytest.raises(ValueError, match="strictly positive"):
        augment_batch(values, [1, 2], AugmentationConfig(gain_min=0.0))
    with pytest.raises(ValueError, match="aligned"):
        augment_batch(values, [1], AugmentationConfig())
