import numpy as np
import pytest

from data.preprocess import (
    AugmentationConfig,
    PreprocessConfig,
    augment_batch,
    preprocess_views,
)


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
