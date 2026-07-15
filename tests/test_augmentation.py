import numpy as np

from data.augmentation import WaveformAugmentationConfig, augment_waveforms


def test_augmentation_is_seeded_and_never_wraps_or_flips_polarity():
    pieces = np.zeros((1, 16000), dtype=np.float32)
    pieces[0, 100] = 10.0
    config = WaveformAugmentationConfig(
        max_shift_samples=32,
        gain_min=0.90,
        gain_max=1.10,
        baseline_drift_fraction=0.0,
        noise_fraction=0.0,
    )
    first = augment_waveforms(pieces, [17], config)
    second = augment_waveforms(pieces, [17], config)
    assert np.array_equal(first, second)
    assert first.max() > 0
    assert first[0, -64:].max() == 0
