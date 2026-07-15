import numpy as np

from data import preprocessing


def negative_pulse():
    waveform = np.zeros(16000, dtype=np.float32)
    waveform[9000] = -20.0
    waveform[10000] = 5.0
    return waveform


def test_multiscale_alignment_uses_largest_absolute_excursion():
    local, global_view = preprocessing.preprocess_multiscale_batch(
        negative_pulse()[None], use_filter=False
    )

    assert local.shape == global_view.shape == (1, 8000)
    assert np.argmin(local[0]) == 2000
    assert local[0, 2000] < 0


def test_robust_signed_scale_preserves_polarity_and_is_finite():
    waveforms = np.stack([negative_pulse(), np.zeros(16000, dtype=np.float32)])

    local, global_view = preprocessing.preprocess_multiscale_batch(
        waveforms, use_filter=False
    )

    assert local[0].min() < 0 < local[0].max()
    assert np.isfinite(local).all()
    assert np.isfinite(global_view).all()
    assert np.count_nonzero(local[1]) == 0


def test_multiscale_preprocessing_accepts_one_dimensional_input():
    local, global_view = preprocessing.preprocess_multiscale_batch(
        negative_pulse(), use_filter=False
    )

    assert local.shape == global_view.shape == (1, 8000)


def test_waveform_quality_distinguishes_clean_pulse_and_clipped_flatline():
    clean = np.full(16000, 32768, dtype=np.float32)
    clean[8000] = 40000
    clipped = np.full(16000, 65535, dtype=np.float32)

    quality = preprocessing.waveform_quality_batch(np.stack([clean, clipped]))

    assert quality.shape == (2, 3)
    assert quality[0, 0] > quality[1, 0]
    assert quality[1, 1] > quality[0, 1]
    assert np.isfinite(quality).all()
