"""Waveform quality diagnostics used by calibrated rejection."""

from __future__ import annotations

import numpy as np


def waveform_quality_batch(pieces):
    """Return SNR, ADC clipping fraction, and baseline-instability scores."""
    pieces = np.asarray(pieces, dtype=np.float32)
    if pieces.ndim == 1:
        pieces = pieces.reshape(1, -1)
    if pieces.ndim != 2 or pieces.shape[1] == 0:
        raise ValueError("pieces must have shape (N, T) with T > 0")

    baseline = np.median(pieces, axis=1, keepdims=True)
    centered = pieces - baseline
    mad = np.median(np.abs(centered), axis=1)
    peak = np.max(np.abs(centered), axis=1)
    snr_score = np.log1p(peak / np.maximum(1.4826 * mad, 1e-6))
    snr_score = np.clip(snr_score, 0.0, 20.0)

    clipping_fraction = np.mean(
        (pieces <= 0.0) | (pieces >= 65535.0),
        axis=1,
    )

    edge = max(1, pieces.shape[1] // 10)
    first = np.median(pieces[:, :edge], axis=1)
    last = np.median(pieces[:, -edge:], axis=1)
    baseline_instability = np.abs(first - last) / np.maximum(1.4826 * mad, 1.0)
    baseline_instability = np.clip(baseline_instability, 0.0, 20.0)

    return np.stack(
        [snr_score, clipping_fraction, baseline_instability],
        axis=1,
    ).astype(np.float32)
