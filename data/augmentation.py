"""Deterministic polarity-safe augmentation for raw lightning waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    """Configuration for bounded raw-waveform augmentation."""

    max_shift_samples: int = 64
    gain_min: float = 0.90
    gain_max: float = 1.10
    baseline_drift_fraction: float = 0.01
    noise_fraction: float = 0.01


def _zero_shift(values: np.ndarray, shift: int) -> np.ndarray:
    result = np.zeros_like(values)
    if shift > 0:
        result[shift:] = values[:-shift]
    elif shift < 0:
        result[:shift] = values[-shift:]
    else:
        result[:] = values
    return result


def augment_waveforms(
    values: np.ndarray,
    seeds: Sequence[int],
    config: WaveformAugmentationConfig,
) -> np.ndarray:
    """Augment aligned raw waveforms deterministically from per-row seeds."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(seeds):
        raise ValueError("waveforms and augmentation seeds must be aligned")
    if not 0 < config.gain_min <= config.gain_max:
        raise ValueError("augmentation gain must remain strictly positive")
    result = np.empty_like(values)
    axis = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    for row_index, (row, seed) in enumerate(zip(values, seeds)):
        rng = np.random.default_rng(int(seed))
        shift = int(
            rng.integers(-config.max_shift_samples, config.max_shift_samples + 1)
        )
        shifted = _zero_shift(row, shift)
        centered = shifted - np.median(shifted)
        amplitude = max(float(np.quantile(np.abs(centered), 0.95)), 1e-6)
        gain = float(rng.uniform(config.gain_min, config.gain_max))
        drift = (
            float(rng.uniform(-1.0, 1.0))
            * config.baseline_drift_fraction
            * amplitude
            * axis
        )
        noise = rng.normal(
            0.0, config.noise_fraction * amplitude, size=len(row)
        ).astype(np.float32)
        result[row_index] = shifted * gain + drift + noise
    return result
