"""Signed synchronized preprocessing and safe waveform augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the signed local and global waveform views."""

    local_length: int = 8000
    global_length: int = 2000
    use_filter: bool = True
    cutoff_hz: float = 120_000.0
    sample_rate_hz: float = 5_000_000.0
    local_center_mode: str = "peak_abs_v1"
    local_energy_window: int = 128


@dataclass(frozen=True)
class AugmentationConfig:
    """Bounds for deterministic polarity-safe training augmentation."""

    max_shift: int = 64
    gain_min: float = 0.90
    gain_max: float = 1.10
    drift_fraction: float = 0.01
    noise_fraction: float = 0.01


def _validate_augmentation(config: AugmentationConfig) -> None:
    if int(config.max_shift) != config.max_shift or config.max_shift < 0:
        raise ValueError("max_shift must be a non-negative integer")
    if not 0 < config.gain_min <= config.gain_max:
        raise ValueError("augmentation gain must remain strictly positive")
    if config.drift_fraction < 0 or config.noise_fraction < 0:
        raise ValueError("drift and noise fractions must be non-negative")
    values = (
        config.gain_min,
        config.gain_max,
        config.drift_fraction,
        config.noise_fraction,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("augmentation bounds must be finite")


def _zero_shift(values: np.ndarray, shift: int) -> np.ndarray:
    """Shift one row with zero fill and no circular wraparound."""
    result = np.zeros_like(values)
    if shift >= len(values) or shift <= -len(values):
        return result
    if shift > 0:
        result[shift:] = values[:-shift]
    elif shift < 0:
        result[:shift] = values[-shift:]
    else:
        result[:] = values
    return result


def augment_batch(
    values: np.ndarray,
    seeds: Sequence[int],
    config: AugmentationConfig,
) -> np.ndarray:
    """Augment raw rows deterministically without polarity inversion."""
    waveforms = np.asarray(values, dtype=np.float32)
    if waveforms.ndim != 2 or len(waveforms) != len(seeds):
        raise ValueError("waveforms and augmentation seeds must be aligned")
    _validate_augmentation(config)

    result = np.empty_like(waveforms)
    drift_axis = np.linspace(
        -1.0, 1.0, waveforms.shape[1], dtype=np.float32
    )
    for row_index, (row, seed) in enumerate(zip(waveforms, seeds)):
        rng = np.random.default_rng(int(seed))
        shift = int(rng.integers(-config.max_shift, config.max_shift + 1))
        shifted = _zero_shift(row, shift)
        centered = shifted - np.median(shifted)
        robust_scale = max(
            float(np.quantile(np.abs(centered), 0.95)), 1e-6
        )
        gain = float(rng.uniform(config.gain_min, config.gain_max))
        drift = (
            float(rng.uniform(-1.0, 1.0))
            * config.drift_fraction
            * robust_scale
            * drift_axis
        )
        noise = rng.normal(
            0.0,
            config.noise_fraction * robust_scale,
            size=len(row),
        ).astype(np.float32)
        result[row_index] = shifted * gain + drift + noise
    return result


def _validate_preprocess(config: PreprocessConfig) -> None:
    if int(config.local_length) != config.local_length or config.local_length <= 0:
        raise ValueError("local_length must be a positive integer")
    if int(config.global_length) != config.global_length or config.global_length <= 0:
        raise ValueError("global_length must be a positive integer")
    if not np.isfinite(config.sample_rate_hz) or config.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if (
        not np.isfinite(config.cutoff_hz)
        or config.cutoff_hz <= 0
        or config.cutoff_hz >= config.sample_rate_hz / 2
    ):
        raise ValueError("cutoff_hz must lie below the Nyquist frequency")
    if config.local_center_mode not in {
        "peak_abs_v1",
        "energy_envelope_v2",
    }:
        raise ValueError(
            "local_center_mode must be peak_abs_v1 or energy_envelope_v2"
        )
    if (
        isinstance(config.local_energy_window, bool)
        or int(config.local_energy_window) != config.local_energy_window
        or config.local_energy_window <= 0
    ):
        raise ValueError("local_energy_window must be a positive integer")


def _filter_batch(values: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if not config.use_filter:
        return values
    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        2,
        config.cutoff_hz / (config.sample_rate_hz / 2),
        btype="low",
        output="sos",
    )
    return sosfiltfilt(sos, values, axis=-1).astype(np.float32)


def _energy_centers(values: np.ndarray, window: int) -> np.ndarray:
    centered = values - np.median(values, axis=1, keepdims=True)
    width = min(int(window), values.shape[1])
    squared = np.square(centered, dtype=np.float64)
    cumulative = np.pad(
        np.cumsum(squared, axis=1),
        ((0, 0), (1, 0)),
        mode="constant",
    )
    energy = cumulative[:, width:] - cumulative[:, :-width]
    return np.argmax(energy, axis=1) + width // 2


def _local_view(
    values: np.ndarray,
    target_length: int,
    center_mode: str,
    energy_window: int,
) -> np.ndarray:
    rows, source_length = values.shape
    result = np.zeros((rows, target_length), dtype=np.float32)
    if center_mode == "peak_abs_v1":
        centered = values - np.median(values, axis=1, keepdims=True)
        centers = np.argmax(np.abs(centered), axis=1)
    else:
        centers = _energy_centers(values, energy_window)
    before = target_length // 4
    for row_index, center in enumerate(centers):
        if source_length <= target_length:
            result[row_index, :source_length] = values[row_index]
            continue
        start = min(
            max(int(center) - before, 0),
            source_length - target_length,
        )
        result[row_index] = values[row_index, start:start + target_length]
    return result


def _global_view(values: np.ndarray, target_length: int) -> np.ndarray:
    rows, source_length = values.shape
    if source_length == target_length:
        return values.copy()
    if source_length % target_length == 0 and source_length > target_length:
        width = source_length // target_length
        return values.reshape(rows, target_length, width).mean(axis=2).astype(
            np.float32
        )

    source_axis = np.linspace(0.0, 1.0, source_length)
    target_axis = np.linspace(0.0, 1.0, target_length)
    return np.stack(
        [np.interp(target_axis, source_axis, row) for row in values]
    ).astype(np.float32)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values, axis=1, keepdims=True)
    scale = np.quantile(np.abs(centered), 0.95, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (centered / scale).astype(np.float32)


def preprocess_views(
    values: np.ndarray,
    config: PreprocessConfig = PreprocessConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Create normalized local/global views from each same signed source row."""
    waveforms = np.asarray(values, dtype=np.float32)
    if waveforms.ndim != 2 or waveforms.shape[1] == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    _validate_preprocess(config)
    signed_source = _filter_batch(waveforms, config)
    local = _local_view(
        signed_source,
        config.local_length,
        config.local_center_mode,
        config.local_energy_window,
    )
    global_view = _global_view(signed_source, config.global_length)
    return _robust_normalize(local), _robust_normalize(global_view)


def legacy_preprocess_batch(values: np.ndarray) -> np.ndarray:
    """Reproduce the retained model's historical 8000-point input view."""
    pieces = np.asarray(values, dtype=np.float32)
    if pieces.ndim != 2 or pieces.shape[1] == 0:
        raise ValueError("waveforms must be a non-empty two-dimensional batch")
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(
            2,
            120_000.0 / (5_000_000.0 / 2.0),
            btype="low",
            output="sos",
        )
        pieces = sosfiltfilt(sos, pieces, axis=-1).astype(np.float32)
    except ImportError:  # pragma: no cover - SciPy is a runtime dependency
        pass
    target_length = 8000
    peaks = np.argmax(pieces, axis=1).astype(np.int64)
    begins = peaks - 2000
    ends = peaks + 6000
    before_source = begins < 0
    begins[before_source] = 0
    ends[before_source] = target_length
    after_source = ends > pieces.shape[1]
    ends[after_source] = pieces.shape[1]
    begins[after_source] = ends[after_source] - target_length
    cropped = np.zeros((len(pieces), target_length), dtype=np.float32)
    for row, (begin, end) in enumerate(zip(begins, ends)):
        segment = pieces[row, begin:end][:target_length]
        cropped[row, :len(segment)] = segment
    denominator = cropped.max(axis=1, keepdims=True) - cropped.min(
        axis=1, keepdims=True
    )
    denominator[denominator < 1e-8] = 1.0
    return (
        (cropped - cropped.mean(axis=1, keepdims=True)) / denominator
    ).astype(np.float32)
