"""Signed synchronized preprocessing and safe waveform augmentation."""

from __future__ import annotations

from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

import numpy as np
import torch


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
    bandwidth_probability: float = 0.0


@dataclass(frozen=True)
class TemporalContextConfig:
    """Bounds for causal high-confidence M/N context promotion."""

    history_size: int = 256
    trigger_count: int = 4
    anchor_confidence: float = 0.98
    min_candidate_probability: float = 0.25

    def __post_init__(self) -> None:
        if type(self.history_size) is not int or self.history_size <= 0:
            raise ValueError("temporal history_size must be a positive integer")
        if (
            type(self.trigger_count) is not int
            or not 1 <= self.trigger_count <= self.history_size
        ):
            raise ValueError("temporal trigger_count must be in 1..history_size")
        for name in ("anchor_confidence", "min_candidate_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"temporal {name} must be in (0, 1]")

    def signature(self) -> dict[str, int | float]:
        """Return settings recorded in the inference resume signature."""
        return {
            "history_size": self.history_size,
            "trigger_count": self.trigger_count,
            "anchor_confidence": self.anchor_confidence,
            "min_candidate_probability": self.min_candidate_probability,
        }


class TemporalContextState:
    """Promote clustered known candidates using only preceding predictions."""

    def __init__(self, config: TemporalContextConfig):
        self.config = config
        self.history: deque[int] = deque()
        self.counts = [0, 0, 0, 0, 0]
        self.promoted_counts = [0, 0, 0, 0, 0]

    def _append(self, type_index: int) -> None:
        if len(self.history) == self.config.history_size:
            self.counts[self.history.popleft()] -= 1
        self.history.append(type_index)
        self.counts[type_index] += 1

    def reset(self) -> None:
        """Clear all causal history and diagnostics."""
        self.history.clear()
        self.counts[:] = [0, 0, 0, 0, 0]
        self.promoted_counts[:] = [0, 0, 0, 0, 0]

    def seed(self, rows: Iterable[tuple[int, float]]) -> None:
        """Reconstruct the last M anchors from durable prediction rows."""
        self.reset()
        for type_index, confidence in rows:
            anchor = (
                type_index
                if 0 < type_index < len(self.counts)
                and confidence >= self.config.anchor_confidence
                else 0
            )
            self._append(anchor)

    def adjust(
        self, type_logits: torch.Tensor, base_types: torch.Tensor
    ) -> torch.Tensor:
        """Causally promote IC rows whose known candidate has recent anchors."""
        if type_logits.ndim != 2 or type_logits.shape[1] != len(self.counts):
            raise ValueError("temporal type_logits must have shape [batch, 5]")
        if base_types.ndim != 1 or len(base_types) != len(type_logits):
            raise ValueError("temporal base_types must align with type_logits")
        probabilities = type_logits.detach().float().softmax(dim=1)
        adjusted = base_types.clone()
        for row in range(len(adjusted)):
            final_type = int(adjusted[row].item())
            candidate = int(probabilities[row, 1:].argmax().item()) + 1
            if (
                final_type == 0
                and self.counts[candidate] >= self.config.trigger_count
                and float(probabilities[row, candidate].item())
                >= self.config.min_candidate_probability
            ):
                final_type = candidate
                adjusted[row] = candidate
                self.promoted_counts[candidate] += 1
            confidence = float(probabilities[row, final_type].item())
            anchor = (
                final_type
                if final_type > 0
                and confidence >= self.config.anchor_confidence
                else 0
            )
            self._append(anchor)
        return adjusted


def is_daylight(timestamp: datetime) -> bool:
    """Return whether one UTC timestamp falls in the UTC+8 daylight window."""
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0


class StreamContextBank:
    """Bounded per-station causal history; commit only successful predictions."""

    def __init__(self, config: TemporalContextConfig | None = None, *,
                 max_gap_seconds: float = 60.0, max_streams: int = 64):
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive")
        if type(max_streams) is not int or max_streams < 1:
            raise ValueError("max_streams must be a positive integer")
        self.config = config
        self.max_gap_seconds, self.max_streams = float(max_gap_seconds), max_streams
        self.streams: OrderedDict[str, tuple[datetime, TemporalContextState | None]] = OrderedDict()

    @staticmethod
    def utc(timestamp: datetime | None) -> datetime | None:
        """Normalize aware timestamps; naive datetimes are UTC by contract."""
        if timestamp is None:
            return None
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp must be datetime or None")
        return (timestamp.astimezone(timezone.utc).replace(tzinfo=None)
                if timestamp.tzinfo is not None else timestamp)

    def prepare(self, stream_id: str, timestamp: datetime | None):
        """Return isolated working state and its reset/ordering reason."""
        if not isinstance(stream_id, str) or not stream_id.strip() or len(stream_id) > 256:
            raise ValueError("stream_id must contain 1..256 characters")
        timestamp = self.utc(timestamp)
        previous = self.streams.get(stream_id)
        if timestamp is None:
            return None, "missing_timestamp"
        if previous is not None and timestamp < previous[0]:
            return None, "late_event"
        reason = "new_stream" if previous is None else "continuous"
        if previous is not None and (timestamp - previous[0]).total_seconds() > self.max_gap_seconds:
            previous, reason = None, "gap_reset"
        state = (TemporalContextState(self.config) if self.config else None)
        if previous is not None:
            state = deepcopy(previous[1])
        return state, reason

    def commit(self, stream_id: str, timestamp: datetime | None,
               state: TemporalContextState | None, reason: str) -> None:
        """Advance a stream after success, without allowing late events to rewind it."""
        if reason == "late_event":
            return
        self.streams.pop(stream_id, None)
        timestamp = self.utc(timestamp)
        if timestamp is not None:
            self.streams[stream_id] = (timestamp, state)
            if len(self.streams) > self.max_streams:
                self.streams.popitem(last=False)

    def snapshot(self) -> dict:
        """Return JSON-serializable bounded state for a controlled restart."""
        return {
            "schema": "stream_context_v1", "max_streams": self.max_streams,
            "max_gap_seconds": self.max_gap_seconds,
            "config": None if self.config is None else self.config.signature(),
            "streams": [{"id": key, "timestamp": timestamp.isoformat(),
                         "history": [] if state is None else list(state.history),
                         "promoted_counts": [0] * 5 if state is None else list(state.promoted_counts)}
                        for key, (timestamp, state) in self.streams.items()],
        }

    def restore(self, payload: dict) -> None:
        """Validate an entire snapshot before replacing the live state."""
        expected = self.snapshot()
        if not isinstance(payload, dict) or set(payload) != set(expected):
            raise ValueError("invalid stream context snapshot")
        if any(payload[k] != expected[k] for k in expected if k != "streams"):
            raise ValueError("stream context configuration mismatch")
        rows = payload["streams"]
        if not isinstance(rows, list) or len(rows) > self.max_streams:
            raise ValueError("invalid stream count")
        restored = StreamContextBank(self.config, max_gap_seconds=self.max_gap_seconds,
                                     max_streams=self.max_streams)
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "timestamp", "history", "promoted_counts"}:
                raise ValueError("invalid stream snapshot row")
            if not isinstance(row["id"], str) or row["id"] in restored.streams:
                raise ValueError("invalid or duplicate stream id")
            if not isinstance(row["timestamp"], str):
                raise ValueError("invalid stream timestamp")
            timestamp = self.utc(datetime.fromisoformat(row["timestamp"]))
            state, reason = restored.prepare(row["id"], timestamp)
            history, counts = row["history"], row["promoted_counts"]
            limit = 0 if self.config is None else self.config.history_size
            if (not isinstance(history, list) or len(history) > limit
                    or any(type(v) is not int or not 0 <= v < 5 for v in history)):
                raise ValueError("invalid stream history")
            if (not isinstance(counts, list) or len(counts) != 5 or counts[0] != 0
                    or any(type(v) is not int or v < 0 for v in counts)):
                raise ValueError("invalid stream promotion counts")
            if self.config is None and any(counts):
                raise ValueError("promotion counts require temporal context")
            if state is not None:
                for value in history:
                    state._append(value)
                state.promoted_counts[:] = counts
            restored.commit(row["id"], timestamp, state, reason)
        self.streams = restored.streams


def daylight_inputs(
    timestamps: Sequence[datetime | None],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Build known daylight values and two alternatives for missing times."""
    missing = torch.tensor(
        [timestamp is None for timestamp in timestamps],
        dtype=torch.bool,
        device=device,
    )
    primary = torch.tensor(
        [[0.0 if value is None else float(is_daylight(value))]
         for value in timestamps],
        dtype=torch.float32,
        device=device,
    )
    if not bool(missing.any()):
        return primary, None, missing
    alternate = primary.clone()
    alternate[missing] = 1.0
    return primary, alternate, missing


def average_unknown_logits(
    primary: torch.Tensor,
    alternate: torch.Tensor,
    missing: torch.Tensor,
) -> torch.Tensor:
    """Average day/night probabilities only for missing timestamps."""
    averaged = torch.logaddexp(
        primary.log_softmax(dim=-1),
        alternate.log_softmax(dim=-1),
    ) - math.log(2.0)
    return torch.where(missing.unsqueeze(1), averaged, primary)


def _validate_augmentation(config: AugmentationConfig) -> None:
    if int(config.max_shift) != config.max_shift or config.max_shift < 0:
        raise ValueError("max_shift must be a non-negative integer")
    if not 0 < config.gain_min <= config.gain_max:
        raise ValueError("augmentation gain must remain strictly positive")
    if config.drift_fraction < 0 or config.noise_fraction < 0:
        raise ValueError("drift and noise fractions must be non-negative")
    if not 0.0 <= config.bandwidth_probability <= 1.0:
        raise ValueError("bandwidth_probability must be in [0, 1]")
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
        if config.bandwidth_probability and rng.random() < config.bandwidth_probability:
            from scipy.signal import butter, sosfiltfilt

            cutoff = float(rng.uniform(60_000.0, 200_000.0))
            sos = butter(2, cutoff, fs=5_000_000.0, output="sos")
            result[row_index] = sosfiltfilt(sos, result[row_index]).astype(np.float32)
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
