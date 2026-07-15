"""Deterministic time-context features for waveform models."""

from __future__ import annotations

import math

import numpy as np


def time_context_batch(timestamps, is_daytime, utc_offset_hours=8.0):
    """Return daylight plus cyclic local-hour features."""
    timestamps = list(timestamps)
    is_daytime = list(is_daytime)
    if len(timestamps) != len(is_daytime):
        raise ValueError("timestamps and is_daytime must have the same length")
    result = np.empty((len(timestamps), 3), dtype=np.float32)
    for index, (timestamp, daylight) in enumerate(zip(timestamps, is_daytime)):
        hour = (
            timestamp.hour
            + timestamp.minute / 60.0
            + timestamp.second / 3600.0
            + utc_offset_hours
        ) % 24.0
        angle = 2.0 * math.pi * hour / 24.0
        result[index] = (float(daylight), math.sin(angle), math.cos(angle))
    return result
