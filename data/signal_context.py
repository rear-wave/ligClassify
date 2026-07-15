"""Deterministic time-context features for waveform models."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np


def time_context_batch(
    timestamps: Iterable[datetime],
    daylight: Iterable[bool | int],
    mode: str = "daylight",
) -> np.ndarray:
    """Return daylight-only context or legacy cyclic local-hour context."""
    timestamps = list(timestamps)
    daylight = list(daylight)
    if len(timestamps) != len(daylight):
        raise ValueError("timestamps and daylight must have the same length")
    daylight_column = np.asarray(daylight, dtype=np.float32).reshape(-1, 1)
    if mode == "daylight":
        return daylight_column
    if mode != "cyclic":
        raise ValueError("time context mode must be daylight or cyclic")
    local_hours = np.asarray([
        (
            stamp.hour
            + 8
            + stamp.minute / 60.0
            + stamp.second / 3600.0
        ) % 24
        for stamp in timestamps
    ])
    angles = 2.0 * np.pi * local_hours / 24.0
    return np.column_stack([
        daylight_column[:, 0], np.sin(angles), np.cos(angles)
    ]).astype(np.float32)
