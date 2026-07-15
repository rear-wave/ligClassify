"""Condition-balanced sampling for trusted lightning waveform pieces."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


CONDITION_DISTANCE_EDGES_KM = np.asarray(
    [0, 300, 600, 1200, 1700, 2400, 3001], dtype=np.int32
)


def interval_coarse_band(low_km, high_km):
    """Map an interval to the coarse band containing its midpoint."""
    low_km, high_km = int(low_km), int(high_km)
    if low_km < 0 or high_km <= low_km:
        return -1
    midpoint = (low_km + high_km) / 2.0
    band = int(
        np.searchsorted(CONDITION_DISTANCE_EDGES_KM, midpoint, side="right") - 1
    )
    return min(max(band, 0), len(CONDITION_DISTANCE_EDGES_KM) - 2)


class ConditionBalancedSampler(Sampler):
    """Balance type, daylight, and coarse distance without file domination."""

    def __init__(
        self,
        type_labels,
        daylight,
        distance_low_km,
        distance_high_km,
        file_ids,
        num_samples,
        max_samples_per_file=256,
        seed=42,
        distance_only=True,
    ):
        arrays = [
            np.asarray(type_labels),
            np.asarray(daylight),
            np.asarray(distance_low_km),
            np.asarray(distance_high_km),
            np.asarray(file_ids),
        ]
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("condition sampler arrays must have the same length")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(max_samples_per_file) <= 0:
            raise ValueError("max_samples_per_file must be positive")
        (
            self.type_labels,
            self.daylight,
            self.distance_low_km,
            self.distance_high_km,
            self.file_ids,
        ) = arrays
        self.num_samples = int(num_samples)
        self.max_samples_per_file = int(max_samples_per_file)
        self.seed = int(seed)
        self.distance_only = bool(distance_only)
        self.epoch = 0
        valid = (
            (self.distance_low_km >= 0)
            & (self.distance_high_km > self.distance_low_km)
        )
        eligible = valid if self.distance_only else np.ones(len(valid), dtype=bool)
        self._eligible = np.flatnonzero(eligible)
        if not len(self._eligible):
            raise ValueError("No eligible waveform pieces are available")
        file_counts = np.bincount(
            self.file_ids[self._eligible].astype(np.int64, copy=False)
        )
        available = np.minimum(file_counts, self.max_samples_per_file).sum()
        self._length = min(self.num_samples, int(available))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def condition_key(self, index):
        """Return the auditable balancing cell for a dataset position."""
        index = int(index)
        return (
            int(self.type_labels[index]),
            int(bool(self.daylight[index])),
            interval_coarse_band(
                self.distance_low_km[index], self.distance_high_km[index]
            ),
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        by_file = defaultdict(list)
        for position in self._eligible:
            by_file[int(self.file_ids[position])].append(int(position))
        kept = []
        for file_id in sorted(by_file):
            positions = np.asarray(by_file[file_id], dtype=np.int64)
            rng.shuffle(positions)
            kept.extend(positions[:self.max_samples_per_file].tolist())

        cells = defaultdict(list)
        for position in kept:
            cells[self.condition_key(position)].append(position)
        for positions in cells.values():
            rng.shuffle(positions)

        selected, cell_deck = [], []
        target_length = len(self)
        while cells and len(selected) < target_length:
            cell_deck = [key for key in cell_deck if key in cells]
            if not cell_deck:
                cell_deck = sorted(cells)
                rng.shuffle(cell_deck)
            key = cell_deck.pop()
            selected.append(cells[key].pop())
            if not cells[key]:
                del cells[key]
        return iter(selected)
