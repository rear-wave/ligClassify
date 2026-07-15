"""Condition-balanced sampling for trusted lightning waveform pieces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib

import numpy as np
from torch.utils.data import Sampler


CONDITION_DISTANCE_EDGES_KM = np.asarray(
    [0, 300, 600, 1200, 1700, 2400, 3001], dtype=np.int32
)


@dataclass(frozen=True)
class SampleRequest:
    """A dataset position paired with its deterministic augmentation seed."""

    position: int
    augmentation_seed: int


def exact_interval_key(low_km, high_km):
    """Return a validated, aligned 100-km distance interval."""
    low_km, high_km = int(low_km), int(high_km)
    if (
        high_km - low_km != 100
        or low_km % 100
        or not 0 <= low_km < high_km <= 3000
    ):
        raise ValueError("joint sampling requires aligned 100-km intervals")
    return low_km, high_km


def _request_seed(seed, epoch, draw_index, position):
    payload = f"{seed}:{epoch}:{draw_index}:{position}".encode("ascii")
    digest = hashlib.blake2b(
        payload, digest_size=8, person=b"lig-augment"
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


class JointConditionSampler(Sampler):
    """Sample type/daylight/exact-distance cells with replacement and file caps."""

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
    ):
        arrays = [
            np.asarray(type_labels),
            np.asarray(daylight),
            np.asarray(distance_low_km),
            np.asarray(distance_high_km),
            np.asarray(file_ids),
        ]
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("joint sampler arrays must have the same length")
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
        self.epoch = 0

        nested = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
        )
        file_cells = defaultdict(set)
        for position in range(len(self.type_labels)):
            type_label = int(self.type_labels[position])
            daylight = int(bool(self.daylight[position]))
            interval = exact_interval_key(
                self.distance_low_km[position], self.distance_high_km[position]
            )
            file_id = int(self.file_ids[position])
            nested[type_label][daylight][interval][file_id].append(position)
            file_cells[file_id].add((type_label, daylight, interval))
        if not file_cells:
            raise ValueError("No eligible waveform pieces are available")
        if self.num_samples > len(file_cells) * self.max_samples_per_file:
            raise ValueError("num_samples exceeds available capacity under file cap")

        self._pools = {
            type_label: {
                daylight: {
                    interval: {
                        file_id: np.asarray(positions, dtype=np.int64)
                        for file_id, positions in sorted(files.items())
                    }
                    for interval, files in sorted(intervals.items())
                }
                for daylight, intervals in sorted(daylights.items())
            }
            for type_label, daylights in sorted(nested.items())
        }
        self._file_cells = {
            file_id: tuple(sorted(cells))
            for file_id, cells in sorted(file_cells.items())
        }
        self._length = self.num_samples

    def set_epoch(self, epoch):
        """Select the deterministic shuffle stream for an epoch."""
        self.epoch = int(epoch)

    def __len__(self):
        return self._length

    def __iter__(self):
        rng = np.random.default_rng(_request_seed(self.seed, self.epoch, -1, -1))
        file_draw_counts = defaultdict(int)
        cell_files = {
            (type_label, daylight, interval): set(files)
            for type_label, daylights in self._pools.items()
            for daylight, intervals in daylights.items()
            for interval, files in intervals.items()
        }
        parent_intervals = {
            (type_label, daylight): set(intervals)
            for type_label, daylights in self._pools.items()
            for daylight, intervals in daylights.items()
        }
        type_daylights = {
            type_label: set(daylights)
            for type_label, daylights in self._pools.items()
        }
        available_types = set(self._pools)
        decks = {}

        def draw_from_deck(deck_key, candidates):
            deck = decks.setdefault(deck_key, [])
            while deck:
                value = deck.pop()
                if value in candidates:
                    return value
            deck.extend(sorted(candidates))
            rng.shuffle(deck)
            return deck.pop()

        used_augmentation_seeds = set()
        for draw_index in range(self._length):
            type_label = draw_from_deck(("type",), available_types)
            daylight = draw_from_deck(
                ("daylight", type_label), type_daylights[type_label]
            )
            interval = draw_from_deck(
                ("interval", type_label, daylight),
                parent_intervals[(type_label, daylight)],
            )
            cell = (type_label, daylight, interval)
            file_id = draw_from_deck(("file", *cell), cell_files[cell])
            positions = self._pools[type_label][daylight][interval][file_id]
            position = int(positions[rng.integers(len(positions))])

            augmentation_seed = _request_seed(
                self.seed, self.epoch, draw_index, position
            )
            while augmentation_seed in used_augmentation_seeds:
                augmentation_seed = (augmentation_seed + 1) % (1 << 64)
            used_augmentation_seeds.add(augmentation_seed)
            yield SampleRequest(position, augmentation_seed)

            file_draw_counts[file_id] += 1
            if file_draw_counts[file_id] < self.max_samples_per_file:
                continue
            for capped_cell in self._file_cells[file_id]:
                files = cell_files[capped_cell]
                files.discard(file_id)
                if files:
                    continue
                capped_type, capped_daylight, capped_interval = capped_cell
                intervals = parent_intervals[(capped_type, capped_daylight)]
                intervals.discard(capped_interval)
                if intervals:
                    continue
                daylights = type_daylights[capped_type]
                daylights.discard(capped_daylight)
                if not daylights:
                    available_types.discard(capped_type)


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
