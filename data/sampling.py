"""Deterministic hierarchical sampling for the five-class trainer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import floor
from numbers import Integral
from typing import Iterable, Iterator, Sequence

import numpy as np

from .manifest import PieceTable, TYPE_NAMES


TYPE_PRIOR = (0.20, 0.20, 0.20, 0.20, 0.20)


@dataclass(frozen=True)
class SampleRequest:
    """One table position and two deterministic augmentation seeds."""

    position: int
    augmentation_seeds: tuple[int, int]


def _largest_remainders(total: int, weights: Sequence[float]) -> list[int]:
    weight_sum = float(sum(weights))
    if total < 0 or not weights or weight_sum <= 0:
        raise ValueError("quota inputs must be non-negative")
    exact = [total * float(weight) / weight_sum for weight in weights]
    quotas = [floor(value) for value in exact]
    remaining = total - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - quotas[index]), index),
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


def _bounded_equal_quotas(total: int, capacities: Sequence[int]) -> list[int]:
    """Distribute a quota evenly without exceeding any source cell."""
    normalized = [int(capacity) for capacity in capacities]
    if (
        total < 0
        or not normalized
        or any(capacity < 0 for capacity in normalized)
        or total > sum(normalized)
    ):
        raise ValueError("quota exceeds available samples")
    quotas = [0] * len(normalized)
    for _ in range(total):
        available = [
            index
            for index, capacity in enumerate(normalized)
            if quotas[index] < capacity
        ]
        selected = min(available, key=lambda index: (quotas[index], index))
        quotas[selected] += 1
    return quotas


def _augmentation_seed(
    seed: int,
    epoch: int,
    draw_index: int,
    position: int,
    view: int,
    attempt: int = 0,
) -> int:
    payload = f"{seed}|{epoch}|{draw_index}|{position}|view={view}"
    if attempt:
        payload += f"|{attempt}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


def _paired_seeds(
    seed: int,
    epoch: int,
    draw_index: int,
    position: int,
    used_seeds: set[int],
) -> tuple[int, int]:
    values: list[int] = []
    for view in range(2):
        attempt = 0
        value = _augmentation_seed(
            seed, epoch, draw_index, position, view
        )
        while value in used_seeds:
            attempt += 1
            value = _augmentation_seed(
                seed,
                epoch,
                draw_index,
                position,
                view,
                attempt,
            )
        used_seeds.add(value)
        values.append(value)
    return values[0], values[1]


class FiveClassSampler:
    """Sample equal type quotas without replacement within an epoch."""

    def __init__(
        self,
        table: PieceTable,
        positions: Iterable[int],
        num_samples: int,
        seed: int,
        balance_distance: bool = True,
    ) -> None:
        self.table = table
        raw_positions = tuple(positions)
        if any(
            isinstance(position, (bool, np.bool_))
            or not isinstance(position, Integral)
            for position in raw_positions
        ):
            raise ValueError("positions must contain integer table positions")
        self.positions = tuple(int(position) for position in raw_positions)
        if isinstance(num_samples, bool) or not isinstance(num_samples, Integral):
            raise ValueError("num_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be a non-negative integer")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.balance_distance = bool(balance_distance)
        self.epoch = 1

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.num_samples % len(TYPE_NAMES):
            raise ValueError("num_samples must be divisible by five")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.positions:
            raise ValueError("positions must not be empty")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        if any(position < 0 or position >= len(table) for position in self.positions):
            raise ValueError("positions must reference the piece table")

        available_types = {
            int(table.type_index[position]) for position in self.positions
        }
        for type_index, type_name in enumerate(TYPE_NAMES):
            if type_index not in available_types:
                raise ValueError(f"missing required type: {type_name}")
            available = sum(
                int(table.type_index[position]) == type_index
                for position in self.positions
            )
            if self.num_samples // len(TYPE_NAMES) > available:
                raise ValueError(
                    "num_samples cannot be drawn without replacement; "
                    f"{type_name} has only {available} pieces"
                )
        if self.balance_distance:
            for position in self.positions:
                type_index = int(table.type_index[position])
                distance_bin = int(table.distance_bin[position])
                if type_index != 0 and not 0 <= distance_bin <= 29:
                    raise ValueError(
                        f"distance_bin outside 0..29 at position {position}"
                    )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Select a positive epoch for deterministic independent draws."""
        normalized = int(epoch)
        if normalized != epoch or normalized <= 0:
            raise ValueError("epoch must be a positive integer")
        self.epoch = normalized

    def _draw_positions(self) -> list[int]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])
        )
        type_quotas = [self.num_samples // len(TYPE_NAMES)] * len(TYPE_NAMES)
        draws: list[int] = []
        for type_index, type_quota in enumerate(type_quotas):
            type_positions = [
                position
                for position in self.positions
                if int(self.table.type_index[position]) == type_index
            ]
            daylight_values = sorted(
                {bool(self.table.daylight[position]) for position in type_positions}
            )
            daylight_groups = [
                [
                    position
                    for position in type_positions
                    if bool(self.table.daylight[position]) == daylight
                ]
                for daylight in daylight_values
            ]
            daylight_quotas = _bounded_equal_quotas(
                type_quota, [len(group) for group in daylight_groups]
            )
            for daylight_positions, daylight_quota in zip(
                daylight_groups, daylight_quotas
            ):
                if type_index == 0 or not self.balance_distance:
                    leaves = [daylight_positions]
                else:
                    distance_bins = sorted(
                        {
                            int(self.table.distance_bin[position])
                            for position in daylight_positions
                        }
                    )
                    leaves = [
                        [
                            position
                            for position in daylight_positions
                            if int(self.table.distance_bin[position]) == distance_bin
                        ]
                        for distance_bin in distance_bins
                    ]
                leaf_quotas = _bounded_equal_quotas(
                    daylight_quota, [len(leaf) for leaf in leaves]
                )
                for leaf, leaf_quota in zip(leaves, leaf_quotas):
                    if leaf_quota:
                        selected = rng.choice(
                            leaf, size=leaf_quota, replace=False
                        )
                        draws.extend(int(position) for position in selected)
        if len(draws) != self.num_samples:
            raise RuntimeError("sampler quota allocation did not sum to epoch size")
        return draws

    def __iter__(self) -> Iterator[SampleRequest]:
        positions = self._draw_positions()
        requests: list[SampleRequest] = []
        used_seeds: set[int] = set()
        for draw_index, position in enumerate(positions):
            seeds = _paired_seeds(
                self.seed,
                self.epoch,
                draw_index,
                position,
                used_seeds,
            )
            requests.append(SampleRequest(position, seeds))

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0x53485546])
        )
        rng.shuffle(requests)
        return iter(requests)


class DistanceExpertSampler:
    """Sample one non-IC type uniformly across observed distance/day cells."""

    def __init__(
        self,
        table: PieceTable,
        positions: Iterable[int],
        type_index: int,
        num_samples: int,
        seed: int,
    ) -> None:
        if (
            isinstance(type_index, bool)
            or not isinstance(type_index, Integral)
            or not 1 <= int(type_index) < len(TYPE_NAMES)
        ):
            raise ValueError("type_index must be an integer in 1..4")
        raw_positions = tuple(positions)
        if any(
            isinstance(position, (bool, np.bool_))
            or not isinstance(position, Integral)
            for position in raw_positions
        ):
            raise ValueError("positions must contain integer table positions")
        normalized = tuple(int(position) for position in raw_positions)
        if len(set(normalized)) != len(normalized):
            raise ValueError("positions must be unique")
        if any(position < 0 or position >= len(table) for position in normalized):
            raise ValueError("positions must reference the piece table")
        if isinstance(num_samples, bool) or not isinstance(num_samples, Integral):
            raise ValueError("num_samples must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be a non-negative integer")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")

        self.table = table
        self.type_index = int(type_index)
        self.positions = tuple(
            position
            for position in normalized
            if int(table.type_index[position]) == self.type_index
        )
        if not self.positions:
            raise ValueError(
                f"no pieces available for distance type "
                f"{TYPE_NAMES[self.type_index]}"
            )
        if any(
            not 0 <= int(table.distance_bin[position]) <= 29
            for position in self.positions
        ):
            raise ValueError("distance_bin must be in 0..29")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 1

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Select a positive epoch for deterministic independent draws."""
        normalized = int(epoch)
        if normalized != epoch or normalized <= 0:
            raise ValueError("epoch must be a positive integer")
        self.epoch = normalized

    def _draw_positions(self) -> list[int]:
        cells: dict[tuple[bool, int], list[int]] = {}
        for position in self.positions:
            key = (
                bool(self.table.daylight[position]),
                int(self.table.distance_bin[position]),
            )
            cells.setdefault(key, []).append(position)
        ordered_cells = [cells[key] for key in sorted(cells)]
        quotas = _largest_remainders(
            self.num_samples, [1.0] * len(ordered_cells)
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])
        )
        draws: list[int] = []
        for cell, quota in zip(ordered_cells, quotas):
            if quota:
                selected = rng.choice(cell, size=quota, replace=True)
                draws.extend(int(position) for position in selected)
        return draws

    def __iter__(self) -> Iterator[SampleRequest]:
        positions = self._draw_positions()
        requests: list[SampleRequest] = []
        used_seeds: set[int] = set()
        for draw_index, position in enumerate(positions):
            seeds = _paired_seeds(
                self.seed,
                self.epoch,
                draw_index,
                position,
                used_seeds,
            )
            requests.append(SampleRequest(position, seeds))
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0x53485546])
        )
        rng.shuffle(requests)
        return iter(requests)
