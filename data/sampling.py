"""Deterministic hierarchical sampling for the five-class trainer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import floor
from numbers import Integral
from typing import Iterable, Iterator, Sequence

import numpy as np

from .manifest import PieceTable, TYPE_NAMES


TYPE_PRIOR = (0.60, 0.10, 0.10, 0.10, 0.10)


@dataclass(frozen=True)
class SampleRequest:
    """One table position and its deterministic augmentation seed."""

    position: int
    augmentation_seed: int


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


def _augmentation_seed(
    seed: int,
    epoch: int,
    draw_index: int,
    position: int,
    attempt: int = 0,
) -> int:
    payload = f"{seed}|{epoch}|{draw_index}|{position}"
    if attempt:
        payload += f"|{attempt}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    )


class FiveClassSampler:
    """Sample type, daylight, then exact distance cells with replacement."""

    def __init__(
        self,
        table: PieceTable,
        positions: Iterable[int],
        num_samples: int,
        seed: int,
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
        self.epoch = 1

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
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
        type_quotas = _largest_remainders(self.num_samples, TYPE_PRIOR)
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
            daylight_quotas = _largest_remainders(
                type_quota, [1.0] * len(daylight_values)
            )
            for daylight, daylight_quota in zip(
                daylight_values, daylight_quotas
            ):
                daylight_positions = [
                    position
                    for position in type_positions
                    if bool(self.table.daylight[position]) == daylight
                ]
                if type_index == 0:
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
                leaf_quotas = _largest_remainders(
                    daylight_quota, [1.0] * len(leaves)
                )
                for leaf, leaf_quota in zip(leaves, leaf_quotas):
                    if leaf_quota:
                        selected = rng.choice(leaf, size=leaf_quota, replace=True)
                        draws.extend(int(position) for position in selected)
        if len(draws) != self.num_samples:
            raise RuntimeError("sampler quota allocation did not sum to epoch size")
        return draws

    def __iter__(self) -> Iterator[SampleRequest]:
        positions = self._draw_positions()
        requests: list[SampleRequest] = []
        used_seeds: set[int] = set()
        for draw_index, position in enumerate(positions):
            attempt = 0
            augmentation_seed = _augmentation_seed(
                self.seed, self.epoch, draw_index, position
            )
            while augmentation_seed in used_seeds:
                attempt += 1
                augmentation_seed = _augmentation_seed(
                    self.seed, self.epoch, draw_index, position, attempt
                )
            used_seeds.add(augmentation_seed)
            requests.append(SampleRequest(position, augmentation_seed))

        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, 0x53485546])
        )
        rng.shuffle(requests)
        return iter(requests)
