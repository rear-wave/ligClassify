"""Balanced sampling helpers for four researched lightning types."""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class BalancedTypeSampler(Sampler):
    """Draw equal counts from contiguous researched type labels."""

    def __init__(self, type_labels, num_samples=180000, seed=42):
        self.type_labels = np.asarray(type_labels)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.classes = sorted(np.unique(self.type_labels).tolist())
        if self.classes != list(range(len(self.classes))):
            raise ValueError("type labels must be contiguous and start at zero")
        if not self.classes or self.num_samples % len(self.classes):
            raise ValueError("num_samples must divide evenly across types")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        per_class = self.num_samples // len(self.classes)
        selected = []
        for value in self.classes:
            pool = np.flatnonzero(self.type_labels == value)
            if per_class > len(pool):
                raise ValueError(
                    f"Type {value} needs {per_class} samples but only "
                    f"{len(pool)} are available"
                )
            selected.extend(
                rng.choice(pool, size=per_class, replace=False).tolist()
            )
        rng.shuffle(selected)
        return iter(selected)


class HierarchicalDistanceSampler(Sampler):
    """Sample type, distance bin, date/file, and piece in balanced cycles."""

    def __init__(
        self,
        type_labels,
        dist_labels,
        date_ids,
        file_ids,
        num_samples,
        max_samples_per_file=256,
        seed=42,
        replacement=False,
    ):
        arrays = [
            np.asarray(type_labels),
            np.asarray(dist_labels),
            np.asarray(date_ids),
            np.asarray(file_ids),
        ]
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("type, distance, date, and file arrays must have the same length")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if max_samples_per_file <= 0:
            raise ValueError("max_samples_per_file must be positive")

        self.type_labels, self.dist_labels, self.date_ids, self.file_ids = arrays
        self.num_samples = int(num_samples)
        self.max_samples_per_file = int(max_samples_per_file)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0

        eligible = np.flatnonzero(self.dist_labels >= 0)
        self._positions_by_file = defaultdict(list)
        for position in eligible:
            self._positions_by_file[int(self.file_ids[position])].append(int(position))
        self._available = sum(
            min(len(positions), self.max_samples_per_file)
            for positions in self._positions_by_file.values()
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.replacement:
            return self.num_samples
        return min(self.num_samples, self._available)

    @staticmethod
    def _next_active(deck, active, rng):
        while deck and deck[-1] not in active:
            deck.pop()
        if not deck:
            deck.extend(active)
            rng.shuffle(deck)
        return deck.pop()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        kept_positions = []
        for positions in self._positions_by_file.values():
            shuffled = np.asarray(positions, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            kept_positions.extend(shuffled[:self.max_samples_per_file].tolist())

        leaves = defaultdict(list)
        for position in kept_positions:
            key = (
                int(self.type_labels[position]),
                int(self.dist_labels[position]),
                int(self.date_ids[position]),
                int(self.file_ids[position]),
            )
            leaves[key].append(position)
        for positions in leaves.values():
            rng.shuffle(positions)

        if self.replacement:
            return iter(self._sample_with_replacement(leaves, rng))

        bin_decks = defaultdict(list)
        date_decks = defaultdict(list)
        file_decks = defaultdict(list)
        type_deck = []
        selected = []

        while leaves and len(selected) < len(self):
            active_types = sorted({key[0] for key in leaves})
            lightning_type = self._next_active(type_deck, active_types, rng)

            active_bins = sorted({key[1] for key in leaves if key[0] == lightning_type})
            dist_bin = self._next_active(
                bin_decks[lightning_type], active_bins, rng
            )

            type_bin = (lightning_type, dist_bin)
            active_dates = sorted({
                key[2] for key in leaves if key[:2] == type_bin
            })
            date_id = self._next_active(date_decks[type_bin], active_dates, rng)

            type_bin_date = (lightning_type, dist_bin, date_id)
            active_files = sorted({
                key[3] for key in leaves if key[:3] == type_bin_date
            })
            file_id = self._next_active(
                file_decks[type_bin_date], active_files, rng
            )

            leaf = (lightning_type, dist_bin, date_id, file_id)
            selected.append(leaves[leaf].pop())
            if not leaves[leaf]:
                del leaves[leaf]

        return iter(selected)

    def _sample_with_replacement(self, leaves, rng):
        """Cycle hierarchy levels uniformly and refill exhausted piece decks."""
        if not leaves:
            raise ValueError("No labelled distance samples are available")
        pools = {key: tuple(values) for key, values in leaves.items()}
        piece_decks = {key: list(values) for key, values in pools.items()}
        bin_decks = defaultdict(list)
        date_decks = defaultdict(list)
        file_decks = defaultdict(list)
        type_deck = []
        selected = []
        active_types = sorted({key[0] for key in pools})

        while len(selected) < self.num_samples:
            lightning_type = self._next_active(type_deck, active_types, rng)
            active_bins = sorted({key[1] for key in pools if key[0] == lightning_type})
            dist_bin = self._next_active(
                bin_decks[lightning_type], active_bins, rng
            )
            type_bin = (lightning_type, dist_bin)
            active_dates = sorted({key[2] for key in pools if key[:2] == type_bin})
            date_id = self._next_active(date_decks[type_bin], active_dates, rng)
            type_bin_date = (lightning_type, dist_bin, date_id)
            active_files = sorted({
                key[3] for key in pools if key[:3] == type_bin_date
            })
            file_id = self._next_active(
                file_decks[type_bin_date], active_files, rng
            )
            leaf = (lightning_type, dist_bin, date_id, file_id)
            if not piece_decks[leaf]:
                piece_decks[leaf] = list(pools[leaf])
                rng.shuffle(piece_decks[leaf])
            selected.append(piece_decks[leaf].pop())
        return selected
