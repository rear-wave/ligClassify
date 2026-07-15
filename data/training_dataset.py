"""Lazy multiscale dataset for trusted lightning waveform pieces."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.lig_parser import LigFileIndex
from data.oof_manifest import oof_row_id
from data.preprocessing import preprocess_multiscale_batch
from data.signal_context import time_context_batch
from data.waveform_quality import waveform_quality_batch


class LightningPieceDataset(Dataset):
    """Read pre-split pieces lazily and attach interval/context metadata."""

    def __init__(
        self,
        entries,
        split="train",
        lig_index=None,
        use_filter=True,
        data_root=None,
    ):
        self.entries = list(entries)
        self.split = str(split)
        self.use_filter = bool(use_filter)
        paths = sorted({entry.filepath for entry in self.entries})
        absolute_paths = [os.path.abspath(path) for path in paths]
        if data_root is not None:
            root = os.path.abspath(os.fspath(data_root))
        elif absolute_paths:
            root = os.path.commonpath(absolute_paths)
            if len(absolute_paths) == 1 and os.path.normcase(root) == os.path.normcase(
                absolute_paths[0]
            ):
                root = os.path.dirname(root)
        else:
            root = os.getcwd()
        self.data_root = root
        self._owns_lig_index = lig_index is None
        self.lig = lig_index or LigFileIndex(paths, validate=False)
        path_to_file = {
            os.path.normcase(os.path.abspath(path)): file_id
            for file_id, path in enumerate(self.lig.filepaths)
        }

        global_indices = []
        type_labels = []
        distance_low_km = []
        distance_high_km = []
        daylight = []
        file_ids = []
        source_paths = []
        piece_indices = []
        piece_keys = []
        timestamps = []
        for entry in self.entries:
            path_key = os.path.normcase(os.path.abspath(entry.filepath))
            if path_key not in path_to_file:
                raise ValueError(f"piece references an unindexed file: {entry.filepath}")
            file_id = path_to_file[path_key]
            piece_count = int(self.lig.num_pieces_per_file[file_id])
            if not 0 <= int(entry.piece_index) < piece_count:
                raise IndexError(
                    f"piece_index={entry.piece_index} outside {entry.filepath}"
                )
            low = -1 if entry.distance_low_km is None else int(entry.distance_low_km)
            high = -1 if entry.distance_high_km is None else int(entry.distance_high_km)
            is_daytime = (
                bool(entry.is_daytime)
                if entry.is_daytime is not None
                else 5.5 <= (entry.timestamp.hour + 8) % 24 < 19.0
            )
            global_indices.append(int(self.lig._cumsum[file_id]) + int(entry.piece_index))
            type_labels.append(int(entry.type_idx))
            distance_low_km.append(low)
            distance_high_km.append(high)
            daylight.append(int(is_daytime))
            file_ids.append(file_id)
            source_path = Path(
                os.path.relpath(os.path.abspath(entry.filepath), root)
            ).as_posix()
            source_paths.append(source_path)
            piece_indices.append(int(entry.piece_index))
            piece_keys.append(oof_row_id(source_path, entry.piece_index))
            timestamps.append(entry.timestamp)

        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.type_labels = np.asarray(type_labels, dtype=np.int64)
        self.distance_low_km = np.asarray(distance_low_km, dtype=np.int32)
        self.distance_high_km = np.asarray(distance_high_km, dtype=np.int32)
        self.distance_labelled = (
            (self.distance_low_km >= 0)
            & (self.distance_high_km > self.distance_low_km)
        )
        self.daylight = np.asarray(daylight, dtype=np.int8)
        self.file_ids = np.asarray(file_ids, dtype=np.int32)
        self.source_paths = np.asarray(source_paths, dtype=object)
        self.piece_indices = np.asarray(piece_indices, dtype=np.int64)
        self.piece_keys = np.asarray(piece_keys, dtype=object)
        self.timestamps = timestamps
        self.context = time_context_batch(self.timestamps, self.daylight)

    def __len__(self):
        return len(self.global_indices)

    def _items_from_positions(self, positions):
        positions = [int(position) for position in positions]
        global_indices = [int(self.global_indices[position]) for position in positions]
        raw = np.stack(self.lig.read_pieces_batch(global_indices), axis=0)
        quality = waveform_quality_batch(raw)
        local, global_view = preprocess_multiscale_batch(
            raw,
            use_filter=self.use_filter,
        )
        items = []
        for row, position in enumerate(positions):
            items.append({
                "local": torch.from_numpy(local[row].copy()).unsqueeze(0),
                "global_view": torch.from_numpy(global_view[row].copy()).unsqueeze(0),
                "context": torch.from_numpy(self.context[position].copy()),
                "quality": torch.from_numpy(quality[row].copy()),
                "type_label": torch.tensor(
                    int(self.type_labels[position]), dtype=torch.long
                ),
                "distance_low_km": torch.tensor(
                    int(self.distance_low_km[position]), dtype=torch.float32
                ),
                "distance_high_km": torch.tensor(
                    int(self.distance_high_km[position]), dtype=torch.float32
                ),
                "file_id": torch.tensor(
                    int(self.file_ids[position]), dtype=torch.long
                ),
                "source_path": str(self.source_paths[position]),
                "piece_index": int(self.piece_indices[position]),
                "piece_key": str(self.piece_keys[position]),
                "timestamp": self.timestamps[position],
            })
        return items

    def __getitem__(self, index):
        return self._items_from_positions([index])[0]

    def __getitems__(self, indices):
        return self._items_from_positions(indices)

    def close(self):
        if getattr(self, "_owns_lig_index", False) and hasattr(self, "lig"):
            self.lig.close()

    def __del__(self):
        self.close()


def collate_training_batch(batch):
    """Stack tensor fields while retaining timestamps for audit reports."""
    keys = (
        "local",
        "global_view",
        "context",
        "quality",
        "type_label",
        "distance_low_km",
        "distance_high_km",
        "file_id",
    )
    result = {key: torch.stack([item[key] for item in batch]) for key in keys}
    result["source_path"] = [item["source_path"] for item in batch]
    result["piece_index"] = [int(item["piece_index"]) for item in batch]
    result["piece_key"] = [item["piece_key"] for item in batch]
    result["timestamp"] = [item["timestamp"] for item in batch]
    return result
