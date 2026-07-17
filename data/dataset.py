"""Lazy piece dataset for signed five-class training and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .lig import LigFileIndex
from .manifest import PieceTable
from .preprocess import (
    AugmentationConfig,
    PreprocessConfig,
    augment_batch,
    preprocess_views,
)
from .sampling import SampleRequest


class FiveClassDataset(Dataset):
    """Read only selected waveform pieces and preserve their stable identity."""

    def __init__(
        self,
        table: PieceTable,
        positions: Iterable[int],
        split: str,
        preprocess_config: PreprocessConfig | None = None,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        self.table = table
        self.positions = np.asarray(
            [int(position) for position in positions], dtype=np.int64
        )
        self.split = str(split)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.augmentation_config = augmentation_config
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if np.any(self.positions < 0) or np.any(self.positions >= len(table)):
            raise ValueError("positions must reference the piece table")
        if len(np.unique(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        self._position_set = set(int(position) for position in self.positions)

        selected_source_ids = sorted(
            {int(table.source_index[position]) for position in self.positions}
        )
        self._source_to_file = {
            source_id: file_index
            for file_index, source_id in enumerate(selected_source_ids)
        }
        selected_paths = [
            table.sources[source_id].path for source_id in selected_source_ids
        ]
        self.lig = LigFileIndex(selected_paths, validate=True)

    def __len__(self) -> int:
        return len(self.positions)

    def _resolve(self, index: int | SampleRequest) -> tuple[int, int | None]:
        if isinstance(index, SampleRequest):
            position = int(index.position)
            if position not in self._position_set:
                raise IndexError("sample request is outside this dataset split")
            return position, int(index.augmentation_seed)
        local_index = int(index)
        if local_index < 0:
            local_index += len(self.positions)
        if local_index < 0 or local_index >= len(self.positions):
            raise IndexError("dataset index out of range")
        return int(self.positions[local_index]), None

    def _global_piece_index(self, position: int) -> int:
        source_id = int(self.table.source_index[position])
        file_index = self._source_to_file[source_id]
        piece_index = int(self.table.piece_index[position])
        if piece_index < 0 or piece_index >= self.lig.num_pieces_per_file[file_index]:
            raise IndexError("piece index is outside its selected source")
        return int(self.lig._cumsum[file_index]) + piece_index

    def _items(self, indices: Sequence[int | SampleRequest]) -> list[dict[str, object]]:
        resolved = [self._resolve(index) for index in indices]
        positions = [position for position, _ in resolved]
        global_indices = [self._global_piece_index(position) for position in positions]
        raw = np.stack(self.lig.read_pieces_batch(global_indices)).astype(
            np.float32, copy=False
        )

        preprocessing_input = raw
        if self.split == "train" and self.augmentation_config is not None:
            augmented_rows = [
                row for row, (_, seed) in enumerate(resolved) if seed is not None
            ]
            if augmented_rows:
                preprocessing_input = raw.copy()
                seeds = [int(resolved[row][1]) for row in augmented_rows]
                preprocessing_input[augmented_rows] = augment_batch(
                    raw[augmented_rows], seeds, self.augmentation_config
                )
        local, global_view = preprocess_views(
            preprocessing_input, self.preprocess_config
        )

        items: list[dict[str, object]] = []
        for row, position in enumerate(positions):
            source = self.table.sources[int(self.table.source_index[position])]
            piece_index = int(self.table.piece_index[position])
            items.append(
                {
                    "local": torch.from_numpy(local[row].copy()).unsqueeze(0),
                    "global": torch.from_numpy(global_view[row].copy()).unsqueeze(0),
                    "daylight": torch.tensor(
                        [float(self.table.daylight[position])], dtype=torch.float32
                    ),
                    "type_label": torch.tensor(
                        int(self.table.type_index[position]), dtype=torch.long
                    ),
                    "distance_bin": torch.tensor(
                        int(self.table.distance_bin[position]), dtype=torch.long
                    ),
                    "source_path": source.relative_path,
                    "piece_index": piece_index,
                    "piece_key": self.table.piece_key(position),
                }
            )
        return items

    def __getitem__(self, index: int | SampleRequest) -> dict[str, object]:
        return self._items([index])[0]

    def __getitems__(
        self, indices: Sequence[int | SampleRequest]
    ) -> list[dict[str, object]]:
        return self._items(indices)

    def close(self) -> None:
        if hasattr(self, "lig"):
            self.lig.close()

    def __del__(self) -> None:
        self.close()


def collate_batch(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    """Stack model tensors while retaining source and piece identities."""
    tensor_keys = (
        "local",
        "global",
        "daylight",
        "type_label",
        "distance_bin",
    )
    result: dict[str, object] = {
        key: torch.stack([item[key] for item in batch]) for key in tensor_keys
    }
    result["source_path"] = [str(item["source_path"]) for item in batch]
    result["piece_index"] = [int(item["piece_index"]) for item in batch]
    result["piece_key"] = [str(item["piece_key"]) for item in batch]
    return result
