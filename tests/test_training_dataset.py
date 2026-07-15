from datetime import datetime

import numpy as np

from data.augmentation import WaveformAugmentationConfig
from data.distance_sampling import SampleRequest
from data.lig_parser import LigFileIndex
from data.training_dataset import LightningPieceDataset, collate_training_batch
from data.training_manifest import PieceManifestEntry
from tests.test_training_manifest import write_lig


class OnePieceIndex:
    def __init__(self, waveform):
        self.filepaths = ["synthetic.lig"]
        self.num_pieces_per_file = np.asarray([1], dtype=np.int64)
        self._cumsum = np.asarray([0, 1], dtype=np.int64)
        self.waveform = np.asarray(waveform, dtype=np.float32)

    def read_pieces_batch(self, indices):
        assert list(indices) == [0]
        return [self.waveform.copy()]


def dataset_item(split, request_seed):
    waveform = np.zeros(16000, dtype=np.float32)
    waveform[100:120] = np.linspace(1.0, 10.0, 20)
    entry = PieceManifestEntry(
        filepath="synthetic.lig",
        piece_index=0,
        type_idx=0,
        dist_bin=0,
        timestamp=datetime(2020, 1, 1, 4),
        distance_low_km=0,
        distance_high_km=100,
        is_daytime=True,
    )
    dataset = LightningPieceDataset(
        [entry],
        split=split,
        lig_index=OnePieceIndex(waveform),
        use_filter=False,
        augmentation=WaveformAugmentationConfig(noise_fraction=0.0),
    )
    return dataset[SampleRequest(position=0, augmentation_seed=request_seed)]


def test_validation_dataset_ignores_augmentation_request():
    train = dataset_item(split="train", request_seed=9)
    first_val = dataset_item(split="val", request_seed=9)
    second_val = dataset_item(split="val", request_seed=123)
    assert not np.array_equal(train["local"].numpy(), first_val["local"].numpy())
    assert np.array_equal(first_val["local"].numpy(), second_val["local"].numpy())


def test_training_dataset_returns_multiscale_interval_and_audit_fields(tmp_path):
    path = write_lig(
        tmp_path / "NNBE" / "night" / "1500-3000km" / "sample.lig",
        pieces=2,
        piece_timestamps=[
            (20, 1, 1, 16, 0, 0),
            (20, 1, 1, 16, 0, 1),
        ],
    )
    entries = [
        PieceManifestEntry(
            filepath=str(path),
            piece_index=piece_index,
            type_idx=1,
            dist_bin=-1,
            timestamp=datetime(2020, 1, 1, 16, 0, piece_index),
            distance_low_km=1500,
            distance_high_km=3000,
            is_daytime=False,
        )
        for piece_index in range(2)
    ]
    index = LigFileIndex([str(path)], validate=False)

    dataset = LightningPieceDataset(
        entries,
        split="train",
        lig_index=index,
        use_filter=False,
        data_root=tmp_path,
    )
    item = dataset[1]

    assert len(dataset) == 2
    assert tuple(item["local"].shape) == (1, 8000)
    assert tuple(item["global_view"].shape) == (1, 8000)
    assert tuple(item["context"].shape) == (3,)
    assert tuple(item["quality"].shape) == (3,)
    assert item["type_label"].item() == 1
    assert item["distance_low_km"].item() == 1500
    assert item["distance_high_km"].item() == 3000
    assert item["file_id"].item() == 0
    assert item["source_path"] == "NNBE/night/1500-3000km/sample.lig"
    assert item["piece_index"] == 1
    assert item["piece_key"] == "NNBE/night/1500-3000km/sample.lig#1"
    assert item["timestamp"] == entries[1].timestamp
    assert np.isfinite(item["local"].numpy()).all()
    assert dataset.distance_labelled.tolist() == [True, True]
    assert dataset.file_ids.tolist() == [0, 0]
    assert dataset.source_paths.tolist() == [
        "NNBE/night/1500-3000km/sample.lig",
        "NNBE/night/1500-3000km/sample.lig",
    ]
    assert dataset.piece_indices.tolist() == [0, 1]
    assert dataset.piece_keys.tolist() == [
        "NNBE/night/1500-3000km/sample.lig#0",
        "NNBE/night/1500-3000km/sample.lig#1",
    ]

    batch = collate_training_batch([dataset[0], dataset[1]])
    assert batch["source_path"] == [
        "NNBE/night/1500-3000km/sample.lig",
        "NNBE/night/1500-3000km/sample.lig",
    ]
    assert batch["piece_index"] == [0, 1]
    assert batch["piece_key"] == [
        "NNBE/night/1500-3000km/sample.lig#0",
        "NNBE/night/1500-3000km/sample.lig#1",
    ]

    dataset.close()
    assert index.read_piece(0).shape == (16000,)
    index.close()


def test_training_dataset_marks_missing_distance_without_losing_type_label(tmp_path):
    path = write_lig(tmp_path / "NCG" / "unlabelled.lig")
    entry = PieceManifestEntry(
        filepath=str(path),
        piece_index=0,
        type_idx=0,
        dist_bin=-1,
        timestamp=datetime(2020, 1, 1),
        is_daytime=True,
    )

    dataset = LightningPieceDataset([entry], use_filter=False)
    item = dataset[0]

    assert item["type_label"].item() == 0
    assert item["distance_low_km"].item() == -1
    assert item["distance_high_km"].item() == -1
    assert item["source_path"] == "unlabelled.lig"
    assert item["piece_index"] == 0
    assert item["piece_key"] == "unlabelled.lig#0"
    assert dataset.distance_labelled.tolist() == [False]
    dataset.close()
