from datetime import datetime

import numpy as np

from data.lig_parser import LigFileIndex
from data.training_dataset import LightningPieceDataset
from data.training_manifest import PieceManifestEntry
from tests.test_training_manifest import write_lig


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
    assert item["timestamp"] == entries[1].timestamp
    assert np.isfinite(item["local"].numpy()).all()
    assert dataset.distance_labelled.tolist() == [True, True]
    assert dataset.file_ids.tolist() == [0, 0]

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
    assert dataset.distance_labelled.tolist() == [False]
    dataset.close()
