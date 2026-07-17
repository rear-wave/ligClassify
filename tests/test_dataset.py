import struct

import numpy as np
import torch

from data.dataset import FiveClassDataset, collate_batch
from data.lig import FILE_HEADER_BYTES, PIECE_BYTES, WAVEFORM_SAMPLES
from data.manifest import PieceTable, SourceRecord
from data.preprocess import AugmentationConfig, PreprocessConfig
from data.sampling import SampleRequest


def _write_source(path, waveform):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(FILE_HEADER_BYTES)
    struct.pack_into("<i", header, 4, 1)
    piece = bytearray(PIECE_BYTES)
    struct.pack_into("<6i4x", piece, 108, 20, 1, 2, 3, 4, 5)
    struct.pack_into("<d", piece, 136, 0.25)
    piece[208:208 + waveform.nbytes] = waveform.astype("<u2").tobytes()
    path.write_bytes(bytes(header) + bytes(piece))


def _selected_piece_table(tmp_path):
    selected_path = tmp_path / "NCG" / "0-100km" / "selected.lig"
    waveform = np.arange(WAVEFORM_SAMPLES, dtype=np.uint16)
    waveform[9000] = 50000
    _write_source(selected_path, waveform)
    missing_path = tmp_path / "IC" / "not-referenced.lig"
    sources = (
        SourceRecord(
            path=str(missing_path),
            relative_path="IC/not-referenced.lig",
            type_index=0,
            distance_bin=-1,
            piece_count=1,
        ),
        SourceRecord(
            path=str(selected_path),
            relative_path="NCG/0-100km/selected.lig",
            type_index=1,
            distance_bin=0,
            piece_count=1,
        ),
    )
    table = PieceTable(
        root=str(tmp_path),
        sources=sources,
        source_index=np.asarray([0, 1], dtype=np.int32),
        piece_index=np.asarray([0, 0], dtype=np.int32),
        type_index=np.asarray([0, 1], dtype=np.int8),
        distance_bin=np.asarray([-1, 0], dtype=np.int8),
        daylight=np.asarray([False, True], dtype=bool),
        timestamp_seconds=np.asarray([0, 1], dtype=np.int64),
    )
    return table


def test_dataset_opens_only_selected_sources_and_returns_exact_contract(tmp_path):
    table = _selected_piece_table(tmp_path)
    dataset = FiveClassDataset(
        table,
        positions=[1],
        split="validation",
        preprocess_config=PreprocessConfig(use_filter=False),
    )

    item = dataset[0]

    assert set(item) == {
        "local",
        "global",
        "daylight",
        "type_label",
        "distance_bin",
        "source_path",
        "piece_index",
        "piece_key",
    }
    assert item["local"].shape == (1, 8000)
    assert item["global"].shape == (1, 2000)
    assert item["daylight"].shape == (1,)
    assert item["daylight"].item() == 1.0
    assert item["type_label"].shape == ()
    assert item["distance_bin"].shape == ()
    assert item["source_path"] == "NCG/0-100km/selected.lig"
    assert item["piece_index"] == 0
    assert item["piece_key"] == "NCG/0-100km/selected.lig#0"
    assert "quality" not in item


def test_only_seeded_training_requests_are_augmented(tmp_path):
    table = _selected_piece_table(tmp_path)
    augmentation = AugmentationConfig(
        max_shift=16,
        gain_min=1.0,
        gain_max=1.0,
        drift_fraction=0.05,
        noise_fraction=0.05,
    )
    kwargs = {
        "preprocess_config": PreprocessConfig(use_filter=False),
        "augmentation_config": augmentation,
    }
    training = FiveClassDataset(table, [1], split="train", **kwargs)
    validation = FiveClassDataset(table, [1], split="validation", **kwargs)

    unseeded_training = training[0]
    seeded_training = training[SampleRequest(position=1, augmentation_seed=77)]
    unseeded_validation = validation[0]
    seeded_validation = validation[
        SampleRequest(position=1, augmentation_seed=77)
    ]

    assert torch.equal(unseeded_training["local"], unseeded_validation["local"])
    assert not torch.equal(seeded_training["local"], unseeded_training["local"])
    assert torch.equal(seeded_validation["local"], unseeded_validation["local"])
    assert torch.equal(seeded_validation["global"], unseeded_validation["global"])


def test_collation_retains_piece_identity(tmp_path):
    table = _selected_piece_table(tmp_path)
    dataset = FiveClassDataset(
        table,
        positions=[1],
        split="test",
        preprocess_config=PreprocessConfig(use_filter=False),
    )

    batch = collate_batch([dataset[0], dataset[0]])

    assert batch["local"].shape == (2, 1, 8000)
    assert batch["global"].shape == (2, 1, 2000)
    assert batch["daylight"].shape == (2, 1)
    assert batch["source_path"] == [
        "NCG/0-100km/selected.lig",
        "NCG/0-100km/selected.lig",
    ]
    assert batch["piece_index"] == [0, 0]
    assert batch["piece_key"] == [
        "NCG/0-100km/selected.lig#0",
        "NCG/0-100km/selected.lig#0",
    ]
