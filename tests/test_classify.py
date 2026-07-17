import csv
import hashlib
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import classify
from checkpoints import LoadedCheckpoint
from data.lig import (
    FILE_HEADER_BYTES,
    MAX_PIECES_PER_FILE,
    PIECE_BYTES,
    WAVEFORM_SAMPLES,
)
from data.preprocessing import preprocess_batch as legacy_reference_preprocess
from models import ModelOutput


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS = tuple(range(0, 3000, 100))


def make_piece(value: int, *, hour: int = 0, marker: int = 0) -> bytes:
    raw = bytearray((marker + index * 17) % 256 for index in range(PIECE_BYTES))
    struct.pack_into("<6i4x", raw, 108, 20, 1, 2, hour, 4, 5)
    struct.pack_into("<d", raw, 136, 0.25)
    waveform = np.full(WAVEFORM_SAMPLES, value, dtype="<u2")
    raw[208:208 + waveform.nbytes] = waveform.tobytes()
    return bytes(raw)


def write_source(path: Path, pieces: list[bytes], *, marker: int = 0) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray((marker + index * 29) % 256 for index in range(FILE_HEADER_BYTES))
    struct.pack_into("<i", header, 4, len(pieces))
    path.write_bytes(bytes(header) + b"".join(pieces))
    return bytes(header)


class ConstantNewModel(nn.Module):
    def __init__(self, type_index: int = 2, distance_bin: int = 5):
        super().__init__()
        self.type_index = type_index
        self.distance_bin = distance_bin
        self.batch_sizes: list[int] = []
        self.daylight_batches: list[torch.Tensor] = []

    def forward(self, local, global_view, daylight):
        count = len(local)
        self.batch_sizes.append(count)
        self.daylight_batches.append(daylight.detach().cpu().clone())
        type_logits = torch.zeros(count, 5, device=local.device)
        type_logits[:, self.type_index] = 9.0
        distance_logits = [
            torch.zeros(count, 30, device=local.device) for _ in range(4)
        ]
        if self.type_index:
            distance_logits[self.type_index - 1][:, self.distance_bin] = 10.0
        return ModelOutput(
            type_logits=type_logits,
            distance_logits=tuple(distance_logits),
            features=torch.zeros(count, 1, device=local.device),
        )


class RecordingLegacyModel(nn.Module):
    def __init__(self, type_index: int = 1, distance_bin: int = 3):
        super().__init__()
        self.type_index = type_index
        self.distance_bin = distance_bin
        self.inputs: list[torch.Tensor] = []

    def forward(self, values):
        self.inputs.append(values.detach().cpu().clone())
        count = len(values)
        type_logits = torch.zeros(count, 5, device=values.device)
        type_logits[:, self.type_index] = 9.0
        heads = [torch.zeros(count, 30, device=values.device) for _ in range(4)]
        if self.type_index:
            heads[self.type_index - 1][:, self.distance_bin] = 10.0
        return type_logits, tuple(heads)


def loaded_new(model: nn.Module) -> LoadedCheckpoint:
    return LoadedCheckpoint(
        model=model,
        schema="five_class_v1",
        type_names=TYPE_NAMES,
        distance_names=DISTANCE_NAMES,
        distance_bins_km=DISTANCE_BINS,
        preprocess_config={
            "local_length": 8000,
            "global_length": 2000,
            "use_filter": False,
        },
        metadata={},
    )


def loaded_legacy(model: nn.Module) -> LoadedCheckpoint:
    return LoadedCheckpoint(
        model=model,
        schema="legacy_five_class",
        type_names=TYPE_NAMES,
        distance_names=DISTANCE_NAMES,
        distance_bins_km=DISTANCE_BINS,
        preprocess_config={"normalize_mode": "minmax", "target_length": 8000},
        metadata={},
    )


def test_direct_argmax_outputs_ic_without_threshold():
    prediction = classify.decode_new_prediction(
        type_logits=torch.tensor([9.0, 1.0, 0.0, 0.0, 0.0]),
        distance_logits=[torch.zeros(30) for _ in range(4)],
    )

    assert prediction.final_type == "IC"
    assert prediction.distance_bin is None
    assert prediction.output_class == "IC"


def test_non_ic_uses_matching_distance_expert():
    heads = [torch.zeros(30) for _ in range(4)]
    heads[1][5] = 10.0

    prediction = classify.decode_new_prediction(
        type_logits=torch.tensor([0.0, 0.0, 9.0, 0.0, 0.0]),
        distance_logits=heads,
    )

    assert prediction.final_type == "NNBE"
    assert prediction.distance_bin == 5
    assert prediction.output_class == "NNBE_500-600km"
    probabilities = torch.softmax(heads[1], dim=0)
    centers = torch.arange(50.0, 3050.0, 100.0)
    assert prediction.expected_distance_km == pytest.approx(
        float((probabilities * centers).sum())
    )


def test_legacy_decode_uses_the_same_direct_prediction_contract():
    heads = [torch.zeros(30) for _ in range(4)]
    heads[3][29] = 10.0

    prediction = classify.decode_legacy_prediction(
        type_logits=torch.tensor([0.0, 0.0, 0.0, 0.0, 9.0]),
        distance_logits=heads,
    )

    assert isinstance(prediction, classify.Prediction)
    assert prediction.final_type == "PNBE"
    assert prediction.distance_bin == 29
    assert prediction.output_class == "PNBE_2900-3000km"


def test_type_only_has_type_directory_and_no_distance_values():
    heads = [torch.zeros(30) for _ in range(4)]
    prediction = classify.decode_new_prediction(
        torch.tensor([0.0, 9.0, 0.0, 0.0, 0.0]),
        heads,
        type_only=True,
    )

    assert prediction.output_class == "NCG"
    assert prediction.distance_bin is None
    assert prediction.expected_distance_km is None
    assert prediction.distance_confidence is None


def test_new_adapter_uses_signed_views_and_each_piece_timestamp():
    model = ConstantNewModel(type_index=0)
    waveforms = np.stack([
        np.linspace(0, 1000, WAVEFORM_SAMPLES, dtype=np.float32),
        np.linspace(1000, 0, WAVEFORM_SAMPLES, dtype=np.float32),
    ])

    predictions = classify.predict_batch(
        loaded_new(model),
        waveforms,
        [datetime(2020, 1, 2, 0), datetime(2020, 1, 2, 16)],
        device=torch.device("cpu"),
        type_only=False,
    )

    assert [item.final_type for item in predictions] == ["IC", "IC"]
    assert model.daylight_batches[0].flatten().tolist() == [1.0, 0.0]


def test_legacy_adapter_matches_retained_8000_point_minmax_preprocessing():
    model = RecordingLegacyModel()
    waveforms = np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32)
    waveforms[0, 3000] = 1000.0
    waveforms[0, 8000] = 500.0
    waveforms[1] = np.linspace(0.0, 2000.0, WAVEFORM_SAMPLES)
    expected = legacy_reference_preprocess(
        waveforms.copy(), target_length=8000, normalize_mode="minmax"
    )

    classify.predict_batch(
        loaded_legacy(model),
        waveforms,
        [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        device=torch.device("cpu"),
        type_only=False,
    )

    assert torch.equal(model.inputs[0], torch.from_numpy(expected).unsqueeze(1))


def test_classification_preserves_bytes_and_writes_exact_csv_contract(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    pieces = [make_piece(11, hour=0, marker=31), make_piece(22, hour=16, marker=97)]
    source = input_dir / "incoming" / "source.lig"
    source_header = write_source(source, pieces, marker=43)
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"synthetic-checkpoint")
    expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = ConstantNewModel(type_index=2, distance_bin=5)
    monkeypatch.setattr(classify, "load_model_checkpoint", lambda *_: loaded_new(model))

    csv_path = classify.classify_directory(
        input_dir,
        output_dir,
        model_path,
        batch_size=1,
        device="cpu",
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert handle.seek(0) == 0
        assert next(csv.reader(handle)) == list(classify.PREDICTION_FIELDS)
    assert len(rows) == 2
    row = rows[0]
    assert row["source_path"] == "incoming/source.lig"
    assert row["piece_index"] == "0"
    assert row["piece_key"] == "incoming/source.lig#0"
    assert row["prob_IC"] != ""
    assert row["prob_NNBE"] != ""
    assert row["checkpoint_schema"] == "five_class_v1"
    assert row["model_sha256"] == expected_hash

    output_file = output_dir / Path(row["output_file"])
    written = output_file.read_bytes()
    written_header = bytearray(written[:FILE_HEADER_BYTES])
    expected_header = bytearray(source_header)
    struct.pack_into("<i", expected_header, 4, 2)
    assert written_header == expected_header
    assert written[FILE_HEADER_BYTES:FILE_HEADER_BYTES + PIECE_BYTES] == pieces[0]
    assert written[FILE_HEADER_BYTES + PIECE_BYTES:] == pieces[1]


def test_regrouping_is_bounded_and_first_piece_owns_each_output_header(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    first_pieces = [make_piece(index, marker=index % 251) for index in range(500)]
    second_pieces = [
        make_piece(1000 + index, marker=(index + 71) % 251) for index in range(15)
    ]
    first_header = write_source(
        input_dir / "incoming" / "a.lig", first_pieces, marker=13
    )
    second_header = write_source(
        input_dir / "incoming" / "b.lig", second_pieces, marker=211
    )
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"bounded")
    model = ConstantNewModel(type_index=1, distance_bin=0)
    monkeypatch.setattr(classify, "load_model_checkpoint", lambda *_: loaded_new(model))

    classify.classify_directory(
        input_dir,
        output_dir,
        model_path,
        batch_size=17,
        device="cpu",
    )

    output_files = sorted((output_dir / "NCG_0-100km").glob("*.lig"))
    counts = [struct.unpack_from("<i", path.read_bytes(), 4)[0] for path in output_files]
    assert counts == [MAX_PIECES_PER_FILE, 3]
    assert max(model.batch_sizes) <= 17
    first_written_header = bytearray(output_files[0].read_bytes()[:FILE_HEADER_BYTES])
    second_written_header = bytearray(output_files[1].read_bytes()[:FILE_HEADER_BYTES])
    first_expected = bytearray(first_header)
    second_expected = bytearray(second_header)
    struct.pack_into("<i", first_expected, 4, MAX_PIECES_PER_FILE)
    struct.pack_into("<i", second_expected, 4, 3)
    assert first_written_header == first_expected
    assert second_written_header == second_expected


def test_cli_is_compact_and_path_validation_rejects_unsafe_nesting(tmp_path):
    args = classify.build_arg_parser().parse_args([
        "--input_dir", "input",
        "--output_dir", "output",
        "--model", "model.pt",
    ])
    assert set(vars(args)) == {
        "input_dir", "output_dir", "model", "batch_size", "type_only", "device"
    }

    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="must not contain"):
        classify.classify_directory(child, parent, model, device="cpu")
    with pytest.raises(ValueError, match="must not contain"):
        classify.classify_directory(parent, child, model, device="cpu")
    with pytest.raises(ValueError, match="batch_size"):
        classify.classify_directory(parent, tmp_path / "elsewhere", model, batch_size=0)
