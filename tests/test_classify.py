import csv
import csv
import hashlib
import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import classify
from checkpoints import LoadedCheckpoint, LoadedModelBundle
from data.lig import (
    FILE_HEADER_BYTES,
    MAX_PIECES_PER_FILE,
    PIECE_BYTES,
    WAVEFORM_SAMPLES,
)
from models import ModelOutput


TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
DISTANCE_BINS = tuple(range(0, 3000, 100))


def legacy_reference_preprocess(
    pieces: np.ndarray,
    *,
    target_length: int,
    normalize_mode: str,
) -> np.ndarray:
    """Frozen oracle for the retained checkpoint's historical input view."""
    from scipy.signal import butter, sosfiltfilt

    values = np.asarray(pieces, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    sos = butter(
        2,
        120_000.0 / (5_000_000.0 / 2.0),
        btype="low",
        output="sos",
    )
    values = sosfiltfilt(sos, values, axis=-1).astype(np.float32)

    peaks = np.argmax(values, axis=1).astype(np.int64)
    begins = peaks - 2000
    ends = peaks + 6000
    before_source = begins < 0
    begins[before_source] = 0
    ends[before_source] = target_length
    after_source = ends > values.shape[1]
    ends[after_source] = values.shape[1]
    begins[after_source] = ends[after_source] - target_length
    cropped = np.empty((len(values), target_length), dtype=np.float32)
    for row in range(len(values)):
        segment = values[row, begins[row]:ends[row]]
        cropped[row, :len(segment)] = segment
        cropped[row, len(segment):] = 0.0

    assert normalize_mode == "minmax"
    pmin = cropped.min(axis=1, keepdims=True)
    pmax = cropped.max(axis=1, keepdims=True)
    pmean = cropped.mean(axis=1, keepdims=True)
    denominator = pmax - pmin
    denominator[denominator < 1e-8] = 1.0
    return ((cropped - pmean) / denominator).astype(np.float32)


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

    def forward_type(self, local, global_view, daylight):
        output = self.forward(local, global_view, daylight)
        return output.type_logits, output.features

    def forward_distance(self, local, global_view, daylight):
        output = self.forward(local, global_view, daylight)
        return output.distance_logits, output.features

    def forward_distance_type(
        self, local, global_view, daylight, *, expert_index
    ):
        heads, features = self.forward_distance(local, global_view, daylight)
        return heads[expert_index], features


class TypeOnlyNewModel(nn.Module):
    def forward_type(self, local, global_view, daylight):
        logits = torch.zeros(len(local), 5, device=local.device)
        logits[:, 1] = 3.0
        return logits, torch.zeros(len(local), 1, device=local.device)

    def forward(self, *_args):
        raise AssertionError("type-only inference must not run the full model")

    def predict_cascade(self, *_args):
        raise AssertionError("type-only inference must not run distance routing")


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
    assert prediction.output_class == "NNBE/0500-0600km"
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
    assert prediction.output_class == "PNBE/2900-3000km"


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


def test_new_type_only_adapter_skips_distance_routing():
    predictions = classify.predict_batch(
        loaded_new(TypeOnlyNewModel()),
        np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32),
        [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        device=torch.device("cpu"),
        type_only=True,
    )

    assert [item.final_type for item in predictions] == ["NCG", "NCG"]


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
    assert row["output_file"] == (
        "NNBE/0500-0600km/GZ_20200102000405.lig"
    )

    output_file = output_dir / Path(row["output_file"])
    written = output_file.read_bytes()
    written_header = bytearray(written[:FILE_HEADER_BYTES])
    expected_header = bytearray(source_header)
    struct.pack_into("<i", expected_header, 4, 2)
    assert written_header == expected_header
    assert written[FILE_HEADER_BYTES:FILE_HEADER_BYTES + PIECE_BYTES] == pieces[0]
    assert written[FILE_HEADER_BYTES + PIECE_BYTES:] == pieces[1]


def test_output_type_filter_writes_only_selected_predictions(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    pieces = [make_piece(11), make_piece(22)]
    write_source(input_dir / "source.lig", pieces)
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"filter")
    monkeypatch.setattr(
        classify,
        "load_model_checkpoint",
        lambda *_: loaded_new(ConstantNewModel()),
    )
    predictions = [
        classify.Prediction(
            "NNBE", "NNBE/0500-0600km", (0, 0, 1, 0, 0), 1, 5, 550, 1
        ),
        classify.Prediction(
            "PNBE", "PNBE/2900-3000km", (0, 0, 0, 0, 1), 1, 29, 2950, 1
        ),
    ]
    monkeypatch.setattr(
        classify,
        "predict_batch",
        lambda *_args, **_kwargs: predictions,
    )

    csv_path = classify.classify_directory(
        input_dir,
        tmp_path / "output",
        model_path,
        batch_size=2,
        output_types=["PNBE"],
        device="cpu",
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["final_type"] for row in rows] == ["NNBE", "PNBE"]
    assert rows[0]["output_file"] == ""
    assert not (tmp_path / "output" / "NNBE").exists()
    output_file = tmp_path / "output" / Path(rows[1]["output_file"])
    assert output_file.read_bytes()[FILE_HEADER_BYTES:] == pieces[1]


def test_classification_warns_and_continues_after_an_empty_lig(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    (input_dir / "a-empty.lig").parent.mkdir(parents=True)
    (input_dir / "a-empty.lig").write_bytes(b"")
    write_source(input_dir / "b-valid.lig", [make_piece(22)])
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"skip-empty")
    monkeypatch.setattr(
        classify, "load_model_checkpoint", lambda *_: loaded_new(ConstantNewModel())
    )

    with pytest.warns(RuntimeWarning, match="skipping invalid LIG file"):
        csv_path = classify.classify_directory(
            input_dir, tmp_path / "output", model_path, device="cpu"
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["source_path"] for row in rows] == ["b-valid.lig"]


def test_resume_skips_completed_sources_and_appends_without_duplicates(tmp_path):
    input_dir = tmp_path / "input"
    first = input_dir / "a.lig"
    second = input_dir / "b.lig"
    write_source(first, [make_piece(11)])
    write_source(second, [make_piece(22)])
    files = [("a.lig", first), ("b.lig", second)]
    output_dir = tmp_path / "output"
    prediction = classify.Prediction(
        "NNBE", "NNBE/0500-0600km", (0, 0, 1, 0, 0), 1, 5, 550, 1
    )
    calls = 0

    def interrupted(waveforms, _timestamps):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return [prediction] * len(waveforms)

    with pytest.raises(RuntimeError, match="interrupted"):
        classify._classify_files(
            files, output_dir, batch_size=1, predict=interrupted,
            checkpoint_schema="test", checkpoint_hash="abc",
            output_types=["NNBE"], run_mode="test",
        )
    assert (output_dir / ".classification_resume.json").is_file()

    resumed_calls = 0

    def resumed(waveforms, _timestamps):
        nonlocal resumed_calls
        resumed_calls += 1
        return [prediction] * len(waveforms)

    csv_path = classify._classify_files(
        files, output_dir, batch_size=1, predict=resumed,
        checkpoint_schema="test", checkpoint_hash="abc",
        output_types=["NNBE"], run_mode="test", resume=True,
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert resumed_calls == 1
    assert [row["piece_key"] for row in rows] == ["a.lig#0", "b.lig#0"]
    assert not (output_dir / ".classification_resume.json").exists()


def test_resume_rejects_changed_model_configuration(tmp_path):
    source = tmp_path / "input" / "a.lig"
    write_source(source, [make_piece(11)])
    files = [("a.lig", source)]
    output_dir = tmp_path / "output"
    prediction = classify.Prediction(
        "NNBE", "NNBE/0500-0600km", (0, 0, 1, 0, 0), 1, 5, 550, 1
    )

    classify._classify_files(
        files, output_dir, batch_size=1,
        predict=lambda waveforms, _: [prediction] * len(waveforms),
        checkpoint_schema="test", checkpoint_hash="abc",
        output_types=["NNBE"], run_mode="test",
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        classify._classify_files(
            files, output_dir, batch_size=1,
            predict=lambda waveforms, _: [prediction] * len(waveforms),
            checkpoint_schema="test", checkpoint_hash="changed",
            output_types=["NNBE"], run_mode="test", resume=True,
        )


def test_classification_preserves_piece_with_missing_timestamp(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "input"
    piece = bytearray(make_piece(11, hour=0, marker=31))
    piece[108:136] = bytes(28)
    source = input_dir / "incoming" / "GZ_000000000000.0128000.lig"
    write_source(source, [bytes(piece)], marker=43)
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"synthetic-checkpoint")
    model = ConstantNewModel(type_index=2)
    monkeypatch.setattr(
        classify,
        "load_model_checkpoint",
        lambda *_: loaded_new(model),
    )

    with pytest.warns(RuntimeWarning, match="invalid timestamps"):
        csv_path = classify.classify_directory(
            input_dir,
            output_dir,
            model_path,
            batch_size=1,
            type_only=True,
            device="cpu",
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    output_file = output_dir / Path(row["output_file"])
    assert output_file.name == "GZ_unknown.lig"
    assert output_file.read_bytes()[FILE_HEADER_BYTES:] == bytes(piece)
    assert [
        batch.flatten().tolist() for batch in model.daylight_batches
    ] == [[0.0], [1.0]]


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

    output_files = sorted(
        (output_dir / "NCG" / "0000-0100km").glob("*.lig")
    )
    counts = [struct.unpack_from("<i", path.read_bytes(), 4)[0] for path in output_files]
    assert counts == [MAX_PIECES_PER_FILE, 3]
    assert [path.name for path in output_files] == [
        "GZ_20200102000405.lig",
        "GZ_20200102000405_002.lig",
    ]
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
        "input_dir",
        "input_root",
        "start_date",
        "end_date",
        "output_dir",
        "model",
        "model_dir",
            "batch_size",
        "type_only",
        "direct_type",
        "decision_config",
        "type_verifier_config",
        "output_type",
        "resume",
        "skip_io_errors",
        "device",
        "prefix",
        "temporal_context",
        "temporal_history",
        "temporal_trigger",
        "temporal_anchor_confidence",
        "temporal_min_probability",
    }
    selected = classify.build_arg_parser().parse_args([
        "--input_dir", "input", "--output_dir", "output", "--model", "model.pt",
        "--output_type", "nnbe", "--output_type", "PNBE", "--resume",
        "--skip_io_errors", "--temporal_context", "--temporal_history", "128",
    ])
    assert selected.output_type == ["NNBE", "PNBE"]
    assert selected.resume is True
    assert selected.skip_io_errors is True
    assert selected.temporal_context is True
    assert selected.temporal_history == 128

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


def test_date_range_is_inclusive_and_excludes_index(tmp_path):
    for name in (
        "GZ_20160702",
        "GZ_20160702Index",
        "GZ_20160703",
        "GZ_20160703Index",
    ):
        (tmp_path / name).mkdir()

    discovered = classify.discover_date_inputs(
        tmp_path, "20160702", "20160703"
    )

    assert discovered == [
        tmp_path / "GZ_20160702",
        tmp_path / "GZ_20160703",
    ]


def test_date_range_reports_every_missing_day(tmp_path):
    (tmp_path / "GZ_20160702").mkdir()

    with pytest.raises(ValueError, match="20160703.*20160704"):
        classify.discover_date_inputs(
            tmp_path, "20160702", "20160704"
        )


def test_bundle_prediction_routes_to_matching_distance_model():
    type_checkpoint = loaded_new(ConstantNewModel(type_index=2))
    distances = tuple(
        loaded_new(
            ConstantNewModel(
                type_index=type_index,
                distance_bin=type_index + 5,
            )
        )
        for type_index in range(1, 5)
    )
    bundle = LoadedModelBundle(
        type_checkpoint=type_checkpoint,
        distance_checkpoints=distances,
        hashes={
            role: role.lower()
            for role in ("type", "NCG", "NNBE", "PCG", "PNBE")
        },
    )
    waveforms = np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32)

    predictions = classify.predict_bundle_batch(
        bundle,
        waveforms,
        [datetime(2016, 7, 8, 1), datetime(2016, 7, 8, 2)],
        device=torch.device("cpu"),
        type_only=False,
    )

    assert [item.final_type for item in predictions] == ["NNBE", "NNBE"]
    assert [item.distance_bin for item in predictions] == [7, 7]


def _loaded_hierarchical_type_checkpoint():
    from checkpoints import (
        HIERARCHICAL_FIVE_CLASS_SCHEMA,
        LoadedCheckpoint,
    )
    from evaluation import (
        conservative_hierarchical_config,
        decision_config_dict,
    )
    from models import create_hierarchical_type_model

    return LoadedCheckpoint(
        model=create_hierarchical_type_model(
            base_channels=8,
            embedding_dim=16,
            prototypes_per_class=2,
        ),
        schema=HIERARCHICAL_FIVE_CLASS_SCHEMA,
        type_names=tuple(classify.TYPE_NAMES),
        distance_names=tuple(classify.DISTANCE_NAMES),
        distance_bins_km=tuple(classify.DISTANCE_BINS_KM),
        preprocess_config={
            "local_length": 8000,
            "global_length": 2000,
            "local_center_mode": "energy_envelope_v2",
            "local_energy_window": 128,
        },
        metadata={
            "decision_config": decision_config_dict(
                conservative_hierarchical_config()
            )
        },
    )


def test_hierarchical_checkpoint_preserves_preprocess_contract():
    config = classify._new_preprocess_config(
        _loaded_hierarchical_type_checkpoint()
    )

    assert config.local_center_mode == "energy_envelope_v2"
    assert config.local_energy_window == 128


def _external_decision_payload():
    return {
        "schema": classify.DECISION_CONFIG_SCHEMA,
        "decision_config": {
            "known_probability_thresholds": [0.3, 0.4, 0.5, 0.6],
            "prototype_similarity_thresholds": [-0.2, -0.1, 0.0, 0.1],
            "max_ic_gate_probabilities": [0.7, 0.8, 0.9, 1.0],
            "max_js_divergences": [0.01, 0.02, 0.03, 0.04],
            "min_branch_votes": 2,
        },
    }


def test_external_decision_config_is_validated_and_does_not_mutate_checkpoint(
    tmp_path,
):
    path = tmp_path / "decision.json"
    path.write_text(
        json.dumps(_external_decision_payload()),
        encoding="utf-8",
    )
    original = _loaded_hierarchical_type_checkpoint()

    loaded = classify.load_decision_config(path)
    overridden = classify._with_decision_config(original, loaded)

    assert overridden is not original
    assert overridden.model is original.model
    assert overridden.metadata["decision_config"] == loaded
    assert original.metadata["decision_config"] != loaded
    assert len(classify._decision_config_sha256(loaded)) == 64


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"decision_config": {}}, "must use schema"),
        (
            {
                **_external_decision_payload(),
                "decision_config": {
                    **_external_decision_payload()["decision_config"],
                    "max_js_divergences": [0.01, 0.02, 0.03, 2.0],
                },
            },
            "max_js_divergences",
        ),
    ],
)
def test_external_decision_config_rejects_malformed_values(
    tmp_path,
    payload,
    match,
):
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        classify.load_decision_config(path)


def test_external_v1_decision_config_is_explicitly_upgraded(tmp_path):
    payload = _external_decision_payload()
    payload["schema"] = "hierarchical_decision_config_v1"
    payload["decision_config"]["max_js_divergence"] = 0.02
    del payload["decision_config"]["max_js_divergences"]
    path = tmp_path / "legacy-decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = classify.load_decision_config(path)

    assert loaded["max_js_divergences"] == [0.02] * 4


def test_external_decision_config_is_recorded_in_csv(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    write_source(input_dir / "source.lig", [make_piece(11)])
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"hierarchical")
    config_path = tmp_path / "decision.json"
    config_path.write_text(
        json.dumps(_external_decision_payload()),
        encoding="utf-8",
    )
    checkpoint = _loaded_hierarchical_type_checkpoint()
    seen = {}
    monkeypatch.setattr(
        classify,
        "load_model_checkpoint",
        lambda *_: checkpoint,
    )

    def infer_with_override(
        configured,
        local,
        _global_view,
        _daylight,
        _alternate_daylight,
        _missing,
    ):
        seen.update(configured.metadata["decision_config"])
        logits = local.new_zeros((len(local), 5))
        return logits, torch.zeros(
            len(local), dtype=torch.long, device=local.device
        )

    monkeypatch.setattr(
        classify,
        "_hierarchical_type_inference",
        infer_with_override,
    )
    csv_path = classify.classify_directory(
        input_dir,
        tmp_path / "output",
        model_path,
        type_only=True,
        decision_config=config_path,
        device="cpu",
    )

    expected = classify.load_decision_config(config_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert seen == expected
    assert row["decision_config_sha256"] == (
        classify._decision_config_sha256(expected)
    )


def test_hierarchical_single_checkpoint_rejects_to_ic():
    checkpoint = _loaded_hierarchical_type_checkpoint()
    waveforms = np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32)

    predictions = classify.predict_batch(
        checkpoint,
        waveforms,
        [datetime(2016, 7, 8, 1), datetime(2016, 7, 8, 2)],
        device=torch.device("cpu"),
        type_only=True,
    )

    assert [item.final_type for item in predictions] == ["IC", "IC"]
    with pytest.raises(ValueError, match="requires --type_only"):
        classify.predict_batch(
            checkpoint,
            waveforms,
            [datetime(2016, 7, 8, 1), datetime(2016, 7, 8, 2)],
            device=torch.device("cpu"),
            type_only=False,
        )


def test_hierarchical_direct_type_uses_five_class_argmax(monkeypatch):
    checkpoint = _loaded_hierarchical_type_checkpoint()

    def rejected_known(
        _checkpoint,
        local,
        _global_view,
        _daylight,
        _alternate_daylight,
        _missing,
    ):
        logits = local.new_zeros((len(local), 5))
        logits[:, 2] = 10.0
        return logits, torch.zeros(
            len(local), dtype=torch.long, device=local.device
        )

    monkeypatch.setattr(
        classify, "_hierarchical_type_inference", rejected_known
    )
    predictions = classify.predict_batch(
        checkpoint,
        np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32),
        [datetime(2016, 7, 8, 1), datetime(2016, 7, 8, 2)],
        device=torch.device("cpu"),
        type_only=True,
        direct_type=True,
    )

    assert [item.final_type for item in predictions] == ["NNBE", "NNBE"]


def test_hierarchical_bundle_routes_stable_known_type(monkeypatch):
    type_checkpoint = _loaded_hierarchical_type_checkpoint()
    distances = tuple(
        loaded_new(
            ConstantNewModel(
                type_index=type_index,
                distance_bin=type_index + 5,
            )
        )
        for type_index in range(1, 5)
    )
    bundle = LoadedModelBundle(
        type_checkpoint=type_checkpoint,
        distance_checkpoints=distances,
        hashes={
            role: role.lower()
            for role in ("type", "NCG", "NNBE", "PCG", "PNBE")
        },
    )

    def stable_nbbe(
        _checkpoint,
        local,
        _global_view,
        _daylight,
        _alternate_daylight,
        _missing,
    ):
        logits = local.new_zeros((len(local), 5))
        logits[:, 2] = 10.0
        return logits, torch.full(
            (len(local),), 2, dtype=torch.long, device=local.device
        )

    monkeypatch.setattr(
        classify, "_hierarchical_type_inference", stable_nbbe
    )
    waveforms = np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32)
    predictions = classify.predict_bundle_batch(
        bundle,
        waveforms,
        [datetime(2016, 7, 8, 1), datetime(2016, 7, 8, 2)],
        device=torch.device("cpu"),
        type_only=False,
    )

    assert [item.final_type for item in predictions] == ["NNBE", "NNBE"]
    assert [item.distance_bin for item in predictions] == [7, 7]


class _IcWithNbeCandidate(ConstantNewModel):
    def forward_type(self, local, global_view, daylight):
        logits = local.new_tensor([1.2, -3, 1.0, -3, -3]).expand(len(local), 5)
        return logits, logits[:, :1]


def _stream_test_bundle():
    return LoadedModelBundle(
        type_checkpoint=loaded_new(_IcWithNbeCandidate()),
        distance_checkpoints=tuple(loaded_new(ConstantNewModel(k, k + 5)) for k in range(1, 5)),
        hashes={role: role for role in ("type", "NCG", "NNBE", "PCG", "PNBE")},
    )


def test_temporal_promotion_runs_the_newly_selected_distance_expert():
    from data.preprocess import TemporalContextConfig, TemporalContextState

    state = TemporalContextState(TemporalContextConfig(
        history_size=4, trigger_count=2, min_candidate_probability=.2))
    state.seed([(2, 1.), (2, 1.)])
    bundle = _stream_test_bundle()
    result = classify.predict_bundle_batch(bundle, np.zeros((1, 16000)),
        [datetime(2024, 1, 1)], device=torch.device("cpu"), type_only=False,
        temporal_context=state)[0]
    assert result.final_type == "NNBE" and result.distance_bin == 7
    assert bundle.distance_checkpoints[1].model.batch_sizes == [1]
    assert not bundle.distance_checkpoints[0].model.batch_sizes


def test_streaming_inference_loads_once_routes_distance_and_restores(monkeypatch):
    from data.preprocess import TemporalContextConfig
    from datetime import timedelta

    loads = []
    def load(path, device):
        loads.append(path)
        return _stream_test_bundle()
    monkeypatch.setattr(classify, "load_model_bundle", load)
    config = TemporalContextConfig(history_size=4, trigger_count=2, min_candidate_probability=.2)
    stream = classify.StreamingClassifier("synthetic", temporal_config=config, max_gap_seconds=10)
    timestamp = datetime(2024, 1, 1)
    state, reason = stream.context.prepare("GZ", timestamp)
    state.seed([(2, 1.0), (2, 1.0)])
    stream.context.commit("GZ", timestamp, state, reason)
    before = stream.snapshot()
    result = stream.predict_piece(np.zeros(16000), stream_id="GZ", timestamp=timestamp + timedelta(seconds=1))
    assert result.prediction.final_type == "NNBE"
    assert result.prediction.distance_bin == 7 and result.temporal_promoted
    assert result.context_status == "continuous" and result.elapsed_ms > 0
    assert stream.predict_piece(np.zeros(16000), stream_id="ZH", timestamp=timestamp).prediction.final_type == "IC"
    assert loads == ["synthetic"]
    restarted = classify.StreamingClassifier("synthetic", temporal_config=config, max_gap_seconds=10)
    restarted.restore(json.loads(json.dumps(before)))
    replay = restarted.predict_piece(np.zeros(16000), stream_id="GZ", timestamp=timestamp + timedelta(seconds=1))
    assert replay.prediction == result.prediction and replay.temporal_promoted
    bad = {**before, "model_hashes": {"type": "changed"}}
    with pytest.raises(ValueError, match="model configuration"):
        restarted.restore(bad)
    current = restarted.snapshot()
    with pytest.raises(ValueError, match="finite"):
        restarted.predict_piece(np.full(16000, np.nan), stream_id="GZ", timestamp=timestamp)
    assert restarted.snapshot() == current


def test_streaming_failure_does_not_advance_context(monkeypatch):
    from data.preprocess import TemporalContextConfig

    monkeypatch.setattr(classify, "load_model_bundle", lambda *_: _stream_test_bundle())
    stream = classify.StreamingClassifier("synthetic", temporal_config=TemporalContextConfig())
    before = stream.snapshot()
    def failed(*_args, temporal_context, **_kwargs):
        temporal_context.seed([(2, 1.)])
        raise OSError("simulated inference failure")
    monkeypatch.setattr(classify, "predict_bundle_batch", failed)
    with pytest.raises(OSError, match="simulated"):
        stream.predict_piece(np.zeros(16000), stream_id="GZ", timestamp=datetime(2024, 1, 1))
    assert stream.snapshot() == before


class _VerificationTypeModel(nn.Module):
    def __init__(self, candidate=2, alternate=None, branch=None, gate=0.01):
        super().__init__()
        self.candidate, self.alternate, self.branch, self.gate = candidate, alternate, branch, gate
        self.batch_sizes = []

    def forward(self, local, global_view, daylight):
        from models import HierarchicalTypeOutput

        candidate = self.alternate if self.alternate is not None and len(self.batch_sizes) % 2 else self.candidate
        self.batch_sizes.append(len(local))
        known = local.new_zeros((len(local), 4))
        known[:, candidate - 1] = 6.
        branch = known.clone()
        if self.branch is not None:
            branch.zero_()
            branch[:, self.branch - 1] = 6.
        gate = local.new_tensor([self.gate, 1. - self.gate]).log().expand(len(local), 2)
        logits = torch.cat((gate[:, :1], gate[:, 1:] + known.log_softmax(1)), 1)
        return HierarchicalTypeOutput(type_logits=logits, gate_logits=gate, known_logits=known,
            prototype_logits=known[:, :, None].expand(-1, -1, 2), prototype_scores=known.new_zeros(known.shape),
            local_known_logits=branch, global_known_logits=branch, embedding=local.new_zeros((len(local), 8)))


def _verification_bundle(*, primary=None, secondary=None, accepted=False):
    from checkpoints import GuardedTypeVerifier, decision_config_sha256

    def checkpoint(model, threshold):
        config = dict(known_probability_thresholds=[threshold] * 4,
                      prototype_similarity_thresholds=[-1.] * 4, max_js_divergences=[.1] * 4,
                      min_branch_votes=2, max_ic_gate_probabilities=[1.] * 4)
        return LoadedCheckpoint(model, "hierarchical_five_class_v2", TYPE_NAMES, DISTANCE_NAMES,
            DISTANCE_BINS, {"local_length": 8000, "global_length": 2000},
            {"decision_config": config, "split_hash": "synthetic", "training_config": {"role": "type"}})
    first = checkpoint(primary or _VerificationTypeModel(), .9 if accepted else 1.)
    second = checkpoint(secondary or _VerificationTypeModel(), .9)
    bundle = LoadedModelBundle(first,
        tuple(loaded_new(ConstantNewModel(k, k + 5)) for k in range(1, 5)),
        {role: "a" * 64 for role in ("type", *DISTANCE_NAMES)})
    verifier = GuardedTypeVerifier(first, second, ((.5, .5, .2, .2),) * 4, "f" * 64, "e" * 64,
        tuple(decision_config_sha256(cp.metadata["decision_config"]) for cp in (first, second)))
    return bundle, verifier


def test_guarded_verification_routes_recovered_distance_and_preserves_probabilities():
    bundle, verifier = _verification_bundle()
    raw = np.zeros((2, WAVEFORM_SAMPLES), dtype=np.float32)
    timestamps = [datetime(2024, 1, 1), None]
    baseline = classify.predict_bundle_batch(bundle, raw, timestamps, device=torch.device("cpu"), type_only=False)
    result = classify.predict_bundle_batch(bundle, raw, timestamps, device=torch.device("cpu"),
                                            type_only=False, verifier=verifier)
    assert [p.final_type for p in baseline] == ["IC", "IC"]
    assert [p.final_type for p in result] == ["NNBE", "IC"]
    assert result[0].distance_bin == 7 and result[1].distance_bin is None
    assert verifier.checkpoint.model.batch_sizes == [1, 1]
    assert bundle.distance_checkpoints[1].model.batch_sizes == [1, 1]
    assert all(not bundle.distance_checkpoints[k].model.batch_sizes for k in (0, 2, 3))
    assert [p.type_probabilities for p in result] == [p.type_probabilities for p in baseline]


@pytest.mark.parametrize("options,expected", [
    ({"accepted": True, "secondary": _VerificationTypeModel(candidate=4)}, "NNBE"),
    ({"primary": _VerificationTypeModel(gate=.9)}, "IC"),
    ({"primary": _VerificationTypeModel(alternate=4)}, "IC"),
    ({"primary": _VerificationTypeModel(branch=4)}, "IC"),
])
def test_verifier_is_not_called_for_accepted_or_unsafe_primary(options, expected):
    bundle, verifier = _verification_bundle(**options)
    result = classify.predict_bundle_batch(bundle, np.zeros((1, 16000)), [datetime(2024, 1, 1)],
        device=torch.device("cpu"), type_only=True, verifier=verifier)
    assert result[0].final_type == expected
    assert not verifier.checkpoint.model.batch_sizes


@pytest.mark.parametrize("secondary", [
    _VerificationTypeModel(candidate=4), _VerificationTypeModel(alternate=4),
    _VerificationTypeModel(gate=.9), _VerificationTypeModel(branch=4),
])
def test_verifier_disagreement_or_failed_evidence_remains_ic(secondary):
    bundle, verifier = _verification_bundle(secondary=secondary)
    result = classify.predict_bundle_batch(bundle, np.zeros((1, 16000)), [datetime(2024, 1, 1)],
        device=torch.device("cpu"), type_only=False, verifier=verifier)
    assert result[0].final_type == "IC" and result[0].distance_bin is None
    assert not any(cp.model.batch_sizes for cp in bundle.distance_checkpoints)


def test_verifier_cannot_be_reused_with_changed_checkpoint_or_thresholds():
    from dataclasses import replace

    bundle, verifier = _verification_bundle()
    raw, times = np.zeros((1, 16000)), [datetime(2024, 1, 1)]
    other = replace(bundle, type_checkpoint=replace(bundle.type_checkpoint))
    with pytest.raises(ValueError, match="configuration changed"):
        classify.predict_bundle_batch(other, raw, times, device=torch.device("cpu"), type_only=True, verifier=verifier)
    verifier.checkpoint.metadata["decision_config"]["max_ic_gate_probabilities"][0] = .01
    with pytest.raises(ValueError, match="configuration changed"):
        classify.predict_bundle_batch(bundle, raw, times, device=torch.device("cpu"), type_only=True, verifier=verifier)


def test_verifier_cli_csv_resume_and_raw_byte_preservation(tmp_path, monkeypatch):
    from dataclasses import replace

    bundle, verifier = _verification_bundle()
    monkeypatch.setattr(classify, "load_model_bundle", lambda *_: bundle)
    monkeypatch.setattr(classify, "load_type_verifier", lambda *_: verifier)
    root, output = tmp_path / "input", tmp_path / "output"
    pieces = [make_piece(100 + k, marker=k) for k in range(3)]
    write_source(root / "GZ_20200102" / "sample.lig", pieces)
    args = ["--input_root", str(root), "--start_date", "20200102", "--end_date", "20200102",
            "--output_dir", str(output), "--model_dir", "synthetic", "--device", "cpu",
            "--type_verifier_config", "synthetic.json", "--output_type", "NNBE"]
    csv_path = classify.main(args)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert len(rows) == 3 and all(r["final_type"] == "NNBE" and r["distance_bin"] == "7" for r in rows)
    assert all(r["decision_config_sha256"] == verifier.signature for r in rows)
    assert json.loads(rows[0]["model_hashes"])["type_verifier"] == verifier.sha256
    files = list(output.rglob("*.lig"))
    assert len(files) == 1 and files[0].read_bytes()[FILE_HEADER_BYTES:] == b"".join(pieces)
    before = csv_path.read_bytes()
    classify.main(args + ["--resume"])
    assert csv_path.read_bytes() == before
    verifier = replace(verifier, signature="b" * 64)
    with pytest.raises(ValueError, match="match|different|signature"):
        classify.main(args + ["--resume"])
    assert csv_path.read_bytes() == before


def test_streaming_verifier_restores_only_identical_recipe_and_loads_once(monkeypatch):
    from dataclasses import replace
    from data.preprocess import TemporalContextConfig

    bundle, verifier = _verification_bundle()
    loads = []
    monkeypatch.setattr(classify, "load_model_bundle", lambda *_: bundle)
    def load(*_args):
        loads.append(True)
        return verifier
    monkeypatch.setattr(classify, "load_type_verifier", load)
    service = classify.StreamingClassifier("synthetic", type_verifier_config="test.json")
    snapshot = service.snapshot()
    for _ in range(2):
        result = service.predict_piece(np.zeros(16000), stream_id="GZ", timestamp=datetime(2024, 1, 1))
        assert result.prediction.final_type == "NNBE" and result.prediction.distance_bin == 7
        assert not result.temporal_promoted
    assert len(loads) == 1
    service.restore(snapshot)
    verifier = replace(verifier, signature="b" * 64)
    changed = classify.StreamingClassifier("synthetic", type_verifier_config="test.json")
    with pytest.raises(ValueError, match="configuration mismatch"):
        changed.restore(snapshot)
    with pytest.raises(ValueError, match="cannot combine"):
        classify.StreamingClassifier("synthetic", type_verifier_config="test.json", temporal_config=TemporalContextConfig())


@pytest.mark.parametrize("options", [{"decision_config": "v3.json"}, {"direct_type": True}, {"temporal_config": object()}])
def test_verifier_rejects_unvalidated_policy_combinations(options):
    bundle, _ = _verification_bundle()
    with pytest.raises(ValueError, match="cannot combine"):
        classify._configured_verifier("test.json", bundle.type_checkpoint, "a" * 64, "cpu", **options)
