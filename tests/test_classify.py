import csv
import struct

import pytest
import torch

import classify
from models import MultiTaskOrdinalResNet, MultiTaskResNet, create_mtl_model
from tests.test_training_manifest import write_lig as write_test_lig


def make_checkpoint(architecture):
    model = create_mtl_model(
        base_channels=8,
        architecture=architecture,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )
    return {
        "model_name": architecture,
        "base_channels": 8,
        "dist_mlp_dim": 12,
        "dist_dropout": 0.0,
        "type_names": ["IC", "NCG", "NNBE", "PCG", "PNBE"],
        "dist_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "dist_bin_starts": [index * 100 for index in range(30)],
        "model_state_dict": model.state_dict(),
    }


def make_four_class_checkpoint(architecture="ordinal_v2"):
    model = create_mtl_model(
        base_channels=8,
        architecture=architecture,
        num_types=4,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )
    return {
        "task_schema": "four_class_rejection_v1",
        "model_name": architecture,
        "base_channels": 8,
        "dist_mlp_dim": 12,
        "dist_dropout": 0.0,
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "dist_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "dist_bin_starts": [index * 100 for index in range(30)],
        "type_rejection": {
            "version": 1,
            "temperature": 1.0,
            "centroids": [[0.0, 0.0]] * 4,
            "scales": [[1.0, 1.0]] * 4,
            "probability_thresholds": [0.5] * 4,
            "margin_thresholds": [0.1] * 4,
            "distance_thresholds": [2.0] * 4,
        },
        "model_state_dict": model.state_dict(),
    }


def make_conditional_checkpoint(context_dim=1, include_model=False):
    checkpoint = {
        "task_schema": "four_class_rejection_v2",
        "model_name": "conditional_expert_v1",
        "model_version": "conditional-test",
        "context_dim": context_dim,
        "time_context": "daylight" if context_dim == 1 else "cyclic",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "rejected_type_name": "IC",
        "combined_split_hash": "split-hash",
        "type_rejection": {
            "version": 2,
            "temperature": 1.0,
            "centroids": [[0.0, 0.0]] * 4,
            "scales": [[1.0, 1.0]] * 4,
            "probability_thresholds": [0.5] * 4,
            "margin_thresholds": [0.1] * 4,
            "distance_thresholds": [2.0] * 4,
            "quality_thresholds": [0.1] * 4,
            "calibration_split_hash": "validation-hash",
        },
        "distance_calibration": {"temperatures": [1.0] * 4},
    }
    if include_model:
        model = create_mtl_model(
            base_channels=8,
            architecture="conditional_expert_v1",
            num_types=4,
            dist_mlp_dim=12,
            dist_dropout=0.0,
            context_dim=context_dim,
        )
        checkpoint.update({
            "base_channels": 8,
            "dist_mlp_dim": 12,
            "dist_dropout": 0.0,
            "model_state_dict": model.state_dict(),
        })
    return checkpoint


def test_load_checkpoint_selects_legacy_and_v2_architectures(tmp_path):
    legacy_path = tmp_path / "legacy.pt"
    v2_path = tmp_path / "v2.pt"
    torch.save(make_checkpoint("mtl_resnet"), legacy_path)
    torch.save(make_checkpoint("ordinal_v2"), v2_path)

    legacy, _ = classify.load_mtl_checkpoint(legacy_path, "cpu")
    v2, _ = classify.load_mtl_checkpoint(v2_path, "cpu")

    assert isinstance(legacy, MultiTaskResNet)
    assert isinstance(v2, MultiTaskOrdinalResNet)


def test_load_checkpoint_uses_four_class_type_head(tmp_path):
    path = tmp_path / "four.pt"
    torch.save(make_four_class_checkpoint(), path)

    model, checkpoint = classify.load_mtl_checkpoint(path, "cpu")

    assert model.type_head.out_features == 4
    assert classify.checkpoint_schema(checkpoint) == "four_class_rejection_v1"


@pytest.mark.parametrize(
    ("context_dim", "stored_context_dim", "expected_mode"),
    [(1, True, "daylight"), (3, False, "cyclic")],
)
def test_load_conditional_checkpoint_preserves_context_compatibility(
    tmp_path, context_dim, stored_context_dim, expected_mode
):
    path = tmp_path / f"conditional-{context_dim}.pt"
    checkpoint = make_conditional_checkpoint(context_dim, include_model=True)
    if not stored_context_dim:
        checkpoint.pop("context_dim")
        checkpoint.pop("time_context")
    torch.save(checkpoint, path)

    model, loaded = classify.load_mtl_checkpoint(path, "cpu")

    assert model.context_dim == context_dim
    assert classify.checkpoint_time_context_mode(loaded) == expected_mode


def test_four_class_rejection_routes_failed_feature_to_ic():
    checkpoint = make_four_class_checkpoint()

    predictions = classify.decode_four_class_predictions(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[20.0, 20.0]]),
        checkpoint,
    )

    assert predictions[0]["raw_type"] == "NCG"
    assert predictions[0]["type"] == "IC"
    assert predictions[0]["class_name"] == "IC"
    assert predictions[0]["rejection_reason"] == "feature_distance"
    assert predictions[0]["status"] == "rejected"


def test_four_class_full_prediction_uses_per_type_distance_mode():
    checkpoint = make_four_class_checkpoint()
    checkpoint["distance_training"] = {
        "prediction": "expected",
        "prediction_by_type": {
            "NCG": "argmax",
            "NNBE": "argmax",
            "PCG": "expected",
            "PNBE": "expected",
        },
    }
    checkpoint["distance_calibration"] = {
        "temperatures": [1.0] * 4,
        "confidence_threshold": 0.0,
    }
    distance_logits = [torch.full((1, 30), -20.0) for _ in range(4)]
    distance_logits[0][0, 4] = 20.0

    predictions = classify.decode_four_class_full_predictions(
        torch.tensor([[8.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        distance_logits,
        checkpoint,
    )

    assert predictions[0]["type"] == "NCG"
    assert predictions[0]["bin_start_km"] == 400
    assert predictions[0]["head_source"] == "single"
    assert predictions[0]["rejection_reason"] == "accepted"


def test_four_class_full_prediction_keeps_rejected_piece_without_distance():
    checkpoint = make_four_class_checkpoint()
    distance_logits = [torch.zeros(1, 30) for _ in range(4)]

    predictions = classify.decode_four_class_full_predictions(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[20.0, 20.0]]),
        distance_logits,
        checkpoint,
    )

    assert predictions[0]["type"] == "IC"
    assert predictions[0]["distance_km"] is None
    assert predictions[0]["class_name"] == "IC"


def test_conditional_checkpoint_routes_context_expert_and_distance():
    checkpoint = make_conditional_checkpoint()
    distance_logits = [torch.full((1, 30), -20.0) for _ in range(4)]
    distance_logits[0][0, 4] = 20.0

    predictions = classify.decode_conditional_batch(
        torch.tensor([[8.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        distance_logits,
        quality=torch.tensor([[10.0, 0.0, 0.0]]),
        checkpoint=checkpoint,
        daylight=torch.tensor([1.0]),
    )

    assert predictions[0]["final_type"] == "NCG"
    assert predictions[0]["expected_distance_km"] == pytest.approx(450.0)
    assert predictions[0]["distance_low_km"] <= predictions[0]["distance_high_km"]
    assert set(predictions[0]["type_probabilities"]) == {
        "NCG", "NNBE", "PCG", "PNBE"
    }
    assert predictions[0]["model_version"] == "conditional-test"
    assert predictions[0]["split_hash"] == "split-hash"


def test_conditional_rejected_piece_has_no_distance_and_keeps_probabilities():
    checkpoint = make_conditional_checkpoint()
    distance_logits = [torch.zeros((1, 30)) for _ in range(4)]

    prediction = classify.decode_conditional_batch(
        torch.tensor([[8.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        distance_logits,
        quality=torch.tensor([[0.1, 1.0, 20.0]]),
        checkpoint=checkpoint,
        daylight=torch.tensor([0.0]),
    )[0]

    assert prediction["final_type"] == prediction["class_name"] == "IC"
    assert prediction["expected_distance_km"] is None
    assert prediction["distance_low_km"] is None
    assert len(prediction["type_probabilities"]) == 4
    assert prediction["rejection_reason"] == "low_quality"


def test_v2_route_marks_low_confidence_prediction_uncertain():
    checkpoint = make_checkpoint("ordinal_v2")
    checkpoint["distance_training"] = {"prediction": "expected"}
    checkpoint["distance_calibration"] = {
        "temperatures": [1.0, 1.0, 1.0, 1.0],
        "confidence_threshold": 0.8,
    }
    distance_logits = [torch.zeros(30) for _ in range(4)]

    prediction = classify.decode_piece_prediction(
        type_index=2,
        distance_logits=distance_logits,
        checkpoint=checkpoint,
    )

    assert prediction["status"] == "uncertain"
    assert prediction["class_name"] == "UNCERTAIN_NNBE"


def test_v2_route_exports_reliable_expected_distance():
    checkpoint = make_checkpoint("ordinal_v2")
    checkpoint["distance_training"] = {"prediction": "expected"}
    checkpoint["distance_calibration"] = {
        "temperatures": [1.0, 1.0, 1.0, 1.0],
        "confidence_threshold": 0.8,
    }
    distance_logits = [torch.full((30,), -20.0) for _ in range(4)]
    distance_logits[1][4] = 20.0

    prediction = classify.decode_piece_prediction(
        type_index=2,
        distance_logits=distance_logits,
        checkpoint=checkpoint,
    )

    assert prediction["status"] == "reliable"
    assert prediction["class_name"] == "NNBE_400-500km"
    assert prediction["distance_km"] == 450.0
    assert prediction["low_km"] == 400.0
    assert prediction["high_km"] == 500.0


def test_hybrid_route_uses_old_type_and_old_ncg_distance():
    old_checkpoint = make_checkpoint("mtl_resnet")
    new_checkpoint = make_checkpoint("ordinal_v2")
    old_type_logits = torch.tensor([0.0, 9.0, 0.0, 0.0, 0.0])
    old_distance_logits = [torch.full((30,), -20.0) for _ in range(4)]
    new_distance_logits = [torch.full((30,), -20.0) for _ in range(4)]
    old_distance_logits[0][3] = 20.0
    new_distance_logits[0][12] = 20.0

    prediction = classify.decode_hybrid_prediction(
        old_type_logits,
        old_distance_logits,
        new_distance_logits,
        old_checkpoint,
        new_checkpoint,
    )

    assert prediction["type"] == "NCG"
    assert prediction["bin_start_km"] == 300
    assert prediction["head_source"] == "old"


def test_hybrid_route_uses_new_distance_for_non_ncg():
    old_checkpoint = make_checkpoint("mtl_resnet")
    new_checkpoint = make_checkpoint("ordinal_v2")
    new_checkpoint["distance_training"] = {"prediction": "expected"}
    old_type_logits = torch.tensor([0.0, 0.0, 9.0, 0.0, 0.0])
    old_distance_logits = [torch.full((30,), -20.0) for _ in range(4)]
    new_distance_logits = [torch.full((30,), -20.0) for _ in range(4)]
    old_distance_logits[1][3] = 20.0
    new_distance_logits[1][7] = 20.0

    prediction = classify.decode_hybrid_prediction(
        old_type_logits,
        old_distance_logits,
        new_distance_logits,
        old_checkpoint,
        new_checkpoint,
    )

    assert prediction["type"] == "NNBE"
    assert prediction["bin_start_km"] == 700
    assert prediction["head_source"] == "new"


def test_hybrid_rejects_incompatible_class_mappings():
    old_checkpoint = make_checkpoint("mtl_resnet")
    new_checkpoint = make_checkpoint("ordinal_v2")
    new_checkpoint["dist_bin_starts"] = [index * 50 for index in range(30)]

    with pytest.raises(ValueError, match="dist_bin_starts"):
        classify.validate_checkpoint_pair(old_checkpoint, new_checkpoint)


def test_type_only_prediction_has_no_distance_fields():
    checkpoint = make_checkpoint("ordinal_v2")
    logits = torch.tensor([0.0, 0.0, 8.0, 0.0, 0.0])

    prediction = classify.decode_type_prediction(logits, checkpoint)

    assert prediction["type"] == prediction["class_name"] == "NNBE"
    assert prediction["distance_km"] is None
    assert prediction["bin_start_km"] is None
    assert prediction["head_source"] == "type"
    assert prediction["status"] == "reliable"


def test_type_only_routes_low_confidence_non_ic_to_uncertained_folder():
    checkpoint = make_checkpoint("ordinal_v2")
    logits = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])

    prediction = classify.decode_type_prediction(
        logits, checkpoint, min_confidence=0.85
    )

    assert prediction["type"] == "NCG"
    assert prediction["class_name"] == "uncertained_NCG"
    assert prediction["status"] == "uncertain"
    assert prediction["confidence"] == prediction["type_confidence"]
    assert prediction["confidence"] < 0.85


def test_type_only_exempts_ic_from_confidence_gate():
    checkpoint = make_checkpoint("ordinal_v2")
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])

    prediction = classify.decode_type_prediction(
        logits, checkpoint, min_confidence=0.85
    )

    assert prediction["type"] == prediction["class_name"] == "IC"
    assert prediction["confidence"] < 0.85
    assert prediction["status"] == "reliable"


def test_type_only_cli_uses_one_explicit_checkpoint():
    args = classify.build_arg_parser().parse_args([
        "--input_dir", "input", "--output_dir", "output",
        "--type_only", "--model", "fresh.pt",
    ])

    assert args.type_only is True
    assert args.model == "fresh.pt"
    assert args.min_type_confidence == 0.0


def test_four_class_cli_uses_single_model_for_type_and_distance():
    args = classify.build_arg_parser().parse_args([
        "--input_dir", "input", "--output_dir", "output",
        "--four_class", "--model", "candidate.pt",
    ])

    assert args.four_class is True
    assert args.type_only is False
    assert args.model == "candidate.pt"


def test_single_checkpoint_is_default_and_legacy_hybrid_is_explicit():
    args = classify.build_arg_parser().parse_args([
        "--input_dir", "input", "--output_dir", "output",
        "--model", "candidate.pt",
    ])

    assert args.legacy_hybrid is False
    assert args.model == "candidate.pt"


def test_four_class_checkpoint_rejects_legacy_confidence_override():
    with pytest.raises(ValueError, match="min_type_confidence"):
        classify.validate_type_only_options(
            make_four_class_checkpoint(), min_type_confidence=0.85
        )


def test_forward_type_bypasses_ordinal_distance_projection():
    model = create_mtl_model(
        base_channels=8, architecture="ordinal_v2", dist_mlp_dim=8, dist_dropout=0.0
    )

    def fail_if_called(*_):
        raise AssertionError("distance projection was called")

    handle = model.distance_projection.register_forward_pre_hook(fail_if_called)
    try:
        logits = model.forward_type(torch.randn(2, 1, 128))
    finally:
        handle.remove()

    assert logits.shape == (2, 5)


def test_prediction_csv_writer_has_reliability_columns(tmp_path):
    output = tmp_path / "predictions.csv"
    with classify.PredictionCsvWriter(output) as writer:
        writer.write({
            "source_file": "sample.lig",
            "piece_index": 7,
            "type": "NNBE",
            "distance_km": 450.0,
            "bin_start_km": 400,
            "low_km": 400.0,
            "high_km": 500.0,
            "confidence": 0.99,
            "status": "reliable",
        })

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert set(row) == set(classify.PREDICTION_FIELDS)
    assert row["status"] == "reliable"
    assert {
        "raw_type",
        "type_margin",
        "feature_distance",
        "rejection_reason",
        "final_type",
    }.issubset(row)
    assert {
        "prob_NCG",
        "prob_NNBE",
        "prob_PCG",
        "prob_PNBE",
        "expected_distance_km",
        "distance_low_km",
        "distance_high_km",
        "daylight",
        "snr_score",
        "clipping_fraction",
        "model_version",
        "split_hash",
    }.issubset(row)


def test_piece_stream_preserves_raw_bytes_and_reads_real_timestamp(tmp_path):
    source = write_test_lig(
        tmp_path / "GZ_160702000049.2536254.lig",
        timestamp=(16, 7, 2, 0, 0, 49),
        sec_frac=0.2536254,
        pieces=2,
    )
    source_bytes = source.read_bytes()

    waveforms, raw_pieces, timestamps, indices = next(
        classify.read_pieces_stream(source, chunk=2)
    )

    assert indices == [0, 1]
    assert timestamps[0].startswith("160702000049.2536254")
    assert raw_pieces[0] == source_bytes[112:112 + 32208]
    assert raw_pieces[1] == source_bytes[112 + 32208:112 + 2 * 32208]
    assert waveforms[0][0] == 1
    assert waveforms[1][0] == 2


def test_write_lig_keeps_each_piece_byte_identical(tmp_path):
    source = write_test_lig(
        tmp_path / "source.lig",
        timestamp=(16, 7, 2, 0, 0, 49),
        sec_frac=0.25,
        pieces=2,
    )
    source_bytes = source.read_bytes()
    raw_pieces = [
        source_bytes[112:112 + 32208],
        source_bytes[112 + 32208:112 + 2 * 32208],
    ]
    output = tmp_path / "classified" / "result.lig"

    classify.write_lig(raw_pieces, output, bytes(112))

    written = output.read_bytes()
    assert struct.unpack_from("<i", written, 4)[0] == 2
    assert written[112:112 + 32208] == raw_pieces[0]
    assert written[112 + 32208:112 + 2 * 32208] == raw_pieces[1]
