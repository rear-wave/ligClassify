import numpy as np
import pytest
import torch

import open_set
from data.split_artifacts import stable_json_hash
from evaluation import evaluate_predictions
from open_set import (
    attach_final_feature_reference,
    decode_with_rejection,
    fit_feature_reference,
    fit_oof_rejection_policy,
    fit_rejection_policy,
    rejection_signals,
)


def separable_oof_signals():
    true_type = np.repeat(np.arange(4), 25)
    predicted = true_type.copy()
    file_id = np.repeat([f"correct-{index}" for index in range(4)], 25)
    true_type = np.concatenate([true_type, np.arange(4)])
    predicted = np.concatenate([predicted, np.roll(np.arange(4), -1)])
    return {
        "true_type": true_type,
        "predicted": predicted,
        "file_id": np.concatenate([file_id, [f"wrong-{i}" for i in range(4)]]),
        "confidence": np.concatenate([np.full(100, 0.99), np.full(4, 0.40)]),
        "margin": np.concatenate([np.full(100, 0.95), np.full(4, 0.05)]),
        "normalized_distance": np.concatenate([np.full(100, 0.1), np.full(4, 3.0)]),
        "quality_score": np.ones(104),
    }


def test_oof_policy_report_matches_evaluator_file_equal_precision():
    signals = separable_oof_signals()
    policy, decisions = fit_oof_rejection_policy(
        signals, target_precision=0.96, min_coverage=0.80
    )
    records = [{
        "file_id": signals["file_id"][index],
        "true_type": int(signals["true_type"][index]),
        "predicted_type": int(signals["predicted"][index]),
        "accepted": bool(decisions["accepted"][index]),
        "distance_low_km": 0,
        "distance_high_km": 100,
        "predicted_distance_km": 50.0,
        "oracle_distance_km": 50.0,
        "daylight": True,
    } for index in range(len(signals["true_type"]))]
    metrics = evaluate_predictions(records)
    assert policy["oof_metrics"]["type_file_equal_precision"] == pytest.approx(
        metrics["type_file_equal_precision"]
    )


def test_raw_feature_coordinates_are_not_transferred_between_models():
    template = {
        "version": 3,
        "temperature": 1.2,
        "probability_thresholds": [0.8] * 4,
        "margin_thresholds": [0.5] * 4,
        "normalized_distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.1] * 4,
        "fold_centroids": [[999.0, 999.0]] * 4,
    }
    final_features = torch.tensor([
        [0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0],
        [20.0, 20.0], [22.0, 22.0], [30.0, 30.0], [32.0, 32.0],
    ])
    labels = torch.repeat_interleave(torch.arange(4), 2)
    final = attach_final_feature_reference(template, final_features, labels)
    assert final["centroids"] != template["fold_centroids"]
    assert final["normalized_distance_thresholds"] == template["normalized_distance_thresholds"]


def test_rejection_signals_have_aligned_tensor_shapes_and_quality_scores():
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    features = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    reference = {
        "centroids": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        "scales": [[1.0, 1.0]] * 4,
    }

    signals = rejection_signals(
        logits, features, reference, temperature=1.0, quality=torch.tensor([0.2, 0.8])
    )

    assert signals["predicted"].dtype == torch.long
    assert signals["predicted"].shape == (2,)
    for name in ("confidence", "margin", "normalized_distance", "quality_score"):
        assert signals[name].dtype == torch.float32
        assert signals[name].shape == (2,)
    assert signals["quality_score"].tolist() == pytest.approx([0.2, 0.8])


def test_oof_policy_v3_fields_and_calibration_hash_are_stable():
    signals = separable_oof_signals()
    signals["piece_key"] = [f"piece-{index}" for index in range(104)]
    fold_temperatures = {2: 1.4, 0: 0.8}
    fold_hashes = {"2": "fold-two", "0": "fold-zero"}

    policy, decisions = fit_oof_rejection_policy(
        signals,
        fold_temperatures=fold_temperatures,
        fold_hashes=fold_hashes,
    )

    assert set(policy) == {
        "version", "temperature", "fold_temperatures",
        "probability_thresholds", "margin_thresholds",
        "normalized_distance_thresholds", "quality_thresholds",
        "target_precision", "minimum_coverage", "oof_metrics", "fold_hashes",
        "calibration_hash",
    }
    assert policy["version"] == 3
    assert policy["fold_temperatures"] == {"0": 0.8, "2": 1.4}
    assert policy["temperature"] == pytest.approx(1.1)
    expected_hash = stable_json_hash({
        "fold_hashes": fold_hashes,
        "piece_keys": list(signals["piece_key"]),
        "accepted": [bool(value) for value in decisions["accepted"]],
        "probability_thresholds": [
            round(float(value), 12) for value in policy["probability_thresholds"]
        ],
        "margin_thresholds": [
            round(float(value), 12) for value in policy["margin_thresholds"]
        ],
        "normalized_distance_thresholds": [
            round(float(value), 12)
            for value in policy["normalized_distance_thresholds"]
        ],
        "quality_thresholds": [
            round(float(value), 12) for value in policy["quality_thresholds"]
        ],
    })
    assert policy["calibration_hash"] == expected_hash


def test_oof_post_fit_verification_hard_rejects_precision_below_contract(monkeypatch):
    signals = separable_oof_signals()
    real_evaluate_predictions = evaluate_predictions

    def below_contract(records):
        metrics = real_evaluate_predictions(records)
        metrics["type_file_equal_precision"] = [0.95] * 4
        return metrics

    monkeypatch.setattr(open_set, "evaluate_predictions", below_contract)

    with pytest.raises(ValueError, match="post-fit.*precision"):
        fit_oof_rejection_policy(signals)


def test_version_three_policy_decodes_with_normalized_distance_thresholds():
    policy = {
        "version": 3,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.40] * 4,
        "margin_thresholds": [0.20] * 4,
        "normalized_distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.0] * 4,
    }

    decoded = decode_with_rejection(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[20.0, 20.0]]),
        policy,
    )

    assert decoded["accepted"].tolist() == [False]
    assert decoded["reason"] == ["feature_distance"]


def test_legacy_policy_ignores_conflicting_normalized_distance_thresholds():
    policy = {
        "version": 2,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.40] * 4,
        "margin_thresholds": [0.20] * 4,
        "distance_thresholds": [30.0] * 4,
        "normalized_distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.0] * 4,
    }

    decoded = decode_with_rejection(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[20.0, 20.0]]),
        policy,
    )

    assert decoded["accepted"].tolist() == [True]


def test_version_three_policy_cannot_fall_back_to_legacy_distance_thresholds():
    policy = {
        "version": 3,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.40] * 4,
        "margin_thresholds": [0.20] * 4,
        "distance_thresholds": [30.0] * 4,
        "quality_thresholds": [0.0] * 4,
    }

    with pytest.raises(
        ValueError, match="version 3.*normalized_distance_thresholds"
    ):
        decode_with_rejection(
            torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[20.0, 20.0]]),
            policy,
        )


def test_oof_policy_preserves_integer_fold_hashes_in_policy_and_hash():
    signals = separable_oof_signals()
    fold_hashes = {2: 202, 0: 100}

    policy, decisions = fit_oof_rejection_policy(signals, fold_hashes=fold_hashes)

    assert policy["fold_hashes"] == fold_hashes
    expected_hash = stable_json_hash({
        "fold_hashes": fold_hashes,
        "piece_keys": [str(index) for index in range(104)],
        "accepted": [bool(value) for value in decisions["accepted"]],
        "probability_thresholds": [
            round(float(value), 12) for value in policy["probability_thresholds"]
        ],
        "margin_thresholds": [
            round(float(value), 12) for value in policy["margin_thresholds"]
        ],
        "normalized_distance_thresholds": [
            round(float(value), 12)
            for value in policy["normalized_distance_thresholds"]
        ],
        "quality_thresholds": [
            round(float(value), 12) for value in policy["quality_thresholds"]
        ],
    })
    assert policy["calibration_hash"] == expected_hash


def test_feature_reference_uses_per_type_center_and_scale():
    features = torch.tensor([
        [0.0, 0.0], [2.0, 2.0],
        [10.0, 20.0], [14.0, 24.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])

    reference = fit_feature_reference(features, labels, num_types=2)

    assert reference["centroids"] == [[1.0, 1.0], [12.0, 22.0]]
    assert reference["scales"] == [[1.0, 1.0], [2.0, 2.0]]


def test_decode_reports_low_margin_before_feature_distance():
    policy = {
        "version": 1,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.40] * 4,
        "margin_thresholds": [0.20] * 4,
        "distance_thresholds": [2.0] * 4,
    }

    decoded = decode_with_rejection(
        torch.tensor([
            [2.0, 1.9, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]),
        torch.tensor([[0.0, 0.0], [20.0, 20.0]]),
        policy,
    )

    assert decoded["accepted"].tolist() == [False, False]
    assert decoded["reason"] == ["low_margin", "feature_distance"]


def test_fit_policy_meets_precision_recall_and_keeps_separable_samples():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    logits = torch.full((8, 4), -4.0)
    logits[torch.arange(8), labels] = 4.0
    features = torch.stack([
        torch.tensor([float(label * 10), float(label * 10)])
        for label in labels
    ])
    reference = fit_feature_reference(features, labels)

    policy = fit_rejection_policy(
        logits,
        features,
        labels,
        reference,
        precision_floor=0.85,
        recall_floor=0.70,
    )

    assert all(value >= 0.85 for value in policy["validation_precision"])
    assert all(value >= 0.70 for value in policy["validation_recall"])
    assert policy["validation_coverage"] == pytest.approx(1.0)
    assert decode_with_rejection(logits, features, policy)["accepted"].all()


def test_fit_policy_fails_when_a_type_has_no_correct_prediction():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    logits = torch.full((8, 4), -2.0)
    logits[:, 0] = 2.0
    features = torch.randn(8, 2)
    reference = fit_feature_reference(features, labels)

    with pytest.raises(ValueError, match="type 1"):
        fit_rejection_policy(logits, features, labels, reference)


def test_low_quality_piece_is_rejected_after_confident_prediction():
    policy = {
        "version": 2,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.40] * 4,
        "margin_thresholds": [0.20] * 4,
        "distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.40] * 4,
        "calibration_split_hash": "validation-hash",
    }

    decoded = decode_with_rejection(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        policy,
        quality=torch.tensor([0.10]),
    )

    assert decoded["accepted"].tolist() == [False]
    assert decoded["final_type"].tolist() == [-1]
    assert decoded["reason"] == ["low_quality"]


def test_fit_policy_stores_quality_thresholds_and_calibration_split_hash():
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    logits = torch.full((8, 4), -4.0)
    logits[torch.arange(8), labels] = 4.0
    features = torch.stack([
        torch.tensor([float(label * 10), float(label * 10)])
        for label in labels
    ])
    quality = torch.tensor([[10.0, 0.0, 0.0]] * 8)
    reference = fit_feature_reference(features, labels)

    policy = fit_rejection_policy(
        logits,
        features,
        labels,
        reference,
        quality=quality,
        target_precision=0.95,
        min_coverage=0.80,
        calibration_split_hash="validation-hash",
    )

    assert policy["version"] == 2
    assert len(policy["quality_thresholds"]) == 4
    assert policy["calibration_split_hash"] == "validation-hash"
    assert min(policy["validation_precision"]) >= 0.95
    assert policy["validation_coverage"] >= 0.80
