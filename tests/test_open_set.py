import pytest
import torch

from open_set import (
    decode_with_rejection,
    fit_feature_reference,
    fit_rejection_policy,
)


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
