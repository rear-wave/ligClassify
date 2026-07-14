import pytest
import torch
import torch.nn.functional as F

from distance_ordinal import (
    aggregate_coarse_probabilities,
    decode_distance_logits,
    fit_temperature_grid,
    is_meaningful_improvement,
    make_selection_key,
    ordinal_distance_loss,
    select_confidence_threshold,
)


def test_coarse_probabilities_are_normalized_groups_of_three():
    probs = torch.full((2, 30), 1 / 30)

    coarse = aggregate_coarse_probabilities(probs)

    assert coarse.shape == (2, 10)
    assert torch.allclose(coarse, torch.full((2, 10), 0.1))


def test_ordinal_loss_prefers_near_error_to_far_error():
    target = torch.tensor([10])
    near = torch.full((1, 30), -8.0)
    far = torch.full((1, 30), -8.0)
    near[0, 11] = 8.0
    far[0, 25] = 8.0

    near_loss, _ = ordinal_distance_loss(near, target)
    far_loss, _ = ordinal_distance_loss(far, target)

    assert near_loss < far_loss


def test_ordinal_loss_returns_named_finite_components():
    logits = torch.randn(4, 30, requires_grad=True)
    targets = torch.tensor([0, 4, 15, 29])

    loss, components = ordinal_distance_loss(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {"soft_ce", "cdf", "huber", "coarse"}
    assert all(torch.isfinite(value) for value in components.values())
    assert logits.grad is not None


def test_decoder_returns_expected_center_interval_and_confidence():
    logits = torch.full((1, 30), -20.0)
    logits[0, 4] = 20.0

    decoded = decode_distance_logits(logits)

    assert decoded["bin"].item() == 4
    assert decoded["distance_km"].item() == pytest.approx(450.0)
    assert decoded["low_km"].item() == pytest.approx(400.0)
    assert decoded["high_km"].item() == pytest.approx(500.0)
    assert decoded["confidence"].item() > 0.99


def test_temperature_grid_does_not_increase_nll():
    logits = torch.tensor([[8.0, -2.0], [8.0, -2.0], [-2.0, 8.0]])
    targets = torch.tensor([1, 0, 1])
    baseline = F.cross_entropy(logits, targets)

    temperature = fit_temperature_grid(logits, targets)
    calibrated = F.cross_entropy(logits / temperature, targets)

    assert 0.5 <= temperature <= 5.0
    assert calibrated <= baseline + 1e-7


def test_threshold_uses_maximum_coverage_that_meets_target():
    confidence = torch.tensor([0.9, 0.8, 0.2, 0.1])
    error_bins = torch.tensor([0.0, 1.0, 5.0, 8.0])

    result = select_confidence_threshold(
        confidence, error_bins, target_w2=0.8
    )

    assert result["threshold"] == pytest.approx(0.8)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["w2"] == pytest.approx(1.0)


def test_selection_key_protects_worst_class_before_guardrail():
    weaker = {
        "per_type_w2": [0.60, 0.90, 0.90, 0.90],
        "w2": 0.82,
        "macro_w2": 0.825,
        "macro_mae_km": 170,
        "type_f1": 0.92,
    }
    safer = {
        "per_type_w2": [0.65, 0.80, 0.80, 0.80],
        "w2": 0.78,
        "macro_w2": 0.7625,
        "macro_mae_km": 190,
        "type_f1": 0.91,
    }

    assert make_selection_key(safer) > make_selection_key(weaker)


def test_selection_key_uses_overall_w2_after_guardrail():
    metrics = {
        "per_type_w2": [0.70, 0.72, 0.74, 0.76],
        "w2": 0.81,
        "macro_w2": 0.73,
        "macro_mae_km": 180,
        "type_f1": 0.91,
    }

    assert make_selection_key(metrics) == (1.0, 0.81, 0.73, -180.0, 0.91)


def test_selection_key_requires_type_f1_eligibility():
    weak_type = {
        "per_type_w2": [0.9] * 4,
        "w2": 0.95,
        "macro_w2": 0.95,
        "macro_mae_km": 50,
        "type_f1": 0.84,
    }
    eligible = {
        "per_type_w2": [0.7] * 4,
        "w2": 0.75,
        "macro_w2": 0.7,
        "macro_mae_km": 200,
        "type_f1": 0.85,
    }

    assert make_selection_key(eligible) > make_selection_key(weak_type)


def test_early_stopping_ignores_tiny_w2_changes():
    best = (0.60, 0.70, -200.0, 0.90)

    assert not is_meaningful_improvement(
        (0.601, 0.70, -200.0, 0.90), best
    )
    assert is_meaningful_improvement(
        (0.603, 0.70, -200.0, 0.90), best
    )


def test_early_stopping_accepts_mae_gain_when_w2_is_tied():
    best = (0.70, 0.75, -210.0, 0.90)

    assert is_meaningful_improvement(
        (0.70, 0.75, -204.0, 0.90), best
    )
