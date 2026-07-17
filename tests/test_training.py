from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from evaluation import evaluate_loader, selection_score
from models import ModelOutput
from training import EarlyStoppingState, compute_joint_loss, train_epoch


def _random_output(batch_size: int = 3) -> ModelOutput:
    return ModelOutput(
        type_logits=torch.randn(batch_size, 5, requires_grad=True),
        distance_logits=tuple(
            torch.randn(batch_size, 30, requires_grad=True) for _ in range(4)
        ),
        features=torch.randn(batch_size, 8),
    )


def test_ic_has_no_distance_loss_and_non_ic_routes_by_true_type():
    output = _random_output()
    batch = {
        "type_label": torch.tensor([0, 1, 4]),
        "distance_bin": torch.tensor([-1, 3, 9]),
    }

    losses = compute_joint_loss(output, batch, distance_weight=0.5)
    losses.total.backward()

    assert losses.distance_count == 2
    assert output.distance_logits[0].grad[1].abs().sum() > 0
    assert output.distance_logits[3].grad[2].abs().sum() > 0
    for head in output.distance_logits:
        if head.grad is not None:
            assert head.grad[0].abs().sum() == 0


@pytest.mark.parametrize(
    ("type_labels", "distance_bins", "message"),
    [
        ([0, 1], [0, 3], "IC"),
        ([0, 1], [-1, -1], "non-IC"),
        ([0, 4], [-1, 30], "non-IC"),
    ],
)
def test_joint_loss_rejects_invalid_distance_labels(
    type_labels, distance_bins, message
):
    output = _random_output(batch_size=2)
    batch = {
        "type_label": torch.tensor(type_labels),
        "distance_bin": torch.tensor(distance_bins),
    }

    with pytest.raises(ValueError, match=message):
        compute_joint_loss(output, batch)


def test_distance_weight_scales_only_the_distance_component():
    output = _random_output(batch_size=2)
    batch = {
        "type_label": torch.tensor([0, 2]),
        "distance_bin": torch.tensor([-1, 5]),
    }

    unweighted = compute_joint_loss(output, batch, distance_weight=1.0)
    half_weighted = compute_joint_loss(output, batch, distance_weight=0.5)

    assert half_weighted.type_loss == pytest.approx(unweighted.type_loss)
    assert half_weighted.distance_loss == pytest.approx(unweighted.distance_loss)
    assert half_weighted.total.item() == pytest.approx(
        half_weighted.type_loss + 0.5 * half_weighted.distance_loss
    )


class _StaticEvaluationModel(nn.Module):
    def __init__(self, output: ModelOutput):
        super().__init__()
        self.output = output

    def forward(self, local, global_view, daylight):
        return self.output


def _logits_with_winners(winners: list[int], columns: int) -> torch.Tensor:
    logits = torch.full((len(winners), columns), -10.0)
    logits[torch.arange(len(winners)), torch.tensor(winners)] = 10.0
    return logits


def _evaluation_fixture():
    true_types = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    true_bins = torch.tensor([3, 3, 8, 8, 12, 12, 20, 20])
    predicted_types = [0, 2, 0, 2, 0, 3, 0, 4]
    heads = []
    for expert in range(4):
        winners = [0] * len(true_types)
        for row, true_type in enumerate(true_types.tolist()):
            if expert == true_type - 1:
                winners[row] = int(true_bins[row])
        heads.append(_logits_with_winners(winners, 30))
    # Row 1 is deliberately routed to the wrong NNBE expert at deployment.
    heads[1][1] = _logits_with_winners([3], 30)[0]
    output = ModelOutput(
        type_logits=_logits_with_winners(predicted_types, 5),
        distance_logits=tuple(heads),
        features=torch.zeros(len(true_types), 1),
    )
    batch = {
        "local": torch.zeros(len(true_types), 1, 4),
        "global": torch.zeros(len(true_types), 1, 2),
        "daylight": torch.zeros(len(true_types), 1),
        "type_label": true_types,
        "distance_bin": true_bins,
    }
    return _StaticEvaluationModel(output), [batch]


def test_evaluation_uses_predicted_type_for_distance_routing():
    model, loader = _evaluation_fixture()

    metrics = evaluate_loader(model, loader, device="cpu")

    assert metrics["type_macro_f1"] < 1.0
    assert metrics["distance_coverage"] == 0.5
    assert metrics["distance_exact_accuracy"] == 0.5
    assert metrics["distance_within_100"] == 0.5
    assert metrics["distance_within_200"] == 0.5
    assert metrics["distance_mae_km"] == 0.0
    assert metrics["per_type_within_200"] == [0.5, 0.5, 0.5, 0.5]
    assert metrics["mean_non_ic_within_200"] == 0.5
    assert selection_score(metrics) == pytest.approx(
        0.5 * metrics["type_macro_f1"]
        + 0.5 * metrics["mean_non_ic_within_200"]
    )


def test_wrong_non_ic_prediction_uses_the_wrong_expert_distance():
    true_types = torch.tensor([1, 2, 3, 4])
    true_bins = torch.tensor([3, 8, 12, 20])
    heads = [
        _logits_with_winners(true_bins.tolist(), 30) for _ in range(4)
    ]
    heads[1][0] = _logits_with_winners([10], 30)[0]
    output = ModelOutput(
        type_logits=_logits_with_winners([2, 2, 3, 4], 5),
        distance_logits=tuple(heads),
        features=torch.zeros(4, 1),
    )
    batch = {
        "local": torch.zeros(4, 1, 4),
        "global": torch.zeros(4, 1, 2),
        "daylight": torch.zeros(4, 1),
        "type_label": true_types,
        "distance_bin": true_bins,
    }

    metrics = evaluate_loader(
        _StaticEvaluationModel(output), [batch], device="cpu"
    )

    assert metrics["distance_coverage"] == 1.0
    assert metrics["per_type_within_200"] == [0.0, 1.0, 1.0, 1.0]
    assert metrics["distance_mae_km"] == 175.0


def test_evaluation_requires_all_four_non_ic_types():
    model, loader = _evaluation_fixture()
    loader[0] = {
        key: value[:-2] if isinstance(value, torch.Tensor) else value
        for key, value in loader[0].items()
    }
    model.output = ModelOutput(
        type_logits=model.output.type_logits[:-2],
        distance_logits=tuple(head[:-2] for head in model.output.distance_logits),
        features=model.output.features[:-2],
    )

    with pytest.raises(ValueError, match="all four non-IC types"):
        evaluate_loader(model, loader, device="cpu")


class _TinyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.forward_count = 0

    def forward(self, local, global_view, daylight):
        self.forward_count += 1
        batch_size = len(local)
        feature = self.weight.expand(batch_size, 1)
        type_logits = self.weight * torch.ones(batch_size, 5)
        distance_logits = tuple(
            self.weight * torch.ones(batch_size, 30) for _ in range(4)
        )
        return ModelOutput(type_logits, distance_logits, feature)


def _training_batch(type_labels, distance_bins):
    batch_size = len(type_labels)
    return {
        "local": torch.zeros(batch_size, 1, 4),
        "global": torch.zeros(batch_size, 1, 2),
        "daylight": torch.zeros(batch_size, 1),
        "type_label": torch.tensor(type_labels),
        "distance_bin": torch.tensor(distance_bins),
    }


def test_train_epoch_uses_one_forward_per_batch_and_aggregates_counts():
    model = _TinyTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loader = [
        _training_batch([0, 1], [-1, 2]),
        _training_batch([2], [4]),
    ]

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        device="cpu",
        scaler=scaler,
        amp=False,
        distance_weight=0.5,
    )

    assert model.forward_count == 2
    assert metrics["sample_count"] == 3
    assert metrics["distance_count"] == 2
    assert set(metrics) == {
        "loss",
        "type_loss",
        "distance_loss",
        "sample_count",
        "distance_count",
    }


def test_early_stopping_requires_more_than_one_millionth_improvement():
    model = nn.Linear(2, 1)
    state = EarlyStoppingState()

    assert state.update(0.5, epoch=2, model=model)
    saved = copy.deepcopy(state.best_state)
    with torch.no_grad():
        model.weight.add_(10)
    assert all(tensor.device.type == "cpu" for tensor in state.best_state.values())
    assert all(
        torch.equal(state.best_state[key], saved[key]) for key in state.best_state
    )

    assert not state.update(0.5000005, epoch=3, model=model)
    assert state.wait == 1
    assert state.best_epoch == 2
    assert state.update(0.500002, epoch=4, model=model)
    assert state.wait == 0
    assert state.best_epoch == 4
