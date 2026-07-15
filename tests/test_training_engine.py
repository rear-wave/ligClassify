import torch

import training_engine
from models import ConditionalExpertNet


def test_distance_weight_stage_boundary():
    assert training_engine.distance_weight_for_epoch(2, 3, 0.25, 1.0) == (
        "type_focus",
        0.25,
    )
    assert training_engine.distance_weight_for_epoch(3, 3, 0.25, 1.0) == (
        "joint",
        1.0,
    )


def test_joint_step_updates_type_and_oracle_distance_heads():
    torch.manual_seed(3)
    model = ConditionalExpertNet(
        base=4, context_dim=1, dist_mlp_dim=8, dist_dropout=0.0
    )
    batch = {
        "local": torch.randn(4, 1, 512),
        "global_view": torch.randn(4, 1, 512),
        "context": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "type_label": torch.arange(4),
        "distance_low_km": torch.tensor([0.0, 300.0, 600.0, 900.0]),
        "distance_high_km": torch.tensor([100.0, 400.0, 700.0, 1000.0]),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    result = training_engine.conditional_joint_train_step(
        model,
        batch,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        distance_loss_weight=0.25,
    )

    assert result["type_count"] == len(batch["type_label"])
    assert result["distance_count"] == len(batch["type_label"])
    assert result["type_loss"] > 0
    assert result["distance_loss"] > 0
