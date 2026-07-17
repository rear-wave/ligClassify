from dataclasses import FrozenInstanceError
import inspect

import pytest
import torch

from models import ModelOutput, create_five_class_model


def _inputs(batch_size=3):
    return (
        torch.randn(batch_size, 1, 8000),
        torch.randn(batch_size, 1, 2000),
        torch.tensor([[float(index % 2)] for index in range(batch_size)]),
    )


def test_five_class_model_has_one_type_head_and_four_distance_experts():
    model = create_five_class_model(base_channels=16)
    output = model(*_inputs())

    assert output.type_logits.shape == (3, 5)
    assert len(output.distance_logits) == 4
    assert all(logits.shape == (3, 30) for logits in output.distance_logits)
    assert isinstance(output.distance_logits, tuple)


def test_model_output_is_immutable():
    output = create_five_class_model(base_channels=8)(*_inputs(batch_size=1))

    with pytest.raises(FrozenInstanceError):
        output.features = torch.zeros_like(output.features)
    assert isinstance(output, ModelOutput)


def test_daylight_is_part_of_the_fused_input():
    torch.manual_seed(7)
    model = create_five_class_model(base_channels=8)
    model.eval()
    local, global_view, _ = _inputs(batch_size=2)

    night = model(local, global_view, torch.zeros(2, 1))
    day = model(local, global_view, torch.ones(2, 1))

    assert not torch.allclose(night.features, day.features)


def test_local_and_global_branches_both_receive_gradients():
    model = create_five_class_model(base_channels=8)
    output = model(*_inputs(batch_size=2))

    loss = output.type_logits.sum()
    loss.backward()

    local_gradient = model.local_branch.network[0].weight.grad
    global_gradient = model.global_branch.network[0].weight.grad
    assert local_gradient is not None
    assert global_gradient is not None
    assert local_gradient.abs().sum() > 0
    assert global_gradient.abs().sum() > 0


def test_model_factory_has_no_warm_start_argument():
    signature = inspect.signature(create_five_class_model)

    assert tuple(signature.parameters) == ("base_channels",)
    with pytest.raises(TypeError):
        create_five_class_model(base_channels=8, init_checkpoint="old.pt")
