import inspect

import pytest
import torch

from models import (
    CASCADE_ARCHITECTURE,
    DISTANCE_BIN_COUNT,
    DISTANCE_EXPERT_COUNT,
    HIERARCHICAL_TYPE_ARCHITECTURE,
    create_anchor_hierarchical_type_model,
    create_five_class_model,
    create_hierarchical_type_model,
)


def _inputs(batch_size=3):
    return (
        torch.randn(batch_size, 1, 8000),
        torch.randn(batch_size, 1, 2000),
        torch.tensor([[1.0], [0.0], [1.0]])[:batch_size],
    )


def test_cascade_has_five_class_type_head_and_four_distance_matchers():
    model = create_five_class_model(base_channels=8)
    output = model(*_inputs())

    assert model.architecture == CASCADE_ARCHITECTURE
    assert output.type_logits.shape == (3, 5)
    assert len(output.distance_logits) == DISTANCE_EXPERT_COUNT
    assert all(
        logits.shape == (3, DISTANCE_BIN_COUNT)
        for logits in output.distance_logits
    )
    assert model.distance_matcher.prototypes.shape == (4, 30, 32)


def test_type_and_distance_stages_do_not_share_encoder_parameters():
    model = create_five_class_model(base_channels=8)
    type_parameters = {id(item) for item in model.type_encoder.parameters()}
    distance_parameters = {
        id(item) for item in model.distance_encoder.parameters()
    }

    assert type_parameters
    assert distance_parameters
    assert type_parameters.isdisjoint(distance_parameters)


def test_stage_freezing_prevents_cross_task_feature_damage():
    model = create_five_class_model(base_channels=8)

    model.set_training_stage("type")
    assert all(item.requires_grad for item in model.type_encoder.parameters())
    assert not any(
        item.requires_grad for item in model.distance_encoder.parameters()
    )
    assert not model.distance_matcher.prototypes.requires_grad

    model.set_training_stage("distance")
    assert not any(item.requires_grad for item in model.type_encoder.parameters())
    assert all(
        item.requires_grad for item in model.distance_encoder.parameters()
    )
    assert model.distance_matcher.prototypes.requires_grad

    model.set_training_stage("joint")
    assert all(item.requires_grad for item in model.parameters())

    with pytest.raises(ValueError, match="type, distance, or joint"):
        model.set_training_stage("unknown")


def test_type_stage_receives_local_global_and_daylight_information():
    model = create_five_class_model(base_channels=8)
    local, global_view, daylight = _inputs()
    local.requires_grad_(True)
    global_view.requires_grad_(True)

    logits, features = model.forward_type(local, global_view, daylight)
    (logits.square().mean() + features.square().mean()).backward()

    assert local.grad.abs().sum() > 0
    assert global_view.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        day_logits, _ = model.forward_type(
            local.detach(), global_view.detach(), torch.ones_like(daylight)
        )
        night_logits, _ = model.forward_type(
            local.detach(), global_view.detach(), torch.zeros_like(daylight)
        )
    assert not torch.allclose(day_logits, night_logits)


def test_distance_stage_is_prototype_matching_and_receives_gradients():
    model = create_five_class_model(base_channels=8)
    distance_logits, features = model.forward_distance(*_inputs())

    loss = sum(head.square().mean() for head in distance_logits)
    loss.backward()

    assert features.shape == (3, 32)
    assert model.distance_matcher.prototypes.grad is not None
    assert model.distance_matcher.prototypes.grad.abs().sum() > 0


def test_cascade_skips_distance_encoder_when_every_piece_is_ic():
    model = create_five_class_model(base_channels=8).eval()
    with torch.no_grad():
        model.type_head.weight.zero_()
        model.type_head.bias.fill_(-10.0)
        model.type_head.bias[0] = 10.0

    calls = []
    hook = model.distance_encoder.register_forward_hook(
        lambda *_args: calls.append(1)
    )
    try:
        with torch.no_grad():
            output = model.predict_cascade(*_inputs())
    finally:
        hook.remove()

    assert output.type_logits.argmax(dim=1).tolist() == [0, 0, 0]
    assert calls == []
    assert all(torch.count_nonzero(head) == 0 for head in output.distance_logits)


def test_cascade_runs_only_the_predicted_non_ic_matcher():
    model = create_five_class_model(base_channels=8).eval()
    with torch.no_grad():
        model.type_head.weight.zero_()
        model.type_head.bias.fill_(-10.0)
        model.type_head.bias[2] = 10.0
        output = model.predict_cascade(*_inputs())

    assert output.type_logits.argmax(dim=1).tolist() == [2, 2, 2]
    assert torch.count_nonzero(output.distance_logits[0]) == 0
    assert torch.count_nonzero(output.distance_logits[1]) > 0
    assert torch.count_nonzero(output.distance_logits[2]) == 0
    assert torch.count_nonzero(output.distance_logits[3]) == 0


def test_factory_has_no_checkpoint_or_warm_start_argument():
    parameters = inspect.signature(create_five_class_model).parameters

    assert set(parameters) == {"base_channels"}


def test_type_role_enables_only_type_parameters():
    model = create_five_class_model(base_channels=8)

    assert model.set_training_role("type") is None
    assert all(
        parameter.requires_grad
        for parameter in model.type_encoder.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.type_head.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.distance_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.distance_matcher.parameters()
    )


@pytest.mark.parametrize(
    ("role", "expert_index"),
    [("NCG", 0), ("NNBE", 1), ("PCG", 2), ("PNBE", 3)],
)
def test_distance_role_enables_only_distance_parameters(role, expert_index):
    model = create_five_class_model(base_channels=8)

    assert model.set_training_role(role) == expert_index
    assert not any(
        parameter.requires_grad
        for parameter in model.type_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.type_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.distance_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.distance_matcher.parameters()
    )


def test_single_distance_role_forward_matches_selected_head():
    model = create_five_class_model(base_channels=8)
    inputs = _inputs(batch_size=2)

    all_heads, _ = model.forward_distance(*inputs)
    selected, _ = model.forward_distance_type(*inputs, expert_index=2)

    assert torch.allclose(selected, all_heads[2])


def test_hierarchical_type_model_returns_complete_probability_contract():
    model = create_hierarchical_type_model(
        base_channels=8,
        embedding_dim=32,
        prototypes_per_class=3,
    )

    output = model(*_inputs(batch_size=2))

    assert model.architecture == HIERARCHICAL_TYPE_ARCHITECTURE
    assert output.type_logits.shape == (2, 5)
    assert output.gate_logits.shape == (2, 2)
    assert output.known_logits.shape == (2, 4)
    assert output.prototype_logits.shape == (2, 4, 3)
    assert output.prototype_scores.shape == (2, 4)
    assert output.local_known_logits.shape == (2, 4)
    assert output.global_known_logits.shape == (2, 4)
    assert output.embedding.shape == (2, 32)
    assert torch.allclose(
        output.type_logits.exp().sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

    type_logits, features = model.forward_type(*_inputs(batch_size=2))
    assert type_logits.shape == (2, 5)
    assert features.shape == (2, 32)


def test_multiscale_model_preserves_probabilities_and_both_view_gradients():
    from models import MultiScaleHierarchicalTypeNet

    model = MultiScaleHierarchicalTypeNet(base_channels=8, embedding_dim=16)
    local, global_view, daylight = _inputs(batch_size=2)
    local.requires_grad_(True)
    global_view.requires_grad_(True)
    output = model(local, global_view, daylight)
    assert output.type_logits.shape == (2, 5)
    assert torch.allclose(output.type_logits.exp().sum(1), torch.ones(2), atol=1e-6)
    (output.type_logits.square().mean() + output.prototype_logits.square().mean()).backward()
    assert local.grad.abs().sum() > 0
    assert global_view.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    model.eval()
    together = model(local, global_view, daylight).type_logits
    separate = torch.cat([model(local[i:i+1], global_view[i:i+1], daylight[i:i+1]).type_logits
                          for i in range(2)])
    assert torch.allclose(together, separate, atol=1e-5)


def test_anchor_moe_type_model_returns_losses_and_sparse_routing():
    model = create_anchor_hierarchical_type_model(
        base_channels=8, embedding_dim=32, patch_length=256,
        patch_stride=128, expert_count=4, expert_topk=2,
    )
    captured = {}
    hook = model.encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.update(output)
    )
    try:
        output = model(*_inputs(batch_size=2))
    finally:
        hook.remove()

    assert output.type_logits.shape == (2, 5)
    assert output.anchor_orth_loss.ndim == 0
    assert output.reliability_loss.ndim == 0
    assert torch.isfinite(output.anchor_orth_loss + output.reliability_loss)
    assert torch.all((captured["routing"] > 0).sum(dim=2) == 2)
    assert torch.allclose(captured["importance"].sum(dim=1), torch.ones(2))


def test_anchor_moe_gradients_reach_waveform_and_experts():
    model = create_anchor_hierarchical_type_model(base_channels=8, embedding_dim=16)
    local, global_view, daylight = _inputs(batch_size=2)
    local.requires_grad_(True)
    output = model(local, global_view, daylight)
    (output.type_logits.square().mean() + output.anchor_orth_loss
     + output.reliability_loss).backward()

    assert local.grad is not None and local.grad.abs().sum() > 0
    assert model.encoder.patch_embedding.weight.grad.abs().sum() > 0
    assert all(next(expert.parameters()).grad is not None
               for expert in model.encoder.experts)


def test_anchor_known_fusion_changes_only_known_decision_evidence():
    model = create_anchor_hierarchical_type_model(base_channels=8)
    inputs = _inputs(batch_size=2)
    baseline = model(*inputs)
    model.known_fusion_weight = 0.5
    hybrid = model(*inputs)

    assert not torch.allclose(hybrid.known_logits, baseline.known_logits)
    assert torch.allclose(hybrid.gate_logits, baseline.gate_logits)


def test_hierarchical_type_gradients_reach_both_scales_and_prototypes():
    model = create_hierarchical_type_model(
        base_channels=8,
        embedding_dim=32,
        prototypes_per_class=4,
    )
    local, global_view, daylight = _inputs(batch_size=2)
    local.requires_grad_(True)
    global_view.requires_grad_(True)

    output = model(local, global_view, daylight)
    loss = (
        output.type_logits.square().mean()
        + output.local_known_logits.square().mean()
        + output.global_known_logits.square().mean()
    )
    loss.backward()

    assert local.grad is not None and local.grad.abs().sum() > 0
    assert global_view.grad is not None and global_view.grad.abs().sum() > 0
    assert model.prototype_matcher.prototypes.grad is not None
    assert model.prototype_matcher.prototypes.grad.abs().sum() > 0
    normalized = model.prototype_matcher.normalized_prototypes()
    assert torch.allclose(
        normalized.norm(dim=-1),
        torch.ones(4, 4),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_channels": 0},
        {"embedding_dim": 0},
        {"prototypes_per_class": 0},
        {"prototype_logit_weight": -0.1},
    ],
)
def test_hierarchical_type_model_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        create_hierarchical_type_model(**kwargs)


def test_hierarchical_type_model_rejects_malformed_inputs():
    model = create_hierarchical_type_model(base_channels=8)
    local, global_view, daylight = _inputs(batch_size=2)

    with pytest.raises(ValueError, match="local"):
        model(local[:, :, :-1], global_view, daylight)
    with pytest.raises(ValueError, match="global_view"):
        model(local, global_view[:, :, :-1], daylight)
    with pytest.raises(ValueError, match="daylight"):
        model(local, global_view, daylight[:, 0])
