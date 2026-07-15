import pytest
import torch

from models import ConditionalExpertNet, MultiTaskResNet, create_mtl_model


def test_ordinal_v2_output_shapes():
    model = create_mtl_model(
        base_channels=8,
        architecture="ordinal_v2",
        dist_mlp_dim=16,
        dist_dropout=0.0,
    )

    type_logits, distance_logits = model(torch.randn(3, 1, 512))

    assert type_logits.shape == (3, 5)
    assert len(distance_logits) == 4
    assert all(item.shape == (3, 30) for item in distance_logits)


def test_default_factory_remains_legacy_model():
    model = create_mtl_model(base_channels=8)

    assert isinstance(model, MultiTaskResNet)


def test_ordinal_v2_contains_distance_specific_projection():
    model = create_mtl_model(
        base_channels=8,
        architecture="ordinal_v2",
        dist_mlp_dim=12,
        dist_dropout=0.1,
    )

    assert model.distance_projection[0].normalized_shape == (32,)
    assert model.d_heads[0].in_features == 12


def test_factory_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="architecture"):
        create_mtl_model(base_channels=8, architecture="transformer")


@pytest.mark.parametrize("architecture", ["mtl_resnet", "ordinal_v2"])
def test_four_class_model_exposes_type_features(architecture):
    model = create_mtl_model(
        base_channels=8,
        architecture=architecture,
        num_types=4,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )

    x = torch.randn(2, 1, 128)
    features = model.extract_type_features(x)
    logits = model.forward_type(x)

    assert features.shape == (2, 16)
    assert logits.shape == (2, 4)


@pytest.mark.parametrize("architecture", ["mtl_resnet", "ordinal_v2"])
def test_forward_with_features_reuses_type_features(architecture):
    model = create_mtl_model(
        base_channels=8,
        architecture=architecture,
        num_types=4,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )
    model.eval()
    x = torch.randn(2, 1, 128)

    features, type_logits, distance_logits = model.forward_with_features(x)

    assert torch.allclose(type_logits, model.type_head(features))
    assert type_logits.shape == (2, 4)
    assert len(distance_logits) == 4
    assert all(logits.shape == (2, 30) for logits in distance_logits)


def test_conditional_expert_output_contract():
    model = create_mtl_model(
        base_channels=8,
        architecture="conditional_expert_v1",
        num_types=4,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )
    model.eval()

    features, type_logits, distance_logits, coarse_logits = (
        model.forward_with_features(
            torch.randn(3, 1, 512),
            torch.randn(3, 1, 512),
            torch.randn(3, 3),
        )
    )

    assert features.shape == (3, 32)
    assert type_logits.shape == (3, 4)
    assert len(distance_logits) == len(coarse_logits) == 4
    assert all(logits.shape == (3, 30) for logits in distance_logits)
    assert all(logits.shape == (3, 6) for logits in coarse_logits)


def test_conditional_type_logits_do_not_use_time_context():
    torch.manual_seed(3)
    model = create_mtl_model(
        base_channels=8,
        architecture="conditional_expert_v1",
        num_types=4,
        dist_mlp_dim=12,
        dist_dropout=0.0,
    )
    model.eval()
    local = torch.randn(2, 1, 512)
    global_view = torch.randn(2, 1, 512)

    first = model.forward_with_features(
        local, global_view, torch.zeros(2, 3)
    )
    second = model.forward_with_features(
        local, global_view, torch.ones(2, 3)
    )

    assert torch.allclose(first[1], second[1])
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(first[2], second[2])
    )


def test_conditional_model_accepts_daylight_only_context():
    model = ConditionalExpertNet(base=8, context_dim=1)

    output = model(
        torch.randn(2, 1, 8000),
        torch.randn(2, 1, 8000),
        torch.ones(2, 1),
    )

    assert output[0].shape == (2, 4)


def test_conditional_factory_keeps_legacy_context_default_and_accepts_explicit_dim():
    legacy = create_mtl_model(
        base_channels=8,
        architecture="conditional_expert_v1",
        num_types=4,
    )
    daylight = create_mtl_model(
        base_channels=8,
        architecture="conditional_expert_v1",
        num_types=4,
        context_dim=1,
    )

    assert legacy.context_dim == 3
    assert daylight.context_dim == 1
