import pytest
import torch

from models import MultiTaskResNet, create_mtl_model


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
