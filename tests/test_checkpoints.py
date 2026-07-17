import hashlib
from pathlib import Path

import pytest
import torch

from checkpoints import (
    FIVE_CLASS_SCHEMA,
    load_model_checkpoint,
    model_sha256,
    save_model_checkpoint,
)
from models import LegacyMultiTaskResNet, create_five_class_model


TYPE_NAMES = ["IC", "NCG", "NNBE", "PCG", "PNBE"]
DISTANCE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]
DISTANCE_BINS = list(range(0, 3000, 100))


def _save_new(path, *, base_channels=8, **updates):
    checkpoint = {
        "schema": FIVE_CLASS_SCHEMA,
        "model_config": {"base_channels": base_channels},
        "model_state": create_five_class_model(
            base_channels=base_channels
        ).state_dict(),
        "type_names": TYPE_NAMES,
        "distance_names": DISTANCE_NAMES,
        "distance_bins_km": DISTANCE_BINS,
        "preprocess_config": {"local_length": 8000, "global_length": 2000},
        "split_hash": "abc",
        "training_config": {"distance_weight": 0.5},
    }
    checkpoint.update(updates)
    torch.save(checkpoint, path)
    return checkpoint


def _legacy_checkpoint(*, base=8):
    legacy = LegacyMultiTaskResNet(base=base)
    return {
        "model_name": "mtl_resnet",
        "base_channels": base,
        "type_names": TYPE_NAMES,
        "dist_names": DISTANCE_NAMES,
        "dist_bin_starts": DISTANCE_BINS,
        "preprocessing": {"normalize_mode": "minmax", "target_length": 8000},
        "model_state_dict": legacy.state_dict(),
    }


def test_new_checkpoint_round_trip(tmp_path):
    model = create_five_class_model(base_channels=16)
    path = tmp_path / "model.pt"
    save_model_checkpoint(
        path,
        model,
        model_config={"base_channels": 16},
        preprocess_config={"local_length": 8000, "global_length": 2000},
        split_hash="abc",
        training_config={"distance_weight": 0.5},
    )

    loaded = load_model_checkpoint(path, "cpu")

    assert loaded.schema == FIVE_CLASS_SCHEMA
    assert loaded.type_names == tuple(TYPE_NAMES)
    assert loaded.distance_names == tuple(DISTANCE_NAMES)
    assert loaded.distance_bins_km == tuple(DISTANCE_BINS)
    assert loaded.preprocess_config == {
        "local_length": 8000,
        "global_length": 2000,
    }
    assert "model_state" not in loaded.metadata
    assert "optimizer_state" not in loaded.metadata
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)


def test_save_validates_temporary_file_before_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    observed = []
    real_replace = __import__("os").replace

    def record_replace(source, target):
        source = Path(source)
        assert source == Path(f"{path}.tmp")
        assert load_model_checkpoint(source, "cpu").schema == FIVE_CLASS_SCHEMA
        observed.append((source, Path(target)))
        real_replace(source, target)

    monkeypatch.setattr("checkpoints.os.replace", record_replace)
    save_model_checkpoint(
        path,
        create_five_class_model(base_channels=8),
        model_config={"base_channels": 8},
        preprocess_config={"local_length": 8000, "global_length": 2000},
        split_hash="split",
        training_config={},
    )

    assert observed == [(Path(f"{path}.tmp"), path)]
    assert path.is_file()
    assert not Path(f"{path}.tmp").exists()


def test_legacy_five_class_checkpoint_is_supported(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(_legacy_checkpoint(base=16), path)

    loaded = load_model_checkpoint(path, "cpu")

    assert loaded.schema == "legacy_five_class"
    assert isinstance(loaded.model, LegacyMultiTaskResNet)
    assert loaded.type_names == tuple(TYPE_NAMES)
    assert loaded.preprocess_config["normalize_mode"] == "minmax"


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"schema": "four_class_cv_v3"},
        {
            "model_name": "mtl_resnet",
            "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        },
        {"weight": torch.ones(1)},
    ],
)
def test_former_four_class_and_raw_state_checkpoints_are_rejected(
    tmp_path, checkpoint
):
    path = tmp_path / "invalid.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="checkpoint"):
        load_model_checkpoint(path, "cpu")


@pytest.mark.parametrize("legacy", [False, True])
def test_mismatched_class_order_is_rejected(tmp_path, legacy):
    path = tmp_path / "wrong-order.pt"
    if legacy:
        checkpoint = _legacy_checkpoint()
        checkpoint["type_names"] = ["NCG", "IC", "NNBE", "PCG", "PNBE"]
        torch.save(checkpoint, path)
    else:
        _save_new(path, type_names=["NCG", "IC", "NNBE", "PCG", "PNBE"])

    with pytest.raises(ValueError, match="type.*order"):
        load_model_checkpoint(path, "cpu")


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_preprocessing_metadata_is_rejected(tmp_path, legacy):
    path = tmp_path / "missing-preprocessing.pt"
    if legacy:
        checkpoint = _legacy_checkpoint()
        checkpoint.pop("preprocessing")
        torch.save(checkpoint, path)
    else:
        checkpoint = _save_new(path)
        checkpoint.pop("preprocess_config")
        torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="preprocess"):
        load_model_checkpoint(path, "cpu")


@pytest.mark.parametrize("legacy", [False, True])
def test_malformed_tensor_shapes_are_rejected(tmp_path, legacy):
    path = tmp_path / "bad-shape.pt"
    if legacy:
        checkpoint = _legacy_checkpoint()
        state_key = "model_state_dict"
    else:
        checkpoint = _save_new(path)
        state_key = "model_state"
    state = dict(checkpoint[state_key])
    first_key = next(iter(state))
    state[first_key] = torch.zeros(1)
    checkpoint[state_key] = state
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="model.*state"):
        load_model_checkpoint(path, "cpu")


@pytest.mark.parametrize(
    "updates",
    [
        {"model_config": {"base_channels": 0}},
        {"model_config": {"base_channels": 8, "init_checkpoint": "old.pt"}},
        {"distance_names": ["NNBE", "NCG", "PCG", "PNBE"]},
        {"distance_bins_km": list(range(100, 3100, 100))},
        {"preprocess_config": {"local_length": 8000}},
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "normalize_mode": "minmax",
            }
        },
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "use_filter": 1,
            }
        },
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "cutoff_hz": float("inf"),
            }
        },
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "cutoff_hz": "120000",
            }
        },
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "cutoff_hz": 120_000.0,
                "sample_rate_hz": 0.0,
            }
        },
        {
            "preprocess_config": {
                "local_length": 8000,
                "global_length": 2000,
                "cutoff_hz": 2_500_000.0,
                "sample_rate_hz": 5_000_000.0,
            }
        },
    ],
)
def test_malformed_new_checkpoint_configuration_is_rejected(tmp_path, updates):
    path = tmp_path / "malformed.pt"
    _save_new(path, **updates)

    with pytest.raises(ValueError, match="checkpoint|config|order|preprocess"):
        load_model_checkpoint(path, "cpu")


@pytest.mark.parametrize(
    "updates",
    [
        {"optimizer_state": {"state": {0: {"step": torch.tensor(1)}}}},
        {
            "training_config": {
                "distance_weight": 0.5,
                "resume": {"optimizer_state_dict": {"state": {}}},
            }
        },
        {
            "metadata": {
                "run": {"components": [{"optimizer_name": "AdamW"}]}
            }
        },
    ],
)
def test_optimizer_metadata_is_rejected_recursively(tmp_path, updates):
    path = tmp_path / "with-optimizer.pt"
    _save_new(path, **updates)

    with pytest.raises(ValueError, match="optimizer"):
        load_model_checkpoint(path, "cpu")


def test_save_rejects_nested_optimizer_metadata_before_replacement(tmp_path):
    path = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="optimizer"):
        save_model_checkpoint(
            path,
            create_five_class_model(base_channels=8),
            model_config={"base_channels": 8},
            preprocess_config={"local_length": 8000, "global_length": 2000},
            split_hash="split",
            training_config={"nested": {"optimizer": {"state": {}}}},
        )

    assert not path.exists()
    assert not Path(f"{path}.tmp").exists()


def test_model_sha256_streams_file_contents(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc" * 400_000)

    assert model_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
