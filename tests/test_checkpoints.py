import hashlib
import hashlib
import json
from pathlib import Path

import pytest
import torch

from checkpoints import (
    FIVE_CLASS_SCHEMA,
    load_model_bundle,
    load_model_checkpoint,
    model_sha256,
    save_model_bundle,
    save_model_checkpoint,
)
from models import (
    ANCHOR_TYPE_VARIANT,
    LegacyMultiTaskResNet,
    create_anchor_hierarchical_type_model,
    create_five_class_model,
)


TYPE_NAMES = ["IC", "NCG", "NNBE", "PCG", "PNBE"]
DISTANCE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]
DISTANCE_BINS = list(range(0, 3000, 100))


def _write_verifier_fixture(tmp_path):
    from checkpoints import decision_config_sha256
    from models import create_hierarchical_type_model

    paths = [tmp_path / name for name in ("primary.pt", "verifier.pt")]
    for path in paths:
        save_model_checkpoint(path, create_hierarchical_type_model(base_channels=4),
            model_config={"base_channels": 4, "embedding_dim": 128,
                          "prototypes_per_class": 4, "prototype_logit_weight": 0.25},
            preprocess_config={"local_length": 8000, "global_length": 2000},
            split_hash="synthetic", training_config={"role": "type"},
            decision_config=dict(known_probability_thresholds=[.9] * 4, prototype_similarity_thresholds=[-.1] * 4,
                max_js_divergences=[.1] * 4, min_branch_votes=2,
                max_ic_gate_probabilities=[.5] * 4))
    primary = load_model_checkpoint(paths[0])
    config_path = tmp_path / "verifier.json"
    payload = {"schema": "guarded_type_verifier_v1", "primary_sha256": model_sha256(paths[0]),
               "primary_decision_sha256": decision_config_sha256(primary.metadata["decision_config"]),
               "verifier_path": "verifier.pt", "verifier_sha256": model_sha256(paths[1]),
               "limits": {name: [.5, .6, .2, .3] for name in DISTANCE_NAMES}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, paths, primary, payload


def test_type_verifier_loader_binds_weights_config_and_relative_path(tmp_path, monkeypatch):
    from checkpoints import load_type_verifier

    path, models, primary, payload = _write_verifier_fixture(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    verifier = load_type_verifier(path, primary, payload["primary_sha256"])
    assert verifier.primary is primary and len(verifier.signature) == 64
    assert verifier.limits == ((.5, .6, .2, .3),) * 4
    assert not verifier.checkpoint.model.training and not primary.model.training
    payload["verifier_path"] = str(models[1])
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_type_verifier(path, primary, payload["primary_sha256"]).signature == verifier.signature
    payload["limits"]["PNBE"][0] = .7
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_type_verifier(path, primary, payload["primary_sha256"]).signature != verifier.signature
    assert load_type_verifier(None, primary, "unused") is None


@pytest.mark.parametrize("change,message", [
    ({"schema": "unknown"}, "schema"), ({"extra": 1}, "fields"),
    ({"primary_sha256": "b" * 64}, "primary checkpoint"),
    ({"primary_decision_sha256": "b" * 64}, "decision configuration"),
    ({"verifier_sha256": "b" * 64}, "hash mismatch"), ({"verifier_sha256": "bad"}, "invalid"),
    ({"verifier_path": "missing.pt"}, "missing"), ({"verifier_path": ""}, "path"),
    ({"limits": {}}, "four known"),
])
def test_type_verifier_loader_rejects_malformed_or_mismatched_recipes(tmp_path, change, message):
    from checkpoints import load_type_verifier

    path, _, primary, payload = _write_verifier_fixture(tmp_path)
    payload.update(change)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_type_verifier(path, primary, model_sha256(tmp_path / "primary.pt"))


@pytest.mark.parametrize("row", [[float("nan"), .5, .2, .2], [True, .5, .2, .2], [-.1, .5, .2, .2], [1.1, .5, .2, .2], [.5]])
def test_type_verifier_limits_reject_invalid_probabilities(tmp_path, row):
    from checkpoints import load_type_verifier

    path, _, primary, payload = _write_verifier_fixture(tmp_path)
    payload["limits"]["NNBE"] = row
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite probabilities"):
        load_type_verifier(path, primary, payload["primary_sha256"])


@pytest.mark.parametrize("field,value", [("split_hash", "different"), ("preprocess_config", {"local_length": 8000, "global_length": 2000, "cutoff_hz": 100000.0})])
def test_type_verifier_rejects_different_split_or_preprocessing(tmp_path, field, value):
    from checkpoints import load_type_verifier

    path, models, primary, payload = _write_verifier_fixture(tmp_path)
    checkpoint = torch.load(models[1], weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, models[1])
    payload["verifier_sha256"] = model_sha256(models[1])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_type_verifier(path, primary, payload["primary_sha256"])


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


def test_anchor_moe_hierarchical_checkpoint_round_trip(tmp_path):
    model = create_anchor_hierarchical_type_model(
        base_channels=8, embedding_dim=16, patch_length=256,
        patch_stride=128, expert_count=4, expert_topk=2,
        known_fusion_weight=0.5,
    )
    path = tmp_path / "anchor.pt"
    config = {
        "base_channels": 8, "embedding_dim": 16, "prototypes_per_class": 4,
        "prototype_logit_weight": 0.25, "type_backbone": ANCHOR_TYPE_VARIANT,
        "patch_length": 256, "patch_stride": 128, "expert_count": 4,
        "expert_topk": 2, "frequency_bands": 4, "use_reliability": True,
        "known_fusion_weight": 0.5,
    }
    save_model_checkpoint(
        path, model, model_config=config,
        preprocess_config={"local_length": 8000, "global_length": 2000},
        split_hash="anchor-split", training_config={"role": "type"},
    )

    loaded = load_model_checkpoint(path, "cpu")
    assert loaded.model.model_variant == ANCHOR_TYPE_VARIANT
    assert loaded.model.known_fusion_weight == 0.5
    assert loaded.metadata["model_config"] == config
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)


def test_multiscale_checkpoint_round_trip_and_explicit_schema(tmp_path):
    from models import MULTISCALE_TYPE_VARIANT, MultiScaleHierarchicalTypeNet

    model = MultiScaleHierarchicalTypeNet(base_channels=8, embedding_dim=16).eval()
    config = {"base_channels": 8, "embedding_dim": 16, "prototypes_per_class": 4,
              "prototype_logit_weight": 0.25, "type_backbone": MULTISCALE_TYPE_VARIANT}
    path = tmp_path / "multiscale.pt"
    save_model_checkpoint(path, model, model_config=config,
                          preprocess_config={"local_length": 8000, "global_length": 2000},
                          split_hash="synthetic", training_config={"role": "type"})
    loaded = load_model_checkpoint(path, "cpu")
    inputs = (torch.randn(2, 1, 8000), torch.randn(2, 1, 2000), torch.zeros(2, 1))
    assert loaded.model.model_variant == MULTISCALE_TYPE_VARIANT
    loaded.model.eval()
    assert torch.equal(model(*inputs).type_logits, loaded.model(*inputs).type_logits)
    payload = torch.load(path, weights_only=False)
    payload["model_config"]["type_backbone"] = "unknown_temporal_v99"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model_config"):
        load_model_checkpoint(path, "cpu")


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


def _write_model_bundle(root):
    role_paths = {}
    preprocess = {"local_length": 8000, "global_length": 2000}
    for role in ("type", "NCG", "NNBE", "PCG", "PNBE"):
        path = root / role / "model.pt"
        save_model_checkpoint(
            path,
            create_five_class_model(base_channels=8),
            model_config={"base_channels": 8},
            preprocess_config=preprocess,
            split_hash="split",
            training_config={"role": role},
        )
        role_paths[role] = path
    save_model_bundle(root, role_paths, preprocess_config=preprocess)
    return root


def test_model_bundle_round_trip_has_five_independent_roles(tmp_path):
    root = _write_model_bundle(tmp_path / "bundle")

    loaded = load_model_bundle(root, "cpu")

    assert loaded.type_checkpoint.schema == FIVE_CLASS_SCHEMA
    assert len(loaded.distance_checkpoints) == 4
    assert set(loaded.hashes) == {"type", "NCG", "NNBE", "PCG", "PNBE"}
    assert len(
        {
            id(loaded.type_checkpoint.model),
            *(id(item.model) for item in loaded.distance_checkpoints),
        }
    ) == 5


def test_model_bundle_rejects_missing_role(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()

    with pytest.raises(ValueError, match="missing.*PNBE|roles"):
        save_model_bundle(
            root,
            {"type": root / "type" / "model.pt"},
            preprocess_config={
                "local_length": 8000,
                "global_length": 2000,
            },
        )


def test_model_bundle_rejects_hash_mismatch(tmp_path):
    root = _write_model_bundle(tmp_path / "bundle")
    with (root / "NCG" / "model.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        load_model_bundle(root, "cpu")


def test_model_bundle_rejects_mismatched_split(tmp_path):
    root = _write_model_bundle(tmp_path / "bundle")
    save_model_checkpoint(
        root / "NCG" / "model.pt",
        create_five_class_model(base_channels=8),
        model_config={"base_channels": 8},
        preprocess_config={
            "local_length": 8000,
            "global_length": 2000,
        },
        split_hash="different-split",
        training_config={"role": "NCG"},
    )
    role_paths = {
        role: root / role / "model.pt"
        for role in ("type", "NCG", "NNBE", "PCG", "PNBE")
    }

    with pytest.raises(ValueError, match="split"):
        save_model_bundle(
            root,
            role_paths,
            preprocess_config={
                "local_length": 8000,
                "global_length": 2000,
            },
        )
