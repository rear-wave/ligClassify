from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import train
import training
from checkpoints import load_model_checkpoint
from evaluation import (
    HierarchicalDecisionConfig,
    calibrate_hierarchical_decision,
    decide_hierarchical_types,
    evaluate_distance_role,
    evaluate_loader,
    evaluate_type_role,
    selection_score,
)
from models import (
    HIERARCHICAL_TYPE_ARCHITECTURE,
    HierarchicalTypeOutput,
    ModelOutput,
    create_hierarchical_type_model,
)
from training import (
    DistanceLossWeights,
    EarlyStoppingState,
    HierarchicalLossWeights,
    distance_expert_loss,
    hierarchical_type_loss,
    load_last_state,
    paired_distance_expert_loss,
    save_last_state,
    train_role_epoch,
)
from .test_lig import make_piece, write_source


def test_runtime_modules_stay_within_approved_line_limit():
    runtime_paths = (
        "train.py",
        "classify.py",
        "audit_data.py",
        "models.py",
        "training.py",
        "evaluation.py",
        "checkpoints.py",
        "data/__init__.py",
        "data/lig.py",
        "data/manifest.py",
        "data/preprocess.py",
        "data/dataset.py",
        "data/sampling.py",
        "data/split.py",
    )

    line_counts = {
        path: sum(
            bool(line.strip())
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        )
        for path in runtime_paths
    }

    assert max(line_counts.values()) <= 800, line_counts


def test_training_defaults_match_five_class_contract():
    args = train.build_parser().parse_args([])

    assert args.task_data == r"..\train_data"
    assert args.output == r".\weights\multi_model"
    assert args.epochs == 50
    assert args.batch_size == 60
    assert args.patience == 10
    assert args.type_samples_per_epoch == 120000
    assert args.distance_samples_per_epoch == 60000
    assert args.num_workers == 0
    assert args.seed == 42
    assert args.stage == "all"
    assert args.base_channels == 64
    assert args.lr == 0.0003
    assert args.weight_decay == 0.0005
    assert args.type_loss_profile == "baseline"
    assert args.type_architecture == "hierarchical"
    assert args.anchor_known_fusion_weight == 0.0
    assert args.resume is None
    assert args.no_amp is False
    assert not hasattr(args, "init_model")


def test_multiscale_drift_training_records_architecture_and_augmentation():
    args = train.build_parser().parse_args([
        "--stage", "type", "--type_architecture", "multiscale",
        "--type_loss_profile", "known_consistency_v1",
        "--augmentation_profile", "sensor_drift_v1", "--defer_test",
    ])
    train._validate_args(args)
    config = train._training_config(args, "type")
    assert config["type_architecture"] == "multiscale"
    assert config["augmentation_config"]["bandwidth_probability"] == 0.35
    assert args.defer_test


def test_inference_view_shift_records_policy_and_preserves_default_hash():
    default = train.build_parser().parse_args(["--stage", "type"])
    aligned = train.build_parser().parse_args(["--stage", "type", "--inference_view_shift"])
    train._validate_args(aligned)
    assert "type_training_views" not in train._training_config(default, "type")
    assert train._training_config(aligned, "type")["type_training_views"] == (
        "raw_augmentation_plus_inference_shift_v1")
    assert train.training_config_hash(default) != train.training_config_hash(aligned)
    invalid = train.build_parser().parse_args(["--stage", "NNBE", "--inference_view_shift"])
    with pytest.raises(ValueError, match="only to type"):
        train._validate_args(invalid)


def test_distance_training_config_records_behavioral_policies():
    args = train.build_parser().parse_args(["--stage", "PCG"])

    config = train._training_config(args, "PCG")

    assert config["schema"] == "five_class_role_training_v3"
    assert config["distance_training_views"] == (
        "paired_augmentation_probability_average_v1"
    )
    assert config["selection_policy"] == "argmax_exact_within_mae_v1"
    assert config["distance_loss"]["consistency"] == 0.10


def test_type_loss_profile_records_all_experimental_weights():
    args = train.build_parser().parse_args(
        ["--stage", "type", "--type_loss_profile", "known_consistency_v1"]
    )

    config = train._training_config(args, "type")

    assert config["type_loss_profile"] == "known_consistency_v1"
    assert config["hierarchical_loss"] == {
        "gate": 1.0,
        "known": 1.25,
        "five_class": 0.20,
        "prototype": 0.25,
        "branch": 0.15,
        "contrastive": 0.15,
        "consistency": 0.25,
        "gate_consistency": 0.15,
        "prototype_consistency": 0.15,
        "ic_margin": 0.25,
        "prototype_diversity": 0.02,
        "label_smoothing": 0.05,
        "temperature": 0.10,
        "ic_similarity_margin": 0.25,
        "prototype_diversity_margin": 0.20,
    }


def test_type_loss_profile_is_rejected_for_distance_only_training():
    args = train.build_parser().parse_args(
        ["--stage", "PCG", "--type_loss_profile", "known_consistency_v1"]
    )

    with pytest.raises(ValueError, match="applies only to type training"):
        train._validate_args(args)


def test_anchor_moe_profile_records_explicit_architecture_and_losses():
    args = train.build_parser().parse_args(
        [
            "--stage", "type",
            "--type_architecture", "anchor_moe",
            "--type_loss_profile", "anchor_moe_v1",
        ]
    )

    train._validate_args(args)
    config = train._training_config(args, "type")

    assert config["type_architecture"] == "anchor_moe"
    assert config["hierarchical_loss"]["anchor_orth"] == 0.05
    assert config["hierarchical_loss"]["reliability"] == 0.01


@pytest.mark.parametrize(
    "options",
    [
        ["--type_architecture", "anchor_moe"],
        ["--type_loss_profile", "anchor_moe_v1"],
    ],
)
def test_anchor_moe_architecture_and_loss_profile_must_be_paired(options):
    args = train.build_parser().parse_args(["--stage", "type", *options])

    with pytest.raises(ValueError, match="requires an anchor loss profile"):
        train._validate_args(args)


def test_anchor_consistency_profile_strengthens_view_objectives():
    weights = train._type_loss_weights("anchor_consistency_v2")

    assert weights.known == 1.25
    assert weights.consistency == 0.25
    assert weights.gate_consistency == 0.15
    assert weights.prototype_consistency == 0.15
    assert weights.anchor_orth == 0.05
    assert weights.reliability == 0.01


def test_anchor_factorized_profile_adds_known_family_objectives():
    weights = train._type_loss_weights("anchor_factorized_v3")

    assert weights.family == 0.10
    assert weights.polarity == 0.10
    assert weights.consistency == 0.25


def test_anchor_fusion_weight_requires_anchor_architecture():
    args = train.build_parser().parse_args(
        ["--stage", "type", "--anchor_known_fusion_weight", "0.5"]
    )

    with pytest.raises(ValueError, match="requires anchor_moe"):
        train._validate_args(args)


def test_moderate_type_loss_profile_changes_only_consistency_weights():
    args = train.build_parser().parse_args(
        ["--stage", "type", "--type_loss_profile", "consistency_moderate_v1"]
    )

    baseline = train._training_config(
        train.build_parser().parse_args(["--stage", "type"]), "type"
    )["hierarchical_loss"]
    candidate = train._training_config(args, "type")["hierarchical_loss"]

    changed = {
        name for name in baseline if baseline[name] != candidate[name]
    }
    assert changed == {
        "consistency", "gate_consistency", "prototype_consistency"
    }
    assert candidate["consistency"] == 0.20
    assert candidate["gate_consistency"] == 0.15
    assert candidate["prototype_consistency"] == 0.15


@pytest.mark.parametrize(
    "option",
    [
        "--resume_cv",
        "--init_model",
        "--folds",
        "--stop_after_oof",
        "--rejection_target_precision",
        "--verify_only",
        "--distance_weight",
        "--samples_per_epoch",
        "--ic_fraction",
    ],
)
def test_old_training_options_are_rejected(option):
    with pytest.raises(SystemExit):
        train.build_parser().parse_args([option])


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


def test_classification_only_evaluation_ignores_distance_labels():
    model, loader = _evaluation_fixture()
    loader[0]["distance_bin"][:] = -1

    metrics = evaluate_loader(
        model, loader, device="cpu", include_distance=False
    )

    assert metrics["piece_count"] == 8
    assert "type_macro_f1" in metrics
    assert not any(key.startswith("distance_") for key in metrics)


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


class _TinyRoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_weight = nn.Parameter(torch.tensor(0.1))
        self.distance_weight = nn.Parameter(torch.tensor(0.2))
        self.type_calls = 0
        self.distance_calls = 0

    def forward_type(self, local, global_view, daylight):
        self.type_calls += 1
        batch_size = len(local)
        logits = self.type_weight * torch.arange(5).expand(batch_size, 5)
        return logits, self.type_weight.expand(batch_size, 1)

    def forward_distance(self, local, global_view, daylight):
        self.distance_calls += 1
        batch_size = len(local)
        bins = torch.arange(30).expand(batch_size, 30)
        heads = tuple(self.distance_weight * bins for _ in range(4))
        return heads, self.distance_weight.expand(batch_size, 1)

    def forward_distance_type(
        self, local, global_view, daylight, *, expert_index
    ):
        heads, features = self.forward_distance(local, global_view, daylight)
        return heads[expert_index], features


def _training_batch(type_labels, distance_bins):
    batch_size = len(type_labels)
    return {
        "local": torch.zeros(batch_size, 1, 4),
        "global": torch.zeros(batch_size, 1, 2),
        "daylight": torch.zeros(batch_size, 1),
        "type_label": torch.tensor(type_labels),
        "distance_bin": torch.tensor(distance_bins),
    }


def _paired_training_batch(type_labels, distance_bins):
    batch = _training_batch(type_labels, distance_bins)
    batch["local_alt"] = batch["local"].clone()
    batch["global_alt"] = batch["global"].clone()
    return batch


def test_type_role_epoch_never_runs_distance_network():
    model = _TinyRoleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_role_epoch(
        model,
        [_training_batch([0, 1], [-1, 3])],
        optimizer,
        "cpu",
        role="type",
        amp=False,
    )

    assert model.type_calls == 1
    assert model.distance_calls == 0
    assert metrics["sample_count"] == 2
    assert model.type_weight.grad is not None
    assert model.distance_weight.grad is None


def test_distance_role_epoch_never_runs_type_network():
    model = _TinyRoleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_role_epoch(
        model,
        [_paired_training_batch([1, 1], [3, 4])],
        optimizer,
        "cpu",
        role="NCG",
        amp=False,
    )

    assert model.type_calls == 0
    assert model.distance_calls == 2
    assert metrics["sample_count"] == 2
    assert model.type_weight.grad is None
    assert model.distance_weight.grad is not None


def test_distance_expert_loss_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="shape"):
        distance_expert_loss(torch.zeros(2, 29), torch.tensor([1, 2]))


def test_paired_distance_loss_penalizes_view_disagreement():
    targets = torch.tensor([2, 4])
    primary = _logits_with_winners([2, 4], 30).requires_grad_()
    matching = primary.detach().clone().requires_grad_()
    disagreeing = _logits_with_winners([8, 10], 30).requires_grad_()
    weights = DistanceLossWeights(
        categorical=0.0, ordered=0.0, expected=0.0, consistency=1.0
    )

    matching_loss, matching_logits = paired_distance_expert_loss(
        primary, matching, targets, weights
    )
    disagreeing_loss, _ = paired_distance_expert_loss(
        primary, disagreeing, targets, weights
    )
    reverse_loss, _ = paired_distance_expert_loss(
        disagreeing, primary, targets, weights
    )

    assert matching_loss.detach().item() == pytest.approx(0.0, abs=1e-7)
    assert disagreeing_loss > 0
    assert disagreeing_loss.detach().item() == pytest.approx(
        reverse_loss.detach().item()
    )
    assert torch.equal(matching_logits.argmax(dim=1), targets)
    disagreeing_loss.backward()
    assert torch.isfinite(primary.grad).all()
    assert torch.isfinite(disagreeing.grad).all()
    assert torch.count_nonzero(primary.grad)
    assert torch.count_nonzero(disagreeing.grad)


def test_distance_role_requires_paired_training_views():
    model = _TinyRoleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="paired views"):
        train_role_epoch(
            model,
            [_training_batch([1, 1], [3, 4])],
            optimizer,
            "cpu",
            role="NCG",
            amp=False,
        )


def test_distance_selection_score_uses_deployed_argmax_metrics():
    better_argmax = {
        "exact_accuracy": 0.70,
        "within_100": 0.80,
        "within_200": 0.90,
        "mae_bins": 0.7,
        "expected_within_200": 0.50,
        "expected_mae_bins": 2.0,
    }
    better_expected = {
        "exact_accuracy": 0.60,
        "within_100": 0.99,
        "within_200": 0.99,
        "mae_bins": 0.3,
        "expected_within_200": 0.99,
        "expected_mae_bins": 0.1,
    }

    assert train._selection_score("PCG", better_argmax) > train._selection_score(
        "PCG", better_expected
    )


def _hierarchical_output(
    known_logits: torch.Tensor,
    embedding: torch.Tensor,
) -> HierarchicalTypeOutput:
    batch_size = len(known_logits)
    gate_logits = torch.zeros(batch_size, 2)
    gate_log = gate_logits.log_softmax(dim=1)
    known_log = known_logits.log_softmax(dim=1)
    type_logits = torch.cat(
        (gate_log[:, :1], gate_log[:, 1:] + known_log),
        dim=1,
    )
    prototype_logits = torch.zeros(batch_size, 4, 2)
    return HierarchicalTypeOutput(
        type_logits=type_logits,
        gate_logits=gate_logits,
        known_logits=known_logits,
        prototype_logits=prototype_logits,
        prototype_scores=torch.zeros(batch_size, 4),
        local_known_logits=known_logits,
        global_known_logits=known_logits,
        embedding=embedding,
    )


def test_hierarchical_decision_masks_exactly_define_acceptance():
    primary = _hierarchical_output(
        torch.tensor([[8.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]),
        torch.eye(2, 4),
    )
    alternate = replace(
        primary,
        known_logits=torch.tensor(
            [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]
        ),
    )
    config = HierarchicalDecisionConfig(
        known_probability_thresholds=(0.25,) * 4,
        prototype_similarity_thresholds=(-1.0,) * 4,
        max_js_divergences=(1.0,) * 4,
        min_branch_votes=0,
        max_ic_gate_probabilities=(1.0,) * 4,
    )

    decision = decide_hierarchical_types(primary, alternate, config)
    expected_stable = torch.stack(
        list(decision.constraint_passes.values())
    ).all(dim=0)

    assert decision.constraint_passes["view_agreement"].tolist() == [True, False]
    assert torch.equal(decision.final_type.ne(0), expected_stable)
    assert decision.final_type.tolist() == [1, 0]


def test_hierarchical_decision_uses_candidate_specific_js_limits():
    primary = _hierarchical_output(
        torch.tensor([[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]),
        torch.eye(2, 4),
    )
    alternate = replace(
        primary,
        known_logits=torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]
        ),
        local_known_logits=torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]
        ),
        global_known_logits=torch.tensor(
            [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]
        ),
    )
    config = HierarchicalDecisionConfig(
        known_probability_thresholds=(0.25,) * 4,
        prototype_similarity_thresholds=(-1.0,) * 4,
        max_js_divergences=(0.0, 1.0, 1.0, 1.0),
        min_branch_votes=1,
        max_ic_gate_probabilities=(1.0,) * 4,
    )

    decision = decide_hierarchical_types(primary, alternate, config)

    assert decision.constraint_passes["js_divergence"].tolist() == [False, True]
    assert decision.final_type.tolist() == [0, 2]


def test_calibration_returns_class_specific_js_limits_without_fixed_ic_prior():
    targets = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
    winners = torch.tensor([1, 0, 0, 1, 1, 2, 2, 3, 3])
    primary_logits = torch.full((len(targets), 4), -3.0)
    alternate_logits = torch.full((len(targets), 4), -3.0)
    primary_logits[torch.arange(len(targets)), winners] = 8.0
    alternate_logits[torch.arange(len(targets)), winners] = torch.tensor(
        [7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5]
    )
    primary = _hierarchical_output(primary_logits, torch.eye(len(targets)))
    alternate = _hierarchical_output(alternate_logits, torch.eye(len(targets)))

    config = calibrate_hierarchical_decision(
        primary,
        alternate,
        targets,
        minimum_precision=0.5,
        minimum_nbe_precision=0.5,
        maximum_ic_false_positive_rate=1.0 - 1e-6,
        maximum_cg_to_nbe_rate=1.0 - 1e-6,
    )

    assert len(config.max_js_divergences) == 4
    assert len(set(config.max_js_divergences)) > 1


def test_calibration_tie_break_keeps_upper_bounds_least_restrictive():
    targets = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4])
    winners = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3])
    known_logits = torch.zeros(len(targets), 4)
    known_logits[torch.arange(len(targets)), winners] = torch.tensor(
        [2.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    )
    primary = _hierarchical_output(known_logits, torch.eye(len(targets)))
    gate_probability = torch.tensor(
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    gate_logits = torch.stack(
        (gate_probability.log(), (1.0 - gate_probability).log()), dim=1
    )
    primary = replace(primary, gate_logits=gate_logits)

    config = calibrate_hierarchical_decision(
        primary,
        primary,
        targets,
        minimum_precision=1.0,
        minimum_nbe_precision=1.0,
        maximum_ic_false_positive_rate=0.0,
        maximum_cg_to_nbe_rate=1.0 - 1e-6,
    )

    # Several gate limits produce the same perfect calibration decisions.  The
    # tie-break must not collapse to the strictest observed value (0.1), which
    # is brittle under deployment shift.
    assert config.max_ic_gate_probabilities[0] > 0.7
    assert decide_hierarchical_types(primary, primary, config).final_type.tolist() == [
        0, 1, 1, 2, 2, 3, 3, 4, 4,
    ]


class _CaptureHierarchicalInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, local, global_view, daylight):
        self.inputs.append((local.clone(), global_view.clone()))
        return _hierarchical_output(
            torch.zeros(len(local), 4), torch.zeros(len(local), 8)
        )


def test_hierarchical_inference_matches_calibration_view_shift():
    from evaluation import (
        conservative_hierarchical_config,
        decision_config_dict,
        infer_hierarchical_types,
    )

    model = _CaptureHierarchicalInferenceModel()
    local = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
    global_view = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    infer_hierarchical_types(
        model,
        decision_config_dict(conservative_hierarchical_config()),
        local,
        global_view,
        torch.zeros(1, 1),
        None,
        torch.zeros(1, dtype=torch.bool),
    )

    assert len(model.inputs) == 2
    assert torch.equal(model.inputs[1][0][..., 16:], local[..., :-16])
    assert torch.equal(
        model.inputs[1][1][..., 4:], global_view[..., :-4]
    )
    assert torch.count_nonzero(model.inputs[1][1][..., :4]) == 0


def test_hierarchical_loss_excludes_ic_from_known_objectives():
    targets = torch.tensor([0, 1, 1, 2, 2])
    known_logits = torch.randn(5, 4)
    primary = _hierarchical_output(known_logits, torch.randn(5, 8))
    alternate = _hierarchical_output(
        known_logits + 0.1 * torch.randn(5, 4),
        torch.randn(5, 8),
    )

    baseline = hierarchical_type_loss(primary, alternate, targets)
    changed_primary = primary.embedding.clone()
    changed_alternate = alternate.embedding.clone()
    changed_primary[0] = 1000.0
    changed_alternate[0] = -1000.0
    changed = hierarchical_type_loss(
        replace(primary, embedding=changed_primary),
        replace(alternate, embedding=changed_alternate),
        targets,
    )

    assert torch.allclose(changed["known"], baseline["known"])
    assert torch.allclose(changed["prototype"], baseline["prototype"])
    assert torch.allclose(changed["branch"], baseline["branch"])
    assert torch.allclose(changed["contrastive"], baseline["contrastive"])
    assert torch.allclose(changed["consistency"], baseline["consistency"])


def test_hierarchical_consistency_increases_when_known_views_disagree():
    targets = torch.tensor([1, 2, 3, 4])
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ]
    )
    primary = _hierarchical_output(logits, torch.eye(4))
    matching = _hierarchical_output(logits.clone(), torch.eye(4))
    disagreeing = _hierarchical_output(
        logits.roll(1, dims=1), torch.eye(4)
    )

    stable = hierarchical_type_loss(primary, matching, targets)
    unstable = hierarchical_type_loss(primary, disagreeing, targets)

    assert stable["consistency"] < unstable["consistency"]


def test_hierarchical_gate_consistency_uses_all_classes():
    targets = torch.tensor([0, 1])
    output = _hierarchical_output(torch.zeros(2, 4), torch.eye(2, 4))
    changed_gate = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])

    stable = hierarchical_type_loss(output, output, targets)
    unstable = hierarchical_type_loss(
        output, replace(output, gate_logits=changed_gate), targets
    )

    assert stable["gate_consistency"] < unstable["gate_consistency"]


def test_hierarchical_ic_margin_rejects_known_prototype_similarity():
    targets = torch.tensor([0, 1])
    output = _hierarchical_output(torch.zeros(2, 4), torch.eye(2, 4))
    high_similarity = output.prototype_scores.clone()
    high_similarity[0, 2] = 0.9

    baseline = hierarchical_type_loss(output, output, targets)
    rejected = hierarchical_type_loss(
        replace(output, prototype_scores=high_similarity), output, targets
    )

    assert baseline["ic_margin"] == 0.0
    assert rejected["ic_margin"] > baseline["ic_margin"]


def test_hierarchical_prototype_diversity_penalizes_collapse():
    targets = torch.tensor([1, 2, 3, 4])
    output = _hierarchical_output(torch.zeros(4, 4), torch.eye(4))
    collapsed = torch.ones(4, 2, 3)
    separated = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 4
    )

    collapsed_loss = hierarchical_type_loss(
        output, output, targets, prototype_vectors=collapsed
    )
    separated_loss = hierarchical_type_loss(
        output, output, targets, prototype_vectors=separated
    )

    assert collapsed_loss["prototype_diversity"] > separated_loss[
        "prototype_diversity"
    ]


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([0, 0]),
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([1, 1, 1]),
    ],
)
def test_hierarchical_loss_is_finite_for_sparse_batch_composition(targets):
    batch_size = len(targets)
    primary = _hierarchical_output(
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 8),
    )
    alternate = _hierarchical_output(
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 8),
    )

    losses = hierarchical_type_loss(primary, alternate, targets)

    assert set(losses) == {
        "total",
        "gate",
        "known",
        "five_class",
        "prototype",
        "branch",
        "contrastive",
        "consistency",
        "gate_consistency",
        "prototype_consistency",
        "ic_margin",
        "prototype_diversity",
    }
    assert all(torch.isfinite(value) for value in losses.values())


def test_anchor_auxiliary_losses_are_weighted_only_when_present():
    targets = torch.tensor([0, 1, 2, 3, 4])
    base = _hierarchical_output(torch.randn(5, 4), torch.randn(5, 8))
    anchor = replace(
        base,
        anchor_orth_loss=torch.tensor(2.0),
        reliability_loss=torch.tensor(3.0),
    )
    weights = replace(
        HierarchicalLossWeights(), anchor_orth=0.05, reliability=0.01
    )

    baseline = hierarchical_type_loss(base, base, targets, weights=weights)
    candidate = hierarchical_type_loss(anchor, anchor, targets, weights=weights)

    assert "anchor_orth" not in baseline
    assert "reliability" not in baseline
    assert candidate["anchor_orth"].item() == pytest.approx(2.0)
    assert candidate["reliability"].item() == pytest.approx(3.0)
    assert candidate["total"].item() == pytest.approx(
        baseline["total"].item() + 0.13
    )


def test_factorized_known_losses_are_finite_and_differentiable():
    targets = torch.tensor([0, 1, 2, 3, 4])
    known_logits = torch.randn(5, 4, requires_grad=True)
    output = _hierarchical_output(known_logits, torch.randn(5, 8))
    weights = replace(HierarchicalLossWeights(), family=0.10, polarity=0.10)

    losses = hierarchical_type_loss(output, output, targets, weights=weights)
    losses["total"].backward()

    assert torch.isfinite(losses["family"] + losses["polarity"])
    assert known_logits.grad is not None and known_logits.grad.abs().sum() > 0


def test_hierarchical_loss_gradients_reach_all_type_components():
    model = create_hierarchical_type_model(
        base_channels=8,
        embedding_dim=16,
        prototypes_per_class=2,
    )
    batch_size = 5
    inputs = (
        torch.randn(batch_size, 1, 8000),
        torch.randn(batch_size, 1, 2000),
        torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0]]),
    )
    alternate_inputs = (
        torch.roll(inputs[0], 16, dims=2),
        torch.roll(inputs[1], 2, dims=2),
        inputs[2],
    )

    losses = hierarchical_type_loss(
        model(*inputs),
        model(*alternate_inputs),
        torch.tensor([0, 1, 2, 3, 4]),
        HierarchicalLossWeights(),
    )
    losses["total"].backward()

    assert model.gate_head.weight.grad is not None
    assert model.known_head.weight.grad is not None
    assert model.prototype_matcher.prototypes.grad is not None
    assert model.encoder.local_branch.network[0].weight.grad is not None
    assert model.encoder.global_branch.network[0].weight.grad is not None


class _TinyHierarchicalModel(nn.Module):
    architecture = HIERARCHICAL_TYPE_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward_hierarchical(self, local, global_view, daylight):
        self.calls += 1
        batch_size = len(local)
        known = self.weight * torch.arange(4).expand(batch_size, 4)
        embedding = self.weight * torch.ones(batch_size, 4)
        return _hierarchical_output(known, embedding)


def test_type_role_epoch_uses_both_hierarchical_views():
    model = _TinyHierarchicalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _training_batch([0, 1, 2, 3, 4], [-1, 1, 2, 3, 4])
    batch["local_alt"] = batch["local"].clone()
    batch["global_alt"] = batch["global"].clone()

    metrics = train_role_epoch(
        model,
        [batch],
        optimizer,
        "cpu",
        role="type",
        amp=False,
    )

    assert model.calls == 2
    assert metrics["sample_count"] == 5
    assert "known_accuracy" in metrics
    assert "known_to_ic_rate" in metrics


@pytest.mark.parametrize("aligned", [False, True])
def test_type_training_view_shift_matches_inference_without_mutating_batch(aligned):
    from evaluation import _shift_tensor_view

    class RecordingModel(_TinyHierarchicalModel):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward_hierarchical(self, local, global_view, daylight):
            self.inputs.append((local.detach().clone(), global_view.detach().clone()))
            return super().forward_hierarchical(local, global_view, daylight)

    model = RecordingModel()
    batch = _paired_training_batch([0, 1, 2, 3, 4], [-1, 1, 2, 3, 4])
    for key, length in (("local", 64), ("global", 16)):
        batch[key] = torch.arange(5 * length, dtype=torch.float32).reshape(5, 1, length)
        batch[key + "_alt"] = -batch[key].clone()
    before = {k: v.clone() for k, v in batch.items()}
    train_role_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=0.01),
                     "cpu", role="type", amp=False, inference_view_shift=aligned)
    for index, (key, shift) in enumerate((("local", 16), ("global", 4))):
        assert torch.equal(model.inputs[0][index], before[key])
        expected = _shift_tensor_view(before[key + "_alt"], shift) if aligned else before[key + "_alt"]
        assert torch.equal(model.inputs[1][index], expected)
    assert all(torch.equal(batch[k], v) for k, v in before.items())
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)


def test_inference_view_shift_rejects_nonhierarchical_or_distance_training():
    for model, role in ((_TinyRoleModel(), "type"), (_TinyHierarchicalModel(), "NNBE")):
        with pytest.raises(ValueError, match="requires hierarchical type"):
            train_role_epoch(model, [], torch.optim.SGD(model.parameters(), lr=0.01),
                             "cpu", role=role, inference_view_shift=True)


class _StaticRoleEvaluationModel(nn.Module):
    def __init__(self, type_logits, distance_heads):
        super().__init__()
        self.type_logits = type_logits
        self.distance_heads = distance_heads

    def forward_type(self, local, global_view, daylight):
        return self.type_logits, torch.zeros(len(local), 1)

    def forward_distance(self, local, global_view, daylight):
        return self.distance_heads, torch.zeros(len(local), 1)

    def forward_distance_type(
        self, local, global_view, daylight, *, expert_index
    ):
        return self.distance_heads[expert_index], torch.zeros(len(local), 1)


def test_type_role_evaluation_uses_only_type_network():
    winners = [0, 1, 2, 3, 4]
    model = _StaticRoleEvaluationModel(
        _logits_with_winners(winners, 5),
        tuple(torch.zeros(5, 30) for _ in range(4)),
    )
    batch = _training_batch(winners, [-1, 1, 2, 3, 4])

    metrics = evaluate_type_role(model, [batch], "cpu")

    assert metrics["piece_count"] == 5
    assert metrics["type_accuracy"] == 1.0
    assert metrics["type_macro_f1"] == 1.0


def test_distance_role_evaluation_reports_point_metrics():
    heads = [torch.zeros(2, 30) for _ in range(4)]
    heads[0] = _logits_with_winners([3, 6], 30)
    model = _StaticRoleEvaluationModel(
        torch.zeros(2, 5), tuple(heads)
    )
    batch = _training_batch([1, 1], [3, 4])

    metrics = evaluate_distance_role(
        model, [batch], "cpu", type_index=1
    )

    assert metrics["piece_count"] == 2
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["within_200"] == 1.0
    assert metrics["mae_km"] == 100.0
    assert metrics["expected_within_200"] == 1.0
    assert metrics["expected_mae_km"] == pytest.approx(100.0, abs=1e-3)


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


def _resume_components(*, scheduler_epoch=1, best_epoch=1, wait=0):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    loss = model(torch.ones(2, 2)).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    for _ in range(scheduler_epoch):
        scheduler.step()
    early_stopping = EarlyStoppingState()
    early_stopping.update(0.75, epoch=best_epoch, model=model)
    early_stopping.wait = wait
    return model, optimizer, scheduler, scaler, early_stopping


class _PartiallyUsedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(2, 1)
        self.unused = nn.Linear(2, 1)

    def forward(self, inputs):
        return self.used(inputs)


def test_exact_resume_accepts_recorded_partial_adamw_state(tmp_path):
    model = _PartiallyUsedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    early_stopping = EarlyStoppingState()
    early_stopping.update(0.75, epoch=1, model=model)
    path = tmp_path / "last.pt"

    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split",
        config_hash="config",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    declared_ids = payload["optimizer_state"]["param_groups"][0]["params"]

    assert payload["optimizer_state_membership"] == sorted(
        payload["optimizer_state"]["state"]
    )
    assert len(payload["optimizer_state_membership"]) == 2
    assert len(payload["optimizer_state_membership"]) < len(declared_ids)

    restored_model = _PartiallyUsedModel()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.01
    )
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=5
    )
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    restored_early_stopping = EarlyStoppingState()

    load_last_state(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        early_stopping=restored_early_stopping,
        expected_split_hash="split",
        expected_config_hash="config",
    )

    assert len(restored_optimizer.state) == 2


def test_exact_resume_restores_full_training_and_rng_state(tmp_path):
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    model, optimizer, scheduler, scaler, early_stopping = _resume_components(
        scheduler_epoch=3, best_epoch=1, wait=2
    )
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_scaler = copy.deepcopy(scaler.state_dict())
    expected_best = copy.deepcopy(early_stopping.best_state)
    path = tmp_path / "last.pt"

    save_last_state(
        path,
        epoch=3,
        sampler_epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split-abc",
        config_hash="config-abc",
    )
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer.param_groups[0]["lr"] = 99.0
    scheduler.last_epoch = 99
    early_stopping.best_score = -1.0
    early_stopping.best_epoch = -1
    early_stopping.wait = 99
    early_stopping.best_state = None
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    restored = load_last_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        expected_split_hash="split-abc",
        expected_config_hash="config-abc",
        device="cpu",
    )

    assert restored == {
        "epoch": 3,
        "sampler_epoch": 3,
        "split_hash": "split-abc",
        "config_hash": "config-abc",
    }
    for name, tensor in expected_model.items():
        assert torch.equal(model.state_dict()[name], tensor)
    restored_optimizer = optimizer.state_dict()
    assert restored_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert restored_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, expected_values in expected_optimizer["state"].items():
        for name, expected in expected_values.items():
            actual = restored_optimizer["state"][parameter_id][name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert scheduler.state_dict() == expected_scheduler
    assert scaler.state_dict() == expected_scaler
    assert early_stopping.best_score == 0.75
    assert early_stopping.best_epoch == 1
    assert early_stopping.wait == 2
    for name, tensor in expected_best.items():
        assert torch.equal(early_stopping.best_state[name], tensor)
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_resume_stages_checkpoint_on_cpu_before_device_restoration(
    tmp_path, monkeypatch
):
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split",
        config_hash="config",
    )
    requested_devices = []
    real_load = training._load_training_payload

    def record_load(checkpoint_path, device):
        requested_devices.append(torch.device(device))
        return real_load(checkpoint_path, "cpu")

    monkeypatch.setattr(training, "_load_training_payload", record_load)

    load_last_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        expected_split_hash="split",
        expected_config_hash="config",
        device="cuda",
    )

    assert requested_devices == [torch.device("cpu")]


@pytest.mark.parametrize(
    "tamper",
    [
        "scheduler_t_max",
        "scheduler_eta_min",
        "scheduler_last_epoch",
        "scheduler_step_count",
        "scheduler_missing_step_count",
        "scheduler_base_lrs",
        "scheduler_missing_last_lr",
        "scheduler_not_mapping",
        "optimizer_membership_missing",
        "optimizer_membership_not_list",
        "optimizer_membership_duplicate",
        "optimizer_membership_outside",
        "optimizer_missing_state_entry",
        "optimizer_added_state_entry",
        "early_wait",
        "sampler_epoch",
    ],
)
def test_resume_rejects_incoherent_metadata_before_live_state_mutation(
    tmp_path, tamper
):
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split",
        config_hash="config",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tamper == "scheduler_t_max":
        payload["scheduler_state"]["T_max"] = 99
    elif tamper == "scheduler_eta_min":
        payload["scheduler_state"]["eta_min"] = 0.25
    elif tamper == "scheduler_last_epoch":
        payload["scheduler_state"]["last_epoch"] = 0
    elif tamper == "scheduler_step_count":
        payload["scheduler_state"]["_step_count"] = 99
    elif tamper == "scheduler_missing_step_count":
        payload["scheduler_state"].pop("_step_count")
    elif tamper == "scheduler_base_lrs":
        payload["scheduler_state"]["base_lrs"] = [0.5]
    elif tamper == "scheduler_missing_last_lr":
        payload["scheduler_state"].pop("_last_lr")
    elif tamper == "scheduler_inconsistent_last_lr":
        payload["scheduler_state"]["_last_lr"] = [0.123]
    elif tamper == "scheduler_not_mapping":
        payload["scheduler_state"] = []
    elif tamper == "optimizer_group_count":
        payload["optimizer_state"]["param_groups"] = []
    elif tamper == "optimizer_parameter_membership":
        payload["optimizer_state"]["param_groups"][0]["params"].pop()
    elif tamper == "optimizer_weight_decay":
        payload["optimizer_state"]["param_groups"][0]["weight_decay"] = 0.25
    elif tamper == "optimizer_betas":
        payload["optimizer_state"]["param_groups"][0]["betas"] = (0.8, 0.9)
    elif tamper == "optimizer_boolean_type":
        payload["optimizer_state"]["param_groups"][0]["maximize"] = 0
    elif tamper == "optimizer_membership_missing":
        payload.pop("optimizer_state_membership")
    elif tamper == "optimizer_membership_not_list":
        payload["optimizer_state_membership"] = {}
    elif tamper == "optimizer_membership_duplicate":
        payload["optimizer_state_membership"].append(
            payload["optimizer_state_membership"][0]
        )
    elif tamper == "optimizer_membership_outside":
        payload["optimizer_state_membership"] = [999]
    elif tamper == "optimizer_missing_state_entry":
        payload["optimizer_state"]["state"].pop(
            next(iter(payload["optimizer_state"]["state"]))
        )
    elif tamper == "optimizer_added_state_entry":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        payload["optimizer_state"]["state"][999] = copy.deepcopy(first_state)
    elif tamper == "optimizer_state_keys":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state.pop("exp_avg")
    elif tamper == "optimizer_state_tensor_shape":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(1)
    elif tamper == "optimizer_step_type":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["step"] = 1.0
    elif tamper == "optimizer_step_shape":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["step"] = torch.ones(1, dtype=first_state["step"].dtype)
    elif tamper == "optimizer_step_dtype":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["step"] = first_state["step"].to(torch.int64)
    elif tamper == "optimizer_lr_only":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 0.123
    elif tamper == "optimizer_and_scheduler_forged_lr":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 0.123
        payload["scheduler_state"]["_last_lr"] = [0.123]
    elif tamper == "early_wait":
        payload["early_stopping"]["wait"] = 1
    elif tamper == "sampler_epoch":
        payload["sampler_epoch"] = 0
    torch.save(payload, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    unchanged = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="resume configuration mismatch"):
        load_last_state(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            expected_split_hash="split",
            expected_config_hash="config",
        )

    for name, expected in unchanged.items():
        assert torch.equal(model.state_dict()[name], expected)


@pytest.mark.parametrize(
    ("option", "changed"),
    [
        ("--batch_size", "65"),
        ("--base_channels", "16"),
        ("--seed", "43"),
    ],
)
def test_resume_rejects_training_configuration_changes(tmp_path, option, changed):
    base_args = train.build_parser().parse_args([])
    changed_args = train.build_parser().parse_args([option, changed])
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="same-split",
        config_hash=train.training_config_hash(base_args),
    )

    with pytest.raises(ValueError, match="resume configuration mismatch"):
        load_last_state(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            expected_split_hash="same-split",
            expected_config_hash=train.training_config_hash(changed_args),
            device="cpu",
        )


def test_resume_rejects_split_hash_mismatch(tmp_path):
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="original",
        config_hash="same-config",
    )

    with pytest.raises(ValueError, match="resume configuration mismatch"):
        load_last_state(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            expected_split_hash="changed",
            expected_config_hash="same-config",
            device="cpu",
        )


def test_resume_rejects_corrupt_best_state(tmp_path):
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split",
        config_hash="config",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["early_stopping"]["best_state"].pop(
        next(iter(payload["early_stopping"]["best_state"]))
    )
    torch.save(payload, path)

    with pytest.raises(ValueError, match="resume configuration mismatch"):
        load_last_state(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            expected_split_hash="split",
            expected_config_hash="config",
        )


def test_training_state_is_never_exposed_as_inference_checkpoint(tmp_path):
    model, optimizer, scheduler, scaler, early_stopping = _resume_components()
    path = tmp_path / "last.pt"
    save_last_state(
        path,
        epoch=1,
        sampler_epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        split_hash="split",
        config_hash="config",
    )

    with pytest.raises(ValueError, match="optimizer|checkpoint"):
        load_model_checkpoint(path, "cpu")


def _write_training_tree(root):
    for type_index, type_name in enumerate(("IC", "NCG", "NNBE", "PCG", "PNBE")):
        for daylight_name, hour in (("day", 3), ("night", 20)):
            for source_index in range(3):
                if type_name == "IC":
                    source = (
                        root
                        / type_name
                        / daylight_name
                        / f"sample_{source_index}.lig"
                    )
                else:
                    source = (
                        root
                        / type_name
                        / daylight_name
                        / f"{type_index * 100}-{(type_index + 1) * 100}km"
                        / f"sample_{source_index}.lig"
                    )
                source.parent.mkdir(parents=True, exist_ok=True)
                write_source(
                    source,
                    [
                        make_piece(
                            type_index * 100
                            + source_index * 10
                            + item,
                            hour=hour,
                        )
                        for item in range(3)
                    ],
                )


def _write_flat_classification_tree(root):
    for type_index, type_name in enumerate(("IC", "NCG", "NNBE", "PCG", "PNBE")):
        for daylight_name, hour in (("day", 3), ("night", 20)):
            source = root / type_name / daylight_name / "sample.lig"
            source.parent.mkdir(parents=True, exist_ok=True)
            write_source(
                source,
                [make_piece(type_index * 10 + item, hour=hour) for item in range(3)],
            )


def _smoke_args(root, output, *extra):
    return [
        "--task_data",
        str(root),
        "--output",
        str(output),
        "--epochs",
        "1",
        "--type_samples_per_epoch",
        "20",
        "--distance_samples_per_epoch",
        "20",
        "--batch_size",
        "5",
        "--num_workers",
        "0",
        "--base_channels",
        "8",
        "--no_amp",
        *extra,
    ]


def test_resume_with_exhausted_patience_does_not_train_another_epoch(
    tmp_path, monkeypatch
):
    root = tmp_path / "train_data"
    output = tmp_path / "output"
    _write_training_tree(root)
    calls = 0

    def restore_exhausted(_path, **kwargs):
        early_stopping = kwargs["early_stopping"]
        model = kwargs["model"]
        early_stopping.best_score = 0.5
        early_stopping.best_epoch = 1
        early_stopping.wait = 10
        early_stopping.best_state = copy.deepcopy(model.state_dict())
        return {
            "epoch": 0,
            "sampler_epoch": 0,
            "split_hash": kwargs["expected_split_hash"],
            "config_hash": kwargs["expected_config_hash"],
        }

    def count_training(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(train, "load_last_state", restore_exhausted)
    monkeypatch.setattr(train, "train_role_epoch", count_training)

    train.main(
        _smoke_args(
            root,
            output,
            "--stage",
            "type",
            "--resume",
            str(output / "type" / "last.pt"),
        )
    )

    assert calls == 0


def test_training_closes_every_dataset_when_validation_fails(tmp_path, monkeypatch):
    root = tmp_path / "train_data"
    output = tmp_path / "output"
    _write_training_tree(root)
    closed = set()
    real_close = train.FiveClassDataset.close

    def record_close(dataset):
        closed.add(dataset.split)
        real_close(dataset)

    monkeypatch.setattr(train.FiveClassDataset, "close", record_close)
    monkeypatch.setattr(train.FiveClassDataset, "__del__", lambda _dataset: None)
    monkeypatch.setattr(
        train,
        "evaluate_type_role",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("coverage")),
    )

    with pytest.raises(ValueError, match="coverage"):
        train.main(_smoke_args(root, output))

    assert closed == {"train", "validation", "test"}


def test_training_closes_constructed_dataset_when_later_constructor_fails(
    tmp_path, monkeypatch
):
    root = tmp_path / "train_data"
    output = tmp_path / "output"
    _write_training_tree(root)
    closed = set()
    real_dataset = train.FiveClassDataset
    real_close = real_dataset.close

    def record_close(dataset):
        closed.add(dataset.split)
        real_close(dataset)

    def fail_validation(*args, **kwargs):
        if kwargs["split"] == "validation":
            raise RuntimeError("validation constructor failed")
        return real_dataset(*args, **kwargs)

    monkeypatch.setattr(real_dataset, "close", record_close)
    monkeypatch.setattr(real_dataset, "__del__", lambda _dataset: None)
    monkeypatch.setattr(train, "FiveClassDataset", fail_validation)

    with pytest.raises(RuntimeError, match="validation constructor failed"):
        train.main(_smoke_args(root, output))

    assert closed == {"train"}


def test_single_split_training_smoke_writes_only_contract_artifacts(tmp_path):
    root = tmp_path / "train_data"
    output = tmp_path / "output"
    _write_training_tree(root)

    result = train.main(_smoke_args(root, output))

    assert set(result["roles"]) == {"type", "NCG", "NNBE", "PCG", "PNBE"}
    assert {path.name for path in output.iterdir()} == {
        "bundle.json",
        "bundle_metrics.json",
        "split.json",
        "type",
        "NCG",
        "NNBE",
        "PCG",
        "PNBE",
    }
    for role in ("type", "NCG", "NNBE", "PCG", "PNBE"):
        assert {path.name for path in (output / role).iterdir()} == {
            "model.pt",
            "last.pt",
            "metrics.json",
        }


def test_type_role_requires_distance_layout_for_shared_split(tmp_path):
    root = tmp_path / "train_data"
    output = tmp_path / "output"
    _write_flat_classification_tree(root)

    with pytest.raises(ValueError, match="100-km interval"):
        train.main(_smoke_args(root, output, "--stage", "type"))
