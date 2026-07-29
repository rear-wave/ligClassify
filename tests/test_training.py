from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import train
import training
from checkpoints import load_model_checkpoint
from evaluation import (
    evaluate_distance_role,
    evaluate_loader,
    evaluate_type_role,
    selection_score,
)
from models import ModelOutput
from training import (
    EarlyStoppingState,
    distance_expert_loss,
    load_last_state,
    save_last_state,
    train_role_epoch,
)
from tests.test_lig import make_piece, write_source


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
        path: len(Path(path).read_text(encoding="utf-8").splitlines())
        for path in runtime_paths
    }

    assert max(line_counts.values()) <= 600, line_counts


def test_training_defaults_match_five_class_contract():
    args = train.build_parser().parse_args([])

    assert args.task_data == r"..\train_data"
    assert args.output == r".\weights\multi_model"
    assert args.epochs == 50
    assert args.batch_size == 64
    assert args.patience == 10
    assert args.type_samples_per_epoch == 120000
    assert args.distance_samples_per_epoch == 60000
    assert args.num_workers == 0
    assert args.seed == 42
    assert args.ic_fraction == 0.60
    assert args.stage == "all"
    assert args.base_channels == 64
    assert args.lr == 0.0003
    assert args.weight_decay == 0.0005
    assert args.resume is None
    assert args.no_amp is False
    assert not hasattr(args, "init_model")


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
        [_training_batch([1, 1], [3, 4])],
        optimizer,
        "cpu",
        role="NCG",
        amp=False,
    )

    assert model.type_calls == 0
    assert model.distance_calls == 1
    assert metrics["sample_count"] == 2
    assert model.type_weight.grad is None
    assert model.distance_weight.grad is not None


def test_distance_expert_loss_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="shape"):
        distance_expert_loss(torch.zeros(2, 29), torch.tensor([1, 2]))


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
        ("--ic_fraction", "0.50"),
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
            if type_name == "IC":
                source = root / type_name / daylight_name / "sample.lig"
            else:
                source = (
                    root
                    / type_name
                    / daylight_name
                    / f"{type_index * 100}-{(type_index + 1) * 100}km"
                    / "sample.lig"
                )
            source.parent.mkdir(parents=True, exist_ok=True)
            write_source(
                source,
                [make_piece(type_index * 10 + item, hour=hour) for item in range(3)],
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
