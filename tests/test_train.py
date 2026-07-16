from types import SimpleNamespace

import pytest
import torch

import conditional_pipeline
import evaluation
import train


def test_research_type_order_is_stable_and_has_no_ic():
    assert train.RESEARCH_TYPE_NAMES == ["NCG", "NNBE", "PCG", "PNBE"]


def test_conditional_training_arguments_have_reliable_defaults():
    args = train.build_arg_parser().parse_args([])

    assert args.task_data == "../train_data"
    assert args.output == "./weights/conditional"
    assert args.init_model == ""
    assert args.no_init is False
    assert args.no_amp is False
    assert args.bootstrap_iterations == 1000
    assert args.rejection_target_precision == 0.96
    assert args.rejection_min_coverage == 0.80
    assert args.samples_per_epoch == 120000
    assert args.max_samples_per_file == 512
    assert args.type_focus_epochs == 3
    assert args.type_focus_distance_weight == 0.25
    assert args.joint_distance_weight == 1.0
    assert args.time_context == "daylight"
    assert args.folds == 3
    assert args.max_epochs == 50
    assert args.resume_cv is False
    assert args.stop_after_oof is False
    assert args.verify_only is False
    assert not hasattr(args, "resume")
    assert not hasattr(args, "baseline_metrics")
    assert not hasattr(args, "baseline_model_name")
    assert not hasattr(args, "target_ic_fraction")
    assert not hasattr(args, "type_samples_per_epoch")
    assert not hasattr(args, "distance_samples_per_epoch")
    assert not hasattr(args, "distance_batch_size")
    assert not hasattr(args, "distance_batch_type_weight")
    for obsolete in (
        "model_arch", "batch_size", "base", "dist_mlp_dim", "dist_dropout",
        "lr", "wd", "lambda_coarse", "val_fraction", "test_fraction",
        "deterministic", "calibration_samples",
    ):
        assert not hasattr(args, obsolete)


def test_cv_rejects_warm_start():
    args = train.build_arg_parser().parse_args([
        "--init_model", "encoder.pt"
    ])

    with pytest.raises(ValueError, match="random initialization"):
        train._validate_args(args)


def test_training_requires_at_least_one_joint_epoch():
    args = train.build_arg_parser().parse_args([
        "--max_epochs", "3", "--type_focus_epochs", "3"
    ])

    with pytest.raises(ValueError, match="joint-stage epoch"):
        train._validate_args(args)


def test_one_epoch_smoke_uses_joint_stage_from_epoch_zero():
    args = train.build_arg_parser().parse_args([
        "--max_epochs", "1",
        "--type_focus_epochs", "0",
        "--patience", "1",
    ])

    train._validate_args(args)


def test_first_joint_epoch_is_selected_after_legacy_resume():
    better_historical_score = (1.0, 1.0)
    current_score = (0.0, 0.0)

    assert conditional_pipeline._joint_checkpoint_improved(
        "joint", current_score, better_historical_score, best_epoch=0
    )
    assert not conditional_pipeline._joint_checkpoint_improved(
        "type_focus", current_score, None, best_epoch=0
    )


def test_conditional_pipeline_uses_shared_checkpoint_selection_key():
    assert (
        conditional_pipeline.checkpoint_selection_key
        is evaluation.checkpoint_selection_key
    )
    assert not hasattr(conditional_pipeline, "_selection_key")


def test_conditional_pipeline_logs_100km_interval_metric():
    metrics = {
        "type_piece_accuracy": 0.90,
        "type_file_macro_accuracy": 0.91,
        "distance_100km_interval_within_200": 0.92,
        "distance_file_macro_within_200": 0.93,
        "distance_interval_mae_km": 100.0,
    }

    conditional_pipeline._log_metrics("validation", metrics)


def test_candidate_release_does_not_require_historical_baseline():
    args = SimpleNamespace(time_context="daylight", skip_test=False)
    metrics = {
        "type_file_equal_precision": [0.96, 0.97, 0.98, 0.99],
        "type_coverage": 0.82,
        "type_file_equal_recall_mean": 0.91,
        "distance_100km_interval_within_200": 0.88,
        "distance_per_type_100km_interval_within_200": [0.80, 0.85, 0.90, 0.95],
        "distance_conditions_100km": {},
    }

    assert conditional_pipeline._evaluate_candidate_release(
        args,
        metrics,
        calibration_error=None,
    ) == (True, [])


def test_conditional_distance_loss_routes_true_type_and_broad_intervals():
    distance_logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    coarse_logits = [torch.zeros(4, 6, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([0, 1, 2, 3])
    low = torch.tensor([0.0, 100.0, 600.0, 1500.0])
    high = torch.tensor([300.0, 200.0, 1200.0, 3000.0])

    loss, components = train.compute_conditional_distance_loss(
        distance_logits, coarse_logits, labels, low, high
    )
    loss.backward()

    assert set(components) == {"interval", "coarse"}
    assert all(head.grad[index].abs().sum() > 0 for index, head in enumerate(distance_logits))
    assert all(head.grad[index].abs().sum() > 0 for index, head in enumerate(coarse_logits))


def test_cuda_backend_defaults_to_fast_tf32_mode():
    old = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        train.configure_cuda_backend(deterministic=False)
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.allow_tf32 is True

        train.configure_cuda_backend(deterministic=True)
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cudnn.benchmark = old[0]
        torch.backends.cudnn.deterministic = old[1]
        torch.backends.cudnn.allow_tf32 = old[2]


def test_main_builds_trusted_manifest_and_cv_config(tmp_path, monkeypatch):
    from tests.test_cv_pipeline import trusted_entries

    captured = {}
    monkeypatch.setattr(
        train,
        "build_manifest",
        lambda task_data, names: (trusted_entries(tmp_path), {"valid_files": 12}),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        train,
        "run_cross_validated_training",
        lambda config, entries, device: captured.update({
            "config": config, "entries": entries, "device": device
        }) or {"passed": True},
    )

    report = train.main([
        "--task_data", str(tmp_path / "train"),
        "--output", str(tmp_path / "weights"),
        "--samples_per_epoch", "24",
        "--max_epochs", "4",
        "--type_focus_epochs", "1",
        "--stop_after_oof",
    ])

    assert report == {"passed": True}
    assert captured["device"] == "cpu"
    assert captured["config"].no_init is True
    assert captured["config"].init_model == ""
    assert captured["config"].samples_per_epoch == 24
    assert captured["config"].stop_after_oof is True
    assert {entry.type_idx for entry in captured["entries"]} == {0, 1, 2, 3}


def test_verify_only_never_queries_cuda_or_starts_training(tmp_path, monkeypatch):
    from tests.test_cv_pipeline import trusted_entries

    captured = {}
    monkeypatch.setattr(
        train,
        "build_manifest",
        lambda task_data, names: (trusted_entries(tmp_path), {"valid_files": 12}),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be queried")),
    )
    monkeypatch.setattr(
        train,
        "run_cross_validated_training",
        lambda *args: (_ for _ in ()).throw(AssertionError("training must not start")),
    )
    monkeypatch.setattr(
        train,
        "verify_cv_artifacts",
        lambda output, expected, hashes, config: captured.update({
            "output": output,
            "expected": expected,
            "hashes": hashes,
            "config": config,
        }) or {"passed": False, "reasons": ["gate"]},
    )

    report = train.main([
        "--task_data", str(tmp_path / "train"),
        "--output", str(tmp_path / "weights"),
        "--verify_only",
    ])

    assert report == {"passed": False, "reasons": ["gate"]}
    assert len(captured["expected"]) == 24
    assert set(captured["hashes"]) == {"0", "1", "2"}


def test_main_rejects_any_non_research_manifest_type_before_cuda(
    tmp_path, monkeypatch
):
    from tests.test_cv_pipeline import trusted_entries

    entries = trusted_entries(tmp_path)
    entries[0] = SimpleNamespace(**{**entries[0].__dict__, "type_idx": 4})
    monkeypatch.setattr(
        train, "build_manifest", lambda *args: (entries, {"valid_files": 12})
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be queried")),
    )
    with pytest.raises(ValueError, match="zero IC"):
        train.main([
            "--task_data", str(tmp_path / "train"),
            "--output", str(tmp_path / "weights"),
        ])
