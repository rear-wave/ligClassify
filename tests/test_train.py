import pytest
import torch

import conditional_pipeline
import train


def test_research_type_order_is_stable_and_has_no_ic():
    assert train.RESEARCH_TYPE_NAMES == ["NCG", "NNBE", "PCG", "PNBE"]


def test_conditional_training_arguments_have_reliable_defaults():
    args = train.build_arg_parser().parse_args([])

    assert args.task_data == "../train_data"
    assert args.output == "./weights/conditional"
    assert args.model_arch == "conditional_expert_v1"
    assert args.init_model == args.resume == ""
    assert args.no_init is False
    assert args.no_amp is False
    assert args.calibration_samples == 20000
    assert args.bootstrap_iterations == 1000
    assert args.rejection_target_precision == 0.95
    assert args.rejection_min_coverage == 0.80
    assert args.samples_per_epoch == 120000
    assert args.max_samples_per_file == 512
    assert args.type_focus_epochs == 3
    assert args.type_focus_distance_weight == 0.25
    assert args.joint_distance_weight == 1.0
    assert args.time_context == "daylight"
    assert args.baseline_metrics == ""
    assert args.baseline_model_name == "old"
    assert not hasattr(args, "target_ic_fraction")
    assert not hasattr(args, "type_samples_per_epoch")
    assert not hasattr(args, "distance_samples_per_epoch")
    assert not hasattr(args, "distance_batch_size")
    assert not hasattr(args, "distance_batch_type_weight")


def test_resume_and_warm_start_are_mutually_exclusive():
    args = train.build_arg_parser().parse_args([
        "--resume", "latest.pt", "--init_model", "encoder.pt"
    ])

    with pytest.raises(ValueError, match="mutually exclusive"):
        train._validate_args(args)


def test_training_requires_at_least_one_joint_epoch():
    args = train.build_arg_parser().parse_args([
        "--epochs", "3", "--type_focus_epochs", "3"
    ])

    with pytest.raises(ValueError, match="joint-stage epoch"):
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
