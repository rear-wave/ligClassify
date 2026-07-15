import pytest
import torch

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
    assert args.baseline_metrics == ""
    assert args.baseline_model_name == "old"
    assert not hasattr(args, "target_ic_fraction")


def test_resume_and_warm_start_are_mutually_exclusive():
    args = train.build_arg_parser().parse_args([
        "--resume", "latest.pt", "--init_model", "encoder.pt"
    ])

    with pytest.raises(ValueError, match="mutually exclusive"):
        train._validate_args(args)


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
