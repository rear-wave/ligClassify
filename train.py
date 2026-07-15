"""Train the file-isolated four-type conditional lightning classifier."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from training_engine import compute_conditional_distance_loss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
RESEARCH_TYPE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]


def configure_cuda_backend(deterministic=False):
    """Select reproducible or RTX-optimized CUDA backend settings."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    if not deterministic:
        torch.set_float32_matmul_precision("high")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument("--output", default="./weights/conditional")
    parser.add_argument(
        "--model_arch",
        choices=["conditional_expert_v1"],
        default="conditional_expert_v1",
    )
    parser.add_argument(
        "--init_model",
        default="",
        help="Explicit compatible encoder checkpoint; empty means random initialization",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Exact latest.pt training state to resume; empty disables",
    )
    parser.add_argument(
        "--no_init",
        action="store_true",
        help="Compatibility flag that forces random initialization",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--samples_per_epoch", type=int, default=120000)
    parser.add_argument("--max_samples_per_file", type=int, default=512)
    parser.add_argument("--type_focus_epochs", type=int, default=3)
    parser.add_argument("--type_focus_distance_weight", type=float, default=0.25)
    parser.add_argument("--joint_distance_weight", type=float, default=1.0)
    parser.add_argument(
        "--time_context",
        choices=["daylight", "cyclic"],
        default="daylight",
        help="Daylight-only training by default; cyclic is ablation-only",
    )
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--dist_mlp_dim", type=int, default=128)
    parser.add_argument("--dist_dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--lambda_coarse", type=float, default=0.5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Windows-safe default; use 2-4 only after a successful smoke run",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--calibration_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--rejection_target_precision", type=float, default=0.95)
    parser.add_argument("--rejection_min_coverage", type=float, default=0.80)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument(
        "--baseline_metrics",
        default="",
        help="Same-split baseline metrics required before model.pt promotion",
    )
    parser.add_argument(
        "--baseline_model_name",
        default="old",
        help="Model key when --baseline_metrics points to benchmark.json",
    )
    return parser


def _validate_args(args):
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.samples_per_epoch <= 0 or args.max_samples_per_file <= 0:
        raise ValueError("sampling limits must be positive")
    if args.type_focus_epochs < 0:
        raise ValueError("--type_focus_epochs must be non-negative")
    if args.epochs <= args.type_focus_epochs:
        raise ValueError("training requires at least one joint-stage epoch")
    if (
        args.type_focus_distance_weight < 0
        or args.joint_distance_weight < 0
    ):
        raise ValueError("distance loss weights must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    if args.resume and args.init_model:
        raise ValueError("--resume and --init_model are mutually exclusive")


def main():
    args = build_arg_parser().parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        configure_cuda_backend(args.deterministic)
    LOGGER.info(
        "device=%s architecture=%s random_init=%s amp=%s",
        device,
        args.model_arch,
        not bool(args.init_model or args.resume),
        device == "cuda" and not args.no_amp,
    )
    from conditional_pipeline import run_conditional_training

    run_conditional_training(args, device)


if __name__ == "__main__":
    main()
