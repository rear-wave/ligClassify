"""Train the file-isolated four-type conditional lightning classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from cv_pipeline import (
    CVConfig,
    run_cross_validated_training,
    training_config_hash,
    verify_cv_artifacts,
)
from data.cross_validation import assign_exact_folds, fold_train_holdout
from data.oof_manifest import expected_oof_rows
from data.split_artifacts import make_fold_manifest, split_hash
from data.training_manifest import build_manifest
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
        "--init_model",
        default="",
        help="Legacy compatibility option; any non-empty value is rejected",
    )
    parser.add_argument(
        "--no_init",
        action="store_true",
        help="Compatibility flag that forces random initialization",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Windows-safe default; use 2-4 only after a successful smoke run",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--rejection_target_precision", type=float, default=0.96)
    parser.add_argument("--rejection_min_coverage", type=float, default=0.80)
    parser.add_argument("--resume_cv", action="store_true")
    parser.add_argument("--stop_after_oof", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    return parser


def _validate_args(args):
    if args.folds != 3:
        raise ValueError("cross-validation requires exactly three folds")
    if args.init_model:
        raise ValueError("cross-validation requires random initialization")
    if args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("--max_epochs and --patience must be positive")
    if args.samples_per_epoch <= 0 or args.max_samples_per_file <= 0:
        raise ValueError("sampling limits must be positive")
    if args.type_focus_epochs < 0:
        raise ValueError("--type_focus_epochs must be non-negative")
    if args.max_epochs <= args.type_focus_epochs:
        raise ValueError("training requires at least one joint-stage epoch")
    if (
        args.type_focus_distance_weight < 0
        or args.joint_distance_weight < 0
    ):
        raise ValueError("distance loss weights must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")


def _cv_config_from_args(args):
    return CVConfig(
        task_data=Path(args.task_data),
        output=Path(args.output),
        folds=args.folds,
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
        max_samples_per_file=args.max_samples_per_file,
        type_focus_epochs=args.type_focus_epochs,
        type_focus_distance_weight=args.type_focus_distance_weight,
        joint_distance_weight=args.joint_distance_weight,
        max_epochs=args.max_epochs,
        patience=args.patience,
        time_context=args.time_context,
        rejection_target_precision=args.rejection_target_precision,
        rejection_min_coverage=args.rejection_min_coverage,
        bootstrap_iterations=args.bootstrap_iterations,
        num_workers=args.num_workers,
        no_amp=args.no_amp,
        no_init=True,
        init_model="",
        resume_cv=args.resume_cv,
        stop_after_oof=args.stop_after_oof,
    )


def _manifest_contract(task_data):
    entries, diagnostics = build_manifest(task_data, RESEARCH_TYPE_NAMES)
    if not entries:
        raise RuntimeError(
            f"No valid trusted-type .lig files found under {task_data}"
        )
    if any(not 0 <= int(entry.type_idx) < 4 for entry in entries):
        raise ValueError("four-class CV manifest must contain zero IC rows")
    represented = {int(entry.type_idx) for entry in entries}
    if represented != {0, 1, 2, 3}:
        raise ValueError("four-class CV manifest must contain all four researched types")
    return entries, diagnostics


def _expected_cv_contract(entries, config):
    folds = assign_exact_folds(
        entries, n_folds=config.folds, seed=config.seed
    )
    manifest = make_fold_manifest(folds, config.task_data, config.seed)
    expected = expected_oof_rows(manifest)
    config_hash = training_config_hash(config)
    hashes = {}
    for fold_index in range(config.folds):
        train_entries, holdout_entries = fold_train_holdout(folds, fold_index)
        hashes[str(fold_index)] = {
            "train_hash": split_hash(train_entries, config.task_data),
            "holdout_hash": split_hash(holdout_entries, config.task_data),
            "config_hash": config_hash,
        }
    return expected, hashes


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    _validate_args(args)
    config = _cv_config_from_args(args)
    entries, diagnostics = _manifest_contract(args.task_data)
    LOGGER.info(
        "manifest=%d trusted files invalid=%d trained_ic=0",
        len(entries),
        int(diagnostics.get("invalid_files", 0)),
    )
    if args.verify_only:
        expected, hashes = _expected_cv_contract(entries, config)
        return verify_cv_artifacts(config.output, expected, hashes, config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        configure_cuda_backend(deterministic=False)
    LOGGER.info(
        "device=%s random_init=%s amp=%s",
        device,
        True,
            device == "cuda" and not args.no_amp,
        )
    return run_cross_validated_training(config, entries, device)


if __name__ == "__main__":
    main()
