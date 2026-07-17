"""Train one randomly initialized five-class lightning classifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from checkpoints import save_model_checkpoint
from data.dataset import FiveClassDataset, collate_batch
from data.manifest import build_piece_table
from data.preprocess import AugmentationConfig, PreprocessConfig
from data.sampling import FiveClassSampler
from data.split import (
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
    write_split_json,
)
from evaluation import evaluate_loader, selection_score
from models import create_five_class_model
from training import (
    EarlyStoppingState,
    load_last_state,
    save_last_state,
    train_epoch,
)


FIXED_IC_FRACTION = 0.60


def build_parser() -> argparse.ArgumentParser:
    """Build the compact five-class training argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default=r"..\train_data")
    parser.add_argument("--output", default=r".\weights\five_class")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples_per_epoch", type=int, default=120000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ic_fraction", type=float, default=FIXED_IC_FRACTION)
    parser.add_argument("--distance_weight", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--resume")
    parser.add_argument("--no_amp", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "patience",
        "samples_per_epoch",
        "base_channels",
    )
    for field in positive_integer_fields:
        value = getattr(args, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{field} must be a positive integer")
    if type(args.num_workers) is not int or args.num_workers < 0:
        raise ValueError("--num_workers must be a non-negative integer")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if float(args.ic_fraction) != FIXED_IC_FRACTION:
        raise ValueError(
            "--ic_fraction must be 0.60 for the fixed 60/10/10/10/10 prior"
        )
    for field in ("distance_weight", "lr", "weight_decay"):
        value = float(getattr(args, field))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{field} must be finite and non-negative")
    if args.lr == 0.0:
        raise ValueError("--lr must be positive")
    if args.resume and Path(args.resume).name.casefold() == "model.pt":
        raise ValueError("model.pt is an inference checkpoint and cannot be resumed")


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "five_class_training_config_v1",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "patience": int(args.patience),
        "samples_per_epoch": int(args.samples_per_epoch),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "ic_fraction": float(args.ic_fraction),
        "distance_weight": float(args.distance_weight),
        "base_channels": int(args.base_channels),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "amp": not bool(args.no_amp),
    }


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_config_hash(args: argparse.Namespace) -> str:
    """Hash every same-run setting that must match exact resume."""
    return _canonical_hash(_training_config(args))


def _split_hash(artifact: Mapping[str, Any]) -> str:
    return _canonical_hash(artifact)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_split(path: Path, artifact: Mapping[str, Any]) -> None:
    temporary = Path(f"{path}.tmp")
    try:
        write_split_json(temporary, artifact)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: FiveClassDataset,
    *,
    batch_size: int,
    num_workers: int,
    sampler: FiveClassSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def _fit_training(
    args: argparse.Namespace,
    output: Path,
    table: Any,
    diagnostics: Mapping[str, Any],
    artifact: Mapping[str, Any],
    train_positions: np.ndarray,
    preprocess_config: PreprocessConfig,
    train_dataset: FiveClassDataset,
    validation_dataset: FiveClassDataset,
    test_dataset: FiveClassDataset,
) -> dict[str, Any]:
    sampler = FiveClassSampler(
        table,
        train_positions,
        num_samples=args.samples_per_epoch,
        seed=args.seed,
    )

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_five_class_model(
        base_channels=args.base_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    early_stopping = EarlyStoppingState()
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = _loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    config = _training_config(args)
    config_hash = training_config_hash(args)
    split_hash = _split_hash(artifact)
    start_epoch = 1
    if args.resume:
        restored = load_last_state(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            expected_split_hash=split_hash,
            expected_config_hash=config_hash,
            device=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        if int(restored["sampler_epoch"]) != int(restored["epoch"]):
            raise ValueError("resume configuration mismatch: sampler continuity")

    validation_metrics: dict[str, Any] | None = None
    epochs_completed = start_epoch - 1
    remaining_epochs = (
        range(start_epoch, args.epochs + 1)
        if early_stopping.wait < args.patience
        else ()
    )
    for epoch in remaining_epochs:
        sampler.set_epoch(epoch)
        train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            amp=amp_enabled,
            distance_weight=args.distance_weight,
        )
        validation_metrics = evaluate_loader(model, validation_loader, device)
        score = selection_score(validation_metrics)
        early_stopping.update(score, epoch, model)
        scheduler.step()
        epochs_completed = epoch
        save_last_state(
            output / "last.pt",
            epoch=epoch,
            sampler_epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            split_hash=split_hash,
            config_hash=config_hash,
        )
        if early_stopping.wait >= args.patience:
            break

    if early_stopping.best_state is None:
        raise ValueError("training produced no validation-selected model")
    model.load_state_dict(early_stopping.best_state, strict=True)
    save_model_checkpoint(
        output / "model.pt",
        model,
        model_config={"base_channels": args.base_channels},
        preprocess_config=asdict(preprocess_config),
        split_hash=split_hash,
        training_config=config,
    )
    validation_metrics = evaluate_loader(model, validation_loader, device)
    test_metrics = evaluate_loader(model, test_loader, device)
    metrics: dict[str, Any] = {
        "schema": "five_class_metrics_v1",
        "best_epoch": early_stopping.best_epoch,
        "best_score": early_stopping.best_score,
        "epochs_completed": epochs_completed,
        "validation": validation_metrics,
        "test": test_metrics,
        "split_hash": split_hash,
        "training_config_hash": config_hash,
        "data": dict(diagnostics),
    }
    _atomic_write_json(output / "metrics.json", metrics)
    return metrics


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run deterministic single-split selection and one final test evaluation."""
    _validate_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    table, diagnostics = build_piece_table(args.task_data)
    assignment = assign_piece_splits(table, seed=args.seed)
    validate_piece_split(table, assignment)
    artifact = split_artifact(table, assignment)
    _atomic_write_split(output / "split.json", artifact)

    train_positions = assignment.positions("train")
    validation_positions = assignment.positions("validation")
    test_positions = assignment.positions("test")

    preprocess_config = PreprocessConfig()
    train_dataset = FiveClassDataset(
        table,
        train_positions,
        split="train",
        preprocess_config=preprocess_config,
        augmentation_config=AugmentationConfig(),
    )
    validation_dataset = FiveClassDataset(
        table,
        validation_positions,
        split="validation",
        preprocess_config=preprocess_config,
    )
    test_dataset = FiveClassDataset(
        table,
        test_positions,
        split="test",
        preprocess_config=preprocess_config,
    )
    try:
        return _fit_training(
            args,
            output,
            table,
            diagnostics,
            artifact,
            train_positions,
            preprocess_config,
            train_dataset,
            validation_dataset,
            test_dataset,
        )
    finally:
        train_dataset.close()
        validation_dataset.close()
        test_dataset.close()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
