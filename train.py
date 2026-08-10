"""Train one type model and four independent distance models."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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

from checkpoints import (
    load_model_bundle,
    save_model_bundle,
    save_model_checkpoint,
)
from data.dataset import FiveClassDataset, collate_batch
from data.manifest import PieceTable, build_piece_table
from data.preprocess import AugmentationConfig, PreprocessConfig
from data.sampling import DistanceExpertSampler, FiveClassSampler, TYPE_PRIOR
from data.split import (
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
    write_split_json,
)
from evaluation import (
    evaluate_distance_role,
    evaluate_model_bundle,
    evaluate_type_role,
    hierarchical_selection_score,
)
from models import DISTANCE_NAMES, create_five_class_model
from training import (
    DistanceLossWeights,
    EarlyStoppingState,
    HierarchicalLossWeights,
    load_last_state,
    save_last_state,
    train_role_epoch,
)


TRAINING_ROLES = ("type", *DISTANCE_NAMES)


def build_parser() -> argparse.ArgumentParser:
    """Build the five-role training command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default=r"..\train_data")
    parser.add_argument("--output", default=r".\weights\multi_model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--type_samples_per_epoch", type=int, default=120000)
    parser.add_argument(
        "--distance_samples_per_epoch", type=int, default=60000
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage", choices=("all", *TRAINING_ROLES), default="all"
    )
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for field in (
        "epochs",
        "batch_size",
        "patience",
        "type_samples_per_epoch",
        "distance_samples_per_epoch",
        "base_channels",
    ):
        value = getattr(args, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{field} must be a positive integer")
    if type(args.num_workers) is not int or args.num_workers < 0:
        raise ValueError("--num_workers must be a non-negative integer")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if args.stage in ("all", "type") and args.batch_size % 5:
        raise ValueError("--batch_size must be divisible by five for type training")
    for field in ("lr", "weight_decay"):
        value = float(getattr(args, field))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{field} must be finite and non-negative")
    if float(args.lr) == 0.0:
        raise ValueError("--lr must be positive")
    if args.resume and args.stage == "all":
        raise ValueError("--resume requires one explicit --stage")
    if args.resume and Path(args.resume).name.casefold() == "model.pt":
        raise ValueError("model.pt is an inference checkpoint and cannot resume")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")


def _training_config(
    args: argparse.Namespace,
    role: str | None = None,
    *,
    effective_samples_per_epoch: int | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    selected_role = role or str(args.stage)
    role_seed = (
        args.seed + TRAINING_ROLES.index(selected_role)
        if selected_role in TRAINING_ROLES
        else args.seed
    )
    samples = (
        args.type_samples_per_epoch
        if selected_role == "type"
        else args.distance_samples_per_epoch
    )
    effective_samples = (
        int(samples)
        if effective_samples_per_epoch is None
        else int(effective_samples_per_epoch)
    )
    config = {
        "schema": "five_class_role_training_v2",
        "role": selected_role,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "patience": int(args.patience),
        "requested_samples_per_epoch": int(samples),
        "samples_per_epoch": effective_samples,
        "num_workers": int(args.num_workers),
        "seed": int(role_seed),
        "sampling_prior": list(TYPE_PRIOR) if selected_role == "type" else None,
        "base_channels": int(args.base_channels),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "amp": not bool(args.no_amp),
        "hierarchical_loss": (
            asdict(HierarchicalLossWeights())
            if selected_role == "type"
            else None
        ),
        "distance_loss": (
            asdict(DistanceLossWeights())
            if selected_role != "type"
            else None
        ),
    }
    if preprocess_config is not None:
        config["preprocess_config"] = asdict(preprocess_config)
    return config


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_config_hash(
    args: argparse.Namespace, role: str | None = None
) -> str:
    """Hash every setting required for exact same-role resume."""
    return _canonical_hash(_training_config(args, role))


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
    sampler: FiveClassSampler | DistanceExpertSampler | None = None,
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


def _role_positions(
    table: PieceTable, positions: np.ndarray, role: str
) -> np.ndarray:
    if role == "type":
        return positions
    type_index = DISTANCE_NAMES.index(role) + 1
    return positions[table.type_index[positions] == type_index]


def _evaluate_role(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    role: str,
) -> dict[str, object]:
    if role == "type":
        return evaluate_type_role(model, loader, device)
    return evaluate_distance_role(
        model,
        loader,
        device,
        type_index=DISTANCE_NAMES.index(role) + 1,
    )


def _selection_score(role: str, metrics: Mapping[str, object]) -> float:
    if role == "type":
        return hierarchical_selection_score(metrics)
    return float(metrics["expected_within_200"]) - 1e-3 * float(
        metrics["expected_mae_bins"]
    )


def _epoch_summary(
    role: str,
    epoch: int,
    train_metrics: Mapping[str, object],
    validation: Mapping[str, object],
    wait: int,
) -> str:
    if role == "type":
        validation_text = (
            f"val_acc={float(validation['type_accuracy']):.4f} "
            f"known_recall={float(validation['known_macro_recall']):.4f} "
            f"false_ic={float(validation['max_known_false_to_ic']):.4f}"
        )
    else:
        validation_text = (
            f"val_w200={float(validation['expected_within_200']):.4f} "
            f"val_mae={float(validation['expected_mae_km']):.0f}km"
        )
    return (
        f"{role} Epoch {epoch:3d} | "
        f"loss={float(train_metrics['loss']):.4f} "
        f"acc={float(train_metrics['accuracy']):.4f} | "
        f"{validation_text} | wait={wait}"
    )


def _fit_role(
    args: argparse.Namespace,
    output: Path,
    role: str,
    table: PieceTable,
    artifact: Mapping[str, Any],
    split_positions: Mapping[str, np.ndarray],
    preprocess_config: PreprocessConfig,
) -> dict[str, Any]:
    role_output = output / role
    role_output.mkdir(parents=True, exist_ok=True)
    role_seed = args.seed + TRAINING_ROLES.index(role)
    _seed_everything(role_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if role == "type":
        from models import create_hierarchical_type_model

        model = create_hierarchical_type_model(
            base_channels=args.base_channels
        ).to(device)
    else:
        model = create_five_class_model(args.base_channels).to(device)
        model.set_training_role(role)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    stopping = EarlyStoppingState()
    split_hash = _canonical_hash(artifact)

    with ExitStack() as stack:
        datasets: dict[str, FiveClassDataset] = {}
        loaders: dict[str, DataLoader] = {}
        for split in ("train", "validation", "test"):
            positions = _role_positions(
                table, split_positions[split], role
            )
            if not len(positions):
                raise ValueError(f"{role} {split} split is empty")
            dataset = FiveClassDataset(
                table,
                positions,
                split=split,
                preprocess_config=preprocess_config,
                augmentation_config=(
                    AugmentationConfig() if split == "train" else None
                ),
            )
            stack.callback(dataset.close)
            datasets[split] = dataset
        if role == "type":
            type_counts = np.bincount(
                table.type_index[
                    _role_positions(
                        table, split_positions["train"], role
                    )
                ],
                minlength=5,
            )
            maximum_type_samples = 5 * int(type_counts.min())
            effective_type_samples = min(
                int(args.type_samples_per_epoch),
                maximum_type_samples,
            )
            effective_type_samples -= effective_type_samples % 5
            if effective_type_samples <= 0:
                raise ValueError(
                    "type training requires at least one piece per class"
                )
            if effective_type_samples < int(args.type_samples_per_epoch):
                print(
                    "type samples per epoch capped at "
                    f"{effective_type_samples} "
                    f"({effective_type_samples // 5} per class; "
                    "no replacement)"
                )
            sampler = FiveClassSampler(
                table,
                split_positions["train"],
                num_samples=effective_type_samples,
                seed=role_seed,
            )
            effective_samples = effective_type_samples
        else:
            sampler = DistanceExpertSampler(
                table,
                split_positions["train"],
                type_index=DISTANCE_NAMES.index(role) + 1,
                num_samples=args.distance_samples_per_epoch,
                seed=role_seed,
            )
            effective_samples = int(args.distance_samples_per_epoch)
        config = _training_config(
            args,
            role,
            effective_samples_per_epoch=effective_samples,
            preprocess_config=preprocess_config,
        )
        config_hash = _canonical_hash(config)
        loaders["train"] = _loader(
            datasets["train"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
        )
        for split in ("validation", "test"):
            loaders[split] = _loader(
                datasets[split],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

        start_epoch = 1
        if args.resume:
            restored = load_last_state(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=stopping,
                expected_split_hash=split_hash,
                expected_config_hash=config_hash,
                device=device,
            )
            start_epoch = int(restored["epoch"]) + 1
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs + 1):
            if stopping.wait >= args.patience:
                break
            sampler.set_epoch(epoch)
            train_metrics = train_role_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                role=role,
                scaler=scaler,
                amp=amp_enabled,
            )
            validation = _evaluate_role(
                model, loaders["validation"], device, role
            )
            stopping.update(
                _selection_score(role, validation), epoch, model
            )
            print(
                _epoch_summary(
                    role, epoch, train_metrics, validation, stopping.wait
                ),
                flush=True,
            )
            scheduler.step()
            last_epoch = epoch
            save_last_state(
                role_output / "last.pt",
                epoch=epoch,
                sampler_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=stopping,
                split_hash=split_hash,
                config_hash=config_hash,
            )
        if stopping.best_state is None:
            raise ValueError(f"{role} training selected no model")
        model.load_state_dict(stopping.best_state, strict=True)
        decision_config = None
        if role == "type":
            from evaluation import (
                decision_config_dict,
                evaluate_hierarchical_type_role,
            )

            validation, calibrated_config = (
                evaluate_hierarchical_type_role(
                    model, loaders["validation"], device
                )
            )
            test, _ = evaluate_hierarchical_type_role(
                model,
                loaders["test"],
                device,
                decision_config=calibrated_config,
            )
            decision_config = decision_config_dict(calibrated_config)
        else:
            validation = _evaluate_role(
                model, loaders["validation"], device, role
            )
            test = _evaluate_role(model, loaders["test"], device, role)
        save_model_checkpoint(
            role_output / "model.pt",
            model,
            model_config={
                "base_channels": args.base_channels,
                **(
                    {
                        "embedding_dim": int(
                            next(
                                value
                                for name, value in model.state_dict().items()
                                if name.endswith("prototypes")
                                and value.ndim == 3
                            ).shape[-1]
                        ),
                        "prototypes_per_class": int(
                            next(
                                value
                                for name, value in model.state_dict().items()
                                if name.endswith("prototypes")
                                and value.ndim == 3
                            ).shape[-2]
                        ),
                        "prototype_logit_weight": float(
                            model.prototype_logit_weight
                        ),
                    }
                    if role == "type"
                    else {}
                ),
            },
            preprocess_config=asdict(preprocess_config),
            split_hash=split_hash,
            training_config=config,
            decision_config=decision_config,
        )

    metrics = {
        "schema": "five_class_role_metrics_v1",
        "role": role,
        "best_epoch": stopping.best_epoch,
        "best_score": stopping.best_score,
        "epochs_completed": last_epoch,
        "validation": validation,
        "test": test,
        "split_hash": split_hash,
        "training_config_hash": config_hash,
    }
    _atomic_write_json(role_output / "metrics.json", metrics)
    return metrics


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train requested roles and create a bundle once all five exist."""
    _validate_args(args)
    output = Path(args.output)
    roles = TRAINING_ROLES if args.stage == "all" else (args.stage,)
    existing = [output / role / "model.pt" for role in roles]
    if (
        not args.resume
        and not args.overwrite
        and any(path.exists() for path in existing)
    ):
        raise ValueError(
            "selected output already contains checkpoints; use --resume, "
            "--overwrite, or a new --output"
        )
    output.mkdir(parents=True, exist_ok=True)
    table, diagnostics = build_piece_table(
        args.task_data, require_distance=True
    )
    assignment = assign_piece_splits(table, seed=args.seed)
    validate_piece_split(table, assignment)
    artifact = split_artifact(table, assignment)
    _atomic_write_split(output / "split.json", artifact)
    split_positions = {
        split: assignment.positions(split)
        for split in ("train", "validation", "test")
    }
    preprocess_config = PreprocessConfig(
        local_center_mode="energy_envelope_v2"
    )
    results: dict[str, Any] = {}
    for role in roles:
        results[role] = _fit_role(
            args,
            output,
            role,
            table,
            artifact,
            split_positions,
            preprocess_config,
        )
    model_paths = {
        role: output / role / "model.pt" for role in TRAINING_ROLES
    }
    if all(path.is_file() for path in model_paths.values()):
        save_model_bundle(
            output,
            model_paths,
            preprocess_config=asdict(preprocess_config),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = load_model_bundle(output, device)
        test_dataset = FiveClassDataset(
            table,
            split_positions["test"],
            split="test",
            preprocess_config=preprocess_config,
        )
        try:
            bundle_metrics = evaluate_model_bundle(
                bundle,
                _loader(
                    test_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                ),
                device,
            )
        finally:
            test_dataset.close()
        _atomic_write_json(output / "bundle_metrics.json", bundle_metrics)
    else:
        bundle_metrics = None
    return {"data": diagnostics, "roles": results, "bundle": bundle_metrics}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    return run_training(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
