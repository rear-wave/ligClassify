"""End-to-end training orchestration for conditional lightning experts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.distance_sampling import ConditionBalancedSampler, JointConditionSampler
from data.group_split import group_stratified_split, validate_group_split
from data.split_artifacts import build_data_audit, make_split_manifest, write_json
from data.training_dataset import LightningPieceDataset, collate_training_batch
from data.training_manifest import build_manifest, build_piece_manifest
from distance_ordinal import (
    decode_distance_distribution,
    fit_interval_temperature_grid,
)
from evaluation import (
    checkpoint_selection_key,
    distance_calibration_is_safe,
    evaluate_predictions,
    evaluate_release,
    file_bootstrap_metrics,
)
from models import create_mtl_model
from open_set import decode_with_rejection, fit_feature_reference, fit_rejection_policy
from training_engine import conditional_joint_train_step, distance_weight_for_epoch


LOGGER = logging.getLogger(__name__)
TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def _joint_checkpoint_improved(
    stage: str,
    score: tuple[float, ...],
    best_score: tuple[float, ...] | None,
    best_epoch: int,
) -> bool:
    """Select improvements only after entering the joint training stage."""
    return stage == "joint" and (
        best_epoch <= 0 or best_score is None or score > best_score
    )


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loader(dataset, batch_size, sampler=None, shuffle=False, workers=0, cuda=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collate_training_batch,
        pin_memory=cuda,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, checkpoint.get("model_state_dict", checkpoint)


def _warm_start(model, path):
    _, source = _load_checkpoint_state(path)
    target = model.state_dict()
    prefixes = ("local_branch.", "global_branch.", "fusion.", "type_head.")
    copied = {
        name: value
        for name, value in source.items()
        if name.startswith(prefixes) and name in target and value.shape == target[name].shape
    }
    if not copied:
        raise ValueError("initial checkpoint has no compatible conditional encoder weights")
    target.update(copied)
    model.load_state_dict(target)
    return sorted(copied)


@torch.no_grad()
def collect_prediction_bundle(model, loader, device, split_hash, fold_index=None):
    """Collect records and calibration tensors from one bounded loader."""
    model.eval()
    records = []
    logits_parts, feature_parts, label_parts, quality_parts = [], [], [], []
    distance_parts, low_parts, high_parts, file_id_parts = [], [], [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        features, type_logits, distance_logits, _ = model.forward_with_features(
            batch["local"], batch["global_view"], batch["context"]
        )
        probabilities = type_logits.softmax(dim=1)
        predicted = probabilities.argmax(dim=1)
        stacked_distance = torch.stack(distance_logits, dim=1)
        rows = torch.arange(len(predicted), device=device)
        routed_logits = stacked_distance[rows, predicted]
        oracle_logits = stacked_distance[rows, batch["type_label"]]
        routed = decode_distance_distribution(routed_logits)
        oracle = decode_distance_distribution(oracle_logits)

        logits_parts.append(type_logits.cpu())
        feature_parts.append(features.cpu())
        label_parts.append(batch["type_label"].cpu())
        quality_parts.append(batch["quality"].cpu())
        distance_parts.append(stacked_distance.cpu())
        low_parts.append(batch["distance_low_km"].cpu())
        high_parts.append(batch["distance_high_km"].cpu())
        file_id_parts.append(batch["file_id"].cpu())
        for row in range(len(predicted)):
            source_path = str(batch["source_path"][row])
            records.append({
                "file_id": source_path,
                "source_path": source_path,
                "piece_index": int(batch["piece_index"][row]),
                "piece_key": str(batch["piece_key"][row]),
                "fold": None if fold_index is None else int(fold_index),
                "true_type": int(batch["type_label"][row].item()),
                "predicted_type": int(predicted[row].item()),
                "accepted": True,
                "distance_low_km": int(batch["distance_low_km"][row].item()),
                "distance_high_km": int(batch["distance_high_km"][row].item()),
                "predicted_distance_km": float(routed["expected_km"][row].item()),
                "oracle_distance_km": float(oracle["expected_km"][row].item()),
                "daylight": bool(batch["context"][row, 0].item() >= 0.5),
                "quality": batch["quality"][row].detach().cpu().tolist(),
                "split_hash": split_hash,
            })
    return {
        "records": records,
        "logits": torch.cat(logits_parts),
        "features": torch.cat(feature_parts),
        "labels": torch.cat(label_parts),
        "quality": torch.cat(quality_parts),
        "distance_logits": torch.cat(distance_parts),
        "distance_low_km": torch.cat(low_parts),
        "distance_high_km": torch.cat(high_parts),
        "file_ids": torch.cat(file_id_parts),
    }


@torch.no_grad()
def collect_reference_features(model, loader, device):
    """Collect a condition-balanced training subset for feature references."""
    model.eval()
    feature_parts, label_parts = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        feature_parts.append(
            model.extract_type_features(batch["local"], batch["global_view"]).cpu()
        )
        label_parts.append(batch["type_label"].cpu())
    return torch.cat(feature_parts), torch.cat(label_parts)


def apply_rejection_policy(bundle, policy):
    """Attach validation-calibrated IC decisions to prediction records."""
    decoded = decode_with_rejection(
        bundle["logits"],
        bundle["features"],
        policy,
        quality=bundle["quality"],
    )
    for row, record in enumerate(bundle["records"]):
        record["accepted"] = bool(decoded["accepted"][row].item())
        record["rejection_reason"] = decoded["reason"][row]
        record["type_confidence"] = float(decoded["confidence"][row].item())
        record["type_margin"] = float(decoded["margin"][row].item())
        record["feature_distance"] = float(decoded["feature_distance"][row].item())
        record["quality_score"] = float(decoded["quality_score"][row].item())
    return decoded


def fit_distance_temperatures(bundle):
    """Fit one interval-likelihood temperature per true-type expert."""
    temperatures = []
    valid = (
        (bundle["distance_low_km"] >= 0)
        & (bundle["distance_high_km"] > bundle["distance_low_km"])
    )
    rows = torch.arange(len(bundle["labels"]))
    oracle_logits = bundle["distance_logits"][rows, bundle["labels"]]
    for type_index in range(4):
        selected = valid & (bundle["labels"] == type_index)
        if not selected.any():
            raise ValueError(
                f"validation split has no distance interval for type {type_index}"
            )
        temperatures.append(fit_interval_temperature_grid(
            oracle_logits[selected],
            bundle["distance_low_km"][selected],
            bundle["distance_high_km"][selected],
        ))
    return temperatures


def apply_distance_temperatures(bundle, temperatures):
    """Re-decode routed and oracle distances with validation temperatures."""
    temperatures = torch.as_tensor(temperatures, dtype=torch.float32)
    labels = bundle["labels"]
    predicted = torch.as_tensor(
        [record["predicted_type"] for record in bundle["records"]],
        dtype=torch.long,
    )
    rows = torch.arange(len(labels))
    routed_logits = bundle["distance_logits"][rows, predicted]
    oracle_logits = bundle["distance_logits"][rows, labels]
    routed = decode_distance_distribution(
        routed_logits / temperatures[predicted].unsqueeze(1)
    )
    oracle = decode_distance_distribution(
        oracle_logits / temperatures[labels].unsqueeze(1)
    )
    for row, record in enumerate(bundle["records"]):
        record["predicted_distance_km"] = float(routed["expected_km"][row].item())
        record["oracle_distance_km"] = float(oracle["expected_km"][row].item())


def _log_metrics(prefix, metrics):
    LOGGER.info(
        "%s type_piece=%.4f type_file=%.4f distance_w200=%.4f "
        "distance_file_w200=%.4f interval_mae=%.0fkm",
        prefix,
        metrics["type_piece_accuracy"],
        metrics["type_file_macro_accuracy"],
        metrics["distance_100km_interval_within_200"],
        metrics["distance_file_macro_within_200"],
        metrics["distance_interval_mae_km"],
    )


def _evaluate_candidate_release(args, metrics, calibration_error):
    """Apply run prerequisites before the candidate-only absolute release gate."""
    if args.time_context != "daylight":
        return False, ["time context must be daylight for promotion"]
    if args.skip_test:
        return False, ["locked test was skipped"]
    if calibration_error is not None:
        return False, [f"IC rejection calibration failed: {calibration_error}"]
    return evaluate_release(metrics)


def run_conditional_training(args, device):
    """Train, select on validation, evaluate locked test, and save candidate."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.init_model:
        raise ValueError("--resume and --init_model are mutually exclusive")

    file_manifest, diagnostics = build_manifest(args.task_data, TYPE_NAMES)
    if not file_manifest:
        raise RuntimeError(f"No valid trusted-type .lig files found under {args.task_data}")
    file_splits = group_stratified_split(
        file_manifest,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    validate_group_split(file_splits)
    split_manifest = make_split_manifest(
        file_splits, args.task_data, args.seed, args.val_fraction, args.test_fraction
    )
    data_audit = build_data_audit(file_splits, TYPE_NAMES)
    data_audit["manifest_diagnostics"] = diagnostics
    data_audit["split_hashes"] = split_manifest["split_hashes"]
    write_json(output / "split_manifest.json", split_manifest)
    write_json(output / "data_audit.json", data_audit)
    LOGGER.info(
        "Manifest %d valid files; split train=%d val=%d test=%d; no shared files",
        len(file_manifest),
        len(file_splits["train"]),
        len(file_splits["val"]),
        len(file_splits["test"]),
    )

    piece_splits = {
        name: build_piece_manifest(entries) for name, entries in file_splits.items()
    }
    datasets = {
        name: LightningPieceDataset(
            entries,
            split=name,
            data_root=args.task_data,
            time_context_mode=args.time_context,
        )
        for name, entries in piece_splits.items()
    }
    train_set, val_set, test_set = (
        datasets["train"], datasets["val"], datasets["test"]
    )
    if not len(val_set) or (not args.skip_test and not len(test_set)):
        raise RuntimeError("file-isolated validation/test split is empty")

    train_sampler = JointConditionSampler(
        train_set.type_labels,
        train_set.daylight,
        train_set.distance_low_km,
        train_set.distance_high_km,
        train_set.file_ids,
        args.samples_per_epoch,
        max_samples_per_file=args.max_samples_per_file,
        seed=args.seed,
    )
    cuda = device == "cuda"
    train_loader = _loader(
        train_set,
        args.batch_size,
        train_sampler,
        workers=args.num_workers,
        cuda=cuda,
    )
    val_loader = _loader(
        val_set, args.batch_size, workers=args.num_workers, cuda=cuda
    )
    test_loader = None if args.skip_test else _loader(
        test_set, args.batch_size, workers=args.num_workers, cuda=cuda
    )

    model = create_mtl_model(
        base_channels=args.base,
        architecture="conditional_expert_v1",
        num_types=4,
        context_dim=1 if args.time_context == "daylight" else 3,
        dist_mlp_dim=args.dist_mlp_dim,
        dist_dropout=args.dist_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = cuda and not args.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        # Compatibility with PyTorch releases that predate torch.amp.GradScaler.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    type_criterion = nn.CrossEntropyLoss()
    start_epoch, best_score, best_state, best_epoch, wait = 0, None, None, 0, 0
    initialized_layers = []
    if args.resume:
        resume, state = _load_checkpoint_state(args.resume)
        model.load_state_dict(state)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        if "scaler_state_dict" in resume:
            scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = tuple(resume.get("best_score", ())) or None
        best_state = resume.get("best_model_state_dict")
        best_epoch = int(resume.get("best_epoch", 0))
        wait = int(resume.get("early_stop_wait", 0))
        LOGGER.info("Resumed exact training state from %s at epoch %d", args.resume, start_epoch + 1)
    elif args.init_model and not args.no_init:
        initialized_layers = _warm_start(model, args.init_model)
        LOGGER.info("Warm-started %d encoder/type tensors from %s", len(initialized_layers), args.init_model)
    else:
        LOGGER.info("Training conditional expert model from random initialization")

    metadata = {
        "task_schema": "four_class_rejection_v2",
        "model_name": "conditional_expert_v1",
        "model_version": "conditional_expert_v1",
        "type_names": list(TYPE_NAMES),
        "rejected_type_name": "IC",
        "base_channels": args.base,
        "dist_mlp_dim": args.dist_mlp_dim,
        "dist_dropout": args.dist_dropout,
        "context_dim": 1 if args.time_context == "daylight" else 3,
        "time_context": args.time_context,
        "distance_bins": 30,
        "coarse_distance_bins": [0, 300, 600, 1200, 1700, 2400, 3000],
        "preprocessing": {
            "name": "signed_multiscale_v1",
            "target_length": 8000,
            "use_filter": True,
        },
        "split_config": {"strategy": "file_isolated_condition_v1", "files_may_overlap": False},
        "split_hashes": split_manifest["split_hashes"],
        "combined_split_hash": split_manifest["combined_split_hash"],
        "initial_model": None if args.no_init else (args.init_model or None),
        "initialized_layers": initialized_layers,
    }

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        stage, distance_loss_weight = distance_weight_for_epoch(
            epoch,
            args.type_focus_epochs,
            args.type_focus_distance_weight,
            args.joint_distance_weight,
        )
        totals = {"type_loss": 0.0, "distance_loss": 0.0, "type_correct": 0, "type_count": 0, "distance_count": 0}
        for batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} ({stage})",
            leave=False,
        ):
            batch = _move_batch(batch, device)
            result = conditional_joint_train_step(
                model,
                batch,
                optimizer,
                type_criterion,
                distance_loss_weight=distance_loss_weight,
                coarse_weight=args.lambda_coarse,
                scaler=scaler,
                amp=not args.no_amp,
            )
            for key in totals:
                totals[key] += result[key]
        scheduler.step()
        validation_bundle = collect_prediction_bundle(
            model, val_loader, device, split_manifest["split_hashes"]["val"]
        )
        validation_metrics = evaluate_predictions(validation_bundle["records"])
        score = checkpoint_selection_key(validation_metrics)
        _log_metrics(f"Epoch {epoch + 1:3d} validation", validation_metrics)
        if _joint_checkpoint_improved(stage, score, best_score, best_epoch):
            best_score = score
            best_state = _clone_state(model)
            best_epoch = epoch + 1
            wait = 0
            torch.save(best_state, output / "best.pt")
            torch.save(
                {
                    **metadata,
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                },
                output / "best_checkpoint.pt",
            )
        elif stage == "joint":
            wait += 1
        torch.save({
            **metadata,
            "epoch": epoch,
            "model_state_dict": _clone_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score,
            "best_model_state_dict": best_state,
            "best_epoch": best_epoch,
            "early_stop_wait": wait,
        }, output / "latest.pt")
        if stage == "joint" and wait >= args.patience:
            LOGGER.info("Early stopping after epoch %d (patience=%d)", epoch + 1, args.patience)
            break

    if best_state is None:
        raise RuntimeError("training completed without a selectable checkpoint")
    model.load_state_dict(best_state)
    validation_bundle = collect_prediction_bundle(
        model, val_loader, device, split_manifest["split_hashes"]["val"]
    )
    uncalibrated_distance_metrics = evaluate_predictions(
        validation_bundle["records"]
    )
    fitted_distance_temperatures = fit_distance_temperatures(validation_bundle)
    apply_distance_temperatures(
        validation_bundle, fitted_distance_temperatures
    )
    calibrated_distance_metrics = evaluate_predictions(
        validation_bundle["records"]
    )
    distance_calibration_safe = distance_calibration_is_safe(
        uncalibrated_distance_metrics, calibrated_distance_metrics
    )
    if distance_calibration_safe:
        distance_temperatures = fitted_distance_temperatures
    else:
        distance_temperatures = [1.0] * len(TYPE_NAMES)
        apply_distance_temperatures(validation_bundle, distance_temperatures)
        LOGGER.warning(
            "Distance temperature calibration was discarded because validation "
            "point metrics regressed"
        )
    metadata["distance_calibration"] = {
        "method": "interval_nll_temperature_guarded_v2",
        "temperatures": distance_temperatures,
        "fitted_temperatures": fitted_distance_temperatures,
        "point_metric_guard_passed": distance_calibration_safe,
        "calibration_split_hash": split_manifest["split_hashes"]["val"],
    }
    reference_sampler = ConditionBalancedSampler(
        train_set.type_labels,
        train_set.daylight,
        train_set.distance_low_km,
        train_set.distance_high_km,
        train_set.file_ids,
        min(args.calibration_samples, len(train_set)),
        max_samples_per_file=min(args.max_samples_per_file, 64),
        seed=args.seed + 991,
        distance_only=False,
    )
    reference_loader = _loader(
        train_set,
        args.batch_size,
        reference_sampler,
        workers=args.num_workers,
        cuda=cuda,
    )
    reference_features, reference_labels = collect_reference_features(
        model, reference_loader, device
    )
    feature_reference = fit_feature_reference(reference_features, reference_labels)
    policy = None
    calibration_error = None
    try:
        policy = fit_rejection_policy(
            validation_bundle["logits"],
            validation_bundle["features"],
            validation_bundle["labels"],
            feature_reference,
            quality=validation_bundle["quality"],
            target_precision=args.rejection_target_precision,
            min_coverage=args.rejection_min_coverage,
            calibration_split_hash=split_manifest["split_hashes"]["val"],
            groups=validation_bundle["file_ids"],
        )
        apply_rejection_policy(validation_bundle, policy)
        metadata["type_rejection"] = policy
    except ValueError as exc:
        calibration_error = str(exc)
        metadata["type_rejection_error"] = calibration_error
        LOGGER.warning("IC rejection calibration failed: %s", calibration_error)
    validation_metrics = evaluate_predictions(validation_bundle["records"])
    metadata["validation_metrics"] = validation_metrics
    evaluated_split = "val" if args.skip_test else "test"
    evaluation_loader = val_loader if args.skip_test else test_loader
    evaluation_bundle = collect_prediction_bundle(
        model,
        evaluation_loader,
        device,
        split_manifest["split_hashes"][evaluated_split],
    )
    apply_distance_temperatures(evaluation_bundle, distance_temperatures)
    if policy is not None:
        apply_rejection_policy(evaluation_bundle, policy)
    metrics = evaluate_predictions(evaluation_bundle["records"])
    metrics.update(file_bootstrap_metrics(
        evaluation_bundle["records"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    ))
    _log_metrics(evaluated_split.title(), metrics)

    passed, reasons = _evaluate_candidate_release(
        args,
        metrics,
        calibration_error,
    )
    checkpoint = {
        **metadata,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output / "candidate.pt")
    report = {"release_passed": passed, "release_reasons": reasons, "evaluated_split": evaluated_split, **metrics}
    write_json(output / "candidate_metrics.json", report)
    if passed:
        torch.save(checkpoint, output / "model.pt")
        write_json(output / "metrics.json", report)
        LOGGER.info("PROMOTED candidate to %s", output / "model.pt")
    else:
        LOGGER.warning("REJECTED candidate; deployed model was not changed")
        for reason in reasons:
            LOGGER.warning("  release gate: %s", reason)
    for dataset in datasets.values():
        dataset.close()
    return report
