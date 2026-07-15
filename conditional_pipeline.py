"""End-to-end training orchestration for conditional lightning experts."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.training_dataset import collate_training_batch
from distance_ordinal import (
    decode_distance_distribution,
    fit_interval_temperature_grid,
)
from evaluation import (
    checkpoint_selection_key,
    evaluate_release,
)
from open_set import decode_with_rejection
from training_engine import conditional_joint_train_step, distance_weight_for_epoch


LOGGER = logging.getLogger(__name__)
TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def select_final_reference_positions(dataset, limit=20000):
    """Select unique pieces by type/day/interval/file/piece round-robin."""
    limit = int(limit)
    if limit <= 0:
        raise ValueError("final feature reference limit must be positive")
    fields = (
        dataset.type_labels,
        dataset.daylight,
        dataset.distance_low_km,
        dataset.distance_high_km,
        dataset.source_paths,
        dataset.piece_indices,
    )
    if any(len(field) != len(dataset) for field in fields):
        raise ValueError("final feature reference metadata is not aligned")
    hierarchy = {}
    for position in range(len(dataset)):
        type_index = int(dataset.type_labels[position])
        daylight = int(dataset.daylight[position])
        interval = (
            int(dataset.distance_low_km[position]),
            int(dataset.distance_high_km[position]),
        )
        source_path = str(dataset.source_paths[position])
        piece_index = int(dataset.piece_indices[position])
        hierarchy.setdefault(type_index, {}).setdefault(daylight, {}).setdefault(
            interval, {}
        ).setdefault(source_path, []).append((piece_index, position))

    def traverse(node):
        if isinstance(node, list):
            for _, position in sorted(node):
                yield position
            return
        active = [iter(traverse(node[key])) for key in sorted(node)]
        while active:
            remaining = []
            for iterator in active:
                try:
                    yield next(iterator)
                    remaining.append(iterator)
                except StopIteration:
                    pass
            active = remaining

    selected = []
    for position in traverse(hierarchy):
        selected.append(int(position))
        if len(selected) >= min(limit, len(dataset)):
            break
    return selected


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


def run_conditional_epoch(
    model,
    loader,
    sampler,
    optimizer,
    type_criterion,
    scaler,
    device,
    epoch,
    type_focus_epochs,
    type_focus_distance_weight,
    joint_distance_weight,
    lambda_coarse=0.5,
    no_amp=False,
):
    """Run one deterministic staged joint-training epoch."""
    sampler.set_epoch(epoch)
    stage, distance_loss_weight = distance_weight_for_epoch(
        epoch,
        type_focus_epochs,
        type_focus_distance_weight,
        joint_distance_weight,
    )
    totals = {
        "type_loss": 0.0,
        "distance_loss": 0.0,
        "type_correct": 0,
        "type_count": 0,
        "distance_count": 0,
    }
    for batch in tqdm(
        loader,
        total=len(loader),
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
            coarse_weight=lambda_coarse,
            scaler=scaler,
            amp=not no_amp,
        )
        for key in totals:
            totals[key] += result[key]
    return stage, distance_loss_weight, totals


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
