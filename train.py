"""Four-class lightning-type and routed 30-bin distance training."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm

from models import create_mtl_model
from open_set import (
    decode_with_rejection,
    fit_feature_reference,
    fit_rejection_policy,
)
from data.lig_parser import LigFileIndex
from data.distance_sampling import BalancedTypeSampler, HierarchicalDistanceSampler
from data.preprocessing import preprocess_batch
from data.training_manifest import (
    build_manifest,
    build_piece_manifest,
    piece_time_split_manifest,
    validate_piece_split_coverage,
    validate_piece_split_isolation,
)
from distance_metrics import summarize_equal_bin_distance_predictions
from distance_ordinal import (
    decode_distance_logits,
    fit_temperature_grid,
    is_meaningful_improvement,
    ordinal_distance_loss,
    select_confidence_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
RESEARCH_TYPE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]
TYPE_NAMES = list(RESEARCH_TYPE_NAMES)
DIST_NAMES = list(RESEARCH_TYPE_NAMES)
# Four research types use the same zero-based type and distance-head index.
DIST_BIN_STARTS = [i * 100 for i in range(30)]       # 0, 100, ..., 2900


def four_class_schema_metadata():
    """Return the checkpoint schema shared by training and inference."""
    return {
        "task_schema": "four_class_rejection_v1",
        "type_names": list(RESEARCH_TYPE_NAMES),
        "rejected_type_name": "IC",
    }


def make_four_class_selection_key(metrics):
    """Order checkpoints by type quality before distance quality."""
    return (
        float(metrics["type_f1"]),
        float(metrics["type_min_recall"]),
        float(metrics["type_min_precision"]),
        float(metrics["dist_equal_bin_macro_w2"]),
        -float(metrics["dist_equal_bin_macro_mae_km"]),
    )


def balanced_type_sample_count(type_labels, requested):
    """Cap an epoch to equal, non-replacement samples from all four types."""
    labels = np.asarray(type_labels)
    counts = [int(np.count_nonzero(labels == index)) for index in range(4)]
    if not all(counts):
        raise ValueError(f"four researched types are required; counts={counts}")
    maximum = min(counts) * 4
    if requested < 0:
        return maximum
    return min(int(requested) - int(requested) % 4, maximum)


# ── Soft distance label ──────────────────────────────────
def make_soft_label(k, num_classes=30, tau=1.0):
    """Soft label centered at bin k: w[j] ∝ exp(-|j-k| / tau)."""
    dists = np.abs(np.arange(num_classes) - k).astype(np.float32)
    w = np.exp(-dists / tau)
    return torch.from_numpy(w / w.sum())


# ── Dataset ──────────────────────────────────────────────
class MultiTaskDataset(Dataset):
    """Lazy waveform view over a pre-split training manifest."""

    def __init__(
        self,
        entries,
        split="train",
        lig_index=None,
        normalize_mode="minmax",
    ):
        self.split = split
        self.normalize_mode = normalize_mode
        paths = sorted({entry.filepath for entry in entries})
        self._owns_lig_index = lig_index is None
        self.lig = lig_index or LigFileIndex(paths, validate=False)
        path_to_file = {
            os.path.normcase(os.path.abspath(path)): index
            for index, path in enumerate(self.lig.filepaths)
        }

        global_indices = []
        type_labels = []
        dist_labels = []
        file_ids = []
        date_ids = []
        for entry in entries:
            key = os.path.normcase(os.path.abspath(entry.filepath))
            if key not in path_to_file:
                raise ValueError(
                    f"piece references an unindexed file: {entry.filepath}"
                )
            file_id = path_to_file[key]
            piece_count = self.lig.num_pieces_per_file[file_id]
            if not 0 <= entry.piece_index < piece_count:
                raise IndexError(
                    f"piece_index={entry.piece_index} outside {entry.filepath}"
                )
            global_indices.append(
                int(self.lig._cumsum[file_id]) + entry.piece_index
            )
            type_labels.append(entry.type_idx)
            dist_labels.append(entry.dist_bin)
            file_ids.append(file_id)
            date_ids.append(int(entry.timestamp.strftime("%Y%m%d")))

        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.type_labels = np.asarray(type_labels, dtype=np.int8)
        self.dist_labels = np.asarray(dist_labels, dtype=np.int8)
        self.file_ids = np.asarray(file_ids, dtype=np.int32)
        self.date_ids = np.asarray(date_ids, dtype=np.int32)
        n_dist = int(np.count_nonzero(self.dist_labels >= 0))
        logger.info(
            "  %s lazy view: %d pieces (%d with distance)",
            split,
            len(self.global_indices),
            n_dist,
        )

    def __len__(self):
        return len(self.global_indices)

    def _items_from_positions(self, positions):
        global_indices = [int(self.global_indices[pos]) for pos in positions]
        waveforms = np.stack(self.lig.read_pieces_batch(global_indices), axis=0)
        processed = preprocess_batch(
            waveforms,
            normalize_mode=self.normalize_mode,
        )
        items = []
        for row, pos in enumerate(positions):
            items.append((
                torch.from_numpy(processed[row].copy()).unsqueeze(0),
                torch.tensor(int(self.type_labels[pos]), dtype=torch.long),
                torch.tensor(int(self.dist_labels[pos]), dtype=torch.long),
            ))
        return items

    def __getitem__(self, index):
        return self._items_from_positions([int(index)])[0]

    def __getitems__(self, indices):
        return self._items_from_positions([int(index) for index in indices])

    def close(self):
        if getattr(self, "_owns_lig_index", False) and hasattr(self, "lig"):
            self.lig.close()

    def __del__(self):
        self.close()


# ── Collate ──────────────────────────────────────────────
def collate(batch):
    xs, ys, ds = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ds)


def alternate_stream_batches(type_loader, distance_loader):
    """Yield each loader once, alternating while both still have batches."""
    type_iterator = iter(type_loader)
    distance_iterator = iter(distance_loader)
    type_done = distance_done = False
    while not (type_done and distance_done):
        if not type_done:
            try:
                yield "type", next(type_iterator)
            except StopIteration:
                type_done = True
        if not distance_done:
            try:
                yield "distance", next(distance_iterator)
            except StopIteration:
                distance_done = True


def compute_distance_head_loss(
    distance_logits,
    type_labels,
    distance_labels,
    tau=1.0,
    lambda_emd=1.0,
    lambda_reg=0.5,
    lambda_coarse=0.5,
):
    """Macro-average ordinal distance loss across represented types."""
    losses = []
    component_values = {name: [] for name in ("soft_ce", "cdf", "huber", "coarse")}
    for lightning_type in range(4):
        mask = (type_labels == lightning_type) & (distance_labels >= 0)
        if not mask.any():
            continue
        loss, components = ordinal_distance_loss(
            distance_logits[lightning_type][mask],
            distance_labels[mask],
            tau=tau,
            lambda_emd=lambda_emd,
            lambda_reg=lambda_reg,
            lambda_coarse=lambda_coarse,
        )
        losses.append(loss)
        for name, value in components.items():
            component_values[name].append(value)

    if not losses:
        zero = distance_logits[0].sum() * 0.0
        return zero, {name: zero for name in component_values}
    return torch.stack(losses).mean(), {
        name: torch.stack(values).mean()
        for name, values in component_values.items()
    }


def train_stream_step(
    model,
    stream,
    x,
    type_labels,
    distance_labels,
    optimizer,
    type_criterion,
    distance_objective="ordinal",
    distance_loss_weight=1.0,
    distance_batch_type_weight=0.1,
    tau=1.0,
    lambda_emd=1.0,
    lambda_reg=0.5,
    lambda_coarse=0.5,
):
    """Optimize exactly one type or balanced-distance batch."""
    if stream not in {"type", "distance"}:
        raise ValueError(f"Unknown training stream: {stream}")
    model.train()
    optimizer.zero_grad()
    type_logits, distance_logits = model(x)
    type_loss = type_criterion(type_logits, type_labels)
    distance_loss = type_loss.new_zeros(())
    components = {}

    if stream == "type":
        total_loss = type_loss
        distance_count = 0
    else:
        distance_count = int((distance_labels >= 0).sum().item())
        if distance_objective == "ordinal":
            distance_loss, components = compute_distance_head_loss(
                distance_logits,
                type_labels,
                distance_labels,
                tau=tau,
                lambda_emd=lambda_emd,
                lambda_reg=lambda_reg,
                lambda_coarse=lambda_coarse,
            )
        elif distance_objective == "ce":
            losses = []
            for lightning_type in range(4):
                mask = (
                    (type_labels == lightning_type)
                    & (distance_labels >= 0)
                )
                if mask.any():
                    losses.append(F.cross_entropy(
                        distance_logits[lightning_type][mask],
                        distance_labels[mask],
                    ))
            distance_loss = (
                torch.stack(losses).mean()
                if losses else distance_logits[0].sum() * 0.0
            )
        else:
            raise ValueError(f"Unknown distance objective: {distance_objective}")
        total_loss = (
            distance_loss_weight * distance_loss
            + distance_batch_type_weight * type_loss
        )

    total_loss.backward()
    optimizer.step()
    return {
        "type_loss": float(type_loss.detach().item()),
        "distance_loss": float(distance_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "type_correct": int((type_logits.argmax(1) == type_labels).sum().item()),
        "type_count": int(len(type_labels)),
        "distance_count": distance_count,
        "components": {
            name: float(value.detach().item())
            for name, value in components.items()
        },
    }


# ── Loss helpers ─────────────────────────────────────────
# ── Training / Eval ──────────────────────────────────────
def _distance_bin_prediction(logits, mode, temperature=1.0):
    if mode == "argmax":
        return int(logits.argmax().item())
    if mode == "expected":
        return int(decode_distance_logits(
            logits.unsqueeze(0), temperature=temperature
        )["bin"].item())
    raise ValueError(f"Unknown distance prediction mode: {mode}")


def route_distance_predictions(
    type_predictions,
    type_labels,
    dist_labels,
    dist_logits,
    prediction_mode="argmax",
    temperatures=None,
):
    """Route distance heads with both ground-truth and predicted type."""
    temperatures = temperatures or [1.0] * len(dist_logits)
    routed = {
        "oracle_predictions": [],
        "end_to_end_predictions": [],
        "true_distances": [],
        "true_types": [],
        "joint_correct": [],
    }
    for index in range(len(dist_labels)):
        true_distance = int(dist_labels[index].item())
        if true_distance < 0:
            continue
        true_type = int(type_labels[index].item())
        predicted_type = int(type_predictions[index].item())
        oracle_prediction = _distance_bin_prediction(
            dist_logits[true_type][index],
            prediction_mode,
            temperatures[true_type],
        )
        if 0 <= predicted_type < len(dist_logits):
            end_to_end_prediction = _distance_bin_prediction(
                dist_logits[predicted_type][index],
                prediction_mode,
                temperatures[predicted_type],
            )
        else:
            end_to_end_prediction = -1

        routed["oracle_predictions"].append(oracle_prediction)
        routed["end_to_end_predictions"].append(end_to_end_prediction)
        routed["true_distances"].append(true_distance)
        routed["true_types"].append(true_type)
        routed["joint_correct"].append(
            predicted_type == true_type and end_to_end_prediction == true_distance
        )
    return routed


def summarize_distance_predictions(predictions, targets):
    """Summarize ordered errors, counting uncovered predictions as failures."""
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("predictions and targets must be aligned vectors")
    if not len(targets):
        return {
            "coverage": 0.0,
            "acc": 0.0,
            "mae_bin": 0.0,
            "mae_km": 0.0,
            "w1": 0.0,
            "w2": 0.0,
        }
    covered = predictions >= 0
    errors = np.zeros(len(targets), dtype=np.float64)
    errors[covered] = np.abs(predictions[covered] - targets[covered])
    return {
        "coverage": float(covered.mean()),
        "acc": float(np.mean(covered & (errors == 0))),
        "mae_bin": float(errors[covered].mean()) if covered.any() else 0.0,
        "mae_km": float(errors[covered].mean() * 100) if covered.any() else 0.0,
        "w1": float(np.mean(covered & (errors <= 1))),
        "w2": float(np.mean(covered & (errors <= 2))),
    }


def group_bootstrap_distance_metrics(
    predictions,
    targets,
    group_ids,
    repetitions=1000,
    seed=42,
):
    """Bootstrap distance metrics by acquisition file, not by correlated piece."""
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    group_ids = np.asarray(group_ids)
    if not (
        predictions.shape == targets.shape == group_ids.shape
        and predictions.ndim == 1
    ):
        raise ValueError("predictions, targets, and group_ids must be aligned vectors")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    unique_groups = np.unique(group_ids)
    if not len(unique_groups):
        return {"mae_km_ci95": [0.0, 0.0], "w2_ci95": [0.0, 0.0]}

    positions = {
        group: np.flatnonzero(group_ids == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    mae_values, w2_values = [], []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled_positions = np.concatenate([
            positions[group] for group in sampled_groups
        ])
        summary = summarize_distance_predictions(
            predictions[sampled_positions], targets[sampled_positions]
        )
        mae_values.append(summary["mae_km"])
        w2_values.append(summary["w2"])
    return {
        "mae_km_ci95": [
            float(value) for value in np.percentile(mae_values, [2.5, 97.5])
        ],
        "w2_ci95": [
            float(value) for value in np.percentile(w2_values, [2.5, 97.5])
        ],
    }


def compute_split_hash(entries, split_name):
    """Return a stable hash of selected piece identities and labels."""
    rows = [
        "|".join([
            split_name,
            os.path.normcase(os.path.abspath(entry.filepath)),
            str(entry.piece_index),
            str(entry.type_idx),
            str(entry.dist_bin),
            entry.timestamp.isoformat(),
        ])
        for entry in entries
    ]
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def count_cross_split_files(splits):
    """Count source files contributing pieces to more than one split."""
    memberships = {}
    for split_name, entries in splits.items():
        for entry in entries:
            path = os.path.normcase(os.path.abspath(entry.filepath))
            memberships.setdefault(path, set()).add(split_name)
    return sum(len(names) > 1 for names in memberships.values())


def fit_distance_calibration(logits_by_head, targets_by_head, target_w2=0.8):
    """Fit per-head temperatures and one validation-only confidence cutoff."""
    if len(logits_by_head) != 4 or len(targets_by_head) != 4:
        raise ValueError("distance calibration requires four heads")
    temperatures = []
    confidence_parts = []
    error_parts = []
    for logits, targets in zip(logits_by_head, targets_by_head):
        if len(targets) == 0:
            temperatures.append(1.0)
            continue
        temperature = fit_temperature_grid(logits, targets)
        temperatures.append(temperature)
        decoded = decode_distance_logits(logits, temperature=temperature)
        confidence_parts.append(decoded["confidence"].cpu())
        error_parts.append((decoded["bin"].cpu() - targets.cpu()).abs().float())

    if confidence_parts:
        threshold = select_confidence_threshold(
            torch.cat(confidence_parts),
            torch.cat(error_parts),
            target_w2=target_w2,
        )
    else:
        threshold = {"threshold": 1.0, "coverage": 0.0, "w2": 0.0}
    return {
        "temperatures": temperatures,
        "confidence_threshold": threshold["threshold"],
        "validation_coverage": threshold["coverage"],
        "validation_w2": threshold["w2"],
    }


@torch.no_grad()
def collect_type_outputs(model, loader, device):
    """Collect type logits, pooled features, and labels from one loader."""
    model.eval()
    logits_parts = []
    feature_parts = []
    label_parts = []
    for x, labels, _ in tqdm(loader, desc="Type calibration", leave=False):
        x = x.to(device)
        features = model.extract_type_features(x)
        logits_parts.append(model.type_head(features).cpu())
        feature_parts.append(features.cpu())
        label_parts.append(labels.cpu())
    return (
        torch.cat(logits_parts),
        torch.cat(feature_parts),
        torch.cat(label_parts),
    )


@torch.no_grad()
def collect_distance_outputs(model, loader, device):
    """Collect validation logits routed by true type for calibration."""
    model.eval()
    logits_by_head = [[] for _ in range(4)]
    targets_by_head = [[] for _ in range(4)]
    for x, type_labels, distance_labels in loader:
        x = x.to(device)
        type_labels = type_labels.to(device)
        distance_labels = distance_labels.to(device)
        _, distance_logits = model(x)
        for head in range(4):
            mask = (type_labels == head) & (distance_labels >= 0)
            if mask.any():
                logits_by_head[head].append(distance_logits[head][mask].cpu())
                targets_by_head[head].append(distance_labels[mask].cpu())
    empty_logits = torch.empty((0, 30), dtype=torch.float32)
    empty_targets = torch.empty((0,), dtype=torch.long)
    return (
        [torch.cat(items) if items else empty_logits.clone() for items in logits_by_head],
        [torch.cat(items) if items else empty_targets.clone() for items in targets_by_head],
    )


@torch.no_grad()
def evaluate(model, loader, type_crit, dist_crit, dev,
             use_soft=False, soft_tau=1.0, prediction_mode="argmax",
             temperatures=None, bootstrap_repetitions=0, bootstrap_seed=42):
    """Evaluate: type metrics + per-class distance metrics."""
    model.eval()
    all_type_preds, all_type_labels = [], []
    # Per-distance-head: collect (pred, true)
    dist_data = {head: {"preds": [], "trues": []} for head in range(4)}
    end_to_end_preds, end_to_end_trues, joint_correct = [], [], []
    oracle_all_preds, oracle_all_trues = [], []

    for x, y, d in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(dev), y.to(dev)
        type_logits, dist_logits = model(x)

        type_preds = type_logits.argmax(1)
        all_type_preds.extend(type_preds.cpu().numpy())
        all_type_labels.extend(y.cpu().numpy())

        routed = route_distance_predictions(
            type_preds,
            y,
            d,
            dist_logits,
            prediction_mode=prediction_mode,
            temperatures=temperatures,
        )
        for pred, true, true_type in zip(
            routed["oracle_predictions"],
            routed["true_distances"],
            routed["true_types"],
        ):
            head_idx = true_type
            dist_data[head_idx]["preds"].append(pred)
            dist_data[head_idx]["trues"].append(true)
        end_to_end_preds.extend(routed["end_to_end_predictions"])
        end_to_end_trues.extend(routed["true_distances"])
        joint_correct.extend(routed["joint_correct"])
        oracle_all_preds.extend(routed["oracle_predictions"])
        oracle_all_trues.extend(routed["true_distances"])

    # ── Type metrics ──
    ap = np.array(all_type_preds)
    al = np.array(all_type_labels)
    type_acc = (ap == al).mean()

    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    type_f1 = f1_score(al, ap, average='macro', zero_division=0)
    type_precision, type_recall, type_per_class_f1, _ = (
        precision_recall_fscore_support(
            al,
            ap,
            labels=list(range(len(TYPE_NAMES))),
            zero_division=0,
        )
    )

    # ── Distance metrics per head ──
    metrics = {
        "type_acc": type_acc,
        "type_f1": type_f1,
        "type_precision": type_precision.tolist(),
        "type_recall": type_recall.tolist(),
        "type_per_class_f1": type_per_class_f1.tolist(),
        "type_min_precision": float(type_precision.min()),
        "type_min_recall": float(type_recall.min()),
        "type_confusion_matrix": confusion_matrix(
            al, ap, labels=list(range(len(TYPE_NAMES)))
        ).tolist(),
    }
    if (
        hasattr(loader, "dataset")
        and hasattr(loader.dataset, "date_ids")
        and len(loader.dataset.date_ids) == len(ap)
    ):
        metrics["type_metrics_by_year"] = summarize_type_metrics_by_year(
            ap, al, loader.dataset.date_ids
        )
    total_dist_ok, total_dist_n, total_mae_bin, total_mae_km = 0, 0, 0, 0
    total_w1, total_w2, total_w1n, total_w2n = 0, 0, 0, 0
    per_type_w2, per_type_mae = [], []
    per_type_equal_bin_w2, per_type_equal_bin_mae = [], []

    for hi, name in enumerate(DIST_NAMES):
        preds = np.array(dist_data[hi]["preds"])
        trues = np.array(dist_data[hi]["trues"])
        n = len(preds)
        key = f"dist_{name}"
        if n == 0:
            metrics[f"{key}_acc"], metrics[f"{key}_mae_km"] = 0.0, 0.0
            metrics[f"{key}_w1"], metrics[f"{key}_w2"] = 0.0, 0.0
            per_type_w2.append(0.0)
            per_type_mae.append(0.0)
            per_type_equal_bin_w2.append(0.0)
            per_type_equal_bin_mae.append(0.0)
            continue

        ok = (preds == trues).sum()
        mae_bin = np.abs(preds - trues).mean()
        mae_km = mae_bin * 100
        w1 = (np.abs(preds - trues) <= 1).mean()
        w2 = (np.abs(preds - trues) <= 2).mean()

        metrics[f"{key}_acc"] = ok / n
        metrics[f"{key}_mae_bin"] = float(mae_bin)
        metrics[f"{key}_mae_km"] = float(mae_km)
        metrics[f"{key}_w1"] = float(w1)
        metrics[f"{key}_w2"] = float(w2)
        per_type_w2.append(float(w2))
        per_type_mae.append(float(mae_km))

        equal_bin = summarize_equal_bin_distance_predictions(preds, trues)
        for metric_name in ("acc", "mae_km", "w1", "w2", "bin_count"):
            metrics[f"{key}_equal_bin_{metric_name}"] = equal_bin[metric_name]
        per_type_equal_bin_w2.append(equal_bin["w2"])
        per_type_equal_bin_mae.append(equal_bin["mae_km"])

        total_dist_ok += ok; total_dist_n += n
        total_mae_bin += mae_bin * n; total_mae_km += mae_km * n
        total_w1 += w1 * n; total_w2 += w2 * n; total_w1n += n; total_w2n += n

    if total_dist_n > 0:
        metrics["dist_acc"] = total_dist_ok / total_dist_n
        metrics["dist_mae_bin"] = total_mae_bin / total_dist_n
        metrics["dist_mae_km"] = total_mae_km / total_dist_n
        metrics["dist_w1"] = total_w1 / total_w1n if total_w1n else 0
        metrics["dist_w2"] = total_w2 / total_w2n if total_w2n else 0
    else:
        metrics["dist_acc"] = metrics["dist_mae_bin"] = metrics["dist_mae_km"] = 0.0
        metrics["dist_w1"] = metrics["dist_w2"] = 0.0

    metrics["per_type_w2"] = per_type_w2
    metrics["dist_macro_w2"] = float(np.mean(per_type_w2))
    metrics["dist_min_type_w2"] = float(np.min(per_type_w2))
    metrics["dist_macro_mae_km"] = float(np.mean(per_type_mae))
    metrics["per_type_equal_bin_w2"] = per_type_equal_bin_w2
    metrics["dist_equal_bin_macro_w2"] = float(
        np.mean(per_type_equal_bin_w2)
    )
    metrics["dist_equal_bin_min_type_w2"] = float(
        np.min(per_type_equal_bin_w2)
    )
    metrics["dist_equal_bin_macro_mae_km"] = float(
        np.mean(per_type_equal_bin_mae)
    )

    e2e_preds = np.asarray(end_to_end_preds, dtype=np.int64)
    e2e_trues = np.asarray(end_to_end_trues, dtype=np.int64)
    if len(e2e_trues):
        e2e = summarize_distance_predictions(e2e_preds, e2e_trues)
        metrics["e2e_dist_acc"] = e2e["acc"]
        metrics["e2e_dist_coverage"] = e2e["coverage"]
        metrics["e2e_dist_mae_km"] = e2e["mae_km"]
        metrics["e2e_dist_w1"] = e2e["w1"]
        metrics["e2e_dist_w2"] = e2e["w2"]
        metrics["joint_acc"] = float(np.mean(joint_correct))
    else:
        metrics["e2e_dist_acc"] = 0.0
        metrics["e2e_dist_coverage"] = 0.0
        metrics["e2e_dist_mae_km"] = 0.0
        metrics["e2e_dist_w1"] = 0.0
        metrics["e2e_dist_w2"] = 0.0
        metrics["joint_acc"] = 0.0

    if bootstrap_repetitions and hasattr(loader, "dataset"):
        dataset = loader.dataset
        if hasattr(dataset, "file_ids") and hasattr(dataset, "dist_labels"):
            group_ids = np.asarray(dataset.file_ids)[
                np.asarray(dataset.dist_labels) >= 0
            ]
            if len(group_ids) == len(oracle_all_preds):
                metrics["dist_group_bootstrap"] = group_bootstrap_distance_metrics(
                    oracle_all_preds,
                    oracle_all_trues,
                    group_ids,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                )

    return metrics


# ── Main ─────────────────────────────────────────────────
def configure_cuda_backend(deterministic=False):
    """Enable fast cuDNN kernels, with an explicit reproducibility override."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def load_initial_weights(model, checkpoint_path):
    """Warm-start the shared encoder and type head from a trusted model."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    prefixes = ("stem.", "layer1.", "layer2.", "layer3.", "type_head.")
    copied = {
        name: value
        for name, value in source.items()
        if name.startswith(prefixes)
        and name in target
        and target[name].shape == value.shape
    }
    if not copied:
        raise ValueError("Initial model has no compatible encoder/type weights")
    target.update(copied)
    model.load_state_dict(target)
    return sorted(copied)


def output_paths(output_dir):
    """Return the stable filenames shared by training and classification."""
    root = Path(output_dir)
    return {
        "model": root / "model.pt",
        "metrics": root / "metrics.json",
        "candidate": root / "candidate.pt",
        "candidate_metrics": root / "candidate_metrics.json",
        "best": root / "best.pt",
        "best_checkpoint": root / "best_checkpoint.pt",
    }


def evaluate_release_gate(
    metrics,
    min_type_f1=0.85,
    min_type_precision=0.85,
    min_type_recall=0.70,
    min_type_w2=0.70,
    min_macro_w2=0.75,
):
    """Return whether locked test metrics are safe to deploy."""
    reasons = []
    type_f1 = float(metrics.get("type_f1", 0.0))
    macro_w2 = float(metrics.get("dist_equal_bin_macro_w2", 0.0))
    per_type = [
        float(value)
        for value in metrics.get("per_type_equal_bin_w2", [])
    ]
    type_precision = [
        float(value) for value in metrics.get("type_precision", [])
    ]
    type_recall = [float(value) for value in metrics.get("type_recall", [])]
    if type_f1 < min_type_f1:
        reasons.append(f"type_f1={type_f1:.4f} below {min_type_f1:.2f}")
    if macro_w2 < min_macro_w2:
        reasons.append(f"macro_w2={macro_w2:.4f} below {min_macro_w2:.2f}")
    if len(type_precision) != 4:
        reasons.append(
            f"type_precision has {len(type_precision)} values; expected 4"
        )
    else:
        for index, value in enumerate(type_precision):
            if value < min_type_precision:
                reasons.append(
                    f"type_precision[{index}]={value:.4f} below "
                    f"{min_type_precision:.2f}"
                )
    if len(type_recall) != 4:
        reasons.append(f"type_recall has {len(type_recall)} values; expected 4")
    else:
        for index, value in enumerate(type_recall):
            if value < min_type_recall:
                reasons.append(
                    f"type_recall[{index}]={value:.4f} below "
                    f"{min_type_recall:.2f}"
                )
    if len(per_type) != 4:
        reasons.append(
            f"per_type_equal_bin_w2 has {len(per_type)} values; expected 4"
        )
    else:
        for index, value in enumerate(per_type):
            if value < min_type_w2:
                reasons.append(
                    f"type_w2[{index}]={value:.4f} below {min_type_w2:.2f}"
                )
    return not reasons, reasons


def compare_baseline_metrics(candidate, baseline_path):
    """Reject promotion when four-class type metrics regress from baseline."""
    if not baseline_path:
        return False, ["baseline metrics file is required for promotion"]
    path = Path(baseline_path)
    if not path.is_file():
        return False, [f"baseline metrics file not found: {baseline_path}"]
    with path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    reasons = []
    candidate_f1 = float(candidate.get("type_f1", 0.0))
    baseline_f1 = float(baseline.get("type_f1", 0.0))
    if candidate_f1 < baseline_f1:
        reasons.append(
            f"type_f1={candidate_f1:.4f} below baseline {baseline_f1:.4f}"
        )
    for field in ("type_precision", "type_recall"):
        candidate_values = candidate.get(field, [])
        baseline_values = baseline.get(field, [])
        if len(candidate_values) != 4 or len(baseline_values) != 4:
            reasons.append(f"{field} baseline comparison requires four values")
            continue
        for index, (value, baseline_value) in enumerate(
            zip(candidate_values, baseline_values)
        ):
            if float(value) < float(baseline_value):
                reasons.append(
                    f"{field}[{index}]={float(value):.4f} below baseline "
                    f"{float(baseline_value):.4f}"
                )
    return not reasons, reasons


def summarize_rejected_types(decoded, labels, num_types=4):
    """Score accepted predictions while counting rejection as a false negative."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = torch.as_tensor(labels, dtype=torch.long).cpu().numpy()
    predicted = torch.as_tensor(decoded["predicted"], dtype=torch.long).cpu()
    accepted = torch.as_tensor(decoded["accepted"], dtype=torch.bool).cpu()
    final_predictions = predicted.clone()
    final_predictions[~accepted] = -1
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        labels,
        final_predictions.numpy(),
        labels=list(range(num_types)),
        zero_division=0,
    )
    return {
        "type_acc": float(
            np.mean(final_predictions.numpy() == labels)
        ),
        "type_f1": float(per_class_f1.mean()),
        "type_precision": precision.tolist(),
        "type_recall": recall.tolist(),
        "type_per_class_f1": per_class_f1.tolist(),
        "type_min_precision": float(precision.min()),
        "type_min_recall": float(recall.min()),
        "type_coverage": float(accepted.float().mean().item()),
    }


def summarize_type_metrics_by_year(predictions, labels, date_ids):
    """Return compact type metrics for each acquisition year."""
    from sklearn.metrics import f1_score

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    date_ids = np.asarray(date_ids)
    if predictions.shape != labels.shape or predictions.shape != date_ids.shape:
        raise ValueError("predictions, labels, and date_ids must align")
    years = date_ids // 10000
    grouped = {}
    for year in sorted(np.unique(years).tolist()):
        mask = years == year
        grouped[str(int(year))] = {
            "count": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == labels[mask])),
            "macro_f1": float(f1_score(
                labels[mask],
                predictions[mask],
                labels=list(range(4)),
                average="macro",
                zero_division=0,
            )),
        }
    return grouped


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_candidate_and_maybe_promote(
    output_dir,
    checkpoint,
    metrics,
    passed,
    reasons,
):
    """Always save the candidate; preserve deployed files after rejection."""
    paths = output_paths(output_dir)
    paths["candidate"].parent.mkdir(parents=True, exist_ok=True)
    report = {"release_passed": bool(passed), "release_reasons": list(reasons), **metrics}
    torch.save(checkpoint, paths["candidate"])
    with paths["candidate_metrics"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    if passed:
        torch.save(checkpoint, paths["model"])
        with paths["metrics"].open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
    return paths


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task_data", default="../train_data")
    p.add_argument("--output", default="./weights/four_class")
    p.add_argument(
        "--init_model",
        default="",
        help="Trusted checkpoint used to initialize the encoder and type head; empty disables",
    )
    p.add_argument(
        "--no_init",
        action="store_true",
        help="Train all model parameters from random initialization",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--wd", type=float, default=0.0005)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Middle piece fraction per type/distance group")
    p.add_argument("--test_fraction", type=float, default=0.15,
                   help="Latest piece fraction per type/distance group")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--lambda_dist", type=float, default=1.0,
                   help="Weight of distance loss")
    p.add_argument("--use_soft_distance_label", action="store_true",
                   help="Use ordinal soft-label CE for distance")
    p.add_argument("--distance_soft_tau", type=float, default=1.0,
                   help="Tau for soft distance label")
    p.add_argument("--model_arch", choices=["mtl_resnet", "ordinal_v2"],
                   default="ordinal_v2")
    p.add_argument("--distance_batch_size", type=int, default=128)
    p.add_argument("--type_samples_per_epoch", type=int, default=180000)
    p.add_argument("--distance_samples_per_epoch", type=int, default=60000)
    p.add_argument("--max_distance_samples_per_file", type=int, default=256)
    p.add_argument("--dist_mlp_dim", type=int, default=128)
    p.add_argument("--dist_dropout", type=float, default=0.2)
    p.add_argument("--lambda_emd", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=0.5)
    p.add_argument("--lambda_coarse", type=float, default=0.5)
    p.add_argument("--distance_batch_type_weight", type=float, default=0.1)
    p.add_argument("--distance_sampling",
                   choices=["uniform", "hierarchical"],
                   default="hierarchical")
    p.add_argument("--distance_objective", choices=["ce", "ordinal"],
                   default="ordinal")
    p.add_argument("--distance_prediction", choices=["argmax", "expected"],
                   default="expected")
    p.add_argument("--skip_test", action="store_true",
                   help="Do not evaluate the locked piece-level test split")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic cuDNN kernels instead of fast benchmark mode")
    p.add_argument("--min_eval_pieces", type=int, default=500)
    p.add_argument("--min_type_f1", type=float, default=0.85)
    p.add_argument("--min_type_precision", type=float, default=0.85)
    p.add_argument("--min_type_recall", type=float, default=0.70)
    p.add_argument("--min_test_type_w2", type=float, default=0.70)
    p.add_argument("--min_test_macro_w2", type=float, default=0.75)
    p.add_argument(
        "--baseline_metrics",
        default="",
        help="Four-class baseline metrics required for automatic promotion",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        configure_cuda_backend(args.deterministic)

    logger.info(f"lambda_dist={args.lambda_dist}, soft_label={args.use_soft_distance_label}, "
                f"soft_tau={args.distance_soft_tau}")

    # Split pieces chronologically inside every type/distance group.
    file_manifest, diagnostics = build_manifest(args.task_data, TYPE_NAMES)
    if not file_manifest:
        raise RuntimeError(f"No valid .lig files found under {args.task_data}")
    logger.info(
        "Manifest: %d/%d valid files, %d invalid, %d timestamp errors, "
        "%d filename fallbacks, %d distance-labelled",
        diagnostics["valid_files"],
        diagnostics["discovered_files"],
        diagnostics["invalid_files"],
        diagnostics["timestamp_errors"],
        diagnostics["filename_timestamp_fallbacks"],
        diagnostics["distance_labeled_files"],
    )
    piece_manifest = build_piece_manifest(file_manifest)
    logger.info("Piece manifest: %d timestamped pieces", len(piece_manifest))
    split_entries = piece_time_split_manifest(
        piece_manifest,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    validate_piece_split_isolation(split_entries)
    validate_piece_split_coverage(
        split_entries,
        TYPE_NAMES,
        min_eval_pieces=args.min_eval_pieces,
    )
    for split_name, entries in split_entries.items():
        logger.info(
            "%s: %d pieces from %d files",
            split_name,
            len(entries),
            len({item.filepath for item in entries}),
        )
        for type_idx, type_name in enumerate(TYPE_NAMES):
            typed = [item for item in entries if item.type_idx == type_idx]
            if not typed:
                logger.warning("  %s: no %s pieces", split_name, type_name)
                continue
            timestamps = sorted(item.timestamp for item in typed)
            dist_bins = {item.dist_bin for item in typed if item.dist_bin >= 0}
            logger.info(
                "  %s: %d pieces, %d files, %s to %s, %d distance bins",
                type_name,
                len(typed),
                len({item.filepath for item in typed}),
                timestamps[0],
                timestamps[-1],
                len(dist_bins),
            )
            missing = sorted(set(range(30)) - dist_bins)
            if missing:
                logger.warning("    missing distance bins: %s", missing)

    cross_split_file_count = count_cross_split_files(split_entries)
    logger.warning(
        "%d source files contribute pieces to multiple splits; "
        "piece identities are disjoint, but evaluation does not measure "
        "cross-file generalization",
        cross_split_file_count,
    )
    shared_lig = LigFileIndex(
        sorted({entry.filepath for entry in piece_manifest}),
        validate=False,
    )
    train_set = MultiTaskDataset(split_entries["train"], "train", shared_lig)
    val_set = MultiTaskDataset(split_entries["val"], "val", shared_lig)
    test_set = (
        None
        if args.skip_test
        else MultiTaskDataset(split_entries["test"], "test", shared_lig)
    )

    type_samples = balanced_type_sample_count(
        train_set.type_labels, args.type_samples_per_epoch
    )
    type_sampler = BalancedTypeSampler(
        train_set.type_labels,
        num_samples=type_samples,
        seed=args.seed,
    )
    train_ld = DataLoader(train_set, batch_size=args.batch_size, sampler=type_sampler,
                          collate_fn=collate, pin_memory=dev == "cuda",
                          num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0)
    labelled_positions = np.flatnonzero(train_set.dist_labels >= 0)
    distance_samples = (
        len(labelled_positions)
        if args.distance_samples_per_epoch < 0
        else args.distance_samples_per_epoch
    )
    if distance_samples <= 0:
        raise RuntimeError("Training split has no labelled non-IC distance pieces")
    if args.distance_sampling == "hierarchical":
        distance_sampler = HierarchicalDistanceSampler(
            train_set.type_labels,
            train_set.dist_labels,
            train_set.date_ids,
            train_set.file_ids,
            num_samples=distance_samples,
            max_samples_per_file=args.max_distance_samples_per_file,
            seed=args.seed,
            replacement=True,
        )
    else:
        rng = np.random.default_rng(args.seed)
        selected_positions = rng.choice(
            labelled_positions,
            size=distance_samples,
            replace=False,
        ).tolist()
        distance_sampler = SubsetRandomSampler(selected_positions)
    distance_ld = DataLoader(
        train_set,
        batch_size=args.distance_batch_size,
        sampler=distance_sampler,
        collate_fn=collate,
        pin_memory=dev == "cuda",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_ld = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, pin_memory=dev == "cuda",
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    test_ld = None if test_set is None else DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=dev == "cuda",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    logger.info(
        "Epoch schedule: %d type samples (%d steps), %d balanced distance "
        "samples (%d steps)",
        len(type_sampler),
        len(train_ld),
        len(distance_sampler),
        len(distance_ld),
    )

    # Model
    model = create_mtl_model(
        base_channels=args.base,
        architecture=args.model_arch,
        num_types=len(TYPE_NAMES),
        dist_mlp_dim=args.dist_mlp_dim,
        dist_dropout=args.dist_dropout,
    ).to(dev)
    initialized_layers = []
    if args.init_model and not args.no_init:
        if os.path.isfile(args.init_model):
            initialized_layers = load_initial_weights(model, args.init_model)
            logger.info(
                "Initialized %d encoder/type tensors from %s",
                len(initialized_layers),
                args.init_model,
            )
        else:
            logger.warning(
                "Initial model not found: %s; training from random weights",
                args.init_model,
            )

    type_crit = nn.CrossEntropyLoss()
    dist_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.output, exist_ok=True)
    paths = output_paths(args.output)
    checkpoint_metadata = {
        **four_class_schema_metadata(),
        "model_name": args.model_arch,
        "base_channels": args.base,
        "dist_mlp_dim": args.dist_mlp_dim,
        "dist_dropout": args.dist_dropout,
        "dist_names": DIST_NAMES,
        "dist_bin_starts": DIST_BIN_STARTS,
        "preprocessing": {"normalize_mode": "minmax", "target_length": 8000},
        "split_config": {
            "strategy": "distance_stratified_piece_time_v1",
            "train_fraction": 1.0 - args.val_fraction - args.test_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "min_eval_pieces": args.min_eval_pieces,
            "files_may_overlap": True,
            "piece_identities_disjoint": True,
            "cross_split_file_count": cross_split_file_count,
        },
        "split_hashes": {
            split_name: compute_split_hash(entries, split_name)
            for split_name, entries in split_entries.items()
        },
        "lambda_dist": args.lambda_dist,
        "initial_model": None if args.no_init else (args.init_model or None),
        "initialized_layers": initialized_layers,
        "release_thresholds": {
            "min_type_f1": args.min_type_f1,
            "min_type_precision": args.min_type_precision,
            "min_type_recall": args.min_type_recall,
            "min_test_type_w2": args.min_test_type_w2,
            "min_test_macro_w2": args.min_test_macro_w2,
        },
        "distance_training": {
            "sampling": args.distance_sampling,
            "samples_per_epoch": distance_samples,
            "max_samples_per_file": args.max_distance_samples_per_file,
            "objective": args.distance_objective,
            "prediction": args.distance_prediction,
            "soft_tau": args.distance_soft_tau,
            "lambda_emd": args.lambda_emd,
            "lambda_reg": args.lambda_reg,
            "lambda_coarse": args.lambda_coarse,
            "distance_batch_type_weight": args.distance_batch_type_weight,
        },
    }
    best_score, best_state, wait = None, None, 0

    for epoch in range(args.epochs):
        # ── Train one epoch ──
        model.train()
        type_sampler.set_epoch(epoch)
        if hasattr(distance_sampler, "set_epoch"):
            distance_sampler.set_epoch(epoch)
        t_loss_sum, d_loss_sum = 0.0, 0.0
        t_ok, t_n, d_n = 0, 0, 0
        batches = alternate_stream_batches(train_ld, distance_ld)
        for stream, (x, y, dl) in tqdm(
            batches,
            total=len(train_ld) + len(distance_ld),
            desc=f"Epoch {epoch + 1}",
            leave=False,
        ):
            x, y, dl = x.to(dev), y.to(dev), dl.to(dev)
            batch_metrics = train_stream_step(
                model=model,
                stream=stream,
                x=x,
                type_labels=y,
                distance_labels=dl,
                optimizer=opt,
                type_criterion=type_crit,
                distance_objective=args.distance_objective,
                distance_loss_weight=args.lambda_dist,
                distance_batch_type_weight=args.distance_batch_type_weight,
                tau=args.distance_soft_tau,
                lambda_emd=args.lambda_emd,
                lambda_reg=args.lambda_reg,
                lambda_coarse=args.lambda_coarse,
            )
            bs = batch_metrics["type_count"]
            t_loss_sum += batch_metrics["type_loss"] * bs
            t_ok += batch_metrics["type_correct"]
            t_n += bs
            distance_count = batch_metrics["distance_count"]
            d_loss_sum += batch_metrics["distance_loss"] * distance_count
            d_n += distance_count
        train_type_acc = t_ok / t_n
        train_type_loss = t_loss_sum / t_n
        train_dist_loss = d_loss_sum / d_n if d_n else 0

        # ── Validate ──
        val = evaluate(
            model,
            val_ld,
            type_crit,
            dist_crit,
            dev,
            args.use_soft_distance_label,
            args.distance_soft_tau,
            prediction_mode=args.distance_prediction,
        )
        sched.step()

        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"T_loss={train_type_loss:.4f} D_loss={train_dist_loss:.4f} T_acc={train_type_acc:.4f} | "
            f"VT_acc={val['type_acc']:.4f} VT_f1={val['type_f1']:.4f} | "
            f"VD_acc={val.get('dist_acc',0):.4f} VD_mae={val.get('dist_mae_km',0):.0f}km "
            f"VD_w1={val.get('dist_w1',0):.4f} "
            f"VD_bin_w2={val.get('dist_equal_bin_macro_w2',0):.4f} | "
            f"E2E_D_acc={val.get('e2e_dist_acc',0):.4f} "
            f"Joint={val.get('joint_acc',0):.4f}")

        # Per-head details every 10 epochs
        if (epoch + 1) % 10 == 0:
            for name in DIST_NAMES:
                k = f"dist_{name}"
                logger.info(f"      {name}: acc={val.get(k+'_acc',0):.4f} "
                            f"mae={val.get(k+'_mae_km',0):.0f}km "
                            f"w1={val.get(k+'_w1',0):.4f} w2={val.get(k+'_w2',0):.4f}")

        # ── Checkpoint ──
        score = make_four_class_selection_key(val)
        if is_meaningful_improvement(score, best_score):
            best_score = score
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, paths["best"])
            torch.save(
                {**checkpoint_metadata, "model_state_dict": best_state},
                paths["best_checkpoint"],
            )
            logger.info("  Best model saved (score=%s)", score)
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.info(f"Early stop at epoch {epoch + 1}")
                break

    # Evaluate the selected state on validation, or on test exactly once.
    model.load_state_dict(best_state)
    _, training_features, training_type_targets = collect_type_outputs(
        model, train_ld, dev
    )
    validation_type_logits, validation_features, validation_type_targets = (
        collect_type_outputs(model, val_ld, dev)
    )
    feature_reference = fit_feature_reference(
        training_features,
        training_type_targets,
        num_types=len(TYPE_NAMES),
    )
    checkpoint_metadata["type_rejection"] = fit_rejection_policy(
        validation_type_logits,
        validation_features,
        validation_type_targets,
        feature_reference,
        precision_floor=args.min_type_precision,
        recall_floor=args.min_type_recall,
    )
    validation_logits, validation_targets = collect_distance_outputs(
        model, val_ld, dev
    )
    calibration = fit_distance_calibration(
        validation_logits,
        validation_targets,
        target_w2=0.8,
    )
    checkpoint_metadata["distance_calibration"] = calibration
    best_validation = evaluate(
        model,
        val_ld,
        type_crit,
        dist_crit,
        dev,
        args.use_soft_distance_label,
        args.distance_soft_tau,
        prediction_mode=args.distance_prediction,
        temperatures=calibration["temperatures"],
        bootstrap_repetitions=1000,
        bootstrap_seed=args.seed,
    )
    checkpoint_metadata["validation_metrics"] = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in best_validation.items()
    }
    torch.save(
        {**checkpoint_metadata, "model_state_dict": best_state},
        paths["best_checkpoint"],
    )
    logger.info(
        "Distance calibration: temperatures=%s threshold=%.4f "
        "coverage=%.4f w2=%.4f",
        [round(value, 3) for value in calibration["temperatures"]],
        calibration["confidence_threshold"],
        calibration["validation_coverage"],
        calibration["validation_w2"],
    )
    evaluated_split = "validation" if args.skip_test else "test"
    evaluation_loader = val_ld if args.skip_test else test_ld
    test = evaluate(
        model,
        evaluation_loader,
        type_crit,
        dist_crit,
        dev,
        args.use_soft_distance_label,
        args.distance_soft_tau,
        prediction_mode=args.distance_prediction,
        temperatures=calibration["temperatures"],
        bootstrap_repetitions=1000,
        bootstrap_seed=args.seed,
    )
    raw_type_metrics = {
        f"raw_{key}": test[key]
        for key in (
            "type_acc",
            "type_f1",
            "type_precision",
            "type_recall",
            "type_per_class_f1",
            "type_min_precision",
            "type_min_recall",
            "type_confusion_matrix",
        )
    }
    evaluation_type_logits, evaluation_features, evaluation_type_targets = (
        collect_type_outputs(model, evaluation_loader, dev)
    )
    rejected_type_metrics = summarize_rejected_types(
        decode_with_rejection(
            evaluation_type_logits,
            evaluation_features,
            checkpoint_metadata["type_rejection"],
        ),
        evaluation_type_targets,
    )
    test.update(raw_type_metrics)
    test.update(rejected_type_metrics)

    logger.info(f"\n{'='*50}")
    logger.info("%s Results:", evaluated_split.title())
    logger.info(f"  Type: acc={test['type_acc']:.4f} f1={test['type_f1']:.4f}")
    logger.info(f"  Distance (all): acc={test.get('dist_acc',0):.4f} "
                f"mae_bin={test.get('dist_mae_bin',0):.2f} "
                f"mae_km={test.get('dist_mae_km',0):.0f}km "
                f"w1={test.get('dist_w1',0):.4f} w2={test.get('dist_w2',0):.4f} "
                f"macro_w2={test.get('dist_macro_w2',0):.4f} "
                f"min_w2={test.get('dist_min_type_w2',0):.4f} "
                f"equal_bin_w2={test.get('dist_equal_bin_macro_w2',0):.4f} "
                f"equal_bin_min={test.get('dist_equal_bin_min_type_w2',0):.4f}")
    logger.info(
        f"  End-to-end distance: acc={test.get('e2e_dist_acc',0):.4f} "
        f"coverage={test.get('e2e_dist_coverage',0):.4f} "
        f"mae={test.get('e2e_dist_mae_km',0):.0f}km "
        f"w2={test.get('e2e_dist_w2',0):.4f} "
        f"joint={test.get('joint_acc',0):.4f}"
    )
    for name in DIST_NAMES:
        k = f"dist_{name}"
        if test.get(f"{k}_mae_km", -1) >= 0:
            logger.info(f"  {name}: acc={test.get(k+'_acc',0):.4f} "
                        f"mae_bin={test.get(k+'_mae_bin',0):.2f} "
                        f"mae_km={test.get(k+'_mae_km',0):.0f}km "
                        f"w1={test.get(k+'_w1',0):.4f} w2={test.get(k+'_w2',0):.4f} "
                        f"equal_bin_mae={test.get(k+'_equal_bin_mae_km',0):.0f}km "
                        f"equal_bin_w2={test.get(k+'_equal_bin_w2',0):.4f}")

    report = {"evaluated_split": evaluated_split, **test}
    if args.skip_test:
        passed = False
        reasons = ["locked test was skipped"]
    else:
        passed, reasons = evaluate_release_gate(
            test,
            min_type_f1=args.min_type_f1,
            min_type_precision=args.min_type_precision,
            min_type_recall=args.min_type_recall,
            min_type_w2=args.min_test_type_w2,
            min_macro_w2=args.min_test_macro_w2,
        )
        baseline_passed, baseline_reasons = compare_baseline_metrics(
            test, args.baseline_metrics
        )
        passed = passed and baseline_passed
        reasons.extend(baseline_reasons)
    save_candidate_and_maybe_promote(
        args.output,
        {**checkpoint_metadata, "model_state_dict": best_state},
        report,
        passed,
        reasons,
    )
    if passed:
        logger.info("PROMOTED candidate to %s", paths["model"])
    else:
        logger.warning("REJECTED candidate; deployed model was not changed")
        for reason in reasons:
            logger.warning("  release gate: %s", reason)


if __name__ == "__main__":
    main()
