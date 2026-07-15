"""Evaluate structured checkpoints on one immutable file-isolated split."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import classify
from conditional_pipeline import (
    apply_distance_temperatures,
    apply_rejection_policy,
    collect_prediction_bundle,
)
from data.lig_parser import LigFileIndex
from data.preprocessing import preprocess_batch
from data.split_artifacts import split_hash, write_json
from data.training_dataset import LightningPieceDataset, collate_training_batch
from data.training_manifest import ManifestEntry, build_piece_manifest
from evaluation import evaluate_predictions, file_bootstrap_metrics


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")

def _file_entries(split_manifest, split_name, task_data):
    rows = split_manifest["splits"][split_name]
    entries = []
    for row in rows:
        low, high = row.get("distance_low_km"), row.get("distance_high_km")
        entries.append(ManifestEntry(
            filepath=str(Path(task_data) / Path(row["path"])),
            type_idx=int(row["type_idx"]),
            dist_bin=(int(low) // 100 if low is not None and high - low == 100 else -1),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            n_pieces=int(row["n_pieces"]),
            distance_low_km=low,
            distance_high_km=high,
            is_daytime=bool(row["is_daytime"]),
        ))
    expected = split_manifest["split_hashes"][split_name]
    actual = split_hash(entries, task_data)
    if actual != expected:
        raise ValueError(
            f"split hash mismatch for {split_name}: manifest={expected}, files={actual}"
        )
    return entries


class LegacyBenchmarkDataset(Dataset):
    """Lazy legacy-preprocessed view of a locked piece manifest."""

    def __init__(self, entries, normalize_mode="minmax"):
        self.entries = list(entries)
        self.normalize_mode = normalize_mode
        paths = sorted({entry.filepath for entry in self.entries})
        self.lig = LigFileIndex(paths, validate=False)
        path_to_file = {
            os.path.normcase(os.path.abspath(path)): index
            for index, path in enumerate(self.lig.filepaths)
        }
        self.global_indices, self.type_labels = [], []
        self.low_km, self.high_km, self.file_ids, self.daylight = [], [], [], []
        for entry in self.entries:
            file_id = path_to_file[os.path.normcase(os.path.abspath(entry.filepath))]
            self.global_indices.append(
                int(self.lig._cumsum[file_id]) + int(entry.piece_index)
            )
            self.type_labels.append(int(entry.type_idx))
            self.low_km.append(-1 if entry.distance_low_km is None else entry.distance_low_km)
            self.high_km.append(-1 if entry.distance_high_km is None else entry.distance_high_km)
            self.file_ids.append(file_id)
            self.daylight.append(bool(entry.is_daytime))

    def __len__(self):
        return len(self.entries)

    def _items(self, positions):
        positions = [int(position) for position in positions]
        raw = np.stack(self.lig.read_pieces_batch([
            int(self.global_indices[position]) for position in positions
        ]))
        processed = preprocess_batch(raw, normalize_mode=self.normalize_mode)
        return [{
            "x": torch.from_numpy(processed[row].copy()).unsqueeze(0),
            "type_label": torch.tensor(self.type_labels[position], dtype=torch.long),
            "distance_low_km": torch.tensor(self.low_km[position], dtype=torch.float32),
            "distance_high_km": torch.tensor(self.high_km[position], dtype=torch.float32),
            "file_id": torch.tensor(self.file_ids[position], dtype=torch.long),
            "daylight": torch.tensor(self.daylight[position], dtype=torch.bool),
        } for row, position in enumerate(positions)]

    def __getitem__(self, index):
        return self._items([index])[0]

    def __getitems__(self, indices):
        return self._items(indices)

    def close(self):
        self.lig.close()


def _legacy_collate(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }


def _distance_points(distance_logits, type_indices, checkpoint):
    stacked = torch.stack(distance_logits, dim=1)
    rows = torch.arange(len(type_indices), device=type_indices.device)
    safe_types = type_indices.clamp(0, 3)
    selected = stacked[rows, safe_types]
    temperatures = torch.tensor(
        checkpoint.get("distance_calibration", {}).get("temperatures", [1.0] * 4),
        device=selected.device,
        dtype=selected.dtype,
    )
    probabilities = torch.softmax(
        selected / temperatures[safe_types].unsqueeze(1), dim=1
    )
    centers = torch.arange(50, 3000, 100, device=selected.device, dtype=selected.dtype)
    expected = (probabilities * centers.unsqueeze(0)).sum(dim=1)
    modal = probabilities.argmax(dim=1).to(selected.dtype) * 100.0 + 50.0
    training = checkpoint.get("distance_training", {})
    default_mode = training.get("prediction", "argmax")
    per_type = training.get("prediction_by_type", {})
    use_expected = torch.tensor([
        per_type.get(TYPE_NAMES[int(index)], default_mode) == "expected"
        for index in safe_types.detach().cpu().tolist()
    ], device=selected.device)
    return torch.where(use_expected, expected, modal)


@torch.no_grad()
def _evaluate_legacy(model, checkpoint, loader, device, locked_hash):
    schema = classify.checkpoint_schema(checkpoint)
    records = []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        if schema == "legacy_five_class":
            type_logits, distance_logits = model(batch["x"])
            legacy_prediction = type_logits.argmax(dim=1)
            predicted = legacy_prediction - 1
            accepted = legacy_prediction > 0
        else:
            features, type_logits, distance_logits = model.forward_with_features(batch["x"])
            decoded = classify.decode_with_rejection(
                type_logits, features, checkpoint["type_rejection"]
            )
            predicted = decoded["predicted"].to(device)
            accepted = decoded["accepted"].to(device)
        routed_distance = _distance_points(distance_logits, predicted, checkpoint)
        oracle_distance = _distance_points(
            distance_logits, batch["type_label"], checkpoint
        )
        for row in range(len(predicted)):
            records.append({
                "file_id": int(batch["file_id"][row].item()),
                "true_type": int(batch["type_label"][row].item()),
                "predicted_type": int(predicted[row].item()),
                "accepted": bool(accepted[row].item()),
                "distance_low_km": int(batch["distance_low_km"][row].item()),
                "distance_high_km": int(batch["distance_high_km"][row].item()),
                "predicted_distance_km": float(routed_distance[row].item()),
                "oracle_distance_km": float(oracle_distance[row].item()),
                "daylight": bool(batch["daylight"][row].item()),
                "split_hash": locked_hash,
            })
    return records


def evaluate_checkpoint(
    model_path,
    file_entries,
    task_data,
    locked_hash,
    batch_size=256,
    workers=0,
    bootstrap_iterations=1000,
    device=None,
):
    """Evaluate one legacy or conditional structured checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = classify.load_mtl_checkpoint(model_path, device)
    pieces = build_piece_manifest(file_entries)
    schema = classify.checkpoint_schema(checkpoint)
    if schema == "four_class_rejection_v2":
        dataset = LightningPieceDataset(
            pieces,
            split="benchmark",
            data_root=task_data,
            time_context_mode=classify.checkpoint_time_context_mode(checkpoint),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_training_batch,
            pin_memory=device == "cuda",
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        bundle = collect_prediction_bundle(model, loader, device, locked_hash)
        temperatures = checkpoint.get("distance_calibration", {}).get(
            "temperatures", [1.0] * 4
        )
        apply_distance_temperatures(bundle, temperatures)
        if checkpoint.get("type_rejection"):
            apply_rejection_policy(bundle, checkpoint["type_rejection"])
        records = bundle["records"]
    else:
        dataset = LegacyBenchmarkDataset(
            pieces,
            checkpoint.get("preprocessing", {}).get("normalize_mode", "minmax"),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_legacy_collate,
            pin_memory=device == "cuda",
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        records = _evaluate_legacy(model, checkpoint, loader, device, locked_hash)
    metrics = evaluate_predictions(records)
    metrics.update(file_bootstrap_metrics(
        records, iterations=bootstrap_iterations, seed=42
    ))
    metrics["checkpoint"] = str(model_path)
    metrics["checkpoint_schema"] = schema
    dataset.close()
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="./weights/conditional/benchmark.json")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    return parser


def main():
    args = build_arg_parser().parse_args()
    with Path(args.split_manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = _file_entries(manifest, args.split, args.task_data)
    locked_hash = manifest["split_hashes"][args.split]
    results = {
        "reference_only": True,
        "split": args.split,
        "split_hash": locked_hash,
        "models": {},
    }
    for specification in args.model:
        if "=" not in specification:
            raise ValueError("--model must use NAME=PATH")
        name, path = specification.split("=", 1)
        print(f"Evaluating {name}: {path}")
        metrics = evaluate_checkpoint(
            path,
            entries,
            args.task_data,
            locked_hash,
            args.batch_size,
            args.num_workers,
            args.bootstrap_iterations,
        )
        metrics["reference_only"] = True
        results["models"][name] = metrics
    write_json(args.output, results)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
