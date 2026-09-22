"""Reproducible validation-only accuracy, corruption, and latency experiment."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from checkpoints import load_model_checkpoint, model_sha256
from data.dataset import FiveClassDataset
from data.manifest import build_piece_table
from data.preprocess import PreprocessConfig, preprocess_views
from data.split import assign_piece_splits, split_artifact
from evaluation import (
    HierarchicalDecisionConfig, _type_metrics, decide_hierarchical_types,
    infer_hierarchical_types,
)
from models import HierarchicalTypeOutput

OUT = ROOT / "weights" / "streaming_research_v1"
PROFILES = [
    ("hierarchical32", "hierarchical", "standard"),
    ("multiscale32", "multiscale", "standard"),
    ("multiscale32_drift", "multiscale", "sensor_drift_v1"),
]
CONDITIONS = ("clean", "noise_20db", "baseline_drift", "lowpass_80khz", "impulse_interference")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def make_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "validation_raw.npz"
    if cache.exists():
        return
    table, _ = build_piece_table(ROOT.parent / "train_data", require_distance=True)
    assignment = assign_piece_splits(table, seed=42)
    positions = assignment.positions("validation")
    rng = np.random.default_rng(20260921)
    selected = np.sort(np.concatenate([
        rng.choice(positions[table.type_index[positions] == label], 400, replace=False)
        for label in range(5)
    ]))
    dataset = FiveClassDataset(table, selected, "validation")
    try:
        raw = np.stack(dataset.lig.read_pieces_batch([
            dataset._global_piece_index(int(p)) for p in selected
        ])).astype(np.float32)
    finally:
        dataset.close()
    keys = [table.piece_key(int(p)) for p in selected]
    np.savez(cache, raw=raw, labels=table.type_index[selected],
             daylight=table.daylight[selected], keys=np.array(keys))
    write_json(OUT / "validation_manifest.json", {
        "split": "validation", "seed": 42, "selection_seed": 20260921,
        "per_class": 400, "piece_keys": keys,
        "split_hash": hashlib.sha256(json.dumps(split_artifact(table, assignment),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "test_used_for_selection": False,
    })
    print("Cached 2000 validation raw pieces; test data untouched", flush=True)


def perturb(raw, condition, *, seed=20260921):
    rng = np.random.default_rng(seed)
    centered = raw - np.median(raw, axis=1, keepdims=True)
    scale = np.quantile(np.abs(centered), .95, axis=1, keepdims=True).clip(1e-6)
    if condition == "clean":
        return raw
    if condition == "noise_20db":
        rms = np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True))
        return (raw + rng.normal(size=raw.shape) * rms * .1).astype(np.float32)
    if condition == "baseline_drift":
        axis = np.linspace(-1, 1, raw.shape[1], dtype=np.float32)[None, :]
        return (raw + axis * scale * .2).astype(np.float32)
    if condition == "lowpass_80khz":
        return sosfiltfilt(butter(2, 80_000, fs=5_000_000, output="sos"), raw).astype(np.float32)
    if condition == "impulse_interference":
        changed = raw.copy()
        for i in range(len(raw)):
            positions = rng.integers(0, raw.shape[1], size=8)
            changed[i, positions] += rng.normal(0, 8 * scale[i, 0], size=8)
        return changed
    raise ValueError(condition)


def score(labels, predictions):
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    result = _type_metrics(confusion)
    support = confusion.sum(1).clip(1)
    result.update(
        known_macro_recall=float(np.mean(result["type_recall"][1:])),
        known_false_to_ic=(confusion[1:, 0] / support[1:]).tolist(),
        cg_to_nbe_rate=float(confusion[np.ix_([1, 3], [2, 4])].sum() / support[[1, 3]].sum()),
        ic_to_nbe_rate=float(confusion[0, [2, 4]].sum() / support[0]),
        nbe_to_cg_rate=float(confusion[np.ix_([2, 4], [1, 3])].sum() / support[[2, 4]].sum()),
        nbe_polarity_confusion=float((confusion[2, 4] + confusion[4, 2]) / support[[2, 4]].sum()),
        cg_polarity_confusion=float((confusion[1, 3] + confusion[3, 1]) / support[[1, 3]].sum()),
    )
    return result


def shifted(values, amount):
    result = torch.zeros_like(values)
    result[..., amount:] = values[..., :-amount]
    return result


def evaluate(name, path):
    cache = np.load(OUT / "validation_raw.npz")
    checkpoint = load_model_checkpoint(path, "cuda")
    checkpoint.model.eval()
    config = PreprocessConfig(**checkpoint.preprocess_config)
    decision = HierarchicalDecisionConfig(**checkpoint.metadata["decision_config"])
    result = {"checkpoint": str(Path(path).relative_to(ROOT)),
              "model_sha256": model_sha256(path), "partition": "validation",
              "piece_count": len(cache["labels"]), "conditions": {}}
    for condition in CONDITIONS:
        local, global_view = preprocess_views(perturb(cache["raw"], condition), config)
        predictions, agreement = [], []
        with torch.inference_mode():
            for start in range(0, len(local), 128):
                x = torch.from_numpy(local[start:start+128, None]).to("cuda")
                g = torch.from_numpy(global_view[start:start+128, None]).to("cuda")
                d = torch.tensor(cache["daylight"][start:start+128, None], device="cuda", dtype=torch.float32)
                primary = checkpoint.model(x, g, d)
                alternate = checkpoint.model(shifted(x, 16), shifted(g, 4), d)
                predictions.extend(decide_hierarchical_types(primary, alternate, decision).final_type.cpu().tolist())
                agreement.extend(primary.known_logits.argmax(1).eq(alternate.known_logits.argmax(1)).cpu().tolist())
        metrics = score(cache["labels"], np.array(predictions))
        metrics["known_consistency"] = float(np.array(agreement)[cache["labels"] > 0].mean())
        result["conditions"][condition] = metrics
        print(f"{name} {condition}: recall={metrics['known_macro_recall']:.4f} "
              f"NBE_precision={min(metrics['type_precision'][2], metrics['type_precision'][4]):.4f}", flush=True)
    # Completed waveform -> preprocess + paired type decision. No file I/O or distance inference.
    timing = {}
    for device in ("cpu", "cuda"):
        checkpoint.model.to(device)
        for batch_size in (1, 32):
            raw_batch = cache["raw"][:batch_size]
            samples = []
            with torch.inference_mode():
                for repeat in range(35):
                    if device == "cuda": torch.cuda.synchronize()
                    start = time.perf_counter()
                    x, g = preprocess_views(raw_batch, config)
                    tensors = [torch.from_numpy(v[:, None]).to(device) for v in (x, g)]
                    day = torch.tensor(cache["daylight"][:batch_size, None], device=device, dtype=torch.float32)
                    infer_hierarchical_types(checkpoint.model, checkpoint.metadata["decision_config"],
                        *tensors, day, None, torch.zeros(batch_size, dtype=torch.bool, device=device))
                    if device == "cuda": torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    if repeat >= 5: samples.append(elapsed)
            timing[f"{device}_batch{batch_size}"] = {
                "median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, .95)),
                "pieces_per_second": batch_size * 1000 / float(np.median(samples)), "repeats": len(samples),
            }
    result["latency"] = timing
    result["parameters"] = sum(p.numel() for p in checkpoint.model.parameters())
    write_json(OUT / f"{name}_validation.json", result)
    print(f"{name} CPU single p95={timing['cpu_batch1']['p95_ms']:.2f} ms", flush=True)
    del checkpoint
    torch.cuda.empty_cache()


def train_candidates():
    for name, architecture, augmentation in PROFILES:
        target = OUT / name
        finished = target / "type" / "metrics.json"
        if finished.exists():
            print(f"Already trained {name}", flush=True)
            if not (OUT / f"{name}_validation.json").exists():
                evaluate(name, target / "type" / "model.pt")
            continue
        args = [sys.executable, "-u", "train.py", "--task_data", str(ROOT.parent / "train_data"),
            "--output", str(target), "--stage", "type", "--type_architecture", architecture,
            "--type_loss_profile", "known_consistency_v1", "--augmentation_profile", augmentation,
            "--base_channels", "32", "--epochs", "20", "--patience", "6", "--batch_size", "60",
            "--defer_test"]
        last = target / "type" / "last.pt"
        if last.exists(): args.extend(["--resume", str(last)])
        write_json(OUT / "current_job.json", {"candidate": name, "command": args, "started": time.time()})
        print(f"Training from random initialization: {name}", flush=True)
        with (OUT / f"{name}_train.log").open("a", encoding="utf-8") as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        evaluate(name, target / "type" / "model.pt")


def compare_candidates():
    records = {}
    for path in sorted(OUT.glob("*_validation.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        conditions = result["conditions"]
        clean = conditions["clean"]
        stress = [v for k, v in conditions.items() if k != "clean"]
        checks = {
            "clean_nbe_precision_ge_099": min(clean["type_precision"][2], clean["type_precision"][4]) >= .99,
            "stress_nbe_precision_ge_097": min(min(v["type_precision"][2], v["type_precision"][4]) for v in stress) >= .97,
            "clean_known_recall_ge_096": clean["known_macro_recall"] >= .96,
            "cpu_single_p95_le_10ms": result["latency"]["cpu_batch1"]["p95_ms"] <= 10,
        }
        records[path.stem.replace("_validation", "")] = {
            "clean_known_recall": clean["known_macro_recall"],
            "worst_stress_known_recall": min(v["known_macro_recall"] for v in stress),
            "clean_nbe_min_precision": min(clean["type_precision"][2], clean["type_precision"][4]),
            "worst_stress_nbe_min_precision": min(min(v["type_precision"][2], v["type_precision"][4]) for v in stress),
            "cpu_single_p95_ms": result["latency"]["cpu_batch1"]["p95_ms"],
            "gates": checks, "eligible": all(checks.values()),
        }
    eligible = [name for name, v in records.items() if v["eligible"]]
    selected = max(eligible, key=lambda n: records[n]["worst_stress_known_recall"]) if eligible else None
    write_json(OUT / "comparison.json", {
        "selection_partition": "validation", "selected": selected, "candidates": records,
        "policy": "Maximize worst-stress known recall subject to frozen precision, clean recall and CPU latency gates.",
        "promotion": "pending full validation, independent test, natural drift and stream replay; no deployed bundle replaced",
    })
    print(f"Validation selection: {selected or 'no candidate passed every gate'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "train", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(OUT / "environment.json", {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(), "latency_scope": "preprocess + paired type decision; excludes I/O, distance and queueing"})
    make_cache()
    if args.phase in ("baseline", "all"):
        for name, directory in (("old_baseline", "multi_model"), ("anchor_baseline", "anchor_moe_best_bundle")):
            if not (OUT / f"{name}_validation.json").exists():
                evaluate(name, ROOT / "weights" / directory / "type" / "model.pt")
    if args.phase in ("train", "all"):
        train_candidates()
    compare_candidates()
    write_json(OUT / f"{args.phase}_complete.json", {"completed": time.time()})


if __name__ == "__main__":
    main()
