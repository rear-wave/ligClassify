"""Three-fold training, OOF artifact, and release-gate orchestration."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.cross_validation import (
    assign_exact_folds,
    build_support_map,
    fold_train_holdout,
    validate_fold_assignment,
)
from data.oof_manifest import expected_oof_rows, validate_oof_rows
from data.split_artifacts import make_fold_manifest, split_hash, stable_json_hash
from data.training_manifest import ManifestEntry
from data.training_manifest import build_piece_manifest
from data.distance_sampling import ConditionBalancedSampler, JointConditionSampler
from data.training_dataset import LightningPieceDataset
from conditional_pipeline import (
    _clone_state,
    _joint_checkpoint_improved,
    _loader,
    apply_distance_temperatures,
    collect_prediction_bundle,
    collect_reference_features,
    fit_distance_temperatures,
    run_conditional_epoch,
)
from evaluation import (
    checkpoint_selection_key,
    distance_calibration_is_safe,
    evaluate_predictions,
    evaluate_release,
    file_bootstrap_metrics,
    round_metrics,
)
from models import create_mtl_model
from open_set import (
    _fit_temperature,
    fit_feature_reference,
    fit_oof_rejection_policy,
    rejection_signals,
)


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
FOLD_BATCH_SIZE = 128
FOLD_MODEL_CONFIG = {
    "base_channels": 64,
    "architecture": "conditional_expert_v1",
    "num_types": 4,
    "context_dim": 1,
    "dist_mlp_dim": 128,
    "dist_dropout": 0.2,
}
FOLD_LEARNING_RATE = 3e-4
FOLD_WEIGHT_DECAY = 5e-4
FOLD_COARSE_WEIGHT = 0.5
OOF_FIELDS = (
    "piece_key", "source_path", "piece_index", "fold", "true_type",
    "predicted_type", "final_type", "accepted", "rejection_reason",
    "logit_NCG", "logit_NNBE", "logit_PCG", "logit_PNBE",
    "prob_NCG", "prob_NNBE", "prob_PCG", "prob_PNBE",
    "confidence", "margin", "normalized_feature_distance", "quality_score",
    "distance_low_km", "distance_high_km", "predicted_distance_km",
    "oracle_distance_km", "distance_temperature", "daylight",
    "support_status", "train_hash", "holdout_hash", "config_hash",
)
OOF_INTEGER_FIELDS = {
    "piece_index", "fold", "true_type", "predicted_type", "final_type"
}
OOF_FLOAT_FIELDS = {
    name for name in OOF_FIELDS
    if name.startswith("logit_") or name.startswith("prob_")
} | {
    "confidence", "margin", "normalized_feature_distance", "quality_score",
    "distance_low_km", "distance_high_km", "predicted_distance_km",
    "oracle_distance_km", "distance_temperature",
}


@dataclass(frozen=True)
class CVConfig:
    """Immutable configuration whose training fields define resume identity."""

    task_data: Path
    output: Path
    folds: int = 3
    seed: int = 42
    samples_per_epoch: int = 120000
    max_samples_per_file: int = 512
    type_focus_epochs: int = 3
    type_focus_distance_weight: float = 0.25
    joint_distance_weight: float = 1.0
    max_epochs: int = 50
    patience: int = 10
    time_context: str = "daylight"
    rejection_target_precision: float = 0.96
    rejection_min_coverage: float = 0.80
    bootstrap_iterations: int = 1000
    num_workers: int = 2
    no_amp: bool = False
    no_init: bool = True
    init_model: str = ""
    resume_cv: bool = False
    stop_after_oof: bool = False


def training_config_hash(config: CVConfig) -> str:
    """Hash only settings that affect fitted fold parameters."""
    payload = dataclasses.asdict(config)
    for key in ("task_data", "output", "resume_cv", "stop_after_oof"):
        payload.pop(key, None)
    return stable_json_hash(payload)


@dataclass
class FoldResult:
    """Verified artifact paths and selection metadata for one fold."""

    fold_index: int
    best_epoch: int
    train_hash: str
    holdout_hash: str
    config_hash: str
    checkpoint_path: str
    oof_path: str
    metrics: dict


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_oof_csv(path):
    """Read OOF rows and convert every field in the stable typed contract."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OOF CSV has no header")
        rows = []
        for raw in reader:
            row = dict(raw)
            for name in OOF_INTEGER_FIELDS:
                if name in row and row[name] != "":
                    row[name] = int(row[name])
            for name in OOF_FLOAT_FIELDS:
                if name in row:
                    row[name] = None if row[name] == "" else float(row[name])
            for name in ("accepted", "daylight"):
                if name in row:
                    if row[name] not in {"True", "False", "1", "0"}:
                        raise ValueError(
                            f"invalid OOF boolean {name}={row[name]!r}"
                        )
                    row[name] = row[name] in {"True", "1"}
            rows.append(row)
    return rows


def validate_fold_checkpoint(checkpoint, fold_index, expected_hashes):
    """Validate fold identity and prove its model state loads strictly."""
    if checkpoint.get("schema") != "conditional_expert_cv_fold_v1":
        raise ValueError("invalid fold checkpoint schema")
    if int(checkpoint.get("fold_index", -1)) != int(fold_index):
        raise ValueError("fold checkpoint index mismatch")
    for key in ("train_hash", "holdout_hash", "config_hash"):
        if checkpoint.get(key) != expected_hashes[key]:
            raise ValueError(f"fold checkpoint {key} mismatch")
    if checkpoint.get("random_initialization") is not True:
        raise ValueError("fold checkpoint was not random-initialized")
    try:
        model = create_mtl_model(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except KeyError as exc:
        raise ValueError(f"fold checkpoint missing field: {exc.args[0]}") from exc


def load_verified_fold(output_dir, fold_index, expected_hashes, expected_rows):
    """Load only a completed fold whose state, model, and OOF rows all verify."""
    directory = Path(output_dir) / "folds" / f"fold_{int(fold_index)}"
    state_path = directory / "fold_state.json"
    best_path = directory / "best.pt"
    oof_path = directory / "oof.csv"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid fold state for fold {fold_index}: {exc}") from exc
    if state.get("status") != "complete":
        return None
    if not (best_path.is_file() and oof_path.is_file()):
        raise ValueError(f"completed fold is missing artifacts: fold {fold_index}")
    for key in ("train_hash", "holdout_hash", "config_hash"):
        if state.get(key) != expected_hashes[key]:
            return False
    try:
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_fold_checkpoint(checkpoint, fold_index, expected_hashes)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(
            f"invalid fold checkpoint for fold {fold_index}: {exc}"
        ) from exc
    try:
        state_best_epoch = int(state["best_epoch"])
        checkpoint_best_epoch = int(checkpoint["best_epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid completed fold best_epoch: fold {fold_index}: {exc}"
        ) from exc
    if state_best_epoch <= 0 or checkpoint_best_epoch != state_best_epoch:
        raise ValueError(f"fold checkpoint best_epoch mismatch: fold {fold_index}")
    try:
        rows = read_oof_csv(oof_path)
        validate_oof_rows(rows, expected_rows)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid fold OOF for fold {fold_index}: {exc}") from exc
    try:
        return FoldResult(
            fold_index=int(fold_index),
            best_epoch=int(state["best_epoch"]),
            train_hash=state["train_hash"],
            holdout_hash=state["holdout_hash"],
            config_hash=state["config_hash"],
            checkpoint_path=str(best_path),
            oof_path=str(oof_path),
            metrics=dict(state["metrics"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid completed fold metadata for fold {fold_index}: {exc}"
        ) from exc


def train_conditional_fold(
    train_entries, holdout_entries, output_dir, config, device, fold_index,
):
    """Train one random-init fold, select on holdout, and write its OOF rows."""
    if config.init_model or not config.no_init:
        raise ValueError("cross-validation folds require random initialization")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_index = int(fold_index)
    hashes = {
        "train_hash": split_hash(train_entries, config.task_data),
        "holdout_hash": split_hash(holdout_entries, config.task_data),
        "config_hash": training_config_hash(config),
    }
    stage_config = {
        "type_focus_epochs": int(config.type_focus_epochs),
        "type_focus_distance_weight": float(config.type_focus_distance_weight),
        "joint_distance_weight": float(config.joint_distance_weight),
        "max_epochs": int(config.max_epochs),
        "patience": int(config.patience),
    }
    state_base = {
        "fold_index": fold_index,
        **hashes,
        "random_initialization": True,
        "stage_config": stage_config,
    }
    state_path = output_dir / "fold_state.json"
    prior_state = None
    if config.resume_cv and state_path.is_file():
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid fold state for fold {fold_index}: {exc}"
            ) from exc

    train_pieces = build_piece_manifest(train_entries)
    holdout_pieces = build_piece_manifest(holdout_entries)
    if not train_pieces or not holdout_pieces:
        raise RuntimeError(f"fold {fold_index} has an empty train or holdout split")
    train_set = LightningPieceDataset(
        train_pieces,
        split="train",
        data_root=config.task_data,
        time_context_mode=config.time_context,
    )
    holdout_set = LightningPieceDataset(
        holdout_pieces,
        split="val",
        data_root=config.task_data,
        time_context_mode=config.time_context,
    )
    train_sampler = JointConditionSampler(
        train_set.type_labels,
        train_set.daylight,
        train_set.distance_low_km,
        train_set.distance_high_km,
        train_set.file_ids,
        config.samples_per_epoch,
        max_samples_per_file=config.max_samples_per_file,
        seed=config.seed + fold_index,
    )
    cuda = str(device).startswith("cuda")
    train_loader = _loader(
        train_set,
        FOLD_BATCH_SIZE,
        train_sampler,
        workers=config.num_workers,
        cuda=cuda,
    )
    holdout_loader = _loader(
        holdout_set,
        FOLD_BATCH_SIZE,
        workers=config.num_workers,
        cuda=cuda,
    )

    fold_seed = int(config.seed) + fold_index
    torch.manual_seed(fold_seed)
    if cuda:
        torch.cuda.manual_seed_all(fold_seed)
    model_config = dict(FOLD_MODEL_CONFIG)
    model_config["context_dim"] = 1 if config.time_context == "daylight" else 3
    model = create_mtl_model(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FOLD_LEARNING_RATE, weight_decay=FOLD_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )
    amp_enabled = cuda and not config.no_amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_score = None
    best_state = None
    best_epoch = 0
    best_metrics = {}
    wait = 0

    latest_path = output_dir / "latest.pt"
    resume_state = None
    if config.resume_cv and prior_state is not None and latest_path.is_file():
        hashes_match = all(
            prior_state.get(key) == value for key, value in hashes.items()
        )
        if prior_state.get("status") != "complete" and hashes_match:
            try:
                resume_state = torch.load(
                    latest_path, map_location="cpu", weights_only=False
                )
            except (OSError, RuntimeError, EOFError) as exc:
                raise ValueError(
                    f"invalid latest fold checkpoint for fold {fold_index}: {exc}"
                ) from exc
    if resume_state is not None:
        if resume_state.get("schema") != "conditional_expert_cv_fold_state_v1":
            raise ValueError(f"invalid latest fold state schema: fold {fold_index}")
        if int(resume_state.get("fold_index", -1)) != fold_index:
            raise ValueError(f"latest fold checkpoint index mismatch: fold {fold_index}")
        for key, value in hashes.items():
            if resume_state.get(key) != value:
                raise ValueError(f"latest fold checkpoint {key} mismatch: fold {fold_index}")
        if resume_state.get("random_initialization") is not True:
            raise ValueError(f"latest fold checkpoint was not random-initialized: fold {fold_index}")
        model.load_state_dict(resume_state["model_state"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        scaler.load_state_dict(resume_state.get("scaler_state", {}))
        start_epoch = int(resume_state["completed_epochs"])
        best_score = (
            tuple(resume_state["best_score"])
            if resume_state.get("best_score") is not None
            else None
        )
        best_state = resume_state.get("best_model_state")
        best_epoch = int(resume_state.get("best_epoch", 0))
        best_metrics = dict(resume_state.get("best_metrics", {}))
        wait = int(resume_state.get("early_stop_wait", 0))
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if cuda and "cuda_rng_state_all" in resume_state:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

    _atomic_write_json(state_path, {
        **state_base,
        "status": "in_progress",
        "best_epoch": best_epoch,
        "metrics": best_metrics,
    })

    try:
        for epoch in range(start_epoch, config.max_epochs):
            stage, _, _ = run_conditional_epoch(
                model,
                train_loader,
                train_sampler,
                optimizer,
                criterion,
                scaler,
                device,
                epoch,
                config.type_focus_epochs,
                config.type_focus_distance_weight,
                config.joint_distance_weight,
                lambda_coarse=FOLD_COARSE_WEIGHT,
                no_amp=config.no_amp,
            )
            scheduler.step()
            bundle = collect_prediction_bundle(
                model, holdout_loader, device, hashes["holdout_hash"], fold_index
            )
            validation_metrics = evaluate_predictions(bundle["records"])
            score = checkpoint_selection_key(validation_metrics)
            if _joint_checkpoint_improved(stage, score, best_score, best_epoch):
                best_score = score
                best_state = _clone_state(model)
                best_epoch = epoch + 1
                best_metrics = validation_metrics
                wait = 0
                checkpoint = {
                    "schema": "conditional_expert_cv_fold_v1",
                    **state_base,
                    "best_epoch": best_epoch,
                    "model_config": model_config,
                    "model_state": best_state,
                    "metrics": best_metrics,
                }
                _atomic_torch_save(output_dir / "best.pt", checkpoint)
            elif stage == "joint":
                wait += 1
            latest = {
                "schema": "conditional_expert_cv_fold_state_v1",
                **state_base,
                "completed_epochs": epoch + 1,
                "model_config": model_config,
                "model_state": _clone_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_score": best_score,
                "best_model_state": best_state,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "early_stop_wait": wait,
                "torch_rng_state": torch.get_rng_state(),
            }
            if cuda:
                latest["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            _atomic_torch_save(latest_path, latest)
            _atomic_write_json(state_path, {
                **state_base,
                "status": "in_progress",
                "best_epoch": best_epoch,
                "metrics": best_metrics,
            })
            if stage == "joint" and wait >= config.patience:
                break

        if best_state is None:
            raise RuntimeError(
                f"fold {fold_index} completed without a selectable joint checkpoint"
            )
        model.load_state_dict(best_state, strict=True)
        bundle = collect_prediction_bundle(
            model, holdout_loader, device, hashes["holdout_hash"], fold_index
        )
        reference_sampler = ConditionBalancedSampler(
            train_set.type_labels,
            train_set.daylight,
            train_set.distance_low_km,
            train_set.distance_high_km,
            train_set.file_ids,
            min(20000, len(train_set)),
            max_samples_per_file=config.max_samples_per_file,
            seed=config.seed + fold_index,
            distance_only=False,
        )
        reference_loader = _loader(
            train_set,
            FOLD_BATCH_SIZE,
            reference_sampler,
            workers=config.num_workers,
            cuda=cuda,
        )
        reference_features, reference_labels = collect_reference_features(
            model, reference_loader, device
        )
        reference = fit_feature_reference(
            reference_features, reference_labels, num_types=4
        )
        signals = rejection_signals(
            bundle["logits"], bundle["features"], reference, 1.0,
            quality=bundle["quality"],
        )

        before = evaluate_predictions(bundle["records"])
        fitted_temperatures = fit_distance_temperatures(bundle)
        apply_distance_temperatures(bundle, fitted_temperatures)
        after = evaluate_predictions(bundle["records"])
        safe = distance_calibration_is_safe(before, after)
        accepted_temperatures = (
            [float(value) for value in fitted_temperatures]
            if safe
            else [1.0, 1.0, 1.0, 1.0]
        )
        if not safe:
            apply_distance_temperatures(bundle, accepted_temperatures)
        distance_calibration = {
            "before": before,
            "after": after,
            "fitted_temperatures": [
                float(value) for value in fitted_temperatures
            ],
            "accepted_temperatures": accepted_temperatures,
            "accepted": bool(safe),
            "guard_reason": "accepted" if safe else "point_metrics_regressed",
        }
        support_map = build_support_map(
            [*train_entries, *holdout_entries], TYPE_NAMES
        )
        raw_rows = []
        probabilities = bundle["logits"].softmax(dim=1)
        for row_index, record in enumerate(bundle["records"]):
            true_name = TYPE_NAMES[int(record["true_type"])]
            support_key = (
                f"{true_name}/{'day' if record['daylight'] else 'night'}/"
                f"{int(record['distance_low_km'])}-"
                f"{int(record['distance_high_km'])}km"
            )
            predicted_type = int(record["predicted_type"])
            row = {name: "" for name in OOF_FIELDS}
            row.update({
                "piece_key": record["piece_key"],
                "source_path": record["source_path"],
                "piece_index": record["piece_index"],
                "fold": fold_index,
                "true_type": record["true_type"],
                "predicted_type": predicted_type,
                "final_type": predicted_type,
                "accepted": True,
                "rejection_reason": "uncalibrated",
                "confidence": float(probabilities[row_index].max().item()),
                "margin": float(
                    probabilities[row_index].topk(2).values.diff().abs().item()
                ),
                "normalized_feature_distance": float(
                    signals["normalized_distance"][row_index].item()
                ),
                "quality_score": float(signals["quality_score"][row_index].item()),
                "distance_low_km": record["distance_low_km"],
                "distance_high_km": record["distance_high_km"],
                "predicted_distance_km": record["predicted_distance_km"],
                "oracle_distance_km": record["oracle_distance_km"],
                "distance_temperature": accepted_temperatures[predicted_type],
                "daylight": record["daylight"],
                "support_status": support_map[support_key]["status"],
                **hashes,
            })
            for type_position, type_name in enumerate(TYPE_NAMES):
                row[f"logit_{type_name}"] = float(
                    bundle["logits"][row_index, type_position].item()
                )
                row[f"prob_{type_name}"] = float(
                    probabilities[row_index, type_position].item()
                )
            raw_rows.append(row)
        oof_path = output_dir / "oof.csv"
        _atomic_write_csv(oof_path, raw_rows, OOF_FIELDS)
        fold_metrics = {
            "selection": best_metrics,
            "distance_calibration": distance_calibration,
            "feature_reference": reference,
        }
        checkpoint = torch.load(
            output_dir / "best.pt", map_location="cpu", weights_only=False
        )
        checkpoint["metrics"] = fold_metrics
        _atomic_torch_save(output_dir / "best.pt", checkpoint)
        _atomic_write_json(state_path, {
            **state_base,
            "status": "complete",
            "best_epoch": best_epoch,
            "metrics": fold_metrics,
        })
        return FoldResult(
            fold_index=fold_index,
            best_epoch=best_epoch,
            train_hash=hashes["train_hash"],
            holdout_hash=hashes["holdout_hash"],
            config_hash=hashes["config_hash"],
            checkpoint_path=str(output_dir / "best.pt"),
            oof_path=str(oof_path),
            metrics=fold_metrics,
        )
    finally:
        train_set.close()
        holdout_set.close()


def _fold_hashes_from_rows(rows):
    fold_hashes = {}
    for row in rows:
        fold = str(int(row["fold"]))
        hashes = {
            key: str(row[key])
            for key in ("train_hash", "holdout_hash", "config_hash")
        }
        previous = fold_hashes.setdefault(fold, hashes)
        if previous != hashes:
            raise ValueError(f"OOF rows contain inconsistent hashes for fold {fold}")
    if set(fold_hashes) != {"0", "1", "2"}:
        raise ValueError("OOF artifacts must contain exactly folds 0, 1, and 2")
    return fold_hashes


def _distance_calibration_summary(output_dir, fold_hashes):
    folds = {}
    accepted_vectors = []
    for fold in range(3):
        state_path = (
            Path(output_dir) / "folds" / f"fold_{fold}" / "fold_state.json"
        )
        if not state_path.is_file():
            raise ValueError(f"missing fold calibration state: fold {fold}")
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for key, expected in fold_hashes[str(fold)].items():
                if state.get(key) != expected:
                    raise ValueError(f"fold calibration {key} mismatch")
            calibration = dict(state["metrics"]["distance_calibration"])
            vector = [
                float(value) for value in calibration["accepted_temperatures"]
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid fold calibration state: fold {fold}: {exc}"
            ) from exc
        if len(vector) != 4 or any(
            not np.isfinite(value) or value <= 0 for value in vector
        ):
            raise ValueError(
                f"fold {fold} distance temperatures must be four positive values"
            )
        calibration["accepted_temperatures"] = vector
        calibration["fitted_temperatures"] = [
            float(value) for value in calibration["fitted_temperatures"]
        ]
        folds[str(fold)] = calibration
        accepted_vectors.append(vector)
    final = np.median(np.asarray(accepted_vectors, dtype=np.float64), axis=0)
    return {
        "folds": folds,
        "final_temperatures": [float(value) for value in final],
        "aggregation": "elementwise_median_with_unsafe_fold_fallback_ones_v1",
    }


def verify_cv_artifacts(output_dir, expected_rows, expected_hashes, config):
    """Recompute release evidence from saved OOF rows before authorization."""
    rows = read_oof_csv(Path(output_dir) / "oof_predictions.csv")
    validate_oof_rows(rows, expected_rows)
    metrics = evaluate_predictions(rows)
    metrics.update(file_bootstrap_metrics(
        rows, iterations=config.bootstrap_iterations, seed=config.seed
    ))
    passed, reasons = evaluate_release(metrics)
    try:
        saved = json.loads(
            (Path(output_dir) / "cv_metrics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid saved CV metrics: {exc}") from exc
    recomputed_hash = stable_json_hash(round_metrics(metrics))
    if recomputed_hash != saved.get("recomputed_metrics_hash"):
        raise ValueError("saved OOF metrics do not match oof_predictions.csv")
    if expected_hashes != saved.get("fold_hashes"):
        raise ValueError("fold hashes changed after OOF evaluation")
    return {
        "passed": bool(passed),
        "reasons": list(reasons),
        "metrics": metrics,
        "oof_piece_count": len(rows),
        "expected_piece_count": len(expected_rows),
    }


def evaluate_oof_artifacts(rows, expected, output_dir, config) -> dict:
    """Calibrate pooled OOF rows, write artifacts, and return verified gates."""
    rows = [dict(row) for row in rows]
    validate_oof_rows(rows, expected)
    fold_hashes = _fold_hashes_from_rows(rows)
    fold_temperatures = {}
    predicted = np.empty(len(rows), dtype=np.int64)
    confidence = np.empty(len(rows), dtype=np.float64)
    margin = np.empty(len(rows), dtype=np.float64)
    probabilities = np.empty((len(rows), len(TYPE_NAMES)), dtype=np.float64)
    for fold in range(3):
        positions = [
            index for index, row in enumerate(rows)
            if int(row["fold"]) == fold
        ]
        logits = torch.as_tensor([
            [float(rows[index][f"logit_{name}"]) for name in TYPE_NAMES]
            for index in positions
        ], dtype=torch.float32)
        labels = torch.as_tensor([
            int(rows[index]["true_type"]) for index in positions
        ], dtype=torch.long)
        temperature = _fit_temperature(logits, labels)
        fold_temperatures[fold] = float(temperature)
        fold_probabilities = (logits / temperature).softmax(dim=1)
        fold_predicted = fold_probabilities.argmax(dim=1)
        top_two = fold_probabilities.topk(2, dim=1).values
        for local_index, row_index in enumerate(positions):
            predicted[row_index] = int(fold_predicted[local_index].item())
            probabilities[row_index] = fold_probabilities[local_index].numpy()
            confidence[row_index] = float(top_two[local_index, 0].item())
            margin[row_index] = float(
                (top_two[local_index, 0] - top_two[local_index, 1]).item()
            )

    signals = {
        "true_type": [int(row["true_type"]) for row in rows],
        "predicted": predicted,
        "file_id": [str(row["source_path"]) for row in rows],
        "confidence": confidence,
        "margin": margin,
        "normalized_distance": [
            float(row["normalized_feature_distance"]) for row in rows
        ],
        "quality_score": [float(row["quality_score"]) for row in rows],
        "piece_key": [str(row["piece_key"]) for row in rows],
    }
    policy, decisions = fit_oof_rejection_policy(
        signals,
        target_precision=config.rejection_target_precision,
        min_coverage=config.rejection_min_coverage,
        fold_temperatures=fold_temperatures,
        fold_hashes=fold_hashes,
    )
    final_rows = []
    for index, raw in enumerate(rows):
        row = {name: raw.get(name, "") for name in OOF_FIELDS}
        row["predicted_type"] = int(decisions["predicted"][index].item())
        row["final_type"] = int(decisions["final_type"][index].item())
        row["accepted"] = bool(decisions["accepted"][index].item())
        row["rejection_reason"] = decisions["reason"][index]
        row["confidence"] = float(decisions["confidence"][index].item())
        row["margin"] = float(decisions["margin"][index].item())
        row["normalized_feature_distance"] = float(
            decisions["normalized_distance"][index].item()
        )
        row["quality_score"] = float(decisions["quality_score"][index].item())
        for type_index, type_name in enumerate(TYPE_NAMES):
            row[f"prob_{type_name}"] = float(probabilities[index, type_index])
        final_rows.append(row)
    final_rows.sort(key=lambda row: str(row["piece_key"]))
    output_dir = Path(output_dir)
    _atomic_write_csv(output_dir / "oof_predictions.csv", final_rows, OOF_FIELDS)

    metrics = evaluate_predictions(final_rows)
    metrics.update(file_bootstrap_metrics(
        final_rows, iterations=config.bootstrap_iterations, seed=config.seed
    ))
    distance_calibration = _distance_calibration_summary(
        output_dir, fold_hashes
    )
    saved = {
        "schema": "conditional_expert_cv_metrics_v1",
        "rounding_decimals": 12,
        "fold_hashes": fold_hashes,
        "type_rejection": policy,
        "distance_calibration": distance_calibration,
        "metrics": round_metrics(metrics),
        "recomputed_metrics_hash": stable_json_hash(round_metrics(metrics)),
        "historical_metrics": {
            "status": "reference_only",
            "used_for_release_gate": False,
        },
    }
    _atomic_write_json(output_dir / "cv_metrics.json", saved)
    verified = verify_cv_artifacts(output_dir, expected, fold_hashes, config)
    saved["verified_release"] = {
        "passed": verified["passed"],
        "reasons": verified["reasons"],
        "source": "verify_cv_artifacts_v1",
    }
    _atomic_write_json(output_dir / "cv_metrics.json", saved)
    return verified


def run_cross_validated_training(
    config: CVConfig,
    entries: list[ManifestEntry],
    device,
    fold_trainer=train_conditional_fold,
    final_trainer=None,
    oof_evaluator=evaluate_oof_artifacts,
) -> dict:
    """Run verified folds, OOF release, and optionally final training."""
    if int(config.folds) != 3:
        raise ValueError("cross-validation requires exactly three folds")
    folds_directory = Path(config.output) / "folds"
    if (
        not config.resume_cv
        and folds_directory.is_dir()
        and any(path.is_file() for path in folds_directory.rglob("*"))
    ):
        raise ValueError(
            "fold training artifacts already exist; pass --resume_cv to verify them"
        )
    folds = assign_exact_folds(entries, n_folds=config.folds, seed=config.seed)
    validate_fold_assignment(folds, entries, n_folds=config.folds)
    fold_manifest = make_fold_manifest(folds, config.task_data, config.seed)
    expected = expected_oof_rows(fold_manifest)
    _atomic_write_json(Path(config.output) / "fold_manifest.json", fold_manifest)
    config_hash = training_config_hash(config)
    results = []
    rows = []
    for fold_index in range(config.folds):
        train_entries, holdout_entries = fold_train_holdout(folds, fold_index)
        expected_hashes = {
            "train_hash": split_hash(train_entries, config.task_data),
            "holdout_hash": split_hash(holdout_entries, config.task_data),
            "config_hash": config_hash,
        }
        expected_fold_rows = {
            key: value
            for key, value in expected.items()
            if int(value["fold"]) == fold_index
        }
        result = None
        if config.resume_cv:
            resumed = load_verified_fold(
                config.output, fold_index, expected_hashes, expected_fold_rows
            )
            if isinstance(resumed, FoldResult):
                result = resumed
        if result is None:
            result = fold_trainer(
                train_entries,
                holdout_entries,
                Path(config.output) / "folds" / f"fold_{fold_index}",
                config,
                device,
                fold_index,
            )
        results.append(result)
        rows.extend(read_oof_csv(result.oof_path))

    validate_oof_rows(rows, expected)
    report = oof_evaluator(rows, expected, config.output, config)
    report.setdefault("oof_piece_count", len(rows))
    report.setdefault("expected_piece_count", len(expected))
    report["fold_results"] = [dataclasses.asdict(result) for result in results]
    report["median_best_epoch"] = int(np.median([
        int(result.best_epoch) for result in results
    ]))
    if (
        report.get("passed")
        and not config.stop_after_oof
        and final_trainer is not None
    ):
        report["final_result"] = final_trainer(entries, config, device, results)
    return report
