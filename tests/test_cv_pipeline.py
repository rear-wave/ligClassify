import csv
import json
from datetime import datetime
from pathlib import Path

import pytest
import torch

from cv_pipeline import (
    CVConfig,
    FoldResult,
    OOF_FIELDS,
    evaluate_oof_artifacts,
    load_verified_fold,
    read_oof_csv,
    run_cross_validated_training,
    training_config_hash,
    validate_fold_checkpoint,
    verify_cv_artifacts,
)
from data.cross_validation import assign_exact_folds, fold_train_holdout
from data.oof_manifest import oof_row_id
from data.split_artifacts import split_hash
from data.training_manifest import ManifestEntry
from data.training_manifest import build_manifest
from tests.test_training_manifest import write_lig


def trusted_entries(tmp_path):
    rows = []
    for type_index, type_name in enumerate(("NCG", "NNBE", "PCG", "PNBE")):
        for file_index in range(3):
            rows.append(ManifestEntry(
                filepath=str(
                    tmp_path / "train" / type_name / "day" / "0-100km"
                    / f"file-{file_index}.lig"
                ),
                type_idx=type_index,
                dist_bin=0,
                timestamp=datetime(2020, 1, 1),
                n_pieces=2,
                distance_low_km=0,
                distance_high_km=100,
                is_daytime=True,
            ))
    return rows


def config(tmp_path, resume_cv=False, stop_after_oof=True):
    return CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        folds=3,
        seed=7,
        resume_cv=resume_cv,
        stop_after_oof=stop_after_oof,
    )


def fake_fold_trainer(calls):
    def train(train_entries, holdout_entries, output_dir, config, device, fold_index):
        calls.append(fold_index)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        oof_path = output_dir / "oof.csv"
        fields = ["piece_key", "fold", "true_type"]
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for entry in holdout_entries:
                relative = Path(entry.filepath).relative_to(config.task_data).as_posix()
                for piece_index in range(entry.n_pieces):
                    writer.writerow({
                        "piece_key": oof_row_id(relative, piece_index),
                        "fold": fold_index,
                        "true_type": entry.type_idx,
                    })
        return FoldResult(
            fold_index=fold_index,
            best_epoch=fold_index + 2,
            train_hash=split_hash(train_entries, config.task_data),
            holdout_hash=split_hash(holdout_entries, config.task_data),
            config_hash=training_config_hash(config),
            checkpoint_path=str(output_dir / "best.pt"),
            oof_path=str(oof_path),
            metrics={},
        )
    return train


def passing_oof_evaluator(rows, expected, output_dir, config):
    return {
        "passed": True,
        "reasons": [],
        "oof_piece_count": len(rows),
        "expected_piece_count": len(expected),
    }


def failing_oof_evaluator(rows, expected, output_dir, config):
    return {
        "passed": False,
        "reasons": ["type_file_equal_precision[0]=0.90 below 0.95"],
        "oof_piece_count": len(rows),
        "expected_piece_count": len(expected),
    }


def fail_if_called(*args, **kwargs):
    raise AssertionError("final trainer must not be called")


def test_cv_runs_each_fold_once_and_validates_all_oof_rows(tmp_path):
    calls = []
    report = run_cross_validated_training(
        config(tmp_path), trusted_entries(tmp_path), device="cpu",
        fold_trainer=fake_fold_trainer(calls), final_trainer=fail_if_called,
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [0, 1, 2]
    assert report["oof_piece_count"] == report["expected_piece_count"]
    assert report["median_best_epoch"] == 3
    saved_manifest = json.loads(
        (config(tmp_path).output / "fold_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["schema"] == "file_isolated_exact_interval_cv_v1"
    assert saved_manifest["fold_count"] == 3


def test_resume_skips_only_hash_verified_complete_folds(tmp_path, monkeypatch):
    entries = trusted_entries(tmp_path)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    train_entries, holdout_entries = fold_train_holdout(folds, 0)
    completed = fake_fold_trainer([])(
        train_entries,
        holdout_entries,
        config(tmp_path).output / "folds" / "fold_0",
        config(tmp_path),
        "cpu",
        0,
    )
    monkeypatch.setattr(
        "cv_pipeline.load_verified_fold",
        lambda output_dir, fold_index, expected_hashes, expected_rows: (
            completed if fold_index == 0 else None
        ),
    )
    calls = []
    run_cross_validated_training(
        config(tmp_path, resume_cv=True), entries, "cpu",
        fold_trainer=fake_fold_trainer(calls),
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [1, 2]


def test_resume_hash_mismatch_false_does_not_skip_fold(tmp_path, monkeypatch):
    monkeypatch.setattr("cv_pipeline.load_verified_fold", lambda *args: False)
    calls = []
    run_cross_validated_training(
        config(tmp_path, resume_cv=True), trusted_entries(tmp_path), "cpu",
        fold_trainer=fake_fold_trainer(calls),
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [0, 1, 2]


def test_failed_oof_gate_never_calls_final_trainer_or_replaces_model(tmp_path):
    deployed = config(tmp_path, stop_after_oof=False).output / "model.pt"
    deployed.parent.mkdir(parents=True)
    deployed.write_bytes(b"existing")
    run_cross_validated_training(
        config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
        fold_trainer=fake_fold_trainer([]), final_trainer=fail_if_called,
        oof_evaluator=failing_oof_evaluator,
    )
    assert deployed.read_bytes() == b"existing"


def test_training_config_hash_excludes_paths_and_run_control_flags(tmp_path):
    first = config(tmp_path, resume_cv=False, stop_after_oof=False)
    second = CVConfig(
        **{
            **first.__dict__,
            "task_data": tmp_path / "other-data",
            "output": tmp_path / "other-output",
            "resume_cv": True,
            "stop_after_oof": True,
        }
    )
    assert training_config_hash(first) == training_config_hash(second)


def test_cv_requires_exactly_three_folds(tmp_path):
    invalid = CVConfig(
        task_data=tmp_path / "train", output=tmp_path / "weights", folds=2
    )
    with pytest.raises(ValueError, match="exactly three folds"):
        run_cross_validated_training(
            invalid, trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]),
            oof_evaluator=passing_oof_evaluator,
        )


def _one_expected_row():
    return {
        "NCG/day/0-100km/file.lig#0": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/file.lig",
            "piece_index": 0,
        }
    }


def _write_minimal_oof(path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "piece_key", "source_path", "piece_index", "fold", "true_type"
        ))
        writer.writeheader()
        writer.writerow({
            "piece_key": "NCG/day/0-100km/file.lig#0",
            "source_path": "NCG/day/0-100km/file.lig",
            "piece_index": 0,
            "fold": 0,
            "true_type": 0,
        })


def test_read_oof_csv_converts_bound_fields(tmp_path):
    path = tmp_path / "oof.csv"
    row = {name: "" for name in OOF_FIELDS}
    row.update({
        "piece_key": "x#0",
        "piece_index": "0",
        "fold": "1",
        "true_type": "2",
        "predicted_type": "3",
        "final_type": "-1",
        "accepted": "False",
        "daylight": "1",
        "logit_NCG": "0.25",
        "prob_PNBE": "0.75",
        "oracle_distance_km": "",
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    parsed = read_oof_csv(path)[0]
    assert parsed["fold"] == 1
    assert parsed["final_type"] == -1
    assert parsed["accepted"] is False
    assert parsed["daylight"] is True
    assert parsed["logit_NCG"] == 0.25
    assert parsed["prob_PNBE"] == 0.75
    assert parsed["oracle_distance_km"] is None


def test_read_oof_csv_rejects_invalid_boolean(tmp_path):
    path = tmp_path / "oof.csv"
    path.write_text("piece_key,accepted\nx#0,yes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid OOF boolean accepted"):
        read_oof_csv(path)


def test_validate_fold_checkpoint_rejects_wrong_schema():
    with pytest.raises(ValueError, match="invalid fold checkpoint schema"):
        validate_fold_checkpoint({}, 0, {
            "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
        })


def test_load_verified_fold_returns_verified_complete_result(tmp_path, monkeypatch):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    hashes = {
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
    }
    state = {
        "status": "complete",
        "best_epoch": 4,
        "metrics": {"score": 1.0},
        **hashes,
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save({"checkpoint": True, "best_epoch": 4}, directory / "best.pt")
    _write_minimal_oof(directory / "oof.csv")
    monkeypatch.setattr("cv_pipeline.validate_fold_checkpoint", lambda *args: None)

    result = load_verified_fold(tmp_path, 0, hashes, _one_expected_row())

    assert isinstance(result, FoldResult)
    assert result.best_epoch == 4
    assert result.metrics == {"score": 1.0}


def test_load_verified_fold_returns_false_for_hash_mismatch(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    state = {
        "status": "complete", "best_epoch": 1, "metrics": {},
        "train_hash": "old", "holdout_hash": "holdout", "config_hash": "config",
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    (directory / "best.pt").write_bytes(b"present")
    _write_minimal_oof(directory / "oof.csv")
    assert load_verified_fold(tmp_path, 0, {
        "train_hash": "new", "holdout_hash": "holdout", "config_hash": "config"
    }, _one_expected_row()) is False


def test_load_verified_fold_rejects_best_epoch_mismatch(tmp_path, monkeypatch):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    hashes = {
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
    }
    state = {
        "status": "complete", "best_epoch": 4, "metrics": {}, **hashes,
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save({"best_epoch": 3}, directory / "best.pt")
    _write_minimal_oof(directory / "oof.csv")
    monkeypatch.setattr("cv_pipeline.validate_fold_checkpoint", lambda *args: None)

    with pytest.raises(ValueError, match="best_epoch mismatch"):
        load_verified_fold(tmp_path, 0, hashes, _one_expected_row())


def test_load_verified_fold_rejects_missing_completed_artifact(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    state = {
        "status": "complete", "best_epoch": 1, "metrics": {},
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config",
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="completed fold is missing artifacts: fold 0"):
        load_verified_fold(tmp_path, 0, state, _one_expected_row())


def test_without_resume_existing_fold_artifact_aborts(tmp_path):
    artifact = config(tmp_path).output / "folds" / "fold_0" / "latest.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"old")
    with pytest.raises(ValueError, match="--resume_cv"):
        run_cross_validated_training(
            config(tmp_path), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]),
            oof_evaluator=passing_oof_evaluator,
        )


def test_train_conditional_fold_writes_resumable_checkpoint_and_raw_oof(
    tmp_path, monkeypatch
):
    from cv_pipeline import train_conditional_fold

    data_root = tmp_path / "train"
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for file_index in range(3):
            write_lig(
                data_root / type_name / "day" / "0-100km"
                / f"file-{file_index}.lig"
            )
    entries, _ = build_manifest(data_root, ("NCG", "NNBE", "PCG", "PNBE"))
    train_entries = [entry for entry in entries if "file-2.lig" not in entry.filepath]
    holdout_entries = [entry for entry in entries if "file-2.lig" in entry.filepath]
    fold_config = CVConfig(
        task_data=data_root,
        output=tmp_path / "weights",
        samples_per_epoch=8,
        max_samples_per_file=2,
        type_focus_epochs=1,
        max_epochs=2,
        patience=1,
        num_workers=0,
        no_amp=True,
    )
    monkeypatch.setattr("cv_pipeline.FOLD_BATCH_SIZE", 4)
    monkeypatch.setattr("cv_pipeline.FOLD_MODEL_CONFIG", {
        "base_channels": 1,
        "architecture": "conditional_expert_v1",
        "num_types": 4,
        "context_dim": 1,
        "dist_mlp_dim": 2,
        "dist_dropout": 0.0,
    })

    result = train_conditional_fold(
        train_entries,
        holdout_entries,
        fold_config.output / "folds" / "fold_0",
        fold_config,
        "cpu",
        0,
    )

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    state = json.loads(
        (fold_config.output / "folds" / "fold_0" / "fold_state.json")
        .read_text(encoding="utf-8")
    )
    rows = read_oof_csv(result.oof_path)
    assert result.best_epoch == 2
    assert checkpoint["schema"] == "conditional_expert_cv_fold_v1"
    assert checkpoint["random_initialization"] is True
    assert checkpoint["best_epoch"] == 2
    assert state["status"] == "complete"
    assert len(rows) == 4
    assert set(OOF_FIELDS).issubset(rows[0])
    assert all(row["source_path"].endswith("file-2.lig") for row in rows)


def test_completed_hash_mismatch_retrains_from_random_epoch_zero(
    tmp_path, monkeypatch
):
    import cv_pipeline

    data_root = tmp_path / "train"
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for file_index in range(3):
            write_lig(
                data_root / type_name / "day" / "0-100km"
                / f"file-{file_index}.lig"
            )
    entries, _ = build_manifest(data_root, ("NCG", "NNBE", "PCG", "PNBE"))
    train_entries = [entry for entry in entries if "file-2.lig" not in entry.filepath]
    holdout_entries = [entry for entry in entries if "file-2.lig" in entry.filepath]
    first_config = CVConfig(
        task_data=data_root,
        output=tmp_path / "weights",
        samples_per_epoch=8,
        max_samples_per_file=2,
        type_focus_epochs=1,
        max_epochs=2,
        patience=1,
        num_workers=0,
        no_amp=True,
    )
    monkeypatch.setattr("cv_pipeline.FOLD_BATCH_SIZE", 4)
    monkeypatch.setattr("cv_pipeline.FOLD_MODEL_CONFIG", {
        "base_channels": 1,
        "architecture": "conditional_expert_v1",
        "num_types": 4,
        "context_dim": 1,
        "dist_mlp_dim": 2,
        "dist_dropout": 0.0,
    })
    output_dir = first_config.output / "folds" / "fold_0"
    cv_pipeline.train_conditional_fold(
        train_entries, holdout_entries, output_dir, first_config, "cpu", 0
    )
    original_epoch = cv_pipeline.run_conditional_epoch
    seen_epochs = []

    def recording_epoch(*args, **kwargs):
        seen_epochs.append(int(args[7]))
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(cv_pipeline, "run_conditional_epoch", recording_epoch)
    changed_config = CVConfig(
        **{
            **first_config.__dict__,
            "max_epochs": 3,
            "resume_cv": True,
        }
    )
    cv_pipeline.train_conditional_fold(
        train_entries, holdout_entries, output_dir, changed_config, "cpu", 0
    )

    assert seen_epochs == [0, 1, 2]


def _perfect_raw_oof(tmp_path):
    rows = []
    expected = {}
    hashes = {}
    for fold in range(3):
        fold_hash = {
            "train_hash": f"train-{fold}",
            "holdout_hash": f"holdout-{fold}",
            "config_hash": "config",
        }
        hashes[str(fold)] = fold_hash
        directory = tmp_path / "weights" / "folds" / f"fold_{fold}"
        directory.mkdir(parents=True)
        fitted = [2.0 + fold] * 4
        accepted = [1.0] * 4 if fold == 1 else fitted
        state = {
            "status": "complete",
            "best_epoch": fold + 2,
            "metrics": {
                "distance_calibration": {
                    "before": {"distance_interval_mae_km": 0.0},
                    "after": {"distance_interval_mae_km": 0.0},
                    "fitted_temperatures": fitted,
                    "accepted_temperatures": accepted,
                    "accepted": fold != 1,
                    "guard_reason": "accepted" if fold != 1 else "point_metrics_regressed",
                }
            },
            **fold_hash,
        }
        (directory / "fold_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        for type_index, type_name in enumerate(("NCG", "NNBE", "PCG", "PNBE")):
            source_path = f"{type_name}/day/0-100km/fold-{fold}.lig"
            piece_key = f"{source_path}#0"
            expected[piece_key] = {
                "fold": fold,
                "type_idx": type_index,
                "source_path": source_path,
                "piece_index": 0,
            }
            row = {name: "" for name in OOF_FIELDS}
            row.update({
                "piece_key": piece_key,
                "source_path": source_path,
                "piece_index": 0,
                "fold": fold,
                "true_type": type_index,
                "predicted_type": type_index,
                "final_type": type_index,
                "accepted": True,
                "rejection_reason": "uncalibrated",
                "normalized_feature_distance": 0.1,
                "quality_score": 1.0,
                "distance_low_km": 0.0,
                "distance_high_km": 100.0,
                "predicted_distance_km": 50.0,
                "oracle_distance_km": 50.0,
                "distance_temperature": accepted[type_index],
                "daylight": True,
                "support_status": "supported",
                **fold_hash,
            })
            for logit_index, logit_name in enumerate(("NCG", "NNBE", "PCG", "PNBE")):
                row[f"logit_{logit_name}"] = 8.0 if logit_index == type_index else 0.0
                row[f"prob_{logit_name}"] = ""
            rows.append(row)
    return rows, expected, hashes


def test_evaluate_oof_writes_verified_csv_metrics_and_guarded_medians(tmp_path):
    rows, expected, hashes = _perfect_raw_oof(tmp_path)
    fold_config = CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        bootstrap_iterations=10,
        stop_after_oof=True,
    )

    report = evaluate_oof_artifacts(rows, expected, fold_config.output, fold_config)

    final_rows = read_oof_csv(fold_config.output / "oof_predictions.csv")
    saved = json.loads(
        (fold_config.output / "cv_metrics.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert len(final_rows) == len(expected) == 12
    assert tuple(final_rows[0]) == OOF_FIELDS
    assert all(row["accepted"] for row in final_rows)
    assert saved["fold_hashes"] == hashes
    assert saved["distance_calibration"]["final_temperatures"] == [2.0] * 4
    assert saved["distance_calibration"]["folds"]["1"]["accepted_temperatures"] == [1.0] * 4
    assert saved["type_rejection"]["version"] == 3
    assert saved["historical_metrics"]["status"] == "reference_only"
    assert saved["verified_release"] == {
        "passed": True,
        "reasons": [],
        "source": "verify_cv_artifacts_v1",
    }


def test_verify_cv_artifacts_rejects_tampered_final_csv(tmp_path):
    rows, expected, hashes = _perfect_raw_oof(tmp_path)
    fold_config = CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        bootstrap_iterations=5,
        stop_after_oof=True,
    )
    evaluate_oof_artifacts(rows, expected, fold_config.output, fold_config)
    final_path = fold_config.output / "oof_predictions.csv"
    final_rows = read_oof_csv(final_path)
    final_rows[0]["predicted_type"] = 1
    _write_rows = []
    for row in final_rows:
        _write_rows.append({
            name: "" if row.get(name) is None else row.get(name, "")
            for name in OOF_FIELDS
        })
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerows(_write_rows)

    with pytest.raises(ValueError, match="saved OOF metrics"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_cv_artifacts_rejects_changed_fold_hashes(tmp_path):
    rows, expected, hashes = _perfect_raw_oof(tmp_path)
    fold_config = CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        bootstrap_iterations=5,
        stop_after_oof=True,
    )
    evaluate_oof_artifacts(rows, expected, fold_config.output, fold_config)
    changed = json.loads(json.dumps(hashes))
    changed["0"]["train_hash"] = "changed"
    with pytest.raises(ValueError, match="fold hashes changed"):
        verify_cv_artifacts(fold_config.output, expected, changed, fold_config)
