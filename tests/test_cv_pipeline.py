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
from data.split_artifacts import make_fold_manifest, split_hash
from data.training_manifest import ManifestEntry
from data.training_manifest import build_manifest
from models import create_mtl_model
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
        bootstrap_iterations=5,
    )


def _distance_calibration_fixture(fitted=None, accepted=None, *, safe=True):
    fitted = [1.0] * 4 if fitted is None else list(fitted)
    accepted = fitted if accepted is None else list(accepted)
    before = {
        "distance_file_macro_within_200": 1.0,
        "distance_100km_interval_within_200": 1.0,
        "distance_interval_mae_km": 0.0,
    }
    after = dict(before) if safe else {
        "distance_file_macro_within_200": 0.0,
        "distance_100km_interval_within_200": 0.0,
        "distance_interval_mae_km": 1.0,
    }
    return {
        "before": before,
        "after": after,
        "fitted_temperatures": fitted,
        "accepted_temperatures": accepted,
        "accepted": bool(safe),
        "guard_reason": "accepted" if safe else "point_metrics_regressed",
    }


def fake_fold_trainer(calls):
    def train(train_entries, holdout_entries, output_dir, config, device, fold_index):
        calls.append(fold_index)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        oof_path = output_dir / "oof.csv"
        train_hash = split_hash(train_entries, config.task_data)
        holdout_hash = split_hash(holdout_entries, config.task_data)
        config_hash = training_config_hash(config)
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
            writer.writeheader()
            for entry in holdout_entries:
                relative = Path(entry.filepath).relative_to(config.task_data).as_posix()
                for piece_index in range(entry.n_pieces):
                    writer.writerow(_complete_oof_row(
                        source_path=relative,
                        piece_index=piece_index,
                        fold=fold_index,
                        true_type=entry.type_idx,
                        train_hash=train_hash,
                        holdout_hash=holdout_hash,
                        config_hash=config_hash,
                    ))
        fold_hashes = {
            "train_hash": train_hash,
            "holdout_hash": holdout_hash,
            "config_hash": config_hash,
        }
        fold_metrics = {
            "distance_calibration": _distance_calibration_fixture()
        }
        checkpoint = _checkpoint_fixture(
            fold_hashes,
            fold_index=fold_index,
            best_epoch=fold_index + 2,
            metrics=fold_metrics,
        )
        torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "fold_state.json").write_text(json.dumps({
            "status": "complete",
            "fold_index": fold_index,
            **fold_hashes,
            "random_initialization": True,
            "best_epoch": checkpoint["best_epoch"],
            "stage_config": checkpoint["stage_config"],
            "model_config": checkpoint["model_config"],
            "metrics": fold_metrics,
        }), encoding="utf-8")
        return FoldResult(
            fold_index=fold_index,
            best_epoch=fold_index + 2,
            train_hash=train_hash,
            holdout_hash=holdout_hash,
            config_hash=config_hash,
            checkpoint_path=str(output_dir / "best.pt"),
            oof_path=str(oof_path),
            metrics=fold_metrics,
        )
    return train


def _write_fake_fold_calibrations(rows, output_dir):
    for fold in range(3):
        fold_row = next(row for row in rows if int(row["fold"]) == fold)
        fold_hashes = {
            name: str(fold_row[name])
            for name in ("train_hash", "holdout_hash", "config_hash")
        }
        directory = Path(output_dir) / "folds" / f"fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fold_state.json").write_text(json.dumps({
            "status": "complete",
            "fold_index": fold,
            **fold_hashes,
            "metrics": {
                "distance_calibration": _distance_calibration_fixture()
            },
        }), encoding="utf-8")


def passing_oof_evaluator(rows, expected, output_dir, config):
    return evaluate_oof_artifacts(rows, expected, output_dir, config)


def failing_oof_evaluator(rows, expected, output_dir, config):
    return evaluate_oof_artifacts(rows, expected, output_dir, config)


def failing_distance_fold_trainer(calls):
    trainer = fake_fold_trainer(calls)

    def train(*args, **kwargs):
        result = trainer(*args, **kwargs)
        rows = read_oof_csv(result.oof_path)
        for row in rows:
            row["predicted_distance_km"] = 5000.0
            row["oracle_distance_km"] = 5000.0
        _write_oof_rows(Path(result.oof_path), rows)
        return result

    return train


def fail_if_called(*args, **kwargs):
    raise AssertionError("final trainer must not be called")


def test_evaluator_true_without_artifacts_cannot_authorize_final_training(tmp_path):
    final_calls = []

    def malicious_evaluator(*args, **kwargs):
        return {"passed": True, "reasons": []}

    def final_trainer(*args, **kwargs):
        final_calls.append(True)

    with pytest.raises(ValueError, match="OOF artifact verification failed"):
        run_cross_validated_training(
            config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]), final_trainer=final_trainer,
            oof_evaluator=malicious_evaluator,
        )
    assert final_calls == []


def test_evaluator_true_after_tampering_cannot_authorize_final_training(tmp_path):
    final_calls = []

    def malicious_evaluator(rows, expected, output_dir, fold_config):
        passing_oof_evaluator(rows, expected, output_dir, fold_config)
        metrics_path = Path(output_dir) / "cv_metrics.json"
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved["type_rejection"]["calibration_hash"] = "tampered"
        metrics_path.write_text(json.dumps(saved), encoding="utf-8")
        return {"passed": True, "reasons": []}

    def final_trainer(*args, **kwargs):
        final_calls.append(True)

    with pytest.raises(ValueError, match="OOF artifact verification failed"):
        run_cross_validated_training(
            config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]), final_trainer=final_trainer,
            oof_evaluator=malicious_evaluator,
        )
    assert final_calls == []


def test_evaluator_cannot_redefine_expected_fold_hashes(tmp_path):
    final_calls = []

    def malicious_evaluator(rows, expected, output_dir, fold_config):
        for row in rows:
            row["train_hash"] = f"attacker-{int(row['fold'])}"
        _write_fake_fold_calibrations(rows, output_dir)
        evaluate_oof_artifacts(rows, expected, output_dir, fold_config)
        return {"passed": True, "reasons": []}

    def final_trainer(*args, **kwargs):
        final_calls.append(True)

    with pytest.raises(ValueError, match="OOF artifact verification failed"):
        run_cross_validated_training(
            config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]), final_trainer=final_trainer,
            oof_evaluator=malicious_evaluator,
        )
    assert final_calls == []


def test_evaluator_cannot_rewrite_canonical_fold_artifact_bytes(tmp_path):
    final_calls = []

    def malicious_evaluator(rows, expected, output_dir, fold_config):
        passing_oof_evaluator(rows, expected, output_dir, fold_config)
        raw_path = Path(output_dir) / "folds" / "fold_0" / "oof.csv"
        raw_path.write_bytes(raw_path.read_bytes() + b"\r\n")
        return {"passed": True, "reasons": []}

    def final_trainer(*args, **kwargs):
        final_calls.append(True)

    with pytest.raises(ValueError, match="OOF artifact verification failed"):
        run_cross_validated_training(
            config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]), final_trainer=final_trainer,
            oof_evaluator=malicious_evaluator,
        )
    assert final_calls == []


def test_evaluator_false_return_cannot_veto_verified_artifacts(tmp_path):
    final_calls = []

    def lying_evaluator(rows, expected, output_dir, fold_config):
        passing_oof_evaluator(rows, expected, output_dir, fold_config)
        return {"passed": False, "reasons": ["fabricated"]}

    def final_trainer(*args, **kwargs):
        final_calls.append(True)
        return {"trained": True}

    report = run_cross_validated_training(
        config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
        fold_trainer=fake_fold_trainer([]), final_trainer=final_trainer,
        oof_evaluator=lying_evaluator,
    )

    assert report["passed"] is True
    assert report["final_result"] == {"trained": True}
    assert final_calls == [True]


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


def test_resume_skips_only_hash_verified_complete_folds(tmp_path):
    entries = trusted_entries(tmp_path)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    train_entries, holdout_entries = fold_train_holdout(folds, 0)
    fake_fold_trainer([])(
        train_entries,
        holdout_entries,
        config(tmp_path).output / "folds" / "fold_0",
        config(tmp_path),
        "cpu",
        0,
    )
    calls = []
    run_cross_validated_training(
        config(tmp_path, resume_cv=True), entries, "cpu",
        fold_trainer=fake_fold_trainer(calls),
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [1, 2]


def test_resume_hash_mismatch_false_does_not_skip_fold(tmp_path):
    entries = trusted_entries(tmp_path)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    train_entries, holdout_entries = fold_train_holdout(folds, 0)
    output_dir = config(tmp_path).output / "folds" / "fold_0"
    fake_fold_trainer([])(
        train_entries, holdout_entries, output_dir,
        config(tmp_path), "cpu", 0,
    )
    state_path = output_dir / "fold_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["config_hash"] = "mismatch"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    run_cross_validated_training(
        config(tmp_path, resume_cv=True), entries, "cpu",
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
        fold_trainer=failing_distance_fold_trainer([]), final_trainer=fail_if_called,
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


def _complete_oof_row(
    *, source_path="NCG/day/0-100km/file.lig", piece_index=0, fold=0,
    true_type=0, train_hash="train", holdout_hash="holdout",
    config_hash="config",
):
    row = {name: "" for name in OOF_FIELDS}
    row.update({
        "piece_key": oof_row_id(source_path, piece_index),
        "source_path": source_path,
        "piece_index": piece_index,
        "fold": fold,
        "true_type": true_type,
        "predicted_type": true_type,
        "final_type": true_type,
        "accepted": True,
        "rejection_reason": "accepted",
        "confidence": 1.0,
        "margin": 1.0,
        "normalized_feature_distance": 0.0,
        "quality_score": 1.0,
        "distance_low_km": 0.0,
        "distance_high_km": 100.0,
        "predicted_distance_km": 50.0,
        "oracle_distance_km": 50.0,
        "distance_temperature": 1.0,
        "daylight": True,
        "support_status": "supported",
        "train_hash": train_hash,
        "holdout_hash": holdout_hash,
        "config_hash": config_hash,
    })
    for index, name in enumerate(("NCG", "NNBE", "PCG", "PNBE")):
        row[f"logit_{name}"] = 8.0 if index == true_type else 0.0
        row[f"prob_{name}"] = 1.0 if index == true_type else 0.0
    return row


def _checkpoint_fixture(
    hashes=None, *, fold_index=0, best_epoch=4, metrics=None
):
    hashes = hashes or {
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
    }
    model_config = {
        "base_channels": 1,
        "architecture": "conditional_expert_v1",
        "num_types": 4,
        "context_dim": 1,
        "dist_mlp_dim": 2,
        "dist_dropout": 0.0,
    }
    model = create_mtl_model(**model_config)
    return {
        "schema": "conditional_expert_cv_fold_v1",
        "fold_index": fold_index,
        **hashes,
        "random_initialization": True,
        "best_epoch": best_epoch,
        "stage_config": {
            "type_focus_epochs": 1,
            "type_focus_distance_weight": 0.25,
            "joint_distance_weight": 1.0,
            "max_epochs": 5,
            "patience": 2,
        },
        "model_config": model_config,
        "model_state": model.state_dict(),
        "metrics": {"score": 1.0} if metrics is None else metrics,
    }


def _write_minimal_oof(path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerow(_complete_oof_row())


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
    row = _complete_oof_row()
    row["accepted"] = "yes"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="invalid OOF boolean accepted"):
        read_oof_csv(path)


@pytest.mark.parametrize("fieldnames", [
    OOF_FIELDS[:-1],
    OOF_FIELDS + ("extra",),
    (OOF_FIELDS[1], OOF_FIELDS[0], *OOF_FIELDS[2:]),
])
def test_read_oof_csv_requires_exact_ordered_header(tmp_path, fieldnames):
    path = tmp_path / "oof.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    with pytest.raises(ValueError, match="OOF CSV header mismatch"):
        read_oof_csv(path)


def test_validate_fold_checkpoint_rejects_wrong_schema():
    with pytest.raises(ValueError, match="invalid fold checkpoint schema"):
        validate_fold_checkpoint({}, 0, {
            "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
        })


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(best_epoch=0), "best_epoch"),
        (lambda row: row.pop("stage_config"), "stage_config"),
        (lambda row: row.pop("metrics"), "metrics"),
        (lambda row: row.pop("model_config"), "model_config"),
        (lambda row: row.pop("model_state"), "model_state"),
    ],
)
def test_validate_fold_checkpoint_requires_complete_schema(mutation, message):
    checkpoint = _checkpoint_fixture()
    mutation(checkpoint)
    with pytest.raises(ValueError, match=message):
        validate_fold_checkpoint(checkpoint, 0, {
            "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
        })


def test_load_verified_fold_rejects_checkpoint_state_metrics_mismatch(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    checkpoint = _checkpoint_fixture()
    state = {
        "status": "complete",
        "fold_index": 0,
        "best_epoch": 4,
        "metrics": {"score": 2.0},
        "stage_config": checkpoint["stage_config"],
        "model_config": checkpoint["model_config"],
        "random_initialization": True,
        "train_hash": "train",
        "holdout_hash": "holdout",
        "config_hash": "config",
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save(checkpoint, directory / "best.pt")
    with (directory / "oof.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerow(_complete_oof_row())
    with pytest.raises(ValueError, match="metrics mismatch"):
        load_verified_fold(
            tmp_path, 0,
            {"train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"},
            _one_expected_row(),
        )


@pytest.mark.parametrize("field", ["stage_config", "model_config"])
def test_load_verified_fold_rejects_checkpoint_state_config_mismatch(
    tmp_path, field
):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    checkpoint = _checkpoint_fixture()
    state = {
        "status": "complete",
        "fold_index": 0,
        "best_epoch": checkpoint["best_epoch"],
        "metrics": checkpoint["metrics"],
        "stage_config": dict(checkpoint["stage_config"]),
        "model_config": dict(checkpoint["model_config"]),
        "random_initialization": True,
        "train_hash": "train",
        "holdout_hash": "holdout",
        "config_hash": "config",
    }
    if field == "stage_config":
        state[field]["patience"] += 1
    else:
        state[field]["base_channels"] += 1
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save(checkpoint, directory / "best.pt")
    _write_minimal_oof(directory / "oof.csv")

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        load_verified_fold(
            tmp_path, 0,
            {"train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"},
            _one_expected_row(),
        )


def test_fold_oof_requires_explicit_source_identity_and_hashes(tmp_path):
    entries = trusted_entries(tmp_path)

    def bad_trainer(train_entries, holdout_entries, output_dir, cfg, device, fold_index):
        result = fake_fold_trainer([])(
            train_entries, holdout_entries, output_dir, cfg, device, fold_index
        )
        rows = []
        for entry in holdout_entries:
            relative = Path(entry.filepath).relative_to(cfg.task_data).as_posix()
            for piece_index in range(entry.n_pieces):
                rows.append(_complete_oof_row(
                    source_path=relative,
                    piece_index=piece_index,
                    fold=fold_index,
                    true_type=entry.type_idx,
                    train_hash=result.train_hash,
                    holdout_hash=result.holdout_hash,
                    config_hash=result.config_hash,
                ))
        rows[0]["source_path"] = ""
        with Path(result.oof_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return result

    with pytest.raises(ValueError, match="OOF source_path is required"):
        run_cross_validated_training(
            config(tmp_path), entries, "cpu",
            fold_trainer=bad_trainer,
            oof_evaluator=passing_oof_evaluator,
        )


def test_load_verified_fold_returns_verified_complete_result(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    hashes = {
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
    }
    checkpoint = _checkpoint_fixture(hashes)
    state = {
        "status": "complete",
        "fold_index": 0,
        "best_epoch": 4,
        "metrics": {"score": 1.0},
        "stage_config": checkpoint["stage_config"],
        "model_config": checkpoint["model_config"],
        "random_initialization": True,
        **hashes,
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save(checkpoint, directory / "best.pt")
    _write_minimal_oof(directory / "oof.csv")

    result = load_verified_fold(tmp_path, 0, hashes, _one_expected_row())

    assert isinstance(result, FoldResult)
    assert result.best_epoch == 4
    assert result.metrics == {"score": 1.0}


def test_load_verified_fold_returns_false_for_hash_mismatch(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    checkpoint = _checkpoint_fixture()
    state = {
        "status": "complete", "best_epoch": 1, "metrics": {},
        "fold_index": 0,
        "stage_config": checkpoint["stage_config"],
        "model_config": checkpoint["model_config"],
        "random_initialization": True,
        "train_hash": "old", "holdout_hash": "holdout", "config_hash": "config",
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    (directory / "best.pt").write_bytes(b"present")
    _write_minimal_oof(directory / "oof.csv")
    assert load_verified_fold(tmp_path, 0, {
        "train_hash": "new", "holdout_hash": "holdout", "config_hash": "config"
    }, _one_expected_row()) is False


def test_load_verified_fold_rejects_best_epoch_mismatch(tmp_path):
    directory = tmp_path / "folds" / "fold_0"
    directory.mkdir(parents=True)
    hashes = {
        "train_hash": "train", "holdout_hash": "holdout", "config_hash": "config"
    }
    checkpoint = _checkpoint_fixture(hashes)
    checkpoint["best_epoch"] = 3
    state = {
        "status": "complete", "fold_index": 0, "best_epoch": 4,
        "metrics": checkpoint["metrics"],
        "stage_config": checkpoint["stage_config"],
        "model_config": checkpoint["model_config"],
        "random_initialization": True,
        **hashes,
    }
    (directory / "fold_state.json").write_text(json.dumps(state), encoding="utf-8")
    torch.save(checkpoint, directory / "best.pt")
    _write_minimal_oof(directory / "oof.csv")

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


@pytest.mark.parametrize("artifact_name", ["oof_predictions.csv", "cv_metrics.json"])
def test_without_resume_root_cv_artifact_aborts_before_training(
    tmp_path, artifact_name
):
    artifact = config(tmp_path).output / artifact_name
    artifact.parent.mkdir(parents=True)
    artifact.write_text("old", encoding="utf-8")
    calls = []
    with pytest.raises(ValueError, match="--resume_cv"):
        run_cross_validated_training(
            config(tmp_path), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer(calls),
            oof_evaluator=passing_oof_evaluator,
        )
    assert calls == []


def test_without_resume_mismatched_fold_manifest_aborts(tmp_path):
    manifest_path = config(tmp_path).output / "fold_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"schema":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="fold_manifest.json does not match"):
        run_cross_validated_training(
            config(tmp_path), trusted_entries(tmp_path), "cpu",
            fold_trainer=fake_fold_trainer([]),
            oof_evaluator=passing_oof_evaluator,
        )


def test_without_resume_reuses_identical_audit_fold_manifest_without_rewrite(
    tmp_path
):
    entries = trusted_entries(tmp_path)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    manifest = make_fold_manifest(folds, config(tmp_path).task_data, seed=7)
    manifest_path = config(tmp_path).output / "fold_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    original = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_path.write_text(original, encoding="utf-8")

    run_cross_validated_training(
        config(tmp_path), entries, "cpu",
        fold_trainer=fake_fold_trainer([]),
        oof_evaluator=passing_oof_evaluator,
    )

    assert manifest_path.read_text(encoding="utf-8") == original


def _real_fold_case(tmp_path, monkeypatch, *, max_epochs=2, patience=1):
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
        max_epochs=max_epochs,
        patience=patience,
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
    return fold_config, train_entries, holdout_entries


def test_train_conditional_fold_writes_resumable_checkpoint_and_raw_oof(
    tmp_path, monkeypatch
):
    from cv_pipeline import train_conditional_fold

    fold_config, train_entries, holdout_entries = _real_fold_case(
        tmp_path, monkeypatch
    )

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
    latest = torch.load(
        fold_config.output / "folds" / "fold_0" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert result.best_epoch == 2
    assert checkpoint["schema"] == "conditional_expert_cv_fold_v1"
    assert checkpoint["random_initialization"] is True
    assert checkpoint["best_epoch"] == 2
    assert state["status"] == "complete"
    assert len(rows) == 4
    assert latest["execution_device_type"] == "cpu"
    assert set(OOF_FIELDS).issubset(rows[0])
    assert all(row["source_path"].endswith("file-2.lig") for row in rows)


def test_resume_with_progress_but_missing_latest_checkpoint_fails(
    tmp_path, monkeypatch
):
    import cv_pipeline

    fold_config, train_entries, holdout_entries = _real_fold_case(
        tmp_path, monkeypatch, max_epochs=3
    )
    fold_config = CVConfig(**{**fold_config.__dict__, "resume_cv": True})
    output_dir = fold_config.output / "folds" / "fold_0"
    output_dir.mkdir(parents=True)
    hashes = {
        "train_hash": split_hash(train_entries, fold_config.task_data),
        "holdout_hash": split_hash(holdout_entries, fold_config.task_data),
        "config_hash": training_config_hash(fold_config),
    }
    state = {
        "status": "in_progress",
        "fold_index": 0,
        **hashes,
        "random_initialization": True,
        "stage_config": {
            "type_focus_epochs": 1,
            "type_focus_distance_weight": 0.25,
            "joint_distance_weight": 1.0,
            "max_epochs": 3,
            "patience": 1,
        },
        "model_config": dict(cv_pipeline.FOLD_MODEL_CONFIG),
        "best_epoch": 1,
        "metrics": {"score": 1.0},
        "completed_epochs": 1,
        "early_stop_wait": 0,
    }
    (output_dir / "fold_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="in-progress fold is missing latest.pt"):
        cv_pipeline.train_conditional_fold(
            train_entries, holdout_entries, output_dir, fold_config, "cpu", 0
        )


def test_resume_already_at_patience_does_not_run_an_extra_epoch(
    tmp_path, monkeypatch
):
    import cv_pipeline

    fold_config, train_entries, holdout_entries = _real_fold_case(
        tmp_path, monkeypatch, max_epochs=3, patience=1
    )
    output_dir = fold_config.output / "folds" / "fold_0"
    original_epoch = cv_pipeline.run_conditional_epoch

    def interrupt_after_two_epochs(*args, **kwargs):
        if int(args[7]) == 2:
            raise RuntimeError("simulated interruption")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(
        cv_pipeline, "run_conditional_epoch", interrupt_after_two_epochs
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cv_pipeline.train_conditional_fold(
            train_entries, holdout_entries, output_dir, fold_config, "cpu", 0
        )

    latest_path = output_dir / "latest.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    latest["early_stop_wait"] = fold_config.patience
    torch.save(latest, latest_path)
    state_path = output_dir / "fold_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "completed_epochs": latest["completed_epochs"],
        "early_stop_wait": fold_config.patience,
        "best_epoch": latest["best_epoch"],
        "metrics": latest["best_metrics"],
    })
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def unexpected_epoch(*args, **kwargs):
        raise AssertionError("resume ran an extra epoch after patience was met")

    monkeypatch.setattr(cv_pipeline, "run_conditional_epoch", unexpected_epoch)
    resumed = CVConfig(**{**fold_config.__dict__, "resume_cv": True})
    result = cv_pipeline.train_conditional_fold(
        train_entries, holdout_entries, output_dir, resumed, "cpu", 0
    )

    assert result.best_epoch == latest["best_epoch"]
    assert read_oof_csv(result.oof_path)


def test_progressed_resume_requires_cpu_torch_rng_state(tmp_path, monkeypatch):
    import cv_pipeline

    fold_config, train_entries, holdout_entries = _real_fold_case(
        tmp_path, monkeypatch, max_epochs=3, patience=5
    )
    output_dir = fold_config.output / "folds" / "fold_0"
    original_epoch = cv_pipeline.run_conditional_epoch

    def interrupt_after_two_epochs(*args, **kwargs):
        if int(args[7]) == 2:
            raise RuntimeError("simulated interruption")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(
        cv_pipeline, "run_conditional_epoch", interrupt_after_two_epochs
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cv_pipeline.train_conditional_fold(
            train_entries, holdout_entries, output_dir, fold_config, "cpu", 0
        )

    latest_path = output_dir / "latest.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert int(latest["completed_epochs"]) > 0
    latest.pop("torch_rng_state")
    torch.save(latest, latest_path)

    def unexpected_epoch(*args, **kwargs):
        raise AssertionError("resume trained without restoring CPU RNG")

    monkeypatch.setattr(cv_pipeline, "run_conditional_epoch", unexpected_epoch)
    resumed = CVConfig(**{**fold_config.__dict__, "resume_cv": True})
    with pytest.raises(ValueError, match="torch_rng_state"):
        cv_pipeline.train_conditional_fold(
            train_entries, holdout_entries, output_dir, resumed, "cpu", 0
        )


def test_progressed_cuda_resume_requires_cuda_rng_state(monkeypatch):
    import cv_pipeline

    restore = getattr(cv_pipeline, "_restore_resume_rng_state", None)
    assert callable(restore), "resume RNG restoration helper is required"
    environment = {
        "device_count": 1,
        "devices": [{"name": "synthetic", "capability": [8, 0]}],
        "torch_cuda_version": "synthetic",
        "cudnn_version": 1,
    }
    monkeypatch.setattr(
        cv_pipeline, "_cuda_environment_identity", lambda: environment,
        raising=False,
    )
    state = {
        "completed_epochs": 1,
        "execution_device_type": "cuda",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_environment": environment,
    }
    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    with pytest.raises(ValueError, match="cuda_rng_state_all"):
        restore(state, cuda=True)
    assert setter_calls == []


def test_progressed_cuda_resume_rejects_device_environment_mismatch(monkeypatch):
    import cv_pipeline

    restore = getattr(cv_pipeline, "_restore_resume_rng_state", None)
    assert callable(restore), "resume RNG restoration helper is required"
    current = {
        "device_count": 1,
        "devices": [{"name": "current", "capability": [8, 0]}],
        "torch_cuda_version": "synthetic",
        "cudnn_version": 1,
    }
    saved = {
        **current,
        "device_count": 2,
        "devices": [
            {"name": "current", "capability": [8, 0]},
            {"name": "second", "capability": [8, 0]},
        ],
    }
    monkeypatch.setattr(
        cv_pipeline, "_cuda_environment_identity", lambda: current,
        raising=False,
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("CUDA RNG restored before environment validation")
        ),
    )
    state = {
        "completed_epochs": 1,
        "execution_device_type": "cuda",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [torch.get_rng_state(), torch.get_rng_state()],
        "cuda_environment": saved,
    }

    with pytest.raises(ValueError, match="CUDA environment mismatch"):
        restore(state, cuda=True)


@pytest.mark.parametrize(
    ("saved_mode", "current_cuda"),
    [("cuda", False), ("cpu", True)],
)
def test_progressed_resume_rejects_execution_mode_change_before_rng_setters(
    monkeypatch, saved_mode, current_cuda
):
    import cv_pipeline

    state = {
        "completed_epochs": 1,
        "execution_device_type": saved_mode,
        "torch_rng_state": torch.get_rng_state(),
    }
    if saved_mode == "cuda":
        state.update({
            "cuda_rng_state_all": [torch.get_rng_state()],
            "cuda_environment": {"device_count": 1},
        })
    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    with pytest.raises(ValueError, match="execution device mismatch"):
        cv_pipeline._restore_resume_rng_state(state, cuda=current_cuda)
    assert setter_calls == []


def test_progressed_resume_requires_execution_device_type_before_rng_setters(
    monkeypatch
):
    import cv_pipeline

    state = {
        "completed_epochs": 1,
        "torch_rng_state": torch.get_rng_state(),
    }
    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    with pytest.raises(ValueError, match="execution_device_type"):
        cv_pipeline._restore_resume_rng_state(state, cuda=False)
    assert setter_calls == []


def test_progressed_cpu_resume_rejects_contradictory_cuda_metadata(monkeypatch):
    import cv_pipeline

    state = {
        "completed_epochs": 1,
        "execution_device_type": "cpu",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [torch.get_rng_state()],
        "cuda_environment": {"device_count": 1},
    }
    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    with pytest.raises(ValueError, match="CPU latest contains CUDA metadata"):
        cv_pipeline._restore_resume_rng_state(state, cuda=False)
    assert setter_calls == []


def test_empty_initial_rng_state_needs_no_execution_mode_or_setters(monkeypatch):
    import cv_pipeline

    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    cv_pipeline._restore_resume_rng_state({"completed_epochs": 0}, cuda=False)
    assert setter_calls == []


@pytest.mark.parametrize("current_cuda", [False, True])
def test_progressed_resume_validates_all_rng_tensors_before_setters(
    monkeypatch, current_cuda
):
    import cv_pipeline

    environment = {"device_count": 1}
    monkeypatch.setattr(
        cv_pipeline, "_cuda_environment_identity", lambda: environment
    )
    state = {
        "completed_epochs": 1,
        "execution_device_type": "cuda" if current_cuda else "cpu",
        "torch_rng_state": torch.get_rng_state() if current_cuda else "invalid",
    }
    if current_cuda:
        state.update({
            "cuda_rng_state_all": ["invalid"],
            "cuda_environment": environment,
        })
    setter_calls = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda *args: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda *args: setter_calls.append("cuda"),
    )

    with pytest.raises(ValueError, match="RNG state tensor"):
        cv_pipeline._restore_resume_rng_state(state, cuda=current_cuda)
    assert setter_calls == []


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
        calibration = _distance_calibration_fixture(
            fitted,
            accepted,
            safe=fold != 1,
        )
        checkpoint = _checkpoint_fixture(
            fold_hash,
            fold_index=fold,
            best_epoch=fold + 2,
            metrics={"distance_calibration": calibration},
        )
        state = {
            "status": "complete",
            "fold_index": fold,
            "best_epoch": fold + 2,
            "metrics": checkpoint["metrics"],
            "stage_config": checkpoint["stage_config"],
            "model_config": checkpoint["model_config"],
            "random_initialization": True,
            **fold_hash,
        }
        torch.save(checkpoint, directory / "best.pt")
        (directory / "fold_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        fold_rows = []
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
            fold_rows.append(row)
        _write_oof_rows(directory / "oof.csv", fold_rows)
    return rows, expected, hashes


def _write_oof_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_FIELDS)
        writer.writeheader()
        writer.writerows([
            {name: "" if row.get(name) is None else row.get(name, "")
             for name in OOF_FIELDS}
            for row in rows
        ])


def _evaluated_oof_case(tmp_path):
    rows, expected, hashes = _perfect_raw_oof(tmp_path)
    fold_config = CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        bootstrap_iterations=5,
        stop_after_oof=True,
    )
    evaluate_oof_artifacts(rows, expected, fold_config.output, fold_config)
    return fold_config, expected, hashes


def test_evaluate_oof_writes_csv_metrics_and_guarded_medians(tmp_path):
    rows, expected, hashes = _perfect_raw_oof(tmp_path)
    fold_config = CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        bootstrap_iterations=10,
        stop_after_oof=True,
    )

    generated = evaluate_oof_artifacts(
        rows, expected, fold_config.output, fold_config
    )
    report = verify_cv_artifacts(
        fold_config.output, expected, hashes, fold_config
    )

    final_rows = read_oof_csv(fold_config.output / "oof_predictions.csv")
    saved = json.loads(
        (fold_config.output / "cv_metrics.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert generated["oof_piece_count"] == len(expected)
    assert len(final_rows) == len(expected) == 12
    assert tuple(final_rows[0]) == OOF_FIELDS
    assert all(row["accepted"] for row in final_rows)
    assert saved["fold_hashes"] == hashes
    assert saved["distance_calibration"]["final_temperatures"] == [2.0] * 4
    assert saved["distance_calibration"]["folds"]["1"]["accepted_temperatures"] == [1.0] * 4
    assert saved["type_rejection"]["version"] == 3
    assert saved["historical_metrics"]["status"] == "reference_only"
    assert "verified_release" not in saved


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


def test_verify_cv_artifacts_rejects_tampered_saved_metrics(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved["metrics"]["piece_count"] = 999
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="saved metrics hash"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda saved: saved.update(schema="wrong"), "saved CV metrics schema"),
        (lambda saved: saved.update(rounding_decimals=6), "rounding_decimals"),
        (
            lambda saved: saved["type_rejection"].update(version=2),
            "rejection policy version",
        ),
        (
            lambda saved: saved["type_rejection"].update(target_precision=0.5),
            "target_precision",
        ),
    ],
)
def test_verify_cv_artifacts_rejects_tampered_saved_schema_metadata(
    tmp_path, tamper, message
):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    tamper(saved)
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


@pytest.mark.parametrize("tamper", ["threshold", "calibration_hash"])
def test_verify_cv_artifacts_rejects_tampered_rejection_calibration(
    tmp_path, tamper
):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if tamper == "threshold":
        saved["type_rejection"]["probability_thresholds"][0] += 0.1
    else:
        saved["type_rejection"]["calibration_hash"] = "tampered"
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration_hash"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_cv_artifacts_rejects_tampered_policy_oof_metrics(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved["type_rejection"]["oof_metrics"]["type_coverage"] = 0.0
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="policy OOF metrics"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_cv_artifacts_rejects_tampered_final_distance_vector(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved["distance_calibration"]["final_temperatures"] = [9.0] * 4
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="distance temperature median"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_cv_artifacts_binds_distance_vectors_to_fold_states(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = saved["distance_calibration"]
    calibration["folds"]["0"]["accepted_temperatures"] = [8.0] * 4
    calibration["folds"]["0"]["fitted_temperatures"] = [8.0] * 4
    calibration["final_temperatures"] = [4.0] * 4
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")
    final_path = fold_config.output / "oof_predictions.csv"
    rows = read_oof_csv(final_path)
    for row in rows:
        if int(row["fold"]) == 0:
            row["distance_temperature"] = 8.0
    _write_oof_rows(final_path, rows)

    with pytest.raises(ValueError, match="raw fold OOF"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def _coherently_mutate_fold_zero_distance(output_dir, *, mutate_checkpoint):
    fold_dir = Path(output_dir) / "folds" / "fold_0"
    state_path = fold_dir / "fold_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    calibration = state["metrics"]["distance_calibration"]
    calibration["fitted_temperatures"] = [8.0] * 4
    calibration["accepted_temperatures"] = [8.0] * 4
    calibration["accepted"] = True
    calibration["guard_reason"] = "accepted"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    if mutate_checkpoint:
        checkpoint_path = fold_dir / "best.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint["metrics"] = json.loads(json.dumps(state["metrics"]))
        torch.save(checkpoint, checkpoint_path)

    metrics_path = Path(output_dir) / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved["distance_calibration"]["folds"]["0"] = json.loads(
        json.dumps(calibration)
    )
    saved["distance_calibration"]["final_temperatures"] = [4.0] * 4
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")
    final_path = Path(output_dir) / "oof_predictions.csv"
    rows = read_oof_csv(final_path)
    for row in rows:
        if int(row["fold"]) == 0:
            row["distance_temperature"] = 8.0
    _write_oof_rows(final_path, rows)


def test_verify_rejects_coherent_distance_mutation_not_in_best_checkpoint(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    _coherently_mutate_fold_zero_distance(
        fold_config.output, mutate_checkpoint=False
    )

    with pytest.raises(ValueError, match="checkpoint/state metrics mismatch"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_binds_final_immutable_fields_to_raw_fold_oof(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    _coherently_mutate_fold_zero_distance(
        fold_config.output, mutate_checkpoint=True
    )

    with pytest.raises(ValueError, match="raw fold OOF"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_distance_safety_requires_boolean_accepted_decision(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    fold_dir = fold_config.output / "folds" / "fold_0"
    state_path = fold_dir / "fold_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["metrics"]["distance_calibration"]["accepted"] = "True"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    checkpoint_path = fold_dir / "best.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint["metrics"] = json.loads(json.dumps(state["metrics"]))
    torch.save(checkpoint, checkpoint_path)
    metrics_path = fold_config.output / "cv_metrics.json"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved["distance_calibration"]["folds"]["0"]["accepted"] = "True"
    metrics_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted must be boolean"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)


def test_verify_cv_artifacts_rejects_tampered_csv_row_hash(tmp_path):
    fold_config, expected, hashes = _evaluated_oof_case(tmp_path)
    final_path = fold_config.output / "oof_predictions.csv"
    rows = read_oof_csv(final_path)
    rows[0]["train_hash"] = "tampered"
    _write_oof_rows(final_path, rows)

    with pytest.raises(ValueError, match="row fold hashes"):
        verify_cv_artifacts(fold_config.output, expected, hashes, fold_config)
