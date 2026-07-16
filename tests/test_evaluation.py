import numpy as np
import pytest
import torch

import evaluation
from data.cross_validation import MINIMUM_SUPPORTED_PIECES
from evaluation import (
    distance_calibration_is_safe,
    evaluate_predictions,
    evaluate_release,
    file_bootstrap_metrics,
)


def make_record(
    file_id,
    true_type,
    predicted_type,
    low=0,
    high=100,
    predicted_distance=50.0,
    daylight=True,
    accepted=True,
    source_path=None,
):
    record = {
        "file_id": file_id,
        "true_type": true_type,
        "predicted_type": predicted_type,
        "accepted": accepted,
        "distance_low_km": low,
        "distance_high_km": high,
        "predicted_distance_km": predicted_distance,
        "oracle_distance_km": predicted_distance,
        "daylight": daylight,
    }
    if source_path is not None:
        record["source_path"] = source_path
    return record


def good_release_metrics():
    return {
        "type_file_equal_precision": [0.96, 0.97, 0.98, 0.99],
        "type_coverage": 0.82,
        "type_file_equal_recall_mean": 0.91,
        "distance_100km_interval_within_200": 0.88,
        "distance_per_type_100km_interval_within_200": [0.80, 0.85, 0.90, 0.95],
        "distance_conditions_100km": {},
    }


def test_record_file_identity_prefers_source_path_and_requires_stable_identity():
    assert evaluation.record_file_identity({
        "source_path": "relative/a.lig",
        "file_id": 7,
    }) == "relative/a.lig"
    assert evaluation.record_file_identity({"file_id": 7}) == "7"
    with pytest.raises(ValueError, match="stable source identity"):
        evaluation.record_file_identity({})


def test_file_equal_piece_weights_give_each_file_total_weight_one():
    file_ids = np.asarray(["large"] * 100 + ["small"])

    weights = evaluation.file_equal_piece_weights(file_ids)

    assert weights[file_ids == "large"].sum() == pytest.approx(1.0)
    assert weights[file_ids == "small"].sum() == pytest.approx(1.0)


def test_file_equal_precision_gives_each_file_total_weight_one():
    records = [make_record("large", 0, 0) for _ in range(100)]
    records += [make_record("small", 1, 0)]

    metrics = evaluate_predictions(records)

    assert metrics["type_piece_accuracy"] > 0.99
    assert metrics["type_file_macro_accuracy"] == pytest.approx(0.5)
    assert metrics["type_precision"][0] == pytest.approx(100 / 101)
    assert metrics["type_file_equal_precision"][0] == pytest.approx(0.5)


def test_source_path_is_the_stable_file_equal_identity():
    records = [
        make_record(index, 0, 0, source_path="large.lig")
        for index in range(100)
    ]
    records.append(make_record(100, 1, 0, source_path="small.lig"))

    metrics = evaluate_predictions(records)

    assert metrics["file_count"] == 2
    assert metrics["type_file_equal_precision"][0] == pytest.approx(0.5)


def test_rejected_metrics_are_separate_from_raw_piece_and_file_equal_metrics():
    records = [
        make_record("accepted", 0, 0, accepted=True),
        make_record("rejected", 1, 1, accepted=False),
    ]

    metrics = evaluate_predictions(records)

    assert metrics["type_piece_accuracy"] == pytest.approx(1.0)
    assert metrics["type_file_macro_accuracy"] == pytest.approx(1.0)
    assert metrics["type_coverage"] == pytest.approx(0.5)
    assert metrics["type_precision"][1] == pytest.approx(0.0)
    assert metrics["type_recall"][1] == pytest.approx(0.0)
    assert metrics["type_file_equal_precision"][1] == pytest.approx(0.0)
    assert metrics["type_file_equal_recall"][1] == pytest.approx(0.0)
    assert metrics["raw_type_precision"][1] == pytest.approx(1.0)
    assert metrics["raw_type_recall"][1] == pytest.approx(1.0)
    assert metrics["raw_type_file_equal_precision"][1] == pytest.approx(1.0)
    assert metrics["raw_type_file_equal_recall"][1] == pytest.approx(1.0)


def test_distance_reports_100km_intervals_and_contextual_coarse_groups():
    records = [
        make_record("a", 0, 0, 0, 100, 50, True),
        make_record("b", 0, 0, 300, 400, 750, False),
        make_record("c", 1, 1, 1500, 3000, 2000, False),
    ]

    metrics = evaluate_predictions(records)

    assert "NCG/day/0-300km" in metrics["distance_subgroups"]
    assert "NCG/night/300-600km" in metrics["distance_subgroups"]
    assert metrics["distance_100km_interval_count"] == 2
    assert metrics["distance_100km_interval_within_200"] == pytest.approx(0.5)
    assert metrics["distance_interval_mae_km"] == pytest.approx(350 / 3)
    assert "distance_exact_count" not in metrics
    assert "distance_exact_within_100" not in metrics
    assert "distance_exact_within_200" not in metrics
    assert "distance_per_type_exact_within_200" not in metrics


def test_distance_conditions_use_exact_100km_bins_and_stable_source_files():
    records = [
        make_record(0, 3, 3, 300, 400, 350, False, source_path="a.lig"),
        make_record(1, 3, 3, 300, 400, 700, False, source_path="a.lig"),
        make_record(2, 3, 3, 300, 400, 350, False, source_path="b.lig"),
        make_record(3, 3, 3, 300, 500, 400, False, source_path="c.lig"),
    ]

    metrics = evaluate_predictions(records)

    condition = metrics["distance_conditions_100km"]["PNBE/night/300-400km"]
    assert condition["piece_count"] == 3
    assert condition["file_count"] == 2
    assert condition["file_macro_within_200"] == pytest.approx(0.75)
    assert len(metrics["distance_conditions_100km"]) == 1


def test_rejected_distance_counts_as_uncovered_and_interval_failure():
    records = [
        make_record("a", 0, 0, predicted_distance=50, accepted=True),
        make_record("b", 0, 0, predicted_distance=50, accepted=False),
    ]

    metrics = evaluate_predictions(records)

    assert metrics["distance_coverage"] == pytest.approx(0.5)
    assert metrics["distance_100km_interval_within_200"] == pytest.approx(0.5)


def test_file_bootstrap_is_reproducible_and_returns_renamed_intervals():
    records = [
        make_record("a", 0, 0),
        make_record("a", 0, 0),
        make_record("b", 1, 0, predicted_distance=500),
    ]

    first = file_bootstrap_metrics(records, iterations=100, seed=7)
    second = file_bootstrap_metrics(records, iterations=100, seed=7)

    assert first == second
    assert set(first) == {
        "type_file_macro_accuracy_ci95",
        "distance_100km_interval_within_200_ci95",
    }


def test_bootstrap_duplicate_source_draws_remain_independent_pseudo_files():
    records = [
        make_record(0, 0, 0, source_path="a"),
        make_record(1, 0, 1, source_path="b"),
        make_record(2, 0, 1, source_path="c"),
    ]

    metrics = file_bootstrap_metrics(records, iterations=1, seed=2)

    assert metrics["type_file_macro_accuracy_ci95"] == pytest.approx([2 / 3, 2 / 3])


def test_only_supported_subgroups_are_hard_gates():
    passing = good_release_metrics()
    passing["distance_conditions_100km"] = {
        "PNBE/night/300-400km": {
            "piece_count": 50,
            "file_macro_within_200": 0.10,
        }
    }
    assert evaluate_release(passing)[0]

    passing["distance_conditions_100km"]["PNBE/night/300-400km"]["piece_count"] = 100
    passed, reasons = evaluate_release(passing)

    assert not passed
    assert any("supported subgroup" in reason for reason in reasons)


def test_release_gate_uses_canonical_piece_support_floor():
    assert MINIMUM_SUPPORTED_PIECES == 100
    assert (
        evaluation.RELEASE_GATES["minimum_supported_pieces"]
        == MINIMUM_SUPPORTED_PIECES
    )


def test_supported_subgroup_at_sixty_percent_still_fails_release():
    candidate = good_release_metrics()
    candidate["distance_conditions_100km"] = {
        "NCG/day/300-400km": {
            "piece_count": MINIMUM_SUPPORTED_PIECES,
            "file_macro_within_200": 0.60,
        }
    }

    passed, reasons = evaluate_release(candidate)

    assert passed is False
    assert any("supported subgroup" in reason for reason in reasons)


def test_release_uses_only_fixed_absolute_file_equal_gates():
    candidate = good_release_metrics()

    assert evaluate_release(candidate) == (True, [])

    candidate["type_file_equal_precision"][0] = 0.94
    candidate["type_coverage"] = 0.79
    candidate["type_file_equal_recall_mean"] = 0.89
    candidate["distance_100km_interval_within_200"] = 0.84
    candidate["distance_per_type_100km_interval_within_200"][2] = 0.74
    passed, reasons = evaluate_release(candidate)

    assert passed is False
    assert any("type_file_equal_precision[0]" in reason for reason in reasons)
    assert any("type_coverage" in reason for reason in reasons)
    assert any("type_file_equal_recall_mean" in reason for reason in reasons)
    assert any("distance_100km_interval_within_200" in reason for reason in reasons)
    assert any("distance_within_200[2]" in reason for reason in reasons)


def test_release_requires_four_per_type_values():
    candidate = good_release_metrics()
    candidate["type_file_equal_precision"] = candidate["type_file_equal_precision"][:3]
    candidate["distance_per_type_100km_interval_within_200"] = []

    passed, reasons = evaluate_release(candidate)

    assert not passed
    assert "four file-equal per-type precision values are required" in reasons
    assert "four per-type 100-km interval values are required" in reasons


def test_checkpoint_selection_key_uses_supported_condition_and_readiness_floor():
    metrics = {
        "type_file_equal_recall_mean": 0.86,
        "type_file_equal_recall": [0.75, 0.80, 0.90, 1.0],
        "distance_conditions_100km": {
            "supported": {"piece_count": 150, "file_macro_within_200": 0.71},
            "sparse": {"piece_count": 50, "file_macro_within_200": 0.10},
        },
        "distance_file_macro_within_200": 0.80,
        "distance_per_type_100km_interval_within_200": [0.75, 0.80, 0.85, 0.90],
        "distance_interval_mae_km": 100.0,
    }

    assert evaluation.checkpoint_selection_key(metrics) == (
        1,
        0.86,
        0.75,
        0.71,
        0.80,
        0.75,
        -100.0,
    )

    metrics["type_file_equal_recall_mean"] = 0.849
    assert evaluation.checkpoint_selection_key(metrics)[0] == 0


def test_checkpoint_selection_key_defaults_unsupported_condition_score_to_zero():
    metrics = {
        "type_file_equal_recall_mean": 0.90,
        "type_file_equal_recall": [0.80, 0.80, 0.80, 0.80],
        "distance_conditions_100km": {
            "sparse": {"piece_count": 50, "file_count": 2, "file_macro_within_200": 0.99},
        },
        "distance_file_macro_within_200": 0.85,
        "distance_per_type_100km_interval_within_200": [0.80] * 4,
        "distance_interval_mae_km": 90.0,
    }

    assert evaluation.checkpoint_selection_key(metrics)[3] == 0.0


def test_distance_calibration_must_not_degrade_three_interval_metrics():
    before = {
        "distance_file_macro_within_200": 0.80,
        "distance_100km_interval_within_200": 0.82,
        "distance_interval_mae_km": 120.0,
    }

    assert distance_calibration_is_safe(before, dict(before))
    assert distance_calibration_is_safe(before, {
        **before,
        "distance_file_macro_within_200": 0.80 - 1e-12,
        "distance_100km_interval_within_200": 0.82 - 1e-12,
        "distance_interval_mae_km": 120.0 + 1e-12,
    })
    assert not distance_calibration_is_safe(before, {
        **before,
        "distance_file_macro_within_200": 0.80 - 2e-12,
    })
    assert not distance_calibration_is_safe(before, {
        **before,
        "distance_100km_interval_within_200": 0.82 - 2e-12,
    })
    assert not distance_calibration_is_safe(before, {
        **before,
        "distance_interval_mae_km": 120.0 + 2e-12,
    })


def test_round_metrics_sorts_keys_normalizes_sequences_and_rejects_non_finite():
    rounded = evaluation.round_metrics({
        "z": np.float64(1 / 3),
        "a": (np.int64(2), 0.1234567890128),
    })

    assert list(rounded) == ["a", "z"]
    assert rounded == {
        "a": [2, 0.123456789013],
        "z": 0.333333333333,
    }
    for value in (float("nan"), float("inf"), np.float64("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            evaluation.round_metrics(value)


def test_v3_benchmark_records_retain_support_and_hash_evidence():
    import benchmark

    distance_logits = torch.zeros(2, 4, 30)
    distance_logits[0, 0, 4] = 5.0
    distance_logits[1, 0, 4] = 5.0
    records = [
        {
            "true_type": 0,
            "predicted_type": 0,
            "accepted": True,
            "daylight": True,
            "predicted_distance_km": 450.0,
        },
        {
            "true_type": 0,
            "predicted_type": 0,
            "accepted": False,
            "daylight": True,
            "predicted_distance_km": 450.0,
        },
    ]
    bundle = {"records": records, "distance_logits": distance_logits}
    checkpoint = {
        "schema": "four_class_cv_v3",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "support_map": {
            "NCG/day/400-500km": {
                "file_count": 1,
                "piece_count": 100,
                "status": "supported",
            }
        },
        "fold_manifest_hash": "fold-hash",
        "full_data_hash": "full-hash",
        "rejection_policy": {"calibration_hash": "calibration-hash"},
    }

    assert hasattr(benchmark, "annotate_v3_benchmark_records")
    annotated = benchmark.annotate_v3_benchmark_records(bundle, checkpoint)

    assert annotated is records
    assert annotated[0]["support_status"] == "supported"
    assert annotated[0]["support_file_count"] == 1
    assert annotated[0]["support_condition"] == "NCG/day/400-500km"
    assert annotated[1]["support_status"] == "not_applicable"
    assert annotated[0]["fold_manifest_hash"] == "fold-hash"
    assert annotated[0]["full_data_hash"] == "full-hash"
    assert annotated[0]["calibration_hash"] == "calibration-hash"
    assert [row["accepted"] for row in annotated] == [True, False]
    assert [row["predicted_distance_km"] for row in annotated] == [450.0, 450.0]


def test_historical_benchmark_output_is_reference_only(monkeypatch, tmp_path):
    import json
    import sys

    import benchmark

    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({
        "split_hashes": {"test": "locked"},
    }), encoding="utf-8")
    written = {}
    monkeypatch.setattr(benchmark, "_file_entries", lambda *args: [])
    monkeypatch.setattr(
        benchmark,
        "evaluate_checkpoint",
        lambda *args, **kwargs: {"type_piece_accuracy": 0.99},
    )
    monkeypatch.setattr(
        benchmark,
        "write_json",
        lambda path, payload: written.update(payload),
    )
    monkeypatch.setattr(sys, "argv", [
        "benchmark.py",
        "--split_manifest",
        str(manifest_path),
        "--model",
        "historical=old.pt",
    ])

    benchmark.main()

    assert written["reference_only"] is True
    assert written["models"]["historical"]["reference_only"] is True


def test_historical_benchmark_has_no_automatic_release_comparator():
    import benchmark

    assert not hasattr(benchmark, "compare_model_metrics")
