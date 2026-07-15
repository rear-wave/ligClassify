import pytest

from evaluation import (
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
):
    return {
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


def test_file_macro_prevents_one_large_file_from_dominating():
    records = [make_record("large", 0, 0) for _ in range(100)]
    records.append(make_record("small", 1, 0))

    metrics = evaluate_predictions(records)

    assert metrics["type_piece_accuracy"] > 0.99
    assert metrics["type_file_macro_accuracy"] == pytest.approx(0.5)
    assert metrics["type_file_macro_precision"][0] == pytest.approx(0.5)


def test_distance_reports_type_daylight_and_coarse_band_groups():
    records = [
        make_record("a", 0, 0, 0, 100, 50, True),
        make_record("b", 0, 0, 300, 400, 750, False),
        make_record("c", 1, 1, 1500, 3000, 2000, False),
    ]

    metrics = evaluate_predictions(records)

    assert "NCG/day/0-300km" in metrics["distance_subgroups"]
    assert "NCG/night/300-600km" in metrics["distance_subgroups"]
    assert metrics["distance_exact_within_200"] == pytest.approx(0.5)
    assert metrics["distance_interval_mae_km"] == pytest.approx(350 / 3)


def test_rejection_metrics_count_unaccepted_true_types_in_recall():
    records = [
        make_record("a", 0, 0, accepted=True),
        make_record("b", 0, 0, accepted=False),
        make_record("c", 1, 1, accepted=True),
    ]

    metrics = evaluate_predictions(records)

    assert metrics["type_coverage"] == pytest.approx(2 / 3)
    assert metrics["type_precision"][0] == pytest.approx(1.0)
    assert metrics["type_recall"][0] == pytest.approx(0.5)


def test_rejected_distance_counts_as_uncovered_and_release_failure():
    records = [
        make_record("a", 0, 0, predicted_distance=50, accepted=True),
        make_record("b", 0, 0, predicted_distance=50, accepted=False),
    ]

    metrics = evaluate_predictions(records)

    assert metrics["distance_coverage"] == pytest.approx(0.5)
    assert metrics["distance_exact_within_200"] == pytest.approx(0.5)


def test_file_bootstrap_is_reproducible_and_returns_intervals():
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
        "distance_exact_within_200_ci95",
    }


def good_release_metrics(split_hash="same"):
    return {
        "split_hash": split_hash,
        "type_precision": [0.96, 0.97, 0.98, 0.99],
        "type_coverage": 0.82,
        "type_macro_recall": 0.91,
        "distance_exact_within_200": 0.88,
        "distance_per_type_exact_within_200": [0.80, 0.85, 0.90, 0.95],
        "distance_subgroups": {"NCG/day/0-300km": {"within_200": 0.90}},
    }


def test_release_requires_same_split_and_all_approved_gates():
    candidate = good_release_metrics()
    baseline = good_release_metrics()

    assert evaluate_release(candidate, baseline) == (True, [])

    candidate["type_precision"][0] = 0.94
    candidate["distance_per_type_exact_within_200"][2] = 0.70
    passed, reasons = evaluate_release(candidate, baseline)
    assert passed is False
    assert any("type_precision[0]" in reason for reason in reasons)
    assert any("distance_within_200[2]" in reason for reason in reasons)

    with pytest.raises(ValueError, match="split hash"):
        evaluate_release(good_release_metrics("candidate"), good_release_metrics("baseline"))


def test_release_rejects_reported_subgroup_regression():
    candidate = good_release_metrics()
    baseline = good_release_metrics()
    candidate["distance_subgroups"]["NCG/day/0-300km"]["within_200"] = 0.80

    passed, reasons = evaluate_release(candidate, baseline)

    assert passed is False
    assert any("subgroup regression" in reason for reason in reasons)


def test_release_uses_file_macro_precision_when_available():
    candidate = good_release_metrics()
    candidate["type_file_macro_precision"] = [0.90, 0.97, 0.98, 0.99]

    passed, reasons = evaluate_release(candidate, good_release_metrics())

    assert passed is False
    assert any("type_file_macro_precision[0]" in reason for reason in reasons)
