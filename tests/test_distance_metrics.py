import numpy as np
import pytest

from distance_metrics import (
    interval_distance_errors,
    summarize_equal_bin_distance_predictions,
)


def test_equal_bin_metrics_do_not_let_large_bin_dominate():
    predictions = np.asarray([0] * 100 + [10])
    targets = np.asarray([0] * 100 + [1])

    metrics = summarize_equal_bin_distance_predictions(predictions, targets)

    assert metrics["bin_count"] == 2
    assert metrics["w2"] == pytest.approx(0.5)
    assert metrics["acc"] == pytest.approx(0.5)
    assert metrics["mae_km"] == pytest.approx(450.0)


def test_equal_bin_metrics_require_non_empty_aligned_arrays():
    with pytest.raises(ValueError, match="non-empty aligned"):
        summarize_equal_bin_distance_predictions([], [])
    with pytest.raises(ValueError, match="non-empty aligned"):
        summarize_equal_bin_distance_predictions([0], [0, 1])


def test_interval_distance_error_is_zero_inside_and_nearest_boundary_outside():
    error = interval_distance_errors(
        predictions=[50, 500, 2500],
        low_km=[0, 100, 1500],
        high_km=[100, 300, 3000],
    )

    assert error.tolist() == [0.0, 200.0, 0.0]


def test_interval_distance_error_requires_valid_aligned_intervals():
    with pytest.raises(ValueError, match="aligned"):
        interval_distance_errors([1], [0, 1], [100, 200])
    with pytest.raises(ValueError, match="valid"):
        interval_distance_errors([1], [100], [100])
