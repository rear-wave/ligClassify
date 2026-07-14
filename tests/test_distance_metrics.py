import numpy as np
import pytest

from distance_metrics import summarize_equal_bin_distance_predictions


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
