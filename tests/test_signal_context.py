from datetime import datetime

import numpy as np


def test_time_context_contains_daylight_and_periodic_local_hour():
    from data.signal_context import time_context_batch

    context = time_context_batch(
        [datetime(2019, 1, 1, 0), datetime(2019, 1, 2, 0)],
        [True, False],
    )

    assert context.shape == (2, 3)
    assert context[:, 0].tolist() == [1.0, 0.0]
    assert np.allclose(np.linalg.norm(context[:, 1:], axis=1), 1.0)
    assert np.allclose(context[0, 1:], context[1, 1:])


def test_time_context_rejects_misaligned_inputs():
    from data.signal_context import time_context_batch

    try:
        time_context_batch([datetime(2019, 1, 1)], [True, False])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("misaligned timestamps and daylight were accepted")
