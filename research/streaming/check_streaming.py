"""Verify completed-piece serving equivalence, restart behavior and full-bundle CPU latency."""
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classify import StreamingClassifier, predict_bundle_batch
from data.preprocess import TemporalContextConfig
from experiment import OUT


def run():
    torch.set_num_threads(4)
    cache = np.load(OUT / "validation_raw.npz")
    rng = np.random.default_rng(900)
    indices = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(cache["labels"] == label), 24, replace=False)
        for label in range(5)
    ]))
    raw = cache["raw"][indices]
    timestamps = [datetime(2024, 1, 1, 4 if d else 20) for d in cache["daylight"][indices]]
    report = {
        "scope": "120 held-out validation pieces, complete first-channel waveform to type AND distance; no file I/O or acquisition time",
        "chronology": "Synthetic event times for API tests only; not a continuous raw-stream accuracy evaluation",
        "device": "cpu", "cpu_threads": 4, "results": {},
    }
    for directory in ("multi_model", "anchor_moe_best_bundle"):
        service = StreamingClassifier(ROOT / "weights" / directory)
        expected = []
        for start in range(0, len(raw), 32):
            expected.extend(predict_bundle_batch(service.bundle, raw[start:start+32], timestamps[start:start+32],
                device=torch.device("cpu"), type_only=False))
        for i in range(5):
            service.predict_piece(raw[i], stream_id="warmup", timestamp=timestamps[i])
        results = [service.predict_piece(x, stream_id="GZ", timestamp=t) for x, t in zip(raw, timestamps)]
        labels_equal = all(a.prediction.final_type == b.final_type for a, b in zip(results, expected))
        bins_equal = all(a.prediction.distance_bin == b.distance_bin for a, b in zip(results, expected))
        confidence_delta = max(abs(a.prediction.type_confidence - b.type_confidence) for a, b in zip(results, expected))
        delays = [r.elapsed_ms for r in results]
        assert labels_equal and bins_equal
        # Restart exactly halfway through a synthetic chronological schedule, with optional context enabled.
        config = TemporalContextConfig(32, 12, .95, .1)
        uninterrupted = StreamingClassifier(ROOT / "weights" / directory, temporal_config=config)
        chronological = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=100*i) for i in range(len(raw))]
        full = [uninterrupted.predict_piece(x, stream_id="GZ", timestamp=t) for x, t in zip(raw, chronological)]
        restarted = StreamingClassifier(ROOT / "weights" / directory, temporal_config=config)
        split = len(raw) // 2
        part = [restarted.predict_piece(x, stream_id="GZ", timestamp=t) for x, t in zip(raw[:split], chronological[:split])]
        saved = json.loads(json.dumps(restarted.snapshot()))
        restarted = StreamingClassifier(ROOT / "weights" / directory, temporal_config=config)
        restarted.restore(saved)
        part += [restarted.predict_piece(x, stream_id="GZ", timestamp=t) for x, t in zip(raw[split:], chronological[split:])]
        restart_equal = all(a.prediction == b.prediction and a.temporal_promoted == b.temporal_promoted for a, b in zip(full, part))
        assert restart_equal
        report["results"][directory] = {
            "pieces": len(raw), "type_equal_to_batch": labels_equal, "distance_bin_equal_to_batch": bins_equal,
            "max_type_confidence_difference": confidence_delta, "restart_predictions_identical": restart_equal,
            "median_ms": float(np.median(delays)), "p95_ms": float(np.quantile(delays, .95)),
            "p99_ms": float(np.quantile(delays, .99)), "bundle_hashes": service.bundle.hashes,
        }
        print(directory, json.dumps(report["results"][directory]), flush=True)
    path = OUT / "streaming_api_verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
