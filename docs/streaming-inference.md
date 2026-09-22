# Completed-waveform streaming inference

`classify.StreamingClassifier` loads a validated five-checkpoint bundle once.
Call `predict_piece` when a complete first-channel waveform has arrived. Each
call returns a `StreamPrediction` containing the existing `Prediction` (type,
distance bin, expected distance and confidence), UTC timestamp, context status,
whether temporal promotion occurred, and measured service latency.

```python
from datetime import datetime, timezone
from classify import StreamingClassifier

service = StreamingClassifier("weights/multi_model", device="cpu")
# waveform: one NumPy array of 16,000 first-channel samples at 5 MHz.
result = service.predict_piece(
    waveform, stream_id="GZ", timestamp=datetime.now(timezone.utc)
)
print(result.prediction.final_type, result.prediction.expected_distance_km)
```

For completed-waveform batch or directory inference, `classify.py` also accepts
an optional `--type_verifier_config` recipe. The verifier is deliberately not
enabled by `StreamingClassifier` unless its explicit configuration is passed;
it cannot be combined with temporal context. A verifier recipe is bound to both
type checkpoints, their decision hashes, preprocessing contract, split hash and
the calibrated per-class limits. Missing or mismatched hashes fail closed.

Use the acquisition timestamp, not arrival time, for historical replay. Naive
datetimes are interpreted as UTC; aware datetimes are converted to UTC. Distinct
stations/channels must have distinct stream IDs. Temporal promotion is disabled
by default; explicitly pass `TemporalContextConfig` after validating it on the
intended continuous stream. The previously selected 32/12/.95/.1 configuration
was developed on selected pieces, not validated as an operational arrival stream.

The API accepts only completed 16,000-sample waveforms and rejects malformed or
non-finite values. It does not yet provide a sample accumulator, file watcher,
network endpoint, or support arbitrary sampling rates. Consumers of longer LIG
formats must use the repository's validated reader and the intended event
segmentation before submitting a piece.

History is isolated per stream and bounded to at most 64 stream IDs by default.
A gap longer than `max_gap_seconds` (default 60) starts fresh history. Late events
are classified without historical promotion and cannot rewind the stream's
watermark. Missing timestamps use the model's existing day/night inference rule
and clear that stream's temporal history after a successful inference. Missing
and late records still produce a prediction; their context status is explicit.

A per-instance lock serializes calls and snapshot operations. History commits
only after successful type and distance inference. Errors therefore do not add
anchors. Duplicate event delivery is the producer's responsibility: repeated
calls are repeated events, not an exactly-once message service.

`snapshot()` returns JSON-serializable state bound to model hashes and context
settings. Persist it together with the consumer's acknowledged input offset.
`restore()` validates the whole snapshot before changing live state and rejects
different models/configuration. The caller owns durable storage and atomic
offset/result acknowledgment; the API does not claim crash-safe exactly-once
delivery by itself.

Latency includes input validation, preprocessing, type inference, optional
temporal promotion, selected distance inference and lock waiting. Acquisition,
disk/network reads and downstream persistence are outside the API timer. CPU/GPU
real-time acceptance must include those external costs and the intended arrival
rate. Controlled stress tests do not establish unseen-station accuracy; all
deployment conclusions must cite the appropriate held-out or chronological
evaluation.
