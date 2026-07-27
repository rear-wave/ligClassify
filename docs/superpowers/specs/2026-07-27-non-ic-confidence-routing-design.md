# Non-IC Confidence Routing

## Goal

Add an inference-only confidence gate for type classification. IC predictions
are always retained. A predicted `NCG`, `NNBE`, `PCG`, or `PNBE` is retained
only when its softmax probability reaches the configured threshold; otherwise
the final type is changed to `IC`.

## Interface and Behavior

`classify.py` accepts `--min_non_ic_confidence`, a floating-point value in
`[0, 1]`. Omitting the option preserves existing behavior. With
`--min_non_ic_confidence 0.90 --type_only`:

- raw IC predictions are written below `IC/`;
- non-IC predictions with confidence at least `0.90` are written below their
  predicted type directory;
- non-IC predictions below `0.90` are written below `IC/`;
- no distance model is evaluated and no distance directory is created.

The CSV keeps all five original softmax probability columns. `final_type` and
`output_file` describe the post-threshold routing result. `type_confidence`
remains the original winning-class confidence so the reason for threshold
routing can be audited from the row.

## Validation and Tests

Reject non-finite thresholds and values outside `[0, 1]`. Add synthetic tests
for raw IC, accepted non-IC, rejected non-IC, exact-threshold acceptance,
disabled-gate compatibility, CLI propagation, and confirmation that
`--type_only` skips the distance stage. Existing checkpoint schemas, model
weights, waveform bytes, and training behavior remain unchanged.
