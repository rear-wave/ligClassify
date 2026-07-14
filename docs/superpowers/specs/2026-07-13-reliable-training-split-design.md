# Reliable Training Split Design

## Goal

Prevent a model with incomplete validation coverage or poor future-date performance from replacing the deployed checkpoint. Training remains fully independent of old-model initialization when `--no_init` is used.

## Data Split Contract

The split has two different evaluation roles:

1. **Locked temporal test:** For each lightning type, select the latest 15% of whole acquisition dates (rounded, with at least one earlier training date). A date must belong entirely to train/development or test. Date-count splitting is intentional because piece counts are extremely concentrated in a few IC dates. The test set is never used for early stopping, calibration, or hyperparameter selection.
2. **Coverage validation:** From files earlier than the temporal test boundary, select whole files independently within each `(type, distance_bin)` group. Selection is deterministic and targets 15% of files while retaining at least one training file when possible. IC files are selected deterministically without distance stratification. Pieces from one file can never cross splits.

Train and validation may share acquisition dates; this is intentional because validation measures distance-bin interpolation. Only the locked test is allowed to support claims about future-date generalization. Checkpoint metadata and logs must label these roles explicitly.

Before training starts, distance validation must contain at least 12 bins and 500 pieces for every non-IC type. Failure raises an actionable error instead of silently training with an invalid validation set. Logs report files, pieces, dates, bins, and missing bins for all splits.

## Sampling and Optimization

The type stream uses 180,000 samples per epoch by default, with exactly 80% IC and non-IC samples distributed proportionally to their available counts. Sampling is deterministic by seed and changes each epoch.

The distance stream uses 60,000 samples per epoch by default. It cycles uniformly over lightning type and available distance bin, then over dates and files. Exhausted pools are reshuffled and reused, providing controlled replacement for rare bins. The per-file candidate pool remains capped at 256 pieces.

The distance-stream loss is:

```text
lambda_dist * distance_loss + distance_batch_type_weight * type_loss
```

`lambda_dist` therefore affects gradients and defaults to `1.0`. Logs report type and distance optimizer-step counts.

## Selection and Release Gate

Early stopping uses coverage-validation metrics. Epochs below `--min_type_f1 0.85` are ineligible unless no eligible epoch exists; such a fallback remains a candidate and cannot be promoted automatically.

After selection and calibration, the locked temporal test runs exactly once. The candidate is always written to `candidate.pt` with `candidate_metrics.json`. It replaces `model.pt` and `metrics.json` only when all conditions pass:

- type macro F1 is at least 0.85;
- every non-IC type has temporal-test `w2` (error within two 100 km bins) of at least 0.70;
- temporal-test macro `w2` is at least 0.75.

A failed gate leaves the existing deployed model untouched and logs every failed condition. `--skip_test` can create a candidate for experiments but can never promote it.

## Verification

Synthetic tests cover latest-date temporal test selection, file isolation, validation-bin coverage failure, exact type prior, balanced replacement sampling, applied distance-loss weight, type-F1 selection eligibility, and release-gate behavior. Real-data audit must confirm no test-date overlap, at least 12 validation bins per distance type, deterministic split hashes, and balanced sampled counts. The full pytest suite, compile check, CLI help, and a one-epoch small-sample smoke run are required before handoff.
