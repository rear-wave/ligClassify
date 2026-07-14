# Four-Class Training with IC Rejection

## Objective

Improve reliable classification of NCG, NNBE, PCG, and PNBE without using the low-quality IC dataset for gradient training. At inference time, `IC` means the waveform was rejected from the four researched classes; it is not a validated physical IC diagnosis. Optimize a balance between four-class precision and retained coverage.

## Training Data and Splits

Train the type head only on NCG, NNBE, PCG, and PNBE. Sample hierarchically by type, acquisition date, file, and piece so that each type contributes equally and no large file or date dominates an epoch. Keep distance-labelled sampling separate from type sampling.

Reserve the latest acquisition dates as a locked temporal test. Build validation coverage by type and distance bin from the remaining development period, with file-level isolation. Report per-type precision, recall, F1, confusion matrices, and metrics grouped by acquisition year. Unlabelled 2016 deployment data may be inspected manually but must not select checkpoints or thresholds.

## Model and Optimization

Replace the five-way type head with a four-way head ordered as `NCG`, `NNBE`, `PCG`, `PNBE`. Keep the shared encoder and four distance heads. Type checkpoint selection must prioritize four-class macro-F1 and per-type recall before distance metrics. Distance optimization must not be allowed to select a checkpoint with degraded type performance.

Initialization is explicit: random initialization remains the default for an independent experiment, while encoder initialization from a named checkpoint requires a CLI option and is recorded in checkpoint metadata. A five-class type head is never loaded into the four-class head.

## Rejection at Inference

For each waveform, calculate:

1. temperature-calibrated probability of the predicted type;
2. the probability margin between the highest and second-highest types;
3. distance from the encoded feature to the predicted type's training-feature reference distribution.

Use validation-fitted thresholds per predicted type. Output the predicted researched class only when all configured checks pass; otherwise output `IC`. The audit CSV records the raw four-class prediction, calibrated probability, probability margin, feature distance, final output, and rejection reason. Raw `.lig` piece bytes remain unchanged.

Because no reliable negative class is available, rejection thresholds control false rejection on known classes but cannot by themselves prove unknown-waveform detection. The existing IC data may be used only as a labelled-noisy diagnostic set to estimate false acceptance; it does not affect weights or primary threshold selection.

## Threshold Selection and Release Gates

For each type, search thresholds that satisfy a configurable precision floor, defaulting to 0.85, then choose the candidate with the highest retained coverage. Select the global threshold set by four-class macro-F1 and coverage, while enforcing a default 0.70 recall floor for every type. Report coverage rather than imposing an arbitrary minimum before the first benchmark. Store thresholds, temperatures, feature references, split hashes, and validation metrics in the checkpoint.

A candidate may replace the deployed model only when it:

- matches or exceeds the baseline four-class macro-F1;
- passes every per-type precision and recall floor;
- shows no abnormal prediction collapse toward one type;
- passes the locked temporal test without using it for model selection; and
- performs at least as well as the baseline in a fixed, blinded manual review of 2016 data.

Keep four-class checkpoints in a separate output directory so the deployed five-class model is never overwritten accidentally.

## Verification

Add synthetic tests for IC exclusion, balanced type sampling, four-class label ordering, checkpoint compatibility, threshold fitting, each rejection reason, CSV audit fields, and raw-byte preservation. Run the full test suite, compile check, a bounded inference smoke test, and a reproducible old-versus-new evaluation before deployment.
