# Reliable Distance Classification Design

## Goal and Acceptance Criteria

Improve non-IC distance estimation without weakening the existing type classifier or leaking future acquisition dates into training. The locked chronological test set is the final authority. For oracle-routed distance heads at full coverage, the target is an overall distance MAE of at most 200 km, overall accuracy within +/-200 km (`w2`) of at least 80%, and `w2` of at least 70% for each of NCG, NNBE, PCG, and PNBE. The end-to-end path must additionally retain at least 95% distance coverage and at least 75% `w2` when samples routed to IC are counted as failures. Confidence-based rejection is an additional operational safeguard and cannot be used to hide failure of the full-coverage target.

The current checkpoint is the frozen baseline: 329 km MAE and 52.62% `w2`. Its 7.98% exact-bin accuracy is below the 14.65% per-type majority-bin baseline, while its MAE is only modestly better than the 351 km per-type median baseline. The design therefore optimizes ordinal distance quality rather than exact 30-way classification alone.

## Data and Evaluation Protocol

Keep the existing file-level, per-type chronological split. Never rebalance validation or test data, move files between dates, or tune against test results. Persist a split manifest or stable file-list hash in every experiment so all candidates use identical examples.

Construct a distance-only training stream from labelled non-IC pieces. Sampling is hierarchical: choose a lightning type, then an available distance bin, then an acquisition date/file, then a piece. Use only bins that actually exist; do not synthesize examples for missing bins. Limit repeated selections from a single file in one epoch so sparse bins do not become memorized date signatures. The existing type stream keeps the 80% IC prior. Alternate batches from the two streams until each has supplied one epoch of samples, cycling neither stream. Type batches optimize type loss; distance batches optimize distance loss plus a small type-loss term so the shared representation does not drift. Each optimizer step handles only one stream, keeping memory bounded and the contribution of each task explicit.

Primary model selection uses validation `w2`, MAE, and the worst per-type `w2`. Exact-bin accuracy remains diagnostic. Compare three deterministic seeds and group-bootstrap confidence intervals by acquisition file/date rather than treating correlated pieces as independent. The final candidate is evaluated once on the locked test split. A rolling-origin temporal diagnostic is run for the finalist where each class has enough dates.

## Model Architecture

Add a versioned `MultiTaskOrdinalResNet` while retaining legacy checkpoint loading. Preserve the shared ResNet temporal encoder and type head. The distance branch receives the final temporal feature map and concatenates adaptive average and maximum pooling, followed by a small normalized MLP with dropout. Four type-conditional 30-bin heads then produce ordered distance distributions.

The 10 coarse 300 km probabilities are derived by summing each consecutive group of three fine-bin probabilities. This guarantees consistency between fine and coarse outputs and avoids contradictory auxiliary heads. Date, directory name, distance folder, and file path must never be model inputs. Raw-amplitude or day/night features are optional later ablations because the observed amplitude-distance correlations are weak; they are not part of the first candidate.

## Ordinal Training Objective

For fine-bin probabilities `p`, target bin `y`, and a normalized exponential soft target `q`, use:

`L_dist = L_soft_ce + lambda_emd L_cdf + lambda_reg L_huber + lambda_coarse L_coarse`.

- `L_soft_ce` is cross-entropy against `q`, so adjacent bins receive partial credit.
- `L_cdf` is the mean absolute difference between predicted and target cumulative distributions, directly penalizing how far probability mass moves.
- `L_huber` compares the expected predicted bin with `y` and stabilizes MAE optimization.
- `L_coarse` is negative log probability of the correct 300 km group.

Initial coefficients are `lambda_emd=1.0`, `lambda_reg=0.5`, and `lambda_coarse=0.5`, with soft-label temperature `1.0`; the type-loss coefficient on distance batches is `0.1`. They are fixed before the locked test and changed only through validation ablations. Distance loss is averaged per type so the largest class cannot dominate. Early stopping ranks candidates lexicographically by minimum per-type `w2`, macro-average per-type `w2`, negative macro-average MAE, and finally type macro-F1. After every type reaches the 70% guardrail, overall `w2` replaces minimum per-type `w2` as the first key. Patience resets for at least 0.2 percentage points of `w2`, 5 km of MAE, or 0.1 percentage points of type macro-F1 when preceding keys are tied.

## Prediction and Reliability Output

Use the expected fine-bin center, `100 * (E[bin] + 0.5)` km, as the point estimate and retain argmax-bin metrics for comparison. Fit one validation-only temperature per distance head. Inference returns type, expected distance, 100 km bin probabilities, 10th-90th percentile interval, and confidence defined as calibrated probability mass within +/-200 km of the estimate. A validation-selected confidence threshold may mark a result `uncertain`; reports must show the resulting accuracy-coverage curve and coverage at the selected threshold.

Versioned checkpoints store architecture name, pooling/MLP dimensions, loss coefficients, sampler policy, split hash, calibration temperatures, confidence threshold, and metrics. `infer_mtl.py` detects the version and keeps the current legacy fallback.

## Staged Experiments and Stop Rules

Run cumulative ablations on the unchanged split:

1. Reproduce the current checkpoint and naive majority/median baselines.
2. Add only hierarchical distance sampling.
3. Add the ordinal multi-term objective and expected-bin prediction.
4. Add the distance-specific pooling/MLP branch.
5. Calibrate probabilities and select the rejection threshold.

Promote a stage only when the median of three seeds improves macro `w2` or MAE and no class loses more than two percentage points of `w2`. Type macro-F1 may not fall more than one percentage point from the frozen baseline. If the best validated model misses the test target, report the failing types and date/bin support instead of tuning on test. In particular, persistent NCG failure triggers a data recommendation for multiple independent acquisition dates per distance bin.

## Verification

Add unit tests for hierarchical sampling, empty/missing bins, ordinal loss monotonicity, coarse-probability aggregation, expected-distance decoding, confidence intervals, temperature serialization, early-stopping ordering, and legacy/v2 checkpoint loading. Run `pytest`, `python -m compileall -q .`, CLI help smoke tests, a one-batch CUDA forward/backward test, and a small end-to-end inference export before full experiments.
