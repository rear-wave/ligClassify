# Guarded second-model verification experiment

Status: piece-test-confirmed research candidate, not a deployed classifier replacement. Existing
bundles, thresholds and external classification outputs are unchanged. This
experiment does not use temporal context or retrain the distance models.

## Why this candidate

First-round multi-scale models did not beat the existing deployment tradeoff.
Simply tightening all thresholds under multiple corruptions improved precision
but reduced clean known-class macro recall to about 91%. A second model can
provide additional evidence for a rejected piece without changing a primary
model decision that was already accepted.

The primary is the original hierarchical checkpoint; the verifier is the new
32-channel hierarchical checkpoint from random initialization. Each remains a
five-class IC/known gate, conditional known head and prototype model with local
and global waveform features. This is selective verification, not an AnchorMoE
expert-routing modification or a standalone four-class classifier.

## Fixed rule

1. Keep every primary non-IC result unchanged.
2. For a primary IC result, require its two conditional views to agree and at
   least two local/global branch votes to agree with the candidate.
3. Check the primary candidate probability and IC gate against per-class limits
   before invoking the verifier.
4. The verifier must independently accept the same class under its own existing
   decision rule. Both models must also satisfy joint confidence/gate limits.
5. Otherwise retain IC. If accepted, downstream distance routing must use the
   recovered final class; a future production integration must test this path.

This cannot correct an already accepted wrong primary label. Correlated model
errors remain possible; it is not a guarantee that recovered pieces are correct.

## Data separation and calibration

The original seed-42 piece-level split is unchanged. Both checkpoint split hashes
match the current data split. Calibration uses 17,714 validation pieces, excluding
every key in the balanced 2,000-piece development evaluation subset. No test
labels enter threshold fitting. Checkpoint hashes and frozen recipe hashes are
stored with the outputs.

One fixed set of per-class limits is fitted across clean inputs, baseline drift
and 80 kHz low-pass perturbation. Every calibration condition must satisfy its
precision and incremental false-acceptance constraints; pooled performance
cannot hide failure in one condition. Calibration precision uses uniform class
weights for comparison, not an assumed 80% deployment IC proportion.

The grid-count implementation is checked against brute-force synthetic counts.
Synthetic rescue checks verify that accepted labels cannot change and that
disagreement, unstable views, rejected verifier decisions and gate violations
cannot rescue a piece.

The recipe was frozen before additional noise/impulse evaluation and latency
measurement. Its piece-test confirmation is read-only: thresholds must not be
adjusted using those test results. A piece-level split can share source files
across partitions; it is not an unseen-station, unseen-year or continuous-stream
validation.

## Development validation results

Known-class recall is the mean recall of NCG, NNBE, PCG and PNBE. NBE precision
below is the lower of the separately measured NNBE and PNBE precisions, not a
pooled binary score. Each class has 400 pieces.

| Condition | Primary recall | Guarded recall | Guarded minimum NBE precision |
| --- | ---: | ---: | ---: |
| Clean | 93.81% | 98.06% | 99.74% |
| 20 dB noise | 93.81% | 97.94% | 99.74% |
| Baseline drift | 90.25% | 95.63% | 99.75% |
| 80 kHz low-pass | 92.25% | 96.69% | 97.84% |
| Impulse interference | 93.63% | 97.56% | 99.74% |

These are empirical development-set measurements, not population-level precision
guarantees. The development subset has been inspected during model research.
Some recovered labels are wrong; all confusion matrices and recovery counts are
retained in the local report.

## Frozen piece-test confirmation

The frozen recipe was evaluated once on all 19,673 original test pieces. There
was zero piece-key overlap with calibration or development validation. No test
results were used to change thresholds. The test results confirm this recipe's
within-dataset improvement; they do not establish cross-source independence.

| Known-class recall | Primary | Guarded |
| --- | ---: | ---: |
| NCG | 96.58% | 98.09% |
| NNBE | 92.40% | 97.17% |
| PCG | 94.81% | 98.39% |
| PNBE | 89.79% | 95.18% |
| Four-class macro average | 93.39% | 97.21% |

On clean test inputs, 731 previously rejected pieces were correctly recovered;
18 rejected pieces were accepted with wrong labels. Of those 18, one true IC
became PCG, one true PCG became PNBE, one true PNBE became NNBE, and 15 true PNBE
became PCG. Wrongly accepting a previously rejected known piece is not the same
as turning a previously correct classification into an error; report both the
new false accepts and the complete confusion matrices.

NNBE/PNBE empirical precision was 99.92%/99.94% under the original test class mix,
or 99.77%/99.77% after uniform five-class reweighting. Precision depends on the
class mix and must not be promised for an IC-dominated operational stream.

| Test condition | Primary known recall | Guarded known recall | Lower NBE precision, uniform class weights |
| --- | ---: | ---: | ---: |
| Clean | 93.39% | 97.21% | 99.77% |
| 20 dB noise | 93.31% | 97.17% | 99.77% |
| Baseline drift | 89.76% | 95.75% | 99.47% |
| 80 kHz low-pass | 91.88% | 96.49% | 97.78% |
| Impulse interference | 93.14% | 97.09% | 99.77% |

Clean known-class paired-view agreement was 99.27% for the primary and 98.60% for
the verifier. Per-class false-to-IC rates, paired NBE/CG confusions, counts and
all condition-specific consistency metrics are recorded in the test JSON. The
distance models were not retrained or evaluated by this type-only experiment.

## Runtime scope

Only 4.9% of clean development pieces triggered the verifier after primary
screening. This fraction depends on the incoming data distribution.

On this machine with four CPU inference threads and concurrent training, the
research implementation's 183-piece timing sample had a type-only p95 of 7.97 ms.
The verifier-triggering subset had p95 8.23 ms. The sample deliberately includes
extra verification-triggering cases and is not a natural stream arrival mix.
Preprocessing is included; acquisition, disk/network reads, queueing, temporal
state, distance inference and output persistence are excluded. All 183 single
predictions matched batched predictions.

## Reproducibility and remaining work

The experiment source and execution order are versioned under
[`research/streaming/`](../research/README.md).

Ignored local artifacts are under `weights/streaming_research_v1/`:

- `constraint_diagnosis.json`: network candidates versus individual constraints.
- `robust_calibration_report.json`: stricter multi-condition calibration control.
- `guarded_rescue_frozen.json`: checkpoint-bound fixed recipe.
- `guarded_rescue_validation_report.json`: all five development conditions and latency.
- `guarded_rescue_test_plan.json` / `guarded_rescue_test.json`: frozen piece-test protocol/results.
- `round2_plan.json`: the separate inference-aligned training ablation.

The guarded verifier is now an explicit opt-in path in `classify.py` and
`StreamingClassifier`; the default path is unchanged. It requires a recipe with
relative verifier checkpoint path, SHA-256 hashes and four per-class limits, and
fails closed on mismatched model, split, preprocessing or decision hashes. It
cannot be combined with temporal context, `--direct_type` or a decision override.
The verifier checkpoint is intentionally not committed with source code.

Remaining deployment gates include production integration review,
distance routing and end-to-end latency checks, external labeled negative and
positive data, and chronological unsorted-arrival/restart replay. Do not replace
the default classifier based only on this development result.
