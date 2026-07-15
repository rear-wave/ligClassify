# Cross-Validated Final Training Design

## Goal

Produce one fast deployment model for NCG, NNBE, PCG, and PNBE while using
source-file-isolated cross-validation for honest model selection and
calibration. IC remains an inference-only rejection result. Existing deployed
and candidate checkpoints remain immutable until the new workflow passes all
release gates.

## Evidence and Constraints

The current conditional candidate is stronger than the historical model in
aggregate: test within-200-km accuracy is 89.92% versus 81.60%, file-macro
within-200-km accuracy is 90.85% versus 78.54%, and MAE is 67 km versus 107 km.
It is not ready for release because PNBE/night/300–400 km falls to 25.21% on an
independent file.

The historical checkpoint is not an honest release baseline: 111 of 168
current test files and 12,686 of 17,106 test pieces overlap its reconstructed
training period. The current split also balances only coarse distance bands;
ten test type/day/100-km cells have no training file. Historical metrics will
therefore be reported as contaminated reference data, not used as automatic
promotion gates.

## Three-Fold Data Contract

Assign each trusted `.lig` source file deterministically to exactly one of
three folds. Optimize assignment over type, day/night, and exact 100-km label
interval, then file and piece totals. Every fold trains on two partitions and
predicts the third. Pooled out-of-fold (OOF) predictions must contain every
trusted piece exactly once, with no source-file overlap between a fold's train
and validation views.

Cells with fewer than three independent files remain valid but are marked
`insufficient_support`; their held-out results measure real extrapolation and
must not be hidden through same-file piece splitting. The final deployment
model may learn all trusted files after OOF evaluation passes.

Persist fold ownership, per-fold hashes, one combined hash, exact-bin support
counts, and OOF row identity. Abort before GPU training on duplicate files,
missing pieces, inconsistent labels, or non-deterministic hashes.

## Model and Optimization

Retain `conditional_expert_v1`: polarity-preserving local and global waveform
branches, a four-class type head, and one ordered distance expert per trusted
type. Correct the training and evaluation pipeline before considering a larger
architecture.

Because every trusted four-class file has a distance label, replace duplicate
type and distance streams with one joint stream. Sample with replacement by
the hierarchy type, day/night, exact 100-km interval, source file, then piece.
Limit each file's contribution per epoch while permitting rare cells to be
revisited. Every batch optimizes type and oracle-routed distance losses.

Use two stages: an initial type-focused stage with a reduced distance weight,
followed by joint optimization. Apply only synchronized, physically plausible
training augmentation: small time shifts, gain changes, baseline drift, and
additive noise. Never invert polarity or reverse time. Give distance experts a
day/night flag by default. Retain cyclic hour features only if a three-fold
ablation improves robust OOF metrics.

After basic type readiness, select each fold's checkpoint lexicographically by
the worst supported condition, file-equal within-200-km accuracy, worst
per-type within-200-km accuracy, and MAE. A supported condition has at least
three independent source files. Train the final model from random
initialization for the median best epoch across folds; do not warm-start it or
select its epoch on full-data training predictions.

## OOF Calibration and Metrics

Use one file-equal confusion calculation for calibration and release. Each
source file has total weight one, distributed across its pieces before TP and
FP are accumulated. This replaces the current conditional average in which a
single stray prediction can count like an entire incorrectly classified file.
Post-fit evaluation must call the same implementation and reject inconsistent
calibration claims.

Fit probability temperature, confidence thresholds, and margin thresholds
from pooled OOF predictions with a 96% per-type precision target. Transfer
robust median scalar parameters to the final model. Recompute feature centers
with the final model; transfer only fold-normalized feature-distance quantiles,
not raw embedding coordinates or distances. Keep distance temperature
calibration only when validation point metrics do not regress.

Report raw four-class results separately from IC-rejected end-to-end results.
Report file-equal and piece metrics, per-type results, day/night and distance
conditions, file-bootstrap intervals, rejected counts, and unsupported cells.
The term `100-km interval` is preferred over misleading `exact distance`.

## Release Contract

OOF results must satisfy all of the following before full-data training:

- accepted precision of at least 95% for every researched type;
- overall accepted coverage of at least 80%;
- file-equal macro recall of at least 90%;
- overall within-200-km accuracy of at least 85%;
- within-200-km accuracy of at least 75% for every type;
- within-200-km accuracy of at least 70% for every condition supported by at
  least three independent files.

Sparse conditions do not silently pass: they appear in a separate evidence
list and inference records receive `insufficient_support`. Inference still
returns the estimated distance and type. The contaminated historical model is
shown only as contextual reference.

Only a complete, passing OOF report authorizes final full-data training and
creation of `model.pt`. Any failed fold, missing OOF row, calibration mismatch,
or failed gate leaves the existing deployment unchanged and writes an
actionable candidate report.

## Artifacts, Recovery, and Inference

Use this output contract:

```text
weights/conditional_cv/
├── folds/fold_0/
├── folds/fold_1/
├── folds/fold_2/
├── fold_manifest.json
├── oof_predictions.csv
├── cv_metrics.json
├── support_map.json
├── final_candidate.pt
└── model.pt
```

Each fold stores independent latest and best checkpoints. Resume only
unfinished folds; never infer completion from directory existence alone.
Final checkpoint writes are atomic and followed by schema, hash, calibration,
and state-dict validation. Production inference remains a single-model path,
preserves original piece bytes, and adds support status to the audit CSV.

## Verification

Synthetic tests cover deterministic fold assignment, exact-bin balance,
file isolation, OOF uniqueness, support maps, replacement sampling, per-file
caps, synchronized polarity-safe augmentation, calibration/release metric
identity, worst-condition checkpoint selection, interrupted-fold recovery,
final checkpoint schema, support-status inference, and raw-byte preservation.

Before full training, run the complete unit suite, compile check, and a
three-fold small-sample smoke test. After training, validate every artifact,
recompute OOF gates from the saved CSV, and run bounded production inference.

## Rejected Alternatives

One file-isolated holdout is faster but wastes scarce distance coverage and
produces unstable single-file subgroup gates. Splitting pieces from one file
across train and validation covers every bin but leaks acquisition signatures
and overstates generalization. A three-model production ensemble may improve
accuracy but triples inference cost; the selected design keeps cross-validation
for evidence and deploys one full-data model.
