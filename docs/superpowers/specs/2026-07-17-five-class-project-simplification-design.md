# Five-Class Project Simplification Design

## Goal

Replace the current four-class cross-validation and rejection system with one
compact five-class multi-task pipeline. The new pipeline trains `IC`, `NCG`,
`NNBE`, `PCG`, and `PNBE` directly, predicts IC by ordinary five-class argmax,
and predicts distance only for the four researched non-IC types.

The refactor must materially reduce code, tests, documentation, and generated
weights while preserving the useful modern waveform representation and the
known-good legacy five-class model.

## Non-Goals

- Do not retain three-fold CV, OOF calibration, open-set rejection, promotion
  gates, support maps, fold resumption, or subgroup release machinery.
- Do not reproduce the old `train_mtl.py` implementation verbatim.
- Do not warm-start the new model from any old checkpoint.
- Do not modify `../train_data/` or external classified waveform outputs.
- Do not treat `.lig` files as statistical groups. They are storage
  containers holding at most 512 independent waveform pieces.

## Final Runtime Structure

The intended runtime surface is:

```text
train.py                 five-class training CLI
classify.py              byte-preserving inference CLI
audit_data.py            manifest and split audit CLI
models.py                new and legacy model definitions
training.py              losses, epoch loop, early stopping, resume
evaluation.py            piece-level type and distance metrics
checkpoints.py           new/legacy checkpoint loading and validation
data/
  lig.py                  binary LIG indexing and piece reads
  manifest.py             file and piece labels
  preprocess.py           signed local/global preprocessing
  dataset.py              lazy piece dataset
  sampling.py             five-class hierarchical sampler
  split.py                deterministic piece-level split
tests/                    focused synthetic regression tests
```

These are the final module names. CLI orchestration remains outside `data/`.

## Model

The new checkpoint schema is `five_class_v1`. The model has:

- one signed local waveform branch;
- one signed global waveform branch;
- one scalar daylight context input;
- one five-class type head ordered as `IC`, `NCG`, `NNBE`, `PCG`, `PNBE`;
- four type-specific 30-bin distance experts ordered as `NCG`, `NNBE`, `PCG`,
  `PNBE`.

The shared representation is trained jointly. IC rows contribute only to the
type loss. Non-IC rows contribute to both type loss and an interval-aware,
ordered distance loss, with the ground-truth type selecting the distance expert
during training. Validation, test, and inference route distance through the
predicted type so reported results match deployment behavior.

The new model always starts from random initialization. Non-empty warm-start
arguments are rejected instead of silently loading old encoder tensors.

## Preprocessing and Augmentation

Preprocessing preserves waveform polarity and creates synchronized local and
global views from the same raw or augmented piece. Training-only augmentation
may apply zero-filled time shifts, strictly positive gain, low-amplitude linear
baseline drift, and robust-scale Gaussian noise. It must never negate, reverse,
or circularly shift a waveform. Validation, test, and inference are never
augmented.

Daylight is derived consistently from the piece timestamp and passed as a
single context value because daytime and nighttime waveforms may differ.

## Piece-Level Manifest and Split

Every piece has the stable identity:

```text
<source-relative-path>#<piece-index>
```

The split unit is the waveform piece, not its `.lig` container. Pieces from one
file may appear in train, validation, and test. The split is deterministic and
does not use acquisition year or chronological cutoffs.

Strata are:

- `(type, daylight, exact 100-km interval)` for NCG, NNBE, PCG, and PNBE;
- `(IC, daylight)` for IC.

Within each stratum, piece identities are ordered by a stable seeded hash and
assigned 70% to training, 15% to validation, and 15% to test. A stratum with at
least three pieces receives at least one validation and one test piece. Smaller
strata remain training-only and are reported as insufficient for evaluation.
The split artifact remains compact: it records the algorithm version, seed,
ratios, manifest hash, partition hashes, and per-stratum counts. Piece ownership
is reconstructed from the same stable hash rule, so it does not store millions
of piece rows and does not depend on input enumeration order.

Identity duplication and missing ownership are hard errors. Full raw-waveform
duplicate hashing is available through an explicit
`audit_data.py --check_duplicates` mode and never runs implicitly in the
training hot path.
Exact duplicate waveforms assigned to different partitions cause that audit to
fail with their source identities.

## Sampling and Training

The training sampler produces a fixed number of samples per epoch with this
type prior:

- IC: 60%;
- NCG: 10%;
- NNBE: 10%;
- PCG: 10%;
- PNBE: 10%.

Within IC, day and night are balanced when both exist. Within each non-IC type,
daylight and exact 100-km intervals are sampled hierarchically with replacement.
Files receive no special weight or cap because they are storage containers, not
independent observational groups. Validation and test retain their unmodified
piece distribution.

Training uses one DataLoader, AMP on CUDA, lazy batch reads, AdamW, cosine
learning-rate decay, and deterministic seeds. The total loss is:

```text
type_cross_entropy + distance_weight * interval_distance_loss
```

The default distance loss weight is `0.5`. The default early-stopping score is:

```text
0.5 * five_class_macro_f1
+ 0.5 * mean_non_ic_within_200_km
```

Patience defaults to 10. The best epoch is selected only on validation data.
The test partition is evaluated once after selection. Validation must contain
all four non-IC types; otherwise training stops with a data-coverage error
instead of averaging an incomplete distance score.

Each run writes only:

```text
model.pt       best inference checkpoint
last.pt        exact resumable training state
metrics.json   final validation and test metrics
split.json     compact ownership hashes and per-stratum counts
```

Resume validates the checkpoint schema, model configuration, split hash, and
training configuration before restoring optimizer, scheduler, scaler, epoch,
and RNG state.

## Inference and Checkpoint Compatibility

`classify.py` supports exactly two schemas:

1. `legacy_five_class`, represented by the retained
   `weights/old/model.pt`;
2. `five_class_v1`, produced by the new trainer.

Checkpoint-specific construction and preprocessing are isolated in
`checkpoints.py`. The main inference loop receives one normalized prediction
contract regardless of schema.

Type inference is direct argmax with no rejection threshold. IC predictions go
to `IC/` and have no distance. A non-IC prediction routes through the matching
distance expert and is written to a directory such as `NNBE_500-600km/` using
the modal distance bin.

`predictions.csv` includes source path, piece index, final type, `prob_IC`,
`prob_NCG`, `prob_NNBE`, `prob_PCG`, `prob_PNBE`, type confidence, modal distance
interval, expected distance, distance confidence, checkpoint schema, and model
SHA-256. Output `.lig` files preserve every original piece byte; inference may
regroup pieces but may not reconstruct waveform payloads.

## Cleanup and Data Safety

Before deleting weights, cleanup verifies that `weights/old/model.pt` has SHA-256
`f3d74de7f1e1ad5b7f3a796f3a145c58e6f9fc86730c45a84f74a634a2cf82ea`.
Cleanup then retains only:

```text
weights/old/model.pt
weights/old/metrics.json
```

All other existing weight directories, candidates, fold checkpoints, optimizer
states, OOF CSVs, support maps, and generated audit artifacts are deleted.
Future training writes to a fresh ignored output directory such as
`weights/five_class/`.

The implementation removes CV, OOF, open-set rejection, release-gate, support
map, and historical benchmark code and their dedicated tests. Obsolete
`docs/superpowers/` plans/specifications are deleted except this approved design
and its implementation plan. A concise `README.md` and updated `AGENTS.md`
describe only the resulting workflow.

Per explicit user direction, the uncommitted release-threshold edit is
discarded and the untracked `CLAUDE.md` and `_check_gates.py` files are deleted.
Deletion commands are restricted to the repository and never target training
data or classified outputs outside it.

## Testing and Acceptance

All fixtures remain synthetic. The focused suite covers:

- valid and invalid LIG parsing plus byte-exact piece reads;
- piece identity, deterministic stratified 70/15/15 ownership, and no missing
  or duplicate assignments;
- exact-waveform duplicate audit failures across partitions;
- the 60/10/10/10/10 sampler prior and exact-interval balance;
- polarity-preserving synchronized preprocessing and augmentation;
- five-class and four-expert output shapes;
- IC masking and predicted-type distance routing;
- early stopping and exact resume validation;
- both retained checkpoint schemas;
- IC and non-IC inference routing, CSV audit fields, bounded memory, and raw-byte
  preservation.

Before handoff, run the complete test suite, compile check, CLI help checks, a
bounded synthetic training smoke test, a legacy checkpoint load smoke test, and
`git diff --check`. The final repository must contain no `.lig` fixtures,
generated classifications, new model weights, credentials, or machine-specific
absolute paths.
