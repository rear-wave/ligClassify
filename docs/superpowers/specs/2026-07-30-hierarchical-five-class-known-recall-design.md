# Hierarchical Five-Class Known-Recall Design

## Problem

The deployed type checkpoint classifies many visually typical CG and NBE
waveforms as IC with saturated confidence. On the 2016-07-08 output, 143,539
of 153,985 pieces were classified as IC. At least 90% of those IC predictions
assigned less than 0.000662 probability to every non-IC class. A representative
NNBE piece received 0.999975 IC probability, so an inference-only confidence
threshold cannot correct the failure safely.

The training audit found 5,020 IC pieces versus 22,787–37,340 pieces in each
known class. The current 120,000-sample epoch assigns 60% to IC, repeating each
training IC piece about 20 times per epoch. Piece-level random splitting also
places highly related pieces from the same LIG container into training,
validation, and test, making the reported 99.76% test accuracy optimistic.

## Goals

- Retain IC as a trained type while prioritizing recall for NCG, NNBE, PCG,
  and PNBE.
- Use both high-resolution local morphology and the complete waveform.
- Make similar known waveforms produce consistent features and predictions.
- Send a waveform to IC only after explicit known-class review.
- Preserve the existing four class-specific distance models and LIG byte
  preservation.

IC precision is a monitoring metric, not the primary optimization target.
The referenced 2016 NNBE example remains diagnostic only and is not added to
training, validation, or regression fixtures.

## Model Architecture

One shared dual-scale encoder produces a fused embedding:

- The local branch uses a robust energy-envelope event center and a
  high-resolution window to represent polarity, rise and fall shape, pulse
  width, overshoot, ringing, and pre/post-pulse structure.
- The global branch processes all 16,000 samples to represent multi-pulse
  structure, long decay, background variation, and total energy distribution.
- A learned gate fuses local and global embeddings without allowing either
  branch to silently replace the other.

Three heads consume the fused embedding:

1. An IC gate predicts `IC` versus `known`.
2. A conditional known-type head predicts `NCG`, `NNBE`, `PCG`, or `PNBE`.
3. A metric head compares the embedding with multiple prototypes per known
   type, covering distance and daylight variation.

IC participates in gate and five-class supervision, but IC embeddings are not
forced into one compact cluster. Supervised contrastive learning applies only
to the four known types.

## Training

Each type occupies 20% of a batch. Epoch size is bounded by available IC
training pieces so that IC is sampled without repeated replacement within an
epoch. Known-class samples rotate across epochs until all pieces are covered.

Each waveform produces two safe augmented views using small translations,
positive gain changes, mild baseline drift, noise, and filter variation.
Polarity is never inverted. The objective combines:

- IC/known gate loss;
- four-class conditional cross-entropy;
- five-class monitoring loss;
- known-class supervised contrastive loss;
- prediction consistency across augmented views;
- mild label smoothing to reduce saturated errors.

Validation and test ownership are grouped by source LIG file while retaining
type, daylight, and distance stratification. This isolation is required for
evaluation; after hyperparameters and thresholds are locked, an optional final
deployment fit may use all training files.

## Inference

Every piece always runs through the IC gate, known-type head, metric prototypes,
local/global diagnostics, and two lightweight consistency views.

A known type overrides an IC gate prediction only when:

- the known-type probability is sufficient;
- prototype similarity for the same type is sufficient;
- local and global evidence agree;
- augmented views remain stable.

If known evidence is weak, conflicting, or unstable, the final type is IC.
Final IC pieces do not run a distance model. Known pieces route to the matching
existing distance checkpoint.

`predictions.csv` adds:

- `ic_gate_probability`
- `known_type_probability`
- `prototype_similarity`
- `local_prediction`
- `global_prediction`
- `consistency_score`
- `decision_reason`
- `candidate_known_type`

The principal decision reasons are `known_direct`, `known_evidence_override`,
and `no_stable_known_match`.

## Validation and Release

A candidate type model is accepted only when the source-file-isolated test set
satisfies all of the following:

- known-type macro recall is at least 95%;
- each known type has at most 5% false rejection to IC;
- augmentation consistency is at least 97%;
- NNBE/PNBE and NCG/PCG confusion is reported separately;
- calibrated probabilities do not remain saturated on clear errors.

The current model is evaluated as a baseline under the same isolated protocol.
The candidate must improve known-to-IC rejection without a material regression
in four-class conditional accuracy. Real 2016 data remains an external,
non-training distribution check and is reviewed separately.

## Compatibility and Safety

Training data and external classifications are read-only. Checkpoints use a new
schema version so older five-class checkpoints cannot be loaded silently into
the hierarchical inference path. Existing distance checkpoints remain
unchanged during the first implementation phase. Raw LIG record bytes,
timestamps, and output grouping behavior remain byte-preserving.
