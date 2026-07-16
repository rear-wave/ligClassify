# Piece-Based Support Consistency Design

## Status and Scope

This document is an approved delta to
`2026-07-15-cross-validated-final-training-design.md`. It replaces that
design's independent-source-file support floor with one waveform-piece floor.
All unrelated cross-validation, calibration, release, final-training, and
byte-preserving inference requirements remain unchanged.

The change covers support classification, checkpoint selection, release gates,
checkpoint validation, benchmark audit records, contributor commands, and
regression tests. It does not weaken any accuracy, precision, coverage, or
within-200-km release threshold.

## Canonical Support Contract

Define one shared constant in `data/cross_validation.py`:

```python
MINIMUM_SUPPORTED_PIECES = 100
```

An exact type/daylight/100-km condition is `supported` if and only if its total
`piece_count` is at least 100. Otherwise it is `insufficient_support`.
`file_count` remains present in support maps and CSV audit data, but it is
descriptive metadata and never controls training, checkpoint selection,
release, checkpoint validation, or inference support status.

The same 100-piece threshold applies to:

- data audit and `support_map.json`;
- fold checkpoint selection;
- OOF support annotations and release gates;
- final checkpoint support maps and validation;
- production and benchmark support annotations.

No component may multiply the threshold by the fold count or define a private
support threshold.

## Component Changes

### Support aggregation and release

`build_support_map()` retains `minimum_pieces` as an optional parameter whose
default is `MINIMUM_SUPPORTED_PIECES`. `audit_data.py` uses the default rather
than a three-fold multiplier. `evaluation.py` imports the shared constant and
uses `piece_count` for both fold checkpoint selection and supported-condition
release gates.

`validate_final_checkpoint()` recomputes the expected status from
`piece_count >= MINIMUM_SUPPORTED_PIECES`. It still validates positive file and
piece counts, exact interval geometry, support-map hashes, and all four type
identities.

### Benchmark audit records

V3 benchmark evaluation annotates every internal prediction record after
distance temperature routing and rejection. Accepted records derive their
modal 100-km bin from the routed expert logits and use the checkpoint support
map. Rejected IC records are `not_applicable`.

Each V3 benchmark record retains:

- `support_status`, `support_file_count`, and `support_condition`;
- `fold_manifest_hash` and `full_data_hash`;
- the rejection policy's `calibration_hash`.

These fields are audit evidence only and do not alter benchmark metrics or
release decisions. Legacy V1/V2 behavior remains unchanged.

### CLI validation and documentation

V3 type-only inference rejects a non-zero `--min_type_confidence`, matching the
existing calibrated V1/V2 behavior.

The documented one-epoch smoke command includes `--type_focus_epochs 0`, so
the single epoch is a selectable joint-stage epoch. Contributor documentation
uses random initialization, `--resume_cv`, exact 100-km interval balancing, and
absolute OOF release gates. Historical-model metrics remain reference-only.

## Data Flow and Failure Behavior

Manifest entries are aggregated into exact conditions. Their piece counts
produce one canonical support map. Fold OOF rows, pooled evaluation, final
checkpoint construction, production inference, and benchmark inference all
consume that same status rule.

Sparse conditions remain visible and retain predicted type and distance.
Changing the support unit does not authorize final training: all existing OOF
release gates must still pass. A support-map row whose stored status disagrees
with its piece count causes final checkpoint validation to fail before either
`final_candidate.pt` or `model.pt` is promoted.

## Compatibility

V3 checkpoints continue to expose both `file_count` and `piece_count` in their
support maps. No checkpoint field is renamed. V1/V2 checkpoint loading and CSV
behavior remain compatible. Existing generated support artifacts may be
recomputed under the canonical 100-piece rule; model weights are not silently
promoted or overwritten.

## Test Strategy

Regression tests must demonstrate:

1. One source file with 99 pieces is insufficient and one source file with 100
   pieces is supported.
2. A condition with many source files but fewer than 100 total pieces remains
   insufficient.
3. Audit, fold selection, release, and final checkpoint validation all use the
   same constant.
4. A final checkpoint with a file-sparse but piece-supported row validates,
   while a row whose status contradicts its piece count is rejected.
5. V3 benchmark records contain support and audit hashes without changing
   accepted/rejected decisions or metrics.
6. V3 rejects a non-zero legacy confidence override.
7. The corrected one-epoch smoke arguments pass CLI validation.

Verification consists of focused red-green tests, the complete unit suite,
`compileall`, all three CLI help commands, `git diff --check`, and a read-only
`--verify_only` run when the active PyTorch environment is available. Real OOF
conditions below their fixed accuracy gates remain release blockers.
