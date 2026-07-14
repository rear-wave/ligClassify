# Distance-First Dataset Split Design

## Goal

Make distance-distribution quality the primary split constraint by splitting individual waveform pieces inside every type and distance group. IC remains excluded. Files are allowed to contribute different pieces to multiple splits.

## Split Roles

Training produces three named splits:

- `train` fits the four-class type and routed distance heads.
- `val` selects epochs and fits type rejection, distance temperature, and confidence thresholds.
- `test` is the locked release evaluation and must represent every available distance bin for each type.

Every waveform piece is identified by `(filepath, piece_index)` and reads its own timestamp from the piece header. Pieces are grouped by `(type, distance_bin)`, ordered by `(timestamp, filepath, piece_index)`, and cut contiguously: the earliest 70% enter `train`, the next 15% enter `val`, and the latest 15% enter `test`. The same file may therefore occur in multiple splits, but the same piece identity may occur in only one.

## Distance-Balanced Assignment

The 70/15/15 proportions are configurable and applied independently to every 100 km bin of every type. Integer boundaries use deterministic constrained largest-remainder rounding: one piece is reserved for each split before the remaining count is allocated toward the target proportions. Groups with fewer than three pieces fail validation because they cannot support independent evaluation. An unreadable piece timestamp also fails manifest construction with its file and piece index; filename fallback is not used for piece ordering.

The split index stores only piece identities and timestamps; it does not copy or rewrite source `.lig` files. A shared file index is constructed once, and each dataset view maps its selected identities back to the original bytes. This avoids indexing the same large files three times while preserving lazy waveform loading.

Both evaluation splits must cover every fine 100 km bin available in the source data and at least 500 pieces per type; otherwise training stops. Because splitting occurs inside each bin, the expected current-data coverage is 30 bins for every type in both validation and testing.

## Metrics and Release Policy

Logs and checkpoint metadata record hashes of `(filepath, piece_index)` identities, piece counts, timestamp ranges, covered and missing bins, distance histograms, and the number of source files shared across split views. A validation check proves that piece identities are disjoint even when file paths overlap. Logs explicitly warn that shared-file evaluation does not measure cross-file generalization. Distance reports include both raw piece-weighted metrics and metrics averaged equally over populated 100 km bins. Equal-bin metrics are primary, preventing large files or common distances from dominating the result.

Early stopping and calibration use only `val`. Distance release gates use equal-bin metrics from `test`; type precision and recall are also checked there. The test set is evaluated only after model selection and calibration. A candidate with an invalid or incomplete split is rejected before optimization begins.

## Verification

Synthetic tests must prove deterministic timestamp ordering, exact piece-identity isolation, allowed file overlap, per-group ratio rounding, full distance-bin coverage, invalid timestamp handling, equal-bin metric aggregation, and failure for groups too small to split. A real-data audit must print all three summaries, demonstrate 30-bin validation and test coverage for every type, and report the amount of cross-split file overlap. Before handoff, run the complete pytest suite, compile check, and a bounded split-only or one-epoch smoke test.
