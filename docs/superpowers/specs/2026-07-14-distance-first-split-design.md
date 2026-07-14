# Distance-First Dataset Split Design

## Goal

Make distance-distribution quality the primary split constraint without losing a separate measurement of future-date generalization. IC remains excluded. Every `.lig` file is indivisible, and no file used for training may appear in validation or either test view.

## Split Roles

Training produces four named splits:

- `train` fits the four-class type and routed distance heads.
- `val` selects epochs and fits type rejection, distance temperature, and confidence thresholds.
- `balanced_test` is the primary release evaluation for distance accuracy. It combines the locked temporal files with additional stratified test files so that every source distance bin is represented whenever file isolation makes that possible.
- `temporal_test` contains the latest complete acquisition dates for each type and measures future-date robustness only. It is never used for selection or calibration.

The temporal holdout is reserved first. Remaining files are assigned to `train`, `val`, and a `balanced_extra` pool within each `(type, distance_bin)` group. The public `balanced_test` view is the union of `temporal_test` and `balanced_extra`; their intentional overlap is recorded in metadata. Validation and balanced-extra files may share dates with training files, but never file identities. The temporal test has dates disjoint from training and validation. No evaluation file appears in training.

## Distance-Balanced Assignment

After reserving temporal dates, assignment targets configurable development-piece proportions, initially 70/15/15 for `train`, `val`, and `balanced_extra`, rather than file proportions. Files are ordered deterministically from the seed and assigned greedily to minimize deviation from the target piece histogram for every type and 100 km bin. Training retains at least one development file in every available group.

Rare groups are never duplicated. Allocation priority is training, balanced testing, then validation. A bin already represented by `temporal_test` does not require an extra balanced-test file. If too few independent files exist, the missing validation coverage is reported rather than manufactured through replacement or file leakage.

`balanced_test` must cover all 30 fine 100 km bins for each type when those bins exist in the source data; otherwise training stops. Validation must retain at least 12 fine bins and 500 pieces per type. The report also groups fine bins into ten 300 km bands and requires all source coarse bands in `balanced_test`; temporal-test gaps are reported but are not filled. These constraints are feasible for the current data while acknowledging that NCG validation and future dates cannot independently cover all 30 bins.

## Metrics and Release Policy

Logs and checkpoint metadata record split hashes, overlap between the two test views, file and piece counts, date ranges, covered and missing bins, and per-type distance histograms. Distance reports include both raw piece-weighted metrics and metrics averaged equally over populated 100 km bins. Equal-bin metrics are primary, preventing large NCG files or common distances from dominating the result.

Early stopping and calibration use only `val`. Distance release gates use equal-bin metrics from `balanced_test`; type precision and recall are also checked there. `temporal_test` results are reported separately as a robustness diagnostic and cannot improve the release decision. A candidate with an invalid or incomplete balanced split is rejected before optimization begins.

## Verification

Synthetic tests must prove deterministic assignment, train/evaluation file isolation, intentional test-view overlap, temporal-date isolation from train/validation, approximate piece-ratio matching, retained training coverage, rare-bin diagnostics, equal-bin metric aggregation, and failure on impossible balanced-test coverage. A real-data audit must print all four split summaries, produce 30-bin NCG balanced-test coverage, and preserve the two-date NCG temporal diagnostic. Before handoff, run the complete pytest suite, compile check, and a bounded split-only or one-epoch smoke test.
