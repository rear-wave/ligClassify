# Distance-First Dataset Split Design

## Goal

Make distance-distribution quality the primary split constraint. Acquisition dates do not determine split membership. IC remains excluded, every `.lig` file is indivisible, and no file may appear in more than one split.

## Split Roles

Training produces three named splits:

- `train` fits the four-class type and routed distance heads.
- `val` selects epochs and fits type rejection, distance temperature, and confidence thresholds.
- `test` is the locked release evaluation and must represent every available distance bin for each type.

All files are grouped by `(type, distance_bin)` and assigned directly to `train`, `val`, or `test`. Different files from the same acquisition date may enter different splits. File paths and piece bytes remain isolated across all three splits.

## Distance-Balanced Assignment

Assignment targets configurable piece proportions, initially 70/15/15, rather than file proportions. Files are ordered deterministically from the seed and assigned greedily within each type and 100 km bin to minimize deviation from the target piece histogram. Training retains at least one file in every available group.

Rare groups are never duplicated. Allocation priority is training, test, then validation. A group with two files therefore contributes one file to training and one to testing, while validation records the missing bin. A group with only one file remains in training and makes full independent test coverage impossible.

`test` must cover every fine 100 km bin available in the source data and at least 500 pieces per type; otherwise training stops. Validation must cover at least `max(12, ceil(0.9 * available_bins))` fine bins and 500 pieces per type. The report also groups fine bins into ten 300 km bands and requires all available coarse bands in both evaluation splits. With the current data this permits 30-bin NCG test coverage and 29-bin NCG validation coverage.

## Metrics and Release Policy

Logs and checkpoint metadata record split hashes, file and piece counts, date ranges, covered and missing bins, and per-type distance histograms. Distance reports include both raw piece-weighted metrics and metrics averaged equally over populated 100 km bins. Equal-bin metrics are primary, preventing large NCG files or common distances from dominating the result.

Early stopping and calibration use only `val`. Distance release gates use equal-bin metrics from `test`; type precision and recall are also checked there. The test set is evaluated only after model selection and calibration. A candidate with an invalid or incomplete split is rejected before optimization begins.

## Verification

Synthetic tests must prove deterministic assignment, three-way file isolation, approximate piece-ratio matching, retained training coverage, rare-bin allocation, validation-coverage checks, equal-bin metric aggregation, and failure on impossible test coverage. A real-data audit must print all three split summaries and demonstrate 30-bin NCG test coverage plus at least 27-bin NCG validation coverage. Before handoff, run the complete pytest suite, compile check, and a bounded split-only or one-epoch smoke test.
