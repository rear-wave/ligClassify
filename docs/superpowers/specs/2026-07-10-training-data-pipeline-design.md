# Training Data Pipeline Design

## Goal and Scope

Repair the multi-task training data path around `E:\Guoxing Yang\train_data` so type training reflects the stated real-world prior of approximately 80% IC, distance labels are parsed correctly, data splits are chronological and leakage-resistant, and training no longer loads roughly 2.43 million waveforms into memory. This change covers dataset discovery, timestamp parsing, splitting, sampling, evaluation, logging, and checkpoint metadata; it does not alter or delete the source `.lig` files.

## Confirmed Dataset Facts

The dataset contains 2,430,838 pieces: 2,306,817 IC pieces (94.9%) and 124,021 non-IC pieces. NCG, NNBE, PCG, and PNBE contain 30,719, 33,175, 22,787, and 37,340 pieces respectively. The current distance regular expression misses underscore directories such as `night_2400_2500km`, causing 150 files and 5,455 pieces to lose valid distance labels. Random file splitting also leaves many distance bins absent from validation and test sets.

## Data Manifest and Labels

Create pure helpers that discover `.lig` files and build a manifest using validated files, preventing path/label misalignment when invalid files are skipped. The binary timestamp in the first piece is canonical; a timestamp embedded in the filename is only a fallback. Normalize hour `24` to hour `00` on the following day and log it. Parse both `100-200km` and `100_200km` suffixes. A distance label is valid only when both boundaries are multiples of 100 km, the interval width is exactly 100 km, and the interval lies within 0-3000 km. Broader ranges remain type-only and are reported explicitly.

## Temporal Splitting and IC Sampling

Split at file level and treat `(type, acquisition_date)` as an indivisible group. Within each type, sort dates chronologically: the earliest dates form training, the next dates form validation, and the latest dates form test. Ensure at least one date per split when a type has at least three dates. Never move individual files to improve distance balance because that would reintroduce same-day leakage. Report missing distance bins and types with too few dates instead. A single global cutoff is not used because IC, NCG, NNBE, PCG, and PNBE cover materially different year ranges.

Apply IC downsampling only after the temporal split and only to the training view. Select pieces with a seeded sampler so IC represents 80% of type-training samples, equivalent to an IC:non-IC ratio of 4:1. Validation and test retain their natural distributions; macro-F1, not raw accuracy, drives the type portion of early stopping. Source files remain untouched, and a CLI option allows changing or disabling the target ratio.

## Lazy Loading and Evaluation

Store only manifest metadata and selected global piece indices. `__getitem__` reads and preprocesses one waveform on demand. Keep distance loss limited to valid non-IC distance labels. Report both oracle-routed distance metrics (ground-truth type selects the head) and end-to-end metrics (predicted type selects the head), so deployment performance is not overstated.

## Checkpoints, Diagnostics, and Safety

Save model width, class mappings, preprocessing mode, temporal split fractions, target IC fraction, and distance-bin rules with the state dictionary. Log each split's date range, file/piece counts, IC fraction, distance coverage, missing bins, normalized timestamps, and skipped invalid files. Existing checkpoints remain loadable through a documented legacy fallback.

## Verification

Add synthetic `pytest` coverage for binary timestamps, hour-24 normalization, filename fallback, both distance naming styles, invalid/broad intervals, chronological date grouping, 80% IC sampling, lazy reads, skipped-file alignment, and predicted-vs-oracle distance routing. Run the full tests, `python -m compileall -q .`, CLI help smoke tests, and a manifest-only audit against the real dataset. No test or implementation step may rewrite files under `E:\Guoxing Yang\train_data`.
