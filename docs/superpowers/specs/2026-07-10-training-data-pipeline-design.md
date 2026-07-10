# Training Data Pipeline Design

## Goal and Scope

Repair the multi-task training data path around `E:\Guoxing Yang\train_data` so type training reflects the stated real-world prior of approximately 80% IC, distance labels are parsed correctly, data splits are reproducible and leakage-resistant, and training no longer loads roughly 2.43 million waveforms into memory. This change covers dataset discovery, splitting, sampling, evaluation, logging, and checkpoint metadata in `train_mtl.py`; it does not alter or delete the source `.lig` files.

## Confirmed Dataset Facts

The dataset contains 2,430,838 pieces: 2,306,817 IC pieces (94.9%) and 124,021 non-IC pieces. NCG, NNBE, PCG, and PNBE contain 30,719, 33,175, 22,787, and 37,340 pieces respectively. The current distance regular expression misses underscore directories such as `night_2400_2500km`, causing 150 files and 5,455 pieces to lose valid distance labels. Random file splitting also leaves many distance bins absent from validation and test sets.

## Data Manifest and Labels

Create pure helpers that discover `.lig` files and build a manifest using the files actually accepted by `LigFileIndex`, preventing path/label misalignment when invalid files are skipped. Parse both `100-200km` and `100_200km` suffixes. A distance label is valid only when both boundaries are multiples of 100 km, the interval width is exactly 100 km, and the interval lies within 0-3000 km. Broader ranges remain type-only and are reported explicitly.

## Splitting and IC Sampling

Split at file level to avoid waveform leakage. Stratify deterministically by `(type, distance_bin)` for distance-labelled non-IC files and by type for type-only files. Sparse strata that cannot populate all three splits stay deterministic and are listed in the summary.

Apply IC downsampling only to the training view. Select pieces with a seeded sampler so IC represents 80% of type-training samples, equivalent to an IC:non-IC ratio of 4:1. Validation and test retain their natural distributions; macro-F1, not raw accuracy, drives the type portion of early stopping. Source files remain untouched, and a CLI option allows changing or disabling the target ratio.

## Lazy Loading and Evaluation

Store only manifest metadata and selected global piece indices. `__getitem__` reads and preprocesses one waveform on demand. Keep distance loss limited to valid non-IC distance labels. Report both oracle-routed distance metrics (ground-truth type selects the head) and end-to-end metrics (predicted type selects the head), so deployment performance is not overstated.

## Checkpoints, Diagnostics, and Safety

Save model width, class mappings, preprocessing mode, split seed, target IC fraction, and distance-bin rules with the state dictionary. Log per-split file/piece counts, IC fractions, distance coverage, sparse/missing strata, and skipped invalid files. Existing checkpoints remain loadable through a documented legacy fallback.

## Verification

Add synthetic `pytest` coverage for both distance naming styles, invalid/broad intervals, deterministic stratification, 80% IC sampling, lazy reads, skipped-file alignment, and predicted-vs-oracle distance routing. Run the full tests, `python -m compileall -q .`, CLI help smoke tests, and a manifest-only audit against the real dataset. No test or implementation step may rewrite files under `E:\Guoxing Yang\train_data`.
