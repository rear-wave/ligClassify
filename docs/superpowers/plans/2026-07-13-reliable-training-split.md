# Reliable Training Split Implementation Plan

> **For agentic workers:** Execute each task with red-green-refactor discipline. Do not overwrite unrelated working-tree changes and do not create commits unless the user explicitly requests them.

**Goal:** Build a training pipeline whose validation covers distance bins, whose locked test remains chronological, whose samplers match the intended priors, and whose candidate cannot replace `model.pt` after failing temporal reliability thresholds.

**Architecture:** `data/training_manifest.py` owns deterministic split construction and coverage validation. `data/distance_sampling.py` owns epoch-aware type-prior and balanced-distance samplers. `train.py` integrates these contracts, applies all configured losses, selects an eligible candidate, evaluates the locked test once, and promotes only passing candidates.

**Tech Stack:** Python, NumPy, PyTorch, pytest

## Global Constraints

- Never modify `.lig` source files below `C:\Users\Administrator\Desktop\train_data`.
- A whole file belongs to exactly one split; a temporal-test date cannot appear in train or validation.
- Validation is for interpolation/model selection; only the locked test supports future-date claims.
- `--no_init` must never load old-model parameters.
- A failed release gate must leave an existing `weights/new/model.pt` byte-identical.

---

### Task 1: Coverage-aware split contract

**Files:**
- Modify: `data/training_manifest.py:127`
- Modify: `tests/test_training_manifest.py`

**Interfaces:**
- Produce `coverage_temporal_split_manifest(entries, val_fraction=0.15, test_fraction=0.15, seed=42) -> dict[str, list[ManifestEntry]]`.
- Produce `validate_split_coverage(splits, type_names, min_bins=12, min_pieces=500) -> None`.

- [ ] Add a failing test where dates contain very different piece counts. Assert that the temporal test contains the latest rounded 15% of whole dates, regardless of piece concentration, and that its dates do not occur in train/validation.
- [ ] Run `python -m pytest tests/test_training_manifest.py -q` and verify failure because `coverage_temporal_split_manifest` is absent.
- [ ] Implement date-count temporal-test selection: group by type/date, hold out `max(1, round(n_dates * test_fraction))` latest dates, capped to retain at least one earlier date.
- [ ] Add a failing test with two files per distance bin. Assert deterministic file-level validation selection, all available bins represented, no file overlap, and identical results for the same seed.
- [ ] Implement validation selection from pre-test files. Sort each `(type_idx, dist_bin)` group by SHA-256 of `seed|absolute_filepath`; select `max(1, round(n * val_fraction))` files when `n >= 2`, capped at `n - 1`. Select IC files with the same stable rule as one group.
- [ ] Add failing tests proving `validate_split_coverage` rejects a non-IC validation type with fewer than 12 bins or 500 pieces and names the failing type in the error.
- [ ] Implement the audit and run `python -m pytest tests/test_training_manifest.py -q` until green.

### Task 2: Exact type-prior epoch sampling

**Files:**
- Modify: `data/distance_sampling.py`
- Modify: `tests/test_distance_sampling.py`

**Interfaces:**
- Produce `TypePriorSampler(type_labels, num_samples=180000, target_ic_fraction=0.8, seed=42)` with `set_epoch(epoch)`.

- [ ] Add a failing test using enough IC and four non-IC types. Assert exact sample count, exact 80% IC, no duplicate positions when pools are sufficient, deterministic epoch 0, and changed positions at epoch 1.
- [ ] Run the targeted test and verify failure because `TypePriorSampler` is absent.
- [ ] Implement seeded without-replacement IC selection. Allocate remaining samples proportionally across non-IC types with largest-remainder rounding so totals are exact. Raise a clear error when the requested prior cannot be drawn without replacement.
- [ ] Run `python -m pytest tests/test_distance_sampling.py -q` until green.

### Task 3: Truly balanced distance sampling

**Files:**
- Modify: `data/distance_sampling.py:7`
- Modify: `tests/test_distance_sampling.py`

**Interfaces:**
- Extend `HierarchicalDistanceSampler(..., replacement=False)`; training uses `replacement=True`.

- [ ] Add a failing test with strongly imbalanced pools and `replacement=True`. Assert exact `num_samples`, type counts differing by at most one cycle, per-type bin counts differing by at most one cycle, repeatability by seed, and a different order after `set_epoch(1)`.
- [ ] Verify the test fails because current sampling exhausts rare leaves and returns unequal counts.
- [ ] Build the capped candidate pool once. In replacement mode cycle uniformly through active types and bins, then cycle dates/files and reshuffle a leaf's positions when exhausted. Preserve current no-replacement behavior for compatibility.
- [ ] Run all sampler tests and ensure both replacement modes pass.

### Task 4: Loss weighting and eligible model selection

**Files:**
- Modify: `train.py:237`
- Modify: `distance_ordinal.py:150`
- Modify: `tests/test_train.py`
- Modify: `tests/test_distance_ordinal.py`

**Interfaces:**
- Add `distance_loss_weight=1.0` to `train_stream_step`.
- Extend `make_selection_key(metrics, guardrail=0.70, min_type_f1=0.85)` so eligibility is the first tuple element.

- [ ] Add a failing gradient test comparing otherwise identical distance steps at weights `1.0` and `0.25`; assert the routed distance-head update scales accordingly when type-batch weight is zero.
- [ ] Implement `total_loss = distance_loss_weight * distance_loss + distance_batch_type_weight * type_loss` and pass `args.lambda_dist` from the training loop.
- [ ] Change `--lambda_dist` default to `1.0` and record the applied value in checkpoint metadata.
- [ ] Add failing selection tests proving an epoch at type F1 0.84 cannot outrank an otherwise acceptable epoch at 0.85, while distance ordering still applies between eligible epochs.
- [ ] Prepend the eligibility flag to the selection key and update meaningful-improvement tolerances for the five-element tuple.
- [ ] Run `python -m pytest tests/test_train.py tests/test_distance_ordinal.py -q` until green.

### Task 5: Release gate and safe artifacts

**Files:**
- Modify: `train.py:750-1190`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produce `evaluate_release_gate(metrics, min_type_f1=0.85, min_type_w2=0.70, min_macro_w2=0.75) -> tuple[bool, list[str]]`.
- Extend `output_paths` with `candidate` and `candidate_metrics`.

- [ ] Add failing tests for passing metrics and for separate type-F1, per-type-w2, and macro-w2 failures. Assert reasons include exact failed thresholds.
- [ ] Implement the pure release-gate helper.
- [ ] Add a filesystem test with a sentinel `model.pt`. Simulate candidate writing with a failed gate and assert the sentinel hash/content remains unchanged; assert a passing gate promotes the structured checkpoint and metrics.
- [ ] Extract `save_candidate_and_maybe_promote(...)` so the behavior is testable without running training.
- [ ] Always save `candidate.pt` and `candidate_metrics.json`. Promote to `model.pt`/`metrics.json` only after a non-skipped locked test passes every gate. Log `PROMOTED` or `REJECTED` and each rejection reason.
- [ ] Run `python -m pytest tests/test_train.py -q` until green.

### Task 6: Integrate efficient loaders and coverage audit

**Files:**
- Modify: `train.py:760-1010`
- Modify: `tests/test_train.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Add CLI defaults `--type_samples_per_epoch 180000`, `--distance_samples_per_epoch 60000`, `--min_val_bins 12`, `--min_val_pieces 500`, `--min_type_f1 0.85`, `--min_test_type_w2 0.70`, and `--min_test_macro_w2 0.75`.

- [ ] Add failing CLI-default tests for every new contract.
- [ ] Replace `temporal_split_manifest` with `coverage_temporal_split_manifest`, then call `validate_split_coverage` before constructing datasets.
- [ ] Use `TypePriorSampler` for the type loader and replacement-mode `HierarchicalDistanceSampler` for the distance loader. Call `set_epoch` on both samplers and log their optimizer-step counts.
- [ ] Ensure `--skip_test` always writes only candidate artifacts and never promotes.
- [ ] Update `AGENTS.md` commands and explain coverage validation versus locked temporal testing.
- [ ] Run `python -m pytest -q` and `python -m compileall -q .`.

### Task 7: Real-data audit and smoke verification

**Files:**
- No source-file changes unless an audit exposes a reproducible defect.

- [ ] Build the manifest from `C:\Users\Administrator\Desktop\train_data` and print split files, pieces, dates, bins, and date intersections.
- [ ] Assert all 1,861 files are assigned exactly once, locked-test dates have zero overlap, every distance validation type has at least 12 bins and 500 pieces, and repeated construction yields identical split hashes.
- [ ] Materialize one epoch from both samplers without reading waveforms. Confirm exactly 144,000/180,000 type samples are IC and distance counts are balanced by type/bin to within one cycle.
- [ ] Run `python train.py --help` and confirm all safety arguments appear.
- [ ] Run a bounded one-epoch smoke job with `--type_samples_per_epoch 512 --distance_samples_per_epoch 512 --epochs 1 --skip_test` on a synthetic five-type fixture; confirm only candidate artifacts are produced.
- [ ] Run the full pytest suite and compile check once more, then report exact evidence and the new training command.
