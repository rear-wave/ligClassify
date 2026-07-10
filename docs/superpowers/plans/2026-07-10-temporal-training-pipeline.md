# Temporal Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chronological, leakage-resistant, lazy multi-task training pipeline whose training view contains 80% IC pieces.

**Architecture:** Add timestamp parsing to the binary parser and place manifest, distance-label, temporal-split, and sampling logic in a focused `data/training_manifest.py` module. `train_mtl.py` consumes one manifest, creates lazy split datasets, reports oracle and end-to-end metrics, and saves the full training contract with the best state.

**Tech Stack:** Python 3, NumPy, PyTorch, scikit-learn, pytest.

## Global Constraints

- Never modify or delete files below `E:\Guoxing Yang\train_data`.
- Split indivisible `(type, acquisition_date)` groups chronologically within each type.
- Downsample IC only in training, after splitting, to a default target fraction of `0.80`.
- Preserve the user's existing `train_mtl.py` changes: patience `10` and immediate `mtl_best.pt` writes.
- Keep raw state-dict compatibility while adding structured checkpoint metadata.

---

### Task 1: Timestamp and Manifest Parsing

**Files:**
- Modify: `data/lig_parser.py`
- Create: `data/training_manifest.py`
- Create: `tests/test_training_manifest.py`

**Interfaces:**
- Produces: `read_lig_timestamp(filepath: str, piece_index: int = 0) -> datetime`
- Produces: `ManifestEntry`, `parse_distance_bin(path: str) -> int`, `build_manifest(data_dir, type_names) -> tuple[list[ManifestEntry], dict]`

- [ ] Write tests that synthesize a minimal binary timestamp, verify ordinary time and `24:00` rollover, verify filename fallback, and verify `100-200km`/`100_200km` while rejecting broad ranges.
- [ ] Run `python -m pytest tests/test_training_manifest.py -v`; expect import failures for the new APIs.
- [ ] Implement strict timestamp field validation, hour-24 normalization, exact 100-km distance matching across path components, validated manifest discovery, and diagnostic counters.
- [ ] Re-run the focused tests; expect all Task 1 tests to pass.

### Task 2: Chronological Split and Lazy IC Sampling

**Files:**
- Modify: `data/training_manifest.py`
- Modify: `train_mtl.py`
- Modify: `tests/test_training_manifest.py`
- Create: `tests/test_train_mtl.py`

**Interfaces:**
- Produces: `temporal_split_manifest(entries, val_fraction, test_fraction) -> dict[str, list[ManifestEntry]]`
- Produces: `select_ic_indices(type_labels, target_fraction, seed) -> ndarray`
- Produces: `MultiTaskDataset(entries, split, target_ic_fraction, seed, normalize_mode)`

- [ ] Add tests proving dates never cross splits, dates are chronological per type, three-date classes populate all splits, IC selection is deterministic and exactly 80%, and lazy construction does not materialize waveform tensors.
- [ ] Run the focused tests; expect missing-function or old-constructor failures.
- [ ] Implement chronological date grouping, seeded piece-index selection, accepted-file label alignment, on-demand `__getitem__`, and batched `__getitems__` preprocessing.
- [ ] Re-run both focused test files; expect all Task 2 tests to pass.

### Task 3: End-to-End Metrics and Checkpoint Contract

**Files:**
- Modify: `train_mtl.py`
- Modify: `infer_mtl.py`
- Modify: `tests/test_train_mtl.py`

**Interfaces:**
- Produces: `route_distance_predictions(type_predictions, type_labels, dist_labels, dist_logits) -> dict`
- Structured checkpoint keys: `model_state_dict`, `base_channels`, `preprocessing`, `split_config`, `type_names`, `dist_names`, `dist_bin_starts`.

- [ ] Add a synthetic routing test where oracle distance is correct but a wrong predicted type makes end-to-end distance and joint accuracy fail.
- [ ] Run `python -m pytest tests/test_train_mtl.py -v`; expect the routing API test to fail.
- [ ] Implement oracle/end-to-end metrics, use macro-F1 plus end-to-end distance accuracy for early stopping, create the manifest once, log split dates/counts/coverage, expose `--target_ic_fraction`, `--val_fraction`, `--test_fraction`, `--num_workers`, and load checkpoint model width/preprocessing in inference.
- [ ] Re-run focused tests and both CLI `--help` commands; expect success.

### Task 4: Real-Data Audit and Full Verification

**Files:**
- Modify: `train_mtl.py` only if audit reveals a reproducible defect.

- [ ] Run a manifest-only audit on `E:\Guoxing Yang\train_data` and verify 5,654 files, 2,430,838 pieces, corrected underscore distance labels, disjoint dates per type, and an 80% IC training view.
- [ ] Run `python -m pytest -v`, `python -m compileall -q .`, and `git diff --check`.
- [ ] Review `git diff` to confirm source data was untouched and the pre-existing patience/best-save edits remain.
