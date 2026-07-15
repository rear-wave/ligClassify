# Cross-Validated Final Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the contaminated single-split release workflow with deterministic three-fold, source-file-isolated OOF evaluation and one randomly initialized full-data deployment model.

**Architecture:** A new cross-validation layer assigns complete `.lig` files to exact-distance folds and validates one OOF row per trusted piece. One joint, hierarchical replacement sampler trains the existing conditional expert with deterministic waveform augmentation. Pooled OOF predictions drive file-equal metrics, IC calibration, release gates, and the fixed epoch count for a final single model.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest, CSV/JSON artifacts, Windows/PowerShell-compatible CLIs.

## Global Constraints

- Train only NCG, NNBE, PCG, and PNBE; never train on IC pieces.
- Treat IC only as inference-time rejection.
- Keep each source `.lig` file in exactly one fold and create exactly one OOF prediction per trusted piece.
- Balance folds and training at exact 100-km intervals, not only coarse bands.
- Preserve waveform polarity and prohibit polarity inversion or time reversal.
- Train every fold and the final model from random initialization; never warm-start from old or fold checkpoints.
- Deploy one full-data model; fold models are evaluation artifacts only.
- Preserve original piece bytes during inference.
- Keep existing `weights/old/`, `weights/conditional_final/`, and all user data unchanged.
- Historical old-model results are contaminated reference information, never an automatic release gate.
- Required OOF gates are per-type precision 0.95, coverage 0.80, file-equal macro recall 0.90, overall within-200 km 0.85, per-type within-200 km 0.75, and supported-condition within-200 km 0.70.

## File Map

- `data/cross_validation.py` owns deterministic exact-interval fold assignment and support counts.
- `data/oof_manifest.py` owns stable piece identities and OOF completeness checks.
- `data/distance_sampling.py` and `data/augmentation.py` own joint replacement sampling and waveform-safe augmentation.
- `data/training_dataset.py`, `data/signal_context.py`, `training_engine.py`, and `models.py` own the single-stream model input/training contract.
- `evaluation.py` owns all file-equal metrics, checkpoint ranking, bootstrap summaries, and release gates.
- `open_set.py` owns transferable OOF rejection calibration and final-model feature-reference rebinding.
- `cv_pipeline.py` owns fold execution, recovery, artifact validation, final full-data training, and atomic promotion; `train.py` is only its CLI adapter.
- `classify.py` owns v3 decoding, byte-preserving output, and inference audit fields; `benchmark.py` remains a reporting adapter.
- `tests/` uses only bounded synthetic fixtures and injected trainers; no real `.lig` data or checkpoints are added.

---

### Task 1: Exact-Interval Three-Fold Assignment and Support Map

**Files:**
- Create: `data/cross_validation.py`
- Modify: `data/split_artifacts.py`
- Modify: `audit_data.py`
- Create: `tests/test_cross_validation.py`
- Modify: `tests/test_audit_data.py`

**Interfaces:**
- Consumes: `ManifestEntry` rows from `data.training_manifest.build_manifest()`.
- Produces: `exact_condition_key()`, `assign_exact_folds()`, `fold_train_holdout()`, `validate_fold_assignment()`, `build_support_map()`, and `make_fold_manifest()` for all later tasks.

- [ ] **Step 1: Write failing exact-fold tests**

Create `tests/test_cross_validation.py` with these imports and module-local
fixtures so the test does not depend on another test module:

```python
from datetime import datetime, timedelta

from data.cross_validation import (
    assign_exact_folds,
    build_support_map,
    validate_fold_assignment,
)
from data.training_manifest import ManifestEntry


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def make_entries(lows, files_per_condition, pieces=(50, 100, 150)):
    entries = []
    for low in lows:
        for file_index in range(files_per_condition):
            entries.append(ManifestEntry(
                filepath=f"NCG/day/{low}-{low + 100}km/file-{file_index}.lig",
                type_idx=0,
                dist_bin=low // 100,
                timestamp=datetime(2020, 1, 1) + timedelta(seconds=len(entries)),
                n_pieces=pieces[file_index % len(pieces)],
                distance_low_km=low,
                distance_high_km=low + 100,
                is_daytime=True,
            ))
    return entries


def fold_paths(folds):
    return {
        fold: sorted(entry.filepath for entry in entries)
        for fold, entries in sorted(folds.items())
    }


def test_exact_bins_are_distinct_and_three_file_cells_cover_every_fold():
    entries = make_entries(
        lows=(300, 400), files_per_condition=3, pieces=(50, 100, 150)
    )
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    for low in (300, 400):
        assert [
            sum(entry.distance_low_km == low for entry in folds[fold])
            for fold in range(3)
        ] == [1, 1, 1]


def test_sparse_condition_is_reported_without_splitting_a_file():
    entries = make_entries(lows=(300,), files_per_condition=2)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    support = build_support_map(entries, TYPE_NAMES, minimum_files=3)
    assert sum(len(rows) for rows in folds.values()) == 2
    assert support["NCG/day/300-400km"]["file_count"] == 2
    assert support["NCG/day/300-400km"]["status"] == "insufficient_support"


def test_fold_assignment_is_order_independent_and_complete():
    entries = make_entries(lows=(0, 100, 200), files_per_condition=5)
    first = assign_exact_folds(entries, n_folds=3, seed=11)
    second = assign_exact_folds(reversed(entries), n_folds=3, seed=11)
    assert fold_paths(first) == fold_paths(second)
    validate_fold_assignment(first, entries, n_folds=3)
```

- [ ] **Step 2: Run the new tests and verify the missing-module failure**

Run:

```powershell
python -m pytest -q tests/test_cross_validation.py tests/test_audit_data.py
```

Expected: failure because `data.cross_validation` and fold artifacts do not exist.

- [ ] **Step 3: Implement exact fold assignment**

Create `data/cross_validation.py` with these contracts and validation rules:

```python
import hashlib
import os
from collections import defaultdict
from pathlib import Path


FOLD_COUNT = 3


def exact_condition_key(entry):
    low = entry.distance_low_km
    high = entry.distance_high_km
    if low is None or high is None or high - low != 100:
        raise ValueError("cross-validation requires 100-km interval labels")
    if low % 100 or not 0 <= low < high <= 3000:
        raise ValueError("distance intervals must be aligned inside 0-3000 km")
    return int(entry.type_idx), bool(entry.is_daytime), int(low), int(high)


def fold_train_holdout(folds, held_out):
    holdout = list(folds[int(held_out)])
    train = [
        entry
        for fold_index, entries in sorted(folds.items())
        if fold_index != int(held_out)
        for entry in entries
    ]
    return train, holdout


def _entry_signature(entry):
    return (
        int(entry.type_idx), bool(entry.is_daytime),
        int(entry.distance_low_km), int(entry.distance_high_km),
        int(entry.n_pieces), entry.timestamp.isoformat(),
    )


def _relative_identities(entries):
    absolute = [os.path.abspath(entry.filepath) for entry in entries]
    root = os.path.commonpath(absolute)
    if len(absolute) == 1 or os.path.isfile(root):
        root = os.path.dirname(root)
    return {
        os.path.normcase(path): Path(os.path.relpath(path, root)).as_posix()
        for path in absolute
    }


def assign_exact_folds(entries, n_folds=FOLD_COUNT, seed=42):
    entries = list(entries)
    if int(n_folds) != FOLD_COUNT:
        raise ValueError("this release contract requires exactly three folds")
    identities = _relative_identities(entries)
    groups = defaultdict(list)
    seen = set()
    for entry in entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in seen:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        seen.add(path)
        groups[exact_condition_key(entry)].append(entry)
    folds = {index: [] for index in range(FOLD_COUNT)}
    global_pieces = [0] * FOLD_COUNT
    for condition in sorted(groups):
        condition_files = [0] * FOLD_COUNT
        condition_pieces = [0] * FOLD_COUNT
        ordered = sorted(
            groups[condition],
            key=lambda entry: (
                -int(entry.n_pieces),
                hashlib.sha256(
                    f"{int(seed)}|{identities[os.path.normcase(os.path.abspath(entry.filepath))]}"
                    .encode("utf-8")
                ).hexdigest(),
            ),
        )
        for entry in ordered:
            fold = min(
                range(FOLD_COUNT),
                key=lambda index: (
                    condition_files[index], condition_pieces[index],
                    global_pieces[index], index,
                ),
            )
            folds[fold].append(entry)
            condition_files[fold] += 1
            condition_pieces[fold] += int(entry.n_pieces)
            global_pieces[fold] += int(entry.n_pieces)
    return folds


def validate_fold_assignment(folds, expected_entries, n_folds=FOLD_COUNT):
    if set(folds) != set(range(int(n_folds))):
        raise ValueError("fold keys must be contiguous from zero")
    expected = {}
    for entry in expected_entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in expected:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        expected[path] = _entry_signature(entry)
    observed = set()
    for entries in folds.values():
        for entry in entries:
            path = os.path.normcase(os.path.abspath(entry.filepath))
            if path in observed:
                raise ValueError(f"duplicate fold owner: {entry.filepath}")
            if path not in expected:
                raise ValueError(f"unknown fold file: {entry.filepath}")
            if _entry_signature(entry) != expected[path]:
                raise ValueError(f"fold label mutation: {entry.filepath}")
            observed.add(path)
    missing = sorted(set(expected) - observed)
    if missing:
        raise ValueError(f"missing fold files: {len(missing)}")
```

- [ ] **Step 4: Implement stable fold and support artifacts**

Extend `data/split_artifacts.py`:

```python
def make_fold_manifest(folds, root, seed):
    holdout_hashes = {
        str(index): split_hash(entries, root)
        for index, entries in sorted(folds.items())
    }
    train_hashes = {
        str(index): split_hash(fold_train_holdout(folds, index)[0], root)
        for index in sorted(folds)
    }
    combined = stable_json_hash({
        "schema": "file_isolated_exact_interval_cv_v1",
        "seed": int(seed),
        "holdout_hashes": holdout_hashes,
        "train_hashes": train_hashes,
    })
    return {
        "schema": "file_isolated_exact_interval_cv_v1",
        "fold_count": len(folds),
        "seed": int(seed),
        "holdout_hashes": holdout_hashes,
        "train_hashes": train_hashes,
        "combined_hash": combined,
        "folds": {
            str(index): [serialize_manifest_entry(entry, root) for entry in rows]
            for index, rows in sorted(folds.items())
        },
    }


def stable_json_hash(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Place `stable_json_hash()` above `make_fold_manifest()`, add `hashlib` and
`json` imports, and import `fold_train_holdout` from `data.cross_validation`
in `data/split_artifacts.py`. Add the support aggregation to
`data/cross_validation.py`:

```python
def build_support_map(entries, type_names, minimum_files=3):
    support = defaultdict(lambda: {"file_count": 0, "piece_count": 0})
    seen = set()
    for entry in entries:
        path = os.path.normcase(os.path.abspath(entry.filepath))
        if path in seen:
            raise ValueError(f"duplicate source file: {entry.filepath}")
        seen.add(path)
        type_index, daylight, low_km, high_km = exact_condition_key(entry)
        type_name = type_names[type_index]
        name = (
            f"{type_name}/{'day' if daylight else 'night'}/"
            f"{low_km}-{high_km}km"
        )
        row = support[name]
        row.update({
            "type_index": type_index,
            "daylight": daylight,
            "low_km": low_km,
            "high_km": high_km,
        })
        row["file_count"] += 1
        row["piece_count"] += int(entry.n_pieces)
    result = {}
    for name, row in sorted(support.items()):
        row["status"] = (
            "supported"
            if row["file_count"] >= int(minimum_files)
            else "insufficient_support"
        )
        result[name] = dict(row)
    return result
```

Its call to `exact_condition_key()` rejects missing or misaligned intervals,
keeping fold and support artifacts on the same validated population.

- [ ] **Step 5: Switch `audit_data.py` to fold artifacts**

Make `audit_dataset()` return and optionally write `data_audit.json`,
`fold_manifest.json`, and `support_map.json`. Its report must include the
number and names of insufficient-support cells and must continue proving that
IC contributes zero trained pieces. Build folds once from `entries` and once
from `reversed(entries)`; abort before dataset/GPU construction unless both
combined hashes and per-fold ownership lists are identical. Also call
`validate_fold_assignment()` and compare summed file/piece totals with the
manifest diagnostics before writing any artifact.

- [ ] **Step 6: Run focused tests**

```powershell
python -m pytest -q tests/test_cross_validation.py tests/test_audit_data.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add data/cross_validation.py data/split_artifacts.py audit_data.py tests/test_cross_validation.py tests/test_audit_data.py
git commit -m "Add exact interval cross-validation folds"
```

---

### Task 2: Stable OOF Piece Identity and Completeness Validation

**Files:**
- Create: `data/oof_manifest.py`
- Modify: `data/training_dataset.py`
- Modify: `conditional_pipeline.py`
- Create: `tests/test_oof_manifest.py`
- Modify: `tests/test_training_dataset.py`

**Interfaces:**
- Consumes: fold manifest from Task 1 and piece rows from `build_piece_manifest()`.
- Produces: stable `piece_key`, source identity in every batch, and OOF validation used by `cv_pipeline.py`.

- [ ] **Step 1: Write failing OOF identity tests**

```python
import pytest

from data.oof_manifest import oof_row_id, validate_oof_rows


def test_oof_rows_require_every_piece_exactly_once():
    expected = {
        "NCG/day/0-100km/a.lig#0": {
            "fold": 0, "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig", "piece_index": 0,
        },
        "NCG/day/0-100km/a.lig#1": {
            "fold": 0, "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig", "piece_index": 1,
        },
    }
    rows = [
        {"piece_key": key, "fold": value["fold"], "true_type": value["type_idx"]}
        for key, value in expected.items()
    ]
    validate_oof_rows(rows, expected)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_rows(rows + [rows[0]], expected)
    with pytest.raises(ValueError, match="missing"):
        validate_oof_rows(rows[:1], expected)


def test_piece_key_does_not_use_fold_local_file_id():
    assert oof_row_id("NCG/a.lig", 7) == "NCG/a.lig#7"
    assert oof_row_id("NNBE/a.lig", 7) != oof_row_id("NCG/a.lig", 7)
```

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest -q tests/test_oof_manifest.py tests/test_training_dataset.py
```

Expected: import or missing-field failures.

- [ ] **Step 3: Implement OOF identities and validation**

Create `data/oof_manifest.py`:

```python
from pathlib import Path


def oof_row_id(relative_path, piece_index):
    return f"{Path(relative_path).as_posix()}#{int(piece_index)}"


def validate_oof_rows(rows, expected):
    seen = {}
    for row in rows:
        key = str(row["piece_key"])
        if key in seen:
            raise ValueError(f"duplicate OOF piece: {key}")
        if key not in expected:
            raise ValueError(f"unknown OOF piece: {key}")
        contract = expected[key]
        if int(row["fold"]) != int(contract["fold"]):
            raise ValueError(f"wrong OOF fold for {key}")
        if int(row["true_type"]) != int(contract["type_idx"]):
            raise ValueError(f"OOF label mismatch for {key}")
        if str(row.get("source_path", contract["source_path"])) != contract["source_path"]:
            raise ValueError(f"OOF source mismatch for {key}")
        if int(row.get("piece_index", contract["piece_index"])) != contract["piece_index"]:
            raise ValueError(f"OOF piece index mismatch for {key}")
        seen[key] = row
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ValueError(f"missing OOF pieces: {len(missing)}")


def expected_oof_rows(fold_manifest):
    if fold_manifest.get("schema") != "file_isolated_exact_interval_cv_v1":
        raise ValueError("unexpected fold manifest schema")
    expected = {}
    for fold_text, files in sorted(fold_manifest["folds"].items()):
        fold = int(fold_text)
        for file_row in files:
            source_path = Path(file_row["path"]).as_posix()
            for piece_index in range(int(file_row["n_pieces"])):
                key = oof_row_id(source_path, piece_index)
                if key in expected:
                    raise ValueError(f"duplicate expected OOF piece: {key}")
                expected[key] = {
                    "fold": fold,
                    "type_idx": int(file_row["type_idx"]),
                    "source_path": source_path,
                    "piece_index": piece_index,
                }
    return expected
```

- [ ] **Step 4: Carry stable source fields through the dataset**

Import `Path` and `oof_row_id`. Add `data_root=None` to
`LightningPieceDataset.__init__`. Production callers
must pass the manifest root; when omitted in existing synthetic tests, derive
the common parent of indexed paths. While expanding entries, append:

```python
root = os.path.abspath(data_root or os.path.commonpath(paths))
if len(paths) == 1 and os.path.normcase(root) == os.path.normcase(os.path.abspath(paths[0])):
    root = os.path.dirname(root)
source_path = Path(os.path.relpath(entry.filepath, root)).as_posix()
source_paths.append(source_path)
piece_indices.append(int(entry.piece_index))
piece_keys.append(oof_row_id(source_path, entry.piece_index))
```

Store `self.source_paths`, `self.piece_indices`, and `self.piece_keys` as arrays
aligned with `self.global_indices`. Extend each dataset item with:

```python
{
    "source_path": self.source_paths[position],
    "piece_index": int(self.piece_indices[position]),
    "piece_key": self.piece_keys[position],
}
```

In `collate_training_batch()`, retain these as lists rather than tensors:

```python
result["source_path"] = [item["source_path"] for item in batch]
result["piece_index"] = [int(item["piece_index"]) for item in batch]
result["piece_key"] = [item["piece_key"] for item in batch]
```

Keep numeric `file_id` only for within-dataset batching. Update
`collect_prediction_bundle(model, loader, device, split_hash, fold_index)` so
every record uses stable `source_path` as its file identity and writes
`piece_key` and `fold`.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest -q tests/test_oof_manifest.py tests/test_training_dataset.py
git add data/oof_manifest.py data/training_dataset.py conditional_pipeline.py tests/test_oof_manifest.py tests/test_training_dataset.py
git commit -m "Add stable OOF piece identities"
```

---

### Task 3: Hierarchical Replacement Sampler

**Files:**
- Modify: `data/distance_sampling.py`
- Modify: `tests/test_distance_sampling.py`

**Interfaces:**
- Consumes: dataset arrays and the exact intervals validated by Task 1.
- Produces: `SampleRequest` objects for deterministic training and augmentation.

- [ ] **Step 1: Replace sampler tests with joint hierarchical behavior**

```python
from collections import Counter

import pytest

from data.distance_sampling import JointConditionSampler


def test_joint_sampler_balances_exact_cells_with_replacement_and_caps_files():
    sampler = JointConditionSampler(
        type_labels=[0, 0, 0, 1, 1, 1],
        daylight=[0, 0, 1, 0, 1, 1],
        distance_low_km=[300, 300, 400, 300, 400, 400],
        distance_high_km=[400, 400, 500, 400, 500, 500],
        file_ids=[0, 0, 1, 2, 3, 3],
        num_samples=12,
        max_samples_per_file=4,
        seed=5,
    )
    requests = list(sampler)
    assert len(requests) == 12
    assert len({request.augmentation_seed for request in requests}) == 12
    assert max(Counter(sampler.file_ids[r.position] for r in requests).values()) <= 4
    cells = Counter(
        (
            sampler.type_labels[request.position],
            sampler.daylight[request.position],
            sampler.distance_low_km[request.position],
        )
        for request in requests
    )
    assert max(cells.values()) - min(cells.values()) <= 1


def test_joint_sampler_rejects_impossible_file_cap():
    with pytest.raises(ValueError, match="file cap"):
        JointConditionSampler(
            type_labels=[0, 0],
            daylight=[1, 1],
            distance_low_km=[0, 0],
            distance_high_km=[100, 100],
            file_ids=[0, 1],
            num_samples=9,
            max_samples_per_file=4,
            seed=1,
        )
```

Retain the regression test proving `__len__` is constant-time and not invoked
once per selected sample.

- [ ] **Step 2: Run and confirm failures against `ConditionBalancedSampler`**

```powershell
python -m pytest -q tests/test_distance_sampling.py
```

Expected: missing `JointConditionSampler` and replacement behavior failures.

- [ ] **Step 3: Implement the joint sampler and deterministic requests**

Add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRequest:
    position: int
    augmentation_seed: int


def exact_interval_key(low_km, high_km):
    low_km, high_km = int(low_km), int(high_km)
    if high_km - low_km != 100 or low_km % 100 or not 0 <= low_km < high_km <= 3000:
        raise ValueError("joint sampling requires aligned 100-km intervals")
    return low_km, high_km
```

`JointConditionSampler` must prebuild nested pools in `__init__` using
`type -> daylight -> exact interval -> file -> positions`. Its epoch iterator
cycles shuffled decks at each hierarchy, chooses only files below the hard cap,
samples piece positions with replacement, and emits a unique augmentation seed
derived from `(seed, epoch, draw_index, position)`. Raise before iteration when
`num_samples > unique_file_count * max_samples_per_file`; never silently shorten
an epoch.

- [ ] **Step 4: Run tests and commit**

```powershell
python -m pytest -q tests/test_distance_sampling.py
git add data/distance_sampling.py tests/test_distance_sampling.py
git commit -m "Add hierarchical joint sampler"
```

---

### Task 4: Deterministic Polarity-Safe Waveform Augmentation

**Files:**
- Create: `data/augmentation.py`
- Modify: `data/training_dataset.py`
- Create: `tests/test_augmentation.py`
- Modify: `tests/test_training_dataset.py`

**Interfaces:**
- Consumes: raw waveform arrays and `SampleRequest` seeds from Task 3.
- Produces: one augmented raw waveform used to create both local and global views.

- [ ] **Step 1: Write failing deterministic and physical-safety tests**

```python
from datetime import datetime

import numpy as np

from data.augmentation import WaveformAugmentationConfig, augment_waveforms
from data.distance_sampling import SampleRequest
from data.training_dataset import LightningPieceDataset
from data.training_manifest import PieceManifestEntry


class OnePieceIndex:
    def __init__(self, waveform):
        self.filepaths = ["synthetic.lig"]
        self.num_pieces_per_file = np.asarray([1], dtype=np.int64)
        self._cumsum = np.asarray([0, 1], dtype=np.int64)
        self.waveform = np.asarray(waveform, dtype=np.float32)

    def read_pieces_batch(self, indices):
        assert list(indices) == [0]
        return [self.waveform.copy()]


def dataset_item(split, request_seed):
    waveform = np.zeros(16000, dtype=np.float32)
    waveform[100:120] = np.linspace(1.0, 10.0, 20)
    entry = PieceManifestEntry(
        filepath="synthetic.lig", piece_index=0, type_idx=0, dist_bin=0,
        timestamp=datetime(2020, 1, 1, 4), distance_low_km=0,
        distance_high_km=100, is_daytime=True,
    )
    dataset = LightningPieceDataset(
        [entry], split=split, lig_index=OnePieceIndex(waveform), use_filter=False,
        augmentation=WaveformAugmentationConfig(noise_fraction=0.0),
    )
    return dataset[SampleRequest(position=0, augmentation_seed=request_seed)]


def test_augmentation_is_seeded_and_never_wraps_or_flips_polarity():
    pieces = np.zeros((1, 16000), dtype=np.float32)
    pieces[0, 100] = 10.0
    config = WaveformAugmentationConfig(
        max_shift_samples=32,
        gain_min=0.90,
        gain_max=1.10,
        baseline_drift_fraction=0.0,
        noise_fraction=0.0,
    )
    first = augment_waveforms(pieces, [17], config)
    second = augment_waveforms(pieces, [17], config)
    assert np.array_equal(first, second)
    assert first.max() > 0
    assert first[0, -64:].max() == 0


def test_validation_dataset_ignores_augmentation_request():
    train = dataset_item(split="train", request_seed=9)
    first_val = dataset_item(split="val", request_seed=9)
    second_val = dataset_item(split="val", request_seed=123)
    assert not np.array_equal(train["local"].numpy(), first_val["local"].numpy())
    assert np.array_equal(first_val["local"].numpy(), second_val["local"].numpy())
```

- [ ] **Step 2: Run and verify failures**

```powershell
python -m pytest -q tests/test_augmentation.py tests/test_training_dataset.py
```

- [ ] **Step 3: Implement raw-waveform augmentation**

Create an immutable config and a batch function:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaveformAugmentationConfig:
    max_shift_samples: int = 64
    gain_min: float = 0.90
    gain_max: float = 1.10
    baseline_drift_fraction: float = 0.01
    noise_fraction: float = 0.01


def _zero_shift(values, shift):
    result = np.zeros_like(values)
    if shift > 0:
        result[shift:] = values[:-shift]
    elif shift < 0:
        result[:shift] = values[-shift:]
    else:
        result[:] = values
    return result


def augment_waveforms(values, seeds, config):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(seeds):
        raise ValueError("waveforms and augmentation seeds must be aligned")
    if not 0 < config.gain_min <= config.gain_max:
        raise ValueError("augmentation gain must remain strictly positive")
    result = np.empty_like(values)
    axis = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    for row_index, (row, seed) in enumerate(zip(values, seeds)):
        rng = np.random.default_rng(int(seed))
        shift = int(rng.integers(
            -config.max_shift_samples, config.max_shift_samples + 1
        ))
        shifted = _zero_shift(row, shift)
        centered = shifted - np.median(shifted)
        amplitude = max(float(np.quantile(np.abs(centered), 0.95)), 1e-6)
        gain = float(rng.uniform(config.gain_min, config.gain_max))
        drift = (
            float(rng.uniform(-1.0, 1.0))
            * config.baseline_drift_fraction * amplitude * axis
        )
        noise = rng.normal(
            0.0, config.noise_fraction * amplitude, size=len(row)
        ).astype(np.float32)
        result[row_index] = shifted * gain + drift + noise
    return result
```

For each row, initialize `np.random.default_rng(seed)`, apply `_zero_shift`, a
strictly positive gain, a linear zero-mean drift scaled by robust absolute
amplitude, and Gaussian noise scaled by the same amplitude. Do not call
`np.roll`, negate, or reverse arrays.

- [ ] **Step 4: Integrate before preprocessing**

Teach `LightningPieceDataset._items_from_positions()` to accept integers or
`SampleRequest`. Read raw pieces once, compute quality from unmodified raw data,
augment only when `split == "train"`, then pass the same augmented batch once to
`preprocess_multiscale_batch()` to create synchronized local/global views.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest -q tests/test_augmentation.py tests/test_training_dataset.py tests/test_preprocessing.py
git add data/augmentation.py data/training_dataset.py tests/test_augmentation.py tests/test_training_dataset.py
git commit -m "Add deterministic waveform augmentation"
```

---

### Task 5: Joint Two-Stage Training and Daylight-Only Context

**Files:**
- Modify: `training_engine.py`
- Modify: `models.py`
- Modify: `data/signal_context.py`
- Modify: `data/training_dataset.py`
- Modify: `conditional_pipeline.py`
- Modify: `train.py`
- Modify: `classify.py`
- Modify: `benchmark.py`
- Create: `tests/test_training_engine.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_signal_context.py`
- Modify: `tests/test_training_dataset.py`
- Modify: `tests/test_classify.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: one joint loader from Tasks 3–4.
- Produces: staged joint optimization reusable by every fold and final training.

- [ ] **Step 1: Write failing staged-loss and context tests**

```python
import torch

from models import ConditionalExpertNet
from training_engine import conditional_joint_train_step, distance_weight_for_epoch


def test_distance_weight_stage_boundary():
    assert distance_weight_for_epoch(2, 3, 0.25, 1.0) == ("type_focus", 0.25)
    assert distance_weight_for_epoch(3, 3, 0.25, 1.0) == ("joint", 1.0)


def test_joint_step_updates_type_and_oracle_distance_heads():
    torch.manual_seed(3)
    model = ConditionalExpertNet(
        base=4, context_dim=1, dist_mlp_dim=8, dist_dropout=0.0
    )
    batch = {
        "local": torch.randn(4, 1, 512),
        "global_view": torch.randn(4, 1, 512),
        "context": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "type_label": torch.arange(4),
        "distance_low_km": torch.tensor([0.0, 300.0, 600.0, 900.0]),
        "distance_high_km": torch.tensor([100.0, 400.0, 700.0, 1000.0]),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    result = conditional_joint_train_step(
        model, batch, optimizer, torch.nn.CrossEntropyLoss(),
        distance_loss_weight=0.25,
    )
    assert result["type_count"] == len(batch["type_label"])
    assert result["distance_count"] == len(batch["type_label"])
    assert result["type_loss"] > 0 and result["distance_loss"] > 0


def test_conditional_model_accepts_daylight_only_context():
    model = ConditionalExpertNet(base=8, context_dim=1)
    output = model(torch.randn(2, 1, 8000), torch.randn(2, 1, 8000), torch.ones(2, 1))
    assert output[0].shape == (2, 4)
```

- [ ] **Step 2: Run and verify failures**

```powershell
python -m pytest -q tests/test_training_engine.py tests/test_models.py tests/test_signal_context.py tests/test_training_dataset.py tests/test_classify.py tests/test_train.py
```

- [ ] **Step 3: Implement one joint training step**

Replace stream branching with:

```python
def distance_weight_for_epoch(epoch, type_focus_epochs, type_focus_weight, joint_weight):
    if epoch < type_focus_epochs:
        return "type_focus", float(type_focus_weight)
    return "joint", float(joint_weight)


def conditional_joint_train_step(
    model, batch, optimizer, type_criterion, distance_loss_weight,
    coarse_weight=0.5, ordered_weight=0.2, scaler=None, amp=False,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=batch["local"].device.type,
        enabled=bool(amp and batch["local"].is_cuda),
    ):
        type_logits, distance_logits, coarse_logits = model(
            batch["local"], batch["global_view"], batch["context"]
        )
        type_loss = type_criterion(type_logits, batch["type_label"])
        distance_loss, components = compute_conditional_distance_loss(
            distance_logits, coarse_logits, batch["type_label"],
            batch["distance_low_km"], batch["distance_high_km"],
            coarse_weight=coarse_weight, ordered_weight=ordered_weight,
        )
        total_loss = type_loss + float(distance_loss_weight) * distance_loss
    amp_enabled = bool(amp and batch["local"].is_cuda)
    if scaler is not None and amp_enabled:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
    valid_distance = (
        (batch["distance_low_km"] >= 0)
        & (batch["distance_high_km"] > batch["distance_low_km"])
    )
    return {
        "type_loss": float(type_loss.detach()),
        "distance_loss": float(distance_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "type_correct": int((type_logits.argmax(1) == batch["type_label"]).sum()),
        "type_count": int(len(batch["type_label"])),
        "distance_count": int(valid_distance.sum()),
        "components": {
            name: float(value.detach()) for name, value in components.items()
        },
    }
```

- [ ] **Step 4: Make context dimension explicit**

Add `context_dim` to `create_mtl_model()`, store it in checkpoints, and pass it
through `classify.load_mtl_checkpoint()`. Add
`time_context_batch(..., mode="daylight")`; default output is shape `(N, 1)`.
Mode `cyclic` retains the current three columns for controlled ablations.

```python
def time_context_batch(timestamps, daylight, mode="daylight"):
    if len(timestamps) != len(daylight):
        raise ValueError("timestamps and daylight must have the same length")
    daylight_column = np.asarray(daylight, dtype=np.float32).reshape(-1, 1)
    if mode == "daylight":
        return daylight_column
    if mode != "cyclic":
        raise ValueError("time context mode must be daylight or cyclic")
    local_hours = np.asarray([
        (stamp.hour + 8 + stamp.minute / 60.0 + stamp.second / 3600.0) % 24
        for stamp in timestamps
    ])
    angles = 2.0 * np.pi * local_hours / 24.0
    return np.column_stack([
        daylight_column[:, 0], np.sin(angles), np.cos(angles)
    ]).astype(np.float32)
```

Keep the factory's `context_dim=3` default for loading historical checkpoints,
but pass `context_dim=1` in the new CV model config. Change the existing cyclic
context test to call `mode="cyclic"` and add:

```python
def test_default_time_context_is_daylight_only():
    context = time_context_batch(
        [datetime(2020, 1, 1), datetime(2020, 1, 2)], [True, False]
    )
    assert context.shape == (2, 1)
    assert context[:, 0].tolist() == [1.0, 0.0]
```

Add `time_context_mode="daylight"` to `LightningPieceDataset`. New CV callers
use that default. `benchmark.py` and `classify.py` must choose `"cyclic"` when
an older checkpoint has `context_dim == 3`, so this change does not invalidate
existing deployed/candidate checkpoints.

The new release path validates `config.time_context == "daylight"`. The cyclic
mode remains only for legacy-checkpoint compatibility; it cannot be promoted by
this plan because no three-fold ablation has established an improvement.

- [ ] **Step 5: Replace two loaders with one**

Delete `_alternate`, `type_loader`, `distance_loader`, and stream-specific
weights from `conditional_pipeline.py`. Each epoch calls `sampler.set_epoch()`,
selects its stage weight, and iterates one loader. Update CLI defaults:

```text
--samples_per_epoch 120000
--max_samples_per_file 512
--type_focus_epochs 3
--type_focus_distance_weight 0.25
--joint_distance_weight 1.0
--time_context daylight
```

Remove `--type_samples_per_epoch`, `--distance_samples_per_epoch`,
`--distance_batch_size`, and `--distance_batch_type_weight` from the new CV
entry point.

Fold early stopping begins only after all three type-focus epochs complete.
Reset the counter whenever `checkpoint_selection_key()` improves
lexicographically; stop after 10 consecutive non-improving joint epochs or at
50 total epochs. Always retain at least one joint-stage checkpoint, and store
its one-based epoch number in `FoldResult.best_epoch`.

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest -q tests/test_training_engine.py tests/test_models.py tests/test_signal_context.py tests/test_training_dataset.py tests/test_classify.py tests/test_train.py
git add training_engine.py models.py data/signal_context.py data/training_dataset.py conditional_pipeline.py train.py classify.py benchmark.py tests/test_training_engine.py tests/test_models.py tests/test_signal_context.py tests/test_training_dataset.py tests/test_classify.py tests/test_train.py
git commit -m "Train conditional experts with one staged stream"
```

---

### Task 6: File-Equal Metrics and Supported-Condition Release Gates

**Files:**
- Modify: `evaluation.py`
- Modify: `tests/test_evaluation.py`
- Modify: `benchmark.py`

**Interfaces:**
- Consumes: prediction records with stable source identities.
- Produces: `evaluate_predictions()`, `round_metrics()`, and one file-equal metric implementation shared by OOF calibration, checkpoint selection, and release.

- [ ] **Step 1: Write failing equal-file and supported-condition tests**

```python
def test_file_equal_precision_gives_each_file_total_weight_one():
    records = [make_record("large", 0, 0) for _ in range(100)]
    records += [make_record("small", 1, 0)]
    metrics = evaluate_predictions(records)
    assert metrics["type_file_equal_precision"][0] == pytest.approx(0.5)


def test_only_supported_subgroups_are_hard_gates():
    passing = good_release_metrics()
    passing["distance_conditions_100km"] = {
        "PNBE/night/300-400km": {
            "file_count": 2, "file_macro_within_200": 0.10,
        }
    }
    assert evaluate_release(passing)[0]
    passing["distance_conditions_100km"]["PNBE/night/300-400km"]["file_count"] = 3
    passed, reasons = evaluate_release(passing)
    assert not passed
    assert any("supported subgroup" in reason for reason in reasons)


def good_release_metrics():
    return {
        "type_file_equal_precision": [0.96, 0.97, 0.98, 0.99],
        "type_coverage": 0.82,
        "type_file_equal_recall_mean": 0.91,
        "distance_100km_interval_within_200": 0.88,
        "distance_per_type_100km_interval_within_200": [0.80, 0.85, 0.90, 0.95],
        "distance_conditions_100km": {},
    }
```

Replace the old module helper named `good_release_metrics` with this exact
shape so legacy baseline-only fields cannot accidentally satisfy the new gate.

- [ ] **Step 2: Run and verify current metric/gate failures**

```powershell
python -m pytest -q tests/test_evaluation.py
```

- [ ] **Step 3: Implement shared file-equal weights**

Add:

```python
def record_file_identity(record):
    if record.get("source_path"):
        return str(record["source_path"])
    if "file_id" in record:
        return str(record["file_id"])
    raise ValueError("prediction record has no stable source identity")


def file_equal_piece_weights(file_ids):
    file_ids = np.asarray(file_ids)
    _, inverse, counts = np.unique(file_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def weighted_type_metrics(true_types, predicted_types, accepted, weights, num_types=4):
    precision, recall = [], []
    for type_index in range(num_types):
        predicted = accepted & (predicted_types == type_index)
        actual = true_types == type_index
        correct = predicted & actual
        precision.append(_safe_ratio(weights[correct].sum(), weights[predicted].sum()))
        recall.append(_safe_ratio(weights[correct].sum(), weights[actual].sum()))
    return precision, recall


def round_metrics(value):
    if isinstance(value, dict):
        return {key: round_metrics(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [round_metrics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("metrics contain a non-finite value")
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value
```

Use `record_file_identity()` everywhere `evaluate_predictions()` or
`file_bootstrap_metrics()` groups records. OOF artifacts therefore use stable
relative `source_path`; existing synthetic/legacy records may continue using
`file_id`.

When a file is drawn during bootstrap, overwrite both possible identities so
duplicate draws remain independent pseudo-files:

```python
copied["file_id"] = f"{draw}:{file_identity}"
if "source_path" in copied:
    copied["source_path"] = f"{draw}:{file_identity}"
```

Use this result for `type_file_equal_precision`,
`type_file_equal_recall`, and `type_file_equal_recall_mean`. Keep raw piece
metrics separately. Do not use the old conditional average of per-file
precision.

Call `weighted_type_metrics()` a second time with an all-true accepted mask and
return `raw_type_file_equal_precision`, `raw_type_file_equal_recall`, and
`raw_type_file_equal_recall_mean`. Keep `type_piece_accuracy` and
`type_file_macro_accuracy` as raw four-class accuracy, while all unprefixed
precision/recall/coverage keys describe the IC-rejected end-to-end path.

Rename the misleading exact-distance outputs while updating their tests:

```python
result.update({
    "distance_100km_interval_count": interval_100km_summary["count"],
    "distance_100km_interval_within_100": interval_100km_summary["within_100"],
    "distance_100km_interval_within_200": interval_100km_summary["within_200"],
    "distance_per_type_100km_interval_within_200": per_type_100km,
})
```

Do not emit new `distance_exact_*` aliases; update in-repository consumers to
the 100-km-interval names so metric terminology cannot diverge again.

Update the calibration guard to compare
`distance_100km_interval_within_200`, `distance_file_macro_within_200`, and
`distance_interval_mae_km`; all three must be non-regressing within `1e-12`.

- [ ] **Step 4: Add condition file counts and absolute gates**

Keep coarse `distance_subgroups` for contextual reporting. Add a separate
`distance_conditions_100km` map keyed like `PNBE/night/300-400km`; every entry
must report `file_count`,
`file_macro_within_200`, and its 100-km-interval count. Add the fixed gate
implementation:

```python
def distance_condition_metrics(records):
    conditions = defaultdict(list)
    for record in records:
        low = record.get("distance_low_km")
        high = record.get("distance_high_km")
        if low is None or high is None or float(high) - float(low) != 100:
            continue
        type_name = TYPE_NAMES[int(record["true_type"])]
        light = "day" if bool(record.get("daylight", False)) else "night"
        name = f"{type_name}/{light}/{int(low)}-{int(high)}km"
        conditions[name].append(record)
    result = {}
    for name, selected in sorted(conditions.items()):
        by_file = defaultdict(list)
        for record in selected:
            by_file[record_file_identity(record)].append(record)
        file_scores = [
            _distance_summary(rows, exact_only=True)["within_200"]
            for rows in by_file.values()
        ]
        result[name] = {
            "piece_count": len(selected),
            "file_count": len(by_file),
            "file_macro_within_200": float(np.mean(file_scores)),
        }
    return result


RELEASE_GATES = {
    "min_per_type_precision": 0.95,
    "min_coverage": 0.80,
    "min_file_equal_macro_recall": 0.90,
    "min_100km_interval_within_200": 0.85,
    "min_per_type_100km_interval_within_200": 0.75,
    "min_supported_condition_within_200": 0.70,
    "minimum_supported_files": 3,
}


def absolute_gate_failures(candidate):
    reasons = []
    precision = candidate.get("type_file_equal_precision", [])
    if len(precision) != 4:
        reasons.append("four file-equal per-type precision values are required")
    for index, value in enumerate(precision):
        if float(value) < RELEASE_GATES["min_per_type_precision"]:
            reasons.append(
                f"type_file_equal_precision[{index}]={float(value):.4f} below 0.95"
            )
    scalar_gates = (
        ("type_coverage", "min_coverage"),
        ("type_file_equal_recall_mean", "min_file_equal_macro_recall"),
        ("distance_100km_interval_within_200", "min_100km_interval_within_200"),
    )
    for metric, gate in scalar_gates:
        value = float(candidate.get(metric, 0.0))
        if value < RELEASE_GATES[gate]:
            reasons.append(
                f"{metric}={value:.4f} below {RELEASE_GATES[gate]:.2f}"
            )
    per_type = candidate.get(
        "distance_per_type_100km_interval_within_200", []
    )
    if len(per_type) != 4:
        reasons.append("four per-type 100-km interval values are required")
    for index, value in enumerate(per_type):
        if float(value) < RELEASE_GATES[
            "min_per_type_100km_interval_within_200"
        ]:
            reasons.append(
                f"distance_within_200[{index}]={float(value):.4f} below 0.75"
            )
    return reasons


def evaluate_release(candidate):
    reasons = absolute_gate_failures(candidate)
    for name, group in candidate["distance_conditions_100km"].items():
        if (
            group["file_count"] >= RELEASE_GATES["minimum_supported_files"]
            and group["file_macro_within_200"]
            < RELEASE_GATES["min_supported_condition_within_200"]
        ):
            reasons.append(
                f"supported subgroup {name}={group['file_macro_within_200']:.4f} below 0.70"
            )
    return not reasons, reasons
```

Call `distance_condition_metrics(records)` exactly once in
`evaluate_predictions()` and return it as `distance_conditions_100km`; do not
derive release support from the coarse-band `distance_subgroups` map.

Remove baseline split-hash and subgroup-regression decisions. Keep historical
benchmark output clearly marked `reference_only: true`.

- [ ] **Step 5: Align checkpoint selection with robust conditions**

Move the validation selection key to `evaluation.py`:

```python
def checkpoint_selection_key(metrics):
    supported = [
        group["file_macro_within_200"]
        for group in metrics["distance_conditions_100km"].values()
        if group["file_count"] >= 3
    ]
    readiness = (
        metrics["type_file_equal_recall_mean"] >= 0.85
        and min(metrics["type_file_equal_recall"]) >= 0.75
    )
    return (
        int(readiness),
        metrics["type_file_equal_recall_mean"],
        min(metrics["type_file_equal_recall"]),
        min(supported, default=0.0),
        metrics["distance_file_macro_within_200"],
        min(metrics["distance_per_type_100km_interval_within_200"]),
        -metrics["distance_interval_mae_km"],
    )
```

The explicit non-release readiness floor is file-equal macro recall 0.85 and
minimum file-equal per-type recall 0.75. It only orders fold checkpoints; the
stricter OOF release contract remains unchanged.

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest -q tests/test_evaluation.py
git add evaluation.py benchmark.py tests/test_evaluation.py
git commit -m "Use file-equal OOF release metrics"
```

---

### Task 7: OOF Rejection Calibration With Metric Identity

**Files:**
- Modify: `open_set.py`
- Modify: `conditional_pipeline.py`
- Modify: `tests/test_open_set.py`

**Interfaces:**
- Consumes: per-fold OOF logits, normalized feature distances, quality, labels, and stable file identities.
- Produces: a transferable rejection-policy template and decoded OOF decisions that pass Task 6 metrics.

- [ ] **Step 1: Write failing calibration-identity tests**

```python
import numpy as np
import pytest
import torch

from evaluation import evaluate_predictions
from open_set import attach_final_feature_reference, fit_oof_rejection_policy


def separable_oof_signals():
    true_type = np.repeat(np.arange(4), 25)
    predicted = true_type.copy()
    file_id = np.repeat([f"correct-{index}" for index in range(4)], 25)
    true_type = np.concatenate([true_type, np.arange(4)])
    predicted = np.concatenate([predicted, np.roll(np.arange(4), -1)])
    return {
        "true_type": true_type,
        "predicted": predicted,
        "file_id": np.concatenate([file_id, [f"wrong-{i}" for i in range(4)]]),
        "confidence": np.concatenate([np.full(100, 0.99), np.full(4, 0.40)]),
        "margin": np.concatenate([np.full(100, 0.95), np.full(4, 0.05)]),
        "normalized_distance": np.concatenate([np.full(100, 0.1), np.full(4, 3.0)]),
        "quality_score": np.ones(104),
    }


def test_oof_policy_report_matches_evaluator_file_equal_precision():
    signals = separable_oof_signals()
    policy, decisions = fit_oof_rejection_policy(
        signals, target_precision=0.96, min_coverage=0.80
    )
    records = [{
        "file_id": signals["file_id"][index],
        "true_type": int(signals["true_type"][index]),
        "predicted_type": int(signals["predicted"][index]),
        "accepted": bool(decisions["accepted"][index]),
        "distance_low_km": 0,
        "distance_high_km": 100,
        "predicted_distance_km": 50.0,
        "oracle_distance_km": 50.0,
        "daylight": True,
    } for index in range(len(signals["true_type"]))]
    metrics = evaluate_predictions(records)
    assert policy["oof_metrics"]["type_file_equal_precision"] == pytest.approx(
        metrics["type_file_equal_precision"]
    )


def test_raw_feature_coordinates_are_not_transferred_between_models():
    template = {
        "version": 3,
        "temperature": 1.2,
        "probability_thresholds": [0.8] * 4,
        "margin_thresholds": [0.5] * 4,
        "normalized_distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.1] * 4,
        "fold_centroids": [[999.0, 999.0]] * 4,
    }
    final_features = torch.tensor([
        [0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0],
        [20.0, 20.0], [22.0, 22.0], [30.0, 30.0], [32.0, 32.0],
    ])
    labels = torch.repeat_interleave(torch.arange(4), 2)
    final = attach_final_feature_reference(template, final_features, labels)
    assert final["centroids"] != template["fold_centroids"]
    assert final["normalized_distance_thresholds"] == template["normalized_distance_thresholds"]
```

- [ ] **Step 2: Run and verify failures**

```powershell
python -m pytest -q tests/test_open_set.py
```

- [ ] **Step 3: Separate fold signals from model-specific references**

Expose:

```python
def rejection_signals(logits, features, reference, temperature, quality=None):
    predicted, confidence, margin, normalized_distance = _type_signals(
        _as_float_tensor(logits), _as_float_tensor(features), temperature, reference
    )
    return {
        "predicted": predicted,
        "confidence": confidence,
        "margin": margin,
        "normalized_distance": normalized_distance,
        "quality_score": waveform_quality_score(quality) if quality is not None else torch.ones(len(logits)),
    }
```

Fit one temperature per fold and store the robust median as the final model
temperature. Pool signals after fold-specific temperature scaling. Search
probability, margin, normalized-distance, and quality thresholds, but calculate
every candidate's precision through the Task 6 file-equal piece weights after
the accepted mask is known.

- [ ] **Step 4: Add mandatory post-fit verification**

Expose this entry point:

```python
def fit_oof_rejection_policy(
    signals,
    target_precision=0.96,
    min_coverage=0.80,
    fold_temperatures=None,
    fold_hashes=None,
):
    """Fit transferable thresholds and return (policy, decoded_decisions)."""
```

When the optional dictionaries are omitted in a unit test, use `{0: 1.0}` and
`{}` respectively. The function must decode every OOF row, call
`evaluate_predictions()`, and fail unless every
`type_file_equal_precision >= 0.96` and overall coverage is at least 0.80.
Return policy version 3 with fold temperatures, median temperature,
normalized thresholds, OOF metrics, fold hashes, and calibration hash.

Use these exact field names:

```python
calibration_hash = stable_json_hash({
    "fold_hashes": dict(fold_hashes),
    "piece_keys": list(signals.get(
        "piece_key", [str(index) for index in range(len(signals["true_type"]))]
    )),
    "accepted": [bool(value) for value in decisions["accepted"]],
    "probability_thresholds": [round(float(value), 12) for value in probability_thresholds],
    "margin_thresholds": [round(float(value), 12) for value in margin_thresholds],
    "normalized_distance_thresholds": [
        round(float(value), 12) for value in normalized_distance_thresholds
    ],
    "quality_thresholds": [round(float(value), 12) for value in quality_thresholds],
})

policy = {
    "version": 3,
    "temperature": float(np.median(fold_temperatures)),
    "fold_temperatures": {
        str(index): float(value)
        for index, value in sorted(fold_temperatures.items())
    },
    "probability_thresholds": probability_thresholds,
    "margin_thresholds": margin_thresholds,
    "normalized_distance_thresholds": normalized_distance_thresholds,
    "quality_thresholds": quality_thresholds,
    "target_precision": float(target_precision),
    "minimum_coverage": float(min_coverage),
    "oof_metrics": round_metrics(oof_metrics),
    "fold_hashes": dict(fold_hashes),
    "calibration_hash": calibration_hash,
}
```

- [ ] **Step 5: Rebind the final feature reference**

Implement `attach_final_feature_reference(template, features, labels)` by
calling `fit_feature_reference()` on the final model's full-data features and
copying only transferable scalar/quantile thresholds. Keep v1/v2 policies
decodable for old checkpoint compatibility.

```python
import copy


TRANSFERABLE_POLICY_FIELDS = (
    "version", "temperature", "fold_temperatures",
    "probability_thresholds", "margin_thresholds",
    "normalized_distance_thresholds", "quality_thresholds",
    "target_precision", "minimum_coverage", "oof_metrics",
    "fold_hashes", "calibration_hash",
)


def attach_final_feature_reference(template, features, labels):
    if int(template.get("version", 0)) != 3:
        raise ValueError("final feature rebinding requires policy version 3")
    rebound = {
        key: copy.deepcopy(template[key])
        for key in TRANSFERABLE_POLICY_FIELDS
        if key in template
    }
    rebound.update(fit_feature_reference(features, labels, num_types=4))
    return rebound
```

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest -q tests/test_open_set.py tests/test_evaluation.py
git add open_set.py conditional_pipeline.py tests/test_open_set.py
git commit -m "Calibrate IC rejection from OOF signals"
```

---

### Task 8: Fold Training, Resume, OOF Artifacts, and Gates

**Files:**
- Create: `cv_pipeline.py`
- Refactor: `conditional_pipeline.py`
- Modify: `train.py`
- Create: `tests/test_cv_pipeline.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: fold checkpoints, validated OOF CSV, `cv_metrics.json`, and a passing/failing decision before final training.

- [ ] **Step 1: Write orchestration tests with injected fake trainers**

```python
import csv
from datetime import datetime
from pathlib import Path

import pytest

from cv_pipeline import (
    CVConfig,
    FoldResult,
    run_cross_validated_training,
    training_config_hash,
)
from data.cross_validation import assign_exact_folds, fold_train_holdout
from data.oof_manifest import oof_row_id
from data.split_artifacts import split_hash
from data.training_manifest import ManifestEntry


def trusted_entries(tmp_path):
    rows = []
    for type_index, type_name in enumerate(("NCG", "NNBE", "PCG", "PNBE")):
        for file_index in range(3):
            rows.append(ManifestEntry(
                filepath=str(
                    tmp_path / "train" / type_name / "day" / "0-100km"
                    / f"file-{file_index}.lig"
                ),
                type_idx=type_index,
                dist_bin=0,
                timestamp=datetime(2020, 1, 1),
                n_pieces=2,
                distance_low_km=0,
                distance_high_km=100,
                is_daytime=True,
            ))
    return rows


def config(tmp_path, resume_cv=False, stop_after_oof=True):
    return CVConfig(
        task_data=tmp_path / "train",
        output=tmp_path / "weights",
        folds=3,
        seed=7,
        resume_cv=resume_cv,
        stop_after_oof=stop_after_oof,
    )


def fake_fold_trainer(calls):
    def train(train_entries, holdout_entries, output_dir, config, device, fold_index):
        calls.append(fold_index)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        oof_path = output_dir / "oof.csv"
        fields = ["piece_key", "fold", "true_type"]
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for entry in holdout_entries:
                relative = Path(entry.filepath).relative_to(config.task_data).as_posix()
                for piece_index in range(entry.n_pieces):
                    writer.writerow({
                        "piece_key": oof_row_id(relative, piece_index),
                        "fold": fold_index,
                        "true_type": entry.type_idx,
                    })
        return FoldResult(
            fold_index=fold_index,
            best_epoch=fold_index + 2,
            train_hash=split_hash(train_entries, config.task_data),
            holdout_hash=split_hash(holdout_entries, config.task_data),
            config_hash=training_config_hash(config),
            checkpoint_path=str(output_dir / "best.pt"),
            oof_path=str(oof_path),
            metrics={},
        )
    return train


def passing_oof_evaluator(rows, expected, output_dir, config):
    return {
        "passed": True,
        "reasons": [],
        "oof_piece_count": len(rows),
        "expected_piece_count": len(expected),
    }


def failing_oof_evaluator(rows, expected, output_dir, config):
    return {
        "passed": False,
        "reasons": ["type_file_equal_precision[0]=0.90 below 0.95"],
        "oof_piece_count": len(rows),
        "expected_piece_count": len(expected),
    }


def fail_if_called(*args, **kwargs):
    raise AssertionError("final trainer must not be called")


def test_cv_runs_each_fold_once_and_validates_all_oof_rows(tmp_path):
    calls = []
    report = run_cross_validated_training(
        config(tmp_path), trusted_entries(tmp_path), device="cpu",
        fold_trainer=fake_fold_trainer(calls), final_trainer=fail_if_called,
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [0, 1, 2]
    assert report["oof_piece_count"] == report["expected_piece_count"]


def test_resume_skips_only_hash_verified_complete_folds(tmp_path, monkeypatch):
    entries = trusted_entries(tmp_path)
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    train_entries, holdout_entries = fold_train_holdout(folds, 0)
    completed = fake_fold_trainer([])(
        train_entries,
        holdout_entries,
        config(tmp_path).output / "folds" / "fold_0",
        config(tmp_path),
        "cpu",
        0,
    )
    monkeypatch.setattr(
        "cv_pipeline.load_verified_fold",
        lambda output_dir, fold_index, expected_hashes, expected_rows: (
            completed if fold_index == 0 else None
        ),
    )
    calls = []
    run_cross_validated_training(
        config(tmp_path, resume_cv=True), entries, "cpu",
        fold_trainer=fake_fold_trainer(calls),
        oof_evaluator=passing_oof_evaluator,
    )
    assert calls == [1, 2]


def test_failed_oof_gate_never_calls_final_trainer_or_replaces_model(tmp_path):
    deployed = config(tmp_path, stop_after_oof=False).output / "model.pt"
    deployed.parent.mkdir(parents=True)
    deployed.write_bytes(b"existing")
    run_cross_validated_training(
        config(tmp_path, stop_after_oof=False), trusted_entries(tmp_path), "cpu",
        fold_trainer=fake_fold_trainer([]), final_trainer=fail_if_called,
        oof_evaluator=failing_oof_evaluator,
    )
    assert deployed.read_bytes() == b"existing"
```

- [ ] **Step 2: Run and verify missing-pipeline failures**

```powershell
python -m pytest -q tests/test_cv_pipeline.py tests/test_train.py
```

- [ ] **Step 3: Refactor a reusable fold trainer**

Move model/optimizer/loader epoch logic from `run_conditional_training()` into:

```python
import dataclasses
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CVConfig:
    task_data: Path
    output: Path
    folds: int = 3
    seed: int = 42
    samples_per_epoch: int = 120000
    max_samples_per_file: int = 512
    type_focus_epochs: int = 3
    type_focus_distance_weight: float = 0.25
    joint_distance_weight: float = 1.0
    max_epochs: int = 50
    patience: int = 10
    time_context: str = "daylight"
    rejection_target_precision: float = 0.96
    rejection_min_coverage: float = 0.80
    bootstrap_iterations: int = 1000
    num_workers: int = 2
    no_amp: bool = False
    no_init: bool = True
    init_model: str = ""
    resume_cv: bool = False
    stop_after_oof: bool = False


def training_config_hash(config):
    payload = dataclasses.asdict(config)
    for key in ("task_data", "output", "resume_cv", "stop_after_oof"):
        payload.pop(key, None)
    return stable_json_hash(payload)


@dataclass
class FoldResult:
    fold_index: int
    best_epoch: int
    train_hash: str
    holdout_hash: str
    config_hash: str
    checkpoint_path: str
    oof_path: str
    metrics: dict


def train_conditional_fold(
    train_entries, holdout_entries, output_dir, config, device, fold_index,
):
    """Train one random-init fold, select on holdout, and write its OOF rows."""
```

The fold checkpoint schema is `conditional_expert_cv_fold_v1` and requires
`fold_index`, `best_epoch`, `train_hash`, `holdout_hash`, `config_hash`,
`random_initialization=True`, `model_config`, `model_state`, and `metrics`.
Expose:

```python
def validate_fold_checkpoint(checkpoint, fold_index, expected_hashes):
    if checkpoint.get("schema") != "conditional_expert_cv_fold_v1":
        raise ValueError("invalid fold checkpoint schema")
    if int(checkpoint.get("fold_index", -1)) != int(fold_index):
        raise ValueError("fold checkpoint index mismatch")
    for key in ("train_hash", "holdout_hash", "config_hash"):
        if checkpoint.get(key) != expected_hashes[key]:
            raise ValueError(f"fold checkpoint {key} mismatch")
    if checkpoint.get("random_initialization") is not True:
        raise ValueError("fold checkpoint was not random-initialized")
    model = create_mtl_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
```

Every fold metadata/checkpoint includes fold index, train/holdout/config hashes,
random initialization, stage configuration, and best epoch. `best_epoch` is a
one-based completed-epoch count, not a zero-based loop index; the final median
therefore maps directly to a fixed training epoch count.

Expose the orchestration seam used by the tests:

```python
def run_cross_validated_training(
    config: CVConfig,
    entries: list[ManifestEntry],
    device,
    fold_trainer=train_conditional_fold,
    final_trainer=None,
    oof_evaluator=evaluate_oof_artifacts,
) -> dict:
    """Run verified folds, OOF release, and optionally final training."""
```

`train.py` is responsible for building the four-class manifest, proving it has
zero IC rows, converting parsed arguments to `CVConfig`, and calling this
function.

- [ ] **Step 4: Implement verified resume state**

Implement the exact resume loader:

```python
def load_verified_fold(output_dir, fold_index, expected_hashes, expected_rows):
    directory = Path(output_dir) / "folds" / f"fold_{int(fold_index)}"
    state_path = directory / "fold_state.json"
    best_path = directory / "best.pt"
    oof_path = directory / "oof.csv"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        return None
    if not (best_path.is_file() and oof_path.is_file()):
        raise ValueError(f"completed fold is missing artifacts: fold {fold_index}")
    for key in ("train_hash", "holdout_hash", "config_hash"):
        if state.get(key) != expected_hashes[key]:
            return False
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_fold_checkpoint(checkpoint, fold_index, expected_hashes)
    rows = read_oof_csv(oof_path)
    validate_oof_rows(rows, expected_rows)
    return FoldResult(
        fold_index=int(fold_index),
        best_epoch=int(state["best_epoch"]),
        train_hash=state["train_hash"],
        holdout_hash=state["holdout_hash"],
        config_hash=state["config_hash"],
        checkpoint_path=str(best_path),
        oof_path=str(oof_path),
        metrics=dict(state["metrics"]),
    )
```

`--resume_cv` skips only non-`None` results and passes `latest.pt` only to an
incomplete fold. Without `--resume_cv`, abort if any fold training artifact
already exists. A corrupt state, checkpoint, or CSV must raise an actionable
validation error instead of being silently treated as complete.

- [ ] **Step 5: Implement OOF aggregation and gates**

Expose the evaluator seam used by `run_cross_validated_training()`:

```python
def evaluate_oof_artifacts(rows, expected, output_dir, config) -> dict:
    """Calibrate pooled OOF rows, write artifacts, and return verified gates."""
```

Stream fold CSVs into `oof_predictions.csv`, validate Task 2 identities, fit
Task 7 rejection and guarded distance calibration, compute Task 6 metrics plus
file bootstrap intervals, and write `cv_metrics.json`. Mark historical model
numbers `reference_only` without reading them as gates.

Use this stable final CSV contract (the four logits/probabilities are separate
columns in `TYPE_NAMES` order):

```python
OOF_FIELDS = (
    "piece_key", "source_path", "piece_index", "fold", "true_type",
    "predicted_type", "final_type", "accepted", "rejection_reason",
    "logit_NCG", "logit_NNBE", "logit_PCG", "logit_PNBE",
    "prob_NCG", "prob_NNBE", "prob_PCG", "prob_PNBE",
    "confidence", "margin", "normalized_feature_distance", "quality_score",
    "distance_low_km", "distance_high_km", "predicted_distance_km",
    "oracle_distance_km", "distance_temperature", "daylight",
    "support_status", "train_hash", "holdout_hash", "config_hash",
)

OOF_INTEGER_FIELDS = {"piece_index", "fold", "true_type", "predicted_type", "final_type"}
OOF_FLOAT_FIELDS = {
    name for name in OOF_FIELDS
    if name.startswith("logit_") or name.startswith("prob_")
} | {
    "confidence", "margin", "normalized_feature_distance", "quality_score",
    "distance_low_km", "distance_high_km", "predicted_distance_km",
    "oracle_distance_km", "distance_temperature",
}


def read_oof_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OOF CSV has no header")
        rows = []
        for raw in reader:
            row = dict(raw)
            for name in OOF_INTEGER_FIELDS:
                if name in row and row[name] != "":
                    row[name] = int(row[name])
            for name in OOF_FLOAT_FIELDS:
                if name in row:
                    row[name] = None if row[name] == "" else float(row[name])
            for name in ("accepted", "daylight"):
                if name in row:
                    if row[name] not in {"True", "False", "1", "0"}:
                        raise ValueError(f"invalid OOF boolean {name}={row[name]!r}")
                    row[name] = row[name] in {"True", "1"}
            rows.append(row)
    return rows
```

Each fold first writes raw logits, normalized feature distance, quality, and
both routed/oracle distance predictions to `folds/fold_N/oof.csv`. Aggregation
fits one type temperature per fold, uses their median in the transferable
policy, and rewrites the final probability/decision columns atomically. Fit a
four-value per-type distance-temperature vector per fold and retain it only when
`distance_calibration_is_safe(before, after)` is true; unsafe folds contribute
`[1.0, 1.0, 1.0, 1.0]`, and the final vector is the elementwise median across
the three accepted/fallback vectors. Record before, after, fitted/accepted
vectors, and guard reason for every fold in
`cv_metrics.json`.

Add artifact revalidation:

```python
def verify_cv_artifacts(output_dir, expected_rows, expected_hashes, config):
    rows = read_oof_csv(Path(output_dir) / "oof_predictions.csv")
    validate_oof_rows(rows, expected_rows)
    metrics = evaluate_predictions(rows)
    metrics.update(file_bootstrap_metrics(
        rows, iterations=config.bootstrap_iterations, seed=config.seed
    ))
    passed, reasons = evaluate_release(metrics)
    saved = json.loads(
        (Path(output_dir) / "cv_metrics.json").read_text(encoding="utf-8")
    )
    if stable_json_hash(round_metrics(metrics)) != saved["recomputed_metrics_hash"]:
        raise ValueError("saved OOF metrics do not match oof_predictions.csv")
    if expected_hashes != saved["fold_hashes"]:
        raise ValueError("fold hashes changed after OOF evaluation")
    return {"passed": passed, "reasons": reasons, "metrics": metrics}
```

Round floating metrics to a documented 12 decimal places before hashing in
both writer and verifier. `verify_cv_artifacts()` is the only source of the
final `passed` value, so a report cannot authorize final training without being
recomputed from the saved CSV.

- [ ] **Step 6: Replace the training CLI**

`train.py` calls `run_cross_validated_training()` and exposes:

```text
--folds 3
--samples_per_epoch 120000
--max_samples_per_file 512
--type_focus_epochs 3
--type_focus_distance_weight 0.25
--joint_distance_weight 1.0
--max_epochs 50
--patience 10
--time_context daylight
--rejection_target_precision 0.96
--rejection_min_coverage 0.80
--resume_cv
--stop_after_oof
--verify_only
```

Remove baseline promotion arguments from the CV path. Keep `--no_init` accepted
and record it as true; reject any `--init_model` during CV/final training.
`--verify_only` rebuilds the manifest/fold hashes and calls
`verify_cv_artifacts()` without starting CUDA or changing any checkpoint.

- [ ] **Step 7: Run tests and commit**

```powershell
python -m pytest -q tests/test_cv_pipeline.py tests/test_train.py tests/test_evaluation.py tests/test_open_set.py
git add cv_pipeline.py conditional_pipeline.py train.py tests/test_cv_pipeline.py tests/test_train.py
git commit -m "Add resumable three-fold OOF training"
```

---

### Task 9: Full-Data Final Model and Atomic Promotion

**Files:**
- Modify: `cv_pipeline.py`
- Modify: `conditional_pipeline.py`
- Modify: `models.py`
- Modify: `tests/test_cv_pipeline.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: passing OOF report and fold best epochs.
- Produces: `final_candidate.pt` and atomically promoted `model.pt` using every trusted file.

- [ ] **Step 1: Write failing final-training and atomic-save tests**

```python
from cv_pipeline import atomic_promote_checkpoint, make_final_training_request


def fold_result(index, best_epoch):
    return FoldResult(
        fold_index=index, best_epoch=best_epoch,
        train_hash=f"train-{index}", holdout_hash=f"holdout-{index}",
        config_hash="config", checkpoint_path=f"fold-{index}/best.pt",
        oof_path=f"fold-{index}/oof.csv", metrics={},
    )


def test_final_request_uses_all_files_random_init_and_median_epoch(tmp_path):
    entries = trusted_entries(tmp_path)
    request = make_final_training_request(
        entries,
        [fold_result(0, 8), fold_result(1, 11), fold_result(2, 20)],
    )
    assert request.epochs == 11
    assert len(request.entries) == len(entries)
    assert request.init_model is None


def test_atomic_validation_failure_preserves_existing_model(tmp_path, monkeypatch):
    candidate = tmp_path / "final_candidate.pt"
    existing = tmp_path / "model.pt"
    existing.write_bytes(b"deployed")
    def reject(_checkpoint):
        raise ValueError("checkpoint validation failed")
    monkeypatch.setattr("cv_pipeline.validate_final_checkpoint", reject)
    with pytest.raises(ValueError, match="checkpoint validation"):
        atomic_promote_checkpoint(
            {"schema": "four_class_cv_v3"}, candidate, existing
        )
    assert existing.read_bytes() == b"deployed"
    assert not candidate.exists()
```

- [ ] **Step 2: Run and verify failures**

```powershell
python -m pytest -q tests/test_cv_pipeline.py tests/test_models.py
```

- [ ] **Step 3: Implement fixed-epoch full-data training**

Add the immutable request builder and `train_final_model()`:

```python
import statistics


@dataclass(frozen=True)
class FinalTrainingRequest:
    entries: tuple[ManifestEntry, ...]
    epochs: int
    init_model: str | None = None


def make_final_training_request(entries, fold_results):
    results = sorted(fold_results, key=lambda result: result.fold_index)
    if [result.fold_index for result in results] != [0, 1, 2]:
        raise ValueError("final training requires complete folds 0, 1, and 2")
    epochs = int(statistics.median(result.best_epoch for result in results))
    if epochs <= 0:
        raise ValueError("fold best epochs must be positive")
    return FinalTrainingRequest(tuple(entries), epochs, None)


def train_final_model(request, config, device):
    """Train one random-init model on all trusted files for a fixed epoch count."""
```

Implement `train_final_model()` using the same joint
sampler, augmentation, stage schedule, optimizer, scheduler, and seed contract
as folds. It must call `create_mtl_model()` without loading any state dict. It
has no validation loader and never performs early stopping. Save
`final_latest.pt` each epoch for `--resume_cv`, but never treat it as selected
evidence.

In `run_cross_validated_training()`, gate the call exactly as follows:

```python
verification = verify_cv_artifacts(
    config.output, expected_rows, fold_hashes, config
)
if not verification["passed"] or config.stop_after_oof:
    return verification
request = make_final_training_request(entries, fold_results)
trainer = final_trainer or train_final_model
checkpoint = trainer(request, config, device)
atomic_promote_checkpoint(
    checkpoint,
    config.output / "final_candidate.pt",
    config.output / "model.pt",
)
return {**verification, "final_model": str(config.output / "model.pt")}
```

Use `train_final_model` when no test double is injected. This is the only code
path permitted to call `atomic_promote_checkpoint()`.

- [ ] **Step 4: Bind calibration and support metadata**

After final training, collect a bounded, hierarchical full-data feature
reference, call `attach_final_feature_reference()`, and construct schema
`four_class_cv_v3` containing `context_dim`, fold manifest hashes, OOF metrics,
OOF calibration, full-data hash, final epoch count, support map, preprocessing,
augmentation, and model state.

Use at most 20,000 unaugmented full-data pieces, selected deterministically by
type, daylight, 100-km interval, file, then piece; store their selection hash.
The checkpoint must contain these exact top-level fields:

```python
FINAL_CHECKPOINT_FIELDS = {
    "schema", "model_version", "architecture", "type_names", "context_dim",
    "model_config", "model_state", "preprocessing", "augmentation",
    "fold_manifest_hash", "fold_hashes", "full_data_hash", "final_epochs",
    "random_initialization", "initialization_source", "oof_metrics",
    "oof_metrics_hash", "rejection_policy", "distance_temperatures",
    "support_map", "support_map_hash", "feature_reference_selection_hash",
}
```

Set `random_initialization=True` and `initialization_source=None`; do not store
an old-model path anywhere in the v3 checkpoint.

- [ ] **Step 5: Implement atomic validation and promotion**

```python
def atomic_promote_checkpoint(checkpoint, candidate_path, model_path):
    candidate_path = Path(candidate_path)
    model_path = Path(model_path)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temp = candidate_path.with_name(candidate_path.name + ".tmp")
    model_temp = model_path.with_name(model_path.name + ".tmp")
    try:
        torch.save(checkpoint, candidate_temp)
        validate_final_checkpoint(torch.load(
            candidate_temp, map_location="cpu", weights_only=False
        ))
        os.replace(candidate_temp, candidate_path)
        shutil.copy2(candidate_path, model_temp)
        validate_final_checkpoint(torch.load(
            model_temp, map_location="cpu", weights_only=False
        ))
        os.replace(model_temp, model_path)
    finally:
        candidate_temp.unlink(missing_ok=True)
        model_temp.unlink(missing_ok=True)
```

Validation checks schema, all hashes, four type names, context dimension,
policy version, support map, OOF gates, state-dict key/shape compatibility, and
absence of any old initialization path.

```python
def validate_final_checkpoint(checkpoint):
    missing = sorted(FINAL_CHECKPOINT_FIELDS - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint validation failed: missing {missing}")
    if checkpoint["schema"] != "four_class_cv_v3":
        raise ValueError("checkpoint validation failed: wrong schema")
    if checkpoint["type_names"] != ["NCG", "NNBE", "PCG", "PNBE"]:
        raise ValueError("checkpoint validation failed: wrong type order")
    if checkpoint["context_dim"] != 1:
        raise ValueError("checkpoint validation failed: context_dim must be 1")
    if checkpoint["random_initialization"] is not True:
        raise ValueError("checkpoint validation failed: model was not random-init")
    if checkpoint["initialization_source"] is not None:
        raise ValueError("checkpoint validation failed: initialization source exists")
    if checkpoint["rejection_policy"].get("version") != 3:
        raise ValueError("checkpoint validation failed: rejection policy is not v3")
    policy = checkpoint["rejection_policy"]
    if policy.get("fold_hashes") != checkpoint["fold_hashes"]:
        raise ValueError("checkpoint validation failed: calibration fold hashes")
    calibration_hash = str(policy.get("calibration_hash", ""))
    if len(calibration_hash) != 64 or any(
        char not in "0123456789abcdef" for char in calibration_hash
    ):
        raise ValueError("checkpoint validation failed: calibration hash")
    for key in (
        "fold_manifest_hash", "full_data_hash", "oof_metrics_hash",
        "support_map_hash",
        "feature_reference_selection_hash",
    ):
        value = str(checkpoint[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"checkpoint validation failed: invalid {key}")
    if stable_json_hash(round_metrics(checkpoint["oof_metrics"])) != checkpoint["oof_metrics_hash"]:
        raise ValueError("checkpoint validation failed: OOF metric hash mismatch")
    if set(checkpoint["fold_hashes"]) != {"0", "1", "2"}:
        raise ValueError("checkpoint validation failed: incomplete fold hashes")
    if not isinstance(checkpoint["support_map"], dict):
        raise ValueError("checkpoint validation failed: invalid support map")
    if stable_json_hash(checkpoint["support_map"]) != checkpoint["support_map_hash"]:
        raise ValueError("checkpoint validation failed: support map hash mismatch")
    temperatures = checkpoint["distance_temperatures"]
    if len(temperatures) != 4 or any(float(value) <= 0 for value in temperatures):
        raise ValueError("checkpoint validation failed: distance temperatures")
    passed, reasons = evaluate_release(checkpoint["oof_metrics"])
    if not passed:
        raise ValueError(f"checkpoint validation failed: OOF gates: {reasons}")
    model = create_mtl_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
```

- [ ] **Step 6: Run tests and commit**

```powershell
python -m pytest -q tests/test_cv_pipeline.py tests/test_models.py
git add cv_pipeline.py conditional_pipeline.py models.py tests/test_cv_pipeline.py tests/test_models.py
git commit -m "Train and atomically promote full-data model"
```

---

### Task 10: V3 Inference Support Status and Audit Output

**Files:**
- Modify: `classify.py`
- Modify: `open_set.py`
- Modify: `benchmark.py`
- Modify: `tests/test_classify.py`
- Modify: `tests/test_open_set.py`

**Interfaces:**
- Consumes: `four_class_cv_v3` checkpoint from Task 9.
- Produces: unchanged single-model inference plus explicit distance support evidence.

- [ ] **Step 1: Write failing support-status tests**

```python
import torch

from classify import annotate_distance_support, distance_support_status
from open_set import decode_with_rejection


def test_v3_rejection_reads_normalized_distance_thresholds():
    policy = {
        "version": 3,
        "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.4] * 4,
        "margin_thresholds": [0.2] * 4,
        "normalized_distance_thresholds": [2.0] * 4,
        "quality_thresholds": [0.0] * 4,
    }
    decoded = decode_with_rejection(
        torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        policy,
    )
    assert decoded["accepted"].tolist() == [True]


def test_insufficient_support_keeps_distance_but_marks_audit_status():
    checkpoint = {
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "support_map": {
            "NCG/day/400-500km": {
                "file_count": 2, "status": "insufficient_support"
            }
        },
    }
    prediction = annotate_distance_support(
        {
            "type_index": 0,
            "class_name": "NCG_400-500km",
            "expected_distance_km": 450.0,
            "modal_distance_bin": 4,
            "daylight": True,
        },
        checkpoint,
    )
    assert prediction["expected_distance_km"] == pytest.approx(450.0)
    assert prediction["support_status"] == "insufficient_support"
    assert prediction["support_file_count"] == 2
    assert prediction["class_name"] == "NCG_400-500km"


def test_old_checkpoint_without_support_map_reports_unknown():
    assert distance_support_status(
        {"type_names": ["NCG", "NNBE", "PCG", "PNBE"]}, 0, True, 400
    ) == ("unknown", None, None)
```

- [ ] **Step 2: Run and verify failures**

```powershell
python -m pytest -q tests/test_classify.py tests/test_open_set.py
```

- [ ] **Step 3: Add v3 loading and support lookup**

Accept `four_class_cv_v3` in `checkpoint_schema()` and use checkpoint
`context_dim` when constructing the model. In `decode_with_rejection()`, use
`normalized_distance_thresholds` for policy v3 and retain `distance_thresholds`
for v1/v2:

```python
distance_thresholds = _as_float_tensor(
    policy["normalized_distance_thresholds"]
    if int(policy.get("version", 1)) >= 3
    else policy["distance_thresholds"]
)
```

For v3 distance decoding, divide each routed expert's logits by
`checkpoint["distance_temperatures"][predicted_type]`; for oracle benchmark
decoding index the same vector by `true_type`. Keep the existing v1/v2 metadata
paths unchanged.

Then implement:

```python
def distance_support_status(checkpoint, type_index, daylight, bin_start_km):
    if bin_start_km is None:
        return "not_applicable", None, None
    support_map = checkpoint.get("support_map")
    if not support_map:
        return "unknown", None, None
    type_name = checkpoint["type_names"][int(type_index)]
    condition = f"{type_name}/{'day' if daylight else 'night'}/{int(bin_start_km)}-{int(bin_start_km) + 100}km"
    support = support_map.get(condition)
    if support is None:
        return "insufficient_support", 0, condition
    return support["status"], int(support["file_count"]), condition


def annotate_distance_support(prediction, checkpoint):
    result = dict(prediction)
    if result.get("class_name") == "IC":
        status, file_count, condition = "not_applicable", None, None
    elif result.get("type_only", False):
        status, file_count, condition = "not_evaluated", None, None
    else:
        modal_bin = result.get("modal_distance_bin")
        status, file_count, condition = distance_support_status(
            checkpoint,
            result["type_index"],
            result["daylight"],
            None if modal_bin is None else int(modal_bin) * 100,
        )
    result.update({
        "support_status": status,
        "support_file_count": file_count,
        "support_condition": condition,
    })
    return result
```

Apply after modal distance decoding. IC is `not_applicable`; type-only is
`not_evaluated`; missing v2 maps are `unknown`. Never suppress the predicted
type/distance or change the output folder solely because support is sparse.

- [ ] **Step 4: Extend CSV without changing raw-byte writing**

Add `support_status`, `support_file_count`, `support_condition`,
`fold_manifest_hash`, `full_data_hash`, and `calibration_hash` to
`PREDICTION_FIELDS` and writer rows. Populate the hash fields from the v3
checkpoint and leave them blank for legacy checkpoints. Preserve all current
raw-byte tests.
`benchmark.py` must retain these fields in prediction records but must not use
historical model metrics as release gates.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest -q tests/test_classify.py tests/test_open_set.py
git add classify.py open_set.py benchmark.py tests/test_classify.py tests/test_open_set.py
git commit -m "Report distance support during inference"
```

---

### Task 11: Documentation, Full Verification, and Real-Data Smoke

**Files:**
- Modify: `AGENTS.md`
- Review: `docs/superpowers/specs/2026-07-15-cross-validated-final-training-design.md` without modifying the approved design

**Interfaces:**
- Consumes: completed Tasks 1–10.
- Produces: reproducible contributor commands and verified training/inference handoff.

- [ ] **Step 1: Update repository commands and safety notes**

Document these final commands in `AGENTS.md`:

```powershell
python audit_data.py --task_data ..\train_data --output .\weights\conditional_cv\data_audit.json
python train.py --task_data ..\train_data --output .\weights\cv_smoke --max_epochs 1 --patience 1 --samples_per_epoch 2048 --bootstrap_iterations 50 --num_workers 0 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --max_epochs 50 --patience 10 --samples_per_epoch 120000 --max_samples_per_file 512 --bootstrap_iterations 1000 --num_workers 2 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --resume_cv --num_workers 2 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --verify_only --no_init
python classify.py --input_dir <lig-dir> --output_dir .\classified --model .\weights\conditional_cv\model.pt
```

State that folds are evaluation-only, final training uses every trusted file,
historical old metrics are contaminated reference data, and unsupported
distance cells remain visible in CSV.

- [ ] **Step 2: Run the complete unit suite**

```powershell
python -m pytest -q
```

Expected: all tests pass. If the tool-host Winsock issue prevents importing
PyTorch, run this command in the user's active `ligclassify` terminal and record
the exact result; do not claim a pass from compile-only checks.

- [ ] **Step 3: Run compile and CLI checks**

```powershell
python -m compileall -q .
python train.py --help
python audit_data.py --help
python classify.py --help
```

Expected: all commands exit zero and show the new CV arguments.

- [ ] **Step 4: Audit the real data before GPU work**

```powershell
python audit_data.py --task_data ..\train_data --output .\weights\conditional_cv\data_audit.json
```

Verify 1,142 trusted files, zero IC training pieces, three deterministic fold
hashes, no cross-fold source overlap, exact file/piece totals, and an explicit
insufficient-support list.

- [ ] **Step 5: Run the three-fold bounded smoke**

```powershell
python train.py --task_data ..\train_data --output .\weights\cv_smoke --max_epochs 1 --patience 1 --samples_per_epoch 2048 --max_samples_per_file 64 --bootstrap_iterations 50 --num_workers 0 --no_init
```

Verify each fold completes, OOF rows are unique and complete, a failing smoke
gate does not start final training, no AMP deprecation warning appears, and the
run finishes without touching existing deployed weights.

- [ ] **Step 6: Run bounded v3 inference after a passing final checkpoint exists**

First recompute every saved OOF gate without starting training:

```powershell
python train.py --task_data ..\train_data --output .\weights\conditional_cv --verify_only --no_init
```

Expected: fold/full hashes, OOF row count, recomputed metric hash, calibration
hash, and every release gate match the saved passing report. Then run:

```powershell
python classify.py --input_dir <small-lig-dir> --output_dir .\classified_smoke --model .\weights\conditional_cv\model.pt
```

Verify original bytes, four probabilities, IC rejection reason, distance,
support status, model version, and fold/full-data hashes in the audit CSV.

- [ ] **Step 7: Review the implementation and commit documentation**

Use the `requesting-code-review` and `verification-before-completion` skills,
address only evidenced defects, then run `git diff --check` and `git status`.

```powershell
git add AGENTS.md
git commit -m "Document cross-validated final training"
```

Expected final worktree: clean except user-owned ignored artifacts.
