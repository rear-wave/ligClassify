# Piece-Time Dataset Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace file/date splitting with deterministic 70/15/15 chronological splitting of waveform pieces inside each `(type, distance_bin)` group, and evaluate distance performance with equal-bin metrics.

**Architecture:** Keep file discovery and label parsing at file level, then expand valid files into timestamped `PieceManifestEntry` records. Build one shared `LigFileIndex` and expose three lazy dataset views containing disjoint `(filepath, piece_index)` identities. Training selection uses validation only; release evaluation uses the locked piece-level test split and equal-bin distance metrics.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, existing `.lig` binary parser.

## Global Constraints

- Read only `NCG`, `NNBE`, `PCG`, and `PNBE`; never add IC training data.
- Within every `(type, distance_bin)` group, sort by `(timestamp, filepath, piece_index)` and split earliest 70%, middle 15%, latest 15%.
- The same file may occur in several splits, but one `(filepath, piece_index)` identity may occur in exactly one split.
- Read each piece timestamp from its binary header; an unreadable timestamp fails before training and never falls back to the filename.
- Never rewrite, copy, or commit source `.lig` files, checkpoints, or generated classifications.
- Preserve unrelated modifications in `classify.py`, `models.py`, `tests/test_classify.py`, and `tests/test_models.py`; stage only files named by each task.

---

### Task 1: Read all piece timestamps with one file handle

**Files:**
- Modify: `data/lig_parser.py`
- Modify: `tests/test_training_manifest.py`

**Interfaces:**
- Produces: `read_lig_timestamps(filepath: str) -> list[datetime]`
- Preserves: `read_lig_timestamp(filepath: str, piece_index: int = 0) -> datetime`

- [ ] **Step 1: Extend the synthetic writer and add failing timestamp tests**

Update `write_lig` without breaking existing callers:

```python
def write_lig(
    path: Path,
    timestamp=(19, 1, 2, 3, 4, 5),
    sec_frac=0.25,
    pieces=1,
    piece_timestamps=None,
    piece_sec_fracs=None,
):
    timestamps = piece_timestamps or [timestamp] * pieces
    fractions = piece_sec_fracs or [sec_frac] * pieces
    if len(timestamps) != pieces or len(fractions) != pieces:
        raise ValueError("piece timestamp metadata must match pieces")
    raw = bytearray(FILE_HEADER_BYTES + PIECE_BYTES * pieces)
    for piece_idx, (piece_timestamp, piece_fraction) in enumerate(
        zip(timestamps, fractions)
    ):
        piece_start = FILE_HEADER_BYTES + PIECE_BYTES * piece_idx
        struct.pack_into("i", raw, piece_start, 1001)
        struct.pack_into("6i4x", raw, piece_start + 108, *piece_timestamp)
        struct.pack_into("d", raw, piece_start + 136, piece_fraction)
        struct.pack_into("H", raw, piece_start + 208, piece_idx + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path
```

Add:

```python
def test_read_lig_timestamps_reads_each_piece_in_order(tmp_path):
    path = write_lig(
        tmp_path / "sample.lig",
        pieces=3,
        piece_timestamps=[
            (19, 1, 2, 3, 4, 5),
            (19, 1, 2, 3, 4, 6),
            (19, 1, 2, 3, 4, 7),
        ],
        piece_sec_fracs=[0.1, 0.2, 0.3],
    )
    assert lig_parser.read_lig_timestamps(str(path)) == [
        datetime(2019, 1, 2, 3, 4, 5, 100000),
        datetime(2019, 1, 2, 3, 4, 6, 200000),
        datetime(2019, 1, 2, 3, 4, 7, 300000),
    ]


def test_read_lig_timestamps_reports_bad_piece_index(tmp_path):
    path = write_lig(
        tmp_path / "bad.lig",
        pieces=2,
        piece_timestamps=[
            (19, 1, 2, 3, 4, 5),
            (19, 13, 2, 3, 4, 6),
        ],
    )
    with pytest.raises(lig_parser.LigFormatError, match=r"piece_index=1"):
        lig_parser.read_lig_timestamps(str(path))
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_training_manifest.py -k "read_lig_timestamps" -q`

Expected: FAIL because `read_lig_timestamps` does not exist.

- [ ] **Step 3: Implement one-handle timestamp reading**

Add a private decoder and batch reader, then make the single-piece API use the same decoder:

```python
def _decode_lig_timestamp(raw, filepath, piece_index):
    if len(raw) != _LIG_TIMESTAMP_BYTES + _LIG_SECFRAC_BYTES:
        raise LigFormatError(
            f"incomplete timestamp: {filepath}: piece_index={piece_index}"
        )
    try:
        year, month, day, hour, minute, second = struct.unpack_from(
            "6i4x", raw, 0
        )
        sec_frac = struct.unpack_from("d", raw, _LIG_TIMESTAMP_BYTES)[0]
        return _datetime_from_lig_fields(
            year, month, day, hour, minute, second, sec_frac
        )
    except (struct.error, ValueError) as exc:
        raise LigFormatError(
            f"invalid timestamp: {filepath}: piece_index={piece_index}: {exc}"
        ) from exc


def read_lig_timestamps(filepath):
    """Read every piece timestamp using one sequentially reused file handle."""
    n_pieces = count_lig_pieces(filepath)
    timestamps = []
    try:
        with open(filepath, "rb") as handle:
            for piece_index in range(n_pieces):
                offset = (
                    _LIG_FILE_HEADER_BYTES
                    + piece_index * _LIG_PIECE_BYTES
                    + _LIG_PIECE_HEADER_BYTES
                )
                handle.seek(offset)
                raw = handle.read(_LIG_TIMESTAMP_BYTES + _LIG_SECFRAC_BYTES)
                timestamps.append(
                    _decode_lig_timestamp(raw, filepath, piece_index)
                )
    except OSError as exc:
        raise LigFormatError(f"failed to read timestamp: {filepath}: {exc}") from exc
    return timestamps
```

Keep the bounds check in `read_lig_timestamp`, read its one record, and return `_decode_lig_timestamp(raw, filepath, piece_index)`.

- [ ] **Step 4: Run parser tests**

Run: `python -m pytest tests/test_training_manifest.py -k "lig_timestamp" -q`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit Task 1 only**

```powershell
git add data/lig_parser.py tests/test_training_manifest.py
git commit -m "Add batch piece timestamp reader"
```

---

### Task 2: Build and split the piece manifest

**Files:**
- Modify: `data/training_manifest.py`
- Modify: `tests/test_training_manifest.py`

**Interfaces:**
- Consumes: `read_lig_timestamps(filepath)` from Task 1
- Produces: `PieceManifestEntry`, `build_piece_manifest`, `piece_time_split_manifest`, `validate_piece_split_isolation`, `validate_piece_split_coverage`

- [ ] **Step 1: Add failing tests for expansion, ordering, ratios, and isolation**

Import `datetime` as already used and add helpers/tests:

```python
def make_piece(module, filepath, piece_index, type_idx, dist_bin, second):
    return module.PieceManifestEntry(
        filepath=filepath,
        piece_index=piece_index,
        type_idx=type_idx,
        dist_bin=dist_bin,
        timestamp=datetime(2020, 1, 1, 0, 0, second),
    )


def test_build_piece_manifest_keeps_individual_timestamps(tmp_path):
    module = training_manifest_module()
    path = write_lig(
        tmp_path / "NCG" / "0-100km" / "sample.lig",
        pieces=3,
        piece_timestamps=[
            (20, 1, 1, 0, 0, 2),
            (20, 1, 1, 0, 0, 0),
            (20, 1, 1, 0, 0, 1),
        ],
    )
    files, _ = module.build_manifest(str(tmp_path), ["NCG"])
    pieces = module.build_piece_manifest(files)
    assert [(x.filepath, x.piece_index) for x in pieces] == [
        (str(path), 0), (str(path), 1), (str(path), 2)
    ]
    assert [x.timestamp.second for x in pieces] == [2, 0, 1]


def test_piece_time_split_is_chronological_per_type_and_bin():
    module = training_manifest_module()
    entries = [
        make_piece(module, "shared.lig", index, 0, 5, second)
        for index, second in enumerate([9, 0, 8, 1, 7, 2, 6, 3, 5, 4])
    ]
    splits = module.piece_time_split_manifest(entries, 0.15, 0.15)
    assert [x.timestamp.second for x in splits["train"]] == list(range(7))
    assert [x.timestamp.second for x in splits["val"]] == [7, 8]
    assert [x.timestamp.second for x in splits["test"]] == [9]
    identities = [
        {(x.filepath, x.piece_index) for x in splits[name]}
        for name in ("train", "val", "test")
    ]
    assert identities[0].isdisjoint(identities[1])
    assert identities[0].isdisjoint(identities[2])
    assert identities[1].isdisjoint(identities[2])
    assert {x.filepath for x in splits["train"]} & {
        x.filepath for x in splits["test"]
    } == {"shared.lig"}


def test_piece_time_split_gives_three_piece_group_to_all_splits():
    module = training_manifest_module()
    entries = [make_piece(module, "a.lig", i, 0, 0, i) for i in range(3)]
    splits = module.piece_time_split_manifest(entries, 0.15, 0.15)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [1, 1, 1]


def test_piece_time_split_rejects_group_smaller_than_three():
    module = training_manifest_module()
    entries = [make_piece(module, "a.lig", i, 0, 0, i) for i in range(2)]
    with pytest.raises(ValueError, match=r"type=0.*bin=0.*2 pieces"):
        module.piece_time_split_manifest(entries, 0.15, 0.15)
```

- [ ] **Step 2: Run the new split tests and verify failure**

Run: `python -m pytest tests/test_training_manifest.py -k "piece_manifest or piece_time_split" -q`

Expected: FAIL because the new dataclass and functions are undefined.

- [ ] **Step 3: Implement piece entries and constrained chronological splitting**

Add:

```python
@dataclass(frozen=True)
class PieceManifestEntry:
    filepath: str
    piece_index: int
    type_idx: int
    dist_bin: int
    timestamp: datetime

    @property
    def identity(self):
        return (os.path.normcase(os.path.abspath(self.filepath)), self.piece_index)


def build_piece_manifest(file_entries):
    """Expand validated files into timestamped, lazily readable pieces."""
    pieces = []
    for file_entry in file_entries:
        timestamps = read_lig_timestamps(file_entry.filepath)
        if len(timestamps) != file_entry.n_pieces:
            raise LigFormatError(
                f"piece count changed: {file_entry.filepath}: "
                f"manifest={file_entry.n_pieces}, timestamps={len(timestamps)}"
            )
        pieces.extend(
            PieceManifestEntry(
                filepath=file_entry.filepath,
                piece_index=piece_index,
                type_idx=file_entry.type_idx,
                dist_bin=file_entry.dist_bin,
                timestamp=timestamp,
            )
            for piece_index, timestamp in enumerate(timestamps)
        )
    return pieces


def _constrained_split_counts(size, val_fraction, test_fraction):
    fractions = np.asarray(
        [1.0 - val_fraction - test_fraction, val_fraction, test_fraction],
        dtype=np.float64,
    )
    if size < 3:
        raise ValueError(f"at least 3 pieces are required, got {size}")
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train, validation, and test fractions must be positive")
    raw = fractions * size
    counts = np.maximum(1, np.floor(raw).astype(np.int64))
    while int(counts.sum()) > size:
        candidates = np.flatnonzero(counts > 1)
        index = int(candidates[np.argmax((counts - raw)[candidates])])
        counts[index] -= 1
    while int(counts.sum()) < size:
        index = int(np.argmax(raw - counts))
        counts[index] += 1
    return tuple(int(value) for value in counts)


def piece_time_split_manifest(entries, val_fraction=0.15, test_fraction=0.15):
    """Split pieces chronologically inside every type/distance group."""
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry.type_idx, entry.dist_bin)].append(entry)
    splits = {"train": [], "val": [], "test": []}
    for (type_idx, dist_bin), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda x: (x.timestamp, os.path.normcase(x.filepath), x.piece_index),
        )
        try:
            train_count, val_count, _ = _constrained_split_counts(
                len(ordered), val_fraction, test_fraction
            )
        except ValueError as exc:
            raise ValueError(
                f"type={type_idx} bin={dist_bin} has {len(ordered)} pieces: {exc}"
            ) from exc
        val_end = train_count + val_count
        splits["train"].extend(ordered[:train_count])
        splits["val"].extend(ordered[train_count:val_end])
        splits["test"].extend(ordered[val_end:])
    return splits
```

Add `import numpy as np` and import `read_lig_timestamps`.

- [ ] **Step 4: Add and pass isolation and coverage validation tests**

Add these tests:

```python
def test_piece_split_isolation_rejects_duplicate_identity():
    module = training_manifest_module()
    duplicate = make_piece(module, "same.lig", 0, 0, 0, 0)
    splits = {"train": [duplicate], "val": [], "test": [duplicate]}
    with pytest.raises(ValueError, match=r"train and test"):
        module.validate_piece_split_isolation(splits)


def test_piece_split_coverage_requires_every_source_bin():
    module = training_manifest_module()
    entries = [
        make_piece(
            module,
            f"bin-{dist_bin}.lig",
            piece_index,
            0,
            dist_bin,
            piece_index,
        )
        for dist_bin in range(30)
        for piece_index in range(20)
    ]
    splits = module.piece_time_split_manifest(entries, 0.15, 0.15)
    module.validate_piece_split_isolation(splits)
    module.validate_piece_split_coverage(
        splits, ["NCG"], min_eval_pieces=1
    )
    assert {x.dist_bin for x in splits["val"]} == set(range(30))
    assert {x.dist_bin for x in splits["test"]} == set(range(30))

    splits["test"] = [x for x in splits["test"] if x.dist_bin != 29]
    with pytest.raises(ValueError, match=r"NCG test: missing bins \[29\]"):
        module.validate_piece_split_coverage(
            splits, ["NCG"], min_eval_pieces=1
        )
```

Implement:

```python
def validate_piece_split_isolation(splits):
    """Reject a piece identity assigned to more than one split."""
    owners = {}
    for split_name, entries in splits.items():
        for entry in entries:
            previous = owners.setdefault(entry.identity, split_name)
            if previous != split_name:
                raise ValueError(
                    f"piece identity appears in {previous} and {split_name}: "
                    f"{entry.identity}"
                )


def validate_piece_split_coverage(splits, type_names, min_eval_pieces=500):
    source = splits["train"] + splits["val"] + splits["test"]
    failures = []
    for type_idx, type_name in enumerate(type_names):
        source_bins = {x.dist_bin for x in source if x.type_idx == type_idx}
        for split_name in ("val", "test"):
            selected = [x for x in splits[split_name] if x.type_idx == type_idx]
            selected_bins = {x.dist_bin for x in selected}
            missing = sorted(source_bins - selected_bins)
            if missing:
                failures.append(f"{type_name} {split_name}: missing bins {missing}")
            if len(selected) < min_eval_pieces:
                failures.append(
                    f"{type_name} {split_name}: pieces={len(selected)} "
                    f"below required {min_eval_pieces}"
                )
    if failures:
        raise ValueError("Invalid piece split coverage: " + "; ".join(failures))
```

Run: `python -m pytest tests/test_training_manifest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2 only**

```powershell
git add data/training_manifest.py tests/test_training_manifest.py
git commit -m "Add chronological piece split"
```

---

### Task 3: Make lazy datasets consume piece identities

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `list[PieceManifestEntry]`
- Produces: `MultiTaskDataset(entries, split, lig_index=None, normalize_mode="minmax")`

- [ ] **Step 1: Replace the existing dataset test with a failing piece-view test**

Create one three-piece file with distinct timestamps. Build `PieceManifestEntry` objects for indices 0 and 2, pass a shared `LigFileIndex`, and assert:

```python
dataset = train.MultiTaskDataset(entries, split="test", lig_index=shared_index)
assert dataset.global_indices.tolist() == [0, 2]
assert dataset.type_labels.tolist() == [0, 0]
assert dataset.dist_labels.tolist() == [1, 1]
assert dataset.date_ids.tolist() == [20200101, 20200103]
assert dataset.file_ids.tolist() == [0, 0]
assert len(dataset) == 2
assert int(dataset[1][0][0, 0]) >= 0
```

Also assert `dataset.close()` does not close the externally supplied index by reading global piece 0 after the call.

- [ ] **Step 2: Run the dataset test and verify failure**

Run: `python -m pytest tests/test_train.py -k "multitask_dataset" -q`

Expected: FAIL because the dataset still expands every piece in every input file.

- [ ] **Step 3: Implement selected-piece mapping**

Replace `MultiTaskDataset.__init__` with logic equivalent to:

```python
def __init__(self, entries, split="train", lig_index=None, normalize_mode="minmax"):
    self.split = split
    self.normalize_mode = normalize_mode
    paths = sorted({entry.filepath for entry in entries})
    self._owns_lig_index = lig_index is None
    self.lig = lig_index or LigFileIndex(paths, validate=False)
    path_to_file = {
        os.path.normcase(os.path.abspath(path)): index
        for index, path in enumerate(self.lig.filepaths)
    }
    global_indices, type_labels, dist_labels = [], [], []
    file_ids, date_ids = [], []
    for entry in entries:
        key = os.path.normcase(os.path.abspath(entry.filepath))
        if key not in path_to_file:
            raise ValueError(f"piece references an unindexed file: {entry.filepath}")
        file_id = path_to_file[key]
        if not 0 <= entry.piece_index < self.lig.num_pieces_per_file[file_id]:
            raise IndexError(
                f"piece_index={entry.piece_index} outside {entry.filepath}"
            )
        global_indices.append(int(self.lig._cumsum[file_id]) + entry.piece_index)
        type_labels.append(entry.type_idx)
        dist_labels.append(entry.dist_bin)
        file_ids.append(file_id)
        date_ids.append(int(entry.timestamp.strftime("%Y%m%d")))
    self.global_indices = np.asarray(global_indices, dtype=np.int64)
    self.type_labels = np.asarray(type_labels, dtype=np.int8)
    self.dist_labels = np.asarray(dist_labels, dtype=np.int8)
    self.file_ids = np.asarray(file_ids, dtype=np.int32)
    self.date_ids = np.asarray(date_ids, dtype=np.int32)
```

Change `close` to close only owned indexes:

```python
def close(self):
    if getattr(self, "_owns_lig_index", False) and hasattr(self, "lig"):
        self.lig.close()
```

Keep `_items_from_positions`, `__getitem__`, and `__getitems__` unchanged.

- [ ] **Step 4: Run dataset and sampler regression tests**

Run: `python -m pytest tests/test_train.py tests/test_distance_sampling.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3 only**

```powershell
git add train.py tests/test_train.py
git commit -m "Use lazy piece-indexed dataset views"
```

---

### Task 4: Wire piece splitting into training and metadata

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_piece_manifest`, `piece_time_split_manifest`, `validate_piece_split_isolation`, `validate_piece_split_coverage`
- Updates: `compute_split_hash(entries, split_name)` for piece identities

- [ ] **Step 1: Add failing CLI, hash, and overlap-summary tests**

Assert the parser exposes `--min_eval_pieces` with default 500 and retains `--val_fraction 0.15` and `--test_fraction 0.15`. Replace file-hash fixtures with `PieceManifestEntry` fixtures and assert changing only `piece_index` changes the hash. Add:

```python
def split_piece(path, piece_index):
    return PieceManifestEntry(
        filepath=path,
        piece_index=piece_index,
        type_idx=0,
        dist_bin=0,
        timestamp=datetime(2020, 1, 1),
    )


def test_split_hash_changes_with_piece_identity(tmp_path):
    first = [split_piece(str(tmp_path / "a.lig"), 0)]
    second = [split_piece(str(tmp_path / "a.lig"), 1)]
    assert train.compute_split_hash(first, "train") != train.compute_split_hash(
        second, "train"
    )


def test_shared_file_count_reports_expected_overlap():
    splits = {
        "train": [split_piece("same.lig", 0)],
        "val": [split_piece("same.lig", 1)],
        "test": [split_piece("same.lig", 2), split_piece("other.lig", 0)],
    }
    assert train.count_cross_split_files(splits) == 1
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_train.py -k "arguments or split_hash or shared_file" -q`

Expected: FAIL on the new argument and helper behavior.

- [ ] **Step 3: Replace main split construction and dataset creation**

Import the new manifest functions. Replace the file-level split block with:

```python
file_manifest, diagnostics = build_manifest(args.task_data, TYPE_NAMES)
if not file_manifest:
    raise RuntimeError(f"No valid .lig files found under {args.task_data}")
piece_manifest = build_piece_manifest(file_manifest)
split_entries = piece_time_split_manifest(
    piece_manifest,
    val_fraction=args.val_fraction,
    test_fraction=args.test_fraction,
)
validate_piece_split_isolation(split_entries)
validate_piece_split_coverage(
    split_entries,
    TYPE_NAMES,
    min_eval_pieces=args.min_eval_pieces,
)
shared_lig = LigFileIndex(
    sorted({entry.filepath for entry in piece_manifest}),
    validate=False,
)
train_set = MultiTaskDataset(split_entries["train"], "train", shared_lig)
val_set = MultiTaskDataset(split_entries["val"], "val", shared_lig)
test_set = None if args.skip_test else MultiTaskDataset(
    split_entries["test"], "test", shared_lig
)
```

For every split, log `len(entries)` as pieces, `len({x.filepath ...})` as contributing files, timestamp range, and all covered/missing bins per type. Add:

```python
def count_cross_split_files(splits):
    memberships = defaultdict(set)
    for split_name, entries in splits.items():
        for entry in entries:
            memberships[os.path.normcase(os.path.abspath(entry.filepath))].add(
                split_name
            )
    return sum(len(names) > 1 for names in memberships.values())
```

Log a warning stating that shared-file evaluation does not measure cross-file generalization.

- [ ] **Step 4: Update arguments, split hashes, and checkpoint metadata**

Remove `--min_val_bins` and `--min_val_pieces`; add:

```python
p.add_argument("--min_eval_pieces", type=int, default=500)
```

Replace `compute_split_hash` with:

```python
def compute_split_hash(entries, split_name):
    """Return a stable hash of selected piece identities and labels."""
    rows = [
        "|".join([
            split_name,
            os.path.normcase(os.path.abspath(entry.filepath)),
            str(entry.piece_index),
            str(entry.type_idx),
            str(entry.dist_bin),
            entry.timestamp.isoformat(),
        ])
        for entry in entries
    ]
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
```

Set metadata to:

```python
"split_config": {
    "strategy": "distance_stratified_piece_time_v1",
    "train_fraction": 1.0 - args.val_fraction - args.test_fraction,
    "val_fraction": args.val_fraction,
    "test_fraction": args.test_fraction,
    "min_eval_pieces": args.min_eval_pieces,
    "files_may_overlap": True,
    "piece_identities_disjoint": True,
},
```

Store per-split hashes and `cross_split_file_count`. Change temporal wording in `--skip_test`, release-gate docstrings, result messages, and rejection reason to simply “locked test”.

- [ ] **Step 5: Run integration unit tests**

Run: `python -m pytest tests/test_training_manifest.py tests/test_train.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 4 only**

```powershell
git add train.py tests/test_train.py
git commit -m "Train from chronological piece splits"
```

---

### Task 5: Make equal-distance-bin metrics primary

**Files:**
- Modify: `train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces: `summarize_equal_bin_distance_predictions(predictions, targets)`
- Adds metrics: `per_type_equal_bin_w2`, `dist_equal_bin_macro_w2`, `dist_equal_bin_min_type_w2`, per-type `*_equal_bin_*`

- [ ] **Step 1: Add a failing imbalance test**

```python
def test_equal_bin_metrics_do_not_let_large_bin_dominate():
    predictions = np.asarray([0] * 100 + [10])
    targets = np.asarray([0] * 100 + [1])
    metrics = train.summarize_equal_bin_distance_predictions(
        predictions, targets
    )
    assert metrics["bin_count"] == 2
    assert metrics["w2"] == pytest.approx(0.5)
    assert metrics["acc"] == pytest.approx(0.5)
    assert metrics["mae_km"] == pytest.approx(450.0)
```

Add a release-gate test where raw `dist_macro_w2` and `per_type_w2` pass but `dist_equal_bin_macro_w2` and one value in `per_type_equal_bin_w2` fail; assert the candidate is rejected for the equal-bin value.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_train.py -k "equal_bin" -q`

Expected: FAIL because equal-bin metrics are undefined.

- [ ] **Step 3: Implement equal-bin aggregation**

```python
def summarize_equal_bin_distance_predictions(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if predictions.shape != targets.shape or not len(targets):
        raise ValueError("non-empty aligned predictions and targets are required")
    summaries = []
    for distance_bin in sorted(np.unique(targets)):
        mask = targets == distance_bin
        error = np.abs(predictions[mask] - targets[mask])
        summaries.append({
            "acc": float(np.mean(error == 0)),
            "mae_km": float(np.mean(error) * 100),
            "w1": float(np.mean(error <= 1)),
            "w2": float(np.mean(error <= 2)),
        })
    return {
        key: float(np.mean([summary[key] for summary in summaries]))
        for key in ("acc", "mae_km", "w1", "w2")
    } | {"bin_count": len(summaries)}
```

Before the per-type loop, initialize `per_type_equal_bin_w2 = []` and `per_type_equal_bin_mae = []`. In the non-empty per-type block add:

```python
equal_bin = summarize_equal_bin_distance_predictions(preds, trues)
for metric_name in ("acc", "mae_km", "w1", "w2", "bin_count"):
    metrics[f"{key}_equal_bin_{metric_name}"] = equal_bin[metric_name]
per_type_equal_bin_w2.append(equal_bin["w2"])
per_type_equal_bin_mae.append(equal_bin["mae_km"])
```

In the empty branch append `0.0` to both lists. After the loop add:

```python
metrics["per_type_equal_bin_w2"] = per_type_equal_bin_w2
metrics["dist_equal_bin_macro_w2"] = float(
    np.mean(per_type_equal_bin_w2)
)
metrics["dist_equal_bin_min_type_w2"] = float(
    np.min(per_type_equal_bin_w2)
)
metrics["dist_equal_bin_macro_mae_km"] = float(
    np.mean(per_type_equal_bin_mae)
)
```

- [ ] **Step 4: Route selection, release, and logs through equal-bin keys**

Change `make_four_class_selection_key` to:

```python
return (
    float(metrics["type_f1"]),
    float(metrics["type_min_recall"]),
    float(metrics["type_min_precision"]),
    float(metrics["dist_equal_bin_macro_w2"]),
    -float(metrics["dist_equal_bin_macro_mae_km"]),
)
```

In `evaluate_release_gate`, read:

```python
macro_w2 = float(metrics.get("dist_equal_bin_macro_w2", 0.0))
per_type = [
    float(value)
    for value in metrics.get("per_type_equal_bin_w2", [])
]
```

Keep all existing raw piece-weighted keys in reports for comparison. Extend result logging with `equal_bin_w2`, `equal_bin_min_w2`, and each type's equal-bin W2/MAE.

- [ ] **Step 5: Run all training metric tests**

Run: `python -m pytest tests/test_train.py tests/test_distance_ordinal.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 5 only**

```powershell
git add train.py tests/test_train.py
git commit -m "Use equal-bin distance release metrics"
```

---

### Task 6: Update guidance and verify on current data

**Files:**
- Modify: `AGENTS.md`
- Verify: all repository Python files and tests

**Interfaces:**
- Documents: the actual piece-level split contract and current training command

- [ ] **Step 1: Update repository guidance**

Replace file-level/temporal split wording in `AGENTS.md` with this paragraph:

```markdown
Training expands each valid file into `(filepath, piece_index)` identities. Within every `(type, distance_bin)` group, pieces are sorted by their binary timestamp and split into earliest 70% training, middle 15% validation, and latest 15% test views. Files may occur in several views, but piece identities must remain disjoint. Shared-file evaluation measures held-out pieces and must not be described as cross-file generalization.
```

- [ ] **Step 2: Run a real-data split audit without training**

Run this read-only audit:

```powershell
@'
from collections import Counter
from data.training_manifest import (
    build_manifest,
    build_piece_manifest,
    piece_time_split_manifest,
    validate_piece_split_coverage,
)
names = ["NCG", "NNBE", "PCG", "PNBE"]
files, diagnostics = build_manifest(r"..\train_data", names)
pieces = build_piece_manifest(files)
splits = piece_time_split_manifest(pieces, 0.15, 0.15)
validate_piece_split_coverage(splits, names, min_eval_pieces=500)
identities = {}
for split_name, entries in splits.items():
    identities[split_name] = {entry.identity for entry in entries}
    print(split_name, len(entries), len({entry.filepath for entry in entries}))
    for type_idx, type_name in enumerate(names):
        selected = [entry for entry in entries if entry.type_idx == type_idx]
        print(type_name, len(selected), len({entry.dist_bin for entry in selected}))
assert identities["train"].isdisjoint(identities["val"])
assert identities["train"].isdisjoint(identities["test"])
assert identities["val"].isdisjoint(identities["test"])
'@ | python -
```

Expected: validation succeeds; validation and test each report 30 bins for NCG, NNBE, PCG, and PNBE; all identity assertions pass.

- [ ] **Step 3: Run full verification**

```powershell
python -m pytest -q
python -m compileall -q .
python train.py --help
```

Expected: all tests PASS, compileall emits no errors, and help lists `--min_eval_pieces`.

- [ ] **Step 4: Run a bounded one-epoch smoke training**

```powershell
python train.py --task_data ..\train_data --output .\weights\piece_split_smoke --no_init --epochs 1 --type_samples_per_epoch 512 --distance_samples_per_epoch 512 --batch_size 64 --distance_batch_size 64 --skip_test
```

Expected: logs show the `distance_stratified_piece_time_v1` three-way split, 30 validation bins for all types, no duplicate piece identity error, one completed epoch, and a candidate rejected only because the locked test was skipped/baseline gate was not satisfied. The command must not read or overwrite `weights/four_class/candidate.pt`.

- [ ] **Step 5: Commit documentation only**

```powershell
git add AGENTS.md
git commit -m "Document piece-level training split"
```

- [ ] **Step 6: Inspect final scope**

Run: `git status --short` and `git log -6 --oneline`.

Expected: only the pre-existing unrelated inference changes and `.workbuddy/` remain uncommitted; task commits contain no `.lig`, checkpoint, generated classification, or machine-specific absolute path.
