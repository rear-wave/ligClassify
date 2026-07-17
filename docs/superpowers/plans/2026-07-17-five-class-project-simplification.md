# Five-Class Project Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the four-class CV/rejection system with one compact, randomly initialized five-class joint type-and-distance pipeline using deterministic piece-level splits.

**Architecture:** Build a new focused data path beside the existing implementation, connect a five-class dual-scale network and compact training/inference CLIs, then delete the superseded CV/OOF/open-set stack only after the new path passes its regression suite. Checkpoint adaptation isolates the retained legacy five-class model from the new five_class_v1 schema.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, scikit-learn, tqdm, pytest.

## Global Constraints

- Train exactly IC, NCG, NNBE, PCG, and PNBE.
- IC receives no distance label; NCG, NNBE, PCG, and PNBE require aligned exact 100-km labels in 0-3000 km.
- Sample IC at 60% and each non-IC type at 10% during training.
- Split waveform pieces, not files, at deterministic 70%/15%/15% ratios; one LIG container may contribute to every partition.
- Do not use acquisition year or chronological cutoffs.
- Preserve waveform polarity; never negate, reverse, or circularly shift a waveform.
- Train the new model from random initialization; reject warm-start arguments.
- Infer IC by direct five-class argmax with no rejection threshold.
- Preserve original piece bytes during inference.
- Retain only weights/old/model.pt and weights/old/metrics.json from existing weights.
- Before weight cleanup, require the retained model SHA-256 to equal f3d74de7f1e1ad5b7f3a796f3a145c58e6f9fc86730c45a84f74a634a2cf82ea.
- Never modify or delete ../train_data or external classified outputs.
- Keep fixtures synthetic and do not commit LIG data or generated weights.
- Preserve unrelated user changes until the cleanup task explicitly handles the three items the user approved deleting: the evaluation.py threshold edit, CLAUDE.md, and _check_gates.py.

---

## Locked File Map

The completed runtime repository contains these Python modules:

~~~text
train.py
classify.py
audit_data.py
models.py
training.py
evaluation.py
checkpoints.py
data/__init__.py
data/lig.py
data/manifest.py
data/preprocess.py
data/dataset.py
data/sampling.py
data/split.py
~~~

The completed focused test suite contains:

~~~text
tests/conftest.py
tests/test_lig.py
tests/test_manifest.py
tests/test_split.py
tests/test_preprocess.py
tests/test_dataset.py
tests/test_sampling.py
tests/test_models.py
tests/test_checkpoints.py
tests/test_training.py
tests/test_classify.py
tests/test_audit.py
~~~

---

### Task 1: Compact LIG and Piece Manifest Core

**Files:**
- Create: data/lig.py
- Create: data/manifest.py
- Modify: data/__init__.py
- Create: tests/test_lig.py
- Create: tests/test_manifest.py
- Create: tests/conftest.py

**Interfaces:**
- Produces: LigFileIndex, read_lig_timestamp(), read_file_header(), read_raw_piece(), write_lig_file().
- Produces: SourceRecord, PieceTable, build_piece_table(), piece_key(), TYPE_NAMES, DISTANCE_NAMES.
- Consumes later: every split, dataset, audit, training, and inference task uses PieceTable positions.

- [ ] **Step 1: Add failing binary-read and raw-byte tests**

Create a synthetic writer in tests/test_lig.py using the repository format constants:

~~~python
import struct

import numpy as np

from data.lig import (
    FILE_HEADER_BYTES,
    PIECE_BYTES,
    WAVEFORM_SAMPLES,
    LigFileIndex,
    read_raw_piece,
    write_lig_file,
)


def make_piece(value, year=2020, month=1, day=2, hour=3):
    raw = bytearray(PIECE_BYTES)
    struct.pack_into("<6i4x", raw, 108, year - 2000, month, day, hour, 4, 5)
    struct.pack_into("<d", raw, 136, 0.25)
    waveform = np.full(WAVEFORM_SAMPLES, value, dtype="<u2")
    raw[208:208 + waveform.nbytes] = waveform.tobytes()
    return bytes(raw)


def write_source(path, pieces):
    header = bytearray(FILE_HEADER_BYTES)
    struct.pack_into("<i", header, 4, len(pieces))
    path.write_bytes(bytes(header) + b"".join(pieces))


def test_raw_piece_round_trip_is_byte_exact(tmp_path):
    source = tmp_path / "source.lig"
    pieces = [make_piece(11), make_piece(22)]
    write_source(source, pieces)
    index = LigFileIndex([source], validate=True)

    assert np.all(index.read_piece(1) == 22)
    assert read_raw_piece(source, 1) == pieces[1]

    output = tmp_path / "output.lig"
    write_lig_file(output, source.read_bytes()[:FILE_HEADER_BYTES], [pieces[1]])
    assert output.read_bytes()[FILE_HEADER_BYTES:] == pieces[1]
    assert struct.unpack_from("<i", output.read_bytes(), 4)[0] == 1
~~~

- [ ] **Step 2: Add failing piece-table tests**

Create tests/test_manifest.py:

~~~python
from data.manifest import (
    DISTANCE_NAMES,
    TYPE_NAMES,
    build_piece_table,
    piece_key,
)
from tests.test_lig import make_piece, write_source


def test_manifest_expands_storage_files_to_stable_piece_rows(tmp_path):
    root = tmp_path / "train_data"
    source = root / "NNBE" / "day" / "500-600km" / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1, hour=8), make_piece(2, hour=20)])

    table, diagnostics = build_piece_table(root)

    assert TYPE_NAMES == ("IC", "NCG", "NNBE", "PCG", "PNBE")
    assert DISTANCE_NAMES == ("NCG", "NNBE", "PCG", "PNBE")
    assert len(table) == 2
    assert table.type_index.tolist() == [2, 2]
    assert table.distance_bin.tolist() == [5, 5]
    assert table.daylight.tolist() == [True, False]
    assert table.piece_key(1) == "NNBE/day/500-600km/sample.lig#1"
    assert piece_key(r"NNBE\\day\\a.lig", 7) == "NNBE/day/a.lig#7"
    assert diagnostics["pieces"] == 2
~~~

Add tests that reject a non-IC file without an exact interval, reject intervals wider than 100 km, reject bins outside 0-3000 km, and allow IC without a distance directory.

Create shared metadata-only fixtures in tests/conftest.py. The single-stratum
fixture contains 20 pieces in one source so piece-level splitting is observable;
the general fixture contains every type, both daylight values, and multiple
distance bins:

~~~python
import numpy as np
import pytest

from data.manifest import PieceTable, SourceRecord


@pytest.fixture
def single_stratum_table(tmp_path):
    source = SourceRecord(
        path=str(tmp_path / "NCG" / "day" / "0-100km" / "one.lig"),
        relative_path="NCG/day/0-100km/one.lig",
        type_index=1,
        distance_bin=0,
        piece_count=20,
    )
    return PieceTable(
        root=str(tmp_path),
        sources=(source,),
        source_index=np.zeros(20, dtype=np.int32),
        piece_index=np.arange(20, dtype=np.int32),
        type_index=np.ones(20, dtype=np.int8),
        distance_bin=np.zeros(20, dtype=np.int8),
        daylight=np.ones(20, dtype=bool),
        timestamp_seconds=np.arange(20, dtype=np.int64),
    )


@pytest.fixture
def piece_table(tmp_path):
    per_type = 40
    sources = tuple(
        SourceRecord(
            path=str(tmp_path / f"type_{type_index}.lig"),
            relative_path=f"type_{type_index}.lig",
            type_index=type_index,
            distance_bin=-1 if type_index == 0 else type_index,
            piece_count=per_type,
        )
        for type_index in range(5)
    )
    type_index = np.repeat(np.arange(5, dtype=np.int8), per_type)
    return PieceTable(
        root=str(tmp_path),
        sources=sources,
        source_index=np.repeat(np.arange(5, dtype=np.int32), per_type),
        piece_index=np.tile(np.arange(per_type, dtype=np.int32), 5),
        type_index=type_index,
        distance_bin=np.where(type_index == 0, -1, type_index).astype(np.int8),
        daylight=np.tile(np.arange(per_type) % 2 == 0, 5),
        timestamp_seconds=np.arange(5 * per_type, dtype=np.int64),
    )
~~~

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

~~~powershell
python -m pytest -q tests/test_lig.py tests/test_manifest.py
~~~

Expected: import failures for data.lig and data.manifest.

- [ ] **Step 4: Implement the compact LIG API**

Port only the validated binary indexing and timestamp behavior from data/lig_parser.py. Use these constants and public contracts:

~~~python
FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208
PIECE_HEADER_BYTES = 208
WAVEFORM_SAMPLES = 16000
WAVEFORM_BYTES = WAVEFORM_SAMPLES * 2


def read_file_header(path):
    with open(path, "rb") as handle:
        header = handle.read(FILE_HEADER_BYTES)
    if len(header) != FILE_HEADER_BYTES:
        raise LigFormatError(f"short LIG header: {path}")
    return header


def read_raw_piece(path, piece_index):
    offset = FILE_HEADER_BYTES + int(piece_index) * PIECE_BYTES
    with open(path, "rb") as handle:
        handle.seek(offset)
        raw = handle.read(PIECE_BYTES)
    if len(raw) != PIECE_BYTES:
        raise LigFormatError(f"short piece {piece_index}: {path}")
    return raw


def write_lig_file(path, source_header, raw_pieces):
    header = bytearray(source_header)
    struct.pack_into("<i", header, 4, len(raw_pieces))
    with open(path, "wb") as handle:
        handle.write(header)
        for raw in raw_pieces:
            if len(raw) != PIECE_BYTES:
                raise LigFormatError("refusing to write a reconstructed piece")
            handle.write(raw)
~~~

LigFileIndex must retain batched waveform and timestamp reads, validate source size against FILE_HEADER_BYTES + n * PIECE_BYTES, and never cache all waveform payloads.

- [ ] **Step 5: Implement memory-compact piece metadata**

Use one source record per file and aligned NumPy arrays per piece:

~~~python
@dataclass(frozen=True)
class SourceRecord:
    path: str
    relative_path: str
    type_index: int
    distance_bin: int
    piece_count: int


@dataclass
class PieceTable:
    root: str
    sources: tuple[SourceRecord, ...]
    source_index: np.ndarray
    piece_index: np.ndarray
    type_index: np.ndarray
    distance_bin: np.ndarray
    daylight: np.ndarray
    timestamp_seconds: np.ndarray

    def __len__(self):
        return len(self.source_index)

    def piece_key(self, position):
        source = self.sources[int(self.source_index[position])]
        return piece_key(source.relative_path, self.piece_index[position])
~~~

build_piece_table() must recursively scan only the five named type directories, parse exact interval labels for all non-IC sources, batch-read each piece timestamp, and populate int32/int8/bool/int64 arrays. A path or piece identity appearing twice is a ValueError.

- [ ] **Step 6: Run focused tests and compile**

Run:

~~~powershell
python -m pytest -q tests/test_lig.py tests/test_manifest.py
python -m compileall -q data\lig.py data\manifest.py
~~~

Expected: all focused tests pass and compile emits no output.

- [ ] **Step 7: Commit**

~~~powershell
git add data\lig.py data\manifest.py data\__init__.py tests\conftest.py tests\test_lig.py tests\test_manifest.py
git commit -m "Add compact five-class piece manifest"
~~~

---

### Task 2: Deterministic Piece-Level Split

**Files:**
- Create: data/split.py
- Create: tests/test_split.py

**Interfaces:**
- Consumes: PieceTable from Task 1.
- Produces: SplitAssignment, assign_piece_splits(), validate_piece_split(), split_artifact(), write_split_json().

- [ ] **Step 1: Write failing piece-not-file split tests**

Create tests/test_split.py:

~~~python
import numpy as np

from data.split import (
    TEST,
    TRAIN,
    VALIDATION,
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
)


def test_piece_split_is_deterministic_balanced_and_crosses_files(
    single_stratum_table,
):
    first = assign_piece_splits(single_stratum_table, seed=42)
    second = assign_piece_splits(single_stratum_table, seed=42)

    assert np.array_equal(first.partition, second.partition)
    validate_piece_split(single_stratum_table, first)
    counts = np.bincount(first.partition, minlength=3)
    assert counts.tolist() == [14, 3, 3]

    positions_from_source_zero = np.flatnonzero(
        single_stratum_table.source_index == 0
    )
    assert set(first.partition[positions_from_source_zero]) == {
        TRAIN,
        VALIDATION,
        TEST,
    }


def test_split_artifact_is_compact(single_stratum_table):
    assignment = assign_piece_splits(single_stratum_table, seed=42)
    artifact = split_artifact(single_stratum_table, assignment)

    assert "piece_rows" not in artifact
    assert artifact["schema"] == "piece_stratified_split_v1"
    assert artifact["ratios"] == [0.70, 0.15, 0.15]
    assert set(artifact["partition_hashes"]) == {"train", "validation", "test"}
~~~

The fixture must include IC day/night rows and all four non-IC types in at least one exact interval. Add tests proving input order does not change ownership, different seeds change ownership, a two-piece stratum stays train-only, and missing/duplicate ownership fails validation.

- [ ] **Step 2: Run the split test and confirm RED**

Run:

~~~powershell
python -m pytest -q tests/test_split.py
~~~

Expected: ModuleNotFoundError for data.split.

- [ ] **Step 3: Implement stable per-stratum ownership**

Use exact constants and hashing:

~~~python
TRAIN = 0
VALIDATION = 1
TEST = 2
PARTITION_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class SplitAssignment:
    partition: np.ndarray
    seed: int

    def positions(self, name):
        index = PARTITION_NAMES.index(str(name))
        return np.flatnonzero(self.partition == index)


def _rank(seed, key):
    payload = f"{int(seed)}|{key}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def _partition_counts(size):
    if size < 3:
        return size, 0, 0
    validation = max(1, int(size * 0.15))
    test = max(1, int(size * 0.15))
    train = size - validation - test
    if train < 1:
        raise ValueError("stratum cannot populate training")
    return train, validation, test
~~~

For non-IC rows, the stratum key is (type_index, daylight, distance_bin). For IC, use (0, daylight, -1). Sort each stratum by (_rank(seed, piece_key), piece_key), slice the three counts, and fill one uint8 partition array aligned with PieceTable.

validate_piece_split() must require exactly one valid partition value for every table position, verify every n >= 3 stratum contains all three partitions, and recompute ownership from the stored seed before accepting a persisted artifact.

- [ ] **Step 4: Implement compact split evidence**

split_artifact() returns JSON-safe data only:

~~~python
{
    "schema": "piece_stratified_split_v1",
    "seed": 42,
    "ratios": [0.70, 0.15, 0.15],
    "manifest_hash": "...",
    "partition_hashes": {
        "train": "...",
        "validation": "...",
        "test": "...",
    },
    "counts": {
        "train": 0,
        "validation": 0,
        "test": 0,
    },
    "strata": {},
}
~~~

Partition hashes are SHA-256 over sorted lines piece_key + tab + partition_name. The artifact must not contain per-piece rows or absolute paths.

- [ ] **Step 5: Run focused tests and commit**

~~~powershell
python -m pytest -q tests/test_split.py tests/test_manifest.py
git add data\split.py tests\test_split.py
git commit -m "Add deterministic piece-level split"
~~~

---

### Task 3: Signed Preprocessing, Dataset, and 60% IC Sampler

**Files:**
- Create: data/preprocess.py
- Create: data/dataset.py
- Create: data/sampling.py
- Create: tests/test_preprocess.py
- Create: tests/test_dataset.py
- Create: tests/test_sampling.py

**Interfaces:**
- Consumes: PieceTable and SplitAssignment.
- Produces: PreprocessConfig, AugmentationConfig, preprocess_views(), FiveClassDataset, SampleRequest, FiveClassSampler, collate_batch().

- [ ] **Step 1: Write failing polarity and synchronized-view tests**

Create tests/test_preprocess.py:

~~~python
import numpy as np

from data.preprocess import (
    AugmentationConfig,
    PreprocessConfig,
    augment_batch,
    preprocess_views,
)


def test_augmentation_is_deterministic_polarity_safe_and_non_wrapping():
    values = np.zeros((1, 16000), dtype=np.float32)
    values[0, 100] = 10.0
    config = AugmentationConfig(
        max_shift=32,
        gain_min=0.9,
        gain_max=1.1,
        drift_fraction=0.0,
        noise_fraction=0.0,
    )
    first = augment_batch(values, [17], config)
    second = augment_batch(values, [17], config)

    assert np.array_equal(first, second)
    assert first.max() > 0
    assert first[0, -64:].max() == 0


def test_local_and_global_views_come_from_one_signed_waveform():
    values = np.linspace(-4.0, 8.0, 16000, dtype=np.float32)[None, :]
    local, global_view = preprocess_views(values, PreprocessConfig(use_filter=False))

    assert local.shape == (1, 8000)
    assert global_view.shape == (1, 2000)
    assert local.min() < 0 < local.max()
    assert global_view.min() < 0 < global_view.max()
~~~

- [ ] **Step 2: Write failing dataset and sampler tests**

Create tests/test_sampling.py:

~~~python
from collections import Counter

from data.sampling import FiveClassSampler


def test_sampler_uses_exact_60_10_10_10_10_prior(piece_table):
    sampler = FiveClassSampler(
        piece_table,
        positions=range(len(piece_table)),
        num_samples=1000,
        seed=9,
    )
    requests = list(sampler)
    counts = Counter(piece_table.type_index[item.position] for item in requests)

    assert counts == {0: 600, 1: 100, 2: 100, 3: 100, 4: 100}
    assert len({item.augmentation_seed for item in requests}) == 1000
~~~

Add a case where one type has one daylight group with many distance bins and the other daylight group has one bin; assert day/night counts differ by at most one before interval balancing. Assert repeated piece sampling is allowed and seed + epoch is deterministic.

Create tests/test_dataset.py proving integer indices are unaugmented in validation, SampleRequest seeds augment only training, quality is not returned, context shape is one, and piece identity survives collation.

- [ ] **Step 3: Run focused tests and confirm RED**

~~~powershell
python -m pytest -q tests/test_preprocess.py tests/test_dataset.py tests/test_sampling.py
~~~

Expected: imports fail for the three new modules.

- [ ] **Step 4: Implement signed preprocessing and safe augmentation**

Use immutable configurations:

~~~python
@dataclass(frozen=True)
class PreprocessConfig:
    local_length: int = 8000
    global_length: int = 2000
    use_filter: bool = True
    cutoff_hz: float = 120_000.0
    sample_rate_hz: float = 5_000_000.0


@dataclass(frozen=True)
class AugmentationConfig:
    max_shift: int = 64
    gain_min: float = 0.90
    gain_max: float = 1.10
    drift_fraction: float = 0.01
    noise_fraction: float = 0.01
~~~

Normalize each view as (x - median) / max(quantile(abs(x - median), 0.95), 1e-6). Derive the local peak window and global downsample from the same signed source row. _zero_shift() uses zero-filled slices and never np.roll. Gain validation requires 0 < gain_min <= gain_max. Validation/test calls preprocess_views() directly without augment_batch().

- [ ] **Step 5: Implement the lazy dataset**

FiveClassDataset accepts table, integer positions, split name, and optional augmentation config. It opens only referenced source files through LigFileIndex. Its item contract is:

~~~python
{
    "local": torch.FloatTensor[1, 8000],
    "global": torch.FloatTensor[1, 2000],
    "daylight": torch.FloatTensor[1],
    "type_label": torch.LongTensor[],
    "distance_bin": torch.LongTensor[],
    "source_path": str,
    "piece_index": int,
    "piece_key": str,
}
~~~

For SampleRequest, use request.position and request.augmentation_seed. For an integer, use the integer and no augmentation seed. Training applies augmentation only when a seed is present; validation and test ignore any supplied seed.

- [ ] **Step 6: Implement hierarchical type/day/interval sampling**

Use:

~~~python
TYPE_PRIOR = (0.60, 0.10, 0.10, 0.10, 0.10)


@dataclass(frozen=True)
class SampleRequest:
    position: int
    augmentation_seed: int
~~~

Allocate exact type quotas with largest remainders so they sum to num_samples. For each type, split its quota evenly over available daylight values. For each non-IC type/day cell, split again over exact distance bins. Draw positions with replacement from each leaf, shuffle the completed requests once, and derive unique deterministic 64-bit augmentation seeds from SHA-256(seed, epoch, draw_index, position). Missing required types, non-positive epochs, or a non-IC distance_bin outside 0..29 are hard errors.

- [ ] **Step 7: Run focused tests and commit**

~~~powershell
python -m pytest -q tests/test_preprocess.py tests/test_dataset.py tests/test_sampling.py
git add data\preprocess.py data\dataset.py data\sampling.py tests\test_preprocess.py tests\test_dataset.py tests\test_sampling.py
git commit -m "Add five-class data pipeline"
~~~

---

### Task 4: Five-Class Model and Two-Schema Checkpoints

**Files:**
- Replace: models.py
- Create: checkpoints.py
- Create: tests/test_models.py
- Create: tests/test_checkpoints.py

**Interfaces:**
- Produces: FiveClassNet, LegacyMultiTaskResNet, create_five_class_model(), ModelOutput.
- Produces: FIVE_CLASS_SCHEMA, LoadedCheckpoint, save_model_checkpoint(), load_model_checkpoint(), model_sha256().

- [ ] **Step 1: Replace model tests with exact five-class contracts**

Create tests/test_models.py:

~~~python
import torch

from models import create_five_class_model


def test_five_class_model_has_one_type_head_and_four_distance_experts():
    model = create_five_class_model(base_channels=16)
    output = model(
        torch.randn(3, 1, 8000),
        torch.randn(3, 1, 2000),
        torch.tensor([[1.0], [0.0], [1.0]]),
    )

    assert output.type_logits.shape == (3, 5)
    assert len(output.distance_logits) == 4
    assert all(logits.shape == (3, 30) for logits in output.distance_logits)
~~~

Add tests that daylight changes the fused input, local/global branches both receive gradients, and no model factory accepts an init checkpoint.

- [ ] **Step 2: Write failing checkpoint compatibility tests**

Create tests/test_checkpoints.py:

~~~python
import torch

from checkpoints import (
    FIVE_CLASS_SCHEMA,
    load_model_checkpoint,
    save_model_checkpoint,
)
from models import LegacyMultiTaskResNet, create_five_class_model


def test_new_checkpoint_round_trip(tmp_path):
    model = create_five_class_model(base_channels=16)
    path = tmp_path / "model.pt"
    save_model_checkpoint(
        path,
        model,
        model_config={"base_channels": 16},
        preprocess_config={"local_length": 8000, "global_length": 2000},
        split_hash="abc",
        training_config={"distance_weight": 0.5},
    )
    loaded = load_model_checkpoint(path, "cpu")
    assert loaded.schema == FIVE_CLASS_SCHEMA
    assert loaded.type_names == ("IC", "NCG", "NNBE", "PCG", "PNBE")


def test_legacy_five_class_checkpoint_is_supported(tmp_path):
    legacy = LegacyMultiTaskResNet(base=16)
    path = tmp_path / "legacy.pt"
    torch.save({
        "model_name": "mtl_resnet",
        "base_channels": 16,
        "type_names": ["IC", "NCG", "NNBE", "PCG", "PNBE"],
        "dist_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "dist_bin_starts": list(range(0, 3000, 100)),
        "preprocessing": {"normalize_mode": "minmax", "target_length": 8000},
        "model_state_dict": legacy.state_dict(),
    }, path)
    loaded = load_model_checkpoint(path, "cpu")
    assert loaded.schema == "legacy_five_class"
~~~

Also reject former four-class schemas, raw state dicts, mismatched class ordering, missing preprocessing metadata, and malformed tensor shapes.

- [ ] **Step 3: Run tests and confirm RED**

~~~powershell
python -m pytest -q tests/test_models.py tests/test_checkpoints.py
~~~

- [ ] **Step 4: Implement the focused model**

Retain the proven residual local/global branch pattern but expose one immutable output:

~~~python
@dataclass(frozen=True)
class ModelOutput:
    type_logits: torch.Tensor
    distance_logits: tuple[torch.Tensor, ...]
    features: torch.Tensor


class FiveClassNet(nn.Module):
    def forward(self, local, global_view, daylight):
        local_features = self.local_branch(local)
        global_features = self.global_branch(global_view)
        features = self.fusion(torch.cat(
            [local_features, global_features, daylight], dim=1
        ))
        return ModelOutput(
            type_logits=self.type_head(features),
            distance_logits=tuple(head(features) for head in self.distance_heads),
            features=features,
        )
~~~

LegacyMultiTaskResNet retains only the exact layers needed by weights/old/model.pt. Remove ordinal_v2 and four-class-only model factories.

- [ ] **Step 5: Implement strict schema adapters**

FIVE_CLASS_SCHEMA is "five_class_v1". New checkpoints use:

~~~python
{
    "schema": "five_class_v1",
    "model_config": {},
    "model_state": {},
    "type_names": ["IC", "NCG", "NNBE", "PCG", "PNBE"],
    "distance_names": ["NCG", "NNBE", "PCG", "PNBE"],
    "distance_bins_km": list(range(0, 3000, 100)),
    "preprocess_config": {},
    "split_hash": "...",
    "training_config": {},
}
~~~

LoadedCheckpoint includes model, schema, names, preprocessing mapping, and raw metadata. It must not expose optimizer state to inference. save_model_checkpoint() writes path + ".tmp", reloads and validates it, then uses os.replace().

- [ ] **Step 6: Run focused tests and commit**

~~~powershell
python -m pytest -q tests/test_models.py tests/test_checkpoints.py
git add models.py checkpoints.py tests\test_models.py tests\test_checkpoints.py
git commit -m "Add five-class model checkpoint contract"
~~~

---

### Task 5: Joint Loss and Piece-Level Evaluation

**Files:**
- Create: training.py
- Replace: evaluation.py
- Create: tests/test_training.py

**Interfaces:**
- Consumes: ModelOutput batches from Task 4.
- Produces: compute_joint_loss(), train_epoch(), evaluate_loader(), selection_score(), EarlyStoppingState.

- [ ] **Step 1: Write failing IC-mask and expert-routing tests**

Create tests/test_training.py:

~~~python
import torch

from models import ModelOutput
from training import compute_joint_loss


def test_ic_has_no_distance_loss_and_non_ic_routes_by_true_type():
    output = ModelOutput(
        type_logits=torch.randn(3, 5, requires_grad=True),
        distance_logits=tuple(
            torch.randn(3, 30, requires_grad=True) for _ in range(4)
        ),
        features=torch.randn(3, 8),
    )
    batch = {
        "type_label": torch.tensor([0, 1, 4]),
        "distance_bin": torch.tensor([-1, 3, 9]),
    }
    losses = compute_joint_loss(output, batch, distance_weight=0.5)
    losses.total.backward()

    assert losses.distance_count == 2
    assert output.distance_logits[0].grad[1].abs().sum() > 0
    assert output.distance_logits[3].grad[2].abs().sum() > 0
    for head in output.distance_logits:
        if head.grad is not None:
            assert head.grad[0].abs().sum() == 0
~~~

Add tests that invalid IC distance labels fail, missing non-IC distance labels fail, and distance_weight=0.5 scales only the distance component.

- [ ] **Step 2: Write failing end-to-end evaluation tests**

Construct records where oracle routing is correct but the predicted type is IC or the wrong non-IC type. Assert:

~~~python
assert metrics["type_macro_f1"] < 1.0
assert metrics["distance_coverage"] == 0.5
assert metrics["distance_within_200"] == 0.5
assert metrics["mean_non_ic_within_200"] == 0.5
assert selection_score(metrics) == pytest.approx(
    0.5 * metrics["type_macro_f1"]
    + 0.5 * metrics["mean_non_ic_within_200"]
)
~~~

Missing any of type indices 1..4 in validation must raise ValueError containing "all four non-IC types".

- [ ] **Step 3: Run tests and confirm RED**

~~~powershell
python -m pytest -q tests/test_training.py
~~~

- [ ] **Step 4: Implement interval-aware routed loss**

For each non-IC type_index in 1..4, select expert type_index - 1. Use categorical CE plus ordered CDF error:

~~~python
probabilities = logits.softmax(dim=1)
predicted_cdf = probabilities.cumsum(dim=1)
target_cdf = (
    torch.arange(30, device=logits.device)[None, :]
    >= targets[:, None]
).to(logits.dtype)
ordered = torch.abs(predicted_cdf - target_cdf).mean()
distance_loss = cross_entropy + 0.2 * ordered
total = type_cross_entropy + distance_weight * distance_loss
~~~

Return detached scalar components plus the differentiable total in a JointLoss dataclass.

- [ ] **Step 5: Implement piece-level routed metrics**

evaluate_loader() computes five-class confusion, accuracy, macro precision/recall/F1, and per-type metrics. For each true non-IC piece:

- predicted IC means no distance and counts as a within-200 failure;
- predicted wrong non-IC type routes through that predicted expert and counts the resulting distance normally;
- predicted correct non-IC type uses its matching expert.

Report distance coverage, exact accuracy, MAE km over covered rows, overall within-100/200 with uncovered rows counted false, and per-true-type within-200. mean_non_ic_within_200 is the unweighted mean of all four required true types.

- [ ] **Step 6: Implement one epoch and early stopping state**

train_epoch() performs one forward per batch, AMP autocast, GradScaler, gradient zero/backward/step, and aggregate logging. EarlyStoppingState stores best_score, best_epoch, wait, and best_state CPU tensors. update() resets wait only when score improves by more than 1e-6.

- [ ] **Step 7: Run focused tests and commit**

~~~powershell
python -m pytest -q tests/test_training.py tests/test_models.py
git add training.py evaluation.py tests\test_training.py
git commit -m "Add five-class joint training metrics"
~~~

---

### Task 6: Single-Split Training CLI and Exact Resume

**Files:**
- Replace: train.py
- Modify: training.py
- Modify: checkpoints.py
- Extend: tests/test_training.py

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: train.main(), run_training(), save_last_state(), load_last_state().

- [ ] **Step 1: Write failing CLI-default tests**

Extend tests/test_training.py:

~~~python
import pytest

import train


def test_training_defaults_match_five_class_contract():
    args = train.build_parser().parse_args([])
    assert args.task_data == r"..\train_data"
    assert args.output == r".\weights\five_class"
    assert args.epochs == 50
    assert args.batch_size == 256
    assert args.patience == 10
    assert args.samples_per_epoch == 120000
    assert args.ic_fraction == 0.60
    assert args.distance_weight == 0.5
    assert not hasattr(args, "init_model")


def test_old_cv_options_are_rejected():
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--resume_cv"])
~~~

- [ ] **Step 2: Write failing exact-resume tests**

Use a tiny model and state file. Assert that resume restores epoch, optimizer, scheduler, scaler, best state, early-stop wait, torch RNG, NumPy RNG, split hash, and config hash. Changing ic_fraction, model base width, or split seed must raise ValueError containing "resume configuration mismatch".

- [ ] **Step 3: Run focused tests and confirm RED**

~~~powershell
python -m pytest -q tests/test_training.py
~~~

- [ ] **Step 4: Implement the compact parser**

Expose these defaults:

~~~text
--task_data ..\train_data
--output .\weights\five_class
--epochs 50
--batch_size 256
--patience 10
--samples_per_epoch 120000
--num_workers 2
--seed 42
--ic_fraction 0.60
--distance_weight 0.5
--base_channels 64
--lr 0.0003
--weight_decay 0.0005
--resume
--no_amp
~~~

There is no init-model, CV, OOF, release-gate, rejection, baseline, or promotion argument.

- [ ] **Step 5: Implement run_training()**

The orchestration order is fixed:

~~~python
table, diagnostics = build_piece_table(args.task_data)
assignment = assign_piece_splits(table, seed=args.seed)
validate_piece_split(table, assignment)
write_split_json(output / "split.json", split_artifact(table, assignment))

train_positions = assignment.positions("train")
validation_positions = assignment.positions("validation")
test_positions = assignment.positions("test")

train_dataset = FiveClassDataset(table, train_positions, split="train", ...)
validation_dataset = FiveClassDataset(table, validation_positions, split="validation")
test_dataset = FiveClassDataset(table, test_positions, split="test")
sampler = FiveClassSampler(
    table,
    train_positions,
    num_samples=args.samples_per_epoch,
    seed=args.seed,
)
model = create_five_class_model(base_channels=args.base_channels)
~~~

Never load weights before optional exact same-run resume. Save last.pt atomically after every epoch. When validation improves, update the in-memory best state. After stopping, save model.pt through save_model_checkpoint(), evaluate test once, and write metrics.json atomically.

- [ ] **Step 6: Add a bounded synthetic smoke test**

Generate five synthetic class trees with at least three pieces in every required stratum. Run:

~~~python
result = train.main([
    "--task_data", str(root),
    "--output", str(output),
    "--epochs", "1",
    "--samples_per_epoch", "20",
    "--batch_size", "5",
    "--num_workers", "0",
    "--base_channels", "8",
    "--no_amp",
])
assert (output / "model.pt").is_file()
assert (output / "last.pt").is_file()
assert (output / "metrics.json").is_file()
assert (output / "split.json").is_file()
~~~

- [ ] **Step 7: Run focused tests and CLI help**

~~~powershell
python -m pytest -q tests/test_training.py
python train.py --help
python -m compileall -q train.py training.py evaluation.py checkpoints.py
~~~

- [ ] **Step 8: Commit**

~~~powershell
git add train.py training.py checkpoints.py tests\test_training.py
git commit -m "Add simplified five-class trainer"
~~~

---

### Task 7: Byte-Preserving Two-Schema Inference

**Files:**
- Replace: classify.py
- Create: tests/test_classify.py

**Interfaces:**
- Consumes: data.lig, data.preprocess, models, and checkpoints.
- Produces: classify.main(), classify_directory(), Prediction, PredictionCsvWriter.

- [ ] **Step 1: Write failing direct-IC and distance-routing tests**

Create tests/test_classify.py with deterministic logits:

~~~python
def test_direct_argmax_outputs_ic_without_threshold():
    prediction = decode_new_prediction(
        type_logits=torch.tensor([9.0, 1.0, 0.0, 0.0, 0.0]),
        distance_logits=[torch.zeros(30) for _ in range(4)],
    )
    assert prediction.final_type == "IC"
    assert prediction.distance_bin is None
    assert prediction.output_class == "IC"


def test_non_ic_uses_matching_distance_expert():
    heads = [torch.zeros(30) for _ in range(4)]
    heads[1][5] = 10.0
    prediction = decode_new_prediction(
        type_logits=torch.tensor([0.0, 0.0, 9.0, 0.0, 0.0]),
        distance_logits=heads,
    )
    assert prediction.final_type == "NNBE"
    assert prediction.distance_bin == 5
    assert prediction.output_class == "NNBE_500-600km"
~~~

Add the equivalent legacy checkpoint decode test.

- [ ] **Step 2: Write failing CSV and raw-byte preservation test**

Classify a synthetic source containing distinct nontrivial piece header bytes. Assert:

~~~python
assert output_piece_bytes == source_piece_bytes
assert row["source_path"] == "incoming/source.lig"
assert row["piece_index"] == "0"
assert row["prob_IC"] != ""
assert row["prob_NNBE"] != ""
assert row["checkpoint_schema"] == "five_class_v1"
assert row["model_sha256"] == expected_hash
~~~

Also assert no output LIG contains more than 512 pieces and a batch-size spy never observes more than the requested batch size.

- [ ] **Step 3: Run tests and confirm RED**

~~~powershell
python -m pytest -q tests/test_classify.py
~~~

- [ ] **Step 4: Implement one normalized prediction contract**

~~~python
@dataclass(frozen=True)
class Prediction:
    final_type: str
    output_class: str
    type_probabilities: tuple[float, float, float, float, float]
    type_confidence: float
    distance_bin: int | None
    expected_distance_km: float | None
    distance_confidence: float | None
~~~

New checkpoints receive signed local/global/daylight tensors. Legacy checkpoints receive the legacy 8000-point minmax view. Both adapters return Prediction. There is no uncertain folder, confidence threshold, rejection reason, support status, hybrid routing, or conditional-CV branch.

- [ ] **Step 5: Implement bounded raw-piece regrouping**

For each source batch, retain raw piece bytes and timestamps beside inference tensors. Buffer raw bytes by output_class, flush at 512 pieces, copy the source file header, and alter only the output file piece-count field. Write predictions.csv incrementally with:

~~~text
source_path,piece_index,piece_key,final_type,
prob_IC,prob_NCG,prob_NNBE,prob_PCG,prob_PNBE,type_confidence,
distance_bin,distance_low_km,distance_high_km,expected_distance_km,
distance_confidence,checkpoint_schema,model_sha256,output_file
~~~

- [ ] **Step 6: Implement the compact CLI**

Retain only:

~~~text
--input_dir
--output_dir
--model
--batch_size 256
--type_only
--device
~~~

Reject input_dir equal to or nested inside output_dir. Recursively discover .lig files in deterministic relative-path order and exclude no data based on filename date.

- [ ] **Step 7: Run tests and commit**

~~~powershell
python -m pytest -q tests/test_classify.py tests/test_checkpoints.py tests/test_lig.py
python classify.py --help
git add classify.py tests\test_classify.py
git commit -m "Add byte-preserving five-class inference"
~~~

---

### Task 8: Audit CLI, Repository Cleanup, and Documentation

**Files:**
- Replace: audit_data.py
- Create: tests/test_audit.py
- Create: README.md
- Replace: AGENTS.md
- Delete: benchmark.py
- Delete: conditional_pipeline.py
- Delete: cv_pipeline.py
- Delete: distance_metrics.py
- Delete: distance_ordinal.py
- Delete: open_set.py
- Delete: training_engine.py
- Delete: data/augmentation.py
- Delete: data/cross_validation.py
- Delete: data/distance_sampling.py
- Delete: data/group_split.py
- Delete: data/lig_parser.py
- Delete: data/oof_manifest.py
- Delete: data/preprocessing.py
- Delete: data/signal_context.py
- Delete: data/split_artifacts.py
- Delete: data/training_dataset.py
- Delete: data/training_manifest.py
- Delete: data/waveform_quality.py
- Delete: all superseded tests not listed in Locked File Map
- Delete: obsolete docs/superpowers files except the approved design and this plan
- Delete: CLAUDE.md
- Delete: _check_gates.py

**Interfaces:**
- Produces: audit_data.main(), audit_dataset(), audit_duplicate_waveforms().
- Finalizes: only the locked runtime/test map remains.

- [ ] **Step 1: Write failing audit tests**

Create tests/test_audit.py:

~~~python
def test_audit_reports_piece_split_and_training_prior(tmp_path):
    report = audit_dataset(tmp_path / "train_data", seed=42)
    assert report["split_schema"] == "piece_stratified_split_v1"
    assert report["split_counts"]["train"] > 0
    assert report["training_prior"] == {
        "IC": 0.60,
        "NCG": 0.10,
        "NNBE": 0.10,
        "PCG": 0.10,
        "PNBE": 0.10,
    }


def test_duplicate_audit_rejects_identical_bytes_across_splits(tmp_path):
    root = tmp_path / "train_data"
    source = root / "NCG" / "day" / "0-100km" / "duplicate.lig"
    source.parent.mkdir(parents=True)
    raw = make_piece(17, hour=8)
    write_source(source, [raw, raw, raw, raw])
    table, _ = build_piece_table(root)
    assignment = assign_piece_splits(table, seed=42)
    with pytest.raises(ValueError, match="duplicate waveform crosses partitions"):
        audit_duplicate_waveforms(table, assignment)
~~~

The normal audit test must monkeypatch the duplicate scanner and prove it is not called without --check_duplicates.

- [ ] **Step 2: Implement the compact audit command**

The CLI accepts:

~~~text
--task_data ..\train_data
--output
--seed 42
--check_duplicates
~~~

audit_dataset() builds the table and split, validates ownership, and reports files, pieces, type/day/bin counts, small strata, manifest hash, split hashes, and the fixed training prior. --check_duplicates hashes each complete raw piece with SHA-256, stores the first identity/partition per digest, and raises if a later identical piece belongs to another partition.

- [ ] **Step 3: Run all new tests before deleting old code**

~~~powershell
python -m pytest -q tests\test_lig.py tests\test_manifest.py tests\test_split.py tests\test_preprocess.py tests\test_dataset.py tests\test_sampling.py tests\test_models.py tests\test_checkpoints.py tests\test_training.py tests\test_classify.py tests\test_audit.py
~~~

Expected: all new tests pass.

- [ ] **Step 4: Delete superseded tracked code and tests**

Use apply_patch file deletions, not git reset or checkout. Delete every tracked runtime and test file listed above. Replace evaluation.py with the Task 5 version, which intentionally removes the user's approved obsolete release-threshold change.

Verify no runtime imports remain:

~~~powershell
rg -n "cv_pipeline|conditional_pipeline|open_set|oof_manifest|cross_validation|support_map|release_gate|four_class_rejection" train.py classify.py audit_data.py models.py training.py evaluation.py checkpoints.py data
~~~

Expected: no matches.

- [ ] **Step 5: Delete approved untracked files**

After resolving absolute paths inside the repository, remove only:

~~~powershell
Remove-Item -LiteralPath .\CLAUDE.md
Remove-Item -LiteralPath .\_check_gates.py
~~~

- [ ] **Step 6: Verify and clean ignored weights safely**

Run this safety check before deletion:

~~~powershell
$repo = (Resolve-Path .).Path
$weights = (Resolve-Path .\weights).Path
if (-not $weights.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "weights directory escaped repository"
}
$expected = "F3D74DE7F1E1AD5B7F3A796F3A145C58E6F9FC86730C45A84F74A634A2CF82EA"
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath .\weights\old\model.pt).Hash
if ($actual -ne $expected) { throw "retained legacy model hash mismatch" }
~~~

Then remove these exact repository-local directories if present:

~~~text
weights/conditional
weights/conditional_cv
weights/conditional_final
weights/cv_smoke
weights/four_class
weights/fresh
weights/new
weights/piece_time
weights/smoke
weights/smoke_guard
~~~

Use Resolve-Path on each existing target, require it to start with the verified weights path, then Remove-Item -LiteralPath <resolved> -Recurse. Delete weights/old/four_class_baseline.json. Finally assert the only files below weights are old/model.pt and old/metrics.json.

- [ ] **Step 7: Remove obsolete design history**

Retain only:

~~~text
docs/superpowers/specs/2026-07-17-five-class-project-simplification-design.md
docs/superpowers/plans/2026-07-17-five-class-project-simplification.md
~~~

Delete every other file below docs/superpowers with apply_patch. Remove empty directories.

- [ ] **Step 8: Write concise README and contributor guide**

README.md documents installation, the five-class data layout, piece-level 70/15/15 split, 60% IC sampling, one training command, resume, audit, inference, checkpoint schemas, and output CSV fields.

AGENTS.md documents the locked file map, synthetic-test-only rule, random initialization, data safety, and:

~~~powershell
python audit_data.py --task_data ..\train_data
python train.py --task_data ..\train_data --output .\weights\five_class
python train.py --task_data ..\train_data --output .\weights\five_class --resume .\weights\five_class\last.pt
python classify.py --input_dir <lig-dir> --output_dir .\classified --model .\weights\five_class\model.pt
python -m pytest -q
python -m compileall -q .
~~~

- [ ] **Step 9: Run focused cleanup validation and commit**

~~~powershell
python -m pytest -q
python -m compileall -q .
git diff --check
git status --short
~~~

Expected: only intentional tracked changes remain; no obsolete runtime imports or obsolete tests remain.

~~~powershell
git add -A
git commit -m "Remove obsolete CV and rejection pipeline"
~~~

Ignored weight deletions will not appear in the commit but must be reported.

---

### Task 9: Final Verification and Size Audit

**Files:**
- Modify only if verification exposes a reproducible defect in the new path.

**Interfaces:**
- Verifies every global constraint and produces the handoff evidence.

- [ ] **Step 1: Verify repository shape**

Run:

~~~powershell
rg --files | Sort-Object
Get-ChildItem -Recurse -File .\weights | Select-Object FullName, Length
~~~

Expected: runtime and tests match Locked File Map plus README.md, AGENTS.md, the approved spec, and this plan. weights contains only old/model.pt and old/metrics.json and is under 10 MB.

- [ ] **Step 2: Run the complete suite and compile**

~~~powershell
python -m pytest -q
python -m compileall -q .
git diff --check
~~~

Expected: all tests pass, compile emits no output, and diff check exits zero.

- [ ] **Step 3: Run CLI smoke checks**

~~~powershell
python train.py --help
python classify.py --help
python audit_data.py --help
~~~

Expected: all exit zero and expose no CV, OOF, rejection, baseline, promotion, warm-start, or support-map arguments.

- [ ] **Step 4: Verify the retained real legacy checkpoint read-only**

~~~powershell
python -c "from checkpoints import load_model_checkpoint, model_sha256; p=r'.\weights\old\model.pt'; c=load_model_checkpoint(p,'cpu'); assert c.schema=='legacy_five_class'; assert model_sha256(p)=='f3d74de7f1e1ad5b7f3a796f3a145c58e6f9fc86730c45a84f74a634a2cf82ea'"
~~~

Expected: exit zero with no model modification.

- [ ] **Step 5: Run bounded end-to-end synthetic smoke**

Run the one-epoch synthetic training fixture, load its five_class_v1 model, classify a synthetic multi-piece file, and assert:

~~~text
model.pt, last.pt, metrics.json, split.json exist
predictions.csv contains five probability columns
output piece count equals input piece count
every output raw piece matches exactly one input raw piece
no output file contains more than 512 pieces
~~~

Delete the temporary smoke directory after verifying its resolved path is inside the system temporary directory.

- [ ] **Step 6: Measure simplification**

Run:

~~~powershell
Get-ChildItem -File *.py, data\*.py | Get-Content | Measure-Object -Line
Get-ChildItem -File tests\*.py | Get-Content | Measure-Object -Line
rg -n "cv|oof|open.set|rejection|release.gate|support.map" --glob "*.py"
~~~

Expected: no single runtime file exceeds 600 lines, cv_pipeline.py and its 2164-line implementation are gone, and the obsolete-term search has no runtime matches except an explicit checkpoint rejection test message.

- [ ] **Step 7: Final Git and safety review**

~~~powershell
git status --short
git log --oneline -12
git diff HEAD^ --stat
~~~

Confirm no .lig files, model weights, generated classifications, credentials, external absolute paths, or training data were committed. Confirm ../train_data and external output directories were never written.

- [ ] **Step 8: Commit any verification-only correction**

Only if Step 1-7 required a code correction:

~~~powershell
git add <exact corrected source and test files>
git commit -m "Fix five-class verification findings"
~~~

Otherwise do not create an empty commit.
