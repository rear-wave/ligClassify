# Multi-Model Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one five-class type checkpoint plus four independent
class-specific distance checkpoints, then classify an inclusive date range into
type/distance directories with timestamp-derived LIG filenames.

**Architecture:** Keep every role checkpoint in the supported
`five_class_v1` schema. A strict `bundle.json` assigns the five checkpoints to
roles. Inference loads the bundle, runs type prediction first, routes non-IC
pieces to one matching distance checkpoint, and regroups raw bytes without
reconstruction.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, scikit-learn, tqdm, pytest.

## Global Constraints

- Runtime files remain limited to the locked repository map and at most 600
  lines each.
- Labels remain exactly `IC`, `NCG`, `NNBE`, `PCG`, and `PNBE`.
- All five roles use random initialization and share no trainable parameters.
- Splits remain deterministic, waveform-piece-level, and mutually exclusive.
- Every complete type batch uses a fixed 20/20/20/20/20 prior.
- Distance experts use only their assigned non-IC type and balance observed
  distance-bin/daylight cells.
- Each output LIG file contains at most 512 complete raw pieces.
- Output filenames use the first output piece timestamp as
  `GZ_YYYYMMDDHHMMSS.lig`; collisions use `_002`, `_003`, and so on.
- `../train_data` and external classification outputs are never modified.
- Existing `--input_dir --model` inference remains compatible.

---

### Task 1: Add deterministic class-specific distance sampling

**Files:**
- Modify: `data/sampling.py`
- Modify: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `PieceTable`, piece positions, one type index in `1..4`.
- Produces:
  `DistanceExpertSampler(table, positions, type_index, num_samples, seed)`.

- [ ] **Step 1: Write failing sampling tests**

```python
def test_distance_expert_sampler_draws_only_requested_type():
    sampler = DistanceExpertSampler(
        table, range(len(table)), type_index=1, num_samples=120, seed=7
    )
    draws = list(sampler)
    assert len(draws) == 120
    assert {int(table.type_index[item.position]) for item in draws} == {1}


def test_distance_expert_sampler_balances_observed_daylight_distance_cells():
    sampler = DistanceExpertSampler(
        table, range(len(table)), type_index=2, num_samples=120, seed=7
    )
    draws = list(sampler)
    cells = [
        (
            bool(table.daylight[item.position]),
            int(table.distance_bin[item.position]),
        )
        for item in draws
    ]
    counts = Counter(cells)
    assert max(counts.values()) - min(counts.values()) <= 1
```

- [ ] **Step 2: Verify the tests fail**

Run:

```cmd
python -m pytest -q tests/test_sampling.py
```

Expected: import failure because `DistanceExpertSampler` does not exist.

- [ ] **Step 3: Implement the sampler**

Add a sampler that validates `type_index`, filters the supplied positions once,
groups them by `(daylight, distance_bin)`, allocates equal largest-remainder
quotas, samples with replacement, assigns deterministic augmentation seeds, and
shuffles deterministically:

```python
class DistanceExpertSampler:
    def __init__(
        self,
        table: PieceTable,
        positions: Iterable[int],
        type_index: int,
        num_samples: int,
        seed: int,
    ) -> None:
        if type(type_index) is not int or not 1 <= type_index <= 4:
            raise ValueError("type_index must be in 1..4")
        self.table = table
        self.type_index = type_index
        self.positions = tuple(
            int(position)
            for position in positions
            if int(table.type_index[position]) == type_index
        )
        if not self.positions:
            raise ValueError(
                f"no pieces available for distance type {TYPE_NAMES[type_index]}"
            )
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 1
```

Reuse `_largest_remainders` and `_augmentation_seed`; do not duplicate their
logic.

- [ ] **Step 4: Run sampling tests**

Run:

```cmd
python -m pytest -q tests/test_sampling.py
```

Expected: all sampling tests pass.

- [ ] **Step 5: Commit**

```cmd
git add data\sampling.py tests\test_sampling.py
git commit -m "Add class-specific distance sampling"
```

---

### Task 2: Add isolated type and distance training/evaluation paths

**Files:**
- Modify: `models.py`
- Modify: `training.py`
- Modify: `evaluation.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_training.py`

**Interfaces:**
- Produces:
  `FiveClassNet.set_training_role(role: str) -> int | None`
- Produces:
  `train_role_epoch(model, loader, optimizer, device, role, scaler, amp)`
- Produces:
  `evaluate_type_role(model, loader, device)`
- Produces:
  `evaluate_distance_role(model, loader, device, type_index)`

- [ ] **Step 1: Write failing role-isolation tests**

```python
def test_type_role_enables_only_type_parameters():
    model = create_five_class_model(base_channels=8)
    assert model.set_training_role("type") is None
    assert all(p.requires_grad for p in model.type_encoder.parameters())
    assert not any(p.requires_grad for p in model.distance_encoder.parameters())


@pytest.mark.parametrize(("role", "index"), [
    ("NCG", 0), ("NNBE", 1), ("PCG", 2), ("PNBE", 3),
])
def test_distance_role_enables_independent_distance_network(role, index):
    model = create_five_class_model(base_channels=8)
    assert model.set_training_role(role) == index
    assert not any(p.requires_grad for p in model.type_encoder.parameters())
    assert all(p.requires_grad for p in model.distance_encoder.parameters())
```

Also add tests proving `train_role_epoch(..., role="type")` never calls
`forward_distance`, while `role="NCG"` never calls `forward_type` and updates
only distance parameters.

- [ ] **Step 2: Verify role tests fail**

Run:

```cmd
python -m pytest -q tests/test_models.py tests/test_training.py
```

Expected: failures for missing role APIs.

- [ ] **Step 3: Implement role selection**

Replace the joint-stage public entry point with:

```python
TRAINING_ROLES = ("type", "NCG", "NNBE", "PCG", "PNBE")

def set_training_role(self, role: str) -> int | None:
    if role not in TRAINING_ROLES:
        raise ValueError(f"role must be one of {TRAINING_ROLES}")
    type_enabled = role == "type"
    for parameter in self.type_encoder.parameters():
        parameter.requires_grad_(type_enabled)
    for parameter in self.type_head.parameters():
        parameter.requires_grad_(type_enabled)
    for parameter in self.distance_encoder.parameters():
        parameter.requires_grad_(not type_enabled)
    for parameter in self.distance_matcher.parameters():
        parameter.requires_grad_(not type_enabled)
    return None if type_enabled else DISTANCE_NAMES.index(role)
```

Each training run creates a fresh `FiveClassNet`; therefore distance encoders
are independent across the four saved role checkpoints.

- [ ] **Step 4: Implement role-specific losses and epochs**

Use cross-entropy for type. For one distance expert, retain the current ordered
loss:

```python
def distance_expert_loss(logits, targets):
    categorical = F.cross_entropy(logits, targets)
    probabilities = logits.softmax(dim=1)
    predicted_cdf = probabilities.cumsum(dim=1)
    target_cdf = (
        torch.arange(DISTANCE_BIN_COUNT, device=logits.device)[None, :]
        >= targets[:, None]
    ).to(logits.dtype)
    return categorical + 0.2 * torch.abs(predicted_cdf - target_cdf).mean()
```

`train_role_epoch` calls `forward_type` for `type`; otherwise it calls
`forward_distance`, selects exactly one head, validates that every label equals
the assigned type, and reports rolling `loss`, `accuracy`, and sample count.

- [ ] **Step 5: Implement role-specific evaluation**

`evaluate_type_role` returns piece count, accuracy, macro precision, macro
recall, and macro-F1. `evaluate_distance_role` returns exact accuracy,
within-100, within-200, MAE in bins/km, and piece count. It rejects mixed types
or invalid distance labels.

- [ ] **Step 6: Run role tests and line-limit test**

Run:

```cmd
python -m pytest -q tests/test_models.py tests/test_training.py
```

Expected: pass, including the maximum 600-line runtime-module assertion.

- [ ] **Step 7: Commit**

```cmd
git add models.py training.py evaluation.py tests\test_models.py tests\test_training.py
git commit -m "Separate type and distance role training"
```

---

### Task 3: Orchestrate five independent role runs and write a strict bundle

**Files:**
- Modify: `train.py`
- Modify: `checkpoints.py`
- Modify: `tests/test_training.py`
- Modify: `tests/test_checkpoints.py`

**Interfaces:**
- Produces CLI `--stage all|type|NCG|NNBE|PCG|PNBE`.
- Produces `save_model_bundle(model_dir, role_paths, preprocess_config)`.
- Produces `load_model_bundle(model_dir, device) -> LoadedModelBundle`.

- [ ] **Step 1: Write failing bundle and CLI tests**

```python
def test_training_defaults_to_all_five_roles():
    args = train.build_parser().parse_args([])
    assert args.stage == "all"
    assert args.type_samples_per_epoch == 120000
    assert args.distance_samples_per_epoch == 60000


def test_bundle_rejects_missing_role(tmp_path):
    with pytest.raises(ValueError, match="missing.*PNBE"):
        load_model_bundle(tmp_path, "cpu")


def test_bundle_rejects_hash_mismatch(valid_bundle):
    valid_bundle.joinpath("NCG", "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_model_bundle(valid_bundle, "cpu")
```

Add a one-epoch synthetic smoke test that writes exactly:

```text
bundle.json
split.json
type/{model.pt,last.pt,metrics.json}
NCG/{model.pt,last.pt,metrics.json}
NNBE/{model.pt,last.pt,metrics.json}
PCG/{model.pt,last.pt,metrics.json}
PNBE/{model.pt,last.pt,metrics.json}
```

- [ ] **Step 2: Verify tests fail**

Run:

```cmd
python -m pytest -q tests/test_checkpoints.py tests/test_training.py
```

Expected: missing CLI and bundle APIs.

- [ ] **Step 3: Implement strict bundle loading**

Add:

```python
MODEL_BUNDLE_SCHEMA = "five_class_model_bundle_v1"
BUNDLE_ROLES = ("type", "NCG", "NNBE", "PCG", "PNBE")

@dataclass(frozen=True)
class LoadedModelBundle:
    type_checkpoint: LoadedCheckpoint
    distance_checkpoints: tuple[LoadedCheckpoint, ...]
    hashes: dict[str, str]
```

`bundle.json` contains only relative normalized role paths, SHA-256 values,
labels, bins, and preprocessing configuration. Reject absolute paths, `..`,
missing/extra roles, non-`five_class_v1` checkpoints, inconsistent
preprocessing, labels/bins, and hash mismatches.

- [ ] **Step 4: Replace joint CLI controls with role controls**

Use:

```python
parser.add_argument(
    "--stage",
    choices=("all", "type", "NCG", "NNBE", "PCG", "PNBE"),
    default="all",
)
parser.add_argument("--type_samples_per_epoch", type=int, default=120000)
parser.add_argument("--distance_samples_per_epoch", type=int, default=60000)
```

Remove `--distance_weight`. Permit `--resume` only when exactly one stage is
selected. Build the piece table once with distance labels required, create one
split artifact, and train requested roles sequentially. Seed each role with
`seed + role_index` so reruns are deterministic but independently initialized.

- [ ] **Step 5: Implement per-role fitting**

For `type`, use `FiveClassSampler` and `evaluate_type_role`; for a distance role,
filter dataset positions to that type, use `DistanceExpertSampler`, and call
`evaluate_distance_role`. Instantiate a new model and optimizer for every role.
Save every checkpoint with the existing `save_model_checkpoint` and include
`training_config["role"]`.

Use type macro-F1 as the type score. Use:

```python
distance_score = metrics["within_200"] - 1e-3 * metrics["mae_bins"]
```

for distance early stopping. Write `bundle.json` only after all five validated
role checkpoints exist.

- [ ] **Step 6: Run training and checkpoint tests**

Run:

```cmd
python -m pytest -q tests/test_checkpoints.py tests/test_training.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```cmd
git add train.py checkpoints.py tests\test_training.py tests\test_checkpoints.py
git commit -m "Train and validate five-model bundles"
```

---

### Task 4: Add inclusive date-range discovery and expert routing

**Files:**
- Modify: `classify.py`
- Modify: `tests/test_classify.py`

**Interfaces:**
- Produces:
  `discover_date_inputs(input_root, start_date, end_date) -> list[Path]`
- Extends `classify_directory` with an optional loaded model bundle.
- Produces CLI `--input_root`, `--start_date`, `--end_date`, `--model_dir`.

- [ ] **Step 1: Write failing date discovery tests**

```python
def test_date_range_is_inclusive_and_excludes_index(tmp_path):
    for name in (
        "GZ_20160702", "GZ_20160702Index",
        "GZ_20160703", "GZ_20160703Index",
    ):
        (tmp_path / name).mkdir()
    assert discover_date_inputs(tmp_path, "20160702", "20160703") == [
        tmp_path / "GZ_20160702",
        tmp_path / "GZ_20160703",
    ]


def test_date_range_reports_all_missing_days(tmp_path):
    (tmp_path / "GZ_20160702").mkdir()
    with pytest.raises(ValueError, match="20160703.*20160704"):
        discover_date_inputs(tmp_path, "20160702", "20160704")
```

Add parser tests that require all four range-mode options together and reject
mixing range mode with `--input_dir --model`.

- [ ] **Step 2: Verify date tests fail**

Run:

```cmd
python -m pytest -q tests/test_classify.py
```

Expected: missing date discovery and range CLI support.

- [ ] **Step 3: Implement preflight date discovery**

Parse dates with `datetime.strptime(value, "%Y%m%d")`, reject reversed ranges,
build every inclusive day, require exact `GZ_YYYYMMDD` directories, and return
them in date order. Never glob `*Index`.

- [ ] **Step 4: Implement bundle prediction**

Add `predict_bundle_batch(bundle, waveforms, timestamps, device, type_only)`.
Run the type checkpoint once. If not type-only, group predicted rows by type and
run only the corresponding distance checkpoint's selected matcher. Return
predictions in original input order.

- [ ] **Step 5: Run classification tests**

Run:

```cmd
python -m pytest -q tests/test_classify.py
```

Expected: date and routing tests pass; old single-model tests still pass.

- [ ] **Step 6: Commit**

```cmd
git add classify.py tests\test_classify.py
git commit -m "Add date-range bundle inference"
```

---

### Task 5: Regroup by type/distance with first-piece timestamp filenames

**Files:**
- Modify: `classify.py`
- Modify: `tests/test_classify.py`

**Interfaces:**
- Changes `_OutputRegrouper.add` to consume the piece timestamp.
- Preserves `Prediction.output_class` as `IC` or
  `TYPE/LLLL-HHHHkm`.

- [ ] **Step 1: Write failing output-contract tests**

```python
def test_output_name_uses_first_piece_timestamp(tmp_path):
    # First two pieces enter the same output leaf.
    result = classify_fixture(
        tmp_path,
        timestamps=["2016-07-08T14:23:17", "2016-07-08T14:24:18"],
    )
    assert result.output_files == [
        "NCG/0300-0400km/GZ_20160708142317.lig"
    ]


def test_output_rollover_uses_next_files_first_timestamp(tmp_path):
    result = classify_fixture(tmp_path, piece_count=513)
    assert result.output_files[1].name.startswith(
        result.timestamps[512].strftime("GZ_%Y%m%d%H%M%S")
    )
```

Also test deterministic `_002` collision suffixes and byte equality between
every input piece and its CSV-addressed output piece.

- [ ] **Step 2: Verify output tests fail**

Run:

```cmd
python -m pytest -q tests/test_classify.py
```

Expected: current `TYPE_000001.lig` naming fails.

- [ ] **Step 3: Implement timestamp naming**

When a leaf needs a new buffer, name it from the incoming first timestamp:

```python
stem = timestamp.strftime("GZ_%Y%m%d%H%M%S")
filename = f"{stem}.lig"
```

Track reserved relative paths and existing destinations. Add `_002`, `_003`,
and so on before `.lig` until unique. Pass the corresponding timestamp from
`_source_batches` into `regrouper.add`.

- [ ] **Step 4: Implement nested output leaves**

Use `Prediction.output_class == "IC"` for IC. For non-IC use:

```python
f"{type_name}/{distance_low:04d}-{distance_low + 100:04d}km"
```

CSV `output_file` stores the normalized relative path.

- [ ] **Step 5: Run classification tests**

Run:

```cmd
python -m pytest -q tests/test_classify.py
```

Expected: all old byte-preservation tests and new naming tests pass.

- [ ] **Step 6: Commit**

```cmd
git add classify.py tests\test_classify.py
git commit -m "Write timestamp-named hierarchical LIG outputs"
```

---

### Task 6: Documentation, complete verification, and smoke commands

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: all locked tests

**Interfaces:**
- Documents one full training command, one role-resume command, date-range
  inference, and retained single-directory inference.

- [ ] **Step 1: Update usage documentation**

Document:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --epochs 50 --patience 10 --batch_size 60 --num_workers 0
```

Single role retraining/resume:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --stage NCG --resume .\weights\multi_model\NCG\last.pt
```

Date-range inference:

```cmd
python -u classify.py --input_root "D:/" --start_date 20160702 --end_date 20160709 --output_dir .\classified\2016.0702-2016.0709 --model_dir .\weights\multi_model --batch_size 256 --device cuda
```

- [ ] **Step 2: Compile all runtime modules**

Run:

```cmd
python -m compileall -q .
```

Expected: exit code 0.

- [ ] **Step 3: Run the full synthetic suite**

Run:

```cmd
python -m pytest -q
```

Expected: all tests pass with no real training data access.

- [ ] **Step 4: Run a CPU synthetic smoke training**

Run the existing test fixture through one epoch for all five roles. Confirm
`bundle.json` loads and every role checkpoint uses `five_class_v1`.

- [ ] **Step 5: Inspect final diff and repository safety**

Run:

```cmd
git diff --check
git status --short
```

Expected: no whitespace errors; no `.lig`, `.pt`, external output, or
`../train_data` changes.

- [ ] **Step 6: Commit documentation**

```cmd
git add README.md AGENTS.md
git commit -m "Document multi-model training and range inference"
```
