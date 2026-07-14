# Four-Class Training with IC Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train only on NCG, NNBE, PCG, and PNBE, then reject insufficiently reliable predictions to the non-research output label `IC`.

**Architecture:** Use contiguous model labels `0..3` for the four researched types and keep each distance head aligned to the same index. Add a focused open-set module that fits type-logit temperature, per-type feature references, and per-type probability/margin/distance thresholds using validation data only. Preserve legacy five-class checkpoint inference while storing the new schema and rejection policy explicitly in four-class checkpoints.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn, pytest

## Global Constraints

- IC files never contribute to model weights, feature references, or primary threshold selection.
- Four-class order is exactly `NCG`, `NNBE`, `PCG`, `PNBE`; rejected output is exactly `IC`.
- Default per-type precision floor is `0.85`; default per-type recall floor is `0.70`.
- The locked latest-date test is evaluated once and never selects checkpoints or thresholds.
- Existing five-class checkpoints remain loadable by `classify.py` and are never overwritten.
- Raw `.lig` piece bytes must be preserved exactly.

---

## File Map

- Create `open_set.py`: type temperature fitting, feature references, rejection-policy fitting, and rejection decoding.
- Modify `models.py`: configurable type count and a stable `extract_type_features()` interface.
- Modify `data/training_manifest.py`: determine distance availability from the type name instead of assuming index zero is IC.
- Modify `data/distance_sampling.py`: support balanced four-class type sampling without an IC prior.
- Modify `train.py`: four-class schema, training/evaluation indices, checkpoint selection, calibration, metadata, and release gates.
- Modify `classify.py`: schema-aware legacy/four-class loading and rejection audit output.
- Modify focused tests under `tests/`; never add real waveform fixtures.

### Task 1: Four-Class Data Schema and Manifest

**Files:**
- Modify: `data/training_manifest.py`
- Modify: `train.py`
- Test: `tests/test_training_manifest.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `RESEARCH_TYPE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]` and manifest entries whose `type_idx` is contiguous `0..3`.
- Consumes: existing `build_manifest(data_dir, type_names)` and `parse_distance_bin(filepath)`.

- [ ] **Step 1: Write failing tests for IC exclusion and index-independent distance parsing**

```python
def test_four_class_manifest_excludes_ic_and_keeps_ncg_distance(tmp_path, monkeypatch):
    for name in ["IC", "NCG", "NNBE", "PCG", "PNBE"]:
        (tmp_path / name).mkdir()
    files = [tmp_path / "IC" / "ic.lig", tmp_path / "NCG" / "NCG_500.lig"]
    for path in files:
        path.write_bytes(b"x")
    monkeypatch.setattr(training_manifest, "discover_lig_files", lambda root: [str(p) for p in files if root in str(p)])
    monkeypatch.setattr(training_manifest, "inspect_lig_file", lambda path: {"valid": True, "n_pieces": 1, "timestamp": datetime(2019, 1, 1)})

    entries, _ = training_manifest.build_manifest(
        str(tmp_path), ["NCG", "NNBE", "PCG", "PNBE"]
    )

    assert [(item.type_idx, item.dist_bin) for item in entries] == [(0, 5)]
```

```python
def test_research_type_order_is_stable():
    assert train.RESEARCH_TYPE_NAMES == ["NCG", "NNBE", "PCG", "PNBE"]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest -q tests/test_training_manifest.py tests/test_train.py`

Expected: FAIL because type index zero is currently treated as IC and `RESEARCH_TYPE_NAMES` is absent.

- [ ] **Step 3: Make distance parsing depend on the type name**

```python
# data/training_manifest.py, inside build_manifest
is_ic = type_name.upper() == "IC"
dist_bin = -1 if is_ic else parse_distance_bin(filepath)
```

```python
# train.py
RESEARCH_TYPE_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]
DIST_NAMES = list(RESEARCH_TYPE_NAMES)
```

Call `build_manifest(args.task_data, RESEARCH_TYPE_NAMES)` and update coverage validation to iterate every requested type that has distance labels rather than slicing `type_names[1:]`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q tests/test_training_manifest.py tests/test_train.py`

Expected: PASS.

- [ ] **Step 5: Commit the schema change**

```powershell
git add data/training_manifest.py train.py tests/test_training_manifest.py tests/test_train.py
git commit -m "Use four-class training schema"
```

### Task 2: Four-Class Model and Feature Interface

**Files:**
- Modify: `models.py`
- Test: `tests/test_models.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Produces: `create_mtl_model(..., num_types: int = 5)` and `extract_type_features(x) -> Tensor[B, 128]`.
- Consumes: four-class checkpoint metadata `type_names`.

- [ ] **Step 1: Write failing model-interface tests**

```python
@pytest.mark.parametrize("architecture", ["mtl_resnet", "ordinal_v2"])
def test_four_class_model_exposes_type_features(architecture):
    model = create_mtl_model(architecture=architecture, base_channels=8, num_types=4)
    x = torch.randn(2, 1, 128)
    features = model.extract_type_features(x)
    logits = model.forward_type(x)
    assert features.shape == (2, 16)
    assert logits.shape == (2, 4)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest -q tests/test_models.py::test_four_class_model_exposes_type_features`

Expected: FAIL because the factory has no `num_types` argument or feature interface.

- [ ] **Step 3: Add one feature interface to both architectures**

```python
def extract_type_features(self, x):
    return self.gap(self._encode_map(x)).squeeze(-1)

def forward_type(self, x):
    return self.type_head(self.extract_type_features(x))
```

For `MultiTaskResNet`, implement `extract_type_features()` by returning `_encode(x)`. Pass `num_types` from `create_mtl_model()` into the selected constructor. Do not change distance-head count.

- [ ] **Step 4: Verify model and legacy bypass tests**

Run: `python -m pytest -q tests/test_models.py tests/test_classify.py::test_forward_type_bypasses_ordinal_distance_projection`

Expected: PASS.

- [ ] **Step 5: Commit the model interface**

```powershell
git add models.py tests/test_models.py tests/test_classify.py
git commit -m "Expose four-class type features"
```

### Task 3: Balanced Four-Class Type and Distance Training

**Files:**
- Modify: `data/distance_sampling.py`
- Modify: `train.py`
- Test: `tests/test_distance_sampling.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `BalancedTypeSampler(type_labels, num_samples, seed=42)` and training routes with type/head indices `0..3`.
- Consumes: contiguous four-class manifest labels from Task 1.

- [ ] **Step 1: Write failing balance and routing tests**

```python
def test_balanced_type_sampler_allocates_equal_counts():
    labels = np.repeat(np.arange(4), [50, 40, 30, 20])
    sampler = BalancedTypeSampler(labels, num_samples=80, seed=7)
    sampled = labels[list(iter(sampler))]
    assert np.bincount(sampled, minlength=4).tolist() == [20, 20, 20, 20]
```

```python
def test_distance_loss_routes_zero_based_four_class_heads():
    labels = torch.tensor([0, 1, 2, 3])
    distances = torch.tensor([1, 2, 3, 4])
    logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    loss, _ = train.compute_distance_head_loss(logits, labels, distances)
    loss.backward()
    assert all(head.grad[index].abs().sum() > 0 for index, head in enumerate(logits))
```

- [ ] **Step 2: Verify focused failures**

Run: `python -m pytest -q tests/test_distance_sampling.py tests/test_train.py`

Expected: FAIL because the balanced sampler does not exist and routing assumes labels `1..4`.

- [ ] **Step 3: Implement equal allocation without replacement**

```python
class BalancedTypeSampler(Sampler):
    def __init__(self, type_labels, num_samples, seed=42):
        self.type_labels = np.asarray(type_labels)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        classes = sorted(np.unique(self.type_labels).tolist())
        if classes != list(range(4)) or self.num_samples % 4:
            raise ValueError("four classes and a sample count divisible by four are required")
        per_class = self.num_samples // 4
        selected = []
        for value in classes:
            pool = np.flatnonzero(self.type_labels == value)
            if len(pool) < per_class:
                raise ValueError(f"type {value} has only {len(pool)} pieces")
            selected.extend(rng.choice(pool, per_class, replace=False).tolist())
        rng.shuffle(selected)
        return iter(selected)
```

Change all type/head loops and routing from `range(1, 5)` plus `type - 1` to `range(4)` plus direct indexing. Remove `--target_ic_fraction`; add `--type_samples_per_epoch` help stating that the total is equally divided across four types.

- [ ] **Step 4: Verify samplers and training helpers**

Run: `python -m pytest -q tests/test_distance_sampling.py tests/test_train.py`

Expected: PASS.

- [ ] **Step 5: Commit balanced training**

```powershell
git add data/distance_sampling.py train.py tests/test_distance_sampling.py tests/test_train.py
git commit -m "Balance four-class training streams"
```

### Task 4: Open-Set Calibration and Rejection Policy

**Files:**
- Create: `open_set.py`
- Create: `tests/test_open_set.py`

**Interfaces:**
- Produces: `fit_feature_reference(features, labels, num_types=4) -> dict`, `fit_rejection_policy(logits, features, labels, reference, precision_floor=.85, recall_floor=.70) -> dict`, and `decode_with_rejection(logits, features, policy) -> dict[str, Tensor]`.
- Consumes: validation logits/features/labels only.

- [ ] **Step 1: Write failing tests for feature distance and rejection reasons**

```python
def test_decode_rejects_probability_margin_and_feature_distance():
    policy = {
        "version": 1, "temperature": 1.0,
        "centroids": [[0.0, 0.0]] * 4,
        "scales": [[1.0, 1.0]] * 4,
        "probability_thresholds": [0.50] * 4,
        "margin_thresholds": [0.20] * 4,
        "distance_thresholds": [2.0] * 4,
    }
    decoded = decode_with_rejection(
        torch.tensor([[2.0, 1.9, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [20.0, 20.0]]),
        policy,
    )
    assert decoded["accepted"].tolist() == [False, False]
    assert decoded["reason"] == ["low_margin", "feature_distance"]
```

```python
def test_policy_meets_known_class_precision_and_recall_floors():
    logits, features, labels = separable_validation_example()
    reference = fit_feature_reference(features, labels)
    policy = fit_rejection_policy(logits, features, labels, reference, .85, .70)
    assert all(value >= .85 for value in policy["validation_precision"])
    assert all(value >= .70 for value in policy["validation_recall"])
```

- [ ] **Step 2: Verify tests fail because the module is absent**

Run: `python -m pytest -q tests/test_open_set.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'open_set'`.

- [ ] **Step 3: Implement deterministic calibration primitives**

Use a single validation-fitted scalar temperature. Store per-type centroid and clamped diagonal standard deviation; compute root-mean-square standardized distance:

```python
def feature_distance(features, centroid, scale):
    z = (features - centroid) / scale.clamp_min(1e-6)
    return z.square().mean(dim=1).sqrt()
```

Search finite threshold grids derived from validation quantiles for probability, top-two margin, and feature distance. For each predicted type, retain candidates satisfying precision `>=0.85` and true-class recall `>=0.70`, then select maximum accepted count, breaking ties by macro-F1 and stricter feature distance. Raise `ValueError` naming the failing type when no feasible candidate exists; never silently relax a floor.

Return JSON-safe policy fields: `version`, `temperature`, `centroids`, `scales`, `probability_thresholds`, `margin_thresholds`, `distance_thresholds`, `validation_precision`, `validation_recall`, and `validation_coverage`.

- [ ] **Step 4: Verify all open-set tests**

Run: `python -m pytest -q tests/test_open_set.py`

Expected: PASS.

- [ ] **Step 5: Commit calibration**

```powershell
git add open_set.py tests/test_open_set.py
git commit -m "Add four-class rejection calibration"
```

### Task 5: Training Selection, Checkpoint Metadata, and Release Gates

**Files:**
- Modify: `train.py`
- Modify: `distance_ordinal.py`
- Test: `tests/test_train.py`
- Test: `tests/test_distance_ordinal.py`

**Interfaces:**
- Produces: four-class checkpoint field `type_rejection`, schema field `task_schema = "four_class_rejection_v1"`, and type-first selection/release reports.
- Consumes: calibration interfaces from Task 4.

- [ ] **Step 1: Write failing type-first selection and metadata tests**

```python
def test_selection_prefers_better_type_macro_f1_before_distance():
    better_type = {"type_macro_f1": .91, "type_min_recall": .80,
                   "type_min_precision": .86, "dist_macro_w2": .70,
                   "dist_macro_mae_km": 150}
    better_distance = {"type_macro_f1": .89, "type_min_recall": .80,
                       "type_min_precision": .86, "dist_macro_w2": .95,
                       "dist_macro_mae_km": 80}
    assert make_four_class_selection_key(better_type) > make_four_class_selection_key(better_distance)
```

```python
def test_four_class_checkpoint_records_rejection_schema():
    checkpoint = train.four_class_schema_metadata()
    assert checkpoint["type_names"] == ["NCG", "NNBE", "PCG", "PNBE"]
    assert checkpoint["task_schema"] == "four_class_rejection_v1"
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest -q tests/test_train.py tests/test_distance_ordinal.py`

Expected: FAIL because selection is distance-first and the schema is absent.

- [ ] **Step 3: Implement type-first checkpoint selection**

```python
def make_four_class_selection_key(metrics):
    return (
        float(metrics["type_macro_f1"]),
        float(metrics["type_min_recall"]),
        float(metrics["type_min_precision"]),
        float(metrics["dist_macro_w2"]),
        -float(metrics["dist_macro_mae_km"]),
    )
```

Add the schema helper used by checkpoint construction:

```python
def four_class_schema_metadata():
    return {
        "task_schema": "four_class_rejection_v1",
        "type_names": list(RESEARCH_TYPE_NAMES),
        "rejected_type_name": "IC",
    }
```

Collect validation type logits and encoder features from the selected state, fit `type_rejection`, and save it before any locked-test evaluation. Extend evaluation with per-type precision/recall/F1, confusion matrix, prediction shares, and year-grouped metrics. Reject release when any precision is below `--min_type_precision` (default `.85`), any recall is below `--min_type_recall` (default `.70`), or any candidate metric is below a supplied baseline metrics file. Do not auto-promote when no baseline file is supplied; write `candidate.pt` and report that comparison is pending.

- [ ] **Step 4: Verify selection and release tests**

Run: `python -m pytest -q tests/test_train.py tests/test_distance_ordinal.py`

Expected: PASS.

- [ ] **Step 5: Commit training integration**

```powershell
git add train.py distance_ordinal.py tests/test_train.py tests/test_distance_ordinal.py
git commit -m "Select four-class models by type quality"
```

### Task 6: Schema-Aware Classification and Audit Output

**Files:**
- Modify: `classify.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Produces: four-class routing to researched type or `IC`, with audit fields `raw_type`, `type_confidence`, `type_margin`, `feature_distance`, `rejection_reason`, and `final_type`.
- Consumes: `task_schema`, `type_names`, and `type_rejection` from Task 5; legacy checkpoints remain on the existing path.

- [ ] **Step 1: Write failing routing and compatibility tests**

```python
def test_four_class_rejection_routes_failed_waveform_to_ic():
    checkpoint = {
        "task_schema": "four_class_rejection_v1",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "type_rejection": {
            "version": 1, "temperature": 1.0,
            "centroids": [[0.0, 0.0]] * 4,
            "scales": [[1.0, 1.0]] * 4,
            "probability_thresholds": [0.5] * 4,
            "margin_thresholds": [0.1] * 4,
            "distance_thresholds": [2.0] * 4,
        },
    }
    decoded = classify.decode_four_class_prediction(
        torch.tensor([[4.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[99.0, 99.0]]),
        checkpoint,
    )
    assert decoded[0]["raw_type"] == "NCG"
    assert decoded[0]["final_type"] == "IC"
    assert decoded[0]["rejection_reason"] == "feature_distance"
```

```python
def test_legacy_five_class_checkpoint_uses_legacy_decoder():
    checkpoint = {"type_names": ["IC", "NCG", "NNBE", "PCG", "PNBE"]}
    assert classify.checkpoint_schema(checkpoint) == "legacy_five_class"
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest -q tests/test_classify.py`

Expected: FAIL because schema dispatch and the new audit fields are absent.

- [ ] **Step 3: Add explicit schema dispatch**

```python
def checkpoint_schema(checkpoint):
    schema = checkpoint.get("task_schema")
    if schema == "four_class_rejection_v1":
        return schema
    if checkpoint.get("type_names") == ["IC", "NCG", "NNBE", "PCG", "PNBE"]:
        return "legacy_five_class"
    raise ValueError("Unsupported checkpoint type schema")
```

Instantiate the model with `num_types=len(checkpoint["type_names"])`. In four-class mode, compute `extract_type_features()` once, call `decode_with_rejection()`, route rejected pieces into `IC`, and populate every audit field. Keep `--min_type_confidence` available only for legacy checkpoints; reject combining it with a four-class rejection checkpoint because calibrated per-type thresholds are authoritative.

- [ ] **Step 4: Verify classification and byte-preservation tests**

Run: `python -m pytest -q tests/test_classify.py`

Expected: PASS, including legacy decoding and raw-byte preservation.

- [ ] **Step 5: Commit classification integration**

```powershell
git add classify.py tests/test_classify.py
git commit -m "Route rejected four-class predictions to IC"
```

### Task 7: End-to-End Verification and Operator Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-14-four-class-rejection-design.md` only if verification exposes a genuine design correction
- Test: all files under `tests/`

**Interfaces:**
- Produces: reproducible training/classification commands and verification evidence.
- Consumes: all prior tasks.

- [ ] **Step 1: Add exact operator commands to `AGENTS.md`**

```powershell
python train.py --task_data ..\train_data --output .\weights\four_class --no_init --baseline_metrics .\weights\old\four_class_baseline.json
python classify.py --input_dir D:\GZ_20160702 --output_dir "E:\Guoxing Yang\typhoon_classified\2016.0702-2016.0709\four_class\20160702" --type_only --model .\weights\four_class\model.pt --batch_size 256
```

Document that `IC` means rejected/not researched for this schema and that four-class checkpoints must not be passed `--min_type_confidence`.

- [ ] **Step 2: Run the full automated suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run the compile check**

Run: `python -m compileall -q .`

Expected: exit code `0` with no syntax errors.

- [ ] **Step 4: Run a bounded synthetic smoke test**

Create only temporary synthetic `.lig` inputs through existing test helpers, classify with a four-class test checkpoint, and verify: one accepted researched-class directory, one rejected `IC` directory, complete CSV audit rows, and byte-for-byte piece equality. Delete only the temporary directory created by the smoke test.

Run: `python -m pytest -q tests/test_classify.py -k "four_class or raw_piece"`

Expected: PASS.

- [ ] **Step 5: Produce the real baseline before training**

Run the deployed old model on the unchanged locked temporal test and export a four-researched-class report containing macro-F1, per-type precision/recall/F1, confusion matrix, and prediction shares. Save it outside Git at `weights/old/four_class_baseline.json`. Do not derive this baseline from 2016 manual-review data.

- [ ] **Step 6: Train and evaluate without automatic deployment**

```powershell
python train.py --task_data ..\train_data --output .\weights\four_class --no_init --baseline_metrics .\weights\old\four_class_baseline.json
```

Expected: `candidate.pt` is always written; `model.pt` is written only if every automated release gate passes. Record the locked-test report and do not change thresholds after reading it.

- [ ] **Step 7: Perform blinded 2016 comparison**

Classify the same fixed 2016 file list with old and candidate models into separate directories, randomize model labels in the review sheet, and record per-model accepted-output correctness and coverage. Promote manually only when the candidate is at least as accurate as the old model and automated gates passed.

- [ ] **Step 8: Commit documentation**

```powershell
git add AGENTS.md
git commit -m "Document four-class rejection workflow"
```
