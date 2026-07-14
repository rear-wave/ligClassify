# Reliable Distance Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a versioned ordinal multi-task model that improves time-held-out non-IC distance estimation and reports calibrated uncertainty without breaking legacy checkpoints.

**Architecture:** Keep the chronological manifest and the current 80% IC type stream. Add a capped hierarchical non-IC distance sampler, a four-term ordinal objective, and a v2 distance branch with average/max pooling. Select on validation worst-class `w2`, calibrate only on validation, and evaluate the locked test set once.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn, pytest, Windows PowerShell.

## Global Constraints

- Do not move files across chronological train/validation/test boundaries or rebalance validation/test.
- Do not use date, path, folder name, or distance directory as model input.
- Keep raw `.lig` files and existing checkpoints unchanged.
- Preserve legacy `mtl_resnet` loading in `infer_mtl.py`.
- Full-coverage oracle target: MAE <= 200 km, overall `w2` >= 0.80, every non-IC type `w2` >= 0.70.
- End-to-end target: coverage >= 0.95 and `w2` >= 0.75 with IC routing counted as failure.
- Use tests before implementation and never use the locked test metrics to choose hyperparameters.

## File Map

- Create `data/distance_sampling.py`: deterministic hierarchical distance sampler.
- Create `distance_ordinal.py`: ordinal loss, decoding, calibration, confidence, and selection helpers.
- Modify `train_mtl.py`: dataset metadata, dual streams, v2 losses, metrics, split hash, checkpoint metadata.
- Modify `models.py`: versioned `MultiTaskOrdinalResNet` and architecture factory.
- Modify `infer_mtl.py`: version-aware loading and reliability CSV/output routing.
- Create `tests/test_distance_sampling.py`, `tests/test_distance_ordinal.py`, and `tests/test_models.py`.
- Extend `tests/test_train_mtl.py` for dual-stream updates, evaluation, and compatibility.

---

### Task 1: Hierarchical Distance Sampler

**Files:**
- Create: `data/distance_sampling.py`
- Create: `tests/test_distance_sampling.py`
- Modify: `train_mtl.py` (`MultiTaskDataset` metadata arrays)

**Interfaces:**
- Produces: `HierarchicalDistanceSampler(type_labels, dist_labels, date_ids, file_ids, num_samples, max_samples_per_file, seed)` with `set_epoch(epoch)`.
- Consumed by: the distance `DataLoader` in Task 4.

- [ ] **Step 1: Write failing sampler tests**

```python
import numpy as np

from data.distance_sampling import HierarchicalDistanceSampler


def make_sampler(seed=7):
    return HierarchicalDistanceSampler(
        type_labels=np.array([0, 1, 1, 1, 2, 2, 2, 2]),
        dist_labels=np.array([-1, 0, 0, 2, 0, 1, 1, 1]),
        date_ids=np.array([1, 1, 2, 2, 1, 1, 2, 2]),
        file_ids=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        num_samples=6,
        max_samples_per_file=1,
        seed=seed,
    )


def test_sampler_uses_only_labelled_non_ic_and_caps_files():
    sampler = make_sampler()
    indices = list(sampler)
    assert len(indices) == 6
    assert all(indices.count(index) <= 1 for index in set(indices))
    assert all(index != 0 for index in indices)


def test_sampler_is_reproducible_and_changes_by_epoch():
    first = make_sampler()
    second = make_sampler()
    assert list(first) == list(second)
    second.set_epoch(1)
    assert list(first) != list(second)


def test_sampler_ignores_missing_bins_without_error():
    sampler = make_sampler()
    assert set(list(sampler)).issubset(set(range(1, 8)))
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_distance_sampling.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'data.distance_sampling'`.

- [ ] **Step 3: Implement the sampler**

Create a `torch.utils.data.Sampler` that builds nested pools keyed by `(type, bin, date, file)`, shuffles positions with `numpy.random.default_rng(seed + epoch)`, and draws in round-robin type/bin order. Each file contributes at most `min(original_piece_count, max_samples_per_file)` positions. Stop at `num_samples` or when all capped pools are empty. `__len__` returns the smaller of `num_samples` and the sum of all per-file caps. Reject non-positive `num_samples` or `max_samples_per_file` with `ValueError`.

Add aligned `file_ids` and integer `date_ids` arrays to `MultiTaskDataset` beside `global_indices`, `type_labels`, and `dist_labels`. Apply the same IC-selection index to all five arrays. Encode dates as `YYYYMMDD` integers.

- [ ] **Step 4: Run focused and existing dataset tests**

Run: `pytest tests/test_distance_sampling.py tests/test_train_mtl.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add data/distance_sampling.py train_mtl.py tests/test_distance_sampling.py tests/test_train_mtl.py
git commit -m "feat: add balanced distance sampler"
```

### Task 2: Ordinal Loss and Reliable Decoding

**Files:**
- Create: `distance_ordinal.py`
- Create: `tests/test_distance_ordinal.py`

**Interfaces:**
- Produces: `ordinal_distance_loss`, `aggregate_coarse_probabilities`, `decode_distance_logits`, `fit_temperature_grid`, `select_confidence_threshold`, and `make_selection_key`.
- Consumed by: Tasks 4-7.

- [ ] **Step 1: Write failing mathematical tests**

```python
import torch

from distance_ordinal import (
    aggregate_coarse_probabilities,
    decode_distance_logits,
    make_selection_key,
    ordinal_distance_loss,
    select_confidence_threshold,
)


def test_coarse_probabilities_are_normalized_groups_of_three():
    probs = torch.full((2, 30), 1 / 30)
    coarse = aggregate_coarse_probabilities(probs)
    assert coarse.shape == (2, 10)
    assert torch.allclose(coarse, torch.full((2, 10), 0.1))


def test_ordinal_loss_prefers_near_error_to_far_error():
    target = torch.tensor([10])
    near = torch.full((1, 30), -8.0); near[0, 11] = 8.0
    far = torch.full((1, 30), -8.0); far[0, 25] = 8.0
    near_loss, _ = ordinal_distance_loss(near, target)
    far_loss, _ = ordinal_distance_loss(far, target)
    assert near_loss < far_loss


def test_decoder_returns_expected_center_interval_and_confidence():
    logits = torch.full((1, 30), -20.0); logits[0, 4] = 20.0
    decoded = decode_distance_logits(logits)
    assert decoded["bin"].item() == 4
    assert decoded["distance_km"].item() == 450.0
    assert decoded["low_km"].item() == 400.0
    assert decoded["high_km"].item() == 500.0
    assert decoded["confidence"].item() > 0.99


def test_threshold_uses_maximum_coverage_that_meets_target():
    confidence = torch.tensor([0.9, 0.8, 0.2, 0.1])
    error_bins = torch.tensor([0.0, 1.0, 5.0, 8.0])
    result = select_confidence_threshold(confidence, error_bins, target_w2=0.8)
    assert result["threshold"] == 0.8
    assert result["coverage"] == 0.5
    assert result["w2"] == 1.0


def test_selection_key_protects_worst_class_before_guardrail():
    weaker = {"per_type_w2": [0.60, 0.90, 0.90, 0.90], "w2": 0.82,
              "macro_w2": 0.825, "macro_mae_km": 170, "type_f1": 0.92}
    safer = {"per_type_w2": [0.65, 0.80, 0.80, 0.80], "w2": 0.78,
             "macro_w2": 0.7625, "macro_mae_km": 190, "type_f1": 0.91}
    assert make_selection_key(safer) > make_selection_key(weaker)
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest tests/test_distance_ordinal.py -q`

Expected: collection fails because `distance_ordinal.py` does not exist.

- [ ] **Step 3: Implement vectorized ordinal functions**

Use `softmax(logits / temperature)`. Build exponential soft targets with `exp(-abs(bin-y)/tau)`. Define CDF loss as mean absolute predicted-target CDF difference, Huber loss on expected bin, and coarse NLL from grouped probabilities. Return `(total_loss, {"soft_ce", "cdf", "huber", "coarse"})`. Decode the rounded expected bin and center distance; obtain the 10th and 90th percentile bin edges with `searchsorted`; confidence is probability mass in bins within two of the rounded estimate.

`fit_temperature_grid(logits, targets)` must test `torch.arange(0.50, 5.01, 0.05)` and return the lowest-NLL value without gradients. `select_confidence_threshold` tests unique confidence values in descending coverage order and chooses the largest coverage with `error_bins <= 2` accuracy at least the target. `make_selection_key` returns `(min_w2, macro_w2, -macro_mae, type_f1)` until all types reach 0.70, then `(overall_w2, macro_w2, -macro_mae, type_f1)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_distance_ordinal.py -q`

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add distance_ordinal.py tests/test_distance_ordinal.py
git commit -m "feat: add ordinal distance objective"
```

### Task 3: Versioned Ordinal Model

**Files:**
- Modify: `models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `MultiTaskOrdinalResNet` and `create_mtl_model(base_channels=64, architecture="ordinal_v2", dist_mlp_dim=128, dist_dropout=0.2)`.
- Preserves: existing `create_mtl_model(base_channels=64)` behavior.

- [ ] **Step 1: Write failing shape and compatibility tests**

```python
import torch

from models import create_mtl_model


def test_ordinal_v2_output_shapes():
    model = create_mtl_model(base_channels=8, architecture="ordinal_v2",
                             dist_mlp_dim=16, dist_dropout=0.0)
    type_logits, distance_logits = model(torch.randn(3, 1, 8000))
    assert type_logits.shape == (3, 5)
    assert len(distance_logits) == 4
    assert all(item.shape == (3, 30) for item in distance_logits)


def test_default_factory_remains_legacy_model():
    model = create_mtl_model(base_channels=8)
    assert model.__class__.__name__ == "MultiTaskResNet"
```

- [ ] **Step 2: Verify the v2 test fails**

Run: `pytest tests/test_models.py -q`

Expected: `TypeError` for the unknown `architecture` argument.

- [ ] **Step 3: Add the v2 distance branch**

Reuse the current stem and residual layers. Keep type logits from adaptive average pooling. For distance, concatenate adaptive average and maximum pooling, then apply `LayerNorm(4 * base)`, `Linear(4 * base, dist_mlp_dim)`, `GELU`, and `Dropout`; connect four `Linear(dist_mlp_dim, 30)` heads. Validate `architecture` against `{"mtl_resnet", "ordinal_v2"}` and raise `ValueError` otherwise.

- [ ] **Step 4: Run model tests and a CUDA forward smoke test**

Run: `pytest tests/test_models.py -q`

Run: `python -c "import torch; from models import create_mtl_model; m=create_mtl_model(16,'ordinal_v2',32,0.1).cuda(); y=m(torch.randn(2,1,8000,device='cuda')); print(y[0].shape, y[1][0].shape)"`

Expected: tests pass and CUDA prints `torch.Size([2, 5]) torch.Size([2, 30])`.

- [ ] **Step 5: Commit**

```powershell
git add models.py tests/test_models.py
git commit -m "feat: add ordinal distance model"
```

### Task 4: Dual-Stream Training Integration

**Files:**
- Modify: `train_mtl.py`
- Modify: `tests/test_train_mtl.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `alternate_stream_batches`, `compute_distance_head_loss`, and v2 CLI arguments.

- [ ] **Step 1: Add failing unit tests**

```python
def test_alternate_stream_batches_consumes_each_loader_once():
    batches = list(train_mtl.alternate_stream_batches(["t1", "t2", "t3"], ["d1"]))
    assert batches == [("type", "t1"), ("distance", "d1"),
                       ("type", "t2"), ("type", "t3")]


def test_distance_loss_is_macro_averaged_across_present_types():
    logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([1, 1, 1, 2])
    distances = torch.tensor([0, 1, 2, 5])
    loss, components = train_mtl.compute_distance_head_loss(
        logits, labels, distances,
        tau=1.0, lambda_emd=1.0, lambda_reg=0.5, lambda_coarse=0.5,
    )
    assert loss.ndim == 0
    assert set(components) == {"soft_ce", "cdf", "huber", "coarse"}
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_train_mtl.py -q`

Expected: both new helpers are missing.

- [ ] **Step 3: Implement dual loaders and stream-specific updates**

Add CLI arguments:

```text
--model_arch {mtl_resnet,ordinal_v2}  default ordinal_v2
--distance_batch_size INT             default 128
--distance_samples_per_epoch INT      default -1 (all labelled non-IC pieces)
--max_distance_samples_per_file INT   default 256
--dist_mlp_dim INT                    default 128
--dist_dropout FLOAT                  default 0.2
--lambda_emd FLOAT                    default 1.0
--lambda_reg FLOAT                    default 0.5
--lambda_coarse FLOAT                 default 0.5
--distance_batch_type_weight FLOAT    default 0.1
--distance_sampling {uniform,hierarchical} default hierarchical
--distance_objective {ce,ordinal}     default ordinal
--distance_prediction {argmax,expected} default expected
--skip_test                            do not read/evaluate the locked test split
```

Create the distance loader over `train_set` with `HierarchicalDistanceSampler`. At each epoch call `sampler.set_epoch(epoch)`. Type batches optimize only type CE. Distance batches route by ground-truth non-IC type, macro-average the four present head losses, add `0.1 * type CE`, and make one optimizer step. Keep legacy training available when `--model_arch mtl_resnet` is explicitly selected.

- [ ] **Step 4: Run focused tests and CLI help**

Run: `pytest tests/test_train_mtl.py tests/test_distance_sampling.py tests/test_distance_ordinal.py -q`

Run: `python train_mtl.py --help`

Expected: tests pass and all v2 arguments appear.

- [ ] **Step 5: Commit**

```powershell
git add train_mtl.py tests/test_train_mtl.py
git commit -m "feat: train with balanced ordinal distance stream"
```

### Task 5: Metrics, Early Stopping, and Split Identity

**Files:**
- Modify: `train_mtl.py`
- Modify: `tests/test_train_mtl.py`

**Interfaces:**
- Produces: expected-bin oracle/end-to-end metrics, `compute_split_hash`, and lexicographic early stopping.
- Consumed by: checkpointing and final reports.

- [ ] **Step 1: Write failing metric tests**

Add explicit tests named `test_v2_metrics_use_expected_bin`, `test_e2e_ic_route_counts_as_w2_failure`, `test_macro_w2_is_unweighted_across_types`, and `test_split_hash_is_order_independent_but_membership_sensitive`. The first uses logits with equal mass on bins 4 and 6 and expects decoded bin 5. The second routes a labelled NNBE sample through predicted IC and expects coverage `0.0` and end-to-end `w2 == 0.0`. The third supplies class `w2` values `[0.4, 0.6, 0.8, 1.0]` and expects `0.7`. The fourth constructs two `ManifestEntry` lists in reversed order, asserts equal hashes, then changes one `dist_bin` and asserts unequal hashes.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_train_mtl.py -q`

Expected: missing v2 metric keys and split hash helper.

- [ ] **Step 3: Extend evaluation and selection**

Report for argmax and expected-bin predictions: exact accuracy, MAE, `w1`, `w2`, macro MAE, macro `w2`, minimum per-type `w2`, end-to-end coverage, and end-to-end `w2` with uncovered samples failing. Keep existing metric names as aliases for the selected primary prediction mode. Hash normalized relative path, type, date, distance bin, piece count, and split name with SHA-256. Carry `file_id` and `date_id` through evaluation batches and add `group_bootstrap_distance_metrics(predictions, targets, group_ids, seed=42, repetitions=1000)` that resamples unique file IDs and returns percentile 95% intervals for MAE and `w2`.

Replace the scalar `0.4 * type_f1 + 0.6 * exact_accuracy` score with `make_selection_key`. Store the best tuple and reset patience only when the first differing value improves by its configured epsilon. Do not load or evaluate test data during epoch selection.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_mtl.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add train_mtl.py tests/test_train_mtl.py
git commit -m "feat: select models on ordinal reliability"
```

### Task 6: Validation Calibration and Versioned Checkpoints

**Files:**
- Modify: `train_mtl.py`
- Modify: `infer_mtl.py`
- Extend: `tests/test_distance_ordinal.py`
- Extend: `tests/test_train_mtl.py`

**Interfaces:**
- Consumes: validation logits/labels from Task 5.
- Produces: `distance_temperatures`, `confidence_threshold`, and v2 checkpoint metadata.

- [ ] **Step 1: Write failing calibration and loading tests**

Test that temperature fitting never increases validation NLL over temperature `1.0`; the checkpoint round-trip preserves four temperatures and the confidence threshold; and inference chooses the legacy factory when `model_name == "mtl_resnet"` and v2 when `model_name == "ordinal_v2"`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_distance_ordinal.py tests/test_train_mtl.py -q`

Expected: calibration metadata and version-aware loading assertions fail.

- [ ] **Step 3: Fit and save reliability metadata**

After loading the best state, collect validation logits once. Fit a grid-search temperature independently for each true-type distance head. Decode calibrated validation predictions and select the maximum-coverage confidence threshold satisfying validation `w2 >= 0.80`. Save architecture, MLP dimensions, dropout, all loss coefficients, sampler settings, split hashes, four temperatures, threshold, baseline metrics, and validation metrics in both best and final structured checkpoints. Continue saving the raw `mtl_best.pt` state dictionary for compatibility.

- [ ] **Step 4: Run calibration tests**

Run: `pytest tests/test_distance_ordinal.py tests/test_train_mtl.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add train_mtl.py infer_mtl.py tests/test_distance_ordinal.py tests/test_train_mtl.py
git commit -m "feat: calibrate distance confidence"
```

### Task 7: Reliable Inference Export

**Files:**
- Modify: `infer_mtl.py`
- Create: `tests/test_infer_mtl.py`

**Interfaces:**
- Consumes: v2 checkpoint metadata and `decode_distance_logits`.
- Produces: classified `.lig` output plus `predictions.csv` with reliability fields.

- [ ] **Step 1: Write failing routing/export tests**

Use synthetic logits to assert that a calibrated high-confidence NNBE prediction routes to `NNBE_400-500km`, a below-threshold prediction routes to `UNCERTAIN_NNBE`, IC remains `IC`, and each CSV row contains `source_file`, `piece_index`, `type`, `distance_km`, `bin_start_km`, `low_km`, `high_km`, `confidence`, and `status`.

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_infer_mtl.py -q`

Expected: reliability routing/export helpers are missing.

- [ ] **Step 3: Implement version-aware reliable output**

Add `load_mtl_checkpoint`, `route_prediction`, and `PredictionCsvWriter`. Legacy checkpoints retain argmax routing and blank reliability fields. V2 checkpoints apply per-head temperature, expected-bin decoding, and the stored threshold. Add `--keep_uncertain_in_type_bin` to retain uncertain pieces in the predicted bin; default behavior places them in `UNCERTAIN_<TYPE>`. Preserve batched streaming and `.lig` piece counts.

- [ ] **Step 4: Run inference tests and CLI smoke tests**

Run: `pytest tests/test_infer_mtl.py -q`

Run: `python infer_mtl.py --help`

Expected: tests pass and the uncertainty option is listed.

- [ ] **Step 5: Commit**

```powershell
git add infer_mtl.py tests/test_infer_mtl.py
git commit -m "feat: export calibrated distance predictions"
```

### Task 8: Verification and Controlled Experiments

**Files:**
- Modify: `docs/superpowers/plans/2026-07-10-reliable-distance-classification.md` (check completed boxes and record commands)
- Output only: separate checkpoint directories under `C:\lig_runs\distance_v2\`

**Interfaces:**
- Produces: verified implementation and comparable validation/test artifacts.

- [ ] **Step 1: Run the complete static and unit test suite**

Run:

```powershell
pytest -q
python -m compileall -q .
python train_mtl.py --help
python infer_mtl.py --help
git diff --check
```

Expected: all tests pass, compile/help commands exit zero, and `git diff --check` reports no errors.

- [ ] **Step 2: Run a one-epoch CUDA integration smoke test**

```powershell
python train_mtl.py --task_data C:\lig_data\train_data_subset --output C:\lig_runs\distance_v2\smoke --model_arch ordinal_v2 --epochs 1 --batch_size 128 --distance_batch_size 128 --distance_samples_per_epoch 4096 --max_distance_samples_per_file 32 --num_workers 0 --seed 42
```

Expected: one epoch completes, all losses are finite, and a structured v2 checkpoint contains split hashes and reliability metadata.

- [ ] **Step 3: Run cumulative validation ablations with seeds 42, 43, and 44**

Run these exact validation-only commands:

```powershell
$seeds = 42,43,44
foreach ($seed in $seeds) {
  python train_mtl.py --task_data C:\lig_data\train_data_subset --output "C:\lig_runs\distance_v2\sampling_s$seed" --model_arch mtl_resnet --distance_sampling hierarchical --distance_objective ce --distance_prediction argmax --skip_test --seed $seed
  python train_mtl.py --task_data C:\lig_data\train_data_subset --output "C:\lig_runs\distance_v2\ordinal_s$seed" --model_arch mtl_resnet --distance_sampling hierarchical --distance_objective ordinal --distance_prediction expected --skip_test --seed $seed
  python train_mtl.py --task_data C:\lig_data\train_data_subset --output "C:\lig_runs\distance_v2\v2_s$seed" --model_arch ordinal_v2 --distance_sampling hierarchical --distance_objective ordinal --distance_prediction expected --skip_test --seed $seed
}
```

Record validation macro/worst-class `w2`, macro MAE, and type macro-F1. Promote only when median validation reliability improves and no class loses more than two `w2` percentage points.

- [ ] **Step 4: Evaluate the selected configuration once on the locked test set**

Rerun exactly the selected stage and seed without `--skip_test`, writing to `C:\lig_runs\distance_v2\final_locked_test`. The command must write final `mtl.json` containing all acceptance metrics. Compare it with the frozen checkpoint: 329 km MAE, 52.62% `w2`, and 91.36% type macro-F1. Do not change hyperparameters after reading these final test results.

- [ ] **Step 5: Document outcome and commit final integration**

Record whether each acceptance criterion passed, the three-seed validation spread, final test metrics, confidence coverage, and any failing type/date-bin support. Then run:

```powershell
git add models.py train_mtl.py infer_mtl.py distance_ordinal.py data/distance_sampling.py tests docs/superpowers
git commit -m "feat: improve distance classification reliability"
```
