# Conditional Expert Lightning Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and release an honestly evaluated four-type lightning waveform classifier with type-conditioned distance experts and calibrated IC rejection.

**Architecture:** A polarity-preserving local/global waveform pipeline feeds a multi-scale 1D encoder. A four-class type head uses waveform features, while four routed distance experts additionally consume day/night context and learn from exact or interval-censored distance labels. File-isolated validation calibrates rejection; one locked file-isolated test compares the candidate with the old deployment baseline.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, scikit-learn, pytest, CUDA AMP on NVIDIA RTX 5080.

## Global Constraints

- Train only NCG, NNBE, PCG, and PNBE; never train on the low-quality IC folder.
- `IC` is an inference-only rejection result.
- Do not split one source `.lig` file across train, validation, and test.
- Balance splits and sampling by type, day/night, and distance; do not impose a year split.
- Keep `weights/old/model.pt` and existing candidate checkpoints immutable until release gates pass.
- Preserve original `.lig` piece bytes during inference.
- Preserve and reconcile the existing uncommitted changes in `classify.py`, `models.py`, `tests/test_classify.py`, and `tests/test_models.py`; never discard them.
- Use synthetic fixtures in Git; do not commit waveform data, checkpoints, generated classifications, or absolute machine paths.
- Random initialization remains the training default; resume and warm start require explicit CLI flags.

---

### Task 1: Interval-Aware Manifest and File-Isolated Split

**Files:**
- Modify: `data/training_manifest.py`
- Create: `data/group_split.py`
- Modify: `tests/test_training_manifest.py`
- Create: `tests/test_group_split.py`

**Interfaces:**
- Produces: `parse_distance_interval(path: str) -> tuple[int, int] | None`
- Produces: `infer_daytime(path: str, timestamp: datetime) -> bool`
- Produces: `group_stratified_split(entries, val_fraction, test_fraction, seed) -> dict[str, list[ManifestEntry]]`
- Produces: `validate_group_split(splits) -> None`
- `ManifestEntry` and `PieceManifestEntry` gain `distance_low_km`, `distance_high_km`, and `is_daytime`; the compatibility `dist_bin` value is non-negative only for exact 100 km labels.

- [ ] **Step 1: Write failing interval and context tests**

```python
def test_distance_interval_keeps_exact_and_broad_labels():
    assert parse_distance_interval("NCG/day/day_400-500km/a.lig") == (400, 500)
    assert parse_distance_interval("NNBE/night/night_1500-3000km/a.lig") == (1500, 3000)
    assert parse_distance_interval("NCG/day/a.lig") is None


def test_folder_context_precedes_timestamp_fallback():
    noon = datetime(2019, 1, 1, 12, 0)
    assert infer_daytime("NCG/night/night_0-100km/a.lig", noon) is False
    assert infer_daytime("NCG/day/day_0-100km/a.lig", noon) is True
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_training_manifest.py -q`

Expected: FAIL because `parse_distance_interval` and `infer_daytime` do not exist.

- [ ] **Step 3: Implement interval parsing and manifest fields**

```python
def parse_distance_interval(path: str) -> tuple[int, int] | None:
    matches = list(_DISTANCE_RE.finditer(path))
    if not matches:
        return None
    low, high = map(int, matches[-1].groups())
    if low % 100 or high % 100 or not 0 <= low < high <= 3000:
        return None
    return low, high


def infer_daytime(path: str, timestamp: datetime) -> bool:
    parts = {part.lower() for part in Path(path).parts}
    if "day" in parts:
        return True
    if "night" in parts:
        return False
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 5.5 <= local_hour < 19.0
```

Set `dist_bin = low // 100` only when `high - low == 100`; retain both interval endpoints for all researched-type files.

- [ ] **Step 4: Write the failing group-isolation and balance tests**

```python
def test_group_split_never_shares_a_source_file():
    entries = make_entries(files_per_stratum=6)
    splits = group_stratified_split(entries, 0.2, 0.2, seed=7)
    validate_group_split(splits)
    owners = {}
    for split_name, selected in splits.items():
        for entry in selected:
            assert owners.setdefault(entry.filepath, split_name) == split_name


def test_group_split_covers_type_daylight_and_coarse_distance():
    splits = group_stratified_split(make_entries(files_per_stratum=6), 0.2, 0.2, 7)
    for selected in splits.values():
        assert {(e.type_idx, e.is_daytime, e.distance_low_km // 600) for e in selected} == EXPECTED_STRATA
```

- [ ] **Step 5: Implement deterministic greedy grouped stratification**

In `data/group_split.py`, aggregate piece counts per source file, define the stratum as `(type_idx, is_daytime, coarse_distance_band)`, order files by `(-n_pieces, stable_seed_hash)`, and assign each file to the split minimizing squared deviation from target file, piece, and stratum counts. Reject duplicate file ownership and record deficits instead of silently moving pieces between files.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_training_manifest.py tests/test_group_split.py -q`

Expected: PASS.

Commit: `git commit -m Add-file-isolated-interval-manifest`

---

### Task 2: Polarity-Preserving Multi-Scale Signal Pipeline

**Files:**
- Modify: `data/preprocessing.py`
- Create: `data/signal_context.py`
- Create: `data/waveform_quality.py`
- Modify: `tests/test_preprocessing.py`
- Create: `tests/test_signal_context.py`

**Interfaces:**
- Produces: `preprocess_multiscale_batch(pieces) -> tuple[np.ndarray, np.ndarray]`, both shaped `(N, 8000)`
- Produces: `time_context_batch(timestamps, is_daytime) -> np.ndarray`, shaped `(N, 3)` as `[daylight, sin(local_hour), cos(local_hour)]`
- Produces: `waveform_quality_batch(pieces) -> np.ndarray`, shaped `(N, 3)` as `[snr_score, clipping_fraction, baseline_instability]`

- [ ] **Step 1: Write failing polarity and view tests**

```python
def test_absolute_peak_alignment_preserves_negative_polarity():
    x = np.zeros((1, 16000), dtype=np.float32)
    x[0, 9000] = -20.0
    x[0, 2000] = 5.0
    local, global_view = preprocess_multiscale_batch(x, use_filter=False)
    assert local[0, 2000] < 0
    assert local.shape == global_view.shape == (1, 8000)


def test_signed_normalization_does_not_flip_waveform():
    local, _ = preprocess_multiscale_batch(negative_fixture()[None], use_filter=False)
    assert np.argmin(local[0]) == 2000
```

- [ ] **Step 2: Verify the focused tests fail**

Run: `python -m pytest tests/test_preprocessing.py -q`

Expected: FAIL because the current crop uses `np.argmax` and has no multi-scale API.

- [ ] **Step 3: Implement local/global preprocessing**

```python
peak_indices = np.argmax(np.abs(centered), axis=1)
local = gather_fixed_windows(centered, peak_indices, before=2000, length=8000)
global_view = scipy.signal.resample_poly(centered, up=1, down=2, axis=1)
local = robust_signed_scale(local)
global_view = robust_signed_scale(global_view[:, :8000])
return local.astype(np.float32), global_view.astype(np.float32)
```

`robust_signed_scale` divides centred samples by `max(percentile(abs(x), 99.5), 1e-6)` and clips only after scaling; it never takes absolute waveform values.

- [ ] **Step 4: Add context and quality tests**

```python
def test_time_context_is_periodic_and_contains_daylight():
    values = time_context_batch([datetime(2019, 1, 1, 16)], [False])
    assert values.shape == (1, 3)
    assert values[0, 0] == 0.0
    assert np.isclose(np.linalg.norm(values[0, 1:]), 1.0)


def test_quality_flags_clipping_and_low_snr():
    quality = waveform_quality_batch(np.stack([clean_pulse(), clipped_flatline()]))
    assert quality[0, 0] > quality[1, 0]
    assert quality[1, 1] > quality[0, 1]
```

- [ ] **Step 5: Implement deterministic context and quality features**

Compute local UTC+8 hour from piece timestamps; calculate SNR from peak-to-MAD ratio, clipping from the fraction at the observed digital extrema, and baseline instability from first/last-window median difference normalized by MAD. Clamp finite outputs to documented ranges.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_preprocessing.py tests/test_signal_context.py -q`

Expected: PASS.

Commit: `git commit -m Add-polarity-preserving-multiscale-inputs`

---

### Task 3: Interval-Censored Ordered Distance Objective

**Files:**
- Modify: `distance_ordinal.py`
- Modify: `tests/test_distance_ordinal.py`

**Interfaces:**
- Produces: `interval_distance_loss(logits, low_km, high_km, ordered_weight=0.2) -> tuple[Tensor, dict[str, Tensor]]`
- Produces: `decode_distance_distribution(logits, temperature=1.0) -> dict[str, Tensor]`

- [ ] **Step 1: Write failing exact, broad, and ordering tests**

```python
def test_interval_loss_rewards_probability_anywhere_in_broad_label():
    near = peaked_logits(7)
    far = peaked_logits(20)
    near_loss, _ = interval_distance_loss(near, tensor([600]), tensor([1200]))
    far_loss, _ = interval_distance_loss(far, tensor([600]), tensor([1200]))
    assert near_loss < far_loss


def test_decoder_returns_expected_bin_and_quantile_interval():
    decoded = decode_distance_distribution(peaked_logits(4))
    assert decoded["bin_index"].item() == 4
    assert 400 <= decoded["expected_km"].item() <= 500
    assert decoded["low_km"].item() <= decoded["high_km"].item()
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_distance_ordinal.py -q`

Expected: FAIL because the interval APIs do not exist.

- [ ] **Step 3: Implement the interval likelihood and ordered penalty**

```python
probabilities = logits.softmax(dim=-1)
centers = torch.arange(50, 3000, 100, device=logits.device)
inside = (centers[None] >= low_km[:, None]) & (centers[None] < high_km[:, None])
mass = (probabilities * inside).sum(dim=1).clamp_min(1e-8)
nll = -mass.log()
outside_km = torch.maximum(low_km[:, None] - centers, centers - high_km[:, None]).clamp_min(0)
ordered = (probabilities * outside_km.div(100)).sum(dim=1)
loss = (nll + ordered_weight * ordered).mean()
```

Decode expected kilometres and 10th/90th cumulative-probability bounds; keep the existing decoder for legacy checkpoints.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_distance_ordinal.py -q`

Expected: PASS.

Commit: `git commit -m Add-interval-distance-objective`

---

### Task 4: Multi-Scale Conditional Expert Network

**Files:**
- Modify: `models.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: architecture name `conditional_expert_v1`
- Produces: `ConditionalExpertNet.forward_with_features(local, global_view, context)` returning `(type_features, type_logits, distance_logits, coarse_logits)`
- Distance and coarse outputs are lists of four tensors shaped `(N, 30)` and `(N, 5)`.

- [ ] **Step 1: Write the failing architecture contract test**

```python
def test_conditional_expert_output_contract():
    model = create_mtl_model(architecture="conditional_expert_v1", base_channels=16)
    features, type_logits, distance, coarse = model.forward_with_features(
        torch.randn(3, 1, 8000), torch.randn(3, 1, 8000), torch.randn(3, 3)
    )
    assert type_logits.shape == (3, 4)
    assert len(distance) == len(coarse) == 4
    assert all(x.shape == (3, 30) for x in distance)
    assert all(x.shape == (3, 5) for x in coarse)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_models.py::test_conditional_expert_output_contract -q`

Expected: FAIL because the factory does not recognize the architecture.

- [ ] **Step 3: Implement focused model components**

Add `MultiScaleResidualBlock`, `WaveformBranch`, and `ConditionalExpertNet`. Each residual block uses depthwise kernels 7, 31, and 127 followed by pointwise fusion. Local and global branches share the block design but not stem weights. The type head consumes only fused waveform features; each distance expert consumes `torch.cat([features, context], dim=1)`. Coarse heads share the same conditioned input.

- [ ] **Step 4: Add routing-gradient and context tests**

```python
def test_type_logits_do_not_depend_on_context():
    model = make_eval_conditional_model()
    a = model.forward_with_features(LOCAL, GLOBAL, torch.zeros(2, 3))[1]
    b = model.forward_with_features(LOCAL, GLOBAL, torch.ones(2, 3))[1]
    assert torch.allclose(a, b)


def test_distance_experts_receive_context():
    model = make_eval_conditional_model()
    a = model.forward_with_features(LOCAL, GLOBAL, torch.zeros(2, 3))[2]
    b = model.forward_with_features(LOCAL, GLOBAL, torch.ones(2, 3))[2]
    assert any(not torch.allclose(x, y) for x, y in zip(a, b))
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_models.py -q`

Expected: PASS for legacy and new architecture contracts.

Commit: `git commit -m Add-conditional-distance-expert-network`

---

### Task 5: Dataset, Balanced Streams, and AMP Training

**Files:**
- Create: `data/training_dataset.py`
- Modify: `data/distance_sampling.py`
- Modify: `train.py`
- Modify: `tests/test_distance_sampling.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces: `LightningPieceDataset` returning a dictionary with `local`, `global_view`, `context`, `quality`, `type_label`, `distance_low_km`, `distance_high_km`, `file_id`, and `timestamp`.
- Produces: `ConditionBalancedSampler(type_labels, daylight, distance_low_km, file_ids, num_samples, max_samples_per_file, seed)`.
- `train.py` consumes `group_stratified_split` and defaults to `conditional_expert_v1` with random initialization.

- [ ] **Step 1: Write failing dataset and sampler tests**

```python
def test_dataset_exposes_multiscale_interval_sample(tmp_path):
    dataset = make_piece_dataset(tmp_path, low=600, high=1200, daytime=False)
    sample = dataset[0]
    assert sample["local"].shape == sample["global_view"].shape == (1, 8000)
    assert sample["distance_low_km"] == 600
    assert sample["distance_high_km"] == 1200
    assert sample["context"].shape == (3,)


def test_condition_sampler_balances_type_daylight_and_range():
    sampler = make_condition_sampler()
    keys = [sampler.condition_key(index) for index in list(iter(sampler))]
    assert max(Counter(keys).values()) - min(Counter(keys).values()) <= 1
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_distance_sampling.py tests/test_train.py -q`

Expected: FAIL because the new dataset and sampler are absent.

- [ ] **Step 3: Implement lazy dataset and balanced sampler**

Reuse one `LigFileIndex`; preprocess only requested indices. Define coarse bands `[0, 300, 600, 1200, 1700, 2400, 3000]`, use interval midpoint only for sampling strata, and never replace the interval targets. Allocate equal samples across available `(type, daylight, coarse_band)` keys, then across files, enforcing `max_samples_per_file`.

- [ ] **Step 4: Replace training orchestration**

Update `train.py` to build one deterministic split, write `split_manifest.json` and `data_audit.json`, construct separate type/distance streams, and route interval loss by trusted zero-based type. Use `torch.autocast(device_type="cuda", dtype=torch.float16)` and `GradScaler` on CUDA, pinned memory, persistent workers, non-blocking transfers, and TF32. Add explicit `--resume` and `--init_model`; neither is used unless supplied.

- [ ] **Step 5: Add a one-step gradient test**

```python
def test_interval_batch_updates_only_represented_experts():
    before = clone_expert_weights(model)
    train_conditional_step(model, batch_with_types([0, 2]), optimizer)
    assert expert_changed(model, before, 0)
    assert not expert_changed(model, before, 1)
    assert expert_changed(model, before, 2)
    assert not expert_changed(model, before, 3)
```

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_distance_sampling.py tests/test_train.py -q`

Expected: PASS.

Commit: `git commit -m Train-conditional-experts-with-balanced-streams`

---

### Task 6: Honest File-Macro Evaluation and Release Gates

**Files:**
- Create: `evaluation.py`
- Modify: `distance_metrics.py`
- Modify: `train.py`
- Create: `tests/test_evaluation.py`
- Modify: `tests/test_distance_metrics.py`

**Interfaces:**
- Produces: `evaluate_predictions(records) -> dict`
- Produces: `file_bootstrap_metrics(records, iterations=1000, seed=0) -> dict`
- Produces: `evaluate_release(candidate, baseline) -> tuple[bool, list[str]]`

- [ ] **Step 1: Write failing file-macro and subgroup tests**

```python
def test_file_macro_prevents_one_large_file_from_dominating():
    metrics = evaluate_predictions(large_easy_file_plus_small_failed_file())
    assert metrics["type_piece_accuracy"] > metrics["type_file_macro_accuracy"]


def test_distance_reports_type_daylight_and_band_groups():
    metrics = evaluate_predictions(mixed_condition_records())
    assert "NCG/day/0-600km" in metrics["distance_subgroups"]
    assert "NCG/night/0-600km" in metrics["distance_subgroups"]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_evaluation.py tests/test_distance_metrics.py -q`

Expected: FAIL because `evaluation.py` does not exist.

- [ ] **Step 3: Implement record-based evaluation**

Define one record per piece with file ID, true/predicted type, rejection status, interval bounds, predicted distance, daylight, and quality. Report raw type metrics, accepted precision/recall/coverage, oracle and routed distance MAE, within-100/200 km, per-file macro values, subgroup values, and file-resampled confidence intervals. For broad labels, interval error is `max(low - prediction, 0, prediction - high)`; release distance gates use exact 100 km labels only.

- [ ] **Step 4: Encode the approved release gates**

```python
RELEASE_GATES = {
    "min_per_type_precision": 0.95,
    "min_coverage": 0.80,
    "min_macro_recall": 0.90,
    "min_exact_within_200": 0.85,
    "min_per_type_exact_within_200": 0.75,
}
```

Require split-hash equality before comparing candidate and baseline. Reject unexplained candidate regression in any reported type/daylight/coarse-distance subgroup.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_evaluation.py tests/test_distance_metrics.py tests/test_train.py -q`

Expected: PASS.

Commit: `git commit -m Add-file-macro-release-evaluation`

---

### Task 7: Precision-Constrained IC Rejection

**Files:**
- Modify: `open_set.py`
- Modify: `train.py`
- Modify: `tests/test_open_set.py`

**Interfaces:**
- Extends: `fit_rejection_policy(logits, features, labels, quality, target_precision=0.95, min_coverage=0.80)`
- Extends: `decode_with_rejection(logits, features, quality, policy)`
- Policy stores per-type probability, margin, feature-distance, and minimum-quality thresholds plus calibration split hash.

- [ ] **Step 1: Write failing quality and precision tests**

```python
def test_low_quality_piece_is_rejected_after_confident_type_prediction():
    policy = permissive_policy_with_quality_floor(0.4)
    decoded = decode_with_rejection(LOGITS, FEATURES, torch.tensor([0.1]), policy)
    assert decoded["final_type"].item() == -1
    assert decoded["reason"][0] == "low_quality"


def test_fit_policy_maximizes_coverage_subject_to_per_type_precision():
    policy = fit_rejection_policy(*calibration_fixture(), target_precision=0.95)
    decoded = decode_fixture(policy)
    assert min(decoded.per_type_precision) >= 0.95
    assert decoded.coverage == max_feasible_coverage()
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_open_set.py -q`

Expected: FAIL because current policy has no quality input or split binding.

- [ ] **Step 3: Implement validation-only threshold search**

Temperature-calibrate logits first. For each predicted type, search observed probability, margin, standardized feature distance, and quality cut points. Select the highest-coverage combination meeting target precision; fail calibration when a type cannot meet the constraint rather than defaulting to acceptance. Record rejection reasons in stable priority order.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_open_set.py tests/test_train.py -q`

Expected: PASS.

Commit: `git commit -m Calibrate-precision-constrained-IC-rejection`

---

### Task 8: Production Checkpoint and Inference Path

**Files:**
- Modify: `classify.py`
- Modify: `models.py`
- Modify: `tests/test_classify.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- New checkpoint schema: `conditional_expert_v1`
- New CLI mode: `python classify.py --model .\weights\conditional\candidate.pt --input_dir ..\train_data\NCG\day\day_0-100km --output_dir .\smoke_classified`; architecture is inferred from checkpoint metadata.
- CSV adds `prob_NCG`, `prob_NNBE`, `prob_PCG`, `prob_PNBE`, `expected_distance_km`, `distance_low_km`, `distance_high_km`, `daylight`, `snr_score`, `clipping_fraction`, `model_version`, and `split_hash`.

- [ ] **Step 1: Write failing checkpoint and decode tests**

```python
def test_conditional_checkpoint_routes_type_context_and_distance():
    prediction = decode_conditional_batch(make_checkpoint(), make_model_outputs())
    assert prediction[0]["final_type"] == "NCG"
    assert prediction[0]["expected_distance_km"] == pytest.approx(450, abs=50)
    assert prediction[0]["distance_low_km"] <= prediction[0]["distance_high_km"]


def test_rejected_piece_has_no_distance_but_keeps_raw_probabilities():
    prediction = decode_conditional_batch(make_rejected_outputs())[0]
    assert prediction["final_type"] == "IC"
    assert prediction["expected_distance_km"] is None
    assert set(prediction["type_probabilities"]) == {"NCG", "NNBE", "PCG", "PNBE"}
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_classify.py -q`

Expected: FAIL because the schema and decoder are missing.

- [ ] **Step 3: Implement one-pass streaming inference**

Read bounded raw-piece batches, derive local/global/context/quality arrays once, execute `forward_with_features` once, apply calibrated rejection, route accepted pieces to one distance expert, and stream audit rows and byte-identical output pieces. Fail clearly on missing schema, context metadata, calibration policy, or incompatible state-dict shapes. Retain legacy loaders only for explicit baseline comparison.

- [ ] **Step 4: Add raw-byte and bounded-memory regression tests**

```python
def test_conditional_inference_preserves_every_piece_byte(tmp_path):
    source = write_synthetic_multi_piece_lig(tmp_path)
    run_conditional_inference(source, tmp_path / "out")
    assert sorted(read_raw_pieces(source)) == sorted(read_all_output_raw_pieces(tmp_path / "out"))


def test_inference_stream_never_materializes_whole_directory(monkeypatch):
    tracker = BatchTracker(limit=32)
    run_conditional_inference(MANY_FILES, OUT, batch_size=32, tracker=tracker)
    assert tracker.max_seen <= 32
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_classify.py tests/test_models.py -q`

Expected: PASS.

Commit: `git commit -m Add-conditional-expert-inference`

---

### Task 9: Real-Data Audit, Baselines, Cleanup, and Final Verification

**Files:**
- Create: `audit_data.py`
- Create: `benchmark.py`
- Modify: `AGENTS.md`
- Delete after replacement verification: `compare_wwlln.py`
- Modify: `data/training_manifest.py`
- Modify: `tests/test_training_manifest.py`

**Interfaces:**
- `audit_data.py --task_data ..\train_data --output .\weights\conditional\data_audit.json` performs no training and writes counts, intervals, context, quality, files, and proposed split hashes.
- `benchmark.py --split_manifest .\weights\conditional\split_manifest.json --model old=.\weights\old\model.pt --model candidate=.\weights\conditional\candidate.pt --output .\weights\conditional\benchmark.json` evaluates checkpoints on the identical locked split.

- [ ] **Step 1: Write failing CLI smoke tests**

```python
def test_audit_cli_writes_reproducible_split_hashes(tmp_path):
    first = run_audit(tmp_path, seed=42)
    second = run_audit(tmp_path, seed=42)
    assert first["split_hashes"] == second["split_hashes"]


def test_benchmark_rejects_metrics_from_another_split(tmp_path):
    with pytest.raises(ValueError, match="split hash"):
        compare_model_metrics(candidate_for("a"), baseline_for("b"))
```

- [ ] **Step 2: Implement audit and benchmark entry points**

Both CLIs call library functions from Tasks 1, 2, and 6. `benchmark.py` adapts the five-class old checkpoint by evaluating only trusted four-class samples and records raw IC predictions as rejected. No hard-coded paths, dates, station coordinates, or model names are allowed.

- [ ] **Step 3: Run the real-data audit without training**

Run: `python audit_data.py --task_data ..\train_data --output .\weights\conditional\data_audit.json`

Expected: four trusted types, zero trained IC pieces, no cross-split files, recorded exact/broad interval counts, and non-empty day/night summaries. Review every reported split deficit before training.

- [ ] **Step 4: Remove superseded split code and hard-coded comparison script**

After `rg` confirms no runtime callers, remove `piece_time_split_manifest`, `validate_piece_split_isolation`, and `validate_piece_split_coverage` plus their piece-overlap tests. Delete `compare_wwlln.py` only after `benchmark.py` covers checkpoint comparison. Keep legacy model classes required to load immutable baselines.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
python -m pytest -q
python -m compileall -q .
python audit_data.py --task_data ..\train_data --output .\weights\conditional\data_audit.json
python train.py --task_data ..\train_data --output .\weights\conditional --epochs 1 --type_samples_per_epoch 2048 --distance_samples_per_epoch 2048 --no_init
python classify.py --model .\weights\conditional\candidate.pt --input_dir ..\train_data\NCG\day\day_0-100km --output_dir .\smoke_classified --batch_size 256
```

Expected: tests and compilation pass; audit shows file isolation; bounded training completes on CUDA without old-model initialization; inference preserves the exact number and bytes of input pieces.

- [ ] **Step 6: Train candidates and compare on the locked test**

Train corrected-alignment, multi-scale, interval-loss, context, and rejection ablations against validation only. Select the final configuration before evaluating test. Run `benchmark.py` once on old, current, and final candidates. Promote only when all approved gates pass; otherwise keep deployment unchanged and write the limiting subgroup to `candidate_metrics.json`.

- [ ] **Step 7: Update contributor commands and commit**

Document the new audit, training, benchmark, and inference commands in `AGENTS.md`, including the no-IC and file-isolation rules.

Commit: `git commit -m Verify-and-document-conditional-classifier`

---

## Self-Review

- Spec coverage: manifest intervals, day/night context, file isolation, multi-scale polarity-preserving input, conditional experts, interval loss, balanced training, IC rejection, file-macro evaluation, release gates, byte-preserving inference, cleanup, and real-data verification are each assigned to a task.
- Placeholder scan: the plan contains no deferred implementation markers; each code-changing task names concrete interfaces, tests, commands, and expected outcomes.
- Type consistency: manifest interval endpoints flow through `LightningPieceDataset`, `interval_distance_loss`, evaluation records, checkpoint decoding, and CSV output using the same `distance_low_km`/`distance_high_km` names. Model inputs consistently use `local`, `global_view`, and three-value `context` tensors.
