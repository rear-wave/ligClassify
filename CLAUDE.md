# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Data audit
python audit_data.py --task_data ..\train_data

# Compile check and full test suite
python -m compileall -q .
python -m pytest -q

# Train all five roles (type + NCG/NNBE/PCG/PNBE distance)
python -u train.py --task_data ..\train_data --output .\weights\multi_model --epochs 50 --patience 10 --batch_size 64 --num_workers 0

# Resume a single role
python -u train.py --task_data ..\train_data --output .\weights\multi_model --stage NCG --resume .\weights\multi_model\NCG\last.pt

# Date-range inference (requires five-role bundle)
python -u classify.py --input_root "D:/" --start_date 20160702 --end_date 20160709 --output_dir .\classified --model_dir .\weights\multi_model --batch_size 256 --device cuda

# Single-checkpoint inference
python classify.py --input_dir <lig-dir> --output_dir .\classified --model <model.pt>

# Distance-only classification (for pre-typed non-IC data, across year directories)
python -u classify_distance.py --input_root <typhoon_classified> --output_root <typhoon_classified> --model_dir .\weights\multi_model_retrain --batch_size 64 --device cuda [--only "2018"]

# WWLLN accuracy evaluation (against ground-truth .loc lightning location data)
python -u evaluate_wwlln.py --predictions_root <year_distance_dir> --lig_base <year_lig_dir> --wwlln_dir <wwlln_data_dir>

# Distance distribution analysis (from _distance prediction CSVs)
python analyze_distance_distribution.py --root <typhoon_classified> --output <analysis_output_dir>

# Dedup classified .lig pieces by (timestamp, waveform) identity
python scripts_temp_dedup.py --input <classified_dir> --output <dedup_output_dir>

# Fix GZ_unknown.lig source files (patch invalid timestamps)
python fix_unknown_timestamps.py --root <year_dir>
```

Dependencies: `numpy`, `scipy`, `torch`, `scikit-learn`, `tqdm`, `pytest`. No setup.py — install them directly.

## Architecture

### Type classification: hierarchical IC-gate + known-class model

The type stage (`HierarchicalTypeNet`) uses a **GatedDualViewEncoder** that fuses local (8000-sample) and global (2000-sample) waveform branches plus a daylight scalar via learned per-piece fusion weights. From the fused features:

- A **gate head** (2-class) separates IC from known.
- A **known head** (4-class) picks NCG/NNBE/PCG/PNBE.
- Branch heads on local-only and global-only features provide consistency signals.
- An **embedding head** projects to a unit sphere, matched against multiple learned prototypes per known class (`KnownPrototypeMatcher`). Prototype evidence is added to known-class logits.

Final type logits combine `log_softmax(gate)` and `log_softmax(known)` so IC is always reachable. At inference, `decide_hierarchical_types()` accepts a known-class override only when two shifted views agree, multiple thresholds are met (probability, prototype similarity, JS divergence, IC gate max, branch votes), and the IC gate probability is low — otherwise the piece defaults to IC. The decision config is calibrated on the validation set via `calibrate_hierarchical_decision()` and stored in the checkpoint.

The type loss (`hierarchical_type_loss`) uses paired augmented views and seven weighted objectives: gate CE, known CE, five-class CE, prototype CE, branch CE, supervised contrastive, and KL consistency — but does **not** force the heterogeneous IC class into a compact prototype.

### Distance: four independent experts

Each non-IC type has its own `FiveClassNet` with an independent `DualViewEncoder` + `PrototypeDistanceMatcher` (30 learned prototypes per class, one per 100-km bin). The distance loss combines categorical CE with an ordered CDF-matching term.

### Training

`train.py` trains five independent roles sequentially. The type role uses `HierarchicalTypeNet`; distance roles use `FiveClassNet` with `set_training_role()` which freezes the type path. Each role gets its own optimizer, scheduler, deterministic `FiveClassSampler` (type) or `DistanceExpertSampler` (distance), and early stopping. Resumable `last.pt` checkpoints save full training state including RNG seeds, with strict hash validation on resume.

Type batches use a 20/20/20/20/20 prior; the 60% IC fraction is fixed and validated. Type sampling is without replacement within an epoch, stratified by type × daylight × distance-bin. Distance sampling is with replacement, uniform across daylight × distance cells.

### Checkpoint schemas

Three inference schemas detected by `load_model_checkpoint()`:

- `hierarchical_five_class_v2` — current type model with decision config
- `five_class_v1` — flat two-encoder cascade (`FiveClassNet`)
- `legacy_five_class` — original `LegacyMultiTaskResNet`

**Model bundles** (`bundle.json`) tie a type checkpoint + four distance checkpoints together with SHA-256 validation. New schema loading must be explicit; never silently reinterpret an older checkpoint.

### Data pipeline

- `data/lig.py` — LIG binary format: 112-byte header, 32208 or 32464-byte pieces with 16000 uint16 samples. `_validate_source` auto-detects piece size by checking piece#1 timestamp validity. `LigFileIndex` provides lazy file handle management with batch reading; piece count is clamped to fit the payload when the larger (32464) variant is detected. `LigOutputRegrouper` writes byte-exact 512-piece output groups named by first-piece timestamp (or `GZ_unknown` for pieces with invalid timestamps).
- `data/manifest.py` — `build_piece_table()` scans `TYPE_NAME/` directories, parses 100-km distance intervals from path names, builds a `PieceTable` with compact numpy arrays.
- `data/split.py` — deterministic 70/15/15 train/val/test splits **grouped by source LIG file**. Stratified by type × daylight × distance-bin using greedy cost-minimizing assignment.
- `data/preprocess.py` — `preprocess_views()` produces signed local/global views: local centers on energy envelope peak (or max-abs), global downsamples/resamples. Both are robust-normalized (median-center, divide by 95th-percentile absolute). Augmentation is polarity-safe: shift, gain, drift, noise — no sign flip.
- `data/dataset.py` — `FiveClassDataset` wraps a `PieceTable` + split positions. `SampleRequest` indices carry deterministic paired augmentation seeds for training; plain `int` indices skip augmentation.
- `data/sampling.py` — `FiveClassSampler` and `DistanceExpertSampler` use deterministic SHA-256-derived seeds keyed by `seed|epoch|draw_index|position`.

### Inference and analysis

`classify.py` supports single-checkpoint and date-range modes for full type+distance classification. `classify_distance.py` handles batch distance-only classification for pre-typed non-IC data using the four distance experts, with optional `--only` year filtering. Both produce `predictions.csv` and byte-exact output `.lig` files grouped as `TYPE/LLLL-HHHHkm/`.

`evaluate_wwlln.py` evaluates distance classification accuracy against WWLLN ground-truth lightning location data (`.loc` files), using three-filter matching (11ms time + 3500km + 10% speed-of-light consistency) and spherical great-circle distance from Guangzhou station (113.61E, 23.57N) to the matched WWLLN event. `analyze_distance_distribution.py` produces per-100km-bin distribution reports from classification output CSVs.

## Constraints

- Never commit `.lig` files, checkpoints, or machine-specific paths.
- Training data belongs in `../train_data/`; weights belong under ignored `weights/`.
- Never modify or delete `../train_data/` or external classification outputs.
- Inference must preserve raw piece bytes and waveform polarity.
- Tests use synthetic temporary fixtures only (see `tests/conftest.py`).
- No warm starts, CV/OOF pipelines, or standalone four-class-only type classifier.