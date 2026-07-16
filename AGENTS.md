# Repository Guidelines

## Project Structure & Architecture

This repository classifies binary `.lig` lightning waveforms with PyTorch.

- `train.py` launches the conditional-expert training pipeline; `conditional_pipeline.py` owns split, training, calibration, and checkpoint orchestration.
- `classify.py` performs bounded, byte-preserving production inference and writes `predictions.csv`.
- `models.py`, `training_engine.py`, `distance_ordinal.py`, `open_set.py`, and `evaluation.py` contain model, loss, rejection, and release logic.
- `data/` contains LIG parsing, signed local/global preprocessing, manifests, file-isolated splitting, datasets, sampling, and audit artifacts.
- `audit_data.py` validates data without training. `benchmark.py` compares structured checkpoints on one locked split.
- `tests/` uses synthetic fixtures only. Local datasets and weights belong under `../train_data/` and `weights/`; both are ignored by Git.

Training includes only `NCG`, `NNBE`, `PCG`, and `PNBE`. `IC` is an inference-only rejection result, never a training class. Folds are evaluation-only artifacts; final training uses every trusted file. Historical old-model metrics are contaminated reference data, never an automatic release gate. Unsupported distance cells remain visible in the CSV without suppressing predictions.

## Build, Test, and Development Commands

Install dependencies in the active `ligclassify` environment:

```powershell
pip install numpy scipy torch scikit-learn tqdm pytest
```

Run the standard workflow:

```powershell
python audit_data.py --task_data ..\train_data --output .\weights\conditional_cv\data_audit.json
python train.py --task_data ..\train_data --output .\weights\cv_smoke --max_epochs 1 --type_focus_epochs 0 --patience 1 --samples_per_epoch 2048 --bootstrap_iterations 50 --num_workers 0 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --max_epochs 50 --patience 10 --samples_per_epoch 120000 --max_samples_per_file 512 --bootstrap_iterations 1000 --num_workers 2 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --resume_cv --num_workers 2 --no_init
python train.py --task_data ..\train_data --output .\weights\conditional_cv --verify_only --num_workers 2 --no_init
python classify.py --input_dir <lig-dir> --output_dir .\classified --model .\weights\conditional_cv\model.pt
python -m pytest -q
python -m compileall -q .
```

Cross-validated training and final training always use random initialization;
non-empty `--init_model` values are rejected. Use `--resume_cv` for exact fold
or final-training continuation. Source files are never shared across folds;
fold assignment balances type, daylight, and exact 100-km intervals. Promotion
requires calibrated rejection and every absolute OOF release gate. Historical
model metrics are reference-only and never authorize promotion.

## Coding Style & Testing

Use four-space indentation, `snake_case` functions/variables, `PascalCase` classes, and `UPPER_CASE` constants. Add type hints and short docstrings to reusable public helpers. Keep CLI orchestration outside `data/`.

Name tests `test_<module>.py`. Cover parsing, file isolation, interval labels, polarity, sampler balance, expert routing, calibration, checkpoint compatibility, bounded inference, and raw-byte preservation. Never add real waveforms as fixtures.

v3 checkpoints use `model_config`/`model_state`/`rejection_policy` instead of the v2 `model_state_dict`/`type_rejection` naming. `load_mtl_checkpoint()` accepts both. The CSV writer includes `support_status`, `support_file_count`, `support_condition`, `fold_manifest_hash`, `full_data_hash`, and `calibration_hash` for v3 audit trails.

## Commits, Pull Requests, and Data Safety

Use concise imperative commit subjects, such as `Add conditional expert inference`. Pull requests must list verification commands, required local data/weights, split hashes, and relevant validation/test metrics. Never commit `.lig` data, checkpoints, generated classifications, credentials, or machine-specific paths.
