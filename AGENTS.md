# Repository Guidelines

## Project Structure & Architecture

This repository classifies binary `.lig` lightning waveforms with PyTorch.

- `train.py` launches the conditional-expert training pipeline; `conditional_pipeline.py` owns split, training, calibration, and checkpoint orchestration.
- `classify.py` performs bounded, byte-preserving production inference and writes `predictions.csv`.
- `models.py`, `training_engine.py`, `distance_ordinal.py`, `open_set.py`, and `evaluation.py` contain model, loss, rejection, and release logic.
- `data/` contains LIG parsing, signed local/global preprocessing, manifests, file-isolated splitting, datasets, sampling, and audit artifacts.
- `audit_data.py` validates data without training. `benchmark.py` compares structured checkpoints on one locked split.
- `tests/` uses synthetic fixtures only. Local datasets and weights belong under `../train_data/` and `weights/`; both are ignored by Git.

Training includes only `NCG`, `NNBE`, `PCG`, and `PNBE`. `IC` is an inference-only rejection result, never a training class.

## Build, Test, and Development Commands

Install dependencies in the active `ligclassify` environment:

```powershell
pip install numpy scipy torch scikit-learn tqdm pytest
```

Run the standard workflow:

```powershell
python audit_data.py --task_data ..\train_data --output .\weights\conditional\data_audit.json
python train.py --task_data ..\train_data --output .\weights\conditional --no_init
python classify.py --model .\weights\conditional\candidate.pt --input_dir <lig-dir> --output_dir .\classified
python benchmark.py --split_manifest .\weights\conditional\split_manifest.json --model old=.\weights\old\model.pt --model candidate=.\weights\conditional\candidate.pt
python -m pytest -q
python -m compileall -q .
```

Random initialization is the default. Use `--init_model` only for an explicit warm start and `--resume .\weights\conditional\latest.pt` only for exact continuation. Source files are never shared across train, validation, and test; splits balance type, daylight, and coarse distance without forcing year boundaries. Promotion requires calibrated rejection, all release gates, and same-split baseline metrics.

## Coding Style & Testing

Use four-space indentation, `snake_case` functions/variables, `PascalCase` classes, and `UPPER_CASE` constants. Add type hints and short docstrings to reusable public helpers. Keep CLI orchestration outside `data/`.

Name tests `test_<module>.py`. Cover parsing, file isolation, interval labels, polarity, sampler balance, expert routing, calibration, checkpoint compatibility, bounded inference, and raw-byte preservation. Never add real waveforms as fixtures.

## Commits, Pull Requests, and Data Safety

Use concise imperative commit subjects, such as `Add conditional expert inference`. Pull requests must list verification commands, required local data/weights, split hashes, and relevant validation/test metrics. Never commit `.lig` data, checkpoints, generated classifications, credentials, or machine-specific paths.
