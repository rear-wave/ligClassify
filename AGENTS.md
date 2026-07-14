# Repository Guidelines

## Project Structure

This repository is a PyTorch classifier for binary `.lig` lightning waveforms.

- `train.py` builds chronological train/validation/test splits and trains the type and distance heads.
- `classify.py` is the production inference entry point. It preserves original piece bytes and writes an audit CSV.
- `models.py` defines the 1D ResNet architectures. `distance_ordinal.py` handles distance objectives; `open_set.py` handles four-class calibration and rejection.
- `data/` contains parsing, preprocessing, manifest, and sampling code.
- `tests/` contains synthetic unit tests. Do not add real waveform files as fixtures.
- `weights/old/` and `weights/four_class/` contain local deployment weights and are ignored by Git.

The local dataset is `../train_data/`. Training reads only `NCG`, `NNBE`, `PCG`, and `PNBE`; the low-quality `IC` folder is excluded.

## Development Commands

Install dependencies in the active environment:

```powershell
pip install numpy torch scikit-learn tqdm pytest
```

Common commands:

```powershell
python train.py --help
python train.py --task_data ..\train_data --output .\weights\four_class --no_init --baseline_metrics .\weights\old\four_class_baseline.json
python classify.py --input_dir <lig-dir> --output_dir .\classified --type_only --model .\weights\four_class\model.pt
python -m pytest -q
python -m compileall -q .
```

Training keeps the latest dates as a locked temporal test, uses file-level validation, and samples the four researched types equally. Random initialization is the default; `--init_model <path>` explicitly warm-starts compatible encoder tensors. Every run writes `candidate.pt`; promotion also requires the baseline metrics file. During four-class inference, `IC` means rejected/not researched, and calibrated per-type thresholds replace `--min_type_confidence`.

## Style and Tests

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_CASE` constants. Keep CLI orchestration out of `data/`. Add type hints and short docstrings to reusable public helpers.

Name tests `test_<module>.py`. Cover parsing, split isolation/coverage, sampler balance, output shape, checkpoint compatibility, routing, and raw-byte preservation. Before handoff, run the full test suite, compile check, and a bounded smoke test.

## Commits and Data Safety

Use concise imperative commit subjects, such as `Add hybrid distance routing`. Pull requests should state the change, verification commands, required weights/data, and relevant metrics. Never commit `.lig` datasets, checkpoints, credentials, generated classifications, or machine-specific absolute paths.
