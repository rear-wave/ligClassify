# Repository Guidelines

## Structure and Architecture

This repository classifies LIG lightning waveform pieces with PyTorch. The
locked runtime map is `train.py`, `classify.py`, `audit_data.py`, `models.py`,
`training.py`, `evaluation.py`, `checkpoints.py`, and `data/{__init__,lig,
manifest,preprocess,dataset,sampling,split}.py`. CLI orchestration stays outside
`data/`. The locked test map is `tests/conftest.py` plus `test_lig.py`,
`test_manifest.py`, `test_split.py`, `test_preprocess.py`, `test_dataset.py`,
`test_sampling.py`, `test_models.py`, `test_checkpoints.py`, `test_training.py`,
`test_classify.py`, and `test_audit.py`.

The five type labels are `IC`, `NCG`, `NNBE`, `PCG`, and `PNBE`. Splits are
deterministic at waveform-piece level, never file level. Training uses a fixed
60/10/10/10/10 type prior and random initialization. Do not add warm starts,
CV/OOF pipelines, open-set rejection, release gates, support maps, or four-class
training code.

## Commands

```powershell
python audit_data.py --task_data ..\train_data
python train.py --task_data ..\train_data --output .\weights\five_class
python train.py --task_data ..\train_data --output .\weights\five_class --resume .\weights\five_class\last.pt
python classify.py --input_dir <lig-dir> --output_dir .\classified --model .\weights\five_class\model.pt
python -m pytest -q
python -m compileall -q .
```

Install `numpy`, `scipy`, `torch`, `scikit-learn`, `tqdm`, and `pytest` in the
active environment. Use four-space indentation, `snake_case` functions and
variables, `PascalCase` classes, type hints, and short public-helper docstrings.

## Tests and Data Safety

Tests must use synthetic temporary fixtures only. Never commit real `.lig`
waveforms, checkpoints, classifications, credentials, or machine-specific
paths. Local training data belongs only under `../train_data/`; weights belong
under ignored `weights/`. Never modify or delete `../train_data/` or external
classification outputs. Preserve raw piece bytes during inference and waveform
polarity during preprocessing. Inference supports only `five_class_v1` and the
retained `legacy_five_class` checkpoint schemas.
