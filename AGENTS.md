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

The five type labels are `IC`, `NCG`, `NNBE`, `PCG`, and `PNBE`. The type stage
is one hierarchical five-class checkpoint: a shared local/global encoder feeds
an IC/known gate, a conditional `NCG`/`NNBE`/`PCG`/`PNBE` head, and known-class
prototype similarities. IC remains an observed training class. A stable known
match may override the IC gate; otherwise inference returns IC. Non-IC results
route to one of four independently initialized class-specific distance
checkpoints. A validated `bundle.json` assigns checkpoints to roles.

Evaluation splits assign individual pieces deterministically while retaining
type, daylight, and distance stratification. A source LIG file may contribute
pieces to multiple partitions. Type batches use a
20/20/20/20/20 prior and avoid repeated IC sampling within an epoch. Training
starts from random initialization. Use supervised contrastive and augmentation
consistency objectives for the four known classes, but do not force the
heterogeneous IC class into one compact prototype. Candidate type models must
report known-class recall, per-class false rejection to IC, augmentation
consistency, and paired NBE/CG confusions on the piece-level test set. Do
not add warm starts, CV/OOF pipelines, or a standalone four-class-only type
classifier.

## Commands

```powershell
python audit_data.py --task_data ..\train_data
python train.py --task_data ..\train_data --output .\weights\multi_model
python train.py --task_data ..\train_data --output .\weights\multi_model --stage NCG --resume .\weights\multi_model\NCG\last.pt
python classify.py --input_root "D:/" --start_date 20160702 --end_date 20160709 --output_dir .\classified --model_dir .\weights\multi_model
python classify.py --input_dir <lig-dir> --output_dir .\classified --model <model.pt>
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
polarity during preprocessing. Inference supports the hierarchical five-class
schema plus the retained `five_class_v1` and `legacy_five_class` schemas. New
schema loading must be explicit; never silently reinterpret an older
checkpoint.
