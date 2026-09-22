# ligClassify

`ligClassify` classifies binary LIG waveform pieces as `IC`, `NCG`, `NNBE`,
`PCG`, or `PNBE`, then estimates distance for the four non-IC types.

Research experiment, calibration and WWLLN distance-bin evaluation scripts are
documented in [research/README.md](research/README.md). Their generated data and
weights remain outside version control.

## Data layout

Each `.lig` container may contain at most 512 independent pieces. Keep training
data outside the repository:

```text
../train_data/
├─ IC/**/*.lig
├─ NCG/**/0000-0100km/*.lig
├─ NNBE/**/0000-0100km/*.lig
├─ PCG/**/0000-0100km/*.lig
└─ PNBE/**/0000-0100km/*.lig
```

Distance folders must use aligned 100 km intervals within 0–3000 km. Splits are
deterministic at waveform-piece level within type, daylight, and distance
strata; pieces from one source file may enter different partitions. Every
complete type batch uses a 20/20/20/20/20 prior. Every model starts from
random initialization.

## Training

Install `numpy`, `scipy`, `torch`, `scikit-learn`, `tqdm`, and `pytest`, then
train the type model and four independent distance models:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --epochs 50 --patience 10 --batch_size 60 --num_workers 0
```

The output contains `type/model.pt`, one model under each non-IC role,
`bundle.json`, and deployment-routed test metrics in `bundle_metrics.json`.
Distance experts train two independently augmented views with a consistency
loss. Early stopping uses the deployed argmax distance bin together with
within-100/200 and MAE metrics; bundle reports retain both argmax and expected
distance metrics.
Retrain or resume one role without changing the others:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --stage NCG --resume .\weights\multi_model\NCG\last.pt
python -u train.py --task_data ..\train_data --output .\weights\type_loss_candidate --stage type --type_loss_profile known_consistency_v1
```

## Date-range inference

This command reads the inclusive range `GZ_20160702` through `GZ_20160709`
below `D:\` and never reads matching `Index` directories:

```cmd
python -u classify.py --input_root "D:/" --start_date 20160702 --end_date 20160709 --output_dir .\classified\2016.0702-2016.0709 --model_dir .\weights\multi_model --batch_size 256 --device cuda
```

Use `--output_type` to write LIG files only for selected predicted classes.
Repeat it when more than one class is required, for example `--output_type
NNBE --output_type PNBE`. `predictions.csv` still records every input piece;
an empty `output_file` marks a prediction excluded from LIG output.

Add `--resume` to continue an interrupted output directory. The command checks
the input list, model hashes, decision configuration, inference mode, and
selected output types before appending, then skips every completed source and
prediction already recorded. Without `--resume`, a non-empty output directory
is rejected to prevent accidental mixing.

Output is grouped as `TYPE/LLLL-HHHHkm/`; IC has no distance level. Each output
file is named from its first piece timestamp, for example
`NCG/0300-0400km/GZ_20160708142317.lig`. Collisions receive `_002`, `_003`, and
so on. Complete raw piece bytes and waveform polarity are preserved.

Existing single-checkpoint inference remains available:

```cmd
python classify.py --input_dir <lig-dir> --output_dir .\classified --model <model.pt>
```

Hierarchical checkpoints can use a versioned external decision configuration
without modifying `model.pt`. Copy `configs/decision_config.example.json`, tune
it on a trusted labeled calibration set, and pass it to either inference mode:

```cmd
python classify.py --input_dir <lig-dir> --output_dir .\classified --model <model.pt> --type_only --decision_config .\configs\decision_config.example.json
python classify.py --input_root "F:/" --start_date 20240926 --end_date 20241003 --output_dir "G:\typhoon_classified\2024.0926-2024.1003" --model_dir .\weights\multi_model --device cuda --decision_config .\configs\decision_nbe_balanced_v4.json --output_type NNBE --output_type PNBE
```

The validated configuration hash is recorded as `decision_config_sha256` in
every prediction row. External thresholds cannot be combined with
`--direct_type`, which bypasses the hierarchical acceptance decision.
Decision configuration v2 stores four class-specific JS-divergence limits in
`max_js_divergences` ordered as NCG, NNBE, PCG, PNBE. Version-1 configurations
remain explicitly supported and expand their single JS limit to all four
classes without changing the old behavior.

`decision_nbe_balanced_v4.json` is the recommended NNBE/PNBE configuration. It
keeps conservative NCG/PCG limits and uses precision-first NNBE/PNBE limits
validated against the human-verified July 2024 NBE set and the untouched test
partition. `decision_nbe_human_202407_v3.json` remains available as a
high-recall experimental configuration and may over-accept deployment data.

The optional guarded verifier can be enabled only with an explicit
`guarded_type_verifier_v1` JSON recipe that binds a second hierarchical type
checkpoint by SHA-256. It confirms only primary IC rejections and routes a
confirmed known class through the corresponding distance expert; accepted
primary labels are never changed. It cannot be combined with temporal context,
`--direct_type`, or an external decision override:

```cmd
python -u classify.py --input_root "F:/" --start_date 20240719 --end_date 20240728 --output_dir .\classified\guarded --model_dir .\weights\multi_model --type_verifier_config .\configs\guarded_type_verifier.json --device cuda
```

This is opt-in research functionality. The verifier checkpoint is not included
in source control, and the recipe must be created from a validated checkpoint
pair. The default command and existing bundles remain unchanged.

## Verification

```cmd
python audit_data.py --task_data ..\train_data
python audit_data.py --task_data ..\train_data --type_checkpoint .\weights\multi_model\type\model.pt --output .\output\type_validation_diagnostics.json
python -m compileall -q .
python -m pytest -q
```

Type constraint diagnostics default to the validation partition. An external
decision configuration is rejected on the test partition so candidate
thresholds cannot be tuned against the final audit set.
