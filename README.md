# ligClassify

`ligClassify` classifies binary LIG waveform pieces as `IC`, `NCG`, `NNBE`,
`PCG`, or `PNBE`, then estimates distance for the four non-IC types.

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
strata. Type batches use the fixed 60/10/10/10/10 prior. Every model starts
from random initialization.

## Training

Install `numpy`, `scipy`, `torch`, `scikit-learn`, `tqdm`, and `pytest`, then
train the type model and four independent distance models:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --epochs 50 --patience 10 --batch_size 64 --num_workers 0
```

The output contains `type/model.pt` and one model under each non-IC role plus
`bundle.json`. Retrain or resume one role without changing the others:

```cmd
python -u train.py --task_data ..\train_data --output .\weights\multi_model --stage NCG --resume .\weights\multi_model\NCG\last.pt
```

## Date-range inference

This command reads the inclusive range `GZ_20160702` through `GZ_20160709`
below `D:\` and never reads matching `Index` directories:

```cmd
python -u classify.py --input_root "D:\" --start_date 20160702 --end_date 20160709 --output_dir "E:\Guoxing Yang\typhoon_classified\2016.0702-2016.0709" --model_dir .\weights\multi_model --batch_size 256 --device cuda
```

Output is grouped as `TYPE/LLLL-HHHHkm/`; IC has no distance level. Each output
file is named from its first piece timestamp, for example
`NCG/0300-0400km/GZ_20160708142317.lig`. Collisions receive `_002`, `_003`, and
so on. Complete raw piece bytes and waveform polarity are preserved.

Existing single-checkpoint inference remains available:

```cmd
python classify.py --input_dir <lig-dir> --output_dir .\classified --model <model.pt>
```

## Verification

```cmd
python audit_data.py --task_data ..\train_data
python -m compileall -q .
python -m pytest -q
```
