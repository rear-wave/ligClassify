# ligClassify

`ligClassify` trains and runs a five-class PyTorch classifier for binary LIG
lightning waveforms. The type classes are `IC`, `NCG`, `NNBE`, `PCG`, and
`PNBE`. Non-IC predictions also use one of four type-specific 100 km distance
experts covering 0–3000 km.

## Install

Use the active `ligclassify` environment:

```powershell
pip install numpy scipy torch scikit-learn tqdm pytest
```

## Training data

Keep local data outside this repository. Each `.lig` container may hold up to
512 independent waveform pieces. IC files need no distance label; every non-IC
path must contain one exact aligned interval such as `0-100km`.

```text
../train_data/
  IC/.../*.lig
  NCG/.../0-100km/*.lig
  NNBE/.../500-600km/*.lig
  PCG/.../1200-1300km/*.lig
  PNBE/.../2900-3000km/*.lig
```

The deterministic split assigns pieces—not container files—70%/15%/15% to
train, validation, and test within type/daylight/distance strata. Consequently,
one source file may contribute pieces to all three partitions. Training samples
IC at 60% and each non-IC type at 10%. Daylight comes from each piece timestamp,
and waveform polarity is preserved.

## Commands

```powershell
python audit_data.py --task_data ..\train_data
python audit_data.py --task_data ..\train_data --output .\audit.json --check_duplicates
python train.py --task_data ..\train_data --output .\weights\five_class
python train.py --task_data ..\train_data --output .\weights\five_class --resume .\weights\five_class\last.pt
python classify.py --input_dir <lig-dir> --output_dir .\classified --model .\weights\five_class\model.pt
python -m pytest -q
```

Training always starts from random initialization unless exact same-run state is
restored from `last.pt`. A run writes `model.pt`, `last.pt`, `metrics.json`, and
`split.json`.

## Checkpoints and inference output

Inference accepts exactly two checkpoint schemas:

- `five_class_v1`: the current dual-scale five-class model produced by
  `train.py`.
- `legacy_five_class`: the retained `weights/old/model.pt`, supported for
  inference compatibility only.

Classification uses direct five-class argmax without an IC rejection threshold.
Output LIG files preserve complete source piece bytes. `predictions.csv` records
`source_path`, `piece_index`, `piece_key`, `final_type`, the five `prob_*`
values, `type_confidence`, `distance_bin`, `distance_low_km`,
`distance_high_km`, `expected_distance_km`, `distance_confidence`,
`checkpoint_schema`, `model_sha256`, and `output_file`. IC and `--type_only`
rows leave distance fields blank.
