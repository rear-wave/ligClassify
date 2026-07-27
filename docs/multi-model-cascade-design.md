# Multi-Model Cascade Design

## Objective

Replace the shared distance representation with five independently trained
roles: one five-class type classifier and one distance model for each of NCG,
NNBE, PCG, and PNBE. Inference must support an inclusive date range, preserve
raw waveform bytes, and regroup pieces by type and then distance.

## Architecture

The type model predicts `IC`, `NCG`, `NNBE`, `PCG`, or `PNBE`. Four independent
distance models each own their distance encoder and class-specific prototype
matcher. They share no trainable parameters with the type model or with one
another. IC pieces bypass distance inference.

The output directory contains a `bundle.json` manifest and five role
directories:

```text
weights/multi_model/
├─ bundle.json
├─ type/model.pt
├─ NCG/model.pt
├─ NNBE/model.pt
├─ PCG/model.pt
└─ PNBE/model.pt
```

Each model file retains the supported `five_class_v1` checkpoint schema.
`bundle.json` assigns one checkpoint to each role and records labels,
preprocessing settings, distance bins, and file hashes. Existing single-model
inference remains supported.

## Training

`train.py` trains all five roles by default and accepts a stage selector for
retraining only `type`, `NCG`, `NNBE`, `PCG`, or `PNBE`. Every role starts from
random initialization.

Splits remain deterministic and mutually exclusive at waveform-piece level.
Type batches use the fixed 60/10/10/10/10 prior. Each distance model sees only
its assigned non-IC class and balances sampling across observed distance bins
and daylight state. The type model stops on validation macro-F1. Distance
models stop primarily on validation within-200-km accuracy, with MAE as a
secondary tie-breaker. Training and test metrics are stored separately for
each role.

## Date-Range Inference

The multi-model interface is:

```cmd
python classify.py --input_root "D:\" --start_date 20160702 --end_date 20160709 --output_dir "E:\Guoxing Yang\typhoon_classified\2016.0702-2016.0709" --model_dir ".\weights\multi_model"
```

Dates are inclusive. The program reads only directories named
`GZ_YYYYMMDD`, ignores corresponding `Index` directories, and reports all
missing requested date directories before inference begins.

Each batch first runs through the type model. IC pieces are written directly.
Each other predicted type is routed only to its matching distance model.

## Output Contract

Non-IC output paths use `TYPE/LLLL-HHHHkm/`; IC uses `IC/`. For example:

```text
NCG/0300-0400km/GZ_20160708142317.lig
IC/GZ_20160708142319.lig
```

Each output file contains at most 512 pieces. Its name is derived from the
timestamp of its first piece, with second precision. A deterministic `_002`,
`_003`, and so on suffix prevents collisions. Inference copies complete raw
piece bytes and preserves waveform polarity.

`predictions.csv` records the requested date, source path, piece index, type
probabilities, predicted type, distance prediction, output path, and hashes of
all five role checkpoints.

## Validation and Failure Handling

The bundle loader verifies that all five files exist, use supported checkpoint
schemas, agree on preprocessing and label/bin definitions, and match recorded
hashes. Missing dates, incomplete bundles, incompatible models, unsafe nested
input/output paths, malformed LIG files, and timestamp failures stop inference
with actionable errors. No partial date-range run begins before preflight
validation succeeds.

Synthetic tests cover role isolation, role-specific training, date discovery,
Index exclusion, routing, output hierarchy, timestamp naming, collision
suffixes, 512-piece rollover, CSV provenance, compatibility with single-model
inference, and byte-exact regrouping.
