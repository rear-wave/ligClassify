# Model experiment and evaluation source

These scripts preserve the experiments previously stored beside ignored local
weights and evaluation outputs. They are research entry points, separate from
the supported training and inference CLI. Run commands from the repository root
with the same dependencies as `train.py`.

## Streaming and guarded verification experiments

Source lives in `research/streaming/`. Generated checkpoints, waveform caches,
logs and reports still go to ignored `weights/streaming_research_v1/`; importing
the modules does not start training or evaluate data. The original local copies
and existing results are retained.

Inputs are local `../train_data/` and the baseline bundles
`weights/multi_model/` and `weights/anchor_moe_best_bundle/`. Neither data nor
checkpoints are included in Git. These research training/evaluation scripts
require CUDA; their latency measurements also exercise CPU inference.

Run the stages in this order when reproducing the experiment with those inputs:

```cmd
python research/streaming/experiment.py --phase all
python research/streaming/diagnose_candidates.py
python research/streaming/robust_calibration.py
python research/streaming/guarded_rescue.py
python research/streaming/verify_guarded_rescue.py
python research/streaming/confirm_guarded_rescue.py
```

`experiment.py` trains the three first-round candidates from random
initialization and compares validation recall, precision, perturbation stability
and latency. `diagnose_candidates.py` identifies which decision constraints
reject correct candidates. `robust_calibration.py` evaluates a stricter
multi-condition calibration control and prepares the separate calibration
evidence used by `guarded_rescue.py`. The latter fits second-model confirmation
limits; `verify_guarded_rescue.py` freezes them and measures additional validation
conditions and CPU latency. `confirm_guarded_rescue.py` performs the final frozen
piece-test confirmation without fitting thresholds on test labels.

The optional `round2.py` runs the inference-aligned training ablation after the
first round. `check_streaming.py` uses the validation cache and baseline bundles
to check completed-piece/batch equivalence, snapshot restoration and full-bundle
CPU timing. Its synthetic timestamps are API fixtures, not continuous-stream
accuracy evidence.

These scripts reuse existing outputs when present. Keep the checkpoint, dataset,
split and cached evidence from the same experiment together; do not reuse an old
cache for a changed dataset or model. The research recipe schema and its signed
confidence limits differ from the production `--type_verifier_config` schema;
do not pass `guarded_rescue_frozen.json` directly to `classify.py`.

See [the experiment protocol](../docs/2026-09-21-streaming-robustness-experiments.md)
and [guarded-verification results](../docs/2026-09-22-guarded-rescue-experiment.md).

## WWLLN distance-bin evaluation

Source lives in `research/wwlln/`. `compute_bin_errors.py` reads classification
prediction CSVs, their corresponding LIG files and WWLLN `AEYYYYMMDD.loc` files.
It uses the station location and matching convention defined in
`evaluate_wwlln.py`, groups by true WWLLN distance in 100 km bins, and reports
overall and per-type distance errors. Reusing it for a different station requires
the correct station coordinates. Matching is not one-to-one.

Create a local configuration following `configs/wwlln_bin_errors.example.json`
and replace its relative example paths with your own paths. Paths are resolved
from the command's working directory. Keep local configurations under `output/`.

```cmd
python research/wwlln/compute_bin_errors.py --config output/wwlln_config.json --output output/wwlln_distance_bins.json
```

Each type's CSV is expected at `TYPE/TYPE_distance_predictions.csv`. Set
`row_source` to `source_path` for contiguous source rows, `source_path_sqlite`
for unsorted source rows, or `output_file` for regrouped classified pieces. The
last two modes create scratch SQLite indexes beside the result. An optional
`manifest` supplies legacy distance-folder aliases.

The historical `merge_bin_errors.py` and `replace_bin_dataset.py` helpers combine
overall additive bin statistics for disjoint periods. Their count columns cover
2016, 2017 and 2021. They omit combined medians and per-type `type_bins`, because
those summaries are not recomputed by these helpers. To obtain valid combined
per-type statistics, list all periods in one `compute_bin_errors.py` config.

```cmd
python research/wwlln/merge_bin_errors.py --base output/period_a.json --extra output/period_b.json --output output/combined.json
python research/wwlln/replace_bin_dataset.py --combined output/combined.json --remove output/period_b.json --add output/period_b_updated.json --label 2021_period_b --output output/combined_updated.json
```

Provide disjoint periods to the merge command and the same period to both
replacement inputs. These aggregate-only helpers cannot detect duplicate
waveforms across periods.
