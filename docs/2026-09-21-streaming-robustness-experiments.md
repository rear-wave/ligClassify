# Lightning streaming robustness experiments

Goal: improve false rejection, false acceptance, robustness to deployment shifts,
and latency from a completed waveform to a type and distance prediction.

The first controlled experiment retains the hierarchical five-class contract and
the deterministic piece-level split (seed 42). It trains three candidates from
random initialization with the same 32-channel width, loss profile, sample
budget, 20-epoch limit and 6-epoch early stopping:

1. Original gated local/global encoder, standard augmentation.
2. Multi-scale temporal convolution encoder, standard augmentation.
3. The same multi-scale encoder with sensor-drift augmentation.

Multi-scale blocks use depthwise convolutions at dilations 1, 4 and 16, with
residual mixing, followed by learned attention pooling and max pooling. Local
and global waveform branches, IC/known gating and known-class prototypes remain.
They operate on a completed waveform, not on samples that have not yet arrived.

Sensor-drift augmentation preserves polarity and adds stronger baseline/noise
variation plus randomized bandwidth. It is experimental; reducing information
too aggressively could hurt classification and must be measured.

Validation selects architectures and hyperparameters. `--defer_test` suppresses
test metrics during the search. A fixed, balanced 2,000-piece validation subset
is also evaluated under clean inputs, 20 dB additive noise, baseline drift,
80 kHz bandwidth, and impulse interference. These are controlled stress tests,
not evidence of generalization to an unseen station or year. Full validation
metrics accompany subset results. Match thresholds to each checkpoint and freeze
them under stress; never recalibrate using corrupted evaluation labels.

Record per-class recall, precision, false rejection to IC, IC/CG false acceptance
as NBE, NBE/CG polarity confusion, and augmented-view agreement. Compare both
existing deployed candidates with the same validation subset. Record CPU/GPU
batch-1 and batch-32 median and p95 latency including preprocessing and paired
type inference. File I/O, queueing, acquisition time and distance routing require
a later end-to-end stream benchmark.

Development script and immutable experiment outputs are local ignored artifacts
under `weights/streaming_research_v1/`. Existing model bundles and external
classifications are preserved. No new candidate is a deployment replacement
until it passes clean/stress validation, independent testing, and replay against
real chronological streams including gaps, station boundaries, and restarts.

Remaining work includes a true continuous-stream replay, causal per-station
context with gap resets, quality/drift monitoring, external labeled data checks,
distance robustness, and the final deployment acceptance report. The earlier
temporal-context gain on selected pieces is not a live-stream accuracy claim.

## First-round results and next controlled experiment (2026-09-22)

All three candidates completed; none passed every predeclared gate. On the
2,000-piece development validation sample, the lightweight hierarchical encoder
had 96.81% clean known-class macro recall and 96.12% worst-stress recall. Its
lower NBE precision was 98.45% clean and 96.38% worst-stress, below the empirical
99%/97% gates. Multi-scale standard augmentation reached 92.44% clean recall;
the stronger-augmentation variant fell to 68.69%. Neither is a replacement.

Constraint diagnostics distinguish representation errors from rejected correct
candidates. The old model's clean conditional known-class candidate accuracy was
99.75%, but its calibrated final known recall was 93.81%. Probability and JS
limits rejected many otherwise correct candidates. For the strong-augmentation
multi-scale candidate, conditional known accuracy itself fell to 92.63%, so
threshold relaxation alone cannot fix that experiment.

Raw training augmentation happens before energy-based local recentering. The
deployment consistency check instead shifts already-preprocessed local/global
views by 16/4 samples. A new opt-in `--inference_view_shift` training flag applies
those same zero-padded shifts to the alternate augmented type view. The default
training and deployment behavior is unchanged. The option is recorded in the
training configuration hash, preventing incompatible resume. It applies only to
hierarchical type training, not distance training.

Two random-initialization controls repeat the 32-channel hierarchical and
multi-scale experiments, changing only this view alignment. Test evaluation
remains deferred during their selection. Unit checks cover default behavior,
exact shifts, unchanged source tensors, gradients, role validation and metadata.

A separate fixed-weight guarded second-model verification experiment is
documented in `2026-09-22-guarded-rescue-experiment.md`. It does not modify the
existing bundle or default classification policy.
