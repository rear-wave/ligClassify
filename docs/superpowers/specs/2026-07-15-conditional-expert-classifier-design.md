# Conditional Expert Lightning Classifier Design

## Goal and Scope

Build a reliable classifier for the four researched lightning waveform types: NCG, NNBE, PCG, and PNBE. The four labelled datasets are treated as trusted. Low-quality IC data is excluded from training; `IC` is an inference-only rejection result for waveforms that cannot be classified reliably.

The system must predict both waveform type and propagation distance while accounting for distance-dependent morphology and day/night propagation effects. Calendar year is not a split constraint. The new system replaces the current piece-level evaluation, whose same-file overlap overstates generalization, only after it beats the deployed old model on one locked, file-isolated test set.

## Data Manifest and Splits

Each waveform piece receives a manifest row containing its source file, piece index, timestamp, trusted type, day/night context, distance interval, and quality diagnostics. Exact folders such as `400-500km` produce a 100 km interval. Broad folders such as `0-300km` remain interval-censored labels; they are not converted to a midpoint or discarded.

Corrupt, flat, clipped, malformed, and low-SNR pieces are reported separately. Trusted labels are not silently deleted. Training, validation, and test are assigned at source-file level, so every piece from one `.lig` file has one owner. Assignment is optimized over type, day/night, and coarse distance bands, with fine-bin coverage preserved where the available file count permits. The split is deterministic and records manifest hashes. Validation controls early stopping, calibration, and model selection; test remains sealed until the final candidate is selected.

## Signal Representation

Preprocessing must preserve polarity. The event is centred on the largest absolute excursion instead of the largest positive sample. Robust signed normalization prevents amplitude outliers from dominating without erasing polarity.

The encoder receives two synchronized views:

- a high-resolution local window around the principal pulse;
- a downsampled full-piece view containing longer propagation, reflection, and ringing structure.

Only physically plausible augmentations are allowed: small time shifts and stretches, additive noise, baseline drift, gain variation, and modest filter-cutoff variation. Polarity inversion and arbitrary time reversal are prohibited. Day/night and cyclic local-time features accompany the waveform embedding. Clipping ratio, baseline variation, and SNR-like diagnostics support rejection but do not replace waveform features.

## Model Architecture

A multi-scale residual 1D encoder extracts short-, medium-, and long-duration features from both views and fuses them into a shared representation. A four-class type head predicts NCG, NNBE, PCG, or PNBE. Four independent distance experts are conditioned on the fused representation and time context.

During training, the trusted type routes a sample to its distance expert. During inference, the accepted type prediction selects the expert. Each expert produces an ordered distribution over thirty 100 km bins plus an auxiliary coarse-range prediction. Exact labels use ordinal/distributional supervision. Broad labels use interval likelihood: probability assigned anywhere inside the labelled interval is considered compatible. The decoded result includes expected distance, the most likely 100 km bin, and a calibrated uncertainty interval.

Training uses separate balanced type and distance streams. Sampling balances type, day/night, and coarse distance while limiting the number of pieces contributed by one source file per epoch. Training begins with stable type representation learning, then adds the distance experts, and finally fine-tunes the complete network. Cross-condition consistency encourages the type embedding to retain morphology shared across distance and day/night without forcing the distance representation to discard those effects.

## Rejection and Inference

IC is fitted after four-class training from file-isolated validation predictions. A piece is rejected when calibrated type probability, top-two margin, feature-space support, waveform quality, or optional multi-seed agreement is insufficient. Per-type thresholds maximize coverage subject to the required precision; merely raising one global probability threshold is not a valid correction for class bias.

Inference preserves the original piece bytes and writes an audit row containing source identity, raw and final type, all type probabilities, rejection reason, day/night context, quality diagnostics, routed expert, expected distance, selected bin, uncertainty interval, and model/checkpoint version. Invalid files and timestamps are recorded explicitly instead of being skipped silently.

## Evaluation and Release Gates

Type evaluation reports raw four-class accuracy, macro F1, confusion matrix, and per-class precision/recall. Rejection evaluation adds accepted coverage and rejected counts by reason. Distance evaluation reports MAE, within-100 km, within-200 km, and ordered-bin scores globally and by type, day/night, and distance band. Metrics are reported both per piece and macro-averaged per source file, with file-level bootstrap confidence intervals. Oracle-type and end-to-end routed distance results are kept separate.

The initial release gates are:

- at least 95% accepted precision for every researched type;
- at least 80% overall researched-type coverage;
- at least 90% macro recall;
- at least 85% overall within-200 km accuracy on exact labels;
- at least 75% within-200 km accuracy for every type;
- improvement over the old model on the identical locked test set, with no unexplained severe day/night or distance-band regression.

If the data cannot support a gate, the release is rejected and the limiting subgroup is reported. Test leakage or excessive IC rejection must never be used to manufacture a passing result.

## Implementation and Verification

Development proceeds through auditable ablations: honest old/current baselines, corrected event alignment, multi-scale inputs, interval distance loss, time conditioning, and rejection calibration. A component remains only when it improves file-level validation metrics consistently. Multiple seeds are ensembled only if the measured gain justifies inference cost.

Tests focus on binary parsing and byte preservation, deterministic file isolation, polarity-preserving preprocessing, interval-loss boundaries, balanced sampling, expert routing, rejection calibration, checkpoint schema, and bounded end-to-end inference. Obsolete tests and duplicate model paths may be removed after their supported behavior is either replaced or intentionally retired. Old and current checkpoints remain immutable baselines until the new candidate passes release gates; unrelated user data and generated classifications are never deleted.
