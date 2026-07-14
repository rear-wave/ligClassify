# Type-Only Classification Design

## Goal

Classify `D:\GZ_20160702` through `D:\GZ_20160705` into the five lightning types without loading an old checkpoint, decoding distance, or creating distance-bin folders.

## Interface and Output

`classify.py --type_only --model <checkpoint>` loads one structured MTL checkpoint and uses only its type encoder/head. Normal hybrid inference remains unchanged when `--type_only` is absent.

`--min_type_confidence 0.85` applies a softmax-confidence gate to NCG, NNBE, PCG, and PNBE. A lower-confidence prediction is routed to `uncertained_<TYPE>` and marked `uncertain`; IC is exempt. Both `confidence` and `type_confidence` are written to CSV.

Each source date writes to a separate directory below `E:\Guoxing Yang\typhoon_classified\2016.0702-2016.0709`, containing `IC`, `NCG`, `NNBE`, `PCG`, or `PNBE` folders plus `predictions.csv`. Distance CSV fields remain empty; `type_confidence` is populated and `head_source` is `type`.

Input discovery is recursive within the explicitly supplied non-`Index` directory. Output regrouping preserves every original 32,208-byte piece exactly.

## Verification

Tests prove type-only decoding, confirm `forward_type` bypasses distance projection, preserve existing hybrid routing, and retain raw-piece byte identity. Run the full pytest suite, compile check, CLI help, and a one-file smoke classification before providing the four-date command.
