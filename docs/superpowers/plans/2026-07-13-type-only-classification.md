# Type-Only Classification Implementation Plan

**Goal:** Add a single-checkpoint type-only path without changing hybrid inference.

1. Add failing tests for `forward_type`, type decoding, and CLI defaults.
2. Add `forward_type` to both MTL architectures and reuse it in `forward`.
3. Extract `build_arg_parser`, add `--type_only` and `--model`, and branch loading/inference/output routing.
4. Run focused and full tests, compile check, CLI help, and one-file smoke classification.
5. Remove smoke output and provide a one-line PowerShell loop for 2016-07-02 through 2016-07-05.
