# Cascaded Type and Distance Model

## Decision Flow

The runtime uses a strict two-stage cascade:

1. `type_encoder` extracts local, global, and daylight features and predicts one
   of `IC`, `NCG`, `NNBE`, `PCG`, or `PNBE`.
2. IC predictions stop immediately. Each non-IC prediction is routed to its
   corresponding `NCG`, `NNBE`, `PCG`, or `PNBE` distance expert.
3. `distance_encoder` extracts a separate representation. The selected expert
   compares it with 30 learned, normalized distance prototypes and returns
   cosine-similarity logits.

The encoders do not share parameters. This prevents the distance objective from
moving type features toward distance-specific shortcuts and makes it possible
to freeze either stage with `model.set_training_stage("type")` or
`model.set_training_stage("distance")`.

## Why Prototype Matching

Distance is ordered and waveforms from adjacent distance bins should remain
similar. A learned prototype per type and distance bin expresses this structure
more directly than an unrelated linear classifier for every bin. The matching
scale is learned, while normalized features keep scores numerically stable.

## IC Data Requirement

The classifier remains a real five-class model. Full type training must not run
until IC waveform pieces are present: without positive IC examples, the IC
decision boundary is undefined. The four verified non-IC classes can be used to
develop and validate the distance stage, but low confidence is not silently
renamed IC.

## Checkpoint Compatibility

New checkpoints retain the `five_class_v1` runtime schema and record the cascade
through its parameter layout. Training starts from random initialization. The
retained legacy five-class loader remains inference-only; legacy weights are not
loaded into the new encoders.
