# Hierarchical Five-Class Known-Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task by task. Do not use
> subagents unless the user explicitly authorizes delegation.

**Goal:** Replace the overconfident flat type classifier with a hierarchical
five-class type model that retains IC training, protects NCG/NNBE/PCG/PNBE
recall, and routes accepted known-class pieces to the existing independent
distance experts.

**Architecture:** A new type-only checkpoint uses a shared local/global
encoder, an IC-versus-known gate, a conditional four-known-class head, and
multiple learned prototypes per known class. Training uses source-file-grouped
evaluation splits, equal per-class batches without within-epoch replacement,
two safe waveform views, and known-only metric/consistency losses. Inference
runs both views and permits stable known-class evidence to override the IC
gate; otherwise the result is IC. The four distance-role checkpoints retain
the existing `FiveClassNet` implementation and routing contract.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, SciPy, scikit-learn, tqdm, pytest.

## Global Constraints

- Work only in the runtime and test files listed in `AGENTS.md`.
- Never read real LIG files from tests; use synthetic temporary fixtures.
- Never change or delete `../train_data`, weights, or classification outputs.
- Preserve waveform polarity and byte-exact inference regrouping.
- Preserve `five_class_v1` and `legacy_five_class` checkpoint inference.
- Keep random initialization; do not add warm starts, OOF/CV, release gates,
  open-set rejection, or a standalone four-class classifier.
- Do not use `GZ_20160708000040.lig#7` as a training or test fixture.
- After each task, run the focused tests and inspect `git diff` before commit.

---

### Task 1: Replace piece leakage with deterministic source-grouped splits

**Files:**

- Modify: `data/split.py`
- Modify: `audit_data.py`
- Test: `tests/test_split.py`
- Test: `tests/test_audit.py`

**Step 1: Write failing split tests**

Add synthetic sources containing multiple pieces and mixed daylight strata.
Assert:

```python
assignment = assign_piece_splits(table, seed=42)
owners = {
    source_id: set(assignment.partition[table.source_index == source_id])
    for source_id in range(len(table.sources))
}
assert all(len(values) == 1 for values in owners.values())
assert split_artifact(table, assignment)["schema"] == (
    "source_grouped_stratified_split_v2"
)
```

Also assert deterministic reconstruction, different seeds can change
ownership, every piece is owned exactly once, and validation raises when one
source is manually spread across partitions. Replace old tests that require
every piece stratum to populate all partitions; a stratum with fewer than
three source files must instead be reported as evaluation-limited.

**Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest -q tests/test_split.py tests/test_audit.py
```

Expected: failures because the current splitter assigns individual pieces and
emits `piece_stratified_split_v1`.

**Step 3: Implement grouped iterative stratification**

In `data/split.py`:

- Build one vector per `source_index`, counting pieces in
  `(type_index, daylight, distance_bin)` cells; use distance `-1` for IC.
- Sort groups deterministically by decreasing rare-stratum contribution,
  decreasing piece count, then `_rank(seed, relative_path)`.
- Greedily assign each whole source to train/validation/test. Score candidate
  partitions by normalized squared error from the 70/15/15 per-stratum target
  plus normalized total-piece error; break ties by the seeded source rank and
  partition index.
- Keep `SplitAssignment.partition` piece-aligned so downstream datasets do not
  change.
- Reconstruct the exact expected assignment in `validate_piece_split`.
- Reject any assignment where one `source_index` has multiple owners.
- Emit `source_grouped_stratified_split_v2`, source counts, piece counts,
  per-stratum source support, and `insufficient_source_groups`.

Do not silently move a source after assignment. The same manifest and seed
must always produce byte-identical `split.json`.

**Step 4: Extend the audit output**

Report source and piece counts per partition and list strata without enough
independent source files. Do not treat limited support as a data-format error.

**Step 5: Run focused tests**

```powershell
python -m pytest -q tests/test_split.py tests/test_audit.py
```

Expected: pass.

**Step 6: Commit**

```powershell
git add data/split.py audit_data.py tests/test_split.py tests/test_audit.py
git commit -m "fix: group evaluation splits by source file"
```

---

### Task 2: Add equal, no-replacement type sampling and paired safe views

**Files:**

- Modify: `data/sampling.py`
- Modify: `data/dataset.py`
- Modify: `data/preprocess.py`
- Test: `tests/test_sampling.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write failing sampler tests**

Create five synthetic class pools of unequal size. Assert that a type epoch:

- contains exactly 20% of each type;
- contains no duplicate table position;
- is deterministic for one seed/epoch and changes across epochs;
- rejects a requested per-class quota larger than the smallest class;
- assigns two distinct deterministic augmentation seeds per request.

Change `SampleRequest` to:

```python
@dataclass(frozen=True)
class SampleRequest:
    position: int
    augmentation_seeds: tuple[int, int]
```

**Step 2: Write failing preprocessing and dataset tests**

Add a waveform with one isolated noise spike and one wider pulse. Assert the
new energy-envelope center selects the wider pulse, preserves its sign, and
produces `(1, 8000)` local plus `(1, 2000)` global views.

For a training `SampleRequest`, assert `FiveClassDataset` returns:

```python
{
    "local", "global", "local_alt", "global_alt",
    "daylight", "type_label", "distance_bin", ...
}
```

Validation and test items must not contain alternate tensors. Assert paired
views differ but never invert polarity by multiplying by a negative gain.

**Step 3: Run tests to verify failure**

```powershell
python -m pytest -q tests/test_sampling.py tests/test_dataset.py tests/test_preprocess.py
```

Expected: failures for the 60/10 prior, replacement sampling, one seed, one
view, and peak-only local centering.

**Step 4: Implement balanced sampling**

Set `TYPE_PRIOR = (0.20, 0.20, 0.20, 0.20, 0.20)`. Require
`num_samples % 5 == 0`. For each class, deterministically shuffle its available
positions and take exactly `num_samples // 5` without replacement. Retain the
existing daylight/distance balancing only as a deterministic ordering
preference; it must never duplicate a piece to fill a cell.

Derive paired seeds with domain-separated SHA-256 payloads (`view=0`,
`view=1`). Keep `DistanceExpertSampler` behavior unchanged.

**Step 5: Implement versioned preprocessing**

Extend `PreprocessConfig` with:

```python
local_center_mode: str = "peak_abs_v1"
local_energy_window: int = 128
```

Keep `peak_abs_v1` byte-compatible for old checkpoints. Add
`energy_envelope_v2`, which subtracts the row median, computes a moving mean of
squared amplitude over `local_energy_window`, centers the 8000-sample crop on
the highest-energy window, and never changes waveform sign. Training of all
new roles will explicitly select `energy_envelope_v2`.

**Step 6: Implement paired dataset output**

Read raw bytes once, create two independently augmented signed arrays using
the paired seeds, and preprocess each. Update `collate_batch` to stack
alternate keys only when every item supplies them; reject partially paired
batches.

**Step 7: Run focused tests**

```powershell
python -m pytest -q tests/test_sampling.py tests/test_dataset.py tests/test_preprocess.py
```

Expected: pass.

**Step 8: Commit**

```powershell
git add data/sampling.py data/dataset.py data/preprocess.py tests/test_sampling.py tests/test_dataset.py tests/test_preprocess.py
git commit -m "feat: balance type sampling and add paired waveform views"
```

---

### Task 3: Implement the hierarchical type model

**Files:**

- Modify: `models.py`
- Test: `tests/test_models.py`

**Step 1: Write failing model-contract tests**

Add tests for a new factory:

```python
model = create_hierarchical_type_model(
    base_channels=16,
    embedding_dim=64,
    prototypes_per_class=4,
)
output = model(local, global_view, daylight)
assert output.type_logits.shape == (batch, 5)
assert output.gate_logits.shape == (batch, 2)
assert output.known_logits.shape == (batch, 4)
assert output.prototype_logits.shape == (batch, 4, 4)
assert output.local_known_logits.shape == (batch, 4)
assert output.global_known_logits.shape == (batch, 4)
assert output.embedding.shape == (batch, 64)
```

Assert `output.type_logits.exp().sum(1)` is one when interpreted as joint log
probabilities, gradients reach both local and global branches, prototypes are
unit-normalized for matching, and invalid shapes/configuration raise clear
errors. Retain all current `FiveClassNet` tests unchanged.

**Step 2: Run test to verify failure**

```powershell
python -m pytest -q tests/test_models.py
```

Expected: import/factory failures.

**Step 3: Add explicit model types**

In `models.py`, add:

```python
HIERARCHICAL_TYPE_ARCHITECTURE = "hierarchical_five_class_v2"

@dataclass(frozen=True)
class HierarchicalTypeOutput:
    type_logits: torch.Tensor
    gate_logits: torch.Tensor
    known_logits: torch.Tensor
    prototype_logits: torch.Tensor
    prototype_scores: torch.Tensor
    local_known_logits: torch.Tensor
    global_known_logits: torch.Tensor
    embedding: torch.Tensor
```

Add a dual-scale encoder that returns local, global, and fused features. Fuse
with a learned sigmoid gate over local/global/daylight features rather than a
plain concatenation only.

Add `KnownPrototypeMatcher` with shape
`[4, prototypes_per_class, embedding_dim]`, cosine matching, a bounded learned
temperature, and log-sum-exp aggregation across prototypes.

Add `HierarchicalTypeNet` with:

- an IC/known gate;
- one four-known-class discriminative head;
- local and global four-class diagnostic heads;
- a normalized metric embedding;
- four-by-K prototypes;
- joint five-class log probabilities:
  `log P(IC)` and `log P(known) + log P(known class)`.

The known distribution combines discriminative and prototype logits with a
fixed, constructor-recorded `prototype_logit_weight=0.25`. Expose
`forward_type()` returning `(type_logits, embedding)` for generic evaluation,
and `forward_hierarchical()`/`forward()` returning the full dataclass.

Do not add distance heads to this model.

**Step 4: Run focused tests**

```powershell
python -m pytest -q tests/test_models.py
```

Expected: pass for new and retained models.

**Step 5: Commit**

```powershell
git add models.py tests/test_models.py
git commit -m "feat: add hierarchical five-class type model"
```

---

### Task 4: Add known-only metric and consistency training losses

**Files:**

- Modify: `training.py`
- Test: `tests/test_training.py`

**Step 1: Write failing loss tests**

Add synthetic `HierarchicalTypeOutput` pairs and assert:

- gate targets are IC for label 0 and known for labels 1–4;
- known CE, prototype CE, supervised contrastive, branch CE, and consistency
  receive only labels 1–4;
- changing only IC embeddings does not change known contrastive loss;
- changing a known alternate-view distribution increases consistency loss;
- loss remains finite when a batch contains no IC or only one example of a
  known class;
- gradients reach both encoders, gate, known head, and prototypes.

**Step 2: Run test to verify failure**

```powershell
python -m pytest -q tests/test_training.py
```

Expected: missing loss API and current flat CE behavior.

**Step 3: Implement the loss**

Add a `HierarchicalLossWeights` frozen dataclass with defaults:

```python
gate=1.0
known=1.0
five_class=0.25
prototype=0.25
branch=0.10
contrastive=0.10
consistency=0.10
label_smoothing=0.05
temperature=0.10
```

Implement:

- gate cross-entropy on every sample;
- known/head/prototype/branch CE only for `target > 0`;
- five-class NLL on joint log probabilities for monitoring and gate
  coordination;
- two-view supervised contrastive loss on normalized known embeddings;
- symmetric KL consistency on known-class probabilities for known samples.

Return a mapping with total and every component so the progress bar can show
`gate`, `known`, `metric`, and `consistency` separately.

**Step 4: Branch `train_role_epoch` by model contract**

For role `type`, require paired tensors and call
`forward_hierarchical()` twice when the model architecture is hierarchical.
Keep the existing flat CE path for a retained `FiveClassNet`. Distance-role
training must remain unchanged.

Update progress output to show total loss, type accuracy, known accuracy, and
known-to-IC rate. Keep bounded tqdm refresh behavior.

**Step 5: Run focused tests**

```powershell
python -m pytest -q tests/test_training.py
```

Expected: pass.

**Step 6: Commit**

```powershell
git add training.py tests/test_training.py
git commit -m "feat: train hierarchical type evidence consistently"
```

---

### Task 5: Add hierarchical decisions, calibration, and required metrics

**Files:**

- Modify: `evaluation.py`
- Test: `tests/test_training.py`
- Test: `tests/test_models.py`

**Step 1: Write failing decision tests**

Define synthetic base/alternate outputs covering:

1. gate says known and stable known evidence → known class;
2. gate says IC but both views, a branch, and prototypes stably support NNBE
   → NNBE override;
3. gate says IC and known candidates disagree → IC;
4. gate says known but evidence is unstable → IC;
5. no known labels in calibration → explicit validation error.

Assert decision diagnostics include candidate class, IC gate probability,
known probability, prototype similarity, consistency, branch votes, and a
stable reason string.

**Step 2: Run tests to verify failure**

```powershell
python -m pytest -q tests/test_models.py tests/test_training.py
```

Expected: missing decision/calibration APIs.

**Step 3: Implement decision configuration**

Add frozen dataclasses:

```python
HierarchicalDecisionConfig(
    known_probability_thresholds: tuple[float, float, float, float],
    prototype_similarity_thresholds: tuple[float, float, float, float],
    max_js_divergence: float,
    min_branch_votes: int,
)

HierarchicalDecision(...)
```

The decision rule must always evaluate both views:

- candidate is the mean known distribution argmax;
- both views must select the same candidate;
- mean candidate known probability must meet its class threshold;
- mean candidate prototype similarity must meet its class threshold;
- Jensen–Shannon divergence must be at most the calibrated maximum;
- at least `min_branch_votes` of the four local/global view predictions must
  support the candidate.

If those conditions pass, output the candidate even when the gate selects IC.
Otherwise output IC. This is evidence-based IC fallback, not confidence
rejection of an already accepted known class.

**Step 4: Implement deterministic validation calibration**

Collect paired validation outputs once. Search thresholds only over observed
validation quantiles. For each known class, choose the lowest stable thresholds
that maximize recall while keeping class precision at or above 0.90; break ties
by lower false-to-IC, then lower thresholds. Select the largest admissible
Jensen–Shannon threshold and the smallest branch-vote count that meets the
same precision rule. Store exact selected values; never silently use hardcoded
deployment thresholds when calibration lacks class support.

**Step 5: Extend evaluation metrics**

For hierarchical type evaluation report:

- overall accuracy and five-class macro F1;
- known-class macro recall;
- per-known-class recall and false rejection to IC;
- known-view consistency rate;
- NBE confusion (`NNBE` versus `PNBE`);
- CG confusion (`NCG` versus `PCG`);
- five-by-five confusion matrix;
- sample and source-file counts.

Aggregate file metrics by mean piece probabilities per `source_path`; do not
let a large file count as many independent evaluation units.

Use early-stopping score:

```python
known_macro_recall - 0.50 * max_known_false_to_ic + 0.10 * type_macro_f1
```

This encodes the confirmed known-recall priority without adding a release
gate.

**Step 6: Run focused tests**

```powershell
python -m pytest -q tests/test_models.py tests/test_training.py
```

Expected: pass.

**Step 7: Commit**

```powershell
git add evaluation.py tests/test_models.py tests/test_training.py
git commit -m "feat: calibrate stable known-class overrides"
```

---

### Task 6: Version hierarchical checkpoints and mixed-role bundles

**Files:**

- Modify: `checkpoints.py`
- Test: `tests/test_checkpoints.py`

**Step 1: Write failing checkpoint tests**

Assert:

- a hierarchical type checkpoint round-trips strictly;
- its schema is `hierarchical_five_class_v2`;
- model config and calibrated decision config are required and validated;
- malformed prototype count, thresholds, or tensor shapes are rejected;
- a v2 bundle accepts one hierarchical type checkpoint plus four
  `five_class_v1` distance checkpoints;
- a hierarchical checkpoint cannot occupy a distance role;
- a flat checkpoint cannot occupy the v2 type role;
- existing `five_class_v1`, `legacy_five_class`, and v1 bundle tests still
  pass.

**Step 2: Run test to verify failure**

```powershell
python -m pytest -q tests/test_checkpoints.py
```

Expected: unsupported schema and mixed-bundle rejection.

**Step 3: Add strict schema dispatch**

Add:

```python
HIERARCHICAL_FIVE_CLASS_SCHEMA = "hierarchical_five_class_v2"
HIERARCHICAL_MODEL_BUNDLE_SCHEMA = "hierarchical_model_bundle_v2"
```

Create a separate required-field validator/loader for the hierarchical type
checkpoint. Require `base_channels`, `embedding_dim`,
`prototypes_per_class`, `prototype_logit_weight`, preprocessing config,
decision config, split hash, and role `type`. Build only
`HierarchicalTypeNet`.

Make `save_model_checkpoint` choose schema from the model architecture and
require decision config for hierarchical models. Do not relax tensor-state
strictness.

**Step 4: Preserve preprocessing compatibility**

Allow optional `local_center_mode` and `local_energy_window` in
`five_class_v1`. Missing keys normalize to `peak_abs_v1` and `128`, preserving
old checkpoint behavior. New training records `energy_envelope_v2`.

**Step 5: Add mixed-role bundle v2**

Write v2 when the type role is hierarchical. Validate:

- the type role schema is hierarchical;
- distance roles are `five_class_v1`;
- all roles share split hash, input lengths, filter settings, and the new
  preprocessing settings;
- role names and hashes match exactly.

Continue loading current v1 bundles unchanged. Update bundle forwarding to
return hierarchical diagnostics for the type role without changing distance
head shapes.

**Step 6: Run focused tests**

```powershell
python -m pytest -q tests/test_checkpoints.py
```

Expected: pass.

**Step 7: Commit**

```powershell
git add checkpoints.py tests/test_checkpoints.py
git commit -m "feat: version hierarchical type checkpoints and bundles"
```

---

### Task 7: Integrate hierarchical type training into the CLI

**Files:**

- Modify: `train.py`
- Test: `tests/test_training.py`
- Test: `tests/test_checkpoints.py`

**Step 1: Write failing CLI/config tests**

Assert:

- `--stage type` creates `HierarchicalTypeNet`;
- distance stages still create `FiveClassNet`;
- default type sample count is auto-derived as
  `5 * min(train_piece_count_by_type)`;
- an explicit type sample count must be divisible by five and cannot exceed
  any class pool;
- `--ic_fraction` defaults to and only accepts `0.20`;
- all new roles use `energy_envelope_v2`;
- resume hashes include architecture, paired-view loss weights, prototype
  count, and preprocessing mode;
- type checkpoint is saved only after validation calibration succeeds.

**Step 2: Run tests to verify failure**

```powershell
python -m pytest -q tests/test_training.py tests/test_checkpoints.py
```

Expected: flat type model, 60% IC prior, and absent hierarchical metadata.

**Step 3: Update parser and configuration**

Change:

```text
--type_samples_per_epoch 0     # auto, not unlimited
--ic_fraction 0.20
--embedding_dim 128
--prototypes_per_class 4
--prototype_logit_weight 0.25
--contrastive_weight 0.10
--consistency_weight 0.10
```

Validate all values strictly. `0` is allowed only for automatic type sampling.
Record the resolved sample count, not zero, in `training_config`.

**Step 4: Build role-specific models**

- `type` → random `HierarchicalTypeNet`;
- `NCG`, `NNBE`, `PCG`, `PNBE` → random `FiveClassNet` with only the
  corresponding distance path trainable.

Use paired training views only for type. Use one view for distance training
and all evaluation loaders. Do not initialize any role from another role.

**Step 5: Calibrate and save**

At each type epoch, compute validation hierarchical metrics for early
stopping. After restoring the best state, run calibration on validation,
reevaluate validation/test with the saved decision config, and persist that
config in `type/model.pt` and `metrics.json`.

Print one concise type summary containing validation macro F1, known macro
recall, maximum false-to-IC, consistency, score, and early-stop wait. Distance
logging remains unchanged.

**Step 6: Run focused tests**

```powershell
python -m pytest -q tests/test_training.py tests/test_checkpoints.py
```

Expected: pass.

**Step 7: Commit**

```powershell
git add train.py tests/test_training.py tests/test_checkpoints.py
git commit -m "feat: train hierarchical type stage from random initialization"
```

---

### Task 8: Apply stable hierarchical decisions during inference

**Files:**

- Modify: `data/preprocess.py`
- Modify: `classify.py`
- Modify: `checkpoints.py`
- Test: `tests/test_classify.py`
- Test: `tests/test_checkpoints.py`

**Step 1: Write failing inference tests**

With synthetic checkpoints and LIG fixtures, assert:

- both base and deterministic alternate waveform views are evaluated;
- stable NNBE evidence overrides an IC gate;
- inconsistent evidence remains IC and receives no distance prediction;
- accepted known results route only to their matching distance role;
- raw output piece bytes are exactly equal to input bytes;
- old single-model and old bundle inference remains unchanged;
- CSV contains the added diagnostic columns.

Expected diagnostic fields:

```text
ic_gate_probability
candidate_known_type
known_type_probability
prototype_similarity
local_prediction
global_prediction
consistency_score
decision_reason
```

**Step 2: Run test to verify failure**

```powershell
python -m pytest -q tests/test_classify.py tests/test_checkpoints.py
```

Expected: hierarchical schema unsupported and fields absent.

**Step 3: Add deterministic second inference view**

In `data/preprocess.py`, add a public helper that creates one alternate signed
view using a fixed zero-filled shift of 32 raw samples. It must not add random
noise, change gain, wrap samples, or invert polarity. Preprocess base and
alternate arrays with the checkpoint-recorded configuration.

**Step 4: Extend `Prediction` and decoding**

Keep the current five probabilities and distance fields. Add optional
hierarchical diagnostics. For old schemas, write empty diagnostic CSV values.
For the hierarchical schema, call the shared decision function from
`evaluation.py`; never duplicate threshold logic in `classify.py`.

Unknown timestamps continue to average day/night probabilities. For
hierarchical predictions, evaluate both daylight alternatives before applying
the decision so timestamp fallback cannot bypass consistency checks.

**Step 5: Route bundles after final type decision**

Change bundle inference so distance selection uses the final hierarchical
decision, not `type_logits.argmax`. Run only the accepted candidate's
independent distance checkpoint. `--type_only` skips every distance model.

Keep output folder names and timestamp-derived LIG filenames unchanged.

**Step 6: Run focused tests**

```powershell
python -m pytest -q tests/test_classify.py tests/test_checkpoints.py
```

Expected: pass.

**Step 7: Commit**

```powershell
git add data/preprocess.py classify.py checkpoints.py tests/test_classify.py tests/test_checkpoints.py
git commit -m "feat: infer stable known classes before IC fallback"
```

---

### Task 9: Update audit guidance and verify the complete project

**Files:**

- Modify: `audit_data.py`
- Modify: `AGENTS.md`
- Test: `tests/test_audit.py`
- Test: all locked test files

**Step 1: Add final audit assertions**

Ensure audit output recommends or reports:

- resolved no-replacement type epoch size;
- per-class train availability;
- source-group counts by split;
- known-class distance/daylight support;
- warnings for evaluation-limited source strata.

It must not propose deleting IC or copying real data into the repository.

**Step 2: Update contributor commands**

Keep `AGENTS.md` concise and update architecture/schema descriptions plus the
normal commands:

```powershell
python audit_data.py --task_data ..\train_data
python train.py --task_data ..\train_data --output .\weights\multi_model
python classify.py --input_root "D:/" --start_date 20160702 --end_date 20160709 --output_dir .\classified --model_dir .\weights\multi_model
```

Do not include machine-specific user paths.

**Step 3: Run static and full tests**

```powershell
python -m compileall -q .
python -m pytest -q
```

Expected: compilation succeeds and every locked test passes.

**Step 4: Run a CPU synthetic smoke train**

Use only pytest-generated/synthetic temporary data through the existing test
helpers. Run one small type epoch, calibration, checkpoint reload, and
type-only inference. Expected: finite losses, a valid
`hierarchical_five_class_v2` checkpoint, and byte-exact regrouping.

Do not start a real 50-epoch training run in this implementation task.

**Step 5: Inspect repository safety**

```powershell
git status --short
git diff --check
git ls-files "*.lig" "*.pt" "*.pth"
```

Expected: no tracked real LIG data or weights; `dataset_audit.txt` remains
untracked and untouched.

**Step 6: Commit**

```powershell
git add AGENTS.md audit_data.py tests/test_audit.py
git commit -m "docs: document hierarchical training workflow"
```

**Step 7: Final verification report**

Report:

- exact test count and pass/fail result;
- checkpoint schemas supported;
- resolved training command;
- explicit note that model quality still requires a fresh real-data training
  run and held-out evaluation; do not claim the target metrics before that run.
