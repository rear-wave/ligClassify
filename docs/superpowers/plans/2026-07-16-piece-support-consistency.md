# Piece-Based Support Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 100 waveform pieces the single support floor across audit, selection, release, checkpoint validation, inference/benchmark audit records, and contributor workflows.

**Architecture:** `data.cross_validation.MINIMUM_SUPPORTED_PIECES` is the canonical domain constant. Existing support maps keep both file and piece counts, but only piece count determines status. Small adapters add V3 audit evidence without changing prediction decisions or benchmark metrics.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, Git.

## Global Constraints

- A condition is supported if and only if `piece_count >= 100`.
- `file_count` remains audit metadata and never gates training, selection, release, validation, or inference.
- Accuracy, precision, coverage, and within-200-km release thresholds remain unchanged.
- V1/V2 checkpoint and inference behavior remains backward compatible.
- Sparse conditions remain visible and keep their predicted type and distance.
- Do not commit `.lig` files, weights, generated classifications, `.claude/`, or `CLAUDE.md`.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Unify the 100-Piece Support Contract

**Files:**
- Modify: `data/cross_validation.py:152-188`
- Modify: `audit_data.py:8-75,143-158`
- Modify: `evaluation.py:9-23,377-408`
- Modify: `cv_pipeline.py:18-25,347-374`
- Modify: `tests/test_cross_validation.py:1-65`
- Modify: `tests/test_audit_data.py:55-83`
- Modify: `tests/test_evaluation.py:190-275`
- Modify: `tests/test_cv_pipeline.py:12-35,478-565,596-655`

**Interfaces:**
- Produces: `data.cross_validation.MINIMUM_SUPPORTED_PIECES == 100`.
- Produces: `build_support_map(entries, type_names, minimum_pieces=MINIMUM_SUPPORTED_PIECES)`.
- Consumes: support-map rows containing `piece_count`, `file_count`, and `status`.

- [ ] **Step 1: Write failing support-contract tests**

In `tests/test_cross_validation.py`, import the module as well as its helpers and add exact boundary coverage:

```python
import data.cross_validation as cross_validation


def test_support_contract_uses_one_hundred_total_pieces_not_file_count():
    assert getattr(cross_validation, "MINIMUM_SUPPORTED_PIECES", None) == 100

    one_file_supported = make_entries(
        lows=(300,), files_per_condition=1, pieces=(100,)
    )
    many_files_sparse = make_entries(
        lows=(400,), files_per_condition=3, pieces=(33, 33, 33)
    )
    support = build_support_map(
        [*one_file_supported, *many_files_sparse], TYPE_NAMES
    )

    assert support["NCG/day/300-400km"]["file_count"] == 1
    assert support["NCG/day/300-400km"]["piece_count"] == 100
    assert support["NCG/day/300-400km"]["status"] == "supported"
    assert support["NCG/day/400-500km"]["file_count"] == 3
    assert support["NCG/day/400-500km"]["piece_count"] == 99
    assert support["NCG/day/400-500km"]["status"] == "insufficient_support"
```

Replace the old sparse-condition test's explicit 300-piece override with the canonical default and use 99 total pieces:

```python
def test_sparse_condition_is_reported_without_splitting_a_file():
    entries = make_entries(
        lows=(300,), files_per_condition=2, pieces=(49, 50)
    )
    folds = assign_exact_folds(entries, n_folds=3, seed=7)
    support = build_support_map(entries, TYPE_NAMES)
    assert sum(len(rows) for rows in folds.values()) == 2
    assert support["NCG/day/300-400km"]["piece_count"] == 99
    assert support["NCG/day/300-400km"]["status"] == "insufficient_support"
```

In `tests/test_audit_data.py`, make NCG total 99 pieces and every other type exactly 100 pieces. Assert that only NCG is insufficient:

```python
def test_audit_reports_insufficient_support_cell_names(tmp_path):
    totals = {"NCG": 99, "NNBE": 100, "PCG": 100, "PNBE": 100}
    for type_name, pieces in totals.items():
        write_lig(
            tmp_path / type_name / "day" / "300-400km" / "only.lig",
            pieces=pieces,
        )

    result = audit_dataset(tmp_path, seed=9)

    assert result["data_audit"]["insufficient_support_cells"] == [
        "NCG/day/300-400km"
    ]
    assert result["support_map"]["NNBE/day/300-400km"]["status"] == "supported"
```

In `tests/test_cv_pipeline.py`, change `_valid_final_checkpoint()` support rows to `file_count=1`, `piece_count=100`, `status="supported"`. Add a mismatch regression:

```python
def test_validate_final_checkpoint_rejects_piece_status_mismatch():
    checkpoint = _valid_final_checkpoint()
    row = checkpoint["support_map"]["NCG/day/0-100km"]
    row["piece_count"] = 99
    checkpoint["support_map_hash"] = stable_json_hash(checkpoint["support_map"])

    with pytest.raises(ValueError, match="invalid support map"):
        validate_final_checkpoint(checkpoint)
```

In `tests/test_evaluation.py`, import the canonical constant and bind the release dictionary to it:

```python
from data.cross_validation import MINIMUM_SUPPORTED_PIECES


def test_release_gate_uses_canonical_piece_support_floor():
    assert MINIMUM_SUPPORTED_PIECES == 100
    assert (
        evaluation.RELEASE_GATES["minimum_supported_pieces"]
        == MINIMUM_SUPPORTED_PIECES
    )
```

- [ ] **Step 2: Run the focused tests and verify the expected failures**

Run:

```powershell
python -m pytest -q tests/test_cross_validation.py tests/test_audit_data.py tests/test_evaluation.py tests/test_cv_pipeline.py
```

Expected: failures show the missing shared constant, the audit's 300-piece floor, and final checkpoint validation still deriving support from `file_count`.

- [ ] **Step 3: Implement the canonical support constant**

In `data/cross_validation.py`:

```python
MINIMUM_SUPPORTED_PIECES = 100


def build_support_map(
    entries: Iterable[ManifestEntry],
    type_names: Sequence[str],
    minimum_pieces: int = MINIMUM_SUPPORTED_PIECES,
) -> dict[str, dict[str, int | bool | str]]:
    """Aggregate exact condition support using total waveform pieces."""
```

Keep status calculation piece-based:

```python
row["status"] = (
    "supported"
    if row["piece_count"] >= int(minimum_pieces)
    else "insufficient_support"
)
```

In `audit_data.py`, import `MINIMUM_SUPPORTED_PIECES`, call `build_support_map(entries, TYPE_NAMES)` without an override, and render the shared threshold in the CLI message:

```python
print(
    "Insufficient-support cells "
    f"(fewer than {MINIMUM_SUPPORTED_PIECES} waveform pieces):"
)
```

In `evaluation.py`, import the constant and bind the release configuration to it:

```python
from data.cross_validation import MINIMUM_SUPPORTED_PIECES

RELEASE_GATES = {
    # existing gates unchanged
    "minimum_supported_pieces": MINIMUM_SUPPORTED_PIECES,
}
```

Continue using `group["piece_count"]` in `evaluate_release()` and `checkpoint_selection_key()`.

In `cv_pipeline.py`, import the constant and replace the remaining file-based validator branch:

```python
or row.get("status") != (
    "supported"
    if piece_count >= MINIMUM_SUPPORTED_PIECES
    else "insufficient_support"
)
```

- [ ] **Step 4: Run focused tests and verify green**

Run:

```powershell
python -m pytest -q tests/test_cross_validation.py tests/test_audit_data.py tests/test_evaluation.py tests/test_cv_pipeline.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit only the support-contract files**

```powershell
git add data/cross_validation.py audit_data.py evaluation.py cv_pipeline.py tests/test_cross_validation.py tests/test_audit_data.py tests/test_evaluation.py tests/test_cv_pipeline.py
git commit -m "Unify piece-based support contract"
```

---

### Task 2: Preserve V3 Benchmark Audit Evidence

**Files:**
- Modify: `benchmark.py:12-25,181-247`
- Modify: `tests/test_evaluation.py:329-368`

**Interfaces:**
- Produces: `annotate_v3_benchmark_records(bundle, checkpoint) -> list[dict]`.
- Consumes: a conditional prediction bundle after distance calibration and rejection.
- Preserves: all fields used by `evaluate_predictions()` and `file_bootstrap_metrics()`.

- [ ] **Step 1: Write a failing V3 benchmark annotation test**

Add this test to `tests/test_evaluation.py`:

```python
import torch


def test_v3_benchmark_records_retain_support_and_hash_evidence():
    import benchmark

    distance_logits = torch.zeros(2, 4, 30)
    distance_logits[0, 0, 4] = 5.0
    distance_logits[1, 0, 4] = 5.0
    records = [
        {
            "true_type": 0,
            "predicted_type": 0,
            "accepted": True,
            "daylight": True,
            "predicted_distance_km": 450.0,
        },
        {
            "true_type": 0,
            "predicted_type": 0,
            "accepted": False,
            "daylight": True,
            "predicted_distance_km": 450.0,
        },
    ]
    bundle = {"records": records, "distance_logits": distance_logits}
    checkpoint = {
        "schema": "four_class_cv_v3",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "support_map": {
            "NCG/day/400-500km": {
                "file_count": 1,
                "piece_count": 100,
                "status": "supported",
            }
        },
        "fold_manifest_hash": "fold-hash",
        "full_data_hash": "full-hash",
        "rejection_policy": {"calibration_hash": "calibration-hash"},
    }

    assert hasattr(benchmark, "annotate_v3_benchmark_records")
    annotated = benchmark.annotate_v3_benchmark_records(bundle, checkpoint)

    assert annotated is records
    assert annotated[0]["support_status"] == "supported"
    assert annotated[0]["support_file_count"] == 1
    assert annotated[0]["support_condition"] == "NCG/day/400-500km"
    assert annotated[1]["support_status"] == "not_applicable"
    assert annotated[0]["fold_manifest_hash"] == "fold-hash"
    assert annotated[0]["full_data_hash"] == "full-hash"
    assert annotated[0]["calibration_hash"] == "calibration-hash"
    assert [row["accepted"] for row in annotated] == [True, False]
    assert [row["predicted_distance_km"] for row in annotated] == [450.0, 450.0]
```

- [ ] **Step 2: Run the test and verify red**

Run:

```powershell
python -m pytest -q tests/test_evaluation.py::test_v3_benchmark_records_retain_support_and_hash_evidence
```

Expected: FAIL because `benchmark.annotate_v3_benchmark_records` does not exist.

- [ ] **Step 3: Implement additive V3 benchmark annotations**

Add to `benchmark.py`:

```python
def annotate_v3_benchmark_records(
    bundle: dict, checkpoint: dict
) -> list[dict]:
    """Retain V3 support and identity evidence on benchmark records."""
    records = bundle["records"]
    distance_logits = bundle["distance_logits"]
    for row_index, record in enumerate(records):
        type_index = int(record["predicted_type"])
        modal_bin = int(distance_logits[row_index, type_index].argmax().item())
        class_name = (
            checkpoint.get("rejected_type_name", "IC")
            if not record.get("accepted", True)
            else f"{checkpoint['type_names'][type_index]}_"
                 f"{modal_bin * 100}-{(modal_bin + 1) * 100}km"
        )
        support = classify.annotate_distance_support({
            "type_index": type_index,
            "class_name": class_name,
            "modal_distance_bin": modal_bin,
            "daylight": bool(record["daylight"]),
            "type_only": False,
        }, checkpoint)
        record.update({
            "support_status": support["support_status"],
            "support_file_count": support["support_file_count"],
            "support_condition": support["support_condition"],
            "fold_manifest_hash": checkpoint.get("fold_manifest_hash", ""),
            "full_data_hash": checkpoint.get("full_data_hash", ""),
            "calibration_hash": checkpoint.get(
                "rejection_policy", {}
            ).get("calibration_hash", ""),
        })
    return records
```

Call it only for V3 after temperature application and rejection:

```python
if schema == "four_class_cv_v3":
    annotate_v3_benchmark_records(bundle, checkpoint)
records = bundle["records"]
```

- [ ] **Step 4: Run focused benchmark/evaluation tests**

Run:

```powershell
python -m pytest -q tests/test_evaluation.py tests/test_classify.py
```

Expected: all selected tests pass and legacy benchmark tests remain unchanged.

- [ ] **Step 5: Commit the benchmark adapter**

```powershell
git add benchmark.py tests/test_evaluation.py
git commit -m "Retain V3 benchmark audit evidence"
```

---

### Task 3: Correct CLI Validation and Contributor Commands

**Files:**
- Modify: `classify.py:330-341`
- Modify: `tests/test_classify.py:421-425`
- Modify: `tests/test_train.py:43-64`
- Modify: `AGENTS.md:14-42`

**Interfaces:**
- Consumes: `checkpoint_schema(checkpoint)`.
- Produces: consistent confidence-override rejection for V1, V2, and V3.
- Documents: a valid one-epoch joint-stage smoke and exact resume/verification flags.

- [ ] **Step 1: Write the failing V3 confidence-override test**

Add to `tests/test_classify.py`:

```python
def test_v3_checkpoint_rejects_legacy_confidence_override():
    checkpoint = {
        "schema": "four_class_cv_v3",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
    }

    with pytest.raises(ValueError, match="min_type_confidence"):
        classify.validate_type_only_options(
            checkpoint, min_type_confidence=0.85
        )
```

Add a regression test for the documented smoke arguments to `tests/test_train.py`:

```python
def test_one_epoch_smoke_uses_joint_stage_from_epoch_zero():
    args = train.build_arg_parser().parse_args([
        "--max_epochs", "1",
        "--type_focus_epochs", "0",
        "--patience", "1",
    ])

    train._validate_args(args)
```

- [ ] **Step 2: Run the CLI tests and verify the V3 test fails**

Run:

```powershell
python -m pytest -q tests/test_classify.py::test_v3_checkpoint_rejects_legacy_confidence_override tests/test_train.py::test_one_epoch_smoke_uses_joint_stage_from_epoch_zero
```

Expected: the V3 test fails with `DID NOT RAISE`; the smoke-argument regression already passes because it documents existing valid behavior.

- [ ] **Step 3: Extend calibrated-checkpoint option validation**

In `classify.py`, include V3 in the existing schema set:

```python
if (
    checkpoint_schema(checkpoint) in {
        "four_class_rejection_v1",
        "four_class_rejection_v2",
        "four_class_cv_v3",
    }
    and float(min_type_confidence) != 0.0
):
```

- [ ] **Step 4: Correct `AGENTS.md` commands and safety notes**

Change the smoke command to include:

```powershell
--max_epochs 1 --type_focus_epochs 0 --patience 1
```

Add `--num_workers 2` to the full-data `--verify_only` command so its training configuration matches the documented full run.

Replace the stale warm-start/resume paragraph with:

```markdown
Cross-validated training and final training always use random initialization;
non-empty `--init_model` values are rejected. Use `--resume_cv` for exact fold
or final-training continuation. Source files are never shared across folds;
fold assignment balances type, daylight, and exact 100-km intervals. Promotion
requires calibrated rejection and every absolute OOF release gate. Historical
model metrics are reference-only and never authorize promotion.
```

- [ ] **Step 5: Run focused tests and static documentation checks**

Run:

```powershell
python -m pytest -q tests/test_classify.py tests/test_train.py
rg -n "type_focus_epochs 0|verify_only.*num_workers 2|resume_cv|reference-only" AGENTS.md
```

Expected: all focused tests pass and each corrected workflow phrase is present.

- [ ] **Step 6: Commit CLI and documentation corrections**

```powershell
git add classify.py tests/test_classify.py tests/test_train.py AGENTS.md
git commit -m "Correct V3 CLI and workflow guidance"
```

---

### Task 4: Full Verification and Real-Artifact Audit

**Files:**
- Verify: all tracked Python and Markdown files
- Regenerate locally only: `weights/conditional_cv/data_audit.json`, `weights/conditional_cv/support_map.json`, `weights/conditional_cv/fold_manifest.json`
- Do not commit: anything under `weights/`

**Interfaces:**
- Consumes: the completed Tasks 1-3.
- Produces: fresh command evidence and a final worktree review.

- [ ] **Step 1: Run the complete unit suite**

Run in the active environment containing PyTorch and pytest:

```powershell
python -m pytest -q
```

Expected: all tests pass. If WinError 10106 prevents PyTorch import, record the exact failure and do not substitute compile-only evidence.

- [ ] **Step 2: Run compile and CLI checks**

```powershell
python -m compileall -q audit_data.py benchmark.py classify.py conditional_pipeline.py cv_pipeline.py distance_metrics.py distance_ordinal.py evaluation.py models.py open_set.py train.py training_engine.py data tests
python train.py --help
python audit_data.py --help
python classify.py --help
```

Expected: every command exits zero.

- [ ] **Step 3: Regenerate the real-data audit under the 100-piece rule**

```powershell
python audit_data.py --task_data ..\train_data --output .\weights\conditional_cv\data_audit.json
```

Verify 1,142 trusted files, 124,021 pieces, zero IC training pieces, zero cross-fold files, and support statuses derived from 100 pieces. Do not stage generated weight-directory artifacts.

- [ ] **Step 4: Recompute saved OOF evidence read-only**

```powershell
python train.py --task_data ..\train_data --output .\weights\conditional_cv --verify_only --num_workers 2 --no_init
```

Expected: artifact identity checks complete without starting training or promoting a model. A non-passing release report remains a valid verification outcome; do not weaken gates or create `model.pt`.

- [ ] **Step 5: Inspect the final diff and repository state**

```powershell
git diff --check
git status --short --branch
git log -4 --oneline
```

Expected: no whitespace errors; only intentional tracked changes and user-owned `.claude/` / `CLAUDE.md` remain visible. Confirm no `.lig`, `.pt`, generated CSV, or weight artifact is staged.

- [ ] **Step 6: Request final code review**

Use the `requesting-code-review` skill on the implementation range beginning at `674b870`. Address evidenced Critical and Important findings, rerun the affected focused tests, then repeat Steps 1, 2, and 5 before reporting completion.
