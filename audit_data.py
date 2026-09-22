"""Audit the five-class piece manifest and deterministic split."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from checkpoints import (
    checkpoint_preprocess_config,
    load_decision_config,
    load_model_checkpoint,
    model_sha256,
    validate_decision_config,
)
from data.dataset import FiveClassDataset, collate_batch
from data.lig import read_raw_piece
from data.manifest import TYPE_NAMES, PieceTable, build_piece_table
from data.split import (
    PARTITION_NAMES,
    SplitAssignment,
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
)
from evaluation import (
    HierarchicalDecision,
    HierarchicalDecisionConfig,
    _evaluation_tensors,
    _hierarchical_outputs,
    decide_hierarchical_types,
    decision_config_dict,
)
from models import HIERARCHICAL_TYPE_ARCHITECTURE, HierarchicalTypeOutput


TRAINING_PRIOR = {
    "IC": 0.20,
    "NCG": 0.20,
    "NNBE": 0.20,
    "PCG": 0.20,
    "PNBE": 0.20,
}
TYPE_DIAGNOSTIC_SCHEMA = "hierarchical_constraint_diagnostic_report_v1"
TYPE_CONSTRAINT_SUMMARY_SCHEMA = "hierarchical_constraint_summary_v1"
CONSTRAINT_ORDER = (
    "view_agreement",
    "known_probability",
    "prototype_similarity",
    "ic_gate",
    "js_divergence",
    "branch_votes",
)


def audit_duplicate_waveforms(
    table: PieceTable,
    assignment: SplitAssignment,
) -> dict[str, int | str]:
    """Hash complete raw pieces and reject duplicates crossing partitions."""
    validate_piece_split(table, assignment)
    first_by_digest: dict[str, tuple[int, str]] = {}
    for position in range(len(table)):
        source = table.sources[int(table.source_index[position])]
        piece_index = int(table.piece_index[position])
        identity = table.piece_key(position)
        partition = int(assignment.partition[position])
        digest = hashlib.sha256(
            read_raw_piece(source.path, piece_index)
        ).hexdigest()
        first = first_by_digest.get(digest)
        if first is None:
            first_by_digest[digest] = (partition, identity)
            continue
        first_partition, first_identity = first
        if first_partition != partition:
            raise ValueError(
                "duplicate waveform crosses partitions: "
                f"{first_identity} ({PARTITION_NAMES[first_partition]}) and "
                f"{identity} ({PARTITION_NAMES[partition]})"
            )
    return {
        "algorithm": "sha256_complete_raw_piece",
        "pieces_hashed": len(table),
        "unique_digests": len(first_by_digest),
    }


def _type_counts(table: PieceTable) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(table.type_index == index))
        for index, name in enumerate(TYPE_NAMES)
    }


def _distance_bin_counts(table: PieceTable) -> dict[str, int]:
    counts: dict[str, int] = {}
    for distance_bin in sorted(
        set(int(value) for value in table.distance_bin if int(value) >= 0)
    ):
        low = distance_bin * 100
        counts[f"{low}-{low + 100}km"] = int(
            np.count_nonzero(table.distance_bin == distance_bin)
        )
    return counts


def audit_dataset(
    task_data: str | Path,
    output: str | Path | None = None,
    seed: int = 42,
    check_duplicates: bool = False,
) -> dict[str, object]:
    """Build and validate a piece split, optionally hashing raw waveforms."""
    table, diagnostics = build_piece_table(task_data)
    if not len(table):
        raise RuntimeError(f"No valid .lig pieces found under {task_data}")

    assignment = assign_piece_splits(table, seed=int(seed))
    validate_piece_split(table, assignment)
    artifact = split_artifact(table, assignment)
    strata = artifact["strata"]
    if not isinstance(strata, dict):
        raise TypeError("split artifact strata must be a mapping")

    report: dict[str, object] = {
        "schema": "five_class_data_audit_v1",
        "files": int(diagnostics["files"]),
        "pieces": int(diagnostics["pieces"]),
        "type_counts": _type_counts(table),
        "daylight_counts": {
            "day": int(np.count_nonzero(table.daylight)),
            "night": int(np.count_nonzero(~table.daylight)),
        },
        "distance_bin_counts": _distance_bin_counts(table),
        "evaluation_limited_strata": sorted(
            name
            for name, counts in strata.items()
            if isinstance(counts, dict)
            and int(counts.get("insufficient_pieces", 0)) > 0
        ),
        "split_schema": artifact["schema"],
        "split_seed": artifact["seed"],
        "split_ratios": artifact["ratios"],
        "manifest_hash": artifact["manifest_hash"],
        "split_hashes": artifact["partition_hashes"],
        "split_counts": artifact["piece_counts"],
        "split_represented_source_counts": artifact[
            "represented_source_counts"
        ],
        "strata_counts": strata,
        "training_prior": dict(TRAINING_PRIOR),
    }
    if check_duplicates:
        report["duplicate_waveforms"] = audit_duplicate_waveforms(
            table, assignment
        )

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_type_constraints(
    decision: HierarchicalDecision,
    targets: torch.Tensor,
) -> dict[str, object]:
    """Summarize why hierarchical candidates were accepted or rejected."""
    if targets.ndim != 1 or len(targets) != len(decision.final_type):
        raise ValueError("diagnostic targets must align with decisions")
    if torch.any((targets < 0) | (targets >= len(TYPE_NAMES))):
        raise ValueError("diagnostic targets must use type indices 0..4")
    passes = decision.constraint_passes
    if set(passes) != set(CONSTRAINT_ORDER):
        raise ValueError("decision constraints are unsupported")
    if any(
        mask.dtype != torch.bool or mask.shape != targets.shape
        for mask in passes.values()
    ):
        raise ValueError("decision constraint masks must align with targets")
    pass_matrix = torch.stack([passes[name] for name in CONSTRAINT_ORDER])
    failures = ~pass_matrix
    failure_count = failures.sum(dim=0)
    accepted = failure_count.eq(0)
    if not torch.equal(accepted, decision.final_type.ne(0)):
        raise ValueError("decision result and constraint masks diverge")
    candidate = decision.candidate_known_type
    candidate_correct = candidate.eq(targets)
    candidate_confusion = np.zeros(
        (len(TYPE_NAMES), len(TYPE_NAMES)), dtype=np.int64
    )
    final_confusion = np.zeros_like(candidate_confusion)
    for truth, candidate_value, final in zip(
        targets.detach().cpu().tolist(),
        candidate.detach().cpu().tolist(),
        decision.final_type.detach().cpu().tolist(),
    ):
        candidate_confusion[int(truth), int(candidate_value)] += 1
        final_confusion[int(truth), int(final)] += 1

    by_type: dict[str, object] = {}
    for type_index, type_name in enumerate(TYPE_NAMES):
        rows = targets.eq(type_index)
        support = int(rows.sum())
        scopes = {"all": rows}
        if type_index:
            scopes["correct_candidate"] = rows & candidate_correct
        scope_reports: dict[str, object] = {}
        for scope_name, scope_rows in scopes.items():
            scope_count = int(scope_rows.sum())
            constraints: dict[str, object] = {}
            prior_pass = torch.ones_like(rows)
            for index, name in enumerate(CONSTRAINT_ORDER):
                failed = ~passes[name] & scope_rows
                unique = failed & failure_count.eq(1)
                incremental = failed & prior_pass & scope_rows
                cumulative = failures[: index + 1].any(dim=0) & scope_rows
                failed_count = int(failed.sum())
                unique_count = int(unique.sum())
                incremental_count = int(incremental.sum())
                cumulative_count = int(cumulative.sum())
                constraints[name] = {
                    "failed_count": failed_count,
                    "failed_rate": _ratio(failed_count, scope_count),
                    "unique_failed_count": unique_count,
                    "unique_failed_rate": _ratio(unique_count, scope_count),
                    "incremental_failed_count": incremental_count,
                    "incremental_failed_rate": _ratio(
                        incremental_count, scope_count
                    ),
                    "cumulative_failed_count": cumulative_count,
                    "cumulative_failed_rate": _ratio(
                        cumulative_count, scope_count
                    ),
                }
                prior_pass &= passes[name]
            scope_reports[scope_name] = {
                "count": scope_count,
                "constraints": constraints,
            }
        correct_count = int((candidate_correct & rows).sum())
        by_type[type_name] = {
            "support": support,
            "candidate_correct_count": None if type_index == 0 else correct_count,
            "candidate_correct_rate": (
                None if type_index == 0 else _ratio(correct_count, support)
            ),
            "candidate_distribution": {
                TYPE_NAMES[index]: int(candidate_confusion[type_index, index])
                for index in range(1, len(TYPE_NAMES))
            },
            "accepted_as_known_count": int((accepted & rows).sum()),
            "rejected_to_ic_count": int((~accepted & rows).sum()),
            "rejected_with_correct_candidate_count": (
                None
                if type_index == 0
                else int((~accepted & candidate_correct & rows).sum())
            ),
            "failure_multiplicity": {
                str(number): int((failure_count.eq(number) & rows).sum())
                for number in range(len(CONSTRAINT_ORDER) + 1)
            },
            "scopes": scope_reports,
        }
    return {
        "schema": TYPE_CONSTRAINT_SUMMARY_SCHEMA,
        "class_order": list(TYPE_NAMES),
        "constraint_order": list(CONSTRAINT_ORDER),
        "piece_count": int(len(targets)),
        "accepted_as_known_count": int(accepted.sum()),
        "rejected_to_ic_count": int((~accepted).sum()),
        "candidate_confusion": candidate_confusion.tolist(),
        "final_confusion": final_confusion.tolist(),
        "by_true_type": by_type,
    }


def _combine_hierarchical_outputs(
    items: list[HierarchicalTypeOutput],
) -> HierarchicalTypeOutput:
    return HierarchicalTypeOutput(
        **{
            field.name: torch.cat([getattr(item, field.name) for item in items])
            for field in fields(HierarchicalTypeOutput)
        }
    )


def _diagnose_type_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: HierarchicalDecisionConfig,
) -> dict[str, object]:
    primary_batches: list[HierarchicalTypeOutput] = []
    alternate_batches: list[HierarchicalTypeOutput] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _evaluation_tensors(batch, device)
            primary, alternate = _hierarchical_outputs(model, tensors)
            for output, destination in (
                (primary, primary_batches), (alternate, alternate_batches)
            ):
                destination.append(HierarchicalTypeOutput(**{
                    field.name: getattr(output, field.name).cpu()
                    for field in fields(HierarchicalTypeOutput)
                }))
            targets.append(tensors["type_label"].long().cpu())
    if not targets:
        raise ValueError("diagnostic loader produced no samples")
    decision = decide_hierarchical_types(
        _combine_hierarchical_outputs(primary_batches),
        _combine_hierarchical_outputs(alternate_batches),
        config,
    )
    return summarize_type_constraints(decision, torch.cat(targets))


def diagnose_type_checkpoint(
    task_data: str | Path,
    checkpoint_path: str | Path,
    output: str | Path | None = None,
    *,
    partition: str = "validation",
    decision_config_path: str | Path | None = None,
    batch_size: int = 60,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, object]:
    """Audit one partition; validation is the only default tuning partition."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    if partition == "test" and decision_config_path is not None:
        raise ValueError("external decision_config is validation-only")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    loaded = load_model_checkpoint(checkpoint_path, target_device)
    if loaded.model.architecture != HIERARCHICAL_TYPE_ARCHITECTURE:
        raise ValueError("constraint diagnostics require a hierarchical checkpoint")
    table, diagnostics = build_piece_table(task_data, require_distance=True)
    assignment = assign_piece_splits(table, seed=seed)
    validate_piece_split(table, assignment)
    artifact = split_artifact(table, assignment)
    split_hash = _canonical_hash(artifact)
    if split_hash != loaded.metadata.get("split_hash"):
        raise ValueError("checkpoint split hash does not match task data")
    config_value = (
        load_decision_config(decision_config_path)
        if decision_config_path is not None
        else validate_decision_config(loaded.metadata["decision_config"])
    )
    config = HierarchicalDecisionConfig(**config_value)
    preprocess = checkpoint_preprocess_config(loaded)
    dataset = FiveClassDataset(
        table,
        assignment.positions(partition),
        split=partition,
        preprocess_config=preprocess,
    )
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
            pin_memory=target_device.type == "cuda",
        )
        diagnostics_report = _diagnose_type_loader(
            loaded.model, loader, target_device, config
        )
    finally:
        dataset.close()
    report = {
        "schema": TYPE_DIAGNOSTIC_SCHEMA,
        "checkpoint_sha256": model_sha256(checkpoint_path),
        "checkpoint_schema": loaded.schema,
        "partition": partition,
        "selection_eligible": partition == "validation",
        "decision_config_source": (
            "external" if decision_config_path is not None else "checkpoint"
        ),
        "split_hash": split_hash,
        "seed": int(seed),
        "decision_config": decision_config_dict(config),
        "data": diagnostics,
        "diagnostics": diagnostics_report,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    report, indent=2, ensure_ascii=False, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the compact audit command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_duplicates", action="store_true")
    parser.add_argument("--type_checkpoint")
    parser.add_argument("--decision_config")
    parser.add_argument(
        "--partition", choices=("validation", "test"), default="validation"
    )
    return parser


def main() -> None:
    """Run the audit CLI."""
    args = build_arg_parser().parse_args()
    if args.type_checkpoint:
        report = diagnose_type_checkpoint(
            args.task_data,
            args.type_checkpoint,
            args.output,
            partition=args.partition,
            decision_config_path=args.decision_config,
            seed=args.seed,
        )
        diagnostic = report["diagnostics"]
        print(
            f"Diagnosed {diagnostic['piece_count']} {args.partition} pieces; "
            f"known={diagnostic['accepted_as_known_count']}"
        )
        return
    if args.decision_config or args.partition != "validation":
        raise ValueError("decision options require --type_checkpoint")
    report = audit_dataset(
        task_data=args.task_data,
        output=args.output,
        seed=args.seed,
        check_duplicates=args.check_duplicates,
    )
    print(
        f"Audited {report['files']} files and {report['pieces']} pieces; "
        f"split counts={report['split_counts']}"
    )


if __name__ == "__main__":
    main()
