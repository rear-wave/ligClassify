"""Audit the five-class piece manifest and deterministic split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data.lig import read_raw_piece
from data.manifest import TYPE_NAMES, PieceTable, build_piece_table
from data.split import (
    PARTITION_NAMES,
    SplitAssignment,
    assign_piece_splits,
    split_artifact,
    validate_piece_split,
)


TRAINING_PRIOR = {
    "IC": 0.20,
    "NCG": 0.20,
    "NNBE": 0.20,
    "PCG": 0.20,
    "PNBE": 0.20,
}


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
            and int(counts.get("insufficient_source_groups", 0)) > 0
        ),
        "split_schema": artifact["schema"],
        "split_seed": artifact["seed"],
        "split_ratios": artifact["ratios"],
        "manifest_hash": artifact["manifest_hash"],
        "split_hashes": artifact["partition_hashes"],
        "split_counts": artifact["piece_counts"],
        "split_source_counts": artifact["source_counts"],
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the compact audit command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_duplicates", action="store_true")
    return parser


def main() -> None:
    """Run the audit CLI."""
    args = build_arg_parser().parse_args()
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
