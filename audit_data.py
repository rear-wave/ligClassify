"""Audit trusted waveform data and write reproducible three-fold artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.cross_validation import (
    FOLD_COUNT,
    MINIMUM_SUPPORTED_PIECES,
    assign_exact_folds,
    build_support_map,
    validate_fold_assignment,
)
from data.split_artifacts import build_data_audit, make_fold_manifest, write_json
from data.training_manifest import build_manifest


TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def audit_dataset(
    task_data,
    output=None,
    seed=42,
    val_fraction=0.15,
    test_fraction=0.15,
):
    """Inspect trusted files without loading waveforms or starting training."""
    entries, diagnostics = build_manifest(str(task_data), TYPE_NAMES)
    if not entries:
        raise RuntimeError(f"No valid trusted-type .lig files found under {task_data}")

    folds = assign_exact_folds(entries, n_folds=FOLD_COUNT, seed=seed)
    validate_fold_assignment(folds, entries, n_folds=FOLD_COUNT)
    fold_manifest = make_fold_manifest(folds, str(task_data), seed)

    reversed_folds = assign_exact_folds(
        reversed(entries), n_folds=FOLD_COUNT, seed=seed
    )
    validate_fold_assignment(reversed_folds, entries, n_folds=FOLD_COUNT)
    reversed_manifest = make_fold_manifest(
        reversed_folds, str(task_data), seed
    )
    if fold_manifest["combined_hash"] != reversed_manifest["combined_hash"]:
        raise RuntimeError("cross-validation fold hashes are not deterministic")
    ownership = {
        fold: [row["path"] for row in rows]
        for fold, rows in fold_manifest["folds"].items()
    }
    reversed_ownership = {
        fold: [row["path"] for row in rows]
        for fold, rows in reversed_manifest["folds"].items()
    }
    if ownership != reversed_ownership:
        raise RuntimeError("cross-validation fold ownership is not deterministic")

    manifest_totals = {
        "files": int(diagnostics["valid_files"]),
        "pieces": int(sum(entry.n_pieces for entry in entries)),
    }
    fold_totals = {
        "files": int(sum(len(rows) for rows in folds.values())),
        "pieces": int(
            sum(entry.n_pieces for rows in folds.values() for entry in rows)
        ),
    }
    if fold_totals != manifest_totals:
        raise RuntimeError(
            "cross-validation fold totals do not match the trusted manifest"
        )

    support_map = build_support_map(entries, TYPE_NAMES)
    insufficient_support_cells = sorted(
        name
        for name, row in support_map.items()
        if row["status"] == "insufficient_support"
    )
    named_folds = {
        f"fold_{index}": rows for index, rows in sorted(folds.items())
    }
    data_audit = build_data_audit(named_folds, TYPE_NAMES)
    data_audit["manifest_diagnostics"] = diagnostics
    data_audit["manifest_totals"] = manifest_totals
    data_audit["fold_totals"] = fold_totals
    data_audit["fold_hashes"] = fold_manifest["holdout_hashes"]
    data_audit["combined_fold_hash"] = fold_manifest["combined_hash"]
    data_audit["cross_fold_files"] = data_audit["cross_split_files"]
    data_audit["cv_contract"] = {
        "schema": fold_manifest["schema"],
        "fold_count": FOLD_COUNT,
        "split_fractions_active": False,
    }
    data_audit["legacy_split_arguments"] = {
        "active": False,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
    }
    data_audit["insufficient_support_cell_count"] = len(
        insufficient_support_cells
    )
    data_audit["insufficient_support_cells"] = insufficient_support_cells

    result = {
        "data_audit": data_audit,
        "fold_manifest": fold_manifest,
        "support_map": support_map,
    }
    if output is not None:
        output = Path(output)
        write_json(output, data_audit)
        write_json(output.parent / "fold_manifest.json", fold_manifest)
        write_json(output.parent / "support_map.json", support_map)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument(
        "--output", default="./weights/conditional/data_audit.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Legacy compatibility argument; inactive under fixed three-fold CV.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.15,
        help="Legacy compatibility argument; inactive under fixed three-fold CV.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    result = audit_dataset(
        args.task_data,
        args.output,
        args.seed,
        args.val_fraction,
        args.test_fraction,
    )
    audit = result["data_audit"]
    print(
        f"Audited {audit['manifest_diagnostics']['valid_files']} trusted files; "
        f"cross-fold files={audit['cross_fold_files']}"
    )
    for name, summary in audit["splits"].items():
        print(f"  {name}: {summary['files']} files, {summary['pieces']} pieces")
    if audit["insufficient_support_cells"]:
        print(
            "Insufficient-support cells "
            f"(fewer than {MINIMUM_SUPPORTED_PIECES} waveform pieces):"
        )
        for condition in audit["insufficient_support_cells"]:
            print(f"  {condition}")


if __name__ == "__main__":
    main()
