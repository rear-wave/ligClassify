"""Audit trusted waveform data and write a reproducible file-isolated split."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.group_split import group_stratified_split, validate_group_split
from data.split_artifacts import build_data_audit, make_split_manifest, write_json
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
    splits = group_stratified_split(
        entries,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    validate_group_split(splits)
    split_manifest = make_split_manifest(
        splits, str(task_data), seed, val_fraction, test_fraction
    )
    data_audit = build_data_audit(splits, TYPE_NAMES)
    data_audit["manifest_diagnostics"] = diagnostics
    data_audit["split_hashes"] = split_manifest["split_hashes"]

    source_conditions = {
        condition
        for summary in data_audit["splits"].values()
        for condition in summary["conditions"]
    }
    deficits = []
    for split_name in ("val", "test"):
        available = set(data_audit["splits"][split_name]["conditions"])
        for condition in sorted(source_conditions - available):
            deficits.append(f"{split_name}: missing {condition}")
    data_audit["split_condition_deficits"] = deficits

    result = {"data_audit": data_audit, "split_manifest": split_manifest}
    if output is not None:
        output = Path(output)
        write_json(output, data_audit)
        write_json(output.parent / "split_manifest.json", split_manifest)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_data", default="../train_data")
    parser.add_argument(
        "--output", default="./weights/conditional/data_audit.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
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
        f"cross-split files={audit['cross_split_files']}"
    )
    for name, summary in audit["splits"].items():
        print(f"  {name}: {summary['files']} files, {summary['pieces']} pieces")
    if audit["split_condition_deficits"]:
        print("Condition deficits (review before training):")
        for deficit in audit["split_condition_deficits"]:
            print(f"  {deficit}")


if __name__ == "__main__":
    main()
