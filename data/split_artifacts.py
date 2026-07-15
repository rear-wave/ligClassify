"""Serializable, path-stable records for file-isolated data splits."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.cross_validation import fold_train_holdout
from data.group_split import coarse_distance_band
from data.training_manifest import ManifestEntry


def _relative_path(path, root):
    return Path(os.path.relpath(os.path.abspath(path), os.path.abspath(root))).as_posix()


def serialize_manifest_entry(entry, root):
    """Convert one file entry to stable JSON-safe metadata."""
    return {
        "path": _relative_path(entry.filepath, root),
        "type_idx": int(entry.type_idx),
        "distance_low_km": entry.distance_low_km,
        "distance_high_km": entry.distance_high_km,
        "is_daytime": bool(entry.is_daytime),
        "n_pieces": int(entry.n_pieces),
        "timestamp": entry.timestamp.isoformat(),
    }


def split_hash(entries, root):
    """Hash source identities and labels independently of machine paths."""
    rows = [serialize_manifest_entry(entry, root) for entry in entries]
    payload = json.dumps(
        sorted(rows, key=lambda row: row["path"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_json_hash(payload: Any) -> str:
    """Hash a JSON-safe payload with deterministic key and separator rules."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_fold_manifest(
    folds: Mapping[int, Sequence[ManifestEntry]],
    root: str | os.PathLike[str],
    seed: int,
) -> dict[str, Any]:
    """Build the canonical exact-interval fold ownership artifact."""
    holdout_hashes = {
        str(index): split_hash(entries, root)
        for index, entries in sorted(folds.items())
    }
    train_hashes = {
        str(index): split_hash(fold_train_holdout(folds, index)[0], root)
        for index in sorted(folds)
    }
    combined = stable_json_hash({
        "schema": "file_isolated_exact_interval_cv_v1",
        "seed": int(seed),
        "holdout_hashes": holdout_hashes,
        "train_hashes": train_hashes,
    })
    return {
        "schema": "file_isolated_exact_interval_cv_v1",
        "fold_count": len(folds),
        "seed": int(seed),
        "holdout_hashes": holdout_hashes,
        "train_hashes": train_hashes,
        "combined_hash": combined,
        "folds": {
            str(index): sorted(
                (serialize_manifest_entry(entry, root) for entry in rows),
                key=lambda row: row["path"],
            )
            for index, rows in sorted(folds.items())
        },
    }


def make_split_manifest(splits, root, seed, val_fraction, test_fraction):
    """Build the complete immutable split description and hashes."""
    serialized = {
        name: [serialize_manifest_entry(entry, root) for entry in entries]
        for name, entries in splits.items()
    }
    hashes = {name: split_hash(entries, root) for name, entries in splits.items()}
    combined = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "file_isolated_condition_split_v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "split_hashes": hashes,
        "combined_split_hash": combined,
        "splits": serialized,
    }


def build_data_audit(splits, type_names):
    """Count files and pieces by type, daylight, and interval status."""
    report = {"trained_types": list(type_names), "trained_ic_pieces": 0, "splits": {}}
    owners = {}
    for split_name, entries in splits.items():
        summary = {
            "files": len(entries),
            "pieces": int(sum(entry.n_pieces for entry in entries)),
            "types": {},
            "conditions": {},
        }
        conditions = defaultdict(lambda: {"files": 0, "pieces": 0})
        for entry in entries:
            path = os.path.normcase(os.path.abspath(entry.filepath))
            previous = owners.setdefault(path, split_name)
            if previous != split_name:
                raise ValueError(f"source file appears in {previous} and {split_name}")
            type_name = type_names[int(entry.type_idx)]
            typed = summary["types"].setdefault(type_name, {
                "files": 0,
                "pieces": 0,
                "exact_interval_files": 0,
                "broad_interval_files": 0,
                "missing_interval_files": 0,
                "day_files": 0,
                "night_files": 0,
            })
            typed["files"] += 1
            typed["pieces"] += int(entry.n_pieces)
            if entry.distance_low_km is None:
                typed["missing_interval_files"] += 1
            elif entry.distance_high_km - entry.distance_low_km == 100:
                typed["exact_interval_files"] += 1
            else:
                typed["broad_interval_files"] += 1
            typed["day_files" if entry.is_daytime else "night_files"] += 1
            condition = (
                f"{type_name}/"
                f"{'day' if entry.is_daytime else 'night'}/"
                f"band_{coarse_distance_band(entry)}"
            )
            conditions[condition]["files"] += 1
            conditions[condition]["pieces"] += int(entry.n_pieces)
        summary["conditions"] = dict(sorted(conditions.items()))
        report["splits"][split_name] = summary
    report["cross_split_files"] = 0
    return report


def write_json(path, payload):
    """Write a deterministic UTF-8 JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
