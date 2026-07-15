"""Stable piece identities and completeness checks for OOF predictions."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any


TRAINING_TYPE_COUNT = 4


def _canonical_source_path(path: str | os.PathLike[str]) -> str:
    """Normalize native or foreign separators to the artifact path format."""
    return PurePosixPath(os.fspath(path).replace("\\", "/")).as_posix()


def oof_row_id(relative_path: str | os.PathLike[str], piece_index: int) -> str:
    """Return the source-relative identity for one waveform piece."""
    return f"{_canonical_source_path(relative_path)}#{int(piece_index)}"


def expected_oof_rows(fold_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Expand a canonical fold manifest into one contract per trusted piece."""
    if fold_manifest.get("schema") != "file_isolated_exact_interval_cv_v1":
        raise ValueError("unexpected fold manifest schema")
    folds = fold_manifest.get("folds")
    if not isinstance(folds, Mapping):
        raise ValueError("fold manifest has no fold rows")

    expected = {}
    for fold_text, files in sorted(folds.items(), key=lambda item: int(item[0])):
        fold = int(fold_text)
        for file_row in files:
            source_path = _canonical_source_path(file_row["path"])
            type_idx = int(file_row["type_idx"])
            if not 0 <= type_idx < TRAINING_TYPE_COUNT:
                raise ValueError(f"unexpected OOF training type: {type_idx}")
            n_pieces = int(file_row["n_pieces"])
            if n_pieces < 0:
                raise ValueError(f"negative OOF piece count for {source_path}")
            for piece_index in range(n_pieces):
                key = oof_row_id(source_path, piece_index)
                if key in expected:
                    raise ValueError(f"duplicate expected OOF piece: {key}")
                expected[key] = {
                    "fold": fold,
                    "type_idx": type_idx,
                    "source_path": source_path,
                    "piece_index": piece_index,
                }
    return expected


def validate_oof_rows(
    rows: Iterable[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require exactly one correctly labelled prediction per expected piece."""
    seen = {}
    for row in rows:
        key = str(row["piece_key"])
        if key in seen:
            raise ValueError(f"duplicate OOF piece: {key}")
        if key not in expected:
            raise ValueError(f"unknown OOF piece: {key}")
        contract = expected[key]
        if int(row["fold"]) != int(contract["fold"]):
            raise ValueError(f"wrong OOF fold for {key}")
        if int(row["true_type"]) != int(contract["type_idx"]):
            raise ValueError(f"OOF label mismatch for {key}")
        if str(row.get("source_path", contract["source_path"])) != str(
            contract["source_path"]
        ):
            raise ValueError(f"OOF source mismatch for {key}")
        if int(row.get("piece_index", contract["piece_index"])) != int(
            contract["piece_index"]
        ):
            raise ValueError(f"OOF piece index mismatch for {key}")
        seen[key] = row

    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ValueError(f"missing OOF pieces: {len(missing)}")
