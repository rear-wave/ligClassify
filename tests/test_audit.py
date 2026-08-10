import json

import pytest

import audit_data
from audit_data import audit_dataset, audit_duplicate_waveforms
from data.manifest import TYPE_NAMES, build_piece_table
from data.split import PARTITION_NAMES, assign_piece_splits
from .test_lig import make_piece, write_source


def _write_audit_corpus(root):
    for type_index, type_name in enumerate(TYPE_NAMES):
        distance = (
            ""
            if type_name == "IC"
            else f"/{type_index * 100}-{(type_index + 1) * 100}km"
        )
        directory = root / type_name / "mixed"
        if distance:
            directory = directory / distance.removeprefix("/")
        directory.mkdir(parents=True)
        for file_index in range(3):
            write_source(
                directory / f"source_{file_index}.lig",
                [
                    make_piece(type_index * 10 + file_index * 2 + 1, hour=0),
                    make_piece(type_index * 10 + file_index * 2 + 2, hour=16),
                ],
            )


def test_audit_reports_piece_split_and_training_prior(tmp_path, monkeypatch):
    root = tmp_path / "train_data"
    _write_audit_corpus(root)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("normal audit must not hash waveform bytes")

    monkeypatch.setattr(audit_data, "audit_duplicate_waveforms", fail_if_called)
    report = audit_dataset(root, seed=42)

    assert report["split_schema"] == "piece_stratified_split_v2"
    assert report["files"] == 15
    assert report["pieces"] == 30
    assert report["type_counts"] == {name: 6 for name in TYPE_NAMES}
    assert report["daylight_counts"] == {"day": 15, "night": 15}
    assert report["distance_bin_counts"] == {
        "100-200km": 6,
        "200-300km": 6,
        "300-400km": 6,
        "400-500km": 6,
    }
    assert set(report["split_counts"]) == set(PARTITION_NAMES)
    assert report["split_counts"]["train"] > 0
    assert set(report["split_hashes"]) == set(PARTITION_NAMES)
    assert len(report["manifest_hash"]) == 64
    assert report["evaluation_limited_strata"] == []
    assert report["training_prior"] == {
        "IC": 0.20,
        "NCG": 0.20,
        "NNBE": 0.20,
        "PCG": 0.20,
        "PNBE": 0.20,
    }
    assert set(report["split_represented_source_counts"]) == set(
        PARTITION_NAMES
    )
    assert "duplicate_waveforms" not in report


def test_duplicate_audit_rejects_identical_bytes_across_splits(tmp_path):
    directory = (
        tmp_path / "train_data" / "NCG" / "day" / "0-100km"
    )
    directory.mkdir(parents=True)
    raw = make_piece(17, hour=8)
    for index in range(3):
        write_source(directory / f"duplicate_{index}.lig", [raw])
    table, _ = build_piece_table(tmp_path / "train_data")
    assignment = assign_piece_splits(table, seed=42)

    with pytest.raises(ValueError, match="duplicate waveform crosses partitions"):
        audit_duplicate_waveforms(table, assignment)


def test_duplicate_mode_hashes_complete_pieces_and_output_is_json(tmp_path):
    root = tmp_path / "train_data"
    source = root / "IC" / "source.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1), make_piece(2)])
    output = tmp_path / "reports" / "audit.json"

    report = audit_dataset(root, output=output, seed=9, check_duplicates=True)

    assert report["duplicate_waveforms"] == {
        "algorithm": "sha256_complete_raw_piece",
        "pieces_hashed": 2,
        "unique_digests": 2,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_audit_cli_has_only_the_compact_contract():
    args = audit_data.build_arg_parser().parse_args([])

    assert set(vars(args)) == {
        "task_data",
        "output",
        "seed",
        "check_duplicates",
    }
    assert args.task_data == "../train_data"
    assert args.output is None
    assert args.seed == 42
    assert args.check_duplicates is False
