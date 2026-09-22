import json

import pytest
import torch

import audit_data
from audit_data import (
    audit_dataset,
    audit_duplicate_waveforms,
    summarize_type_constraints,
)
from data.manifest import TYPE_NAMES, build_piece_table
from data.split import PARTITION_NAMES, assign_piece_splits
from evaluation import HierarchicalDecision
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
        "type_checkpoint",
        "decision_config",
        "partition",
    }
    assert args.task_data == "../train_data"
    assert args.output is None
    assert args.seed == 42
    assert args.check_duplicates is False
    assert args.type_checkpoint is None
    assert args.decision_config is None
    assert args.partition == "validation"


def _constraint_decision():
    targets = torch.tensor([1, 1, 1, 1, 1, 1, 1, 2, 0])
    passes = {
        name: torch.ones(len(targets), dtype=torch.bool)
        for name in audit_data.CONSTRAINT_ORDER
    }
    for row, name in enumerate(audit_data.CONSTRAINT_ORDER, start=1):
        passes[name][row] = False
    passes["known_probability"][7] = False
    passes["prototype_similarity"][7] = False
    stable = torch.stack(list(passes.values())).all(dim=0)
    candidate = torch.tensor([1, 1, 1, 1, 1, 1, 1, 2, 1])
    decision = HierarchicalDecision(
        final_type=torch.where(stable, candidate, torch.zeros_like(candidate)),
        candidate_known_type=candidate,
        ic_gate_probability=torch.zeros(len(targets)),
        known_type_probability=torch.ones(len(targets)),
        prototype_similarity=torch.ones(len(targets)),
        local_prediction=candidate,
        global_prediction=candidate,
        consistency_score=torch.ones(len(targets)),
        decision_reason=tuple("test" for _ in targets),
        constraint_passes=passes,
    )
    return decision, targets


def test_type_constraint_audit_separates_unique_and_overlapping_failures():
    decision, targets = _constraint_decision()

    report = summarize_type_constraints(decision, targets)
    ncg = report["by_true_type"]["NCG"]
    probability = ncg["scopes"]["all"]["constraints"]["known_probability"]
    prototype = ncg["scopes"]["all"]["constraints"][
        "prototype_similarity"
    ]

    assert report["piece_count"] == 9
    assert report["accepted_as_known_count"] == 2
    assert ncg["candidate_correct_count"] == 7
    assert ncg["rejected_with_correct_candidate_count"] == 6
    assert probability["failed_count"] == 1
    assert probability["unique_failed_count"] == 1
    assert prototype["failed_count"] == 1
    assert prototype["unique_failed_count"] == 1
    nnbe = report["by_true_type"]["NNBE"]
    assert nnbe["failure_multiplicity"]["2"] == 1
    assert nnbe["scopes"]["all"]["constraints"]["known_probability"][
        "unique_failed_count"
    ] == 0
    assert report["candidate_confusion"][0][1] == 1
    assert report["final_confusion"][0][1] == 1
    json.dumps(report, allow_nan=False)


def test_type_constraint_audit_rejects_divergent_masks():
    decision, targets = _constraint_decision()
    decision.constraint_passes["view_agreement"][0] = False

    with pytest.raises(ValueError, match="diverge"):
        summarize_type_constraints(decision, targets)


def test_type_constraint_audit_forbids_tuning_on_test_partition():
    with pytest.raises(ValueError, match="validation-only"):
        audit_data.diagnose_type_checkpoint(
            "unused",
            "unused.pt",
            partition="test",
            decision_config_path="candidate.json",
        )
