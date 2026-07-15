import copy

import pytest
import torch

from conditional_pipeline import collect_prediction_bundle
from data.oof_manifest import expected_oof_rows, oof_row_id, validate_oof_rows


def _fold_manifest():
    return {
        "schema": "file_isolated_exact_interval_cv_v1",
        "folds": {
            "0": [
                {
                    "path": "NCG/day/0-100km/a.lig",
                    "n_pieces": 2,
                    "type_idx": 0,
                }
            ],
            "1": [
                {
                    "path": "NNBE/night/100-200km/b.lig",
                    "n_pieces": 1,
                    "type_idx": 1,
                }
            ],
        },
    }


def _complete_rows(expected):
    return [
        {
            "piece_key": key,
            "fold": contract["fold"],
            "true_type": contract["type_idx"],
            "source_path": contract["source_path"],
            "piece_index": contract["piece_index"],
        }
        for key, contract in expected.items()
    ]


def test_oof_rows_require_every_piece_exactly_once():
    expected = {
        "NCG/day/0-100km/a.lig#0": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig",
            "piece_index": 0,
        },
        "NCG/day/0-100km/a.lig#1": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig",
            "piece_index": 1,
        },
    }
    rows = [
        {
            "piece_key": key,
            "fold": value["fold"],
            "true_type": value["type_idx"],
        }
        for key, value in expected.items()
    ]

    validate_oof_rows(rows, expected)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_rows(rows + [rows[0]], expected)
    with pytest.raises(ValueError, match="missing"):
        validate_oof_rows(rows[:1], expected)


def test_piece_key_does_not_use_fold_local_file_id():
    assert oof_row_id("NCG/a.lig", 7) == "NCG/a.lig#7"
    assert oof_row_id(r"NCG\a.lig", 7) == "NCG/a.lig#7"
    assert oof_row_id("NNBE/a.lig", 7) != oof_row_id("NCG/a.lig", 7)


def test_expected_oof_rows_expand_canonical_fold_paths():
    expected = expected_oof_rows(_fold_manifest())

    assert expected == {
        "NCG/day/0-100km/a.lig#0": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig",
            "piece_index": 0,
        },
        "NCG/day/0-100km/a.lig#1": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig",
            "piece_index": 1,
        },
        "NNBE/night/100-200km/b.lig#0": {
            "fold": 1,
            "type_idx": 1,
            "source_path": "NNBE/night/100-200km/b.lig",
            "piece_index": 0,
        },
    }


def test_expected_oof_rows_normalizes_foreign_source_separators():
    manifest = {
        "schema": "file_isolated_exact_interval_cv_v1",
        "folds": {
            "0": [
                {
                    "path": r"NCG\day/0-100km\a.lig",
                    "n_pieces": 1,
                    "type_idx": 0,
                }
            ]
        },
    }

    expected = expected_oof_rows(manifest)

    assert expected == {
        "NCG/day/0-100km/a.lig#0": {
            "fold": 0,
            "type_idx": 0,
            "source_path": "NCG/day/0-100km/a.lig",
            "piece_index": 0,
        }
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold", 2, "wrong OOF fold"),
        ("true_type", 3, "OOF label mismatch"),
        ("source_path", "NCG/day/0-100km/other.lig", "OOF source mismatch"),
        ("piece_index", 9, "OOF piece index mismatch"),
    ],
)
def test_oof_rows_reject_contract_mismatches(field, value, message):
    expected = expected_oof_rows(_fold_manifest())
    rows = _complete_rows(expected)
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_oof_rows(rows, expected)


def test_oof_rows_reject_unknown_piece():
    expected = expected_oof_rows(_fold_manifest())
    rows = _complete_rows(expected)
    rows[0]["piece_key"] = "NCG/day/0-100km/unknown.lig#0"

    with pytest.raises(ValueError, match="unknown"):
        validate_oof_rows(rows, expected)


def test_expected_oof_rows_rejects_wrong_schema_and_ic_type():
    wrong_schema = _fold_manifest()
    wrong_schema["schema"] = "other"
    with pytest.raises(ValueError, match="schema"):
        expected_oof_rows(wrong_schema)

    with_ic = copy.deepcopy(_fold_manifest())
    with_ic["folds"]["2"] = [
        {"path": "IC/rejected.lig", "n_pieces": 1, "type_idx": 4}
    ]
    with pytest.raises(ValueError, match="training type"):
        expected_oof_rows(with_ic)


class _PredictionModel:
    def eval(self):
        return self

    def forward_with_features(self, local, global_view, context):
        count = len(local)
        features = torch.zeros(count, 2)
        type_logits = torch.tensor([[4.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
        distance_logits = [torch.zeros(count, 30) for _ in range(4)]
        return features, type_logits, distance_logits, None


def test_prediction_bundle_records_stable_piece_identity_and_fold():
    batch = {
        "local": torch.zeros(1, 1, 8000),
        "global_view": torch.zeros(1, 1, 8000),
        "context": torch.tensor([[1.0, 0.0, 1.0]]),
        "quality": torch.ones(1, 3),
        "type_label": torch.tensor([0]),
        "distance_low_km": torch.tensor([0.0]),
        "distance_high_km": torch.tensor([100.0]),
        "file_id": torch.tensor([17]),
        "source_path": ["NCG/day/0-100km/a.lig"],
        "piece_index": [1],
        "piece_key": ["NCG/day/0-100km/a.lig#1"],
    }

    bundle = collect_prediction_bundle(
        _PredictionModel(), [batch], "cpu", "split-hash", fold_index=2
    )

    assert bundle["file_ids"].tolist() == [17]
    assert bundle["records"][0]["file_id"] == "NCG/day/0-100km/a.lig"
    assert bundle["records"][0]["source_path"] == "NCG/day/0-100km/a.lig"
    assert bundle["records"][0]["piece_index"] == 1
    assert bundle["records"][0]["piece_key"] == "NCG/day/0-100km/a.lig#1"
    assert bundle["records"][0]["fold"] == 2
