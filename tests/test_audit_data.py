from pathlib import Path

from audit_data import audit_dataset
from tests.test_training_manifest import write_lig


def test_audit_is_reproducible_file_isolated_and_excludes_ic(tmp_path):
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for index in range(6):
            write_lig(
                tmp_path / type_name / "day" / "0-100km" / f"{index}.lig",
                pieces=index + 1,
            )
    write_lig(tmp_path / "IC" / "ignored.lig", pieces=10)

    first = audit_dataset(tmp_path, seed=17)
    second = audit_dataset(tmp_path, seed=17)

    assert first["fold_manifest"] == second["fold_manifest"]
    assert first["fold_manifest"]["fold_count"] == 3
    assert set(first["fold_manifest"]["folds"]) == {"0", "1", "2"}
    assert first["data_audit"]["trained_ic_pieces"] == 0
    assert first["data_audit"]["cross_fold_files"] == 0
    assert set(first["data_audit"]["trained_types"]) == {
        "NCG", "NNBE", "PCG", "PNBE"
    }
    assert first["data_audit"]["manifest_totals"] == {"files": 24, "pieces": 84}
    assert first["data_audit"]["fold_totals"] == {"files": 24, "pieces": 84}
    assert first["data_audit"]["cv_contract"] == {
        "schema": "file_isolated_exact_interval_cv_v1",
        "fold_count": 3,
        "split_fractions_active": False,
    }
    assert all(
        first["data_audit"]["splits"][f"fold_{index}"]["files"]
        for index in range(3)
    )


def test_audit_writes_requested_report_and_sibling_fold_artifacts(tmp_path):
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for index in range(3):
            write_lig(tmp_path / type_name / "night" / "100-200km" / f"{index}.lig")
    output = tmp_path / "reports" / "data_audit.json"

    audit_dataset(tmp_path, output=output, seed=5)

    assert output.is_file()
    assert (output.parent / "fold_manifest.json").is_file()
    assert (output.parent / "support_map.json").is_file()
    assert not (output.parent / "split_manifest.json").exists()


def test_audit_reports_insufficient_support_cell_names(tmp_path):
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        file_count = 2 if type_name == "NCG" else 3
        for index in range(file_count):
            write_lig(
                tmp_path / type_name / "day" / "300-400km" / f"{index}.lig",
                pieces=index + 1,
            )

    result = audit_dataset(tmp_path, seed=9)

    assert result["data_audit"]["insufficient_support_cell_count"] == 1
    assert result["data_audit"]["insufficient_support_cells"] == [
        "NCG/day/300-400km"
    ]
    assert result["support_map"]["NCG/day/300-400km"]["status"] == (
        "insufficient_support"
    )


def test_audit_marks_legacy_split_fractions_inactive(tmp_path):
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for index in range(3):
            write_lig(
                tmp_path / type_name / "day" / "0-100km" / f"{index}.lig"
            )

    result = audit_dataset(
        tmp_path, seed=3, val_fraction=0.4, test_fraction=0.1
    )

    assert result["data_audit"]["cv_contract"]["split_fractions_active"] is False
