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

    assert first["split_manifest"]["split_hashes"] == second["split_manifest"]["split_hashes"]
    assert first["data_audit"]["trained_ic_pieces"] == 0
    assert first["data_audit"]["cross_split_files"] == 0
    assert set(first["data_audit"]["trained_types"]) == {
        "NCG", "NNBE", "PCG", "PNBE"
    }
    assert all(first["data_audit"]["splits"][name]["files"] for name in ("train", "val", "test"))


def test_audit_writes_requested_report_and_sibling_split_manifest(tmp_path):
    for type_name in ("NCG", "NNBE", "PCG", "PNBE"):
        for index in range(3):
            write_lig(tmp_path / type_name / "night" / "100-200km" / f"{index}.lig")
    output = tmp_path / "reports" / "data_audit.json"

    audit_dataset(tmp_path, output=output, seed=5)

    assert output.is_file()
    assert (output.parent / "split_manifest.json").is_file()
