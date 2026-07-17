import pytest

from data.manifest import (
    DISTANCE_NAMES,
    TYPE_NAMES,
    build_piece_table,
    piece_key,
)
from tests.test_lig import make_piece, write_source


def test_manifest_expands_storage_files_to_stable_piece_rows(tmp_path):
    root = tmp_path / "train_data"
    source = root / "NNBE" / "day" / "500-600km" / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1, hour=8), make_piece(2, hour=20)])

    table, diagnostics = build_piece_table(root)

    assert TYPE_NAMES == ("IC", "NCG", "NNBE", "PCG", "PNBE")
    assert DISTANCE_NAMES == ("NCG", "NNBE", "PCG", "PNBE")
    assert len(table) == 2
    assert table.type_index.tolist() == [2, 2]
    assert table.distance_bin.tolist() == [5, 5]
    assert table.daylight.tolist() == [True, False]
    assert table.piece_key(1) == "NNBE/day/500-600km/sample.lig#1"
    assert piece_key(r"NNBE\day\a.lig", 7) == "NNBE/day/a.lig#7"
    assert diagnostics["pieces"] == 2


def test_manifest_rejects_non_ic_file_without_exact_interval(tmp_path):
    source = tmp_path / "NCG" / "day" / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1)])

    with pytest.raises(ValueError, match="exact 100-km interval"):
        build_piece_table(tmp_path)


def test_manifest_rejects_interval_wider_than_100_km(tmp_path):
    source = tmp_path / "PCG" / "night" / "0-300km" / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1)])

    with pytest.raises(ValueError, match="exact 100-km interval"):
        build_piece_table(tmp_path)


@pytest.mark.parametrize("interval", ["2900-3100km", "3000-3100km"])
def test_manifest_rejects_bins_outside_0_to_3000_km(tmp_path, interval):
    source = tmp_path / "PNBE" / "day" / interval / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1)])

    with pytest.raises(ValueError, match="0-3000 km"):
        build_piece_table(tmp_path)


def test_manifest_allows_ic_without_distance_directory(tmp_path):
    source = tmp_path / "IC" / "events" / "sample.lig"
    source.parent.mkdir(parents=True)
    write_source(source, [make_piece(1, hour=8), make_piece(2, hour=20)])

    table, diagnostics = build_piece_table(tmp_path)

    assert table.type_index.tolist() == [0, 0]
    assert table.distance_bin.tolist() == [-1, -1]
    assert table.daylight.tolist() == [True, False]
    assert diagnostics == {"files": 1, "pieces": 2}
