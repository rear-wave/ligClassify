import importlib
import struct
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from data import lig_parser


FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208
TIMESTAMP_OFFSET = FILE_HEADER_BYTES + 108
WAVEFORM_OFFSET = FILE_HEADER_BYTES + 208


def write_lig(
    path: Path,
    timestamp=(19, 1, 2, 3, 4, 5),
    sec_frac=0.25,
    pieces=1,
    piece_timestamps=None,
    piece_sec_fracs=None,
):
    timestamps = piece_timestamps or [timestamp] * pieces
    fractions = piece_sec_fracs or [sec_frac] * pieces
    if len(timestamps) != pieces or len(fractions) != pieces:
        raise ValueError("piece timestamp metadata must match pieces")
    raw = bytearray(FILE_HEADER_BYTES + PIECE_BYTES * pieces)
    for piece_idx, (piece_timestamp, piece_fraction) in enumerate(
        zip(timestamps, fractions)
    ):
        piece_start = FILE_HEADER_BYTES + PIECE_BYTES * piece_idx
        struct.pack_into("i", raw, piece_start, 1001)
        struct.pack_into("6i4x", raw, piece_start + 108, *piece_timestamp)
        struct.pack_into("d", raw, piece_start + 136, piece_fraction)
        struct.pack_into("H", raw, piece_start + 208, piece_idx + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def training_manifest_module():
    try:
        return importlib.import_module("data.training_manifest")
    except ModuleNotFoundError:
        pytest.fail("data.training_manifest is not implemented")


def test_read_lig_timestamp_supports_two_digit_year(tmp_path):
    path = write_lig(tmp_path / "sample.lig")

    assert hasattr(lig_parser, "read_lig_timestamp")
    assert lig_parser.read_lig_timestamp(str(path)) == datetime(2019, 1, 2, 3, 4, 5, 250000)


def test_read_lig_timestamp_normalizes_hour_24(tmp_path):
    path = write_lig(tmp_path / "sample.lig", (17, 8, 21, 24, 0, 7), 0.5)

    assert hasattr(lig_parser, "read_lig_timestamp")
    assert lig_parser.read_lig_timestamp(str(path)) == datetime(2017, 8, 22, 0, 0, 7, 500000)


def test_read_lig_timestamps_reads_each_piece_in_order(tmp_path):
    path = write_lig(
        tmp_path / "sample.lig",
        pieces=3,
        piece_timestamps=[
            (19, 1, 2, 3, 4, 5),
            (19, 1, 2, 3, 4, 6),
            (19, 1, 2, 3, 4, 7),
        ],
        piece_sec_fracs=[0.1, 0.2, 0.3],
    )

    assert lig_parser.read_lig_timestamps(str(path)) == [
        datetime(2019, 1, 2, 3, 4, 5, 100000),
        datetime(2019, 1, 2, 3, 4, 6, 200000),
        datetime(2019, 1, 2, 3, 4, 7, 300000),
    ]


def test_read_lig_timestamps_reports_bad_piece_index(tmp_path):
    path = write_lig(
        tmp_path / "bad.lig",
        pieces=2,
        piece_timestamps=[
            (19, 1, 2, 3, 4, 5),
            (19, 13, 2, 3, 4, 6),
        ],
    )

    with pytest.raises(lig_parser.LigFormatError, match=r"piece_index=1"):
        lig_parser.read_lig_timestamps(str(path))


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"NNBE\day\100-200km\a.lig", 1),
        (r"NNBE\night_2400_2500km\a.lig", 24),
        (r"PNBE\day\0-300km\day_300-400km_events1.lig", 3),
        (r"PNBE\day\0-300km\events.lig", -1),
        (r"PNBE\day\2950-3050km\events.lig", -1),
    ],
)
def test_parse_distance_bin_accepts_only_exact_100km_ranges(path, expected):
    module = training_manifest_module()

    assert module.parse_distance_bin(path) == expected


def test_build_manifest_falls_back_to_filename_timestamp(tmp_path):
    module = training_manifest_module()
    path = write_lig(
        tmp_path / "NCG" / "100-200km" / "GZ_190102030405.1234567.lig",
        (19, 13, 2, 3, 4, 5),
    )

    entries, diagnostics = module.build_manifest(str(tmp_path), ["IC", "NCG"])

    assert len(entries) == 1
    assert entries[0].filepath == str(path)
    assert entries[0].timestamp == datetime(2019, 1, 2, 3, 4, 5, 123456)
    assert entries[0].dist_bin == 1
    assert diagnostics["filename_timestamp_fallbacks"] == 1


def test_four_class_manifest_treats_zero_index_ncg_as_distance_labelled(tmp_path):
    path = write_lig(
        tmp_path / "NCG" / "500-600km" / "GZ_190102030405.1234567.lig"
    )

    module = training_manifest_module()
    entries, diagnostics = module.build_manifest(
        str(tmp_path), ["NCG", "NNBE", "PCG", "PNBE"]
    )

    assert len(entries) == 1
    assert entries[0].filepath == str(path)
    assert entries[0].type_idx == 0
    assert entries[0].dist_bin == 5
    assert diagnostics["distance_labeled_files"] == 1


def make_entry(module, type_idx, day, name, n_pieces=1, dist_bin=-1):
    return module.ManifestEntry(
        filepath=name,
        type_idx=type_idx,
        dist_bin=dist_bin,
        timestamp=datetime(2020, 1, 1) + timedelta(days=day),
        n_pieces=n_pieces,
    )


def test_coverage_split_uses_latest_date_count_despite_piece_concentration():
    module = training_manifest_module()
    entries = [
        make_entry(module, 1, 0, "old.lig", n_pieces=100, dist_bin=0),
        make_entry(module, 1, 1, "new-1.lig", n_pieces=10, dist_bin=0),
        make_entry(module, 1, 2, "new-2.lig", n_pieces=10, dist_bin=0),
    ]

    splits = module.coverage_temporal_split_manifest(
        entries, val_fraction=0.0, test_fraction=0.15, seed=7
    )

    test_dates = {item.acquisition_date for item in splits["test"]}
    development_dates = {
        item.acquisition_date for split in ("train", "val") for item in splits[split]
    }
    assert test_dates == {entries[2].acquisition_date}
    assert test_dates.isdisjoint(development_dates)


def test_coverage_split_is_deterministic_and_holds_out_whole_files():
    module = training_manifest_module()
    entries = []
    for dist_bin in range(4):
        for file_index in range(3):
            entries.append(make_entry(
                module, 1, file_index, f"bin-{dist_bin}-{file_index}.lig",
                n_pieces=100, dist_bin=dist_bin,
            ))
    entries.append(make_entry(module, 1, 10, "future.lig", 100, 0))

    first = module.coverage_temporal_split_manifest(
        entries, val_fraction=0.34, test_fraction=0.05, seed=11
    )
    second = module.coverage_temporal_split_manifest(
        list(reversed(entries)), val_fraction=0.34, test_fraction=0.05, seed=11
    )

    assert {x.filepath for x in first["val"]} == {x.filepath for x in second["val"]}
    assert {x.dist_bin for x in first["val"]} == {0, 1, 2, 3}
    paths = [{x.filepath for x in first[name]} for name in ("train", "val", "test")]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])


def test_four_class_split_preserves_distance_bin_coverage_for_zero_index():
    module = training_manifest_module()
    entries = []
    for dist_bin in range(4):
        for file_index in range(3):
            entries.append(make_entry(
                module, 0, file_index,
                f"ncg-bin-{dist_bin}-{file_index}.lig",
                n_pieces=20,
                dist_bin=dist_bin,
            ))
    entries.append(make_entry(module, 0, 10, "future.lig", 20, 0))

    splits = module.coverage_temporal_split_manifest(
        entries, val_fraction=0.10, test_fraction=0.05, seed=11
    )

    assert {item.dist_bin for item in splits["val"]} == {0, 1, 2, 3}


def test_split_coverage_rejects_missing_bins_and_too_few_pieces():
    module = training_manifest_module()
    validation = [
        make_entry(module, 1, 0, f"ncg-{index}.lig", 20, index)
        for index in range(3)
    ]
    splits = {"train": [], "val": validation, "test": []}

    with pytest.raises(ValueError, match="NCG.*bins"):
        module.validate_split_coverage(
            splits, ["IC", "NCG"], min_bins=4, min_pieces=1
        )
    with pytest.raises(ValueError, match="NCG.*pieces"):
        module.validate_split_coverage(
            splits, ["IC", "NCG"], min_bins=3, min_pieces=100
        )


def test_four_class_coverage_validates_zero_index_ncg():
    module = training_manifest_module()
    validation = [
        make_entry(module, 0, 0, f"ncg-{index}.lig", 20, index)
        for index in range(3)
    ]

    with pytest.raises(ValueError, match="NCG.*bins"):
        module.validate_split_coverage(
            {"train": [], "val": validation, "test": []},
            ["NCG", "NNBE", "PCG", "PNBE"],
            min_bins=4,
            min_pieces=1,
        )
