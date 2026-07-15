import struct
from datetime import datetime
from pathlib import Path

import pytest

from data import lig_parser
from data.training_manifest import (
    build_manifest,
    build_piece_manifest,
    infer_daytime,
    parse_distance_bin,
    parse_distance_interval,
)


FILE_HEADER_BYTES = 112
PIECE_BYTES = 32208


def write_lig(
    path: Path,
    timestamp=(19, 1, 2, 3, 4, 5),
    sec_frac=0.25,
    pieces=1,
    piece_timestamps=None,
    piece_sec_fracs=None,
):
    """Write a bounded synthetic LIG fixture without real waveform data."""
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


def test_read_lig_timestamp_supports_two_digit_year(tmp_path):
    path = write_lig(tmp_path / "sample.lig")

    assert lig_parser.read_lig_timestamp(str(path)) == datetime(
        2019, 1, 2, 3, 4, 5, 250000
    )


def test_read_lig_timestamp_normalizes_hour_24(tmp_path):
    path = write_lig(tmp_path / "sample.lig", (17, 8, 21, 24, 0, 7), 0.5)

    assert lig_parser.read_lig_timestamp(str(path)) == datetime(
        2017, 8, 22, 0, 0, 7, 500000
    )


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
        piece_timestamps=[(19, 1, 2, 3, 4, 5), (19, 13, 2, 3, 4, 6)],
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
    assert parse_distance_bin(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"NCG\day\day_400-500km\a.lig", (400, 500)),
        (r"NNBE\night\night_1500-3000km\a.lig", (1500, 3000)),
        (r"PNBE\day\0-300km\day_300-400km_events.lig", (300, 400)),
        (r"NCG\day\a.lig", None),
        (r"PCG\day\2950-3050km\a.lig", None),
    ],
)
def test_parse_distance_interval_keeps_exact_and_broad_ranges(path, expected):
    assert parse_distance_interval(path) == expected


def test_infer_daytime_prefers_folder_then_utc_plus_eight():
    noon = datetime(2019, 1, 1, 12)
    assert infer_daytime(r"NCG\day\0-100km\a.lig", noon) is True
    assert infer_daytime(r"NCG\night\0-100km\a.lig", noon) is False
    assert infer_daytime("NCG/a.lig", datetime(2019, 1, 1, 0)) is True
    assert infer_daytime("NCG/a.lig", datetime(2019, 1, 1, 16)) is False


def test_build_manifest_falls_back_to_filename_timestamp(tmp_path):
    path = write_lig(
        tmp_path / "NCG" / "100-200km" / "GZ_190102030405.1234567.lig",
        (19, 13, 2, 3, 4, 5),
    )

    entries, diagnostics = build_manifest(str(tmp_path), ["NCG"])

    assert entries[0].filepath == str(path)
    assert entries[0].timestamp == datetime(2019, 1, 2, 3, 4, 5, 123456)
    assert diagnostics["filename_timestamp_fallbacks"] == 1


def test_four_class_manifest_preserves_broad_interval_and_context(tmp_path):
    path = write_lig(
        tmp_path / "NNBE" / "night" / "night_1500-3000km" / "sample.lig"
    )

    entries, diagnostics = build_manifest(str(tmp_path), ["NNBE"])

    assert entries[0].filepath == str(path)
    assert entries[0].type_idx == 0
    assert entries[0].dist_bin == -1
    assert entries[0].distance_low_km == 1500
    assert entries[0].distance_high_km == 3000
    assert entries[0].is_daytime is False
    assert diagnostics["distance_labeled_files"] == 1


def test_build_piece_manifest_keeps_individual_timestamps_and_interval(tmp_path):
    path = write_lig(
        tmp_path / "NCG" / "0-100km" / "sample.lig",
        pieces=3,
        piece_timestamps=[
            (20, 1, 1, 0, 0, 2),
            (20, 1, 1, 0, 0, 0),
            (20, 1, 1, 0, 0, 1),
        ],
    )
    files, _ = build_manifest(str(tmp_path), ["NCG"])

    pieces = build_piece_manifest(files)

    assert [(item.filepath, item.piece_index) for item in pieces] == [
        (str(path), 0), (str(path), 1), (str(path), 2)
    ]
    assert [item.timestamp.second for item in pieces] == [2, 0, 1]
    assert all(item.distance_low_km == 0 for item in pieces)
