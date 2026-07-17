import struct
from datetime import datetime

import numpy as np
import pytest

from data.lig import (
    FILE_HEADER_BYTES,
    PIECE_BYTES,
    WAVEFORM_SAMPLES,
    LigFileIndex,
    LigFormatError,
    read_file_header,
    read_lig_timestamp,
    read_raw_piece,
    write_lig_file,
)


def make_piece(value, year=2020, month=1, day=2, hour=3):
    raw = bytearray(PIECE_BYTES)
    struct.pack_into("<6i4x", raw, 108, year - 2000, month, day, hour, 4, 5)
    struct.pack_into("<d", raw, 136, 0.25)
    waveform = np.full(WAVEFORM_SAMPLES, value, dtype="<u2")
    raw[208:208 + waveform.nbytes] = waveform.tobytes()
    return bytes(raw)


def write_source(path, pieces):
    header = bytearray(FILE_HEADER_BYTES)
    struct.pack_into("<i", header, 4, len(pieces))
    path.write_bytes(bytes(header) + b"".join(pieces))


def test_raw_piece_round_trip_is_byte_exact(tmp_path):
    source = tmp_path / "source.lig"
    pieces = [make_piece(11), make_piece(22)]
    write_source(source, pieces)
    index = LigFileIndex([source], validate=True)

    assert np.all(index.read_piece(1) == 22)
    assert read_raw_piece(source, 1) == pieces[1]

    output = tmp_path / "output.lig"
    write_lig_file(output, source.read_bytes()[:FILE_HEADER_BYTES], [pieces[1]])
    assert output.read_bytes()[FILE_HEADER_BYTES:] == pieces[1]
    assert struct.unpack_from("<i", output.read_bytes(), 4)[0] == 1


def test_index_batch_reads_waveforms_and_piece_timestamps(tmp_path):
    first = tmp_path / "first.lig"
    second = tmp_path / "second.lig"
    write_source(first, [make_piece(3, hour=8), make_piece(4, hour=9)])
    write_source(second, [make_piece(5, hour=20)])
    index = LigFileIndex([first, second], validate=True)

    waveforms = index.read_pieces_batch([2, 0])

    assert len(waveforms) == 2
    assert waveforms[0].dtype == np.float32
    assert np.all(waveforms[0] == 5)
    assert np.all(waveforms[1] == 3)
    assert index.read_timestamps_batch([1, 2]) == [
        datetime(2020, 1, 2, 9, 4, 5, 250000),
        datetime(2020, 1, 2, 20, 4, 5, 250000),
    ]
    assert read_lig_timestamp(first, 0) == datetime(
        2020, 1, 2, 8, 4, 5, 250000
    )


def test_validation_rejects_declared_piece_count_size_mismatch(tmp_path):
    source = tmp_path / "truncated.lig"
    write_source(source, [make_piece(7)])
    raw = bytearray(source.read_bytes())
    struct.pack_into("<i", raw, 4, 2)
    source.write_bytes(raw)

    with pytest.raises(LigFormatError, match="size mismatch"):
        LigFileIndex([source], validate=True)


def test_short_header_and_reconstructed_piece_are_rejected(tmp_path):
    short = tmp_path / "short.lig"
    short.write_bytes(b"short")

    with pytest.raises(LigFormatError, match="short LIG header"):
        read_file_header(short)
    with pytest.raises(LigFormatError, match="reconstructed piece"):
        write_lig_file(tmp_path / "bad.lig", bytes(FILE_HEADER_BYTES), [b"short"])
