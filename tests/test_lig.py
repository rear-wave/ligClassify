import json
import struct
from datetime import datetime

import numpy as np
import pytest
import data.lig as lig_module

from data.lig import (
    FILE_HEADER_BYTES,
    PIECE_BYTES,
    THREE_CHANNEL_PIECE_BYTES,
    WAVEFORM_SAMPLES,
    LigFileIndex,
    LigFormatError,
    iter_inference_lig_batches,
    iter_lig_batches,
    read_file_header,
    read_lig_timestamp,
    read_raw_piece,
    write_lig_file,
)


def make_piece(
    value,
    year=2020,
    month=1,
    day=2,
    hour=3,
    piece_bytes=PIECE_BYTES,
    waveform_offset=208,
):
    raw = bytearray(piece_bytes)
    struct.pack_into("<6i4x", raw, 108, year - 2000, month, day, hour, 4, 5)
    struct.pack_into("<d", raw, 136, 0.25)
    waveform = np.full(WAVEFORM_SAMPLES, value, dtype="<u2")
    raw[waveform_offset:waveform_offset + waveform.nbytes] = waveform.tobytes()
    return bytes(raw)


def write_source(path, pieces):
    header = bytearray(FILE_HEADER_BYTES)
    struct.pack_into("<i", header, 4, len(pieces))
    path.write_bytes(bytes(header) + b"".join(pieces))


def test_resume_state_retries_a_transient_windows_replace_lock(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    journal = lig_module.LigInferenceJournal(
        output_dir, ["piece_key"], {}, {}, None, resume=False
    )
    real_replace = lig_module.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "temporarily locked")
        return real_replace(source, destination)

    monkeypatch.setattr(lig_module.os, "replace", flaky_replace)
    monkeypatch.setattr(lig_module.time, "sleep", lambda _delay: None)
    with journal:
        journal.complete_source("ZH_20240901/a.lig")

    payload = json.loads(journal.state_path.read_text(encoding="utf-8"))
    assert attempts == 3
    assert payload["completed_source"] == "ZH_20240901/a.lig"
    assert not list(output_dir.glob("*.tmp"))


def test_resume_state_retries_a_transient_windows_delete_lock(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    journal = lig_module.LigInferenceJournal(
        output_dir, ["piece_key"], {}, {}, None, resume=False
    )
    with journal:
        journal.complete_source("ZH_20240901/a.lig")

    path_type = type(journal.state_path)
    real_unlink = path_type.unlink
    attempts = 0

    def flaky_unlink(path, *args, **kwargs):
        nonlocal attempts
        if path == journal.state_path:
            attempts += 1
            if attempts < 3:
                raise PermissionError(13, "temporarily locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "unlink", flaky_unlink)
    monkeypatch.setattr(lig_module.time, "sleep", lambda _delay: None)
    journal.finish()

    assert attempts == 3
    assert not journal.state_path.exists()


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


def test_extended_header_layout_preserves_waveform_timestamp_and_raw_bytes(
    tmp_path,
):
    source = tmp_path / "extended.lig"
    piece = make_piece(
        37,
        hour=11,
        piece_bytes=PIECE_BYTES + 256,
        waveform_offset=208 + 256,
    )
    write_source(source, [piece])

    with LigFileIndex([source], validate=True) as index:
        assert np.all(index.read_piece(0) == 37)
        assert index.read_timestamps_batch([0]) == [
            datetime(2020, 1, 2, 11, 4, 5, 250000)
        ]
    assert read_raw_piece(source, 0) == piece

    output = tmp_path / "extended-output.lig"
    write_lig_file(output, source.read_bytes()[:FILE_HEADER_BYTES], [piece])
    assert output.read_bytes()[FILE_HEADER_BYTES:] == piece


def test_three_channel_layout_reads_first_channel_and_preserves_raw_bytes(
    tmp_path,
):
    source = tmp_path / "three-channel.lig"
    pieces = []
    for hour in (11, 12):
        piece = bytearray(
            make_piece(
                17,
                hour=hour,
                piece_bytes=THREE_CHANNEL_PIECE_BYTES,
                waveform_offset=208 + 256,
            )
        )
        second = np.full(WAVEFORM_SAMPLES, 29, dtype="<u2")
        third = np.full(WAVEFORM_SAMPLES, 41, dtype="<u2")
        piece[32464:64464] = second.tobytes()
        piece[64464:96464] = third.tobytes()
        pieces.append(bytes(piece))
    write_source(source, pieces)

    with LigFileIndex([source], validate=True) as index:
        assert np.all(index.read_piece(0) == 17)
        assert index.read_timestamps_batch([1]) == [
            datetime(2020, 1, 2, 12, 4, 5, 250000)
        ]
    assert read_lig_timestamp(source, 1) == datetime(
        2020, 1, 2, 12, 4, 5, 250000
    )
    assert read_raw_piece(source, 1) == pieces[1]

    output = tmp_path / "three-channel-output.lig"
    write_lig_file(output, source.read_bytes()[:FILE_HEADER_BYTES], [pieces[1]])
    assert output.read_bytes()[FILE_HEADER_BYTES:] == pieces[1]


def test_long_three_channel_layout_crops_first_channel_and_preserves_raw_bytes(
    tmp_path,
):
    source = tmp_path / "long-three-channel.lig"
    sample_count = 40_000
    piece = bytearray(464 + 3 * sample_count * 2)
    struct.pack_into("<2i", piece, 20, sample_count, 3)
    struct.pack_into("<6i4x", piece, 108, 20, 1, 2, 11, 4, 5)
    struct.pack_into("<d", piece, 136, 0.25)
    first = np.full(sample_count, 2100, dtype="<u2")
    first[20_000] = 2600
    second = np.full(sample_count, 3100, dtype="<u2")
    third = np.full(sample_count, 4100, dtype="<u2")
    piece[464:] = first.tobytes() + second.tobytes() + third.tobytes()
    write_source(source, [bytes(piece)])

    with LigFileIndex([source], validate=True) as index:
        waveform = index.read_piece(0)
        assert waveform.shape == (WAVEFORM_SAMPLES,)
        assert waveform[WAVEFORM_SAMPLES // 4] == 2600
        assert waveform.max() < 3000
    assert read_raw_piece(source, 0) == bytes(piece)

    output = tmp_path / "long-three-channel-output.lig"
    write_lig_file(output, source.read_bytes()[:FILE_HEADER_BYTES], [bytes(piece)])
    assert output.read_bytes()[FILE_HEADER_BYTES:] == bytes(piece)


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


def test_timestamp_normalizes_bounded_fractional_second_overflow(tmp_path):
    source = tmp_path / "overflow.lig"
    piece = bytearray(make_piece(3, hour=8))
    struct.pack_into("<d", piece, 136, 2.5195954)
    write_source(source, [bytes(piece)])

    assert read_lig_timestamp(source, 0) == datetime(
        2020, 1, 2, 8, 4, 7, 519595
    )


def test_inference_batches_can_coerce_missing_timestamp(tmp_path):
    source = tmp_path / "missing-time.lig"
    piece = bytearray(make_piece(3, hour=8))
    piece[108:136] = bytes(28)
    write_source(source, [bytes(piece)])

    batch = next(
        iter_lig_batches(
            source,
            batch_size=1,
            allow_invalid_timestamps=True,
        )
    )

    assert batch[2] == [None]


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


def test_inference_batches_warn_and_skip_an_empty_lig_file(tmp_path):
    source = tmp_path / "empty.lig"
    source.write_bytes(b"")

    with pytest.warns(RuntimeWarning, match="skipping invalid LIG file"):
        batches = list(iter_inference_lig_batches(source, batch_size=8))

    assert batches == []


def test_inference_batches_retry_then_skip_an_unreadable_lig(
    tmp_path, monkeypatch
):
    source = tmp_path / "unreadable.lig"
    source.write_bytes(b"synthetic")
    attempts = 0

    def denied(_path):
        nonlocal attempts
        attempts += 1
        raise PermissionError("locked")

    monkeypatch.setattr(lig_module, "read_file_header", denied)
    monkeypatch.setattr(lig_module.time, "sleep", lambda _delay: None)
    with pytest.warns(RuntimeWarning, match="after 3 attempts"):
        batches = list(
            lig_module.iter_inference_lig_batches(
                source,
                batch_size=8,
                skip_io_errors=True,
            )
        )

    assert attempts == 3
    assert batches == []


def test_inference_batches_resume_at_failed_batch_without_duplicates(
    tmp_path, monkeypatch
):
    source = tmp_path / "transient.lig"
    write_source(source, [make_piece(1), make_piece(2), make_piece(3)])
    real_batches = lig_module.iter_lig_batches
    starts = []

    def fail_after_first_batch(path, batch_size, **kwargs):
        starts.append(kwargs.get("start_piece", 0))
        for batch in real_batches(path, batch_size, **kwargs):
            yield batch
            if len(starts) == 1:
                raise PermissionError(13, "transient USB read failure")

    monkeypatch.setattr(lig_module, "iter_lig_batches", fail_after_first_batch)
    monkeypatch.setattr(lig_module.time, "sleep", lambda _delay: None)
    batches = list(
        lig_module.iter_inference_lig_batches(source, batch_size=1)
    )

    assert starts == [0, 1]
    assert [batch[1] for batch in batches] == [0, 1, 2]


def test_inference_batches_can_skip_an_unreadable_partial_remainder(
    tmp_path, monkeypatch
):
    source = tmp_path / "partial.lig"
    write_source(source, [make_piece(1), make_piece(2)])
    real_batches = lig_module.iter_lig_batches
    attempts = 0

    def fail_from_second_piece(path, batch_size, **kwargs):
        nonlocal attempts
        attempts += 1
        if kwargs.get("start_piece", 0) == 0:
            yield next(real_batches(path, batch_size, **kwargs))
        raise PermissionError(13, "persistent USB read failure")

    monkeypatch.setattr(lig_module, "iter_lig_batches", fail_from_second_piece)
    monkeypatch.setattr(lig_module.time, "sleep", lambda _delay: None)
    with pytest.warns(RuntimeWarning, match="from piece 1"):
        batches = list(
            lig_module.iter_inference_lig_batches(
                source,
                batch_size=1,
                skip_io_errors=True,
            )
        )

    assert attempts == 3
    assert [batch[1] for batch in batches] == [0]
