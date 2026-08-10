"""Fix GZ_unknown.lig source files in 2018 by writing valid timestamps.

For each GZ_unknown*.lig under a type directory, every piece's timestamp
field (offset 108..144) is garbled. This script rewrites each piece's
timestamp to a valid value: base date inferred from a sibling GZ_YYYYMMDD*.lig
file + the waveform peak offset (same method as code/2.py repacklig), so the
file becomes openable. Piece bytes are otherwise preserved.
"""

from __future__ import annotations

import argparse
import re
import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from data.lig import LigFileIndex

TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")
SAMPLE_RATE = 5_000_000.0


def _sibling_date(path: Path) -> datetime | None:
    for sib in sorted(path.parent.glob("GZ_*.lig")):
        m = re.search(r"GZ_(\d{8})", sib.name)
        if m:
            d = m.group(1)
            return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]))
    return None


def _encode_ts(ts: datetime) -> bytes:
    return struct.pack(
        "<6i4x", ts.year % 100, ts.month, ts.day, ts.hour, ts.minute, ts.second
    ) + struct.pack("<d", ts.microsecond / 1e6)


def _peak_offset_seconds(waveform: np.ndarray) -> float:
    piece = waveform - float(np.mean(waveform))
    idx_max = int(np.where(piece == piece.max())[0][0])
    begin = max(0, min(idx_max - 4000, len(piece) - 16000))
    window = piece[begin:begin + 16000]
    fc = 700_000.0 / (SAMPLE_RATE / 2.0)
    b, a = butter(3, fc, btype="low")
    filtered = filtfilt(b, a, window)
    peak_index = int(np.argmax(np.abs(filtered)))
    return peak_index * 0.0002 * 0.001


def fix_file(path: Path, base: datetime) -> int:
    with LigFileIndex([path], validate=True) as idx:
        pb = idx.piece_bytes_per_file[0]
        n = len(idx)
        waveforms = [idx.read_piece(i) for i in range(n)]
    raw = bytearray(path.read_bytes())
    fixed = 0
    for i in range(n):
        try:
            off = 112 + i * pb + 108
            # keep increasing time across pieces within the file
            ts = base + timedelta(seconds=i * 0.2) + timedelta(
                seconds=_peak_offset_seconds(waveforms[i])
            )
            raw[off:off + 36] = _encode_ts(ts)
            fixed += 1
        except Exception:
            continue
    tmp = path.with_suffix(".lig.tmp")
    tmp.write_bytes(bytes(raw))
    tmp.replace(path)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="year dir containing NCG/NNBE/PCG/PNBE")
    args = parser.parse_args()
    root = Path(args.root)
    total_files = 0
    total_pieces = 0
    for t in TYPE_NAMES:
        tdir = root / t
        if not tdir.is_dir():
            continue
        unk_files = sorted(tdir.glob("GZ_unknown*.lig"))
        if not unk_files:
            continue
        # one base date per type dir (from any sibling with a date)
        base = _sibling_date(tdir)
        if base is None:
            print(f"[skip] {t}: no sibling date found", flush=True)
            continue
        for f in unk_files:
            n = fix_file(f, base)
            total_files += 1
            total_pieces += n
        print(f"[{t}] fixed {len(unk_files)} files ({base.date()})", flush=True)
    print(f"DONE: {total_files} files, {total_pieces} pieces", flush=True)


if __name__ == "__main__":
    main()