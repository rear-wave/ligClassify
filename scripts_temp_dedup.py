"""Dedup .lig waveforms by (timestamp, waveform) identity — memory-efficient.

Reads a classified directory (IC/NCG/NNBE/PCG/PNBE subdirs), and for each type
keeps only the FIRST occurrence of each unique (timestamp_bytes, waveform_bytes)
event. Writes a new directory mirroring the type structure, with deduplicated
pieces repacked into output .lig groups (up to 512 pieces per file) named by
first-piece timestamp, preserving byte-exact piece content.

Memory-efficient: the dedup key is a SHA-256 digest of (timestamp + waveform),
NOT the raw 32KB waveform bytes, so memory stays tiny even for millions of
pieces.

Original data is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import struct
from datetime import datetime, timedelta
from pathlib import Path

from data.lig import LigFileIndex, LigOutputRegrouper, read_file_header

TYPE_NAMES = ("IC", "NCG", "NNBE", "PCG", "PNBE")
HEADER_BYTES = 112
PIECE_BYTES = 32208


def _decode_ts(raw: bytes) -> datetime | None:
    try:
        year, month, day, hour, minute, second = struct.unpack_from("<6i4x", raw, 0)
        sec_frac = struct.unpack_from("<d", raw, 28)[0]
        if year < 100:
            year += 2000
        return datetime(year, month, day) + timedelta(
            hours=hour, minutes=minute, seconds=second + sec_frac
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    if dst == src or dst.resolve() in [p.resolve() for p in src.parents]:
        raise ValueError("output must not be inside or equal to input")
    dst.mkdir(parents=True, exist_ok=True)

    for type_name in TYPE_NAMES:
        type_dir = src / type_name
        if not type_dir.is_dir():
            continue
        files = sorted(type_dir.rglob("*.lig"))
        if not files:
            continue
        if type_name == "IC":
            # IC is never deduplicated and stays in the original directory;
            # it is skipped entirely here (no copy, no dedup).
            print(f"[{type_name}] skipped (IC not deduplicated)", flush=True)
            continue

        seen: set[str] = set()      # sha256 digests, small memory
        kept = removed = 0
        pipeline: list[list] = []   # per-file buffered kept raw pieces
        # For each source file independently to keep the output structured,
        # we repack kept pieces with LigOutputRegrouper (type-level grouping).
        regrouper = LigOutputRegrouper(dst)
        for source_path in files:
            header = read_file_header(source_path)
            with LigFileIndex([source_path], validate=False) as index:
                with open(source_path, "rb") as raw:
                    for i in range(len(index)):
                        piece_off = HEADER_BYTES + i * PIECE_BYTES
                        raw.seek(piece_off + 108)
                        ts_bytes = raw.read(36)
                        raw.seek(piece_off + 208)
                        wf_bytes = raw.read(PIECE_BYTES - 208)
                        digest = hashlib.sha256(ts_bytes + wf_bytes).hexdigest()
                        if digest in seen:
                            removed += 1
                            continue
                        seen.add(digest)
                        raw.seek(piece_off)
                        piece = raw.read(PIECE_BYTES)
                        regrouper.add(type_name, header, piece, _decode_ts(ts_bytes))
                        kept += 1
        regrouper.flush_all()
        print(f"[{type_name}] kept={kept} removed_duplicates={removed} -> {dst}",
              flush=True)

    print("DEDUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()