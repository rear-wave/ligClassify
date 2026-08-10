"""Classify distance for all non-IC .lig pieces across all typhoon year dirs.

For each input year directory (and each GZ date subdir if present), for each of
the four non-IC type subdirs (NCG/NNBE/PCG/PNBE), run the matching distance
expert model and write:
  - <output>/<TYPE>/<LLLL-HHHHkm>/GZ_*.lig  (byte-exact regrouped by distance)
  - <output>/<TYPE>_distance_predictions.csv
using the same PreprocessConfig and forward_distance_type path as inference.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from checkpoints import load_model_checkpoint
from data.lig import LigFileIndex, LigOutputRegrouper, iter_lig_batches, read_file_header, read_lig_timestamp
from data.preprocess import PreprocessConfig, preprocess_views
from models import DISTANCE_NAMES

DISTANCE_BINS_KM = tuple(range(0, 3000, 100))

PRED_FIELDS = (
    "source_path piece_index piece_key final_type distance_bin "
    "distance_low_km distance_high_km expected_distance_km "
    "distance_confidence output_file".split()
)

TYPE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")


def _new_preprocess_config(checkpoint) -> PreprocessConfig:
    c = checkpoint.preprocess_config
    return PreprocessConfig(
        local_length=int(c["local_length"]),
        global_length=int(c["global_length"]),
        use_filter=bool(c.get("use_filter", True)),
        cutoff_hz=float(c.get("cutoff_hz", 120_000.0)),
        sample_rate_hz=float(c.get("sample_rate_hz", 5_000_000.0)),
        local_center_mode=str(c.get("local_center_mode", "peak_abs_v1")),
        local_energy_window=int(c.get("local_energy_window", 128)),
    )


def _is_daylight(timestamp: datetime | None) -> float:
    if timestamp is None:
        return 0.0
    local_hour = (timestamp.hour + 8 + timestamp.minute / 60.0) % 24
    return 1.0 if 5.5 <= local_hour < 19.0 else 0.0


def _event_time(path: Path, piece_index: int, waveform: np.ndarray,
                base_time: datetime | None) -> datetime | None:
    """Compute a valid event time (m_FirstPointTime + peak offset) like repacklig.

    Falls back to the file's piece-0 timestamp if this piece's timestamp is
    invalid, so every output piece gets a valid timestamp (openable file).
    """
    if base_time is None:
        try:
            base_time = read_lig_timestamp(path, 0)
        except Exception:
            base_time = None
    if base_time is None:
        # Source file has no valid timestamp at all (e.g. GZ_unknown.lig).
        # Fall back to a date inferred from a sibling file's name so the
        # output piece still gets a valid (date-accurate) timestamp.
        base_time = _fallback_date_from_siblings(path)
    if base_time is None:
        return None
    try:
        piece = waveform - float(np.mean(waveform))
        idx_max = int(np.where(piece == piece.max())[0][0])
        begin = max(0, min(idx_max - 4000, len(piece) - 16000))
        window = piece[begin:begin + 16000]
        fc = 700_000.0 / (5_000_000.0 / 2.0)
        b, a = butter(3, fc, btype="low")
        filtered = filtfilt(b, a, window)
        peak_index = int(np.argmax(np.abs(filtered)))
        offset = peak_index * 0.0002 * 0.001  # ms -> seconds (replicates original)
        return base_time + timedelta(seconds=offset)
    except Exception:
        return base_time


def _fallback_date_from_siblings(path: Path) -> datetime | None:
    """Infer a date from a sibling .lig file named GZ_YYYYMMDD... (best effort)."""
    try:
        import re
        for sib in sorted(path.parent.glob("GZ_*.lig")):
            m = re.search(r"GZ_(\d{8})", sib.name)
            if m:
                d = m.group(1)
                return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]))
    except Exception:
        pass
    return None


def _encode_timestamp(ts: datetime) -> bytes:
    """Encode a datetime as the 36-byte LIG timestamp field (6i4x + double)."""
    year = ts.year % 100
    return struct.pack(
        "<6i4x", year, ts.month, ts.day, ts.hour, ts.minute, ts.second
    ) + struct.pack("<d", ts.microsecond / 1e6)


def _patch_piece_timestamp(raw_piece: bytes, ts: datetime) -> bytes:
    """Return raw_piece with its timestamp field (offset 108..144) set to ts."""
    if ts is None:
        return raw_piece
    encoded = _encode_timestamp(ts)
    return raw_piece[:108] + encoded + raw_piece[144:]



def _collect_inputs(year_dir: Path):
    """Return list of (root_dir, label) where label prefixes output classes.

    - If the year dir directly has NCG/... subdirs -> [(year_dir, "")]
    - If it has GZ_* date subdirs (2016) -> one entry per GZ dir
    """
    entries = []
    for t in TYPE_NAMES:
        if (year_dir / t).is_dir():
            entries.append((year_dir, ""))
            break
    else:
        for sub in sorted(year_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("GZ_"):
                entries.append((sub, ""))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default=None,
                        help="only process year dirs whose name contains this substring")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve()
    model_root = Path(args.model_dir).resolve()

    # pre-load the four distance experts once
    experts = {}
    for type_name in DISTANCE_NAMES:
        ckpt = load_model_checkpoint(model_root / type_name / "model.pt", device)
        experts[type_name] = (ckpt.model.eval(), _new_preprocess_config(ckpt))

    year_dirs = sorted(p for p in in_root.iterdir() if p.is_dir())
    for year_dir in year_dirs:
        # skip obvious backups
        if year_dir.name.endswith("_old"):
            print(f"[skip] {year_dir.name} (_old)", flush=True)
            continue
        if args.only and args.only not in year_dir.name:
            print(f"[skip] {year_dir.name} (not --only)", flush=True)
            continue
        for src_root, _lbl in _collect_inputs(year_dir):
            if not _collect_inputs(year_dir):
                continue
            for type_name in TYPE_NAMES:
                type_dir = src_root / type_name
                if not type_dir.is_dir():
                    continue
                files = sorted(type_dir.rglob("*.lig"))
                if not files:
                    continue
                model, pc = experts[type_name]
                expert_index = DISTANCE_NAMES.index(type_name)
                # output grouped under <year>_distance/<TYPE>/...
                out_dir = out_root / f"{year_dir.name}_distance"
                type_out = out_dir / type_name
                type_out.mkdir(parents=True, exist_ok=True)
                regrouper = LigOutputRegrouper(out_dir)
                csv_path = type_out / f"{type_name}_distance_predictions.csv"
                n = 0
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=PRED_FIELDS, extrasaction="raise")
                    writer.writeheader()
                    for source_path in files:
                        header = read_file_header(source_path)
                        try:
                            base_time = read_lig_timestamp(source_path, 0)
                        except Exception:
                            base_time = None
                        for start, waveforms, timestamps, raw_pieces in iter_lig_batches(
                            source_path, args.batch_size, allow_invalid_timestamps=True
                        ):
                            values = np.asarray(waveforms, dtype=np.float32)
                            local, gview = preprocess_views(values, pc)
                            local_t = torch.from_numpy(local).unsqueeze(1).to(device)
                            gview_t = torch.from_numpy(gview).unsqueeze(1).to(device)
                            daylight = torch.tensor(
                                [[_is_daylight(ts)] for ts in timestamps],
                                dtype=torch.float32, device=device,
                            )
                            missing = torch.tensor(
                                [ts is None for ts in timestamps],
                                dtype=torch.bool, device=device,
                            )
                            with torch.inference_mode():
                                logits, _ = model.forward_distance_type(
                                    local_t, gview_t, daylight, expert_index=expert_index
                                )
                                alt = daylight.clone(); alt[missing] = 1.0
                                alt_logits, _ = model.forward_distance_type(
                                    local_t, gview_t, alt, expert_index=expert_index
                                )
                            probs = torch.softmax(logits.detach().float(), dim=1)
                            alt_probs = torch.softmax(alt_logits.detach().float(), dim=1)
                            mean_probs = torch.where(
                                missing.unsqueeze(1), 0.5 * (probs + alt_probs), probs
                            )
                            bins = mean_probs.argmax(dim=1).cpu().tolist()
                            centers = np.arange(50.0, 3050.0, 100.0)
                            exp_km = (mean_probs.cpu().numpy() * centers).sum(axis=1).tolist()
                            probs_np = mean_probs.cpu().numpy()
                            rel = source_path.relative_to(src_root).as_posix()
                            for i, (raw_piece, ts) in enumerate(zip(raw_pieces, timestamps)):
                                bin_i = bins[i]
                                low = DISTANCE_BINS_KM[bin_i]
                                out_class = f"{type_name}/{low:04d}-{low+100:04d}km"
                                # Repack-style: give every output piece a valid
                                # timestamp (m_FirstPointTime + peak offset) so
                                # the output .lig is openable (no GZ_unknown).
                                evt = _event_time(source_path, start + i,
                                                  values[i], base_time)
                                if evt is not None:
                                    raw_piece = _patch_piece_timestamp(raw_piece, evt)
                                    ts = evt
                                out_file = regrouper.add(out_class, header, raw_piece, ts)
                                writer.writerow({
                                    "source_path": rel,
                                    "piece_index": start + i,
                                    "piece_key": f"{rel}#{start+i}",
                                    "final_type": type_name,
                                    "distance_bin": bin_i,
                                    "distance_low_km": low,
                                    "distance_high_km": low + 100,
                                    "expected_distance_km": round(exp_km[i], 3),
                                    "distance_confidence": round(float(probs_np[i, bin_i]), 4),
                                    "output_file": out_file,
                                })
                                n += 1
                    regrouper.flush_all()
                print(f"[done] {year_dir.name} {type_name}: {n} pieces -> {csv_path.name}",
                      flush=True)
    print("ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()