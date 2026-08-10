"""Fast WWLLN distance accuracy evaluation with batch processing.

Uses iter_lig_batches to read each source file's timestamps + waveforms in
batches, computes peak offsets in batch, then matches against WWLLN. Correct
event_time = piece_timestamp + peak_offset (replicates code/2.py repacklig).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from data.lig import iter_lig_batches

STATION_LAT = 23.568582
STATION_LON = 113.61469
EARTH_RADIUS_KM = 6371.0
C_KM_S = 299792.458
MAX_TIME_DIFF_S = 0.011
MAX_DISTANCE_KM = 3500.0
TOLERANCE_RATIO = 0.10
DISTANCE_NAMES = ("NCG", "NNBE", "PCG", "PNBE")

_FC_NORM = 700_000.0 / (5_000_000.0 / 2.0)
_B, _A = butter(3, _FC_NORM, btype="low")


def spherical_distance(lat1, lon1, lat2, lon2):
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    ca = (math.sin(lat1r) * math.sin(lat2r)
          + math.cos(lat1r) * math.cos(lat2r) * math.cos(lon1r - lon2r))
    return EARTH_RADIUS_KM * math.acos(max(min(ca, 1.0), -1.0))


def load_wwlln(path):
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 4:
                continue
            try:
                lat, lon = float(p[2]), float(p[3])
                yy, mo, dd = [int(x) for x in p[0].split("/")]
                hm = p[1].split(".")
                hms = [int(x) for x in hm[0].split(":")]
                micro = int(hm[1].ljust(6, "0")[:6]) if len(hm) > 1 else 0
                if yy < 100:
                    yy += 2000
                events.append((datetime(yy, mo, dd, hms[0], hms[1], hms[2])
                               + timedelta(microseconds=micro), lat, lon))
            except Exception:
                continue
    events.sort(key=lambda e: e[0])
    return events


def match_wwlln(lig_time, events, times):
    i = bisect.bisect_right(times, lig_time)
    if i <= 0:
        return None
    evt_time, lat, lon = events[i - 1]
    dt = (lig_time - evt_time).total_seconds()
    if not (0 < dt <= MAX_TIME_DIFF_S):
        return None
    actual = spherical_distance(STATION_LAT, STATION_LON, lat, lon)
    if actual > MAX_DISTANCE_KM:
        return None
    if abs(actual - dt * C_KM_S) <= actual * TOLERANCE_RATIO:
        return (lat, lon, actual)
    return None


def peak_offset(waveform: np.ndarray) -> float:
    piece = waveform - float(np.mean(waveform))
    idx_max = int(np.where(piece == piece.max())[0][0])
    begin = max(0, min(idx_max - 4000, len(piece) - 16000))
    window = piece[begin:begin + 16000]
    filtered = filtfilt(_B, _A, window)
    return int(np.argmax(np.abs(filtered))) * 0.0002 * 0.001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_root", required=True)
    parser.add_argument("--lig_base", required=True)
    parser.add_argument("--wwlln_dir", default="E:/Guoxing Yang/WWLLN")
    args = parser.parse_args()

    pred_root = Path(args.predictions_root)
    lig_base = Path(args.lig_base)
    wwlln_dir = Path(args.wwlln_dir)

    wwlln_cache: dict = {}
    all_err_bin, all_err_km = [], []
    exact = w100 = w200 = w500 = matched = total = 0
    per_type = {t: {"n": 0, "exact": 0, "w100": 0, "w200": 0, "mae": 0.0} for t in DISTANCE_NAMES}

    for type_name in DISTANCE_NAMES:
        csv_path = pred_root / type_name / f"{type_name}_distance_predictions.csv"
        if not csv_path.is_file():
            continue
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        print(f"{type_name}: {len(rows)} predictions", flush=True)
        # index predictions by (source_path, piece_index)
        pred_map: dict = {}
        for r in rows:
            pred_map.setdefault(r["source_path"], {})[int(r["piece_index"])] = r

        for src_rel, piece_preds in pred_map.items():
            date = "".join(c for c in src_rel if c.isdigit())[:8]
            gz_dir = lig_base / f"GZ_{date}"
            if not gz_dir.is_dir():
                gz_dir = lig_base
            src = gz_dir / src_rel
            if not src.is_file():
                continue
            if date not in wwlln_cache:
                loc = wwlln_dir / f"AE{date}.loc"
                wwlln_cache[date] = (load_wwlln(loc), ) if loc.is_file() else None
            cached = wwlln_cache[date]
            if not cached:
                continue
            events = cached[0]
            times = [e[0] for e in events]
            # batch read this file's waveforms + timestamps
            for start, waveforms, timestamps, raw_pieces in iter_lig_batches(
                src, 256, allow_invalid_timestamps=True
            ):
                for i, (wf, ts) in enumerate(zip(waveforms, timestamps)):
                    pi = start + i
                    r = piece_preds.get(pi)
                    if r is None:
                        continue
                    total += 1
                    if ts is None:
                        continue
                    try:
                        off = peak_offset(wf)
                    except Exception:
                        off = 0.0
                    lig_t = ts + timedelta(seconds=off)
                    res = match_wwlln(lig_t, events, times)
                    if res is None:
                        continue
                    _, _, true_km = res
                    pred_km = float(r["expected_distance_km"])
                    pred_bin = int(r["distance_bin"])
                    true_bin = min(int(true_km // 100), 29)
                    matched += 1
                    eb = pred_bin - true_bin
                    ek = pred_km - true_km
                    all_err_bin.append(eb)
                    all_err_km.append(ek)
                    if eb == 0: exact += 1
                    if abs(eb) <= 1: w100 += 1
                    if abs(eb) <= 2: w200 += 1
                    if abs(eb) <= 5: w500 += 1
                    pt = per_type[type_name]
                    pt["n"] += 1
                    if eb == 0: pt["exact"] += 1
                    if abs(eb) <= 1: pt["w100"] += 1
                    if abs(eb) <= 2: pt["w200"] += 1
                    pt["mae"] += abs(ek)
        pt = per_type[type_name]
        if pt["n"]:
            print(f"  {type_name} matched={pt['n']} exact={pt['exact']/pt['n']:.3f} "
                  f"w200={pt['w200']/pt['n']:.3f} mae={pt['mae']/pt['n']:.1f}", flush=True)

    if matched == 0:
        print("NO MATCHED EVENTS")
        return
    eb = np.asarray(all_err_bin, dtype=np.float64)
    ek = np.asarray(all_err_km, dtype=np.float64)
    print("\n===== WWLLN DISTANCE ACCURACY / ERROR =====", flush=True)
    print(f"matched: {matched}", flush=True)
    print(f"exact bin accuracy : {exact/matched:.4f}", flush=True)
    print(f"within 100 km      : {w100/matched:.4f}", flush=True)
    print(f"within 200 km      : {w200/matched:.4f}", flush=True)
    print(f"within 500 km      : {w500/matched:.4f}", flush=True)
    print(f"MAE bins           : {np.abs(eb).mean():.3f}", flush=True)
    print(f"MAE km             : {np.abs(ek).mean():.1f}", flush=True)
    print(f"RMSE km            : {np.sqrt((ek**2).mean()):.1f}", flush=True)
    print(f"bias km            : {ek.mean():.1f}", flush=True)
    print(f"median abs err km  : {np.median(np.abs(ek)):.1f}", flush=True)
    print("\n--- per type ---", flush=True)
    for t in DISTANCE_NAMES:
        pt = per_type[t]
        if pt["n"]:
            print(f"{t}: n={pt['n']} exact={pt['exact']/pt['n']:.3f} "
                  f"w100={pt['w100']/pt['n']:.3f} w200={pt['w200']/pt['n']:.3f} "
                  f"mae_km={pt['mae']/pt['n']:.1f}", flush=True)


if __name__ == "__main__":
    main()