"""Compare MTL model distance predictions with WWLLN ground truth."""
import os, sys, struct, time as time_mod
import numpy as np
import torch
from models import create_mtl_model
from data.preprocessing import preprocess_batch

# Config
MODEL_PATH = r"C:\Users\Administrator\Desktop\ligClassify\checkpoints_v2\mtl.pt"
LOC_FILE = r"E:\Guoxing Yang\2021.0413-type&distance\AE20210413.loc"
INPUT_DIR = r"E:\Guoxing Yang\typhoon_classified\2021.0413-2021.0425"
DATE_FILTER = "210413"

STATION_LAT, STATION_LON = 23.568582, 113.61469  # Guangzhou
MATCH_WINDOW_S = 0.05  # ±50ms
MAX_MATCH_DIST_KM = 3500  # only match WWLLN events within this range

BIN_STARTS = [i * 100 for i in range(30)]
NUM_CLASSES = 30


def parse_loc_date_sec(date_str, time_str):
    """Convert '2021/4/13' '00:00:00.022515' → seconds since midnight Apr 13."""
    parts = date_str.strip().split('/')
    month, day = int(parts[1]), int(parts[2])
    tparts = time_str.strip().split(':')
    h, m = int(tparts[0]), int(tparts[1])
    s = float(tparts[2])
    return h * 3600 + m * 60 + s


def load_wwlln(path):
    """Load WWLLN .loc file → numpy array [time_sec, dist_km]."""
    times, dists = [], []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            t_sec = parse_loc_date_sec(parts[0], parts[1])
            lat, lon = float(parts[2]), float(parts[3])
            dist = spherical_dist(lat, lon)
            times.append(t_sec)
            dists.append(dist)
    arr = np.array(list(zip(times, dists)))
    arr = arr[arr[:, 0].argsort()]  # sort by time
    return arr


def spherical_dist(lat, lon):
    """Distance from Guangzhou station to (lat, lon) in km."""
    from math import sin, cos, acos, pi
    r = 6371.0
    lat1, lon1 = STATION_LAT * pi / 180, STATION_LON * pi / 180
    lat2, lon2 = lat * pi / 180, lon * pi / 180
    cos_a = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)
    cos_a = max(min(cos_a, 1), -1)
    return r * acos(cos_a)


def find_match(wwlln, t_sec):
    """Find closest WWLLN event within MATCH_WINDOW_S. Returns dist or None."""
    lo = np.searchsorted(wwlln[:, 0], t_sec - MATCH_WINDOW_S)
    hi = np.searchsorted(wwlln[:, 0], t_sec + MATCH_WINDOW_S)
    if lo >= len(wwlln) or hi <= lo:
        return None
    window = wwlln[lo:hi]
    diffs = np.abs(window[:, 0] - t_sec)
    best_idx = np.argmin(diffs)
    if diffs[best_idx] > MATCH_WINDOW_S:
        return None
    dist = window[best_idx, 1]
    if dist > MAX_MATCH_DIST_KM:
        return None  # filter: only local events
    return dist


def parse_timestamp(ts_str):
    """Convert '210413045057.1570816' → seconds since midnight Apr 13."""
    h = int(ts_str[6:8])
    m = int(ts_str[8:10])
    s = int(ts_str[10:12])
    frac = float(ts_str[12:])
    return h * 3600 + m * 60 + s + frac


def read_piece_timestamps(fpath):
    """Read all piece timestamps from a .lig file."""
    fsize = os.path.getsize(fpath)
    n = (fsize - 112) // 32208
    with open(fpath, 'rb') as f:
        for pi in range(n):
            f.seek(112 + pi * 32208 + 108)
            Y, M, D, h, m, s = struct.unpack('<6i', f.read(24))
            _ = f.read(4)
            sec = struct.unpack('<d', f.read(8))[0]
            ts = f'{Y:02d}{M:02d}{D:02d}{h:02d}{m:02d}{s:02d}{sec:010.7f}'
            yield pi, ts


def read_waveform(fpath, pi):
    """Read one piece waveform as float32."""
    with open(fpath, 'rb') as f:
        f.seek(112 + pi * 32208 + 208)
        return np.frombuffer(f.read(32000), dtype=np.uint16).astype(np.float32)


def main():
    print("Loading WWLLN data...")
    wwlln = load_wwlln(LOC_FILE)
    print(f"  {len(wwlln)} events, time range: {wwlln[0,0]:.1f}s – {wwlln[-1,0]:.1f}s")

    print("Loading MTL model...")
    dev = "cuda"
    torch.backends.cudnn.enabled = False
    ckpt = torch.load(MODEL_PATH, map_location=dev, weights_only=False)
    types = ckpt['type_names']
    dist_names = ckpt['dist_names']
    model = create_mtl_model().to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Collect input files
    files = []
    for cls_dir in sorted(os.listdir(INPUT_DIR)):
        d = os.path.join(INPUT_DIR, cls_dir)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith('.lig') and DATE_FILTER in fn:
                    files.append(os.path.join(d, fn))

    print(f"Processing {len(files)} files...")

    results = []  # [(wwlln_dist, model_bin), ...]
    matched, unmatched = 0, 0
    batch_wf, batch_meta = [], []
    last_print = time_mod.time()

    for fi, fp in enumerate(sorted(files)):
        try:
            pieces = list(read_piece_timestamps(fp))
        except (PermissionError, OSError):
            continue
        for pi, ts_str in pieces:
            try:
                batch_wf.append(read_waveform(fp, pi))
            except (PermissionError, OSError):
                continue
            batch_meta.append((fp, pi, ts_str))

            if len(batch_wf) >= 256:
                # Process batch
                wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')
                x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
                with torch.no_grad():
                    t_logits, d_logits = model(x)
                    tp = torch.softmax(t_logits, -1).cpu().numpy()
                    dp = [torch.softmax(h, -1).cpu().numpy() for h in d_logits]

                for i in range(len(batch_wf)):
                    _, pi_i, ts_str = batch_meta[i]
                    t_sec = parse_timestamp(ts_str)
                    ww_dist = find_match(wwlln, t_sec)
                    if ww_dist is not None:
                        t_idx = tp[i].argmax()
                        tname = types[t_idx]
                        if tname != 'IC':
                            hi = dist_names.index(tname)
                            d_idx = dp[hi][i].argmax()
                            bin_center = BIN_STARTS[d_idx] + 50
                            results.append((ww_dist, d_idx, bin_center))
                        else:
                            results.append((ww_dist, -1, -1))
                        matched += 1
                    else:
                        unmatched += 1

                batch_wf, batch_meta = [], []

            now = time_mod.time()
            if now - last_print > 10:
                print(f"  [{fi+1}/{len(files)}] matched={matched} unmatched={unmatched}", flush=True)
                last_print = now

    # Final batch
    if batch_wf:
        wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')
        x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
        with torch.no_grad():
            t_logits, d_logits = model(x)
            tp = torch.softmax(t_logits, -1).cpu().numpy()
            dp = [torch.softmax(h, -1).cpu().numpy() for h in d_logits]
        for i in range(len(batch_wf)):
            _, pi_i, ts_str = batch_meta[i]
            t_sec = parse_timestamp(ts_str)
            ww_dist = find_match(wwlln, t_sec)
            if ww_dist is not None:
                t_idx = tp[i].argmax()
                tname = types[t_idx]
                if tname != 'IC':
                    hi = dist_names.index(tname)
                    d_idx = dp[hi][i].argmax()
                    bin_center = BIN_STARTS[d_idx] + 50
                    results.append((ww_dist, d_idx, bin_center))
                else:
                    results.append((ww_dist, -1, -1))
                matched += 1

    # ── Analysis ──
    print(f"\n{'='*60}")
    print(f"Total matched: {matched}, unmatched: {unmatched}")
    if not results:
        print("No results to analyze.")
        return

    arr = np.array(results)
    non_ic = arr[arr[:, 1] >= 0]

    # Save matched results
    match_file = LOC_FILE.replace('.loc', '_model_matched.csv')
    ww_km_all = non_ic[:, 0]
    mod_bin = non_ic[:, 1].astype(int)
    mod_km = non_ic[:, 2].astype(int)
    with open(match_file, 'w') as f:
        f.write('WWLLN_km,Model_Bin,Model_km,Error_km\n')
        for i in range(len(non_ic)):
            err = abs(ww_km_all[i] - mod_km[i])
            f.write(f'{ww_km_all[i]:.1f},{mod_bin[i]},{mod_km[i]},{err:.0f}\n')
    print(f'Matched results saved: {match_file}\n')

    # Distance comparison
    ww_km = non_ic[:, 0]
    model_km = non_ic[:, 2]
    abs_err = np.abs(ww_km - model_km)
    rel_err = abs_err / (ww_km + 1e-6) * 100

    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    medae = np.median(abs_err)
    n = len(non_ic)

    within_100 = (abs_err <= 100).sum() / n * 100
    within_200 = (abs_err <= 200).sum() / n * 100
    within_500 = (abs_err <= 500).sum() / n * 100

    print(f"\nNon-IC pieces: {n}")
    print(f"MAE: {mae:.1f} km")
    print(f"RMSE: {rmse:.1f} km")
    print(f"MedAE: {medae:.1f} km")
    print(f"Mean Rel Error: {rel_err.mean():.1f}%")
    print(f"Within ±100km: {within_100:.1f}%")
    print(f"Within ±200km: {within_200:.1f}%")
    print(f"Within ±500km: {within_500:.1f}%")

    # Per distance range
    for lo, hi, label in [(0, 500, "0-500km"), (500, 1500, "500-1500km"),
                           (1500, 3000, "1500-3000km"), (3000, 9999, ">3000km")]:
        mask = (ww_km >= lo) & (ww_km < hi)
        if mask.sum() > 0:
            m = abs_err[mask].mean()
            print(f"  {label}: {mask.sum()} pieces, MAE={m:.1f}km")

    # Confusion: model bin vs WWLLN bin
    ww_bin = np.clip((ww_km / 100).astype(int), 0, 29)
    mod_bin = non_ic[:, 1].astype(int)
    same_bin = (ww_bin == mod_bin).sum() / n * 100
    within_1 = (np.abs(ww_bin - mod_bin) <= 1).sum() / n * 100
    within_2 = (np.abs(ww_bin - mod_bin) <= 2).sum() / n * 100
    print(f"\nBin accuracy: {same_bin:.1f}%")
    print(f"Within ±1 bin: {within_1:.1f}%")
    print(f"Within ±2 bin: {within_2:.1f}%")

    # IC rate
    ic_rate = (arr[:, 1] < 0).sum() / len(arr) * 100
    print(f"\nIC rate: {ic_rate:.1f}%")


if __name__ == "__main__":
    main()
