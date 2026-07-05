"""
ligClassify — classify .lig files and output by class
"""

import os, struct, argparse, logging
from collections import defaultdict

import numpy as np
import torch
from models import create_model
from data.preprocessing import preprocess_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LIG_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\LigHead.lig"
PIECE_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\Limitbyt"
MAX_PER_FILE = 512


def get_file_piece_size(fpath):
    """Determine piece size from file header version, or compute from actual layout."""
    with open(fpath, 'rb') as f:
        version = struct.unpack('<i', f.read(4))[0]
    std_psz = 32208 if version == 1001 else 32464
    # Verify: does file size = 112 + NumOfPiece * std_psz?
    fsize = os.path.getsize(fpath)
    with open(fpath, 'rb') as f:
        f.seek(4)
        n = struct.unpack('<i', f.read(4))[0]
    if n > 0 and (fsize - 112) % n == 0:
        actual_psz = (fsize - 112) // n
        if actual_psz != std_psz:
            return actual_psz
    return std_psz


def read_file_pieces(fpath):
    """Read all pieces from a .lig file.
    Uses standard piece size for each version, with waveform at offset hdr_size.
    Piece 0 has valid timestamp; pieces 1+ may have garbage timestamps,
    in which case we increment from piece 0.
    """
    try:
        with open(fpath, 'rb') as f:
            f.seek(0, 2)
            fsize = f.tell()
            if fsize < 116:  # too small: 112 header + 4 bytes minimum
                return
            f.seek(0)
            raw4 = f.read(4)
            if len(raw4) < 4:
                return
            file_ver = struct.unpack('<i', raw4)[0]
    except Exception:
        return
    hdr_size = 208 if file_ver == 1001 else 464
    psz = 32208 if file_ver == 1001 else 32464

    with open(fpath, 'rb') as f:
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(0)
        n_pieces = max(1, (fsize - 112) // psz)
        f.seek(112)
        try:
            raw0 = f.read(psz)
        except OSError:
            return
        if len(raw0) < hdr_size + 32000:
            return

        # Base timestamp from piece 0
        Y0, M0, D0, h0, m0, s0 = struct.unpack_from('<6i', raw0, 108)
        sec0 = struct.unpack_from('<d', raw0, 136)[0]
        ts0 = f'{Y0:02d}{M0:02d}{D0:02d}{h0:02d}{m0:02d}{s0:02d}{sec0:010.7f}'
        wf0 = np.frombuffer(raw0[hdr_size:hdr_size+32000], dtype=np.uint16)
        yield wf0.astype(np.float32), wf0, ts0

        for pi in range(1, n_pieces):
            try:
                f.seek(112 + pi * psz)
                raw = f.read(psz)
            except OSError:
                break  # file truncated or I/O error
            if len(raw) < psz:
                continue
            wf = np.frombuffer(raw[hdr_size:hdr_size+32000], dtype=np.uint16)
            # Try parse timestamp; if invalid, increment from piece 0
            try:
                Y, M, D, h, m, s = struct.unpack_from('<6i', raw, 108)
                sec = struct.unpack_from('<d', raw, 136)[0]
                if 0 <= Y <= 99 and 1 <= M <= 12 and 1 <= D <= 31 and 0 <= h <= 23:
                    ts = f'{Y:02d}{M:02d}{D:02d}{h:02d}{m:02d}{s:02d}{sec:010.7f}'
                else:
                    raise ValueError
            except Exception:
                ts = f'{Y0:02d}{M0:02d}{D0:02d}{h0:02d}{m0:02d}{s0:02d}{sec0 + pi * 0.0032:010.7f}'
            yield wf.astype(np.float32), wf, ts


def repack(wf, ts, ph):
    piece = np.asarray(wf, dtype=np.uint16)
    if len(piece) < 16000:
        piece = np.pad(piece, (0, 16000 - len(piece)), 'constant')
    else:
        piece = piece[:16000]
    try:
        Y = int(ts[:2]); M = int(ts[2:4]); D = int(ts[4:6])
        h = int(ts[6:8]); m = int(ts[8:10])
        si = int(float(ts[10:12])); S = float(ts[12:])
    except Exception:
        Y = M = D = h = m = si = 0; S = 0.0
    out = ph + struct.pack('16000H', *piece.tolist())
    out = out[:108] + struct.pack('6i4x', Y, M, D, h, m, si) + struct.pack('d', S) + out[144:]
    return out


def write_file(evts, out_dir, cls, lh, ph):
    if not evts:
        return
    ts = evts[0][1]
    ip, dp = (ts.split('.', 1) + [''])[:2]
    ip = ip.zfill(12)[:12]; dp = dp.ljust(7, '0')[:7]
    fn = f"GZ_{ip}.{dp}.lig"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fn)
    n = 1
    base, ext = os.path.splitext(fn)
    while os.path.exists(path):
        path = os.path.join(out_dir, f"{base}_{n}{ext}"); n += 1
    with open(path, 'wb') as f:
        fhdr = bytearray(lh)
        struct.pack_into('i', fhdr, 4, len(evts))
        f.write(fhdr)
        for wf, ts, _ in evts:
            f.write(repack(wf, ts, ph))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--date_start", type=str, default=None, help="e.g. 20210922")
    p.add_argument("--date_end", type=str, default=None, help="e.g. 20211006")
    p.add_argument("--only_class", type=str, default=None, help="Only output this class (e.g. PCG)")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False

    with open(LIG_HEAD, 'rb') as f:
        lh = f.read()
    with open(PIECE_HEAD, 'rb') as f:
        ph = f.read()

    ckpt = torch.load(args.model, map_location=dev, weights_only=False)
    classes = ckpt["class_names"]; nc = len(classes)
    model = create_model(nc).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(os.path.join(r, f)
                   for r, _, fs in os.walk(args.input_dir)
                   for f in fs if f.endswith('.lig')
                   if not 'Index' in r)  # skip Index folders
    if args.date_start and args.date_end:
        import re
        pattern = re.compile(r'GZ_(\d{8})')
        files = [f for f in files
                 if (m := pattern.search(f)) and args.date_start <= m.group(1) <= args.date_end]
    logger.info(f"{len(files)} files")

    # --- resume support ---
    processed_file = os.path.join(args.output_dir, "_processed.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    processed = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as pf:
            processed = set(line.strip() for line in pf if line.strip())
        logger.info(f"Resuming: {len(processed)} files already processed, {len(files) - len(set(files) & processed)} remaining")
    files = [f for f in files if f not in processed]

    all_classes = classes
    cache = {c: [] for c in all_classes}
    processed_batch = set()

    batch_wf, batch_meta = [], []
    for fpath in files:
        for wf_f32, wf_u16, ts in read_file_pieces(fpath):
            # Ensure uniform length (pad/truncate to 16000)
            if len(wf_f32) < 16000:
                wf_f32 = np.pad(wf_f32, (0, 16000 - len(wf_f32)), 'constant')
                wf_u16 = np.pad(wf_u16, (0, 16000 - len(wf_u16)), 'constant')
            elif len(wf_f32) > 16000:
                wf_f32 = wf_f32[:16000]
                wf_u16 = wf_u16[:16000]
            batch_wf.append(wf_f32)
            batch_meta.append((wf_u16, ts))
            if len(batch_wf) >= args.batch_size:
                # Preprocess & classify
                x = torch.from_numpy(preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')).unsqueeze(1).to(dev)
                with torch.no_grad():
                    probs = torch.softmax(model(x), -1).cpu().numpy()
                for i, (wf_u16, ts) in enumerate(batch_meta):
                    cls = classes[probs[i].argmax()]
                    if args.only_class and cls != args.only_class:
                        continue
                    cache[cls].append((wf_u16, ts, float(probs[i].max())))
                    if len(cache[cls]) >= MAX_PER_FILE:
                        chunk = cache[cls][:MAX_PER_FILE]
                        path = write_file(chunk, os.path.join(args.output_dir, cls), cls, lh, ph)
                        logger.info(f"  {cls}: {len(chunk)}p → {os.path.basename(path)}")
                        cache[cls] = cache[cls][MAX_PER_FILE:]
                batch_wf, batch_meta = [], []

        # Source file fully read → mark as processed
        processed_batch.add(fpath)
        if len(processed_batch) >= 10:
            processed |= processed_batch
            with open(processed_file, 'w') as pf:
                for p in sorted(processed):
                    pf.write(p + '\n')
            processed_batch.clear()

    # Flush remaining batch
    if batch_wf:
        x = torch.from_numpy(preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')).unsqueeze(1).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(x), -1).cpu().numpy()
        for i, (wf_u16, ts) in enumerate(batch_meta):
            cls = classes[probs[i].argmax()]
            mp = float(probs[i].max())
            if args.only_class and cls != args.only_class:
                continue
            cache[cls].append((wf_u16, ts, float(mp)))
            if len(cache[cls]) >= MAX_PER_FILE:
                chunk = cache[cls][:MAX_PER_FILE]
                path = write_file(chunk, os.path.join(args.output_dir, cls), cls, lh, ph)
                logger.info(f"  {cls}: {len(chunk)}p → {os.path.basename(path)}")
                cache[cls] = cache[cls][MAX_PER_FILE:]
    # Flush remaining processed files
    if processed_batch:
        processed |= processed_batch
        with open(processed_file, 'w') as pf:
            for p in sorted(processed):
                pf.write(p + '\n')
        processed_batch.clear()

    # Flush remaining
    for cls in all_classes:
        if cache[cls]:
            path = write_file(cache[cls], os.path.join(args.output_dir, cls), cls, lh, ph)
            logger.info(f"  {cls}: {len(cache[cls])}p → {os.path.basename(path)}")

    print(f"\n{'='*45}")
    for cls in all_classes:
        d = os.path.join(args.output_dir, cls)
        if os.path.isdir(d):
            fs = [f for f in os.listdir(d) if f.endswith('.lig')]
            print(f"  {cls:>5}: {len(fs)} file(s)")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
