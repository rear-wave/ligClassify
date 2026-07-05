"""
Quick MTL inference: type + distance for existing classified .lig files.
Filters by date in filename (YYMMDD pattern).
"""
import os, sys, re, struct, argparse
import numpy as np
import torch
from models import create_mtl_model
from data.preprocessing import preprocess_batch

LIG_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\LigHead.lig"
PIECE_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\Limitbyt"
MAX_PER_FILE = 512


def read_pieces_stream(fpath, chunk=256):
    """Generator: yields batches of (wf_float32, time_str) from a .lig file."""
    fsize = os.path.getsize(fpath)
    n_pieces = (fsize - 112) // 32208
    with open(fpath, 'rb') as f:
        batch_wf, batch_ts = [], []
        for pi in range(n_pieces):
            f.seek(112 + pi * 32208 + 208)
            wf = np.frombuffer(f.read(32000), dtype=np.uint16).astype(np.float32)
            ts = f"210413000000.{pi:010d}"
            batch_wf.append(wf)
            batch_ts.append(ts)
            if len(batch_wf) >= chunk:
                yield batch_wf, batch_ts
                batch_wf, batch_ts = [], []
        if batch_wf:
            yield batch_wf, batch_ts


def write_lig(pieces, outpath, lh, ph):
    """Write pieces to a .lig file, up to MAX_PER_FILE per file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    idx = 0
    base = outpath.rsplit('.', 1)[0]
    while idx < len(pieces):
        chunk = pieces[idx:idx + MAX_PER_FILE]
        fname = f"{base}.lig" if idx == 0 else f"{base}_{idx//MAX_PER_FILE+1}.lig"
        with open(fname, 'wb') as f:
            gh = bytearray(lh)
            struct.pack_into('<i', gh, 4, len(chunk))
            f.write(gh)
            for wf, ts in chunk:
                piece = np.asarray(wf[:16000], dtype=np.uint16)
                if len(piece) < 16000:
                    piece = np.pad(piece, (0, 16000 - len(piece)), 'constant')
                out = ph + struct.pack('16000H', *piece.tolist())
                Y = int(ts[:2]); M = int(ts[2:4]); D = int(ts[4:6])
                h = int(ts[6:8]); m = int(ts[8:10])
                si = int(float(ts[10:12])); S = float(ts[12:])
                out = out[:108] + struct.pack('6i4x', Y, M, D, h, m, si) + struct.pack('d', S) + out[144:]
                f.write(out)
        idx += MAX_PER_FILE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--date", required=True, help="e.g. 210413")
    p.add_argument("--model", default="./checkpoints_v2/mtl.pt")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.enabled = False

    # Load model
    ckpt = torch.load(args.model, map_location=dev, weights_only=False)
    types = ckpt['type_names']
    dist_names = ckpt['dist_names']
    bins = ckpt['dist_bin_starts']
    model = create_mtl_model().to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load templates
    with open(LIG_HEAD, 'rb') as f:
        lh = f.read()
    with open(PIECE_HEAD, 'rb') as f:
        ph = f.read()

    # Collect files (avoid slow os.walk on USB drives)
    files = []
    for cls_dir in sorted(os.listdir(args.input_dir)):
        d = os.path.join(args.input_dir, cls_dir)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith('.lig') and args.date in fn:
                    files.append(os.path.join(d, fn))
    print(f"Found {len(files)} files for date {args.date}")

    # Cache per output class
    os.makedirs(args.output_dir, exist_ok=True)
    cache = {}
    total, processed = 0, 0

    for fp in sorted(files):
        for batch_wf, batch_ts in read_pieces_stream(fp, args.batch_size):
            if not batch_wf:
                continue
            wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')
            x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
            with torch.no_grad():
                t_logits, d_logits = model(x)
                tp = torch.softmax(t_logits, -1)
                dp = [torch.softmax(h, -1) for h in d_logits]

            for i in range(len(batch_wf)):
                ti = tp[i].argmax().item()
                tname = types[ti]
                if tname == 'IC':
                    cls_out = 'IC'
                else:
                    hi = dist_names.index(tname)
                    di = dp[hi][i].argmax().item()
                    lo = bins[di]
                    cls_out = f'{tname}_{lo}-{lo+100}km'

                if cls_out not in cache:
                    cache[cls_out] = []
                cache[cls_out].append((batch_wf[i], batch_ts[i]))

                if len(cache[cls_out]) >= MAX_PER_FILE:
                    chunk = cache[cls_out][:MAX_PER_FILE]
                    path = os.path.join(args.output_dir, cls_out,
                                       f"GZ_{batch_ts[i].replace('.','')}.lig")
                    write_lig(chunk, path, lh, ph)
                    cache[cls_out] = cache[cls_out][MAX_PER_FILE:]

                processed += 1
                if processed % 10000 == 0:
                    print(f"  Processed {processed} pieces...")

    # Flush remaining
    for cls_out, pieces in cache.items():
        if pieces:
            ts = pieces[0][1]
            path = os.path.join(args.output_dir, cls_out,
                               f"GZ_{ts.replace('.','')}.lig")
            write_lig(pieces, path, lh, ph)

    print(f"\nDone. {processed} pieces classified into {len(cache)} categories.")
    for cls_out in sorted(os.listdir(args.output_dir)):
        d = os.path.join(args.output_dir, cls_out)
        if os.path.isdir(d):
            n = len(os.listdir(d))
            print(f"  {cls_out}: {n} files")


if __name__ == "__main__":
    main()
