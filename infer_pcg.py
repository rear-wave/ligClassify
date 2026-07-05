"""Stage 2: MTL distance inference on pre-classified PCG .lig files."""
import os, struct, argparse, numpy as np, torch
from models import create_mtl_model
from data.preprocessing import preprocess_batch

LIG_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\LigHead.lig"
PIECE_HEAD = r"C:\Users\Administrator\Desktop\LigEdit\Limitbyt"
MAX_PER_FILE = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="./checkpoints_v2/mtl.pt")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.enabled = False

    ckpt = torch.load(args.model, map_location=dev, weights_only=False)
    dist_names = ckpt['dist_names']
    bins = ckpt['dist_bin_starts']
    model = create_mtl_model().to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with open(LIG_HEAD, 'rb') as f:
        lh = f.read()
    with open(PIECE_HEAD, 'rb') as f:
        ph = f.read()

    # Collect PCG .lig files
    files = []
    for root, _, fns in os.walk(args.input_dir):
        for fn in fns:
            if fn.endswith('.lig'):
                files.append(os.path.join(root, fn))
    files.sort()
    print(f"Found {len(files)} PCG source files")

    os.makedirs(args.output_dir, exist_ok=True)
    cache = {}  # {label: [(wf, ts), ...]}
    processed, out_files = 0, 0
    pcg_idx = dist_names.index('PCG')

    for fp in files:
        fsize = os.path.getsize(fp)
        n_pieces = (fsize - 112) // 32208
        if n_pieces < 1:
            continue

        batch_wf, batch_data = [], []
        with open(fp, 'rb') as f:
            for pi in range(n_pieces):
                f.seek(112 + pi * 32208 + 208)
                wf = np.frombuffer(f.read(32000), dtype=np.uint16).astype(np.float32)
                f.seek(112 + pi * 32208 + 108)
                Y, M, D, h, m, s = struct.unpack('<6i', f.read(24))
                _ = f.read(4); sec = struct.unpack('<d', f.read(8))[0]
                ts = f'{Y:02d}{M:02d}{D:02d}{h:02d}{m:02d}{s:02d}{sec:010.7f}'
                batch_wf.append(wf); batch_data.append((wf, ts))

                if len(batch_wf) >= args.batch_size:
                    wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')
                    x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
                    with torch.no_grad():
                        _, d_logits = model(x)
                        dp = torch.softmax(d_logits[pcg_idx], -1).cpu().numpy()
                    for i in range(len(batch_wf)):
                        di = dp[i].argmax()
                        lo = bins[di]
                        label = f'PCG_{lo}-{lo+100}km'
                        wf_u16, ts = batch_data[i]
                        if label not in cache:
                            cache[label] = []
                        cache[label].append((wf_u16, ts))
                    processed += len(batch_wf)
                    batch_wf, batch_data = [], []

        if batch_wf:
            wf_pp = preprocess_batch(np.stack(batch_wf), normalize_mode='minmax')
            x = torch.from_numpy(wf_pp).unsqueeze(1).to(dev)
            with torch.no_grad():
                _, d_logits = model(x)
                dp = torch.softmax(d_logits[pcg_idx], -1).cpu().numpy()
            for i in range(len(batch_wf)):
                di = dp[i].argmax()
                lo = bins[di]
                label = f'PCG_{lo}-{lo+100}km'
                wf_u16, ts = batch_data[i]
                if label not in cache:
                    cache[label] = []
                cache[label].append((wf_u16, ts))
            processed += len(batch_wf)

        # Flush periodically
        for label in list(cache.keys()):
            if len(cache[label]) >= MAX_PER_FILE:
                write_batch(cache[label], label, args.output_dir, lh, ph)
                cache[label] = []
                out_files += 1

        if processed % 5000 == 0:
            print(f"  Processed {processed} pieces, {out_files} output files...")

    # Final flush
    for label, pieces in cache.items():
        if pieces:
            write_batch(pieces, label, args.output_dir, lh, ph)
            out_files += 1

    print(f"Done. {out_files} output files → {args.output_dir}")
    for d in sorted(os.listdir(args.output_dir)):
        dd = os.path.join(args.output_dir, d)
        if os.path.isdir(dd):
            print(f"  {d}: {len(os.listdir(dd))} files")


def write_batch(pieces, label, output_dir, lh, ph):
    d = os.path.join(output_dir, label)
    os.makedirs(d, exist_ok=True)
    idx = len(os.listdir(d))
    fname = os.path.join(d, f'PCG_{idx + 1:04d}.lig')
    gh = bytearray(lh)
    struct.pack_into('<i', gh, 4, len(pieces))
    with open(fname, 'wb') as f:
        f.write(gh)
        for wf, ts in pieces:
            p = np.asarray(wf[:16000], dtype=np.uint16)
            if len(p) < 16000:
                p = np.pad(p, (0, 16000 - len(p)), 'constant')
            out = ph + struct.pack('16000H', *p.tolist())
            Y = int(ts[:2]); M = int(ts[2:4]); D = int(ts[4:6])
            h = int(ts[6:8]); m = int(ts[8:10])
            si = int(float(ts[10:12])); S = float(ts[12:])
            out = out[:108] + struct.pack('6i4x', Y, M, D, h, m, si) + struct.pack('d', S) + out[144:]
            f.write(out)


if __name__ == "__main__":
    main()
