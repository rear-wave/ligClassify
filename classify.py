"""
ligClassify — piece-level classification (CSV output)
"""

import os, csv, argparse, logging
import numpy as np, torch

from models import create_model
from data.lig_parser import LigFileIndex
from data.preprocessing import preprocess_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", default="./classified")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_pieces", type=int, default=None)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.enabled = False

    ckpt = torch.load(args.model, map_location=dev, weights_only=False)
    classes = ckpt["class_names"]; nc = len(classes)
    model = create_model(nc).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(os.path.join(r, f)
                   for r, _, fs in os.walk(args.input_dir)
                   for f in fs if f.endswith('.lig'))
    lig = LigFileIndex(files, validate=False)
    total = lig.total_pieces
    if args.max_pieces and args.max_pieces < total:
        total = args.max_pieces
    logger.info(f"{len(files)} files, {total} pieces")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece_index", "class", "max_prob"] + [f"prob_{c}" for c in classes])

        counts = {c: 0 for c in classes}
        for start in range(0, total, 5000):
            end = min(start + 5000, total)
            idx = list(range(start, end))
            wf = lig.read_pieces_batch(idx)
            wf = np.stack(wf, axis=0)
            wf = preprocess_batch(wf, normalize_mode='minmax')

            for s in range(0, len(wf), args.batch_size):
                e = min(s + args.batch_size, len(wf))
                x = torch.from_numpy(wf[s:e]).unsqueeze(1).to(dev)
                with torch.no_grad():
                    probs = torch.softmax(model(x), -1).cpu().numpy()

                for i, gi in enumerate(range(start + s, start + e)):
                    cls = classes[probs[i].argmax()]
                    counts[cls] += 1
                    w.writerow([gi, cls, f"{probs[i].max():.4f}"] +
                                [f"{probs[i,j]:.4f}" for j in range(nc)])
            del wf

    lig.close()
    print(f"\n{'='*50}")
    for c in classes:
        n = counts.get(c, 0)
        print(f"  {c:<8}: {n:>7d}  ({n/max(sum(counts.values()),1)*100:5.1f}%)")
    print(f"{'='*50}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
