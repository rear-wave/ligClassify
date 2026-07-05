"""
ligClassify — Multi-task training: type (5-class) + per-class distance (30-bin).
IC samples have dist_label = -1 and are excluded from distance loss.
"""

import os, re, glob, argparse, logging, json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from models import create_mtl_model
from data.lig_parser import LigFileIndex
from data.preprocessing import preprocess_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
TYPE_NAMES = ["IC", "NCG", "NNBE", "PCG", "PNBE"]  # 0..4
DIST_NAMES = ["NCG", "NNBE", "PCG", "PNBE"]         # maps to d_heads[0..3]
# IC=0 is not in DIST_NAMES → dist_head index = type_idx - 1
DIST_BIN_STARTS = [i * 100 for i in range(30)]       # 0, 100, ..., 2900


# ── Soft distance label ──────────────────────────────────
def make_soft_label(k, num_classes=30, tau=1.0):
    """Soft label centered at bin k: w[j] ∝ exp(-|j-k| / tau)."""
    dists = np.abs(np.arange(num_classes) - k).astype(np.float32)
    w = np.exp(-dists / tau)
    return torch.from_numpy(w / w.sum())


# ── Dataset ──────────────────────────────────────────────
class MultiTaskDataset(Dataset):
    """Combined type + distance labels from train_data/ folder structure.
    IC samples: type=0, dist_label=-1 (no distance head).
    Non-IC samples: type from parent folder, distance label from folder name.
    """

    def __init__(self, data_dir, split="train", val=0.15, test=0.15, seed=42):
        # ----- Collect type-classified files (all of them) -----
        type_files = {}  # class_name → [paths]
        for cls in TYPE_NAMES:
            d = os.path.join(data_dir, cls)
            if os.path.isdir(d):
                fs = sorted(glob.glob(os.path.join(d, "**", "*.lig"), recursive=True))
                if fs:
                    type_files[cls] = fs

        # ----- Collect distance-labeled files (day + night subdirs) -----
        dist_re = re.compile(r'(\d+)-(\d+)km')
        dist_files = defaultdict(list)  # (type_name, dist_bin) → [paths]
        for cls in DIST_NAMES:
            for tod in ["day", "night"]:
                dn_dir = os.path.join(data_dir, cls, tod)
                if not os.path.isdir(dn_dir):
                    continue
                for root, _, files in os.walk(dn_dir):
                    ligs = [f for f in files if f.endswith('.lig')]
                    if not ligs:
                        continue
                    m = dist_re.search(os.path.basename(root))
                    if not m:
                        continue
                    lo = int(m.group(1))
                    bin_idx = lo // 100
                    for fn in ligs:
                        dist_files[(cls, bin_idx)].append(os.path.join(root, fn))

        # ----- Merge: distance-labeled files override type-only files -----
        all_files = {}  # filepath → (type_idx, dist_bin)
        # Add distance-labeled files first (they have the correct dist label)
        for (cls, bin_idx), fpaths in dist_files.items():
            type_idx = TYPE_NAMES.index(cls)
            for fp in fpaths:
                all_files[fp] = (type_idx, bin_idx)
        # Add remaining type-only files (IC + non-IC without distance labels)
        for cls in TYPE_NAMES:
            type_idx = TYPE_NAMES.index(cls)
            for fp in type_files.get(cls, []):
                if fp not in all_files:
                    all_files[fp] = (type_idx, -1)
        # Convert to list
        all_files = [(fp, t, d) for fp, (t, d) in all_files.items()]

        # Shuffle and split
        rng = np.random.RandomState(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n = len(indices)
        vn, tn = max(1, int(n * val)), max(1, int(n * test))
        tr_n = n - vn - tn
        split_ranges = {"train": (0, tr_n), "val": (tr_n, tr_n + vn), "test": (tr_n + vn, n)}
        lo, hi = split_ranges[split]
        split_files = [all_files[i] for i in indices[lo:hi]]
        logger.info(f"  {split}: {len(split_files)} files")

        # Count per-type stats
        type_counts = defaultdict(int)
        dist_counts = defaultdict(int)
        for _, t, d in split_files:
            type_counts[t] += 1
            if d >= 0:
                dist_counts[t] += 1
        for t, c in sorted(type_counts.items()):
            extras = f", with dist: {dist_counts.get(t,0)}" if t > 0 else ""
            logger.info(f"    {TYPE_NAMES[t]}: {c} files{extras}")

        # ----- Load pieces via LigFileIndex -----
        self.data, self.type_labels, self.dist_labels = [], [], []
        filepaths = [f[0] for f in split_files]
        lig = LigFileIndex(filepaths, validate=False)

        cs = 2000
        total_pieces = lig.total_pieces
        for start in range(0, total_pieces, cs):
            end = min(start + cs, total_pieces)
            chunk = list(range(start, end))
            wf = lig.read_pieces_batch(chunk)
            wf = np.stack(wf, axis=0)
            wf = preprocess_batch(wf, normalize_mode='minmax')
            self.data.append(torch.from_numpy(wf.copy()).unsqueeze(1))

            # Map piece indices → file labels
            pieces_per_file = [int(lig._cumsum[i + 1] - lig._cumsum[i])
                               for i in range(len(filepaths))]
            file_of_piece = np.repeat(np.arange(len(filepaths)), pieces_per_file)
            chunk_file_idx = file_of_piece[start:end]
            self.type_labels.append(torch.tensor([split_files[i][1] for i in chunk_file_idx],
                                                dtype=torch.long))
            self.dist_labels.append(torch.tensor([split_files[i][2] for i in chunk_file_idx],
                                                dtype=torch.long))
            del wf
        lig.close()

        self.data = torch.cat(self.data, dim=0)
        self.type_labels = torch.cat(self.type_labels, dim=0)
        self.dist_labels = torch.cat(self.dist_labels, dim=0)
        n_ic = (self.type_labels == 0).sum().item()
        n_dist = (self.dist_labels >= 0).sum().item()
        logger.info(f"  Loaded: {len(self.data)} pieces ({n_ic} IC, {n_dist} with dist)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.type_labels[i], self.dist_labels[i]


# ── Collate ──────────────────────────────────────────────
def collate(batch):
    xs, ys, ds = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ds)


# ── Loss helpers ─────────────────────────────────────────
def distance_loss_fn(logits, targets, ce_crit, soft_tau=None):
    """Distance loss. If soft_tau is set, uses soft-label CE."""
    if soft_tau is not None and soft_tau > 0:
        B, C = logits.shape
        soft_targets = torch.zeros(B, C, device=logits.device)
        for i in range(B):
            soft_targets[i] = make_soft_label(targets[i].item(), C, soft_tau).to(logits.device)
        return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    return ce_crit(logits, targets)


# ── Training / Eval ──────────────────────────────────────
def train_step(model, x, y, d, opt, type_crit, dist_crit,
               lambda_dist, use_soft, soft_tau):
    """Train one batch. Returns (type_loss, dist_loss, type_correct, total)."""
    model.train()
    opt.zero_grad()
    type_logits, dist_logits = model(x)

    type_loss = type_crit(type_logits, y)
    type_ok = (type_logits.argmax(1) == y).sum().item()

    # Distance loss: only non-IC (y > 0), use GT type to pick head
    mask = d >= 0
    dist_loss = torch.tensor(0.0, device=x.device)
    if mask.any():
        dist_losses = []
        for i in range(len(y)):
            if mask[i]:
                t = y[i].item()  # ground-truth type (1..4)
                head_idx = t - 1  # 0..3
                logit = dist_logits[head_idx][i:i + 1]  # [1, 30]
                target = d[i:i + 1]
                if use_soft:
                    dl = distance_loss_fn(logit, target, dist_crit, soft_tau)
                else:
                    dl = dist_crit(logit, target)
                dist_losses.append(dl)
        dist_loss = torch.stack(dist_losses).mean() if dist_losses else 0.0

    total_loss = type_loss + lambda_dist * dist_loss
    total_loss.backward()
    opt.step()
    return type_loss.item(), dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss, type_ok, len(y)


@torch.no_grad()
def evaluate(model, loader, type_crit, dist_crit, dev,
             use_soft=False, soft_tau=1.0):
    """Evaluate: type metrics + per-class distance metrics."""
    model.eval()
    all_type_preds, all_type_labels = [], []
    # Per-distance-head: collect (pred, true)
    dist_data = {head: {"preds": [], "trues": []} for head in range(4)}

    for x, y, d in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(dev), y.to(dev)
        type_logits, dist_logits = model(x)

        type_preds = type_logits.argmax(1)
        all_type_preds.extend(type_preds.cpu().numpy())
        all_type_labels.extend(y.cpu().numpy())

        # Distance: use GT type to select head (for training eval)
        mask = d >= 0
        if mask.any():
            for i in (mask.nonzero(as_tuple=True)[0]):
                t = y[i].item()
                head_idx = t - 1
                dist_data[head_idx]["preds"].append(
                    dist_logits[head_idx][i].argmax().item())
                dist_data[head_idx]["trues"].append(d[i].item())

    # ── Type metrics ──
    ap = np.array(all_type_preds)
    al = np.array(all_type_labels)
    type_acc = (ap == al).mean()

    from sklearn.metrics import f1_score
    type_f1 = f1_score(al, ap, average='macro', zero_division=0)

    # ── Distance metrics per head ──
    metrics = {"type_acc": type_acc, "type_f1": type_f1}
    dist_bin_starts = np.array(DIST_BIN_STARTS)
    total_dist_ok, total_dist_n, total_mae_bin, total_mae_km = 0, 0, 0, 0
    total_w1, total_w2, total_w1n, total_w2n = 0, 0, 0, 0

    for hi, name in enumerate(DIST_NAMES):
        preds = np.array(dist_data[hi]["preds"])
        trues = np.array(dist_data[hi]["trues"])
        n = len(preds)
        key = f"dist_{name}"
        if n == 0:
            metrics[f"{key}_acc"], metrics[f"{key}_mae_km"] = 0.0, 0.0
            metrics[f"{key}_w1"], metrics[f"{key}_w2"] = 0.0, 0.0
            continue

        ok = (preds == trues).sum()
        mae_bin = np.abs(preds - trues).mean()
        mae_km = mae_bin * 100
        w1 = (np.abs(preds - trues) <= 1).mean()
        w2 = (np.abs(preds - trues) <= 2).mean()

        metrics[f"{key}_acc"] = ok / n
        metrics[f"{key}_mae_bin"] = float(mae_bin)
        metrics[f"{key}_mae_km"] = float(mae_km)
        metrics[f"{key}_w1"] = float(w1)
        metrics[f"{key}_w2"] = float(w2)

        total_dist_ok += ok; total_dist_n += n
        total_mae_bin += mae_bin * n; total_mae_km += mae_km * n
        total_w1 += w1 * n; total_w2 += w2 * n; total_w1n += n; total_w2n += n

    if total_dist_n > 0:
        metrics["dist_acc"] = total_dist_ok / total_dist_n
        metrics["dist_mae_bin"] = total_mae_bin / total_dist_n
        metrics["dist_mae_km"] = total_mae_km / total_dist_n
        metrics["dist_w1"] = total_w1 / total_w1n if total_w1n else 0
        metrics["dist_w2"] = total_w2 / total_w2n if total_w2n else 0
    else:
        metrics["dist_acc"] = metrics["dist_mae_bin"] = metrics["dist_mae_km"] = 0.0
        metrics["dist_w1"] = metrics["dist_w2"] = 0.0

    return metrics


# ── Main ─────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_data", default="./train_data")
    p.add_argument("--output", default="./checkpoints_v2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--wd", type=float, default=0.0005)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_dist", type=float, default=0.5,
                   help="Weight of distance loss")
    p.add_argument("--use_soft_distance_label", action="store_true",
                   help="Use ordinal soft-label CE for distance")
    p.add_argument("--distance_soft_tau", type=float, default=1.0,
                   help="Tau for soft distance label")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.enabled = False

    logger.info(f"lambda_dist={args.lambda_dist}, soft_label={args.use_soft_distance_label}, "
                f"soft_tau={args.distance_soft_tau}")

    # Datasets
    train_set = MultiTaskDataset(args.task_data, "train", seed=args.seed)
    val_set = MultiTaskDataset(args.task_data, "val", seed=args.seed)
    test_set = MultiTaskDataset(args.task_data, "test", seed=args.seed)

    train_ld = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, pin_memory=True)
    val_ld = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, pin_memory=True)
    test_ld = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate, pin_memory=True)

    # Model
    model = create_mtl_model(base_channels=args.base).to(dev)

    type_crit = nn.CrossEntropyLoss()
    dist_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.output, exist_ok=True)
    best_acc, best_state, wait = 0.0, None, 0

    for epoch in range(args.epochs):
        # ── Train one epoch ──
        model.train()
        t_loss_sum, d_loss_sum, t_ok, t_n = 0.0, 0.0, 0, 0
        for x, y, dl in tqdm(train_ld, desc=f"Epoch {epoch + 1}", leave=False):
            x, y, dl = x.to(dev), y.to(dev), dl.to(dev)
            tl, dl_val, ok, bs = train_step(
                model, x, y, dl, opt, type_crit, dist_crit,
                args.lambda_dist, args.use_soft_distance_label, args.distance_soft_tau)
            t_loss_sum += tl * bs; d_loss_sum += dl_val * bs
            t_ok += ok; t_n += bs
        train_type_acc = t_ok / t_n
        train_type_loss = t_loss_sum / t_n
        train_dist_loss = d_loss_sum / t_n if t_n else 0

        # ── Validate ──
        val = evaluate(model, val_ld, type_crit, dist_crit, dev,
                       args.use_soft_distance_label, args.distance_soft_tau)
        sched.step()

        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"T_loss={train_type_loss:.4f} D_loss={train_dist_loss:.4f} T_acc={train_type_acc:.4f} | "
            f"VT_acc={val['type_acc']:.4f} VT_f1={val['type_f1']:.4f} | "
            f"VD_acc={val.get('dist_acc',0):.4f} VD_mae={val.get('dist_mae_km',0):.0f}km "
            f"VD_w1={val.get('dist_w1',0):.4f}")

        # Per-head details every 10 epochs
        if (epoch + 1) % 10 == 0:
            for name in DIST_NAMES:
                k = f"dist_{name}"
                logger.info(f"      {name}: acc={val.get(k+'_acc',0):.4f} "
                            f"mae={val.get(k+'_mae_km',0):.0f}km "
                            f"w1={val.get(k+'_w1',0):.4f} w2={val.get(k+'_w2',0):.4f}")

        # ── Checkpoint ──
        score = val['type_acc'] * 0.4 + val.get('dist_acc', 0) * 0.6
        if score > best_acc + 1e-6:
            best_acc = score
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.info(f"Early stop at epoch {epoch + 1}")
                break

    # ── Final test ──
    model.load_state_dict(best_state)
    test = evaluate(model, test_ld, type_crit, dist_crit, dev,
                    args.use_soft_distance_label, args.distance_soft_tau)

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results:")
    logger.info(f"  Type: acc={test['type_acc']:.4f} f1={test['type_f1']:.4f}")
    logger.info(f"  Distance (all): acc={test.get('dist_acc',0):.4f} "
                f"mae_bin={test.get('dist_mae_bin',0):.2f} "
                f"mae_km={test.get('dist_mae_km',0):.0f}km "
                f"w1={test.get('dist_w1',0):.4f} w2={test.get('dist_w2',0):.4f}")
    for name in DIST_NAMES:
        k = f"dist_{name}"
        if test.get(f"{k}_mae_km", -1) >= 0:
            logger.info(f"  {name}: acc={test.get(k+'_acc',0):.4f} "
                        f"mae_bin={test.get(k+'_mae_bin',0):.2f} "
                        f"mae_km={test.get(k+'_mae_km',0):.0f}km "
                        f"w1={test.get(k+'_w1',0):.4f} w2={test.get(k+'_w2',0):.4f}")

    # Save model
    torch.save({
        "model_state_dict": {k: v.clone().cpu() for k, v in model.state_dict().items()},
        "model_name": "mtl_resnet",
        "type_names": TYPE_NAMES,
        "dist_names": DIST_NAMES,
        "dist_bin_starts": DIST_BIN_STARTS,
        "lambda_dist": args.lambda_dist,
    }, os.path.join(args.output, "mtl.pt"))
    with open(os.path.join(args.output, "mtl.json"), "w") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    main()
