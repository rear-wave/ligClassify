"""
ligClassify training
====================
"""

import os, json, time, argparse, logging, glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from data.lig_parser import LigFileIndex
from data.preprocessing import preprocess_batch
from models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Dataset ──────────────────────────────────────────────
class WaveformDataset(Dataset):
    def __init__(self, data_dir, split="train", val=0.15, test=0.15, seed=42, max_pieces=None):
        classes = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
        self.class_names = classes
        rng = np.random.RandomState(seed)

        # Collect files per class
        class_files = {}
        for c in classes:
            d = os.path.join(data_dir, c)
            fs = sorted(glob.glob(os.path.join(d, "**", "*.lig"), recursive=True))
            if fs:
                class_files[c] = fs

        # File-level split
        selected, file_labels = [], []
        for label, c in enumerate(classes):
            fs = class_files.get(c, [])
            n = len(fs)
            vn, tn = max(1, int(n * val)), max(1, int(n * test))
            train_n = n - vn - tn
            idx = list(range(n)); rng.shuffle(idx)
            pick = {"train": set(idx[:train_n]),
                    "val": set(idx[train_n:train_n+vn]),
                    "test": set(idx[train_n+vn:])}[split]
            logger.info(f"  {c}: {len(pick)}/{n} files")
            for i in sorted(pick):
                selected.append(fs[i]); file_labels.append(label)

        lig = LigFileIndex(selected, validate=False)
        self.labels = []
        cum = lig._cumsum
        for i in range(len(selected)):
            self.labels += [file_labels[i]] * int(cum[i+1] - cum[i])

        total = lig.total_pieces
        self.indices = rng.choice(total, max_pieces, replace=False).tolist() if max_pieces and max_pieces < total else list(range(total))
        logger.info(f"  {split}: {len(self.indices)} pieces, {len(selected)} files")

        # Chunked preload
        cs = 5000
        data = []
        for start in range(0, len(self.indices), cs):
            end = min(start + cs, len(self.indices))
            chunk = self.indices[start:end]
            wf = lig.read_pieces_batch(chunk)
            wf = np.stack(wf, axis=0)
            wf = preprocess_batch(wf, normalize_mode='minmax')
            data.append(torch.from_numpy(wf.copy()).unsqueeze(1))
            del wf
        lig.close()

        self.data = torch.cat(data, dim=0)
        self.labels_tensor = torch.tensor([self.labels[i] for i in self.indices], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels_tensor[i]


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ── Training loop ────────────────────────────────────────
def train_epoch(model, loader, opt, crit, dev):
    model.train()
    loss_sum, ok, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(y)
        ok += (out.detach().argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, ok / n


@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum, ok, n = 0.0, 0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        p = out.argmax(1)
        ok += (p == y).sum().item()
        n += len(y)
        preds.append(p.cpu()); labels.append(y.cpu())
    ap = torch.cat(preds).numpy(); al = torch.cat(labels).numpy()
    return {"loss": loss_sum / n, "accuracy": accuracy_score(al, ap),
            "f1": f1_score(al, ap, average="weighted"),
            "f1_per_class": f1_score(al, ap, average=None).tolist(),
            "confusion": confusion_matrix(al, ap).tolist()}


# ── Main ─────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_data", default="./train_data")
    p.add_argument("--output", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--wd", type=float, default=0.0005)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_pieces", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.enabled = False

    train_set = WaveformDataset(args.task_data, "train", seed=args.seed, max_pieces=args.max_pieces)
    val_set = WaveformDataset(args.task_data, "val", seed=args.seed, max_pieces=args.max_pieces)
    test_set = WaveformDataset(args.task_data, "test", seed=args.seed, max_pieces=args.max_pieces)
    nc = len(train_set.class_names)
    logger.info(f"Classes: {train_set.class_names}")

    train_ld = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_ld = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_ld = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = create_model(nc, args.base).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state, wait = 0, None, 0
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Training {args.epochs} epochs, batch={args.batch_size}")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_ld, opt, crit, dev)
        val = evaluate(model, val_ld, crit, dev)
        sched.step()

        logger.info(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val_loss={val['loss']:.4f} acc={val['accuracy']:.4f} f1={val['f1']:.4f}")

        if val["accuracy"] > best_acc + 1e-6:
            best_acc = val["accuracy"]
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.output, f"resnet_epoch{epoch+1}.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "model_name": "resnet",
                        "class_names": train_set.class_names}, path)
            logger.info(f"  Saved periodic checkpoint: {path}")

    model.load_state_dict(best_state)
    test = evaluate(model, test_ld, crit, dev)
    logger.info(f"Test acc={test['accuracy']:.4f} f1={test['f1']:.4f}")

    ckpt = {"model_state_dict": model.state_dict(), "model_name": "resnet",
            "class_names": train_set.class_names, "test_metrics": test}
    path = os.path.join(args.output, "resnet.pt")
    torch.save(ckpt, path)

    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump({"model": "resnet", "accuracy": test["accuracy"], "f1": test["f1"],
                   "f1_per_class": test["f1_per_class"], "confusion": test["confusion"],
                   "class_names": train_set.class_names}, f, indent=2)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    main()
