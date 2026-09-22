"""Controlled inference-aligned training ablation, validation selection only."""
import json
import subprocess
import sys
import time

import torch

from experiment import ROOT, OUT, evaluate, compare_candidates, write_json


def main():
    torch.set_num_threads(4)
    jobs = (("hierarchical32_aligned", "hierarchical"),
            ("multiscale32_aligned", "multiscale"))
    write_json(OUT / "round2_plan.json", {
        "hypothesis": "Raw augmentation is followed by recentering; training should also cover the zero-padded 16/4 shifts used by deployment decisions.",
        "control": "First-round matching architecture, width, seed, split, losses, augmentation, epochs and patience; only alternate post-preprocess view shift changes.",
        "jobs": jobs, "test_used_for_selection": False,
        "default_inference_unchanged": True,
    })
    for name, architecture in jobs:
        target = OUT / name
        if not (target / "type/metrics.json").exists():
            args = [sys.executable, "-u", "train.py", "--task_data", str(ROOT.parent / "train_data"),
                    "--output", str(target), "--stage", "type", "--type_architecture", architecture,
                    "--type_loss_profile", "known_consistency_v1", "--augmentation_profile", "standard",
                    "--base_channels", "32", "--epochs", "20", "--patience", "6", "--batch_size", "60",
                    "--defer_test", "--inference_view_shift"]
            if (target / "type/last.pt").exists():
                args.extend(["--resume", str(target / "type/last.pt")])
            write_json(OUT / "round2_current_job.json", {
                "candidate": name, "command": args, "started": time.time()})
            print(f"Training: {name}", flush=True)
            with (OUT / f"{name}_train.log").open("a", encoding="utf-8") as log:
                subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (OUT / f"{name}_validation.json").exists():
            evaluate(name, target / "type/model.pt")
        compare_candidates()
    write_json(OUT / "round2_complete.json", {"completed": time.time()})


if __name__ == "__main__":
    main()
