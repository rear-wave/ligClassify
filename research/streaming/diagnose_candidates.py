"""Validation-only evidence audit; never relax deployed decision limits."""
from dataclasses import fields
import json

import numpy as np
import torch

from experiment import (OUT, ROOT, load_model_checkpoint, PreprocessConfig,
                        preprocess_views, perturb, shifted, score, write_json,
                        HierarchicalDecisionConfig, decide_hierarchical_types)
from audit_data import summarize_type_constraints
from models import HierarchicalTypeOutput


def combined(batches):
    return HierarchicalTypeOutput(**{
        field.name: (torch.cat([getattr(batch, field.name) for batch in batches])
                     if getattr(batches[0], field.name) is not None
                     and getattr(batches[0], field.name).ndim > 0 else None)
        for field in fields(HierarchicalTypeOutput)
    })


def run():
    torch.set_num_threads(4)
    data = np.load(OUT / "validation_raw.npz")
    targets = torch.from_numpy(data["labels"].astype(np.int64))
    records = {}
    paths = {
        "old_baseline": ROOT / "weights/multi_model/type/model.pt",
        "anchor_baseline": ROOT / "weights/anchor_moe_best_bundle/type/model.pt",
        **{name: OUT / name / "type/model.pt"
           for name in ("hierarchical32", "multiscale32", "multiscale32_drift")},
    }
    for name, path in paths.items():
        checkpoint = load_model_checkpoint(path, "cuda")
        checkpoint.model.eval()
        config = PreprocessConfig(**checkpoint.preprocess_config)
        decision_config = HierarchicalDecisionConfig(**checkpoint.metadata["decision_config"])
        records[name] = {}
        for condition in ("clean", "baseline_drift", "lowpass_80khz"):
            cache = OUT / f"evidence_{name}_{condition}.pt"
            if cache.exists():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                primary, alternate = [HierarchicalTypeOutput(**saved[k]) for k in ("primary", "alternate")]
            else:
                local, global_view = preprocess_views(perturb(data["raw"], condition), config)
                batches = [[], []]
                with torch.inference_mode():
                    for start in range(0, len(local), 128):
                        x = torch.from_numpy(local[start:start+128, None]).to("cuda")
                        g = torch.from_numpy(global_view[start:start+128, None]).to("cuda")
                        d = torch.tensor(data["daylight"][start:start+128, None], device="cuda", dtype=torch.float32)
                        outputs = (checkpoint.model(x, g, d), checkpoint.model(shifted(x, 16), shifted(g, 4), d))
                        for output, destination in zip(outputs, batches):
                            destination.append(HierarchicalTypeOutput(**{
                                f.name: getattr(output, f.name).detach().cpu()
                                if getattr(output, f.name) is not None else None
                                for f in fields(HierarchicalTypeOutput)}))
                primary, alternate = map(combined, batches)
                torch.save({k: {f.name: getattr(v, f.name) for f in fields(HierarchicalTypeOutput)}
                            for k, v in (("primary", primary), ("alternate", alternate))}, cache)
            decision = decide_hierarchical_types(primary, alternate, decision_config)
            summary = summarize_type_constraints(decision, targets)
            joint = (primary.type_logits + alternate.type_logits).argmax(1)
            record = {"joint_argmax": score(targets.numpy(), joint.numpy()),
                      "calibrated": score(targets.numpy(), decision.final_type.numpy()),
                      "constraints": summary, "drop_one_constraint": {}}
            for omitted in decision.constraint_passes:
                accepted = torch.stack([v for k, v in decision.constraint_passes.items() if k != omitted]).all(0)
                result = torch.where(accepted, decision.candidate_known_type, 0)
                record["drop_one_constraint"][omitted] = score(targets.numpy(), result.numpy())
            records[name][condition] = record
            known_correct = sum(v["candidate_correct_count"] or 0 for v in summary["by_true_type"].values()) / 1600
            print(name, condition, json.dumps({
                "known_candidate_accuracy": round(known_correct, 4),
                "joint_known_recall": round(record["joint_argmax"]["known_macro_recall"], 4),
                "final_known_recall": round(record["calibrated"]["known_macro_recall"], 4),
                "correct_candidates_rejected_by_constraint": {
                    constraint: sum(summary["by_true_type"][label]["scopes"]["correct_candidate"]
                        ["constraints"][constraint]["failed_count"] for label in ("NCG", "NNBE", "PCG", "PNBE"))
                    for constraint in decision.constraint_passes},
            }), flush=True)
        del checkpoint
        torch.cuda.empty_cache()
    write_json(OUT / "constraint_diagnosis.json", records)


if __name__ == "__main__":
    run()
