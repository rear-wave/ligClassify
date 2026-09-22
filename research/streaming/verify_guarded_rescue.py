"""Freeze and verify guarded rescue on all validation stresses and CPU latency."""
import hashlib
import json
import time

import numpy as np
import torch

from guarded_rescue import original_predictions, predict
from robust_calibration import MODELS, OUT, evidence, model_sha256, load_model_checkpoint, write_json, score
from experiment import CONDITIONS, perturb, preprocess_views, PreprocessConfig, shifted


def collect(checkpoint, raw, daylight, device, batch_size=128):
    local, global_view = preprocess_views(raw, PreprocessConfig(**checkpoint.preprocess_config))
    parts = [[], [], [], []]
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            x, g = [torch.from_numpy(v[start:start+batch_size, None]).to(device) for v in (local, global_view)]
            d = torch.tensor(daylight[start:start+batch_size, None], device=device, dtype=torch.float32)
            primary = checkpoint.model(x, g, d)
            alternate = checkpoint.model(shifted(x, 16), shifted(g, 4), d)
            result = (*evidence(primary, alternate), primary.known_logits.argmax(1).eq(
                alternate.known_logits.argmax(1)).cpu().numpy())
            for target, value in zip(parts, result):
                target.append(value)
    return dict(zip(("features", "candidate", "stable", "view_agreement"), (np.concatenate(p) for p in parts)))


def needs_verifier(first, config, limits):
    base = original_predictions(first, config)
    needed = np.zeros(len(base), dtype=bool)
    for label, thresholds in limits.items():
        limit = np.array(thresholds, dtype=np.float32)
        needed |= ((base == 0) & first["stable"] & (first["candidate"] == int(label))
                   & (first["features"][:, 0] <= limit[0]) & (first["features"][:, 2] <= limit[2]))
    return base, needed


def single_piece(checkpoints, raw, day, limits):
    """Return one type; preprocess once when checkpoint configurations match."""
    config = PreprocessConfig(**checkpoints[0].preprocess_config)
    local, global_view = preprocess_views(raw, config)
    x, g = [torch.from_numpy(v[:, None]) for v in (local, global_view)]
    d = torch.tensor(day[:, None], dtype=torch.float32)
    with torch.inference_mode():
        model = checkpoints[0].model
        f, c, s = evidence(model(x, g, d), model(shifted(x, 16), shifted(g, 4), d))
        first = {"features": f, "candidate": c, "stable": s}
        base, needed = needs_verifier(first, checkpoints[0].metadata["decision_config"], limits)
        if not needed[0]:
            return int(base[0]), False
        if checkpoints[0].preprocess_config != checkpoints[1].preprocess_config:
            local, global_view = preprocess_views(raw, PreprocessConfig(**checkpoints[1].preprocess_config))
            x, g = [torch.from_numpy(v[:, None]) for v in (local, global_view)]
        model = checkpoints[1].model
        f, c, s = evidence(model(x, g, d), model(shifted(x, 16), shifted(g, 4), d))
        second = {"features": f, "candidate": c, "stable": s}
        _, result = predict(first, second, checkpoints[0].metadata["decision_config"],
                            checkpoints[1].metadata["decision_config"], limits)
        return int(result[0]), True


def main():
    torch.set_num_threads(4)
    prior = json.loads((OUT / "guarded_rescue_report.json").read_text(encoding="utf-8"))
    frozen = {"limits": prior["limits"], "models": prior["models"],
              "schema": "guarded_rescue_frozen_v1", "frozen_before_extra_stress_and_latency": True}
    frozen["sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    write_json(OUT / "guarded_rescue_frozen.json", frozen)
    checkpoints = [load_model_checkpoint(path, "cuda") for path in MODELS.values()]
    for checkpoint, (name, path) in zip(checkpoints, MODELS.items()):
        assert model_sha256(path) == frozen["models"][name]
        checkpoint.model.eval()
    with np.load(OUT / "validation_raw.npz") as archive:
        data = dict(archive)
    configs = [cp.metadata["decision_config"] for cp in checkpoints]
    results, clean_prediction, needed_clean = {}, None, None
    for condition in CONDITIONS:
        raw = perturb(data["raw"], condition)
        first, second = [collect(cp, raw, data["daylight"], "cuda") for cp in checkpoints]
        base, prediction = predict(first, second, *configs, frozen["limits"])
        _, needed = needs_verifier(first, configs[0], frozen["limits"])
        metrics = score(data["labels"], prediction)
        metrics.update(secondary_call_fraction=float(needed.mean()),
                       recovered_correct=int(((base == 0) & (prediction == data["labels"]) & (data["labels"] > 0)).sum()),
                       introduced_wrong=int(((base == 0) & (prediction != 0) & (prediction != data["labels"])).sum()))
        results[condition] = metrics
        if condition == "clean":
            clean_prediction, needed_clean = prediction, needed
        print(condition, "recall", round(metrics["known_macro_recall"], 4),
              "NBE precision", [round(metrics["type_precision"][i], 4) for i in (2, 4)],
              "secondary fraction", round(float(needed.mean()), 4), flush=True)
    for cp in checkpoints:
        cp.model.to("cpu")
    indices = np.concatenate([np.flatnonzero(data["labels"] == label)[:24] for label in range(5)])
    indices = np.unique(np.r_[indices, np.flatnonzero(needed_clean)[:64]])
    for index in indices[:10]:
        single_piece(checkpoints, data["raw"][index:index+1], data["daylight"][index:index+1], frozen["limits"])
    timing, called, mismatches = [], [], 0
    for index in indices:
        start = time.perf_counter()
        prediction, second_used = single_piece(checkpoints, data["raw"][index:index+1],
                                                data["daylight"][index:index+1], frozen["limits"])
        timing.append((time.perf_counter() - start) * 1000)
        called.append(second_used)
        mismatches += int(prediction != clean_prediction[index])
    timing, called = np.array(timing), np.array(called)
    latency = {"scope": "CPU completed waveform to type, includes preprocessing and conditional verification; excludes distance, I/O, queueing; concurrent training on this host",
               "sampling": "24 per true class plus up to 64 verification-triggering pieces; not natural arrival proportions",
               "single_vs_batch_mismatches": mismatches, "sample_count": len(indices), "cpu_threads": 4}
    for name, mask in (("all", np.ones(len(timing), dtype=bool)), ("primary_only", ~called), ("with_verifier", called)):
        if mask.any():
            latency[name] = {"count": int(mask.sum()), "median_ms": float(np.median(timing[mask])),
                             "p95_ms": float(np.quantile(timing[mask], .95)), "p99_ms": float(np.quantile(timing[mask], .99))}
    report = {"frozen_recipe": frozen, "partition": "validation", "conditions": results,
               "latency": latency, "test_used": False, "deployment_ready": False}
    write_json(OUT / "guarded_rescue_validation_report.json", report)
    print(json.dumps(latency), flush=True)


if __name__ == "__main__":
    main()
