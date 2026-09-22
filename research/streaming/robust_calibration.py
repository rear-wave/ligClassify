"""Fit one fixed rule on separate validation pieces under multiple conditions.

All evaluation piece keys are excluded from calibration. This is development,
not an independent test or a claim about unknown deployment class proportions.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiment import (ROOT, OUT, make_cache, perturb, preprocess_views, shifted,
                        score, write_json, load_model_checkpoint, model_sha256,
                        PreprocessConfig, HierarchicalDecisionConfig,
                        decide_hierarchical_types)
from data.dataset import FiveClassDataset
from data.manifest import build_piece_table
from data.split import assign_piece_splits
from evaluation import _js_divergence
from models import HierarchicalTypeOutput

CONDITIONS = ("clean", "baseline_drift", "lowpass_80khz")
MODELS = {
    "old_baseline": ROOT / "weights/multi_model/type/model.pt",
    "hierarchical32": OUT / "hierarchical32/type/model.pt",
}
LIMITS = {
    "clean_precision": [.97, .995, .97, .995],
    "stress_precision": [.95, .98, .95, .98],
    "ic_false_accept_per_class": [.01, .005, .01, .005],
    "cg_to_nbe_per_class": .002,
}


def evidence(primary, alternate):
    broad = HierarchicalDecisionConfig((0.,) * 4, (-1.,) * 4, (1.,) * 4, 2, (1.,) * 4)
    decision = decide_hierarchical_types(primary, alternate, broad)
    features = torch.stack((-decision.known_type_probability,
                            -decision.prototype_similarity,
                            decision.ic_gate_probability,
                            _js_divergence(primary.known_logits, alternate.known_logits)), 1)
    stable = decision.constraint_passes["view_agreement"] & decision.constraint_passes["branch_votes"]
    return features.cpu().numpy(), decision.candidate_known_type.cpu().numpy(), stable.cpu().numpy()


def accepted_grid(values, targets, groups, axes, group_count):
    """Exact accepted counts at every four-dimensional upper-limit grid point."""
    shape = (group_count, 5, *(len(axis) for axis in axes))
    coordinates = [groups, targets] + [np.searchsorted(axis, values[:, i], side="left")
                                     for i, axis in enumerate(axes)]
    if any(np.any(v >= shape[i]) for i, v in enumerate(coordinates)):
        raise ValueError("grid does not cover every evidence value")
    flat = np.ravel_multi_index(coordinates, shape)
    histogram = np.bincount(flat, minlength=np.prod(shape)).reshape(shape)
    for axis in range(2, 6):
        histogram = histogram.cumsum(axis=axis)
    return histogram


def self_test():
    rng = np.random.default_rng(711)
    values = rng.normal(size=(120, 4))
    labels = rng.integers(0, 5, 120)
    groups = rng.integers(0, 3, 120)
    axes = [np.quantile(values[:, i], [0, .3, .7, 1]) for i in range(4)]
    counts = accepted_grid(values, labels, groups, axes, 3)
    for cell in np.ndindex(*(len(a) for a in axes)):
        accepted = (values <= np.array([a[k] for a, k in zip(axes, cell)])).all(1)
        for group in range(3):
            for label in range(5):
                assert counts[(group, label, *cell)] == np.sum(accepted & (labels == label) & (groups == group))


def cache_calibration(name, path):
    cache_path = OUT / f"robust_calibration_{name}_evidence_v2.npz"
    if cache_path.exists():
        return dict(np.load(cache_path))
    table, _ = build_piece_table(ROOT.parent / "train_data", require_distance=True)
    assignment = assign_piece_splits(table, seed=42)
    evaluation_keys = set(np.load(OUT / "validation_raw.npz")["keys"].tolist())
    positions = np.array([p for p in assignment.positions("validation")
                          if table.piece_key(int(p)) not in evaluation_keys])
    keys = np.array([table.piece_key(int(p)) for p in positions])
    assert not evaluation_keys.intersection(keys.tolist())
    checkpoint = load_model_checkpoint(path, "cuda")
    checkpoint.model.eval()
    config = PreprocessConfig(**checkpoint.preprocess_config)
    dataset = FiveClassDataset(table, positions, "validation")
    collected = {key: [] for key in ("features", "candidate", "stable", "targets", "groups")}
    try:
        with torch.inference_mode():
            for start in range(0, len(positions), 128):
                selected = positions[start:start + 128]
                raw = np.stack(dataset.lig.read_pieces_batch([
                    dataset._global_piece_index(int(p)) for p in selected])).astype(np.float32)
                for condition_index, condition in enumerate(CONDITIONS):
                    local, global_view = preprocess_views(perturb(raw, condition), config)
                    x, g = [torch.from_numpy(v[:, None]).to("cuda") for v in (local, global_view)]
                    d = torch.tensor(table.daylight[selected, None], device="cuda", dtype=torch.float32)
                    primary = checkpoint.model(x, g, d)
                    alternate = checkpoint.model(shifted(x, 16), shifted(g, 4), d)
                    features, candidate, stable = evidence(primary, alternate)
                    for key, value in (("features", features), ("candidate", candidate), ("stable", stable),
                                       ("targets", table.type_index[selected]),
                                       ("groups", np.full(len(selected), condition_index))):
                        collected[key].append(value)
                if start % 2048 == 0:
                    print(f"{name}: separate calibration {start}/{len(positions)}", flush=True)
    finally:
        dataset.close()
    result = {key: np.concatenate(value) for key, value in collected.items()}
    result.update(keys=keys, model_sha256=np.array(model_sha256(path)))
    np.savez_compressed(cache_path, **result)
    return result


def fit(data):
    features, labels, groups = data["features"], data["targets"], data["groups"]
    support = np.array([np.bincount(labels[groups == group], minlength=5) for group in range(3)])
    if np.any(support == 0):
        raise ValueError("every calibration condition requires all five classes")
    values = [[], [], [], []]
    details = {}
    for label in range(1, 5):
        mask = data["stable"] & (data["candidate"] == label)
        candidate_values = features[mask]
        levels = ([0., .01, .025, .05, .1, .2, .4, .6, .8, .9, .95, .98, 1.],
                  [0., .01, .025, .05, .1, .2, .4, .6, .8, .9, .95, .98, 1.],
                  [0., .1, .2, .4, .6, .8, .9, .95, .97, .98, .99, .995, 1.],
                  [0., .5, .8, .9, .95, .97, .99, 1.])
        if not len(candidate_values):
            for out, value in zip(values, (-1., -1., 0., 0.)):
                out.append(value)
            details[str(label)] = {"feasible": False}
            continue
        axes = [np.unique(np.quantile(candidate_values[:, i], q).astype(np.float32)) for i, q in enumerate(levels)]
        axes[3] = np.unique(np.maximum(axes[3], np.float32(1e-8)))
        counts = accepted_grid(candidate_values, labels[mask], groups[mask], axes, 3)
        shape = (3, 5) + (1,) * 4
        rates = counts / support.reshape(shape)
        recall = rates[:, label]
        # Uniform class weighting is for development comparison, not an assumed deployment prior.
        precision = rates[:, label] / rates.sum(1).clip(1e-12)
        ic_rate = rates[:, 0]
        cg_rate = (counts[:, 1] + counts[:, 3]) / support[:, [1, 3]].sum(1).reshape((3,) + (1,) * 4)
        required = np.array([LIMITS["clean_precision"][label - 1],
                             LIMITS["stress_precision"][label - 1],
                             LIMITS["stress_precision"][label - 1]]).reshape((3,) + (1,) * 4)
        feasible = ((precision >= required) & (ic_rate <= LIMITS["ic_false_accept_per_class"][label - 1])).all(0)
        if label in (2, 4):
            feasible &= (cg_rate <= LIMITS["cg_to_nbe_per_class"]).all(0)
        options = np.flatnonzero(feasible)
        if not len(options):
            chosen = (-1., -1., 0., 0.)
            details[str(label)] = {"feasible": False}
        else:
            objectives = [recall.min(0), recall[0], recall.mean(0), precision.min(0)]
            ordering = np.lexsort(tuple(value.ravel()[options] for value in objectives[::-1]))
            cell = np.unravel_index(options[ordering[-1]], feasible.shape)
            chosen = [float(axis[index]) for axis, index in zip(axes, cell)]
            details[str(label)] = {"feasible": True,
                "per_condition_recall": recall[(slice(None), *cell)].tolist(),
                "per_condition_balanced_precision": precision[(slice(None), *cell)].tolist(),
                "per_condition_ic_false_accept": ic_rate[(slice(None), *cell)].tolist()}
        for out, value in zip(values, chosen):
            out.append(value)
    config = HierarchicalDecisionConfig(tuple(-v for v in values[0]), tuple(-v for v in values[1]),
                                         tuple(max(v, 1e-8) for v in values[3]), 2, tuple(values[2]))
    return config, details


def verify(name, config):
    targets = np.load(OUT / "validation_raw.npz")["labels"]
    result = {}
    for condition in CONDITIONS:
        payload = torch.load(OUT / f"evidence_{name}_{condition}.pt", map_location="cpu", weights_only=True)
        primary, alternate = [HierarchicalTypeOutput(**payload[key]) for key in ("primary", "alternate")]
        decision = decide_hierarchical_types(primary, alternate, config)
        result[condition] = score(targets, decision.final_type.numpy())
        print(name, "robust-calibration", condition,
              "recall", round(result[condition]["known_macro_recall"], 4),
              "NBE precision", [round(result[condition]["type_precision"][i], 4) for i in (2, 4)], flush=True)
    return result


def main():
    torch.set_num_threads(2)
    self_test()
    make_cache()
    write_json(OUT / "robust_calibration_plan.json", {
        "limits": LIMITS, "calibration_conditions": CONDITIONS,
        "calibration_partition": "seed42 validation, excluding every 2000-piece evaluation key",
        "precision_weighting": "uniform five-class; not a deployment class-prior claim",
        "test_used": False, "deployed_weights_modified": False,
        "selection": "per class maximize worst-condition recall subject to EVERY condition's risk constraints",
    })
    report = {}
    for name, path in MODELS.items():
        data = cache_calibration(name, path)
        assert str(data["model_sha256"]) == model_sha256(path)
        config, details = fit(data)
        write_json(OUT / f"robust_calibration_{name}_config.json", asdict(config))
        report[name] = {"calibration_piece_count": len(data["keys"]), "calibration_details": details,
                        "decision_config": asdict(config), "evaluation": verify(name, config)}
        write_json(OUT / "robust_calibration_report.json", report)


if __name__ == "__main__":
    main()
