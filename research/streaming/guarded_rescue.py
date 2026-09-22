"""Validation experiment: second-model confirmation only for rejected pieces."""
from dataclasses import asdict
import json

import numpy as np
import torch

from robust_calibration import (OUT, MODELS, CONDITIONS, accepted_grid, evidence,
                                load_model_checkpoint, model_sha256, write_json,
                                HierarchicalTypeOutput, score)


def original_predictions(data, decision):
    candidate = data["candidate"].astype(np.int64)
    limits = np.stack((-np.array(decision["known_probability_thresholds"]),
                       -np.array(decision["prototype_similarity_thresholds"]),
                       np.array(decision["max_ic_gate_probabilities"]),
                       np.array(decision["max_js_divergences"])), 1).astype(np.float32)
    accepted = data["stable"] & (data["features"] <= limits[candidate - 1]).all(1)
    return np.where(accepted, candidate, 0)


def fit(primary, secondary, first_config, second_config):
    y, groups = primary["targets"], primary["groups"]
    assert np.array_equal(y, secondary["targets"]) and np.array_equal(groups, secondary["groups"])
    base = original_predictions(primary, first_config)
    second = original_predictions(secondary, second_config)
    feature = np.column_stack((primary["features"][:, 0], secondary["features"][:, 0],
                               primary["features"][:, 2], secondary["features"][:, 2]))
    opportunity = (base == 0) & (second > 0) & (primary["candidate"] == second) & primary["stable"]
    support = np.array([np.bincount(y[groups == g], minlength=5) for g in range(3)])
    details, limits = {}, {}
    for label in range(1, 5):
        rows = opportunity & (second == label)
        baseline = np.array([np.bincount(y[(groups == g) & (base == label)], minlength=5) for g in range(3)])
        if not rows.any():
            details[str(label)] = {"feasible": False}
            continue
        axes = [np.unique(np.quantile(feature[rows, i], [0, .05, .1, .2, .4, .6, .8, .9, .95, .99, 1]).astype(np.float32))
                for i in range(4)]
        extra = accepted_grid(feature[rows], y[rows], groups[rows], axes, 3)
        counts = extra + baseline.reshape((3, 5) + (1,) * 4)
        rates = counts / support.reshape((3, 5) + (1,) * 4)
        recall = rates[:, label]
        precision = rates[:, label] / rates.sum(1).clip(1e-12)
        requirements = np.array([.99, .97, .97] if label in (2, 4) else [.95, .93, .93])
        extra_ic_rate = extra[:, 0] / support[:, 0].reshape((3,) + (1,) * 4)
        extra_cg_rate = (extra[:, 1] + extra[:, 3]) / support[:, [1, 3]].sum(1).reshape((3,) + (1,) * 4)
        feasible = ((precision >= requirements.reshape((3,) + (1,) * 4)) & (extra_ic_rate <= .002)).all(0)
        if label in (2, 4):
            feasible &= (extra_cg_rate <= .001).all(0)
        indices = np.flatnonzero(feasible)
        if not len(indices):
            details[str(label)] = {"feasible": False}
            continue
        objectives = [recall.min(0), recall[0], recall.mean(0), precision.min(0)]
        ordering = np.lexsort(tuple(value.ravel()[indices] for value in objectives[::-1]))
        cell = np.unravel_index(indices[ordering[-1]], feasible.shape)
        limits[str(label)] = [float(axis[k]) for axis, k in zip(axes, cell)]
        details[str(label)] = {"feasible": True, "calibration_recall": recall[(slice(None), *cell)].tolist(),
            "calibration_balanced_precision": precision[(slice(None), *cell)].tolist(),
            "extra_ic_false_accept_rate": extra_ic_rate[(slice(None), *cell)].tolist()}
    return limits, details


def predict(primary, secondary, first_config, second_config, limits):
    base = original_predictions(primary, first_config)
    second = original_predictions(secondary, second_config)
    features = np.column_stack((primary["features"][:, 0], secondary["features"][:, 0],
                               primary["features"][:, 2], secondary["features"][:, 2]))
    result = base.copy()
    for label, limit in limits.items():
        label = int(label)
        rows = ((base == 0) & (second == label) & (primary["candidate"] == label)
                & primary["stable"] & (features <= np.array(limit, dtype=np.float32)).all(1))
        result[rows] = label
    assert np.array_equal(result[base != 0], base[base != 0])
    return base, result


def self_test():
    config = {"known_probability_thresholds": [.9] * 4,
              "prototype_similarity_thresholds": [-1.] * 4,
              "max_ic_gate_probabilities": [.5] * 4, "max_js_divergences": [1.] * 4}
    first = {"candidate": np.array([2, 2, 2, 2, 2, 4]),
             "stable": np.array([True, True, True, False, True, True]),
             "features": np.array([[-.95, 0, .01, 0], [-.7, 0, .01, 0], [-.7, 0, .01, 0],
                                    [-.7, 0, .01, 0], [-.7, 0, .01, 0], [-.7, 0, .3, 0]], dtype=np.float32)}
    second = {"candidate": np.array([4, 2, 4, 2, 2, 4]), "stable": np.ones(6, dtype=bool),
              "features": np.array([[-.95, 0, .01, 0]] * 4 + [[-.85, 0, .01, 0], [-.95, 0, .01, 0]], dtype=np.float32)}
    limits = {"2": [-.6, -.9, .1, .1], "4": [-.6, -.9, .1, .1]}
    before, after = predict(first, second, config, config, limits)
    assert before.tolist() == [2, 0, 0, 0, 0, 0]
    assert after.tolist() == [2, 2, 0, 0, 0, 0]


def main():
    torch.set_num_threads(2)
    self_test()
    names = ("old_baseline", "hierarchical32")
    configs = [load_model_checkpoint(MODELS[name], "cpu").metadata["decision_config"] for name in names]
    calibration = [dict(np.load(OUT / f"robust_calibration_{name}_evidence_v2.npz")) for name in names]
    assert np.array_equal(calibration[0]["keys"], calibration[1]["keys"])
    for name, data in zip(names, calibration):
        assert str(data["model_sha256"]) == model_sha256(MODELS[name])
    limits, detail = fit(*calibration, *configs)
    report = {"schema": "guarded_rescue_experiment_v1", "limits": limits, "calibration": detail,
              "models": {name: model_sha256(MODELS[name]) for name in names},
              "selection_partition": "17714 validation pieces excluding evaluation keys",
              "strategy": "Keep primary known predictions. Rescue IC only with both conditional candidates agreeing, stable primary views, accepted secondary evidence, and calibrated joint confidence/gates.",
              "deployment_ready": False, "test_used": False, "evaluation": {}}
    for condition in CONDITIONS:
        datasets = []
        for name in names:
            saved = torch.load(OUT / f"evidence_{name}_{condition}.pt", map_location="cpu", weights_only=True)
            primary, alternate = [HierarchicalTypeOutput(**saved[k]) for k in ("primary", "alternate")]
            features, candidate, stable = evidence(primary, alternate)
            datasets.append({"features": features, "candidate": candidate, "stable": stable})
        y = np.load(OUT / "validation_raw.npz")["labels"]
        base, predictions = predict(*datasets, *configs, limits)
        before, after = score(y, base), score(y, predictions)
        report["evaluation"][condition] = {"baseline": before, "guarded_rescue": after,
            "rescued_correct": int(((base == 0) & (predictions == y) & (y > 0)).sum()),
            "rescued_wrong": int(((base == 0) & (predictions != 0) & (predictions != y)).sum()),
            "secondary_call_fraction": float((base == 0).mean())}
        print(condition, "recall", round(before["known_macro_recall"], 4), "->", round(after["known_macro_recall"], 4),
              "NBE precision", [round(after["type_precision"][i], 4) for i in (2, 4)], flush=True)
    write_json(OUT / "guarded_rescue_report.json", report)


if __name__ == "__main__":
    main()
