"""One confirmatory piece-test evaluation after freezing the rescue recipe."""
import hashlib
import json

import numpy as np
import torch

from verify_guarded_rescue import collect, needs_verifier
from guarded_rescue import predict
from robust_calibration import (ROOT, OUT, MODELS, load_model_checkpoint,
                                model_sha256, write_json, score)
from experiment import CONDITIONS, perturb
from data.dataset import FiveClassDataset
from data.manifest import build_piece_table
from data.split import assign_piece_splits


def main():
    torch.set_num_threads(2)
    frozen = json.loads((OUT / "guarded_rescue_frozen.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps({k: v for k, v in frozen.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()
    assert digest == frozen["sha256"]
    for name, path in MODELS.items():
        assert model_sha256(path) == frozen["models"][name]
    report_path = OUT / "guarded_rescue_test.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        assert existing["recipe_sha256"] == digest
        print("Frozen recipe already tested; refusing repeat selection on test labels.", flush=True)
        return
    write_json(OUT / "guarded_rescue_test_plan.json", {
        "recipe_sha256": digest, "threshold_updates_allowed": False,
        "partition": "original seed42 piece-level test; not source/year/station independent",
        "conditions": CONDITIONS, "corruption_seed": "20260922 + batch_start_position",
        "distance_models_unchanged": True,
    })
    table, _ = build_piece_table(ROOT.parent / "train_data", require_distance=True)
    assignment = assign_piece_splits(table, seed=42)
    positions = assignment.positions("test")
    keys = [table.piece_key(int(p)) for p in positions]
    calibration = np.load(OUT / "robust_calibration_old_baseline_evidence_v2.npz")["keys"]
    validation = np.load(OUT / "validation_raw.npz")["keys"]
    assert not set(keys).intersection(calibration.tolist())
    assert not set(keys).intersection(validation.tolist())
    models = [load_model_checkpoint(path, "cuda") for path in MODELS.values()]
    for cp in models:
        cp.model.eval()
    configs = [cp.metadata["decision_config"] for cp in models]
    dataset = FiveClassDataset(table, positions, "test")
    records = {condition: {key: [] for key in ("baseline", "rescue", "needed", "first_agreement", "second_agreement")}
               for condition in CONDITIONS}
    try:
        for start in range(0, len(positions), 128):
            selected = positions[start:start + 128]
            raw = np.stack(dataset.lig.read_pieces_batch([
                dataset._global_piece_index(int(p)) for p in selected])).astype(np.float32)
            for condition in CONDITIONS:
                changed = perturb(raw, condition, seed=20260922 + start)
                first, second = [collect(cp, changed, table.daylight[selected], "cuda") for cp in models]
                base, result = predict(first, second, *configs, frozen["limits"])
                _, needed = needs_verifier(first, configs[0], frozen["limits"])
                for key, value in (("baseline", base), ("rescue", result), ("needed", needed),
                                   ("first_agreement", first["view_agreement"]),
                                   ("second_agreement", second["view_agreement"])):
                    records[condition][key].append(value)
            if start % 2048 == 0:
                print(f"Fixed-recipe test {start}/{len(positions)}", flush=True)
    finally:
        dataset.close()
    targets = table.type_index[positions]
    result = {"recipe_sha256": digest, "partition": "test", "piece_count": len(positions),
              "piece_keys_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
              "calibration_key_overlap": 0, "validation_key_overlap": 0,
              "models": frozen["models"], "threshold_updates_after_test": False,
              "deployment_ready": False, "conditions": {}}
    for condition, lists in records.items():
        arrays = {key: np.concatenate(value) for key, value in lists.items()}
        condition_result = {}
        for method in ("baseline", "rescue"):
            metrics = score(targets, arrays[method])
            confusion = np.array(metrics["type_confusion"])
            rates = confusion / confusion.sum(1, keepdims=True)
            metrics["uniform_class_weighted_precision"] = (rates.diagonal() / rates.sum(0).clip(1e-12)).tolist()
            condition_result[method] = metrics
        condition_result.update(
            recovered_correct=int(((arrays["baseline"] == 0) & (arrays["rescue"] == targets) & (targets > 0)).sum()),
            introduced_wrong=int(((arrays["baseline"] == 0) & (arrays["rescue"] != 0) & (arrays["rescue"] != targets)).sum()),
            known_view_consistency_primary=float(arrays["first_agreement"][targets > 0].mean()),
            known_view_consistency_secondary=float(arrays["second_agreement"][targets > 0].mean()),
            secondary_call_fraction=float(arrays["needed"].mean()))
        result["conditions"][condition] = condition_result
        print(condition, "known recall", round(condition_result["baseline"]["known_macro_recall"], 4), "->",
              round(condition_result["rescue"]["known_macro_recall"], 4), "NBE precision",
              [round(condition_result["rescue"]["type_precision"][i], 4) for i in (2, 4)], flush=True)
    write_json(report_path, result)


if __name__ == "__main__":
    main()
