from datetime import datetime
import inspect

import numpy as np
import pytest
import torch

import train
from data.training_manifest import PieceManifestEntry
from distance_ordinal import ordinal_distance_loss
from models import create_mtl_model
from tests.test_training_manifest import write_lig


def test_research_type_order_is_stable():
    assert train.RESEARCH_TYPE_NAMES == ["NCG", "NNBE", "PCG", "PNBE"]


def test_multitask_dataset_has_no_ic_sampling_parameter():
    assert "target_ic_fraction" not in inspect.signature(
        train.MultiTaskDataset
    ).parameters


def test_multitask_dataset_is_lazy_for_four_class_entries(tmp_path):
    ncg_path = write_lig(
        tmp_path / "NCG" / "100-200km" / "ncg.lig",
        pieces=3,
        piece_timestamps=[
            (20, 1, 1, 0, 0, 0),
            (20, 1, 2, 0, 0, 0),
            (20, 1, 3, 0, 0, 0),
        ],
    )
    entries = [
        PieceManifestEntry(str(ncg_path), 0, 0, 1, datetime(2020, 1, 1)),
        PieceManifestEntry(str(ncg_path), 2, 0, 1, datetime(2020, 1, 3)),
    ]
    shared_index = train.LigFileIndex([str(ncg_path)], validate=False)

    dataset = train.MultiTaskDataset(
        entries,
        split="test",
        lig_index=shared_index,
    )

    assert not hasattr(dataset, "data")
    assert len(dataset) == 2
    assert dataset.global_indices.tolist() == [0, 2]
    assert dataset.type_labels.tolist() == [0, 0]
    assert dataset.dist_labels.tolist() == [1, 1]
    assert len(dataset.file_ids) == len(dataset)
    assert len(dataset.date_ids) == len(dataset)
    assert dataset.date_ids.tolist() == [20200101, 20200103]
    assert dataset.file_ids.tolist() == [0, 0]
    waveform, type_label, dist_label = dataset[1]
    assert tuple(waveform.shape) == (1, 8000)
    assert type_label.ndim == 0
    assert dist_label.ndim == 0
    dataset.close()
    assert shared_index.read_piece(0).shape == (8000,)
    shared_index.close()


def test_distance_routing_separates_oracle_from_end_to_end_results():
    type_predictions = torch.tensor([1, 0])
    type_labels = torch.tensor([0, 0])
    dist_labels = torch.tensor([3, 4])
    dist_logits = [torch.zeros(2, 30) for _ in range(4)]
    dist_logits[0][0, 3] = 10
    dist_logits[0][1, 4] = 10
    dist_logits[1][0, 9] = 10

    routed = train.route_distance_predictions(
        type_predictions,
        type_labels,
        dist_labels,
        dist_logits,
    )

    assert routed["oracle_predictions"] == [3, 4]
    assert routed["end_to_end_predictions"] == [9, 4]
    assert routed["joint_correct"] == [False, True]


def test_alternate_stream_batches_consumes_each_loader_once():
    batches = list(
        train.alternate_stream_batches(["t1", "t2", "t3"], ["d1"])
    )

    assert batches == [
        ("type", "t1"),
        ("distance", "d1"),
        ("type", "t2"),
        ("type", "t3"),
    ]


def test_distance_loss_is_macro_averaged_across_present_types():
    logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([1, 1, 1, 2])
    distances = torch.tensor([0, 1, 2, 5])

    loss, components = train.compute_distance_head_loss(
        logits,
        labels,
        distances,
        tau=1.0,
        lambda_emd=1.0,
        lambda_reg=0.5,
        lambda_coarse=0.5,
    )
    type_one_loss, _ = ordinal_distance_loss(logits[0][:3], distances[:3])
    type_two_loss, _ = ordinal_distance_loss(logits[1][3:], distances[3:])

    assert torch.allclose(loss, (type_one_loss + type_two_loss) / 2)
    assert set(components) == {"soft_ce", "cdf", "huber", "coarse"}


def test_four_class_distance_loss_routes_zero_based_heads():
    logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([0, 1, 2, 3])
    distances = torch.tensor([1, 2, 3, 4])

    loss, _ = train.compute_distance_head_loss(logits, labels, distances)
    loss.backward()

    assert all(
        head.grad[index].abs().sum() > 0
        for index, head in enumerate(logits)
    )


def test_conditional_training_arguments_have_reliable_defaults():
    args = train.build_arg_parser().parse_args([])

    assert args.model_arch == "conditional_expert_v1"
    assert args.distance_sampling == "condition"
    assert args.distance_objective == "interval"
    assert args.distance_prediction == "expected"
    assert args.distance_batch_size == 128
    assert args.max_distance_samples_per_file == 256
    assert args.type_samples_per_epoch == 180000
    assert args.distance_samples_per_epoch == 60000
    assert args.min_eval_pieces == 500
    assert args.min_type_f1 == 0.85
    assert args.min_test_type_w2 == 0.70
    assert args.min_test_macro_w2 == 0.75
    assert args.lambda_dist == 1.0
    assert args.skip_test is False
    assert args.deterministic is False
    assert args.task_data == "../train_data"
    assert args.output == "./weights/conditional"
    assert args.init_model == ""
    assert args.resume == ""
    assert args.no_amp is False
    assert args.no_init is False
    assert args.min_type_precision == 0.85
    assert args.min_type_recall == 0.70
    assert args.baseline_metrics == ""
    assert train.build_arg_parser().parse_args(["--no_init"]).no_init is True


def test_conditional_distance_loss_routes_true_type_and_broad_intervals():
    distance_logits = [torch.zeros(4, 30, requires_grad=True) for _ in range(4)]
    coarse_logits = [torch.zeros(4, 6, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([0, 1, 2, 3])
    low = torch.tensor([0.0, 100.0, 600.0, 1500.0])
    high = torch.tensor([300.0, 200.0, 1200.0, 3000.0])

    loss, components = train.compute_conditional_distance_loss(
        distance_logits,
        coarse_logits,
        labels,
        low,
        high,
    )
    loss.backward()

    assert set(components) == {"interval", "coarse"}
    assert all(head.grad[index].abs().sum() > 0 for index, head in enumerate(distance_logits))
    assert all(head.grad[index].abs().sum() > 0 for index, head in enumerate(coarse_logits))


def test_four_class_training_has_no_ic_sampling_argument():
    args = train.build_arg_parser().parse_args([])

    assert not hasattr(args, "target_ic_fraction")


def test_balanced_type_sample_count_caps_to_smallest_class():
    labels = np.repeat(np.arange(4), [100, 80, 60, 20])

    assert train.balanced_type_sample_count(labels, 180) == 80
    assert train.balanced_type_sample_count(labels, 40) == 40


def test_initial_weights_copy_shared_encoder_and_type_head(tmp_path):
    old = create_mtl_model(base_channels=8, architecture="mtl_resnet")
    with torch.no_grad():
        old.stem[0].weight.fill_(0.25)
        old.type_head.weight.fill_(0.5)
        old.d_heads[0].weight.fill_(0.75)
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": old.state_dict()}, path)
    new = create_mtl_model(
        base_channels=8,
        architecture="ordinal_v2",
        dist_mlp_dim=8,
        dist_dropout=0.0,
    )
    distance_before = new.d_heads[0].weight.detach().clone()

    copied = train.load_initial_weights(new, path)

    assert "stem.0.weight" in copied
    assert "type_head.weight" in copied
    assert torch.all(new.stem[0].weight == 0.25)
    assert torch.all(new.type_head.weight == 0.5)
    assert torch.equal(new.d_heads[0].weight, distance_before)


def test_output_paths_use_classifier_checkpoint_names(tmp_path):
    paths = train.output_paths(tmp_path)

    assert paths["model"].name == "model.pt"
    assert paths["metrics"].name == "metrics.json"
    assert paths["best"].name == "best.pt"


def test_release_gate_reports_each_failed_threshold():
    metrics = {
        "type_f1": 0.84,
        "dist_equal_bin_macro_w2": 0.74,
        "per_type_equal_bin_w2": [0.80, 0.69, 0.90, 0.88],
    }

    passed, reasons = train.evaluate_release_gate(metrics)

    assert passed is False
    assert any("type_f1" in reason and "0.85" in reason for reason in reasons)
    assert any("macro_w2" in reason and "0.75" in reason for reason in reasons)
    assert any("type_w2[1]" in reason and "0.70" in reason for reason in reasons)
    good = {
        "type_f1": 0.9,
        "type_precision": [0.85] * 4,
        "type_recall": [0.70] * 4,
        "dist_equal_bin_macro_w2": 0.8,
        "per_type_equal_bin_w2": [0.7] * 4,
    }
    assert train.evaluate_release_gate(good) == (True, [])


def test_release_gate_enforces_four_class_precision_and_recall():
    metrics = {
        "type_f1": 0.90,
        "type_precision": [0.90, 0.84, 0.91, 0.88],
        "type_recall": [0.80, 0.82, 0.69, 0.75],
        "dist_equal_bin_macro_w2": 0.80,
        "per_type_equal_bin_w2": [0.80] * 4,
    }

    passed, reasons = train.evaluate_release_gate(metrics)

    assert passed is False
    assert any("type_precision[1]" in reason for reason in reasons)
    assert any("type_recall[2]" in reason for reason in reasons)


def test_four_class_selection_prefers_type_quality_before_distance():
    better_type = {
        "type_f1": 0.91,
        "type_min_recall": 0.80,
        "type_min_precision": 0.86,
        "dist_equal_bin_macro_w2": 0.70,
        "dist_equal_bin_macro_mae_km": 150.0,
    }
    better_distance = {
        "type_f1": 0.89,
        "type_min_recall": 0.80,
        "type_min_precision": 0.86,
        "dist_equal_bin_macro_w2": 0.95,
        "dist_equal_bin_macro_mae_km": 80.0,
    }

    assert train.make_four_class_selection_key(
        better_type
    ) > train.make_four_class_selection_key(better_distance)


def test_release_gate_uses_equal_bin_metrics_instead_of_raw_piece_metrics():
    metrics = {
        "type_f1": 0.90,
        "type_precision": [0.90] * 4,
        "type_recall": [0.80] * 4,
        "dist_macro_w2": 0.99,
        "per_type_w2": [0.99] * 4,
        "dist_equal_bin_macro_w2": 0.74,
        "per_type_equal_bin_w2": [0.80, 0.69, 0.90, 0.88],
    }

    passed, reasons = train.evaluate_release_gate(metrics)

    assert passed is False
    assert any("macro_w2" in reason for reason in reasons)
    assert any("type_w2[1]" in reason for reason in reasons)


def test_four_class_schema_metadata_is_explicit():
    metadata = train.four_class_schema_metadata()

    assert metadata == {
        "task_schema": "four_class_rejection_v1",
        "type_names": ["NCG", "NNBE", "PCG", "PNBE"],
        "rejected_type_name": "IC",
    }


def test_baseline_comparison_requires_file_and_rejects_type_regression(tmp_path):
    candidate = {
        "type_f1": 0.89,
        "type_precision": [0.90] * 4,
        "type_recall": [0.80] * 4,
    }
    passed, reasons = train.compare_baseline_metrics(candidate, "")
    assert passed is False
    assert reasons == ["baseline metrics file is required for promotion"]

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        '{"type_f1": 0.90, "type_precision": [0.85, 0.85, 0.85, 0.85], '
        '"type_recall": [0.70, 0.70, 0.70, 0.70]}',
        encoding="utf-8",
    )
    passed, reasons = train.compare_baseline_metrics(
        candidate, str(baseline_path)
    )
    assert passed is False
    assert any("type_f1" in reason for reason in reasons)


def test_rejected_type_summary_counts_rejection_as_false_negative():
    decoded = {
        "predicted": torch.tensor([0, 1, 2, 3, 0]),
        "accepted": torch.tensor([True, False, True, True, True]),
    }
    labels = torch.tensor([0, 1, 2, 0, 3])

    metrics = train.summarize_rejected_types(decoded, labels)

    assert metrics["type_coverage"] == pytest.approx(0.80)
    assert metrics["type_precision"] == [0.5, 0.0, 1.0, 0.0]
    assert metrics["type_recall"] == [0.5, 0.0, 1.0, 0.0]
    assert metrics["type_f1"] == pytest.approx((0.5 + 0.0 + 1.0 + 0.0) / 4)


def test_type_metrics_are_grouped_by_acquisition_year():
    grouped = train.summarize_type_metrics_by_year(
        predictions=np.array([0, 1, 0, 1]),
        labels=np.array([0, 1, 1, 1]),
        date_ids=np.array([20190101, 20190201, 20200101, 20200201]),
    )

    assert grouped["2019"]["accuracy"] == pytest.approx(1.0)
    assert grouped["2020"]["accuracy"] == pytest.approx(0.5)
    assert grouped["2019"]["count"] == 2


def test_failed_candidate_does_not_replace_deployed_model(tmp_path):
    deployed = tmp_path / "model.pt"
    deployed.write_bytes(b"trusted-model")
    checkpoint = {"model_state_dict": {"weight": torch.tensor([1.0])}}

    train.save_candidate_and_maybe_promote(
        tmp_path, checkpoint, {"type_f1": 0.5}, passed=False,
        reasons=["type_f1 failed"],
    )

    paths = train.output_paths(tmp_path)
    assert deployed.read_bytes() == b"trusted-model"
    assert paths["candidate"].is_file()
    assert paths["candidate_metrics"].is_file()

    train.save_candidate_and_maybe_promote(
        tmp_path, checkpoint, {"type_f1": 0.9}, passed=True, reasons=[]
    )
    loaded = torch.load(deployed, map_location="cpu", weights_only=False)
    assert torch.equal(loaded["model_state_dict"]["weight"], torch.tensor([1.0]))
    assert paths["metrics"].is_file()


def test_cuda_backend_defaults_to_fast_cudnn_mode():
    old = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        train.configure_cuda_backend(deterministic=False)
        assert torch.backends.cudnn.enabled is True
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False

        train.configure_cuda_backend(deterministic=True)
        assert torch.backends.cudnn.enabled is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.backends.cudnn.enabled = old[0]
        torch.backends.cudnn.benchmark = old[1]
        torch.backends.cudnn.deterministic = old[2]


def test_type_stream_does_not_update_distance_heads():
    torch.manual_seed(3)
    model = create_mtl_model(
        base_channels=8,
        architecture="ordinal_v2",
        dist_mlp_dim=8,
        dist_dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.d_heads[0].weight.detach().clone()

    metrics = train.train_stream_step(
        model=model,
        stream="type",
        x=torch.randn(2, 1, 128),
        type_labels=torch.tensor([0, 1]),
        distance_labels=torch.tensor([-1, 2]),
        optimizer=optimizer,
        type_criterion=torch.nn.CrossEntropyLoss(),
    )

    assert metrics["type_count"] == 2
    assert metrics["distance_count"] == 0
    assert torch.equal(before, model.d_heads[0].weight.detach())


def test_distance_stream_updates_routed_distance_head():
    torch.manual_seed(4)
    model = create_mtl_model(
        base_channels=8,
        architecture="ordinal_v2",
        dist_mlp_dim=8,
        dist_dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.d_heads[0].weight.detach().clone()

    metrics = train.train_stream_step(
        model=model,
        stream="distance",
        x=torch.randn(2, 1, 128),
        type_labels=torch.tensor([0, 0]),
        distance_labels=torch.tensor([2, 3]),
        optimizer=optimizer,
        type_criterion=torch.nn.CrossEntropyLoss(),
        distance_batch_type_weight=0.1,
    )

    assert metrics["type_count"] == 2
    assert metrics["distance_count"] == 2
    assert not torch.equal(before, model.d_heads[0].weight.detach())


def test_distance_loss_weight_scales_distance_head_update():
    torch.manual_seed(41)
    full = create_mtl_model(
        base_channels=8, architecture="ordinal_v2", dist_mlp_dim=8, dist_dropout=0.0
    )
    quarter = create_mtl_model(
        base_channels=8, architecture="ordinal_v2", dist_mlp_dim=8, dist_dropout=0.0
    )
    quarter.load_state_dict(full.state_dict())
    start = full.d_heads[0].weight.detach().clone()
    x = torch.randn(2, 1, 128)
    labels = torch.tensor([1, 1])
    distances = torch.tensor([2, 3])
    for model, weight in ((full, 1.0), (quarter, 0.25)):
        train.train_stream_step(
            model=model,
            stream="distance",
            x=x,
            type_labels=labels,
            distance_labels=distances,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            type_criterion=torch.nn.CrossEntropyLoss(),
            distance_batch_type_weight=0.0,
            distance_loss_weight=weight,
        )

    full_delta = start - full.d_heads[0].weight.detach()
    quarter_delta = start - quarter.d_heads[0].weight.detach()
    assert torch.allclose(quarter_delta, full_delta * 0.25, atol=1e-7)


def test_distance_routing_can_use_expected_bin():
    type_predictions = torch.tensor([0])
    type_labels = torch.tensor([0])
    dist_labels = torch.tensor([5])
    dist_logits = [torch.full((1, 30), -20.0) for _ in range(4)]
    dist_logits[0][0, 4] = 10.0
    dist_logits[0][0, 6] = 10.0

    routed = train.route_distance_predictions(
        type_predictions,
        type_labels,
        dist_labels,
        dist_logits,
        prediction_mode="expected",
    )

    assert routed["oracle_predictions"] == [5]
    assert routed["end_to_end_predictions"] == [5]


def test_uncovered_end_to_end_prediction_counts_as_w2_failure():
    metrics = train.summarize_distance_predictions(
        predictions=np.array([-1, 5]),
        targets=np.array([5, 5]),
    )

    assert metrics["coverage"] == 0.5
    assert metrics["w2"] == 0.5
    assert metrics["mae_km"] == 0.0


def test_split_hash_is_order_independent_but_label_sensitive(tmp_path):
    entries = [
        PieceManifestEntry(
            str(tmp_path / "b.lig"), 2, 2, 4, datetime(2020, 1, 2)
        ),
        PieceManifestEntry(
            str(tmp_path / "a.lig"), 1, 1, 2, datetime(2020, 1, 1)
        ),
    ]

    first = train.compute_split_hash(entries, "train")
    second = train.compute_split_hash(list(reversed(entries)), "train")
    changed_entries = [
        entries[0],
        PieceManifestEntry(
            entries[1].filepath,
            2,
            entries[1].type_idx,
            entries[1].dist_bin,
            entries[1].timestamp,
        ),
    ]

    assert first == second
    assert train.compute_split_hash(changed_entries, "train") != first


def test_shared_file_count_reports_expected_overlap():
    def piece(path, piece_index):
        return PieceManifestEntry(
            path,
            piece_index,
            0,
            0,
            datetime(2020, 1, 1),
        )

    splits = {
        "train": [piece("same.lig", 0)],
        "val": [piece("same.lig", 1)],
        "test": [piece("same.lig", 2), piece("other.lig", 0)],
    }

    assert train.count_cross_split_files(splits) == 1


def test_evaluate_reports_macro_worst_and_end_to_end_w2():
    class FixedModel(torch.nn.Module):
        def forward(self, x):
            type_logits = torch.full((4, 4), -10.0)
            distance_logits = [torch.full((4, 30), -10.0) for _ in range(4)]
            for row, lightning_type in enumerate([0, 1, 2, 3]):
                type_logits[row, lightning_type] = 10.0
                predicted_bin = 5 + row
                distance_logits[lightning_type][row, predicted_bin] = 10.0
            return type_logits, distance_logits

    loader = [(
        torch.zeros(4, 1, 32),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([5, 5, 5, 5]),
    )]

    metrics = train.evaluate(
        FixedModel(),
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.nn.CrossEntropyLoss(),
        "cpu",
        prediction_mode="expected",
    )

    assert metrics["dist_macro_w2"] == pytest.approx(0.75)
    assert metrics["dist_min_type_w2"] == pytest.approx(0.0)
    assert metrics["dist_equal_bin_macro_w2"] == pytest.approx(0.75)
    assert metrics["dist_equal_bin_min_type_w2"] == pytest.approx(0.0)
    assert metrics["per_type_equal_bin_w2"] == [1.0, 1.0, 1.0, 0.0]
    assert metrics["e2e_dist_w2"] == pytest.approx(0.75)
    assert metrics["e2e_dist_coverage"] == pytest.approx(1.0)
    assert metrics["type_precision"] == [1.0, 1.0, 1.0, 1.0]
    assert metrics["type_recall"] == [1.0, 1.0, 1.0, 1.0]
    assert metrics["type_min_precision"] == pytest.approx(1.0)
    assert metrics["type_min_recall"] == pytest.approx(1.0)
    assert metrics["type_confusion_matrix"] == np.eye(4, dtype=int).tolist()


def test_fit_distance_calibration_returns_four_temperatures_and_threshold():
    logits_by_head = []
    targets_by_head = []
    for head in range(4):
        logits = torch.full((3, 30), -8.0)
        targets = torch.tensor([head + 2, head + 3, head + 4])
        logits[torch.arange(3), targets] = 8.0
        logits_by_head.append(logits)
        targets_by_head.append(targets)

    calibration = train.fit_distance_calibration(
        logits_by_head,
        targets_by_head,
        target_w2=0.8,
    )

    assert len(calibration["temperatures"]) == 4
    assert all(0.5 <= value <= 5.0 for value in calibration["temperatures"])
    assert calibration["confidence_threshold"] <= 1.0
    assert calibration["validation_coverage"] == pytest.approx(1.0)
    assert calibration["validation_w2"] == pytest.approx(1.0)


def test_collect_distance_outputs_routes_zero_based_type_labels():
    class FixedModel:
        def eval(self):
            return self

        def __call__(self, x):
            return (
                torch.zeros((4, 4)),
                [torch.zeros((4, 30)) for _ in range(4)],
            )

    loader = [(
        torch.zeros((4, 1, 32)),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
    )]

    _, targets = train.collect_distance_outputs(FixedModel(), loader, "cpu")

    assert [values.tolist() for values in targets] == [[4], [5], [6], [7]]


def test_group_bootstrap_is_reproducible_and_returns_metric_intervals():
    predictions = np.array([5, 5, 10, 10])
    targets = np.array([5, 5, 5, 5])
    groups = np.array([0, 0, 1, 1])

    first = train.group_bootstrap_distance_metrics(
        predictions, targets, groups, repetitions=100, seed=9
    )
    second = train.group_bootstrap_distance_metrics(
        predictions, targets, groups, repetitions=100, seed=9
    )

    assert first == second
    assert set(first) == {"mae_km_ci95", "w2_ci95"}
    assert first["mae_km_ci95"][0] <= first["mae_km_ci95"][1]
    assert first["w2_ci95"][0] <= first["w2_ci95"][1]
