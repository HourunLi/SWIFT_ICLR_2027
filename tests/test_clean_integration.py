import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from evaluate_clir import evaluate
from extract_hidden_states import extract_row, validate_token_ids
from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    resolve_feature_metadata,
    write_jsonl,
)
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    dual_prior_losses,
    hallucination_localization_losses,
    path_no_hallucination_log_probability,
    path_level_hallucination_mil,
)
from train_clir import (
    DEFAULT_CONFIG,
    feature_reference_state,
    load_config,
    split_indices_by_query,
    validate_feature_contract,
)


ABLATION_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "clean_ablation_v1"
)
GATE_ABLATION_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "clean_gate_ablation_v1"
)
GATE_TUNING_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "configs" / "clean_gate_tuning_v2"
)


def test_best_current_is_compact_and_uses_retained_defaults():
    config, training = load_config(DEFAULT_CONFIG)
    model = ConsistencyLocalizedReward(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert config.hidden_dim == 33 * 3072
    assert config.model_dim == 768
    assert config.encoder_type == "layer_transformer"
    assert parameter_count < 10_000_000
    assert config.progress_score_weight == 0.0
    assert config.mil_weight == 0.0
    assert config.pseudo_tail_weight == 0.0
    assert config.prior_weight == 1.0
    assert config.prior_distill_weight == 0.25
    assert config.gate_prior_weight == 0.0
    assert training["prior_phase_mode"] == "joint"


def test_clean_ablation_v1_changes_only_declared_loss_families():
    config_names = (
        "c0_correctness_only",
        "c1_consistency",
        "h0_onset_bce",
        "h1_onset_tail",
        "p0_direct_prior",
        "p1_mutual_prior",
        "full_integration",
    )
    payloads = {
        name: json.loads((ABLATION_CONFIG_DIR / f"{name}.json").read_text())
        for name in config_names
    }
    training = payloads["c0_correctness_only"]["training"]
    assert training["epochs"] == 3
    assert all(payload["training"] == training for payload in payloads.values())

    factor_keys = {
        "consistency_weight",
        "hallucination_weight",
        "token_reward_weight",
        "tail_weight",
        "prior_weight",
        "prior_distill_weight",
    }
    invariant = {
        key: value
        for key, value in payloads["c0_correctness_only"]["model"].items()
        if key not in factor_keys
    }
    for payload in payloads.values():
        assert {
            key: value
            for key, value in payload["model"].items()
            if key not in factor_keys
        } == invariant

    factor_order = (
        "consistency_weight",
        "hallucination_weight",
        "token_reward_weight",
        "tail_weight",
        "prior_weight",
        "prior_distill_weight",
    )
    expected = {
        "c0_correctness_only": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "c1_consistency": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "h0_onset_bce": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "h1_onset_tail": (0.0, 1.0, 0.5, 0.5, 0.0, 0.0),
        "p0_direct_prior": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "p1_mutual_prior": (0.0, 0.0, 0.0, 0.0, 1.0, 0.25),
        "full_integration": (1.0, 1.0, 0.5, 0.5, 1.0, 0.25),
    }
    for name, values in expected.items():
        model = payloads[name]["model"]
        assert tuple(model[key] for key in factor_order) == values

    best = json.loads(DEFAULT_CONFIG.read_text())
    assert payloads["full_integration"]["model"] == best["model"]


def test_clean_gate_ablation_changes_only_main_scale_gate_alignment():
    p0 = json.loads((GATE_ABLATION_CONFIG_DIR / "p0_direct_prior.json").read_text())
    pg0 = json.loads(
        (GATE_ABLATION_CONFIG_DIR / "pg0_direct_prior_gate.json").read_text()
    )

    assert p0["training"] == pg0["training"]
    p0_model = dict(p0["model"])
    pg0_model = dict(pg0["model"])
    assert p0_model.pop("gate_prior_weight") == 0.0
    assert pg0_model.pop("gate_prior_weight") == 0.0625
    assert p0_model == pg0_model
    assert pg0_model["prior_weight"] == 1.0
    assert pg0_model["prior_distill_weight"] == 0.0
    assert 1.0 * 0.0625 == 0.25 * 0.25


def test_clean_gate_tuning_v2_changes_only_gate_alignment_strength():
    names_and_weights = {
        "g025_main_inner": 0.25,
        "g100_balanced": 1.0,
        "g400_intermediate": 4.0,
        "g1000_historical_strong": 10.0,
    }
    baseline = json.loads(
        (GATE_ABLATION_CONFIG_DIR / "p0_direct_prior.json").read_text()
    )
    baseline_model = dict(baseline["model"])
    assert baseline_model.pop("gate_prior_weight") == 0.0

    for name, weight in names_and_weights.items():
        payload = json.loads((GATE_TUNING_CONFIG_DIR / f"{name}.json").read_text())
        assert payload["training"] == baseline["training"]
        model = dict(payload["model"])
        assert model.pop("gate_prior_weight") == weight
        assert model == baseline_model


def test_layer_axis_encoder_forward_and_gradient():
    config = RewardConfig(
        hidden_dim=24,
        encoder_type="layer_transformer",
        model_dim=8,
        num_feature_layers=3,
        per_layer_dim=8,
        layer_encoder_dim=8,
        layer_encoder_blocks=1,
        layer_encoder_heads=2,
        layer_pool_queries=2,
        projection_dim=4,
        condition_attention_dim=4,
    )
    model = ConsistencyLocalizedReward(config)
    outputs, losses = model.training_step(
        {
            "hidden_states": torch.randn(2, 4, 24),
            "mask": torch.ones(2, 4),
            "correctness": torch.tensor([1.0, 0.0]),
        }
    )
    losses["total"].backward()

    assert outputs["token_features"].shape == (2, 4, 8)
    assert outputs["layer_attention"].shape == (2, 4, 2, 3)
    assert torch.allclose(outputs["layer_attention"].sum(-1), torch.ones(2, 4, 2))
    assert model.feature_encoder.input_projection.weight.grad is not None


def test_main_hallucination_onset_shapes_post_onset_reward_path():
    logits = torch.zeros(1, 4, requires_grad=True)
    token_values = torch.zeros(1, 4, requires_grad=True)
    losses = hallucination_localization_losses(
        logits,
        token_values,
        mask=torch.ones(1, 4),
        onset=torch.tensor([2]),
        onset_label_mask=torch.tensor([True]),
        token_advantage=None,
        token_advantage_mask=None,
        negative_tail_margin=0.5,
    )
    (losses["token_bce"] + losses["token_reward"] + losses["tail_margin"]).backward()

    assert token_values.grad[0, :2].abs().sum() == 0
    assert token_values.grad[0, 2:].abs().sum() > 0
    assert logits.grad[0, :2].abs().sum() > 0
    assert logits.grad[0, 2:].abs().sum() > 0


def test_token_advantage_trains_without_an_onset_row_in_the_batch():
    config = RewardConfig(
        hidden_dim=4,
        projection_dim=4,
        condition_attention_dim=4,
        final_weight=0.0,
        consistency_weight=0.0,
        hallucination_weight=1.0,
        token_reward_weight=0.5,
        tail_weight=0.5,
        prior_weight=0.0,
    )
    model = ConsistencyLocalizedReward(config)
    _, losses = model.training_step(
        {
            "hidden_states": torch.randn(2, 3, 4),
            "mask": torch.ones(2, 3),
            "token_advantage": torch.ones(2, 3),
            "token_advantage_mask": torch.ones(2, 3, dtype=torch.bool),
        }
    )

    assert losses["localization_token_bce"].item() == 0.0
    assert losses["localization_tail_margin"].item() == 0.0
    assert losses["localization_token_reward"].item() > 0.0
    assert losses["total"].requires_grad


def test_log_space_mil_is_finite_for_long_paths():
    logits = torch.full((2, 4096), -20.0, requires_grad=True)
    loss = path_level_hallucination_mil(
        logits,
        torch.ones_like(logits),
        torch.tensor([1.0, 0.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_log_clean_path_probability_remains_informative_for_long_paths():
    logits = torch.zeros(2, 4096)
    mask = torch.ones_like(logits)
    log_clean = path_no_hallucination_log_probability(logits, mask)

    assert torch.isfinite(log_clean).all()
    assert torch.all(log_clean < -1000)


def test_disabled_prior_branches_do_not_evaluate_nan_targets():
    outputs = {
        "scores": torch.zeros(1),
        "mask": torch.ones(1, 2),
        "key_prior_logits": torch.zeros(1, 2),
        "complete_prior_logits": torch.zeros(1, 2),
        "key_prior": torch.full((1, 2), 0.5),
        "complete_prior": torch.full((1, 2), 0.5),
        "gates": torch.full((1, 2), 0.5),
        "fused_prior": torch.full((1, 2), 0.5),
        "complete_reconstruction": torch.zeros(1, 4),
    }
    losses = dual_prior_losses(
        outputs,
        {"complete_reconstruction_target": torch.full((1, 1), float("nan"))},
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=0.0,
        gate_weight=0.0,
        reconstruction_weight=0.0,
    )

    assert losses["total"].item() == 0.0
    assert torch.isfinite(losses["reconstruction"])


def test_disabled_progress_head_cannot_poison_score_or_receive_gradient():
    config = RewardConfig(
        hidden_dim=4,
        projection_dim=4,
        condition_attention_dim=4,
        consistency_weight=0.0,
        hallucination_weight=0.0,
        mil_weight=0.0,
        token_reward_weight=0.0,
        tail_weight=0.0,
        pseudo_tail_weight=0.0,
        progress_weight=0.0,
        progress_score_weight=0.0,
        prior_weight=0.0,
    )
    model = ConsistencyLocalizedReward(config)
    with torch.no_grad():
        model.progress_head.weight.fill_(float("nan"))
        model.progress_head.bias.fill_(float("nan"))
    outputs, losses = model.training_step(
        {
            "hidden_states": torch.randn(2, 3, 4),
            "mask": torch.ones(2, 3),
            "correctness": torch.tensor([1.0, 0.0]),
        }
    )
    losses["total"].backward()

    assert torch.isfinite(outputs["scores"]).all()
    assert model.progress_head.weight.grad is None
    assert model.progress_head.bias.grad is None


def test_dataset_preserves_bfloat16_and_masks_missing_correctness(tmp_path: Path):
    rows = []
    for index in range(2):
        feature = tmp_path / f"{index}.pt"
        torch.save(torch.randn(3, 8, dtype=torch.bfloat16), feature)
        row = {
            "id": str(index),
            "query_id": "q",
            "hidden_states_path": str(feature),
            "key_prior_target": [1, 0, 0],
        }
        if index == 0:
            row["correctness"] = 1
        rows.append(row)
    data = tmp_path / "data.jsonl"
    write_jsonl(data, rows)
    batch = next(
        iter(
            DataLoader(
                CLIRTrajectoryDataset(data), batch_size=2, collate_fn=clir_collate
            )
        )
    )

    assert batch["hidden_states"].dtype == torch.bfloat16
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["correctness_mask"].tolist() == [True, False]


def test_token_labels_must_match_feature_length(tmp_path: Path):
    feature = tmp_path / "feature.pt"
    torch.save(torch.randn(3, 8), feature)
    data = tmp_path / "bad.jsonl"
    write_jsonl(
        data,
        [
            {
                "id": "bad",
                "query_id": "q",
                "hidden_states_path": str(feature),
                "key_prior_target": [1, 0, 0, 0],
            }
        ],
    )
    dataset = CLIRTrajectoryDataset(data)
    try:
        dataset[0]
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("Expected strict token-label validation")


def test_collate_cannot_silently_truncate_custom_sequence_targets():
    item = {
        "row_index": 0,
        "id": "bad",
        "query_id": "q",
        "hidden_states": torch.randn(3, 4),
        "token_advantage": torch.ones(2),
    }
    with pytest.raises(ValueError, match="token_advantage must have length 3"):
        clir_collate([item])


def test_legacy_nested_feature_metadata_and_checksums_are_supported(tmp_path: Path):
    feature = tmp_path / "feature.pt"
    torch.save(torch.randn(2, 6), feature)
    common = {
        "id": "row",
        "query_id": "q",
        "hidden_states_path": str(feature),
        "feature_metadata": {
            "feature_dim": 6,
            "layer_count": 2,
            "per_layer_hidden_size": 3,
        },
    }
    legacy_manifest = tmp_path / "legacy.jsonl"
    canonical_manifest = tmp_path / "canonical.jsonl"
    write_jsonl(legacy_manifest, [{**common, "feature_sha256": "abc"}])
    write_jsonl(canonical_manifest, [{**common, "hidden_states_sha256": "abc"}])
    legacy = CLIRTrajectoryDataset(legacy_manifest)
    canonical = CLIRTrajectoryDataset(canonical_manifest)
    config = RewardConfig(
        hidden_dim=6,
        encoder_type="layer_transformer",
        model_dim=4,
        num_feature_layers=2,
        per_layer_dim=3,
        layer_encoder_dim=4,
        layer_encoder_blocks=1,
        layer_encoder_heads=2,
        layer_pool_queries=1,
        projection_dim=4,
        condition_attention_dim=4,
    )

    assert resolve_feature_metadata(common) == {
        "feature_dim": 6,
        "num_feature_layers": 2,
        "per_layer_dim": 3,
    }
    validate_feature_contract(legacy, config, "legacy")
    assert feature_reference_state(legacy, [0]) == feature_reference_state(
        canonical, [0]
    )


def test_conflicting_feature_metadata_is_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_feature_metadata(
            {
                "feature_dim": 8,
                "feature_metadata": {"feature_dim": 6},
            }
        )


def test_query_fraction_split_is_disjoint(tmp_path: Path):
    rows = []
    for query in range(4):
        for candidate in range(2):
            feature = tmp_path / f"{query}-{candidate}.pt"
            torch.save(torch.randn(2, 4), feature)
            rows.append(
                {
                    "id": f"{query}-{candidate}",
                    "query_id": f"q{query}",
                    "hidden_states_path": str(feature),
                }
            )
    data = tmp_path / "split.jsonl"
    write_jsonl(data, rows)
    dataset = CLIRTrajectoryDataset(data)
    train, val = split_indices_by_query(dataset, 0.25, seed=42)
    train_queries = {dataset.rows[index]["query_id"] for index in train}
    val_queries = {dataset.rows[index]["query_id"] for index in val}

    assert train_queries.isdisjoint(val_queries)
    assert sorted(train + val) == list(range(8))


class FakeCausalModel:
    def __call__(self, input_ids, **kwargs):
        del kwargs
        length = input_ids.shape[1]
        layers = tuple(torch.full((1, length, 4), float(layer)) for layer in range(3))
        return SimpleNamespace(hidden_states=layers)


def test_exact_id_extraction_slices_prompt_and_output_without_tokenizer():
    trajectory, condition, layers, width = extract_row(
        FakeCausalModel(),
        prompt_token_ids=[10, 11, 12],
        output_token_ids=[20, 21],
        device=torch.device("cpu"),
    )

    assert trajectory.shape == (2, 12)
    assert condition.shape == (3, 12)
    assert layers == 3
    assert width == 4


@pytest.mark.parametrize("bad_id", [1.5, "2", True])
def test_exact_id_validation_rejects_coercible_non_integer_ids(bad_id):
    with pytest.raises(ValueError, match="integer token IDs"):
        validate_token_ids([1, bad_id], "output_token_ids", "row")


def test_query_level_evaluator_uses_stable_ties_and_pairwise_metric():
    rows = [
        {"query_id": "q0", "candidate_index": 0, "correctness": 1, "clir_score": 0.5},
        {"query_id": "q0", "candidate_index": 1, "correctness": 0, "clir_score": 0.5},
        {"query_id": "q1", "candidate_index": 0, "correctness": 0, "clir_score": 0.1},
        {"query_id": "q1", "candidate_index": 1, "correctness": 1, "clir_score": 0.9},
    ]
    report = evaluate(rows, "clir_score", "correctness", [2], 100, 7)

    assert report["by_k"]["2"]["bon_accuracy"] == 1.0
    assert report["within_query_pairwise"]["comparisons"] == 2
    assert report["within_query_pairwise"]["ties"] == 1
    assert report["within_query_pairwise"]["accuracy"] == 0.75


def test_evaluator_requires_a_common_query_population_by_default():
    rows = [
        {"query_id": "q0", "candidate_index": 0, "correctness": 1, "clir_score": 0.5},
        {"query_id": "q0", "candidate_index": 1, "correctness": 0, "clir_score": 0.4},
        {"query_id": "q1", "candidate_index": 0, "correctness": 1, "clir_score": 0.6},
    ]
    with pytest.raises(ValueError, match="fewer than max K=2"):
        evaluate(rows, "clir_score", "correctness", [1, 2], 0, 7)

    report = evaluate(
        rows,
        "clir_score",
        "correctness",
        [1, 2],
        0,
        7,
        allow_incomplete_queries=True,
    )
    assert report["by_k"]["1"]["queries"] == 2
    assert report["by_k"]["2"]["queries"] == 1


@pytest.mark.parametrize("candidate_index", [1.5, "1", True])
def test_evaluator_rejects_coercible_non_integer_candidate_indices(candidate_index):
    row = {
        "query_id": "q",
        "candidate_index": candidate_index,
        "correctness": 1,
        "clir_score": 0.0,
    }
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate([row], "clir_score", "correctness", [1], 0, 7)


@pytest.mark.parametrize(
    ("field", "value"),
    [("clir_score", float("nan")), ("correctness", 0.5)],
)
def test_evaluator_rejects_non_finite_scores_and_non_binary_labels(field, value):
    row = {
        "query_id": "q",
        "candidate_index": 0,
        "correctness": 1,
        "clir_score": 0.0,
    }
    row[field] = value
    with pytest.raises(ValueError):
        evaluate([row], "clir_score", "correctness", [1], 0, 7)
