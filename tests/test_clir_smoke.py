from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.clir_data import (
    CLIRTrajectoryDataset,
    SemanticGroupBatchSampler,
    clir_collate,
    write_jsonl,
)
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    dual_prior_losses,
)
from train_clir import DEFAULT_CONFIG, load_config, parse_args


def test_clir_forward_and_loss():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    batch = {
        "hidden_states": torch.randn(4, 6, 8),
        "condition_states": torch.randn(4, 3, 8),
        "condition_mask": torch.ones(4, 3),
        "mask": torch.ones(4, 6),
        "correctness": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "semantic_ids": torch.tensor([1, 1, 2, 2]),
        "style_ids": torch.tensor([1, 2, 1, 2]),
        "hallucination_onset": torch.tensor([-1, 3, -1, 2]),
        "onset_label_mask": torch.tensor([True, True, True, True]),
        "path_hallucinated": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "path_label_mask": torch.tensor([True, True, True, True]),
        "token_advantage": torch.zeros(4, 6),
        "token_advantage_mask": torch.ones(4, 6, dtype=torch.bool),
        "progress_targets": torch.zeros(4, 6),
        "progress_mask": torch.ones(4, 6, dtype=torch.bool),
        "key_prior_target": torch.zeros(4, 6),
        "complete_prior_target": torch.ones(4, 6),
    }

    outputs, losses = model.training_step(batch)
    assert outputs["scores"].shape == (4,)
    assert outputs["token_rewards"].shape == (4, 6)
    assert outputs["key_prior"].shape == (4, 6)
    assert outputs["condition_relevance"].shape == (4, 6)
    assert losses["total"].ndim == 0
    losses["total"].backward()


def test_condition_changes_prior_distribution():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    hidden_states = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5)
    condition_a = torch.randn(2, 3, 8)
    condition_b = condition_a.clone()
    condition_b[:, 0] = condition_b[:, 0] + 5.0

    out_a = model(hidden_states, mask=mask, condition_states=condition_a)
    out_b = model(hidden_states, mask=mask, condition_states=condition_b)

    assert not torch.allclose(out_a["key_prior"], out_b["key_prior"])


def test_prior_loss_is_zero_without_external_targets():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    outputs, losses = model.training_step(
        {
            "hidden_states": torch.randn(2, 4, 8),
            "mask": torch.ones(2, 4),
            "correctness": torch.tensor([1.0, 0.0]),
        }
    )

    assert outputs["key_prior"].shape == (2, 4)
    assert losses["prior_total"].item() == 0.0


def test_dual_prior_partial_mask_preserves_full_attention_mass():
    scores = torch.zeros(1)
    mask = torch.ones(1, 4)
    outputs = {
        "scores": scores,
        "mask": mask,
        "key_prior_logits": torch.zeros(1, 4),
        "complete_prior_logits": torch.zeros(1, 4),
        "key_prior": torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        "complete_prior": torch.tensor([[0.2, 0.2, 0.3, 0.3]]),
        "gates": torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        "fused_prior": torch.tensor([[0.2, 0.2, 0.3, 0.3]]),
        "complete_reconstruction": torch.zeros(1, 4),
    }
    batch = {
        "key_prior_target": torch.zeros(1, 4),
        "complete_prior_target": torch.zeros(1, 4),
        "key_prior_mask": torch.tensor([[True, True, False, False]]),
        "complete_prior_mask": torch.tensor([[True, True, False, False]]),
    }

    losses = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=1.0,
        gate_weight=1.0,
        reconstruction_weight=0.0,
        phase="key",
    )

    expected = torch.tensor((0.7 - 0.2) ** 2 + (0.1 - 0.2) ** 2)
    assert torch.isclose(losses["distill"], expected)
    assert torch.isclose(losses["gate"], expected)


def test_gate_alignment_trains_gate_and_detaches_fused_prior_target():
    gates = torch.tensor([[0.8, 0.2]], requires_grad=True)
    fused_prior = torch.tensor([[0.2, 0.8]], requires_grad=True)
    outputs = {
        "scores": torch.zeros(1),
        "mask": torch.ones(1, 2),
        "key_prior_logits": torch.zeros(1, 2),
        "complete_prior_logits": torch.zeros(1, 2),
        "key_prior": torch.tensor([[0.2, 0.8]]),
        "complete_prior": torch.tensor([[0.2, 0.8]]),
        "gates": gates,
        "fused_prior": fused_prior,
        "complete_reconstruction": torch.zeros(1, 2),
    }
    batch = {
        "key_prior_target": torch.zeros(1, 2),
        "complete_prior_target": torch.zeros(1, 2),
    }

    losses = dual_prior_losses(
        outputs,
        batch,
        key_weight=0.0,
        complete_weight=0.0,
        distill_weight=0.0,
        gate_weight=1.0,
        reconstruction_weight=0.0,
    )
    losses["total"].backward()

    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()
    assert gates.grad.abs().sum() > 0
    assert fused_prior.grad is None


def test_condition_module_params_scale_linearly_with_hidden_dim():
    # Regression test: the condition attention/fusion/reconstruction layers must
    # stay O(hidden_dim) via the condition_attention_dim bottleneck, not
    # O(hidden_dim^2). hidden_dim is meant to be a SWIFT-style concatenation of
    # many transformer layers (~1.3e5 for a 33-layer 8B model); a quadratic
    # regression there would balloon this "lightweight" reward model to ~1.8e11
    # parameters (bigger than GPT-3) instead of a few hundred million.
    def count_params(hidden_dim: int) -> int:
        config = RewardConfig(
            hidden_dim=hidden_dim, projection_dim=64, condition_attention_dim=64
        )
        return sum(p.numel() for p in ConsistencyLocalizedReward(config).parameters())

    small, large = 512, 4096
    n_small, n_large = count_params(small), count_params(large)
    ratio = n_large / n_small
    hidden_dim_ratio = large / small

    # Linear scaling keeps ratio close to hidden_dim_ratio (measured ~7.5 for 8x);
    # quadratic scaling would push it toward hidden_dim_ratio**2 (64 for 8x).
    assert ratio < 2 * hidden_dim_ratio, (
        f"condition module parameter count grew {ratio:.1f}x for a {hidden_dim_ratio:.0f}x "
        "hidden_dim increase -- looks quadratic again, check condition_attention_dim wiring"
    )


def test_invalid_hallucination_onset_raises():
    config = RewardConfig(hidden_dim=8, projection_dim=4)
    model = ConsistencyLocalizedReward(config)
    batch = {
        "hidden_states": torch.randn(1, 3, 8),
        "mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "hallucination_onset": torch.tensor([2]),
        "onset_label_mask": torch.tensor([True]),
    }

    try:
        model.training_step(batch)
    except ValueError as exc:
        assert "hallucination_onset" in str(exc)
    else:
        raise AssertionError("Expected invalid hallucination onset to raise ValueError")


def test_jsonl_dataset_collate(tmp_path: Path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows = []
    for idx in range(2):
        hidden_path = feature_dir / f"sample{idx}.pt"
        torch.save(torch.randn(3 + idx, 8), hidden_path)
        rows.append(
            {
                "id": f"sample{idx}",
                "query_id": "q0",
                "hidden_states_path": str(hidden_path),
                "correctness": idx == 0,
                "semantic_id": "q0",
                "style_id": f"style{idx}",
                "path_hallucinated": idx == 1,
                "key_prior_target": [1] + [0] * (2 + idx),
            }
        )

    jsonl_path = tmp_path / "data.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    loader = DataLoader(dataset, batch_size=2, collate_fn=clir_collate)
    batch = next(iter(loader))

    assert batch["hidden_states"].shape == (2, 4, 8)
    assert batch["mask"].sum().item() == 7
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["consistency_mask"].tolist() == [True, True]


def test_semantic_group_batch_sampler(tmp_path: Path):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for semantic_id in ["a", "b"]:
        for style_id in ["direct", "verbose"]:
            path = feature_dir / f"{semantic_id}_{style_id}.pt"
            torch.save(torch.randn(2, 8), path)
            rows.append(
                {
                    "id": f"{semantic_id}-{style_id}",
                    "query_id": semantic_id,
                    "hidden_states_path": str(path),
                    "semantic_id": semantic_id,
                    "style_id": style_id,
                }
            )

    jsonl_path = tmp_path / "grouped.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=2, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1], [2, 3]]

    sampler = SemanticGroupBatchSampler(dataset, batch_size=4, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3]]


def test_semantic_group_batch_sampler_packs_groups(tmp_path: Path):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for idx, semantic_id in enumerate(["a", "a", "a", "b", "b"]):
        path = feature_dir / f"sample_{idx}.pt"
        torch.save(torch.randn(2, 8), path)
        rows.append(
            {
                "id": f"sample-{idx}",
                "query_id": semantic_id,
                "hidden_states_path": str(path),
                "semantic_id": semantic_id,
                "style_id": f"style-{idx}",
            }
        )

    jsonl_path = tmp_path / "packed.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=4, shuffle=False)
    batches = list(iter(sampler))

    assert batches[0] == [0, 1, 3, 4]
    assert batches[1] == [2]


def test_semantic_group_batch_sampler_len_matches_iter_for_uneven_groups(
    tmp_path: Path,
):
    rows = []
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for group_idx in range(5):
        for sample_idx in range(2):
            row_idx = len(rows)
            path = feature_dir / f"sample_{row_idx}.pt"
            torch.save(torch.randn(2, 8), path)
            rows.append(
                {
                    "id": f"sample-{row_idx}",
                    "query_id": f"q-{group_idx}",
                    "hidden_states_path": str(path),
                    "semantic_id": f"q-{group_idx}",
                    "style_id": f"style-{sample_idx}",
                }
            )

    jsonl_path = tmp_path / "uneven.jsonl"
    write_jsonl(jsonl_path, rows)
    dataset = CLIRTrajectoryDataset(jsonl_path)
    sampler = SemanticGroupBatchSampler(dataset, batch_size=5, shuffle=False)
    batches = list(iter(sampler))

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert len(sampler) == len(batches)


def test_train_cli_uses_single_config_and_identity_development_override():
    args = parse_args(
        [
            "--train_jsonl",
            "train.jsonl",
            "--output_model",
            "model.pt",
            "--hidden_dim",
            "8",
        ]
    )
    config, training = load_config(args.config, args.hidden_dim)

    assert Path(args.config) == DEFAULT_CONFIG
    assert config.hidden_dim == 8
    assert config.encoder_type == "identity"
    assert config.progress_score_weight == 0.0
    assert config.gate_prior_weight == 0.0
    assert training["prior_phase_mode"] == "joint"
