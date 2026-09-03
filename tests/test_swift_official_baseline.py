"""Regression tests for the official-SWIFT baseline adapter and its protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from run_swift_official_baseline_training import (
    CELL,
    _model_config,
    _training_contract,
    load_protocol,
)
from src.clir_smoke import file_sha256
from src.swift_official_baseline import (
    UPSTREAM_COMMIT,
    SwiftLinearRewardModel,
    load_feature_tensor,
    query_disjoint_split,
    stacked_swift_scores,
    swift_collate,
)
from summarize_clir_prior_ablation_v2 import _holm


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/swift_official_baseline_v1/protocol.json"
VENDOR_UTILS = ROOT / "run_artifacts/vendor/swift-41f7c9f7/utils.py"
FEATURE_DIM = 24
TRAIN_MANIFEST_SHA256 = (
    "ef3bd3a2f5aada2809830181940f464b235c88bd933b83be08c7eb4bf68d6fcf"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _upstream_linear_reward_model():
    """Import the vendored upstream module directly, without a package import."""

    if not VENDOR_UTILS.exists():  # pragma: no cover - vendor checkout is optional.
        pytest.skip(f"vendored SWIFT checkout is absent: {VENDOR_UTILS}")
    spec = importlib.util.spec_from_file_location("swift_vendor_utils", VENDOR_UTILS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LinearRewardModel


def _batch(seed: int, *, width: int = FEATURE_DIM):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(3, 7, width, generator=generator)
    lengths = [7, 4, 1]
    for index, length in enumerate(lengths):
        hidden_states[index, length:] = 0.0
    return hidden_states, lengths


@pytest.mark.parametrize("disable_gate", [False, True])
def test_adapter_matches_the_vendored_upstream_head_bit_for_bit(disable_gate) -> None:
    upstream_class = _upstream_linear_reward_model()
    torch.manual_seed(0)
    adapter = SwiftLinearRewardModel(FEATURE_DIM, disable_gate=disable_gate).eval()
    upstream = upstream_class(FEATURE_DIM, disable_gate=disable_gate).eval()
    upstream.load_state_dict(adapter.state_dict())
    hidden_states, lengths = _batch(11)
    with torch.no_grad():
        ours = adapter(hidden_states, lengths)
        theirs = upstream(hidden_states, lengths)
    assert ours.shape == (3,)
    assert float((ours - theirs).abs().max()) == 0.0


def test_stacked_scoring_equals_the_per_checkpoint_loop_elementwise() -> None:
    hidden_states, lengths = _batch(23)
    models = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        models.append(SwiftLinearRewardModel(FEATURE_DIM).eval())
    state_dicts = [model.state_dict() for model in models]
    stacked = stacked_swift_scores(hidden_states, lengths, state_dicts)
    assert stacked.shape == (3, 3)
    for position, model in enumerate(models):
        with torch.no_grad():
            reference = model(hidden_states, lengths)
        assert float((stacked[:, position] - reference).abs().max()) == 0.0


def test_stacked_scoring_rejects_an_empty_checkpoint_set() -> None:
    hidden_states, lengths = _batch(24)
    with pytest.raises(ValueError, match="at least one SWIFT checkpoint"):
        stacked_swift_scores(hidden_states, lengths, [])


def test_gate_denominator_clamp_keeps_scores_finite_and_zero_padding_is_inert() -> None:
    model = SwiftLinearRewardModel(FEATURE_DIM).eval()
    # Drive every gate logit to -inf so the gate denominator collapses to zero.
    with torch.no_grad():
        model.fused_layer.weight.zero_()
        model.fused_layer.bias[0] = -1e9
        model.fused_layer.bias[1] = 3.0
    hidden_states, lengths = _batch(37)
    with torch.no_grad():
        scores = model(hidden_states, lengths)
    assert bool(torch.isfinite(scores).all())
    assert float(scores.abs().max()) == 0.0

    ungated = SwiftLinearRewardModel(FEATURE_DIM, disable_gate=True).eval()
    with torch.no_grad():
        ungated.reward_layer.weight.zero_()
        ungated.reward_layer.bias.fill_(2.0)
        # A zero-length candidate must not divide by zero.
        empty = ungated(hidden_states, [0, 0, 0])
    assert bool(torch.isfinite(empty).all())
    assert float(empty.abs().max()) == 0.0


def test_padding_does_not_change_a_candidate_score() -> None:
    torch.manual_seed(5)
    model = SwiftLinearRewardModel(FEATURE_DIM).eval()
    generator = torch.Generator().manual_seed(6)
    short = torch.randn(1, 3, FEATURE_DIM, generator=generator)
    padded = torch.cat([short, torch.randn(1, 4, FEATURE_DIM, generator=generator)], dim=1)
    with torch.no_grad():
        tight = model(short, [3])
        loose = model(padded, [3])
    assert torch.equal(tight, loose)


def _feature_row(tmp_path: Path, tensor: torch.Tensor, *, tokens: int) -> dict:
    path = tmp_path / "feature.pt"
    torch.save({"hidden_states": tensor}, path)
    return {
        "id": "row-0",
        "hidden_states_path": str(path),
        "output_token_ids": list(range(tokens)),
        "feature_dim": FEATURE_DIM,
    }


def test_feature_loader_rejects_shape_length_width_and_dtype_drift(tmp_path) -> None:
    good = torch.randn(5, FEATURE_DIM)
    row = _feature_row(tmp_path, good, tokens=5)
    assert load_feature_tensor(row, tmp_path).shape == (5, FEATURE_DIM)

    with pytest.raises(ValueError, match="must be \\[time, width\\]"):
        load_feature_tensor(
            _feature_row(tmp_path, torch.randn(5, 2, FEATURE_DIM), tokens=5), tmp_path
        )
    with pytest.raises(ValueError, match="output-token feature axis drift"):
        load_feature_tensor(_feature_row(tmp_path, good, tokens=4), tmp_path)
    with pytest.raises(ValueError, match="feature width drift"):
        load_feature_tensor(
            _feature_row(tmp_path, torch.randn(5, FEATURE_DIM - 1), tokens=5), tmp_path
        )
    with pytest.raises(ValueError, match="not floating point"):
        load_feature_tensor(
            _feature_row(tmp_path, torch.zeros(5, FEATURE_DIM, dtype=torch.int64), tokens=5),
            tmp_path,
        )
    with pytest.raises(ValueError, match="hidden_states_path is missing"):
        load_feature_tensor({"id": "row-1"}, tmp_path)


def _item(row_index: int, tokens: int, width: int, label: float) -> dict:
    return {
        "row_index": row_index,
        "id": f"row-{row_index}",
        "query_id": f"q-{row_index}",
        "candidate_index": row_index,
        "hidden_states": torch.randn(tokens, width),
        "length": tokens,
        "correctness": label,
    }


def test_collate_pads_but_reports_unpadded_lengths_and_rejects_mixed_widths() -> None:
    batch = swift_collate([_item(0, 5, FEATURE_DIM, 1.0), _item(1, 2, FEATURE_DIM, 0.0)])
    assert batch["hidden_states"].shape == (2, 5, FEATURE_DIM)
    assert batch["lengths"] == [5, 2]
    assert float(batch["hidden_states"][1, 2:].abs().max()) == 0.0
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["row_indices"] == [0, 1]

    with pytest.raises(ValueError, match="mixed feature widths"):
        swift_collate([_item(0, 5, FEATURE_DIM, 1.0), _item(1, 5, FEATURE_DIM + 1, 0.0)])
    with pytest.raises(ValueError, match="empty SWIFT batch"):
        swift_collate([])


def test_query_disjoint_split_is_deterministic_and_label_blind() -> None:
    rows = [
        {"query_id": f"q-{index // 3}", "correctness": index % 2}
        for index in range(30)
    ]
    first = query_disjoint_split(rows, validation_fraction=0.2, namespace="swift")
    second = query_disjoint_split(rows, validation_fraction=0.2, namespace="swift")
    assert first == second

    flipped = [
        {"query_id": row["query_id"], "correctness": 1 - int(row["correctness"])}
        for row in rows
    ]
    assert query_disjoint_split(flipped, validation_fraction=0.2, namespace="swift") == first

    train_queries = set(first["train_query_ids"])
    validation_queries = set(first["validation_query_ids"])
    assert train_queries and validation_queries
    assert not train_queries & validation_queries
    assert not set(first["train_indices"]) & set(first["validation_indices"])
    assert len(first["train_indices"]) + len(first["validation_indices"]) == len(rows)
    assert query_disjoint_split(
        rows, validation_fraction=0.2, namespace="other"
    ) != first

    with pytest.raises(ValueError, match="strictly between zero and one"):
        query_disjoint_split(rows, validation_fraction=0.0, namespace="swift")


def test_protocol_freezes_one_cell_at_the_matched_u0_budget() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert list(protocol["cells"]) == [CELL] == ["swift_official"]
    training = protocol["training"]
    assert [int(value) for value in training["seeds"]] == [42, 43, 44]
    assert int(training["run_count"]) == 3
    assert int(training["epochs"]) == 3
    assert int(training["batch_size"]) == 4
    assert training["validation_split"] is None
    assert training["early_stopping"] is None
    assert training["fixed_epoch_no_checkpoint_selection"] is True
    assert training["group_by_semantic_id"] is False
    assert protocol["upstream"]["commit"] == UPSTREAM_COMMIT

    declared = {entry["field"] for entry in training["declared_upstream_deviations"]}
    assert declared == {
        "epochs",
        "batch_size",
        "weight_decay",
        "max_grad_norm",
        "train_val_split_ratio_and_early_stopping",
        "feature_dtype_and_autocast",
    }
    assert all(
        entry.get("reason") for entry in training["declared_upstream_deviations"]
    )

    model = protocol["model"]
    assert int(model["feature_dim"]) == 33 * 3072 == 101376
    assert int(model["num_feature_layers"]) == 33
    assert model["disable_gate"] is False
    assert int(model["trainable_parameters"]) == 202754
    assert int(model["matched_budget_reference_trainable_parameters"]) == 5347593
    assert len(model["retained_from_swift"]) == 3
    assert "layer_axis_transformer_encoder" in model["deliberately_absent_clir_components"]
    assert model["condition_states_are_never_loaded"] is True

    assert (
        protocol["frozen_parents"]["training_manifest"]["file_sha256"]
        == TRAIN_MANIFEST_SHA256
    )
    assert int(protocol["frozen_parents"]["training_manifest"]["rows"]) == 5552
    assert protocol["frozen_parents"]["u0_comparison_scores"]["read_only_never_regenerated"]
    assert protocol["evaluation"]["primary_contrasts"] == ["u0_minus_swift_official"]
    assert protocol["evaluation"]["secondary_contrasts"] == []
    assert protocol["gpu_policy"]["never_preempt_or_compete_with_the_current_user_workload"]
    assert protocol["gpu_policy"]["launch_only_after_all_eight_are_idle"]
    assert protocol["ranking_population"]["feature_extraction_required"] is False
    assert protocol["ranking_population"]["new_rollout_required"] is False
    assert protocol["evidence_boundary"]["no_existing_clir_cell_is_retrained"]
    assert protocol["evidence_boundary"][
        "therefore_no_post_result_tuning_of_any_kind_is_permitted"
    ]


def test_protocol_loader_rejects_a_relaxed_budget(tmp_path) -> None:
    payload = _protocol()
    payload["training"]["validation_split"] = 0.2
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no validation split or early stopping"):
        load_protocol(path)

    payload = _protocol()
    payload["cells"] = ["swift_official", "swift_upstream_native"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the swift_official cell"):
        load_protocol(path)

    payload = _protocol()
    payload["training"]["run_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="run count differs"):
        load_protocol(path)

    payload = _protocol()
    payload["upstream"]["commit"] = "0" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="upstream commit differs"):
        load_protocol(path)


def test_derived_contracts_track_the_protocol_and_the_frozen_head_size() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    config = _model_config(protocol)
    model = SwiftLinearRewardModel(
        config["feature_dim"], disable_gate=config["disable_gate"]
    )
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    assert parameters == int(protocol["model"]["trainable_parameters"])
    assert tuple(model.fused_layer.weight.shape) == (2, config["feature_dim"])

    contract = _training_contract(protocol, 42)
    assert contract["seed"] == 42
    assert contract["epochs"] == 3
    assert contract["validation_split"] is None
    assert contract["early_stopping"] is None
    assert contract["loss"] == "BCEWithLogitsLoss"


def test_matched_budget_reference_config_is_the_hash_bound_u0_cell() -> None:
    protocol = _protocol()
    spec = protocol["frozen_parents"]["matched_budget_reference_config"]
    path = ROOT / spec["path"]
    if not path.exists():  # pragma: no cover - config tree is optional in checkouts.
        pytest.skip(f"reference config is absent: {path}")
    assert file_sha256(path) == spec["file_sha256"]
    reference = json.loads(path.read_text(encoding="utf-8"))["training"]
    training = protocol["training"]
    for field in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "amp_dtype",
        "num_workers",
        "pin_memory",
    ):
        assert reference[field] == training[field], field
    # The only intentional data-pipeline difference: semantic grouping exists for
    # CLIR's consistency loss, which this baseline does not have.
    assert reference["group_by_semantic_id"] is True
    assert training["group_by_semantic_id"] is False
    assert protocol["training"]["declared_reference_cell_deviations"] == [
        {
            "field": "group_by_semantic_id",
            "this_run": False,
            "u0_reference": True,
            "reason": "semantic grouping exists only to build the consistency-loss batches; this baseline has no consistency loss, so grouping would add a data-order difference with no matching objective",
        }
    ]


def test_holm_is_the_identity_for_a_single_hypothesis_family() -> None:
    assert _holm({"u0_minus_swift_official": 0.031}) == {
        "u0_minus_swift_official": 0.031
    }
    assert _holm({"u0_minus_swift_official": 1.0}) == {"u0_minus_swift_official": 1.0}
    # Two hypotheses must still be adjusted, so the identity is family-size bound.
    adjusted = _holm({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.04)
