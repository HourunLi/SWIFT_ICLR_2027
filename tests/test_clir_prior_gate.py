import json
from pathlib import Path

import pytest

from prepare_clir_prior_gate import build_parser, load_protocol, verify_config_pair
from summarize_clir_prior_gate import _prior_run_metrics, load_ranking_authorization


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/data_expansion_prior_v12/posthoc_v1"
PROTOCOL = CONFIG_ROOT / "gate_v1/protocol.json"
COMPLETION = CONFIG_ROOT / "gate_v1/completion.json"


def test_fixed_gate_configs_differ_only_in_gate_weight() -> None:
    p0 = json.loads((CONFIG_ROOT / "p0_direct_prior.json").read_text())
    pg0 = json.loads((CONFIG_ROOT / "gate_v1/pg0_fixed_025.json").read_text())
    assert p0["model"].pop("gate_prior_weight") == 0.0
    assert pg0["model"].pop("gate_prior_weight") == 0.25
    assert p0 == pg0


def test_gate_protocol_freezes_one_factor_and_defers_full() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["status"] == "AUTHORIZED_FIXED_025_P0_PG0_REPLICATION"
    assert verify_config_pair(protocol)["pg0"]["file_sha256"] == (
        protocol["cells"]["pg0"]["file_sha256"]
    )
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert protocol["training"]["epochs"] == 3
    assert protocol["next_stage"][
        "three_method_combination_user_authorized_in_principle"
    ] is True
    assert protocol["next_stage"]["must_be_separately_frozen_after_gate_result"] is True


def test_gate_preflight_parser_defaults_to_frozen_protocol() -> None:
    args = build_parser().parse_args(["preflight", "--device", "cpu"])
    assert args.protocol.endswith(
        "configs/data_expansion_prior_v12/posthoc_v1/gate_v1/protocol.json"
    )
    assert args.command == "preflight"
    assert args.device == "cpu"


def test_gate_ranking_authorization_freezes_primary_and_defers_full() -> None:
    path = CONFIG_ROOT / "gate_v1/ranking_evaluation_authorization.json"
    authorization = load_ranking_authorization(path)
    assert authorization["frozen_before_pg0_scored_outputs"] is True
    assert authorization["grid"]["cells"] == ["p0", "pg0"]
    assert authorization["grid"]["seeds"] == [42, 43, 44]
    assert authorization["metrics"]["primary"] == (
        "paired_query_level_pg0_minus_p0_Best_of_N_accuracy_at_K_16"
    )
    assert authorization["metrics"]["bootstrap_replicates"] == 10_000
    assert authorization["decision_and_claim_rules"][
        "three_module_full_requires_a_separate_post_gate_protocol"
    ] is True


def test_gate_completion_separates_mechanism_from_ranking_decision() -> None:
    completion = json.loads(COMPLETION.read_text())
    assert completion["status"] == (
        "COMPLETE_PRIOR_V12_POSTHOC_FIXED_025_GATE_EXPLORATORY_SCREEN"
    )
    assert completion["mechanism"]["gate_alignment_learned"] is True
    assert completion["mechanism"]["prior_protection_pass"] is True
    assert completion["mechanism"]["gate_collapse_guard_pass"] is True
    assert completion["ranking"]["pg0_minus_p0"]["8"] > 0.0
    assert completion["ranking"]["pg0_minus_p0"]["16"] < 0.0
    assert all(
        delta < 0.0
        for delta in completion["ranking"]["bon16_delta_by_seed"].values()
    )
    assert completion["decision"]["exploratory_fixed_025_ranking_benefit"] is False
    assert completion["decision"][
        "three_module_full_requires_separate_frozen_protocol"
    ] is True


def test_prior_run_metrics_include_alignment_and_protection_targets() -> None:
    rows = []
    for index, correctness in enumerate((0, 1)):
        rows.append(
            {
                "output_token_ids": [10, 11],
                "key_prior_target": [1, 0],
                "key_prior_mask": [1, 1],
                "complete_prior_target": [1, 1 if index else 0],
                "complete_prior_mask": [1, 1],
                "clir_key_prior_membership": [0.9, 0.1],
                "clir_complete_prior_membership": [0.8, 0.7 if index else 0.2],
                "correctness": correctness,
                "clir_score": -1.0 if not correctness else 1.0,
                "clir_gate_attention": [0.75, 0.25],
                "clir_prior_gate_squared_l2": 0.1 + index * 0.1,
                "clir_prior_gate_alignment": 0.6,
                "clir_mean_gate": 0.5,
            }
        )
    report = _prior_run_metrics(rows)
    assert report["rows"] == 2
    assert report["key"]["auroc"] == 1.0
    assert report["correctness"]["auroc"] == 1.0
    assert report["gate"]["full_trajectory_squared_l2_mean"] == pytest.approx(0.15)
    assert 0.0 < report["gate"]["attention_effective_token_fraction_mean"] <= 1.0
