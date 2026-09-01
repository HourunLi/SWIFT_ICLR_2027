import json
from pathlib import Path

import numpy as np

from prepare_clir_three_module import (
    build_parser,
    load_training_authorization,
    verify_factorial_configs,
)
from evaluate_clir_three_module_factorial import (
    factorial_effects,
    h_metrics,
    prior_metrics,
)
from evaluate_clir_mechanisms import average_precision
from score_clir_factorial import _add_global_selections
from src.clir_three_module import build_unified_data


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/three_module_expansion_v1/protocol.json"
AUTHORIZATION = ROOT / "configs/three_module_expansion_v1/training_authorization.json"


def _row(row_id: str, query_id: str, **extra: object) -> dict:
    return {
        "id": row_id,
        "query_id": query_id,
        "candidate_index": 0,
        "correctness": 1,
        "prompt_token_ids": [1],
        "output_token_ids": [2, 3],
        "hidden_states_path": f"features/{row_id}.pt",
        "condition_states_path": f"features/{query_id}.pt",
        **extra,
    }


def _prior(row: dict) -> dict:
    return {
        **row,
        "key_prior_target": [1, 0],
        "complete_prior_target": [1, 1],
    }


def test_three_module_configs_form_complete_factorial() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    observed = verify_factorial_configs(protocol)
    assert set(observed) == {"u0", "c", "h", "p", "ch", "cp", "hp", "full"}
    assert observed["u0"]["factors"] == [0, 0, 0]
    assert observed["full"]["factors"] == [1, 1, 1]


def test_three_module_parser_exposes_frozen_training_gates() -> None:
    args = build_parser().parse_args(["materialize"])
    assert args.command == "materialize"
    assert args.protocol.endswith("configs/three_module_expansion_v1/protocol.json")
    preflight = build_parser().parse_args(["preflight", "--device", "cpu"])
    assert preflight.command == "preflight"
    assert preflight.authorization.endswith(
        "configs/three_module_expansion_v1/training_authorization.json"
    )


def test_three_module_training_authorization_binds_complete_grid() -> None:
    authorization = load_training_authorization(AUTHORIZATION)
    assert authorization["cell_order"] == [
        "u0",
        "c",
        "h",
        "p",
        "ch",
        "cp",
        "hp",
        "full",
    ]
    assert authorization["training"]["runs"] == 24
    assert authorization["cells"]["full"]["factors"] == [1, 1, 1]


def test_unified_merge_enriches_shared_prior_and_removes_cross_task_dev() -> None:
    shared0 = _row("base-0", "q-base-0")
    shared1 = _row("base-1", "q-base-1")
    consistency = [
        _row(
            "c-0",
            "q-c",
            semantic_id="relation-0",
            consistency_supervision=True,
        ),
        _row(
            "c-1",
            "q-c",
            semantic_id="relation-0",
            consistency_supervision=True,
        ),
    ]
    h_rows = [
        _row(
            "h-positive",
            "q-h-positive",
            path_hallucinated=1,
            hallucination_onset=1,
        ),
        _row(
            "h-clean",
            "q-h-clean",
            path_hallucinated=0,
            hallucination_onset=-1,
        ),
    ]
    h_train = [shared0, shared1, *consistency, *h_rows]
    prior_train = [
        _prior(shared0),
        shared1,
        _prior(_row("prior-new", "q-prior-new")),
    ]
    h_dev = [
        _row("hdev-keep", "q-hdev-keep", path_hallucinated=0, hallucination_onset=-1),
        _row("hdev-drop", "q-prior-new", path_hallucinated=0, hallucination_onset=-1),
    ]
    prior_dev = [
        _prior(_row("pdev-keep", "q-pdev-keep")),
        _prior(_row("pdev-drop", "q-h-positive")),
    ]
    expected = {
        "shared_historical_rows": 2,
        "legacy_prior_rows": 1,
        "new_prior_rows": 1,
        "train_rows": 7,
        "train_queries": 6,
        "consistency_endpoint_rows": 2,
        "consistency_relations": 1,
        "h_rows": 2,
        "h_positive_rows": 1,
        "h_clean_rows": 1,
        "prior_rows": 2,
        "clean_h_dev_rows": 1,
        "clean_prior_dev_rows": 1,
    }
    result = build_unified_data(
        consistency_h0_train=h_train,
        prior_train=prior_train,
        h_dev=h_dev,
        prior_dev=prior_dev,
        consistency_h0_parent=Path("/source/h"),
        prior_parent=Path("/source/p"),
        h_dev_parent=Path("/source/h"),
        prior_dev_parent=Path("/source/p"),
        target_parent=Path("/target/data"),
        expected=expected,
    )
    assert [row["id"] for row in result["train"]] == [
        "base-0",
        "base-1",
        "c-0",
        "c-1",
        "h-positive",
        "h-clean",
        "prior-new",
    ]
    assert result["train"][0]["key_prior_target"] == [1, 0]
    assert result["train"][0]["prior_merge_origin"] == ("legacy_shared_historical_row")
    assert [row["id"] for row in result["h_dev"]] == ["hdev-keep"]
    assert [row["id"] for row in result["prior_dev"]] == ["pdev-keep"]
    assert result["report"]["removed_h_dev_queries"] == ["q-prior-new"]
    assert result["report"]["removed_prior_dev_queries"] == ["q-h-positive"]


def test_factorial_effects_use_frozen_averaged_contrasts() -> None:
    cells = {
        "u0": 0.0,
        "c": 2.0,
        "h": 3.0,
        "p": 5.0,
        "ch": 2.0 + 3.0 + 7.0,
        "cp": 2.0 + 5.0 + 11.0,
        "hp": 3.0 + 5.0 + 13.0,
        "full": 2.0 + 3.0 + 5.0 + 7.0 + 11.0 + 13.0 + 17.0,
    }
    effects = factorial_effects(cells)
    assert effects["C_main"] == 2.0 + 7.0 / 2 + 11.0 / 2 + 17.0 / 4
    assert effects["H_main"] == 3.0 + 7.0 / 2 + 13.0 / 2 + 17.0 / 4
    assert effects["P_main"] == 5.0 + 11.0 / 2 + 13.0 / 2 + 17.0 / 4
    assert effects["C_x_H"] == 7.0 + 17.0 / 2
    assert effects["C_x_P"] == 11.0 + 17.0 / 2
    assert effects["H_x_P"] == 13.0 + 17.0 / 2
    assert effects["C_x_H_x_P"] == 17.0


def test_h_metrics_accept_query_disjoint_balance_after_leakage_removal() -> None:
    rows = [
        {
            "query_id": "q-positive",
            "source": "gsm8k",
            "output_token_ids": [1, 2, 3],
            "hallucination_onset": 1,
            "path_hallucinated": 1,
            "clir_checkpoint_sha256": "checkpoint",
            "clir_hallucination_prob": [0.1, 0.8, 0.9],
            "clir_path_hallucination_prob": 0.95,
            "clir_pseudo_onset": 1,
        },
        {
            "query_id": "q-clean",
            "source": "gsm8k",
            "output_token_ids": [4, 5],
            "hallucination_onset": -1,
            "path_hallucinated": 0,
            "clir_checkpoint_sha256": "checkpoint",
            "clir_hallucination_prob": [0.1, 0.2],
            "clir_path_hallucination_prob": 0.25,
            "clir_pseudo_onset": -1,
        },
    ]
    report = h_metrics(rows)
    assert report["class_counts"] == {0: 1, 1: 1}
    assert report["onset"]["positive_exact_start_rate"] == 1.0
    assert report["onset"]["clean_no_onset_rate"] == 1.0


def test_factorial_global_selection_is_query_wide_and_stable_on_ties() -> None:
    rows = [
        {"query_id": "q0", "clir_score": 0.2},
        {"query_id": "q1", "clir_score": 0.9},
        {"query_id": "q0", "clir_score": 0.2},
        {"query_id": "q1", "clir_score": 0.1},
    ]
    _add_global_selections(rows)
    assert [row["clir_selected_best_of_n"] for row in rows] == [True, True, False, False]


def test_fast_tie_aware_average_precision_matches_threshold_definition() -> None:
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 2, size=500).tolist()
    scores = rng.integers(0, 9, size=500).astype(float).tolist()
    targets = np.asarray(labels)
    values = np.asarray(scores)
    expected = 0.0
    positives = int(targets.sum())
    for threshold in np.unique(values[targets == 1]):
        selected = values >= threshold
        tied_positives = int(((values == threshold) & (targets == 1)).sum())
        selected_positives = int(targets[selected].sum())
        expected += tied_positives * selected_positives / int(selected.sum())
    expected /= positives
    assert average_precision(labels, scores) == expected


def test_gate_scale_aware_diagnostic_uses_same_learned_prior() -> None:
    row = {
        "output_token_ids": [1, 2],
        "key_prior_target": [1, 0],
        "key_prior_mask": [1, 1],
        "complete_prior_target": [1, 0],
        "complete_prior_mask": [1, 1],
        "clir_key_prior_membership": [0.9, 0.1],
        "clir_complete_prior_membership": [0.8, 0.2],
        "correctness": 1,
        "clir_score": 1.0,
        "clir_gate_attention": [0.8, 0.2],
        "clir_key_prior": [0.9, 0.1],
        "clir_complete_prior": [0.7, 0.3],
        "clir_prior_gate_squared_l2": 0.0,
        "clir_prior_gate_alignment": 0.68,
        "clir_mean_gate": 0.5,
    }
    report = prior_metrics([row])
    assert report["gate"]["uniform_to_same_fused_prior_squared_l2_mean"] > 0
    assert report["gate"]["learned_gate_advantage_over_uniform_l2_mean"] > 0
    assert report["gate"]["learned_gate_beats_uniform_rows"] == 1
