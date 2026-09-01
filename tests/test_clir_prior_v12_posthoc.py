import json
from pathlib import Path

from prepare_clir_prior_v12_posthoc import build_parser
from summarize_clir_prior_v12_posthoc import _selection, load_authorization
from src.clir_prior_v12_posthoc import (
    LABEL_NAME,
    ORIGINAL_V12_STATUS,
    construct_posthoc_rows,
    feature_inventory,
    inventory_statistics,
)


def _package(item_id: str) -> dict:
    return {
        "schema_version": "clir-prior-v12-annotation-package",
        "item_id": item_id,
        "question": "What is one plus one?",
        "response": "One plus one is two. Therefore the answer is two.",
        "units": [
            {"unit_index": 0, "kind": "material_claim", "text": "1+1=2"},
            {"unit_index": 1, "kind": "material_claim", "text": "answer=2"},
        ],
    }


def _label(item_id: str, complete=(0, 1)) -> dict:
    return {
        "item_id": item_id,
        "eligibility": "usable",
        "key_unit_indices": [1],
        "complete_unit_indices": list(complete),
        "confidence": "high",
        "rationale": "unit 0 computes the value and unit 1 states the requested result",
    }


def _proposal(item_id: str, trajectory_id: str, priority: str) -> dict:
    return {
        "schema_version": "clir-prior-v12-natural-proposal",
        "proposal_id": item_id,
        "trajectory_id": trajectory_id,
        "query_id": f"query-{item_id}",
        "cluster_id": f"cluster-{item_id}",
        "source": "gsm8k",
        "source_record_id": item_id,
        "checker_status": "numeric_match",
        "prior_label_split": "train",
        "candidate_index": 0,
        "question": "What is one plus one?",
        "response": "One plus one is two. Therefore the answer is two.",
        "material_claim_count": 2,
        "output_token_count": 4,
        "units": _package(item_id)["units"],
        "selection_priority": priority,
    }


def _materialized(item_id: str, trajectory_id: str) -> dict:
    return {
        "id": trajectory_id,
        "query_id": f"query-{item_id}",
        "cluster_id": f"cluster-{item_id}",
        "candidate_index": 0,
        "checker_status": "numeric_match",
        "correctness": 1,
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "prompt_token_ids": [10, 11],
        "output_token_ids": [20, 21, 22, 23],
        "units": [
            {
                "unit_index": 0,
                "kind": "material_claim",
                "text": "1+1=2",
                "token_start": 0,
                "token_end": 2,
            },
            {
                "unit_index": 1,
                "kind": "material_claim",
                "text": "answer=2",
                "token_start": 2,
                "token_end": 4,
            },
        ],
    }


def test_exact_consensus_excludes_an_available_failed_repeat() -> None:
    proposals = [_proposal("keep", "traj-keep", "b"), _proposal("drop", "traj-drop", "a")]
    materialized = [
        _materialized("keep", "traj-keep"),
        _materialized("drop", "traj-drop"),
    ]
    private = []
    packages = {"a": [], "b": []}
    labels = {"a": [], "b": []}
    for annotator in ("a", "b"):
        for item_id in ("keep", "drop"):
            private.append(
                {
                    "annotator": annotator,
                    "item_id": item_id,
                    "natural_item_id": item_id,
                    "kind": "natural",
                }
            )
            packages[annotator].append(_package(item_id))
            labels[annotator].append(_label(item_id))

    private.append(
        {
            "annotator": "a",
            "item_id": "drop-repeat-a",
            "natural_item_id": "drop",
            "kind": "repeat",
        }
    )
    packages["a"].append(_package("drop-repeat-a"))
    labels["a"].append(_label("drop-repeat-a", complete=(1,)))

    rows, report = construct_posthoc_rows(
        proposals=proposals,
        materialized_rows=materialized,
        private_index=private,
        packages=packages,
        labels=labels,
    )
    assert [row["proposal_id"] for row in rows] == ["keep"]
    row = rows[0]
    assert row["key_prior_target"] == [0, 0, 1, 1]
    assert row["complete_prior_target"] == [1, 1, 1, 1]
    assert row["key_prior_mask"] == [1, 1, 1, 1]
    assert row["complete_prior_mask"] == [1, 1, 1, 1]
    assert row["prior_label_name"] == LABEL_NAME
    assert row["prior_original_v12_status"] == ORIGINAL_V12_STATUS
    assert row["prior_posthoc_exploratory"] is True
    assert row["prior_human_verified"] is False
    assert report["selected_rows"] == 1
    assert report["excluded_reasons"]["available_self_repeat_failed"] == 1
    assert report["repeat_metrics"]["a"] == {"passed": 0, "total": 1}


def test_inventory_statistics_preserve_exact_feature_cost() -> None:
    row = {
        "id": "trajectory",
        "query_id": "query",
        "feature_role": "prior_dev",
        "source": "math",
        "prompt_token_count": 3,
        "output_token_count": 5,
    }
    inventory = feature_inventory([row])
    assert inventory[0]["feature_inventory_index"] == 0
    statistics = inventory_statistics(inventory)
    assert statistics == {
        "trajectory_count": 1,
        "query_count": 1,
        "condition_count": 1,
        "role_row_counts": {"prior_dev": 1},
        "source_row_counts": {"math": 1},
        "output_token_count": 5,
        "prompt_token_count": 3,
        "total_feature_token_count": 8,
    }


def test_r0_p0_configs_differ_only_in_outer_prior_weight() -> None:
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configs/data_expansion_prior_v12/posthoc_v1"
    r0 = json.loads((config_root / "r0_correctness_only.json").read_text())
    p0 = json.loads((config_root / "p0_direct_prior.json").read_text())
    assert r0["model"].pop("prior_weight") == 0.0
    assert p0["model"].pop("prior_weight") == 1.0
    assert r0 == p0


def test_parser_defaults_to_posthoc_v1_protocol() -> None:
    args = build_parser().parse_args(["prepare"])
    assert args.protocol.endswith("configs/data_expansion_prior_v12/posthoc_v1/protocol.json")
    assert args.output_root.endswith("run_artifacts/data_expansion_prior_v12/posthoc_v1")


def test_training_authorization_freezes_legacy_plus_new_prior_supervision() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "configs/data_expansion_prior_v12/posthoc_v1/training_authorization.json"
    )
    authorization = json.loads(path.read_text())
    assert authorization["status"] == (
        "AUTHORIZED_POSTHOC_EXPLORATORY_R0_P0_THREE_SEED_TRAINING"
    )
    supervision = authorization["supervision_contract"]
    assert supervision["legacy_prior_rows"] == 48
    assert supervision["new_v12_posthoc_prior_rows"] == 202
    assert supervision["total_prior_rows_seen_by_p0"] == 250
    assert supervision["legacy_prior_target_tokens"] == 14307
    assert supervision["new_prior_target_tokens"] == 63298
    assert supervision["total_prior_target_tokens"] == 77605
    assert authorization["next_gate"]["mutual_gate_or_full_unlocked"] is False


def test_ranking_authorization_freezes_paired_primary_before_results() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "configs/data_expansion_prior_v12/posthoc_v1/"
        "ranking_evaluation_authorization.json"
    )
    authorization = json.loads(path.read_text())
    assert authorization["status"] == (
        "AUTHORIZED_FROZEN_EXPLORATORY_R0_P0_RANKING_EVALUATION"
    )
    assert authorization["frozen_before_scored_outputs_completed"] is True
    assert authorization["grid"] == {
        "cells": ["r0", "p0"],
        "seeds": [42, 43, 44],
        "same_candidate_population_required": True,
        "all_six_runs_required": True,
    }
    metrics = authorization["metrics"]
    assert metrics["primary"] == (
        "paired_query_level_p0_minus_r0_Best_of_N_accuracy_at_K_16"
    )
    assert metrics["k"] == [1, 2, 4, 8, 16]
    assert metrics["bootstrap_replicates"] == 10_000
    assert authorization["decision_and_claim_rules"][
        "mutual_gate_or_full_remain_locked_by_this_authorization"
    ] is True
    assert load_authorization(path) == authorization


def test_posthoc_ranking_selection_uses_stable_frozen_prefixes() -> None:
    rows = []
    for query in ("q0", "q1"):
        for index in range(16):
            score = float(index)
            if query == "q1" and index in (14, 15):
                score = 20.0
            rows.append(
                {
                    "id": f"{query}-{index}",
                    "query_id": query,
                    "candidate_index": index,
                    "correctness": int(index % 2 == 0),
                    "clir_score": score,
                    "clir_selected_best_of_n": index == (15 if query == "q0" else 14),
                }
            )
    queries, labels, indices = _selection(rows, [1, 2, 4, 8, 16])
    assert queries == ["q0", "q1"]
    assert indices[16].tolist() == [15, 14]
    assert labels[16].tolist() == [0.0, 1.0]
