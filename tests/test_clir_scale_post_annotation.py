from __future__ import annotations

from copy import deepcopy

import networkx as nx

from src.clir_scale_post_annotation import (
    build_scale_post_annotation_plan,
    build_scale_post_annotation_plan_v6_1,
    deterministic_preferred_maximum_matching,
)


def _fixture() -> tuple[list[dict], list[dict], dict, dict]:
    proposals: list[dict] = []
    rows: list[dict] = []
    specs = [
        ("train-a", "train_acquisition", "gsm8k", "11/1"),
        ("train-b", "train_acquisition", "math", "12/1"),
        ("held-a", "heldout_acquisition", "gsm8k", "21/1"),
        ("held-b", "heldout_acquisition", "gsm8k", "22/1"),
    ]
    for index, (item_id, split, source, answer) in enumerate(specs):
        query_id = f"query:{item_id}"
        cluster_id = f"cluster:{item_id}"
        left_id = f"{query_id}:left"
        right_id = f"{query_id}:right"
        proposals.append(
            {
                "proposal_id": item_id,
                "query_id": query_id,
                "cluster_id": cluster_id,
                "source": source,
                "source_subject": None,
                "source_level": None,
                "acquisition_split": split,
                "left_id": left_id,
                "right_id": right_id,
                "left_candidate_index": 0,
                "right_candidate_index": 1,
                "annotation_priority": f"{index:02d}",
            }
        )
        for side, trajectory_id in (("left", left_id), ("right", right_id)):
            if item_id == "held-a":
                response = "apples crates balance result"
            elif item_id == "held-b":
                response = "apples crates remaining result"
            else:
                response = f"unique {item_id} {side} reasoning"
            rows.append(
                {
                    "id": trajectory_id,
                    "query_id": query_id,
                    "cluster_id": cluster_id,
                    "source": source,
                    "source_subject": None,
                    "source_level": None,
                    "acquisition_split": split,
                    "normalized_candidate_answer": [answer],
                    "output_token_ids": list(range(10)),
                    "prompt_token_ids": [100, 101, 102],
                    "response": response,
                    "candidate_index": 0 if side == "left" else 1,
                    "checker_status": "numeric_match",
                    "numeric_value_match": 1,
                    "eligible_for_supervision": True,
                    "unitization_status": "ok",
                    "finish_reason": "stop",
                }
            )
    raw_gate_report = {
        "status": "PASS_SCALE_V6_RAW_ANNOTATION_GATES",
        "common_accept_item_ids": [
            "held-a",
            "train-a",
            "held-b",
            "train-b",
        ],
    }
    protocol = {
        "final_positive_selection": {
            "train_select_first_by_frozen_annotation_order": 2,
            "heldout_select_first_by_frozen_annotation_order": 2,
        },
        "heldout_hard_negatives": {
            "count": 2,
            "different_query_and_template_cluster_required": True,
            "different_normalized_final_answer_required": True,
            "view_token_length_ratio_min": 0.8,
            "view_token_length_ratio_max": 1.25,
            "surface_bigram_jaccard_min": 0.1,
            "surface_bigram_jaccard_max": 0.4,
            "matching": (
                "deterministic_greedy_by_source_stratum_length_distance_then_sha256"
            ),
        },
    }
    return proposals, rows, raw_gate_report, protocol


def _v6_1_amendment(*, count: int) -> dict:
    return {
        "hard_negative_contract": {
            "count": count,
            "source_pool": (
                "all_existing_heldout_numeric_match_supervision_eligible_views"
            ),
            "different_query_and_template_cluster_required": True,
            "different_normalized_final_answer_required": True,
            "view_token_length_ratio_min": 0.8,
            "view_token_length_ratio_max": 1.25,
            "surface_bigram_jaccard_min": 0.1,
            "surface_bigram_jaccard_max": 0.4,
            "matching": (
                "networkx_max_weight_matching_maxcardinality_then_preference_v1"
            ),
            "networkx_version": nx.__version__,
            "negative_pairs_are_evaluation_only": True,
        },
        "storage_contract": {"full_feature_bytes_per_token": 202752},
    }


def test_post_annotation_plan_selects_first_n_and_builds_negatives() -> None:
    proposals, rows, raw_gate_report, protocol = _fixture()
    artifacts, report = build_scale_post_annotation_plan(
        proposals=proposals,
        materialized_rows=rows,
        raw_gate_report=raw_gate_report,
        protocol=protocol,
    )
    assert report["status"] == "PASS_SCALE_V6_POST_ANNOTATION_PLAN"
    assert [row["relation_id"] for row in artifacts["train"]] == [
        "train-a",
        "train-b",
    ]
    assert [row["relation_id"] for row in artifacts["heldout"]] == [
        "held-a",
        "held-b",
    ]
    assert len(artifacts["negatives"]) == 2
    assert report["heldout_hard_negatives"]["selected_count"] == 2
    assert len(
        {
            endpoint
            for row in artifacts["negatives"]
            for endpoint in (row["left_id"], row["right_id"])
        }
    ) == 4


def test_post_annotation_plan_stops_when_frozen_negative_yield_is_low() -> None:
    proposals, rows, raw_gate_report, protocol = _fixture()
    failed_protocol = deepcopy(protocol)
    failed_protocol["heldout_hard_negatives"][
        "surface_bigram_jaccard_min"
    ] = 0.9
    artifacts, report = build_scale_post_annotation_plan(
        proposals=proposals,
        materialized_rows=rows,
        raw_gate_report=raw_gate_report,
        protocol=failed_protocol,
    )
    assert report["status"] == "STOP_SCALE_V6_POST_ANNOTATION_PLAN"
    assert report["publishable_relation_manifests_allowed"] is False
    assert report["feature_extraction_allowed"] is False
    assert report["heldout_hard_negatives"]["selected_count"] == 0
    assert artifacts["train"]
    assert artifacts["heldout"]
    assert artifacts["negatives"] == []


def test_preferred_matching_uses_maximum_cardinality_before_edge_preference() -> None:
    # A greedy choice of edge 0 would leave only one edge.  The frozen matcher
    # must instead take ranks 1 and 2 to reach cardinality two.
    ranks = deterministic_preferred_maximum_matching(
        node_count=4,
        ordered_edges=[(0, 1), (0, 2), (1, 3)],
        required_networkx_version=nx.__version__,
    )
    assert ranks == [1, 2]


def test_v6_1_expands_only_the_negative_pool_and_builds_exact_inventory() -> None:
    proposals, rows, raw_gate_report, protocol = _fixture()
    query_id = "query:held-c"
    for candidate_index, side in enumerate(("left", "right")):
        rows.append(
            {
                "id": f"{query_id}:{side}",
                "query_id": query_id,
                "cluster_id": "cluster:held-c",
                "source": "gsm8k",
                "source_subject": None,
                "source_level": None,
                "acquisition_split": "heldout_acquisition",
                "normalized_candidate_answer": ["23/1"],
                "output_token_ids": list(range(10)),
                "prompt_token_ids": [200, 201, 202],
                "response": "apples crates final count",
                "candidate_index": candidate_index,
                "checker_status": "numeric_match",
                "numeric_value_match": 1,
                "eligible_for_supervision": True,
                "unitization_status": "ok",
                "finish_reason": "stop",
            }
        )

    artifacts, report = build_scale_post_annotation_plan_v6_1(
        proposals=proposals,
        materialized_rows=rows,
        raw_gate_report=raw_gate_report,
        protocol=protocol,
        amendment=_v6_1_amendment(count=3),
    )
    assert report["status"] == "PASS_SCALE_V6_1_POST_ANNOTATION_PLAN"
    assert [row["relation_id"] for row in artifacts["train"]] == [
        "train-a",
        "train-b",
    ]
    assert [row["relation_id"] for row in artifacts["heldout"]] == [
        "held-a",
        "held-b",
    ]
    assert len(artifacts["negatives"]) == 3
    assert report["heldout_hard_negatives"]["endpoint_count"] == 6
    assert report["heldout_hard_negatives"]["maximum_matching_size"] == 3
    assert report["feature_extraction_allowed"] is False
    assert report["training_allowed"] is False
    storage = report["selected_feature_inventory"]
    assert storage["positive_only"]["trajectory_count"] == 8
    assert storage["positive_only"]["unique_prompt_count"] == 4
    assert storage["final_selected"]["trajectory_count"] == 10
    assert storage["final_selected"]["unique_prompt_count"] == 5
    assert len(artifacts["inventory"]) == 10
    assert sum(row["condition_feature_owner"] for row in artifacts["inventory"]) == 5


def test_v6_1_is_invariant_to_materialized_row_order() -> None:
    proposals, rows, raw_gate_report, protocol = _fixture()
    first, first_report = build_scale_post_annotation_plan_v6_1(
        proposals=proposals,
        materialized_rows=rows,
        raw_gate_report=raw_gate_report,
        protocol=protocol,
        amendment=_v6_1_amendment(count=2),
    )
    second, second_report = build_scale_post_annotation_plan_v6_1(
        proposals=proposals,
        materialized_rows=list(reversed(rows)),
        raw_gate_report=raw_gate_report,
        protocol=protocol,
        amendment=_v6_1_amendment(count=2),
    )
    assert first == second
    assert first_report == second_report
