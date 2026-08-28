from __future__ import annotations

from copy import deepcopy

from src.clir_scale_post_annotation import build_scale_post_annotation_plan


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
                    "response": response,
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
