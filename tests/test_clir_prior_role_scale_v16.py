from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.clir_prior_role_scale_v16 import (
    PACKAGE_SCHEMA,
    build_blind_shards_v16,
    build_hidden_controls_v16,
    construct_silver_rows_v16,
    evaluate_role_scale_v16,
    select_scale_rows_v16,
)


def _proposal(index: int) -> dict:
    texts = ["Let x = 4 + 3.", "x = 7.", "2x = 14.", "The answer is 14."]
    return {
        "schema_version": "clir-prior-role-only-scale-proposal-v16",
        "item_id": f"prior-v16-natural-{index:04d}",
        "source_row_id": f"trajectory-{index:04d}",
        "query_id": f"query-{index:04d}",
        "cluster_id": f"cluster-{index:04d}",
        "source": "gsm8k",
        "checker_status": "numeric_match",
        "prior_label_split": "train",
        "candidate_index": 0,
        "source_record_id": index,
        "material_claim_count": 4,
        "selection_priority": f"{index:064x}",
        "question": "A value is 4. Add 3 and double the result.",
        "response": "\n".join(texts),
        "units": [
            {"unit_index": 2 * i, "kind": "material_claim", "text": text}
            for i, text in enumerate(texts)
        ],
    }


def _natural_label(item: dict) -> dict:
    assert item["structure"]["block_count"] == 4
    return {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": "supported",
        "block_roles": [
            {"block_id": 0, "role": "main_step"},
            {"block_id": 1, "role": "main_step"},
            {"block_id": 2, "role": "main_step"},
            {"block_id": 3, "role": "answer_wrapper"},
        ],
        "final_block_id": 2,
        "confidence": "high",
        "rationale": "the first three blocks form the answer path",
    }


def _fixture():
    proposals = [_proposal(index) for index in range(600)]
    packages, private, construction = build_blind_shards_v16(proposals)
    private_map = {(row["annotator"], row["item_id"]): row for row in private}
    labels = {"a": [], "b": []}
    fields = (
        "item_id",
        "eligibility",
        "path_status",
        "block_roles",
        "final_block_id",
        "confidence",
        "rationale",
    )
    for annotator in ("a", "b"):
        for shard in packages[annotator]:
            for item in shard:
                hidden = private_map[(annotator, item["item_id"])]
                if hidden["kind"] == "control":
                    labels[annotator].append(
                        {key: deepcopy(hidden["expected_label"][key]) for key in fields}
                    )
                else:
                    labels[annotator].append(_natural_label(item))
    flat_packages = {
        side: [row for shard in packages[side] for row in shard]
        for side in ("a", "b")
    }
    return proposals, flat_packages, private, labels, construction


def _gates() -> dict:
    return {
        "controls_min_pass": 11,
        "self_repeat_target_exact_min": 0.95,
        "eligibility_exact_min": 0.95,
        "common_usable_nonlow_min": 550,
        "path_exact_min": 0.90,
        "final_block_exact_min": 0.95,
        "role_decision_agreement_min": 0.85,
        "selected_complete_iou_min": 0.80,
        "selected_complete_mask_coverage_min": 0.90,
        "selected_all_material_union_rate_max": 0.25,
    }


def test_v16_selects_fresh_query_and_cluster_distinct_rows() -> None:
    rows = []
    for index in range(4):
        proposal = _proposal(index)
        rows.append(
            {
                "id": proposal["source_row_id"],
                "query_id": proposal["query_id"],
                "cluster_id": proposal["cluster_id"],
                "source": "gsm8k",
                "checker_status": "numeric_match",
                "prior_label_split": "train",
                "eligible_for_supervision": True,
                "unitization_status": "ok",
                "status": "ok",
                "finish_reason": "stop",
                "material_claim_count": 4,
                "candidate_index": 0,
                "source_record_id": index,
                "question": proposal["question"],
                "response": proposal["response"],
                "units": proposal["units"],
            }
        )
    selected, report = select_scale_rows_v16(
        rows,
        excluded_query_ids={"query-0000"},
        excluded_cluster_ids={"cluster-0001"},
        strata=[
            {
                "source": "gsm8k",
                "checker_status": "numeric_match",
                "split": "train",
                "count": 2,
            }
        ],
        minimum_material_claims=4,
    )
    assert len(selected) == 2
    assert report["distinct_queries"] == report["distinct_clusters"] == 2
    assert not {"query-0000", "query-0001"} & {
        row["query_id"] for row in selected
    }


def test_v16_controls_packages_and_perfect_scale_gate() -> None:
    controls = build_hidden_controls_v16("a")
    assert len(controls) == 12
    assert controls[-1][1]["path_status"] == "flawed"
    proposals, packages, private, labels, construction = _fixture()
    assert construction["rows_per_shard"] == 56
    assert construction["repeats_per_annotator"] == 60
    assert all(row["schema_version"] == PACKAGE_SCHEMA for row in packages["a"])
    report = evaluate_role_scale_v16(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        final_strata=[
            {
                "source": "gsm8k",
                "checker_status": "numeric_match",
                "split": "train",
                "count": 500,
            }
        ],
        gates=_gates(),
    )
    assert report["status"] == "PASS_PRIOR_V16_ROLE_ONLY_SCALE"
    assert report["prospective_frozen_selection"]["selected_rows"] == 500
    assert report["cross_annotator_natural"]["final_block_exact_rate"] == 1.0


def test_v16_materialization_masks_only_main_nonmain_disagreement() -> None:
    proposals, packages, private, labels, _ = _fixture()
    repeat_parents = {
        row["natural_item_id"] for row in private if row["kind"] == "repeat"
    }
    ambiguous_id = next(
        row["item_id"] for row in proposals if row["item_id"] not in repeat_parents
    )
    for row in labels["b"]:
        if row["item_id"] == ambiguous_id:
            row["block_roles"][0]["role"] = "unused_branch"
            row["rationale"] = "block zero is treated as unused in this fixture"
    report = evaluate_role_scale_v16(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        final_strata=[
            {
                "source": "gsm8k",
                "checker_status": "numeric_match",
                "split": "train",
                "count": 600,
            }
        ],
        gates={**_gates(), "selected_all_material_union_rate_max": 1.0},
    )
    assert report["status"] == "PASS_PRIOR_V16_ROLE_ONLY_SCALE"
    materialized = []
    for index, proposal in enumerate(proposals):
        materialized.append(
            {
                "id": proposal["source_row_id"],
                "query_id": proposal["query_id"],
                "cluster_id": proposal["cluster_id"],
                "checker_status": proposal["checker_status"],
                "candidate_index": 0,
                "correctness": 1,
                "prompt_token_ids": [100 + index],
                "output_token_ids": list(range(8)),
                "units": [
                    {
                        **unit,
                        "token_start": 2 * unit_index,
                        "token_end": 2 * unit_index + 2,
                    }
                    for unit_index, unit in enumerate(proposal["units"])
                ],
            }
        )
    rows, materialization = construct_silver_rows_v16(
        proposals=proposals,
        materialized_rows=materialized,
        packages=packages,
        private_index=private,
        labels=labels,
        evaluation_report=report,
    )
    target = next(row for row in rows if row["proposal_id"] == ambiguous_id)
    assert target["complete_ambiguous_unit_indices"] == [0]
    assert target["complete_prior_mask"][:2] == [0, 0]
    assert target["complete_prior_mask"][2:] == [1] * 6
    assert target["key_prior_mask"] == [1] * 8
    assert materialization["selected_rows"] == 600


def test_v16_protocol_freezes_role_only_scale_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/data_expansion_prior_v16/protocol.json").read_text()
    )
    assert protocol["status"] == "FROZEN_BEFORE_ANY_V16_LABEL"
    assert protocol["proposal_pool"]["natural_count"] == 600
    assert protocol["prospective_selection"]["final_target_rows"] == 500
    assert protocol["target"]["ai_does_not_output"] == [
        "dependency_edges",
        "key",
        "complete",
    ]
    assert protocol["claim_boundary"]["human_verified"] is False
    prompt = (root / "configs/data_expansion_prior_v16/annotation_prompt.md").read_text()
    assert "会被程序遮住" in prompt
    assert "最早错误完全属于 Hallucination" in prompt
