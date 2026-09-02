from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.clir_prior_role_v15 import (
    build_blind_shards_v15,
    build_hidden_controls_v15,
    evaluate_blind_labels_v15,
    public_package_item_v15,
    validate_role_audit_annotation,
)


def _raw(item_id: str = "role-v15-toy") -> dict:
    texts = ["Let x = 4 + 3.", "x = 7.", "2x = 14.", "The answer is 14."]
    return {
        "item_id": item_id,
        "question": "A value is 4. Add 3 and double the result.",
        "response": "\n".join(texts),
        "units": [
            {"unit_index": 2 * i, "kind": "material_claim", "text": text}
            for i, text in enumerate(texts)
        ],
    }


def _label(item: dict, *, path_status: str = "supported") -> dict:
    return {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": path_status,
        "block_roles": [
            {"block_id": 0, "role": "main_step"},
            {"block_id": 1, "role": "main_step"},
            {"block_id": 2, "role": "main_step"},
            {"block_id": 3, "role": "answer_wrapper"},
        ],
        "final_block_id": 2,
        "confidence": "high",
        "rationale": "the introduced variable and its transformations produce 14",
    }


def test_role_only_derives_structural_key_and_complete_for_flawed_path() -> None:
    item = public_package_item_v15(_raw())
    supported = validate_role_audit_annotation(_label(item), item)
    flawed = validate_role_audit_annotation(_label(item, path_status="flawed"), item)
    assert supported["key_unit_indices"] == flawed["key_unit_indices"] == [4]
    assert supported["complete_unit_indices"] == [0, 2, 4]
    assert supported["key_unit_indices"] != [0]
    assert "candidate_edges" not in item["structure"]


def test_role_only_rejects_non_main_final_and_extra_fields() -> None:
    item = public_package_item_v15(_raw())
    bad = _label(item)
    bad["final_block_id"] = 3
    with pytest.raises(ValueError, match="final block must have role main_step"):
        validate_role_audit_annotation(bad, item)
    extra = _label(item)
    extra["key_unit_index"] = 4
    with pytest.raises(ValueError, match="strict schema"):
        validate_role_audit_annotation(extra, item)


def test_v15_controls_cover_structural_flaw_separation() -> None:
    controls = build_hidden_controls_v15("a")
    assert len(controls) == 8
    by_name = {name: expected for _, expected, name in controls}
    assert by_name["introduced_variable_is_main"]["complete_block_indices"] == [
        0,
        1,
        2,
    ]
    flawed = by_name["flawed_path_structural_key"]
    assert flawed["path_status"] == "flawed"
    assert flawed["key_block_indices"] == [1]
    assert flawed["key_unit_indices"] == [2]
    assert flawed["complete_unit_indices"] == [0, 2]


def test_v15_blind_shards_and_evaluator_pass_perfect_fixture() -> None:
    proposals = []
    for index in range(48):
        row = _raw(f"prior-v15-natural-{index:02d}")
        row.update(
            {
                "query_id": f"q-{index}",
                "cluster_id": f"c-{index}",
                "source_row_id": f"source-{index}",
            }
        )
        proposals.append(row)
    packages, private, construction = build_blind_shards_v15(proposals)
    assert construction["dependency_edges_present"] is False
    assert all(len(shard) == 18 for side in packages.values() for shard in side)

    labels = {"a": [], "b": []}
    private_by_id = {(row["annotator"], row["item_id"]): row for row in private}
    for annotator in ("a", "b"):
        for shard in packages[annotator]:
            for item in shard:
                info = private_by_id[(annotator, item["item_id"])]
                if info["kind"] == "control":
                    expected = info["expected_label"]
                    labels[annotator].append(
                        {
                            key: deepcopy(expected[key])
                            for key in (
                                "item_id",
                                "eligibility",
                                "path_status",
                                "block_roles",
                                "final_block_id",
                                "confidence",
                                "rationale",
                            )
                        }
                    )
                else:
                    labels[annotator].append(_label(item))

    report = evaluate_blind_labels_v15(
        packages={side: [row for shard in packages[side] for row in shard] for side in ("a", "b")},
        private_index=private,
        labels=labels,
        gates={
            "controls_min_pass": 8,
            "self_repeat_target_exact_min": 0.9375,
            "common_usable_nonlow_min": 40,
            "path_exact_min": 0.90,
            "final_block_exact_min": 0.90,
            "key_macro_f1_min": 0.90,
            "complete_macro_f1_min": 0.90,
            "complete_macro_iou_min": 0.80,
            "complete_mask_coverage_min": 0.90,
            "role_decision_agreement_min": 0.85,
            "all_material_union_rate_max": 0.25,
        },
    )
    assert report["status"] == "PASS_PRIOR_V15_ROLE_ONLY_SMOKE"
    assert report["cross_annotator_natural"]["key_exact_rate"] == 1.0
    assert report["trainable_labels_published"] is False


def test_v15_protocol_and_prompt_freeze_role_only_target() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/data_expansion_prior_v15/protocol.json").read_text()
    )
    assert protocol["status"] == "FROZEN_BEFORE_ANY_V15_LABEL"
    assert protocol["target"]["ai_does_not_output"] == [
        "dependency_edges",
        "key",
        "complete",
    ]
    assert protocol["target"]["hallucination_h0_exclusively_owns_first_error_localization"]
    assert protocol["gates"]["controls_min_pass"] == 8
    assert protocol["claim_boundary"]["smoke_rows_trainable"] is False
    prompt = (root / "configs/data_expansion_prior_v15/annotation_prompt.md").read_text()
    assert "不要输出 `dependency_edges`" in prompt
    assert "最早错误完全属于 Hallucination" in prompt
