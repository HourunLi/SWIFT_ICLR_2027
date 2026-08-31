from __future__ import annotations

import pytest

from src.clir_prior_mechanical import (
    PACKAGE_SCHEMA,
    compile_reasoning_structure,
    derive_complete_consensus,
    expand_block_indices_to_units,
    project_unit_indices_to_blocks,
    validate_local_audit_annotation,
)


def _item(units: list[str], *, question: str = "") -> dict:
    return {
        "item_id": "item",
        "question": question,
        "response": "\n".join(units),
        "units": [
            {"unit_index": index, "kind": "material_claim", "text": text}
            for index, text in enumerate(units)
        ],
    }


def test_factorial_fragments_compile_to_one_block() -> None:
    structure = compile_reasoning_structure(
        _item(["C(10, 1) = 10!", "/ (1!", "* (10-1)!)", "C(10,1)=10"])
    )
    assert structure["block_count"] == 2
    assert structure["blocks"][0]["unit_indices"] == [0, 1, 2]
    assert structure["blocks"][0]["merge_reasons"] == [
        "same_line_operator_continuation",
        "same_line_operator_continuation",
    ]
    assert project_unit_indices_to_blocks(structure, [0, 2]) == [0]
    assert expand_block_indices_to_units(structure, [0]) == [0, 1, 2]


def test_result_only_line_attaches_to_instantiated_expression() -> None:
    structure = compile_reasoning_structure(_item(["260 - 195", "= 65", "65 + 5 = 70"]))
    assert structure["blocks"][0]["unit_indices"] == [0, 1]
    assert structure["blocks"][0]["merge_reasons"] == ["result_only_continuation"]
    assert structure["blocks"][1]["unit_indices"] == [2]


def test_role_hints_do_not_delete_original_units() -> None:
    structure = compile_reasoning_structure(
        _item(
            [
                "We need to calculate the total.",
                "9 + 6 = 15",
                "The problem says 2 were returned.",
                "15 - 2 = 13",
                "So the answer is \\boxed{13}.",
            ],
            question="Nine were sold, six more were sold, and 2 were returned.",
        )
    )
    assert structure["material_unit_count"] == 5
    assert set(map(int, structure["unit_to_block"])) == set(range(5))
    roles = [block["role_hint"] for block in structure["blocks"]]
    assert roles[0] == "possible_plan_or_heading"
    assert roles[-1] == "possible_answer_wrapper"


def test_candidate_edge_uses_defined_symbol_and_value() -> None:
    structure = compile_reasoning_structure(
        _item(["total time = 8 + 24 + 6 = 38", "charge = total time * 15 = 570"])
    )
    edge = next(
        row
        for row in structure["candidate_edges"]
        if row["parent_block_id"] == 0 and row["child_block_id"] == 1
    )
    assert edge["strength"] == "high"
    assert any(
        reason.startswith("nearest_defined_symbol:total") for reason in edge["evidence"]
    )


def test_implicit_dependency_gets_one_local_predecessor_candidate() -> None:
    structure = compile_reasoning_structure(
        _item(["The subtotal is 12.", "Adding the remaining charge gives 19."])
    )
    assert structure["candidate_edges"] == [
        {
            "parent_block_id": 0,
            "child_block_id": 1,
            "strength": "medium",
            "evidence": ["nearest_substantive_predecessor"],
        }
    ]


def test_consensus_closure_maps_back_to_units_and_masks_uncertainty() -> None:
    structure = compile_reasoning_structure(
        _item(["3 * 4", "= 12", "12 + 2 = 14", "14 * 2 = 28"])
    )
    edges = {
        (row["parent_block_id"], row["child_block_id"])
        for row in structure["candidate_edges"]
    }
    assert (0, 1) in edges
    assert (1, 2) in edges
    decisions_a = {edge: "keep" for edge in edges}
    decisions_b = {edge: "keep" for edge in edges}
    decisions_b[(0, 1)] = "uncertain"
    result = derive_complete_consensus(
        structure=structure,
        final_block_id=2,
        decisions_a=decisions_a,
        decisions_b=decisions_b,
    )
    assert result["positive_block_indices"] == [1, 2]
    assert result["masked_block_indices"] == [0]
    assert result["positive_unit_indices"] == [2, 3]
    assert result["masked_unit_indices"] == [0, 1]


def test_consensus_requires_every_candidate_edge_decision() -> None:
    structure = compile_reasoning_structure(_item(["x = 3", "y = x + 2"]))
    with pytest.raises(ValueError, match="every proposed edge"):
        derive_complete_consensus(
            structure=structure,
            final_block_id=1,
            decisions_a={},
            decisions_b={},
        )


def _audit_item(raw: dict) -> dict:
    return {
        "schema_version": PACKAGE_SCHEMA,
        "item_id": raw["item_id"],
        "question": raw["question"],
        "response": raw["response"],
        "structure": compile_reasoning_structure(raw),
    }


def test_local_audit_validation_derives_complete_and_raw_unit_key() -> None:
    item = _audit_item(_item(["3 * 4", "= 12", "12 + 2 = 14", "Answer: 14"]))
    structure = item["structure"]
    decisions = [
        {
            "parent_block_id": edge["parent_block_id"],
            "child_block_id": edge["child_block_id"],
            "decision": (
                "keep"
                if (edge["parent_block_id"], edge["child_block_id"]) == (0, 1)
                else "drop"
            ),
        }
        for edge in structure["candidate_edges"]
    ]
    normalized = validate_local_audit_annotation(
        {
            "item_id": "item",
            "eligibility": "usable",
            "path_status": "supported",
            "block_roles": [
                {"block_id": 0, "role": "main_step"},
                {"block_id": 1, "role": "main_step"},
                {"block_id": 2, "role": "answer_wrapper"},
            ],
            "final_block_id": 1,
            "edge_decisions": decisions,
            "missing_edges": [],
            "key_unit_index": 2,
            "confidence": "high",
            "rationale": "the second calculation completes the answer",
        },
        item,
    )
    assert normalized["complete_block_indices"] == [0, 1]
    assert normalized["complete_unit_indices"] == [0, 1, 2]
    assert normalized["key_unit_indices"] == [2]


def test_ineligible_local_audit_must_leave_structure_empty() -> None:
    item = _audit_item(_item(["\\boxed{42}"]))
    normalized = validate_local_audit_annotation(
        {
            "item_id": "item",
            "eligibility": "no_auditable_reasoning",
            "path_status": None,
            "block_roles": [],
            "final_block_id": None,
            "edge_decisions": [],
            "missing_edges": [],
            "key_unit_index": None,
            "confidence": "high",
            "rationale": "answer only",
        },
        item,
    )
    assert normalized["complete_unit_indices"] == []
