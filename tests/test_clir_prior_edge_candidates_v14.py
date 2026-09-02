from __future__ import annotations

from collections import Counter

import pytest

from src.clir_prior_edge_candidates_v14 import (
    EDGE_PROPOSAL_SCHEMA,
    propose_dependency_edges_v14,
)
from src.clir_prior_mechanical import compile_reasoning_structure


def _item(units: list[str], *, question: str = "") -> dict:
    raw = {
        "item_id": "item",
        "question": question,
        "response": "\n".join(units),
        "units": [
            {"unit_index": index, "kind": "material_claim", "text": text}
            for index, text in enumerate(units)
        ],
    }
    return {
        "item_id": raw["item_id"],
        "question": question,
        "response": raw["response"],
        "structure": compile_reasoning_structure(raw),
    }


def _pairs(item: dict) -> set[tuple[int, int]]:
    return {
        (int(edge["parent_block_id"]), int(edge["child_block_id"]))
        for edge in propose_dependency_edges_v14(item)
    }


def test_multi_operand_subtraction_receives_both_numeric_producers() -> None:
    item = _item(
        [
            "Current daily revenue = $375,000",
            "Potential daily revenue = $450,000",
            "Lost revenue = $450,000 - $375,000 = $75,000",
        ]
    )
    pairs = _pairs(item)
    assert (0, 2) in pairs
    assert (1, 2) in pairs


def test_simple_latex_fraction_definition_is_preferred_as_a_candidate() -> None:
    item = _item(
        [
            r"$b=\frac{2}{9}$",
            r"$2a=3(\frac{2}{9})-\frac{1}{3}$",
            r"$z=12(\frac{2}{9})$",
        ]
    )
    assert (0, 2) in _pairs(item)


def test_plan_line_does_not_hide_previous_calculation() -> None:
    item = _item(["x = 5", "Calculate y.", "y = x + 1 = 6"])
    assert (0, 2) in _pairs(item)


def test_variable_rewrite_keeps_previous_equation_visible() -> None:
    item = _item(
        ["5x = 12x - 4", "Move all x terms to one side.", "-7x = -4"]
    )
    assert (0, 2) in _pairs(item)


def test_candidates_are_deterministic_bounded_and_forward() -> None:
    item = _item([f"v{i} = {i} + 1 = {i + 1}" for i in range(12)])
    first = propose_dependency_edges_v14(item)
    second = propose_dependency_edges_v14(item)
    assert first == second
    assert all(edge["schema_version"] == EDGE_PROPOSAL_SCHEMA for edge in first)
    assert all(edge["parent_block_id"] < edge["child_block_id"] for edge in first)
    per_child = Counter(int(edge["child_block_id"]) for edge in first)
    assert per_child
    assert max(per_child.values()) <= 6


def test_parent_limit_contract_is_fail_closed() -> None:
    item = _item(["x = 1", "y = x + 1"])
    with pytest.raises(ValueError, match="1 <= min <= max"):
        propose_dependency_edges_v14(item, min_parents=4, max_parents=3)
    with pytest.raises(ValueError, match="unsafe audit burden"):
        propose_dependency_edges_v14(item, max_parents=9)
