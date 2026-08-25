from __future__ import annotations

from copy import deepcopy

from prepare_clir_smoke import (
    build_consistency_v5_packages,
    evaluate_consistency_v5_audit,
)
from src.clir_smoke import (
    build_mechanical_consistency_proposals,
    consistency_mechanical_metrics,
)


def _candidate(
    query_id: str,
    candidate_index: int,
    response: str,
    *,
    token_count: int,
    numeric_match: int = 1,
) -> dict:
    return {
        "id": f"{query_id}:cand:{candidate_index:03d}",
        "query_id": query_id,
        "candidate_index": candidate_index,
        "source": "math",
        "question": "Three children need juice on five days for 25 weeks.",
        "response": response,
        "output_token_ids": list(range(token_count)),
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "units": [
            {"unit_index": index, "kind": "material_claim", "text": f"u{index}"}
            for index in range(4)
        ],
        "numeric_value_match": numeric_match,
        "normalized_candidate_answer": ["375/1"],
    }


def test_mechanical_metrics_and_one_pair_per_query() -> None:
    left = (
        "Compute the weekly supply carefully. $3*5=15$. "
        "Across the school year, $15*25=375$. Therefore 375 boxes are needed."
    )
    right = (
        "Start with all children during one week. $3*5=15$. "
        "Extending that weekly amount through the year gives $15*25=375$. "
        "The requested count is 375 boxes."
    )
    metrics = consistency_mechanical_metrics(
        left, right, left_token_count=100, right_token_count=120
    )
    assert metrics["metric_version"] == "clir_consistency_mechanical_v1"
    assert metrics["token_length_ratio"] == 1.2
    assert metrics["math_trace_similarity"] == 1.0

    rows = [
        _candidate("math:q1", 0, left, token_count=100),
        _candidate("math:q1", 1, right, token_count=120),
        _candidate("math:q1", 2, right + " Extra detail.", token_count=130),
        _candidate("math:q1", 3, right, token_count=120, numeric_match=0),
    ]
    proposals, report = build_mechanical_consistency_proposals(
        rows,
        source="math",
        min_material_units=4,
        min_length_ratio=1.1,
        max_length_ratio=3.0,
        min_math_tokens=4,
        min_numeric_tokens=4,
        min_surface_bigrams=2,
        min_math_similarity=0.6,
        min_numeric_similarity=0.7,
        min_surface_jaccard=0.0,
        max_surface_jaccard=1.0,
    )
    assert len(proposals) == 1
    assert proposals[0]["query_id"] == "math:q1"
    assert report["admitted_query_distinct_pairs"] == 1
    assert proposals[0]["mechanical_metrics"]["metric_version"] == (
        "clir_consistency_mechanical_v1"
    )


def _natural_items() -> list[dict]:
    return [
        {
            "item_id": f"natural-{index:02d}",
            "query_id": f"math:q{index:02d}",
            "problem": "What is 2+3?",
            "audit_scope": "substantive_claim_validity_only",
            "left": {"id": "left", "trajectory": "2+3=5", "units": []},
            "right": {"id": "right", "trajectory": "Adding gives 5", "units": []},
        }
        for index in range(12)
    ]


def _labels(package: list[dict], private: dict, *, slot: str) -> list[dict]:
    expected_controls = {
        control["item_id"]: control["expected_annotation"]["decision"]
        for control in private["controls"]
    }
    decisions = {item_id: "accept" for item_id in private["natural_item_ids"]}
    decisions.update(expected_controls)
    if slot == "a":
        for repeat in private["self_repeats_a"]:
            decisions[repeat["repeat_item_id"]] = decisions[repeat["original_item_id"]]
    prefixes = {
        "accept": "[ACCEPT_VALID] correct",
        "reject": "[REJECT_ERROR] explicit error",
        "review": "[REVIEW] uncertain",
    }
    return [
        {
            "item_id": item["item_id"],
            "decision": decisions[item["item_id"]],
            "confidence": "high",
            "rationale": prefixes[decisions[item["item_id"]]],
        }
        for item in package
    ]


def test_v5_packages_and_fail_closed_audit_gate() -> None:
    packages, private = build_consistency_v5_packages(_natural_items(), repeat_count=3)
    assert len(packages["a"]) == 19
    assert len(packages["b"]) == 16
    labels_a = _labels(packages["a"], private, slot="a")
    labels_b = _labels(packages["b"], private, slot="b")
    gates = {
        "natural_decision_agreement_min_count": 11,
        "review_count_max_per_annotator": 1,
        "auto_agree_accept_min_count": 8,
    }
    report = evaluate_consistency_v5_audit(
        labels_a=labels_a,
        labels_b=labels_b,
        private=private,
        gates=gates,
    )
    assert report["status"] == "PASS_FRESH_MECHANICAL_AUDIT"
    assert report["scaled_protocol_allowed"] is True
    assert report["eligible_for_training"] is False

    bad_labels_b = deepcopy(labels_b)
    control_id = private["controls"][0]["item_id"]
    for label in bad_labels_b:
        if label["item_id"] == control_id:
            label["decision"] = "reject"
            label["rationale"] = "[REJECT_ERROR] wrong scope"
    failed = evaluate_consistency_v5_audit(
        labels_a=labels_a,
        labels_b=bad_labels_b,
        private=private,
        gates=gates,
    )
    assert failed["status"] == "STOP_FRESH_MECHANICAL_AUDIT_FAILURE"
    assert "hidden_controls_b" in failed["failed_gate_names"]
