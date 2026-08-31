"""Canonical direct-set smoke for CLIR Key/Complete priors (v10).

V10 keeps the direct-set target from v9 but removes two annotation degrees of
freedom: usable rows have one Key anchor, and Complete follows one prescribed
backward-slice policy.  The module prepares controls and evaluates blind A/B
labels; it has no provider, feature-extraction, finalization, or training path.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from src.clir_prior_partial import (
    build_blind_packages,
    evaluate_partial_prior_labels,
    target_signature,
    validate_partial_prior_annotation,
)
from src.clir_smoke import stable_priority


PROPOSAL_SCHEMA = "clir-prior-canonical-smoke-natural-v10"
PACKAGE_SCHEMA = "clir-prior-canonical-smoke-package-v10"
PRIVATE_SCHEMA = "clir-prior-canonical-smoke-private-index-v10"
LABEL_SCHEMA = "clir-prior-canonical-smoke-label-v10"
REPORT_SCHEMA = "clir-prior-canonical-smoke-gate-v10"
NAMESPACE = "clir-prior-v10"


def validate_canonical_prior_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_partial_prior_annotation(annotation, item)
    if normalized["eligibility"] == "usable" and len(
        normalized["key_unit_indices"]
    ) != 1:
        raise ValueError("Prior v10 usable annotation requires exactly one Key unit")
    normalized["schema_version"] = LABEL_SCHEMA
    return normalized


def canonical_control_items() -> list[dict[str, Any]]:
    definitions = [
        (
            "self_contained_linear",
            "Three boxes hold four apples each, and there are two loose apples. How many apples are there?",
            [
                "Three boxes times four apples gives 3×4=12 apples.",
                "Adding the two loose apples gives 12+2=14 apples.",
                "Therefore the answer is 14 apples.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "split_calculation",
            "Three boxes hold four apples each, and there are two loose apples. How many apples are there?",
            [
                "The first calculation is 3×4.",
                "Evaluating it gives 12 apples in the boxes.",
                "The final calculation is 12+2.",
                "Evaluating it gives 14 apples.",
                "Therefore the answer is 14 apples.",
            ],
            "usable",
            [3],
            [0, 1, 2, 3],
        ),
        (
            "given_restatement",
            "A shop sold 9 pens in the morning and 6 later, with 2 returned. How many stayed sold?",
            [
                "The two sales periods total 9+6=15 pens.",
                "The problem says that 2 pens were returned.",
                "Subtracting returns gives 15-2=13 pens.",
                "The answer is 13 pens.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "unused_branch",
            "A ticket costs $7 and three snacks cost $3 each. What is the total cost?",
            [
                "Three snacks cost 3×3=9 dollars.",
                "The word ticket has 6 letters.",
                "Adding the ticket gives 9+7=16 dollars.",
                "The final answer is 16 dollars.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "earliest_fatal_error",
            "Mia has 7 red and 5 blue beads, then doubles the total. How many beads does she have?",
            [
                "Adding the colors gives 7+5=13 beads.",
                "Doubling gives 13×2=26 beads.",
                "Therefore Mia has 26 beads.",
            ],
            "usable",
            [0],
            [0, 1],
        ),
        (
            "late_semantic_error",
            "Kevin travels 600 miles at 50 mph. How much faster must he travel to shorten the trip by 4 hours?",
            [
                "The original travel time is 600/50=12 hours.",
                "The shorter travel time is 12-4=8 hours.",
                "The required new speed is 600/8=75 mph.",
                "Therefore Kevin must travel 75 mph faster.",
                "The final answer is 75.",
            ],
            "usable",
            [3],
            [0, 1, 2, 3],
        ),
        (
            "duplicate_result",
            "A carton has 6 rows of 8 eggs and two loose eggs. How many eggs are there?",
            [
                "The carton contains 6×8=48 eggs.",
                "So the carton count is 48 eggs.",
                "Adding the loose eggs gives 48+2=50 eggs.",
                "Thus the answer is 50 eggs.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "answer_only",
            "What is 9 plus 5?",
            ["14"],
            "no_auditable_reasoning",
            [],
            [],
        ),
    ]
    output = []
    for name, question, texts, eligibility, key, complete in definitions:
        output.append(
            {
                "schema_version": PACKAGE_SCHEMA,
                "item_id": stable_priority(f"{NAMESPACE}-control", name),
                "question": question,
                "response": "\n".join(texts),
                "units": [
                    {"unit_index": index, "kind": "material_claim", "text": text}
                    for index, text in enumerate(texts)
                ],
                "expected_signature": (
                    eligibility,
                    tuple(key),
                    tuple(complete),
                ),
            }
        )
    return output


def build_canonical_blind_packages(
    proposals: Sequence[Mapping[str, Any]],
    *,
    repeat_count_a: int,
    repeat_count_b: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    return build_blind_packages(
        proposals,
        repeat_count_a=repeat_count_a,
        repeat_count_b=repeat_count_b,
        namespace=NAMESPACE,
        package_schema=PACKAGE_SCHEMA,
        private_schema=PRIVATE_SCHEMA,
        control_items=canonical_control_items(),
    )


def _normalized_population(
    package: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    annotation_validator: Callable[
        [Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ] = validate_canonical_prior_annotation,
) -> dict[str, dict[str, Any]]:
    package_by_id = {str(row["item_id"]): row for row in package}
    if len(package_by_id) != len(package):
        raise ValueError("package item IDs are not unique")
    output: dict[str, dict[str, Any]] = {}
    for row in labels:
        item_id = str(row.get("item_id"))
        if item_id in output:
            raise ValueError("duplicate label item_id")
        if item_id not in package_by_id:
            raise ValueError("unknown label item_id")
        output[item_id] = annotation_validator(row, package_by_id[item_id])
    if set(output) != set(package_by_id):
        raise ValueError("label population is incomplete")
    return output


def _repeat_score(
    *,
    annotator: str,
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    repeats = [
        row
        for row in private_index
        if row["annotator"] == annotator and row["kind"] == "repeat"
    ]
    passed = sum(
        target_signature(labels[str(row["item_id"])])
        == target_signature(labels[str(row["natural_item_id"])])
        for row in repeats
    )
    return {
        "passed": passed,
        "total": len(repeats),
        "rate": passed / max(1, len(repeats)),
    }


def evaluate_canonical_prior_labels(
    *,
    package_a: Sequence[Mapping[str, Any]],
    package_b: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    annotation_validator: Callable[
        [Mapping[str, Any], Mapping[str, Any]], dict[str, Any]
    ] = validate_canonical_prior_annotation,
    report_schema: str = REPORT_SCHEMA,
    pass_status: str = "PASS_PRIOR_CANONICAL_SMOKE_V10",
    yield_only_status: str = "STOP_PRIOR_CANONICAL_SMOKE_V10_YIELD_ONLY",
    definition_failure_status: str = (
        "STOP_PRIOR_CANONICAL_SMOKE_V10_DEFINITION_FAILURE"
    ),
    claim_boundary: str = "canonical dual-AI Prior target operability only; not Gold, factual accuracy, learnability, gate efficacy, or Best-of-N evidence",
) -> dict[str, Any]:
    normalized_a = _normalized_population(
        package_a, labels_a, annotation_validator=annotation_validator
    )
    normalized_b = _normalized_population(
        package_b, labels_b, annotation_validator=annotation_validator
    )
    report = evaluate_partial_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private_index,
        labels_a=labels_a,
        labels_b=labels_b,
        gates=gates,
    )

    repeats = {
        "a": _repeat_score(
            annotator="a", private_index=private_index, labels=normalized_a
        ),
        "b": _repeat_score(
            annotator="b", private_index=private_index, labels=normalized_b
        ),
    }
    report["schema_version"] = report_schema
    report["metrics"]["self_repeat"] = repeats
    report["gates"]["self_repeat_a"] = {
        "pass": repeats["a"]["rate"] >= float(gates["self_repeat_min"])
    }
    report["gates"]["self_repeat_b"] = {
        "pass": repeats["b"]["rate"] >= float(gates["self_repeat_min"])
    }

    failed = {
        name for name, result in report["gates"].items() if not result["pass"]
    }
    yield_only_gates = {
        "common_usable",
        "common_nonlow",
        "key_exact_rows",
        "complete_consensus_rows",
        "partial_paired_rows",
    }
    if not failed:
        status = pass_status
        failure_class = None
        scale_protocol_allowed = True
        oversampled_scale_protocol_allowed = True
    elif failed.issubset(yield_only_gates):
        status = yield_only_status
        failure_class = "yield_only"
        scale_protocol_allowed = False
        oversampled_scale_protocol_allowed = True
    else:
        status = definition_failure_status
        failure_class = "definition_or_stability"
        scale_protocol_allowed = False
        oversampled_scale_protocol_allowed = False

    report.update(
        {
            "status": status,
            "failure_class": failure_class,
            "scale_annotation_allowed": False,
            "scale_protocol_preparation_allowed": scale_protocol_allowed,
            "oversampled_scale_protocol_preparation_allowed": (
                oversampled_scale_protocol_allowed
            ),
            "feature_extraction_allowed": False,
            "training_allowed": False,
            "claim_boundary": claim_boundary,
        }
    )
    return report


__all__ = [
    "LABEL_SCHEMA",
    "NAMESPACE",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "REPORT_SCHEMA",
    "build_canonical_blind_packages",
    "canonical_control_items",
    "evaluate_canonical_prior_labels",
    "validate_canonical_prior_annotation",
]
