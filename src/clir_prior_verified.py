"""Verification-first canonical CLIR Prior smoke (v11).

V11 preserves v10's singleton Key and canonical Complete definitions.  It adds
fresh hidden controls and a prompt-level requirement to verify every material
claim before choosing the earliest fatal error.  The module contains no model
provider, feature extraction, finalization, or training path.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.clir_prior_canonical import evaluate_canonical_prior_labels
from src.clir_prior_partial import (
    build_blind_packages,
    validate_partial_prior_annotation,
)
from src.clir_smoke import stable_priority


PROPOSAL_SCHEMA = "clir-prior-verified-smoke-natural-v11"
PACKAGE_SCHEMA = "clir-prior-verified-smoke-package-v11"
PRIVATE_SCHEMA = "clir-prior-verified-smoke-private-index-v11"
LABEL_SCHEMA = "clir-prior-verified-smoke-label-v11"
REPORT_SCHEMA = "clir-prior-verified-smoke-gate-v11"
NAMESPACE = "clir-prior-v11"


def validate_verified_prior_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = validate_partial_prior_annotation(annotation, item)
    if normalized["eligibility"] == "usable" and len(
        normalized["key_unit_indices"]
    ) != 1:
        raise ValueError("Prior v11 usable annotation requires exactly one Key unit")
    normalized["schema_version"] = LABEL_SCHEMA
    return normalized


def verification_control_items() -> list[dict[str, Any]]:
    """Return eight fresh controls that were not exposed in v10 packages."""

    definitions = [
        (
            "early_arithmetic_error",
            "Six boxes hold eight pencils each, and five loose pencils are added. How many pencils are there?",
            [
                "Six boxes contain 6×8=46 pencils.",
                "Adding the loose pencils gives 46+5=51 pencils.",
                "Therefore there are 51 pencils.",
            ],
            "usable",
            [0],
            [0, 1],
        ),
        (
            "later_arithmetic_error",
            "Seven trays hold nine cookies each, and there are four extra cookies. How many cookies are there?",
            [
                "The trays contain 7×9=63 cookies.",
                "Adding the extras gives 63+4=68 cookies.",
                "Therefore there are 68 cookies.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "late_unit_semantic_error",
            "A cyclist covers 120 miles in 3 hours. What is the speed in miles per hour?",
            [
                "Dividing distance by time gives 120/3=40 miles per hour.",
                "Therefore the cyclist travels 40 miles in total.",
                "The final answer is 40.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "fresh_split_calculation",
            "Five crates hold seven bottles each, with three loose bottles. How many bottles are there?",
            [
                "The crate calculation is 5×7.",
                "Evaluating it gives 35 bottles in crates.",
                "The total calculation is 35+3.",
                "Evaluating it gives 38 bottles.",
                "Therefore the answer is 38 bottles.",
            ],
            "usable",
            [3],
            [0, 1, 2, 3],
        ),
        (
            "fresh_given_restatement",
            "Nora read 12 pages on Monday and 8 on Tuesday, but 3 Tuesday pages were rereads. How many different pages did she read?",
            [
                "The two daily counts total 12+8=20 pages.",
                "The problem says that 3 pages were rereads.",
                "Subtracting the rereads gives 20-3=17 different pages.",
                "The answer is 17 pages.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "fresh_unused_branch",
            "One ticket costs 4 dollars and five drinks cost 2 dollars each. What is the total cost?",
            [
                "Five drinks cost 5×2=10 dollars.",
                "The word drink has five letters.",
                "Adding the ticket gives 10+4=14 dollars.",
                "The final answer is 14 dollars.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "fresh_duplicate_result",
            "A shelf has four rows of nine books and three loose books. How many books are there?",
            [
                "The rows contain 4×9=36 books.",
                "So the row count is 36 books.",
                "Adding the loose books gives 36+3=39 books.",
                "Thus the answer is 39 books.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "fresh_answer_only",
            "What is 11 plus 8?",
            ["19"],
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


def build_verified_blind_packages(
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
        control_items=verification_control_items(),
    )


def evaluate_verified_prior_labels(
    *,
    package_a: Sequence[Mapping[str, Any]],
    package_b: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    return evaluate_canonical_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private_index,
        labels_a=labels_a,
        labels_b=labels_b,
        gates=gates,
        annotation_validator=validate_verified_prior_annotation,
        report_schema=REPORT_SCHEMA,
        pass_status="PASS_PRIOR_VERIFIED_SMOKE_V11",
        yield_only_status="STOP_PRIOR_VERIFIED_SMOKE_V11_YIELD_ONLY",
        definition_failure_status="STOP_PRIOR_VERIFIED_SMOKE_V11_DEFINITION_FAILURE",
        claim_boundary="verification-first canonical dual-AI Prior target operability only; not Gold, factual accuracy, learnability, gate efficacy, or Best-of-N evidence",
    )


__all__ = [
    "LABEL_SCHEMA",
    "NAMESPACE",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "REPORT_SCHEMA",
    "build_verified_blind_packages",
    "evaluate_verified_prior_labels",
    "validate_verified_prior_annotation",
    "verification_control_items",
]
