"""Fresh Prior-v14 smoke using recall-first mechanical dependency candidates.

This module deliberately leaves the frozen v13 compiler, packages, labels, and
terminal report untouched.  It selects fresh query/template-cluster-disjoint
rows, reuses only v13's deterministic block projection, replaces the old top-2
edge proposals with the prospectively frozen v14 recall-first proposer, and
builds new blind controls/repeats for two independent max-reasoning annotators.

The smoke can establish annotation operability only.  It never publishes
training labels or changes any v12/v13 decision.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_edge_candidates_v14 import (
    DEFAULT_MAX_PARENTS,
    DEFAULT_MIN_PARENTS,
    FROZEN_EDGE_PROPOSAL_SCHEMA,
    propose_dependency_edges_v14,
)
from src.clir_prior_mechanical import (
    local_audit_target_signature,
    validate_local_audit_annotation,
)
from src.clir_prior_mechanical_smoke import (
    evaluate_blind_labels,
    public_package_item as public_package_item_v13,
    select_fresh_natural_rows,
)
from src.clir_smoke import stable_priority


PROPOSAL_SCHEMA = "clir-prior-mechanical-recall-smoke-proposal-v14"
PACKAGE_SCHEMA = "clir-prior-mechanical-local-audit-package-v14"
PRIVATE_SCHEMA = "clir-prior-mechanical-recall-smoke-private-index-v14"
CONTROL_SCHEMA = "clir-prior-mechanical-recall-control-v14"
EVALUATION_SCHEMA = "clir-prior-mechanical-recall-smoke-evaluation-v14"
DEFAULT_NAMESPACE = "clir-prior-mechanical-recall-smoke-v14"


def select_fresh_natural_rows_v14(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: Iterable[str],
    excluded_cluster_ids: Iterable[str],
    strata: Sequence[Mapping[str, Any]],
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select fresh rows with v13's frozen field-only policy and new IDs."""

    selected, report = select_fresh_natural_rows(
        rows,
        excluded_query_ids=excluded_query_ids,
        excluded_cluster_ids=excluded_cluster_ids,
        strata=strata,
        namespace=namespace,
    )
    output: list[dict[str, Any]] = []
    for row in selected:
        item = dict(row)
        item["schema_version"] = PROPOSAL_SCHEMA
        item["item_id"] = (
            "prior-v14-natural-"
            + stable_priority(f"{namespace}:natural-id", item["source_row_id"])[
                :20
            ]
        )
        output.append(item)
    if len({row["item_id"] for row in output}) != len(output):
        raise AssertionError("v14 natural item IDs are not unique")
    report = {
        **report,
        "proposal_schema": PROPOSAL_SCHEMA,
        "selection_policy": "v13_field_only_policy_with_v14_namespace",
    }
    return output, report


def public_package_item_v14(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    """Compile one public item with v13 blocks and frozen v14 edge proposals."""

    item = public_package_item_v13(source, item_id=item_id)
    edges = propose_dependency_edges_v14(
        item,
        min_parents=DEFAULT_MIN_PARENTS,
        max_parents=DEFAULT_MAX_PARENTS,
        proposal_schema=FROZEN_EDGE_PROPOSAL_SCHEMA,
    )
    item["schema_version"] = PACKAGE_SCHEMA
    item["structure"]["candidate_edges"] = edges
    item["structure"]["candidate_edge_schema"] = FROZEN_EDGE_PROPOSAL_SCHEMA
    item["structure"]["candidate_parent_min_target"] = DEFAULT_MIN_PARENTS
    item["structure"]["candidate_parent_max"] = DEFAULT_MAX_PARENTS
    item["structure"]["claim_boundary"] = (
        "fresh v14 smoke: v13 blocks plus frozen recall-first edge candidates; "
        "every role and edge remains a blind-AI audit decision"
    )
    counts: dict[int, int] = {}
    for edge in edges:
        child = int(edge["child_block_id"])
        counts[child] = counts.get(child, 0) + 1
    if any(count > DEFAULT_MAX_PARENTS for count in counts.values()):
        raise AssertionError("v14 public candidate-parent cap drift")
    return item


def _raw_control(
    control_id: str,
    *,
    question: str,
    unit_texts: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA,
        "item_id": control_id,
        "question": question,
        "response": "\n".join(unit_texts),
        "units": [
            {
                "unit_index": 2 * index,
                "kind": "material_claim",
                "text": text,
            }
            for index, text in enumerate(unit_texts)
        ],
    }


def _expected_audit(
    item: Mapping[str, Any],
    *,
    eligibility: str = "usable",
    path_status: str | None = "supported",
    roles: Sequence[str] = (),
    final_block_id: int | None = None,
    key_unit_index: int | None = None,
    kept_edges: Iterable[tuple[int, int]] = (),
    missing_edges: Sequence[Sequence[int]] = (),
    rationale: str,
) -> dict[str, Any]:
    if eligibility != "usable":
        return validate_local_audit_annotation(
            {
                "item_id": item["item_id"],
                "eligibility": eligibility,
                "path_status": None,
                "block_roles": [],
                "final_block_id": None,
                "edge_decisions": [],
                "missing_edges": [],
                "key_unit_index": None,
                "confidence": "high",
                "rationale": rationale,
            },
            item,
        )

    structure = item["structure"]
    if len(roles) != int(structure["block_count"]):
        raise ValueError(f"{item['item_id']}: control role count drift")
    candidates = {
        (int(edge["parent_block_id"]), int(edge["child_block_id"]))
        for edge in structure["candidate_edges"]
    }
    kept = {tuple(edge) for edge in kept_edges}
    missing = {tuple(int(value) for value in edge) for edge in missing_edges}
    if not kept <= candidates:
        raise ValueError(
            f"{item['item_id']}: a required control edge was not proposed"
        )
    if candidates & missing:
        raise ValueError(f"{item['item_id']}: missing edge was already proposed")
    annotation = {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": path_status,
        "block_roles": [
            {"block_id": index, "role": role}
            for index, role in enumerate(roles)
        ],
        "final_block_id": final_block_id,
        "edge_decisions": [
            {
                "parent_block_id": int(edge["parent_block_id"]),
                "child_block_id": int(edge["child_block_id"]),
                "decision": (
                    "keep"
                    if (
                        int(edge["parent_block_id"]),
                        int(edge["child_block_id"]),
                    )
                    in kept
                    else "drop"
                ),
            }
            for edge in structure["candidate_edges"]
        ],
        "missing_edges": [list(edge) for edge in sorted(missing)],
        "key_unit_index": key_unit_index,
        "confidence": "high",
        "rationale": rationale,
    }
    return validate_local_audit_annotation(annotation, item)


def build_hidden_controls_v14(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Build eight fresh controls targeting the v14 failure modes."""

    if annotator not in {"a", "b"}:
        raise ValueError("annotator must be a or b")
    specs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    specs.append(
        (
            "three_numeric_producers",
            _raw_control(
                "",
                question=(
                    "Five packs have four red items each, three packs have six "
                    "blue items each, and two packs have seven green items each. "
                    "How many items are there?"
                ),
                unit_texts=[
                    "5 * 4 = 20 red items.",
                    "3 * 6 = 18 blue items.",
                    "2 * 7 = 14 green items.",
                    "20 + 18 + 14 = 52 items.",
                    "The answer is 52.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "main_step",
                    "main_step",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 3,
                "key_unit_index": 6,
                "kept_edges": {(0, 3), (1, 3), (2, 3)},
                "rationale": "all three calculated subtotals directly feed the total",
            },
        )
    )
    specs.append(
        (
            "comma_number_producer",
            _raw_control(
                "",
                question=(
                    "Thirty boxes contain forty parts each. After shipping 375 "
                    "parts, how many remain?"
                ),
                unit_texts=[
                    "30 * 40 = 1,200 parts.",
                    "1,200 - 375 = 825 parts.",
                    "The answer is 825 parts.",
                ],
            ),
            {
                "roles": ["main_step", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 2,
                "kept_edges": {(0, 1)},
                "rationale": "the comma-formatted subtotal directly feeds subtraction",
            },
        )
    )
    specs.append(
        (
            "latex_fraction_chain",
            _raw_control(
                "",
                question=(
                    "A tank holds 84 liters. Half is used, then 9 liters are "
                    "added. How many liters are in the tank?"
                ),
                unit_texts=[
                    "\\frac{1}{2} * 84 = 42 liters are used.",
                    "84 - 42 = 42 liters remain.",
                    "42 + 9 = 51 liters.",
                    "The answer is 51 liters.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "main_step",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 1), (1, 2)},
                "rationale": "the half-tank calculation feeds the remaining and final amounts",
            },
        )
    )
    specs.append(
        (
            "plan_is_not_a_parent",
            _raw_control(
                "",
                question="Seven trays hold eight items each and nine are added. How many?",
                unit_texts=[
                    "7 * 8 = 56 items.",
                    "Next, calculate the total after adding 9.",
                    "56 + 9 = 65 items.",
                    "The answer is 65.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "plan_or_heading",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 2)},
                "rationale": "the calculation, not the planning sentence, feeds the sum",
            },
        )
    )
    specs.append(
        (
            "variable_rewrite",
            _raw_control(
                "",
                question="Solve 3x + 4 = 19.",
                unit_texts=[
                    "3x + 4 = 19.",
                    "3x = 15.",
                    "x = 5.",
                    "The answer is 5.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "main_step",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 1), (1, 2)},
                "rationale": "each algebraic rewrite directly depends on the previous equation",
            },
        )
    )
    specs.append(
        (
            "earliest_fatal_error",
            _raw_control(
                "",
                question="Eight boxes contain seven items each, plus three extras. How many?",
                unit_texts=[
                    "8 * 7 = 54.",
                    "54 + 3 = 57.",
                    "The answer is 57.",
                ],
            ),
            {
                "path_status": "flawed",
                "roles": ["main_step", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 0,
                "kept_edges": {(0, 1)},
                "rationale": "8 times 7 is 56, so the first calculation is the fatal error",
            },
        )
    )
    specs.append(
        (
            "unused_branch",
            _raw_control(
                "",
                question="Four trays hold six items each and five are added. How many?",
                unit_texts=[
                    "4 * 6 = 24.",
                    "9 * 9 = 81.",
                    "24 + 5 = 29.",
                    "The answer is 29.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "unused_branch",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 2)},
                "rationale": "the 81 calculation is unused by the answer path",
            },
        )
    )
    specs.append(
        (
            "answer_only_empty_structure",
            _raw_control(
                "",
                question="What is nine times six?",
                unit_texts=["The answer is \\boxed{54}."],
            ),
            {
                "eligibility": "no_auditable_reasoning",
                "rationale": "the response states only an answer and has no auditable path",
            },
        )
    )

    output: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (name, raw, expected_kwargs) in enumerate(specs):
        item_id = f"prior-v14-control-{annotator}-{index:02d}"
        raw["item_id"] = item_id
        public = public_package_item_v14(raw)
        expected = _expected_audit(public, **expected_kwargs)
        output.append((public, expected, name))
    return output


def build_blind_shards_v14(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 4,
    repeats_per_shard: int = 4,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    """Build four 12-natural +2-control +4-repeat shards per annotator."""

    if shard_count != 4 or len(proposals) != 48 or repeats_per_shard != 4:
        raise ValueError("v14 smoke freezes 48 natural rows in four 18-row shards")
    public_natural = [public_package_item_v14(row) for row in proposals]
    natural_shards = [
        public_natural[index::shard_count] for index in range(shard_count)
    ]
    if any(len(rows) != 12 for rows in natural_shards):
        raise AssertionError("v14 natural shard balance drift")

    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        controls = build_hidden_controls_v14(annotator)
        shard_rows: list[list[dict[str, Any]]] = [
            list(rows) for rows in natural_shards
        ]
        for control_index, (public, expected, name) in enumerate(controls):
            destination = control_index % shard_count
            shard_rows[destination].append(public)
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "shard_index": destination,
                    "kind": "control",
                    "item_id": public["item_id"],
                    "natural_item_id": None,
                    "control_name": name,
                    "expected_label": expected,
                }
            )

        repeat_counter = 0
        for parent_shard, parent_rows in enumerate(natural_shards):
            destination = (parent_shard + 1) % shard_count
            for local_index, parent in enumerate(parent_rows[:repeats_per_shard]):
                repeat_id = (
                    f"{parent['item_id']}:repeat:{annotator}:{repeat_counter:02d}"
                )
                repeated = public_package_item_v14(parent, item_id=repeat_id)
                if (
                    repeated["structure"]["source_sha256"]
                    != parent["structure"]["source_sha256"]
                ):
                    raise AssertionError("v14 repeat source hash differs from parent")
                if repeated["structure"] != {
                    **parent["structure"],
                    "item_id": repeat_id,
                }:
                    raise AssertionError("v14 repeat structure differs from parent")
                shard_rows[destination].append(repeated)
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": destination,
                        "kind": "repeat",
                        "item_id": repeat_id,
                        "natural_item_id": parent["item_id"],
                        "control_name": None,
                        "expected_label": None,
                        "parent_shard_index": parent_shard,
                        "repeat_local_index": local_index,
                    }
                )
                repeat_counter += 1

        for shard_index, rows in enumerate(shard_rows):
            for natural in natural_shards[shard_index]:
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": shard_index,
                        "kind": "natural",
                        "item_id": natural["item_id"],
                        "natural_item_id": natural["item_id"],
                        "control_name": None,
                        "expected_label": None,
                    }
                )
            if len(rows) != 18:
                raise AssertionError("v14 public shard composition drift")
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}:package:{annotator}:{shard_index}",
                    row["item_id"],
                )
            )
        packages[annotator] = shard_rows

    private.sort(
        key=lambda row: (row["annotator"], row["shard_index"], row["item_id"])
    )
    construction = {
        "annotators": ["a", "b"],
        "shards_per_annotator": shard_count,
        "natural_rows_per_annotator": 48,
        "controls_per_annotator": 8,
        "repeats_per_annotator": 16,
        "rows_per_shard": 18,
        "rows_per_annotator": 72,
        "parent_repeat_same_shard": 0,
        "edge_candidate_schema": FROZEN_EDGE_PROPOSAL_SCHEMA,
        "candidate_parent_max": DEFAULT_MAX_PARENTS,
    }
    return packages, private, construction


def summarize_candidate_burden(
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the public natural candidate burden before annotation."""

    items = [public_package_item_v14(row) for row in proposals]
    edge_counts = [len(item["structure"]["candidate_edges"]) for item in items]
    parent_counts: list[int] = []
    for item in items:
        per_child: dict[int, int] = {}
        for edge in item["structure"]["candidate_edges"]:
            child = int(edge["child_block_id"])
            per_child[child] = per_child.get(child, 0) + 1
        parent_counts.extend(per_child.values())
    return {
        "natural_rows": len(items),
        "total_candidate_edges": sum(edge_counts),
        "min_edges_per_row": min(edge_counts) if edge_counts else 0,
        "mean_edges_per_row": (
            sum(edge_counts) / len(edge_counts) if edge_counts else 0.0
        ),
        "max_edges_per_row": max(edge_counts) if edge_counts else 0,
        "max_parents_for_any_child": max(parent_counts) if parent_counts else 0,
        "candidate_parent_cap": DEFAULT_MAX_PARENTS,
    }


def evaluate_blind_labels_v14(
    *,
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate fresh v14 labels while preserving the v13 terminal decision."""

    report = evaluate_blind_labels(
        packages=packages,
        private_index=private_index,
        labels=labels,
        gates=gates,
    )
    status_map = {
        "FAIL_PRIOR_V13_SCHEMA": "FAIL_PRIOR_V14_SCHEMA",
        "PASS_PRIOR_V13_MECHANICAL_SMOKE": (
            "PASS_PRIOR_V14_MECHANICAL_RECALL_SMOKE"
        ),
        "STOP_PRIOR_V13_MECHANICAL_SMOKE": (
            "STOP_PRIOR_V14_MECHANICAL_RECALL_SMOKE"
        ),
    }
    output = deepcopy(report)
    output["schema_version"] = EVALUATION_SCHEMA
    output["status"] = status_map.get(str(report.get("status")), report.get("status"))
    output["v13_terminal_decision_unchanged"] = True
    output["v13_bridge_labels_used_for_training"] = False
    output["fresh_v14_natural_rows"] = True
    output["edge_candidate_schema"] = FROZEN_EDGE_PROPOSAL_SCHEMA
    output["trainable_labels_published"] = False
    if str(output["status"]).startswith("PASS"):
        output["next_step"] = "freeze_separate_prior_scale_v15_before_more_annotation"
    else:
        output["next_step"] = "stop_v14_without_relabel_or_adjudication"
    return output


def target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    """Expose the unchanged v13 target signature for v14 tests/audits."""

    return local_audit_target_signature(annotation)
