"""Role-only structural targets for a future CLIR Prior smoke.

Prior-v14 showed that block roles and the derived Complete mask were stable,
while bounded dependency proposals still omitted many edges.  It also asked
Prior to choose the earliest fatal error on flawed paths, duplicating the
Hallucination target.  This module deliberately changes that representation:

* annotators audit one role per deterministic reasoning block;
* ``Complete`` is the union of every block labelled ``main_step``;
* ``Key`` is the final answer-producing ``main_step`` block;
* path correctness remains diagnostic and never changes structural Key.

Original unit indices remain the only bridge to exact token labels.  This
module never repairs v14 labels or publishes training data.  A fresh smoke must
freeze and pass before any scale or feature extraction is considered.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Integral
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_mechanical import (
    BLOCK_ROLE_VALUES,
    CONFIDENCE_VALUES,
    ELIGIBILITY_VALUES,
    PATH_STATUS_VALUES,
    expand_block_indices_to_units,
)
from src.clir_prior_mechanical_smoke import (
    public_package_item as public_package_item_v13,
    select_fresh_natural_rows,
)
from src.clir_smoke import stable_priority


PROPOSAL_SCHEMA = "clir-prior-role-only-smoke-proposal-v15"
PACKAGE_SCHEMA = "clir-prior-role-only-audit-package-v15"
STRUCTURE_SCHEMA = "clir-prior-role-only-structure-v15"
LABEL_SCHEMA = "clir-prior-role-only-audit-label-v15"
PRIVATE_SCHEMA = "clir-prior-role-only-smoke-private-index-v15"
CONTROL_SCHEMA = "clir-prior-role-only-control-v15"
EVALUATION_SCHEMA = "clir-prior-role-only-smoke-evaluation-v15"
DEFAULT_NAMESPACE = "clir-prior-role-only-smoke-v15"


def select_fresh_natural_rows_v15(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: Iterable[str],
    excluded_cluster_ids: Iterable[str],
    strata: Sequence[Mapping[str, Any]],
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select fresh rows with the frozen v13 eligibility and split contract."""

    selected, report = select_fresh_natural_rows(
        rows,
        excluded_query_ids=excluded_query_ids,
        excluded_cluster_ids=excluded_cluster_ids,
        strata=strata,
        namespace=namespace,
    )
    output: list[dict[str, Any]] = []
    for row in selected:
        item = deepcopy(row)
        item["schema_version"] = PROPOSAL_SCHEMA
        item["item_id"] = (
            "prior-v15-natural-"
            + stable_priority(f"{namespace}:natural-id", row["source_row_id"])[
                :20
            ]
        )
        output.append(item)
    report = dict(report)
    report["target_representation"] = "role_only_structural_key_complete"
    return output, report


def public_package_item_v15(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    """Compile a public role-audit item without dependency candidates."""

    base = public_package_item_v13(source, item_id=item_id)
    structure = deepcopy(base["structure"])
    structure["schema_version"] = STRUCTURE_SCHEMA
    structure.pop("candidate_edges", None)
    structure["target_derivation"] = {
        "complete": "all blocks annotated main_step",
        "key": "all raw units in final_block_id",
        "path_status_changes_key": False,
    }
    structure["claim_boundary"] = (
        "role hints and proposed final candidates are nonbinding; annotators "
        "audit roles and final block, while Key/Complete are programmatic"
    )
    return {
        "schema_version": PACKAGE_SCHEMA,
        "item_id": str(base["item_id"]),
        "question": str(base["question"]),
        "response": str(base["response"]),
        "units": deepcopy(base["units"]),
        "structure": structure,
    }


def validate_role_audit_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one role audit and derive structural Key/Complete targets."""

    expected_fields = {
        "item_id",
        "eligibility",
        "path_status",
        "block_roles",
        "final_block_id",
        "confidence",
        "rationale",
    }
    if set(annotation) != expected_fields:
        raise ValueError("role-audit label fields differ from the strict schema")
    if annotation.get("item_id") != item.get("item_id"):
        raise ValueError("annotation item_id does not match package item")
    structure = item.get("structure")
    if (
        not isinstance(structure, Mapping)
        or structure.get("schema_version") != STRUCTURE_SCHEMA
    ):
        raise ValueError("package lacks a supported v15 role structure")

    eligibility = annotation.get("eligibility")
    confidence = annotation.get("confidence")
    rationale = annotation.get("rationale")
    roles = annotation.get("block_roles")
    final_block = annotation.get("final_block_id")
    path_status = annotation.get("path_status")
    if eligibility not in ELIGIBILITY_VALUES:
        raise ValueError("invalid eligibility")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("invalid confidence")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty")
    if not isinstance(roles, list):
        raise ValueError("block_roles must be an array")

    if eligibility != "usable":
        if path_status is not None or final_block is not None or roles:
            raise ValueError("ineligible role audit must clear all structure fields")
        return {
            "schema_version": LABEL_SCHEMA,
            "item_id": str(annotation["item_id"]),
            "eligibility": str(eligibility),
            "path_status": None,
            "block_roles": [],
            "final_block_id": None,
            "key_block_indices": [],
            "key_unit_indices": [],
            "complete_block_indices": [],
            "complete_unit_indices": [],
            "confidence": str(confidence),
            "rationale": rationale.strip(),
        }

    if path_status not in PATH_STATUS_VALUES:
        raise ValueError("usable role audit has invalid path_status")
    block_count = int(structure["block_count"])
    normalized_roles: list[dict[str, Any]] = []
    for row in roles:
        if not isinstance(row, Mapping) or set(row) != {"block_id", "role"}:
            raise ValueError("each block role must contain only block_id and role")
        block_id = row.get("block_id")
        role = row.get("role")
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, Integral)
            or not 0 <= int(block_id) < block_count
            or role not in BLOCK_ROLE_VALUES
        ):
            raise ValueError("invalid block role entry")
        normalized_roles.append({"block_id": int(block_id), "role": str(role)})
    if [row["block_id"] for row in normalized_roles] != list(range(block_count)):
        raise ValueError("block_roles must decide every block exactly once in order")

    if isinstance(final_block, bool) or not isinstance(final_block, Integral):
        raise ValueError("usable role audit requires an integer final_block_id")
    final_integer = int(final_block)
    if not 0 <= final_integer < block_count:
        raise ValueError("final_block_id is out of range")
    role_by_id = {row["block_id"]: row["role"] for row in normalized_roles}
    if role_by_id[final_integer] != "main_step":
        raise ValueError("final block must have role main_step")

    complete_blocks = sorted(
        block_id for block_id, role in role_by_id.items() if role == "main_step"
    )
    if final_integer not in complete_blocks:
        raise AssertionError("final block must be part of Complete")
    key_blocks = [final_integer]
    complete_units = expand_block_indices_to_units(structure, complete_blocks)
    key_units = expand_block_indices_to_units(structure, key_blocks)
    if not set(key_units) <= set(complete_units):
        raise AssertionError("mechanical Key must be a subset of Complete")

    return {
        "schema_version": LABEL_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "eligibility": "usable",
        "path_status": str(path_status),
        "block_roles": normalized_roles,
        "final_block_id": final_integer,
        "key_block_indices": key_blocks,
        "key_unit_indices": key_units,
        "complete_block_indices": complete_blocks,
        "complete_unit_indices": complete_units,
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


def role_audit_target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the exact semantic target used for controls and repeats."""

    return (
        annotation.get("eligibility"),
        annotation.get("path_status"),
        tuple(
            (row.get("block_id"), row.get("role"))
            for row in annotation.get("block_roles", [])
        ),
        annotation.get("final_block_id"),
        tuple(annotation.get("key_unit_indices", [])),
        tuple(annotation.get("complete_unit_indices", [])),
    )


def _raw_control(
    *, question: str, unit_texts: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema_version": CONTROL_SCHEMA,
        "item_id": "",
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
    rationale: str,
) -> dict[str, Any]:
    if eligibility != "usable":
        return validate_role_audit_annotation(
            {
                "item_id": item["item_id"],
                "eligibility": eligibility,
                "path_status": None,
                "block_roles": [],
                "final_block_id": None,
                "confidence": "high",
                "rationale": rationale,
            },
            item,
        )
    if len(roles) != int(item["structure"]["block_count"]):
        raise ValueError(f"{item['item_id']}: control role count drift")
    return validate_role_audit_annotation(
        {
            "item_id": item["item_id"],
            "eligibility": "usable",
            "path_status": path_status,
            "block_roles": [
                {"block_id": index, "role": role}
                for index, role in enumerate(roles)
            ],
            "final_block_id": final_block_id,
            "confidence": "high",
            "rationale": rationale,
        },
        item,
    )


def build_hidden_controls_v15(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Build fresh controls for structural roles and H/P separation."""

    if annotator not in {"a", "b"}:
        raise ValueError("annotator must be a or b")
    specs: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "introduced_variable_is_main",
            _raw_control(
                question="A value is 4. Add 3 and then double the result.",
                unit_texts=[
                    "Let x = 4 + 3.",
                    "x = 7.",
                    "2x = 14.",
                    "The answer is 14.",
                ],
            ),
            {
                "roles": ["main_step", "main_step", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "rationale": "the introduced variable and both later uses form the answer path",
            },
        ),
        (
            "algebra_rewrite_chain",
            _raw_control(
                question="A word problem gives five times a value minus ten as twenty. Find the value.",
                unit_texts=[
                    "5y - 10 = 20.",
                    "5y = 30.",
                    "y = 6.",
                    "The answer is 6.",
                ],
            ),
            {
                "roles": ["main_step", "main_step", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "rationale": "the setup and both algebraic rewrites are used",
            },
        ),
        (
            "three_branch_sum",
            _raw_control(
                question="Two packs have nine red items, four packs have five blue items, and three packs have seven green items. How many total?",
                unit_texts=[
                    "2 * 9 = 18 red items.",
                    "4 * 5 = 20 blue items.",
                    "3 * 7 = 21 green items.",
                    "18 + 20 + 21 = 59 items.",
                    "The answer is 59.",
                ],
            ),
            {
                "roles": ["main_step", "main_step", "main_step", "main_step", "answer_wrapper"],
                "final_block_id": 3,
                "rationale": "all three subtotals and their sum are on the answer path",
            },
        ),
        (
            "premise_restatement_excluded",
            _raw_control(
                question="Nine boxes contain four items each. How many items?",
                unit_texts=[
                    "There are nine boxes with four items in each box.",
                    "9 * 4 = 36 items.",
                    "The answer is 36.",
                ],
            ),
            {
                "roles": ["premise_restatement", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "rationale": "the first line only copies the question and the multiplication completes the answer",
            },
        ),
        (
            "duplicate_excluded",
            _raw_control(
                question="Seven groups have eight items each, then five are added. How many?",
                unit_texts=[
                    "7 * 8 = 56.",
                    "Thus seven times eight is 56.",
                    "56 + 5 = 61.",
                    "The answer is 61.",
                ],
            ),
            {
                "roles": ["main_step", "duplicate", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "rationale": "the repeated 56 statement adds no step",
            },
        ),
        (
            "unused_branch_excluded",
            _raw_control(
                question="Three trays hold nine items each and two more are added. How many?",
                unit_texts=[
                    "3 * 9 = 27.",
                    "100 / 4 = 25.",
                    "27 + 2 = 29.",
                    "The answer is 29.",
                ],
            ),
            {
                "roles": ["main_step", "unused_branch", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "rationale": "the unrelated division is not on the answer path",
            },
        ),
        (
            "flawed_path_structural_key",
            _raw_control(
                question="Six boxes hold eight items each and two more are added. How many?",
                unit_texts=[
                    "6 * 8 = 47.",
                    "47 + 2 = 49.",
                    "The answer is 49.",
                ],
            ),
            {
                "path_status": "flawed",
                "roles": ["main_step", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "rationale": "the path is flawed, but structural Key remains the final calculation rather than the first error",
            },
        ),
        (
            "answer_only_empty_structure",
            _raw_control(
                question="What is eleven times five?",
                unit_texts=["The answer is \\boxed{55}."],
            ),
            {
                "eligibility": "no_auditable_reasoning",
                "rationale": "the response contains only an answer wrapper",
            },
        ),
    ]

    output: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (name, raw, expected_kwargs) in enumerate(specs):
        item_id = f"prior-v15-control-{annotator}-{index:02d}"
        raw["item_id"] = item_id
        public = public_package_item_v15(raw)
        expected = _expected_audit(public, **expected_kwargs)
        output.append((public, expected, name))
    return output


def build_blind_shards_v15(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 4,
    repeats_per_shard: int = 4,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    """Build four 12-natural +2-control +4-repeat shards per annotator."""

    if shard_count != 4 or len(proposals) != 48 or repeats_per_shard != 4:
        raise ValueError("v15 smoke freezes 48 natural rows in four 18-row shards")
    public_natural = [public_package_item_v15(row) for row in proposals]
    natural_shards = [
        public_natural[index::shard_count] for index in range(shard_count)
    ]
    if any(len(rows) != 12 for rows in natural_shards):
        raise AssertionError("v15 natural shard balance drift")

    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        controls = build_hidden_controls_v15(annotator)
        shard_rows = [list(rows) for rows in natural_shards]
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
                repeated = public_package_item_v15(parent, item_id=repeat_id)
                if repeated["structure"]["source_sha256"] != parent["structure"][
                    "source_sha256"
                ]:
                    raise AssertionError("v15 repeat source hash differs from parent")
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
                raise AssertionError("v15 public shard composition drift")
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
        "ai_outputs_key_or_complete": False,
        "dependency_edges_present": False,
        "target_derivation": "roles_to_complete_and_final_block_to_key",
    }
    return packages, private, construction


def summarize_role_burden(proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = [public_package_item_v15(row) for row in proposals]
    counts = [int(item["structure"]["block_count"]) for item in items]
    return {
        "natural_rows": len(items),
        "min_blocks_per_row": min(counts) if counts else 0,
        "mean_blocks_per_row": mean(counts) if counts else 0.0,
        "max_blocks_per_row": max(counts) if counts else 0,
        "role_decisions_per_row_equals_block_count": True,
        "edge_decisions_per_row": 0,
        "key_decisions_per_row": 0,
        "complete_set_decisions_per_row": 0,
    }


def _set_f1(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _set_iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def evaluate_blind_labels_v15(
    *,
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen role-only smoke without publishing targets."""

    normalized: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    package_maps: dict[str, dict[str, dict[str, Any]]] = {}
    private_maps: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for row in private_index:
        private_maps[str(row["annotator"])][str(row["item_id"])] = dict(row)

    schema_errors: list[str] = []
    for annotator in ("a", "b"):
        package_map = {str(row["item_id"]): dict(row) for row in packages[annotator]}
        package_maps[annotator] = package_map
        label_map = {str(row.get("item_id")): dict(row) for row in labels[annotator]}
        if len(label_map) != len(labels[annotator]) or set(label_map) != set(
            package_map
        ):
            raise ValueError(f"annotator {annotator} label/package population mismatch")
        for item_id, item in package_map.items():
            try:
                normalized[annotator][item_id] = validate_role_audit_annotation(
                    label_map[item_id], item
                )
            except ValueError as exc:
                schema_errors.append(f"{annotator}:{item_id}:{exc}")
    if schema_errors:
        return {
            "schema_version": EVALUATION_SCHEMA,
            "status": "FAIL_PRIOR_V15_ROLE_SCHEMA",
            "schema_errors": schema_errors,
            "trainable_labels_published": False,
            "next_step": "stop_v15_without_relabel_or_adjudication",
        }

    controls_report: dict[str, Any] = {}
    repeats_report: dict[str, Any] = {}
    for annotator in ("a", "b"):
        controls = [
            row for row in private_maps[annotator].values() if row["kind"] == "control"
        ]
        control_items: list[dict[str, Any]] = []
        for row in controls:
            actual = normalized[annotator][str(row["item_id"])]
            passed = role_audit_target_signature(actual) == role_audit_target_signature(
                row["expected_label"]
            )
            control_items.append(
                {"name": row["control_name"], "item_id": row["item_id"], "pass": passed}
            )
        controls_report[annotator] = {
            "passed": sum(row["pass"] for row in control_items),
            "total": len(control_items),
            "items": control_items,
        }

        repeats = [
            row for row in private_maps[annotator].values() if row["kind"] == "repeat"
        ]
        exact = complete_exact = key_exact = role_exact = 0
        for row in repeats:
            parent = normalized[annotator][str(row["natural_item_id"])]
            repeated = normalized[annotator][str(row["item_id"])]
            exact += role_audit_target_signature(parent) == role_audit_target_signature(
                repeated
            )
            complete_exact += parent["complete_unit_indices"] == repeated[
                "complete_unit_indices"
            ]
            key_exact += parent["key_unit_indices"] == repeated["key_unit_indices"]
            role_exact += parent["block_roles"] == repeated["block_roles"]
        total = len(repeats)
        repeats_report[annotator] = {
            "total": total,
            "target_signature_exact": exact,
            "target_signature_exact_rate": exact / max(1, total),
            "complete_exact": complete_exact,
            "key_exact": key_exact,
            "role_exact": role_exact,
        }

    natural_ids = sorted(
        str(row["item_id"])
        for row in private_maps["a"].values()
        if row["kind"] == "natural"
    )
    if set(natural_ids) != {
        str(row["item_id"])
        for row in private_maps["b"].values()
        if row["kind"] == "natural"
    }:
        raise ValueError("A/B natural populations differ")

    eligibility_exact = 0
    common: list[str] = []
    path_exact = final_exact = key_exact = 0
    role_agree = role_total = 0
    complete_f1: list[float] = []
    complete_iou: list[float] = []
    complete_coverage: list[float] = []
    key_f1: list[float] = []
    all_material_rows = 0
    main_step_rates: dict[str, list[float]] = {"a": [], "b": []}
    for item_id in natural_ids:
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        eligibility_exact += left["eligibility"] == right["eligibility"]
        if not (
            left["eligibility"] == right["eligibility"] == "usable"
            and left["confidence"] != "low"
            and right["confidence"] != "low"
        ):
            continue
        common.append(item_id)
        path_exact += left["path_status"] == right["path_status"]
        final_exact += left["final_block_id"] == right["final_block_id"]
        key_exact += left["key_unit_indices"] == right["key_unit_indices"]
        key_f1.append(_set_f1(left["key_unit_indices"], right["key_unit_indices"]))
        complete_f1.append(
            _set_f1(left["complete_unit_indices"], right["complete_unit_indices"])
        )
        complete_iou.append(
            _set_iou(left["complete_unit_indices"], right["complete_unit_indices"])
        )
        size = int(package_maps["a"][item_id]["structure"]["material_unit_count"])
        complete_coverage.append(
            1.0
            - len(
                set(left["complete_unit_indices"])
                ^ set(right["complete_unit_indices"])
            )
            / max(1, size)
        )
        all_material_rows += (
            len(
                set(left["complete_unit_indices"])
                | set(right["complete_unit_indices"])
            )
            == size
        )
        left_roles = {row["block_id"]: row["role"] for row in left["block_roles"]}
        right_roles = {row["block_id"]: row["role"] for row in right["block_roles"]}
        role_total += len(left_roles)
        role_agree += sum(left_roles[key] == right_roles[key] for key in left_roles)
        for annotator, label in (("a", left), ("b", right)):
            main_step_rates[annotator].append(
                len(label["complete_block_indices"]) / max(1, len(label["block_roles"]))
            )

    common_count = len(common)
    cross = {
        "natural_rows": len(natural_ids),
        "eligibility_exact": eligibility_exact,
        "eligibility_exact_rate": eligibility_exact / max(1, len(natural_ids)),
        "common_usable_nonlow": common_count,
        "path_exact_rate": path_exact / max(1, common_count),
        "final_block_exact_rate": final_exact / max(1, common_count),
        "key_exact_rate": key_exact / max(1, common_count),
        "key_macro_f1": mean(key_f1) if key_f1 else 0.0,
        "complete_macro_f1": mean(complete_f1) if complete_f1 else 0.0,
        "complete_macro_iou": mean(complete_iou) if complete_iou else 0.0,
        "complete_mask_coverage": (
            mean(complete_coverage) if complete_coverage else 0.0
        ),
        "role_decision_agreement": role_agree / max(1, role_total),
        "all_material_union_rate": all_material_rows / max(1, common_count),
        "mean_main_step_rate_a": (
            mean(main_step_rates["a"]) if main_step_rates["a"] else 0.0
        ),
        "mean_main_step_rate_b": (
            mean(main_step_rates["b"]) if main_step_rates["b"] else 0.0
        ),
    }
    checks = {
        "controls_a": controls_report["a"]["passed"]
        >= int(gates["controls_min_pass"]),
        "controls_b": controls_report["b"]["passed"]
        >= int(gates["controls_min_pass"]),
        "self_repeat_a": repeats_report["a"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "self_repeat_b": repeats_report["b"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "common_usable": common_count >= int(gates["common_usable_nonlow_min"]),
        "path": cross["path_exact_rate"] >= float(gates["path_exact_min"]),
        "final_block": cross["final_block_exact_rate"]
        >= float(gates["final_block_exact_min"]),
        "key": cross["key_macro_f1"] >= float(gates["key_macro_f1_min"]),
        "complete_f1": cross["complete_macro_f1"]
        >= float(gates["complete_macro_f1_min"]),
        "complete_iou": cross["complete_macro_iou"]
        >= float(gates["complete_macro_iou_min"]),
        "coverage": cross["complete_mask_coverage"]
        >= float(gates["complete_mask_coverage_min"]),
        "roles": cross["role_decision_agreement"]
        >= float(gates["role_decision_agreement_min"]),
        "all_material": cross["all_material_union_rate"]
        <= float(gates["all_material_union_rate_max"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": (
            "PASS_PRIOR_V15_ROLE_ONLY_SMOKE"
            if passed
            else "STOP_PRIOR_V15_ROLE_ONLY_SMOKE"
        ),
        "schema_errors": [],
        "controls": controls_report,
        "self_repeats": repeats_report,
        "cross_annotator_natural": cross,
        "gates": dict(gates),
        "gate_checks": checks,
        "target_definition": {
            "key": "raw units of the final answer-producing main_step block",
            "complete": "raw units of every block labelled main_step",
            "flaw_localization_owned_by": "Hallucination_H0_not_Prior",
            "dependency_edges_used": False,
        },
        "trainable_labels_published": False,
        "next_step": (
            "freeze_separate_prior_role_scale_v16"
            if passed
            else "stop_v15_without_relabel_or_adjudication"
        ),
        "v14_terminal_decision_unchanged": True,
    }


__all__ = [
    "CONTROL_SCHEMA",
    "DEFAULT_NAMESPACE",
    "EVALUATION_SCHEMA",
    "LABEL_SCHEMA",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "STRUCTURE_SCHEMA",
    "build_blind_shards_v15",
    "build_hidden_controls_v15",
    "evaluate_blind_labels_v15",
    "public_package_item_v15",
    "role_audit_target_signature",
    "select_fresh_natural_rows_v15",
    "summarize_role_burden",
    "validate_role_audit_annotation",
]
