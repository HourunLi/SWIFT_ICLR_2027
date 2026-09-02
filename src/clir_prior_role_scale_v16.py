"""Prospective role-only CLIR Prior scale-v16 contracts.

V16 scales the role audit that passed the fresh v15 smoke.  Annotators never
draw dependency edges and never emit Key/Complete sets.  They audit one role
per deterministic block and identify the final answer-producing block.  The
program then derives exact-token targets:

* Key is the shared final block and is fully supervised;
* Complete is positive where both annotators say ``main_step``;
* Complete is negative where both say non-main;
* a main/non-main disagreement is masked, rather than silently adjudicated.

The module is intentionally independent of label files and model inference so
that selection, package construction, gates, and target materialization can be
frozen before either annotator runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from numbers import Integral
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_role_v15 import (
    build_hidden_controls_v15,
    public_package_item_v15,
    role_audit_target_signature,
    validate_role_audit_annotation,
)
from src.clir_smoke import canonical_sha256, stable_priority


PROTOCOL_SCHEMA = "clir-prior-role-only-scale-v16"
PROPOSAL_SCHEMA = "clir-prior-role-only-scale-proposal-v16"
PACKAGE_SCHEMA = "clir-prior-role-only-scale-package-v16"
PRIVATE_SCHEMA = "clir-prior-role-only-scale-private-index-v16"
EVALUATION_SCHEMA = "clir-prior-role-only-scale-evaluation-v16"
ROW_SCHEMA = "clir-prior-role-only-silver-row-v16"
DEFAULT_NAMESPACE = "clir-prior-role-only-scale-v16"
LABEL_NAME = "silver_dual_ai_role_only_prior_v16_no_human_verification"


def _material_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = row.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ValueError("row units must be an array")
    output: list[dict[str, Any]] = []
    for raw in units:
        if not isinstance(raw, Mapping):
            raise ValueError("unit must be an object")
        if raw.get("kind") != "material_claim":
            continue
        index = raw.get("unit_index")
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("material unit_index must be an integer")
        output.append(
            {
                "unit_index": int(index),
                "kind": "material_claim",
                "text": str(raw.get("text", "")),
            }
        )
    output.sort(key=lambda unit: unit["unit_index"])
    if len({unit["unit_index"] for unit in output}) != len(output):
        raise ValueError("material unit indices are not unique")
    return output


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source")),
        str(row.get("checker_status")),
        str(row.get("prior_label_split")),
    )


def _stratum_name(value: Sequence[str]) -> str:
    return "|".join(value)


def select_scale_rows_v16(
    materialized_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: Iterable[str],
    excluded_cluster_ids: Iterable[str],
    strata: Sequence[Mapping[str, Any]],
    minimum_material_claims: int = 6,
    maximum_material_claims: int = 40,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one fresh trajectory per query and template cluster."""

    quotas = {
        (str(row["source"]), str(row["checker_status"]), str(row["split"])): int(
            row["count"]
        )
        for row in strata
    }
    if len(quotas) != len(strata) or any(count <= 0 for count in quotas.values()):
        raise ValueError("v16 proposal strata must be unique and positive")
    excluded_queries = {str(value) for value in excluded_query_ids}
    excluded_clusters = {str(value) for value in excluded_cluster_ids}
    by_stratum_query: dict[
        tuple[str, str, str], dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    rejection: Counter[str] = Counter()
    for row in materialized_rows:
        stratum = _stratum(row)
        if stratum not in quotas:
            rejection["outside_frozen_strata"] += 1
            continue
        query_id = str(row.get("query_id", ""))
        cluster_id = str(row.get("cluster_id", ""))
        if not query_id or not cluster_id:
            rejection["missing_identity"] += 1
            continue
        if query_id in excluded_queries or cluster_id in excluded_clusters:
            rejection["historically_used_query_or_cluster"] += 1
            continue
        if row.get("eligible_for_supervision") is not True:
            rejection["not_supervision_eligible"] += 1
            continue
        if row.get("unitization_status") != "ok" or row.get("status") != "ok":
            rejection["materialization_not_ok"] += 1
            continue
        if row.get("finish_reason") != "stop":
            rejection["finish_reason"] += 1
            continue
        count = row.get("material_claim_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not minimum_material_claims <= int(count) <= maximum_material_claims
        ):
            rejection["material_claim_count"] += 1
            continue
        if len(_material_units(row)) != int(count):
            raise ValueError(f"{row.get('id')}: material claim count drift")
        by_stratum_query[stratum][query_id].append(row)

    candidates: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for stratum, by_query in by_stratum_query.items():
        representatives = [
            min(
                rows,
                key=lambda row: stable_priority(
                    f"{namespace}:candidate", query_id, str(row.get("id"))
                ),
            )
            for query_id, rows in by_query.items()
        ]
        candidates[stratum] = sorted(
            representatives,
            key=lambda row: stable_priority(
                f"{namespace}:query", *stratum, str(row["query_id"]), str(row["id"])
            ),
        )

    protocol_order = [
        (str(row["source"]), str(row["checker_status"]), str(row["split"]))
        for row in strata
    ]
    # Scarce strata reserve their clusters first.  The acquisition registry is
    # already globally cluster-disjoint, but the check remains explicit.
    ordered_strata = sorted(
        quotas,
        key=lambda value: (len(candidates.get(value, [])), protocol_order.index(value)),
    )
    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    for stratum in ordered_strata:
        chosen = 0
        for row in candidates.get(stratum, []):
            query_id = str(row["query_id"])
            cluster_id = str(row["cluster_id"])
            if query_id in used_queries or cluster_id in used_clusters:
                continue
            source_row_id = str(row["id"])
            item_id = "prior-v16-natural-" + stable_priority(
                f"{namespace}:item", source_row_id
            )[:24]
            material = _material_units(row)
            selected.append(
                {
                    "schema_version": PROPOSAL_SCHEMA,
                    "item_id": item_id,
                    "source_row_id": source_row_id,
                    "query_id": query_id,
                    "cluster_id": cluster_id,
                    "source": stratum[0],
                    "checker_status": stratum[1],
                    "prior_label_split": stratum[2],
                    "candidate_index": int(row["candidate_index"]),
                    "source_record_id": row.get("source_record_id"),
                    "material_claim_count": len(material),
                    "selection_priority": stable_priority(
                        f"{namespace}:selection", source_row_id
                    ),
                    "question": str(row["question"]),
                    "response": str(row["response"]),
                    "units": material,
                }
            )
            used_queries.add(query_id)
            used_clusters.add(cluster_id)
            chosen += 1
            if chosen == quotas[stratum]:
                break
        if chosen != quotas[stratum]:
            raise ValueError(
                f"insufficient v16 capacity for {_stratum_name(stratum)}: "
                f"{chosen}/{quotas[stratum]}"
            )

    selected.sort(key=lambda row: str(row["selection_priority"]))
    expected = sum(quotas.values())
    if len(selected) != expected:
        raise AssertionError("v16 selected row count drift")
    if len(used_queries) != expected or len(used_clusters) != expected:
        raise AssertionError("v16 selection is not query/cluster distinct")
    counts = Counter(_stratum(row) for row in selected)
    return selected, {
        "namespace": namespace,
        "selected": len(selected),
        "distinct_queries": len(used_queries),
        "distinct_clusters": len(used_clusters),
        "excluded_query_count": len(excluded_queries),
        "excluded_cluster_count": len(excluded_clusters),
        "available_query_counts": {
            _stratum_name(value): len(candidates.get(value, []))
            for value in sorted(quotas)
        },
        "selected_by_stratum": {
            _stratum_name(value): counts[value] for value in sorted(quotas)
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "ordered_item_ids_sha256": canonical_sha256(
            [row["item_id"] for row in selected]
        ),
    }


def public_package_item_v16(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    item = public_package_item_v15(source, item_id=item_id)
    item["schema_version"] = PACKAGE_SCHEMA
    return item


def _raw_expected_label(
    item: Mapping[str, Any],
    *,
    roles: Sequence[str] = (),
    final_block_id: int | None = None,
    path_status: str | None = "supported",
    eligibility: str = "usable",
    rationale: str,
) -> dict[str, Any]:
    return validate_role_audit_annotation(
        {
            "item_id": item["item_id"],
            "eligibility": eligibility,
            "path_status": path_status if eligibility == "usable" else None,
            "block_roles": (
                [
                    {"block_id": index, "role": role}
                    for index, role in enumerate(roles)
                ]
                if eligibility == "usable"
                else []
            ),
            "final_block_id": final_block_id if eligibility == "usable" else None,
            "confidence": "high",
            "rationale": rationale,
        },
        item,
    )


def build_hidden_controls_v16(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Build twelve role controls per annotator, four fresh beyond v15."""

    if annotator not in {"a", "b"}:
        raise ValueError("annotator must be a or b")
    output: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (old_item, old_expected, name) in enumerate(
        build_hidden_controls_v15(annotator)
    ):
        new_id = f"prior-v16-control-{annotator}-{index:02d}"
        item = public_package_item_v16(old_item, item_id=new_id)
        annotation = {
            key: deepcopy(old_expected[key])
            for key in (
                "eligibility",
                "path_status",
                "block_roles",
                "final_block_id",
                "confidence",
                "rationale",
            )
        }
        annotation["item_id"] = new_id
        expected = validate_role_audit_annotation(annotation, item)
        output.append((item, expected, f"v15_{name}"))

    fresh = [
        (
            "formula_and_plan_excluded",
            "A rectangle is 12 meters long and 5 meters wide. Find its area.",
            [
                "Area of a rectangle is length times width.",
                "Now substitute the values.",
                "12 * 5 = 60 square meters.",
                "The answer is 60 square meters.",
            ],
            ["formula_only", "plan_or_heading", "main_step", "answer_wrapper"],
            2,
            "the generic formula and plan do not themselves calculate this answer",
        ),
        (
            "used_definition_is_main",
            "A number is three more than 8, then multiplied by 4. Find the result.",
            [
                "Let z be three more than 8, so z = 8 + 3.",
                "z = 11.",
                "4z = 44.",
                "Therefore the result is 44.",
            ],
            ["main_step", "main_step", "main_step", "answer_wrapper"],
            2,
            "the introduced symbol and both calculations are used by the answer",
        ),
        (
            "two_used_branches_one_unused",
            "Four packs hold six cards each and two packs hold five cards each. How many cards total?",
            [
                "4 * 6 = 24 cards.",
                "2 * 5 = 10 cards.",
                "9 * 9 = 81.",
                "24 + 10 = 34 cards.",
                "The answer is 34 cards.",
            ],
            ["main_step", "main_step", "unused_branch", "main_step", "answer_wrapper"],
            3,
            "both subtotals feed the sum while the unrelated multiplication is unused",
        ),
        (
            "late_flaw_keeps_structural_final",
            "Five boxes hold seven items each and three are added. How many items?",
            [
                "5 * 7 = 35 items.",
                "35 + 3 = 39 items.",
                "The answer is 39 items.",
            ],
            ["main_step", "main_step", "answer_wrapper"],
            1,
            "the addition is wrong but it remains the final structural calculation",
        ),
    ]
    for offset, (name, question, texts, roles, final_block, rationale) in enumerate(
        fresh, start=8
    ):
        raw = {
            "item_id": f"prior-v16-control-{annotator}-{offset:02d}",
            "question": question,
            "response": "\n".join(texts),
            "units": [
                {"unit_index": 2 * index, "kind": "material_claim", "text": text}
                for index, text in enumerate(texts)
            ],
        }
        item = public_package_item_v16(raw)
        expected = _raw_expected_label(
            item,
            roles=roles,
            final_block_id=final_block,
            path_status=("flawed" if name == "late_flaw_keeps_structural_final" else "supported"),
            rationale=rationale,
        )
        output.append((item, expected, name))
    if len(output) != 12:
        raise AssertionError("v16 control count drift")
    return output


def build_blind_shards_v16(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 12,
    natural_per_shard: int = 50,
    repeats_per_shard: int = 5,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    """Build twelve 50-natural +1-control +5-repeat shards per side."""

    expected_natural = shard_count * natural_per_shard
    if (
        shard_count != 12
        or natural_per_shard != 50
        or repeats_per_shard != 5
        or len(proposals) != expected_natural
    ):
        raise ValueError("v16 freezes 600 natural rows in twelve 56-row shards")
    ordered = sorted(
        (dict(row) for row in proposals),
        key=lambda row: stable_priority(f"{namespace}:natural-shard", row["item_id"]),
    )
    natural_shards = [
        [public_package_item_v16(row) for row in ordered[index::shard_count]]
        for index in range(shard_count)
    ]
    if any(len(rows) != natural_per_shard for rows in natural_shards):
        raise AssertionError("v16 natural shard balance drift")

    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        shard_rows = [list(rows) for rows in natural_shards]
        controls = build_hidden_controls_v16(annotator)
        for shard_index, (item, expected, name) in enumerate(controls):
            shard_rows[shard_index].append(item)
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "shard_index": shard_index,
                    "kind": "control",
                    "item_id": item["item_id"],
                    "natural_item_id": None,
                    "control_name": name,
                    "expected_label": expected,
                }
            )

        for parent_shard, parents in enumerate(natural_shards):
            destination = (parent_shard + 1) % shard_count
            for local_index, parent in enumerate(parents[:repeats_per_shard]):
                repeat_id = (
                    f"{parent['item_id']}:repeat:{annotator}:"
                    f"{parent_shard:02d}:{local_index:02d}"
                )
                repeated = public_package_item_v16(parent, item_id=repeat_id)
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
            if len(rows) != natural_per_shard + 1 + repeats_per_shard:
                raise AssertionError("v16 public shard composition drift")
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}:package:{annotator}:{shard_index}", row["item_id"]
                )
            )
        packages[annotator] = shard_rows

    private.sort(key=lambda row: (row["annotator"], row["shard_index"], row["item_id"]))
    construction = {
        "annotators": ["a", "b"],
        "shards_per_annotator": shard_count,
        "natural_rows_per_shard": natural_per_shard,
        "controls_per_shard": 1,
        "repeats_per_shard": repeats_per_shard,
        "rows_per_shard": natural_per_shard + 1 + repeats_per_shard,
        "natural_rows_per_annotator": expected_natural,
        "controls_per_annotator": shard_count,
        "repeats_per_annotator": shard_count * repeats_per_shard,
        "rows_per_annotator": expected_natural + shard_count * (1 + repeats_per_shard),
        "parent_repeat_same_shard": 0,
        "repeat_parent_population_shared_across_annotators": True,
        "ai_outputs_key_or_complete": False,
        "dependency_edges_present": False,
    }
    return packages, private, construction


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


def _normalize_population(
    *,
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    normalized: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    package_maps: dict[str, dict[str, dict[str, Any]]] = {}
    private_maps: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for raw in private_index:
        row = dict(raw)
        annotator = str(row.get("annotator"))
        if annotator not in private_maps or row.get("schema_version") != PRIVATE_SCHEMA:
            raise ValueError("v16 private index schema or annotator drift")
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in private_maps[annotator]:
            raise ValueError("v16 private item identity is missing or duplicated")
        private_maps[annotator][item_id] = row

    for annotator in ("a", "b"):
        package_map: dict[str, dict[str, Any]] = {}
        for raw in packages[annotator]:
            row = dict(raw)
            item_id = str(row.get("item_id", ""))
            if (
                row.get("schema_version") != PACKAGE_SCHEMA
                or not item_id
                or item_id in package_map
            ):
                raise ValueError(f"v16 package {annotator} schema or ID drift")
            package_map[item_id] = row
        label_map = {str(row.get("item_id")): dict(row) for row in labels[annotator]}
        if len(label_map) != len(labels[annotator]) or set(label_map) != set(package_map):
            raise ValueError(f"v16 annotator {annotator} label/package population mismatch")
        if set(private_maps[annotator]) != set(package_map):
            raise ValueError(f"v16 annotator {annotator} package/private mismatch")
        package_maps[annotator] = package_map
        for item_id, item in package_map.items():
            normalized[annotator][item_id] = validate_role_audit_annotation(
                label_map[item_id], item
            )
    return normalized, package_maps, private_maps


def _pair_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    left_complete = set(left["complete_unit_indices"])
    right_complete = set(right["complete_unit_indices"])
    material_count = int(item["structure"]["material_unit_count"])
    ambiguous = left_complete ^ right_complete
    union = left_complete | right_complete
    return {
        "complete_f1": _set_f1(left_complete, right_complete),
        "complete_iou": _set_iou(left_complete, right_complete),
        "complete_mask_coverage": 1 - len(ambiguous) / max(1, material_count),
        "all_material_union": len(union) == material_count,
    }


def evaluate_role_scale_v16(
    *,
    proposals: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    final_strata: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate prospective gates and freeze the label-independent quota pick."""

    try:
        normalized, package_maps, private_maps = _normalize_population(
            packages=packages, private_index=private_index, labels=labels
        )
    except ValueError as exc:
        return {
            "schema_version": EVALUATION_SCHEMA,
            "status": "FAIL_PRIOR_V16_ROLE_SCHEMA",
            "schema_errors": [str(exc)],
            "trainable_labels_published": False,
            "next_step": "stop_v16_without_relabel_adjudication_or_subset_salvage",
        }

    proposal_by_id = {str(row["item_id"]): dict(row) for row in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("v16 proposal IDs are duplicated")
    natural_ids = {
        annotator: {
            item_id
            for item_id, row in private_maps[annotator].items()
            if row["kind"] == "natural"
        }
        for annotator in ("a", "b")
    }
    if natural_ids["a"] != natural_ids["b"] or natural_ids["a"] != set(proposal_by_id):
        raise ValueError("v16 proposal/A/B natural populations differ")

    controls_report: dict[str, Any] = {}
    repeats_report: dict[str, Any] = {}
    repeat_failed: defaultdict[str, set[str]] = defaultdict(set)
    for annotator in ("a", "b"):
        control_rows = [
            row for row in private_maps[annotator].values() if row["kind"] == "control"
        ]
        control_items = []
        for row in control_rows:
            passed = role_audit_target_signature(
                normalized[annotator][str(row["item_id"])]
            ) == role_audit_target_signature(row["expected_label"])
            control_items.append(
                {"name": row["control_name"], "item_id": row["item_id"], "pass": passed}
            )
        controls_report[annotator] = {
            "passed": sum(item["pass"] for item in control_items),
            "total": len(control_items),
            "items": control_items,
        }
        repeat_rows = [
            row for row in private_maps[annotator].values() if row["kind"] == "repeat"
        ]
        exact = complete_exact = key_exact = role_exact = 0
        for row in repeat_rows:
            parent_id = str(row["natural_item_id"])
            parent = normalized[annotator][parent_id]
            repeated = normalized[annotator][str(row["item_id"])]
            is_exact = role_audit_target_signature(parent) == role_audit_target_signature(
                repeated
            )
            exact += is_exact
            complete_exact += parent["complete_unit_indices"] == repeated["complete_unit_indices"]
            key_exact += parent["key_unit_indices"] == repeated["key_unit_indices"]
            role_exact += parent["block_roles"] == repeated["block_roles"]
            if not is_exact:
                repeat_failed[parent_id].add(annotator)
        repeats_report[annotator] = {
            "total": len(repeat_rows),
            "target_signature_exact": exact,
            "target_signature_exact_rate": exact / max(1, len(repeat_rows)),
            "complete_exact": complete_exact,
            "key_exact": key_exact,
            "role_exact": role_exact,
        }

    ordered_natural = sorted(proposal_by_id)
    eligibility_exact = path_exact = final_exact = key_exact = 0
    common: list[str] = []
    complete_f1: list[float] = []
    complete_iou: list[float] = []
    coverage: list[float] = []
    role_agree = role_total = all_material_union = 0
    eligible_by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    excluded = Counter()
    for item_id in ordered_natural:
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        eligibility_exact += left["eligibility"] == right["eligibility"]
        if not (
            left["eligibility"] == right["eligibility"] == "usable"
            and left["confidence"] != "low"
            and right["confidence"] != "low"
        ):
            excluded["not_common_nonlow_usable"] += 1
            continue
        common.append(item_id)
        path_exact += left["path_status"] == right["path_status"]
        same_final = left["final_block_id"] == right["final_block_id"]
        final_exact += same_final
        key_exact += left["key_unit_indices"] == right["key_unit_indices"]
        pair = _pair_metrics(left, right, package_maps["a"][item_id])
        complete_f1.append(pair["complete_f1"])
        complete_iou.append(pair["complete_iou"])
        coverage.append(pair["complete_mask_coverage"])
        all_material_union += pair["all_material_union"]
        left_roles = {row["block_id"]: row["role"] for row in left["block_roles"]}
        right_roles = {row["block_id"]: row["role"] for row in right["block_roles"]}
        role_total += len(left_roles)
        role_agree += sum(left_roles[key] == right_roles[key] for key in left_roles)
        if not same_final:
            excluded["final_block_disagreement"] += 1
            continue
        if item_id in repeat_failed:
            excluded["available_self_repeat_failed"] += 1
            continue
        eligible_by_stratum[_stratum(proposal_by_id[item_id])].append(item_id)

    final_quotas = {
        (str(row["source"]), str(row["checker_status"]), str(row["split"])): int(
            row["count"]
        )
        for row in final_strata
    }
    if len(final_quotas) != len(final_strata):
        raise ValueError("v16 final strata are duplicated")
    selected_ids: list[str] = []
    quota_feasible = True
    for stratum in [
        (str(row["source"]), str(row["checker_status"]), str(row["split"]))
        for row in final_strata
    ]:
        ordered = sorted(
            eligible_by_stratum.get(stratum, []),
            key=lambda item_id: (
                str(proposal_by_id[item_id]["selection_priority"]), item_id
            ),
        )
        quota = final_quotas[stratum]
        if len(ordered) < quota:
            quota_feasible = False
        selected_ids.extend(ordered[:quota])

    selected_pairs = [
        _pair_metrics(
            normalized["a"][item_id],
            normalized["b"][item_id],
            package_maps["a"][item_id],
        )
        for item_id in selected_ids
    ]
    selected_iou = mean(row["complete_iou"] for row in selected_pairs) if selected_pairs else 0.0
    selected_coverage = (
        mean(row["complete_mask_coverage"] for row in selected_pairs)
        if selected_pairs
        else 0.0
    )
    selected_all_material = (
        mean(float(row["all_material_union"]) for row in selected_pairs)
        if selected_pairs
        else 1.0
    )
    common_count = len(common)
    cross = {
        "natural_rows": len(ordered_natural),
        "eligibility_exact": eligibility_exact,
        "eligibility_exact_rate": eligibility_exact / max(1, len(ordered_natural)),
        "common_usable_nonlow": common_count,
        "path_exact_rate": path_exact / max(1, common_count),
        "final_block_exact_rate": final_exact / max(1, common_count),
        "key_exact_rate": key_exact / max(1, common_count),
        "complete_macro_f1": mean(complete_f1) if complete_f1 else 0.0,
        "complete_macro_iou": mean(complete_iou) if complete_iou else 0.0,
        "complete_mask_coverage": mean(coverage) if coverage else 0.0,
        "role_decision_agreement": role_agree / max(1, role_total),
        "all_material_union_rate": all_material_union / max(1, common_count),
    }
    selected_counts = Counter(_stratum(proposal_by_id[item_id]) for item_id in selected_ids)
    selected = {
        "target_rows": sum(final_quotas.values()),
        "selected_rows": len(selected_ids),
        "selected_train_rows": sum(
            proposal_by_id[item_id]["prior_label_split"] == "train"
            for item_id in selected_ids
        ),
        "selected_dev_rows": sum(
            proposal_by_id[item_id]["prior_label_split"] == "dev"
            for item_id in selected_ids
        ),
        "selected_by_stratum": {
            _stratum_name(value): selected_counts[value] for value in final_quotas
        },
        "eligible_by_stratum": {
            _stratum_name(value): len(eligible_by_stratum.get(value, []))
            for value in final_quotas
        },
        "selected_complete_iou_mean": selected_iou,
        "selected_complete_mask_coverage_mean": selected_coverage,
        "selected_all_material_union_rate": selected_all_material,
        "selected_ordered_item_ids": selected_ids,
        "selected_ordered_item_ids_sha256": canonical_sha256(selected_ids),
    }
    checks = {
        "controls_a": controls_report["a"]["passed"] >= int(gates["controls_min_pass"]),
        "controls_b": controls_report["b"]["passed"] >= int(gates["controls_min_pass"]),
        "self_repeat_a": repeats_report["a"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "self_repeat_b": repeats_report["b"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "eligibility": cross["eligibility_exact_rate"]
        >= float(gates["eligibility_exact_min"]),
        "common_usable": common_count >= int(gates["common_usable_nonlow_min"]),
        "path": cross["path_exact_rate"] >= float(gates["path_exact_min"]),
        "final_block": cross["final_block_exact_rate"]
        >= float(gates["final_block_exact_min"]),
        "roles": cross["role_decision_agreement"]
        >= float(gates["role_decision_agreement_min"]),
        "quota_feasible": quota_feasible and len(selected_ids) == sum(final_quotas.values()),
        "selected_iou": selected_iou >= float(gates["selected_complete_iou_min"]),
        "selected_coverage": selected_coverage
        >= float(gates["selected_complete_mask_coverage_min"]),
        "selected_all_material": selected_all_material
        <= float(gates["selected_all_material_union_rate_max"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": (
            "PASS_PRIOR_V16_ROLE_ONLY_SCALE"
            if passed
            else "STOP_PRIOR_V16_ROLE_ONLY_SCALE"
        ),
        "schema_errors": [],
        "controls": controls_report,
        "self_repeats": repeats_report,
        "cross_annotator_natural": cross,
        "prospective_frozen_selection": selected,
        "excluded_reasons": dict(sorted(excluded.items())),
        "gates": dict(gates),
        "gate_checks": checks,
        "target_definition": {
            "key": "shared final answer-producing main_step block; full token mask",
            "complete_positive": "blocks both annotators label main_step",
            "complete_negative": "blocks both annotators label non-main_step",
            "complete_masked": "blocks with main/non-main disagreement",
            "flaw_localization_owned_by": "Hallucination_H0_not_Prior",
            "dependency_edges_used": False,
        },
        "trainable_labels_published": False,
        "next_step": (
            "materialize_only_the_frozen_selected_ids"
            if passed
            else "stop_v16_without_relabel_adjudication_or_subset_salvage"
        ),
    }


def _integer_tokens(value: Any, *, field: str, row_id: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be an integer array")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0:
            raise ValueError(f"{row_id}: {field} contains an invalid token ID")
        output.append(int(item))
    if not output:
        raise ValueError(f"{row_id}: {field} is empty")
    return output


def _unit_token_spans(row: Mapping[str, Any], output_count: int) -> dict[int, tuple[int, int, str]]:
    row_id = str(row.get("id", ""))
    units = row.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ValueError(f"{row_id}: units must be an array")
    spans: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    for raw in units:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{row_id}: unit is not an object")
        index = raw.get("unit_index")
        start = raw.get("token_start")
        end = raw.get("token_end")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (index, start, end)):
            raise ValueError(f"{row_id}: unit indices/spans must be integers")
        index, start, end = int(index), int(start), int(end)
        if index in spans or start != cursor or not start <= end <= output_count:
            raise ValueError(f"{row_id}: unit partition is invalid")
        spans[index] = (start, end, str(raw.get("kind")))
        cursor = end
    if cursor != output_count:
        raise ValueError(f"{row_id}: units do not cover the exact output token axis")
    return spans


def construct_silver_rows_v16(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluation_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize only the prospectively selected v16 IDs to exact tokens."""

    if evaluation_report.get("status") != "PASS_PRIOR_V16_ROLE_ONLY_SCALE":
        raise ValueError("v16 Silver materialization requires a passing frozen report")
    normalized, package_maps, _ = _normalize_population(
        packages=packages, private_index=private_index, labels=labels
    )
    proposal_by_id = {str(row["item_id"]): dict(row) for row in proposals}
    materialized_by_id = {str(row["id"]): dict(row) for row in materialized_rows}
    selected_ids = list(
        evaluation_report["prospective_frozen_selection"]["selected_ordered_item_ids"]
    )
    if canonical_sha256(selected_ids) != evaluation_report["prospective_frozen_selection"][
        "selected_ordered_item_ids_sha256"
    ]:
        raise ValueError("v16 selected-ID report binding drift")

    output: list[dict[str, Any]] = []
    strata = Counter()
    for selection_index, item_id in enumerate(selected_ids):
        proposal = proposal_by_id[item_id]
        row_id = str(proposal["source_row_id"])
        source = materialized_by_id[row_id]
        for field in ("query_id", "cluster_id", "checker_status", "candidate_index"):
            if source.get(field) != proposal.get(field):
                raise ValueError(f"{row_id}: proposal/materialized {field} drift")
        prompt_ids = _integer_tokens(source.get("prompt_token_ids"), field="prompt_token_ids", row_id=row_id)
        output_ids = _integer_tokens(source.get("output_token_ids"), field="output_token_ids", row_id=row_id)
        spans = _unit_token_spans(source, len(output_ids))
        left, right = normalized["a"][item_id], normalized["b"][item_id]
        if not (
            left["eligibility"] == right["eligibility"] == "usable"
            and left["confidence"] != "low"
            and right["confidence"] != "low"
            and left["final_block_id"] == right["final_block_id"]
        ):
            raise ValueError(f"{item_id}: selected row no longer satisfies v16 consensus")
        left_roles = {entry["block_id"]: entry["role"] for entry in left["block_roles"]}
        right_roles = {entry["block_id"]: entry["role"] for entry in right["block_roles"]}
        structure = package_maps["a"][item_id]["structure"]
        blocks = {int(block["block_id"]): block for block in structure["blocks"]}
        positive_blocks = sorted(
            block_id
            for block_id in blocks
            if left_roles[block_id] == right_roles[block_id] == "main_step"
        )
        ambiguous_blocks = sorted(
            block_id
            for block_id in blocks
            if (left_roles[block_id] == "main_step")
            != (right_roles[block_id] == "main_step")
        )
        final_block = int(left["final_block_id"])
        if final_block not in positive_blocks:
            raise AssertionError("v16 shared final block must be Complete-positive")
        positive_units = sorted(
            unit
            for block_id in positive_blocks
            for unit in blocks[block_id]["unit_indices"]
        )
        ambiguous_units = sorted(
            unit
            for block_id in ambiguous_blocks
            for unit in blocks[block_id]["unit_indices"]
        )
        key_units = sorted(int(value) for value in blocks[final_block]["unit_indices"])
        key_target = [0] * len(output_ids)
        complete_target = [0] * len(output_ids)
        key_mask = [1] * len(output_ids)
        complete_mask = [1] * len(output_ids)
        for unit_index in key_units:
            start, end, kind = spans[unit_index]
            if kind != "material_claim":
                raise ValueError(f"{row_id}: Key unit is not material")
            key_target[start:end] = [1] * (end - start)
        for unit_index in positive_units:
            start, end, kind = spans[unit_index]
            if kind != "material_claim":
                raise ValueError(f"{row_id}: Complete unit is not material")
            complete_target[start:end] = [1] * (end - start)
        for unit_index in ambiguous_units:
            start, end, kind = spans[unit_index]
            if kind != "material_claim":
                raise ValueError(f"{row_id}: ambiguous Complete unit is not material")
            complete_mask[start:end] = [0] * (end - start)
        if any(k and not c for k, c in zip(key_target, complete_target, strict=True)):
            raise AssertionError("v16 Key is not nested in Complete")
        split = str(proposal["prior_label_split"])
        result = {
            "schema_version": ROW_SCHEMA,
            "selection_index": selection_index,
            "id": row_id,
            "trajectory_id": row_id,
            "proposal_id": item_id,
            "query_id": str(proposal["query_id"]),
            "cluster_id": str(proposal["cluster_id"]),
            "candidate_index": int(proposal["candidate_index"]),
            "source": proposal["source"],
            "source_record_id": proposal.get("source_record_id"),
            "split": split,
            "prior_label_split": split,
            "checker_status": proposal["checker_status"],
            "correctness": int(source["correctness"]),
            "prompt_token_ids": prompt_ids,
            "output_token_ids": output_ids,
            "prompt_token_count": len(prompt_ids),
            "output_token_count": len(output_ids),
            "key_unit_indices": key_units,
            "complete_unit_indices": positive_units,
            "complete_ambiguous_unit_indices": ambiguous_units,
            "key_prior_target": key_target,
            "key_prior_mask": key_mask,
            "complete_prior_target": complete_target,
            "complete_prior_mask": complete_mask,
            "prior_label_name": LABEL_NAME,
            "prior_label_source": "blind_dual_ai_role_consensus_with_disagreement_mask",
            "prior_human_verified": False,
            "feature_role": "prior_train" if split == "train" else "prior_dev",
        }
        output.append(result)
        strata[_stratum_name(_stratum(proposal))] += 1

    if len({row["query_id"] for row in output}) != len(output) or len(
        {row["cluster_id"] for row in output}
    ) != len(output):
        raise ValueError("v16 Silver rows are not query/cluster distinct")
    report = {
        "schema_version": "clir-prior-role-only-materialization-report-v16",
        "status": "PASS_PRIOR_V16_SILVER_TARGET_MATERIALIZATION",
        "selected_rows": len(output),
        "train_rows": sum(row["split"] == "train" for row in output),
        "dev_rows": sum(row["split"] == "dev" for row in output),
        "selected_by_stratum": dict(sorted(strata.items())),
        "selected_ordered_item_ids_sha256": canonical_sha256(selected_ids),
        "output_token_count": sum(row["output_token_count"] for row in output),
        "key_positive_token_count": sum(sum(row["key_prior_target"]) for row in output),
        "complete_positive_token_count": sum(
            sum(row["complete_prior_target"]) for row in output
        ),
        "complete_supervised_token_count": sum(
            sum(row["complete_prior_mask"]) for row in output
        ),
        "human_verified": False,
        "feature_extraction_started": False,
        "training_started": False,
    }
    return output, report


__all__ = [
    "DEFAULT_NAMESPACE",
    "EVALUATION_SCHEMA",
    "LABEL_NAME",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PROTOCOL_SCHEMA",
    "ROW_SCHEMA",
    "build_blind_shards_v16",
    "build_hidden_controls_v16",
    "construct_silver_rows_v16",
    "evaluate_role_scale_v16",
    "public_package_item_v16",
    "select_scale_rows_v16",
]
