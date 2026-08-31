"""Fresh partial-consensus smoke for CLIR Key/Complete priors.

Annotators independently mark direct Key and Complete unit sets.  The raw gate
measures their agreement without adjudication.  A later scale stage may keep
exact non-low Key consensus and may mask only Complete units on which the two
annotators disagree.  This module has no provider client, feature extraction,
or training publication path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Integral
from statistics import mean
from typing import Any, Mapping, Sequence

from src.clir_smoke import canonical_sha256, stable_priority


PROPOSAL_SCHEMA = "clir-prior-partial-smoke-natural-v9"
PACKAGE_SCHEMA = "clir-prior-partial-smoke-package-v9"
PRIVATE_SCHEMA = "clir-prior-partial-smoke-private-index-v9"
LABEL_SCHEMA = "clir-prior-partial-smoke-label-v9"
REPORT_SCHEMA = "clir-prior-partial-smoke-gate-v9"

ELIGIBILITY_VALUES = {
    "usable",
    "no_auditable_reasoning",
    "insufficient_unitization",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}


def _material_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = row.get("units")
    if not isinstance(units, list):
        raise ValueError(f"{row.get('id', row.get('item_id'))}: units must be a list")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for unit in units:
        if not isinstance(unit, Mapping) or unit.get("kind") != "material_claim":
            continue
        index = unit.get("unit_index")
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("material unit index must be an integer")
        integer = int(index)
        if integer in seen:
            raise ValueError("material unit indices must be unique")
        text = unit.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("material unit text must be non-empty")
        seen.add(integer)
        output.append({"unit_index": integer, "kind": "material_claim", "text": text})
    output.sort(key=lambda unit: unit["unit_index"])
    if not output:
        raise ValueError("row has no material units")
    return output


def _index_list(value: Any, *, field: str, valid: set[int]) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result: list[int] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, Integral):
            raise ValueError(f"{field} must contain integers")
        index = int(element)
        if index not in valid:
            raise ValueError(f"{field} references a missing material unit: {index}")
        result.append(index)
    if result != sorted(set(result)):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def validate_partial_prior_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation must be a JSON object")
    if annotation.get("item_id") != item.get("item_id"):
        raise ValueError("annotation item_id does not match package item")
    eligibility = annotation.get("eligibility")
    if eligibility not in ELIGIBILITY_VALUES:
        raise ValueError("invalid eligibility")
    confidence = annotation.get("confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("invalid confidence")
    rationale = annotation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty")

    valid = {unit["unit_index"] for unit in _material_units(item)}
    key = _index_list(
        annotation.get("key_unit_indices"), field="key_unit_indices", valid=valid
    )
    complete = _index_list(
        annotation.get("complete_unit_indices"),
        field="complete_unit_indices",
        valid=valid,
    )
    if eligibility == "usable":
        if not key or not complete:
            raise ValueError("usable annotation requires non-empty Key and Complete")
        if not set(key).issubset(complete):
            raise ValueError("Key must be a subset of Complete")
    elif key or complete:
        raise ValueError("ineligible annotation must use empty Key and Complete")

    return {
        "schema_version": LABEL_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "eligibility": str(eligibility),
        "key_unit_indices": key,
        "complete_unit_indices": complete,
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


def target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation["eligibility"],
        tuple(annotation.get("key_unit_indices", [])),
        tuple(annotation.get("complete_unit_indices", [])),
    )


def derive_partial_consensus_unit_targets(
    left: Mapping[str, Any], right: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive unit-level partial supervision without adjudicating a dispute."""

    valid = {unit["unit_index"] for unit in _material_units(item)}
    common_nonlow = (
        left.get("eligibility") == right.get("eligibility") == "usable"
        and left.get("confidence") != "low"
        and right.get("confidence") != "low"
    )
    if not common_nonlow:
        return {
            "common_nonlow_usable": False,
            "key_trainable": False,
            "key_positive_units": [],
            "key_covered_units": [],
            "complete_trainable": False,
            "complete_positive_units": [],
            "complete_negative_units": [],
            "complete_ambiguous_units": [],
            "complete_covered_units": [],
        }

    left_key = set(left["key_unit_indices"])
    right_key = set(right["key_unit_indices"])
    left_complete = set(left["complete_unit_indices"])
    right_complete = set(right["complete_unit_indices"])
    key_trainable = left_key == right_key and bool(left_key)
    complete_positive = left_complete & right_complete
    complete_ambiguous = left_complete ^ right_complete
    complete_negative = valid - (left_complete | right_complete)
    complete_covered = valid - complete_ambiguous
    return {
        "common_nonlow_usable": True,
        "key_trainable": key_trainable,
        "key_positive_units": sorted(left_key) if key_trainable else [],
        "key_covered_units": sorted(valid) if key_trainable else [],
        "complete_trainable": bool(complete_positive),
        "complete_positive_units": sorted(complete_positive),
        "complete_negative_units": sorted(complete_negative),
        "complete_ambiguous_units": sorted(complete_ambiguous),
        "complete_covered_units": sorted(complete_covered),
    }


def select_partial_prior_smoke_rows(
    *,
    materialized_rows: Sequence[Mapping[str, Any]],
    excluded_query_ids: set[str],
    excluded_cluster_ids: set[str],
    selection: Mapping[str, Any],
    namespace: str = "clir-prior-v9",
    proposal_schema: str = PROPOSAL_SCHEMA,
    version_label: str = "Prior v9",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        (str(entry["source"]), str(entry["checker_status"])): int(entry["count"])
        for entry in selection["strata"]
    }
    if sum(quotas.values()) != int(selection["natural_count"]):
        raise ValueError(f"{version_label} strata do not sum to natural_count")
    min_claims = int(selection["minimum_material_claims"])
    max_claims = int(selection["maximum_material_claims"])

    by_stratum_query: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    rejection = Counter()
    for row in materialized_rows:
        stratum = (str(row.get("source")), str(row.get("checker_status")))
        if stratum not in quotas:
            rejection["outside_strata"] += 1
            continue
        if row.get("acquisition_split") != "train_acquisition":
            rejection["not_train_acquisition"] += 1
            continue
        if str(row.get("query_id")) in excluded_query_ids:
            rejection["excluded_query"] += 1
            continue
        if str(row.get("cluster_id")) in excluded_cluster_ids:
            rejection["excluded_cluster"] += 1
            continue
        if not row.get("eligible_for_supervision"):
            rejection["not_supervision_eligible"] += 1
            continue
        if row.get("unitization_status") != "ok":
            rejection["unitization"] += 1
            continue
        if row.get("finish_reason") != "stop":
            rejection["finish_reason"] += 1
            continue
        claim_count = int(row.get("material_claim_count", 0))
        if not min_claims <= claim_count <= max_claims:
            rejection["material_claim_count"] += 1
            continue
        _material_units(row)
        by_stratum_query[stratum][str(row["query_id"])].append(row)

    candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for stratum, by_query in by_stratum_query.items():
        one_per_query = []
        for query_id, rows in by_query.items():
            chosen = min(
                rows,
                key=lambda row: stable_priority(
                    f"{namespace}-candidate", query_id, row["id"]
                ),
            )
            one_per_query.append(chosen)
        candidates[stratum] = sorted(
            one_per_query,
            key=lambda row: stable_priority(
                f"{namespace}-query",
                stratum[0],
                stratum[1],
                row["query_id"],
                row["id"],
            ),
        )

    protocol_order = [
        (str(entry["source"]), str(entry["checker_status"]))
        for entry in selection["strata"]
    ]
    ordered_strata = sorted(
        quotas,
        key=lambda stratum: (
            len(candidates.get(stratum, [])),
            protocol_order.index(stratum),
        ),
    )
    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    for stratum in ordered_strata:
        selected_here = 0
        for row in candidates.get(stratum, []):
            query_id = str(row["query_id"])
            cluster_id = str(row["cluster_id"])
            if query_id in used_queries or cluster_id in used_clusters:
                continue
            proposal_id = stable_priority(f"{namespace}-proposal", row["id"])
            selected.append(
                {
                    "schema_version": proposal_schema,
                    "proposal_id": proposal_id,
                    "trajectory_id": str(row["id"]),
                    "query_id": query_id,
                    "cluster_id": cluster_id,
                    "source": str(row["source"]),
                    "source_record_id": row.get("source_record_id"),
                    "checker_status": str(row["checker_status"]),
                    "candidate_index": int(row["candidate_index"]),
                    "question": str(row["question"]),
                    "response": str(row["response"]),
                    "material_claim_count": int(row["material_claim_count"]),
                    "output_token_count": int(row["output_token_count"]),
                    "units": _material_units(row),
                    "selection_priority": stable_priority(
                        f"{namespace}-query",
                        stratum[0],
                        stratum[1],
                        query_id,
                        row["id"],
                    ),
                }
            )
            used_queries.add(query_id)
            used_clusters.add(cluster_id)
            selected_here += 1
            if selected_here == quotas[stratum]:
                break
        if selected_here != quotas[stratum]:
            raise ValueError(
                f"insufficient {version_label} capacity for {stratum}: "
                f"{selected_here}/{quotas[stratum]}"
            )

    selected.sort(key=lambda row: row["proposal_id"])
    counts = Counter((row["source"], row["checker_status"]) for row in selected)
    report = {
        "natural_selected": len(selected),
        "unique_queries": len(used_queries),
        "unique_clusters": len(used_clusters),
        "selected_by_stratum": {
            f"{source}:{status}": count
            for (source, status), count in sorted(counts.items())
        },
        "available_query_counts": {
            f"{source}:{status}": len(candidates.get((source, status), []))
            for source, status in sorted(quotas)
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "ordered_rows_sha256": canonical_sha256(selected),
    }
    return selected, report


def _control_items() -> list[dict[str, Any]]:
    definitions = [
        (
            "linear",
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
            "unused_branch",
            "A ticket costs $7 and three snacks cost $3 each. What is the total cost?",
            [
                "Three snacks cost 3×3=9 dollars.",
                "The letters in the word ticket total 6.",
                "Adding the ticket gives 9+7=16 dollars.",
                "The final answer is 16 dollars.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "plan_and_wrapper",
            "What is 8 times 6?",
            [
                "We need to multiply the two numbers.",
                "Computing gives 8×6=48.",
                "Thus the answer is 48.",
            ],
            "usable",
            [1],
            [1],
        ),
        (
            "flawed_chain",
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
            "two_inputs",
            "A shop sold 9 pens in the morning and 6 later, with 2 returned. How many stayed sold?",
            [
                "The two sales periods total 9+6=15 pens.",
                "There were 2 returned pens.",
                "Subtracting returns gives 15-2=13 pens.",
                "The answer is 13 pens.",
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
        item_id = stable_priority("clir-prior-v9-control", name)
        output.append(
            {
                "schema_version": PACKAGE_SCHEMA,
                "item_id": item_id,
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


def build_blind_packages(
    proposals: Sequence[Mapping[str, Any]],
    *,
    repeat_count_a: int,
    repeat_count_b: int = 0,
    namespace: str = "clir-prior-v9",
    package_schema: str = PACKAGE_SCHEMA,
    private_schema: str = PRIVATE_SCHEMA,
    control_items: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    natural = [
        {
            "schema_version": package_schema,
            "item_id": str(row["proposal_id"]),
            "question": str(row["question"]),
            "response": str(row["response"]),
            "units": [dict(unit) for unit in row["units"]],
        }
        for row in proposals
    ]
    controls = [dict(row) for row in (control_items or _control_items())]
    repeat_namespace_a = (
        "clir-prior-v9-repeat"
        if namespace == "clir-prior-v9"
        else f"{namespace}-repeat-a"
    )
    repeat_item_namespace_a = (
        "clir-prior-v9-repeat-item"
        if namespace == "clir-prior-v9"
        else f"{namespace}-repeat-item-a"
    )
    repeats_a = sorted(
        natural,
        key=lambda row: stable_priority(repeat_namespace_a, row["item_id"]),
    )[:repeat_count_a]
    repeats_b = sorted(
        natural,
        key=lambda row: stable_priority(f"{namespace}-repeat-b", row["item_id"]),
    )[:repeat_count_b]

    private: list[dict[str, Any]] = []
    package_a = [dict(row) for row in natural]
    package_b = [dict(row) for row in natural]
    for row in natural:
        for annotator in ("a", "b"):
            private.append(
                {
                    "schema_version": private_schema,
                    "annotator": annotator,
                    "item_id": row["item_id"],
                    "kind": "natural",
                    "natural_item_id": row["item_id"],
                }
            )
    for control in controls:
        public = {
            key: value for key, value in control.items() if key != "expected_signature"
        }
        package_a.append(dict(public))
        package_b.append(dict(public))
        for annotator in ("a", "b"):
            private.append(
                {
                    "schema_version": private_schema,
                    "annotator": annotator,
                    "item_id": control["item_id"],
                    "kind": "control",
                    "expected_signature": list(control["expected_signature"]),
                }
            )
    for annotator, repeats, package in (
        ("a", repeats_a, package_a),
        ("b", repeats_b, package_b),
    ):
        for parent in repeats:
            repeat_item_namespace = (
                repeat_item_namespace_a
                if annotator == "a"
                else f"{namespace}-repeat-item-b"
            )
            repeat_id = stable_priority(repeat_item_namespace, parent["item_id"])
            repeated = dict(parent)
            repeated["item_id"] = repeat_id
            package.append(repeated)
            private.append(
                {
                    "schema_version": private_schema,
                    "annotator": annotator,
                    "item_id": repeat_id,
                    "kind": "repeat",
                    "natural_item_id": parent["item_id"],
                }
            )

    package_a.sort(
        key=lambda row: stable_priority(f"{namespace}-package-a", row["item_id"])
    )
    package_b.sort(
        key=lambda row: stable_priority(f"{namespace}-package-b", row["item_id"])
    )
    private.sort(key=lambda row: (row["annotator"], row["item_id"]))
    report = {
        "natural": len(natural),
        "controls_per_annotator": len(controls),
        "a_repeats": len(repeats_a),
        "b_repeats": len(repeats_b),
        "package_a_rows": len(package_a),
        "package_b_rows": len(package_b),
        "package_a_ordered_rows_sha256": canonical_sha256(package_a),
        "package_b_ordered_rows_sha256": canonical_sha256(package_b),
        "private_index_ordered_rows_sha256": canonical_sha256(private),
    }
    return package_a, package_b, private, report


def _set_f1(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def evaluate_partial_prior_labels(
    *,
    package_a: Sequence[Mapping[str, Any]],
    package_b: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    packages = {
        "a": {str(row["item_id"]): row for row in package_a},
        "b": {str(row["item_id"]): row for row in package_b},
    }
    raw_labels = {"a": labels_a, "b": labels_b}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for annotator in ("a", "b"):
        expected_count = len(package_a if annotator == "a" else package_b)
        if len(packages[annotator]) != expected_count:
            raise ValueError("package item IDs are not unique")
        by_id: dict[str, dict[str, Any]] = {}
        for row in raw_labels[annotator]:
            item_id = str(row.get("item_id"))
            if item_id in by_id:
                raise ValueError(f"duplicate label item_id for annotator {annotator}")
            if item_id not in packages[annotator]:
                raise ValueError(f"unknown label item_id for annotator {annotator}")
            by_id[item_id] = validate_partial_prior_annotation(
                row, packages[annotator][item_id]
            )
        if set(by_id) != set(packages[annotator]):
            missing = sorted(set(packages[annotator]) - set(by_id))
            raise ValueError(
                f"annotator {annotator} label population incomplete: {missing[:3]}"
            )
        normalized[annotator] = by_id

    private_by_annotator = {
        annotator: {
            str(row["item_id"]): row
            for row in private_index
            if row["annotator"] == annotator
        }
        for annotator in ("a", "b")
    }
    natural_ids = sorted(
        item_id
        for item_id, row in private_by_annotator["a"].items()
        if row["kind"] == "natural"
    )
    if natural_ids != sorted(
        item_id
        for item_id, row in private_by_annotator["b"].items()
        if row["kind"] == "natural"
    ):
        raise ValueError("A/B natural populations differ")

    natural_a = [normalized["a"][item_id] for item_id in natural_ids]
    natural_b = [normalized["b"][item_id] for item_id in natural_ids]
    eligibility_agree = sum(
        a["eligibility"] == b["eligibility"] for a, b in zip(natural_a, natural_b)
    )
    usable_pairs = [
        (item_id, a, b, packages["a"][item_id])
        for item_id, a, b in zip(natural_ids, natural_a, natural_b)
        if a["eligibility"] == b["eligibility"] == "usable"
    ]
    common_nonlow = [
        row
        for row in usable_pairs
        if row[1]["confidence"] != "low" and row[2]["confidence"] != "low"
    ]
    key_f1 = sum(
        _set_f1(a["key_unit_indices"], b["key_unit_indices"])
        for _, a, b, _ in usable_pairs
    ) / max(1, len(usable_pairs))
    complete_f1 = sum(
        _set_f1(a["complete_unit_indices"], b["complete_unit_indices"])
        for _, a, b, _ in usable_pairs
    ) / max(1, len(usable_pairs))

    key_exact_nonlow = 0
    complete_exact_nonlow = 0
    exact_joint_nonlow = 0
    complete_consensus_rows = 0
    partial_paired_rows = 0
    material_units = 0
    complete_agree_units = 0
    complete_intersection_units = 0
    complete_union_units = 0
    row_mask_coverage: list[float] = []
    complete_relations = Counter()
    for _, a, b, item in common_nonlow:
        material = {unit["unit_index"] for unit in _material_units(item)}
        left_key = set(a["key_unit_indices"])
        right_key = set(b["key_unit_indices"])
        left_complete = set(a["complete_unit_indices"])
        right_complete = set(b["complete_unit_indices"])
        key_exact = left_key == right_key and bool(left_key)
        complete_exact = left_complete == right_complete
        complete_intersection = left_complete & right_complete
        complete_union = left_complete | right_complete
        complete_ambiguous = left_complete ^ right_complete
        key_exact_nonlow += key_exact
        complete_exact_nonlow += complete_exact
        exact_joint_nonlow += key_exact and complete_exact
        complete_consensus_rows += bool(complete_intersection)
        partial_paired_rows += key_exact and bool(complete_intersection)
        material_units += len(material)
        complete_agree_units += len(material) - len(complete_ambiguous)
        complete_intersection_units += len(complete_intersection)
        complete_union_units += len(complete_union)
        row_mask_coverage.append(
            (len(material) - len(complete_ambiguous)) / max(1, len(material))
        )
        if left_complete == right_complete:
            complete_relations["equal"] += 1
        elif left_complete < right_complete:
            complete_relations["a_strict_subset_b"] += 1
        elif right_complete < left_complete:
            complete_relations["b_strict_subset_a"] += 1
        else:
            complete_relations["overlap_or_disjoint"] += 1

    control_scores = {}
    for annotator in ("a", "b"):
        controls = [
            row
            for row in private_by_annotator[annotator].values()
            if row["kind"] == "control"
        ]
        passed = 0
        for row in controls:
            expected = row["expected_signature"]
            expected_signature = (
                expected[0],
                tuple(expected[1]),
                tuple(expected[2]),
            )
            passed += (
                target_signature(normalized[annotator][row["item_id"]])
                == expected_signature
            )
        control_scores[annotator] = {
            "passed": passed,
            "total": len(controls),
            "rate": passed / max(1, len(controls)),
        }

    repeats = [
        row for row in private_by_annotator["a"].values() if row["kind"] == "repeat"
    ]
    repeat_passed = sum(
        target_signature(normalized["a"][row["item_id"]])
        == target_signature(normalized["a"][row["natural_item_id"]])
        for row in repeats
    )

    all_material_rates = {}
    for annotator in ("a", "b"):
        all_selected = 0
        usable = 0
        for item_id in natural_ids:
            label = normalized[annotator][item_id]
            if label["eligibility"] != "usable":
                continue
            usable += 1
            material = {
                unit["unit_index"]
                for unit in _material_units(packages[annotator][item_id])
            }
            all_selected += set(label["complete_unit_indices"]) == material
        all_material_rates[annotator] = all_selected / max(1, usable)

    complete_unit_agreement = complete_agree_units / max(1, material_units)
    complete_positive_overlap = complete_intersection_units / max(
        1, complete_union_units
    )
    metrics = {
        "natural_denominator": len(natural_ids),
        "eligibility_agreement": eligibility_agree / max(1, len(natural_ids)),
        "common_usable": len(usable_pairs),
        "common_nonlow_usable": len(common_nonlow),
        "key_macro_f1": key_f1,
        "complete_macro_f1": complete_f1,
        "key_exact_nonlow_rows": key_exact_nonlow,
        "complete_exact_nonlow_rows_diagnostic": complete_exact_nonlow,
        "exact_joint_nonlow_rows_diagnostic": exact_joint_nonlow,
        "complete_nonempty_consensus_rows": complete_consensus_rows,
        "partial_paired_trainable_rows": partial_paired_rows,
        "complete_unit_decision_agreement": complete_unit_agreement,
        "complete_ambiguous_unit_fraction": 1.0 - complete_unit_agreement,
        "complete_positive_intersection_over_union": complete_positive_overlap,
        "complete_row_mask_coverage_mean": (
            mean(row_mask_coverage) if row_mask_coverage else 0.0
        ),
        "complete_set_relations": dict(sorted(complete_relations.items())),
        "complete_all_material_rate": all_material_rates,
        "controls": control_scores,
        "a_self_repeat": {
            "passed": repeat_passed,
            "total": len(repeats),
            "rate": repeat_passed / max(1, len(repeats)),
        },
        "confidence": {
            "a": dict(sorted(Counter(a["confidence"] for a in natural_a).items())),
            "b": dict(sorted(Counter(b["confidence"] for b in natural_b).items())),
        },
    }
    checks = {
        "eligibility": metrics["eligibility_agreement"]
        >= float(gates["eligibility_agreement_min"]),
        "common_usable": metrics["common_usable"] >= int(gates["common_usable_min"]),
        "common_nonlow": metrics["common_nonlow_usable"]
        >= int(gates["common_nonlow_usable_min"]),
        "key_f1": metrics["key_macro_f1"] >= float(gates["key_macro_f1_min"]),
        "complete_f1": metrics["complete_macro_f1"]
        >= float(gates["complete_macro_f1_min"]),
        "key_exact_rows": metrics["key_exact_nonlow_rows"]
        >= int(gates["key_exact_nonlow_rows_min"]),
        "complete_consensus_rows": metrics["complete_nonempty_consensus_rows"]
        >= int(gates["complete_nonempty_consensus_rows_min"]),
        "partial_paired_rows": metrics["partial_paired_trainable_rows"]
        >= int(gates["partial_paired_trainable_rows_min"]),
        "complete_unit_agreement": metrics["complete_unit_decision_agreement"]
        >= float(gates["complete_unit_decision_agreement_min"]),
        "complete_ambiguity": metrics["complete_ambiguous_unit_fraction"]
        <= float(gates["complete_ambiguous_unit_fraction_max"]),
        "complete_positive_overlap": metrics[
            "complete_positive_intersection_over_union"
        ]
        >= float(gates["complete_positive_intersection_over_union_min"]),
        "complete_mask_coverage": metrics["complete_row_mask_coverage_mean"]
        >= float(gates["complete_row_mask_coverage_mean_min"]),
        "controls_a": control_scores["a"]["rate"] == 1.0,
        "controls_b": control_scores["b"]["rate"] == 1.0,
        "self_repeat_a": metrics["a_self_repeat"]["rate"]
        >= float(gates["self_repeat_min"]),
        "anti_all_material_a": all_material_rates["a"]
        <= float(gates["complete_all_material_rate_max"]),
        "anti_all_material_b": all_material_rates["b"]
        <= float(gates["complete_all_material_rate_max"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "PASS_PRIOR_PARTIAL_SMOKE_V9"
            if passed
            else "STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE"
        ),
        "metrics": metrics,
        "gates": {name: {"pass": value} for name, value in checks.items()},
        "scale_annotation_allowed": passed,
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "claim_boundary": "dual-AI partial-consensus target operability only; not Gold, factual accuracy, learnability, gate efficacy, or Best-of-N evidence",
    }


__all__ = [
    "CONFIDENCE_VALUES",
    "ELIGIBILITY_VALUES",
    "LABEL_SCHEMA",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "REPORT_SCHEMA",
    "build_blind_packages",
    "derive_partial_consensus_unit_targets",
    "evaluate_partial_prior_labels",
    "select_partial_prior_smoke_rows",
    "target_signature",
    "validate_partial_prior_annotation",
]
