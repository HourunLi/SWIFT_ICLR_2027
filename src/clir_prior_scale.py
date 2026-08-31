"""Deterministic dependency-graph smoke for CLIR Key/Complete priors.

The annotators describe causal edges and conclusion/flaw anchors.  Code derives
Key and Complete sets from those annotations, avoiding the ambiguous request to
guess a minimum Complete set directly.  This module has no provider client and
does not publish trainable labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Integral
from typing import Any, Mapping, Sequence

from src.clir_smoke import canonical_sha256, stable_priority


PROPOSAL_SCHEMA = "clir-prior-dependency-smoke-natural-v8"
PACKAGE_SCHEMA = "clir-prior-dependency-smoke-package-v8"
PRIVATE_SCHEMA = "clir-prior-dependency-smoke-private-index-v8"
LABEL_SCHEMA = "clir-prior-dependency-smoke-label-v8"
REPORT_SCHEMA = "clir-prior-dependency-smoke-gate-v8"

ELIGIBILITY_VALUES = {
    "usable",
    "no_auditable_reasoning",
    "insufficient_unitization",
}
PATH_STATUS_VALUES = {"supported", "flawed", "uncertain"}
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


def derive_prior_targets(
    *,
    material_unit_indices: Sequence[int],
    conclusion_unit_indices: Sequence[int],
    dependency_edges: Sequence[Sequence[int]],
    path_status: str,
    first_flaw_unit_index: int | None,
) -> tuple[list[int], list[int]]:
    """Derive Key and Complete from a forward dependency graph.

    Complete is the backward transitive closure of the conclusion anchors.
    For a supported path, Key is the conclusion anchor set.  For a flawed path,
    Key is the first causal flaw on that closure.  Uncertain paths are not
    trainable and therefore have an empty Key.
    """

    valid = {int(index) for index in material_unit_indices}
    conclusions = [int(index) for index in conclusion_unit_indices]
    if not conclusions or any(index not in valid for index in conclusions):
        raise ValueError("usable graph requires valid conclusion units")
    if conclusions != sorted(set(conclusions)):
        raise ValueError("conclusion_unit_indices must be sorted and unique")

    parents: dict[int, set[int]] = defaultdict(set)
    seen_edges: set[tuple[int, int]] = set()
    for edge in dependency_edges:
        if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)):
            raise ValueError("each dependency edge must be [parent, child]")
        if len(edge) != 2:
            raise ValueError("each dependency edge must contain two indices")
        parent, child = edge
        if (
            isinstance(parent, bool)
            or isinstance(child, bool)
            or not isinstance(parent, Integral)
            or not isinstance(child, Integral)
        ):
            raise ValueError("dependency edge indices must be integers")
        pair = (int(parent), int(child))
        if pair[0] not in valid or pair[1] not in valid:
            raise ValueError("dependency edge references a missing material unit")
        if pair[0] >= pair[1]:
            raise ValueError("dependency edges must point forward in unit order")
        if pair in seen_edges:
            raise ValueError("dependency edges must be unique")
        seen_edges.add(pair)
        parents[pair[1]].add(pair[0])

    complete = set(conclusions)
    frontier = list(conclusions)
    while frontier:
        child = frontier.pop()
        for parent in parents.get(child, set()):
            if parent not in complete:
                complete.add(parent)
                frontier.append(parent)

    if path_status == "supported":
        if first_flaw_unit_index is not None:
            raise ValueError("supported path must not set first_flaw_unit_index")
        key = set(conclusions)
    elif path_status == "flawed":
        if (
            isinstance(first_flaw_unit_index, bool)
            or not isinstance(first_flaw_unit_index, Integral)
        ):
            raise ValueError("flawed path requires an integer first flaw")
        flaw = int(first_flaw_unit_index)
        if flaw not in complete:
            raise ValueError("first flaw must lie on the derived Complete closure")
        key = {flaw}
    elif path_status == "uncertain":
        if first_flaw_unit_index is not None:
            raise ValueError("uncertain path must not set first_flaw_unit_index")
        key = set()
    else:
        raise ValueError(f"unsupported path_status: {path_status!r}")

    return sorted(key), sorted(complete)


def validate_prior_annotation(
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

    units = _material_units(item)
    valid = {unit["unit_index"] for unit in units}
    conclusions = _index_list(
        annotation.get("conclusion_unit_indices"),
        field="conclusion_unit_indices",
        valid=valid,
    )
    edges = annotation.get("dependency_edges")
    if not isinstance(edges, list):
        raise ValueError("dependency_edges must be a JSON array")
    flaw = annotation.get("first_flaw_unit_index")
    path_status = annotation.get("path_status")

    if eligibility != "usable":
        if conclusions or edges or flaw is not None or path_status is not None:
            raise ValueError("ineligible annotation must use empty graph fields")
        key: list[int] = []
        complete: list[int] = []
        normalized_edges: list[list[int]] = []
    else:
        if path_status not in PATH_STATUS_VALUES:
            raise ValueError("usable annotation has invalid path_status")
        normalized_edges = []
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("dependency edge must be [parent, child]")
            parent_child = _index_list(
                edge, field="dependency_edge", valid=valid
            )
            # _index_list sorts, which is correct because edges must point forward.
            if len(parent_child) != 2:
                raise ValueError("dependency edge endpoints must be distinct")
            normalized_edges.append(parent_child)
        if [tuple(edge) for edge in normalized_edges] != sorted(
            set(map(tuple, normalized_edges))
        ):
            raise ValueError("dependency_edges must be sorted and unique")
        key, complete = derive_prior_targets(
            material_unit_indices=sorted(valid),
            conclusion_unit_indices=conclusions,
            dependency_edges=normalized_edges,
            path_status=str(path_status),
            first_flaw_unit_index=flaw,
        )
        if path_status == "uncertain" and confidence != "low":
            raise ValueError("uncertain path must use low confidence")

    return {
        "schema_version": LABEL_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "eligibility": str(eligibility),
        "path_status": path_status,
        "conclusion_unit_indices": conclusions,
        "dependency_edges": normalized_edges,
        "first_flaw_unit_index": flaw,
        "key_unit_indices": key,
        "complete_unit_indices": complete,
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


def target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation["eligibility"],
        annotation.get("path_status"),
        tuple(annotation.get("key_unit_indices", [])),
        tuple(annotation.get("complete_unit_indices", [])),
    )


def select_prior_smoke_rows(
    *,
    materialized_rows: Sequence[Mapping[str, Any]],
    excluded_query_ids: set[str],
    excluded_cluster_ids: set[str],
    selection: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        (str(entry["source"]), str(entry["checker_status"])): int(entry["count"])
        for entry in selection["strata"]
    }
    if sum(quotas.values()) != int(selection["natural_count"]):
        raise ValueError("Prior smoke strata do not sum to natural_count")
    min_claims = int(selection["minimum_material_claims"])
    max_claims = int(selection["maximum_material_claims"])

    by_stratum_query: dict[
        tuple[str, str], dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
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
                    "clir-prior-v8-candidate", query_id, row["id"]
                ),
            )
            one_per_query.append(chosen)
        candidates[stratum] = sorted(
            one_per_query,
            key=lambda row: stable_priority(
                "clir-prior-v8-query",
                stratum[0],
                stratum[1],
                row["query_id"],
                row["id"],
            ),
        )

    # Start with the scarcest fixed stratum, then use the protocol order.
    ordered_strata = sorted(
        quotas,
        key=lambda stratum: (
            len(candidates.get(stratum, [])),
            [tuple((str(x["source"]), str(x["checker_status"]))) for x in selection["strata"]].index(stratum),
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
            proposal_id = stable_priority("clir-prior-v8-proposal", row["id"])
            selected.append(
                {
                    "schema_version": PROPOSAL_SCHEMA,
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
                        "clir-prior-v8-query",
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
                f"insufficient Prior smoke capacity for {stratum}: "
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
            "supported",
            [1],
            [[0, 1]],
            None,
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
            "supported",
            [2],
            [[0, 2]],
            None,
        ),
        (
            "wrapper_and_plan",
            "What is 8 times 6?",
            [
                "We need to multiply the two numbers.",
                "The problem asks for 8 times 6.",
                "Computing gives 8×6=48.",
                "Thus the answer is 48.",
            ],
            "supported",
            [2],
            [],
            None,
        ),
        (
            "flawed_chain",
            "Mia has 7 red and 5 blue beads, then doubles the total. How many beads does she have?",
            [
                "Adding the colors gives 7+5=13 beads.",
                "Doubling gives 13×2=26 beads.",
                "Therefore Mia has 26 beads.",
            ],
            "flawed",
            [1],
            [[0, 1]],
            0,
        ),
        (
            "two_parents",
            "A shop sold 9 pens in the morning and 6 later, with 2 returned. How many stayed sold?",
            [
                "The two sales periods total 9+6=15 pens.",
                "There were 2 returned pens.",
                "Subtracting returns gives 15-2=13 pens.",
                "The answer is 13 pens.",
            ],
            "supported",
            [2],
            [[0, 2], [1, 2]],
            None,
        ),
        (
            "duplicate_claim",
            "A pack has 5 cards and there are 4 packs. How many cards are there?",
            [
                "Four packs of five cards gives 4×5=20 cards.",
                "In other words, the packs contain twenty cards altogether.",
                "Using the product, the requested total is 20 cards.",
                "Final answer: 20.",
            ],
            "supported",
            [0],
            [],
            None,
        ),
    ]
    output = []
    for name, question, texts, status, conclusions, edges, flaw in definitions:
        item_id = stable_priority("clir-prior-v8-control", name)
        units = [
            {"unit_index": index, "kind": "material_claim", "text": text}
            for index, text in enumerate(texts)
        ]
        key, complete = derive_prior_targets(
            material_unit_indices=list(range(len(texts))),
            conclusion_unit_indices=conclusions,
            dependency_edges=edges,
            path_status=status,
            first_flaw_unit_index=flaw,
        )
        output.append(
            {
                "schema_version": PACKAGE_SCHEMA,
                "item_id": item_id,
                "question": question,
                "response": "\n".join(texts),
                "units": units,
                "expected_signature": ("usable", status, tuple(key), tuple(complete)),
            }
        )
    return output


def build_blind_packages(
    proposals: Sequence[Mapping[str, Any]], *, repeat_count_a: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    natural = []
    for row in proposals:
        natural.append(
            {
                "schema_version": PACKAGE_SCHEMA,
                "item_id": str(row["proposal_id"]),
                "question": str(row["question"]),
                "response": str(row["response"]),
                "units": [dict(unit) for unit in row["units"]],
            }
        )
    controls = _control_items()
    repeats = sorted(
        natural,
        key=lambda row: stable_priority("clir-prior-v8-repeat", row["item_id"]),
    )[:repeat_count_a]

    private: list[dict[str, Any]] = []
    package_a = [dict(row) for row in natural]
    package_b = [dict(row) for row in natural]
    for row in natural:
        for annotator in ("a", "b"):
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "item_id": row["item_id"],
                    "kind": "natural",
                    "natural_item_id": row["item_id"],
                }
            )
    for control in controls:
        public = {key: value for key, value in control.items() if key != "expected_signature"}
        package_a.append(dict(public))
        package_b.append(dict(public))
        for annotator in ("a", "b"):
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "item_id": control["item_id"],
                    "kind": "control",
                    "expected_signature": list(control["expected_signature"]),
                }
            )
    for parent in repeats:
        repeat_id = stable_priority("clir-prior-v8-repeat-item", parent["item_id"])
        repeated = dict(parent)
        repeated["item_id"] = repeat_id
        package_a.append(repeated)
        private.append(
            {
                "schema_version": PRIVATE_SCHEMA,
                "annotator": "a",
                "item_id": repeat_id,
                "kind": "repeat",
                "natural_item_id": parent["item_id"],
            }
        )

    package_a.sort(key=lambda row: stable_priority("clir-prior-v8-package-a", row["item_id"]))
    package_b.sort(key=lambda row: stable_priority("clir-prior-v8-package-b", row["item_id"]))
    private.sort(key=lambda row: (row["annotator"], row["item_id"]))
    report = {
        "natural": len(natural),
        "controls_per_annotator": len(controls),
        "a_repeats": len(repeats),
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


def evaluate_prior_labels(
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
        if len(packages[annotator]) != len(package_a if annotator == "a" else package_b):
            raise ValueError("package item IDs are not unique")
        by_id: dict[str, dict[str, Any]] = {}
        for row in raw_labels[annotator]:
            item_id = str(row.get("item_id"))
            if item_id in by_id:
                raise ValueError(f"duplicate label item_id for annotator {annotator}")
            if item_id not in packages[annotator]:
                raise ValueError(f"unknown label item_id for annotator {annotator}")
            by_id[item_id] = validate_prior_annotation(row, packages[annotator][item_id])
        if set(by_id) != set(packages[annotator]):
            missing = sorted(set(packages[annotator]) - set(by_id))
            raise ValueError(f"annotator {annotator} label population incomplete: {missing[:3]}")
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
        (a, b, packages["a"][item_id])
        for item_id, a, b in zip(natural_ids, natural_a, natural_b)
        if a["eligibility"] == b["eligibility"] == "usable"
    ]
    path_agree = sum(a["path_status"] == b["path_status"] for a, b, _ in usable_pairs)
    key_f1 = sum(_set_f1(a["key_unit_indices"], b["key_unit_indices"]) for a, b, _ in usable_pairs) / max(1, len(usable_pairs))
    complete_f1 = sum(_set_f1(a["complete_unit_indices"], b["complete_unit_indices"]) for a, b, _ in usable_pairs) / max(1, len(usable_pairs))
    exact_consensus = [
        (a, b, item)
        for a, b, item in usable_pairs
        if target_signature(a) == target_signature(b)
        and a["confidence"] != "low"
        and b["confidence"] != "low"
    ]
    common_flawed = [
        (a, b) for a, b, _ in usable_pairs if a["path_status"] == b["path_status"] == "flawed"
    ]
    flaw_exact = sum(
        a["first_flaw_unit_index"] == b["first_flaw_unit_index"]
        for a, b in common_flawed
    )

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
                expected[0], expected[1], tuple(expected[2]), tuple(expected[3])
            )
            passed += target_signature(normalized[annotator][row["item_id"]]) == expected_signature
        control_scores[annotator] = {"passed": passed, "total": len(controls), "rate": passed / max(1, len(controls))}

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
            material = {unit["unit_index"] for unit in _material_units(packages[annotator][item_id])}
            all_selected += set(label["complete_unit_indices"]) == material
        all_material_rates[annotator] = all_selected / max(1, usable)

    metrics = {
        "natural_denominator": len(natural_ids),
        "eligibility_agreement": eligibility_agree / max(1, len(natural_ids)),
        "common_usable": len(usable_pairs),
        "path_agreement": path_agree / max(1, len(usable_pairs)),
        "key_macro_f1": key_f1,
        "complete_macro_f1": complete_f1,
        "exact_training_consensus": len(exact_consensus),
        "exact_training_consensus_rate": len(exact_consensus) / max(1, len(natural_ids)),
        "minimum_adjudication_fraction": 1.0 - len(exact_consensus) / max(1, len(natural_ids)),
        "common_flawed": len(common_flawed),
        "first_flaw_exact_rate": flaw_exact / max(1, len(common_flawed)),
        "complete_all_material_rate": all_material_rates,
        "controls": control_scores,
        "a_self_repeat": {
            "passed": repeat_passed,
            "total": len(repeats),
            "rate": repeat_passed / max(1, len(repeats)),
        },
    }
    checks = {
        "eligibility": metrics["eligibility_agreement"] >= float(gates["eligibility_agreement_min"]),
        "common_usable": metrics["common_usable"] >= int(gates["common_usable_min"]),
        "path": metrics["path_agreement"] >= float(gates["path_agreement_min"]),
        "key_f1": metrics["key_macro_f1"] >= float(gates["key_macro_f1_min"]),
        "complete_f1": metrics["complete_macro_f1"] >= float(gates["complete_macro_f1_min"]),
        "exact_consensus": (
            metrics["exact_training_consensus"] >= int(gates["exact_training_consensus_min"])
            and metrics["exact_training_consensus_rate"] >= float(gates["exact_training_consensus_rate_min"])
        ),
        "adjudication_fraction": metrics["minimum_adjudication_fraction"] <= float(gates["minimum_adjudication_fraction_max"]),
        "flaw_support": metrics["common_flawed"] >= int(gates["common_flawed_min"]),
        "flaw_exact": metrics["first_flaw_exact_rate"] >= float(gates["first_flaw_exact_rate_min"]),
        "controls_a": control_scores["a"]["rate"] == 1.0,
        "controls_b": control_scores["b"]["rate"] == 1.0,
        "self_repeat_a": metrics["a_self_repeat"]["rate"] >= float(gates["self_repeat_min"]),
        "anti_all_material_a": all_material_rates["a"] <= float(gates["complete_all_material_rate_max"]),
        "anti_all_material_b": all_material_rates["b"] <= float(gates["complete_all_material_rate_max"]),
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_PRIOR_DEPENDENCY_SMOKE_V8" if all(checks.values()) else "STOP_PRIOR_DEPENDENCY_SMOKE_V8_RAW_GATE_FAILURE",
        "metrics": metrics,
        "gates": {name: {"pass": passed} for name, passed in checks.items()},
        "scale_annotation_allowed": all(checks.values()),
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "claim_boundary": "dual-AI dependency-target operability only; not Gold, factual accuracy, learnability, gate efficacy, or Best-of-N evidence",
    }
