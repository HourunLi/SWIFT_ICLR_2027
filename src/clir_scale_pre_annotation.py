"""CPU-only pre-annotation contracts for CLIR Consistency scale v6.

This module materializes the frozen raw token axis, applies the unchanged v5
mechanical filter, and constructs isolated A/B packages.  It deliberately has
no model-provider, label-finalization, feature-extraction, or training client.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from src.clir_smoke import (
    UNITIZER_VERSION,
    agreement_report,
    build_mechanical_consistency_proposals,
    canonical_sha256,
    check_numeric_response,
    consistency_item,
    stable_priority,
    tokenize_visible_response,
    unitize_exact_tokens,
    validate_annotation,
    validate_rollout_population,
)


PRE_ANNOTATION_AUTHORIZATION_SCHEMA = (
    "clir-consistency-scale-v6-pre-annotation-authorization"
)
MATERIALIZED_SCHEMA = "clir-consistency-scale-materialized-rollouts-v6"
PROPOSAL_SCHEMA = "clir-consistency-scale-mechanical-proposals-v6"
NATURAL_ITEM_SCHEMA = "clir-consistency-scale-natural-audit-items-v6"
PACKAGE_SCHEMA = "clir-consistency-scale-blind-package-v6"


def _counter_by(
    rows: Sequence[Mapping[str, Any]], field: str, value_field: str
) -> dict[str, dict[str, int]]:
    output: dict[str, Counter[str]] = {}
    for row in rows:
        key = str(row.get(field, "missing"))
        output.setdefault(key, Counter())[str(row.get(value_field, "missing"))] += 1
    return {
        key: dict(sorted(counts.items())) for key, counts in sorted(output.items())
    }


def materialize_scale_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    checker_version: str,
    unitizer_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen checker and exact-token unitizer on every raw audit row."""

    if unitizer_version != UNITIZER_VERSION:
        raise ValueError(
            f"scale-v6 requires {UNITIZER_VERSION}, got {unitizer_version}"
        )
    processed: list[dict[str, Any]] = []
    unitization_failures: Counter[str] = Counter()
    for raw in raw_rows:
        row = dict(raw)
        row["raw_reference_answer"] = str(row["reference_answer"])
        checker = check_numeric_response(
            response=str(row["response"]),
            raw_reference=row["raw_reference_answer"],
            source=str(row["source"]),
            finish_reason=row.get("finish_reason"),
            checker_version=checker_version,
        )
        # Scale-v6 explicitly treats parser failure as audit-only rather than a
        # negative label.  This is stricter than the generic smoke helper.
        if checker.get("checker_status") == "parse_failed":
            checker["eligible_for_supervision"] = False
        row.update(checker)
        try:
            mapping = tokenize_visible_response(
                tokenizer, str(row["response"]), row["output_token_ids"]
            )
            row["token_mapping_mode"] = mapping.pop("mapping_mode")
            row.update(
                unitize_exact_tokens(
                    response=str(row["response"]),
                    output_token_ids=row["output_token_ids"],
                    **mapping,
                )
            )
            row["unitization_status"] = "ok"
        except (KeyError, TypeError, ValueError) as exc:
            row["unitization_status"] = "failed"
            row["unitization_error"] = f"{type(exc).__name__}: {exc}"
            row["units"] = []
            row["material_claim_count"] = 0
            row["eligible_for_supervision"] = False
            unitization_failures[type(exc).__name__] += 1
        processed.append(row)

    report = {
        "rows": len(processed),
        "queries": len({str(row["query_id"]) for row in processed}),
        "unitization_ok": sum(
            row["unitization_status"] == "ok" for row in processed
        ),
        "unitization_failures": dict(sorted(unitization_failures.items())),
        "token_mapping_modes": dict(
            sorted(
                Counter(
                    str(row.get("token_mapping_mode", "failed"))
                    for row in processed
                ).items()
            )
        ),
        "checker_statuses": dict(
            sorted(Counter(str(row["checker_status"]) for row in processed).items())
        ),
        "finish_reason_counts": dict(
            sorted(
                Counter(str(row.get("finish_reason")) for row in processed).items()
            )
        ),
        "eligible_rows": sum(
            bool(row.get("eligible_for_supervision")) for row in processed
        ),
        "material_claim_count_histogram": dict(
            sorted(
                Counter(int(row["material_claim_count"]) for row in processed).items()
            )
        ),
        "checker_status_by_source": _counter_by(
            processed, "source", "checker_status"
        ),
        "checker_status_by_split": _counter_by(
            processed, "acquisition_split", "checker_status"
        ),
    }
    report["exact_contract_pass_rate"] = (
        report["unitization_ok"] / len(processed) if processed else 0.0
    )
    return processed, report


def _validate_units(row: Mapping[str, Any]) -> None:
    units = row.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError(f"{row.get('id')}: successful unitization has no units")
    cursor = 0
    material_count = 0
    output_length = len(row["output_token_ids"])
    for expected_index, unit in enumerate(units):
        if int(unit.get("unit_index", -1)) != expected_index:
            raise ValueError(f"{row['id']}: unit indices are not contiguous")
        token_start = int(unit.get("token_start", -1))
        token_end = int(unit.get("token_end", -1))
        if token_start != cursor or not token_start < token_end <= output_length:
            raise ValueError(f"{row['id']}: units do not partition the token axis")
        if unit.get("kind") not in {"material_claim", "non_claim"}:
            raise ValueError(f"{row['id']}: unsupported unit kind")
        if unit.get("kind") == "material_claim":
            material_count += 1
        cursor = token_end
    if cursor != output_length:
        raise ValueError(f"{row['id']}: units do not cover the output-token suffix")
    if material_count != int(row.get("material_claim_count", -1)):
        raise ValueError(f"{row['id']}: material claim count is inconsistent")


def validate_scale_materialized_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
    checker_version: str,
    unitizer_version: str,
) -> dict[str, Any]:
    """Recheck raw identity, checker eligibility, and the complete token partition."""

    if len(rows) != len(raw_rows):
        raise ValueError("materialized and raw row counts differ")
    population = validate_rollout_population(rows, candidate_count=candidate_count)
    mapping_modes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for raw, row in zip(raw_rows, rows):
        for field in (
            "id",
            "query_id",
            "candidate_index",
            "source",
            "question",
            "response",
            "finish_reason",
            "acquisition_split",
            "cluster_id",
        ):
            if row.get(field) != raw.get(field):
                raise ValueError(f"{raw.get('id')}: materialization changed {field}")
        if row.get("prompt_token_ids") != raw.get("prompt_token_ids"):
            raise ValueError(f"{raw['id']}: prompt token IDs changed")
        if row.get("output_token_ids") != raw.get("output_token_ids"):
            raise ValueError(f"{raw['id']}: output token IDs changed")
        if row.get("raw_reference_answer") != str(raw.get("reference_answer")):
            raise ValueError(f"{raw['id']}: raw reference was not preserved")
        if row.get("checker_version") != checker_version:
            raise ValueError(f"{raw['id']}: checker version drift")
        status = str(row.get("checker_status"))
        statuses[status] += 1
        unitization_status = row.get("unitization_status")
        if unitization_status == "ok":
            if row.get("unitizer_version") != unitizer_version:
                raise ValueError(f"{raw['id']}: unitizer version drift")
            _validate_units(row)
            mapping_modes[str(row.get("token_mapping_mode"))] += 1
        elif unitization_status == "failed":
            if row.get("units") != [] or row.get("eligible_for_supervision"):
                raise ValueError(f"{raw['id']}: failed unitization remained eligible")
            mapping_modes["failed"] += 1
        else:
            raise ValueError(f"{raw['id']}: invalid unitization status")
        if status in {
            "truncated",
            "parse_failed",
            "ambiguous_multiple_answers",
            "invalid_reference",
            "empty_output",
        } and row.get("eligible_for_supervision"):
            raise ValueError(f"{raw['id']}: audit-only checker row remained eligible")
        if raw.get("finish_reason") == "length" and row.get(
            "eligible_for_supervision"
        ):
            raise ValueError(f"{raw['id']}: truncated row remained eligible")
    return {
        **population,
        "raw_identity_rows_verified": len(rows),
        "checker_statuses": dict(sorted(statuses.items())),
        "token_mapping_modes": dict(sorted(mapping_modes.items())),
        "exact_partition_rows": sum(
            row.get("unitization_status") == "ok" for row in rows
        ),
    }


def build_scale_consistency_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    mechanical: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the unchanged v5 filter and freeze one pair per query."""

    if mechanical.get("version") != "clir_consistency_mechanical_v1":
        raise ValueError("scale-v6 mechanical metric version drift")
    kwargs = {
        "min_material_units": int(
            mechanical["minimum_material_claim_units_per_view"]
        ),
        "min_length_ratio": float(mechanical["token_length_ratio_min"]),
        "max_length_ratio": float(mechanical["token_length_ratio_max"]),
        "min_math_tokens": int(mechanical["math_trace_token_count_min_per_view"]),
        "min_numeric_tokens": int(
            mechanical["numeric_trace_token_count_min_per_view"]
        ),
        "min_surface_bigrams": int(
            mechanical["surface_bigram_count_min_per_view"]
        ),
        "min_math_similarity": float(mechanical["math_trace_similarity_min"]),
        "min_numeric_similarity": float(
            mechanical["numeric_trace_similarity_min"]
        ),
        "min_surface_jaccard": float(
            mechanical["surface_bigram_jaccard_min"]
        ),
        "max_surface_jaccard": float(
            mechanical["surface_bigram_jaccard_max"]
        ),
    }
    admitted: list[dict[str, Any]] = []
    source_reports: dict[str, Any] = {}
    for source in ("math", "gsm8k"):
        source_admitted, source_report = build_mechanical_consistency_proposals(
            rows, source=source, **kwargs
        )
        admitted.extend(source_admitted)
        source_reports[source] = source_report

    row_by_id = {str(row["id"]): row for row in rows}
    decorated: list[dict[str, Any]] = []
    for proposal in admitted:
        left = row_by_id[str(proposal["left_id"])]
        right = row_by_id[str(proposal["right_id"])]
        for field in ("source", "acquisition_split", "cluster_id", "query_id"):
            if left.get(field) != right.get(field):
                raise ValueError(
                    f"{proposal['proposal_id']}: pair disagrees on {field}"
                )
        split = str(left["acquisition_split"])
        priority = stable_priority(
            "clir-C-v6-audit",
            split,
            proposal["query_id"],
            proposal["left_candidate_index"],
            proposal["right_candidate_index"],
        )
        value = dict(proposal)
        value.update(
            {
                "schema_version": "clir-consistency-proposal-v6",
                "source": str(left["source"]),
                "source_subject": left.get("source_subject"),
                "source_level": left.get("source_level"),
                "acquisition_split": split,
                "cluster_id": str(left["cluster_id"]),
                "annotation_priority": priority,
            }
        )
        decorated.append(value)
    decorated.sort(key=lambda row: (row["annotation_priority"], row["proposal_id"]))
    query_ids = [str(row["query_id"]) for row in decorated]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("scale-v6 emitted more than one pair for a query")
    split_counts = Counter(str(row["acquisition_split"]) for row in decorated)
    source_counts = Counter(str(row["source"]) for row in decorated)
    split_source_counts = Counter(
        f"{row['acquisition_split']}|{row['source']}" for row in decorated
    )
    return decorated, {
        "mechanical_version": str(mechanical["version"]),
        "input_rows": len(rows),
        "admitted_query_distinct_pairs": len(decorated),
        "admitted_by_split": dict(sorted(split_counts.items())),
        "admitted_by_source": dict(sorted(source_counts.items())),
        "admitted_by_split_source": dict(sorted(split_source_counts.items())),
        "source_reports": source_reports,
    }


def build_scale_natural_items(
    proposals: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    row_by_id = {str(row["id"]): row for row in rows}
    items: list[dict[str, Any]] = []
    for proposal in proposals:
        item = consistency_item(proposal, row_by_id)
        item["audit_scope"] = "substantive_claim_validity_only"
        items.append(item)
    if len({str(item["item_id"]) for item in items}) != len(items):
        raise ValueError("scale-v6 natural item IDs are not unique")
    return items


def _scale_controls(
    shard_id: str, shard_index: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    children = 2 + shard_index % 4
    days = 4 + shard_index % 3
    weeks = 10 + shard_index % 11
    weekly = children * days
    total = weekly * weeks
    start = 12 + shard_index % 9
    added = 3 + shard_index % 5
    summed = start + added
    bags = 3 + shard_index % 5
    per_bag = 4 + shard_index % 4
    bag_total = bags * per_bag
    removed = 2 + shard_index % 5
    original = 15 + shard_index % 8
    remain = original - removed
    cases = [
        {
            "problem": (
                f"There are {children} children. Each needs one juice box on "
                f"{days} days per week for {weeks} weeks. How many boxes are needed?"
            ),
            "left": (
                f"Each child needs {days}*{weeks}={days * weeks} boxes. For "
                f"{children} children, {days * weeks}*{children}={total}."
            ),
            "right": (
                f"All children use {children}*{days}={weekly} boxes weekly. "
                f"Across {weeks} weeks, {weekly}*{weeks}={total}."
            ),
            "decision": "accept",
        },
        {
            "problem": (
                f"Mina has {start} apples and receives {added} more. How many apples?"
            ),
            "left": f"Add the new apples: {start}+{added}={summed}.",
            "right": f"The total is {start}+{added}={summed} apples.",
            "decision": "accept",
        },
        {
            "problem": (
                f"There are {bags} bags with {per_bag} apples in each. "
                "How many apples are there?"
            ),
            "left": f"Multiply: {bags}*{per_bag}={bag_total} apples.",
            "right": (
                f"Multiply: {bags}*{per_bag}={bag_total}. Therefore there are "
                f"{bag_total} oranges."
            ),
            "decision": "reject",
        },
        {
            "problem": (
                f"A basket has {original} apples and {removed} are removed. "
                "How many remain?"
            ),
            "left": (
                f"Subtract: {original}-{removed}={remain}, so {remain} remain."
            ),
            "right": (
                f"Subtracting gives {original}-{removed}={remain + 1}. "
                f"The final answer is {remain}."
            ),
            "decision": "reject",
        },
    ]
    output: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for control_index, case in enumerate(cases):
        item_id = stable_priority(
            "clir-consistency-v6-control", shard_id, control_index
        )
        item = {
            "item_id": item_id,
            "query_id": f"control:consistency-v6:{shard_id}:{control_index}",
            "problem": case["problem"],
            "audit_scope": "substantive_claim_validity_only",
            "left": {
                "id": f"{item_id}:left",
                "trajectory": case["left"],
                "units": [],
            },
            "right": {
                "id": f"{item_id}:right",
                "trajectory": case["right"],
                "units": [],
            },
        }
        expected = validate_annotation(
            "consistency",
            {
                "item_id": item_id,
                "decision": case["decision"],
                "confidence": "high",
                "rationale": (
                    "[ACCEPT_VALID] hidden scale-v6 control"
                    if case["decision"] == "accept"
                    else "[REJECT_ERROR] hidden scale-v6 control"
                ),
            },
            item,
        )
        output.append((item, expected))
    return output


def build_scale_annotation_packages(
    natural_items: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    *,
    max_natural_per_shard: int,
    controls_per_shard: int,
    repeat_fraction_per_annotator: float,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Build deterministic isolated packages with later-shard self repeats."""

    if max_natural_per_shard <= 0:
        raise ValueError("maximum natural pair count per shard must be positive")
    if controls_per_shard != 4:
        raise ValueError("scale-v6 freezes exactly four controls per shard")
    if not 0 <= repeat_fraction_per_annotator <= 1:
        raise ValueError("self-repeat fraction must lie inside [0,1]")
    natural = [dict(item) for item in natural_items]
    if [str(item["item_id"]) for item in natural] != [
        str(row["proposal_id"]) for row in proposals
    ]:
        raise ValueError("natural items are not aligned with frozen proposals")
    shard_count = math.ceil(len(natural) / max_natural_per_shard)
    if not shard_count:
        raise ValueError("cannot package an empty natural population")
    shard_ids = [f"shard-{index:03d}" for index in range(shard_count)]
    base_shards: dict[str, list[dict[str, Any]]] = {}
    natural_membership: dict[str, str] = {}
    for index, shard_id in enumerate(shard_ids):
        start = index * max_natural_per_shard
        rows = natural[start : start + max_natural_per_shard]
        base_shards[shard_id] = rows
        for item in rows:
            natural_membership[str(item["item_id"])] = shard_id

    controls: dict[str, list[dict[str, Any]]] = {}
    control_items: dict[str, list[dict[str, Any]]] = {}
    for index, shard_id in enumerate(shard_ids):
        pairs = _scale_controls(shard_id, index)
        control_items[shard_id] = [item for item, _ in pairs]
        controls[shard_id] = [
            {"item_id": item["item_id"], "expected_annotation": expected}
            for item, expected in pairs
        ]

    packages: dict[str, dict[str, list[dict[str, Any]]]] = {"a": {}, "b": {}}
    repeats_by_slot: dict[str, list[dict[str, str]]] = {}
    repeat_count = math.ceil(len(natural) * repeat_fraction_per_annotator)
    shard_index_by_id = {shard_id: index for index, shard_id in enumerate(shard_ids)}
    for slot in ("a", "b"):
        eligible_sources = [
            item
            for item in natural
            if shard_index_by_id[natural_membership[str(item["item_id"])]]
            < shard_count - 1
        ]
        eligible_sources.sort(
            key=lambda item: stable_priority(
                "clir-consistency-v6-repeat-source", slot, item["item_id"]
            )
        )
        if repeat_count > len(eligible_sources):
            raise ValueError("too few earlier-shard items for frozen self repeats")
        repeats_for_shard: dict[str, list[dict[str, Any]]] = {
            shard_id: [] for shard_id in shard_ids
        }
        repeat_map: list[dict[str, str]] = []
        for item in eligible_sources[:repeat_count]:
            original_id = str(item["item_id"])
            original_shard = natural_membership[original_id]
            original_index = shard_index_by_id[original_shard]
            later = shard_ids[original_index + 1 :]
            target_index = int(
                stable_priority(
                    "clir-consistency-v6-repeat-target", slot, original_id
                )[:16],
                16,
            ) % len(later)
            target_shard = later[target_index]
            repeated = dict(item)
            repeat_id = stable_priority(
                "clir-consistency-v6-repeat", slot, original_id
            )
            repeated["item_id"] = repeat_id
            repeats_for_shard[target_shard].append(repeated)
            repeat_map.append(
                {
                    "original_item_id": original_id,
                    "repeat_item_id": repeat_id,
                    "original_shard_id": original_shard,
                    "repeat_shard_id": target_shard,
                }
            )
        repeats_by_slot[slot] = repeat_map
        for shard_id in shard_ids:
            rows = [
                *base_shards[shard_id],
                *control_items[shard_id],
                *repeats_for_shard[shard_id],
            ]
            rows.sort(
                key=lambda item: stable_priority(
                    "clir-consistency-v6-package", slot, shard_id, item["item_id"]
                )
            )
            item_ids = [str(item["item_id"]) for item in rows]
            if len(item_ids) != len(set(item_ids)):
                raise ValueError(f"{slot}/{shard_id}: package item IDs overlap")
            packages[slot][shard_id] = rows

    natural_metadata = []
    for proposal in proposals:
        natural_metadata.append(
            {
                "item_id": str(proposal["proposal_id"]),
                "query_id": str(proposal["query_id"]),
                "source": str(proposal["source"]),
                "acquisition_split": str(proposal["acquisition_split"]),
                "cluster_id": str(proposal["cluster_id"]),
                "annotation_priority": str(proposal["annotation_priority"]),
                "base_shard_id": natural_membership[str(proposal["proposal_id"])],
            }
        )
    private = {
        "schema_version": "clir-consistency-scale-private-package-manifest-v6",
        "warning": "PRIVATE: never send this file to either annotator",
        "natural_item_ids": [str(item["item_id"]) for item in natural],
        "natural_metadata": natural_metadata,
        "controls_by_shard": controls,
        "self_repeats": repeats_by_slot,
        "natural_pair_count": len(natural),
        "annotation_shard_count": shard_count,
        "max_natural_per_shard": max_natural_per_shard,
        "repeat_fraction_per_annotator": repeat_fraction_per_annotator,
        "package_ordered_rows_sha256": {
            slot: {
                shard_id: canonical_sha256(packages[slot][shard_id])
                for shard_id in shard_ids
            }
            for slot in ("a", "b")
        },
    }
    return packages, private


def validate_scale_package_labels(
    package_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate one shard's strict schema and exact item population."""

    expected_by_id = {str(row["item_id"]): row for row in package_rows}
    if len(expected_by_id) != len(package_rows):
        raise ValueError("annotation package contains duplicate item IDs")
    label_by_id: dict[str, Mapping[str, Any]] = {}
    required_fields = {"item_id", "decision", "confidence", "rationale"}
    for label in labels:
        if set(label) != required_fields:
            raise ValueError("annotation label fields differ from the strict schema")
        item_id = str(label.get("item_id"))
        if item_id in label_by_id:
            raise ValueError("annotation label contains a duplicate item ID")
        label_by_id[item_id] = label
    if set(label_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(label_by_id))
        extra = sorted(set(label_by_id) - set(expected_by_id))
        raise ValueError(
            f"annotation label population differs from package: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return [
        validate_annotation(
            "consistency", label_by_id[item_id], expected_by_id[item_id]
        )
        for item_id in expected_by_id
    ]


def evaluate_scale_annotations(
    *,
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    private: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the preregistered v6 raw gates without adjudication or rescue."""

    natural_ids = [str(value) for value in private["natural_item_ids"]]
    natural_id_set = set(natural_ids)
    by_slot = {
        "a": {str(row["item_id"]): dict(row) for row in labels_a},
        "b": {str(row["item_id"]): dict(row) for row in labels_b},
    }
    if len(by_slot["a"]) != len(labels_a) or len(by_slot["b"]) != len(labels_b):
        raise ValueError("annotation labels contain duplicate item IDs")
    for slot in ("a", "b"):
        if not natural_id_set.issubset(by_slot[slot]):
            missing = sorted(natural_id_set - set(by_slot[slot]))
            raise ValueError(f"annotator {slot} is missing natural IDs: {missing[:5]}")
    natural = {
        slot: [by_slot[slot][item_id] for item_id in natural_ids]
        for slot in ("a", "b")
    }
    agreement = agreement_report("consistency", natural["a"], natural["b"])
    natural_metadata = {
        str(row["item_id"]): row for row in private["natural_metadata"]
    }
    if set(natural_metadata) != natural_id_set:
        raise ValueError("private natural metadata differs from natural IDs")

    expected_prefix = {
        "accept": "[ACCEPT_VALID]",
        "reject": "[REJECT_ERROR]",
        "review": "[REVIEW]",
    }
    gate_rows: list[dict[str, Any]] = []

    def add_gate(name: str, passed: bool, **details: Any) -> None:
        gate_rows.append(
            {"name": name, "status": "PASS" if passed else "FAIL", **details}
        )

    agreement_rate = (
        agreement["exact_target_agree"] / len(natural_ids) if natural_ids else 0.0
    )
    add_gate(
        "natural_decision_agreement",
        agreement_rate >= float(gates["natural_decision_agreement_min"]),
        value=agreement_rate,
        numerator=agreement["exact_target_agree"],
        denominator=len(natural_ids),
        threshold=float(gates["natural_decision_agreement_min"]),
    )

    annotator_reports: dict[str, Any] = {}
    all_controls = [
        control
        for shard_controls in private["controls_by_shard"].values()
        for control in shard_controls
    ]
    for slot in ("a", "b"):
        natural_counts = Counter(str(row["decision"]) for row in natural[slot])
        review_fraction = (
            natural_counts["review"] / len(natural_ids) if natural_ids else 0.0
        )
        control_correct = sum(
            by_slot[slot][str(control["item_id"])]["decision"]
            == control["expected_annotation"]["decision"]
            for control in all_controls
        )
        repeats = private["self_repeats"][slot]
        repeat_correct = sum(
            by_slot[slot][str(repeat["original_item_id"])]["decision"]
            == by_slot[slot][str(repeat["repeat_item_id"])]["decision"]
            for repeat in repeats
        )
        repeat_rate = repeat_correct / len(repeats) if repeats else 1.0
        invalid_prefixes = [
            str(row["item_id"])
            for row in by_slot[slot].values()
            if not str(row["rationale"]).startswith(
                expected_prefix[str(row["decision"])]
            )
        ]
        annotator_reports[slot] = {
            "natural_decision_counts": dict(sorted(natural_counts.items())),
            "natural_review_fraction": review_fraction,
            "hidden_control_correct": control_correct,
            "hidden_control_total": len(all_controls),
            "self_repeat_correct": repeat_correct,
            "self_repeat_total": len(repeats),
            "self_repeat_agreement": repeat_rate,
            "invalid_rationale_prefix_item_ids": invalid_prefixes,
        }
        add_gate(
            f"natural_review_fraction_{slot}",
            review_fraction <= float(gates["review_fraction_max_per_annotator"]),
            value=review_fraction,
            numerator=natural_counts["review"],
            denominator=len(natural_ids),
            threshold=float(gates["review_fraction_max_per_annotator"]),
        )
        add_gate(
            f"hidden_control_accuracy_{slot}",
            control_correct == len(all_controls),
            value=(control_correct / len(all_controls) if all_controls else 1.0),
            numerator=control_correct,
            denominator=len(all_controls),
            threshold=float(gates["hidden_control_accuracy_required_per_annotator"]),
        )
        add_gate(
            f"self_repeat_agreement_{slot}",
            repeat_rate >= float(gates["self_repeat_agreement_min_per_annotator"]),
            value=repeat_rate,
            numerator=repeat_correct,
            denominator=len(repeats),
            threshold=float(gates["self_repeat_agreement_min_per_annotator"]),
        )
        add_gate(
            f"rationale_prefixes_{slot}",
            not invalid_prefixes,
            invalid_item_ids=invalid_prefixes,
        )

    common_accept_ids = [
        item_id
        for item_id in natural_ids
        if by_slot["a"][item_id]["decision"] == "accept"
        and by_slot["b"][item_id]["decision"] == "accept"
        and by_slot["a"][item_id]["confidence"] != "low"
        and by_slot["b"][item_id]["confidence"] != "low"
    ]
    common_by_split = Counter(
        str(natural_metadata[item_id]["acquisition_split"])
        for item_id in common_accept_ids
    )
    common_by_source = Counter(
        str(natural_metadata[item_id]["source"]) for item_id in common_accept_ids
    )
    add_gate(
        "train_common_accept_count",
        common_by_split["train_acquisition"]
        >= int(gates["train_common_accept_count_min"]),
        value=common_by_split["train_acquisition"],
        threshold=int(gates["train_common_accept_count_min"]),
    )
    add_gate(
        "heldout_common_accept_count",
        common_by_split["heldout_acquisition"]
        >= int(gates["heldout_common_accept_count_min"]),
        value=common_by_split["heldout_acquisition"],
        threshold=int(gates["heldout_common_accept_count_min"]),
    )
    failed = [row["name"] for row in gate_rows if row["status"] == "FAIL"]
    return {
        "schema_version": "clir-consistency-scale-raw-annotation-gate-report-v6",
        "status": (
            "PASS_SCALE_V6_RAW_ANNOTATION_GATES"
            if not failed
            else "STOP_SCALE_V6_RAW_ANNOTATION_GATE_FAILURE"
        ),
        "failed_gate_names": failed,
        "third_model_rescue_allowed": False,
        "finalization_allowed": not failed,
        "natural_agreement": agreement,
        "natural_agreement_rate": agreement_rate,
        "annotators": annotator_reports,
        "common_accept_count": len(common_accept_ids),
        "common_accept_by_split": dict(sorted(common_by_split.items())),
        "common_accept_by_source": dict(sorted(common_by_source.items())),
        "common_accept_item_ids": common_accept_ids,
        "gates": gate_rows,
    }


__all__ = [
    "MATERIALIZED_SCHEMA",
    "NATURAL_ITEM_SCHEMA",
    "PACKAGE_SCHEMA",
    "PRE_ANNOTATION_AUTHORIZATION_SCHEMA",
    "PROPOSAL_SCHEMA",
    "build_scale_annotation_packages",
    "build_scale_consistency_proposals",
    "build_scale_natural_items",
    "evaluate_scale_annotations",
    "materialize_scale_rows",
    "validate_scale_package_labels",
    "validate_scale_materialized_rows",
]
