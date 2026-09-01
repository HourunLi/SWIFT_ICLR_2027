"""Deterministic contracts for the exploratory Prior-v12 exact-consensus subset.

The prospective v12 acquisition remains a terminal failure.  This module does
not re-evaluate its gates.  It implements a separately named, post-hoc route
that keeps only natural rows on which both blind annotators gave exactly the
same singleton Key and exactly the same non-empty Complete set.  A natural row
is also removed when either annotator failed an available exact self-repeat for
that row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from numbers import Integral
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_consensus_scale import validate_prior_v12_annotation
from src.clir_prior_partial import target_signature
from src.clir_smoke import canonical_sha256


LABEL_NAME = (
    "silver_posthoc_dual_ai_exact_prior_v12_repeat_fail_excluded_"
    "no_human_verification"
)
ORIGINAL_V12_STATUS = "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE"
ROW_SCHEMA = "clir-prior-v12-posthoc-exact-row-v1"
FEATURE_SCHEMA = "clir-prior-v12-posthoc-feature-input-v1"


def _index_unique(
    rows: Iterable[Mapping[str, Any]], field: str, description: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        value = str(row.get(field, ""))
        if not value or value in indexed:
            raise ValueError(f"missing or duplicate {description} {field}: {value!r}")
        indexed[value] = row
    return indexed


def _integer(value: Any, *, field: str, row_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{row_id}: {field} must be an integer")
    return int(value)


def _token_ids(value: Any, *, field: str, row_id: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be an integer sequence")
    result = [_integer(item, field=field, row_id=row_id) for item in value]
    if not result or any(item < 0 for item in result):
        raise ValueError(f"{row_id}: {field} must be non-empty and non-negative")
    return result


def _normalize_all_labels(
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for annotator in ("a", "b"):
        if annotator not in packages or annotator not in labels:
            raise ValueError("both annotator populations are required")
        package_map = _index_unique(
            packages[annotator], "item_id", f"annotator-{annotator} package"
        )
        label_map = _index_unique(
            labels[annotator], "item_id", f"annotator-{annotator} label"
        )
        if set(package_map) != set(label_map):
            raise ValueError(f"annotator-{annotator} package/label IDs differ")
        normalized[annotator] = {
            item_id: validate_prior_v12_annotation(label_map[item_id], item)
            for item_id, item in package_map.items()
        }
    return normalized


def _materialize_targets(
    row: Mapping[str, Any], key_units: Sequence[int], complete_units: Sequence[int]
) -> dict[str, list[int]]:
    row_id = str(row.get("id", ""))
    output_ids = _token_ids(
        row.get("output_token_ids"), field="output_token_ids", row_id=row_id
    )
    units = row.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ValueError(f"{row_id}: units must be a sequence")
    by_index: dict[int, Mapping[str, Any]] = {}
    cursor = 0
    for raw in units:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{row_id}: unit must be an object")
        index = _integer(raw.get("unit_index"), field="unit_index", row_id=row_id)
        start = _integer(raw.get("token_start"), field="token_start", row_id=row_id)
        end = _integer(raw.get("token_end"), field="token_end", row_id=row_id)
        if index in by_index or start != cursor or not start <= end <= len(output_ids):
            raise ValueError(f"{row_id}: invalid unit partition")
        by_index[index] = raw
        cursor = end
    if cursor != len(output_ids):
        raise ValueError(f"{row_id}: units do not cover the exact output-token axis")

    targets: dict[str, list[int]] = {
        "key_prior_target": [0] * len(output_ids),
        "complete_prior_target": [0] * len(output_ids),
    }
    for field, indices in (
        ("key_prior_target", key_units),
        ("complete_prior_target", complete_units),
    ):
        for unit_index in indices:
            if unit_index not in by_index:
                raise ValueError(f"{row_id}: target unit is absent: {unit_index}")
            unit = by_index[unit_index]
            if unit.get("kind") != "material_claim":
                raise ValueError(f"{row_id}: target unit is not a material claim")
            start, end = int(unit["token_start"]), int(unit["token_end"])
            for token_index in range(start, end):
                targets[field][token_index] = 1
    if any(
        key and not complete
        for key, complete in zip(
            targets["key_prior_target"],
            targets["complete_prior_target"],
            strict=True,
        )
    ):
        raise ValueError(f"{row_id}: Key target is not nested in Complete")
    targets["key_prior_mask"] = [1] * len(output_ids)
    targets["complete_prior_mask"] = [1] * len(output_ids)
    return targets


def construct_posthoc_rows(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute the complete post-hoc membership and exact-token targets."""

    proposal_by_id = _index_unique(proposals, "proposal_id", "proposal")
    materialized_by_id = _index_unique(materialized_rows, "id", "materialized row")
    normalized = _normalize_all_labels(packages, labels)

    private_by_annotator: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for source in private_index:
        row = dict(source)
        annotator = str(row.get("annotator", ""))
        if annotator not in private_by_annotator:
            raise ValueError("private index has an unsupported annotator")
        private_by_annotator[annotator].append(row)

    natural_ids: dict[str, set[str]] = {}
    repeat_tested: defaultdict[str, set[str]] = defaultdict(set)
    repeat_failed: defaultdict[str, set[str]] = defaultdict(set)
    repeat_metrics: dict[str, dict[str, int]] = {}
    for annotator in ("a", "b"):
        private_item_ids = [
            str(row.get("item_id", "")) for row in private_by_annotator[annotator]
        ]
        if (
            any(not item_id for item_id in private_item_ids)
            or len(private_item_ids) != len(set(private_item_ids))
            or set(private_item_ids) != set(normalized[annotator])
        ):
            raise ValueError(f"annotator-{annotator} private/package population differs")
        if any(
            row.get("kind") == "natural"
            and str(row.get("item_id")) != str(row.get("natural_item_id"))
            for row in private_by_annotator[annotator]
        ):
            raise ValueError(f"annotator-{annotator} natural parent identity drift")
        natural_ids[annotator] = {
            str(row["natural_item_id"])
            for row in private_by_annotator[annotator]
            if row.get("kind") == "natural"
        }
        passed = total = 0
        for row in private_by_annotator[annotator]:
            if row.get("kind") != "repeat":
                continue
            total += 1
            item_id = str(row["item_id"])
            parent_id = str(row["natural_item_id"])
            repeat_tested[parent_id].add(annotator)
            if (
                target_signature(normalized[annotator][item_id])
                == target_signature(normalized[annotator][parent_id])
            ):
                passed += 1
            else:
                repeat_failed[parent_id].add(annotator)
        repeat_metrics[annotator] = {"passed": passed, "total": total}
    if natural_ids["a"] != natural_ids["b"] or set(proposal_by_id) != natural_ids["a"]:
        raise ValueError("natural proposal/package populations differ")

    candidates: list[str] = []
    excluded = Counter()
    for item_id in sorted(proposal_by_id):
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        if not (
            left["eligibility"] == right["eligibility"] == "usable"
            and left["confidence"] != "low"
            and right["confidence"] != "low"
        ):
            excluded["not_common_nonlow_usable"] += 1
            continue
        if not (
            len(left["key_unit_indices"]) == 1
            and left["key_unit_indices"] == right["key_unit_indices"]
        ):
            excluded["key_not_exact_singleton"] += 1
            continue
        if not left["complete_unit_indices"] or (
            left["complete_unit_indices"] != right["complete_unit_indices"]
        ):
            excluded["complete_not_exact_nonempty"] += 1
            continue
        if item_id in repeat_failed:
            excluded["available_self_repeat_failed"] += 1
            continue
        candidates.append(item_id)
    candidates.sort(
        key=lambda item_id: (
            str(proposal_by_id[item_id]["selection_priority"]),
            item_id,
        )
    )

    result: list[dict[str, Any]] = []
    strata: Counter[str] = Counter()
    repeat_statuses: Counter[str] = Counter()
    for selection_index, item_id in enumerate(candidates):
        proposal = proposal_by_id[item_id]
        trajectory_id = str(proposal["trajectory_id"])
        if trajectory_id not in materialized_by_id:
            raise ValueError(f"selected trajectory is absent: {trajectory_id}")
        materialized = materialized_by_id[trajectory_id]
        for field in ("query_id", "cluster_id", "candidate_index", "checker_status"):
            if materialized.get(field) != proposal.get(field):
                raise ValueError(f"{trajectory_id}: proposal/materialized {field} drift")
        if (
            materialized.get("eligible_for_supervision") is not True
            or materialized.get("unitization_status") != "ok"
            or materialized.get("correctness") not in {0, 1}
        ):
            raise ValueError(f"{trajectory_id}: selected materialized row is ineligible")
        prompt_ids = _token_ids(
            materialized.get("prompt_token_ids"),
            field="prompt_token_ids",
            row_id=trajectory_id,
        )
        output_ids = _token_ids(
            materialized.get("output_token_ids"),
            field="output_token_ids",
            row_id=trajectory_id,
        )
        label = normalized["a"][item_id]
        key_units = list(label["key_unit_indices"])
        complete_units = list(label["complete_unit_indices"])
        targets = _materialize_targets(materialized, key_units, complete_units)
        if not any(targets["key_prior_target"]) or not any(
            targets["complete_prior_target"]
        ):
            raise ValueError(f"{trajectory_id}: selected Prior target has no positive token")
        tested = repeat_tested.get(item_id, set())
        repeat_status = (
            "passed_for_" + "_and_".join(sorted(tested)) if tested else "not_sampled"
        )
        repeat_statuses[repeat_status] += 1
        split = str(proposal["prior_label_split"])
        if split not in {"train", "dev"}:
            raise ValueError(f"{trajectory_id}: unsupported Prior split")
        row = {
            "schema_version": ROW_SCHEMA,
            "selection_index": selection_index,
            "id": trajectory_id,
            "trajectory_id": trajectory_id,
            "proposal_id": item_id,
            "query_id": str(proposal["query_id"]),
            "cluster_id": str(proposal["cluster_id"]),
            "candidate_index": int(proposal["candidate_index"]),
            "source": proposal.get("source"),
            "source_record_id": proposal.get("source_record_id"),
            "source_subject": materialized.get("source_subject"),
            "source_level": materialized.get("source_level"),
            "split": split,
            "prior_label_split": split,
            "checker_status": proposal["checker_status"],
            "correctness": int(materialized["correctness"]),
            "prompt_token_ids": prompt_ids,
            "output_token_ids": output_ids,
            "prompt_token_count": len(prompt_ids),
            "output_token_count": len(output_ids),
            "key_unit_indices": key_units,
            "complete_unit_indices": complete_units,
            **targets,
            "prior_label_name": LABEL_NAME,
            "prior_label_source": "blind_dual_ai_exact_a_b",
            "prior_human_verified": False,
            "prior_posthoc_exploratory": True,
            "prior_original_v12_status": ORIGINAL_V12_STATUS,
            "available_self_repeat_status": repeat_status,
            "annotator_a_confidence": normalized["a"][item_id]["confidence"],
            "annotator_b_confidence": normalized["b"][item_id]["confidence"],
            "selection_priority": proposal["selection_priority"],
            "feature_role": "prior_train" if split == "train" else "prior_dev",
        }
        result.append(row)
        strata[f"{row['source']}|{row['checker_status']}|{split}"] += 1

    query_ids = {str(row["query_id"]) for row in result}
    cluster_ids = {str(row["cluster_id"]) for row in result}
    if len(query_ids) != len(result) or len(cluster_ids) != len(result):
        raise ValueError("selected post-hoc rows are not query/cluster unique")
    train_queries = {str(row["query_id"]) for row in result if row["split"] == "train"}
    dev_queries = {str(row["query_id"]) for row in result if row["split"] == "dev"}
    if train_queries & dev_queries:
        raise ValueError("post-hoc Prior train/dev query overlap")

    report = {
        "input_natural_rows": len(proposal_by_id),
        "selected_rows": len(result),
        "selected_train_rows": sum(row["split"] == "train" for row in result),
        "selected_dev_rows": sum(row["split"] == "dev" for row in result),
        "selected_ordered_proposal_ids_sha256": canonical_sha256(candidates),
        "selected_strata": dict(sorted(strata.items())),
        "excluded_reasons": dict(sorted(excluded.items())),
        "repeat_metrics": repeat_metrics,
        "repeat_failed_parent_union": len(repeat_failed),
        "selected_repeat_statuses": dict(sorted(repeat_statuses.items())),
        "output_token_count": sum(row["output_token_count"] for row in result),
        "prompt_token_count": sum(row["prompt_token_count"] for row in result),
        "key_positive_token_count": sum(sum(row["key_prior_target"]) for row in result),
        "complete_positive_token_count": sum(
            sum(row["complete_prior_target"]) for row in result
        ),
    }
    return result, report


def feature_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["schema_version"] = FEATURE_SCHEMA
        row["feature_inventory_index"] = index
        output.append(row)
    return output


def inventory_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    roles = Counter(str(row["feature_role"]) for row in rows)
    sources = Counter(str(row.get("source")) for row in rows)
    output_tokens = sum(int(row["output_token_count"]) for row in rows)
    prompt_tokens = sum(int(row["prompt_token_count"]) for row in rows)
    return {
        "trajectory_count": len(rows),
        "query_count": len({str(row["query_id"]) for row in rows}),
        "condition_count": len({str(row["query_id"]) for row in rows}),
        "role_row_counts": dict(sorted(roles.items())),
        "source_row_counts": dict(sorted(sources.items())),
        "output_token_count": output_tokens,
        "prompt_token_count": prompt_tokens,
        "total_feature_token_count": output_tokens + prompt_tokens,
    }


def rebase_feature_paths(
    row: Mapping[str, Any], *, source_parent: str | Path, target_parent: str | Path
) -> dict[str, Any]:
    output = dict(row)
    source_root = Path(source_parent).resolve()
    target_root = Path(target_parent).resolve()
    for field in ("hidden_states_path", "condition_states_path"):
        raw = Path(str(output[field]))
        absolute = raw.resolve() if raw.is_absolute() else (source_root / raw).resolve()
        output[field] = os.path.relpath(absolute, target_root)
    return output


def stable_worker_name(identifier: str) -> str:
    return hashlib.sha256(
        f"clir-prior-v12-posthoc-feature|{identifier}".encode("utf-8")
    ).hexdigest()


__all__ = [
    "FEATURE_SCHEMA",
    "LABEL_NAME",
    "ORIGINAL_V12_STATUS",
    "ROW_SCHEMA",
    "construct_posthoc_rows",
    "feature_inventory",
    "inventory_statistics",
    "rebase_feature_paths",
    "stable_worker_name",
]
