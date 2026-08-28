"""Fail-closed post-annotation planning for CLIR Consistency scale v6.

This module consumes already validated blind A/B labels.  It evaluates no
provider and extracts no hidden states.  Positive relations are selected only
from common non-low-confidence accepts in the frozen annotation order.  The
held-out hard-negative planner applies the response-surface contract literally
and reports an explicit stop when the preregistered target is infeasible.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from src.clir_smoke import (
    canonical_sha256,
    consistency_mechanical_metrics,
    stable_priority,
)


POSITIVE_RELATION_SCHEMA = "clir-consistency-scale-positive-relation-v6"
HARD_NEGATIVE_RELATION_SCHEMA = (
    "clir-consistency-scale-heldout-hard-negative-v6"
)
POST_ANNOTATION_PLAN_SCHEMA = "clir-consistency-scale-post-annotation-plan-v6"


def _unique_by(
    rows: Sequence[Mapping[str, Any]], field: str, *, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in output:
            raise ValueError(f"{label} contains duplicate {field}: {key}")
        output[key] = row
    return output


def _normalized_answer(row: Mapping[str, Any]) -> tuple[str, ...]:
    answer = row.get("normalized_candidate_answer")
    if not isinstance(answer, list) or not answer:
        raise ValueError(f"{row.get('id')}: missing normalized candidate answer")
    return tuple(str(value) for value in answer)


def _positive_relation(
    proposal: Mapping[str, Any], row_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    left = row_by_id[str(proposal["left_id"])]
    right = row_by_id[str(proposal["right_id"])]
    for field in ("query_id", "cluster_id", "source", "acquisition_split"):
        if left.get(field) != right.get(field):
            raise ValueError(
                f"{proposal['proposal_id']}: selected views disagree on {field}"
            )
    left_answer = _normalized_answer(left)
    right_answer = _normalized_answer(right)
    if left_answer != right_answer:
        raise ValueError(
            f"{proposal['proposal_id']}: selected positive answers disagree"
        )
    return {
        "schema_version": POSITIVE_RELATION_SCHEMA,
        "relation_id": str(proposal["proposal_id"]),
        "label": 1,
        "label_tier": "silver_dual_ai_consistency_v6",
        "acquisition_split": str(proposal["acquisition_split"]),
        "query_id": str(proposal["query_id"]),
        "cluster_id": str(proposal["cluster_id"]),
        "source": str(proposal["source"]),
        "source_subject": proposal.get("source_subject"),
        "source_level": proposal.get("source_level"),
        "left_id": str(proposal["left_id"]),
        "right_id": str(proposal["right_id"]),
        "left_candidate_index": int(proposal["left_candidate_index"]),
        "right_candidate_index": int(proposal["right_candidate_index"]),
        "normalized_candidate_answer": list(left_answer),
        "annotation_priority": str(proposal["annotation_priority"]),
        "selection_rule": "first_common_accept_in_frozen_annotation_order",
    }


def select_scale_positive_relations(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    common_accept_item_ids: Sequence[str],
    train_count: int,
    heldout_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen first-N train/heldout common accepts."""

    proposal_by_id = _unique_by(proposals, "proposal_id", label="proposals")
    row_by_id = _unique_by(materialized_rows, "id", label="materialized rows")
    ordered_common = [str(item_id) for item_id in common_accept_item_ids]
    if len(ordered_common) != len(set(ordered_common)):
        raise ValueError("common-accept item IDs are not unique")
    missing = [item_id for item_id in ordered_common if item_id not in proposal_by_id]
    if missing:
        raise ValueError(f"common accepts are missing proposals: {missing[:5]}")

    by_split: dict[str, list[str]] = {
        "train_acquisition": [],
        "heldout_acquisition": [],
    }
    for item_id in ordered_common:
        split = str(proposal_by_id[item_id]["acquisition_split"])
        if split not in by_split:
            raise ValueError(f"unsupported acquisition split: {split}")
        by_split[split].append(item_id)
    if len(by_split["train_acquisition"]) < train_count:
        raise ValueError("insufficient train common accepts for frozen selection")
    if len(by_split["heldout_acquisition"]) < heldout_count:
        raise ValueError("insufficient heldout common accepts for frozen selection")

    train_ids = by_split["train_acquisition"][:train_count]
    heldout_ids = by_split["heldout_acquisition"][:heldout_count]
    train = [
        _positive_relation(proposal_by_id[item_id], row_by_id)
        for item_id in train_ids
    ]
    heldout = [
        _positive_relation(proposal_by_id[item_id], row_by_id)
        for item_id in heldout_ids
    ]

    train_queries = {row["query_id"] for row in train}
    heldout_queries = {row["query_id"] for row in heldout}
    train_clusters = {row["cluster_id"] for row in train}
    heldout_clusters = {row["cluster_id"] for row in heldout}
    if train_queries & heldout_queries:
        raise ValueError("selected train and heldout queries overlap")
    if train_clusters & heldout_clusters:
        raise ValueError("selected train and heldout template clusters overlap")
    if len(train_queries) != len(train) or len(heldout_queries) != len(heldout):
        raise ValueError("selected positives are not query-distinct within split")

    report = {
        "train_selected": len(train),
        "heldout_selected": len(heldout),
        "train_available_common_accepts": len(by_split["train_acquisition"]),
        "heldout_available_common_accepts": len(by_split["heldout_acquisition"]),
        "train_by_source": dict(
            sorted(Counter(row["source"] for row in train).items())
        ),
        "heldout_by_source": dict(
            sorted(Counter(row["source"] for row in heldout).items())
        ),
        "train_unique_queries": len(train_queries),
        "heldout_unique_queries": len(heldout_queries),
        "train_unique_clusters": len(train_clusters),
        "heldout_unique_clusters": len(heldout_clusters),
        "cross_split_query_overlap": 0,
        "cross_split_cluster_overlap": 0,
        "train_ordered_rows_sha256": canonical_sha256(train),
        "heldout_ordered_rows_sha256": canonical_sha256(heldout),
    }
    return train, heldout, report


def _hard_negative_endpoint(
    relation: Mapping[str, Any],
    side: str,
    row_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = row_by_id[str(relation[f"{side}_id"])]
    if str(row["query_id"]) != str(relation["query_id"]):
        raise ValueError(f"{relation['relation_id']}: endpoint query drift")
    return {
        "positive_relation_id": str(relation["relation_id"]),
        "side": side,
        "trajectory_id": str(row["id"]),
        "query_id": str(row["query_id"]),
        "cluster_id": str(row["cluster_id"]),
        "source": str(row["source"]),
        "source_subject": row.get("source_subject"),
        "source_level": row.get("source_level"),
        "normalized_candidate_answer": _normalized_answer(row),
        "output_token_count": len(row["output_token_ids"]),
        "response": str(row["response"]),
    }


def build_scale_heldout_hard_negatives(
    *,
    heldout_positive_relations: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen response-surface hard-negative matching literally.

    Every selected held-out positive contributes its two already budgeted views.
    Candidate edges use different queries/clusters/answers, a symmetric token
    length ratio, and the existing v5 response-surface bigram metric.  Greedy
    matching never reuses a view, preventing a small number of trajectories from
    dominating the collapse diagnostic.
    """

    target = int(contract["count"])
    min_ratio = float(contract["view_token_length_ratio_min"])
    max_ratio = float(contract["view_token_length_ratio_max"])
    min_surface = float(contract["surface_bigram_jaccard_min"])
    max_surface = float(contract["surface_bigram_jaccard_max"])
    if contract.get("matching") != (
        "deterministic_greedy_by_source_stratum_length_distance_then_sha256"
    ):
        raise ValueError("hard-negative matching contract drift")
    if not contract.get("different_query_and_template_cluster_required"):
        raise ValueError("hard-negative query/cluster separation was disabled")
    if not contract.get("different_normalized_final_answer_required"):
        raise ValueError("hard-negative answer separation was disabled")

    row_by_id = _unique_by(materialized_rows, "id", label="materialized rows")
    endpoints: list[dict[str, Any]] = []
    for relation in heldout_positive_relations:
        if relation.get("acquisition_split") != "heldout_acquisition":
            raise ValueError("hard-negative endpoint is not heldout")
        endpoints.append(_hard_negative_endpoint(relation, "left", row_by_id))
        endpoints.append(_hard_negative_endpoint(relation, "right", row_by_id))
    if len({row["trajectory_id"] for row in endpoints}) != len(endpoints):
        raise ValueError("heldout positive endpoints are not unique")

    candidate_edges: list[tuple[tuple[Any, ...], dict[str, Any], int, int]] = []
    rejection_counts: Counter[str] = Counter()
    for left_index, left in enumerate(endpoints):
        for right_index in range(left_index + 1, len(endpoints)):
            right = endpoints[right_index]
            if left["query_id"] == right["query_id"]:
                rejection_counts["same_query"] += 1
                continue
            if left["cluster_id"] == right["cluster_id"]:
                rejection_counts["same_template_cluster"] += 1
                continue
            if (
                left["normalized_candidate_answer"]
                == right["normalized_candidate_answer"]
            ):
                rejection_counts["same_normalized_answer"] += 1
                continue
            shorter_to_longer = min(
                left["output_token_count"], right["output_token_count"]
            ) / max(left["output_token_count"], right["output_token_count"])
            # [0.8, 1.25] is the frozen reciprocal-style interval; the
            # symmetric shorter/longer value is in (0, 1].
            if not min_ratio <= shorter_to_longer <= min(max_ratio, 1.0):
                rejection_counts["token_length_ratio"] += 1
                continue
            metrics = consistency_mechanical_metrics(
                left["response"],
                right["response"],
                left_token_count=left["output_token_count"],
                right_token_count=right["output_token_count"],
            )
            surface = float(metrics["surface_bigram_jaccard"])
            if not min_surface <= surface <= max_surface:
                rejection_counts["surface_bigram_jaccard"] += 1
                continue

            left_stratum = (left["source"], left["source_subject"])
            right_stratum = (right["source"], right["source_subject"])
            source_penalty = int(left["source"] != right["source"])
            stratum_penalty = int(left_stratum != right_stratum)
            length_distance = 1.0 - shorter_to_longer
            edge_priority = stable_priority(
                "clir-C-v6-hard-negative",
                left["trajectory_id"],
                right["trajectory_id"],
            )
            relation_id = stable_priority(
                "clir-C-v6-hard-negative-relation",
                left["trajectory_id"],
                right["trajectory_id"],
            )
            row = {
                "schema_version": HARD_NEGATIVE_RELATION_SCHEMA,
                "relation_id": relation_id,
                "label": 0,
                "label_tier": "deterministic_heldout_hard_negative_v6",
                "evaluation_only": True,
                "acquisition_split": "heldout_acquisition",
                "left_id": left["trajectory_id"],
                "right_id": right["trajectory_id"],
                "left_positive_relation_id": left["positive_relation_id"],
                "right_positive_relation_id": right["positive_relation_id"],
                "left_query_id": left["query_id"],
                "right_query_id": right["query_id"],
                "left_cluster_id": left["cluster_id"],
                "right_cluster_id": right["cluster_id"],
                "left_normalized_candidate_answer": list(
                    left["normalized_candidate_answer"]
                ),
                "right_normalized_candidate_answer": list(
                    right["normalized_candidate_answer"]
                ),
                "left_source": left["source"],
                "right_source": right["source"],
                "left_source_subject": left["source_subject"],
                "right_source_subject": right["source_subject"],
                "left_output_token_count": left["output_token_count"],
                "right_output_token_count": right["output_token_count"],
                "shorter_to_longer_token_ratio": shorter_to_longer,
                "surface_bigram_jaccard": surface,
                "matching_priority": {
                    "source_mismatch_penalty": source_penalty,
                    "source_stratum_mismatch_penalty": stratum_penalty,
                    "relative_length_distance": length_distance,
                    "sha256_tiebreak": edge_priority,
                },
            }
            sort_key = (
                source_penalty,
                stratum_penalty,
                length_distance,
                edge_priority,
            )
            candidate_edges.append((sort_key, row, left_index, right_index))

    candidate_edges.sort(key=lambda value: value[0])
    used_endpoints: set[int] = set()
    selected: list[dict[str, Any]] = []
    for _, row, left_index, right_index in candidate_edges:
        if left_index in used_endpoints or right_index in used_endpoints:
            continue
        used_endpoints.update((left_index, right_index))
        selected.append(row)
        if len(selected) == target:
            break

    candidate_query_pairs = {
        tuple(sorted((row[1]["left_query_id"], row[1]["right_query_id"])))
        for row in candidate_edges
    }
    status = (
        "PASS_SCALE_V6_HELDOUT_HARD_NEGATIVES"
        if len(selected) == target
        else "STOP_SCALE_V6_HARD_NEGATIVE_YIELD"
    )
    report = {
        "status": status,
        "target_count": target,
        "selected_count": len(selected),
        "endpoint_count": len(endpoints),
        "unique_endpoint_count": len(
            {row["trajectory_id"] for row in endpoints}
        ),
        "eligible_candidate_edge_count": len(candidate_edges),
        "eligible_unique_query_pair_count": len(candidate_query_pairs),
        "used_endpoint_count": len(used_endpoints),
        "response_surface_metric": "clir_consistency_mechanical_v1",
        "selection_reuses_endpoint": False,
        "thresholds_changed": False,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selected_ordered_rows_sha256": canonical_sha256(selected),
    }
    return selected, report


def build_scale_post_annotation_plan(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    raw_gate_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build the frozen plan and fail closed before publishing manifests."""

    if raw_gate_report.get("status") != "PASS_SCALE_V6_RAW_ANNOTATION_GATES":
        raise ValueError("raw annotation gates did not pass")
    positive_contract = protocol["final_positive_selection"]
    train, heldout, positive_report = select_scale_positive_relations(
        proposals=proposals,
        materialized_rows=materialized_rows,
        common_accept_item_ids=raw_gate_report["common_accept_item_ids"],
        train_count=int(
            positive_contract["train_select_first_by_frozen_annotation_order"]
        ),
        heldout_count=int(
            positive_contract[
                "heldout_select_first_by_frozen_annotation_order"
            ]
        ),
    )
    negatives, negative_report = build_scale_heldout_hard_negatives(
        heldout_positive_relations=heldout,
        materialized_rows=materialized_rows,
        contract=protocol["heldout_hard_negatives"],
    )
    passed = negative_report["status"] == (
        "PASS_SCALE_V6_HELDOUT_HARD_NEGATIVES"
    )
    report = {
        "schema_version": POST_ANNOTATION_PLAN_SCHEMA,
        "status": (
            "PASS_SCALE_V6_POST_ANNOTATION_PLAN"
            if passed
            else "STOP_SCALE_V6_POST_ANNOTATION_PLAN"
        ),
        "positive_selection": positive_report,
        "heldout_hard_negatives": negative_report,
        "publishable_relation_manifests_allowed": passed,
        "feature_extraction_allowed": passed,
        "training_allowed": False,
        "third_model_rescue_allowed": False,
        "threshold_changes_applied": False,
    }
    return {"train": train, "heldout": heldout, "negatives": negatives}, report


__all__ = [
    "HARD_NEGATIVE_RELATION_SCHEMA",
    "POSITIVE_RELATION_SCHEMA",
    "POST_ANNOTATION_PLAN_SCHEMA",
    "build_scale_heldout_hard_negatives",
    "build_scale_post_annotation_plan",
    "select_scale_positive_relations",
]
