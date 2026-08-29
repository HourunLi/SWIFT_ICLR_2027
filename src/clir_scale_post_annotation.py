"""Fail-closed post-annotation planning for CLIR Consistency scale v6.

This module consumes already validated blind A/B labels.  It evaluates no
provider and extracts no hidden states.  Positive relations are selected only
from common non-low-confidence accepts in the frozen annotation order.  The
held-out hard-negative planner applies the response-surface contract literally
and reports an explicit stop when the preregistered target is infeasible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import networkx as nx

from src.clir_smoke import (
    canonical_sha256,
    consistency_mechanical_metrics,
    consistency_surface_bigrams,
    stable_priority,
)


POSITIVE_RELATION_SCHEMA = "clir-consistency-scale-positive-relation-v6"
HARD_NEGATIVE_RELATION_SCHEMA = (
    "clir-consistency-scale-heldout-hard-negative-v6"
)
POST_ANNOTATION_PLAN_SCHEMA = "clir-consistency-scale-post-annotation-plan-v6"
HARD_NEGATIVE_RELATION_SCHEMA_V6_1 = (
    "clir-consistency-scale-heldout-hard-negative-v6.1"
)
FEATURE_INVENTORY_SCHEMA_V6_1 = "clir-consistency-scale-feature-inventory-v6.1"
POST_ANNOTATION_PLAN_SCHEMA_V6_1 = (
    "clir-consistency-scale-post-annotation-plan-v6.1"
)


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


def deterministic_preferred_maximum_matching(
    *,
    node_count: int,
    ordered_edges: Sequence[tuple[int, int]],
    required_networkx_version: str,
) -> list[int]:
    """Return preference ranks in a deterministic maximum-cardinality matching.

    ``ordered_edges`` is already sorted from most to least preferred.  Cardinality
    is the primary objective.  Among maximum-cardinality matchings, NetworkX
    maximizes the sum of the monotone integer preference weights.  Node order,
    edge insertion order, the final SHA-256 tie-break, and the NetworkX version
    are all frozen, so identical inputs reproduce the same selected edge set.
    """

    if nx.__version__ != required_networkx_version:
        raise RuntimeError(
            "scale-v6.1 requires networkx "
            f"{required_networkx_version}, found {nx.__version__}"
        )
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    seen: set[tuple[int, int]] = set()
    edge_count = len(ordered_edges)
    for rank, edge in enumerate(ordered_edges):
        left, right = sorted((int(edge[0]), int(edge[1])))
        if left < 0 or right >= node_count or left == right:
            raise ValueError(f"invalid matching edge: {(left, right)}")
        if (left, right) in seen:
            raise ValueError(f"duplicate matching edge: {(left, right)}")
        seen.add((left, right))
        graph.add_edge(
            left,
            right,
            weight=edge_count - rank,
            preference_rank=rank,
        )
    matched = nx.max_weight_matching(
        graph, maxcardinality=True, weight="weight"
    )
    ranks = sorted(
        int(graph[min(left, right)][max(left, right)]["preference_rank"])
        for left, right in matched
    )
    used: set[int] = set()
    for rank in ranks:
        left, right = ordered_edges[rank]
        if left in used or right in used:
            raise RuntimeError("NetworkX returned a non-matching edge set")
        used.update((left, right))
    return ranks


def _v6_1_pool_endpoint(
    row: Mapping[str, Any],
    selected_positive_by_trajectory: Mapping[str, str],
) -> dict[str, Any]:
    trajectory_id = str(row["id"])
    return {
        "trajectory_id": trajectory_id,
        "query_id": str(row["query_id"]),
        "cluster_id": str(row["cluster_id"]),
        "source": str(row["source"]),
        "source_subject": row.get("source_subject"),
        "source_level": row.get("source_level"),
        "candidate_index": int(row["candidate_index"]),
        "normalized_candidate_answer": _normalized_answer(row),
        "output_token_count": len(row["output_token_ids"]),
        "response": str(row["response"]),
        "selected_positive_relation_id": selected_positive_by_trajectory.get(
            trajectory_id
        ),
    }


def _eligible_v6_1_pool_rows(
    materialized_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    eligible: list[Mapping[str, Any]] = []
    for row in materialized_rows:
        if row.get("acquisition_split") != "heldout_acquisition":
            continue
        if row.get("checker_status") != "numeric_match":
            continue
        if row.get("numeric_value_match") != 1:
            continue
        if row.get("eligible_for_supervision") is not True:
            continue
        if row.get("unitization_status") != "ok":
            continue
        if row.get("finish_reason") == "length":
            continue
        if not isinstance(row.get("output_token_ids"), list) or not row[
            "output_token_ids"
        ]:
            raise ValueError(f"{row.get('id')}: eligible endpoint has no output IDs")
        if not isinstance(row.get("prompt_token_ids"), list) or not row[
            "prompt_token_ids"
        ]:
            raise ValueError(f"{row.get('id')}: eligible endpoint has no prompt IDs")
        _normalized_answer(row)
        eligible.append(row)
    return sorted(eligible, key=lambda row: str(row["id"]))


def build_scale_heldout_hard_negatives_v6_1(
    *,
    heldout_positive_relations: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build v6.1 negatives from every existing eligible held-out match view."""

    expected_pool = (
        "all_existing_heldout_numeric_match_supervision_eligible_views"
    )
    expected_matching = (
        "networkx_max_weight_matching_maxcardinality_then_preference_v1"
    )
    if contract.get("source_pool") != expected_pool:
        raise ValueError("scale-v6.1 hard-negative source-pool drift")
    if contract.get("matching") != expected_matching:
        raise ValueError("scale-v6.1 hard-negative matcher drift")
    if not contract.get("different_query_and_template_cluster_required"):
        raise ValueError("hard-negative query/cluster separation was disabled")
    if not contract.get("different_normalized_final_answer_required"):
        raise ValueError("hard-negative answer separation was disabled")
    if contract.get("negative_pairs_are_evaluation_only") is not True:
        raise ValueError("scale-v6.1 hard negatives must remain evaluation-only")

    target = int(contract["count"])
    min_ratio = float(contract["view_token_length_ratio_min"])
    max_ratio = float(contract["view_token_length_ratio_max"])
    min_surface = float(contract["surface_bigram_jaccard_min"])
    max_surface = float(contract["surface_bigram_jaccard_max"])
    if not 0 < min_surface <= max_surface <= 1:
        raise ValueError("hard-negative surface band is invalid")
    if not 0 < min_ratio <= 1 <= max_ratio:
        raise ValueError("hard-negative reciprocal length band is invalid")

    selected_positive_by_trajectory: dict[str, str] = {}
    for relation in heldout_positive_relations:
        if relation.get("acquisition_split") != "heldout_acquisition":
            raise ValueError("selected positive is not heldout")
        for side in ("left", "right"):
            trajectory_id = str(relation[f"{side}_id"])
            if trajectory_id in selected_positive_by_trajectory:
                raise ValueError("selected heldout positives reuse a trajectory")
            selected_positive_by_trajectory[trajectory_id] = str(
                relation["relation_id"]
            )

    pool_rows = _eligible_v6_1_pool_rows(materialized_rows)
    endpoints = [
        _v6_1_pool_endpoint(row, selected_positive_by_trajectory)
        for row in pool_rows
    ]
    if len({row["trajectory_id"] for row in endpoints}) != len(endpoints):
        raise ValueError("scale-v6.1 endpoint pool repeats trajectory IDs")
    missing_positive_views = sorted(
        set(selected_positive_by_trajectory)
        - {row["trajectory_id"] for row in endpoints}
    )
    if missing_positive_views:
        raise ValueError(
            "selected heldout positives are absent from the v6.1 eligible pool: "
            f"{missing_positive_views[:5]}"
        )

    endpoint_count = len(endpoints)
    inverted: dict[tuple[str, str], list[int]] = defaultdict(list)
    for endpoint_index, endpoint in enumerate(endpoints):
        for bigram in sorted(consistency_surface_bigrams(endpoint["response"])):
            inverted[bigram].append(endpoint_index)
    encoded_pairs: set[int] = set()
    for indices in inverted.values():
        for offset, left_index in enumerate(indices):
            for right_index in indices[offset + 1 :]:
                encoded_pairs.add(left_index * endpoint_count + right_index)

    candidate_edges: list[tuple[tuple[Any, ...], dict[str, Any], int, int]] = []
    rejection_counts: Counter[str] = Counter()
    for encoded in sorted(encoded_pairs):
        left_index, right_index = divmod(encoded, endpoint_count)
        left = endpoints[left_index]
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

        source_penalty = int(left["source"] != right["source"])
        left_stratum = (left["source"], left["source_subject"])
        right_stratum = (right["source"], right["source_subject"])
        stratum_penalty = int(left_stratum != right_stratum)
        length_distance = 1.0 - shorter_to_longer
        edge_priority = stable_priority(
            "clir-C-v6.1-hard-negative",
            left["trajectory_id"],
            right["trajectory_id"],
        )
        relation_id = stable_priority(
            "clir-C-v6.1-hard-negative-relation",
            left["trajectory_id"],
            right["trajectory_id"],
        )
        row = {
            "schema_version": HARD_NEGATIVE_RELATION_SCHEMA_V6_1,
            "relation_id": relation_id,
            "label": 0,
            "label_tier": "deterministic_heldout_hard_negative_v6_1",
            "evaluation_only": True,
            "acquisition_split": "heldout_acquisition",
            "endpoint_source_pool": expected_pool,
            "left_id": left["trajectory_id"],
            "right_id": right["trajectory_id"],
            "left_selected_positive_relation_id": left[
                "selected_positive_relation_id"
            ],
            "right_selected_positive_relation_id": right[
                "selected_positive_relation_id"
            ],
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
    matching_ranks = deterministic_preferred_maximum_matching(
        node_count=endpoint_count,
        ordered_edges=[(row[2], row[3]) for row in candidate_edges],
        required_networkx_version=str(contract["networkx_version"]),
    )
    maximum_matching_rows: list[dict[str, Any]] = []
    for rank in matching_ranks:
        row = dict(candidate_edges[rank][1])
        row["maximum_matching_preference_rank"] = rank
        maximum_matching_rows.append(row)
    selected = maximum_matching_rows[:target]

    candidate_query_pairs = {
        tuple(sorted((row[1]["left_query_id"], row[1]["right_query_id"])))
        for row in candidate_edges
    }
    selected_endpoint_ids = {
        endpoint
        for row in selected
        for endpoint in (row["left_id"], row["right_id"])
    }
    selected_query_ids = {
        query
        for row in selected
        for query in (row["left_query_id"], row["right_query_id"])
    }
    source_pairs = Counter(
        "/".join(sorted((row["left_source"], row["right_source"])))
        for row in selected
    )
    status = (
        "PASS_SCALE_V6_1_HELDOUT_HARD_NEGATIVES"
        if len(selected) == target
        else "STOP_SCALE_V6_1_HARD_NEGATIVE_YIELD"
    )
    all_possible_pairs = endpoint_count * (endpoint_count - 1) // 2
    report = {
        "status": status,
        "target_count": target,
        "selected_count": len(selected),
        "endpoint_source_pool": expected_pool,
        "endpoint_count": endpoint_count,
        "endpoint_unique_query_count": len(
            {row["query_id"] for row in endpoints}
        ),
        "endpoint_by_source": dict(
            sorted(Counter(row["source"] for row in endpoints).items())
        ),
        "all_possible_view_pair_count": all_possible_pairs,
        "shared_surface_bigram_pair_count": len(encoded_pairs),
        "eligible_candidate_edge_count": len(candidate_edges),
        "eligible_unique_query_pair_count": len(candidate_query_pairs),
        "maximum_matching_size": len(maximum_matching_rows),
        "maximum_matching_ordered_rows_sha256": canonical_sha256(
            maximum_matching_rows
        ),
        "used_endpoint_count": len(selected_endpoint_ids),
        "selected_unique_query_count": len(selected_query_ids),
        "selected_positive_view_overlap_count": len(
            selected_endpoint_ids & set(selected_positive_by_trajectory)
        ),
        "selected_by_source_pair": dict(sorted(source_pairs.items())),
        "response_surface_metric": "clir_consistency_mechanical_v1",
        "matching": expected_matching,
        "networkx_version": nx.__version__,
        "selection_reuses_endpoint": False,
        "thresholds_changed": False,
        "negative_pairs_are_evaluation_only": True,
        "rejection_counts_among_shared_bigram_pairs": dict(
            sorted(rejection_counts.items())
        ),
        "no_shared_surface_bigram_pair_count": (
            all_possible_pairs - len(encoded_pairs)
        ),
        "selected_ordered_rows_sha256": canonical_sha256(selected),
    }
    return selected, report


def build_scale_feature_inventory_v6_1(
    *,
    train_positive_relations: Sequence[Mapping[str, Any]],
    heldout_positive_relations: Sequence[Mapping[str, Any]],
    heldout_hard_negatives: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    feature_bytes_per_token: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the exact selected-view inventory and revised storage budget."""

    if feature_bytes_per_token <= 0:
        raise ValueError("feature bytes per token must be positive")
    row_by_id = _unique_by(materialized_rows, "id", label="materialized rows")
    usage_by_id: dict[str, set[str]] = defaultdict(set)
    relation_ids_by_id: dict[str, set[str]] = defaultdict(set)

    def add_relations(
        relations: Sequence[Mapping[str, Any]], usage: str
    ) -> None:
        for relation in relations:
            for side in ("left", "right"):
                trajectory_id = str(relation[f"{side}_id"])
                if trajectory_id not in row_by_id:
                    raise ValueError(
                        f"selected trajectory is absent from materialization: {trajectory_id}"
                    )
                usage_by_id[trajectory_id].add(usage)
                relation_ids_by_id[trajectory_id].add(
                    str(relation["relation_id"])
                )

    add_relations(train_positive_relations, "train_positive")
    add_relations(heldout_positive_relations, "heldout_positive")
    add_relations(heldout_hard_negatives, "heldout_hard_negative")

    positive_ids = {
        str(relation[side])
        for relation in train_positive_relations
        for side in ("left_id", "right_id")
    } | {
        str(relation[side])
        for relation in heldout_positive_relations
        for side in ("left_id", "right_id")
    }
    negative_ids = {
        str(relation[side])
        for relation in heldout_hard_negatives
        for side in ("left_id", "right_id")
    }
    selected_ids = sorted(positive_ids | negative_ids)

    prompt_ids_by_query: dict[str, tuple[int, ...]] = {}
    rows_by_query: dict[str, list[str]] = defaultdict(list)
    for trajectory_id in selected_ids:
        row = row_by_id[trajectory_id]
        query_id = str(row["query_id"])
        prompt_ids = tuple(int(value) for value in row["prompt_token_ids"])
        existing = prompt_ids_by_query.setdefault(query_id, prompt_ids)
        if existing != prompt_ids:
            raise ValueError(f"selected query has inconsistent prompt IDs: {query_id}")
        rows_by_query[query_id].append(trajectory_id)
    condition_owner = {
        query_id: min(trajectory_ids)
        for query_id, trajectory_ids in rows_by_query.items()
    }

    inventory: list[dict[str, Any]] = []
    for trajectory_id in selected_ids:
        row = row_by_id[trajectory_id]
        query_id = str(row["query_id"])
        inventory.append(
            {
                "schema_version": FEATURE_INVENTORY_SCHEMA_V6_1,
                "trajectory_id": trajectory_id,
                "query_id": query_id,
                "cluster_id": str(row["cluster_id"]),
                "acquisition_split": str(row["acquisition_split"]),
                "source": str(row["source"]),
                "source_subject": row.get("source_subject"),
                "source_level": row.get("source_level"),
                "candidate_index": int(row["candidate_index"]),
                "output_token_count": len(row["output_token_ids"]),
                "prompt_token_count": len(row["prompt_token_ids"]),
                "condition_feature_owner": (
                    condition_owner[query_id] == trajectory_id
                ),
                "uses": sorted(usage_by_id[trajectory_id]),
                "relation_ids": sorted(relation_ids_by_id[trajectory_id]),
            }
        )

    def storage_slice(
        view_ids: set[str], query_ids: set[str]
    ) -> dict[str, Any]:
        output_tokens = sum(
            len(row_by_id[trajectory_id]["output_token_ids"])
            for trajectory_id in view_ids
        )
        prompt_tokens = sum(len(prompt_ids_by_query[query_id]) for query_id in query_ids)
        total_tokens = output_tokens + prompt_tokens
        feature_bytes = total_tokens * feature_bytes_per_token
        return {
            "trajectory_count": len(view_ids),
            "unique_prompt_count": len(query_ids),
            "output_token_count": output_tokens,
            "prompt_token_count": prompt_tokens,
            "total_feature_token_count": total_tokens,
            "feature_bytes": feature_bytes,
            "feature_gib": feature_bytes / (1024**3),
            "feature_gb": feature_bytes / 1_000_000_000,
        }

    positive_queries = {str(row_by_id[value]["query_id"]) for value in positive_ids}
    negative_queries = {str(row_by_id[value]["query_id"]) for value in negative_ids}
    extra_negative_ids = negative_ids - positive_ids
    extra_negative_queries = negative_queries - positive_queries
    selected_queries = positive_queries | negative_queries
    report = {
        "feature_bytes_per_token": feature_bytes_per_token,
        "positive_only": storage_slice(positive_ids, positive_queries),
        "hard_negative_increment": storage_slice(
            extra_negative_ids, extra_negative_queries
        ),
        "final_selected": storage_slice(set(selected_ids), selected_queries),
        "hard_negative_endpoint_count": len(negative_ids),
        "hard_negative_overlap_with_positive_views": len(
            negative_ids & positive_ids
        ),
        "inventory_ordered_rows_sha256": canonical_sha256(inventory),
    }
    return inventory, report


def build_scale_post_annotation_plan_v6_1(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    raw_gate_report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Apply the user-approved hard-negative-only v6.1 amendment fail closed."""

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
            positive_contract["heldout_select_first_by_frozen_annotation_order"]
        ),
    )
    negatives, negative_report = build_scale_heldout_hard_negatives_v6_1(
        heldout_positive_relations=heldout,
        materialized_rows=materialized_rows,
        contract=amendment["hard_negative_contract"],
    )
    passed = negative_report["status"] == (
        "PASS_SCALE_V6_1_HELDOUT_HARD_NEGATIVES"
    )
    inventory: list[dict[str, Any]] = []
    storage_report: dict[str, Any] = {}
    if passed:
        inventory, storage_report = build_scale_feature_inventory_v6_1(
            train_positive_relations=train,
            heldout_positive_relations=heldout,
            heldout_hard_negatives=negatives,
            materialized_rows=materialized_rows,
            feature_bytes_per_token=int(
                amendment["storage_contract"]["full_feature_bytes_per_token"]
            ),
        )
    train_queries = {str(row["query_id"]) for row in train}
    train_clusters = {str(row["cluster_id"]) for row in train}
    negative_queries = {
        str(row[field])
        for row in negatives
        for field in ("left_query_id", "right_query_id")
    }
    negative_clusters = {
        str(row[field])
        for row in negatives
        for field in ("left_cluster_id", "right_cluster_id")
    }
    cross_split_negative_query_overlap = len(train_queries & negative_queries)
    cross_split_negative_cluster_overlap = len(train_clusters & negative_clusters)
    if cross_split_negative_query_overlap or cross_split_negative_cluster_overlap:
        raise ValueError("scale-v6.1 negative endpoints leak across acquisition splits")
    report = {
        "schema_version": POST_ANNOTATION_PLAN_SCHEMA_V6_1,
        "status": (
            "PASS_SCALE_V6_1_POST_ANNOTATION_PLAN"
            if passed
            else "STOP_SCALE_V6_1_POST_ANNOTATION_PLAN"
        ),
        "amendment_evidence_tier": (
            "post_failure_engineering_amendment_not_blind_preregistration"
        ),
        "positive_selection": positive_report,
        "positive_selection_contract_changed": False,
        "heldout_hard_negatives": negative_report,
        "selected_feature_inventory": storage_report,
        "negative_endpoint_cross_split_query_overlap": (
            cross_split_negative_query_overlap
        ),
        "negative_endpoint_cross_split_cluster_overlap": (
            cross_split_negative_cluster_overlap
        ),
        "publishable_relation_manifests_allowed": passed,
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "next_gate": (
            "SEPARATE_FEATURE_EXTRACTION_AUTHORIZATION"
            if passed
            else "STOP_NO_THRESHOLD_REPAIR"
        ),
        "provider_or_third_model_call_used": False,
        "threshold_changes_applied": False,
        "raw_annotation_gates_reinterpreted": False,
    }
    return {
        "train": train,
        "heldout": heldout,
        "negatives": negatives,
        "inventory": inventory,
    }, report


__all__ = [
    "FEATURE_INVENTORY_SCHEMA_V6_1",
    "HARD_NEGATIVE_RELATION_SCHEMA",
    "HARD_NEGATIVE_RELATION_SCHEMA_V6_1",
    "POSITIVE_RELATION_SCHEMA",
    "POST_ANNOTATION_PLAN_SCHEMA",
    "POST_ANNOTATION_PLAN_SCHEMA_V6_1",
    "build_scale_heldout_hard_negatives",
    "build_scale_heldout_hard_negatives_v6_1",
    "build_scale_feature_inventory_v6_1",
    "build_scale_post_annotation_plan",
    "build_scale_post_annotation_plan_v6_1",
    "deterministic_preferred_maximum_matching",
    "select_scale_positive_relations",
]
