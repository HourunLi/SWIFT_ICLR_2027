"""Prospective population planning for CLIR Prior/Gate attribution and tuning.

The helpers in this module are deliberately model-free.  They select fresh,
template-cluster-disjoint tuning and sealed-confirmation queries before any
rollout, and later apply a checker-only yield rule without looking at CLIR
scores.  No AI annotation is involved in this ranking-only stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_smoke import (
    canonical_sha256,
    check_numeric_response,
    stable_priority,
    validate_rollout_population,
)


PROTOCOL_SCHEMA = "clir-prior-gate-tuning-v1"
TUNING_ROLE = "weight_tuning"
CONFIRMATION_ROLE = "sealed_confirmation"
ROLE_ORDER = (TUNING_ROLE, CONFIRMATION_ROLE)
GSM_LENGTH_QUANTILES = 4


class YieldGateError(ValueError):
    """Raised when a pre-frozen raw population cannot meet a final quota."""


def build_selected_feature_inventory(
    tuning_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_count: int,
    worker_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strip outcomes and freeze a query-atomic selected-only GPU inventory."""

    from src.clir_scale_features import assign_workers, selected_statistics

    role_rows = {
        TUNING_ROLE: [dict(row) for row in tuning_rows],
        CONFIRMATION_ROLE: [dict(row) for row in confirmation_rows],
    }
    query_sets: dict[str, set[str]] = {}
    trajectory_ids: set[str] = set()
    inventory: list[dict[str, Any]] = []
    role_statistics: dict[str, Any] = {}
    for role in ROLE_ORDER:
        rows = role_rows[role]
        population = validate_rollout_population(rows, candidate_count=candidate_count)
        query_ids = {str(row["query_id"]) for row in rows}
        query_sets[role] = query_ids
        expected_sealed = role == CONFIRMATION_ROLE
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("role") != role:
                raise ValueError(f"{row.get('id')}: feature role drift")
            if bool(row.get("sealed_until_weight_lock")) != expected_sealed:
                raise ValueError(f"{row.get('id')}: feature sealing marker drift")
            trajectory_id = str(row["id"])
            if trajectory_id in trajectory_ids:
                raise ValueError(f"duplicate selected trajectory: {trajectory_id}")
            trajectory_ids.add(trajectory_id)
            by_query[str(row["query_id"])].append(row)
        role_inventory: list[dict[str, Any]] = []
        for query_id in sorted(query_ids):
            query_rows = sorted(
                by_query[query_id], key=lambda row: int(row["candidate_index"])
            )
            if [int(row["candidate_index"]) for row in query_rows] != list(
                range(candidate_count)
            ):
                raise ValueError(f"{query_id}: selected feature candidate axis drift")
            prompts = {
                tuple(int(value) for value in row["prompt_token_ids"])
                for row in query_rows
            }
            if len(prompts) != 1:
                raise ValueError(f"{query_id}: selected feature prompt axis drift")
            for row in query_rows:
                prompt_ids = [int(value) for value in row["prompt_token_ids"]]
                output_ids = [int(value) for value in row["output_token_ids"]]
                if not prompt_ids or not output_ids:
                    raise ValueError(f"{row['id']}: selected feature token axis empty")
                item = {
                    "schema_version": "clir-gate-tuning-v1-feature-inventory-row",
                    "inventory_index": len(inventory) + len(role_inventory),
                    "id": str(row["id"]),
                    "trajectory_id": str(row["id"]),
                    "query_id": query_id,
                    "candidate_index": int(row["candidate_index"]),
                    "role": role,
                    "evaluation_split": str(row["evaluation_split"]),
                    "sealed_until_weight_lock": expected_sealed,
                    "source": str(row["source"]),
                    "cluster_id": str(row["cluster_id"]),
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": output_ids,
                    "prompt_token_count": len(prompt_ids),
                    "output_token_count": len(output_ids),
                    "condition_feature_owner": int(row["candidate_index"]) == 0,
                }
                role_inventory.append(item)
        inventory.extend(role_inventory)
        role_stats = selected_statistics(role_inventory)
        role_statistics[role] = {
            **population,
            **role_stats,
            "source_counts": _counts(
                [by_query[query_id][0] for query_id in sorted(query_ids)], "source"
            ),
        }
    if query_sets[TUNING_ROLE] & query_sets[CONFIRMATION_ROLE]:
        raise ValueError("tuning and confirmation feature queries overlap")
    assigned, worker_statistics = assign_workers(inventory, worker_count)
    total_statistics = selected_statistics(assigned)
    return assigned, {
        "roles": role_statistics,
        "total": total_statistics,
        "worker_count": worker_count,
        "worker_statistics": worker_statistics,
        "query_overlap": 0,
        "trajectory_count": len(trajectory_ids),
        "confirmation_correctness_copied_into_inventory": False,
        "clir_scores_copied_into_inventory": False,
    }


def materialize_numeric_checker_rows(
    raw_rows: Sequence[Mapping[str, Any]], *, checker_version: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen numeric checker without unitization or CLIR scoring.

    A parser failure is deliberately *not* converted into an incorrect
    candidate for this experiment.  It remains auditable in ``correctness``
    but is excluded by the pre-registered all-16-binary eligibility rule.
    """

    processed: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
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
        checker["eligible_for_gate_tuning_population"] = checker.get(
            "checker_status"
        ) in {"numeric_match", "numeric_mismatch"}
        row.update(checker)
        processed.append(row)
        statuses[str(row["checker_status"])] += 1
        finish_reasons[str(row.get("finish_reason"))] += 1
    return processed, {
        "rows": len(processed),
        "queries": len({str(row["query_id"]) for row in processed}),
        "checker_version": checker_version,
        "checker_statuses": dict(sorted(statuses.items())),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "binary_rows": sum(
            status in {"numeric_match", "numeric_mismatch"}
            for status in (str(row["checker_status"]) for row in processed)
        ),
        "clir_scoring_run": False,
        "unitization_run": False,
        "ai_annotation_run": False,
    }


def validate_numeric_checker_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
    checker_version: str,
) -> dict[str, Any]:
    """Recompute and verify checker rows against the immutable rollout axis."""

    if len(rows) != len(raw_rows):
        raise ValueError("checker and raw rollout row counts differ")
    population = validate_rollout_population(rows, candidate_count=candidate_count)
    statuses: Counter[str] = Counter()
    for raw, checked in zip(raw_rows, rows):
        row_id = str(raw.get("id"))
        for field, value in raw.items():
            if field == "reference_answer":
                continue
            if checked.get(field) != value:
                raise ValueError(f"{row_id}: checker materialization changed {field}")
        if checked.get("raw_reference_answer") != str(raw.get("reference_answer")):
            raise ValueError(f"{row_id}: raw reference answer was not preserved")
        expected = check_numeric_response(
            response=str(raw["response"]),
            raw_reference=str(raw["reference_answer"]),
            source=str(raw["source"]),
            finish_reason=raw.get("finish_reason"),
            checker_version=checker_version,
        )
        expected["eligible_for_gate_tuning_population"] = expected.get(
            "checker_status"
        ) in {"numeric_match", "numeric_mismatch"}
        for field, value in expected.items():
            if checked.get(field) != value:
                raise ValueError(f"{row_id}: checker field drift: {field}")
        status = str(checked["checker_status"])
        statuses[status] += 1
        if bool(checked["eligible_for_gate_tuning_population"]) != (
            status in {"numeric_match", "numeric_mismatch"}
        ):
            raise ValueError(f"{row_id}: binary checker eligibility drift")
    return {
        **population,
        "raw_identity_rows_verified": len(rows),
        "checker_rows_recomputed": len(rows),
        "checker_statuses": dict(sorted(statuses.items())),
        "candidate_axis_verified": True,
        "clir_scoring_run": False,
    }


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _proportional_quotas(
    available: Mapping[str, int], target: int, *, namespace: str
) -> dict[str, int]:
    total = sum(int(value) for value in available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot allocate {target} rows from capacity {total}")
    if target == 0:
        return {key: 0 for key in sorted(available)}
    raw = {key: target * int(value) / total for key, value in available.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        available,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            stable_priority(namespace, key),
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if any(quotas[key] > int(available[key]) for key in quotas):
        raise AssertionError("stratified quota exceeded capacity")
    return dict(sorted(quotas.items()))


def one_query_per_cluster(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one deterministic representative from each selectable cluster."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        for field in ("cluster_id", "query_id", "source"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"selectable row lacks {field}")
        grouped[str(row["cluster_id"])].append(row)
    selected: list[dict[str, Any]] = []
    dropped: list[str] = []
    for cluster_id, members in sorted(grouped.items()):
        ordered = sorted(
            members,
            key=lambda row: stable_priority(
                f"{namespace}-cluster-representative", str(row["query_id"])
            ),
        )
        selected.append(ordered[0])
        dropped.extend(str(row["query_id"]) for row in ordered[1:])
    selected.sort(
        key=lambda row: stable_priority(
            f"{namespace}-representative-order", str(row["query_id"])
        )
    )
    return selected, {
        "input_rows": len(rows),
        "unique_clusters": len(grouped),
        "selected_rows": len(selected),
        "dropped_same_cluster_rows": len(dropped),
        "dropped_query_ids_sha256": canonical_sha256(sorted(dropped)),
        "source_counts": _counts(selected, "source"),
    }


def _attach_strata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            stable_priority("clir-gate-tuning-v1-gsm-length", row["query_id"]),
        ),
    )
    for index, row in enumerate(gsm):
        quantile = min(
            GSM_LENGTH_QUANTILES - 1,
            index * GSM_LENGTH_QUANTILES // max(1, len(gsm)),
        )
        row["selection_stratum"] = f"reasoning_length_q{quantile + 1}"
    for row in output:
        if row["source"] == "math":
            subject = str(row.get("source_subject", ""))
            level = int(row.get("source_level", 0))
            if not subject or level not in {1, 2, 3, 4, 5}:
                raise ValueError("MATH row lacks a supported subject/level stratum")
            row["selection_stratum"] = f"{subject}|level_{level}"
        if "selection_stratum" not in row:
            raise ValueError(f"unsupported source {row.get('source')}")
    return output


def _select_stratified(
    rows: Sequence[Mapping[str, Any]], target: int, *, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["selection_stratum"])].append(dict(row))
    available = {key: len(values) for key, values in grouped.items()}
    quotas = _proportional_quotas(available, target, namespace=f"{namespace}-quota")
    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        ordered = sorted(
            grouped[stratum],
            key=lambda row: stable_priority(f"{namespace}-row", str(row["query_id"])),
        )
        selected.extend(ordered[:quota])
    selected.sort(
        key=lambda row: stable_priority(f"{namespace}-selected", str(row["query_id"]))
    )
    return selected, {
        "available_by_stratum": dict(sorted(available.items())),
        "selected_by_stratum": quotas,
    }


def build_query_manifests(
    selectable_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze raw tuning and confirmation populations before rollout."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("gate-tuning population requires protocol v1")
    namespace = str(protocol["population"]["selection_namespace"])
    representatives, representative_report = one_query_per_cluster(
        selectable_rows, namespace=namespace
    )
    representatives = _attach_strata(representatives)
    selected_by_role: dict[str, list[dict[str, Any]]] = {
        role: [] for role in ROLE_ORDER
    }
    report: dict[str, Any] = {"representatives": representative_report}
    used_ids: set[str] = set()
    used_clusters: set[str] = set()
    raw_counts = protocol["population"]["raw_source_counts"]
    for source in ("gsm8k", "math"):
        remaining = [row for row in representatives if row["source"] == source]
        source_report: dict[str, Any] = {"initial_capacity": len(remaining)}
        for role in ROLE_ORDER:
            target = int(raw_counts[role][source])
            chosen, selection_report = _select_stratified(
                remaining,
                target,
                namespace=f"{namespace}-{source}-{role}",
            )
            chosen_ids = {str(row["query_id"]) for row in chosen}
            chosen_clusters = {str(row["cluster_id"]) for row in chosen}
            if chosen_ids & used_ids or chosen_clusters & used_clusters:
                raise AssertionError("tuning/confirmation query or cluster overlap")
            used_ids.update(chosen_ids)
            used_clusters.update(chosen_clusters)
            remaining = [
                row for row in remaining if str(row["query_id"]) not in chosen_ids
            ]
            for raw in chosen:
                row = dict(raw)
                row["role"] = role
                row["evaluation_split"] = (
                    "tuning" if role == TUNING_ROLE else "confirmation"
                )
                row["evaluation_only"] = True
                row["sealed_until_weight_lock"] = role == CONFIRMATION_ROLE
                row["role_priority"] = stable_priority(
                    f"{namespace}-{role}-order", str(row["query_id"])
                )
                selected_by_role[role].append(row)
            source_report[role] = selection_report
        source_report["unused_representatives"] = len(remaining)
        report[source] = source_report

    for role in ROLE_ORDER:
        selected_by_role[role].sort(key=lambda row: str(row["role_priority"]))
        expected = sum(int(value) for value in raw_counts[role].values())
        if len(selected_by_role[role]) != expected:
            raise AssertionError(f"{role} raw query count differs from protocol")
    tuning = selected_by_role[TUNING_ROLE]
    confirmation = selected_by_role[CONFIRMATION_ROLE]
    tuning_ids = {str(row["query_id"]) for row in tuning}
    confirmation_ids = {str(row["query_id"]) for row in confirmation}
    tuning_clusters = {str(row["cluster_id"]) for row in tuning}
    confirmation_clusters = {str(row["cluster_id"]) for row in confirmation}
    if tuning_ids & confirmation_ids or tuning_clusters & confirmation_clusters:
        raise AssertionError("tuning and confirmation populations overlap")
    report["selected"] = {
        "tuning_query_count": len(tuning),
        "confirmation_query_count": len(confirmation),
        "tuning_source_counts": _counts(tuning, "source"),
        "confirmation_source_counts": _counts(confirmation, "source"),
        "query_overlap": 0,
        "cluster_overlap": 0,
    }
    return tuning, confirmation, report


def build_rollout_shards(
    tuning: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generation = protocol["generation"]
    shard_size = int(generation["queries_per_shard"])
    candidate_count = int(generation["candidate_count"])
    shards: list[dict[str, Any]] = []
    for role, rows in ((TUNING_ROLE, tuning), (CONFIRMATION_ROLE, confirmation)):
        if len(rows) % shard_size:
            raise ValueError(f"{role} query count is not divisible by shard size")
        for offset in range(0, len(rows), shard_size):
            index = offset // shard_size
            shard_rows = rows[offset : offset + shard_size]
            shard_id = f"{role}-{index:03d}"
            shards.append(
                {
                    "shard_id": shard_id,
                    "role": role,
                    "query_ids": [str(row["query_id"]) for row in shard_rows],
                    "query_count": len(shard_rows),
                    "candidate_count": candidate_count,
                    "expected_candidate_rows": len(shard_rows) * candidate_count,
                    "output_path": f"rollouts/shards/{shard_id}.jsonl",
                }
            )
    expected = int(generation["rollout_shards"])
    if len(shards) != expected:
        raise AssertionError("rollout shard count differs from protocol")
    return shards


def select_checker_eligible_rows(
    materialized_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the pre-frozen all-16-valid yield rule and source quotas."""

    candidate_count = int(protocol["generation"]["candidate_count"])
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in materialized_rows:
        by_query[str(raw["query_id"])].append(dict(raw))
    query_by_id = {str(row["query_id"]): dict(row) for row in query_rows}
    if set(by_query) != set(query_by_id):
        raise ValueError("materialized/query-manifest query population mismatch")
    binary = {"numeric_match", "numeric_mismatch"}
    eligible: dict[str, list[dict[str, Any]]] = {}
    failure_reasons: Counter[str] = Counter()
    for query_id, rows in by_query.items():
        ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
        indices = [int(row["candidate_index"]) for row in ordered]
        if indices != list(range(candidate_count)):
            failure_reasons["candidate_axis"] += 1
            continue
        if any(row.get("checker_status") not in binary for row in ordered):
            failure_reasons["nonbinary_checker_status"] += 1
            continue
        if any(int(row.get("correctness", -1)) not in {0, 1} for row in ordered):
            failure_reasons["nonbinary_correctness"] += 1
            continue
        eligible[query_id] = ordered

    final_counts = protocol["population"]["final_source_counts"]
    selected_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    selected_queries: dict[str, list[str]] = {role: [] for role in ROLE_ORDER}
    eligibility_counts: dict[str, Any] = {}
    namespace = str(protocol["population"]["final_selection_namespace"])
    for role in ROLE_ORDER:
        role_report: dict[str, Any] = {}
        for source in ("gsm8k", "math"):
            candidates = [
                row
                for query_id, row in query_by_id.items()
                if row["role"] == role
                and row["source"] == source
                and query_id in eligible
            ]
            candidates.sort(
                key=lambda row: stable_priority(
                    f"{namespace}-{role}-{source}", str(row["query_id"])
                )
            )
            target = int(final_counts[role][source])
            if len(candidates) < target:
                raise YieldGateError(
                    f"{role}/{source}: {len(candidates)} eligible queries < {target}"
                )
            chosen = candidates[:target]
            for query in chosen:
                query_id = str(query["query_id"])
                selected_queries[role].append(query_id)
                selected_rows[role].extend(eligible[query_id])
            role_report[source] = {
                "raw_queries": sum(
                    1
                    for row in query_by_id.values()
                    if row["role"] == role and row["source"] == source
                ),
                "eligible_queries": len(candidates),
                "selected_queries": target,
            }
        eligibility_counts[role] = role_report
        selected_rows[role].sort(
            key=lambda row: (
                selected_queries[role].index(str(row["query_id"])),
                int(row["candidate_index"]),
            )
        )
    tuning = selected_rows[TUNING_ROLE]
    confirmation = selected_rows[CONFIRMATION_ROLE]
    expected_tuning = sum(int(v) for v in final_counts[TUNING_ROLE].values())
    expected_confirmation = sum(
        int(v) for v in final_counts[CONFIRMATION_ROLE].values()
    )
    if len(tuning) != expected_tuning * candidate_count:
        raise AssertionError("tuning final row count drift")
    if len(confirmation) != expected_confirmation * candidate_count:
        raise AssertionError("confirmation final row count drift")
    return (
        tuning,
        confirmation,
        {
            "raw_query_count": len(query_rows),
            "eligible_query_count": len(eligible),
            "ineligible_query_count": len(query_rows) - len(eligible),
            "ineligible_reasons": dict(sorted(failure_reasons.items())),
            "by_role_source": eligibility_counts,
            "tuning_selected_query_ids_sha256": canonical_sha256(
                selected_queries[TUNING_ROLE]
            ),
            "confirmation_selected_query_ids_sha256": canonical_sha256(
                selected_queries[CONFIRMATION_ROLE]
            ),
            "selection_used_clir_scores": False,
        },
    )


def choose_tuning_axis(
    ch_k16_by_seed: Mapping[str, float],
    direct_gate0_k16_by_seed: Mapping[str, float],
    full_k16_by_seed: Mapping[str, float],
) -> dict[str, Any]:
    """Choose at most one tuning axis from fresh tuning-set attribution.

    ``direct_effect`` isolates adding Key/Complete supervision with Gate off.
    ``gate_effect`` isolates adding the fixed 0.25 Gate on top of direct Prior.
    If both are non-negative no tuning is opened.  Otherwise only the more
    negative axis is opened, which avoids a post-result two-dimensional grid.
    """

    seeds = sorted(ch_k16_by_seed)
    if seeds != sorted(direct_gate0_k16_by_seed) or seeds != sorted(full_k16_by_seed):
        raise ValueError("attribution cells must have the same seeds")
    if len(seeds) < 3:
        raise ValueError("attribution requires at least three seeds")
    direct_by_seed = {
        seed: float(direct_gate0_k16_by_seed[seed]) - float(ch_k16_by_seed[seed])
        for seed in seeds
    }
    gate_by_seed = {
        seed: float(full_k16_by_seed[seed]) - float(direct_gate0_k16_by_seed[seed])
        for seed in seeds
    }
    direct_mean = sum(direct_by_seed.values()) / len(seeds)
    gate_mean = sum(gate_by_seed.values()) / len(seeds)
    if direct_mean >= 0 and gate_mean >= 0:
        axis = "none"
    elif direct_mean <= gate_mean:
        axis = "direct_prior"
    else:
        axis = "gate"
    return {
        "seeds": seeds,
        "direct_effect_by_seed": direct_by_seed,
        "direct_effect_mean": direct_mean,
        "gate_effect_by_seed": gate_by_seed,
        "gate_effect_mean": gate_mean,
        "selected_tuning_axis": axis,
    }
