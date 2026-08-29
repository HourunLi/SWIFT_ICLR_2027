"""Deterministic query-role planning for CLIR ranking/H expansion v7.

The module is model-free.  It consumes already filtered, template-clustered
train-source rows, keeps at most one query per cluster, and freezes disjoint
ranking-evaluation and hallucination-acquisition populations before rollout.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_smoke import canonical_sha256, stable_priority


RANKING_V7_SCHEMA = "clir-ranking-h-expansion-v7"
RANKING_ROLE = "ranking_evaluation"
H_ROLE = "hallucination_acquisition"
GSM_LENGTH_QUANTILES = 4


def _source_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["source"]) for row in rows).items()))


def _proportional_quotas(
    available: Mapping[str, int], target: int, *, namespace: str
) -> dict[str, int]:
    total = sum(available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot allocate {target} rows from capacity {total}")
    if target == 0:
        return {key: 0 for key in sorted(available)}
    raw = {key: target * count / total for key, count in available.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        available,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            stable_priority(namespace, key),
        ),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    if any(quotas[key] > available[key] for key in quotas):
        raise AssertionError("stratified quota exceeded capacity")
    return dict(sorted(quotas.items()))


def one_query_per_cluster(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        cluster_id = row.get("cluster_id")
        source = row.get("source")
        query_id = row.get("query_id")
        if not all(
            isinstance(value, str) and value for value in (cluster_id, source, query_id)
        ):
            raise ValueError("selectable rows require cluster_id, source, and query_id")
        by_cluster[str(cluster_id)].append(row)

    selected: list[dict[str, Any]] = []
    dropped: list[str] = []
    for cluster_id, members in sorted(by_cluster.items()):
        sources = {str(row["source"]) for row in members}
        if len(sources) != 1:
            raise ValueError(f"cluster {cluster_id} mixes source families")
        ordered = sorted(
            members,
            key=lambda row: stable_priority(
                "clir-ranking-v7-cluster-representative", str(row["query_id"])
            ),
        )
        selected.append(ordered[0])
        dropped.extend(str(row["query_id"]) for row in ordered[1:])
    selected.sort(
        key=lambda row: stable_priority(
            "clir-ranking-v7-representative-order", str(row["query_id"])
        )
    )
    return selected, {
        "input_rows": len(rows),
        "unique_clusters": len(by_cluster),
        "selected_rows": len(selected),
        "dropped_same_cluster_rows": len(dropped),
        "dropped_query_ids_sha256": canonical_sha256(sorted(dropped)),
        "source_counts": _source_counts(selected),
    }


def _attach_selection_strata(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            stable_priority("clir-ranking-v7-gsm-length", str(row["query_id"])),
        ),
    )
    if gsm:
        for index, row in enumerate(gsm):
            quantile = min(
                GSM_LENGTH_QUANTILES - 1,
                index * GSM_LENGTH_QUANTILES // len(gsm),
            )
            row["selection_stratum"] = f"reasoning_length_q{quantile + 1}"
    for row in output:
        if row["source"] == "math":
            subject = str(row.get("source_subject", ""))
            level = int(row.get("source_level", 0))
            if not subject or level not in {3, 4, 5}:
                raise ValueError("MATH row lacks an allowed subject/level stratum")
            row["selection_stratum"] = f"{subject}|level_{level}"
    return output


def _select_stratified(
    rows: Sequence[Mapping[str, Any]],
    target: int,
    *,
    namespace: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_stratum[str(raw["selection_stratum"])].append(dict(raw))
    available = {key: len(values) for key, values in by_stratum.items()}
    quotas = _proportional_quotas(available, target, namespace=f"{namespace}-quota")
    selected: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        ordered = sorted(
            by_stratum[stratum],
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


def _assign_h_cells(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells = protocol["h_acquisition"]["preassigned_cells"]
    output: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for source in ("math", "gsm8k"):
        source_rows = sorted(
            (dict(row) for row in rows if row["source"] == source),
            key=lambda row: stable_priority(
                "clir-ranking-v7-h-cell-order", source, str(row["query_id"])
            ),
        )
        cursor = 0
        source_report: dict[str, int] = {}
        for cell_name in sorted(cells[source]):
            count = int(cells[source][cell_name])
            checker_target, label_split = cell_name.split("|", 1)
            if checker_target not in {"numeric_match", "numeric_mismatch"}:
                raise ValueError(f"invalid H checker target {checker_target}")
            if label_split not in {"train", "dev"}:
                raise ValueError(f"invalid H label split {label_split}")
            chosen = source_rows[cursor : cursor + count]
            if len(chosen) != count:
                raise ValueError(f"insufficient {source} rows for H cell {cell_name}")
            cursor += count
            for row in chosen:
                row["role"] = H_ROLE
                row["h_target_checker_status"] = checker_target
                row["h_label_split"] = label_split
                row["role_priority"] = stable_priority(
                    "clir-ranking-v7-h-role", str(row["query_id"])
                )
                output.append(row)
            source_report[cell_name] = count
        if cursor != len(source_rows):
            raise ValueError(f"H cell quotas do not consume every {source} row")
        report[source] = source_report
    output.sort(key=lambda row: row["role_priority"])
    return output, report


def build_role_manifests(
    selectable_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze cluster-disjoint ranking and H-acquisition query populations."""

    if protocol.get("schema_version") != RANKING_V7_SCHEMA:
        raise ValueError("ranking role planning requires protocol v7")
    representatives, cluster_report = one_query_per_cluster(selectable_rows)
    representatives = _attach_selection_strata(representatives)
    h_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    selection_report: dict[str, Any] = {}
    used_ids: set[str] = set()
    for source in ("math", "gsm8k"):
        pool = [row for row in representatives if row["source"] == source]
        h_target = int(protocol["roles"][H_ROLE]["source_counts"][source])
        rank_target = int(protocol["roles"][RANKING_ROLE]["source_counts"][source])
        selected_h, h_report = _select_stratified(
            pool,
            h_target,
            namespace=f"clir-ranking-v7-{source}-h",
        )
        h_ids = {str(row["query_id"]) for row in selected_h}
        remaining = [row for row in pool if str(row["query_id"]) not in h_ids]
        selected_rank, rank_report = _select_stratified(
            remaining,
            rank_target,
            namespace=f"clir-ranking-v7-{source}-ranking",
        )
        h_rows.extend(selected_h)
        ranking_rows.extend(selected_rank)
        used_ids.update(h_ids)
        used_ids.update(str(row["query_id"]) for row in selected_rank)
        selection_report[source] = {
            "available": len(pool),
            "h": h_report,
            "ranking": rank_report,
            "unused": len(pool) - h_target - rank_target,
        }

    h_rows, h_cell_report = _assign_h_cells(h_rows, protocol)
    ranking_output: list[dict[str, Any]] = []
    for raw in ranking_rows:
        row = dict(raw)
        row["role"] = RANKING_ROLE
        row["evaluation_only"] = True
        row["role_priority"] = stable_priority(
            "clir-ranking-v7-ranking-role", str(row["query_id"])
        )
        ranking_output.append(row)
    ranking_output.sort(key=lambda row: row["role_priority"])

    h_ids = {str(row["query_id"]) for row in h_rows}
    rank_ids = {str(row["query_id"]) for row in ranking_output}
    h_clusters = {str(row["cluster_id"]) for row in h_rows}
    rank_clusters = {str(row["cluster_id"]) for row in ranking_output}
    if h_ids & rank_ids or h_clusters & rank_clusters:
        raise AssertionError("ranking and H roles overlap by query or cluster")
    expected_h = int(protocol["roles"][H_ROLE]["query_count"])
    expected_rank = int(protocol["roles"][RANKING_ROLE]["query_count"])
    if len(h_rows) != expected_h or len(ranking_output) != expected_rank:
        raise AssertionError("role query counts differ from protocol")
    if len(used_ids) != expected_h + expected_rank:
        raise AssertionError("selected role IDs are not unique")
    return (
        ranking_output,
        h_rows,
        {
            "one_query_per_cluster": cluster_report,
            "selection": selection_report,
            "h_preassigned_cells": h_cell_report,
            "ranking_source_counts": _source_counts(ranking_output),
            "h_source_counts": _source_counts(h_rows),
            "ranking_query_count": len(ranking_output),
            "h_query_count": len(h_rows),
            "query_overlap": 0,
            "cluster_overlap": 0,
        },
    )


def _shard_source_counts(total: int, shards: int, shard_size: int) -> list[int]:
    base, remainder = divmod(total, shards)
    counts = [base] * shards
    order = sorted(
        range(shards),
        key=lambda index: stable_priority(
            "clir-ranking-v7-shard-source-remainder", index
        ),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if any(value < 0 or value > shard_size for value in counts):
        raise ValueError("invalid per-shard source allocation")
    return counts


def build_rollout_shards(
    ranking_rows: Sequence[Mapping[str, Any]],
    h_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for role, rows in ((RANKING_ROLE, ranking_rows), (H_ROLE, h_rows)):
        cfg = protocol["roles"][role]
        shard_count = int(cfg["rollout_shards"])
        shard_size = int(cfg["queries_per_shard"])
        candidate_count = int(cfg["candidate_count"])
        if len(rows) != shard_count * shard_size:
            raise ValueError(f"{role} rows do not fill frozen shards")
        expected_sources = {
            key: int(value) for key, value in cfg["source_counts"].items()
        }
        if _source_counts(rows) != expected_sources:
            raise ValueError(f"{role} source counts differ from protocol")
        math_counts = _shard_source_counts(
            expected_sources["math"], shard_count, shard_size
        )
        math_rows = sorted(
            (dict(row) for row in rows if row["source"] == "math"),
            key=lambda row: stable_priority(
                "clir-ranking-v7-shard-math", role, str(row["query_id"])
            ),
        )
        gsm_rows = sorted(
            (dict(row) for row in rows if row["source"] == "gsm8k"),
            key=lambda row: stable_priority(
                "clir-ranking-v7-shard-gsm", role, str(row["query_id"])
            ),
        )
        math_cursor = 0
        gsm_cursor = 0
        prefix = "ranking" if role == RANKING_ROLE else "h"
        for index, math_count in enumerate(math_counts):
            gsm_count = shard_size - math_count
            query_rows = [
                *math_rows[math_cursor : math_cursor + math_count],
                *gsm_rows[gsm_cursor : gsm_cursor + gsm_count],
            ]
            math_cursor += math_count
            gsm_cursor += gsm_count
            query_rows.sort(
                key=lambda row: stable_priority(
                    "clir-ranking-v7-shard-row", role, str(row["query_id"])
                )
            )
            shard_id = f"{prefix}-{index:03d}"
            shards.append(
                {
                    "shard_id": shard_id,
                    "role": role,
                    "query_count": shard_size,
                    "candidate_count": candidate_count,
                    "source_counts": {
                        "math": math_count,
                        "gsm8k": gsm_count,
                    },
                    "query_ids": [str(row["query_id"]) for row in query_rows],
                    "ordered_query_ids_sha256": canonical_sha256(
                        [str(row["query_id"]) for row in query_rows]
                    ),
                    "expected_candidate_rows": shard_size * candidate_count,
                    "output_path": f"rollouts/{shard_id}.jsonl",
                }
            )
        if math_cursor != len(math_rows) or gsm_cursor != len(gsm_rows):
            raise AssertionError(f"{role} sharding did not consume every row")
    expected = sum(
        int(protocol["roles"][role]["rollout_shards"])
        for role in (RANKING_ROLE, H_ROLE)
    )
    if len(shards) != expected:
        raise AssertionError("rollout shard count differs from protocol")
    return shards


def compute_budget(
    ranking_rows: Sequence[Mapping[str, Any]],
    h_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    assumptions = protocol["budget"]
    expected_by_source = assumptions["expected_output_tokens_per_candidate"]
    bytes_per_token = int(assumptions["full_feature_bytes_per_token"])
    role_rows = {RANKING_ROLE: ranking_rows, H_ROLE: h_rows}
    per_role: dict[str, Any] = {}
    expected_tokens_total = 0
    worst_tokens_total = 0
    for role, rows in role_rows.items():
        candidate_count = int(protocol["roles"][role]["candidate_count"])
        counts = Counter(str(row["source"]) for row in rows)
        expected_tokens = sum(
            counts[source] * candidate_count * int(expected_by_source[source])
            for source in counts
        )
        worst_tokens = (
            len(rows) * candidate_count * int(protocol["generation"]["max_new_tokens"])
        )
        expected_tokens_total += expected_tokens
        worst_tokens_total += worst_tokens
        per_role[role] = {
            "query_count": len(rows),
            "source_counts": dict(sorted(counts.items())),
            "candidate_count": candidate_count,
            "candidate_rows": len(rows) * candidate_count,
            "expected_output_tokens": expected_tokens,
            "worst_case_output_tokens": worst_tokens,
        }
    ranking_expected = int(per_role[RANKING_ROLE]["expected_output_tokens"])
    h_selected = int(protocol["h_acquisition"]["proposal_target_total"])
    h_expected_per_candidate = int(
        per_role[H_ROLE]["expected_output_tokens"] / per_role[H_ROLE]["candidate_rows"]
    )
    condition_tokens = sum(
        int(row.get("prompt_token_count", 0)) for row in [*ranking_rows, *h_rows]
    )
    selected_feature_tokens = (
        ranking_expected + h_selected * h_expected_per_candidate + condition_tokens
    )
    return {
        "per_role": per_role,
        "candidate_rows_total": sum(
            int(value["candidate_rows"]) for value in per_role.values()
        ),
        "expected_output_tokens_total": expected_tokens_total,
        "worst_case_output_tokens_total": worst_tokens_total,
        "selected_feature_scope": (
            "all ranking trajectories plus only frozen H proposals and one condition per query"
        ),
        "selected_feature_storage_tb_decimal": (
            selected_feature_tokens * bytes_per_token / 1_000_000_000_000
        ),
        "forbidden_all_rollout_feature_storage_tb_decimal": (
            (expected_tokens_total + condition_tokens)
            * bytes_per_token
            / 1_000_000_000_000
        ),
        "bytes_per_full_feature_token": bytes_per_token,
        "maximum_concurrent_l20z_jobs": int(
            assumptions["maximum_concurrent_l20z_jobs"]
        ),
    }


__all__ = [
    "GSM_LENGTH_QUANTILES",
    "H_ROLE",
    "RANKING_ROLE",
    "RANKING_V7_SCHEMA",
    "build_role_manifests",
    "build_rollout_shards",
    "compute_budget",
    "one_query_per_cluster",
]
