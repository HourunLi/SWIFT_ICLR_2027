"""Pure contracts for the post-hoc exploratory H0 v7.4 experiment.

The original v7 dual-AI H0 collection remains a terminal failure.  This module
only supports the separately authorized, post-hoc triple-consensus subset.  It
also freezes a mechanically evaluable ranking population: a query is retained
only when all sixteen candidates have finite binary checker labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from numbers import Integral
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VALID_RANKING_CHECKER_STATUSES = {"numeric_match", "numeric_mismatch"}
H_LABEL_NAME = "silver_posthoc_triple_consensus_h0_v7_4"


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


def _index_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        row_id = str(row.get("id", ""))
        if not row_id or row_id in seen:
            raise ValueError(f"missing or duplicate trajectory id: {row_id!r}")
        seen.add(row_id)
        output.append(row)
    return output


def validate_h_partition(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    expected_clean: int,
    expected_positive: int,
) -> dict[str, Any]:
    """Validate one frozen H train/dev partition without changing its order."""

    indexed = _index_rows(rows)
    queries: set[str] = set()
    statuses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in indexed:
        row_id = str(row["id"])
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in queries:
            raise ValueError(f"{row_id}: missing or duplicate H query_id")
        queries.add(query_id)
        if row.get("h_label_name") != H_LABEL_NAME:
            raise ValueError(f"{row_id}: unexpected H label name")
        if row.get("h_label_split") != split:
            raise ValueError(f"{row_id}: H split drift")
        if row.get("h_posthoc_exploratory") is not True:
            raise ValueError(f"{row_id}: H row is not marked post-hoc exploratory")
        if row.get("h_original_v7_status") != "FAIL_H0_V7_RESERVE":
            raise ValueError(f"{row_id}: original v7 terminal status was erased")
        status = str(row.get("h_status", ""))
        if status not in {"clean", "hallucinated"}:
            raise ValueError(f"{row_id}: unsupported H status")
        onset = _integer(row.get("hallucination_onset"), field="hallucination_onset", row_id=row_id)
        output_ids = _token_ids(
            row.get("output_token_ids"), field="output_token_ids", row_id=row_id
        )
        _token_ids(row.get("prompt_token_ids"), field="prompt_token_ids", row_id=row_id)
        if status == "clean" and onset != -1:
            raise ValueError(f"{row_id}: clean H row must use onset -1")
        if status == "hallucinated" and not 0 <= onset < len(output_ids):
            raise ValueError(f"{row_id}: positive H onset is outside the output")
        correctness = row.get("correctness")
        if correctness not in {0, 1}:
            raise ValueError(f"{row_id}: H correctness must be binary")
        statuses[status] += 1
        sources[str(row.get("source"))] += 1
    expected = {"clean": expected_clean, "hallucinated": expected_positive}
    if dict(statuses) != expected:
        raise ValueError(f"H {split} class counts drift: {dict(statuses)} != {expected}")
    return {
        "rows": len(indexed),
        "queries": len(queries),
        "status_counts": dict(sorted(statuses.items())),
        "source_counts": dict(sorted(sources.items())),
    }


def select_fully_labeled_ranking(
    rows: Sequence[Mapping[str, Any]], *, candidate_count: int = 16
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep whole queries whose frozen candidates all have binary checker labels."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    indexed = _index_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in indexed:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            raise ValueError(f"{row['id']}: ranking row is missing query_id")
        grouped[query_id].append(row)

    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    informative_queries = 0
    source_queries: Counter[str] = Counter()
    for query_id, query_rows in grouped.items():
        candidate_indices = [
            _integer(row.get("candidate_index"), field="candidate_index", row_id=str(row["id"]))
            for row in query_rows
        ]
        if len(query_rows) != candidate_count or sorted(candidate_indices) != list(
            range(candidate_count)
        ):
            rejected["candidate_contract"] += 1
            continue
        query_rows.sort(key=lambda row: int(row["candidate_index"]))
        if any(row.get("evaluation_only") is not True for row in query_rows):
            raise ValueError(f"{query_id}: ranking row is not evaluation-only")
        if any(
            row.get("checker_status") not in VALID_RANKING_CHECKER_STATUSES
            or row.get("eligible_for_supervision") is not True
            or row.get("correctness") not in {0, 1}
            for row in query_rows
        ):
            rejected["non_binary_checker_label"] += 1
            continue
        prompts = {
            tuple(
                _token_ids(
                    row.get("prompt_token_ids"),
                    field="prompt_token_ids",
                    row_id=str(row["id"]),
                )
            )
            for row in query_rows
        }
        if len(prompts) != 1:
            raise ValueError(f"{query_id}: ranking prompt token IDs drift within query")
        for row in query_rows:
            _token_ids(
                row.get("output_token_ids"),
                field="output_token_ids",
                row_id=str(row["id"]),
            )
        labels = {int(row["correctness"]) for row in query_rows}
        informative_queries += int(labels == {0, 1})
        source_queries[str(query_rows[0].get("source"))] += 1
        selected.extend(query_rows)

    statistics = {
        "input_rows": len(indexed),
        "input_queries": len(grouped),
        "selected_rows": len(selected),
        "selected_queries": len(selected) // candidate_count,
        "candidate_count": candidate_count,
        "informative_queries_with_both_labels": informative_queries,
        "source_query_counts": dict(sorted(source_queries.items())),
        "rejected_query_counts": dict(sorted(rejected.items())),
        "selection_uses_clir_scores": False,
    }
    return selected, statistics


def _minimal_feature_row(source: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    row_id = str(source["id"])
    output = {
        "schema_version": "clir-h0-v7.4-selected-feature-input",
        "id": row_id,
        "trajectory_id": row_id,
        "query_id": str(source["query_id"]),
        "candidate_index": int(source["candidate_index"]),
        "source": source.get("source"),
        "source_record_id": source.get("source_record_id"),
        "source_subject": source.get("source_subject"),
        "source_level": source.get("source_level"),
        "cluster_id": source.get("cluster_id"),
        "prompt_token_ids": list(source["prompt_token_ids"]),
        "output_token_ids": list(source["output_token_ids"]),
        "prompt_token_count": len(source["prompt_token_ids"]),
        "output_token_count": len(source["output_token_ids"]),
        "correctness": int(source["correctness"]),
        "feature_role": role,
    }
    if role in {"h_train", "h_dev"}:
        output.update(
            {
                "split": "train" if role == "h_train" else "dev",
                "hallucination_onset": int(source["hallucination_onset"]),
                "path_hallucinated": int(source["path_hallucinated"]),
                "h_status": source["h_status"],
                "h_label_name": source["h_label_name"],
                "hallucination_label_tier": source.get(
                    "hallucination_label_tier"
                ),
                "h_posthoc_exploratory": True,
                "h_original_v7_status": "FAIL_H0_V7_RESERVE",
            }
        )
    elif role == "ranking_evaluation":
        output.update(
            {
                "split": "evaluation",
                "checker_status": source["checker_status"],
                "evaluation_only": True,
            }
        )
    else:  # pragma: no cover - guarded by build_feature_inventory
        raise ValueError(f"unsupported feature role: {role}")
    return output


def build_feature_inventory(
    h_train: Sequence[Mapping[str, Any]],
    h_dev: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the selected-only exact-token inventory in frozen role order."""

    parts = (("h_train", h_train), ("h_dev", h_dev), ("ranking_evaluation", ranking))
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_queries_by_role: dict[str, set[str]] = {}
    for role, rows in parts:
        role_queries: set[str] = set()
        for source in rows:
            row = _minimal_feature_row(source, role=role)
            if row["id"] in seen_ids:
                raise ValueError(f"feature inventory repeats trajectory {row['id']}")
            seen_ids.add(row["id"])
            role_queries.add(row["query_id"])
            row["feature_inventory_index"] = len(result)
            result.append(row)
        seen_queries_by_role[role] = role_queries
    roles = list(seen_queries_by_role)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            overlap = seen_queries_by_role[left] & seen_queries_by_role[right]
            if overlap:
                raise ValueError(f"feature-role query overlap {left}/{right}: {sorted(overlap)[:3]}")
    return result


def inventory_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    roles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in rows:
        grouped[str(row["query_id"])].append(row)
        roles[str(row["feature_role"])] += 1
        sources[str(row.get("source"))] += 1
    output_tokens = sum(int(row["output_token_count"]) for row in rows)
    prompt_tokens = 0
    for query_id, query_rows in grouped.items():
        prompt_counts = {int(row["prompt_token_count"]) for row in query_rows}
        prompt_ids = {tuple(row["prompt_token_ids"]) for row in query_rows}
        if len(prompt_counts) != 1 or len(prompt_ids) != 1:
            raise ValueError(f"{query_id}: prompt contract drift")
        prompt_tokens += next(iter(prompt_counts))
    return {
        "trajectory_count": len(rows),
        "query_count": len(grouped),
        "condition_count": len(grouped),
        "role_row_counts": dict(sorted(roles.items())),
        "source_row_counts": dict(sorted(sources.items())),
        "output_token_count": output_tokens,
        "prompt_token_count": prompt_tokens,
        "total_feature_token_count": output_tokens + prompt_tokens,
    }


def _stable_worker_tiebreak(query_id: str) -> str:
    return hashlib.sha256(
        f"clir-h0-v7.4-feature-worker|{query_id}".encode("utf-8")
    ).hexdigest()


def assign_feature_workers(
    rows: Sequence[Mapping[str, Any]], worker_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Token-balance whole queries across workers with a stable tie-break."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    costs = {
        query_id: int(query_rows[0]["prompt_token_count"])
        + sum(int(row["output_token_count"]) for row in query_rows)
        for query_id, query_rows in grouped.items()
    }
    loads = [0] * worker_count
    query_counts = [0] * worker_count
    row_counts = [0] * worker_count
    assignment: dict[str, int] = {}
    for query_id in sorted(
        grouped, key=lambda value: (-costs[value], _stable_worker_tiebreak(value))
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignment[query_id] = worker
        loads[worker] += costs[query_id]
        query_counts[worker] += 1
        row_counts[worker] += len(grouped[query_id])
    assigned: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["feature_worker_index"] = assignment[str(row["query_id"])]
        assigned.append(row)
    stats = [
        {
            "worker_index": index,
            "query_count": query_counts[index],
            "trajectory_count": row_counts[index],
            "feature_token_count": loads[index],
        }
        for index in range(worker_count)
    ]
    return assigned, stats


def rebase_feature_paths(
    row: Mapping[str, Any], *, source_parent: Path, target_parent: Path
) -> dict[str, Any]:
    """Copy a row while preserving what its relative feature paths point to."""

    result = dict(row)
    for field in ("hidden_states_path", "condition_states_path"):
        raw = Path(str(result[field]))
        absolute = raw if raw.is_absolute() else (source_parent / raw).resolve()
        result[field] = os.path.relpath(absolute, target_parent)
    return result
