"""One-shot, query-preserving acquisition rescue after the H0 v7 FAIL-yield."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from src.clir_smoke import canonical_sha256, stable_priority


RESCUE_SCHEMA = "clir-h0-yield-rescue-v7.1"


def _cell(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["h_target_checker_status"]),
            str(row["h_label_split"]),
            str(row["source"]),
        )
    )


def _eligible(row: Mapping[str, Any], minimum_units: int) -> bool:
    return (
        row.get("unitization_status") == "ok"
        and row.get("checker_status") == row.get("h_target_checker_status")
        and row.get("finish_reason") != "length"
        and bool(row.get("eligible_for_supervision"))
        and int(row.get("material_claim_count", 0)) >= minimum_units
    )


def proposal_quotas(protocol: Mapping[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for status_split, sources in protocol["h_acquisition"]["proposal_target"].items():
        status, split = status_split.split("|", 1)
        for source, count in sources.items():
            output[f"{status}|{split}|{source}"] = int(count)
    return dict(sorted(output.items()))


def build_rescue_plan(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze every zero-yield query in a quota-short cell, exactly once."""

    if amendment.get("schema_version") != RESCUE_SCHEMA:
        raise ValueError("unsupported H0 yield-rescue amendment")
    minimum_units = int(protocol["h_acquisition"]["minimum_material_units"])
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_query[str(raw["query_id"])].append(dict(raw))
    survivors_by_cell: Counter[str] = Counter()
    failed_by_cell: dict[str, list[str]] = defaultdict(list)
    for query_id, candidates in sorted(by_query.items()):
        if len(candidates) != int(amendment["parent_candidate_count"]):
            raise ValueError(f"{query_id}: parent candidate count drift")
        indices = sorted(int(row["candidate_index"]) for row in candidates)
        if indices != list(range(int(amendment["parent_candidate_count"]))):
            raise ValueError(f"{query_id}: parent candidate indices drift")
        for row in candidates:
            index = int(row["candidate_index"])
            if row.get("id") != f"{query_id}:cand:{index:03d}":
                raise ValueError(f"{query_id}: parent trajectory ID drift")
        cells = {_cell(row) for row in candidates}
        if len(cells) != 1:
            raise ValueError(f"{query_id}: preassigned H cell drift")
        frozen_fields = (
            "role",
            "source",
            "source_record_id",
            "source_subject",
            "source_level",
            "source_license",
            "question",
            "reference_answer",
            "cluster_id",
            "h_target_checker_status",
            "h_label_split",
        )
        for field in frozen_fields:
            if len({canonical_sha256(row.get(field)) for row in candidates}) != 1:
                raise ValueError(f"{query_id}: frozen parent field {field} drift")
        prompt_hashes = {
            canonical_sha256(row.get("prompt_token_ids")) for row in candidates
        }
        if len(prompt_hashes) != 1 or not isinstance(
            candidates[0].get("prompt_token_ids"), list
        ):
            raise ValueError(f"{query_id}: frozen parent prompt token IDs drift")
        cell = next(iter(cells))
        if any(_eligible(row, minimum_units) for row in candidates):
            survivors_by_cell[cell] += 1
        else:
            failed_by_cell[cell].append(query_id)
    quotas = proposal_quotas(protocol)
    shortages = {
        cell: {
            "target": target,
            "available": int(survivors_by_cell[cell]),
            "shortage": target - int(survivors_by_cell[cell]),
        }
        for cell, target in quotas.items()
        if survivors_by_cell[cell] < target
    }
    expected_shortages = amendment["observed_fail_yield"]["shortages"]
    if shortages != expected_shortages:
        raise ValueError(
            f"observed shortage drift: expected {expected_shortages}, got {shortages}"
        )
    rescue_ids = sorted(
        query_id for cell in shortages for query_id in failed_by_cell[cell]
    )
    if len(rescue_ids) != int(amendment["rescue_query_count"]):
        raise ValueError("rescue query count differs from amendment")
    first_by_query = {query_id: by_query[query_id][0] for query_id in rescue_ids}
    queries: list[dict[str, Any]] = []
    for query_id in rescue_ids:
        parent = first_by_query[query_id]
        queries.append(
            {
                key: parent[key]
                for key in (
                    "query_id",
                    "role",
                    "source",
                    "source_record_id",
                    "source_subject",
                    "source_level",
                    "source_license",
                    "question",
                    "reference_answer",
                    "cluster_id",
                    "prompt_token_count",
                    "prompt_token_ids",
                    "h_target_checker_status",
                    "h_label_split",
                )
                if key in parent
            }
        )
        queries[-1]["prompt_token_count"] = len(parent["prompt_token_ids"])
        queries[-1]["rescue_cell"] = _cell(parent)
        queries[-1]["rescue_priority"] = stable_priority(
            "clir-H-v7.1-rescue-query", query_id
        )
    queries.sort(key=lambda row: row["rescue_priority"])
    shard_count = int(amendment["rollout_shards"])
    shard_members: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(queries):
        shard_members[index % shard_count].append(row)
    candidate_count = int(amendment["additional_candidates_per_query"])
    start = int(amendment["candidate_index_start"])
    shards: list[dict[str, Any]] = []
    for index, members in enumerate(shard_members):
        shard_id = f"rescue-{index:03d}"
        shards.append(
            {
                "shard_id": shard_id,
                "query_count": len(members),
                "candidate_count": candidate_count,
                "candidate_index_start": start,
                "candidate_index_end_exclusive": start + candidate_count,
                "query_ids": [str(row["query_id"]) for row in members],
                "ordered_query_ids_sha256": canonical_sha256(
                    [str(row["query_id"]) for row in members]
                ),
                "source_counts": dict(
                    sorted(Counter(str(row["source"]) for row in members).items())
                ),
                "cell_counts": dict(
                    sorted(Counter(str(row["rescue_cell"]) for row in members).items())
                ),
                "expected_candidate_rows": len(members) * candidate_count,
                "output_path": f"yield_rescue/rollouts/{shard_id}.jsonl",
            }
        )
    if sum(int(row["expected_candidate_rows"]) for row in shards) != int(
        amendment["expected_additional_candidate_rows"]
    ):
        raise ValueError("rescue candidate-row budget differs from amendment")
    return (
        queries,
        shards,
        {
            "parent_queries": len(by_query),
            "minimum_material_units": minimum_units,
            "survivors_by_cell": dict(sorted(survivors_by_cell.items())),
            "failed_queries_by_cell": {
                cell: len(values) for cell, values in sorted(failed_by_cell.items())
            },
            "shortages": shortages,
            "rescue_query_count": len(queries),
            "rescue_queries_by_cell": dict(
                sorted(Counter(str(row["rescue_cell"]) for row in queries).items())
            ),
            "rescue_query_ids_sha256": canonical_sha256(
                [row["query_id"] for row in queries]
            ),
            "rollout_shards": len(shards),
            "expected_additional_candidate_rows": sum(
                int(row["expected_candidate_rows"]) for row in shards
            ),
        },
    )


__all__ = ["RESCUE_SCHEMA", "build_rescue_plan", "proposal_quotas"]
