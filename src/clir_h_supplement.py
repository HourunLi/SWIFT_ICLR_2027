"""Deterministic planning for the fresh H0 v7.2 acquisition supplement."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_smoke import canonical_sha256, stable_priority


SUPPLEMENT_SCHEMA = "clir-h0-fresh-supplement-v7.2"


def _proportional_quotas(
    available: Mapping[str, int], target: int, *, namespace: str
) -> dict[str, int]:
    total = sum(available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot select {target} rows from capacity {total}")
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
    return dict(sorted(quotas.items()))


def _one_per_cluster(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        by_cluster[str(row["cluster_id"])].append(row)
    output: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items()):
        chosen = min(
            members,
            key=lambda row: stable_priority(
                f"{namespace}-cluster-representative", str(row["query_id"])
            ),
        )
        output.append(chosen)
    return output


def _attach_strata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            stable_priority("clir-H-v7.2-gsm-length", str(row["query_id"])),
        ),
    )
    for index, row in enumerate(gsm):
        quantile = min(3, index * 4 // len(gsm))
        row["supplement_stratum"] = f"reasoning_length_q{quantile + 1}"
    for row in output:
        if row["source"] == "math":
            row["supplement_stratum"] = str(row["source_subject"])
    return output


def _select_stratified(
    rows: Sequence[Mapping[str, Any]], target: int, *, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_stratum[str(raw["supplement_stratum"])].append(dict(raw))
    available = {key: len(value) for key, value in by_stratum.items()}
    quotas = _proportional_quotas(available, target, namespace=f"{namespace}-quota")
    selected: list[dict[str, Any]] = []
    for stratum, count in quotas.items():
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: stable_priority(f"{namespace}-row", str(row["query_id"])),
        )
        selected.extend(ordered[:count])
    return selected, {
        "available_by_stratum": dict(sorted(available.items())),
        "selected_by_stratum": quotas,
    }


def select_supplement_queries(
    selectable_rows: Sequence[Mapping[str, Any]],
    supplement: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select fresh, one-per-cluster queries for only the three short cells."""

    if supplement.get("schema_version") != SUPPLEMENT_SCHEMA:
        raise ValueError("unsupported H0 supplement protocol")
    source_cfg = supplement["fresh_source_pool"]
    minimum_words = int(source_cfg["math"]["minimum_official_solution_words"])
    math_level = int(source_cfg["math"]["selected_level"])
    eligible: list[dict[str, Any]] = []
    for raw in selectable_rows:
        row = dict(raw)
        if row["source"] == "math":
            if int(row.get("source_level", -1)) != math_level:
                continue
            if len(str(row.get("source_solution", "")).split()) < minimum_words:
                continue
        elif row["source"] != "gsm8k":
            continue
        eligible.append(row)
    representatives = _attach_strata(
        _one_per_cluster(
            eligible,
            namespace=str(source_cfg["cluster_namespace"]),
        )
    )
    remaining = {str(row["query_id"]): row for row in representatives}
    selected: list[dict[str, Any]] = []
    cell_reports: dict[str, Any] = {}
    for cell in supplement["preassigned_cells"]:
        source = str(cell["source"])
        checker_status = str(cell["checker_status"])
        label_split = str(cell["label_split"])
        count = int(cell["query_count"])
        cell_name = f"{checker_status}|{label_split}|{source}"
        pool = [row for row in remaining.values() if row["source"] == source]
        chosen, report = _select_stratified(
            pool,
            count,
            namespace=f"clir-H-v7.2-{cell_name}",
        )
        for raw in chosen:
            row = dict(raw)
            row["role"] = "hallucination_acquisition_supplement"
            row["h_target_checker_status"] = checker_status
            row["h_label_split"] = label_split
            row["supplement_cell"] = cell_name
            row["role_priority"] = stable_priority(
                "clir-H-v7.2-role", cell_name, str(row["query_id"])
            )
            selected.append(row)
            del remaining[str(row["query_id"])]
        cell_reports[cell_name] = report
    selected.sort(key=lambda row: str(row["role_priority"]))
    expected = int(supplement["query_count"])
    ids = [str(row["query_id"]) for row in selected]
    clusters = [str(row["cluster_id"]) for row in selected]
    if len(selected) != expected or len(set(ids)) != expected:
        raise AssertionError("fresh H0 supplement query count or identity drift")
    if len(set(clusters)) != expected:
        raise AssertionError("fresh H0 supplement reuses a template cluster")
    return selected, {
        "selectable_input_rows": len(selectable_rows),
        "eligible_rows_before_one_per_cluster": len(eligible),
        "eligible_cluster_representatives": len(representatives),
        "eligible_representatives_by_source": dict(
            sorted(Counter(str(row["source"]) for row in representatives).items())
        ),
        "selected_queries": len(selected),
        "selected_by_cell": dict(
            sorted(Counter(str(row["supplement_cell"]) for row in selected).items())
        ),
        "selected_by_source": dict(
            sorted(Counter(str(row["source"]) for row in selected).items())
        ),
        "cell_strata": cell_reports,
        "selected_query_ids_sha256": canonical_sha256(ids),
        "selected_cluster_ids_sha256": canonical_sha256(clusters),
        "remaining_eligible_representatives": len(remaining),
    }


def build_supplement_shards(
    rows: Sequence[Mapping[str, Any]], supplement: Mapping[str, Any]
) -> list[dict[str, Any]]:
    shard_count = int(supplement["rollout_shards"])
    candidate_count = int(supplement["candidate_count"])
    members: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_priority(
            "clir-H-v7.2-shard-order", str(row["query_id"])
        ),
    )
    for index, row in enumerate(ordered):
        members[index % shard_count].append(row)
    shards: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(members):
        shard_id = f"supplement-{index:03d}"
        shards.append(
            {
                "shard_id": shard_id,
                "query_count": len(shard_rows),
                "candidate_count": candidate_count,
                "candidate_index_start": 0,
                "candidate_index_end_exclusive": candidate_count,
                "query_ids": [str(row["query_id"]) for row in shard_rows],
                "ordered_query_ids_sha256": canonical_sha256(
                    [str(row["query_id"]) for row in shard_rows]
                ),
                "source_counts": dict(
                    sorted(Counter(str(row["source"]) for row in shard_rows).items())
                ),
                "cell_counts": dict(
                    sorted(
                        Counter(
                            str(row["supplement_cell"]) for row in shard_rows
                        ).items()
                    )
                ),
                "expected_candidate_rows": len(shard_rows) * candidate_count,
                "output_path": f"supplement_v7_2/rollouts/{shard_id}.jsonl",
            }
        )
    if sum(int(row["expected_candidate_rows"]) for row in shards) != int(
        supplement["expected_candidate_rows"]
    ):
        raise AssertionError("fresh H0 supplement candidate-row budget drift")
    return shards


__all__ = [
    "SUPPLEMENT_SCHEMA",
    "build_supplement_shards",
    "select_supplement_queries",
]
