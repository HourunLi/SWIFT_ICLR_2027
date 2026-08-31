"""Deterministic planning for CLIR Prior strict-consensus scale v12.

V12 is a prospective acquisition protocol, not a repair or salvage of the
failed v8--v11 smoke rows.  It freezes fresh query/split identities before
rollout, then selects one checker/unitizer-valid trajectory per query for a
large blind dual-AI annotation pool.  A later stage may retain only exact
singleton-Key consensus with non-empty partial Complete consensus.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_prior_partial import _material_units
from src.clir_smoke import canonical_sha256, stable_priority


PROTOCOL_SCHEMA = "clir-prior-strict-consensus-scale-v12"
QUERY_SCHEMA = "clir-prior-v12-acquisition-query"
PROPOSAL_SCHEMA = "clir-prior-v12-natural-proposal"
PACKAGE_SCHEMA = "clir-prior-v12-annotation-package"
PRIVATE_SCHEMA = "clir-prior-v12-private-index"
LABEL_SCHEMA = "clir-prior-v12-label"


def _proportional_quotas(
    available: Mapping[str, int], target: int, *, namespace: str
) -> dict[str, int]:
    total = sum(int(value) for value in available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot select {target} rows from capacity {total}")
    if not available:
        if target:
            raise ValueError("cannot select from an empty stratum map")
        return {}
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


def _one_query_per_cluster(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        for field in ("cluster_id", "query_id", "source"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"selectable row lacks {field}")
        by_cluster[str(row["cluster_id"])].append(row)
    output = []
    for cluster_id, members in sorted(by_cluster.items()):
        sources = {str(row["source"]) for row in members}
        if len(sources) != 1:
            raise ValueError(f"template cluster {cluster_id} mixes sources")
        output.append(
            min(
                members,
                key=lambda row: stable_priority(
                    f"{namespace}-cluster-representative", str(row["query_id"])
                ),
            )
        )
    return output


def _attach_source_strata(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            stable_priority(f"{namespace}-gsm-length", str(row["query_id"])),
        ),
    )
    for index, row in enumerate(gsm):
        quantile = min(3, index * 4 // max(1, len(gsm)))
        row["prior_source_stratum"] = f"reasoning_length_q{quantile + 1}"
    for row in output:
        if row["source"] == "math":
            subject = str(row.get("source_subject", ""))
            level = int(row.get("source_level", 0))
            if not subject or level not in {2, 3, 4, 5}:
                raise ValueError("MATH acquisition row lacks an allowed subject/level")
            row["prior_source_stratum"] = f"{subject}|level_{level}"
        elif row["source"] != "gsm8k":
            raise ValueError(f"unsupported Prior v12 source: {row['source']}")
    return output


def _select_stratified(
    rows: Sequence[Mapping[str, Any]], target: int, *, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_stratum[str(raw["prior_source_stratum"])].append(dict(raw))
    available = {key: len(values) for key, values in by_stratum.items()}
    quotas = _proportional_quotas(
        available, target, namespace=f"{namespace}-quota"
    )
    selected = []
    for stratum, count in quotas.items():
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: stable_priority(
                f"{namespace}-row", str(row["query_id"])
            ),
        )
        selected.extend(ordered[:count])
    return selected, {
        "available_by_stratum": dict(sorted(available.items())),
        "selected_by_stratum": quotas,
    }


def select_acquisition_queries(
    selectable_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze fresh train/dev query identities before any v12 rollout."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior v12 protocol")
    namespace = str(protocol["query_pool"]["namespace"])
    representatives = _attach_source_strata(
        _one_query_per_cluster(selectable_rows, namespace=namespace),
        namespace=namespace,
    )
    remaining = {str(row["query_id"]): row for row in representatives}
    selected: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    source_split_counts = protocol["query_pool"]["source_split_counts"]
    # Dev is frozen first and can never be consumed by train selection.
    for source in ("gsm8k", "math"):
        source_report: dict[str, Any] = {}
        for split in ("dev", "train"):
            count = int(source_split_counts[source][split])
            pool = [
                row for row in remaining.values() if row["source"] == source
            ]
            chosen, report = _select_stratified(
                pool,
                count,
                namespace=f"{namespace}-{source}-{split}",
            )
            for raw in chosen:
                row = dict(raw)
                row["schema_version"] = QUERY_SCHEMA
                row["role"] = "prior_acquisition"
                row["prior_label_split"] = split
                row["role_priority"] = stable_priority(
                    f"{namespace}-role", split, str(row["query_id"])
                )
                selected.append(row)
                del remaining[str(row["query_id"])]
            source_report[split] = report
        reports[source] = source_report

    selected.sort(key=lambda row: str(row["role_priority"]))
    expected = sum(
        int(count)
        for values in source_split_counts.values()
        for count in values.values()
    )
    query_ids = [str(row["query_id"]) for row in selected]
    cluster_ids = [str(row["cluster_id"]) for row in selected]
    if len(selected) != expected or len(set(query_ids)) != expected:
        raise AssertionError("Prior v12 acquisition query count/identity drift")
    if len(set(cluster_ids)) != expected:
        raise AssertionError("Prior v12 acquisition reuses a template cluster")
    return selected, {
        "selectable_rows": len(selectable_rows),
        "cluster_representatives": len(representatives),
        "selected_queries": len(selected),
        "selected_by_source": dict(
            sorted(Counter(str(row["source"]) for row in selected).items())
        ),
        "selected_by_split": dict(
            sorted(
                Counter(str(row["prior_label_split"]) for row in selected).items()
            )
        ),
        "selected_by_source_split": dict(
            sorted(
                Counter(
                    f"{row['source']}|{row['prior_label_split']}" for row in selected
                ).items()
            )
        ),
        "source_strata": reports,
        "selected_query_ids_sha256": canonical_sha256(query_ids),
        "selected_cluster_ids_sha256": canonical_sha256(cluster_ids),
        "remaining_cluster_representatives": len(remaining),
    }


def build_acquisition_shards(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cfg = protocol["generation"]
    shard_count = int(cfg["rollout_shards"])
    candidate_count = int(cfg["candidate_count"])
    members: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_priority(
            "clir-prior-v12-rollout-shard-order", str(row["query_id"])
        ),
    )
    for index, row in enumerate(ordered):
        members[index % shard_count].append(row)
    shards = []
    for index, shard_rows in enumerate(members):
        shard_id = f"prior-{index:03d}"
        query_ids = [str(row["query_id"]) for row in shard_rows]
        shards.append(
            {
                "shard_id": shard_id,
                "query_count": len(shard_rows),
                "candidate_count": candidate_count,
                "candidate_index_start": 0,
                "candidate_index_end_exclusive": candidate_count,
                "query_ids": query_ids,
                "ordered_query_ids_sha256": canonical_sha256(query_ids),
                "source_counts": dict(
                    sorted(Counter(str(row["source"]) for row in shard_rows).items())
                ),
                "split_counts": dict(
                    sorted(
                        Counter(
                            str(row["prior_label_split"]) for row in shard_rows
                        ).items()
                    )
                ),
                "expected_candidate_rows": len(shard_rows) * candidate_count,
                "output_path": f"rollouts/{shard_id}.jsonl",
            }
        )
    expected_queries = int(protocol["query_pool"]["query_count"])
    expected_rows = expected_queries * candidate_count
    if sum(int(row["query_count"]) for row in shards) != expected_queries:
        raise AssertionError("Prior v12 shard query count drift")
    if sum(int(row["expected_candidate_rows"]) for row in shards) != expected_rows:
        raise AssertionError("Prior v12 shard candidate-row count drift")
    return shards


def select_prior_proposals(
    materialized_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen 800-row annotation population after materialization."""

    pool = protocol["proposal_pool"]
    quotas = {
        (
            str(entry["source"]),
            str(entry["checker_status"]),
            str(entry["split"]),
        ): int(entry["count"])
        for entry in pool["strata"]
    }
    if sum(quotas.values()) != int(pool["natural_count"]):
        raise ValueError("Prior v12 proposal strata do not sum to natural_count")
    minimum = int(pool["minimum_material_claims"])
    maximum = int(pool["maximum_material_claims"])
    by_stratum_query: dict[
        tuple[str, str, str], dict[str, list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    rejection: Counter[str] = Counter()
    for row in materialized_rows:
        stratum = (
            str(row.get("source")),
            str(row.get("checker_status")),
            str(row.get("prior_label_split")),
        )
        if stratum not in quotas:
            rejection["outside_strata"] += 1
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
        claims = int(row.get("material_claim_count", 0))
        if not minimum <= claims <= maximum:
            rejection["material_claim_count"] += 1
            continue
        _material_units(row)
        by_stratum_query[stratum][str(row["query_id"])].append(row)

    candidates: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    namespace = str(pool["selection_namespace"])
    for stratum, by_query in by_stratum_query.items():
        one_per_query = []
        for query_id, rows in by_query.items():
            one_per_query.append(
                min(
                    rows,
                    key=lambda row: stable_priority(
                        f"{namespace}-candidate", query_id, str(row["id"])
                    ),
                )
            )
        candidates[stratum] = sorted(
            one_per_query,
            key=lambda row: stable_priority(
                f"{namespace}-query", *stratum, str(row["query_id"]), str(row["id"])
            ),
        )

    protocol_order = [
        (str(row["source"]), str(row["checker_status"]), str(row["split"]))
        for row in pool["strata"]
    ]
    ordered_strata = sorted(
        quotas,
        key=lambda stratum: (
            len(candidates.get(stratum, [])),
            protocol_order.index(stratum),
        ),
    )
    selected = []
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    for stratum in ordered_strata:
        selected_here = 0
        for row in candidates.get(stratum, []):
            query_id = str(row["query_id"])
            cluster_id = str(row["cluster_id"])
            if query_id in used_queries or cluster_id in used_clusters:
                continue
            proposal_id = stable_priority(f"{namespace}-proposal", str(row["id"]))
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
                    "prior_label_split": str(row["prior_label_split"]),
                    "candidate_index": int(row["candidate_index"]),
                    "question": str(row["question"]),
                    "response": str(row["response"]),
                    "material_claim_count": int(row["material_claim_count"]),
                    "output_token_count": int(row["output_token_count"]),
                    "units": _material_units(row),
                    "selection_priority": stable_priority(
                        f"{namespace}-query",
                        *stratum,
                        query_id,
                        str(row["id"]),
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
                f"insufficient Prior v12 proposal capacity for {stratum}: "
                f"{selected_here}/{quotas[stratum]}"
            )
    selected.sort(key=lambda row: str(row["proposal_id"]))
    counts = Counter(
        (row["source"], row["checker_status"], row["prior_label_split"])
        for row in selected
    )
    return selected, {
        "natural_selected": len(selected),
        "unique_queries": len(used_queries),
        "unique_clusters": len(used_clusters),
        "selected_by_stratum": {
            "|".join(stratum): count for stratum, count in sorted(counts.items())
        },
        "available_query_counts": {
            "|".join(stratum): len(candidates.get(stratum, []))
            for stratum in sorted(quotas)
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "ordered_rows_sha256": canonical_sha256(selected),
    }


__all__ = [
    "LABEL_SCHEMA",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PROTOCOL_SCHEMA",
    "QUERY_SCHEMA",
    "build_acquisition_shards",
    "select_acquisition_queries",
    "select_prior_proposals",
]
