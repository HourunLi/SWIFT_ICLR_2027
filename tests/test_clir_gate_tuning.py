from __future__ import annotations

from collections import Counter

import pytest

from src.clir_gate_tuning import (
    CONFIRMATION_ROLE,
    PROTOCOL_SCHEMA,
    TUNING_ROLE,
    YieldGateError,
    build_query_manifests,
    build_rollout_shards,
    choose_tuning_axis,
    select_checker_eligible_rows,
)


def _protocol() -> dict:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "population": {
            "selection_namespace": "test-gate-tuning",
            "raw_source_counts": {
                TUNING_ROLE: {"gsm8k": 2, "math": 2},
                CONFIRMATION_ROLE: {"gsm8k": 2, "math": 2},
            },
            "final_selection_namespace": "test-gate-tuning-final",
            "final_source_counts": {
                TUNING_ROLE: {"gsm8k": 1, "math": 1},
                CONFIRMATION_ROLE: {"gsm8k": 1, "math": 1},
            },
        },
        "generation": {
            "candidate_count": 2,
            "queries_per_shard": 2,
            "rollout_shards": 4,
        },
    }


def _selectable() -> list[dict]:
    rows = []
    for source in ("gsm8k", "math"):
        for index in range(6):
            row = {
                "query_id": f"{source}:train:{index:05d}",
                "source": source,
                "cluster_id": f"cluster-{source}-{index}",
                "reference_reasoning_word_count": 30 + index,
            }
            if source == "math":
                row.update(source_subject="algebra", source_level=index % 5 + 1)
            rows.append(row)
    return rows


def test_population_is_deterministic_and_split_cluster_disjoint() -> None:
    protocol = _protocol()
    first = build_query_manifests(_selectable(), protocol)
    second = build_query_manifests(_selectable(), protocol)
    assert first == second
    tuning, confirmation, report = first
    assert Counter(row["source"] for row in tuning) == {"gsm8k": 2, "math": 2}
    assert Counter(row["source"] for row in confirmation) == {
        "gsm8k": 2,
        "math": 2,
    }
    assert {row["query_id"] for row in tuning}.isdisjoint(
        row["query_id"] for row in confirmation
    )
    assert {row["cluster_id"] for row in tuning}.isdisjoint(
        row["cluster_id"] for row in confirmation
    )
    assert all(not row["sealed_until_weight_lock"] for row in tuning)
    assert all(row["sealed_until_weight_lock"] for row in confirmation)
    assert report["selected"]["query_overlap"] == 0


def test_rollout_shards_exactly_partition_both_roles() -> None:
    protocol = _protocol()
    tuning, confirmation, _ = build_query_manifests(_selectable(), protocol)
    shards = build_rollout_shards(tuning, confirmation, protocol)
    assert len(shards) == 4
    assert sum(row["expected_candidate_rows"] for row in shards) == 16
    ids = [query_id for shard in shards for query_id in shard["query_ids"]]
    assert len(ids) == len(set(ids)) == 8


def _materialized(query_rows: list[dict], bad_query: str | None = None) -> list[dict]:
    rows = []
    for query in query_rows:
        for candidate_index in range(2):
            status = "numeric_match" if candidate_index == 0 else "numeric_mismatch"
            correctness = 1 if candidate_index == 0 else 0
            if query["query_id"] == bad_query and candidate_index == 1:
                status = "parse_failure"
                correctness = -1
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{candidate_index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": candidate_index,
                    "checker_status": status,
                    "correctness": correctness,
                }
            )
    return rows


def test_checker_only_final_selection_keeps_whole_candidate_groups() -> None:
    protocol = _protocol()
    tuning, confirmation, _ = build_query_manifests(_selectable(), protocol)
    queries = [*tuning, *confirmation]
    bad_query = next(row["query_id"] for row in tuning if row["source"] == "gsm8k")
    selected_tuning, selected_confirmation, report = select_checker_eligible_rows(
        _materialized(queries, bad_query=bad_query), queries, protocol
    )
    assert len(selected_tuning) == 4
    assert len(selected_confirmation) == 4
    assert all(
        Counter(row["candidate_index"] for row in selected if row["query_id"] == query)
        == {0: 1, 1: 1}
        for selected in (selected_tuning, selected_confirmation)
        for query in {row["query_id"] for row in selected}
    )
    assert report["ineligible_reasons"] == {"nonbinary_checker_status": 1}
    assert report["selection_used_clir_scores"] is False


def test_checker_yield_shortfall_is_terminal() -> None:
    protocol = _protocol()
    tuning, confirmation, _ = build_query_manifests(_selectable(), protocol)
    queries = [*tuning, *confirmation]
    bad_ids = [row["query_id"] for row in tuning if row["source"] == "math"]
    rows = _materialized(queries)
    for row in rows:
        if row["query_id"] in bad_ids and row["candidate_index"] == 1:
            row["checker_status"] = "ambiguous"
            row["correctness"] = -1
    with pytest.raises(YieldGateError):
        select_checker_eligible_rows(rows, queries, protocol)


def test_attribution_opens_only_the_more_negative_axis() -> None:
    ch = {"42": 0.86, "43": 0.85, "44": 0.87}
    direct = {"42": 0.84, "43": 0.84, "44": 0.85}
    full = {"42": 0.83, "43": 0.835, "44": 0.845}
    result = choose_tuning_axis(ch, direct, full)
    assert result["direct_effect_mean"] < result["gate_effect_mean"] < 0
    assert result["selected_tuning_axis"] == "direct_prior"


def test_attribution_skips_tuning_if_both_increments_are_nonnegative() -> None:
    ch = {"42": 0.84, "43": 0.85, "44": 0.86}
    direct = {"42": 0.85, "43": 0.86, "44": 0.87}
    full = {"42": 0.86, "43": 0.87, "44": 0.88}
    assert choose_tuning_axis(ch, direct, full)["selected_tuning_axis"] == "none"
