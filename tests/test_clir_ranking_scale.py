from __future__ import annotations

from collections import Counter

import pytest

from src.clir_ranking_scale import (
    H_ROLE,
    RANKING_ROLE,
    RANKING_V7_SCHEMA,
    build_role_manifests,
    build_rollout_shards,
    compute_budget,
    one_query_per_cluster,
)


def _protocol() -> dict:
    return {
        "schema_version": RANKING_V7_SCHEMA,
        "roles": {
            RANKING_ROLE: {
                "query_count": 8,
                "source_counts": {"math": 4, "gsm8k": 4},
                "candidate_count": 4,
                "rollout_shards": 2,
                "queries_per_shard": 4,
            },
            H_ROLE: {
                "query_count": 8,
                "source_counts": {"math": 4, "gsm8k": 4},
                "candidate_count": 2,
                "rollout_shards": 2,
                "queries_per_shard": 4,
            },
        },
        "h_acquisition": {
            "preassigned_cells": {
                source: {
                    "numeric_match|dev": 1,
                    "numeric_match|train": 1,
                    "numeric_mismatch|dev": 1,
                    "numeric_mismatch|train": 1,
                }
                for source in ("math", "gsm8k")
            },
            "proposal_target_total": 4,
        },
        "generation": {"max_new_tokens": 16},
        "budget": {
            "expected_output_tokens_per_candidate": {"math": 10, "gsm8k": 5},
            "full_feature_bytes_per_token": 100,
            "maximum_concurrent_l20z_jobs": 2,
        },
    }


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(8):
        rows.append(
            {
                "query_id": f"math:train:algebra:{index:05d}",
                "source": "math",
                "cluster_id": f"math-cluster-{index}",
                "source_subject": "algebra",
                "source_level": 3 + index % 3,
                "prompt_token_count": 10,
            }
        )
        rows.append(
            {
                "query_id": f"gsm8k:train:{index:05d}",
                "source": "gsm8k",
                "cluster_id": f"gsm-cluster-{index}",
                "reference_reasoning_word_count": 50 + index,
                "prompt_token_count": 8,
            }
        )
    return rows


def test_role_manifests_are_query_and_cluster_disjoint() -> None:
    protocol = _protocol()
    ranking, h_rows, report = build_role_manifests(_rows(), protocol)

    assert len(ranking) == len(h_rows) == 8
    assert Counter(row["source"] for row in ranking) == {"math": 4, "gsm8k": 4}
    assert Counter(row["source"] for row in h_rows) == {"math": 4, "gsm8k": 4}
    assert not (
        {row["query_id"] for row in ranking} & {row["query_id"] for row in h_rows}
    )
    assert not (
        {row["cluster_id"] for row in ranking} & {row["cluster_id"] for row in h_rows}
    )
    assert Counter(
        (row["source"], row["h_target_checker_status"], row["h_label_split"])
        for row in h_rows
    ) == {
        (source, checker, split): 1
        for source in ("math", "gsm8k")
        for checker in ("numeric_match", "numeric_mismatch")
        for split in ("dev", "train")
    }
    assert report["query_overlap"] == report["cluster_overlap"] == 0


def test_shards_and_budget_respect_role_specific_candidate_counts() -> None:
    protocol = _protocol()
    ranking, h_rows, _ = build_role_manifests(_rows(), protocol)
    shards = build_rollout_shards(ranking, h_rows, protocol)
    budget = compute_budget(ranking, h_rows, protocol)

    assert len(shards) == 4
    assert Counter(row["role"] for row in shards) == {
        RANKING_ROLE: 2,
        H_ROLE: 2,
    }
    assert {
        row["expected_candidate_rows"] for row in shards if row["role"] == RANKING_ROLE
    } == {16}
    assert {
        row["expected_candidate_rows"] for row in shards if row["role"] == H_ROLE
    } == {8}
    assert budget["candidate_rows_total"] == 48
    assert budget["per_role"][RANKING_ROLE]["candidate_rows"] == 32
    assert budget["per_role"][H_ROLE]["candidate_rows"] == 16


def test_one_query_per_cluster_rejects_cross_source_cluster() -> None:
    rows = _rows()[:2]
    rows[1]["cluster_id"] = rows[0]["cluster_id"]
    with pytest.raises(ValueError, match="mixes source families"):
        one_query_per_cluster(rows)
