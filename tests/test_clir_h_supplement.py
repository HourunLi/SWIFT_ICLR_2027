from __future__ import annotations

from src.clir_h_supplement import (
    build_supplement_shards,
    select_supplement_queries,
)


def _protocol() -> dict:
    return {
        "schema_version": "clir-h0-fresh-supplement-v7.2",
        "fresh_source_pool": {
            "cluster_namespace": "test-supplement",
            "math": {"selected_level": 2, "minimum_official_solution_words": 5},
        },
        "preassigned_cells": [
            {
                "source": "math",
                "checker_status": "numeric_match",
                "label_split": "train",
                "query_count": 4,
            },
            {
                "source": "math",
                "checker_status": "numeric_match",
                "label_split": "dev",
                "query_count": 2,
            },
            {
                "source": "gsm8k",
                "checker_status": "numeric_mismatch",
                "label_split": "dev",
                "query_count": 4,
            },
        ],
        "query_count": 10,
        "candidate_count": 16,
        "rollout_shards": 2,
        "expected_candidate_rows": 160,
    }


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(10):
        rows.append(
            {
                "query_id": f"math:train:algebra:{index:05d}",
                "cluster_id": f"math-cluster-{index}",
                "source": "math",
                "source_subject": "algebra" if index % 2 else "geometry",
                "source_level": 2,
                "source_solution": "one two three four five six",
            }
        )
    rows.append(
        {
            "query_id": "math:wrong-level",
            "cluster_id": "math-wrong-level",
            "source": "math",
            "source_subject": "algebra",
            "source_level": 3,
            "source_solution": "one two three four five six",
        }
    )
    for index in range(8):
        rows.append(
            {
                "query_id": f"gsm8k:train:{index:05d}",
                "cluster_id": f"gsm-cluster-{index}",
                "source": "gsm8k",
                "reference_reasoning_word_count": 45 + index,
            }
        )
    duplicate_cluster = dict(rows[0])
    duplicate_cluster["query_id"] = "math:train:duplicate-cluster"
    rows.append(duplicate_cluster)
    return rows


def test_fresh_supplement_is_stratified_and_cluster_unique() -> None:
    selected, report = select_supplement_queries(_rows(), _protocol())

    assert len(selected) == 10
    assert len({row["query_id"] for row in selected}) == 10
    assert len({row["cluster_id"] for row in selected}) == 10
    assert all(row["source"] != "math" or row["source_level"] == 2 for row in selected)
    assert report["selected_by_cell"] == {
        "numeric_match|dev|math": 2,
        "numeric_match|train|math": 4,
        "numeric_mismatch|dev|gsm8k": 4,
    }

    shards = build_supplement_shards(selected, _protocol())
    assert len(shards) == 2
    assert [row["query_count"] for row in shards] == [5, 5]
    assert sum(row["expected_candidate_rows"] for row in shards) == 160
