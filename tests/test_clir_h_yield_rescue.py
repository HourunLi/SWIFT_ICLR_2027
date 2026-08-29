from __future__ import annotations

import pytest

from src.clir_h_yield_rescue import build_rescue_plan


def _candidate_rows(
    query_id: str,
    *,
    source: str,
    target: str,
    label_split: str,
    survives: bool,
) -> list[dict]:
    rows: list[dict] = []
    other = "numeric_mismatch" if target == "numeric_match" else "numeric_match"
    for candidate_index in range(8):
        rows.append(
            {
                "id": f"{query_id}:cand:{candidate_index:03d}",
                "query_id": query_id,
                "candidate_index": candidate_index,
                "role": "hallucination_acquisition",
                "source": source,
                "source_record_id": query_id,
                "source_subject": "arithmetic",
                "source_level": "1",
                "source_license": "test-only",
                "question": "What is 2 + 3?",
                "reference_answer": "5",
                "cluster_id": f"cluster:{query_id}",
                "prompt_token_count": 12,
                "prompt_token_ids": list(range(12)),
                "h_target_checker_status": target,
                "h_label_split": label_split,
                "checker_status": target
                if survives and candidate_index == 0
                else other,
                "finish_reason": "stop",
                "eligible_for_supervision": True,
                "unitization_status": "ok",
                "material_claim_count": 5,
            }
        )
    return rows


def _protocol() -> dict:
    return {
        "h_acquisition": {
            "minimum_material_units": 5,
            "proposal_target": {"numeric_match|train": {"math": 2, "gsm8k": 1}},
        }
    }


def _amendment() -> dict:
    return {
        "schema_version": "clir-h0-yield-rescue-v7.1",
        "parent_candidate_count": 8,
        "additional_candidates_per_query": 24,
        "candidate_index_start": 8,
        "rollout_shards": 2,
        "rescue_query_count": 2,
        "expected_additional_candidate_rows": 48,
        "observed_fail_yield": {
            "shortages": {
                "numeric_match|train|math": {
                    "target": 2,
                    "available": 1,
                    "shortage": 1,
                }
            }
        },
    }


def test_rescue_freezes_all_and_only_zero_yield_queries_in_short_cells() -> None:
    rows = [
        *_candidate_rows(
            "math:ok",
            source="math",
            target="numeric_match",
            label_split="train",
            survives=True,
        ),
        *_candidate_rows(
            "math:failed-a",
            source="math",
            target="numeric_match",
            label_split="train",
            survives=False,
        ),
        *_candidate_rows(
            "math:failed-b",
            source="math",
            target="numeric_match",
            label_split="train",
            survives=False,
        ),
        *_candidate_rows(
            "gsm:ok",
            source="gsm8k",
            target="numeric_match",
            label_split="train",
            survives=True,
        ),
        *_candidate_rows(
            "gsm:failed",
            source="gsm8k",
            target="numeric_match",
            label_split="train",
            survives=False,
        ),
    ]

    queries, shards, report = build_rescue_plan(rows, _protocol(), _amendment())

    assert {row["query_id"] for row in queries} == {
        "math:failed-a",
        "math:failed-b",
    }
    assert all(row["rescue_cell"] == "numeric_match|train|math" for row in queries)
    assert [row["query_count"] for row in shards] == [1, 1]
    assert [row["expected_candidate_rows"] for row in shards] == [24, 24]
    assert report["expected_additional_candidate_rows"] == 48
    assert report["failed_queries_by_cell"]["numeric_match|train|gsm8k"] == 1


def test_rescue_refuses_observed_shortage_drift() -> None:
    rows = _candidate_rows(
        "math:failed",
        source="math",
        target="numeric_match",
        label_split="train",
        survives=False,
    )
    amendment = _amendment()
    amendment["rescue_query_count"] = 1
    amendment["expected_additional_candidate_rows"] = 24

    with pytest.raises(ValueError, match="observed shortage drift"):
        build_rescue_plan(rows, _protocol(), amendment)
