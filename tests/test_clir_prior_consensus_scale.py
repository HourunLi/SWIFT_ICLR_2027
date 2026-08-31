from __future__ import annotations

import pytest

from src.clir_prior_consensus_scale import (
    PROTOCOL_SCHEMA,
    build_acquisition_shards,
    select_acquisition_queries,
    select_prior_proposals,
)


def _query(index: int, source: str) -> dict:
    row = {
        "query_id": f"{source}:train:{index:05d}",
        "cluster_id": f"cluster-{source}-{index}",
        "source": source,
        "source_record_id": index,
        "question": f"question {index}",
        "reference_answer": str(index),
        "source_license": "MIT",
        "cluster_split_priority": f"cluster-priority-{index}",
        "query_priority": f"query-priority-{index}",
    }
    if source == "gsm8k":
        row.update(
            {
                "reference_reasoning_word_count": 50 + index,
                "reference_calculation_marker_count": 3,
                "reference_distinct_intermediate_numeric_count": 3,
            }
        )
    else:
        row.update(
            {
                "source_subject": "algebra" if index % 2 else "number_theory",
                "source_level": 2 + index % 4,
                "source_solution": "word " * 30,
            }
        )
    return row


def _protocol() -> dict:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "query_pool": {
            "namespace": "test-prior-v12",
            "query_count": 6,
            "source_split_counts": {
                "gsm8k": {"train": 2, "dev": 1},
                "math": {"train": 2, "dev": 1},
            },
        },
        "generation": {"rollout_shards": 3, "candidate_count": 2},
    }


def test_select_acquisition_queries_freezes_split_and_cluster_identity() -> None:
    rows = [_query(index, "gsm8k") for index in range(8)] + [
        _query(index, "math") for index in range(8, 16)
    ]
    selected, report = select_acquisition_queries(rows, _protocol())
    assert len(selected) == 6
    assert len({row["query_id"] for row in selected}) == 6
    assert len({row["cluster_id"] for row in selected}) == 6
    assert report["selected_by_source"] == {"gsm8k": 3, "math": 3}
    assert report["selected_by_split"] == {"dev": 2, "train": 4}
    assert all(row["role"] == "prior_acquisition" for row in selected)

    shards = build_acquisition_shards(selected, _protocol())
    assert len(shards) == 3
    assert sum(row["query_count"] for row in shards) == 6
    assert sum(row["expected_candidate_rows"] for row in shards) == 12
    assert all(row["query_count"] == 2 for row in shards)


def _materialized(
    index: int, source: str, status: str, split: str, *, cluster: str | None = None
) -> dict:
    return {
        "id": f"trajectory-{index}",
        "query_id": f"query-{index}",
        "cluster_id": cluster or f"cluster-{index}",
        "source": source,
        "source_record_id": index,
        "checker_status": status,
        "prior_label_split": split,
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "finish_reason": "stop",
        "material_claim_count": 6,
        "candidate_index": 0,
        "question": f"question {index}",
        "response": f"response {index}",
        "output_token_count": 20,
        "units": [
            {
                "unit_index": unit * 2,
                "kind": "material_claim",
                "text": f"claim {unit}",
            }
            for unit in range(6)
        ],
    }


def test_select_prior_proposals_uses_all_prefrozen_strata() -> None:
    keys = [
        ("gsm8k", "numeric_match", "train"),
        ("gsm8k", "numeric_match", "dev"),
        ("gsm8k", "numeric_mismatch", "train"),
        ("gsm8k", "numeric_mismatch", "dev"),
        ("math", "numeric_match", "train"),
        ("math", "numeric_match", "dev"),
        ("math", "numeric_mismatch", "train"),
        ("math", "numeric_mismatch", "dev"),
    ]
    rows = [
        _materialized(index, source, status, split)
        for index, (source, status, split) in enumerate(keys)
    ]
    protocol = {
        "proposal_pool": {
            "natural_count": 8,
            "minimum_material_claims": 6,
            "maximum_material_claims": 40,
            "selection_namespace": "test-prior-proposal",
            "strata": [
                {
                    "source": source,
                    "checker_status": status,
                    "split": split,
                    "count": 1,
                }
                for source, status, split in keys
            ],
        }
    }
    selected, report = select_prior_proposals(rows, protocol)
    assert len(selected) == 8
    assert report["unique_queries"] == 8
    assert report["unique_clusters"] == 8
    assert set(report["selected_by_stratum"].values()) == {1}

    with pytest.raises(ValueError, match="insufficient Prior v12 proposal capacity"):
        select_prior_proposals(rows[:-1], protocol)
