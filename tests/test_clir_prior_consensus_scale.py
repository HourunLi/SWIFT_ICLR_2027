from __future__ import annotations

import pytest

from prepare_clir_prior_scale_v12 import _derive_query_seed, _validate_shard_rows
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


def test_prior_v12_rollout_rows_bind_exact_frozen_prompt_and_provenance() -> None:
    protocol = {
        "generation": {
            "candidate_count": 2,
            "seed_namespace": "test-prior-v12-seed",
            "base_seed": 17,
            "model_revision": "model-revision",
            "tokenizer_revision": "tokenizer-revision",
            "backend_version": "vllm-version",
        }
    }
    query = {
        "query_id": "gsm8k:train:00001",
        "source": "gsm8k",
        "question": "What is 1+1?",
        "reference_answer": "2",
        "cluster_id": "cluster-1",
        "prior_label_split": "train",
        "prompt_token_ids": [1, 2, 3],
    }
    shard = {
        "shard_id": "prior-000",
        "query_ids": [query["query_id"]],
        "expected_candidate_rows": 2,
    }
    provenance = {
        "protocol_file_sha256": "protocol-hash",
        "pre_rollout_registry_file_sha256": "registry-hash",
        "authorization_file_sha256": "authorization-hash",
        "code_commit": "commit",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "vllm_version": "vllm-version",
    }
    rows = []
    for candidate_index in range(2):
        rows.append(
            {
                "id": f"{query['query_id']}:cand:{candidate_index:03d}",
                "query_id": query["query_id"],
                "candidate_index": candidate_index,
                "shard_id": shard["shard_id"],
                "source": query["source"],
                "question": query["question"],
                "reference_answer": query["reference_answer"],
                "cluster_id": query["cluster_id"],
                "prior_label_split": query["prior_label_split"],
                "prompt_token_ids": list(query["prompt_token_ids"]),
                "output_token_ids": [10 + candidate_index],
                "response": str(candidate_index),
                "sampling_seed": _derive_query_seed(protocol, query["query_id"]),
                "finish_reason": "stop",
                "decode_matches_backend_text": True,
                "provenance": dict(provenance),
            }
        )
    report = _validate_shard_rows(
        rows,
        shard=shard,
        query_by_id={query["query_id"]: query},
        protocol=protocol,
        protocol_file_sha256="protocol-hash",
        authorization_file_sha256="authorization-hash",
        registry_file_sha256="registry-hash",
    )
    assert report["queries"] == 1
    assert report["rows"] == 2
    assert report["exact_prompt_token_ids_match_freeze"] is True

    rows[0]["prompt_token_ids"] = [1, 2, 4]
    rows[1]["prompt_token_ids"] = [1, 2, 4]
    with pytest.raises(ValueError, match="exact prompt token IDs drift"):
        _validate_shard_rows(
            rows,
            shard=shard,
            query_by_id={query["query_id"]: query},
            protocol=protocol,
            protocol_file_sha256="protocol-hash",
            authorization_file_sha256="authorization-hash",
            registry_file_sha256="registry-hash",
        )
