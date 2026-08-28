from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepare_clir_scale import _derive_query_seed, _validate_shard_rows
from src.clir_scale import (
    build_rollout_shards,
    build_source_candidates,
    build_template_clusters,
    combine_permanent_exclusions,
    entity_template_signature,
    gsm8k_long_chain_metrics,
)
from src.clir_smoke import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "configs/data_expansion_scale_v6/protocol.json").read_text(
        encoding="utf-8"
    )
)
AUTHORIZATION_PATH = (
    ROOT / "configs/data_expansion_scale_v6/rollout_authorization.json"
)


def _source_row(source: str, query_id: str, question: str) -> dict:
    row = {
        "source": source,
        "query_id": query_id,
        "source_record_id": query_id,
        "question": question,
        "reference_answer": "1",
        "source_license": "MIT",
    }
    if source == "math":
        row.update(
            {
                "source_subject": "algebra",
                "source_level": 3,
                "source_solution": "word " * 50,
            }
        )
    return row


def test_entity_template_and_long_chain_contract() -> None:
    left = entity_template_signature("Alice bought 12 apples in Paris.")
    right = entity_template_signature("Bob bought 7 apples in London.")
    assert left == right == "<ent> bought <num> apples in <ent>"

    cfg = PROTOCOL["sources"]["gsm8k"]["long_chain_filter"]
    reasoning = (
        ("Carefully determine the intermediate amounts before combining them. " * 5)
        + "First <<2+3=5>>. Then <<5*4=20>>. Finally <<20-6=14>>. #### 14"
    )
    metrics = gsm8k_long_chain_metrics(reasoning, cfg)
    assert metrics["long_chain_filter_pass"] is True
    assert metrics["reference_distinct_intermediate_numeric_count"] == 3
    assert gsm8k_long_chain_metrics("Short <<1+1=2>>. #### 2", cfg)[
        "long_chain_filter_pass"
    ] is False


def test_source_filter_and_permanent_exclusion_merge() -> None:
    math = _source_row(
        "math", "math:train:algebra:99990", "Find 2 plus 3."
    )
    gsm = _source_row(
        "gsm8k", "gsm8k:train:99990", "Alice has a long arithmetic story."
    )
    gsm["reference_answer"] = (
        ("Carefully determine each intermediate amount before the total. " * 6)
        + "<<2+3=5>> <<5*4=20>> <<20-6=14>> #### 14"
    )
    candidates, report = build_source_candidates([math, gsm], PROTOCOL)
    assert {row["source"] for row in candidates} == {"math", "gsm8k"}
    assert report["counts"]["math_eligible"] == 1
    assert report["counts"]["gsm8k_eligible"] == 1

    exclusions = combine_permanent_exclusions(
        [{"query_id": gsm["query_id"], "reasons": ["old_train"]}],
        [dict(math, acquisition_batch="reserve")],
    )
    assert [row["query_id"] for row in exclusions] == sorted(
        [math["query_id"], gsm["query_id"]]
    )
    assert exclusions[0]["reasons"] or exclusions[1]["reasons"]


def test_excluded_anchor_propagates_through_template_cluster() -> None:
    candidate_a = _source_row(
        "math",
        "math:train:algebra:99991",
        "Alice bought 12 apples in Paris.",
    )
    candidate_b = _source_row(
        "math",
        "math:train:algebra:99992",
        "Bob bought 7 apples in London.",
    )
    anchor = _source_row(
        "gsm8k",
        "gsm8k:train:99991",
        "Carol bought 4 apples in Rome.",
    )
    clusters, selectable, report = build_template_clusters(
        [candidate_a, candidate_b], [anchor], [anchor["query_id"]]
    )
    containing = [
        row for row in clusters if candidate_a["query_id"] in row["member_query_ids"]
    ]
    assert len(containing) == 1
    assert containing[0]["excluded_by_prior_membership"] is True
    assert containing[0]["selectable_query_ids"] == []
    assert selectable == []
    assert report["excluded_clusters"] == 1


def test_rollout_shards_are_source_balanced_and_exact_partition() -> None:
    train = [
        {"source": "math", "query_id": f"math:train:algebra:{index:05d}"}
        for index in range(1050)
    ] + [
        {"source": "gsm8k", "query_id": f"gsm8k:train:{index:05d}"}
        for index in range(450)
    ]
    heldout = [
        {"source": "math", "query_id": f"math:train:prealgebra:{index:05d}"}
        for index in range(350)
    ] + [
        {"source": "gsm8k", "query_id": f"gsm8k:train:{index + 10000:05d}"}
        for index in range(150)
    ]
    shards = build_rollout_shards(train, heldout, PROTOCOL)
    assert len(shards) == 40
    assert all(row["query_count"] == 50 for row in shards)
    assert all(row["source_counts"] == {"math": 35, "gsm8k": 15} for row in shards)
    query_ids = [query_id for shard in shards for query_id in shard["query_ids"]]
    assert len(query_ids) == len(set(query_ids)) == 2000
    assert all(row["expected_candidate_rows"] == 400 for row in shards)


def test_rollout_authorization_is_rollout_only() -> None:
    authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    assert authorization["status"] == "AUTHORIZED_ROLLOUT_ONLY"
    assert authorization["authorized_scope"] == {
        "rollout": True,
        "checker_and_unitizer_materialization": False,
        "annotation": False,
        "feature_extraction": False,
        "training": False,
        "threshold_or_query_manifest_change": False,
    }
    assert authorization["runtime_contract"]["first_calibration_shard"] == (
        "train-000"
    )


def test_rollout_shard_row_contract_rejects_prompt_drift() -> None:
    query_id = "math:train:algebra:99993"
    query = {
        "query_id": query_id,
        "source": "math",
        "question": "What is 2 plus 3?",
        "reference_answer": "5",
        "cluster_id": "cluster-1",
        "acquisition_split": "train_acquisition",
        "prompt_token_count": 3,
    }
    shard = {
        "shard_id": "train-test",
        "query_ids": [query_id],
        "expected_candidate_rows": 8,
    }
    authorization_sha = file_sha256(AUTHORIZATION_PATH)
    protocol_sha = file_sha256(
        ROOT / "configs/data_expansion_scale_v6/protocol.json"
    )
    provenance = {
        "protocol_file_sha256": protocol_sha,
        "pre_rollout_registry_file_sha256": "registry-sha",
        "authorization_file_sha256": authorization_sha,
        "code_commit": "test-commit",
        "model_revision": PROTOCOL["generation"]["model_revision"],
        "tokenizer_revision": PROTOCOL["generation"]["tokenizer_revision"],
        "vllm_version": PROTOCOL["generation"]["backend_version"],
    }
    seed = _derive_query_seed(PROTOCOL["generation"]["seed"], query_id)
    rows = [
        {
            "id": f"{query_id}:cand:{index:03d}",
            "query_id": query_id,
            "candidate_index": index,
            "shard_id": "train-test",
            "acquisition_split": "train_acquisition",
            "cluster_id": "cluster-1",
            "source": "math",
            "question": query["question"],
            "reference_answer": "5",
            "prompt_token_ids": [1, 2, 3],
            "output_token_ids": [100 + index],
            "response": str(index),
            "sampling_seed": seed,
            "decode_matches_backend_text": True,
            "finish_reason": "stop",
            "provenance": provenance,
        }
        for index in range(8)
    ]
    report = _validate_shard_rows(
        rows,
        shard=shard,
        query_by_id={query_id: query},
        protocol=PROTOCOL,
        protocol_file_sha256=protocol_sha,
        authorization_file_sha256=authorization_sha,
        registry_file_sha256="registry-sha",
    )
    assert report["rows"] == 8
    assert report["total_output_tokens"] == 8
    assert report["finish_reason_counts"] == {"stop": 8}

    drifted = [dict(row, prompt_token_ids=[1, 2]) for row in rows]
    with pytest.raises(ValueError, match="prompt token count differs"):
        _validate_shard_rows(
            drifted,
            shard=shard,
            query_by_id={query_id: query},
            protocol=PROTOCOL,
            protocol_file_sha256=protocol_sha,
            authorization_file_sha256=authorization_sha,
            registry_file_sha256="registry-sha",
        )
