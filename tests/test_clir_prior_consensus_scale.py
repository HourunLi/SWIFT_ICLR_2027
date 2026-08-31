from __future__ import annotations

import pytest

from prepare_clir_prior_scale_v12 import (
    _derive_query_seed,
    _raw_rows_for_shared_materializer,
    _validate_shard_rows,
)
from src.clir_prior_consensus_scale import (
    LABEL_SCHEMA,
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROTOCOL_SCHEMA,
    build_acquisition_shards,
    build_prior_annotation_shards,
    evaluate_prior_v12_labels,
    select_acquisition_queries,
    select_prior_proposals,
    validate_prior_v12_annotation,
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


def test_prior_v12_shared_materializer_aliases_only_the_frozen_split() -> None:
    raw = [{"id": "row-1", "prior_label_split": "dev", "value": 3}]
    aliased = _raw_rows_for_shared_materializer(raw)
    assert aliased == [
        {
            "id": "row-1",
            "prior_label_split": "dev",
            "acquisition_split": "dev",
            "value": 3,
        }
    ]
    assert "acquisition_split" not in raw[0]
    with pytest.raises(ValueError, match="invalid Prior v12 label split"):
        _raw_rows_for_shared_materializer(
            [{"id": "row-2", "prior_label_split": "test"}]
        )


def test_prior_v12_annotation_shards_have_frozen_natural_control_repeat_mix() -> None:
    proposals = [
        {
            "proposal_id": f"proposal-{index:04d}",
            "question": f"question {index}",
            "response": f"response {index}",
            "units": [{"unit_index": 0, "kind": "material_claim", "text": "claim"}],
        }
        for index in range(800)
    ]
    protocol = {
        "annotation": {
            "natural_shards_per_annotator": 16,
            "natural_rows_per_shard": 50,
            "hidden_controls_total_per_annotator": 16,
            "self_repeats_total_per_annotator": 80,
        }
    }
    packages, private, report = build_prior_annotation_shards(proposals, protocol)
    assert set(packages) == {"a", "b"}
    assert report["rows_per_shard"] == 56
    assert len(private) == 2 * (800 + 16 + 80)
    for annotator in ("a", "b"):
        assert len(packages[annotator]) == 16
        assert all(len(rows) == 56 for rows in packages[annotator])
        assert all(
            "expected_signature" not in row
            for rows in packages[annotator]
            for row in rows
        )
        assert all(
            set(row) == {"schema_version", "item_id", "question", "response", "units"}
            for rows in packages[annotator]
            for row in rows
        )
        shard_by_item = {
            row["item_id"]: shard_index
            for shard_index, rows in enumerate(packages[annotator])
            for row in rows
        }
        private_by_item = {
            row["item_id"]: row for row in private if row["annotator"] == annotator
        }
        assert all(
            shard_by_item[item_id] != shard_by_item[private_row["natural_item_id"]]
            for item_id, private_row in private_by_item.items()
            if private_row["kind"] == "repeat"
        )
    control_private = [row for row in private if row["kind"] == "control"]
    assert len(control_private) == 32
    assert all("expected_signature" in row for row in control_private)


def _evaluation_fixture() -> dict:
    strata = [
        ("gsm8k", "numeric_match", "train"),
        ("gsm8k", "numeric_match", "dev"),
        ("gsm8k", "numeric_mismatch", "train"),
        ("gsm8k", "numeric_mismatch", "dev"),
        ("math", "numeric_match", "train"),
        ("math", "numeric_match", "dev"),
        ("math", "numeric_mismatch", "train"),
        ("math", "numeric_mismatch", "dev"),
    ]
    units = [
        {"unit_index": 0, "kind": "material_claim", "text": "first"},
        {"unit_index": 1, "kind": "material_claim", "text": "answer"},
        {"unit_index": 2, "kind": "material_claim", "text": "wrapper"},
    ]
    proposals = []
    packages = {"a": [], "b": []}
    private = []
    labels = {"a": [], "b": []}
    for index, (source, checker_status, split) in enumerate(strata):
        item_id = f"natural-{index}"
        proposal = {
            "schema_version": "clir-prior-v12-natural-proposal",
            "proposal_id": item_id,
            "trajectory_id": f"trajectory-{index}",
            "query_id": f"query-{index}",
            "cluster_id": f"cluster-{index}",
            "source": source,
            "source_record_id": index,
            "checker_status": checker_status,
            "prior_label_split": split,
            "candidate_index": 0,
            "question": f"question {index}",
            "response": f"response {index}",
            "material_claim_count": 3,
            "output_token_count": 12,
            "units": units,
            "selection_priority": f"{index:02d}",
        }
        proposals.append(proposal)
        for annotator in ("a", "b"):
            packages[annotator].append(
                {
                    "schema_version": PACKAGE_SCHEMA,
                    "item_id": item_id,
                    "question": proposal["question"],
                    "response": proposal["response"],
                    "units": units,
                }
            )
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "item_id": item_id,
                    "kind": "natural",
                    "natural_item_id": item_id,
                    "annotation_shard_id": f"{annotator}-00",
                }
            )
            labels[annotator].append(
                {
                    "item_id": item_id,
                    "eligibility": "usable",
                    "key_unit_indices": [1],
                    "complete_unit_indices": [0, 1],
                    "confidence": "high",
                    "rationale": "checked",
                }
            )

    for annotator in ("a", "b"):
        for control_index in range(2):
            item_id = f"control-{annotator}-{control_index}"
            packages[annotator].append(
                {
                    "schema_version": PACKAGE_SCHEMA,
                    "item_id": item_id,
                    "question": "control question",
                    "response": "control response",
                    "units": units,
                }
            )
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "item_id": item_id,
                    "kind": "control",
                    "expected_signature": ["usable", [1], [0, 1]],
                    "annotation_shard_id": f"{annotator}-00",
                }
            )
            labels[annotator].append(
                {
                    "item_id": item_id,
                    "eligibility": "usable",
                    "key_unit_indices": [1],
                    "complete_unit_indices": [0, 1],
                    "confidence": "high",
                    "rationale": "checked control",
                }
            )
        for repeat_index in range(2):
            parent_id = f"natural-{repeat_index}"
            item_id = f"repeat-{annotator}-{repeat_index}"
            parent = next(
                row for row in packages[annotator] if row["item_id"] == parent_id
            )
            packages[annotator].append({**parent, "item_id": item_id})
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "item_id": item_id,
                    "kind": "repeat",
                    "natural_item_id": parent_id,
                    "annotation_shard_id": f"{annotator}-01",
                }
            )
            labels[annotator].append(
                {
                    "item_id": item_id,
                    "eligibility": "usable",
                    "key_unit_indices": [1],
                    "complete_unit_indices": [0, 1],
                    "confidence": "medium",
                    "rationale": "checked repeat",
                }
            )

    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "proposal_pool": {
            "natural_count": 8,
            "strata": [
                {
                    "source": source,
                    "checker_status": checker_status,
                    "split": split,
                    "count": 1,
                }
                for source, checker_status, split in strata
            ],
        },
        "annotation": {
            "hidden_controls_total_per_annotator": 2,
            "self_repeats_total_per_annotator": 2,
        },
        "strict_consensus": {
            "label_name": "silver-test",
            "final_target_rows": 8,
            "final_strata": [
                {
                    "source": source,
                    "checker_status": checker_status,
                    "split": split,
                    "count": 1,
                }
                for source, checker_status, split in strata
            ],
        },
        "gates": {
            "controls_min_per_annotator": "2/2",
            "self_repeat_min": 0.95,
            "selected_complete_positive_iou_mean_min": 0.80,
            "selected_complete_mask_coverage_mean_min": 0.90,
            "selected_complete_all_material_rate_max": 0.25,
        },
    }
    return {
        "proposals": proposals,
        "package_a": packages["a"],
        "package_b": packages["b"],
        "private_index": private,
        "labels_a": labels["a"],
        "labels_b": labels["b"],
        "protocol": protocol,
    }


def test_prior_v12_annotation_requires_exact_fields_and_singleton_key() -> None:
    item = {
        "schema_version": PACKAGE_SCHEMA,
        "item_id": "item",
        "question": "question",
        "response": "response",
        "units": [
            {"unit_index": 0, "kind": "material_claim", "text": "claim"},
            {"unit_index": 1, "kind": "material_claim", "text": "answer"},
        ],
    }
    label = {
        "item_id": "item",
        "eligibility": "usable",
        "key_unit_indices": [1],
        "complete_unit_indices": [0, 1],
        "confidence": "high",
        "rationale": "checked",
    }
    assert validate_prior_v12_annotation(label, item)["schema_version"] == LABEL_SCHEMA
    with pytest.raises(ValueError, match="field mismatch"):
        validate_prior_v12_annotation({**label, "extra": True}, item)
    with pytest.raises(ValueError, match="exactly one Key"):
        validate_prior_v12_annotation({**label, "key_unit_indices": [0, 1]}, item)


def test_prior_v12_strict_consensus_gate_passes_frozen_selection() -> None:
    fixture = _evaluation_fixture()
    report = evaluate_prior_v12_labels(**fixture)
    assert report["status"] == "PASS_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE"
    assert report["failed_gates"] == []
    assert report["metrics"]["raw_population"]["strict_eligible_rows"] == 8
    selected = report["metrics"]["prospective_frozen_selection"]
    assert selected["selected_rows"] == 8
    assert selected["complete_positive_iou_mean"] == 1.0
    assert selected["complete_mask_coverage_mean"] == 1.0
    assert report["target_publication_authorized"] is False
    assert report["training_allowed"] is False


def test_prior_v12_gate_stops_when_one_prefrozen_stratum_lacks_yield() -> None:
    fixture = _evaluation_fixture()
    fixture["labels_b"][0]["confidence"] = "low"
    report = evaluate_prior_v12_labels(**fixture)
    assert report["status"] == "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE"
    assert "every_final_stratum_quota" in report["failed_gates"]
    assert report["metrics"]["prospective_frozen_selection"]["selected_rows"] == 0
    assert report["failure_is_terminal"] is True


def test_prior_v12_gate_uses_fixed_selected_iou_and_coverage_guards() -> None:
    fixture = _evaluation_fixture()
    for row in fixture["labels_b"]:
        if row["item_id"].startswith("natural-") or row["item_id"].startswith(
            "repeat-b-"
        ):
            row["complete_unit_indices"] = [1, 2]
    report = evaluate_prior_v12_labels(**fixture)
    assert report["gates"]["every_final_stratum_quota"]["pass"] is True
    assert report["gates"]["selected_complete_positive_iou_mean"]["pass"] is False
    assert report["gates"]["selected_complete_mask_coverage_mean"]["pass"] is False
    selected = report["metrics"]["prospective_frozen_selection"]
    assert selected["complete_positive_iou_mean"] == pytest.approx(1 / 3)
    assert selected["complete_mask_coverage_mean"] == pytest.approx(1 / 3)
