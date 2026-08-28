from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.clir_scale_pre_annotation import (
    PRE_ANNOTATION_AUTHORIZATION_SCHEMA,
    build_scale_annotation_packages,
    build_scale_consistency_proposals,
    build_scale_natural_items,
    evaluate_scale_annotations,
    materialize_scale_rows,
    validate_scale_package_labels,
    validate_scale_materialized_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class _CharacterTokenizer:
    is_fast = True

    def __call__(self, text: str, **_: object) -> dict:
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(value) for value in token_ids)


def _raw_candidate(index: int, response: str, finish_reason: str = "stop") -> dict:
    return {
        "id": f"math:train:algebra:99999:cand:{index:03d}",
        "query_id": "math:train:algebra:99999",
        "candidate_index": index,
        "source": "math",
        "question": "What is 2+3?",
        "reference_answer": "5",
        "response": response,
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [ord(char) for char in response],
        "finish_reason": finish_reason,
        "acquisition_split": "train_acquisition",
        "cluster_id": "cluster-99999",
    }


def test_scale_materialization_preserves_axis_and_excludes_audit_only_rows() -> None:
    raw = [
        _raw_candidate(0, "Compute 2+3=5. The answer is \\boxed{5}."),
        _raw_candidate(1, "No numeric answer is supplied."),
        _raw_candidate(2, "Compute 2+3=5 but continue forever.", "length"),
    ]
    rows, report = materialize_scale_rows(
        raw,
        _CharacterTokenizer(),
        checker_version="clir_numeric_multisource_v3",
        unitizer_version="clir_material_claim_unitizer_v2",
    )
    assert report["unitization_ok"] == 3
    assert report["checker_statuses"] == {
        "numeric_match": 1,
        "parse_failed": 1,
        "truncated": 1,
    }
    assert rows[0]["eligible_for_supervision"] is True
    assert rows[1]["eligible_for_supervision"] is False
    assert rows[2]["eligible_for_supervision"] is False
    validation = validate_scale_materialized_rows(
        rows,
        raw_rows=raw,
        candidate_count=3,
        checker_version="clir_numeric_multisource_v3",
        unitizer_version="clir_material_claim_unitizer_v2",
    )
    assert validation["raw_identity_rows_verified"] == 3
    assert validation["exact_partition_rows"] == 3


def _materialized_candidate(
    query_id: str,
    candidate_index: int,
    response: str,
    token_count: int,
    *,
    source: str,
    split: str,
) -> dict:
    return {
        "id": f"{query_id}:cand:{candidate_index:03d}",
        "query_id": query_id,
        "candidate_index": candidate_index,
        "source": source,
        "question": "Three children need juice on five days for 25 weeks.",
        "response": response,
        "output_token_ids": list(range(token_count)),
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "units": [
            {"unit_index": index, "kind": "material_claim", "text": f"u{index}"}
            for index in range(4)
        ],
        "numeric_value_match": 1,
        "normalized_candidate_answer": ["375/1"],
        "acquisition_split": split,
        "cluster_id": f"cluster:{query_id}",
        "source_subject": "algebra" if source == "math" else None,
        "source_level": 3 if source == "math" else None,
    }


def test_scale_proposals_keep_source_split_and_frozen_annotation_order() -> None:
    left = (
        "Compute the weekly supply carefully. $3*5=15$. "
        "Across the school year, $15*25=375$. Therefore 375 boxes are needed."
    )
    right = (
        "Start with all children during one week. $3*5=15$. "
        "Extending that weekly amount through the year gives $15*25=375$. "
        "The requested count is 375 boxes."
    )
    rows = []
    for query_id, source, split in (
        ("math:train:algebra:99997", "math", "train_acquisition"),
        ("gsm8k:train:99997", "gsm8k", "heldout_acquisition"),
    ):
        rows.extend(
            [
                _materialized_candidate(
                    query_id, 0, left, 100, source=source, split=split
                ),
                _materialized_candidate(
                    query_id, 1, right, 120, source=source, split=split
                ),
            ]
        )
    mechanical = {
        "version": "clir_consistency_mechanical_v1",
        "minimum_material_claim_units_per_view": 4,
        "token_length_ratio_min": 1.1,
        "token_length_ratio_max": 3.0,
        "math_trace_token_count_min_per_view": 4,
        "numeric_trace_token_count_min_per_view": 4,
        "surface_bigram_count_min_per_view": 2,
        "math_trace_similarity_min": 0.6,
        "numeric_trace_similarity_min": 0.7,
        "surface_bigram_jaccard_min": 0.0,
        "surface_bigram_jaccard_max": 1.0,
    }
    proposals, report = build_scale_consistency_proposals(
        rows, mechanical=mechanical
    )
    assert len(proposals) == 2
    assert report["admitted_by_split"] == {
        "heldout_acquisition": 1,
        "train_acquisition": 1,
    }
    assert all(row["annotation_priority"] for row in proposals)
    items = build_scale_natural_items(proposals, rows)
    assert [item["item_id"] for item in items] == [
        proposal["proposal_id"] for proposal in proposals
    ]


def test_scale_packages_have_balanced_controls_and_later_self_repeats() -> None:
    proposals = []
    natural = []
    for index in range(120):
        item_id = f"natural-{index:03d}"
        proposals.append(
            {
                "proposal_id": item_id,
                "query_id": f"math:train:algebra:{index:05d}",
                "source": "math",
                "acquisition_split": (
                    "train_acquisition" if index < 80 else "heldout_acquisition"
                ),
                "cluster_id": f"cluster-{index:03d}",
                "annotation_priority": f"{index:064d}",
            }
        )
        natural.append(
            {
                "item_id": item_id,
                "query_id": f"math:train:algebra:{index:05d}",
                "problem": "What is 2+3?",
                "audit_scope": "substantive_claim_validity_only",
                "left": {"id": "left", "trajectory": "2+3=5", "units": []},
                "right": {"id": "right", "trajectory": "Adding gives 5", "units": []},
            }
        )
    packages, private = build_scale_annotation_packages(
        natural,
        proposals,
        max_natural_per_shard=50,
        controls_per_shard=4,
        repeat_fraction_per_annotator=0.1,
    )
    assert private["annotation_shard_count"] == 3
    assert len(private["self_repeats"]["a"]) == 12
    assert len(private["self_repeats"]["b"]) == 12
    for controls in private["controls_by_shard"].values():
        decisions = [row["expected_annotation"]["decision"] for row in controls]
        assert decisions.count("accept") == decisions.count("reject") == 2
    shard_number = {f"shard-{index:03d}": index for index in range(3)}
    for slot in ("a", "b"):
        for repeat in private["self_repeats"][slot]:
            assert shard_number[repeat["repeat_shard_id"]] > shard_number[
                repeat["original_shard_id"]
            ]
        for rows in packages[slot].values():
            ids = [row["item_id"] for row in rows]
            assert len(ids) == len(set(ids))

    expected_controls = {
        control["item_id"]: control["expected_annotation"]["decision"]
        for shard_controls in private["controls_by_shard"].values()
        for control in shard_controls
    }

    def labels_for(slot: str) -> list[dict]:
        labels = []
        for package_rows in packages[slot].values():
            for item in package_rows:
                decision = expected_controls.get(item["item_id"], "accept")
                labels.append(
                    {
                        "item_id": item["item_id"],
                        "decision": decision,
                        "confidence": "high",
                        "rationale": (
                            "[ACCEPT_VALID] correct"
                            if decision == "accept"
                            else "[REJECT_ERROR] explicit error"
                        ),
                    }
                )
        return labels

    labels_a = labels_for("a")
    labels_b = labels_for("b")
    first_shard = next(iter(packages["a"]))
    first_ids = {row["item_id"] for row in packages["a"][first_shard]}
    first_labels = [row for row in labels_a if row["item_id"] in first_ids]
    validated = validate_scale_package_labels(
        packages["a"][first_shard],
        first_labels,
    )
    assert len(validated) == len(packages["a"][first_shard])
    with pytest.raises(ValueError, match="strict schema"):
        malformed = deepcopy(first_labels)
        malformed[0]["extra"] = True
        validate_scale_package_labels(packages["a"][first_shard], malformed)
    gates = {
        "natural_decision_agreement_min": 0.95,
        "review_fraction_max_per_annotator": 0.02,
        "hidden_control_accuracy_required_per_annotator": 1.0,
        "self_repeat_agreement_min_per_annotator": 0.95,
        "train_common_accept_count_min": 70,
        "heldout_common_accept_count_min": 35,
    }
    report = evaluate_scale_annotations(
        labels_a=labels_a, labels_b=labels_b, private=private, gates=gates
    )
    assert report["status"] == "PASS_SCALE_V6_RAW_ANNOTATION_GATES"
    assert report["common_accept_by_split"] == {
        "heldout_acquisition": 40,
        "train_acquisition": 80,
    }

    failed_b = deepcopy(labels_b)
    first_control = next(iter(expected_controls))
    for label in failed_b:
        if label["item_id"] == first_control:
            label["decision"] = (
                "reject" if label["decision"] == "accept" else "accept"
            )
            label["rationale"] = (
                "[REJECT_ERROR] wrong"
                if label["decision"] == "reject"
                else "[ACCEPT_VALID] wrong"
            )
    failed = evaluate_scale_annotations(
        labels_a=labels_a, labels_b=failed_b, private=private, gates=gates
    )
    assert failed["status"] == "STOP_SCALE_V6_RAW_ANNOTATION_GATE_FAILURE"
    assert "hidden_control_accuracy_b" in failed["failed_gate_names"]


def test_pre_annotation_authorization_stops_before_ai_calls() -> None:
    authorization = json.loads(
        (
            ROOT
            / "configs/data_expansion_scale_v6/pre_annotation_authorization.json"
        ).read_text(encoding="utf-8")
    )
    assert authorization["schema_version"] == PRE_ANNOTATION_AUTHORIZATION_SCHEMA
    assert authorization["status"] == "AUTHORIZED_PRE_ANNOTATION_ONLY"
    scope = authorization["authorized_scope"]
    assert scope["checker_and_unitizer_materialization"] is True
    assert scope["blind_package_construction"] is True
    assert scope["ai_annotation_or_provider_call"] is False
    assert scope["feature_extraction"] is False
    assert scope["training"] is False
