from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.clir_prior_binary_v17 import (
    PACKAGE_SCHEMA,
    build_blind_shards_v17,
    build_hidden_controls_v17,
    compile_binary_structure_v17,
    evaluate_binary_smoke_v17,
    public_package_item_v17,
    select_fresh_rows_v17,
    validate_binary_annotation_v17,
)


def _raw(index: int, *, source: str = "gsm8k") -> dict:
    texts = [
        "To calculate the answer, follow the quantities below.",
        "Let x = 4 + 3.",
        "x = 7.",
        "An unrelated check gives 9 * 9 = 81.",
        "2 * x = 14.",
        "The answer is \\boxed{14}.",
        "The result is 14.",
        "Final response complete.",
    ]
    return {
        "id": f"trajectory-{index:04d}",
        "item_id": f"prior-v17-natural-{index:04d}",
        "query_id": f"query-{index:04d}",
        "cluster_id": f"cluster-{index:04d}",
        "source": source,
        "checker_status": "numeric_match",
        "acquisition_split": "train",
        "prior_label_split": "train",
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "status": "ok",
        "finish_reason": "stop",
        "candidate_index": 0,
        "source_record_id": index,
        "material_claim_count": len(texts),
        "question": "Start with 4, add 3, then double the result.",
        "response": "\n".join(texts),
        "parsed_answer": "14",
        "units": [
            {"unit_index": 2 * i, "kind": "material_claim", "text": text}
            for i, text in enumerate(texts)
        ],
    }


def _raw_label(item: dict) -> dict:
    residual = item["structure"]["residual_block_ids"]
    decisions = []
    for block_id in residual:
        text = item["structure"]["blocks"][block_id]["text"]
        decisions.append(
            {
                "block_id": block_id,
                "decision": "not_used" if "unrelated" in text else "used",
            }
        )
    return {
        "item_id": item["item_id"],
        "residual_decisions": decisions,
        "confidence": "high",
        "rationale": "the variable chain is used and the unrelated check is not used",
    }


def _fixture():
    proposals = []
    for index in range(96):
        raw = _raw(index)
        raw["schema_version"] = "clir-prior-mechanical-key-binary-proposal-v17"
        raw["source_row_id"] = raw.pop("id")
        raw["length_band"] = "medium"
        raw["selection_priority"] = f"{index:064x}"
        structure = compile_binary_structure_v17(raw)
        raw["mechanical_key_block_id"] = structure["key_block_id"]
        raw["residual_block_count"] = len(structure["residual_block_ids"])
        raw["fixed_non_main_block_count"] = len(
            structure["fixed_non_main_block_ids"]
        )
        proposals.append(raw)
    packages, private, construction = build_blind_shards_v17(proposals)
    private_map = {(row["annotator"], row["item_id"]): row for row in private}
    labels = {"a": [], "b": []}
    for annotator in ("a", "b"):
        for shard in packages[annotator]:
            for item in shard:
                hidden = private_map[(annotator, item["item_id"])]
                if hidden["kind"] == "control":
                    expected = hidden["expected_label"]
                    labels[annotator].append(
                        {
                            "item_id": expected["item_id"],
                            "residual_decisions": deepcopy(expected["residual_decisions"]),
                            "confidence": "high",
                            "rationale": "control follows the deletion test",
                        }
                    )
                else:
                    labels[annotator].append(_raw_label(item))
    flattened = {
        side: [row for shard in packages[side] for row in shard] for side in ("a", "b")
    }
    return proposals, flattened, private, labels, construction


def _gates() -> dict:
    return {
        "natural_rows_required": 96,
        "controls_min_pass": 11,
        "self_repeat_exact_min": 0.95,
        "low_confidence_rate_max": 0.10,
        "residual_agreement_min": 0.90,
        "residual_kappa_min": 0.65,
        "row_exact_min": 0.45,
        "complete_unit_iou_min": 0.85,
        "complete_unit_mask_coverage_min": 0.95,
        "both_used_rate_min": 0.05,
        "both_not_used_rate_min": 0.05,
        "all_material_union_rate_max": 0.25,
    }


def test_v17_compiler_fixes_key_plan_duplicate_and_suffix() -> None:
    item = public_package_item_v17(_raw(0))
    assert item["schema_version"] == PACKAGE_SCHEMA
    structure = item["structure"]
    assert structure["key_block_id"] == 4
    assert structure["blocks"][0]["mechanical_status"] == "fixed_non_main"
    assert structure["blocks"][4]["mechanical_status"] == "fixed_key"
    assert structure["fixed_non_main_block_ids"][-3:] == [5, 6, 7]
    assert structure["residual_block_ids"] == [1, 2, 3]


def test_v17_schema_only_accepts_exact_ordered_residual_binary_decisions() -> None:
    item = public_package_item_v17(_raw(1))
    valid = _raw_label(item)
    normalized = validate_binary_annotation_v17(valid, item)
    assert normalized["complete_block_indices"] == [1, 2, 4]
    broken = deepcopy(valid)
    broken["residual_decisions"] = broken["residual_decisions"][:-1]
    with pytest.raises(ValueError, match="every residual block"):
        validate_binary_annotation_v17(broken, item)


def test_v17_controls_shards_and_perfect_gate() -> None:
    assert len(build_hidden_controls_v17("a")) == 12
    proposals, packages, private, labels, construction = _fixture()
    assert construction["rows_per_shard"] == 22
    assert construction["repeats_per_annotator"] == 24
    report = evaluate_binary_smoke_v17(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        gates=_gates(),
    )
    assert report["status"] == "PASS_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE"
    assert report["cross_annotator_natural"]["residual_decision_agreement"] == 1.0
    assert report["cross_annotator_natural"]["both_not_used_rate"] > 0
    assert report["trainable_labels_published"] is False


def test_v17_selects_fresh_compilable_query_cluster_distinct_rows() -> None:
    rows = [_raw(index) for index in range(5)]
    selected, report = select_fresh_rows_v17(
        rows,
        excluded_query_ids={"query-0000"},
        excluded_cluster_ids={"cluster-0001"},
        strata=[
            {
                "source": "gsm8k",
                "checker_status": "numeric_match",
                "length_band": "medium",
                "count": 3,
            }
        ],
    )
    assert len(selected) == 3
    assert report["distinct_queries"] == report["distinct_clusters"] == 3
    assert not {"query-0000", "query-0001"} & {row["query_id"] for row in selected}


def test_v17_protocol_freezes_nontrainable_binary_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "configs/data_expansion_prior_v17/protocol.json").read_text()
    )
    assert protocol["status"] == "FROZEN_BEFORE_ANY_V17_LABEL"
    assert protocol["proposal_pool"]["natural_count"] == 96
    assert protocol["claim_boundary"]["v17_rows_are_prompt_development_smoke_and_never_trainable"]
    assert protocol["fallback"]["only_if_v17_terminally_fails"].endswith(
        "v16_posthoc_replay"
    )
    prompt = (root / "configs/data_expansion_prior_v17/annotation_prompt.md").read_text()
    assert "反向删除检查" in prompt
    assert "不要输出 Key、Complete" in prompt
