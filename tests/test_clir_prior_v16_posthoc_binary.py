from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.clir_prior_v16_posthoc_binary import (
    LABEL_NAME,
    PACKAGE_SCHEMA,
    PROPOSAL_SCHEMA,
    build_posthoc_controls,
    build_posthoc_shards,
    construct_posthoc_silver_rows,
    evaluate_posthoc_replay,
    public_posthoc_item,
    validate_posthoc_annotation,
    validate_posthoc_silver_rows,
)


def _proposal(index: int) -> dict:
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
        "schema_version": PROPOSAL_SCHEMA,
        "item_id": f"prior-v16-posthoc-natural-{index:04d}",
        "original_v16_item_id": f"prior-v16-natural-{index:04d}",
        "source_row_id": f"trajectory-{index:04d}",
        "query_id": f"query-{index:04d}",
        "cluster_id": f"cluster-{index:04d}",
        "source": "gsm8k" if index < 450 else "math",
        "checker_status": "numeric_match",
        "prior_label_split": "train" if index < 386 else "dev",
        "candidate_index": 0,
        "source_record_id": index,
        "material_claim_count": len(texts),
        "selection_priority": f"{index:064x}",
        "question": "Start with 4, add 3, then double the result.",
        "response": "\n".join(texts),
        "parsed_answer": "14",
        "units": [
            {"unit_index": 2 * i, "kind": "material_claim", "text": text}
            for i, text in enumerate(texts)
        ],
        "mechanical_key_block_id": 4,
        "residual_block_count": 3,
        "fixed_non_main_block_count": 4,
    }


def _materialized(proposal: dict) -> dict:
    units = []
    material = proposal["units"]
    for i, raw in enumerate(material):
        units.extend(
            [
                {
                    **raw,
                    "token_start": 2 * i,
                    "token_end": 2 * i + 1,
                },
                {
                    "unit_index": 2 * i + 1,
                    "kind": "non_claim",
                    "text": "\n",
                    "token_start": 2 * i + 1,
                    "token_end": 2 * i + 2,
                },
            ]
        )
    return {
        "id": proposal["source_row_id"],
        "query_id": proposal["query_id"],
        "cluster_id": proposal["cluster_id"],
        "source": proposal["source"],
        "checker_status": proposal["checker_status"],
        "candidate_index": proposal["candidate_index"],
        "correctness": 1,
        "prompt_token_ids": [10, 11],
        "output_token_ids": list(range(100, 100 + len(units))),
        "units": units,
    }


def _natural_label(item: dict) -> dict:
    decisions = []
    for block_id in item["structure"]["residual_block_ids"]:
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
        "rationale": "the variable chain is used and the unrelated check is removable",
    }


def _fixture():
    proposals = [_proposal(index) for index in range(490)]
    packages_by_shard, private, construction = build_posthoc_shards(proposals)
    private_map = {(row["annotator"], row["item_id"]): row for row in private}
    labels = {"a": [], "b": []}
    for side in ("a", "b"):
        for shard in packages_by_shard[side]:
            for item in shard:
                hidden = private_map[(side, item["item_id"])]
                if hidden["kind"] == "control":
                    expected = hidden["expected_label"]
                    label = {
                        "item_id": expected["item_id"],
                        "residual_decisions": deepcopy(expected["residual_decisions"]),
                        "confidence": "high",
                        "rationale": "the control follows the frozen deletion rule",
                    }
                else:
                    label = _natural_label(item)
                labels[side].append(label)
    packages = {
        side: [row for shard in packages_by_shard[side] for row in shard]
        for side in ("a", "b")
    }
    return proposals, packages, private, labels, construction


def _gates() -> dict:
    return {
        "natural_rows_required": 490,
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
        "publishable_train_rows_min": 360,
        "publishable_dev_rows_min": 90,
    }


def test_posthoc_controls_follow_question_visible_rule() -> None:
    controls = {name: expected for _, expected, name in build_posthoc_controls("a")}
    early = dict(
        (row["block_id"], row["decision"])
        for row in controls["unused_early_guess"]["residual_decisions"]
    )
    alternative = dict(
        (row["block_id"], row["decision"])
        for row in controls["unused_alternative"]["residual_decisions"]
    )
    case = dict(
        (row["block_id"], row["decision"])
        for row in controls["used_case_choice"]["residual_decisions"]
    )
    assert early == {0: "not_used", 1: "not_used", 2: "not_used"}
    assert alternative == {0: "not_used", 1: "not_used", 2: "not_used"}
    assert case == {0: "not_used", 1: "used", 2: "used"}


def test_posthoc_schema_is_strict() -> None:
    item = public_posthoc_item(_proposal(0))
    assert item["schema_version"] == PACKAGE_SCHEMA
    normalized = validate_posthoc_annotation(_natural_label(item), item)
    assert normalized["complete_block_indices"] == [1, 2, 4]
    broken = _natural_label(item)
    broken["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        validate_posthoc_annotation(broken, item)


def test_posthoc_perfect_replay_materializes_exact_token_rows() -> None:
    proposals, packages, private, labels, construction = _fixture()
    assert construction["rows_per_shard"] == [56, 56, 55, 55, 55, 55, 55, 55, 55, 55]
    report = evaluate_posthoc_replay(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        gates=_gates(),
    )
    assert report["status"] == "PASS_PRIOR_V16_POSTHOC_BINARY_REPLAY"
    assert report["publishable_population"]["train_rows"] == 386
    assert report["publishable_population"]["dev_rows"] == 104
    rows, materialization = construct_posthoc_silver_rows(
        proposals=proposals,
        materialized_rows=[_materialized(row) for row in proposals],
        packages=packages,
        labels=labels,
        evaluation_report=report,
    )
    validation = validate_posthoc_silver_rows(
        rows,
        expected_item_ids_sha256=report["publishable_population"][
            "ordered_item_ids_sha256"
        ],
    )
    assert materialization["selected_rows"] == validation["rows"] == 490
    assert rows[0]["prior_label_name"] == LABEL_NAME
    assert len(rows[0]["complete_prior_target"]) == len(rows[0]["output_token_ids"])
    assert rows[0]["key_prior_mask"] == [1] * len(rows[0]["output_token_ids"])

    repeat = next(
        row for row in private if row["annotator"] == "a" and row["kind"] == "repeat"
    )
    repeated_label = next(
        row for row in labels["a"] if row["item_id"] == repeat["item_id"]
    )
    repeated_label["confidence"] = "low"
    with_low_repeat = evaluate_posthoc_replay(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        gates=_gates(),
    )
    assert (
        repeat["natural_item_id"]
        in with_low_repeat["publishable_population"]["repeat_failed_parent_ids"]
    )
    assert with_low_repeat["publishable_population"]["eligible_rows"] == 489


def test_posthoc_protocol_freezes_population_publication_and_claim_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (
            root / "configs/data_expansion_prior_v16/posthoc_binary_v1/protocol.json"
        ).read_text()
    )
    assert protocol["status"] == "FROZEN_POSTHOC_BEFORE_ANY_REPLAY_LABEL"
    assert protocol["replay_population"]["mechanically_compilable_rows"] == 490
    assert protocol["publication"][
        "do_not_select_rows_by_cross_annotator_agreement_iou_source_correctness_or_difficulty"
    ]
    assert protocol["claim_boundary"][
        "replay_is_posthoc_and_population_was_used_for_target_development"
    ]
    assert protocol["claim_boundary"]["labels_are_silver_not_gold"]
    prompt = (
        root / "configs/data_expansion_prior_v16/posthoc_binary_v1/annotation_prompt.md"
    ).read_text()
    assert "题目 `question` 在阅读和核验时始终可见" in prompt
    assert "换算常数" in prompt
