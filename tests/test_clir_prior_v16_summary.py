import json
from pathlib import Path

import pytest

from summarize_clir_prior_v16_training import (
    FULL_SCORE_FIELDS,
    STAGE_1_METRICS,
    TOKEN_SCORE_FIELDS,
    _validate_prior_dev_projection,
    _validate_scored_rows,
    aggregate_contrast,
    evaluate_stage_1_gate,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/data_expansion_prior_v16/posthoc_training_v1/protocol.json"
)


def _prior_payload(
    *, key_ap: float, key_auc: float, key_bce: float,
    complete_ap: float, complete_auc: float, complete_bce: float,
    correctness_auc: float,
) -> dict:
    return {
        "prior": {
            "key": {
                "average_precision": key_ap,
                "auroc": key_auc,
                "binary_cross_entropy": key_bce,
            },
            "complete": {
                "average_precision": complete_ap,
                "auroc": complete_auc,
                "binary_cross_entropy": complete_bce,
            },
            "correctness": {
                "average_precision": 0.8,
                "auroc": correctness_auc,
                "binary_cross_entropy": 0.5,
            },
        }
    }


def test_protocol_keeps_ranking_deferred_and_original_stops() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert protocol["cells"]["full"]["factors"] == [1, 1, 1]
    assert protocol["evaluation"]["ranking"].startswith("deferred_until_fresh")
    assert protocol["evidence_boundary"]["original_terminal_statuses_are_unchanged"]


def test_stage_1_gate_passes_only_when_every_frozen_check_passes() -> None:
    seeds = [42, 43, 44]
    runs = {}
    for seed in seeds:
        runs[("r0", seed)] = _prior_payload(
            key_ap=0.10,
            key_auc=0.50,
            key_bce=0.70,
            complete_ap=0.30,
            complete_auc=0.50,
            complete_bce=0.75,
            correctness_auc=0.80,
        )
        runs[("p0", seed)] = _prior_payload(
            key_ap=0.80,
            key_auc=0.95,
            key_bce=0.10,
            complete_ap=0.90,
            complete_auc=0.96,
            complete_bce=0.20,
            correctness_auc=0.79,
        )
    stage = aggregate_contrast(
        runs,
        control="r0",
        treatment="p0",
        seeds=seeds,
        metrics=STAGE_1_METRICS,
    )
    gate = load_protocol(PROTOCOL)["stage_1_gate"]
    result = evaluate_stage_1_gate(stage, gate)
    assert result["status"] == "PASS_PRIOR_V16_POSTHOC_STAGE_1_GATE"
    assert result["all_checks_pass"]

    stage["metrics"]["correctness_auroc"]["p0_minus_r0"][
        "mean_paired_delta"
    ] = -0.051
    failed = evaluate_stage_1_gate(stage, gate)
    assert failed["status"] == "FAIL_PRIOR_V16_POSTHOC_STAGE_1_GATE"
    assert not failed["checks"]["correctness_auroc_delta"]


def test_full_score_validator_rejects_frozen_input_drift(tmp_path: Path) -> None:
    reference = {
        "id": "row-1",
        "query_id": "query-1",
        "output_token_ids": [1, 2],
        "correctness": 1,
    }
    scored = dict(reference)
    for field in FULL_SCORE_FIELDS:
        if field in TOKEN_SCORE_FIELDS:
            scored[field] = [0.25, 0.75]
        else:
            scored[field] = 0.25
    scored.update(
        {
            "clir_checkpoint_sha256": "checkpoint",
            "clir_scoring_mode": "full",
            "clir_pseudo_onset": -1,
            "clir_selected_best_of_n": True,
        }
    )
    reference_path = tmp_path / "reference.jsonl"
    scored_path = tmp_path / "scored.jsonl"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    scored_path.write_text(json.dumps(scored) + "\n", encoding="utf-8")
    rows, identity = _validate_scored_rows(
        reference_path, scored_path, "checkpoint"
    )
    assert rows[0]["id"] == "row-1"
    assert identity["rows"] == 1

    scored["correctness"] = 0
    scored_path.write_text(json.dumps(scored) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen score input drift"):
        _validate_scored_rows(reference_path, scored_path, "checkpoint")


def test_prior_dev_projection_allows_only_one_provenance_field(
    tmp_path: Path,
) -> None:
    base = {
        "id": "row-1",
        "query_id": "query-1",
        "correctness": 1,
        "schema_version": "clir-prior-v16-posthoc-training-row-v1",
        "experiment_population": "prior_v16_posthoc_binary_v1",
    }
    projected = {
        **base,
        "schema_version": "clir-three-module-v16-posthoc-row-v1",
        "experiment_population": "three_module_v16_posthoc_v1",
        "source_experiment_population": "prior_v16_posthoc_binary_v1",
    }
    base_path = tmp_path / "base.jsonl"
    projected_path = tmp_path / "projected.jsonl"
    base_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    projected_path.write_text(json.dumps(projected) + "\n", encoding="utf-8")
    _validate_prior_dev_projection(base_path, projected_path)

    projected["correctness"] = 0
    projected_path.write_text(json.dumps(projected) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="projection value drift"):
        _validate_prior_dev_projection(base_path, projected_path)
