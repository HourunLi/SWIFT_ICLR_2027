from __future__ import annotations

import json
from pathlib import Path

import pytest

from prepare_clir_gate_tuning_stage_a import (
    COMPLETION_STATUS,
    _assert_direct_preflight,
    _validate_direct_config,
)
from score_clir import file_sha256
from score_clir_checkpoint_set import _load_bound_contract


ROOT = Path(__file__).resolve().parents[1]


def test_direct_gate0_config_is_the_frozen_stage_a_diagnostic() -> None:
    config, training = _validate_direct_config(
        ROOT / "configs/prior_gate_tuning_v1/ch_direct_prior_gate0.json"
    )
    assert config.consistency_weight == 1.0
    assert config.hallucination_weight == 1.0
    assert config.prior_weight == 1.0
    assert config.key_prior_weight == 1.0
    assert config.complete_prior_weight == 1.0
    assert config.gate_prior_weight == 0.0
    assert training["epochs"] == 3


def _preflight_reports() -> dict:
    base_total = {"feature_encoder": 1.0, "final_score_head": 1.0}
    return {
        "consistency": {
            "losses": {"consistency_total": 0.5},
            "objective_gradient_norms": {"projector": 1.0},
            "total_gradient_norms": base_total,
        },
        "hallucination": {
            "losses": {"localization_token_bce": 0.5},
            "objective_gradient_norms": {"hallucination_head": 1.0},
            "total_gradient_norms": base_total,
        },
        "prior": {
            "losses": {
                "prior_key": 0.4,
                "prior_complete": 0.3,
                "prior_gate": 0.0,
                "prior_total": 0.7,
            },
            "objective_gradient_norms": {
                "feature_encoder": 1.0,
                "key_prior_head": 1.0,
                "complete_prior_head": 1.0,
                "token_reward_head": 0.0,
            },
            "total_gradient_norms": base_total,
        },
    }


def test_direct_preflight_requires_prior_gradients_but_no_gate_gradient() -> None:
    _assert_direct_preflight(_preflight_reports())
    bad = _preflight_reports()
    bad["prior"]["objective_gradient_norms"]["token_reward_head"] = 1.0
    with pytest.raises(ValueError, match="unexpectedly trained"):
        _assert_direct_preflight(bad)


def test_factorial_scorer_accepts_only_exact_stage_a_3x3_grid(
    tmp_path: Path,
) -> None:
    runs = [
        {"cell": cell, "seed": seed}
        for cell in ("ch", "direct_gate0", "full_025")
        for seed in (42, 43, 44)
    ]
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(
        json.dumps({"status": COMPLETION_STATUS, "runs": runs}), encoding="utf-8"
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "q:cand:000",
                "query_id": "q",
                "candidate_index": 0,
                "sealed_until_weight_lock": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "status": "AUTHORIZED_PRIOR_GATE_TUNING_V1_STAGE_A_SCORING",
        "confirmation_scoring_allowed": False,
        "scorer_sha256": file_sha256(ROOT / "score_clir_checkpoint_set.py"),
        "training_completion_path": str(completion_path),
        "training_completion_sha256": file_sha256(completion_path),
        "training_completion_status": COMPLETION_STATUS,
        "ranking_input_path": str(input_path),
        "ranking_input_sha256": file_sha256(input_path),
        "ranking_rows": 1,
        "ranking_queries": 1,
        "candidates_per_query": 1,
        "cells": ["ch", "direct_gate0", "full_025"],
        "seeds": [42, 43, 44],
        "run_count": 9,
        "runtime": {
            "num_shards": 1,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "amp_dtype": "bfloat16",
            "shard_output_root": str(tmp_path / "shards"),
            "merged_output_root": str(tmp_path / "merged"),
        },
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    _, _, completion, digest, _, _ = _load_bound_contract(
        authorization_path=authorization_path,
        completion_path=completion_path,
        input_path=input_path,
    )
    assert completion["runs"] == runs
    assert len(digest) == 64
    runs[-1]["seed"] = 43
    completion_path.write_text(
        json.dumps({"status": COMPLETION_STATUS, "runs": runs}), encoding="utf-8"
    )
    authorization["training_completion_sha256"] = file_sha256(completion_path)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="grid"):
        _load_bound_contract(
            authorization_path=authorization_path,
            completion_path=completion_path,
            input_path=input_path,
        )
