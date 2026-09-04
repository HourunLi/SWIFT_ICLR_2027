"""Regression tests for the protected MATH-hard fair-baseline addendum."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from score_clir_math_hard_baselines import (
    ADDENDUM_SCHEMA,
    ADDENDUM_STATUS,
    EXPECTED_CELLS,
    load_addendum,
)
from summarize_clir_math_hard_v2 import (
    EXPECTED_FACTORIAL_TERMS,
    _decision,
    _summarize_family,
)


def _valid_addendum() -> dict:
    runs = []
    for cell in EXPECTED_CELLS:
        for seed in (42, 43, 44):
            runs.append(
                {
                    "cell": cell,
                    "source_cell": "swift_official" if cell == "swift_random" else cell,
                    "seed": seed,
                    "epoch": 3,
                    "model_kind": "u0_clir" if cell == "u0_random" else "plain_swift",
                    "checkpoint_path": f"ignored/{cell}/{seed}.pt",
                    "checkpoint_file_sha256": "0" * 64,
                }
            )
    return {
        "schema_version": ADDENDUM_SCHEMA,
        "status": ADDENDUM_STATUS,
        "evidence_boundary": {
            "test_questions_already_accessed_for_deterministic_selection": True,
            "no_rollout_correctness_or_reward_scores_opened_before_this_addendum": True,
            "model_set_locked_before_first_rollout": True,
            "no_post_result_checkpoint_epoch_weight_subset_or_seed_selection": True,
        },
        "model_set": {
            "seeds": [42, 43, 44],
            "additional_checkpoint_count": 9,
            "total_checkpoint_count": 66,
            "runs": runs,
        },
        "frozen_parent": {
            "math_hard_protocol": {"path": "ignored", "file_sha256": "0" * 64},
            "pre_rollout_registry": {"path": "ignored", "file_sha256": "0" * 64},
            "training_completions": [],
        },
    }


def test_addendum_requires_complete_three_by_three_grid(tmp_path: Path) -> None:
    payload = _valid_addendum()
    path = tmp_path / "addendum.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_addendum(path, verify_files=False)
    assert len(loaded["model_set"]["runs"]) == 9
    assert {row["source_cell"] for row in loaded["model_set"]["runs"]} >= {
        "swift_official",
        "u0_random",
        "swift_grouped",
    }

    payload["model_set"]["runs"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint grid drift"):
        load_addendum(path, verify_files=False)


def test_addendum_refuses_false_unopened_claim(tmp_path: Path) -> None:
    payload = _valid_addendum()
    payload["evidence_boundary"][
        "test_questions_already_accessed_for_deterministic_selection"
    ] = False
    path = tmp_path / "addendum.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence boundary"):
        load_addendum(path, verify_files=False)


def test_factorial_terms_cover_architecture_sampler_and_interaction() -> None:
    assert tuple(EXPECTED_FACTORIAL_TERMS) == (
        "architecture_random",
        "architecture_grouped",
        "sampler_u0",
        "sampler_swift",
        "architecture_by_sampler_interaction",
    )
    interaction = EXPECTED_FACTORIAL_TERMS["architecture_by_sampler_interaction"]
    assert sum(interaction.values()) == 0
    assert set(interaction) == {
        "u0_grouped",
        "u0_random",
        "swift_grouped",
        "swift_random",
    }


def test_joint_family_summary_keeps_seed_and_query_axes() -> None:
    loaded = {}
    for seed, shift in zip((42, 43, 44), (0.0, 0.1, -0.1)):
        loaded[("left", seed)] = {
            "selected": {1: np.asarray([0.0, 0.0, 1.0, 1.0])},
        }
        loaded[("right", seed)] = {
            "selected": {
                1: np.asarray([1.0, 0.0, 1.0, 1.0]) + shift * 0.0,
            },
        }
    report = _summarize_family(
        "toy",
        {"right_minus_left": {"right": 1, "left": -1}},
        loaded,
        {"all": np.arange(4, dtype=np.int64)},
        [42, 43, 44],
        [1],
        1,
        100,
        7,
    )
    metric = report["contrasts"]["right_minus_left"]["by_k"]["1"]
    assert metric["mean_delta"] == pytest.approx(0.25)
    assert metric["per_seed_delta"] == {"42": 0.25, "43": 0.25, "44": 0.25}
    assert "fixed_seed_query_95_ci" in metric
    assert "hierarchical_seed_query_95_ci" in metric


def test_decision_requires_both_intervals_and_multiplicity() -> None:
    metric = {
        "mean_delta": 0.1,
        "fixed_seed_query_95_ci": [0.01, 0.2],
        "hierarchical_seed_query_95_ci": [0.001, 0.25],
        "seed_direction_counts": {"positive": 3, "zero": 0, "negative": 0},
    }
    assert _decision(metric, 0.01) == "benefit_on_locked_math_hard_population"
    metric["hierarchical_seed_query_95_ci"] = [-0.01, 0.25]
    assert _decision(metric, 0.01) == "inconclusive_on_locked_math_hard_population"
