"""Regression tests for the SWIFT/U0 architecture-by-sampler repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from run_swift_u0_sampler_factorial import (
    NEW_CELLS,
    _batch_audit,
    _bound_training_rows,
    load_protocol,
)
from src.clir_data import CLIRTrajectoryDataset
from summarize_swift_u0_sampler_factorial import _decision, _linear_combination


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/swift_u0_sampler_factorial_v1/protocol.json"


def test_protocol_freezes_only_the_two_missing_cells_and_epoch_three() -> None:
    protocol = load_protocol(PROTOCOL)
    assert tuple(protocol["factorial"]["new_cells"]) == NEW_CELLS == (
        "u0_random",
        "swift_grouped",
    )
    assert set(protocol["factorial"]["cells"]) == {
        "swift_random",
        "swift_grouped",
        "u0_random",
        "u0_grouped",
    }
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert protocol["training"]["saved_epochs"] == [1, 2, 3]
    assert protocol["training"]["primary_epoch"] == 3
    assert protocol["evaluation"]["ranking_scored_epochs"] == [3]
    assert protocol["evidence_boundary"]["ranking_population_already_inspected"]
    assert protocol["evidence_boundary"]["math_hard_eval_v1_remains_sealed"]


def test_u0_random_config_changes_only_the_sampler_flag() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    grouped_path = ROOT / protocol["frozen_parents"]["u0_grouped_config"]["path"]
    random_path = ROOT / protocol["frozen_parents"]["u0_random_config"]["path"]
    grouped = json.loads(grouped_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    assert grouped["training"]["group_by_semantic_id"] is True
    assert random["training"]["group_by_semantic_id"] is False
    grouped["training"]["group_by_semantic_id"] = False
    assert grouped == random


def test_real_manifest_sampler_inventory_and_step_parity() -> None:
    protocol = load_protocol(PROTOCOL)
    candidate = ROOT / protocol["frozen_parents"]["training_manifest"]["path"]
    if not candidate.exists():  # pragma: no cover - large artifacts are optional.
        pytest.skip("ignored training artifact is absent")
    train_path, rows = _bound_training_rows(protocol)
    report = _batch_audit(
        CLIRTrajectoryDataset(train_path), rows, protocol, seed=42
    )
    assert report["semantic_groups"] == 400
    assert report["semantic_rows"] == 800
    assert report["grouped"]["batches"] == report["random"]["batches"] == 1388
    assert report["grouped"]["rows"] == report["random"]["rows"] == 5552
    assert report["grouped"]["semantic_pairs_colocated"] == 400
    assert report["random"]["semantic_pairs_colocated"] == 0


def test_factorial_interaction_uses_the_predeclared_signs() -> None:
    loaded = {}
    values = {
        "u0_grouped": [1.0, 1.0],
        "u0_random": [0.5, 0.5],
        "swift_grouped": [0.8, 0.8],
        "swift_random": [0.6, 0.6],
    }
    for cell, selected in values.items():
        loaded[(cell, 42)] = {"selected": {16: np.asarray(selected)}}
    terms = {
        "u0_grouped": 1,
        "u0_random": -1,
        "swift_grouped": -1,
        "swift_random": 1,
    }
    observed = _linear_combination(loaded, terms, seed=42, k=16)
    assert np.allclose(observed, [0.3, 0.3])


def test_diagnostic_decision_requires_direction_interval_and_adjusted_p() -> None:
    metric = {
        "mean_delta": 0.01,
        "fixed_seed_query_95_ci": [0.001, 0.02],
        "seed_direction_counts": {"positive": 3, "zero": 0, "negative": 0},
    }
    assert _decision(metric, 0.01) == "positive_stable_on_inspected_diagnostic"
    assert _decision(metric, 0.10) == "inconclusive_on_inspected_diagnostic"
    metric["mean_delta"] = -0.01
    metric["fixed_seed_query_95_ci"] = [-0.02, -0.001]
    metric["seed_direction_counts"] = {"positive": 0, "zero": 0, "negative": 3}
    assert _decision(metric, 0.01) == "negative_stable_on_inspected_diagnostic"
