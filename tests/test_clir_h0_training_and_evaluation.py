from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from evaluate_clir_h0 import evaluate_h0
from prepare_clir_h0_training import (
    _assert_objective_routing,
    _representative_indices,
)
from summarize_clir_h0_experiment import _hierarchical_bootstrap


def _h_dev_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(200):
        positive = index >= 100
        rows.append(
            {
                "id": f"h-{index}",
                "query_id": f"q-{index}",
                "source": "gsm8k" if index % 2 == 0 else "math",
                "output_token_ids": [1, 2, 3],
                "hallucination_onset": 1 if positive else -1,
                "path_hallucinated": int(positive),
                "hallucination_label_tier": (
                    "silver_posthoc_triple_consensus_h0_v7_4"
                ),
                "clir_checkpoint_sha256": "checkpoint",
                "clir_hallucination_prob": (
                    [0.1, 0.9, 0.9] if positive else [0.1, 0.1, 0.1]
                ),
                "clir_path_hallucination_prob": 0.9 if positive else 0.1,
                "clir_pseudo_onset": 1 if positive else -1,
            }
        )
    return rows


def test_h0_evaluator_handles_h_only_labels_and_exact_onsets() -> None:
    report = evaluate_h0(_h_dev_rows(), onset_threshold=0.5, onset_window_tokens=5)
    assert report["rows"] == 200
    assert report["token"]["auroc"] == 1.0
    assert report["path"]["average_precision"] == 1.0
    assert report["onset"]["positive_detection_rate"] == 1.0
    assert report["onset"]["positive_exact_start_rate"] == 1.0
    assert report["onset"]["clean_no_onset_rate"] == 1.0
    assert report["source_counts"] == {"gsm8k": 100, "math": 100}


def test_h0_evaluator_fails_when_scoring_threshold_drifted() -> None:
    rows = _h_dev_rows()
    broken = deepcopy(rows)
    broken[100]["clir_pseudo_onset"] = 2
    with pytest.raises(ValueError, match="frozen threshold"):
        evaluate_h0(broken, onset_threshold=0.5)


class _RowsOnlyDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows


def test_representative_batches_are_deterministic_and_disjoint() -> None:
    rows = [
        {
            "id": "c0-a",
            "consistency_relation_id": "r0",
            "style_id": "relative_compact",
        },
        {
            "id": "c0-b",
            "consistency_relation_id": "r0",
            "style_id": "relative_expanded",
        },
        {
            "id": "c1-a",
            "consistency_relation_id": "r1",
            "style_id": "relative_compact",
        },
        {
            "id": "c1-b",
            "consistency_relation_id": "r1",
            "style_id": "relative_expanded",
        },
        {"id": "clean-0", "feature_role": "h_train", "hallucination_onset": -1},
        {"id": "positive-0", "feature_role": "h_train", "hallucination_onset": 2},
        {"id": "clean-1", "feature_role": "h_train", "hallucination_onset": -1},
        {"id": "positive-1", "feature_role": "h_train", "hallucination_onset": 1},
    ]
    selected = _representative_indices(_RowsOnlyDataset(rows))  # type: ignore[arg-type]
    assert selected == {
        "consistency": [0, 1, 2, 3],
        "hallucination": [4, 6, 5, 7],
    }
    assert not set(selected["consistency"]) & set(selected["hallucination"])


def test_objective_routing_requires_only_the_enabled_head_gradients() -> None:
    report = {
        "batches": {
            "consistency": {
                "losses": {"final": 1.0, "consistency_total": 0.5, "total": 1.5},
                "gradient_norms": {
                    "feature_encoder": 1.0,
                    "projector": 0.25,
                    "hallucination_head": 0.0,
                    "final_score_head": 0.5,
                },
            },
            "hallucination": {
                "losses": {
                    "final": 1.0,
                    "localization_token_bce": 0.7,
                    "total": 1.7,
                },
                "gradient_norms": {
                    "feature_encoder": 1.0,
                    "projector": 0.0,
                    "hallucination_head": 0.3,
                    "final_score_head": 0.5,
                },
            },
        }
    }
    _assert_objective_routing(report, consistency_enabled=True, h0_enabled=True)
    broken = deepcopy(report)
    broken["batches"]["hallucination"]["gradient_norms"][
        "hallucination_head"
    ] = 0.0
    with pytest.raises(ValueError, match="hallucination-head"):
        _assert_objective_routing(
            broken, consistency_enabled=True, h0_enabled=True
        )


def test_hierarchical_bootstrap_preserves_paired_effect_and_interaction() -> None:
    shape = (3, 7)
    values = {
        "c0": np.zeros(shape),
        "c1": np.ones(shape),
        "h0": np.full(shape, 0.25),
        "ch0": np.full(shape, 0.75),
    }
    intervals = _hierarchical_bootstrap(values, replicates=100, seed=7)
    assert intervals["c1-c0"] == [1.0, 1.0]
    assert intervals["h0-c0"] == [0.25, 0.25]
    assert intervals["ch0-c1-h0+c0"] == [-0.5, -0.5]
