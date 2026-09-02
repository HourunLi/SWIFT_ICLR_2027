import json
from pathlib import Path

import numpy as np

from prepare_clir_prior_ablation_v2_checker import _select
from src.clir_prior_ablation import (
    CONTRAST_TERMS,
    EXPECTED_CELLS,
    contrast_vector,
    derive_config,
    select_query_rows,
    validate_protocol,
)
from summarize_clir_prior_ablation_v2 import _holm


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "configs/prior_ablation_v2/protocol.json").read_text(encoding="utf-8")
    )


def test_protocol_and_anchor_derivations_are_exact() -> None:
    protocol = _protocol()
    validate_protocol(protocol)
    assert tuple(protocol["cells"]) == EXPECTED_CELLS
    base = json.loads(
        (ROOT / "configs/three_module_expansion_v1/u0_correctness_only.json").read_text()
    )
    anchors = {
        "u0": "u0_correctness_only.json",
        "c": "c_consistency.json",
        "h": "h_h0_onset_bce.json",
        "ch": "ch_consistency_h0.json",
        "full": "full_consistency_h0_prior_gate.json",
    }
    for cell, filename in anchors.items():
        expected = json.loads(
            (ROOT / "configs/three_module_expansion_v1" / filename).read_text()
        )
        assert derive_config(protocol, base, cell) == expected


def test_generated_prior_cells_route_only_declared_losses() -> None:
    protocol = _protocol()
    base = json.loads(
        (ROOT / "configs/three_module_expansion_v1/u0_correctness_only.json").read_text()
    )
    expected = {
        "k": (1.0, 1.0, 0.0, 0.0, 0.5),
        "complete": (1.0, 0.0, 1.0, 0.0, 0.5),
        "kc": (1.0, 1.0, 1.0, 0.0, 0.5),
        "kcm": (1.0, 1.0, 1.0, 0.25, 0.5),
        "kcg": (1.0, 1.0, 1.0, 0.0, 0.5),
        "ch_kcg_key": (1.0, 1.0, 1.0, 0.0, 1.0),
        "ch_kcg_complete": (1.0, 1.0, 1.0, 0.0, 0.0),
    }
    for cell, values in expected.items():
        model = derive_config(protocol, base, cell)["model"]
        observed = (
            model["prior_weight"],
            model["key_prior_weight"],
            model["complete_prior_weight"],
            model["prior_distill_weight"],
            model["prior_fusion_alpha"],
        )
        assert observed == values
        assert model["gate_prior_weight"] == (0.25 if "g" in cell else 0.0)
        for key in (
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "progress_weight",
            "reconstruction_weight",
        ):
            assert model[key] == 0.0


def test_query_selection_keeps_one_per_cluster_before_hash() -> None:
    rows = [
        {"query_id": "q1", "cluster_id": "a"},
        {"query_id": "q2", "cluster_id": "a"},
        {"query_id": "q3", "cluster_id": "b"},
        {"query_id": "q4", "cluster_id": "c"},
    ]
    first = select_query_rows(rows, 3, namespace="test")
    second = select_query_rows(list(reversed(rows)), 3, namespace="test")
    assert [row["query_id"] for row in first] == [row["query_id"] for row in second]
    assert len({row["cluster_id"] for row in first}) == 3


def _candidate(query: str, source: str, index: int, priority: str):
    return {
        "id": f"{query}:cand:{index:03d}",
        "query_id": query,
        "candidate_index": index,
        "source": source,
        "cluster_id": query,
        "prompt_token_ids": [1],
        "output_token_ids": [2],
        "response": "2",
        "checker_status": "numeric_match" if index == 0 else "numeric_mismatch",
        "correctness": int(index == 0),
        "prior_ablation_final_priority": priority,
        "prior_ablation_selection_priority": priority,
    }


def test_checker_selection_is_priority_only_and_source_exact() -> None:
    protocol = _protocol()
    protocol["ranking_population"]["source_query_counts"] = {
        "gsm8k": 1,
        "asdiv-a": 1,
        "math": 1,
    }
    protocol["ranking_population"]["total_queries"] = 3
    protocol["ranking_population"]["selected_candidate_rows"] = 48
    checked = []
    for query, source, priority in (
        ("g2", "gsm8k", "b"),
        ("g1", "gsm8k", "a"),
        ("a1", "asdiv-a", "a"),
    ):
        checked.extend(_candidate(query, source, index, priority) for index in range(16))
    reserve = [_candidate("m1", "math", index, "a") for index in range(16)]
    selected, report = _select(checked, reserve, protocol)
    assert [selected[index]["query_id"] for index in range(0, 48, 16)] == [
        "g1",
        "a1",
        "m1",
    ]
    assert report["selection_used_clir_scores"] is False


def test_contrast_algebra_and_holm_are_deterministic() -> None:
    selections = {
        "kc": np.array([1.0, 0.0]),
        "u0": np.array([0.0, 0.0]),
    }
    assert np.array_equal(
        contrast_vector(selections, "kc_minus_u0"), np.array([1.0, 0.0])
    )
    assert list(CONTRAST_TERMS) == _protocol()["evaluation"]["primary_contrasts"] + _protocol()[
        "evaluation"
    ]["secondary_contrasts"]
    adjusted = _holm({"a": 0.01, "b": 0.03, "c": 0.02})
    assert adjusted == {"a": 0.03, "c": 0.04, "b": 0.04}
