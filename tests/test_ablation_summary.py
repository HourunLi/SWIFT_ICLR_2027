import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_clir import evaluate, file_sha256
from evaluate_clir_mechanisms import evaluate_mechanisms
from src.clir_data import write_jsonl
from summarize_clir_ablation import paired_bootstrap_ci, summarize


def write_run(
    root: Path,
    seed: int,
    cell: str,
    scores: list[float],
    labels: list[float] | None = None,
) -> None:
    labels = labels or [0.0, 1.0, 1.0, 0.0]
    rows = []
    for row_index, (label, score) in enumerate(zip(labels, scores)):
        query_index = row_index // 2
        candidate = row_index % 2
        rows.append(
            {
                "id": f"q{query_index}-c{candidate}",
                "query_id": f"q{query_index}",
                "candidate_index": candidate,
                "correctness": label,
                "clir_score": score,
                "clir_checkpoint_sha256": f"checkpoint-{cell}-{seed}",
            }
        )
    run_dir = root / f"seed_{seed}" / cell
    run_dir.mkdir(parents=True)
    scored = run_dir / "validation_scored.jsonl"
    write_jsonl(scored, rows)
    metrics = evaluate(
        rows,
        score_field="clir_score",
        correctness_field="correctness",
        k_values=(1, 2),
        bootstrap_replicates=0,
        seed=seed,
    )
    metrics["input_jsonl_sha256"] = file_sha256(scored)
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )


def test_paired_multi_seed_summary(tmp_path: Path):
    # Left selects candidate 0 for both queries. Right improves q0 for both
    # seeds, while seed 12 also regresses q1.
    for seed in (11, 12):
        write_run(tmp_path, seed, "left", [2.0, 1.0, 2.0, 1.0])
    write_run(tmp_path, 11, "right", [1.0, 2.0, 2.0, 1.0])
    write_run(tmp_path, 12, "right", [1.0, 2.0, 1.0, 2.0])

    report = summarize(
        tmp_path,
        cells=("left", "right"),
        seeds=(11, 12),
        contrasts=(("effect", "left", "right"),),
        k_values=(1, 2),
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    effect = report["contrasts"]["effect"]["by_k"]["2"]
    assert effect["mean_delta"] == pytest.approx(0.25)
    assert effect["per_seed_delta"] == {"11": 0.5, "12": 0.0}
    assert effect["seed_direction_counts"] == {
        "positive": 1,
        "zero": 1,
        "negative": 0,
    }
    assert len(effect["fixed_seed_query_95_ci"]) == 2
    assert len(effect["hierarchical_seed_query_95_ci"]) == 2
    assert report["cell_summary"]["right"]["by_k"]["2"]["mean"] == 0.75


def test_summary_rejects_candidate_population_mismatch(tmp_path: Path):
    write_run(tmp_path, 11, "left", [2.0, 1.0, 2.0, 1.0])
    write_run(
        tmp_path,
        11,
        "right",
        [1.0, 2.0, 2.0, 1.0],
        labels=[1.0, 1.0, 1.0, 0.0],
    )

    with pytest.raises(ValueError, match="Candidate population mismatch"):
        summarize(
            tmp_path,
            cells=("left", "right"),
            seeds=(11,),
            contrasts=(("effect", "left", "right"),),
            k_values=(1, 2),
            bootstrap_replicates=0,
        )


def test_paired_bootstrap_can_be_disabled():
    result = paired_bootstrap_ci(
        deltas=np.array([[1.0, -1.0]]),
        replicates=0,
        seed=1,
    )
    assert result == {
        "fixed_seed_query_95_ci": [],
        "hierarchical_seed_query_95_ci": [],
    }


def test_mechanism_diagnostics_separate_localization_and_value_shift():
    common = {
        "clir_checkpoint_sha256": "checkpoint",
        "token_hallucination_mask": [1, 1],
        "key_prior_target": [1, 0],
        "complete_prior_target": [0, 1],
        "clir_key_prior_membership": [0.9, 0.1],
        "clir_complete_prior_membership": [0.1, 0.9],
        "clir_gate_attention": [0.5, 0.5],
        "clir_key_prior": [0.8, 0.2],
        "clir_complete_prior": [0.2, 0.8],
        "clir_prior_gate_squared_l2": 0.0,
        "clir_prior_gate_alignment": 0.5,
        "clir_mean_gate": 0.6,
    }
    rows = [
        {
            **common,
            "token_hallucination_target": [0, 0],
            "clir_hallucination_prob": [0.1, 0.2],
            "path_hallucinated": 0,
            "clir_path_no_hallucination_log_prob": -0.1,
            "clir_path_hallucination_prob": 0.1,
            "hallucination_onset": -1,
            "clir_pseudo_onset": -1,
            "clir_token_value": [1.0, 1.0],
        },
        {
            **common,
            "token_hallucination_target": [0, 1],
            "clir_hallucination_prob": [0.2, 0.9],
            "path_hallucinated": 1,
            "clir_path_no_hallucination_log_prob": -3.0,
            "clir_path_hallucination_prob": 0.9,
            "hallucination_onset": 1,
            "clir_pseudo_onset": 1,
            "clir_token_value": [1.0, -1.0],
        },
    ]

    report = evaluate_mechanisms(rows, onset_window=0)

    assert report["hallucination"]["token"]["average_precision"] == 1.0
    assert report["hallucination"]["path"]["auroc"] == 1.0
    assert report["hallucination"]["onset_threshold_0_5"][
        "positive_within_window_rate"
    ] == 1.0
    assert report["hallucination"]["token_value"]["post_minus_pre"] == -2.0
    assert report["dual_prior"]["key"]["auroc"] == 1.0
    assert report["dual_prior"]["complete"]["auroc"] == 1.0
    gate = report["dual_prior"]["gate_alignment"]
    assert gate["full_trajectory_squared_l2_mean"] == 0.0
    assert gate["dot_product_mean"] == 0.5
    assert gate["raw_sigmoid_gate_mean"] == 0.6
    assert gate["attention_normalized_entropy_mean"] == pytest.approx(1.0)
    assert gate["attention_effective_tokens_mean"] == 2.0
    assert gate["attention_effective_fraction_mean"] == 1.0
