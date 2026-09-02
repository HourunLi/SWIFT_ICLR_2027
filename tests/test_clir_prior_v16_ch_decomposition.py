from pathlib import Path

import torch

import score_clir_prior_v16_ch_decomposition as scoring
from summarize_clir_prior_v16_ch_decomposition import _evaluate_unbalanced_h_dev


def test_checkpoint_audit_expands_two_factor_protocol_for_shared_scorer(
    monkeypatch,
) -> None:
    authorization = {
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
            "amp_dtype": "bfloat16",
        },
        "training_input": {
            "file_sha256": "train-sha",
            "rows": 10,
            "queries": 5,
        },
        "cells": {
            "c": {
                "factors": [1, 0],
                "config": "config.json",
                "config_sha256": "config-sha",
                "checkpoint_pattern": "checkpoint-{seed}.pt",
            }
        },
    }
    model = {
        "consistency_weight": 1.0,
        "hallucination_weight": 0.0,
        "prior_weight": 0.0,
        "gate_prior_weight": 0.0,
        "token_reward_weight": 0.0,
        "tail_weight": 0.0,
        "mil_weight": 0.0,
        "pseudo_tail_weight": 0.0,
        "prior_distill_weight": 0.0,
        "progress_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    checkpoint = {
        "completed_epoch": 3,
        "metrics": [{"loss": 1.0}, {"loss": 0.8}, {"loss": 0.7}],
        "training_contract": {
            "seed": 42,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
            "amp_dtype": "bfloat16",
        },
        "data_state": {
            "train_sha256": "train-sha",
            "train_rows": 10,
            "train_queries": 5,
        },
        "model_config": model,
        "run_provenance": {
            "config": {"sha256": "config-sha"},
            "code": {"branch": "clir-clean-integration", "dirty": False},
        },
        "state_dict": {"weight": torch.ones(1)},
    }
    monkeypatch.setattr(
        scoring,
        "file_sha256",
        lambda path: "config-sha" if Path(path).name == "config.json" else "checkpoint-sha",
    )
    monkeypatch.setattr(scoring.torch, "load", lambda *args, **kwargs: checkpoint)

    audited = scoring._audit_checkpoint(authorization, "c", 42)
    assert audited["factors"] == [1.0, 0.0, 0.0]


def test_unbalanced_h_dev_evaluator_keeps_every_row() -> None:
    rows = [
        {
            "query_id": "positive",
            "source": "gsm8k",
            "output_token_ids": [1, 2, 3],
            "hallucination_onset": 1,
            "path_hallucinated": 1,
            "clir_hallucination_prob": [0.1, 0.7, 0.8],
            "clir_path_hallucination_prob": 0.9,
            "clir_pseudo_onset": 1,
            "clir_checkpoint_sha256": "checkpoint-sha",
        },
        {
            "query_id": "clean-1",
            "source": "gsm8k",
            "output_token_ids": [1, 2, 3],
            "hallucination_onset": -1,
            "path_hallucinated": 0,
            "clir_hallucination_prob": [0.1, 0.2, 0.3],
            "clir_path_hallucination_prob": 0.1,
            "clir_pseudo_onset": -1,
            "clir_checkpoint_sha256": "checkpoint-sha",
        },
        {
            "query_id": "clean-2",
            "source": "math",
            "output_token_ids": [1, 2, 3],
            "hallucination_onset": -1,
            "path_hallucinated": 0,
            "clir_hallucination_prob": [0.2, 0.1, 0.4],
            "clir_path_hallucination_prob": 0.2,
            "clir_pseudo_onset": -1,
            "clir_checkpoint_sha256": "checkpoint-sha",
        },
    ]

    report = _evaluate_unbalanced_h_dev(
        rows, onset_threshold=0.5, onset_window_tokens=5
    )
    assert report["rows"] == 3
    assert report["class_counts"] == {0: 2, 1: 1}
    assert report["checkpoint_sha256"] == "checkpoint-sha"
