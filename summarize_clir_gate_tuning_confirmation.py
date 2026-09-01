#!/usr/bin/env python
"""Summarize the one-time sealed Prior/Gate confirmation population."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, evaluate, file_sha256
from score_clir_checkpoint_set import MERGE_STATUS
from src.clir_data import read_jsonl
from summarize_clir_three_module_ranking import (
    _effect_summary,
    _sample_sd,
    _validate_compact_rows,
    selection_vector,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/prior_gate_tuning_v1/confirmation_summary_authorization.json"
)
DEFAULT_MERGE = (
    PROJECT_ROOT
    / "run_artifacts/prior_gate_tuning_v1/confirmation/scoring/merged/merge_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/confirmation/summary.json"
)
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_CONFIRMATION_SUMMARY"
SCORING_AUTHORIZATION_STATUS = (
    "AUTHORIZED_PRIOR_GATE_TUNING_V1_CONFIRMATION_SCORING"
)
LOCK_STATUS = "LOCKED_PRIOR_GATE_TUNING_V1_CONFIRMATION_12_CHECKPOINTS"
CELLS = ("locked_candidate", "ch", "u0", "full_025")
SEEDS = (42, 43, 44)
K_VALUES = (1, 2, 4, 8, 16)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def confirmation_decision(primary: Mapping[str, Any]) -> dict[str, Any]:
    interval = primary["fixed_seed_query_95_ci"]
    directions = primary["seed_direction_counts"]
    mean = float(primary["mean_paired_effect"])
    benefit = mean > 0 and int(directions["positive"]) >= 2 and interval[0] > 0
    harm = mean < 0 and int(directions["negative"]) >= 2
    result = (
        "CONFIRMATION_BENEFIT"
        if benefit
        else "CONFIRMATION_HARM"
        if harm
        else "CONFIRMATION_INCONCLUSIVE"
    )
    return {
        "contrast": "locked_candidate_minus_ch_at_K16",
        "result": result,
        "benefit": benefit,
        "harm": harm,
        "rule": (
            "benefit requires positive mean, at least two positive seed effects, "
            "and fixed-seed paired-query 95% CI lower bound above zero; harm "
            "requires negative mean and at least two negative seed effects"
        ),
    }


def summarize(
    scored: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    selected_direct_weight: float,
    gate_prior_weight: float,
) -> dict[str, Any]:
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if set(scored) != expected:
        raise ValueError("confirmation scores must form the complete 4x3 grid")
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        k: {cell: [] for cell in CELLS} for k in K_VALUES
    }
    indices: dict[int, dict[tuple[str, int], np.ndarray]] = {
        k: {} for k in K_VALUES
    }
    reference_queries: list[str] | None = None
    for cell in CELLS:
        for seed in SEEDS:
            rows = scored[(cell, seed)]
            reports[(cell, seed)] = evaluate(
                rows,
                score_field="clir_score",
                correctness_field="correctness",
                k_values=K_VALUES,
                bootstrap_replicates=0,
                seed=bootstrap_seed,
            )
            for k in K_VALUES:
                query_ids, selected, selected_indices = selection_vector(rows, k)
                if reference_queries is None:
                    reference_queries = query_ids
                elif query_ids != reference_queries:
                    raise ValueError("confirmation query population/order drift")
                vectors[k][cell].append(selected)
                indices[k][(cell, seed)] = selected_indices
    assert reference_queries is not None
    if len(reference_queries) != 800:
        raise ValueError("confirmation population must contain 800 queries")

    by_k: dict[str, Any] = {}
    for k_index, k in enumerate(K_VALUES):
        arrays = {cell: np.stack(vectors[k][cell], axis=0) for cell in CELLS}
        contrasts = {
            "locked_candidate_minus_ch": arrays["locked_candidate"] - arrays["ch"],
            "locked_candidate_minus_u0": arrays["locked_candidate"] - arrays["u0"],
            "locked_candidate_minus_full_025": (
                arrays["locked_candidate"] - arrays["full_025"]
            ),
        }
        first = reports[("locked_candidate", 42)]["by_k"][str(k)]
        by_k[str(k)] = {
            "queries": len(reference_queries),
            "random_expected_accuracy": first["random_expected_accuracy"],
            "oracle_accuracy": first["oracle_accuracy"],
            "cells": {
                cell: {
                    "mean_accuracy": float(arrays[cell].mean()),
                    "sample_sd_across_seeds": _sample_sd(
                        arrays[cell].mean(axis=1).tolist()
                    ),
                    "by_seed": {
                        str(seed): float(value)
                        for seed, value in zip(
                            SEEDS, arrays[cell].mean(axis=1), strict=True
                        )
                    },
                }
                for cell in CELLS
            },
            "paired_contrasts": {
                name: _effect_summary(
                    values,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed + 100 * k_index + index,
                )
                for index, (name, values) in enumerate(contrasts.items())
            },
        }

    changes: dict[str, Any] = {}
    for comparator in ("ch", "u0", "full_025"):
        by_seed: dict[str, Any] = {}
        for seed_index, seed in enumerate(SEEDS):
            candidate_index = indices[16][("locked_candidate", seed)]
            comparator_index = indices[16][(comparator, seed)]
            candidate_label = vectors[16]["locked_candidate"][seed_index]
            comparator_label = vectors[16][comparator][seed_index]
            changed = candidate_index != comparator_index
            by_seed[str(seed)] = {
                "queries": len(reference_queries),
                "changed_candidate": int(changed.sum()),
                "changed_fraction": float(changed.mean()),
                "wrong_to_correct": int(
                    ((comparator_label == 0) & (candidate_label == 1)).sum()
                ),
                "correct_to_wrong": int(
                    ((comparator_label == 1) & (candidate_label == 0)).sum()
                ),
                "changed_same_correctness": int(
                    (changed & (candidate_label == comparator_label)).sum()
                ),
            }
        changes[f"locked_candidate_vs_{comparator}"] = by_seed

    primary = by_k["16"]["paired_contrasts"]["locked_candidate_minus_ch"]
    return {
        "schema_version": "clir-prior-gate-tuning-v1-confirmation-summary",
        "status": "COMPLETE_PRIOR_GATE_TUNING_V1_ONE_TIME_CONFIRMATION",
        "created_at_utc": _utc_now(),
        "evidence_tier": (
            "fresh_query_and_cluster_disjoint_train_source_confirmation_"
            "numeric_checker_no_human_verification"
        ),
        "confirmation_queries": len(reference_queries),
        "locked_candidate": {
            "direct_prior_weight": selected_direct_weight,
            "gate_prior_weight": gate_prior_weight,
        },
        "ranking": {"by_k": by_k},
        "selection_changes_at_k16": changes,
        "primary_decision": confirmation_decision(primary),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "fixed_seed_query": "resample paired queries then average three fixed seeds",
            "hierarchical": "exploratory resample seeds and paired queries",
        },
        "confirmation_outcomes_opened": True,
        "no_second_weight_selection_allowed": True,
        "claim_boundary": (
            "fresh confirmation for the locked train-source weight decision; "
            "not human Gold, a protected test, or external-generalization evidence"
        ),
    }


def command_summarize(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    merge_path = Path(args.merge_report).resolve()
    authorization = _load_json(authorization_path)
    if (
        authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("summarizer_sha256") != file_sha256(__file__)
    ):
        raise ValueError("confirmation summary is unauthorized or hash-stale")
    scoring_path = _project_path(authorization["scoring_authorization_path"])
    if file_sha256(scoring_path) != authorization["scoring_authorization_sha256"]:
        raise ValueError("confirmation scoring authorization hash drift")
    scoring = _load_json(scoring_path)
    if (
        scoring.get("status") != SCORING_AUTHORIZATION_STATUS
        or scoring.get("confirmation_scoring_allowed") is not True
        or scoring.get("cells") != list(CELLS)
        or scoring.get("seeds") != list(SEEDS)
        or scoring.get("k") != list(K_VALUES)
    ):
        raise ValueError("confirmation scoring contract drift")
    lock_path = _project_path(authorization["weight_lock_path"])
    if file_sha256(lock_path) != authorization["weight_lock_sha256"]:
        raise ValueError("confirmation weight-lock hash drift")
    lock = _load_json(lock_path)
    if lock.get("status") != LOCK_STATUS:
        raise ValueError("confirmation weight lock is inactive")
    if (
        merge_path != _project_path(authorization["merge_report_path"])
        or file_sha256(merge_path) != authorization["merge_report_sha256"]
    ):
        raise ValueError("confirmation merge report hash drift")

    input_path = _project_path(scoring["ranking_input_path"])
    source = read_jsonl(input_path)
    merge = _load_json(merge_path)
    if (
        len(source) != 12_800
        or merge.get("status") != MERGE_STATUS
        or merge.get("confirmation_scoring") is not True
        or merge.get("authorization_file_sha256") != file_sha256(scoring_path)
        or merge.get("input_jsonl_sha256") != scoring["ranking_input_sha256"]
        or merge.get("completion_report_sha256")
        != scoring["training_completion_sha256"]
        or merge.get("scorer_sha256") != scoring["scorer_sha256"]
        or len(merge.get("outputs", {})) != 12
    ):
        raise ValueError("confirmation score merge is stale or unauthorized")
    run_spec = {
        (str(run["cell"]), int(run["seed"])): run for run in lock["runs"]
    }
    scored: dict[tuple[str, int], list[dict[str, Any]]] = {}
    inputs: dict[str, Any] = {}
    for cell in CELLS:
        for seed in SEEDS:
            key = f"{cell}/seed-{seed}"
            record = merge["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"confirmation merge lacks {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"confirmation scored output drift: {path}")
            rows = read_jsonl(path)
            checkpoint = str(run_spec[(cell, seed)]["checkpoint_file_sha256"])
            _validate_compact_rows(rows, source, checkpoint)
            scored[(cell, seed)] = rows
            inputs[key] = {
                "path": str(path),
                "file_sha256": record["file_sha256"],
                "checkpoint_sha256": checkpoint,
            }
    report = summarize(
        scored,
        bootstrap_replicates=int(scoring["bootstrap_replicates"]),
        bootstrap_seed=int(scoring["bootstrap_seed"]),
        selected_direct_weight=float(
            lock["weight_selection"]["selected_direct_weight"]
        ),
        gate_prior_weight=float(lock["weight_selection"]["gate_prior_weight"]),
    )
    report.update(
        {
            "authorization_file_sha256": file_sha256(authorization_path),
            "scoring_authorization_file_sha256": file_sha256(scoring_path),
            "weight_lock_file_sha256": file_sha256(lock_path),
            "ranking_input_sha256": scoring["ranking_input_sha256"],
            "score_merge_sha256": file_sha256(merge_path),
            "runs": inputs,
        }
    )
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"confirmation summary already exists: {output}")
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "k16": report["ranking"]["by_k"]["16"],
                "primary_decision": report["primary_decision"],
                "confirmation_outcomes_opened": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--merge-report", default=str(DEFAULT_MERGE))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    command_summarize(build_parser().parse_args())
