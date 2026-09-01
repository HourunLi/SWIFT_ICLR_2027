#!/usr/bin/env python
"""Select the pre-registered direct-Prior weight on the open tuning split."""

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
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/weight_grid_summary_authorization.json"
)
DEFAULT_MERGE = (
    PROJECT_ROOT
    / "run_artifacts/prior_gate_tuning_v1/weight_grid/scoring/merged/merge_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/weight_grid/selection.json"
)
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_WEIGHT_GRID_SUMMARY"
SCORING_AUTHORIZATION_STATUS = (
    "AUTHORIZED_PRIOR_GATE_TUNING_V1_WEIGHT_GRID_SCORING"
)
COMPLETION_STATUS = "PASS_PRIOR_GATE_TUNING_V1_WEIGHT_GRID_9_CHECKPOINTS"
CELLS = ("direct_025", "direct_050", "direct_100")
WEIGHTS = {"direct_025": 0.25, "direct_050": 0.5, "direct_100": 1.0}
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


def _load_authorization(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    payload = _load_json(path)
    if payload.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("direct-weight-grid summary is not authorized")
    if payload.get("summarizer_sha256") != file_sha256(__file__):
        raise ValueError("weight-grid authorization binds another summarizer")
    scoring_path = _project_path(payload["scoring_authorization_path"])
    if file_sha256(scoring_path) != payload["scoring_authorization_sha256"]:
        raise ValueError("weight-grid scoring authorization hash drift")
    scoring = _load_json(scoring_path)
    if (
        scoring.get("status") != SCORING_AUTHORIZATION_STATUS
        or scoring.get("cells") != list(CELLS)
        or scoring.get("seeds") != list(SEEDS)
        or scoring.get("k") != list(K_VALUES)
        or scoring.get("confirmation_scoring_allowed") is not False
    ):
        raise ValueError("weight-grid scoring contract drift")
    return payload, scoring, scoring_path


def choose_weight(k16_means: Mapping[str, float]) -> dict[str, Any]:
    if set(k16_means) != set(CELLS):
        raise ValueError("weight selection requires all three frozen cells")
    selected = max(CELLS, key=lambda cell: (float(k16_means[cell]), WEIGHTS[cell]))
    return {
        "selected_cell": selected,
        "selected_direct_weight": WEIGHTS[selected],
        "gate_prior_weight": 0.25,
        "metric": "mean_across_three_seeds_Best_of_N_accuracy_at_K16",
        "tie_break": "larger_direct_weight_on_exact_mean_tie",
    }


def summarize(
    scored: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ch_k16: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if set(scored) != expected:
        raise ValueError("weight-grid scores must form the complete 3x3 grid")
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        k: {cell: [] for cell in CELLS} for k in K_VALUES
    }
    reports: dict[tuple[str, int], dict[str, Any]] = {}
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
                query_ids, selected, _ = selection_vector(rows, k)
                if reference_queries is None:
                    reference_queries = query_ids
                elif query_ids != reference_queries:
                    raise ValueError("weight-grid query population/order drift")
                vectors[k][cell].append(selected)
    assert reference_queries is not None
    if len(reference_queries) != 800:
        raise ValueError("weight-grid tuning population must contain 800 queries")

    by_k: dict[str, Any] = {}
    for k_index, k in enumerate(K_VALUES):
        arrays = {cell: np.stack(vectors[k][cell], axis=0) for cell in CELLS}
        first = reports[("direct_025", 42)]["by_k"][str(k)]
        effects = {
            "direct_025_minus_direct_100": arrays["direct_025"]
            - arrays["direct_100"],
            "direct_050_minus_direct_100": arrays["direct_050"]
            - arrays["direct_100"],
        }
        by_k[str(k)] = {
            "queries": 800,
            "random_expected_accuracy": first["random_expected_accuracy"],
            "oracle_accuracy": first["oracle_accuracy"],
            "cells": {
                cell: {
                    "direct_weight": WEIGHTS[cell],
                    "gate_prior_weight": 0.25,
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
            "paired_diagnostics_vs_direct_100": {
                name: _effect_summary(
                    values,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed + 100 * k_index + effect_index,
                )
                for effect_index, (name, values) in enumerate(effects.items())
            },
        }

    k16_means = {
        cell: by_k["16"]["cells"][cell]["mean_accuracy"] for cell in CELLS
    }
    decision = choose_weight(k16_means)
    selected_mean = k16_means[decision["selected_cell"]]
    ch_mean = float(ch_k16["mean_accuracy"])
    return {
        "schema_version": "clir-prior-gate-tuning-v1-direct-grid-selection",
        "status": "COMPLETE_PRIOR_GATE_TUNING_V1_DIRECT_WEIGHT_SELECTION",
        "created_at_utc": _utc_now(),
        "evidence_tier": "prospective_train_source_weight_tuning_not_confirmation",
        "tuning_queries": 800,
        "ranking": {"by_k": by_k},
        "selection": {
            **decision,
            "selected_mean_k16_accuracy": selected_mean,
            "ch_reference_mean_k16_accuracy": ch_mean,
            "selected_minus_ch_mean": selected_mean - ch_mean,
            "at_least_one_prior_candidate_beats_ch": max(k16_means.values()) > ch_mean,
        },
        "confirmation_outcomes_opened": False,
        "claim_boundary": (
            "weight selected on the open tuning split; benefit must be judged only "
            "after the one-time sealed confirmation"
        ),
    }


def command_summarize(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    merge_path = Path(args.merge_report).resolve()
    authorization, scoring, scoring_path = _load_authorization(authorization_path)
    if (
        merge_path != _project_path(authorization["merge_report_path"])
        or file_sha256(merge_path) != authorization["merge_report_sha256"]
    ):
        raise ValueError("weight-grid merge report hash drift")
    attribution_path = _project_path(authorization["stage_a_attribution_path"])
    if file_sha256(attribution_path) != authorization["stage_a_attribution_sha256"]:
        raise ValueError("Stage-A attribution hash drift")
    attribution = _load_json(attribution_path)
    ch_k16 = attribution["ranking"]["by_k"]["16"]["cells"]["ch"]

    completion_path = _project_path(scoring["training_completion_path"])
    input_path = _project_path(scoring["ranking_input_path"])
    if file_sha256(completion_path) != scoring["training_completion_sha256"]:
        raise ValueError("weight-grid completion hash drift")
    if file_sha256(input_path) != scoring["ranking_input_sha256"]:
        raise ValueError("weight-grid input hash drift")
    completion = _load_json(completion_path)
    if completion.get("status") != COMPLETION_STATUS:
        raise ValueError("weight-grid checkpoint set did not pass")
    source = read_jsonl(input_path)
    merge = _load_json(merge_path)
    if (
        len(source) != 12_800
        or merge.get("status") != MERGE_STATUS
        or merge.get("authorization_file_sha256") != file_sha256(scoring_path)
        or merge.get("input_jsonl_sha256") != scoring["ranking_input_sha256"]
        or merge.get("completion_report_sha256")
        != scoring["training_completion_sha256"]
        or merge.get("scorer_sha256") != scoring["scorer_sha256"]
        or len(merge.get("outputs", {})) != 9
    ):
        raise ValueError("weight-grid score merge is stale or unauthorized")
    run_spec = {
        (str(run["cell"]), int(run["seed"])): run for run in completion["runs"]
    }
    scored: dict[tuple[str, int], list[dict[str, Any]]] = {}
    input_files: dict[str, Any] = {}
    for cell in CELLS:
        for seed in SEEDS:
            key = f"{cell}/seed-{seed}"
            record = merge["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"weight-grid merge lacks {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"weight-grid scored output drift: {path}")
            rows = read_jsonl(path)
            checkpoint = str(run_spec[(cell, seed)]["checkpoint_file_sha256"])
            _validate_compact_rows(rows, source, checkpoint)
            scored[(cell, seed)] = rows
            input_files[key] = {
                "path": str(path),
                "file_sha256": record["file_sha256"],
                "checkpoint_sha256": checkpoint,
            }
    report = summarize(
        scored,
        bootstrap_replicates=int(scoring["bootstrap_replicates"]),
        bootstrap_seed=int(scoring["bootstrap_seed"]),
        ch_k16=ch_k16,
    )
    report.update(
        {
            "authorization_file_sha256": file_sha256(authorization_path),
            "scoring_authorization_file_sha256": file_sha256(scoring_path),
            "training_completion_sha256": scoring["training_completion_sha256"],
            "ranking_input_sha256": scoring["ranking_input_sha256"],
            "score_merge_sha256": file_sha256(merge_path),
            "stage_a_attribution_sha256": file_sha256(attribution_path),
            "runs": input_files,
        }
    )
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"weight-grid selection already exists: {output}")
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "k16": report["ranking"]["by_k"]["16"],
                "selection": report["selection"],
                "confirmation_outcomes_opened": False,
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
