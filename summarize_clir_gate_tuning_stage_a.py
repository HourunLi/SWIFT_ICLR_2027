#!/usr/bin/env python
"""Summarize fresh CH vs direct-P/Gate=0 vs Full(.25) attribution."""

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
from src.clir_gate_tuning import choose_tuning_axis
from summarize_clir_three_module_ranking import (
    _effect_summary,
    _sample_sd,
    _validate_compact_rows,
    selection_vector,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/stage_a_summary_authorization.json"
)
DEFAULT_MERGE = (
    PROJECT_ROOT
    / "run_artifacts/prior_gate_tuning_v1/stage_a/scoring/merged/merge_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/stage_a/attribution.json"
)
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_STAGE_A_SUMMARY"
SCORING_AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_STAGE_A_SCORING"
STAGE_A_COMPLETION_STATUS = "PASS_PRIOR_GATE_TUNING_V1_STAGE_A_9_CHECKPOINTS"
CELLS = ("ch", "direct_gate0", "full_025")
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


def _validate_scoring_authorization(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != SCORING_AUTHORIZATION_STATUS:
        raise ValueError("Prior/Gate Stage-A scoring did not pass authorization")
    if payload.get("cells") != list(CELLS) or payload.get("seeds") != list(SEEDS):
        raise ValueError("Stage-A cell/seed grid drift")
    if payload.get("k") != list(K_VALUES):
        raise ValueError("Stage-A K grid drift")
    if int(payload.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Stage-A bootstrap budget drift")
    if payload.get("confirmation_scoring_allowed") is not False:
        raise ValueError("Stage-A scoring must keep confirmation sealed")


def _load_authorization(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    payload = _load_json(path)
    if payload.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("Prior/Gate Stage-A summarization is not authorized")
    if payload.get("summarizer_sha256") != file_sha256(__file__):
        raise ValueError("Stage-A authorization binds another summarizer")
    scoring_path = _project_path(payload["scoring_authorization_path"])
    if file_sha256(scoring_path) != payload["scoring_authorization_sha256"]:
        raise ValueError("Stage-A scoring authorization hash drift")
    scoring = _load_json(scoring_path)
    _validate_scoring_authorization(scoring)
    return payload, scoring, scoring_path


def _source_accuracy(
    rows: Sequence[Mapping[str, Any]], query_ids: Sequence[str], selected: np.ndarray
) -> dict[str, Any]:
    source_by_query: dict[str, str] = {}
    for row in rows:
        query_id = str(row["query_id"])
        source = str(row.get("source") or query_id.split(":", 1)[0])
        if source not in {"gsm8k", "math"}:
            raise ValueError(f"cannot recover source namespace for {query_id}")
        if query_id in source_by_query and source_by_query[query_id] != source:
            raise ValueError("query spans multiple sources")
        source_by_query[query_id] = source
    output: dict[str, Any] = {}
    for source in sorted(set(source_by_query.values())):
        mask = np.asarray(
            [source_by_query[query_id] == source for query_id in query_ids],
            dtype=bool,
        )
        output[source] = {
            "queries": int(mask.sum()),
            "accuracy": float(selected[mask].mean()),
        }
    return output


def summarize(
    scored: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if set(scored) != expected:
        raise ValueError("Stage-A scores must form the complete 3x3 grid")
    vectors: dict[int, dict[str, list[np.ndarray]]] = {
        k: {cell: [] for cell in CELLS} for k in K_VALUES
    }
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    reference_queries: list[str] | None = None
    source_diagnostics: dict[str, Any] = {}
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
                    raise ValueError("Stage-A query order/population differs across runs")
                vectors[k][cell].append(selected)
                if k == 16:
                    source_diagnostics[f"{cell}/seed-{seed}"] = _source_accuracy(
                        rows, query_ids, selected
                    )
    assert reference_queries is not None
    if len(reference_queries) != 800:
        raise ValueError("Stage-A tuning population must contain 800 queries")

    by_k: dict[str, Any] = {}
    for k_index, k in enumerate(K_VALUES):
        arrays = {
            cell: np.stack(vectors[k][cell], axis=0) for cell in CELLS
        }
        effects = {
            "direct_prior_effect": arrays["direct_gate0"] - arrays["ch"],
            "gate_effect_given_direct_prior": (
                arrays["full_025"] - arrays["direct_gate0"]
            ),
            "full_025_minus_ch": arrays["full_025"] - arrays["ch"],
        }
        first = reports[("ch", 42)]["by_k"][str(k)]
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
            "paired_effects": {
                name: _effect_summary(
                    values,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed + 100 * k_index + effect_index,
                )
                for effect_index, (name, values) in enumerate(effects.items())
            },
        }

    k16 = by_k["16"]["cells"]
    axis = choose_tuning_axis(
        k16["ch"]["by_seed"],
        k16["direct_gate0"]["by_seed"],
        k16["full_025"]["by_seed"],
    )
    return {
        "schema_version": "clir-prior-gate-tuning-v1-stage-a-attribution",
        "status": "COMPLETE_PRIOR_GATE_TUNING_V1_STAGE_A_ATTRIBUTION",
        "created_at_utc": _utc_now(),
        "evidence_tier": "prospective_train_source_weight_tuning_not_confirmation",
        "tuning_queries": len(reference_queries),
        "ranking": {"by_k": by_k},
        "source_diagnostics_at_k16": source_diagnostics,
        "axis_decision": axis,
        "axis_rule": (
            "if both K16 increment means are nonnegative, tune neither; otherwise "
            "open only the more-negative of direct_prior and gate"
        ),
        "confirmation_outcomes_opened": False,
        "claim_boundary": (
            "fresh query/template-disjoint train-source tuning evidence only; "
            "not protected-test evidence and not the sealed confirmation result"
        ),
    }


def command_summarize(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    merge_path = Path(args.merge_report).resolve()
    authorization, scoring_authorization, scoring_authorization_path = (
        _load_authorization(authorization_path)
    )
    expected_merge = _project_path(authorization["merge_report_path"])
    if merge_path != expected_merge or file_sha256(merge_path) != authorization[
        "merge_report_sha256"
    ]:
        raise ValueError("Stage-A merge report hash drift")
    completion_path = _project_path(
        scoring_authorization["training_completion_path"]
    )
    input_path = _project_path(scoring_authorization["ranking_input_path"])
    if (
        file_sha256(completion_path)
        != scoring_authorization["training_completion_sha256"]
    ):
        raise ValueError("Stage-A training completion hash drift")
    if file_sha256(input_path) != scoring_authorization["ranking_input_sha256"]:
        raise ValueError("Stage-A tuning population hash drift")
    completion = _load_json(completion_path)
    if completion.get("status") != STAGE_A_COMPLETION_STATUS:
        raise ValueError("Stage-A checkpoint set did not pass")
    source = read_jsonl(input_path)
    if len(source) != 12_800 or len({str(row["query_id"]) for row in source}) != 800:
        raise ValueError("Stage-A tuning input inventory drift")
    merge = _load_json(merge_path)
    if (
        merge.get("status") != MERGE_STATUS
        or merge.get("input_jsonl_sha256")
        != scoring_authorization["ranking_input_sha256"]
        or merge.get("completion_report_sha256")
        != scoring_authorization["training_completion_sha256"]
        or merge.get("authorization_file_sha256")
        != file_sha256(scoring_authorization_path)
        or merge.get("scorer_sha256") != scoring_authorization["scorer_sha256"]
        or len(merge.get("outputs", {})) != 9
    ):
        raise ValueError("Stage-A score merge is stale or unauthorized")

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
                raise ValueError(f"Stage-A score merge lacks {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"Stage-A scored output hash drift: {path}")
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
        bootstrap_replicates=int(scoring_authorization["bootstrap_replicates"]),
        bootstrap_seed=int(scoring_authorization["bootstrap_seed"]),
    )
    report.update(
        {
            "authorization_file_sha256": file_sha256(authorization_path),
            "scoring_authorization_file_sha256": file_sha256(
                scoring_authorization_path
            ),
            "training_completion_sha256": scoring_authorization[
                "training_completion_sha256"
            ],
            "ranking_input_sha256": scoring_authorization["ranking_input_sha256"],
            "score_merge_sha256": file_sha256(merge_path),
            "runs": input_files,
        }
    )
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Stage-A attribution already exists: {output}")
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "k16": report["ranking"]["by_k"]["16"],
                "axis_decision": report["axis_decision"],
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
