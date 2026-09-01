#!/usr/bin/env python
"""Summarize the frozen 8-cell x 3-seed CLIR ranking factorial."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, evaluate, file_sha256, group_rows
from src.clir_data import read_jsonl
from summarize_clir_ablation import paired_bootstrap_ci


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/three_module_expansion_v1/ranking_evaluation_authorization.json"
)
DEFAULT_MERGE = (
    PROJECT_ROOT
    / "run_artifacts/three_module_expansion_v1/evaluation/ranking_scoring/merged/merge_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "run_artifacts/three_module_expansion_v1/evaluation/ranking_summary.json"
)
AUTHORIZATION_STATUS = "AUTHORIZED_THREE_MODULE_FACTORIAL_RANKING_V1"
CELLS = ("u0", "c", "h", "p", "ch", "cp", "hp", "full")
SEEDS = (42, 43, 44)
K_VALUES = (1, 2, 4, 8, 16)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def selection_vector(
    rows: Sequence[Mapping[str, Any]], k: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped = group_rows(rows)
    query_ids = sorted(grouped)
    labels: list[float] = []
    candidate_indices: list[int] = []
    for query_id in query_ids:
        candidates = grouped[query_id]
        if len(candidates) != 16:
            raise ValueError(f"{query_id} does not have exactly 16 candidates")
        prefix = candidates[:k]
        scores = [float(row["clir_score"]) for row in prefix]
        correctness = [float(row["correctness"]) for row in prefix]
        if not np.isfinite(scores).all() or not np.isin(
            correctness, [0.0, 1.0]
        ).all():
            raise ValueError(f"invalid ranking values for {query_id}")
        best = max(range(k), key=lambda index: scores[index])
        labels.append(correctness[best])
        candidate_indices.append(int(prefix[best]["candidate_index"]))
    return (
        query_ids,
        np.asarray(labels, dtype=np.float64),
        np.asarray(candidate_indices, dtype=np.int64),
    )


def factorial_vectors(cells: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(cells) != set(CELLS):
        raise ValueError("factorial vectors require all eight cells")
    u0, c, h, p, ch, cp, hp, full = (cells[cell] for cell in CELLS)
    shapes = {array.shape for array in cells.values()}
    if len(shapes) != 1:
        raise ValueError("factorial cell vectors are not paired")
    return {
        "C_main": (c + ch + cp + full - u0 - h - p - hp) / 4,
        "H_main": (h + ch + hp + full - u0 - c - p - cp) / 4,
        "P_main": (p + cp + hp + full - u0 - c - h - ch) / 4,
        "C_x_H": (u0 - c - h + ch + p - cp - hp + full) / 2,
        "C_x_P": (u0 - c + h - ch - p + cp - hp + full) / 2,
        "H_x_P": (u0 + c - h - ch - p - cp + hp + full) / 2,
        "C_x_H_x_P": full - ch - cp - hp + c + h + p - u0,
        "Full_minus_U0": full - u0,
        "Full_minus_C": full - c,
        "Full_minus_H": full - h,
        "Full_minus_P": full - p,
        "Full_minus_CH": full - ch,
        "Full_minus_CP": full - cp,
        "Full_minus_HP": full - hp,
    }


def _effect_summary(
    values: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] != 3 or values.shape[1] <= 0:
        raise ValueError("paired effect values must have shape [3, queries]")
    per_seed = values.mean(axis=1)
    return {
        "mean_paired_effect": float(values.mean()),
        "sample_sd_across_seed_effects": _sample_sd(per_seed.tolist()),
        "by_seed": {
            str(seed): float(value) for seed, value in zip(SEEDS, per_seed)
        },
        "seed_direction_counts": {
            "positive": int((per_seed > 0).sum()),
            "zero": int((per_seed == 0).sum()),
            "negative": int((per_seed < 0).sum()),
        },
        **paired_bootstrap_ci(values, bootstrap_replicates, bootstrap_seed),
    }


def _validate_compact_rows(
    rows: Sequence[Mapping[str, Any]],
    source: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> None:
    if len(rows) != len(source):
        raise ValueError("ranking scored/source row-count drift")
    for index, (row, original) in enumerate(zip(rows, source)):
        if int(row.get("source_row_index", -1)) != index:
            raise ValueError("ranking source-row index drift")
        for field in ("id", "query_id", "candidate_index", "correctness"):
            if row.get(field) != original.get(field):
                raise ValueError(f"ranking compact field drift: {field}")
        if row.get("clir_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("ranking checkpoint identity drift")
        if row.get("clir_scoring_mode") != "scalar_only":
            raise ValueError("ranking score is not scalar-only")
        if not math.isfinite(float(row["clir_score"])):
            raise FloatingPointError("ranking score is non-finite")


def _load_authorization(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("three-module ranking is not authorized")
    if payload.get("summarizer_sha256") != file_sha256(__file__):
        raise ValueError("ranking authorization binds another summarizer")
    if payload.get("k") != list(K_VALUES):
        raise ValueError("ranking K grid drift")
    if payload.get("seeds") != list(SEEDS) or payload.get("cells") != list(CELLS):
        raise ValueError("ranking factorial grid drift")
    if int(payload.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("ranking bootstrap budget drift")
    return payload


def summarize(
    scored: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if set(scored) != expected:
        raise ValueError("ranking scores must form the complete 8x3 grid")
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
                    raise ValueError("ranking query order/population differs across runs")
                vectors[k][cell].append(selected)
                indices[k][(cell, seed)] = selected_indices
    assert reference_queries is not None
    if len(reference_queries) != 892:
        raise ValueError("ranking population must contain exactly 892 queries")

    by_k: dict[str, Any] = {}
    for k_index, k in enumerate(K_VALUES):
        arrays = {cell: np.stack(vectors[k][cell], axis=0) for cell in CELLS}
        effects = factorial_vectors(arrays)
        first = reports[("u0", 42)]["by_k"][str(k)]
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
                        for seed, value in zip(SEEDS, arrays[cell].mean(axis=1))
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

    pairwise = {
        cell: {
            "comparisons": reports[(cell, 42)]["within_query_pairwise"][
                "comparisons"
            ],
            "mean_accuracy": float(
                np.mean(
                    [
                        reports[(cell, seed)]["within_query_pairwise"]["accuracy"]
                        for seed in SEEDS
                    ]
                )
            ),
            "by_seed": {
                str(seed): reports[(cell, seed)]["within_query_pairwise"]["accuracy"]
                for seed in SEEDS
            },
        }
        for cell in CELLS
    }

    selection_changes: dict[str, Any] = {}
    for seed in SEEDS:
        u0_indices = indices[16][("u0", seed)]
        full_indices = indices[16][("full", seed)]
        u0_labels = vectors[16]["u0"][SEEDS.index(seed)]
        full_labels = vectors[16]["full"][SEEDS.index(seed)]
        changed = u0_indices != full_indices
        selection_changes[str(seed)] = {
            "queries": len(reference_queries),
            "changed_candidate": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
            "wrong_to_correct": int(((u0_labels == 0) & (full_labels == 1)).sum()),
            "correct_to_wrong": int(((u0_labels == 1) & (full_labels == 0)).sum()),
            "changed_same_correctness": int(
                (changed & (u0_labels == full_labels)).sum()
            ),
        }

    primary = by_k["16"]["paired_effects"]["Full_minus_U0"]
    fixed_interval = primary["fixed_seed_query_95_ci"]
    benefit = (
        primary["mean_paired_effect"] > 0
        and primary["seed_direction_counts"]["positive"] >= 2
        and fixed_interval[0] > 0
    )
    harm = (
        primary["mean_paired_effect"] < 0
        and primary["seed_direction_counts"]["negative"] >= 2
    )
    decision = (
        "EXPLORATORY_BENEFIT_CANDIDATE"
        if benefit
        else "EXPLORATORY_HARM_SCREEN"
        if harm
        else "EXPLORATORY_INCONCLUSIVE"
    )
    return {
        "schema_version": "clir-three-module-factorial-ranking-summary-v1",
        "status": "COMPLETE_THREE_MODULE_FACTORIAL_EXPLORATORY_RANKING",
        "created_at_utc": _utc_now(),
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "ranking_queries": len(reference_queries),
        "ranking": {"by_k": by_k, "within_query_pairwise": pairwise},
        "selection_changes_full_vs_u0_at_k16": selection_changes,
        "primary_decision": {
            "contrast": "Full_minus_U0_at_K16",
            "result": decision,
            "benefit_candidate": benefit,
            "harm_screen": harm,
            "rule": (
                "benefit requires positive mean, at least two positive seed effects, "
                "and fixed-seed paired-query 95% CI lower bound above zero; harm "
                "requires negative mean and at least two negative seed effects"
            ),
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "fixed_seed_query": "resample paired queries then average three fixed seeds",
            "hierarchical": "exploratory resample seeds and paired queries",
        },
        "factorial_effect_definition": {
            "main": "average on-minus-off over the other two factors within seed/query",
            "two_way": "average difference-in-differences over the third factor",
            "three_way": "difference-in-difference-in-differences",
        },
        "claim_boundary": (
            "reused exploratory 892-query ranking only; not Gold, human verified, "
            "fresh confirmatory, protected test, or a repair of v7/v12/v13"
        ),
    }


def command_summarize(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    merge_path = Path(args.merge_report).resolve()
    authorization = _load_authorization(authorization_path)
    completion_path = _project_path(authorization["training_completion_path"])
    mechanism_path = _project_path(authorization["mechanism_report_path"])
    ranking_path = _project_path(authorization["ranking_input_path"])
    if file_sha256(completion_path) != authorization["training_completion_sha256"]:
        raise ValueError("ranking authorization training completion drift")
    if file_sha256(mechanism_path) != authorization["mechanism_report_sha256"]:
        raise ValueError("ranking authorization mechanism report drift")
    if file_sha256(ranking_path) != authorization["ranking_input_sha256"]:
        raise ValueError("ranking authorization population drift")
    source = read_jsonl(ranking_path)
    if (
        len(source) != int(authorization["ranking_rows"])
        or len({str(row["query_id"]) for row in source})
        != int(authorization["ranking_queries"])
    ):
        raise ValueError("ranking source inventory drift")
    merge = _load_json(merge_path)
    if (
        merge.get("status") != "PASS_FACTORIAL_SCORING_MERGE"
        or merge.get("mode") != "scalar"
        or merge.get("input_jsonl_sha256") != authorization["ranking_input_sha256"]
        or merge.get("completion_report_sha256")
        != authorization["training_completion_sha256"]
        or merge.get("ranking_authorization_sha256")
        != file_sha256(authorization_path)
        or merge.get("scorer_sha256") != authorization["scorer_sha256"]
        or len(merge.get("outputs", {})) != 24
    ):
        raise ValueError("ranking score merge is stale or unauthorized")

    completion = _load_json(completion_path)
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
                raise ValueError(f"ranking merge lacks {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"ranking scored output hash drift: {path}")
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
        bootstrap_replicates=int(authorization["bootstrap_replicates"]),
        bootstrap_seed=int(authorization["bootstrap_seed"]),
    )
    report.update(
        {
            "authorization_file_sha256": file_sha256(authorization_path),
            "training_completion_sha256": authorization[
                "training_completion_sha256"
            ],
            "mechanism_report_sha256": authorization["mechanism_report_sha256"],
            "ranking_input_sha256": authorization["ranking_input_sha256"],
            "score_merge_sha256": file_sha256(merge_path),
            "runs": input_files,
        }
    )
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"ranking summary already exists: {output}")
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "primary_decision": report["primary_decision"],
            },
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


def main() -> None:
    command_summarize(build_parser().parse_args())


if __name__ == "__main__":
    main()
