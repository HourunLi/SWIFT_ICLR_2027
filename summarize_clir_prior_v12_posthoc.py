#!/usr/bin/env python
"""Validate and summarize the frozen Prior-v12 post-hoc R0/P0 ranking run."""

from __future__ import annotations

import argparse
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
    / "configs/data_expansion_prior_v12/posthoc_v1/"
    "ranking_evaluation_authorization.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12/posthoc_v1"
)
EXPECTED_EXTRA_FIELDS = {
    "clir_checkpoint_sha256",
    "clir_score",
    "clir_scoring_mode",
    "clir_selected_best_of_n",
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        authorization.get("schema_version")
        != "clir-prior-v12-posthoc-ranking-evaluation-authorization-v1"
        or authorization.get("status")
        != "AUTHORIZED_FROZEN_EXPLORATORY_R0_P0_RANKING_EVALUATION"
        or authorization.get("frozen_before_scored_outputs_completed") is not True
    ):
        raise ValueError("unsupported or inactive ranking authorization")
    boundary = authorization["evidence_boundary"]
    if (
        boundary.get("tier")
        != "posthoc_exploratory_silver_no_human_verification"
        or boundary.get("confirmatory_or_protected_test") is not False
        or boundary.get("original_v12_status")
        != "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE"
        or boundary.get("original_v13_status") != "FAIL_PRIOR_V13_SCHEMA"
    ):
        raise ValueError("ranking evidence boundary drift")
    if authorization["grid"] != {
        "cells": ["r0", "p0"],
        "seeds": [42, 43, 44],
        "same_candidate_population_required": True,
        "all_six_runs_required": True,
    }:
        raise ValueError("ranking run grid drift")
    metrics = authorization["metrics"]
    if (
        metrics.get("primary")
        != "paired_query_level_p0_minus_r0_Best_of_N_accuracy_at_K_16"
        or metrics.get("k") != [1, 2, 4, 8, 16]
        or int(metrics.get("bootstrap_replicates", -1)) != 10_000
    ):
        raise ValueError("ranking metric contract drift")
    return authorization


def _validate_frozen_inputs(authorization: Mapping[str, Any]) -> Path:
    frozen = authorization["frozen_inputs"]
    fixed_paths = {
        "training_authorization_sha256": (
            PROJECT_ROOT
            / "configs/data_expansion_prior_v12/posthoc_v1/"
            "training_authorization.json"
        ),
        "training_completion_report_sha256": (
            DEFAULT_OUTPUT_ROOT / "training/completion_report.json"
        ),
        "prior_dev_mechanism_report_sha256": (
            DEFAULT_OUTPUT_ROOT / "evaluation/dev_mechanism_report.json"
        ),
    }
    for key, path in fixed_paths.items():
        _assert_hash(path, str(frozen[key]), key)

    ranking = frozen["ranking_manifest"]
    ranking_path = _project_path(ranking["path"])
    _assert_hash(ranking_path, str(ranking["sha256"]), "ranking manifest")

    checkpoint_paths = {
        f"{cell}_seed_{seed}": (
            DEFAULT_OUTPUT_ROOT / f"training/{cell}/seed-{seed}/checkpoint.pt"
        )
        for cell in ("r0", "p0")
        for seed in (42, 43, 44)
    }
    for key, path in checkpoint_paths.items():
        _assert_hash(path, str(frozen["checkpoints"][key]), key)
    return ranking_path


def _selection(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> tuple[list[str], dict[int, np.ndarray], dict[int, np.ndarray]]:
    grouped = group_rows(rows)
    query_ids = sorted(grouped)
    labels = {k: [] for k in k_values}
    indices = {k: [] for k in k_values}
    for query_id in query_ids:
        candidates = grouped[query_id]
        if len(candidates) != 16:
            raise ValueError(f"{query_id} does not have 16 frozen candidates")
        for k in k_values:
            prefix = candidates[:k]
            best = max(range(k), key=lambda index: float(prefix[index]["clir_score"]))
            labels[k].append(float(prefix[best]["correctness"]))
            indices[k].append(int(prefix[best]["candidate_index"]))
        selected_flags = [bool(row["clir_selected_best_of_n"]) for row in candidates]
        if sum(selected_flags) != 1 or not selected_flags[indices[16][-1]]:
            raise ValueError(f"{query_id} has an invalid full Best-of-N marker")
    return (
        query_ids,
        {k: np.asarray(values, dtype=np.float64) for k, values in labels.items()},
        {k: np.asarray(values, dtype=np.int64) for k, values in indices.items()},
    )


def _load_and_validate_run(
    scored_path: Path,
    reference_rows: Sequence[Mapping[str, Any]],
    expected_checkpoint_sha256: str,
    k_values: Sequence[int],
) -> dict[str, Any]:
    rows = read_jsonl(scored_path)
    if len(rows) != len(reference_rows):
        raise ValueError(f"row-count drift in {scored_path}")
    for index, (reference, row) in enumerate(zip(reference_rows, rows)):
        if set(row) != set(reference) | EXPECTED_EXTRA_FIELDS:
            raise ValueError(f"row {index} field drift in {scored_path}")
        if any(row[key] != value for key, value in reference.items()):
            raise ValueError(f"row {index} frozen-input drift in {scored_path}")
        if row["clir_checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError(f"row {index} checkpoint drift in {scored_path}")
        if row["clir_scoring_mode"] != "scalar_only":
            raise ValueError(f"row {index} is not scalar-only in {scored_path}")
        if not math.isfinite(float(row["clir_score"])):
            raise ValueError(f"row {index} has a non-finite score in {scored_path}")
        if not isinstance(row["clir_selected_best_of_n"], bool):
            raise ValueError(f"row {index} has a non-boolean selection marker")

    report = evaluate(
        rows,
        score_field="clir_score",
        correctness_field="correctness",
        k_values=k_values,
        bootstrap_replicates=0,
        seed=0,
    )
    query_ids, selected, selected_indices = _selection(rows, k_values)
    return {
        "query_ids": query_ids,
        "selected": selected,
        "selected_indices": selected_indices,
        "metrics": report,
        "scored_jsonl_sha256": file_sha256(scored_path),
    }


def _sample_sd(values: Sequence[float]) -> float | None:
    return (
        float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
        if len(values) > 1
        else None
    )


def summarize(
    authorization_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    authorization_path = Path(authorization_path).resolve()
    authorization = load_authorization(authorization_path)
    ranking_path = _validate_frozen_inputs(authorization)
    reference_rows = read_jsonl(ranking_path)
    ranking_contract = authorization["frozen_inputs"]["ranking_manifest"]
    if len(reference_rows) != int(ranking_contract["rows"]):
        raise ValueError("frozen ranking row-count drift")
    if len({str(row["query_id"]) for row in reference_rows}) != int(
        ranking_contract["queries"]
    ):
        raise ValueError("frozen ranking query-count drift")

    root = Path(output_root).resolve()
    cells = list(authorization["grid"]["cells"])
    seeds = [int(value) for value in authorization["grid"]["seeds"]]
    k_values = [int(value) for value in authorization["metrics"]["k"]]
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    reference_query_ids: list[str] | None = None
    for cell in cells:
        for seed in seeds:
            key = f"{cell}_seed_{seed}"
            run = _load_and_validate_run(
                root / f"evaluation/ranking_scored/{cell}_seed-{seed}.jsonl",
                reference_rows,
                authorization["frozen_inputs"]["checkpoints"][key],
                k_values,
            )
            if reference_query_ids is None:
                reference_query_ids = run["query_ids"]
            elif run["query_ids"] != reference_query_ids:
                raise ValueError("query identity/order differs across runs")
            loaded[(cell, seed)] = run

    by_k: dict[str, Any] = {}
    bootstrap = authorization["metrics"]
    for k in k_values:
        cell_payload: dict[str, Any] = {}
        for cell in cells:
            values = [
                float(loaded[(cell, seed)]["selected"][k].mean()) for seed in seeds
            ]
            cell_payload[cell] = {
                "mean_accuracy": float(np.mean(values)),
                "sample_sd_across_seeds": _sample_sd(values),
                "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
        deltas = np.stack(
            [
                loaded[("p0", seed)]["selected"][k]
                - loaded[("r0", seed)]["selected"][k]
                for seed in seeds
            ]
        )
        per_seed_delta = deltas.mean(axis=1)
        first_metrics = loaded[("r0", seeds[0])]["metrics"]["by_k"][str(k)]
        by_k[str(k)] = {
            "queries": len(reference_query_ids or []),
            "random_expected_accuracy": first_metrics["random_expected_accuracy"],
            "oracle_accuracy": first_metrics["oracle_accuracy"],
            "cells": cell_payload,
            "p0_minus_r0": {
                "mean_paired_delta": float(deltas.mean()),
                "sample_sd_across_seed_deltas": _sample_sd(per_seed_delta.tolist()),
                "by_seed": {
                    str(seed): float(value)
                    for seed, value in zip(seeds, per_seed_delta.tolist())
                },
                "seed_direction_counts": {
                    "positive": int((per_seed_delta > 0).sum()),
                    "zero": int((per_seed_delta == 0).sum()),
                    "negative": int((per_seed_delta < 0).sum()),
                },
                **paired_bootstrap_ci(
                    deltas,
                    int(bootstrap["bootstrap_replicates"]),
                    int(bootstrap["bootstrap_seed"]) + k,
                ),
            },
        }

    pairwise: dict[str, Any] = {}
    for cell in cells:
        values = [
            float(loaded[(cell, seed)]["metrics"]["within_query_pairwise"]["accuracy"])
            for seed in seeds
        ]
        pairwise[cell] = {
            "comparisons": loaded[(cell, seeds[0])]["metrics"][
                "within_query_pairwise"
            ]["comparisons"],
            "mean_accuracy": float(np.mean(values)),
            "sample_sd_across_seeds": _sample_sd(values),
            "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
        }

    runs = {
        f"{cell}_seed_{seed}": {
            "checkpoint_sha256": authorization["frozen_inputs"]["checkpoints"][
                f"{cell}_seed_{seed}"
            ],
            "scored_jsonl_sha256": loaded[(cell, seed)]["scored_jsonl_sha256"],
        }
        for cell in cells
        for seed in seeds
    }
    return {
        "schema_version": "clir-prior-v12-posthoc-ranking-summary-v1",
        "status": "COMPLETE_PRIOR_V12_POSTHOC_EXPLORATORY_R0_P0_RANKING_EVALUATION",
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "original_v12_status": "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE",
        "original_v13_status": "FAIL_PRIOR_V13_SCHEMA",
        "authorization_file_sha256": file_sha256(authorization_path),
        "ranking_manifest_sha256": ranking_contract["sha256"],
        "ranking_rows": len(reference_rows),
        "ranking_queries": len(reference_query_ids or []),
        "runs": runs,
        "ranking": {"by_k": by_k, "within_query_pairwise": pairwise},
        "bootstrap": {
            "replicates": int(bootstrap["bootstrap_replicates"]),
            "seed": int(bootstrap["bootstrap_seed"]),
            "fixed_seed_unit": "paired query averaged across the three fixed seeds",
            "hierarchical_unit": "exploratory paired seed and query resampling",
        },
        "claim_boundary": (
            "reused ranking diagnostics only; not Gold, human-verified, fresh "
            "confirmatory, protected-test, mutual-prior, gate, or Full evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen Prior-v12 post-hoc ranking comparison."
    )
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output = (
        Path(args.output_json).resolve()
        if args.output_json
        else output_root / "evaluation/ranking_summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    report = summarize(args.authorization, output_root)
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
