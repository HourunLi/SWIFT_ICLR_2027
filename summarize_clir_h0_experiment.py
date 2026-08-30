"""Produce the frozen paired summary for the CLIR H0 v7.4 experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, evaluate, file_sha256, group_rows
from evaluate_clir_h0 import evaluate_h0
from src.clir_data import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/ranking_expansion_v7/h0_experiment_v7_4/training_authorization.json"
)


def _selection_vector(
    rows: Sequence[Mapping[str, Any]], k: int
) -> tuple[list[str], np.ndarray]:
    grouped = group_rows(rows)
    query_ids = sorted(grouped)
    selected: list[float] = []
    for query_id in query_ids:
        candidates = grouped[query_id]
        if len(candidates) != 16:
            raise ValueError(f"{query_id} does not have the frozen 16 candidates")
        prefix = candidates[:k]
        scores = [float(row["clir_score"]) for row in prefix]
        correctness = [float(row["correctness"]) for row in prefix]
        if not np.isfinite(scores).all() or not np.isin(correctness, [0.0, 1.0]).all():
            raise ValueError(f"{query_id} has invalid score or correctness")
        # Python max is stable, so ties use the lowest frozen candidate index.
        best = max(range(k), key=lambda index: scores[index])
        selected.append(correctness[best])
    return query_ids, np.asarray(selected, dtype=np.float64)


def _hierarchical_bootstrap(
    values: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    cells = ("c0", "c1", "h0", "ch0")
    shapes = {cell: values[cell].shape for cell in cells}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"paired value shapes differ: {shapes}")
    seed_count, query_count = next(iter(shapes.values()))
    if seed_count != 3 or query_count <= 0:
        raise ValueError("frozen bootstrap expects three seeds and non-empty queries")
    rng = np.random.default_rng(seed)
    draws = {
        "c0": np.empty(replicates, dtype=np.float64),
        "c1": np.empty(replicates, dtype=np.float64),
        "h0": np.empty(replicates, dtype=np.float64),
        "ch0": np.empty(replicates, dtype=np.float64),
        "c1-c0": np.empty(replicates, dtype=np.float64),
        "h0-c0": np.empty(replicates, dtype=np.float64),
        "ch0-c0": np.empty(replicates, dtype=np.float64),
        "ch0-c1-h0+c0": np.empty(replicates, dtype=np.float64),
    }
    for replicate in range(replicates):
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        query_indices = rng.integers(0, query_count, size=query_count)
        means = {
            cell: float(
                values[cell][seed_indices][:, query_indices].mean()
            )
            for cell in cells
        }
        for cell in cells:
            draws[cell][replicate] = means[cell]
        draws["c1-c0"][replicate] = means["c1"] - means["c0"]
        draws["h0-c0"][replicate] = means["h0"] - means["c0"]
        draws["ch0-c0"][replicate] = means["ch0"] - means["c0"]
        draws["ch0-c1-h0+c0"][replicate] = (
            means["ch0"] - means["c1"] - means["h0"] + means["c0"]
        )
    return {
        name: [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ]
        for name, samples in draws.items()
    }


def _metric_path(payload: Mapping[str, Any], path: str) -> float | None:
    value: Any = payload
    for component in path.split("."):
        value = value[component]
    return None if value is None else float(value)


def _summarize_seed_metric(
    reports: Sequence[Mapping[str, Any]], path: str
) -> dict[str, Any]:
    values = [_metric_path(report, path) for report in reports]
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return {
        "path": path,
        "by_seed": values,
        "mean": float(np.mean(finite)) if finite else None,
        "min": float(np.min(finite)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
    }


def summarize(
    scored: Mapping[tuple[str, int], Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    k_values: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    onset_threshold: float,
    onset_window_tokens: int,
) -> dict[str, Any]:
    cells = ("c0", "c1", "h0", "ch0")
    seeds = (42, 43, 44)
    expected = {(cell, seed) for cell in cells for seed in seeds}
    if set(scored) != expected:
        raise ValueError("scored run grid must be exactly four cells by three seeds")

    h_reports: dict[tuple[str, int], dict[str, Any]] = {}
    ranking_reports: dict[tuple[str, int], dict[str, Any]] = {}
    ranking_vectors: dict[int, dict[str, list[np.ndarray]]] = {
        k: {cell: [] for cell in cells} for k in k_values
    }
    reference_queries: list[str] | None = None
    checkpoint_hashes: set[str] = set()
    for cell in cells:
        for seed in seeds:
            run = scored[(cell, seed)]
            h_report = evaluate_h0(
                run["h_dev"],
                onset_threshold=onset_threshold,
                onset_window_tokens=onset_window_tokens,
            )
            ranking_report = evaluate(
                run["ranking"],
                score_field="clir_score",
                correctness_field="correctness",
                k_values=k_values,
                bootstrap_replicates=0,
                seed=bootstrap_seed,
            )
            ranking_checkpoints = {
                str(row["clir_checkpoint_sha256"]) for row in run["ranking"]
            }
            if ranking_checkpoints != {h_report["checkpoint_sha256"]}:
                raise ValueError(f"{cell}/{seed} H and ranking checkpoint mismatch")
            checkpoint_hashes.update(ranking_checkpoints)
            h_reports[(cell, seed)] = h_report
            ranking_reports[(cell, seed)] = ranking_report
            for k in k_values:
                queries, vector = _selection_vector(run["ranking"], k)
                if reference_queries is None:
                    reference_queries = queries
                elif queries != reference_queries:
                    raise ValueError("ranking query population/order differs across runs")
                ranking_vectors[k][cell].append(vector)
    if len(checkpoint_hashes) != 12:
        raise ValueError("the 12 runs must have 12 distinct checkpoint files")
    assert reference_queries is not None
    if len(reference_queries) != 892:
        raise ValueError("frozen ranking population must contain 892 queries")

    ranking_summary: dict[str, Any] = {}
    for k_index, k in enumerate(k_values):
        arrays = {
            cell: np.stack(ranking_vectors[k][cell], axis=0) for cell in cells
        }
        intervals = _hierarchical_bootstrap(
            arrays,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + k_index,
        )
        cell_summary = {
            cell: {
                "by_seed": [float(value) for value in arrays[cell].mean(axis=1)],
                "mean_accuracy": float(arrays[cell].mean()),
                "hierarchical_bootstrap_95_ci": intervals[cell],
            }
            for cell in cells
        }
        effects: dict[str, Any] = {}
        for name, left, right in (
            ("c1-c0", "c1", "c0"),
            ("h0-c0", "h0", "c0"),
            ("ch0-c0", "ch0", "c0"),
        ):
            difference = arrays[left] - arrays[right]
            effects[name] = {
                "by_seed": [float(value) for value in difference.mean(axis=1)],
                "mean_paired_difference": float(difference.mean()),
                "hierarchical_bootstrap_95_ci": intervals[name],
            }
        interaction = arrays["ch0"] - arrays["c1"] - arrays["h0"] + arrays["c0"]
        effects["ch0-c1-h0+c0"] = {
            "by_seed": [float(value) for value in interaction.mean(axis=1)],
            "mean_paired_difference": float(interaction.mean()),
            "hierarchical_bootstrap_95_ci": intervals["ch0-c1-h0+c0"],
        }
        first_report = ranking_reports[("c0", 42)]["by_k"][str(k)]
        ranking_summary[str(k)] = {
            "queries": len(reference_queries),
            "cells": cell_summary,
            "paired_effects": effects,
            "random_expected_accuracy": first_report["random_expected_accuracy"],
            "oracle_accuracy": first_report["oracle_accuracy"],
        }

    h_metric_paths = (
        "token.auroc",
        "token.average_precision",
        "token.binary_cross_entropy",
        "path.auroc",
        "path.average_precision",
        "path.binary_cross_entropy",
        "onset.positive_detection_rate",
        "onset.positive_exact_start_rate",
        "onset.positive_within_window_rate",
        "onset.clean_no_onset_rate",
        "onset.balanced_path_decision_accuracy",
    )
    h_summary = {
        cell: {
            path: _summarize_seed_metric(
                [h_reports[(cell, seed)] for seed in seeds], path
            )
            for path in h_metric_paths
        }
        for cell in cells
    }
    pairwise_summary = {
        cell: {
            "by_seed": [
                ranking_reports[(cell, seed)]["within_query_pairwise"]["accuracy"]
                for seed in seeds
            ],
            "mean": float(
                np.mean(
                    [
                        ranking_reports[(cell, seed)]["within_query_pairwise"][
                            "accuracy"
                        ]
                        for seed in seeds
                    ]
                )
            ),
            "comparisons": ranking_reports[(cell, 42)]["within_query_pairwise"][
                "comparisons"
            ],
        }
        for cell in cells
    }
    return {
        "schema_version": "clir-h0-v7.4-posthoc-exploratory-summary",
        "status": "COMPLETE_H0_V7_4_POSTHOC_EXPLORATORY_EVALUATION",
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "cells": list(cells),
        "seeds": list(seeds),
        "ranking_queries": len(reference_queries),
        "ranking": {
            "by_k": ranking_summary,
            "within_query_pairwise": pairwise_summary,
        },
        "h_dev": h_summary,
        "bootstrap": {
            "method": "paired hierarchical resampling of seeds and query_ids",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "claim_boundary": (
            "posthoc exploratory Silver evidence only; not Gold, confirmatory, "
            "protected-test, H1, Dual-Prior, or Full evidence"
        ),
    }


def _load_authorization(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "clir-h0-v7.4-training-authorization"
        or payload.get("status")
        != "AUTHORIZED_POSTHOC_EXPLORATORY_FOUR_CELL_TRAINING"
    ):
        raise ValueError("invalid H0 v7.4 training authorization")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen four-cell by three-seed H0 experiment."
    )
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    authorization_path = Path(args.authorization).resolve()
    authorization = _load_authorization(authorization_path)
    output_root = (
        Path(args.run_root).resolve()
        if args.run_root
        else (PROJECT_ROOT / authorization["runtime"]["output_root"]).resolve()
    )
    scored: dict[
        tuple[str, int], dict[str, Sequence[Mapping[str, Any]]]
    ] = {}
    input_files: dict[str, dict[str, str]] = {}
    for cell in authorization["cells"]:
        for seed in authorization["training"]["seeds"]:
            directory = output_root / f"scores/{cell}/seed-{seed}"
            h_path = directory / "h_dev.scored.jsonl"
            ranking_path = directory / "ranking.scored.jsonl"
            scored[(cell, int(seed))] = {
                "h_dev": read_jsonl(h_path),
                "ranking": read_jsonl(ranking_path),
            }
            input_files[f"{cell}/seed-{seed}"] = {
                "h_dev_sha256": file_sha256(h_path),
                "ranking_sha256": file_sha256(ranking_path),
            }
    evaluation = authorization["evaluation"]
    report = summarize(
        scored,
        k_values=[int(value) for value in evaluation["ranking_k"]],
        bootstrap_replicates=int(evaluation["bootstrap_replicates"]),
        bootstrap_seed=int(evaluation["bootstrap_seed"]),
        onset_threshold=float(evaluation["onset_threshold"]),
        onset_window_tokens=int(evaluation["onset_window_tokens"]),
    )
    report["authorization_file_sha256"] = file_sha256(authorization_path)
    report["input_files"] = input_files
    output = (
        Path(args.output_json).resolve()
        if args.output_json
        else output_root / "evaluation/summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
