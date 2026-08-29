#!/usr/bin/env python
"""Paired multi-seed summary for the frozen C0/C1 relation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json
from evaluate_clir_consistency import REPORT_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired C0/C1 held-out relation reports."
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--cells", default="c0,c1")
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=42061)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _load_report(path: Path, cell: str, seed: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"Unexpected report schema: {path}")
    if report.get("status") != "PASS_HELDOUT_RELATION_EVALUATION":
        raise ValueError(f"Held-out relation report did not pass: {path}")
    if report.get("cell") != cell or int(report.get("seed", -1)) != seed:
        raise ValueError(f"Cell/seed identity drift: {path}")
    return report


def _relation_index(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in report["relations"]:
        relation_id = str(row["relation_id"])
        if relation_id in result:
            raise ValueError(f"Duplicate relation_id in report: {relation_id}")
        result[relation_id] = row
    return result


def _paired_separation_bootstrap(
    positive_deltas: np.ndarray,
    negative_deltas: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    """Bootstrap positive and negative relation units separately.

    Arrays are [seed, relation].  Endpoint/query reuse means this interval is a
    relation-level descriptive interval, not an independence proof.
    """

    if positive_deltas.ndim != 2 or negative_deltas.ndim != 2:
        raise ValueError("Bootstrap inputs must be [seed, relation]")
    if positive_deltas.shape[0] != negative_deltas.shape[0]:
        raise ValueError("Positive/negative seed counts differ")
    if replicates <= 0:
        return {
            "fixed_seed_relation_95_ci": [],
            "hierarchical_seed_relation_95_ci": [],
        }
    rng = np.random.default_rng(seed)
    seed_count, positive_count = positive_deltas.shape
    negative_count = negative_deltas.shape[1]
    positive_mean = positive_deltas.mean(axis=0)
    negative_mean = negative_deltas.mean(axis=0)
    fixed = np.empty(replicates, dtype=np.float64)
    hierarchical = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        positive_indices = rng.integers(0, positive_count, size=positive_count)
        negative_indices = rng.integers(0, negative_count, size=negative_count)
        fixed[index] = (
            positive_mean[positive_indices].mean()
            - negative_mean[negative_indices].mean()
        )
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        hierarchical[index] = (
            positive_deltas[seed_indices][:, positive_indices].mean()
            - negative_deltas[seed_indices][:, negative_indices].mean()
        )
    return {
        "fixed_seed_relation_95_ci": [
            float(np.quantile(fixed, 0.025)),
            float(np.quantile(fixed, 0.975)),
        ],
        "hierarchical_seed_relation_95_ci": [
            float(np.quantile(hierarchical, 0.025)),
            float(np.quantile(hierarchical, 0.975)),
        ],
    }


def summarize(
    input_root: str | Path,
    seeds: Sequence[int],
    cells: Sequence[str] = ("c0", "c1"),
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42061,
) -> dict[str, Any]:
    if list(cells) != ["c0", "c1"]:
        raise ValueError("Frozen comparison requires cells c0,c1 in that order")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be non-empty and unique")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    root = Path(input_root)
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    reference_signature: str | None = None
    reference_inputs: dict[str, Any] | None = None
    positive_ids: list[str] | None = None
    negative_ids: list[str] | None = None
    for seed in seeds:
        for cell in cells:
            path = root / f"seed_{seed}" / cell / "heldout_relations.json"
            report = _load_report(path, cell, seed)
            signature = str(report["relation_signature_sha256"])
            input_identity = {
                name: report["inputs"][name]["file_sha256"]
                for name in (
                    "endpoint_manifest",
                    "positive_relations",
                    "negative_relations",
                    "expected_train_manifest",
                )
            }
            index = _relation_index(report)
            current_positive = sorted(
                relation_id
                for relation_id, row in index.items()
                if int(row["label"]) == 1
            )
            current_negative = sorted(
                relation_id
                for relation_id, row in index.items()
                if int(row["label"]) == 0
            )
            if reference_signature is None:
                reference_signature = signature
                reference_inputs = input_identity
                positive_ids = current_positive
                negative_ids = current_negative
            elif (
                signature != reference_signature
                or input_identity != reference_inputs
                or current_positive != positive_ids
                or current_negative != negative_ids
            ):
                raise ValueError(
                    f"Relation population drift at cell={cell}, seed={seed}"
                )
            reports[(cell, seed)] = report
    assert positive_ids is not None and negative_ids is not None

    cell_summary: dict[str, Any] = {}
    metric_paths = {
        "cosine_separation": (
            "representation",
            "mean_separation_positive_minus_negative",
        ),
        "cosine_auroc": ("representation", "relation_classification_auroc"),
        "cosine_average_precision": (
            "representation",
            "relation_classification_average_precision",
        ),
        "positive_cosine": ("representation", "positive_cosine", "mean"),
        "negative_cosine": ("representation", "negative_cosine", "mean"),
        "score_gap_separation": (
            "score",
            "mean_gap_separation_negative_minus_positive",
        ),
        "score_gap_auroc": (
            "score",
            "relation_classification_auroc_from_negative_gap",
        ),
    }

    def value_at(report: Mapping[str, Any], path: Sequence[str]) -> float:
        value: Any = report
        for component in path:
            value = value[component]
        return float(value)

    for cell in cells:
        metrics: dict[str, Any] = {}
        for name, path in metric_paths.items():
            values = [value_at(reports[(cell, seed)], path) for seed in seeds]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "sample_sd_across_seeds": _sample_sd(values),
                "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
        cell_summary[cell] = metrics

    positive_delta_rows: list[list[float]] = []
    negative_delta_rows: list[list[float]] = []
    positive_gap_delta_rows: list[list[float]] = []
    negative_gap_delta_rows: list[list[float]] = []
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        c0 = _relation_index(reports[("c0", seed)])
        c1 = _relation_index(reports[("c1", seed)])
        positive_delta = np.asarray(
            [
                float(c1[rid]["cosine_similarity"])
                - float(c0[rid]["cosine_similarity"])
                for rid in positive_ids
            ]
        )
        negative_delta = np.asarray(
            [
                float(c1[rid]["cosine_similarity"])
                - float(c0[rid]["cosine_similarity"])
                for rid in negative_ids
            ]
        )
        positive_gap_delta = np.asarray(
            [
                float(c1[rid]["absolute_score_gap"])
                - float(c0[rid]["absolute_score_gap"])
                for rid in positive_ids
            ]
        )
        negative_gap_delta = np.asarray(
            [
                float(c1[rid]["absolute_score_gap"])
                - float(c0[rid]["absolute_score_gap"])
                for rid in negative_ids
            ]
        )
        positive_delta_rows.append(positive_delta.tolist())
        negative_delta_rows.append(negative_delta.tolist())
        positive_gap_delta_rows.append(positive_gap_delta.tolist())
        negative_gap_delta_rows.append(negative_gap_delta.tolist())
        per_seed[str(seed)] = {
            "cosine_separation_delta": float(
                positive_delta.mean() - negative_delta.mean()
            ),
            "positive_cosine_delta": float(positive_delta.mean()),
            "negative_cosine_delta": float(negative_delta.mean()),
            "score_gap_separation_delta": float(
                negative_gap_delta.mean() - positive_gap_delta.mean()
            ),
            "cosine_auroc_delta": value_at(
                reports[("c1", seed)], metric_paths["cosine_auroc"]
            )
            - value_at(reports[("c0", seed)], metric_paths["cosine_auroc"]),
        }
    positive_deltas = np.asarray(positive_delta_rows)
    negative_deltas = np.asarray(negative_delta_rows)
    positive_gap_deltas = np.asarray(positive_gap_delta_rows)
    negative_gap_deltas = np.asarray(negative_gap_delta_rows)
    uncertainty = _paired_separation_bootstrap(
        positive_deltas,
        negative_deltas,
        bootstrap_replicates,
        bootstrap_seed,
    )
    score_uncertainty = _paired_separation_bootstrap(
        negative_gap_deltas,
        positive_gap_deltas,
        bootstrap_replicates,
        bootstrap_seed + 1,
    )
    seed_separation = np.asarray(
        [per_seed[str(seed)]["cosine_separation_delta"] for seed in seeds]
    )
    fixed_ci = uncertainty["fixed_seed_relation_95_ci"]
    if fixed_ci and bool((seed_separation > 0.0).all()) and fixed_ci[0] > 0.0:
        decision = "SUPPORTS_C1_HELDOUT_RELATION_SEPARATION"
    elif fixed_ci and bool((seed_separation < 0.0).all()) and fixed_ci[1] < 0.0:
        decision = "EVIDENCE_C1_REDUCES_HELDOUT_RELATION_SEPARATION"
    else:
        decision = "INCONCLUSIVE_C1_HELDOUT_RELATION_SEPARATION"
    return {
        "schema_version": "clir-consistency-c0-c1-paired-summary-v6.1",
        "status": "PASS_PAIRED_C0_C1_SUMMARY",
        "input_root": str(root.resolve()),
        "cells": list(cells),
        "seeds": list(seeds),
        "relation_signature_sha256": reference_signature,
        "input_file_hashes": reference_inputs,
        "relation_counts": {
            "positive": len(positive_ids),
            "negative": len(negative_ids),
        },
        "cell_summary": cell_summary,
        "contrast_c1_minus_c0": {
            "primary_metric": "positive_mean_cosine_minus_negative_mean_cosine",
            "mean_cosine_separation_delta": float(seed_separation.mean()),
            "sample_sd_across_seed_deltas": _sample_sd(seed_separation.tolist()),
            "per_seed": per_seed,
            **uncertainty,
            "mean_score_gap_separation_delta": float(
                (
                    negative_gap_deltas.mean(axis=1) - positive_gap_deltas.mean(axis=1)
                ).mean()
            ),
            "score_gap_separation_uncertainty": score_uncertainty,
            "decision": decision,
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "paired_relation_with_positive_and_negative_strata_resampled_separately",
            "dependency_warning": (
                "heldout_positive_and_negative_relations_reuse_some_endpoints_queries; "
                "intervals_are_descriptive_relation_level_uncertainty"
            ),
        },
        "claim_boundary": (
            "consistency_relation_mechanism_only_not_best_of_n_or_reward_ranking_efficacy"
        ),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cells = [value.strip() for value in args.cells.split(",") if value.strip()]
    report = summarize(
        args.input_root,
        seeds,
        cells,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    atomic_write_json(output, report)
    print(json.dumps(report["contrast_c1_minus_c0"], indent=2))


if __name__ == "__main__":
    main()
