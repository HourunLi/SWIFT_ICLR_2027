"""Strict paired, multi-seed summary for matched CLIR ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, candidate_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired Best-of-N outcomes across CLIR cells and seeds."
    )
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--cells", required=True, help="Comma-separated cell names")
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds")
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="Repeat name:left_cell:right_cell; delta is right minus left",
    )
    parser.add_argument("--k", default="1,2,4,8,16")
    parser.add_argument("--scored_filename", default="validation_scored.jsonl")
    parser.add_argument("--metrics_filename", default="validation_metrics.json")
    parser.add_argument("--score_field", default="clir_score")
    parser.add_argument("--correctness_field", default="correctness")
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_contrasts(
    specifications: Sequence[str], cells: Sequence[str]
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    names: set[str] = set()
    cell_set = set(cells)
    for specification in specifications:
        parts = specification.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"contrast must be name:left_cell:right_cell, got {specification!r}"
            )
        name, left, right = parts
        if name in names:
            raise ValueError(f"Duplicate contrast name: {name}")
        if left not in cell_set or right not in cell_set:
            raise ValueError(
                f"Contrast {name} references cells outside --cells: {left}, {right}"
            )
        names.add(name)
        result.append((name, left, right))
    return result


def _load_run(
    scored_path: Path,
    metrics_path: Path,
    score_field: str,
    correctness_field: str,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Load only ranking fields while hashing and validating the large scored file."""
    grouped: dict[str, list[tuple[int, bool, str, float, float]]] = {}
    digest = hashlib.sha256()
    checkpoint_hashes: set[str] = set()
    row_count = 0
    with scored_path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            row = json.loads(raw_line)
            if "query_id" not in row or "id" not in row:
                raise ValueError(f"Row {row_count} in {scored_path} lacks query_id/id")
            if score_field not in row or correctness_field not in row:
                raise ValueError(
                    f"Row {row_count} in {scored_path} lacks "
                    f"{score_field}/{correctness_field}"
                )
            index, explicit = candidate_index(row, row_count)
            label = float(row[correctness_field])
            score = float(row[score_field])
            if label not in {0.0, 1.0} or not math.isfinite(score):
                raise ValueError(
                    f"Row {row_count} in {scored_path} has invalid label/score"
                )
            checkpoint_hash = row.get("clir_checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
                raise ValueError(
                    f"Row {row_count} in {scored_path} lacks checkpoint provenance"
                )
            checkpoint_hashes.add(checkpoint_hash)
            grouped.setdefault(str(row["query_id"]), []).append(
                (index, explicit, str(row["id"]), label, score)
            )
            row_count += 1
    if not grouped:
        raise ValueError(f"No rows in {scored_path}")
    if len(checkpoint_hashes) != 1:
        raise ValueError(
            f"Expected one checkpoint hash in {scored_path}, got {checkpoint_hashes}"
        )

    query_ids = sorted(grouped)
    signatures: list[tuple[str, tuple[tuple[int, str, float], ...]]] = []
    selected: dict[int, list[float]] = {k: [] for k in k_values}
    for query_id in query_ids:
        candidates = grouped[query_id]
        indices = [item[0] for item in candidates]
        explicit = [item[1] for item in candidates]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate candidate index for {query_id} in {scored_path}")
        if any(explicit) and not all(explicit):
            raise ValueError(f"Mixed explicit candidate indices for {query_id}")
        if all(explicit) and sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Non-contiguous candidate indices for {query_id}")
        candidates.sort(key=lambda item: item[0])
        signatures.append(
            (
                query_id,
                tuple((index, row_id, label) for index, _, row_id, label, _ in candidates),
            )
        )
        for k in k_values:
            if len(candidates) < k:
                raise ValueError(
                    f"Query {query_id} in {scored_path} has fewer than K={k} rows"
                )
            prefix = candidates[:k]
            # Python max is stable, matching evaluate_clir's frozen tie policy.
            best = max(range(k), key=lambda offset: prefix[offset][4])
            selected[k].append(prefix[best][3])

    scored_sha256 = digest.hexdigest()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("input_jsonl_sha256") != scored_sha256:
        raise ValueError(
            f"Metric/input hash mismatch for {metrics_path}: "
            f"{metrics.get('input_jsonl_sha256')} != {scored_sha256}"
        )
    if metrics.get("rows") != row_count or metrics.get("queries") != len(query_ids):
        raise ValueError(f"Metric row/query counts do not match {scored_path}")
    for k, outcomes in selected.items():
        reported = metrics.get("by_k", {}).get(str(k), {}).get("bon_accuracy")
        actual = float(np.mean(outcomes))
        if reported is None or not math.isclose(
            float(reported), actual, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Metric BoN@{k} does not match paired outcomes for {scored_path}"
            )

    return {
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "scored_jsonl_sha256": scored_sha256,
        "candidate_signature_sha256": _sha256_json(signatures),
        "rows": row_count,
        "query_ids": query_ids,
        "selected": {
            k: np.asarray(outcomes, dtype=np.float64) for k, outcomes in selected.items()
        },
        "within_query_pairwise_accuracy": float(
            metrics["within_query_pairwise"]["accuracy"]
        ),
    }


def paired_bootstrap_ci(
    deltas: np.ndarray, replicates: int, seed: int
) -> dict[str, list[float]]:
    """Bootstrap paired query deltas with fixed seeds and with seed resampling."""
    if deltas.ndim != 2 or deltas.shape[0] == 0 or deltas.shape[1] == 0:
        raise ValueError("deltas must have shape [seeds, queries]")
    if replicates <= 0:
        return {"fixed_seed_query_95_ci": [], "hierarchical_seed_query_95_ci": []}
    rng = np.random.default_rng(seed)
    seed_count, query_count = deltas.shape
    query_means = deltas.mean(axis=0)
    query_indices = rng.integers(0, query_count, size=(replicates, query_count))
    fixed_samples = query_means[query_indices].mean(axis=1)

    hierarchical_samples = np.empty(replicates, dtype=np.float64)
    batch_size = min(512, replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        count = stop - start
        seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
        sampled_queries = rng.integers(0, query_count, size=(count, query_count))
        values = deltas[
            seed_indices[:, :, None], sampled_queries[:, None, :]
        ]
        hierarchical_samples[start:stop] = values.mean(axis=(1, 2))

    return {
        "fixed_seed_query_95_ci": [
            float(np.quantile(fixed_samples, 0.025)),
            float(np.quantile(fixed_samples, 0.975)),
        ],
        "hierarchical_seed_query_95_ci": [
            float(np.quantile(hierarchical_samples, 0.025)),
            float(np.quantile(hierarchical_samples, 0.975)),
        ],
    }


def summarize(
    input_root: str | Path,
    cells: Sequence[str],
    seeds: Sequence[int],
    contrasts: Sequence[tuple[str, str, str]],
    k_values: Sequence[int],
    scored_filename: str = "validation_scored.jsonl",
    metrics_filename: str = "validation_metrics.json",
    score_field: str = "clir_score",
    correctness_field: str = "correctness",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("cells must be non-empty and unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    k_values = sorted(set(k_values))
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")

    root = Path(input_root)
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    reference_signature: str | None = None
    reference_queries: list[str] | None = None
    for cell in cells:
        for seed in seeds:
            run_dir = root / f"seed_{seed}" / cell
            run = _load_run(
                run_dir / scored_filename,
                run_dir / metrics_filename,
                score_field,
                correctness_field,
                k_values,
            )
            if reference_signature is None:
                reference_signature = run["candidate_signature_sha256"]
                reference_queries = run["query_ids"]
            elif (
                run["candidate_signature_sha256"] != reference_signature
                or run["query_ids"] != reference_queries
            ):
                raise ValueError(
                    f"Candidate population mismatch at cell={cell}, seed={seed}"
                )
            loaded[(cell, seed)] = run

    runs: dict[str, dict[str, Any]] = {}
    cell_summary: dict[str, Any] = {}
    for cell in cells:
        runs[cell] = {}
        pairwise_values: list[float] = []
        for seed in seeds:
            run = loaded[(cell, seed)]
            accuracy = {
                str(k): float(run["selected"][k].mean()) for k in k_values
            }
            pairwise = run["within_query_pairwise_accuracy"]
            pairwise_values.append(pairwise)
            runs[cell][str(seed)] = {
                "checkpoint_sha256": run["checkpoint_sha256"],
                "scored_jsonl_sha256": run["scored_jsonl_sha256"],
                "bon_accuracy_by_k": accuracy,
                "within_query_pairwise_accuracy": pairwise,
            }
        by_k: dict[str, Any] = {}
        for k in k_values:
            values = [float(loaded[(cell, seed)]["selected"][k].mean()) for seed in seeds]
            by_k[str(k)] = {
                "mean": float(np.mean(values)),
                "sample_sd_across_seeds": _sample_sd(values),
                "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
        cell_summary[cell] = {
            "by_k": by_k,
            "within_query_pairwise": {
                "mean": float(np.mean(pairwise_values)),
                "sample_sd_across_seeds": _sample_sd(pairwise_values),
                "per_seed": {
                    str(seed): value for seed, value in zip(seeds, pairwise_values)
                },
            },
        }

    contrast_summary: dict[str, Any] = {}
    for contrast_index, (name, left, right) in enumerate(contrasts):
        by_k = {}
        for k in k_values:
            deltas = np.stack(
                [
                    loaded[(right, seed)]["selected"][k]
                    - loaded[(left, seed)]["selected"][k]
                    for seed in seeds
                ]
            )
            per_seed = deltas.mean(axis=1)
            uncertainty = paired_bootstrap_ci(
                deltas,
                bootstrap_replicates,
                bootstrap_seed + contrast_index * 10_000 + k,
            )
            by_k[str(k)] = {
                "mean_delta": float(deltas.mean()),
                "sample_sd_across_seed_deltas": _sample_sd(per_seed.tolist()),
                "per_seed_delta": {
                    str(seed): float(delta)
                    for seed, delta in zip(seeds, per_seed.tolist())
                },
                "seed_direction_counts": {
                    "positive": int((per_seed > 0).sum()),
                    "zero": int((per_seed == 0).sum()),
                    "negative": int((per_seed < 0).sum()),
                },
                **uncertainty,
            }
        contrast_summary[name] = {"left": left, "right": right, "by_k": by_k}

    return {
        "schema_version": "clir-paired-ablation-summary-v1",
        "input_root": str(root),
        "cells": list(cells),
        "seeds": list(seeds),
        "k_values": k_values,
        "score_field": score_field,
        "correctness_field": correctness_field,
        "queries": len(reference_queries or []),
        "candidate_signature_sha256": reference_signature,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "paired query",
            "fixed_seed_query_95_ci": "resamples queries and averages fixed seeds",
            "hierarchical_seed_query_95_ci": (
                "resamples both seeds and paired queries; exploratory with few seeds"
            ),
        },
        "runs": runs,
        "cell_summary": cell_summary,
        "contrasts": contrast_summary,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    cells = [value.strip() for value in args.cells.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    contrasts = _parse_contrasts(args.contrast, cells)
    report = summarize(
        input_root=args.input_root,
        cells=cells,
        seeds=seeds,
        contrasts=contrasts,
        k_values=k_values,
        scored_filename=args.scored_filename,
        metrics_filename=args.metrics_filename,
        score_field=args.score_field,
        correctness_field=args.correctness_field,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
