"""Query-level Best-of-N evaluation for scored CLIR trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from src.clir_data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CLIR scored JSONL file.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--score_field", default="clir_score")
    parser.add_argument("--correctness_field", default="correctness")
    parser.add_argument("--k", default="1,2,4,8,16")
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_incomplete_queries",
        action="store_true",
        help="Opt out of the default common-query population across all K values.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_index(row: Mapping[str, Any], fallback: int) -> tuple[int, bool]:
    for key in ("candidate_index", "completion_index", "vllm_completion_output_index"):
        if key in row:
            value = row[key]
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer, got {value!r}")
            return int(value), True
    return fallback, False


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, list[Mapping[str, Any]]]:
    grouped: Dict[str, list[tuple[int, bool, Mapping[str, Any]]]] = {}
    for row_number, row in enumerate(rows):
        if "query_id" not in row:
            raise ValueError(f"Row {row_number} is missing query_id")
        index, explicit = candidate_index(row, row_number)
        grouped.setdefault(str(row["query_id"]), []).append((index, explicit, row))
    result: Dict[str, list[Mapping[str, Any]]] = {}
    for query_id, indexed in grouped.items():
        indices = [index for index, _, _ in indexed]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate candidate index for query {query_id}")
        explicit_flags = [explicit for _, explicit, _ in indexed]
        if any(explicit_flags) and not all(explicit_flags):
            raise ValueError(
                f"Mixed explicit/implicit candidate indices for query {query_id}"
            )
        if all(explicit_flags) and sorted(indices) != list(range(len(indices))):
            raise ValueError(
                f"Candidate indices for query {query_id} must be contiguous from 0"
            )
        indexed.sort(key=lambda item: item[0])
        result[query_id] = [row for _, _, row in indexed]
    return result


def bootstrap_mean_ci(
    values: Sequence[float], replicates: int, seed: int
) -> list[float]:
    if not values or replicates <= 0:
        return []
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, len(array)), replace=True).mean(
        axis=1
    )
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    score_field: str,
    correctness_field: str,
    k_values: Iterable[int],
    bootstrap_replicates: int,
    seed: int,
    allow_incomplete_queries: bool = False,
) -> Dict[str, Any]:
    grouped = group_rows(rows)
    if not grouped:
        raise ValueError("No scored rows to evaluate")
    unique_k = sorted(set(k_values))
    if not unique_k or any(k <= 0 for k in unique_k):
        raise ValueError("At least one positive k value is required")
    max_k = max(unique_k)
    incomplete = {
        query: len(candidates)
        for query, candidates in grouped.items()
        if len(candidates) < max_k
    }
    if incomplete and not allow_incomplete_queries:
        examples = ", ".join(
            f"{query}({count})" for query, count in list(incomplete.items())[:5]
        )
        raise ValueError(
            f"{len(incomplete)} queries have fewer than max K={max_k} candidates: "
            f"{examples}. Use --allow_incomplete_queries only for exploratory reports."
        )
    result: Dict[str, Any] = {
        "rows": len(rows),
        "queries": len(grouped),
        "score_field": score_field,
        "common_query_population": not allow_incomplete_queries,
        "max_k": max_k,
        "by_k": {},
    }
    all_pairwise: list[float] = []
    ties = 0
    comparisons = 0

    for candidates in grouped.values():
        for row in candidates:
            if score_field not in row or correctness_field not in row:
                raise ValueError(
                    f"Every row requires {score_field} and {correctness_field}"
                )
        correct = [float(row[correctness_field]) for row in candidates]
        scores = [float(row[score_field]) for row in candidates]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError(f"Query contains a non-finite {score_field}")
        if any(
            not math.isfinite(value) or value not in {0.0, 1.0} for value in correct
        ):
            raise ValueError(f"{correctness_field} must be finite binary 0/1")
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if correct[left] == correct[right]:
                    continue
                comparisons += 1
                correct_score = (
                    scores[left] if correct[left] > correct[right] else scores[right]
                )
                wrong_score = (
                    scores[right] if correct[left] > correct[right] else scores[left]
                )
                if correct_score == wrong_score:
                    ties += 1
                    all_pairwise.append(0.5)
                else:
                    all_pairwise.append(float(correct_score > wrong_score))

    for k in unique_k:
        selected: list[float] = []
        random_expected: list[float] = []
        oracle: list[float] = []
        eligible = (
            {
                query: candidates
                for query, candidates in grouped.items()
                if len(candidates) >= k
            }
            if allow_incomplete_queries
            else grouped
        )
        for candidates in eligible.values():
            prefix = candidates[:k]
            # max is stable: score ties choose the earliest frozen candidate.
            best = max(range(k), key=lambda index: float(prefix[index][score_field]))
            labels = [float(row[correctness_field]) for row in prefix]
            selected.append(labels[best])
            random_expected.append(float(np.mean(labels)))
            oracle.append(float(any(label > 0.5 for label in labels)))
        result["by_k"][str(k)] = {
            "queries": len(eligible),
            "bon_accuracy": float(np.mean(selected)) if selected else None,
            "bon_bootstrap_95_ci": bootstrap_mean_ci(
                selected, bootstrap_replicates, seed + k
            ),
            "random_expected_accuracy": (
                float(np.mean(random_expected)) if random_expected else None
            ),
            "oracle_accuracy": float(np.mean(oracle)) if oracle else None,
        }

    result["within_query_pairwise"] = {
        "comparisons": comparisons,
        "ties": ties,
        "accuracy": float(np.mean(all_pairwise)) if all_pairwise else None,
    }
    return result


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be non-negative")
    output_path = Path(args.output_json)
    if output_path.resolve() == Path(args.input_jsonl).resolve():
        raise ValueError("output_json must not overwrite input_jsonl")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")
    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    report = evaluate(
        read_jsonl(args.input_jsonl),
        score_field=args.score_field,
        correctness_field=args.correctness_field,
        k_values=k_values,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        allow_incomplete_queries=args.allow_incomplete_queries,
    )
    report["input_jsonl_sha256"] = file_sha256(args.input_jsonl)
    atomic_write_json(output_path, report)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
