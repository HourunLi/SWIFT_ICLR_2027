#!/usr/bin/env python
"""Validate and summarize the v16 checkpoints on the reused 892-query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, evaluate, file_sha256, group_rows
from src.clir_data import read_jsonl
from summarize_clir_ablation import paired_bootstrap_ci


PROJECT_ROOT = Path(__file__).resolve().parent
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_V16_POSTHOC_REUSED_892_RANKING_V1"
MERGE_STATUS = "PASS_PRIOR_V16_POSTHOC_REUSED_892_SCORING_MERGE"
SUMMARY_STATUS = "COMPLETE_PRIOR_V16_POSTHOC_REUSED_892_RANKING_V1"
REQUIRED_SCORE_FIELDS = {
    "source_row_index",
    "id",
    "query_id",
    "candidate_index",
    "correctness",
    "clir_checkpoint_sha256",
    "clir_scoring_mode",
    "clir_score",
    "clir_selected_best_of_n",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _candidate_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        (
            str(row["query_id"]),
            int(row["candidate_index"]),
            str(row["id"]),
            int(row["correctness"]),
        )
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_hashes(
    authorization: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[tuple[str, int], str]:
    output: dict[tuple[str, int], str] = {}
    for cell in authorization["cells"]:
        for seed in authorization["seeds"]:
            key = f"{cell}/seed-{seed}"
            try:
                checkpoint = summary["runs"][key]["checkpoint"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"training summary lacks {key}") from exc
            if (
                checkpoint.get("status") != "PASS_CHECKPOINT_AUDIT"
                or checkpoint.get("train_jsonl_sha256")
                != authorization["cell_train_sha256"][cell]
            ):
                raise ValueError(f"training checkpoint provenance drift: {key}")
            output[(str(cell), int(seed))] = str(checkpoint["file_sha256"])
    if len(output) != int(authorization["run_count"]):
        raise ValueError("checkpoint run-count drift")
    return output


def _selected_vectors(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> tuple[list[str], dict[int, np.ndarray], dict[int, np.ndarray]]:
    grouped = group_rows(rows)
    query_ids = sorted(grouped)
    labels = {k: [] for k in k_values}
    indices = {k: [] for k in k_values}
    for query_id in query_ids:
        candidates = grouped[query_id]
        for k in k_values:
            prefix = candidates[:k]
            best = max(range(k), key=lambda index: float(prefix[index]["clir_score"]))
            labels[k].append(float(prefix[best]["correctness"]))
            indices[k].append(int(prefix[best]["candidate_index"]))
        flags = [bool(row["clir_selected_best_of_n"]) for row in candidates]
        if sum(flags) != 1 or not flags[indices[max(k_values)][-1]]:
            raise ValueError(f"invalid global selection marker: {query_id}")
    return (
        query_ids,
        {k: np.asarray(values, dtype=np.float64) for k, values in labels.items()},
        {k: np.asarray(values, dtype=np.int64) for k, values in indices.items()},
    )


def _load_run(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_checkpoint_sha256: str,
    reference_rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    if file_sha256(path) != expected_file_sha256:
        raise ValueError(f"scored file hash drift: {path}")
    rows = read_jsonl(path)
    if len(rows) != len(reference_rows):
        raise ValueError(f"scored row-count drift: {path}")
    for index, (row, reference) in enumerate(zip(rows, reference_rows, strict=True)):
        if set(row) != REQUIRED_SCORE_FIELDS:
            raise ValueError(f"scored field drift at row {index}: {path}")
        if (
            int(row["source_row_index"]) != index
            or row["id"] != reference["id"]
            or row["query_id"] != reference["query_id"]
            or row["candidate_index"] != reference["candidate_index"]
            or row["correctness"] != reference["correctness"]
        ):
            raise ValueError(f"frozen candidate drift at row {index}: {path}")
        if (
            row["clir_checkpoint_sha256"] != expected_checkpoint_sha256
            or row["clir_scoring_mode"] != "scalar_only"
            or not math.isfinite(float(row["clir_score"]))
            or not isinstance(row["clir_selected_best_of_n"], bool)
        ):
            raise ValueError(f"invalid CLIR score provenance at row {index}: {path}")
    metrics = evaluate(
        rows,
        score_field="clir_score",
        correctness_field="correctness",
        k_values=k_values,
        bootstrap_replicates=0,
        seed=0,
    )
    query_ids, selected, selected_indices = _selected_vectors(rows, k_values)
    return {
        "query_ids": query_ids,
        "selected": selected,
        "selected_indices": selected_indices,
        "metrics": metrics,
        "file_sha256": expected_file_sha256,
    }


def summarize(
    authorization_path: Path,
    merge_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    authorization = _load_json(authorization_path)
    if (
        authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("evidence_tier")
        != "posthoc_exploratory_reused_not_fresh"
        or authorization.get("confirmation_scoring_allowed") is not False
    ):
        raise ValueError("inactive or malformed reused-ranking authorization")
    authorization_sha = file_sha256(authorization_path)
    ranking_path = _project_path(authorization["ranking_input_path"])
    training_summary_path = _project_path(authorization["training_completion_path"])
    if file_sha256(ranking_path) != authorization["ranking_input_sha256"]:
        raise ValueError("ranking input hash drift")
    if file_sha256(training_summary_path) != authorization["training_completion_sha256"]:
        raise ValueError("training summary hash drift")
    reference_rows = read_jsonl(ranking_path)
    if (
        len(reference_rows) != int(authorization["ranking_rows"])
        or len({str(row["query_id"]) for row in reference_rows})
        != int(authorization["ranking_queries"])
    ):
        raise ValueError("ranking population inventory drift")
    signature = _candidate_signature(reference_rows)

    merge = _load_json(merge_path)
    if (
        merge.get("status") != MERGE_STATUS
        or merge.get("authorization_file_sha256") != authorization_sha
        or merge.get("input_jsonl_sha256") != authorization["ranking_input_sha256"]
        or merge.get("completion_report_sha256")
        != authorization["training_completion_sha256"]
        or int(merge.get("num_shards", -1))
        != int(authorization["runtime"]["num_shards"])
    ):
        raise ValueError("score merge report violates the frozen authorization")

    training_summary = _load_json(training_summary_path)
    checkpoint_hashes = _checkpoint_hashes(authorization, training_summary)
    cells = [str(value) for value in authorization["cells"]]
    seeds = [int(value) for value in authorization["seeds"]]
    k_values = [int(value) for value in authorization["evaluation"]["k"]]
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    reference_queries: list[str] | None = None
    for cell in cells:
        for seed in seeds:
            key = f"{cell}/seed-{seed}"
            record = merge.get("outputs", {}).get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"score merge lacks {key}")
            run = _load_run(
                path=Path(str(record["path"])),
                expected_file_sha256=str(record["file_sha256"]),
                expected_checkpoint_sha256=checkpoint_hashes[(cell, seed)],
                reference_rows=reference_rows,
                k_values=k_values,
            )
            if reference_queries is None:
                reference_queries = run["query_ids"]
            elif run["query_ids"] != reference_queries:
                raise ValueError(f"query population drift at {key}")
            loaded[(cell, seed)] = run

    by_k: dict[str, Any] = {}
    for k in k_values:
        reference_metrics = loaded[(cells[0], seeds[0])]["metrics"]["by_k"][str(k)]
        cell_metrics: dict[str, Any] = {}
        for cell in cells:
            values = [
                float(loaded[(cell, seed)]["selected"][k].mean()) for seed in seeds
            ]
            cell_metrics[cell] = {
                "mean_accuracy": float(np.mean(values)),
                "sample_sd_across_seeds": _sample_sd(values),
                "by_seed": {
                    str(seed): value for seed, value in zip(seeds, values, strict=True)
                },
            }
            for seed in seeds:
                run_metrics = loaded[(cell, seed)]["metrics"]["by_k"][str(k)]
                if (
                    run_metrics["random_expected_accuracy"]
                    != reference_metrics["random_expected_accuracy"]
                    or run_metrics["oracle_accuracy"] != reference_metrics["oracle_accuracy"]
                ):
                    raise ValueError("random/oracle baselines drift across checkpoints")
        by_k[str(k)] = {
            "queries": len(reference_queries or []),
            "random_expected_accuracy": reference_metrics["random_expected_accuracy"],
            "oracle_accuracy": reference_metrics["oracle_accuracy"],
            "cells": cell_metrics,
        }

    contrasts: dict[str, Any] = {}
    contrast_specs = {
        "p0_minus_r0": ("r0", "p0"),
        "full_minus_ch": ("ch", "full"),
    }
    bootstrap_replicates = int(authorization["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(authorization["evaluation"]["bootstrap_seed"])
    for contrast_index, (name, (left, right)) in enumerate(contrast_specs.items()):
        contrast_by_k: dict[str, Any] = {}
        for k in k_values:
            left_vectors = [loaded[(left, seed)]["selected"][k] for seed in seeds]
            right_vectors = [loaded[(right, seed)]["selected"][k] for seed in seeds]
            deltas = np.stack(
                [
                    right_values - left_values
                    for left_values, right_values in zip(
                        left_vectors, right_vectors, strict=True
                    )
                ]
            )
            per_seed = deltas.mean(axis=1)
            transition_by_seed: dict[str, dict[str, int]] = {}
            transition_total = {
                "wrong_to_wrong": 0,
                "wrong_to_correct": 0,
                "correct_to_wrong": 0,
                "correct_to_correct": 0,
            }
            for seed, left_values, right_values in zip(
                seeds, left_vectors, right_vectors, strict=True
            ):
                transitions = {
                    "wrong_to_wrong": int(
                        np.sum((left_values == 0) & (right_values == 0))
                    ),
                    "wrong_to_correct": int(
                        np.sum((left_values == 0) & (right_values == 1))
                    ),
                    "correct_to_wrong": int(
                        np.sum((left_values == 1) & (right_values == 0))
                    ),
                    "correct_to_correct": int(
                        np.sum((left_values == 1) & (right_values == 1))
                    ),
                }
                transition_by_seed[str(seed)] = transitions
                for label, count in transitions.items():
                    transition_total[label] += count
            selection_changes = [
                float(
                    np.mean(
                        loaded[(right, seed)]["selected_indices"][k]
                        != loaded[(left, seed)]["selected_indices"][k]
                    )
                )
                for seed in seeds
            ]
            contrast_by_k[str(k)] = {
                "mean_paired_accuracy_delta": float(deltas.mean()),
                "sample_sd_across_seed_deltas": _sample_sd(per_seed.tolist()),
                "by_seed": {
                    str(seed): float(value)
                    for seed, value in zip(seeds, per_seed.tolist(), strict=True)
                },
                "seed_direction_counts": {
                    "positive": int((per_seed > 0).sum()),
                    "zero": int((per_seed == 0).sum()),
                    "negative": int((per_seed < 0).sum()),
                },
                "mean_selected_candidate_change_rate": float(
                    np.mean(selection_changes)
                ),
                "selected_candidate_change_rate_by_seed": {
                    str(seed): value
                    for seed, value in zip(seeds, selection_changes, strict=True)
                },
                "correctness_transition_counts": {
                    "aggregate_across_seeds": transition_total,
                    "by_seed": transition_by_seed,
                },
                **paired_bootstrap_ci(
                    deltas,
                    bootstrap_replicates,
                    bootstrap_seed + contrast_index * 10000 + k,
                ),
            }
        contrasts[name] = {
            "left": left,
            "right": right,
            "same_training_manifest": True,
            "by_k": contrast_by_k,
        }

    pairwise: dict[str, Any] = {}
    for cell in cells:
        values = [
            float(loaded[(cell, seed)]["metrics"]["within_query_pairwise"]["accuracy"])
            for seed in seeds
        ]
        pairwise[cell] = {
            "comparisons": int(
                loaded[(cell, seeds[0])]["metrics"]["within_query_pairwise"][
                    "comparisons"
                ]
            ),
            "mean_accuracy": float(np.mean(values)),
            "sample_sd_across_seeds": _sample_sd(values),
            "by_seed": {
                str(seed): value for seed, value in zip(seeds, values, strict=True)
            },
        }

    report = {
        "schema_version": "clir-prior-v16-posthoc-reused-ranking-summary-v1",
        "status": SUMMARY_STATUS,
        "evidence_tier": "posthoc_exploratory_reused_not_fresh",
        "authorization_file_sha256": authorization_sha,
        "merge_report_sha256": file_sha256(merge_path),
        "training_completion_sha256": authorization["training_completion_sha256"],
        "ranking_input_sha256": authorization["ranking_input_sha256"],
        "candidate_signature_sha256": signature,
        "ranking_rows": len(reference_rows),
        "ranking_queries": len(reference_queries or []),
        "candidates_per_query": int(authorization["candidates_per_query"]),
        "cells": cells,
        "seeds": seeds,
        "k_values": k_values,
        "ranking": {
            "by_k": by_k,
            "within_query_pairwise": pairwise,
            "matched_contrasts": contrasts,
        },
        "runs": {
            f"{cell}/seed-{seed}": {
                "checkpoint_sha256": checkpoint_hashes[(cell, seed)],
                "scored_jsonl_sha256": loaded[(cell, seed)]["file_sha256"],
            }
            for cell in cells
            for seed in seeds
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "fixed_seed_query_95_ci": "resample paired queries and average the three fixed seeds",
            "hierarchical_seed_query_95_ci": "exploratory resampling of both seeds and paired queries",
        },
        "interpretation_constraints": {
            "primary": "Full minus CH tests direct Prior plus the fixed 0.25 main-style Gate on top of C+H0",
            "secondary": "P0 minus R0 tests direct Prior without the Gate",
            "forbidden_cross_stage_contrast": "R0/P0 versus CH/Full because those pairs used different training manifests",
            "not_fresh": True,
            "not_protected": True,
            "no_human_verification": True,
            "no_retuning_after_opening": True,
        },
        "summary_implementation_sha256": file_sha256(__file__),
    }
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"summary already exists: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--merge-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    report = summarize(
        Path(args.authorization).resolve(),
        Path(args.merge_report).resolve(),
        output,
        overwrite=args.overwrite,
    )
    atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
