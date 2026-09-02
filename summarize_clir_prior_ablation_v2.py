#!/usr/bin/env python
"""Summarize the frozen 19-cell Prior ablation with paired uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prepare_clir_prior_ablation_v2 import load_protocol
from src.clir_prior_ablation import CONTRAST_TERMS, contrast_vector
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(values), ddof=1)) if len(values) > 1 else None


def _bootstrap(
    deltas: np.ndarray, replicates: int, seed: int
) -> dict[str, list[float]]:
    if deltas.ndim != 2 or not deltas.shape[0] or not deltas.shape[1]:
        raise ValueError("bootstrap deltas must have shape [seeds, queries]")
    rng = np.random.default_rng(seed)
    seed_count, query_count = deltas.shape
    query_mean = deltas.mean(axis=0)
    fixed = np.empty(replicates, dtype=np.float64)
    hierarchical = np.empty(replicates, dtype=np.float64)
    batch = 256
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        count = stop - start
        queries = rng.integers(0, query_count, size=(count, query_count))
        fixed[start:stop] = query_mean[queries].mean(axis=1)
        seeds = rng.integers(0, seed_count, size=(count, seed_count))
        values = deltas[seeds[:, :, None], queries[:, None, :]]
        hierarchical[start:stop] = values.mean(axis=(1, 2))
    return {
        "fixed_seed_query_95_ci": [
            float(np.quantile(fixed, 0.025)),
            float(np.quantile(fixed, 0.975)),
        ],
        "hierarchical_seed_query_95_ci": [
            float(np.quantile(hierarchical, 0.025)),
            float(np.quantile(hierarchical, 0.975)),
        ],
    }


def _sign_flip_pvalue(deltas: np.ndarray, replicates: int, seed: int) -> float:
    values = deltas.mean(axis=0)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch = 256
    while completed < replicates:
        count = min(batch, replicates - completed)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        samples = np.abs((signs * values).mean(axis=1))
        exceed += int((samples >= observed - 1e-15).sum())
        completed += count
    return (exceed + 1.0) / (replicates + 1.0)


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda name: (raw[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * float(raw[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _load_run(
    path: Path,
    expected_hash: str,
    source: Sequence[Mapping[str, Any]],
    query_order: Sequence[str],
    k_values: Sequence[int],
) -> dict[str, Any]:
    if file_sha256(path) != expected_hash:
        raise ValueError(f"scored run hash drift: {path}")
    rows = read_jsonl(path)
    if len(rows) != len(source):
        raise ValueError(f"scored run row count drift: {path}")
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    pair_wins = 0.0
    pair_count = 0
    for index, (row, reference) in enumerate(zip(rows, source, strict=True)):
        for field in ("id", "query_id", "candidate_index", "correctness"):
            if row.get(field) != reference.get(field):
                raise ValueError(f"source identity drift at {path}:{index}:{field}")
        score = float(row["clir_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite CLIR score at {path}:{index}")
        grouped[str(row["query_id"])].append(
            (int(row["candidate_index"]), int(row["correctness"]), score)
        )
    selected = {k: [] for k in k_values}
    for query_id in query_order:
        candidates = sorted(grouped[query_id])
        if [value[0] for value in candidates] != list(range(len(candidates))):
            raise ValueError(f"candidate axis drift: {path}:{query_id}")
        correct = [value for value in candidates if value[1] == 1]
        wrong = [value for value in candidates if value[1] == 0]
        for left in correct:
            for right in wrong:
                pair_count += 1
                pair_wins += float(left[2] > right[2]) + 0.5 * float(left[2] == right[2])
        for k in k_values:
            prefix = candidates[:k]
            best = max(range(k), key=lambda offset: prefix[offset][2])
            selected[k].append(float(prefix[best][1]))
    return {
        "selected": {k: np.asarray(values, dtype=np.float64) for k, values in selected.items()},
        "pairwise_comparisons": pair_count,
        "pairwise_accuracy": pair_wins / pair_count if pair_count else float("nan"),
    }


def _effect_summary(
    deltas: np.ndarray,
    *,
    with_interval: bool,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    per_seed = deltas.mean(axis=1)
    result = {
        "mean_delta": float(deltas.mean()),
        "sample_sd_across_seed_deltas": _sample_sd(per_seed.tolist()),
        "per_seed_delta": {
            str(seed_value): float(value)
            for seed_value, value in zip((42, 43, 44), per_seed.tolist())
        },
        "seed_direction_counts": {
            "positive": int((per_seed > 0).sum()),
            "zero": int((per_seed == 0).sum()),
            "negative": int((per_seed < 0).sum()),
        },
    }
    if with_interval:
        result.update(_bootstrap(deltas, replicates, seed))
    return result


def command_summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    target = root / "summary/final.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists: {target}")
    protocol = load_protocol(protocol_path)
    merge_path = root / "ranking/scored/merge_report.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if merge.get("status") != "PASS_PRIOR_ABLATION_V2_SCORING_MERGE":
        raise ValueError("v2 score merge is incomplete")
    feature_path = root / "features_v2/final/tuning_features.jsonl"
    if merge.get("input_jsonl_sha256") != file_sha256(feature_path):
        raise ValueError("score merge/ranking feature hash drift")
    source = read_jsonl(feature_path)
    by_query_source: dict[str, str] = {}
    query_order: list[str] = []
    labels: dict[str, list[int]] = defaultdict(list)
    for row in source:
        query_id = str(row["query_id"])
        if query_id not in by_query_source:
            query_order.append(query_id)
            by_query_source[query_id] = str(row["source"])
        elif by_query_source[query_id] != row["source"]:
            raise ValueError("query source drift")
        labels[query_id].append(int(row["correctness"]))
    if len(query_order) != int(protocol["ranking_population"]["total_queries"]):
        raise ValueError("summary query count drift")
    k_values = [int(value) for value in protocol["evaluation"]["k_values"]]
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    cells = list(protocol["cells"])
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        for seed in seeds:
            key = f"{cell}/seed-{seed}"
            spec = merge["outputs"].get(key)
            if not isinstance(spec, Mapping):
                raise ValueError(f"score merge lacks {key}")
            loaded[(cell, seed)] = _load_run(
                Path(spec["path"]), spec["file_sha256"], source, query_order, k_values
            )

    source_indices = {
        "all": np.arange(len(query_order)),
        **{
            name: np.asarray(
                [index for index, query_id in enumerate(query_order) if by_query_source[query_id] == name],
                dtype=np.int64,
            )
            for name in ("gsm8k", "asdiv-a", "math")
        },
    }
    baselines: dict[str, Any] = {}
    for source_name, indices in source_indices.items():
        by_k = {}
        for k in k_values:
            random_values = [np.mean(labels[query_order[index]][:k]) for index in indices]
            oracle_values = [max(labels[query_order[index]][:k]) for index in indices]
            by_k[str(k)] = {
                "random_expected_accuracy": float(np.mean(random_values)),
                "oracle_accuracy": float(np.mean(oracle_values)),
            }
        baselines[source_name] = {"queries": len(indices), "by_k": by_k}

    cell_summary: dict[str, Any] = {}
    for cell in cells:
        strata = {}
        for source_name, indices in source_indices.items():
            by_k = {}
            for k in k_values:
                values = [float(loaded[(cell, seed)]["selected"][k][indices].mean()) for seed in seeds]
                by_k[str(k)] = {
                    "mean": float(np.mean(values)),
                    "sample_sd_across_seeds": _sample_sd(values),
                    "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
                }
            strata[source_name] = {"queries": len(indices), "by_k": by_k}
        pairwise = [loaded[(cell, seed)]["pairwise_accuracy"] for seed in seeds]
        cell_summary[cell] = {
            "strata": strata,
            "within_query_correct_wrong_pairwise": {
                "comparisons_per_seed": loaded[(cell, seeds[0])]["pairwise_comparisons"],
                "mean": float(np.mean(pairwise)),
                "sample_sd_across_seeds": _sample_sd(pairwise),
                "per_seed": {str(seed): value for seed, value in zip(seeds, pairwise)},
            },
        }

    primary = list(protocol["evaluation"]["primary_contrasts"])
    secondary = list(protocol["evaluation"]["secondary_contrasts"])
    if primary + secondary != list(CONTRAST_TERMS):
        raise ValueError("protocol contrast order differs from implementation")
    contrast_summary: dict[str, Any] = {}
    primary_p: dict[str, float] = {}
    primary_k = int(protocol["evaluation"]["primary_k"])
    replicates = int(protocol["evaluation"]["uncertainty"]["paired_query_bootstrap_replicates"])
    base_seed = 20260903
    for contrast_index, name in enumerate(primary + secondary):
        terms = CONTRAST_TERMS[name]
        by_k = {}
        deltas_by_k: dict[int, np.ndarray] = {}
        for k in k_values:
            deltas = np.stack(
                [
                    contrast_vector(
                        {cell: loaded[(cell, seed)]["selected"][k] for cell in terms},
                        name,
                    )
                    for seed in seeds
                ]
            )
            deltas_by_k[k] = deltas
            by_k[str(k)] = _effect_summary(
                deltas,
                with_interval=(name in primary or k == primary_k),
                replicates=replicates,
                seed=base_seed + contrast_index * 1000 + k,
            )
        source_primary = {}
        for source_offset, source_name in enumerate(("gsm8k", "asdiv-a", "math")):
            indices = source_indices[source_name]
            source_primary[source_name] = _effect_summary(
                deltas_by_k[primary_k][:, indices],
                with_interval=True,
                replicates=replicates,
                seed=base_seed + contrast_index * 1000 + 100 + source_offset,
            )
        transitions = None
        if len(terms) == 2 and set(terms.values()) == {-1, 1}:
            left = next(cell for cell, coefficient in terms.items() if coefficient == -1)
            right = next(cell for cell, coefficient in terms.items() if coefficient == 1)
            left_values = np.stack([loaded[(left, seed)]["selected"][primary_k] for seed in seeds])
            right_values = np.stack([loaded[(right, seed)]["selected"][primary_k] for seed in seeds])
            transitions = {
                "wrong_to_correct": int(((left_values == 0) & (right_values == 1)).sum()),
                "correct_to_wrong": int(((left_values == 1) & (right_values == 0)).sum()),
                "unchanged": int((left_values == right_values).sum()),
                "denominator_seed_queries": int(left_values.size),
            }
        contrast_summary[name] = {
            "terms": terms,
            "family": "primary" if name in primary else "secondary",
            "by_k": by_k,
            "source_strata_at_primary_k": source_primary,
            "selection_transitions_at_primary_k": transitions,
        }
        if name in primary:
            primary_p[name] = _sign_flip_pvalue(
                deltas_by_k[primary_k], replicates, base_seed + 900_000 + contrast_index
            )
    adjusted = _holm(primary_p)
    for name in primary:
        metric = contrast_summary[name]["by_k"][str(primary_k)]
        metric["paired_sign_flip_p_value"] = primary_p[name]
        metric["holm_adjusted_p_value"] = adjusted[name]
        fixed = metric["fixed_seed_query_95_ci"]
        directions = metric["seed_direction_counts"]
        if (
            metric["mean_delta"] > 0
            and directions["positive"] >= 2
            and fixed[0] > 0
            and adjusted[name] < 0.05
        ):
            decision = "benefit"
        elif (
            metric["mean_delta"] < 0
            and directions["negative"] >= 2
            and fixed[1] < 0
            and adjusted[name] < 0.05
        ):
            decision = "harm"
        else:
            decision = "inconclusive"
        metric["preregistered_evidence_decision"] = decision

    mechanism_path = root / "mechanisms/prior_dev.json"
    if not mechanism_path.is_file():
        raise FileNotFoundError("Prior mechanism report is missing")
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if mechanism.get("status") != "PASS_PRIOR_ABLATION_V2_PRIOR_MECHANISM_EVALUATION":
        raise ValueError("Prior mechanism evaluation is incomplete")
    report = {
        "schema_version": "clir-prior-ablation-v2-final-summary",
        "status": "COMPLETE_PRIOR_ABLATION_V2",
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_completion_file_sha256": file_sha256(root / "training/completion.json"),
        "feature_manifest_file_sha256": file_sha256(feature_path),
        "score_merge_file_sha256": file_sha256(merge_path),
        "mechanism_report_file_sha256": file_sha256(mechanism_path),
        "population": {
            "queries": len(query_order),
            "candidates_per_query": int(protocol["generation"]["candidate_count"]),
            "rows": len(source),
            "source_query_counts": {name: len(indices) for name, indices in source_indices.items() if name != "all"},
            "score_unseen_but_not_protected_test": True,
        },
        "baselines": baselines,
        "cells": cell_summary,
        "contrasts": contrast_summary,
        "primary_multiplicity": {
            "method": "Holm adjustment over two-sided paired query sign-flip tests at K=16",
            "raw_p_values": primary_p,
            "adjusted_p_values": adjusted,
        },
        "mechanisms": {
            "path": str(mechanism_path),
            "file_sha256": file_sha256(mechanism_path),
            "dev_is_already_inspected_and_descriptive_only": True,
        },
        "claim_boundary": {
            "prior_labels": "dual-AI Silver v16 post-hoc binary; no human verification",
            "correctness": "numeric value match, not full semantic correctness",
            "external_or_protected_test": False,
            "no_cell_weight_epoch_or_threshold_selected_from_this_population": True,
        },
    }
    atomic_write_json(target, report)
    print(json.dumps({**report, "cells": f"{len(cells)} cells", "contrasts": f"{len(contrast_summary)} contrasts"}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
