#!/usr/bin/env python
"""Summarize the one-shot protected MATH-hard evaluation with paired intervals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from prepare_clir_math_hard_eval import load_protocol
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from summarize_clir_prior_ablation_v2 import (
    _bootstrap,
    _holm,
    _load_run,
    _sample_sd,
    _sign_flip_pvalue,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/math_hard_eval_v1/protocol.json"
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1"
PRIMARY_TERMS = {
    "c_minus_u0": {"c": 1, "u0": -1},
    "h_minus_u0": {"h": 1, "u0": -1},
    "ch_minus_u0": {"ch": 1, "u0": -1},
    "kc_minus_u0": {"kc": 1, "u0": -1},
    "kcg_minus_kc": {"kcg": 1, "kc": -1},
    "full_minus_ch": {"full": 1, "ch": -1},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _effect(deltas: np.ndarray, replicates: int, seed: int) -> dict[str, Any]:
    per_seed = deltas.mean(axis=1)
    result = {
        "mean_delta": float(deltas.mean()),
        "sample_sd_across_seed_deltas": _sample_sd(per_seed.tolist()),
        "per_seed_delta": {
            str(value): float(delta)
            for value, delta in zip((42, 43, 44), per_seed.tolist())
        },
        "seed_direction_counts": {
            "positive": int((per_seed > 0).sum()),
            "zero": int((per_seed == 0).sum()),
            "negative": int((per_seed < 0).sum()),
        },
    }
    result.update(_bootstrap(deltas, replicates, seed))
    return result


def _accuracy(values: list[float], seeds: list[int]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd_across_seeds": _sample_sd(values),
        "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
    }


def command_summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    target = root / "summary/final.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"protected summary exists: {target}")
    protocol = load_protocol(protocol_path)
    merge_path = root / "ranking/scored/merge_report.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if merge.get("status") != "PASS_MATH_HARD_EVAL_V1_SCORING_MERGE":
        raise ValueError("protected all-cell score merge is incomplete")
    feature_path = root / "features_v1/final/tuning_features.jsonl"
    if merge.get("input_jsonl_sha256") != file_sha256(feature_path):
        raise ValueError("protected score/feature hash drift")
    source = read_jsonl(feature_path)
    query_order: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[int]] = defaultdict(list)
    for row in source:
        query_id = str(row["query_id"])
        if query_id not in metadata:
            query_order.append(query_id)
            metadata[query_id] = {
                "level": int(row["source_level"]),
                "subject": str(row["source_subject"]),
            }
        labels[query_id].append(int(row["correctness"]))
    if len(query_order) != int(protocol["source"]["total_queries"]):
        raise ValueError("protected summary query count drift")

    prior_protocol = json.loads(
        (PROJECT_ROOT / protocol["frozen_models"]["prior_ablation_protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    cells = list(prior_protocol["cells"])
    seeds = [int(value) for value in prior_protocol["training"]["seeds"]]
    k_values = [int(value) for value in protocol["evaluation"]["k_values"]]
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        for seed in seeds:
            key = f"{cell}/seed-{seed}"
            spec = merge["outputs"].get(key)
            if not isinstance(spec, Mapping):
                raise ValueError(f"protected score merge lacks {key}")
            loaded[(cell, seed)] = _load_run(
                Path(spec["path"]), spec["file_sha256"], source, query_order, k_values
            )

    strata: dict[str, np.ndarray] = {
        "all": np.arange(len(query_order), dtype=np.int64),
        "level_4": np.asarray(
            [i for i, q in enumerate(query_order) if metadata[q]["level"] == 4],
            dtype=np.int64,
        ),
        "level_5": np.asarray(
            [i for i, q in enumerate(query_order) if metadata[q]["level"] == 5],
            dtype=np.int64,
        ),
    }
    for subject in protocol["source"]["subjects"]:
        strata[f"subject:{subject}"] = np.asarray(
            [i for i, q in enumerate(query_order) if metadata[q]["subject"] == subject],
            dtype=np.int64,
        )

    baselines: dict[str, Any] = {}
    for name, indices in strata.items():
        if not len(indices):
            continue
        baselines[name] = {
            "queries": len(indices),
            "by_k": {
                str(k): {
                    "random_expected_accuracy": float(
                        np.mean([np.mean(labels[query_order[i]][:k]) for i in indices])
                    ),
                    "oracle_accuracy": float(
                        np.mean([max(labels[query_order[i]][:k]) for i in indices])
                    ),
                }
                for k in k_values
            },
        }

    cell_summary: dict[str, Any] = {}
    primary_k = int(protocol["evaluation"]["primary_k"])
    for cell in cells:
        all_by_k = {}
        for k in k_values:
            values = [
                float(loaded[(cell, seed)]["selected"][k].mean()) for seed in seeds
            ]
            all_by_k[str(k)] = _accuracy(values, seeds)
        level_by_k = {}
        for name in ("level_4", "level_5"):
            indices = strata[name]
            level_by_k[name] = {
                str(k): _accuracy(
                    [
                        float(loaded[(cell, seed)]["selected"][k][indices].mean())
                        for seed in seeds
                    ],
                    seeds,
                )
                for k in k_values
            }
        subject_at_k = {}
        for name, indices in strata.items():
            if not name.startswith("subject:") or not len(indices):
                continue
            subject_at_k[name.removeprefix("subject:")] = {
                "queries": len(indices),
                **_accuracy(
                    [
                        float(
                            loaded[(cell, seed)]["selected"][primary_k][indices].mean()
                        )
                        for seed in seeds
                    ],
                    seeds,
                ),
            }
        pairwise = [loaded[(cell, seed)]["pairwise_accuracy"] for seed in seeds]
        cell_summary[cell] = {
            "all_by_k": all_by_k,
            "level_by_k": level_by_k,
            "subject_at_primary_k": subject_at_k,
            "within_query_correct_wrong_pairwise": _accuracy(pairwise, seeds),
            "pairwise_comparisons_per_seed": loaded[(cell, seeds[0])][
                "pairwise_comparisons"
            ],
        }

    primary = list(protocol["evaluation"]["primary_contrasts"])
    if primary != list(PRIMARY_TERMS):
        raise ValueError("protected primary contrast order differs from implementation")
    replicates = int(protocol["evaluation"]["paired_query_bootstrap_replicates"])
    contrasts: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for contrast_index, name in enumerate(primary):
        terms = PRIMARY_TERMS[name]
        by_k: dict[str, Any] = {}
        delta_by_k: dict[int, np.ndarray] = {}
        for k in k_values:
            deltas = np.stack(
                [
                    sum(
                        coefficient * loaded[(cell, seed)]["selected"][k]
                        for cell, coefficient in terms.items()
                    )
                    for seed in seeds
                ]
            )
            delta_by_k[k] = deltas
            by_k[str(k)] = _effect(
                deltas, replicates, 20260903 + contrast_index * 1000 + k
            )
        hard = delta_by_k[primary_k]
        stratum_effects = {
            stratum: _effect(
                hard[:, indices],
                replicates,
                30260903 + contrast_index * 1000 + offset,
            )
            for offset, (stratum, indices) in enumerate(strata.items())
            if len(indices)
        }
        left = next(cell for cell, value in terms.items() if value == -1)
        right = next(cell for cell, value in terms.items() if value == 1)
        left_values = np.stack(
            [loaded[(left, seed)]["selected"][primary_k] for seed in seeds]
        )
        right_values = np.stack(
            [loaded[(right, seed)]["selected"][primary_k] for seed in seeds]
        )
        contrasts[name] = {
            "terms": terms,
            "by_k": by_k,
            "strata_at_primary_k": stratum_effects,
            "selection_transitions_at_primary_k": {
                "wrong_to_correct": int(((left_values == 0) & (right_values == 1)).sum()),
                "correct_to_wrong": int(((left_values == 1) & (right_values == 0)).sum()),
                "unchanged": int((left_values == right_values).sum()),
                "denominator_seed_queries": int(left_values.size),
            },
        }
        raw_p[name] = _sign_flip_pvalue(
            hard, replicates, 40260903 + contrast_index
        )
    adjusted = _holm(raw_p)
    for name in primary:
        metric = contrasts[name]["by_k"][str(primary_k)]
        metric["paired_sign_flip_p_value"] = raw_p[name]
        metric["holm_adjusted_p_value"] = adjusted[name]
        fixed = metric["fixed_seed_query_95_ci"]
        hierarchical = metric["hierarchical_seed_query_95_ci"]
        directions = metric["seed_direction_counts"]
        if (
            metric["mean_delta"] > 0
            and directions["positive"] >= 2
            and fixed[0] > 0
            and hierarchical[0] > 0
            and adjusted[name] < 0.05
        ):
            decision = "benefit"
        elif (
            metric["mean_delta"] < 0
            and directions["negative"] >= 2
            and fixed[1] < 0
            and hierarchical[1] < 0
            and adjusted[name] < 0.05
        ):
            decision = "harm"
        else:
            decision = "inconclusive"
        metric["preregistered_evidence_decision"] = decision

    checker = json.loads((root / "checker/completion.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "clir-math-hard-eval-v1-final-summary",
        "status": "COMPLETE_MATH_HARD_EVAL_V1",
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "score_merge_file_sha256": file_sha256(merge_path),
        "feature_manifest_file_sha256": file_sha256(feature_path),
        "checker_completion_file_sha256": file_sha256(root / "checker/completion.json"),
        "population": {
            "queries": len(query_order),
            "rows": len(source),
            "candidates_per_query": int(protocol["generation"]["candidate_count"]),
            "levels": dict(sorted(Counter(metadata[q]["level"] for q in query_order).items())),
            "subjects": dict(
                sorted(Counter(metadata[q]["subject"] for q in query_order).items())
            ),
            "protected_official_math_test": True,
            "checker_health": checker["health"],
        },
        "baselines": baselines,
        "cells": cell_summary,
        "contrasts": contrasts,
        "primary_multiplicity": {
            "method": "Holm over two-sided paired-query sign-flip tests at K=16",
            "raw_p_values": raw_p,
            "adjusted_p_values": adjusted,
        },
        "claim_boundary": {
            "one_shot_protected_test": True,
            "no_weight_epoch_threshold_subset_or_checkpoint_selection": True,
            "hard_level_4_5_only": True,
            "published_SWIFT_numbers_directly_comparable": False,
            "silver_auxiliary_labels_human_verified": False,
        },
    }
    atomic_write_json(target, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "population": report["population"],
                "primary": {
                    name: contrasts[name]["by_k"][str(primary_k)] for name in primary
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
