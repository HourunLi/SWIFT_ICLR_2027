#!/usr/bin/env python
"""Joint protected summary for all CLIR cells and fair plain-SWIFT controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from score_clir_math_hard_baselines import (
    DEFAULT_ADDENDUM,
    MERGE_STATUS as BASELINE_MERGE_STATUS,
    _require_clean_branch,
    _resolve,
    load_addendum,
)
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from summarize_clir_math_hard_eval import PRIMARY_TERMS
from summarize_clir_prior_ablation_v2 import (
    _effect_summary,
    _holm,
    _load_run,
    _sample_sd,
    _sign_flip_pvalue,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1"
STATUS = "COMPLETE_MATH_HARD_EVAL_V2_WITH_FAIR_SWIFT_CONTROLS"
CLIR_MERGE_STATUS = "PASS_MATH_HARD_EVAL_V1_SCORING_MERGE"
FACTORIAL_CELLS = ("swift_random", "swift_grouped", "u0_random", "u0_grouped")
EXPECTED_FACTORIAL_TERMS = {
    "architecture_random": {"u0_random": 1, "swift_random": -1},
    "architecture_grouped": {"u0_grouped": 1, "swift_grouped": -1},
    "sampler_u0": {"u0_grouped": 1, "u0_random": -1},
    "sampler_swift": {"swift_grouped": 1, "swift_random": -1},
    "architecture_by_sampler_interaction": {
        "u0_grouped": 1,
        "u0_random": -1,
        "swift_grouped": -1,
        "swift_random": 1,
    },
}
BASE_SEED = 20260904


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _accuracy(values: Sequence[float], seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd_across_seeds": _sample_sd(values),
        "per_seed": {str(seed): float(value) for seed, value in zip(seeds, values)},
    }


def _linear_combination(
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    terms: Mapping[str, int],
    seed: int,
    k: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for cell, coefficient in terms.items():
        value = float(coefficient) * loaded[(cell, seed)]["selected"][k]
        result = value.copy() if result is None else result + value
    if result is None:
        raise ValueError("contrast cannot be empty")
    return result


def _decision(metric: Mapping[str, Any], adjusted_p: float) -> str:
    fixed = metric["fixed_seed_query_95_ci"]
    hierarchical = metric["hierarchical_seed_query_95_ci"]
    direction = metric["seed_direction_counts"]
    if (
        float(metric["mean_delta"]) > 0
        and int(direction["positive"]) >= 2
        and float(fixed[0]) > 0
        and float(hierarchical[0]) > 0
        and adjusted_p < 0.05
    ):
        return "benefit_on_locked_math_hard_population"
    if (
        float(metric["mean_delta"]) < 0
        and int(direction["negative"]) >= 2
        and float(fixed[1]) < 0
        and float(hierarchical[1]) < 0
        and adjusted_p < 0.05
    ):
        return "harm_on_locked_math_hard_population"
    return "inconclusive_on_locked_math_hard_population"


def _transition_summary(
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    positive: str,
    negative: str,
    seeds: Sequence[int],
    k: int,
) -> dict[str, int]:
    left = np.stack([loaded[(negative, seed)]["selected"][k] for seed in seeds])
    right = np.stack([loaded[(positive, seed)]["selected"][k] for seed in seeds])
    return {
        f"{negative}_wrong_to_{positive}_correct": int(
            ((left == 0) & (right == 1)).sum()
        ),
        f"{negative}_correct_to_{positive}_wrong": int(
            ((left == 1) & (right == 0)).sum()
        ),
        "unchanged": int((left == right).sum()),
        "denominator_seed_queries": int(left.size),
    }


def _summarize_cells(
    cells: Sequence[str],
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    strata: Mapping[str, np.ndarray],
    seeds: Sequence[int],
    k_values: Sequence[int],
    primary_k: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cell in cells:
        by_stratum: dict[str, Any] = {}
        for name, indices in strata.items():
            if not len(indices):
                continue
            by_stratum[name] = {
                "queries": int(len(indices)),
                "by_k": {
                    str(k): _accuracy(
                        [
                            float(
                                loaded[(cell, seed)]["selected"][k][indices].mean()
                            )
                            for seed in seeds
                        ],
                        seeds,
                    )
                    for k in k_values
                },
            }
        pairwise = [float(loaded[(cell, seed)]["pairwise_accuracy"]) for seed in seeds]
        result[cell] = {
            "strata": by_stratum,
            "within_query_correct_wrong_pairwise": {
                "comparisons_per_seed": int(
                    loaded[(cell, seeds[0])]["pairwise_comparisons"]
                ),
                **_accuracy(pairwise, seeds),
            },
            "primary_accuracy": by_stratum["all"]["by_k"][str(primary_k)],
        }
    return result


def _summarize_family(
    name: str,
    terms_by_name: Mapping[str, Mapping[str, int]],
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    strata: Mapping[str, np.ndarray],
    seeds: Sequence[int],
    k_values: Sequence[int],
    primary_k: int,
    replicates: int,
    base_seed: int,
) -> dict[str, Any]:
    arrays: dict[str, dict[int, np.ndarray]] = {}
    result: dict[str, Any] = {}
    for offset, (contrast, terms) in enumerate(terms_by_name.items()):
        by_k: dict[str, Any] = {}
        arrays[contrast] = {}
        for k in k_values:
            delta = np.stack(
                [_linear_combination(loaded, terms, seed, k) for seed in seeds]
            )
            arrays[contrast][k] = delta
            by_k[str(k)] = _effect_summary(
                delta,
                with_interval=True,
                replicates=replicates,
                seed=base_seed + offset * 1000 + k,
            )
        result[contrast] = {
            "terms": dict(terms),
            "by_k": by_k,
            "strata_at_primary_k": {
                stratum: _effect_summary(
                    arrays[contrast][primary_k][:, indices],
                    with_interval=True,
                    replicates=replicates,
                    seed=base_seed + 100_000 + offset * 1000 + index,
                )
                for index, (stratum, indices) in enumerate(strata.items())
                if len(indices)
            },
        }

    raw = {
        contrast: _sign_flip_pvalue(
            arrays[contrast][primary_k],
            replicates,
            base_seed + 200_000 + offset,
        )
        for offset, contrast in enumerate(terms_by_name)
    }
    adjusted = _holm(raw)
    for contrast in terms_by_name:
        metric = result[contrast]["by_k"][str(primary_k)]
        metric["paired_sign_flip_p_value"] = raw[contrast]
        metric["holm_adjusted_p_value"] = adjusted[contrast]
        metric["preregistered_evidence_decision"] = _decision(
            metric, adjusted[contrast]
        )
    return {
        "family": name,
        "contrasts": result,
        "multiplicity": {
            "method": "Holm within this predeclared family at primary K",
            "raw_p_values": raw,
            "adjusted_p_values": adjusted,
        },
    }


def command_summarize(args: argparse.Namespace) -> None:
    addendum_path = Path(args.addendum).resolve()
    addendum = load_addendum(addendum_path, verify_files=True)
    runtime_code = _require_clean_branch(addendum)
    root = Path(args.output_root).resolve()
    if root != _resolve(addendum["runtime"]["math_hard_output_root"]):
        raise ValueError("joint MATH-hard summary output-root drift")
    target = root / "summary/final_v2.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"protected joint summary exists: {target}")

    base_protocol_path = _resolve(
        addendum["frozen_parent"]["math_hard_protocol"]["path"]
    )
    base_protocol = _load_json(base_protocol_path)
    prior_protocol_path = _resolve(
        base_protocol["frozen_models"]["prior_ablation_protocol"]["path"]
    )
    prior_protocol = _load_json(prior_protocol_path)
    clir_cells = [str(value) for value in prior_protocol["cells"]]
    seeds = [int(value) for value in prior_protocol["training"]["seeds"]]
    k_values = [int(value) for value in base_protocol["evaluation"]["k_values"]]
    primary_k = int(base_protocol["evaluation"]["primary_k"])
    if seeds != [42, 43, 44] or len(clir_cells) != 19:
        raise ValueError("protected CLIR grid drift")

    feature_path = root / "features_v1/final/tuning_features.jsonl"
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
    if len(source) != 8000 or len(query_order) != 500:
        raise ValueError("protected joint summary population drift")

    clir_merge_path = root / "ranking/scored/merge_report.json"
    baseline_merge_path = _resolve(addendum["runtime"]["baseline_merge_report"])
    clir_merge = _load_json(clir_merge_path)
    baseline_merge = _load_json(baseline_merge_path)
    if (
        clir_merge.get("status") != CLIR_MERGE_STATUS
        or clir_merge.get("input_jsonl_sha256") != file_sha256(feature_path)
        or baseline_merge.get("status") != BASELINE_MERGE_STATUS
        or baseline_merge.get("addendum_file_sha256") != file_sha256(addendum_path)
        or baseline_merge.get("ranking_input_file_sha256") != file_sha256(feature_path)
    ):
        raise ValueError("protected score-merge binding drift")

    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in clir_cells:
        for seed in seeds:
            spec = clir_merge["outputs"].get(f"{cell}/seed-{seed}")
            if not isinstance(spec, Mapping):
                raise ValueError(f"CLIR merge lacks {cell}/seed-{seed}")
            loaded[(cell, seed)] = _load_run(
                Path(str(spec["path"])),
                str(spec["file_sha256"]),
                source,
                query_order,
                k_values,
            )
    for cell in ("u0_random", "swift_random", "swift_grouped"):
        for seed in seeds:
            spec = baseline_merge["outputs"].get(f"{cell}/seed-{seed}")
            if not isinstance(spec, Mapping):
                raise ValueError(f"baseline merge lacks {cell}/seed-{seed}")
            loaded[(cell, seed)] = _load_run(
                Path(str(spec["path"])),
                str(spec["file_sha256"]),
                source,
                query_order,
                k_values,
            )
    for seed in seeds:
        loaded[("u0_grouped", seed)] = loaded[("u0", seed)]

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
    for subject in base_protocol["source"]["subjects"]:
        strata[f"subject:{subject}"] = np.asarray(
            [i for i, q in enumerate(query_order) if metadata[q]["subject"] == subject],
            dtype=np.int64,
        )

    baselines = {
        name: {
            "queries": int(len(indices)),
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
        for name, indices in strata.items()
        if len(indices)
    }
    cell_summary = _summarize_cells(
        [*clir_cells, "u0_random", "swift_random", "swift_grouped"],
        loaded,
        strata,
        seeds,
        k_values,
        primary_k,
    )

    factorial_terms = addendum["evaluation"]["sampler_factorial_contrasts"]
    if factorial_terms != EXPECTED_FACTORIAL_TERMS:
        raise ValueError("protected sampler-factorial contrast drift")
    original_terms = {
        name: PRIMARY_TERMS[name]
        for name in base_protocol["evaluation"]["primary_contrasts"]
    }
    all_clir_terms = {
        f"{cell}_minus_swift_grouped": {cell: 1, "swift_grouped": -1}
        for cell in clir_cells
    }
    replicates = int(base_protocol["evaluation"]["paired_query_bootstrap_replicates"])
    families = {
        "original_clir_primary": _summarize_family(
            "original_clir_primary",
            original_terms,
            loaded,
            strata,
            seeds,
            k_values,
            primary_k,
            replicates,
            BASE_SEED,
        ),
        "architecture_by_sampler": _summarize_family(
            "architecture_by_sampler",
            factorial_terms,
            loaded,
            strata,
            seeds,
            k_values,
            primary_k,
            replicates,
            BASE_SEED + 1_000_000,
        ),
        "all_grouped_clir_vs_grouped_swift": _summarize_family(
            "all_grouped_clir_vs_grouped_swift",
            all_clir_terms,
            loaded,
            strata,
            seeds,
            k_values,
            primary_k,
            replicates,
            BASE_SEED + 2_000_000,
        ),
    }
    pairwise_contrasts = {
        "architecture_random": ("u0_random", "swift_random"),
        "architecture_grouped": ("u0_grouped", "swift_grouped"),
        "sampler_u0": ("u0_grouped", "u0_random"),
        "sampler_swift": ("swift_grouped", "swift_random"),
    }
    factor_records = families["architecture_by_sampler"]["contrasts"]
    for name, (positive, negative) in pairwise_contrasts.items():
        factor_records[name]["selection_transitions_at_primary_k"] = (
            _transition_summary(loaded, positive, negative, seeds, primary_k)
        )

    checker_path = root / "checker/completion.json"
    checker = _load_json(checker_path)
    report = {
        "schema_version": "clir-math-hard-eval-v2-joint-summary",
        "status": STATUS,
        "created_at_utc": _utc_now(),
        "code": runtime_code,
        "addendum_file_sha256": file_sha256(addendum_path),
        "base_protocol_file_sha256": file_sha256(base_protocol_path),
        "feature_manifest_file_sha256": file_sha256(feature_path),
        "clir_score_merge_file_sha256": file_sha256(clir_merge_path),
        "baseline_score_merge_file_sha256": file_sha256(baseline_merge_path),
        "checker_completion_file_sha256": file_sha256(checker_path),
        "population": {
            "queries": len(query_order),
            "rows": len(source),
            "candidates_per_query": 16,
            "levels": dict(
                sorted(Counter(metadata[q]["level"] for q in query_order).items())
            ),
            "subjects": dict(
                sorted(Counter(metadata[q]["subject"] for q in query_order).items())
            ),
            "protected_official_math_test": True,
            "checker_health": checker["health"],
        },
        "model_inventory": {
            "clir_checkpoints": 57,
            "additional_fair_control_checkpoints": 9,
            "total_checkpoints": 66,
            "clir_cells": clir_cells,
            "factorial_cells": list(FACTORIAL_CELLS),
            "seeds": seeds,
            "epoch": 3,
        },
        "baselines": baselines,
        "cells": cell_summary,
        "contrast_families": families,
        "claim_boundary": {
            "questions_accessed_only_for_deterministic_selection_before_addendum": True,
            "rollouts_correctness_and_scores_unopened_when_model_set_was_locked": True,
            "single_one_shot_protected_evaluation": True,
            "all_19_clir_cells_use_semantic_group_sampler": True,
            "fair_all_cell_reference_is_swift_grouped": True,
            "swift_random_comparison_to_u0_grouped_is_joint_architecture_plus_sampler_only": True,
            "published_swift_numbers_directly_comparable": False,
            "silver_auxiliary_labels_human_verified": False,
            "no_post_result_tuning_or_checkpoint_selection": True,
        },
    }
    atomic_write_json(target, report)
    print(
        json.dumps(
            {
                "status": STATUS,
                "population": report["population"],
                "bon16": {
                    cell: cell_summary[cell]["primary_accuracy"]["mean"]
                    for cell in [*clir_cells, "u0_random", "swift_random", "swift_grouped"]
                },
                "architecture_factorial_at_16": {
                    name: record["by_k"][str(primary_k)]
                    for name, record in factor_records.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", default=str(DEFAULT_ADDENDUM))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
