#!/usr/bin/env python
"""Summarize the completed SWIFT/U0 architecture-by-sampler diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from run_swift_u0_sampler_factorial import (
    COMPLETION_STATUS,
    _resolve,
    load_protocol,
)
from score_swift_u0_sampler_factorial import MERGE_STATUS
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from summarize_clir_prior_ablation_v2 import (
    _effect_summary,
    _holm,
    _load_run,
    _sample_sd,
    _sign_flip_pvalue,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/swift_u0_sampler_factorial_v1/protocol.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/swift_u0_sampler_factorial_v1"
STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_SUMMARY"
BASE_SEED = 20260904
CELLS = ("swift_random", "swift_grouped", "u0_random", "u0_grouped")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _score_spec(
    protocol: Mapping[str, Any],
    new_merge: Mapping[str, Any],
    cell: str,
    seed: int,
) -> tuple[Path, str]:
    if protocol["factorial"]["cells"][cell]["source"] == "new_run":
        spec = new_merge["outputs"].get(f"{cell}/seed-{seed}")
        if not isinstance(spec, Mapping):
            raise ValueError(f"new score merge lacks {cell}/seed-{seed}")
        return Path(str(spec["path"])).resolve(), str(spec["file_sha256"])
    path, checksum = protocol["immutable_anchors"][cell]["scores"][str(seed)]
    return _resolve(path), str(checksum)


def _checkpoint_spec(
    protocol: Mapping[str, Any],
    completion: Mapping[str, Any],
    cell: str,
    seed: int,
) -> tuple[Path, str]:
    if protocol["factorial"]["cells"][cell]["source"] == "immutable_anchor":
        path, checksum = protocol["immutable_anchors"][cell]["checkpoints"][str(seed)]
        return _resolve(path), str(checksum)
    matches = [
        run
        for run in completion["runs"]
        if run["cell"] == cell and int(run["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"completion lacks one primary checkpoint for {cell}/{seed}")
    return Path(matches[0]["checkpoint_path"]), matches[0]["checkpoint_file_sha256"]


def _training_curve(path: Path, checksum: str, cell: str) -> list[float]:
    if file_sha256(path) != checksum:
        raise ValueError(f"checkpoint hash drift while reading training curve: {cell}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint["metrics"]
    if len(metrics) != 3:
        raise ValueError(f"expected three training epochs: {cell}")
    if cell.startswith("u0_"):
        return [float(record["train"]["final"]) for record in metrics]
    return [float(record["correctness_bce"]) for record in metrics]


def _linear_combination(
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    terms: Mapping[str, int],
    seed: int,
    k: int,
) -> np.ndarray:
    result: np.ndarray | None = None
    for cell, coefficient in terms.items():
        values = loaded[(cell, seed)]["selected"][k]
        contribution = float(coefficient) * values
        result = contribution.copy() if result is None else result + contribution
    if result is None:
        raise ValueError("contrast has no terms")
    return result


def _decision(metric: Mapping[str, Any], adjusted_p: float) -> str:
    fixed = metric["fixed_seed_query_95_ci"]
    directions = metric["seed_direction_counts"]
    if (
        float(metric["mean_delta"]) > 0.0
        and int(directions["positive"]) >= 2
        and float(fixed[0]) > 0.0
        and adjusted_p < 0.05
    ):
        return "positive_stable_on_inspected_diagnostic"
    if (
        float(metric["mean_delta"]) < 0.0
        and int(directions["negative"]) >= 2
        and float(fixed[1]) < 0.0
        and adjusted_p < 0.05
    ):
        return "negative_stable_on_inspected_diagnostic"
    return "inconclusive_on_inspected_diagnostic"


def _transition_summary(
    loaded: Mapping[tuple[str, int], Mapping[str, Any]],
    positive_cell: str,
    negative_cell: str,
    seeds: Sequence[int],
    k: int,
) -> dict[str, int]:
    positive = np.stack([loaded[(positive_cell, seed)]["selected"][k] for seed in seeds])
    negative = np.stack([loaded[(negative_cell, seed)]["selected"][k] for seed in seeds])
    return {
        f"{negative_cell}_wrong_to_{positive_cell}_correct": int(
            ((negative == 0) & (positive == 1)).sum()
        ),
        f"{negative_cell}_correct_to_{positive_cell}_wrong": int(
            ((negative == 1) & (positive == 0)).sum()
        ),
        "unchanged": int((negative == positive).sum()),
        "denominator_seed_queries": int(positive.size),
    }


def command_summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    root = Path(args.output_root).resolve()
    target = root / "summary/final.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists: {target}")

    completion_path = root / "training/completion.json"
    completion = _load_json(completion_path)
    if (
        completion.get("status") != COMPLETION_STATUS
        or completion.get("protocol_file_sha256") != protocol_sha
    ):
        raise ValueError("training completion is missing or stale")
    merge_path = root / "ranking/scored/merge_report.json"
    merge = _load_json(merge_path)
    feature_spec = protocol["frozen_parents"]["ranking_feature_manifest"]
    feature_path = _resolve(feature_spec["path"])
    feature_sha = file_sha256(feature_path)
    if (
        merge.get("status") != MERGE_STATUS
        or merge.get("protocol_file_sha256") != protocol_sha
        or merge.get("training_completion_file_sha256") != file_sha256(completion_path)
        or merge.get("ranking_input_file_sha256") != feature_sha
        or feature_sha != feature_spec["file_sha256"]
    ):
        raise ValueError("score merge or ranking-feature binding drift")

    source = read_jsonl(feature_path)
    query_order: list[str] = []
    query_source: dict[str, str] = {}
    labels: dict[str, list[int]] = defaultdict(list)
    for row in source:
        query_id = str(row["query_id"])
        if query_id not in query_source:
            query_order.append(query_id)
            query_source[query_id] = str(row["source"])
        elif query_source[query_id] != str(row["source"]):
            raise ValueError("query/source drift")
        labels[query_id].append(int(row["correctness"]))
    if len(query_order) != int(feature_spec["queries"]):
        raise ValueError("ranking query-count drift")

    seeds = [int(value) for value in protocol["training"]["seeds"]]
    k_values = [int(value) for value in protocol["evaluation"]["k_values"]]
    primary_k = int(protocol["evaluation"]["primary_k"])
    source_names = [str(value) for value in protocol["evaluation"]["source_strata"]]
    source_indices = {
        "all": np.arange(len(query_order), dtype=np.int64),
        **{
            source_name: np.asarray(
                [
                    index
                    for index, query_id in enumerate(query_order)
                    if query_source[query_id] == source_name
                ],
                dtype=np.int64,
            )
            for source_name in source_names
        },
    }

    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    training_curves: dict[str, Any] = {}
    for cell in CELLS:
        cell_curves: dict[str, list[float]] = {}
        for seed in seeds:
            score_path, score_hash = _score_spec(protocol, merge, cell, seed)
            loaded[(cell, seed)] = _load_run(
                score_path, score_hash, source, query_order, k_values
            )
            checkpoint_path, checkpoint_hash = _checkpoint_spec(
                protocol, completion, cell, seed
            )
            cell_curves[str(seed)] = _training_curve(
                checkpoint_path, checkpoint_hash, cell
            )
        training_curves[cell] = {
            "correctness_bce_per_seed_by_epoch": cell_curves,
            "mean_by_epoch": np.asarray(list(cell_curves.values())).mean(axis=0).tolist(),
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
        baselines[source_name] = {"queries": int(len(indices)), "by_k": by_k}

    cell_summary: dict[str, Any] = {}
    for cell in CELLS:
        strata: dict[str, Any] = {}
        for source_name, indices in source_indices.items():
            by_k = {}
            for k in k_values:
                values = [
                    float(loaded[(cell, seed)]["selected"][k][indices].mean())
                    for seed in seeds
                ]
                by_k[str(k)] = {
                    "mean": float(np.mean(values)),
                    "sample_sd_across_seeds": _sample_sd(values),
                    "per_seed": {
                        str(seed): value for seed, value in zip(seeds, values)
                    },
                }
            strata[source_name] = {"queries": int(len(indices)), "by_k": by_k}
        pairwise = [loaded[(cell, seed)]["pairwise_accuracy"] for seed in seeds]
        cell_summary[cell] = {
            "architecture": protocol["factorial"]["cells"][cell]["architecture"],
            "sampler": protocol["factorial"]["cells"][cell]["sampler"],
            "source": protocol["factorial"]["cells"][cell]["source"],
            "trainable_parameters": int(
                protocol["models"][protocol["factorial"]["cells"][cell]["architecture"]][
                    "trainable_parameters"
                ]
            ),
            "strata": strata,
            "within_query_correct_wrong_pairwise": {
                "comparisons_per_seed": loaded[(cell, seeds[0])]["pairwise_comparisons"],
                "mean": float(np.mean(pairwise)),
                "sample_sd_across_seeds": _sample_sd(pairwise),
                "per_seed": {
                    str(seed): value for seed, value in zip(seeds, pairwise)
                },
            },
        }
        if abs(strata["all"]["by_k"]["1"]["mean"] - 0.955) > 1e-12:
            raise ValueError(f"BoN@1 frozen-candidate sanity failed for {cell}")

    contrasts_spec = protocol["evaluation"]["contrasts"]
    replicates = int(protocol["evaluation"]["paired_query_bootstrap_replicates"])
    contrast_arrays: dict[str, dict[int, np.ndarray]] = {}
    contrast_summary: dict[str, Any] = {}
    for contrast_index, (name, terms) in enumerate(contrasts_spec.items()):
        by_k: dict[str, Any] = {}
        arrays: dict[int, np.ndarray] = {}
        for k in k_values:
            deltas = np.stack(
                [
                    _linear_combination(loaded, terms, seed, k)
                    for seed in seeds
                ]
            )
            arrays[k] = deltas
            by_k[str(k)] = _effect_summary(
                deltas,
                with_interval=True,
                replicates=replicates,
                seed=BASE_SEED + contrast_index * 1000 + k,
            )
        source_effects = {
            source_name: _effect_summary(
                arrays[primary_k][:, indices],
                with_interval=True,
                replicates=replicates,
                seed=BASE_SEED + 100_000 + contrast_index * 1000 + offset,
            )
            for offset, (source_name, indices) in enumerate(
                (item for item in source_indices.items() if item[0] != "all")
            )
        }
        contrast_arrays[name] = arrays
        contrast_summary[name] = {
            "terms": terms,
            "positive_means": "the positively weighted side has higher Best-of-N accuracy",
            "by_k": by_k,
            "source_strata_at_primary_k": source_effects,
        }

    raw_p = {
        name: _sign_flip_pvalue(
            contrast_arrays[name][primary_k],
            int(protocol["evaluation"]["paired_sign_flip_replicates"]),
            BASE_SEED + 900_000 + index,
        )
        for index, name in enumerate(protocol["evaluation"]["primary_multiplicity_family"])
    }
    adjusted = _holm(raw_p)
    for name in protocol["evaluation"]["primary_multiplicity_family"]:
        metric = contrast_summary[name]["by_k"][str(primary_k)]
        metric["paired_sign_flip_p_value"] = raw_p[name]
        metric["holm_adjusted_p_value"] = adjusted[name]
        contrast_summary[name]["decision"] = _decision(metric, adjusted[name])

    pair_contrasts = {
        "architecture_random": ("u0_random", "swift_random"),
        "architecture_grouped": ("u0_grouped", "swift_grouped"),
        "sampler_u0": ("u0_grouped", "u0_random"),
        "sampler_swift": ("swift_grouped", "swift_random"),
    }
    for name, (positive, negative) in pair_contrasts.items():
        contrast_summary[name]["correctness_transitions_at_primary_k"] = (
            _transition_summary(loaded, positive, negative, seeds, primary_k)
        )

    report = {
        "schema_version": "swift-u0-sampler-factorial-v1-summary",
        "status": STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "training_completion_file_sha256": file_sha256(completion_path),
        "score_merge_file_sha256": file_sha256(merge_path),
        "ranking_feature_manifest_file_sha256": feature_sha,
        "population": {
            "queries": len(query_order),
            "rows": len(source),
            "candidates_per_query": int(feature_spec["candidates_per_query"]),
            "source_queries": {
                name: int(len(source_indices[name])) for name in source_names
            },
            "already_inspected": True,
        },
        "sampler_inventory": _load_json(root / "training/preflight.json")[
            "sampler_audit"
        ],
        "training_curves": training_curves,
        "baselines": baselines,
        "cells": cell_summary,
        "contrasts": contrast_summary,
        "primary_multiplicity": {
            "family": protocol["evaluation"]["primary_multiplicity_family"],
            "raw_p_values": raw_p,
            "holm_adjusted_p_values": adjusted,
        },
        "claim_boundary": {
            "tier": protocol["evidence_boundary"]["tier"],
            "epoch_3_was_fixed_before_training": True,
            "epoch_1_and_2_were_not_ranking_scored_or_used_for_selection": True,
            "no_post_result_tuning": True,
            "not_fresh_not_protected_not_external_generalization": True,
            "hard_math_opened": False,
        },
    }
    atomic_write_json(target, report)
    print(
        json.dumps(
            {
                "status": STATUS,
                "population": report["population"],
                "bon16": {
                    cell: cell_summary[cell]["strata"]["all"]["by_k"]["16"]["mean"]
                    for cell in CELLS
                },
                "contrasts_at_16": {
                    name: {
                        "mean_delta": value["by_k"]["16"]["mean_delta"],
                        "fixed_ci": value["by_k"]["16"]["fixed_seed_query_95_ci"],
                        "decision": value["decision"],
                    }
                    for name, value in contrast_summary.items()
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
