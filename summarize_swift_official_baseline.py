#!/usr/bin/env python
"""Summarize the official-SWIFT baseline against the frozen U0 reference cell.

The U0 scores are read from the completed prior-ablation-v2 artifacts and are
never regenerated.  Statistical machinery is imported from the prior-ablation
summarizer so both reports use identical bootstrap and Holm implementations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from run_swift_official_baseline_training import CELL, load_protocol
from score_swift_official_baseline import MERGE_STATUS, _resolve
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from summarize_clir_prior_ablation_v2 import (
    _effect_summary,
    _holm,
    _load_run,
    _sample_sd,
    _sign_flip_pvalue,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/swift_official_baseline_v1/protocol.json"
REFERENCE_CELL = "u0"
PRIMARY_CONTRAST = "u0_minus_swift_official"
SOURCE_STRATA = ("gsm8k", "asdiv-a", "math")
BASE_SEED = 20260904
STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_SUMMARY"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def command_summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    root = _resolve(protocol["runtime"]["output_root"])
    target = root / "summary/final.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"summary exists: {target}")

    parents = protocol["frozen_parents"]
    feature_path = _resolve(parents["ranking_feature_manifest"]["path"])
    feature_sha = file_sha256(feature_path)
    if feature_sha != parents["ranking_feature_manifest"]["file_sha256"]:
        raise ValueError("reused ranking feature manifest hash drift")

    swift_merge_path = root / "ranking/scored/merge_report.json"
    swift_merge = _load_json(swift_merge_path)
    if swift_merge.get("status") != MERGE_STATUS:
        raise ValueError("SWIFT score merge is incomplete")
    if (
        swift_merge.get("protocol_file_sha256") != protocol_sha
        or swift_merge.get("input_jsonl_sha256") != feature_sha
    ):
        raise ValueError("SWIFT score merge binds another protocol or manifest")

    reference_merge_path = _resolve(parents["prior_ablation_v2_score_merge"]["path"])
    if file_sha256(reference_merge_path) != parents["prior_ablation_v2_score_merge"]["file_sha256"]:
        raise ValueError("prior-ablation-v2 score merge hash drift")
    reference_merge = _load_json(reference_merge_path)
    if reference_merge.get("input_jsonl_sha256") != feature_sha:
        raise ValueError("U0 scores were produced on a different ranking manifest")

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
    ranking = protocol["ranking_population"]
    if len(query_order) != int(ranking["total_queries"]):
        raise ValueError("summary query count drift")

    k_values = [int(value) for value in protocol["evaluation"]["k_values"]]
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    primary_k = int(protocol["evaluation"]["primary_k"])
    replicates = int(
        protocol["evaluation"]["uncertainty"]["paired_query_bootstrap_replicates"]
    )

    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for cell, merge in ((CELL, swift_merge), (REFERENCE_CELL, reference_merge)):
        for seed in seeds:
            key = f"{cell}/seed-{seed}"
            spec = merge["outputs"].get(key)
            if not isinstance(spec, Mapping):
                raise ValueError(f"score merge lacks {key}")
            expected = parents["u0_comparison_scores"].get(f"seed_{seed}")
            if cell == REFERENCE_CELL and spec["file_sha256"] != expected["file_sha256"]:
                raise ValueError(f"frozen U0 score hash drift: {key}")
            loaded[(cell, seed)] = _load_run(
                Path(spec["path"]), spec["file_sha256"], source, query_order, k_values
            )

    source_indices = {
        "all": np.arange(len(query_order)),
        **{
            name: np.asarray(
                [
                    index
                    for index, query_id in enumerate(query_order)
                    if by_query_source[query_id] == name
                ],
                dtype=np.int64,
            )
            for name in SOURCE_STRATA
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
    for cell in (CELL, REFERENCE_CELL):
        strata = {}
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

    sanity = protocol["evaluation"]["sanity_checks"]
    swift_k1 = cell_summary[CELL]["strata"]["all"]["by_k"]["1"]["mean"]
    if abs(swift_k1 - float(sanity["bon_at_k1_must_equal_the_frozen_candidate_index_zero_accuracy"])) > 1e-12:
        raise ValueError(f"BoN@1 sanity check failed: {swift_k1}")
    swift_primary = cell_summary[CELL]["strata"]["all"]["by_k"][str(primary_k)]["mean"]
    low, high = (float(value) for value in sanity["bon_at_k16_must_lie_between_random_expected_and_oracle"])
    if not low <= swift_primary <= high:
        raise ValueError(f"BoN@{primary_k} outside the frozen sanity band: {swift_primary}")

    by_k: dict[str, Any] = {}
    deltas_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        deltas = np.stack(
            [
                loaded[(REFERENCE_CELL, seed)]["selected"][k]
                - loaded[(CELL, seed)]["selected"][k]
                for seed in seeds
            ]
        )
        deltas_by_k[k] = deltas
        by_k[str(k)] = _effect_summary(
            deltas, with_interval=True, replicates=replicates, seed=BASE_SEED + k
        )
    source_primary = {
        source_name: _effect_summary(
            deltas_by_k[primary_k][:, source_indices[source_name]],
            with_interval=True,
            replicates=replicates,
            seed=BASE_SEED + 100 + offset,
        )
        for offset, source_name in enumerate(SOURCE_STRATA)
    }
    reference_values = np.stack(
        [loaded[(REFERENCE_CELL, seed)]["selected"][primary_k] for seed in seeds]
    )
    swift_values = np.stack([loaded[(CELL, seed)]["selected"][primary_k] for seed in seeds])
    transitions = {
        "swift_wrong_to_u0_correct": int(((swift_values == 0) & (reference_values == 1)).sum()),
        "swift_correct_to_u0_wrong": int(((swift_values == 1) & (reference_values == 0)).sum()),
        "unchanged": int((swift_values == reference_values).sum()),
        "denominator_seed_queries": int(swift_values.size),
    }

    raw_p = {PRIMARY_CONTRAST: _sign_flip_pvalue(deltas_by_k[primary_k], replicates, BASE_SEED + 900_000)}
    adjusted = _holm(raw_p)
    metric = by_k[str(primary_k)]
    metric["paired_sign_flip_p_value"] = raw_p[PRIMARY_CONTRAST]
    metric["holm_adjusted_p_value"] = adjusted[PRIMARY_CONTRAST]
    fixed = metric["fixed_seed_query_95_ci"]
    directions = metric["seed_direction_counts"]
    if (
        metric["mean_delta"] > 0
        and directions["positive"] >= 2
        and fixed[0] > 0
        and adjusted[PRIMARY_CONTRAST] < 0.05
    ):
        decision = "clir_structure_benefit"
    elif (
        metric["mean_delta"] < 0
        and directions["negative"] >= 2
        and fixed[1] < 0
        and adjusted[PRIMARY_CONTRAST] < 0.05
    ):
        decision = "clir_structure_harm"
    else:
        decision = "inconclusive"

    report = {
        "schema_version": "swift-official-baseline-v1-summary",
        "status": STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "upstream_commit": protocol["upstream"]["commit"],
        "feature_manifest_file_sha256": feature_sha,
        "swift_score_merge_file_sha256": file_sha256(swift_merge_path),
        "reference_score_merge_file_sha256": file_sha256(reference_merge_path),
        "population": {
            "queries": len(query_order),
            "candidates_per_query": int(ranking["candidates_per_query"]),
            "rows": len(source),
            "source_query_counts": {
                name: len(source_indices[name]) for name in SOURCE_STRATA
            },
            "reused_without_re_extraction": True,
            "already_inspected_by_the_nineteen_cell_ablation": True,
        },
        "model_comparison": {
            "swift_official": {
                "architecture": protocol["model"]["architecture"],
                "trainable_parameters": int(protocol["model"]["trainable_parameters"]),
                "retained_from_swift": protocol["model"]["retained_from_swift"],
                "absent_clir_components": protocol["model"][
                    "deliberately_absent_clir_components"
                ],
            },
            "u0_reference": {
                "trainable_parameters": int(
                    protocol["model"]["matched_budget_reference_trainable_parameters"]
                ),
                "config": parents["matched_budget_reference_config"]["path"],
            },
            "matched_training_budget": {
                "epochs": int(protocol["training"]["epochs"]),
                "batch_size": int(protocol["training"]["batch_size"]),
                "seeds": seeds,
                "training_rows": int(parents["training_manifest"]["rows"]),
                "training_manifest_file_sha256": parents["training_manifest"]["file_sha256"],
            },
            "declared_upstream_deviations": protocol["training"][
                "declared_upstream_deviations"
            ],
        },
        "baselines": baselines,
        "cells": cell_summary,
        "contrasts": {
            PRIMARY_CONTRAST: {
                "terms": {REFERENCE_CELL: 1, CELL: -1},
                "family": "primary",
                "interpretation": "positive means the full CLIR structure ranks better than plain SWIFT at the same training budget",
                "by_k": by_k,
                "source_strata_at_primary_k": source_primary,
                "selection_transitions_at_primary_k": transitions,
                "decision": decision,
            }
        },
        "primary_multiplicity": {
            "family": [PRIMARY_CONTRAST],
            "raw_p_values": raw_p,
            "holm_adjusted_p_values": adjusted,
            "holm_is_identity_for_a_single_hypothesis_family": True,
        },
        "claim_boundary": {
            "tier": protocol["evidence_boundary"]["tier"],
            "ranking_population_was_already_inspected": True,
            "no_post_result_tuning_was_performed": True,
            "not_a_replication_of_the_upstream_paper_numbers": True,
            "reader_caveat": protocol["evidence_boundary"]["reader_caveat"],
        },
    }
    atomic_write_json(target, report)
    print(
        json.dumps(
            {
                **report,
                "cells": f"{len(cell_summary)} cells",
                "baselines": f"{len(baselines)} strata",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
