#!/usr/bin/env python
"""Summarize fixed-.25 P0/PG0 Prior mechanism and ranking diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, evaluate, file_sha256
from evaluate_clir_mechanisms import average_precision, binary_auroc
from prepare_clir_prior_gate import PROJECT_ROOT, _project_path, load_protocol
from src.clir_data import read_jsonl
from summarize_clir_ablation import paired_bootstrap_ci
from summarize_clir_prior_v12_posthoc import _selection


DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v12/posthoc_v1/gate_v1/protocol.json"
)
DEFAULT_RANKING_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v12/posthoc_v1/gate_v1/"
    "ranking_evaluation_authorization.json"
)

FULL_SCORE_FIELDS = {
    "clir_checkpoint_sha256",
    "clir_complete_prior",
    "clir_complete_prior_membership",
    "clir_condition_relevance",
    "clir_gate_attention",
    "clir_hallucination_prob",
    "clir_key_prior",
    "clir_key_prior_membership",
    "clir_mean_gate",
    "clir_path_hallucination_prob",
    "clir_path_no_hallucination_log_prob",
    "clir_prior_gate_alignment",
    "clir_prior_gate_squared_l2",
    "clir_pseudo_onset",
    "clir_score",
    "clir_selected_best_of_n",
    "clir_token_reward",
    "clir_token_value",
}
SCALAR_SCORE_FIELDS = {
    "clir_checkpoint_sha256",
    "clir_score",
    "clir_scoring_mode",
    "clir_selected_best_of_n",
}


def _sample_sd(values: Sequence[float]) -> float | None:
    return (
        float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
        if len(values) > 1
        else None
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _bce(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    target = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability)))


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if len(label_array) == 0 or len(label_array) != len(score_array):
        raise ValueError("binary metric arrays must be nonempty and aligned")
    if not np.isin(label_array, [0, 1]).all() or not np.isfinite(score_array).all():
        raise ValueError("binary labels/scores must be finite and binary")
    return {
        "examples": int(len(label_array)),
        "positives": int(label_array.sum()),
        "prevalence": float(label_array.mean()),
        "average_precision": average_precision(label_array, score_array),
        "auroc": binary_auroc(label_array, score_array),
    }


def _validate_rows(
    scored_path: Path,
    reference: Sequence[Mapping[str, Any]],
    expected_extra: set[str],
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    rows = read_jsonl(scored_path)
    if len(rows) != len(reference):
        raise ValueError(f"row-count drift in {scored_path}")
    for index, (source, row) in enumerate(zip(reference, rows)):
        if set(row) != set(source) | expected_extra:
            raise ValueError(f"field drift at row {index} in {scored_path}")
        if any(row[field] != value for field, value in source.items()):
            raise ValueError(f"frozen input drift at row {index} in {scored_path}")
        if row["clir_checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError(f"checkpoint drift at row {index} in {scored_path}")
        if not math.isfinite(float(row["clir_score"])):
            raise FloatingPointError(f"non-finite score at row {index}")
        if "clir_scoring_mode" in row:
            expected_mode = (
                "scalar_only" if expected_extra == SCALAR_SCORE_FIELDS else "full"
            )
            if row["clir_scoring_mode"] != expected_mode:
                raise ValueError(
                    f"unexpected scoring mode at row {index}: "
                    f"{row['clir_scoring_mode']} != {expected_mode}"
                )
    return rows


def _prior_run_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    key_labels: list[int] = []
    key_scores: list[float] = []
    complete_labels: list[int] = []
    complete_scores: list[float] = []
    correctness_labels: list[int] = []
    correctness_scores: list[float] = []
    gate_l2: list[float] = []
    gate_dot: list[float] = []
    gate_entropy: list[float] = []
    gate_normalized_entropy: list[float] = []
    gate_effective_fraction: list[float] = []
    raw_gate_mean: list[float] = []
    for index, row in enumerate(rows):
        length = len(row["output_token_ids"])
        fields = (
            "key_prior_target",
            "key_prior_mask",
            "complete_prior_target",
            "complete_prior_mask",
            "clir_key_prior_membership",
            "clir_complete_prior_membership",
            "clir_gate_attention",
        )
        if any(not isinstance(row.get(field), list) or len(row[field]) != length for field in fields):
            raise ValueError(f"unaligned Prior/Gate token fields at row {index}")
        key_mask = np.asarray(row["key_prior_mask"], dtype=bool)
        complete_mask = np.asarray(row["complete_prior_mask"], dtype=bool)
        key_labels.extend(np.asarray(row["key_prior_target"], dtype=int)[key_mask].tolist())
        key_scores.extend(np.asarray(row["clir_key_prior_membership"], dtype=float)[key_mask].tolist())
        complete_labels.extend(
            np.asarray(row["complete_prior_target"], dtype=int)[complete_mask].tolist()
        )
        complete_scores.extend(
            np.asarray(row["clir_complete_prior_membership"], dtype=float)[complete_mask].tolist()
        )
        correctness_labels.append(int(row["correctness"]))
        correctness_scores.append(float(row["clir_score"]))
        attention = np.asarray(row["clir_gate_attention"], dtype=np.float64)
        if (
            not np.isfinite(attention).all()
            or np.any(attention < 0)
            or not np.isclose(attention.sum(), 1.0, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError(f"invalid Gate attention at row {index}")
        positive = attention[attention > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        effective = float(1.0 / np.square(attention).sum())
        gate_entropy.append(entropy)
        gate_normalized_entropy.append(entropy / math.log(length) if length > 1 else 1.0)
        gate_effective_fraction.append(effective / length)
        gate_l2.append(float(row["clir_prior_gate_squared_l2"]))
        gate_dot.append(float(row["clir_prior_gate_alignment"]))
        raw_gate_mean.append(float(row["clir_mean_gate"]))
    key = _binary_metrics(key_labels, key_scores)
    key["binary_cross_entropy"] = _bce(key_labels, key_scores)
    complete = _binary_metrics(complete_labels, complete_scores)
    complete["binary_cross_entropy"] = _bce(complete_labels, complete_scores)
    correctness = _binary_metrics(correctness_labels, correctness_scores)
    correctness["binary_cross_entropy"] = _bce(
        correctness_labels, [_sigmoid(value) for value in correctness_scores]
    )
    return {
        "rows": len(rows),
        "key": key,
        "complete": complete,
        "correctness": correctness,
        "gate": {
            "full_trajectory_squared_l2_mean": float(np.mean(gate_l2)),
            "dot_product_mean": float(np.mean(gate_dot)),
            "raw_sigmoid_gate_mean": float(np.mean(raw_gate_mean)),
            "attention_entropy_mean": float(np.mean(gate_entropy)),
            "attention_normalized_entropy_mean": float(np.mean(gate_normalized_entropy)),
            "attention_effective_token_fraction_mean": float(np.mean(gate_effective_fraction)),
        },
    }


def summarize_dev(protocol_path: Path, output_json: Path | None) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_root = _project_path(protocol["runtime"]["output_root"])
    completion_path = output_root / "training/completion_report.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_PRIOR_V12_POSTHOC_PG0_THREE_RUNS":
        raise ValueError("PG0 training completion has not passed")
    pg0_hashes = {
        str(run["seed"]): run["checkpoint_file_sha256"] for run in completion["runs"]
    }
    p0_hashes = protocol["cells"]["p0"]["checkpoint_sha256_by_seed"]
    reference = read_jsonl(_project_path(protocol["frozen_inputs"]["prior_dev"]["path"]))
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    run_records = []
    for cell in ("p0", "pg0"):
        for seed in protocol["training"]["seeds"]:
            if cell == "p0":
                scored_path = _project_path(
                    protocol["cells"]["p0"]["dev_scored_by_seed"][str(seed)]
                )
                expected_scored_hash = protocol["cells"]["p0"][
                    "dev_scored_sha256_by_seed"
                ][str(seed)]
                if file_sha256(scored_path) != expected_scored_hash:
                    raise ValueError(f"frozen P0 dev score hash drift for seed {seed}")
                checkpoint = p0_hashes[str(seed)]
            else:
                scored_path = output_root / f"evaluation/dev_scored/pg0_seed-{seed}.jsonl"
                checkpoint = pg0_hashes[str(seed)]
            expected_fields = (
                FULL_SCORE_FIELDS
                if cell == "p0"
                else FULL_SCORE_FIELDS | {"clir_scoring_mode"}
            )
            rows = _validate_rows(
                scored_path, reference, expected_fields, checkpoint
            )
            metrics = _prior_run_metrics(rows)
            loaded[(cell, int(seed))] = metrics
            run_records.append(
                {
                    "cell": cell,
                    "seed": int(seed),
                    "scored_path": str(scored_path.relative_to(PROJECT_ROOT)),
                    "scored_file_sha256": file_sha256(scored_path),
                    "checkpoint_sha256": checkpoint,
                    "metrics": metrics,
                }
            )
    means: dict[str, Any] = {}
    for cell in ("p0", "pg0"):
        means[cell] = {
            target: {
                metric: float(np.mean([loaded[(cell, int(seed))][target][metric] for seed in protocol["training"]["seeds"]]))
                for metric in ("average_precision", "auroc", "binary_cross_entropy")
            }
            for target in ("key", "complete", "correctness")
        }
        means[cell]["gate"] = {
            metric: float(np.mean([loaded[(cell, int(seed))]["gate"][metric] for seed in protocol["training"]["seeds"]]))
            for metric in loaded[(cell, int(protocol["training"]["seeds"][0]))]["gate"]
        }
    l2_deltas = {
        str(seed): loaded[("pg0", int(seed))]["gate"]["full_trajectory_squared_l2_mean"]
        - loaded[("p0", int(seed))]["gate"]["full_trajectory_squared_l2_mean"]
        for seed in protocol["training"]["seeds"]
    }
    rules = protocol["decision_rules"]
    key_ap_drop = means["p0"]["key"]["average_precision"] - means["pg0"]["key"]["average_precision"]
    complete_ap_drop = means["p0"]["complete"]["average_precision"] - means["pg0"]["complete"]["average_precision"]
    protection_pass = (
        key_ap_drop <= float(rules["maximum_key_ap_drop_from_p0"])
        and complete_ap_drop <= float(rules["maximum_complete_ap_drop_from_p0"])
    )
    collapse_pass = (
        means["pg0"]["gate"]["attention_normalized_entropy_mean"]
        >= float(rules["minimum_mean_normalized_gate_entropy"])
        and means["pg0"]["gate"]["attention_effective_token_fraction_mean"]
        >= float(rules["minimum_mean_effective_token_fraction"])
    )
    alignment_learned = (
        means["pg0"]["gate"]["full_trajectory_squared_l2_mean"]
        < means["p0"]["gate"]["full_trajectory_squared_l2_mean"]
        and sum(value < 0 for value in l2_deltas.values()) >= 2
    )
    report = {
        "schema_version": "clir-prior-v12-posthoc-gate-dev-summary-v1",
        "status": "COMPLETE_PRIOR_V12_POSTHOC_FIXED_025_GATE_DEV_EVALUATION",
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_completion_file_sha256": file_sha256(completion_path),
        "runs": run_records,
        "three_seed_means": means,
        "pg0_minus_p0_gate_l2_by_seed": l2_deltas,
        "guards": {
            "key_ap_drop_from_p0": key_ap_drop,
            "complete_ap_drop_from_p0": complete_ap_drop,
            "prior_protection_pass": protection_pass,
            "gate_collapse_guard_pass": collapse_pass,
            "gate_alignment_learned": alignment_learned,
        },
        "ranking_scoring_allowed": protection_pass and collapse_pass,
        "claim_boundary": "posthoc exploratory Silver mechanism diagnostics; not label accuracy, Gold, or ranking efficacy",
    }
    target = output_json or output_root / "evaluation/dev_gate_summary.json"
    if target.exists():
        raise FileExistsError(f"dev summary exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def load_ranking_authorization(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "clir-prior-v12-posthoc-gate-ranking-authorization-v1"
        or payload.get("status") != "AUTHORIZED_FROZEN_EXPLORATORY_P0_PG0_RANKING"
        or payload.get("frozen_before_pg0_scored_outputs") is not True
    ):
        raise ValueError("unsupported or inactive Gate ranking authorization")
    return payload


def summarize_ranking(
    protocol_path: Path,
    authorization_path: Path,
    output_json: Path | None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    authorization = load_ranking_authorization(authorization_path)
    if authorization["frozen_inputs"]["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("Gate ranking authorization protocol drift")
    output_root = _project_path(protocol["runtime"]["output_root"])
    completion_path = output_root / "training/completion_report.json"
    dev_path = output_root / "evaluation/dev_gate_summary.json"
    for key, path in (
        ("training_completion_sha256", completion_path),
        ("dev_gate_summary_sha256", dev_path),
    ):
        if authorization["frozen_inputs"][key] != file_sha256(path):
            raise ValueError(f"Gate ranking authorization {key} drift")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if not dev.get("ranking_scoring_allowed"):
        raise ValueError("Prior protection/collapse guard did not allow ranking scoring")
    ranking_spec = authorization["frozen_inputs"]["ranking_manifest"]
    ranking_path = _project_path(ranking_spec["path"])
    if file_sha256(ranking_path) != ranking_spec["file_sha256"]:
        raise ValueError("ranking manifest hash drift")
    reference = read_jsonl(ranking_path)
    if (
        len(reference) != int(ranking_spec["row_count"])
        or len({str(row["query_id"]) for row in reference}) != int(ranking_spec["query_count"])
    ):
        raise ValueError("ranking population drift")
    seeds = [int(value) for value in authorization["grid"]["seeds"]]
    k_values = [int(value) for value in authorization["metrics"]["k"]]
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    query_ids: list[str] | None = None
    runs: dict[str, Any] = {}
    for cell in ("p0", "pg0"):
        for seed in seeds:
            checkpoint = authorization["frozen_inputs"]["checkpoints"][f"{cell}_seed_{seed}"]
            if cell == "p0":
                scored_path = _project_path(protocol["cells"]["p0"]["ranking_scored_by_seed"][str(seed)])
                expected_scored_hash = protocol["cells"]["p0"][
                    "ranking_scored_sha256_by_seed"
                ][str(seed)]
                if file_sha256(scored_path) != expected_scored_hash:
                    raise ValueError(
                        f"frozen P0 ranking score hash drift for seed {seed}"
                    )
            else:
                scored_path = output_root / f"evaluation/ranking_scored/pg0_seed-{seed}.jsonl"
            rows = _validate_rows(scored_path, reference, SCALAR_SCORE_FIELDS, checkpoint)
            metrics = evaluate(
                rows,
                score_field="clir_score",
                correctness_field="correctness",
                k_values=k_values,
                bootstrap_replicates=0,
                seed=0,
            )
            run_query_ids, selected, selected_indices = _selection(rows, k_values)
            if query_ids is None:
                query_ids = run_query_ids
            elif query_ids != run_query_ids:
                raise ValueError("ranking query identity/order differs across runs")
            loaded[(cell, seed)] = {
                "metrics": metrics,
                "selected": selected,
                "selected_indices": selected_indices,
            }
            runs[f"{cell}_seed_{seed}"] = {
                "checkpoint_sha256": checkpoint,
                "scored_jsonl_sha256": file_sha256(scored_path),
            }
    by_k: dict[str, Any] = {}
    bootstrap = authorization["metrics"]
    for k in k_values:
        cells: dict[str, Any] = {}
        for cell in ("p0", "pg0"):
            values = [float(loaded[(cell, seed)]["selected"][k].mean()) for seed in seeds]
            cells[cell] = {
                "mean_accuracy": float(np.mean(values)),
                "sample_sd_across_seeds": _sample_sd(values),
                "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
            }
        deltas = np.stack(
            [loaded[("pg0", seed)]["selected"][k] - loaded[("p0", seed)]["selected"][k] for seed in seeds]
        )
        per_seed = deltas.mean(axis=1)
        reference_metrics = loaded[("p0", seeds[0])]["metrics"]["by_k"][str(k)]
        by_k[str(k)] = {
            "queries": len(query_ids or []),
            "random_expected_accuracy": reference_metrics["random_expected_accuracy"],
            "oracle_accuracy": reference_metrics["oracle_accuracy"],
            "cells": cells,
            "pg0_minus_p0": {
                "mean_paired_delta": float(deltas.mean()),
                "sample_sd_across_seed_deltas": _sample_sd(per_seed.tolist()),
                "by_seed": {str(seed): float(value) for seed, value in zip(seeds, per_seed.tolist())},
                "seed_direction_counts": {
                    "positive": int((per_seed > 0).sum()),
                    "zero": int((per_seed == 0).sum()),
                    "negative": int((per_seed < 0).sum()),
                },
                **paired_bootstrap_ci(
                    deltas,
                    int(bootstrap["bootstrap_replicates"]),
                    int(bootstrap["bootstrap_seed"]) + k,
                ),
            },
        }
    pairwise = {}
    for cell in ("p0", "pg0"):
        values = [float(loaded[(cell, seed)]["metrics"]["within_query_pairwise"]["accuracy"]) for seed in seeds]
        pairwise[cell] = {
            "comparisons": loaded[(cell, seeds[0])]["metrics"]["within_query_pairwise"]["comparisons"],
            "mean_accuracy": float(np.mean(values)),
            "sample_sd_across_seeds": _sample_sd(values),
            "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
        }
    selection_changes: dict[str, Any] = {}
    for seed in seeds:
        p0_indices = loaded[("p0", seed)]["selected_indices"][16]
        pg0_indices = loaded[("pg0", seed)]["selected_indices"][16]
        p0_labels = loaded[("p0", seed)]["selected"][16]
        pg0_labels = loaded[("pg0", seed)]["selected"][16]
        changed = p0_indices != pg0_indices
        selection_changes[str(seed)] = {
            "queries": len(p0_indices),
            "changed_candidate": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
            "wrong_to_correct": int(((p0_labels == 0) & (pg0_labels == 1)).sum()),
            "correct_to_wrong": int(((p0_labels == 1) & (pg0_labels == 0)).sum()),
            "changed_same_correctness": int((changed & (p0_labels == pg0_labels)).sum()),
        }
    primary = by_k["16"]["pg0_minus_p0"]
    fixed_ci = primary["fixed_seed_query_95_ci"]
    benefit = (
        primary["mean_paired_delta"] > 0
        and primary["seed_direction_counts"]["positive"] >= 2
        and fixed_ci[0] > 0
    )
    rejected = (
        primary["mean_paired_delta"] < 0
        and primary["seed_direction_counts"]["negative"] >= 2
    )
    report = {
        "schema_version": "clir-prior-v12-posthoc-gate-ranking-summary-v1",
        "status": "COMPLETE_PRIOR_V12_POSTHOC_FIXED_025_P0_PG0_RANKING",
        "evidence_tier": protocol["evidence_tier"],
        "terminal_statuses_preserved": protocol["terminal_statuses_preserved"],
        "authorization_file_sha256": file_sha256(authorization_path),
        "ranking_manifest_sha256": ranking_spec["file_sha256"],
        "ranking_rows": len(reference),
        "ranking_queries": len(query_ids or []),
        "runs": runs,
        "ranking": {"by_k": by_k, "within_query_pairwise": pairwise},
        "selection_changes_at_k16": selection_changes,
        "decision": {
            "exploratory_fixed_025_ranking_benefit": benefit,
            "fixed_025_rejected_on_current_exploratory_screen": rejected,
            "gate_alignment_learned": dev["guards"]["gate_alignment_learned"],
            "prior_protection_pass": dev["guards"]["prior_protection_pass"],
            "three_module_full_requires_separate_frozen_stage": True,
            "no_tuning_on_same_dev_or_ranking_population": True,
        },
        "claim_boundary": "reused exploratory ranking only; not Gold, v12/v13 pass, fresh confirmatory, protected test, or three-module Full evidence",
    }
    target = output_json or output_root / "evaluation/ranking_summary.json"
    if target.exists():
        raise FileExistsError(f"ranking summary exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    dev_parser = subparsers.add_parser("summarize-dev")
    dev_parser.add_argument("--output-json", default=None)
    ranking_parser = subparsers.add_parser("summarize-ranking")
    ranking_parser.add_argument(
        "--authorization", default=str(DEFAULT_RANKING_AUTHORIZATION)
    )
    ranking_parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_path = Path(args.protocol).resolve()
    if args.command == "summarize-dev":
        report = summarize_dev(
            protocol_path,
            Path(args.output_json).resolve() if args.output_json else None,
        )
    elif args.command == "summarize-ranking":
        report = summarize_ranking(
            protocol_path,
            Path(args.authorization).resolve(),
            Path(args.output_json).resolve() if args.output_json else None,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
