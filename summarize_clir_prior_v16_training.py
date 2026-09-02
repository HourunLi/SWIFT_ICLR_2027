#!/usr/bin/env python
"""Audit and summarize the frozen Prior-v16 post-hoc staged experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_clir import atomic_write_json, file_sha256
from evaluate_clir_three_module_factorial import h_metrics, prior_metrics
from src.clir_data import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_training_v1/protocol.json"
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
    "clir_scoring_mode",
    "clir_selected_best_of_n",
    "clir_token_reward",
    "clir_token_value",
}
TOKEN_SCORE_FIELDS = {
    "clir_complete_prior",
    "clir_complete_prior_membership",
    "clir_condition_relevance",
    "clir_gate_attention",
    "clir_hallucination_prob",
    "clir_key_prior",
    "clir_key_prior_membership",
    "clir_token_reward",
    "clir_token_value",
}
STAGE_1_METRICS = {
    "key_average_precision": "prior.key.average_precision",
    "key_auroc": "prior.key.auroc",
    "key_bce": "prior.key.binary_cross_entropy",
    "complete_average_precision": "prior.complete.average_precision",
    "complete_auroc": "prior.complete.auroc",
    "complete_bce": "prior.complete.binary_cross_entropy",
    "correctness_average_precision": "prior.correctness.average_precision",
    "correctness_auroc": "prior.correctness.auroc",
    "correctness_bce": "prior.correctness.binary_cross_entropy",
}
STAGE_2_METRICS = {
    **STAGE_1_METRICS,
    "gate_advantage_over_uniform_l2": (
        "prior.gate.learned_gate_advantage_over_uniform_l2_mean"
    ),
    "gate_alignment": "prior.gate.dot_product_mean",
    "gate_normalized_entropy": "prior.gate.attention_normalized_entropy_mean",
    "h_token_average_precision": "hallucination.token.average_precision",
    "h_token_auroc": "hallucination.token.auroc",
    "h_token_bce": "hallucination.token.binary_cross_entropy",
    "h_position_baseline_auroc": "hallucination.token.position_baseline.auroc",
    "h_path_average_precision": "hallucination.path.average_precision",
    "h_path_auroc": "hallucination.path.auroc",
    "h_path_bce": "hallucination.path.binary_cross_entropy",
    "h_balanced_path_accuracy": (
        "hallucination.onset.balanced_path_decision_accuracy"
    ),
    "h_positive_within_5_tokens": (
        "hallucination.onset.positive_within_window_rate"
    ),
    "consistency_representation_separation": (
        "consistency.representation.mean_separation_positive_minus_negative"
    ),
    "consistency_representation_auroc": (
        "consistency.representation.relation_classification_auroc"
    ),
    "consistency_representation_average_precision": (
        "consistency.representation.relation_classification_average_precision"
    ),
    "consistency_score_gap_separation": (
        "consistency.score.mean_gap_separation_negative_minus_positive"
    ),
}


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": head, "branch": branch, "dirty": dirty}


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-prior-v16-posthoc-training-protocol-v1"
        or protocol.get("status")
        != "AUTHORIZED_STAGED_R0_P0_CH_FULL_TRAINING"
    ):
        raise ValueError("unsupported Prior-v16 staged protocol")
    boundary = protocol["evidence_boundary"]
    if (
        boundary.get("tier")
        != "posthoc_exploratory_dual_ai_silver_no_human_verification"
        or boundary.get("original_v16_status")
        != "STOP_PRIOR_V16_ROLE_ONLY_SCALE"
        or boundary.get("original_v17_status")
        != "STOP_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE"
        or boundary.get("original_terminal_statuses_are_unchanged") is not True
        or boundary.get("fresh_query_cluster_ranking_confirmation_required_later")
        is not True
    ):
        raise ValueError("Prior-v16 evidence boundary drift")
    training = protocol["training"]
    if (
        training.get("stage_1_cells") != ["r0", "p0"]
        or training.get("stage_2_cells") != ["ch", "full"]
        or training.get("seeds") != [42, 43, 44]
        or int(training.get("epochs", -1)) != 3
        or training.get("fixed_epoch_no_checkpoint_selection") is not True
        or training.get("adaptive_epoch_weight_or_subset_change") is not False
    ):
        raise ValueError("Prior-v16 training grid drift")
    if not str(protocol["evaluation"]["ranking"]).startswith(
        "deferred_until_fresh"
    ):
        raise ValueError("ranking must remain deferred")
    return protocol


def _nested(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(path)
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite metric: {path}")
    return result


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def aggregate_contrast(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    control: str,
    treatment: str,
    seeds: Sequence[int],
    metrics: Mapping[str, str],
) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    for name, path in metrics.items():
        values = {
            cell: [_nested(runs[(cell, seed)], path) for seed in seeds]
            for cell in (control, treatment)
        }
        deltas = [right - left for left, right in zip(values[control], values[treatment])]
        by_metric[name] = {
            "cells": {
                cell: {
                    "mean": float(np.mean(cell_values)),
                    "sample_sd_across_seeds": _sample_sd(cell_values),
                    "by_seed": {
                        str(seed): value for seed, value in zip(seeds, cell_values)
                    },
                }
                for cell, cell_values in values.items()
            },
            f"{treatment}_minus_{control}": {
                "mean_paired_delta": float(np.mean(deltas)),
                "sample_sd_across_seed_deltas": _sample_sd(deltas),
                "by_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
                "seed_direction_counts": {
                    "positive": int(sum(value > 0 for value in deltas)),
                    "zero": int(sum(value == 0 for value in deltas)),
                    "negative": int(sum(value < 0 for value in deltas)),
                },
            },
        }
    return {"metrics": by_metric}


def evaluate_stage_1_gate(
    stage_1: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = stage_1["metrics"]

    def mean(metric: str, cell: str) -> float:
        return float(metrics[metric]["cells"][cell]["mean"])

    def delta(metric: str) -> float:
        return float(metrics[metric]["p0_minus_r0"]["mean_paired_delta"])

    checks = {
        "p0_mean_key_auroc": mean("key_auroc", "p0")
        >= float(gate["p0_mean_key_auroc_min"]),
        "p0_mean_complete_auroc": mean("complete_auroc", "p0")
        >= float(gate["p0_mean_complete_auroc_min"]),
        "key_average_precision_delta": delta("key_average_precision")
        >= float(gate["p0_minus_r0_key_average_precision_min"]),
        "complete_average_precision_delta": delta("complete_average_precision")
        >= float(gate["p0_minus_r0_complete_average_precision_min"]),
        "key_bce_lower": mean("key_bce", "p0") < mean("key_bce", "r0"),
        "complete_bce_lower": mean("complete_bce", "p0")
        < mean("complete_bce", "r0"),
        "correctness_auroc_delta": delta("correctness_auroc")
        >= float(gate["p0_minus_r0_correctness_auroc_min"]),
    }
    return {
        "status": (
            "PASS_PRIOR_V16_POSTHOC_STAGE_1_GATE"
            if all(checks.values())
            else "FAIL_PRIOR_V16_POSTHOC_STAGE_1_GATE"
        ),
        "all_checks_pass": all(checks.values()),
        "checks": checks,
    }


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _checkpoint_audit(
    path: Path,
    *,
    cell: str,
    seed: int,
    protocol: Mapping[str, Any],
    train_path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = protocol["training"]
    if int(checkpoint.get("completed_epoch", -1)) != int(training["epochs"]):
        raise ValueError(f"{cell}/{seed} epoch drift")
    contract = checkpoint.get("training_contract", {})
    expected_contract = {
        "seed": seed,
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "amp_dtype": str(training["amp_dtype"]),
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"{cell}/{seed} training contract drift: {key}")
    manifest_name = str(protocol["cells"][cell]["train_manifest"])
    data = checkpoint.get("data_state", {})
    expected_rows = int(
        protocol["data_contract"][
            "direct_train_rows" if manifest_name == "direct" else "combined_train_rows"
        ]
    )
    expected_queries = int(
        protocol["data_contract"][
            "direct_train_queries"
            if manifest_name == "direct"
            else "combined_train_queries"
        ]
    )
    if (
        data.get("train_sha256") != file_sha256(train_path)
        or int(data.get("train_rows", -1)) != expected_rows
        or int(data.get("train_queries", -1)) != expected_queries
    ):
        raise ValueError(f"{cell}/{seed} training data drift")
    config_path = _project_path(protocol["cells"][cell]["config"])
    if file_sha256(config_path) != protocol["cells"][cell]["file_sha256"]:
        raise ValueError(f"{cell} config hash drift")
    provenance = checkpoint.get("run_provenance", {})
    if (
        provenance.get("config", {}).get("sha256")
        != protocol["cells"][cell]["file_sha256"]
        or provenance.get("code", {}).get("dirty") is not False
        or provenance.get("code", {}).get("branch") != "clir-clean-integration"
    ):
        raise ValueError(f"{cell}/{seed} run provenance drift")
    expected_weights = {
        "consistency_weight": float(protocol["cells"][cell]["factors"][0]),
        "hallucination_weight": float(protocol["cells"][cell]["factors"][1]),
        "prior_weight": float(protocol["cells"][cell]["factors"][2]),
        "gate_prior_weight": float(protocol["cells"][cell]["gate_prior_weight"]),
    }
    model_config = checkpoint.get("model_config", {})
    if any(model_config.get(key) != value for key, value in expected_weights.items()):
        raise ValueError(f"{cell}/{seed} factor identity drift")
    for disabled in (
        "token_reward_weight",
        "tail_weight",
        "mil_weight",
        "pseudo_tail_weight",
        "prior_distill_weight",
        "progress_weight",
        "reconstruction_weight",
    ):
        if float(model_config.get(disabled, -1.0)) != 0.0:
            raise ValueError(f"{cell}/{seed} unexpectedly enabled {disabled}")
    bad_tensors = [
        name
        for name, tensor in checkpoint.get("state_dict", {}).items()
        if not torch.isfinite(tensor).all()
    ]
    metrics = checkpoint.get("metrics", [])
    if (
        bad_tensors
        or len(metrics) != int(training["epochs"])
        or [int(row.get("epoch", -1)) for row in metrics] != [1, 2, 3]
        or not _all_numeric_finite(metrics)
    ):
        raise FloatingPointError(f"{cell}/{seed} checkpoint is incomplete or non-finite")
    return {
        "status": "PASS_CHECKPOINT_AUDIT",
        "path": str(path),
        "file_sha256": file_sha256(path),
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "training_commit": provenance["code"]["commit"],
        "config_sha256": provenance["config"]["sha256"],
        "train_jsonl_sha256": data["train_sha256"],
        "train_rows": int(data["train_rows"]),
        "train_queries": int(data["train_queries"]),
        "final_train_metrics": metrics[-1]["train"],
    }


def _validate_scored_rows(
    reference_path: Path, scored_path: Path, checkpoint_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = read_jsonl(reference_path)
    scored = read_jsonl(scored_path)
    if len(reference) != len(scored):
        raise ValueError(f"score row-count drift: {scored_path}")
    for index, (source, row) in enumerate(zip(reference, scored)):
        if set(row) != set(source) | FULL_SCORE_FIELDS:
            raise ValueError(f"score field drift at row {index}: {scored_path}")
        if any(row[key] != value for key, value in source.items()):
            raise ValueError(f"frozen score input drift at row {index}: {scored_path}")
        if (
            row["clir_checkpoint_sha256"] != checkpoint_sha256
            or row["clir_scoring_mode"] != "full"
            or not math.isfinite(float(row["clir_score"]))
            or not isinstance(row["clir_selected_best_of_n"], bool)
        ):
            raise ValueError(f"invalid score identity at row {index}: {scored_path}")
        length = len(row["output_token_ids"])
        for field in TOKEN_SCORE_FIELDS:
            values = row[field]
            if (
                not isinstance(values, list)
                or len(values) != length
                or not all(math.isfinite(float(value)) for value in values)
            ):
                raise ValueError(f"unaligned {field} at row {index}: {scored_path}")
    return scored, {
        "path": str(scored_path),
        "file_sha256": file_sha256(scored_path),
        "rows": len(scored),
        "input_jsonl_sha256": file_sha256(reference_path),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _load_consistency_report(
    path: Path,
    *,
    seed: int,
    checkpoint_sha256: str,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS_HELDOUT_RELATION_EVALUATION"
        or report.get("cell") != "c1"
        or int(report.get("seed", -1)) != seed
        or int(report.get("completed_epoch", -1)) != int(protocol["training"]["epochs"])
        or report.get("inputs", {}).get("checkpoint", {}).get("file_sha256")
        != checkpoint_sha256
    ):
        raise ValueError(f"Consistency report identity drift: {path}")
    frozen = protocol["frozen_inputs"]
    expected = {
        "endpoint_manifest": frozen["consistency_heldout_endpoints"]["file_sha256"],
        "positive_relations": frozen["consistency_heldout_positive_relations"][
            "file_sha256"
        ],
        "negative_relations": frozen["consistency_heldout_negative_relations"][
            "file_sha256"
        ],
    }
    for key, expected_hash in expected.items():
        if report["inputs"][key]["file_sha256"] != expected_hash:
            raise ValueError(f"Consistency {key} hash drift: {path}")
    return report, {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "endpoint_rows": int(report["inputs"]["endpoint_manifest"]["rows"]),
        "positive_relations": int(report["inputs"]["positive_relations"]["rows"]),
        "negative_relations": int(report["inputs"]["negative_relations"]["rows"]),
        "checkpoint_sha256": checkpoint_sha256,
    }


def summarize(protocol_path: str | Path) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    output_root = _project_path(protocol["runtime"]["output_root"])
    train_paths = {
        "direct": output_root / "data/train_r0_p0.jsonl",
        "combined": output_root / "data/train_ch_full.jsonl",
    }
    prior_dev = output_root / "data/prior_dev_query_disjoint.jsonl"
    h_dev = output_root / "data/h_dev_query_disjoint.jsonl"
    seeds = [int(seed) for seed in protocol["training"]["seeds"]]
    checkpoints: dict[tuple[str, int], dict[str, Any]] = {}
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    score_artifacts: dict[str, Any] = {}

    for cell in ("r0", "p0", "ch", "full"):
        for seed in seeds:
            checkpoint_path = output_root / f"training/{cell}/seed-{seed}/checkpoint.pt"
            checkpoint = _checkpoint_audit(
                checkpoint_path,
                cell=cell,
                seed=seed,
                protocol=protocol,
                train_path=train_paths[protocol["cells"][cell]["train_manifest"]],
            )
            checkpoints[(cell, seed)] = checkpoint
            prior_path = output_root / f"evaluation/prior_dev_scored/{cell}_seed-{seed}.jsonl"
            prior_rows, prior_artifact = _validate_scored_rows(
                prior_dev, prior_path, checkpoint["file_sha256"]
            )
            score_artifacts[f"prior/{cell}/seed-{seed}"] = prior_artifact
            run_metrics: dict[str, Any] = {"prior": prior_metrics(prior_rows)}
            if cell in {"ch", "full"}:
                h_path = output_root / f"evaluation/h_dev_scored/{cell}_seed-{seed}.jsonl"
                h_rows, h_artifact = _validate_scored_rows(
                    h_dev, h_path, checkpoint["file_sha256"]
                )
                score_artifacts[f"h/{cell}/seed-{seed}"] = h_artifact
                consistency_path = (
                    output_root
                    / f"evaluation/consistency_reports/{cell}_seed-{seed}.json"
                )
                consistency, consistency_artifact = _load_consistency_report(
                    consistency_path,
                    seed=seed,
                    checkpoint_sha256=checkpoint["file_sha256"],
                    protocol=protocol,
                )
                score_artifacts[
                    f"consistency/{cell}/seed-{seed}"
                ] = consistency_artifact
                run_metrics["hallucination"] = h_metrics(h_rows)
                run_metrics["consistency"] = {
                    "representation": consistency["representation"],
                    "score": consistency["score"],
                    "endpoint_representation_diagnostics": consistency[
                        "endpoint_representation_diagnostics"
                    ],
                }
            metrics[(cell, seed)] = run_metrics

    stage_1 = aggregate_contrast(
        metrics,
        control="r0",
        treatment="p0",
        seeds=seeds,
        metrics=STAGE_1_METRICS,
    )
    stage_1["gate"] = evaluate_stage_1_gate(stage_1, protocol["stage_1_gate"])
    if not stage_1["gate"]["all_checks_pass"]:
        raise ValueError("completed Stage 2 despite a failed frozen Stage-1 gate")
    stage_2 = aggregate_contrast(
        metrics,
        control="ch",
        treatment="full",
        seeds=seeds,
        metrics=STAGE_2_METRICS,
    )
    stage_2["status"] = "COMPLETE_DESCRIPTIVE_MECHANISM_COMPATIBILITY_EVALUATION"
    stage_2["ranking_status"] = "DEFERRED_FRESH_QUERY_CLUSTER_POPULATION_REQUIRED"

    run_records = {
        f"{cell}/seed-{seed}": {
            "checkpoint": checkpoints[(cell, seed)],
            "mechanisms": metrics[(cell, seed)],
        }
        for cell in ("r0", "p0", "ch", "full")
        for seed in seeds
    }
    return {
        "schema_version": "clir-prior-v16-posthoc-staged-training-summary-v1",
        "status": (
            "COMPLETE_PRIOR_V16_POSTHOC_STAGED_TRAINING_AND_MECHANISM_EVALUATION"
        ),
        "created_at_utc": _utc_now(),
        "evidence_tier": protocol["evidence_boundary"]["tier"],
        "original_v16_status": protocol["evidence_boundary"]["original_v16_status"],
        "original_v17_status": protocol["evidence_boundary"]["original_v17_status"],
        "original_terminal_statuses_are_unchanged": True,
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": file_sha256(protocol_path),
        },
        "evaluation_code": {
            **_git_state(),
            "summary_script_sha256": file_sha256(Path(__file__)),
        },
        "data": {
            "direct_train": {
                "path": str(train_paths["direct"]),
                "file_sha256": file_sha256(train_paths["direct"]),
                "rows": protocol["data_contract"]["direct_train_rows"],
                "queries": protocol["data_contract"]["direct_train_queries"],
            },
            "combined_train": {
                "path": str(train_paths["combined"]),
                "file_sha256": file_sha256(train_paths["combined"]),
                "rows": protocol["data_contract"]["combined_train_rows"],
                "queries": protocol["data_contract"]["combined_train_queries"],
            },
            "prior_dev": {
                "path": str(prior_dev),
                "file_sha256": file_sha256(prior_dev),
                "rows": protocol["data_contract"]["prior_dev_rows"],
            },
            "h_dev": {
                "path": str(h_dev),
                "file_sha256": file_sha256(h_dev),
                "rows": protocol["data_contract"]["clean_h_dev_rows"],
            },
        },
        "runs": run_records,
        "score_artifacts": score_artifacts,
        "stage_1_r0_vs_p0": stage_1,
        "stage_2_ch_vs_full": stage_2,
        "claim_boundary": (
            "Post-hoc exploratory dual-AI Silver mechanism learnability and "
            "compatibility only. No human verification, fresh Best-of-N ranking "
            "population, protected test, or final-selection efficacy conclusion. "
            "H1 negative-tail, Path MIL, pseudo-onset tail, mutual Prior, progress, "
            "and reconstruction remain disabled."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_protocol(args.protocol)
    output_root = _project_path(protocol["runtime"]["output_root"])
    output = (
        Path(args.output_json).resolve()
        if args.output_json
        else output_root / "evaluation/staged_training_summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    report = summarize(args.protocol)
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
