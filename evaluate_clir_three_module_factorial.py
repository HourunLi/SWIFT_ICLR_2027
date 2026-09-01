#!/usr/bin/env python
"""Summarize frozen mechanism diagnostics for the CLIR 2x2x2 factorial."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, file_sha256
from evaluate_clir_consistency import relation_metrics
from evaluate_clir_h0 import _binary_metrics as h_binary_metrics
from src.clir_data import read_jsonl
from summarize_clir_prior_gate import _prior_run_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/three_module_expansion_v1/protocol.json"
DEFAULT_COMPLETION = (
    PROJECT_ROOT
    / "run_artifacts/three_module_expansion_v1/training/completion_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "run_artifacts/three_module_expansion_v1/evaluation/mechanism_summary.json"
)
CELL_FACTORS = {
    "u0": (0, 0, 0),
    "c": (1, 0, 0),
    "h": (0, 1, 0),
    "p": (0, 0, 1),
    "ch": (1, 1, 0),
    "cp": (1, 0, 1),
    "hp": (0, 1, 1),
    "full": (1, 1, 1),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _bce(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    targets = np.asarray(labels, dtype=np.float64)
    scores = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(scores) + (1 - targets) * np.log(1 - scores)))


def h_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    onset_threshold: float = 0.5,
    onset_window_tokens: int = 5,
) -> dict[str, Any]:
    """Evaluate the query-disjoint H dev without assuming the old 100/100 count."""
    if not rows:
        raise ValueError("no H rows")
    token_labels: list[int] = []
    token_scores: list[float] = []
    token_positions: list[float] = []
    path_labels: list[int] = []
    path_scores: list[float] = []
    positive_detected: list[bool] = []
    positive_exact: list[bool] = []
    positive_window: list[bool] = []
    detected_errors: list[float] = []
    detected_normalized_errors: list[float] = []
    clean_no_onset: list[bool] = []
    query_ids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    source_counts: Counter[str] = Counter()
    source_labels: dict[str, list[int]] = defaultdict(list)
    source_scores: dict[str, list[float]] = defaultdict(list)

    for index, row in enumerate(rows):
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in query_ids:
            raise ValueError(f"H dev query identity drift at row {index}")
        query_ids.add(query_id)
        checkpoint_hashes.add(str(row.get("clir_checkpoint_sha256", "")))
        output_ids = row.get("output_token_ids")
        probabilities = row.get("clir_hallucination_prob")
        if (
            not isinstance(output_ids, list)
            or not output_ids
            or not isinstance(probabilities, list)
            or len(probabilities) != len(output_ids)
        ):
            raise ValueError(f"unaligned H token fields at row {index}")
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(probability_array).all() or np.any(
            (probability_array < 0) | (probability_array > 1)
        ):
            raise ValueError(f"invalid H probabilities at row {index}")
        onset = int(row["hallucination_onset"])
        if onset < -1 or onset >= len(output_ids):
            raise ValueError(f"invalid onset at row {index}")
        path_label = int(row["path_hallucinated"])
        if path_label != int(onset >= 0):
            raise ValueError(f"path/onset mismatch at row {index}")
        target = np.zeros(len(output_ids), dtype=np.int64)
        if onset >= 0:
            target[onset:] = 1
        token_labels.extend(target.tolist())
        token_scores.extend(probability_array.tolist())
        token_positions.extend(
            (np.arange(len(output_ids)) / max(len(output_ids) - 1, 1)).tolist()
        )
        path_probability = float(row["clir_path_hallucination_prob"])
        if not math.isfinite(path_probability) or not 0 <= path_probability <= 1:
            raise ValueError(f"invalid H path probability at row {index}")
        path_labels.append(path_label)
        path_scores.append(path_probability)
        source = str(row.get("source", "unknown"))
        source_counts[source] += 1
        source_labels[source].append(path_label)
        source_scores[source].append(path_probability)
        predicted = int(row["clir_pseudo_onset"])
        expected = next(
            (
                token_index
                for token_index, probability in enumerate(probability_array)
                if probability >= onset_threshold
            ),
            -1,
        )
        if predicted != expected:
            raise ValueError(f"pseudo onset threshold drift at row {index}")
        if onset >= 0:
            detected = predicted >= 0
            positive_detected.append(detected)
            positive_exact.append(predicted == onset)
            positive_window.append(
                detected and abs(predicted - onset) <= onset_window_tokens
            )
            if detected:
                error = float(abs(predicted - onset))
                detected_errors.append(error)
                detected_normalized_errors.append(error / max(len(output_ids) - 1, 1))
        else:
            clean_no_onset.append(predicted == -1)

    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("H score file must bind exactly one checkpoint")
    token = h_binary_metrics(token_labels, token_scores)
    token["position_baseline"] = h_binary_metrics(token_labels, token_positions)
    path = h_binary_metrics(path_labels, path_scores)
    positive_detection = float(np.mean(positive_detected))
    clean_rate = float(np.mean(clean_no_onset))
    return {
        "rows": len(rows),
        "queries": len(query_ids),
        "class_counts": dict(sorted(Counter(path_labels).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "token": token,
        "path": path,
        "onset": {
            "threshold": onset_threshold,
            "window_tokens": onset_window_tokens,
            "positive_rows": len(positive_detected),
            "positive_detection_rate": positive_detection,
            "positive_exact_start_rate": float(np.mean(positive_exact)),
            "positive_within_window_rate": float(np.mean(positive_window)),
            "conditional_mae_tokens_when_detected": (
                float(np.mean(detected_errors)) if detected_errors else None
            ),
            "conditional_normalized_mae_when_detected": (
                float(np.mean(detected_normalized_errors))
                if detected_normalized_errors
                else None
            ),
            "clean_rows": len(clean_no_onset),
            "clean_no_onset_rate": clean_rate,
            "balanced_path_decision_accuracy": (positive_detection + clean_rate) / 2,
        },
        "by_source": {
            source: h_binary_metrics(source_labels[source], source_scores[source])
            for source in sorted(source_counts)
        },
    }


def _load_merge(
    merge_path: Path,
    *,
    expected_mode: str,
    expected_input_sha256: str,
    completion_sha256: str,
) -> dict[str, Any]:
    report = _load_json(merge_path)
    if (
        report.get("status") != "PASS_FACTORIAL_SCORING_MERGE"
        or report.get("mode") != expected_mode
        or report.get("input_jsonl_sha256") != expected_input_sha256
        or report.get("completion_report_sha256") != completion_sha256
        or len(report.get("outputs", {})) != 24
    ):
        raise ValueError(f"invalid factorial scoring merge: {merge_path}")
    return report


def _run_rows(
    merge: Mapping[str, Any], cell: str, seed: int, checkpoint_hash: str
) -> list[dict[str, Any]]:
    key = f"{cell}/seed-{seed}"
    record = merge["outputs"].get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"merge report lacks {key}")
    path = Path(str(record["path"]))
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"merged score hash drift: {path}")
    rows = read_jsonl(path)
    if len(rows) != int(record["rows"]):
        raise ValueError(f"merged score row-count drift: {path}")
    if any(row.get("clir_checkpoint_sha256") != checkpoint_hash for row in rows):
        raise ValueError(f"merged score checkpoint drift: {path}")
    return rows


def _consistency_metrics(
    rows: Sequence[Mapping[str, Any]],
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    representations = {
        str(row["id"]): row["clir_representation"] for row in rows
    }
    scores = {str(row["id"]): float(row["clir_score"]) for row in rows}
    if len(representations) != len(rows):
        raise ValueError("duplicate Consistency endpoint ID")
    required = {
        str(relation[field])
        for relation in [*positive, *negative]
        for field in ("left_id", "right_id")
    }
    if set(representations) != required:
        raise ValueError("Consistency scored rows are not the exact endpoint union")
    return relation_metrics(representations, scores, positive, negative, margin=0.2)


METRIC_PATHS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "consistency.representation_separation": lambda row: row["consistency"]["representation"]["mean_separation_positive_minus_negative"],
    "consistency.representation_auroc": lambda row: row["consistency"]["representation"]["relation_classification_auroc"],
    "consistency.representation_average_precision": lambda row: row["consistency"]["representation"]["relation_classification_average_precision"],
    "consistency.positive_one_minus_cosine": lambda row: row["consistency"]["representation"]["positive_mean_one_minus_cosine"],
    "consistency.score_gap_separation": lambda row: row["consistency"]["score"]["mean_gap_separation_negative_minus_positive"],
    "hallucination.token_average_precision": lambda row: row["hallucination"]["token"]["average_precision"],
    "hallucination.token_auroc": lambda row: row["hallucination"]["token"]["auroc"],
    "hallucination.token_bce": lambda row: row["hallucination"]["token"]["binary_cross_entropy"],
    "hallucination.path_average_precision": lambda row: row["hallucination"]["path"]["average_precision"],
    "hallucination.path_auroc": lambda row: row["hallucination"]["path"]["auroc"],
    "hallucination.path_bce": lambda row: row["hallucination"]["path"]["binary_cross_entropy"],
    "hallucination.balanced_path_accuracy": lambda row: row["hallucination"]["onset"]["balanced_path_decision_accuracy"],
    "hallucination.positive_within_window_rate": lambda row: row["hallucination"]["onset"]["positive_within_window_rate"],
    "prior.key_average_precision": lambda row: row["prior"]["key"]["average_precision"],
    "prior.key_auroc": lambda row: row["prior"]["key"]["auroc"],
    "prior.key_bce": lambda row: row["prior"]["key"]["binary_cross_entropy"],
    "prior.complete_average_precision": lambda row: row["prior"]["complete"]["average_precision"],
    "prior.complete_auroc": lambda row: row["prior"]["complete"]["auroc"],
    "prior.complete_bce": lambda row: row["prior"]["complete"]["binary_cross_entropy"],
    "prior.gate_squared_l2": lambda row: row["prior"]["gate"]["full_trajectory_squared_l2_mean"],
    "prior.gate_alignment": lambda row: row["prior"]["gate"]["dot_product_mean"],
    "prior.gate_normalized_entropy": lambda row: row["prior"]["gate"]["attention_normalized_entropy_mean"],
    "prior.gate_effective_fraction": lambda row: row["prior"]["gate"]["attention_effective_token_fraction_mean"],
}


def factorial_effects(cell_values: Mapping[str, float]) -> dict[str, float]:
    """Return averaged 0/1 main effects, two-way DDs, and the three-way DDD."""
    if set(cell_values) != set(CELL_FACTORS):
        raise ValueError("factorial effects require all eight cells")
    u0, c, h, p = (float(cell_values[key]) for key in ("u0", "c", "h", "p"))
    ch, cp, hp, full = (
        float(cell_values[key]) for key in ("ch", "cp", "hp", "full")
    )
    return {
        "C_main": (c + ch + cp + full - u0 - h - p - hp) / 4,
        "H_main": (h + ch + hp + full - u0 - c - p - cp) / 4,
        "P_main": (p + cp + hp + full - u0 - c - h - ch) / 4,
        "C_x_H": (u0 - c - h + ch + p - cp - hp + full) / 2,
        "C_x_P": (u0 - c + h - ch - p + cp - hp + full) / 2,
        "H_x_P": (u0 + c - h - ch - p - cp + hp + full) / 2,
        "C_x_H_x_P": full - ch - cp - hp + c + h + p - u0,
        "Full_minus_U0": full - u0,
    }


def _summarize_metrics(run_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_identity = {
        (str(row["cell"]), int(row["seed"])): row for row in run_metrics
    }
    seeds = sorted({seed for _, seed in by_identity})
    if len(by_identity) != 24 or len(seeds) != 3:
        raise ValueError("mechanism summary requires the complete 8x3 grid")
    output: dict[str, Any] = {}
    for metric, getter in METRIC_PATHS.items():
        cell_seed = {
            cell: {str(seed): float(getter(by_identity[(cell, seed)])) for seed in seeds}
            for cell in CELL_FACTORS
        }
        cell_means = {
            cell: float(np.mean(list(seed_values.values())))
            for cell, seed_values in cell_seed.items()
        }
        by_seed_effects = {
            str(seed): factorial_effects(
                {cell: float(getter(by_identity[(cell, seed)])) for cell in CELL_FACTORS}
            )
            for seed in seeds
        }
        effect_means = {
            effect: float(np.mean([values[effect] for values in by_seed_effects.values()]))
            for effect in next(iter(by_seed_effects.values()))
        }
        output[metric] = {
            "cell_seed": cell_seed,
            "cell_mean": cell_means,
            "factorial_effect_by_seed": by_seed_effects,
            "factorial_effect_mean": effect_means,
        }
    return output


def command_mechanisms(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    completion_path = Path(args.completion_report).resolve()
    protocol = _load_json(protocol_path)
    completion = _load_json(completion_path)
    if (
        completion.get("status")
        != "PASS_THREE_MODULE_COMPLETE_2X2X2_24_RUN_TRAINING"
        or completion.get("mechanism_evaluation_allowed") is not True
    ):
        raise ValueError("24-run training completion has not authorized mechanisms")
    completion_sha256 = file_sha256(completion_path)
    runs = completion.get("runs", [])
    if len(runs) != 24:
        raise ValueError("training completion run-count drift")
    for run in runs:
        if tuple(run["factors"]) != CELL_FACTORS[str(run["cell"])]:
            raise ValueError("completion factorial identity drift")

    c_input = _project_path(
        protocol["frozen_inputs"]["consistency_heldout_endpoints"]["path"]
    )
    h_input = _project_path(
        "run_artifacts/three_module_expansion_v1/data/h_dev_query_disjoint.jsonl"
    )
    p_input = _project_path(
        "run_artifacts/three_module_expansion_v1/data/prior_dev_query_disjoint.jsonl"
    )
    expected_hashes = {
        "consistency": protocol["frozen_inputs"]["consistency_heldout_endpoints"]["file_sha256"],
        "hallucination": file_sha256(h_input),
        "prior": file_sha256(p_input),
    }
    if file_sha256(c_input) != expected_hashes["consistency"]:
        raise ValueError("Consistency endpoint input hash drift")

    c_merge_path = Path(args.consistency_merge).resolve()
    h_merge_path = Path(args.hallucination_merge).resolve()
    p_merge_path = Path(args.prior_merge).resolve()
    c_merge = _load_merge(
        c_merge_path,
        expected_mode="consistency",
        expected_input_sha256=expected_hashes["consistency"],
        completion_sha256=completion_sha256,
    )
    h_merge = _load_merge(
        h_merge_path,
        expected_mode="full",
        expected_input_sha256=expected_hashes["hallucination"],
        completion_sha256=completion_sha256,
    )
    p_merge = _load_merge(
        p_merge_path,
        expected_mode="full",
        expected_input_sha256=expected_hashes["prior"],
        completion_sha256=completion_sha256,
    )
    positive_path = _project_path(
        protocol["frozen_inputs"]["consistency_heldout_positive_relations"]["path"]
    )
    negative_path = _project_path(
        protocol["frozen_inputs"]["consistency_heldout_negative_relations"]["path"]
    )
    positive = read_jsonl(positive_path)
    negative = read_jsonl(negative_path)

    run_metrics: list[dict[str, Any]] = []
    for run in runs:
        cell, seed = str(run["cell"]), int(run["seed"])
        checkpoint_hash = str(run["checkpoint_file_sha256"])
        c_rows = _run_rows(c_merge, cell, seed, checkpoint_hash)
        h_rows = _run_rows(h_merge, cell, seed, checkpoint_hash)
        p_rows = _run_rows(p_merge, cell, seed, checkpoint_hash)
        run_metrics.append(
            {
                "cell": cell,
                "seed": seed,
                "factors": list(run["factors"]),
                "checkpoint_sha256": checkpoint_hash,
                "consistency": _consistency_metrics(c_rows, positive, negative),
                "hallucination": h_metrics(h_rows),
                "prior": _prior_run_metrics(p_rows),
            }
        )

    metrics = _summarize_metrics(run_metrics)
    checks = {
        "C_relation_separation_main_effect_positive": metrics[
            "consistency.representation_separation"
        ]["factorial_effect_mean"]["C_main"]
        > 0,
        "H_token_AP_main_effect_positive": metrics[
            "hallucination.token_average_precision"
        ]["factorial_effect_mean"]["H_main"]
        > 0,
        "H_token_BCE_main_effect_negative": metrics["hallucination.token_bce"][
            "factorial_effect_mean"
        ]["H_main"]
        < 0,
        "P_key_AP_main_effect_positive": metrics["prior.key_average_precision"][
            "factorial_effect_mean"
        ]["P_main"]
        > 0,
        "P_complete_AP_main_effect_positive": metrics[
            "prior.complete_average_precision"
        ]["factorial_effect_mean"]["P_main"]
        > 0,
        "P_gate_L2_main_effect_negative": metrics["prior.gate_squared_l2"][
            "factorial_effect_mean"
        ]["P_main"]
        < 0,
    }
    report = {
        "schema_version": "clir-three-module-factorial-mechanisms-v1",
        "status": "PASS_THREE_MODULE_MECHANISM_EVALUATION",
        "created_at_utc": _utc_now(),
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "terminal_statuses_preserved": protocol["terminal_statuses_preserved"],
        "inputs": {
            "protocol_sha256": file_sha256(protocol_path),
            "training_completion_sha256": completion_sha256,
            "consistency_merge_sha256": file_sha256(c_merge_path),
            "hallucination_merge_sha256": file_sha256(h_merge_path),
            "prior_merge_sha256": file_sha256(p_merge_path),
            "positive_relations_sha256": file_sha256(positive_path),
            "negative_relations_sha256": file_sha256(negative_path),
        },
        "factorial_effect_definition": {
            "main": "average_on_minus_off_over_the_other_two_factors_within_seed",
            "two_way": "average_difference_in_differences_over_the_third_factor_within_seed",
            "three_way": "difference_in_difference_in_differences_within_seed",
            "seed_summary": "arithmetic_mean_of_the_three_fixed_seed_effects",
        },
        "runs": run_metrics,
        "metrics": metrics,
        "descriptive_mechanism_checks": checks,
        "ranking_authorization_eligible": True,
        "claim_boundary": (
            "mechanism_learnability_on_query_disjoint_dual_ai_silver_dev_only; "
            "not_Best_of_N_efficacy_not_Gold_not_human_verified_and_not_a_repair_"
            "of_v7_v12_v13"
        ),
    }
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"mechanism report already exists: {output}")
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "ranking_authorization_eligible": True,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--completion-report", default=str(DEFAULT_COMPLETION))
    parser.add_argument("--consistency-merge", required=True)
    parser.add_argument("--hallucination-merge", required=True)
    parser.add_argument("--prior-merge", required=True)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    command_mechanisms(build_parser().parse_args())


if __name__ == "__main__":
    main()
