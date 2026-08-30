"""Evaluate the frozen H0 target without requiring Dual-Prior annotations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, file_sha256
from evaluate_clir_mechanisms import average_precision, binary_auroc
from src.clir_data import read_jsonl


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    if len(targets) == 0 or len(targets) != len(probabilities):
        raise ValueError("binary targets and scores must be non-empty and aligned")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("binary targets must be 0/1")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probabilities must be finite and in [0, 1]")
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    binary_cross_entropy = float(
        -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
    )
    return {
        "examples": len(targets),
        "positives": int(targets.sum()),
        "prevalence": float(targets.mean()),
        "auroc": binary_auroc(targets.tolist(), probabilities.tolist()),
        "average_precision": average_precision(
            targets.tolist(), probabilities.tolist()
        ),
        "binary_cross_entropy": binary_cross_entropy,
        "brier": float(np.mean(np.square(probabilities - targets))),
    }


def evaluate_h0(
    rows: Sequence[Mapping[str, Any]],
    *,
    onset_threshold: float = 0.5,
    onset_window_tokens: int = 5,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no H0 rows to evaluate")
    if not 0.0 <= onset_threshold <= 1.0:
        raise ValueError("onset_threshold must be in [0, 1]")
    if onset_window_tokens < 0:
        raise ValueError("onset_window_tokens must be non-negative")

    token_labels: list[int] = []
    token_scores: list[float] = []
    token_positions: list[float] = []
    path_labels: list[int] = []
    path_scores: list[float] = []
    positive_detected: list[bool] = []
    positive_exact: list[bool] = []
    positive_within_window: list[bool] = []
    detected_absolute_errors: list[float] = []
    detected_normalized_errors: list[float] = []
    clean_no_onset: list[bool] = []
    checkpoint_hashes: set[str] = set()
    query_ids: set[str] = set()
    label_tiers: set[str] = set()
    source_counts: Counter[str] = Counter()
    source_paths: dict[str, list[int]] = defaultdict(list)
    source_path_scores: dict[str, list[float]] = defaultdict(list)

    for row_number, row in enumerate(rows):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"row {row_number} lacks query_id")
        if query_id in query_ids:
            raise ValueError(f"H0 dev requires one row per query: {query_id}")
        query_ids.add(query_id)
        checkpoint = row.get("clir_checkpoint_sha256")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"row {row_number} lacks checkpoint hash")
        checkpoint_hashes.add(checkpoint)
        label_tier = row.get("hallucination_label_tier")
        if not isinstance(label_tier, str) or not label_tier:
            raise ValueError(f"row {row_number} lacks hallucination label tier")
        label_tiers.add(label_tier)

        output_ids = row.get("output_token_ids")
        probabilities = row.get("clir_hallucination_prob")
        if not isinstance(output_ids, list) or not output_ids:
            raise ValueError(f"row {row_number} lacks output_token_ids")
        if not isinstance(probabilities, list) or len(probabilities) != len(output_ids):
            raise ValueError(f"row {row_number} has unaligned H0 probabilities")
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(probability_array).all() or np.any(
            (probability_array < 0.0) | (probability_array > 1.0)
        ):
            raise ValueError(f"row {row_number} has invalid H0 probabilities")

        onset = row.get("hallucination_onset")
        if not isinstance(onset, int) or isinstance(onset, bool):
            raise ValueError(f"row {row_number} has invalid hallucination_onset")
        length = len(output_ids)
        if onset < -1 or onset >= length:
            raise ValueError(f"row {row_number} onset is out of range")
        path_label = int(row.get("path_hallucinated", -1))
        expected_path = int(onset >= 0)
        if path_label != expected_path:
            raise ValueError(f"row {row_number} path/onset target mismatch")
        target = np.zeros(length, dtype=np.int64)
        if onset >= 0:
            target[onset:] = 1
        token_labels.extend(target.tolist())
        token_scores.extend(probability_array.tolist())
        token_positions.extend(
            (np.arange(length, dtype=np.float64) / max(length - 1, 1)).tolist()
        )

        path_probability = float(row["clir_path_hallucination_prob"])
        if not math_is_probability(path_probability):
            raise ValueError(f"row {row_number} has invalid path probability")
        path_labels.append(path_label)
        path_scores.append(path_probability)
        source = str(row.get("source", "unknown"))
        source_counts[source] += 1
        source_paths[source].append(path_label)
        source_path_scores[source].append(path_probability)

        predicted_onset = row.get("clir_pseudo_onset")
        if not isinstance(predicted_onset, int) or isinstance(predicted_onset, bool):
            raise ValueError(f"row {row_number} has invalid predicted onset")
        if predicted_onset < -1 or predicted_onset >= length:
            raise ValueError(f"row {row_number} predicted onset is out of range")
        expected_from_probabilities = next(
            (
                index
                for index, probability in enumerate(probability_array)
                if probability >= onset_threshold
            ),
            -1,
        )
        if predicted_onset != expected_from_probabilities:
            raise ValueError(
                f"row {row_number} pseudo onset was not produced at the frozen threshold"
            )
        if onset >= 0:
            detected = predicted_onset >= 0
            positive_detected.append(detected)
            positive_exact.append(predicted_onset == onset)
            positive_within_window.append(
                detected and abs(predicted_onset - onset) <= onset_window_tokens
            )
            if detected:
                error = float(abs(predicted_onset - onset))
                detected_absolute_errors.append(error)
                detected_normalized_errors.append(error / max(length - 1, 1))
        else:
            clean_no_onset.append(predicted_onset == -1)

    if len(checkpoint_hashes) != 1:
        raise ValueError("H0 report must contain exactly one checkpoint")
    if len(label_tiers) != 1:
        raise ValueError("H0 report mixes label tiers")
    if Counter(path_labels) != Counter({0: 100, 1: 100}):
        raise ValueError(f"frozen H0 dev balance drift: {Counter(path_labels)}")

    token = _binary_metrics(token_labels, token_scores)
    token["position_baseline"] = _binary_metrics(token_labels, token_positions)
    path = _binary_metrics(path_labels, path_scores)
    by_source = {
        source: {
            "rows": source_counts[source],
            "path": _binary_metrics(source_paths[source], source_path_scores[source]),
        }
        for source in sorted(source_counts)
    }
    positive_detection_rate = float(np.mean(positive_detected))
    clean_no_onset_rate = float(np.mean(clean_no_onset))
    return {
        "schema_version": "clir-h0-v7.4-heldout-diagnostics",
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "rows": len(rows),
        "queries": len(query_ids),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "hallucination_label_tier": next(iter(label_tiers)),
        "source_counts": dict(sorted(source_counts.items())),
        "token": token,
        "path": path,
        "onset": {
            "threshold": onset_threshold,
            "window_tokens": onset_window_tokens,
            "positive_rows": len(positive_detected),
            "positive_detection_rate": positive_detection_rate,
            "positive_exact_start_rate": float(np.mean(positive_exact)),
            "positive_within_window_rate": float(np.mean(positive_within_window)),
            "conditional_mae_tokens_when_detected": (
                float(np.mean(detected_absolute_errors))
                if detected_absolute_errors
                else None
            ),
            "conditional_normalized_mae_when_detected": (
                float(np.mean(detected_normalized_errors))
                if detected_normalized_errors
                else None
            ),
            "clean_rows": len(clean_no_onset),
            "clean_no_onset_rate": clean_no_onset_rate,
            "balanced_path_decision_accuracy": (
                positive_detection_rate + clean_no_onset_rate
            )
            / 2.0,
        },
        "by_source": by_source,
    }


def math_is_probability(value: float) -> bool:
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate H0 on the frozen 200-query Silver dev set."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--onset_threshold", type=float, default=0.5)
    parser.add_argument("--onset_window_tokens", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_json)
    if output.resolve() == Path(args.input_jsonl).resolve():
        raise ValueError("output_json must not overwrite input_jsonl")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    report = evaluate_h0(
        read_jsonl(args.input_jsonl),
        onset_threshold=args.onset_threshold,
        onset_window_tokens=args.onset_window_tokens,
    )
    report["input_jsonl_sha256"] = file_sha256(args.input_jsonl)
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
