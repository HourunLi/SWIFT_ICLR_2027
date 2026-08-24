"""Diagnostic held-out metrics for CLIR hallucination and dual-prior heads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_clir import atomic_write_json, file_sha256
from src.clir_data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIR mechanism outputs on an annotated scored JSONL."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--onset_window", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    negatives = len(labels_array) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = float(_average_ranks(scores_array)[labels_array == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def average_precision(
    labels: Sequence[int], scores: Sequence[float]
) -> float | None:
    """Tie-aware non-interpolated average precision."""
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    positives = int(labels_array.sum())
    if positives == 0:
        return None
    total = 0.0
    for threshold in np.unique(scores_array[labels_array == 1]):
        selected = scores_array >= threshold
        positives_at_threshold = int((labels_array[selected] == 1).sum())
        tied_positive_count = int(
            ((scores_array == threshold) & (labels_array == 1)).sum()
        )
        total += tied_positive_count * positives_at_threshold / int(selected.sum())
    return total / positives


def _binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(labels_array) != len(scores_array) or len(labels_array) == 0:
        raise ValueError("Binary metric inputs must be non-empty and aligned")
    if not np.isin(labels_array, [0, 1]).all() or not np.isfinite(scores_array).all():
        raise ValueError("Binary labels/scores must be finite and labels must be 0/1")
    return {
        "examples": len(labels_array),
        "positives": int(labels_array.sum()),
        "prevalence": float(labels_array.mean()),
        "average_precision": average_precision(labels_array, scores_array),
        "auroc": binary_auroc(labels_array, scores_array),
    }


def _probability_bce(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities_array, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(
            labels_array * np.log(clipped)
            + (1.0 - labels_array) * np.log(1.0 - clipped)
        )
    )


def _require_aligned_lists(
    row: Mapping[str, Any], row_number: int, fields: Sequence[str]
) -> int:
    lengths = {}
    for field in fields:
        value = row.get(field)
        if not isinstance(value, list):
            raise ValueError(f"Row {row_number} requires list field {field}")
        lengths[field] = len(value)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Row {row_number} has unaligned token fields: {lengths}")
    length = next(iter(lengths.values()))
    if length == 0:
        raise ValueError(f"Row {row_number} has no token outputs")
    return length


def evaluate_mechanisms(
    rows: Sequence[Mapping[str, Any]], onset_window: int = 5
) -> dict[str, Any]:
    if onset_window < 0:
        raise ValueError("onset_window must be non-negative")
    hallucination_labels: list[int] = []
    hallucination_scores: list[float] = []
    token_positions: list[float] = []
    key_labels: list[int] = []
    key_scores: list[float] = []
    complete_labels: list[int] = []
    complete_scores: list[float] = []
    path_labels: list[int] = []
    path_rank_scores: list[float] = []
    path_probabilities: list[float] = []
    positive_detected: list[bool] = []
    positive_within_window: list[bool] = []
    detected_errors: list[float] = []
    clean_no_onset: list[bool] = []
    pre_values: list[float] = []
    tail_values: list[float] = []
    clean_values: list[float] = []
    all_values: list[float] = []
    mutual_squared_l2: list[float] = []
    mutual_l1: list[float] = []
    prior_gate_squared_l2: list[float] = []
    prior_gate_dot_product: list[float] = []
    raw_gate_means: list[float] = []
    gate_entropies: list[float] = []
    gate_normalized_entropies: list[float] = []
    gate_effective_tokens: list[float] = []
    gate_effective_fractions: list[float] = []
    checkpoint_hashes: set[str] = set()

    token_fields = (
        "token_hallucination_target",
        "token_hallucination_mask",
        "key_prior_target",
        "complete_prior_target",
        "clir_hallucination_prob",
        "clir_token_value",
        "clir_key_prior_membership",
        "clir_complete_prior_membership",
        "clir_gate_attention",
        "clir_key_prior",
        "clir_complete_prior",
    )
    for row_number, row in enumerate(rows):
        length = _require_aligned_lists(row, row_number, token_fields)
        checkpoint_hash = row.get("clir_checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            raise ValueError(f"Row {row_number} lacks clir_checkpoint_sha256")
        checkpoint_hashes.add(checkpoint_hash)
        mask = np.asarray(row["token_hallucination_mask"], dtype=bool)
        if not mask.any():
            raise ValueError(f"Row {row_number} has no supervised tokens")
        positions = np.arange(length, dtype=np.float64) / max(length - 1, 1)

        h_labels = np.asarray(row["token_hallucination_target"], dtype=np.float64)
        key_target = np.asarray(row["key_prior_target"], dtype=np.float64)
        complete_target = np.asarray(row["complete_prior_target"], dtype=np.float64)
        for name, target in (
            ("token_hallucination_target", h_labels),
            ("key_prior_target", key_target),
            ("complete_prior_target", complete_target),
        ):
            if not np.isin(target[mask], [0.0, 1.0]).all():
                raise ValueError(f"Row {row_number} has non-binary {name}")

        hallucination_labels.extend(h_labels[mask].astype(int).tolist())
        hallucination_scores.extend(
            np.asarray(row["clir_hallucination_prob"], dtype=np.float64)[mask].tolist()
        )
        token_positions.extend(positions[mask].tolist())
        key_labels.extend(key_target[mask].astype(int).tolist())
        key_scores.extend(
            np.asarray(row["clir_key_prior_membership"], dtype=np.float64)[mask].tolist()
        )
        complete_labels.extend(complete_target[mask].astype(int).tolist())
        complete_scores.extend(
            np.asarray(row["clir_complete_prior_membership"], dtype=np.float64)[
                mask
            ].tolist()
        )

        if "path_hallucinated" not in row:
            raise ValueError(f"Row {row_number} lacks path_hallucinated")
        path_label = int(row["path_hallucinated"])
        if path_label not in {0, 1}:
            raise ValueError(f"Row {row_number} has non-binary path_hallucinated")
        path_labels.append(path_label)
        path_rank_scores.append(-float(row["clir_path_no_hallucination_log_prob"]))
        path_probabilities.append(float(row["clir_path_hallucination_prob"]))

        gold_onset = int(row["hallucination_onset"])
        predicted_onset = int(row["clir_pseudo_onset"])
        if gold_onset >= 0:
            detected = predicted_onset >= 0
            positive_detected.append(detected)
            positive_within_window.append(
                detected and abs(predicted_onset - gold_onset) <= onset_window
            )
            if detected:
                detected_errors.append(float(abs(predicted_onset - gold_onset)))
        else:
            clean_no_onset.append(predicted_onset == -1)

        values = np.asarray(row["clir_token_value"], dtype=np.float64)
        all_values.extend(values[mask].tolist())
        if gold_onset >= 0:
            pre_values.extend(values[:gold_onset].tolist())
            tail_values.extend(values[gold_onset:].tolist())
        else:
            clean_values.extend(values[mask].tolist())

        key_map = np.asarray(row["clir_key_prior"], dtype=np.float64)[mask]
        complete_map = np.asarray(row["clir_complete_prior"], dtype=np.float64)[mask]
        difference = key_map - complete_map
        mutual_squared_l2.append(float(np.square(difference).sum()))
        mutual_l1.append(float(np.abs(difference).sum()))

        gate_attention = np.asarray(row["clir_gate_attention"], dtype=np.float64)
        if not np.isfinite(gate_attention).all() or np.any(gate_attention < 0.0):
            raise ValueError(f"Row {row_number} has invalid clir_gate_attention")
        if not np.isclose(gate_attention.sum(), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"Row {row_number} clir_gate_attention does not sum to one"
            )
        if "clir_prior_gate_squared_l2" in row:
            squared_l2 = float(row["clir_prior_gate_squared_l2"])
        else:
            # Backward compatibility for scored files produced before the exact
            # loss-shaped diagnostic was emitted. Historical clean configs use
            # the fixed 0.5/0.5 prior fusion below.
            fused_prior = 0.5 * (
                np.asarray(row["clir_key_prior"], dtype=np.float64)
                + np.asarray(row["clir_complete_prior"], dtype=np.float64)
            )
            squared_l2 = float(np.square(gate_attention - fused_prior).sum())
        dot_product = float(row["clir_prior_gate_alignment"])
        raw_gate_mean = float(row["clir_mean_gate"])
        if not np.isfinite([squared_l2, dot_product, raw_gate_mean]).all():
            raise ValueError(f"Row {row_number} has non-finite gate diagnostics")
        if squared_l2 < 0.0 or not 0.0 <= raw_gate_mean <= 1.0:
            raise ValueError(f"Row {row_number} has invalid gate diagnostics")
        positive_attention = gate_attention[gate_attention > 0.0]
        entropy = float(-np.sum(positive_attention * np.log(positive_attention)))
        effective_tokens = float(1.0 / np.square(gate_attention).sum())
        prior_gate_squared_l2.append(squared_l2)
        prior_gate_dot_product.append(dot_product)
        raw_gate_means.append(raw_gate_mean)
        gate_entropies.append(entropy)
        gate_normalized_entropies.append(
            entropy / np.log(length) if length > 1 else 1.0
        )
        gate_effective_tokens.append(effective_tokens)
        gate_effective_fractions.append(effective_tokens / length)

    if not rows:
        raise ValueError("No rows to evaluate")
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"Expected one checkpoint hash, got {checkpoint_hashes}")

    token_metrics = _binary_metrics(hallucination_labels, hallucination_scores)
    token_metrics["position_baseline"] = _binary_metrics(
        hallucination_labels, token_positions
    )
    path_metrics = _binary_metrics(path_labels, path_rank_scores)
    path_probabilities_array = np.asarray(path_probabilities, dtype=np.float64)
    if not np.isfinite(path_probabilities_array).all() or np.any(
        (path_probabilities_array < 0.0) | (path_probabilities_array > 1.0)
    ):
        raise ValueError("Path probabilities must be finite and in [0, 1]")
    path_metrics["brier"] = float(
        np.mean(
            np.square(path_probabilities_array - np.asarray(path_labels, dtype=float))
        )
    )

    key_metrics = _binary_metrics(key_labels, key_scores)
    key_metrics["position_baseline"] = _binary_metrics(key_labels, token_positions)
    key_metrics["binary_cross_entropy"] = _probability_bce(key_labels, key_scores)
    complete_metrics = _binary_metrics(complete_labels, complete_scores)
    complete_metrics["position_baseline"] = _binary_metrics(
        complete_labels, token_positions
    )
    complete_metrics["binary_cross_entropy"] = _probability_bce(
        complete_labels, complete_scores
    )
    pre_mean = float(np.mean(pre_values)) if pre_values else None
    tail_mean = float(np.mean(tail_values)) if tail_values else None

    return {
        "schema_version": "clir-mechanism-diagnostics-v2",
        "rows": len(rows),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "hallucination": {
            "token": token_metrics,
            "path": path_metrics,
            "onset_threshold_0_5": {
                "positive_rows": len(positive_detected),
                "positive_detection_rate": float(np.mean(positive_detected)),
                "within_window": onset_window,
                "positive_within_window_rate": float(
                    np.mean(positive_within_window)
                ),
                "conditional_mae_when_detected": (
                    float(np.mean(detected_errors)) if detected_errors else None
                ),
                "clean_rows": len(clean_no_onset),
                "clean_no_onset_rate": float(np.mean(clean_no_onset)),
            },
            "token_value": {
                "pre_onset_pooled_mean": pre_mean,
                "post_onset_pooled_mean": tail_mean,
                "post_minus_pre": (
                    tail_mean - pre_mean
                    if pre_mean is not None and tail_mean is not None
                    else None
                ),
                "clean_pooled_mean": (
                    float(np.mean(clean_values)) if clean_values else None
                ),
                "all_pooled_mean": float(np.mean(all_values)),
            },
        },
        "dual_prior": {
            "key": key_metrics,
            "complete": complete_metrics,
            "mutual_map_discrepancy": {
                "mean_squared_l2": float(np.mean(mutual_squared_l2)),
                "mean_l1": float(np.mean(mutual_l1)),
            },
            "gate_alignment": {
                "full_trajectory_squared_l2_mean": float(
                    np.mean(prior_gate_squared_l2)
                ),
                "dot_product_mean": float(np.mean(prior_gate_dot_product)),
                "raw_sigmoid_gate_mean": float(np.mean(raw_gate_means)),
                "attention_entropy_mean": float(np.mean(gate_entropies)),
                "attention_normalized_entropy_mean": float(
                    np.mean(gate_normalized_entropies)
                ),
                "attention_effective_tokens_mean": float(
                    np.mean(gate_effective_tokens)
                ),
                "attention_effective_fraction_mean": float(
                    np.mean(gate_effective_fractions)
                ),
            },
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json)
    if output.resolve() == Path(args.input_jsonl).resolve():
        raise ValueError("output_json must not overwrite input_jsonl")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    report = evaluate_mechanisms(read_jsonl(args.input_jsonl), args.onset_window)
    report["input_jsonl_sha256"] = file_sha256(args.input_jsonl)
    atomic_write_json(output, report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
