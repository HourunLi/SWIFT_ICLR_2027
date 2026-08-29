#!/usr/bin/env python
"""Evaluate CLIR representations on frozen held-out positive/negative relations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_clir import atomic_write_json
from evaluate_clir_mechanisms import average_precision, binary_auroc
from src.clir_consistency_scale_training import relation_signature
from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
    read_jsonl,
)
from src.clir_smoke import canonical_sha256, file_sha256
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig


REPORT_SCHEMA = "clir-consistency-heldout-relation-evaluation-v6.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one C0/C1 checkpoint on frozen held-out relations."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--positive_relations", required=True)
    parser.add_argument("--negative_relations", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected_train_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--amp_dtype", default="bfloat16", choices=["none", "bfloat16"])
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--cell", required=True, choices=["c0", "c1"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 autocast is supported only on CPU or CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Distribution inputs must be non-empty and finite")
    quantiles = np.quantile(array, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def relation_metrics(
    representations: Mapping[str, Sequence[float]],
    scores: Mapping[str, float],
    positive_relations: Sequence[Mapping[str, Any]],
    negative_relations: Sequence[Mapping[str, Any]],
    *,
    margin: float = 0.2,
) -> dict[str, Any]:
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    all_relations = [*positive_relations, *negative_relations]
    if not all_relations:
        raise ValueError("No relations supplied")
    relation_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for expected_label, relations in ((1, positive_relations), (0, negative_relations)):
        for relation in relations:
            relation_id = str(relation["relation_id"])
            if relation_id in relation_ids:
                raise ValueError(f"Duplicate relation_id: {relation_id}")
            relation_ids.add(relation_id)
            if int(relation["label"]) != expected_label:
                raise ValueError(f"Relation {relation_id} label drift")
            left_id, right_id = str(relation["left_id"]), str(relation["right_id"])
            if left_id not in representations or right_id not in representations:
                raise ValueError(
                    f"Relation {relation_id} lacks an endpoint representation"
                )
            if left_id not in scores or right_id not in scores:
                raise ValueError(f"Relation {relation_id} lacks an endpoint score")
            left = np.asarray(representations[left_id], dtype=np.float64)
            right = np.asarray(representations[right_id], dtype=np.float64)
            if left.shape != right.shape or left.ndim != 1:
                raise ValueError(f"Relation {relation_id} representation shape drift")
            cosine = float(np.dot(left, right))
            score_gap = abs(float(scores[left_id]) - float(scores[right_id]))
            if not math.isfinite(cosine) or not math.isfinite(score_gap):
                raise ValueError(f"Relation {relation_id} has non-finite metrics")
            records.append(
                {
                    "relation_id": relation_id,
                    "label": expected_label,
                    "left_id": left_id,
                    "right_id": right_id,
                    "cosine_similarity": cosine,
                    "absolute_score_gap": score_gap,
                }
            )
    positive = [row for row in records if row["label"] == 1]
    negative = [row for row in records if row["label"] == 0]
    if not positive or not negative:
        raise ValueError("Held-out evaluation requires positive and negative relations")
    labels = [int(row["label"]) for row in records]
    cosines = [float(row["cosine_similarity"]) for row in records]
    score_similarities = [-float(row["absolute_score_gap"]) for row in records]
    positive_cosines = [float(row["cosine_similarity"]) for row in positive]
    negative_cosines = [float(row["cosine_similarity"]) for row in negative]
    positive_gaps = [float(row["absolute_score_gap"]) for row in positive]
    negative_gaps = [float(row["absolute_score_gap"]) for row in negative]
    return {
        "relation_signature_sha256": relation_signature(all_relations),
        "relation_rows_canonical_sha256": canonical_sha256(records),
        "relations": records,
        "representation": {
            "positive_cosine": _distribution(positive_cosines),
            "negative_cosine": _distribution(negative_cosines),
            "mean_separation_positive_minus_negative": float(
                np.mean(positive_cosines) - np.mean(negative_cosines)
            ),
            "relation_classification_auroc": binary_auroc(labels, cosines),
            "relation_classification_average_precision": average_precision(
                labels, cosines
            ),
            "positive_mean_one_minus_cosine": float(
                np.mean(1.0 - np.asarray(positive_cosines))
            ),
            "negative_margin_violation_rate": float(
                np.mean(np.asarray(negative_cosines) > margin)
            ),
            "negative_mean_hinge": float(
                np.mean(np.maximum(np.asarray(negative_cosines) - margin, 0.0))
            ),
            "margin": margin,
        },
        "score": {
            "positive_absolute_gap": _distribution(positive_gaps),
            "negative_absolute_gap": _distribution(negative_gaps),
            "mean_gap_separation_negative_minus_positive": float(
                np.mean(negative_gaps) - np.mean(positive_gaps)
            ),
            "relation_classification_auroc_from_negative_gap": binary_auroc(
                labels, score_similarities
            ),
            "relation_classification_average_precision_from_negative_gap": (
                average_precision(labels, score_similarities)
            ),
            "positive_score_mse": float(np.mean(np.square(positive_gaps))),
        },
    }


def _load_checkpoint(
    path: Path, device: torch.device, expected_train_sha256: str, seed: int, cell: str
) -> tuple[ConsistencyLocalizedReward, Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    data_state = checkpoint.get("data_state", {})
    if data_state.get("train_sha256") != expected_train_sha256:
        raise ValueError("Checkpoint was not trained on the expected shared manifest")
    training_contract = checkpoint.get("training_contract", {})
    if int(training_contract.get("seed", -1)) != seed:
        raise ValueError("Checkpoint seed does not match --seed")
    model_values = checkpoint.get("model_config")
    if not isinstance(model_values, Mapping):
        raise ValueError("Checkpoint lacks model_config")
    expected_weight = 0.0 if cell == "c0" else 1.0
    if float(model_values.get("consistency_weight", -1.0)) != expected_weight:
        raise ValueError("Checkpoint Consistency weight does not match --cell")
    if int(checkpoint.get("completed_epoch", 0)) <= 0:
        raise ValueError("Checkpoint has not completed an epoch")
    model = ConsistencyLocalizedReward(RewardConfig(**model_values)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    output = Path(args.output_json)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    input_path = Path(args.input_jsonl)
    positive_path = Path(args.positive_relations)
    negative_path = Path(args.negative_relations)
    checkpoint_path = Path(args.model)
    train_path = Path(args.expected_train_jsonl)
    protected = {
        input_path.resolve(),
        positive_path.resolve(),
        negative_path.resolve(),
        checkpoint_path.resolve(),
        train_path.resolve(),
    }
    if output.resolve() in protected:
        raise ValueError("Evaluation output must not overwrite an input")
    positive_relations = read_jsonl(positive_path)
    negative_relations = read_jsonl(negative_path)
    required_endpoints = {
        str(relation[field])
        for relation in [*positive_relations, *negative_relations]
        for field in ("left_id", "right_id")
    }
    dataset = CLIRTrajectoryDataset(input_path, feature_root=args.feature_root)
    dataset_ids = [str(row["id"]) for row in dataset.rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Evaluation feature manifest has duplicate IDs")
    if set(dataset_ids) != required_endpoints:
        raise ValueError("Evaluation feature manifest is not the exact endpoint union")
    device = resolve_device(args.device)
    model, checkpoint = _load_checkpoint(
        checkpoint_path,
        device,
        expected_train_sha256=file_sha256(train_path),
        seed=args.seed,
        cell=args.cell,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    representations: dict[str, list[float]] = {}
    scores: dict[str, float] = {}
    for raw_batch in loader:
        ids = list(raw_batch["ids"])
        batch = move_batch_to_device(raw_batch, device)
        if args.amp_dtype == "none":
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].float()
        with autocast_context(device, args.amp_dtype):
            outputs = model(
                batch["hidden_states"],
                mask=batch["mask"],
                condition_states=batch.get("condition_states"),
                condition_mask=batch.get("condition_mask"),
                condition_embedding=batch.get("condition_embedding"),
                condition_embedding_mask=batch.get("condition_embedding_mask"),
            )
        for key in ("scores", "representations"):
            if not torch.isfinite(outputs[key]).all():
                raise FloatingPointError(f"Non-finite held-out output: {key}")
        for offset, endpoint_id in enumerate(ids):
            representations[str(endpoint_id)] = (
                outputs["representations"][offset].detach().float().cpu().tolist()
            )
            scores[str(endpoint_id)] = float(outputs["scores"][offset].float().cpu())
    metrics = relation_metrics(
        representations,
        scores,
        positive_relations,
        negative_relations,
        margin=args.margin,
    )
    matrix = np.asarray([representations[endpoint_id] for endpoint_id in dataset_ids])
    norms = np.linalg.norm(matrix, axis=1)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_HELDOUT_RELATION_EVALUATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": args.cell,
        "seed": args.seed,
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "inputs": {
            "endpoint_manifest": {
                "path": str(input_path.resolve()),
                "file_sha256": file_sha256(input_path),
                "rows": len(dataset),
            },
            "positive_relations": {
                "path": str(positive_path.resolve()),
                "file_sha256": file_sha256(positive_path),
                "rows": len(positive_relations),
            },
            "negative_relations": {
                "path": str(negative_path.resolve()),
                "file_sha256": file_sha256(negative_path),
                "rows": len(negative_relations),
            },
            "expected_train_manifest": {
                "path": str(train_path.resolve()),
                "file_sha256": file_sha256(train_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "file_sha256": file_sha256(checkpoint_path),
            },
        },
        "endpoint_representation_diagnostics": {
            "endpoints": len(dataset_ids),
            "dimension": int(matrix.shape[1]),
            "norm": _distribution(norms.tolist()),
            "centroid_norm": float(np.linalg.norm(matrix.mean(axis=0))),
            "mean_coordinate_variance": float(matrix.var(axis=0).mean()),
        },
        **metrics,
        "claim_boundary": (
            "heldout_relation_mechanism_learnability_only_not_best_of_n_or_ranking_efficacy"
        ),
    }
    atomic_write_json(output, report)
    return report


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cell": report["cell"],
                "seed": report["seed"],
                "epoch": report["completed_epoch"],
                "cosine_separation": report["representation"][
                    "mean_separation_positive_minus_negative"
                ],
                "auroc": report["representation"]["relation_classification_auroc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
