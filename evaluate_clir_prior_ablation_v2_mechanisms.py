#!/usr/bin/env python
"""Evaluate Prior heads, Key/Complete separation, and Gate alignment on v16 dev."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_clir_mechanisms import average_precision, binary_auroc
from prepare_clir_prior_ablation_v2 import load_protocol
import score_clir_factorial as scoring
from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device
from src.clir_smoke import atomic_write_json, file_sha256


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _bce(labels: list[int], probabilities: list[float]) -> float:
    target = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(
        -np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))
    )


def _binary(labels: list[int], scores: list[float]) -> dict[str, Any]:
    label = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if len(label) == 0 or len(label) != len(score) or not np.isin(label, [0, 1]).all():
        raise ValueError("invalid binary Prior metric arrays")
    return {
        "tokens": int(len(label)),
        "positives": int(label.sum()),
        "prevalence": float(label.mean()),
        "auroc": binary_auroc(label, score),
        "average_precision": average_precision(label, score),
        "binary_cross_entropy": _bce(labels, scores),
    }


def _mean(values: list[float]) -> float:
    if not values or not np.isfinite(values).all():
        raise ValueError("empty or non-finite mechanism metric")
    return float(np.mean(values))


def _run_metrics(accumulator: Mapping[str, list[Any]]) -> dict[str, Any]:
    return {
        "key": _binary(accumulator["key_labels"], accumulator["key_scores"]),
        "complete": _binary(
            accumulator["complete_labels"], accumulator["complete_scores"]
        ),
        "maps": {
            "key_complete_squared_l2_mean": _mean(accumulator["map_l2"]),
            "key_complete_cosine_mean": _mean(accumulator["map_cosine"]),
            "key_complete_jensen_shannon_mean": _mean(accumulator["map_js"]),
            "key_entropy_normalized_mean": _mean(accumulator["key_entropy"]),
            "complete_entropy_normalized_mean": _mean(
                accumulator["complete_entropy"]
            ),
        },
        "gate": {
            "gate_fused_prior_squared_l2_mean": _mean(accumulator["gate_l2"]),
            "gate_fused_prior_dot_mean": _mean(accumulator["gate_dot"]),
            "gate_attention_entropy_normalized_mean": _mean(
                accumulator["gate_entropy"]
            ),
            "gate_mass_on_labeled_key_union_complete_mean": _mean(
                accumulator["gate_support"]
            ),
        },
    }


@torch.no_grad()
def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    target = root / "mechanisms/prior_dev.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"mechanism report exists: {target}")
    protocol = load_protocol(protocol_path)
    completion_path = root / "training/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_PRIOR_ABLATION_V2_MATCHED_TRAINING_GRID":
        raise ValueError("v2 training grid is incomplete")
    prior_spec = protocol["frozen_parents"]["prior_dev"]
    prior_path = PROJECT_ROOT / prior_spec["path"]
    if file_sha256(prior_path) != prior_spec["file_sha256"]:
        raise ValueError("Prior dev hash drift")
    dataset = CLIRTrajectoryDataset(prior_path)
    if len(dataset) != int(prior_spec["rows"]):
        raise ValueError("Prior dev row count drift")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=0,
        pin_memory=False,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    runs = []
    for run in completion["runs"]:
        factors = [float(value) for value in run["factors"]]
        prior_on = int(any(factors[index] for index in (2, 3, 4, 5)))
        runs.append({**run, "factors": [int(factors[0]), int(factors[1]), prior_on]})
    loaded = scoring._load_models(runs, device)
    accumulators: dict[str, dict[str, list[Any]]] = {
        scoring._run_key(run): defaultdict(list) for run, _ in loaded
    }
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        for run, model in loaded:
            key = scoring._run_key(run)
            outputs = scoring._forward(model, batch, amp_dtype="bfloat16")
            scoring._require_finite(outputs, "full")
            mask = batch["mask"].bool()
            key_mask = mask & batch["key_prior_mask"].bool()
            complete_mask = mask & batch["complete_prior_mask"].bool()
            accumulator = accumulators[key]
            accumulator["key_labels"].extend(
                batch["key_prior_target"][key_mask].long().cpu().tolist()
            )
            accumulator["key_scores"].extend(
                torch.sigmoid(outputs["key_prior_logits"])[key_mask]
                .float()
                .cpu()
                .tolist()
            )
            accumulator["complete_labels"].extend(
                batch["complete_prior_target"][complete_mask].long().cpu().tolist()
            )
            accumulator["complete_scores"].extend(
                torch.sigmoid(outputs["complete_prior_logits"])[complete_mask]
                .float()
                .cpu()
                .tolist()
            )
            gate = outputs["gates"] * mask
            gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
            for offset in range(mask.shape[0]):
                length = int(mask[offset].sum().item())
                key_map = outputs["key_prior"][offset, :length].float()
                complete_map = outputs["complete_prior"][offset, :length].float()
                fused = outputs["fused_prior"][offset, :length].float()
                gate_map = gate[offset, :length].float()
                accumulator["map_l2"].append(
                    float(torch.square(key_map - complete_map).sum().cpu())
                )
                accumulator["map_cosine"].append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            key_map[None, :], complete_map[None, :]
                        ).cpu()
                    )
                )
                midpoint = 0.5 * (key_map + complete_map)
                eps = 1e-8
                js = 0.5 * (
                    torch.sum(key_map * torch.log((key_map + eps) / (midpoint + eps)))
                    + torch.sum(
                        complete_map
                        * torch.log((complete_map + eps) / (midpoint + eps))
                    )
                )
                accumulator["map_js"].append(float(js.cpu()))
                normalizer = math.log(length) if length > 1 else 1.0
                accumulator["key_entropy"].append(
                    float((-(key_map * torch.log(key_map + eps)).sum() / normalizer).cpu())
                )
                accumulator["complete_entropy"].append(
                    float(
                        (-(complete_map * torch.log(complete_map + eps)).sum() / normalizer).cpu()
                    )
                )
                accumulator["gate_l2"].append(
                    float(torch.square(gate_map - fused).sum().cpu())
                )
                accumulator["gate_dot"].append(float(torch.sum(gate_map * fused).cpu()))
                accumulator["gate_entropy"].append(
                    float((-(gate_map * torch.log(gate_map + eps)).sum() / normalizer).cpu())
                )
                support = (
                    batch["key_prior_target"][offset, :length].bool()
                    | batch["complete_prior_target"][offset, :length].bool()
                ) & (
                    batch["key_prior_mask"][offset, :length].bool()
                    & batch["complete_prior_mask"][offset, :length].bool()
                )
                accumulator["gate_support"].append(
                    float(gate_map[support].sum().cpu()) if support.any() else 0.0
                )
            del outputs
    reports = []
    for run, _ in loaded:
        reports.append(
            {
                "cell": run["cell"],
                "seed": int(run["seed"]),
                "checkpoint_file_sha256": run["checkpoint_file_sha256"],
                "metrics": _run_metrics(accumulators[scoring._run_key(run)]),
            }
        )
    scalar_paths = (
        "key.auroc",
        "key.average_precision",
        "key.binary_cross_entropy",
        "complete.auroc",
        "complete.average_precision",
        "complete.binary_cross_entropy",
        "maps.key_complete_squared_l2_mean",
        "maps.key_complete_cosine_mean",
        "maps.key_complete_jensen_shannon_mean",
        "gate.gate_fused_prior_squared_l2_mean",
        "gate.gate_fused_prior_dot_mean",
        "gate.gate_mass_on_labeled_key_union_complete_mean",
    )

    def nested(payload: Mapping[str, Any], path: str) -> float:
        value: Any = payload
        for part in path.split("."):
            value = value[part]
        return float(value)

    by_cell: dict[str, Any] = {}
    for cell in protocol["cells"]:
        rows = [row for row in reports if row["cell"] == cell]
        by_cell[cell] = {
            path: {
                "mean": float(np.mean([nested(row["metrics"], path) for row in rows])),
                "per_seed": {
                    str(row["seed"]): nested(row["metrics"], path) for row in rows
                },
            }
            for path in scalar_paths
        }
    report = {
        "schema_version": "clir-prior-ablation-v2-mechanism-report",
        "status": "PASS_PRIOR_ABLATION_V2_PRIOR_MECHANISM_EVALUATION",
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_completion_file_sha256": file_sha256(completion_path),
        "prior_dev": {
            "path": str(prior_path),
            "file_sha256": file_sha256(prior_path),
            "rows": len(dataset),
            "already_inspected_descriptive_only": True,
            "human_verified": False,
        },
        "runs": reports,
        "by_cell": by_cell,
    }
    atomic_write_json(target, report)
    print(json.dumps({**report, "runs": f"{len(reports)} runs"}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
