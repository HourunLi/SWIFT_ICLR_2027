#!/usr/bin/env python
"""Validate and preflight the frozen CLIR H0 v7.4 four-cell run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from src.clir_data import CLIRTrajectoryDataset, clir_collate, read_jsonl
from src.clir_smoke import atomic_write_json, file_sha256
from src.consistency_localized_reward import ConsistencyLocalizedReward
from train_clir import (
    apply_training_overrides,
    autocast_context,
    load_config,
    prepare_batch,
    query_ids,
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/ranking_expansion_v7/h0_experiment_v7_4/training_authorization.json"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_authorization(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "clir-h0-v7.4-training-authorization"
        or payload.get("status")
        != "AUTHORIZED_POSTHOC_EXPLORATORY_FOUR_CELL_TRAINING"
        or payload.get("evidence_tier")
        != "posthoc_exploratory_silver_no_human_verification"
        or payload.get("original_v7_status") != "FAIL_H0_V7_RESERVE"
    ):
        raise ValueError("unsupported, inactive, or claim-drifting authorization")
    if set(payload.get("cells", {})) != {"c0", "c1", "h0", "ch0"}:
        raise ValueError("authorization must contain exactly c0/c1/h0/ch0")
    return payload


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def _git_state(authorization: Mapping[str, Any]) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("full-width preflight requires a clean implementation commit")
    minimum = str(authorization["minimum_code_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", minimum, commit],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if ancestor.returncode:
        raise ValueError("current implementation does not descend from frozen feature code")
    return {"commit": commit, "dirty": False, "minimum_commit": minimum}


def verify_authorization_inputs(
    authorization: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, specification in authorization["frozen_inputs"].items():
        path = _project_path(str(specification["path"]))
        _assert_hash(path, str(specification["file_sha256"]), name)
        if "row_count" in specification:
            rows = read_jsonl(path)
            if len(rows) != int(specification["row_count"]):
                raise ValueError(f"{name} row-count drift")
        verified[name] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            **(
                {"row_count": int(specification["row_count"])}
                if "row_count" in specification
                else {}
            ),
        }
    finalization_path = _project_path(
        authorization["frozen_inputs"]["feature_and_data_finalization"]["path"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    if (
        finalization.get("status") != "PASS_H0_V7_4_FEATURES_AND_MATCHED_DATA"
        or not finalization.get("training_allowed")
        or finalization.get("original_v7_status") != "FAIL_H0_V7_RESERVE"
    ):
        raise ValueError("feature/data finalization does not authorize training")
    return verified


def verify_config_grid(
    authorization: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    result: dict[str, dict[str, Any]] = {}
    for cell, specification in authorization["cells"].items():
        path = _project_path(str(specification["config"]))
        _assert_hash(path, str(specification["file_sha256"]), f"{cell} config")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"model", "training"}:
            raise ValueError(f"{cell} config structure drift")
        if (
            float(payload["model"]["consistency_weight"])
            != float(specification["consistency_weight"])
            or float(payload["model"]["hallucination_weight"])
            != float(specification["hallucination_weight"])
        ):
            raise ValueError(f"{cell} enabled objective drift")
        for disabled in (
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "prior_weight",
            "progress_weight",
        ):
            if float(payload["model"][disabled]) != 0.0:
                raise ValueError(f"{cell} unexpectedly enables {disabled}")
        payloads[cell] = payload
        result[cell] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
        }

    reference: dict[str, Any] | None = None
    observed_weights: dict[str, tuple[float, float]] = {}
    for cell, payload in payloads.items():
        clone = json.loads(json.dumps(payload))
        observed_weights[cell] = (
            float(clone["model"].pop("consistency_weight")),
            float(clone["model"].pop("hallucination_weight")),
        )
        if reference is None:
            reference = clone
        elif clone != reference:
            raise ValueError("cell configs differ outside the two frozen loss weights")
    expected = {
        "c0": (0.0, 0.0),
        "c1": (1.0, 0.0),
        "h0": (0.0, 1.0),
        "ch0": (1.0, 1.0),
    }
    if observed_weights != expected:
        raise ValueError(f"four-cell objective grid drift: {observed_weights}")
    assert reference is not None
    training = reference["training"]
    frozen_training = authorization["training"]
    if (
        int(training["epochs"]) != int(frozen_training["epochs"])
        or int(training["batch_size"]) != int(frozen_training["batch_size"])
        or float(training["learning_rate"])
        != float(frozen_training["learning_rate"])
    ):
        raise ValueError("shared training hyperparameter drift")
    return result


def _representative_indices(
    dataset: CLIRTrajectoryDataset,
) -> dict[str, list[int]]:
    relations: dict[str, list[int]] = {}
    clean: list[int] = []
    positive: list[int] = []
    for index, row in enumerate(dataset.rows):
        relation = row.get("consistency_relation_id")
        if relation is not None:
            relations.setdefault(str(relation), []).append(index)
        if row.get("feature_role") == "h_train":
            onset = int(row["hallucination_onset"])
            (clean if onset == -1 else positive).append(index)
    usable_relations: list[list[int]] = []
    for indices in relations.values():
        if len(indices) != 2:
            continue
        styles = Counter(str(dataset.rows[index].get("style_id")) for index in indices)
        if styles == Counter({"relative_compact": 1, "relative_expanded": 1}):
            usable_relations.append(indices)
    if len(usable_relations) < 2:
        raise ValueError("preflight requires two valid Consistency relations")
    if len(clean) < 2 or len(positive) < 2:
        raise ValueError("preflight requires two clean and two positive H rows")
    consistency = usable_relations[0] + usable_relations[1]
    hallucination = clean[:2] + positive[:2]
    if len(set(consistency + hallucination)) != 8:
        raise ValueError("preflight batch rows unexpectedly overlap")
    return {"consistency": consistency, "hallucination": hallucination}


def _gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix) or parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        squared += float(torch.sum(value * value).cpu())
    return math.sqrt(squared)


def _run_batch(
    model: ConsistencyLocalizedReward,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    batch = prepare_batch(dict(raw_batch), device, amp_dtype)
    with autocast_context(device, amp_dtype):
        outputs, losses = model.training_step(batch, prior_phase="joint")
        total = losses["total"]
    if not torch.isfinite(total):
        raise FloatingPointError("preflight total loss is non-finite")
    total.backward()
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise FloatingPointError(
            "preflight non-finite gradients: " + ", ".join(nonfinite_gradients[:8])
        )
    finite_outputs = {
        key: bool(torch.isfinite(outputs[key]).all().detach().cpu())
        for key in (
            "scores",
            "representations",
            "token_rewards",
            "hallucination_logits",
            "gates",
        )
    }
    if not all(finite_outputs.values()):
        raise FloatingPointError("preflight model outputs are non-finite")
    loss_values = {
        key: float(value.detach().float().cpu()) for key, value in losses.items()
    }
    if any(not math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError("preflight loss component is non-finite")
    return {
        "row_ids": list(raw_batch["ids"]),
        "hidden_shape": list(raw_batch["hidden_states"].shape),
        "condition_shape": list(raw_batch["condition_states"].shape),
        "losses": loss_values,
        "finite_outputs": finite_outputs,
        "gradient_norms": {
            "feature_encoder": _gradient_norm(model, "feature_encoder."),
            "projector": _gradient_norm(model, "projector."),
            "hallucination_head": _gradient_norm(model, "hallucination_head."),
            "final_score_head": _gradient_norm(model, "final_score_head."),
        },
    }


def _assert_objective_routing(
    report: Mapping[str, Any], consistency_enabled: bool, h0_enabled: bool
) -> None:
    c_batch = report["batches"]["consistency"]
    h_batch = report["batches"]["hallucination"]
    c_has_loss = "consistency_total" in c_batch["losses"]
    h_has_loss = "localization_token_bce" in h_batch["losses"]
    if c_has_loss != consistency_enabled:
        raise ValueError("Consistency loss routing does not match the frozen cell")
    if h_has_loss != h0_enabled:
        raise ValueError("H0 loss routing does not match the frozen cell")
    if (c_batch["gradient_norms"]["projector"] > 0.0) != consistency_enabled:
        raise ValueError("projector gradient does not match Consistency enablement")
    if (h_batch["gradient_norms"]["hallucination_head"] > 0.0) != h0_enabled:
        raise ValueError("hallucination-head gradient does not match H0 enablement")
    if "localization_token_bce" in c_batch["losses"]:
        raise ValueError("non-H preflight rows unexpectedly activate H0")
    if "consistency_total" in h_batch["losses"]:
        raise ValueError("H-only preflight rows unexpectedly activate Consistency")
    for batch in (c_batch, h_batch):
        if batch["gradient_norms"]["feature_encoder"] <= 0.0:
            raise ValueError("shared feature encoder did not receive gradients")
        if batch["gradient_norms"]["final_score_head"] <= 0.0:
            raise ValueError("final score head did not receive correctness gradients")


def preflight(
    authorization_path: Path,
    cell: str,
    device_name: str,
    output_json: Path | None,
) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    verified_inputs = verify_authorization_inputs(authorization)
    configs = verify_config_grid(authorization)
    git = _git_state(authorization)
    cell_specification = authorization["cells"][cell]
    config_path = _project_path(cell_specification["config"])
    model_config, configured_training = load_config(config_path)
    override_args = argparse.Namespace(
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        max_grad_norm=None,
        seed=None,
        num_workers=None,
        pin_memory=None,
        amp_dtype=None,
        group_by_semantic_id=None,
        prior_phase_mode=None,
    )
    training = apply_training_overrides(configured_training, override_args)
    set_seed(int(training["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    cuda_index: int | None = None
    if device.type == "cuda":
        torch.empty(1, device=device)
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_index)

    train_path = Path(verified_inputs["train"]["path"])
    h_dev_path = Path(verified_inputs["h_dev"]["path"])
    dataset = CLIRTrajectoryDataset(train_path)
    h_dev_rows = read_jsonl(h_dev_path)
    validate_feature_contract(dataset, model_config, "h0-v7.4-preflight-train")
    summary = supervision_summary(dataset, list(range(len(dataset))))
    validate_supervision_coverage(summary, model_config)
    if (
        summary["rows"] != 5168
        or summary["correctness_rows"] != 5168
        or summary["consistency_rows"] != 800
        or summary["consistency_positive_pairs"] != 400
        or summary["onset_rows"] != 400
    ):
        raise ValueError(f"shared supervision inventory drift: {summary}")
    train_queries = query_ids(dataset)
    dev_queries = {str(row["query_id"]) for row in h_dev_rows}
    if train_queries & dev_queries:
        raise ValueError("H train/dev query overlap")
    if Counter(
        "clean" if int(row["hallucination_onset"]) == -1 else "hallucinated"
        for row in h_dev_rows
    ) != Counter({"clean": 100, "hallucinated": 100}):
        raise ValueError("H dev class balance drift")

    indices = _representative_indices(dataset)
    raw_batches = {
        name: clir_collate([dataset[index] for index in selected])
        for name, selected in indices.items()
    }
    model = ConsistencyLocalizedReward(model_config).to(device)
    model.train()
    batch_reports = {
        name: _run_batch(
            model,
            raw_batch,
            device,
            str(training["amp_dtype"]),
        )
        for name, raw_batch in raw_batches.items()
    }
    report: dict[str, Any] = {
        "schema_version": "clir-h0-v7.4-four-cell-full-width-preflight",
        "status": "PASS_H0_V7_4_FULL_WIDTH_CELL_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "cell": cell,
        "evidence_tier": authorization["evidence_tier"],
        "original_v7_status": authorization["original_v7_status"],
        "git": git,
        "authorization": {
            "path": str(authorization_path.resolve()),
            "file_sha256": file_sha256(authorization_path),
        },
        "config": configs[cell],
        "enabled": {
            "consistency": bool(float(cell_specification["consistency_weight"])),
            "h0_onset_bce": bool(float(cell_specification["hallucination_weight"])),
        },
        "shared_train_manifest": verified_inputs["train"],
        "shared_h_dev_manifest": verified_inputs["h_dev"],
        "supervision_per_epoch": summary,
        "batches": batch_reports,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None
        ),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None
            else None
        ),
        "training_allowed_for_cell": True,
    }
    _assert_objective_routing(
        report,
        consistency_enabled=report["enabled"]["consistency"],
        h0_enabled=report["enabled"]["h0_onset_bce"],
    )
    output_root = _project_path(authorization["runtime"]["output_root"])
    target = output_json or output_root / f"training_preflight/{cell}.json"
    if target.exists():
        raise FileExistsError(f"preflight report already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def finalize_preflight(authorization_path: Path) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    verify_authorization_inputs(authorization)
    verify_config_grid(authorization)
    git = _git_state(authorization)
    output_root = _project_path(authorization["runtime"]["output_root"])
    reports: dict[str, dict[str, Any]] = {}
    for cell in authorization["cells"]:
        path = output_root / f"training_preflight/{cell}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "PASS_H0_V7_4_FULL_WIDTH_CELL_PREFLIGHT"
            or payload.get("cell") != cell
            or not payload.get("training_allowed_for_cell")
            or payload.get("git", {}).get("commit") != git["commit"]
            or payload.get("authorization", {}).get("file_sha256")
            != file_sha256(authorization_path)
        ):
            raise ValueError(f"invalid or stale {cell} preflight report")
        reports[cell] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "peak_cuda_memory_allocated_bytes": payload[
                "peak_cuda_memory_allocated_bytes"
            ],
        }
    gate = {
        "schema_version": "clir-h0-v7.4-four-cell-training-gate",
        "status": "PASS_H0_V7_4_FOUR_CELL_TRAINING_GATE",
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "preflights": reports,
        "cells": list(authorization["cells"]),
        "seeds": list(authorization["training"]["seeds"]),
        "training_runs": len(authorization["cells"])
        * len(authorization["training"]["seeds"]),
        "training_allowed": True,
    }
    target = output_root / "training_preflight/training_gate.json"
    if target.exists():
        raise FileExistsError(f"training gate already exists: {target}")
    atomic_write_json(target, gate)
    return gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight the frozen post-hoc exploratory H0 four-cell run."
    )
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--cell", required=True, choices=["c0", "c1", "h0", "ch0"]
    )
    preflight_parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda", "mps"]
    )
    preflight_parser.add_argument("--output-json", default=None)
    subparsers.add_parser("finalize-preflight")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    authorization_path = Path(args.authorization).resolve()
    if args.command == "preflight":
        report = preflight(
            authorization_path,
            args.cell,
            args.device,
            Path(args.output_json).resolve() if args.output_json else None,
        )
    elif args.command == "finalize-preflight":
        report = finalize_preflight(authorization_path)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
