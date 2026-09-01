#!/usr/bin/env python
"""Preflight and validate the fresh Prior/Gate Stage-A checkpoint set.

Stage A reuses the frozen CH and Full(.25) checkpoints and trains only the
three-seed ``direct Prior, Gate=0`` diagnostic.  This script keeps that small
training action hash-bound and then publishes the exact nine-checkpoint set
used by the fresh tuning scorer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from prepare_clir_three_module import (
    _all_state_tensors_finite,
    _git_state,
    _representative_indices,
    _run_preflight_batch,
)
from src.clir_data import CLIRTrajectoryDataset, clir_collate, read_jsonl
from src.clir_smoke import atomic_write_json, file_sha256
from src.consistency_localized_reward import ConsistencyLocalizedReward
from train_clir import (
    load_config,
    query_ids,
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/stage_a_authorization.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/stage_a"
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_STAGE_A"
PREFLIGHT_STATUS = "PASS_PRIOR_GATE_TUNING_V1_STAGE_A_PREFLIGHT"
COMPLETION_STATUS = "PASS_PRIOR_GATE_TUNING_V1_STAGE_A_9_CHECKPOINTS"
PARENT_COMPLETION_STATUS = "PASS_THREE_MODULE_COMPLETE_2X2X2_24_RUN_TRAINING"
FEATURE_COMPLETION_STATUS = "PASS_GATE_TUNING_V1_SELECTED_FEATURES"
SEEDS = (42, 43, 44)
CELLS = ("ch", "direct_gate0", "full_025")


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _assert_file(specification: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(specification["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != specification["file_sha256"]:
        raise ValueError(f"{label} hash drift: {observed}")
    if "row_count" in specification:
        if len(read_jsonl(path)) != int(specification["row_count"]):
            raise ValueError(f"{label} row-count drift")
    return path


def _validate_direct_config(path: Path) -> tuple[Any, dict[str, Any]]:
    model_config, training = load_config(path)
    expected_on = {
        "final_weight": 1.0,
        "consistency_weight": 1.0,
        "hallucination_weight": 1.0,
        "prior_weight": 1.0,
        "key_prior_weight": 1.0,
        "complete_prior_weight": 1.0,
    }
    expected_off = {
        "gate_prior_weight": 0.0,
        "token_reward_weight": 0.0,
        "tail_weight": 0.0,
        "mil_weight": 0.0,
        "pseudo_tail_weight": 0.0,
        "progress_weight": 0.0,
        "prior_distill_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    for name, expected in {**expected_on, **expected_off}.items():
        if float(getattr(model_config, name)) != expected:
            raise ValueError(f"direct Gate=0 config drift: {name}")
    expected_training = {
        "seed": 42,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "amp_dtype": "bfloat16",
        "num_workers": 4,
        "pin_memory": True,
        "group_by_semantic_id": True,
        "prior_phase_mode": "joint",
    }
    if training != expected_training:
        raise ValueError("direct Gate=0 training schedule drift")
    return model_config, training


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization_path = Path(path).resolve()
    payload = _load_json(authorization_path)
    if (
        payload.get("schema_version")
        != "clir-prior-gate-tuning-v1-stage-a-authorization"
        or payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("evidence_tier")
        != "prospective_train_source_tuning_with_sealed_confirmation"
    ):
        raise ValueError("unsupported or inactive Prior/Gate Stage-A authorization")

    frozen = payload["frozen_inputs"]
    protocol_path = _assert_file(frozen["protocol"], "protocol")
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema_version") != "clir-prior-gate-tuning-v1"
        or protocol.get("attribution_and_tuning", {}).get("seeds") != list(SEEDS)
    ):
        raise ValueError("Prior/Gate protocol drift")

    feature_path = _assert_file(frozen["feature_completion"], "feature completion")
    feature = _load_json(feature_path)
    if (
        feature.get("status") != FEATURE_COMPLETION_STATUS
        or feature.get("confirmation_outcomes_opened") is not False
        or feature.get("training_allowed") is not False
    ):
        raise ValueError("selected feature completion is not sealed and complete")
    tuning_path = _assert_file(frozen["tuning_features"], "tuning feature manifest")
    if (
        feature.get("tuning_manifest", {}).get("file_sha256")
        != frozen["tuning_features"]["file_sha256"]
        or int(feature.get("tuning_manifest", {}).get("row_count", -1)) != 12_800
    ):
        raise ValueError("tuning feature completion binding drift")
    if len({str(row["query_id"]) for row in read_jsonl(tuning_path)}) != 800:
        raise ValueError("tuning feature query-count drift")

    train_path = _assert_file(frozen["train_manifest"], "training manifest")
    parent_path = _assert_file(
        frozen["parent_training_completion"], "parent training completion"
    )
    parent = _load_json(parent_path)
    if parent.get("status") != PARENT_COMPLETION_STATUS:
        raise ValueError("parent three-module training did not pass")
    parent_runs = {
        (str(run["cell"]), int(run["seed"])): run for run in parent["runs"]
    }
    for cell in ("ch", "full"):
        for seed in SEEDS:
            key = f"{cell}/seed-{seed}"
            expected = payload["reused_checkpoints"][key]
            observed = parent_runs[(cell, seed)]
            if (
                observed["checkpoint_path"] != expected["path"]
                or observed["checkpoint_file_sha256"] != expected["file_sha256"]
            ):
                raise ValueError(f"reused checkpoint registry drift: {key}")
            _assert_file(expected, f"reused checkpoint {key}")

    direct_path = _assert_file(frozen["direct_config"], "direct Gate=0 config")
    _validate_direct_config(direct_path)
    for name, specification in payload["implementation"].items():
        _assert_file(specification, f"implementation {name}")

    training = payload["training"]
    if (
        training.get("seeds") != list(SEEDS)
        or int(training.get("new_runs", -1)) != 3
        or int(training.get("epochs", -1)) != 3
        or int(training.get("train_rows", -1)) != 5_370
        or int(training.get("train_queries", -1)) != 1_493
        or training.get("supervision_per_epoch")
        != supervision_summary(CLIRTrajectoryDataset(train_path), list(range(5_370)))
    ):
        raise ValueError("Stage-A training inventory or schedule drift")
    return payload


def _assert_direct_preflight(reports: Mapping[str, Mapping[str, Any]]) -> None:
    consistency = reports["consistency"]
    hallucination = reports["hallucination"]
    prior = reports["prior"]
    if consistency["losses"].get("consistency_total", 0.0) <= 0.0:
        raise ValueError("Stage A did not exercise Consistency")
    if (consistency["objective_gradient_norms"] or {}).get("projector", 0.0) <= 0.0:
        raise ValueError("Stage A Consistency did not train the projector")
    if hallucination["losses"].get("localization_token_bce", 0.0) <= 0.0:
        raise ValueError("Stage A did not exercise H0 onset BCE")
    if (hallucination["objective_gradient_norms"] or {}).get(
        "hallucination_head", 0.0
    ) <= 0.0:
        raise ValueError("Stage A H0 did not train the hallucination head")
    for key in ("prior_key", "prior_complete", "prior_total"):
        if prior["losses"].get(key, 0.0) <= 0.0:
            raise ValueError(f"Stage A direct Prior did not exercise {key}")
    if prior["losses"].get("prior_gate") != 0.0:
        raise ValueError("Stage A diagnostic unexpectedly executed Gate loss")
    gradients = prior["objective_gradient_norms"] or {}
    for key in ("feature_encoder", "key_prior_head", "complete_prior_head"):
        if gradients.get(key, 0.0) <= 0.0:
            raise ValueError(f"Stage A direct Prior did not train {key}")
    if gradients.get("token_reward_head", 0.0) != 0.0:
        raise ValueError("Gate=0 Prior objective unexpectedly trained token reward Gate")
    for name, report in reports.items():
        total_gradients = report["total_gradient_norms"]
        if (
            total_gradients.get("feature_encoder", 0.0) <= 0.0
            or total_gradients.get("final_score_head", 0.0) <= 0.0
        ):
            raise ValueError(f"Stage A base reward gradient missing in {name}")


def command_preflight(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    authorization = load_authorization(authorization_path)
    git = _git_state(authorization["minimum_parent_commit"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device.index)

    train_path = _project_path(
        authorization["frozen_inputs"]["train_manifest"]["path"]
    )
    config_path = _project_path(
        authorization["frozen_inputs"]["direct_config"]["path"]
    )
    dataset = CLIRTrajectoryDataset(train_path)
    model_config, training = _validate_direct_config(config_path)
    validate_feature_contract(dataset, model_config, "Prior/Gate Stage A")
    summary = supervision_summary(dataset, list(range(len(dataset))))
    validate_supervision_coverage(summary, model_config)
    indices = _representative_indices(dataset)
    batches = {
        name: clir_collate([dataset[index] for index in selected])
        for name, selected in indices.items()
    }
    set_seed(int(training["seed"]))
    model = ConsistencyLocalizedReward(model_config).to(device)
    model.train()
    reports = {
        "consistency": _run_preflight_batch(
            model,
            batches["consistency"],
            device,
            str(training["amp_dtype"]),
            "consistency_total",
        ),
        "hallucination": _run_preflight_batch(
            model,
            batches["hallucination"],
            device,
            str(training["amp_dtype"]),
            "localization_token_bce",
        ),
        "prior": _run_preflight_batch(
            model,
            batches["prior"],
            device,
            str(training["amp_dtype"]),
            "prior_total",
        ),
    }
    _assert_direct_preflight(reports)
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-stage-a-preflight",
        "status": PREFLIGHT_STATUS,
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "supervision_per_epoch": summary,
        "batches": reports,
        "gate_zero_verified": True,
        "direct_prior_heads_and_shared_encoder_receive_gradients": True,
        "token_reward_gate_receives_no_prior_objective_gradient": True,
        "device": str(device),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device.index))
            if device.type == "cuda"
            else None
        ),
        "training_allowed": True,
    }
    target = _project_path(authorization["runtime"]["preflight_report"])
    if target.exists():
        raise FileExistsError(f"Stage-A preflight already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_direct_run(
    *,
    authorization: Mapping[str, Any],
    git: Mapping[str, Any],
    seed: int,
    config_path: Path,
    model_config: Any,
    configured_training: Mapping[str, Any],
) -> dict[str, Any]:
    root = _project_path(authorization["runtime"]["direct_training_root"])
    run_root = root / f"seed-{seed}"
    checkpoint_path = run_root / "checkpoint.pt"
    metrics_path = run_root / "metrics.jsonl"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete Stage-A direct run seed-{seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("completed_epoch", -1)) != 3:
        raise ValueError(f"direct seed-{seed}: incomplete epoch budget")
    if checkpoint.get("model_config") != model_config.__dict__:
        raise ValueError(f"direct seed-{seed}: model config drift")
    if not _all_state_tensors_finite(checkpoint.get("state_dict", {})):
        raise FloatingPointError(f"direct seed-{seed}: non-finite model state")
    if not _all_state_tensors_finite(checkpoint.get("optimizer_state_dict", {})):
        raise FloatingPointError(f"direct seed-{seed}: non-finite optimizer state")
    expected_contract = dict(configured_training)
    expected_contract["seed"] = seed
    expected_contract.pop("epochs")
    if checkpoint.get("training_contract") != expected_contract:
        raise ValueError(f"direct seed-{seed}: training contract drift")
    data = checkpoint["data_state"]
    frozen_train = authorization["frozen_inputs"]["train_manifest"]
    if (
        data.get("train_sha256") != frozen_train["file_sha256"]
        or int(data.get("train_rows", -1)) != 5_370
        or int(data.get("train_queries", -1)) != 1_493
        or data.get("train_supervision_per_epoch")
        != authorization["training"]["supervision_per_epoch"]
        or data.get("val_sha256") is not None
        or int(data.get("val_rows", -1)) != 0
    ):
        raise ValueError(f"direct seed-{seed}: data-state drift")
    provenance = checkpoint["run_provenance"]
    if (
        provenance.get("code", {}).get("commit") != git["commit"]
        or provenance.get("code", {}).get("dirty") is not False
        or provenance.get("config", {}).get("sha256") != file_sha256(config_path)
    ):
        raise ValueError(f"direct seed-{seed}: run provenance drift")
    metrics = read_jsonl(metrics_path)
    if len(metrics) != 3:
        raise ValueError(f"direct seed-{seed}: metric epoch-count drift")
    required = {
        "final",
        "consistency_total",
        "localization_token_bce",
        "prior_key",
        "prior_complete",
        "prior_gate",
        "prior_total",
        "total",
    }
    for row in metrics:
        values = row["train"]
        if not required <= set(values):
            raise ValueError(f"direct seed-{seed}: enabled loss missing")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError(f"direct seed-{seed}: non-finite train metric")
        if float(values["prior_gate"]) != 0.0:
            raise ValueError(f"direct seed-{seed}: Gate loss was not zero")
    return {
        "cell": "direct_gate0",
        "factors": [1, 1, 1],
        "seed": seed,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "metrics_file_sha256": file_sha256(metrics_path),
        "completed_epoch": 3,
        "final_train_total": float(metrics[-1]["train"]["total"]),
        "mechanism": {
            "direct_key_weight": 1.0,
            "direct_complete_weight": 1.0,
            "gate_prior_weight": 0.0,
        },
        "all_state_tensors_finite": True,
    }


def command_validate_training(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    authorization = load_authorization(authorization_path)
    git = _git_state(authorization["minimum_parent_commit"])
    preflight_path = _project_path(authorization["runtime"]["preflight_report"])
    preflight = _load_json(preflight_path)
    if (
        preflight.get("status") != PREFLIGHT_STATUS
        or preflight.get("git", {}).get("commit") != git["commit"]
        or preflight.get("authorization_file_sha256")
        != file_sha256(authorization_path)
        or preflight.get("training_allowed") is not True
    ):
        raise ValueError("missing or stale Stage-A preflight")

    config_path = _project_path(
        authorization["frozen_inputs"]["direct_config"]["path"]
    )
    model_config, configured_training = _validate_direct_config(config_path)
    direct_runs = [
        _validate_direct_run(
            authorization=authorization,
            git=git,
            seed=seed,
            config_path=config_path,
            model_config=model_config,
            configured_training=configured_training,
        )
        for seed in SEEDS
    ]

    parent_path = _project_path(
        authorization["frozen_inputs"]["parent_training_completion"]["path"]
    )
    parent = _load_json(parent_path)
    parent_runs = {
        (str(run["cell"]), int(run["seed"])): dict(run) for run in parent["runs"]
    }
    runs: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in SEEDS:
            if cell == "direct_gate0":
                run = next(item for item in direct_runs if item["seed"] == seed)
            else:
                parent_cell = "ch" if cell == "ch" else "full"
                run = dict(parent_runs[(parent_cell, seed)])
                run["source_parent_cell"] = parent_cell
                run["cell"] = cell
                run["mechanism"] = {
                    "direct_key_weight": 0.0 if cell == "ch" else 1.0,
                    "direct_complete_weight": 0.0 if cell == "ch" else 1.0,
                    "gate_prior_weight": 0.0 if cell == "ch" else 0.25,
                }
            runs.append(run)
    if {(run["cell"], run["seed"]) for run in runs} != {
        (cell, seed) for cell in CELLS for seed in SEEDS
    }:
        raise ValueError("Stage-A nine-checkpoint grid drift")
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-stage-a-completion",
        "status": COMPLETION_STATUS,
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "preflight_file_sha256": file_sha256(preflight_path),
        "feature_completion_file_sha256": authorization["frozen_inputs"][
            "feature_completion"
        ]["file_sha256"],
        "tuning_input_sha256": authorization["frozen_inputs"]["tuning_features"][
            "file_sha256"
        ],
        "runs": runs,
        "cells": list(CELLS),
        "seeds": list(SEEDS),
        "new_training_runs": 3,
        "reused_training_runs": 6,
        "same_frozen_training_manifest_for_all_checkpoints": True,
        "all_checkpoints_load_and_are_finite": True,
        "confirmation_outcomes_opened": False,
        "mechanism_evaluation_allowed": False,
        "tuning_scoring_allowed": False,
        "next_gate": "separate_hash_bound_stage_a_tuning_scoring_authorization",
    }
    target = _project_path(authorization["runtime"]["completion_report"])
    if target.exists():
        raise FileExistsError(f"Stage-A completion already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--device", default="cuda")
    subparsers.add_parser("validate-training")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        command_preflight(args)
    elif args.command == "validate-training":
        command_validate_training(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
