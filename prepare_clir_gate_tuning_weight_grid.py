#!/usr/bin/env python
"""Preflight and validate the selected direct-Prior weight grid."""

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
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/weight_grid_authorization.json"
)
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_GATE_TUNING_V1_DIRECT_WEIGHT_GRID"
PREFLIGHT_STATUS = "PASS_PRIOR_GATE_TUNING_V1_DIRECT_WEIGHT_GRID_PREFLIGHT"
COMPLETION_STATUS = "PASS_PRIOR_GATE_TUNING_V1_WEIGHT_GRID_9_CHECKPOINTS"
STAGE_A_STATUS = "PASS_PRIOR_GATE_TUNING_V1_STAGE_A_9_CHECKPOINTS"
ATTRIBUTION_STATUS = "COMPLETE_PRIOR_GATE_TUNING_V1_STAGE_A_ATTRIBUTION"
SEEDS = (42, 43, 44)
NEW_CELLS = {"direct_025": 0.25, "direct_050": 0.5}
ALL_CELLS = {**NEW_CELLS, "direct_100": 1.0}


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
    if file_sha256(path) != specification["file_sha256"]:
        raise ValueError(f"{label} hash drift")
    if "row_count" in specification and len(read_jsonl(path)) != int(
        specification["row_count"]
    ):
        raise ValueError(f"{label} row-count drift")
    return path


def _validate_weight_config(path: Path, expected_weight: float) -> tuple[Any, dict]:
    model_config, training = load_config(path)
    expected = {
        "final_weight": 1.0,
        "consistency_weight": 1.0,
        "hallucination_weight": 1.0,
        "prior_weight": 1.0,
        "key_prior_weight": expected_weight,
        "complete_prior_weight": expected_weight,
        "gate_prior_weight": 0.25,
        "token_reward_weight": 0.0,
        "tail_weight": 0.0,
        "mil_weight": 0.0,
        "pseudo_tail_weight": 0.0,
        "progress_weight": 0.0,
        "prior_distill_weight": 0.0,
        "reconstruction_weight": 0.0,
    }
    for name, value in expected.items():
        if float(getattr(model_config, name)) != value:
            raise ValueError(f"weight-grid config drift: {path.name}/{name}")
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
        raise ValueError(f"weight-grid training schedule drift: {path.name}")
    return model_config, training


def load_authorization(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path).resolve())
    if (
        payload.get("schema_version")
        != "clir-prior-gate-tuning-v1-direct-weight-grid-authorization"
        or payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("evidence_tier")
        != "prospective_train_source_weight_tuning_not_confirmation"
    ):
        raise ValueError("unsupported or inactive direct-weight-grid authorization")
    frozen = payload["frozen_inputs"]
    _assert_file(frozen["protocol"], "protocol")
    attribution_path = _assert_file(frozen["attribution"], "Stage-A attribution")
    attribution = _load_json(attribution_path)
    if (
        attribution.get("status") != ATTRIBUTION_STATUS
        or attribution.get("axis_decision", {}).get("selected_tuning_axis")
        != "direct_prior"
        or attribution.get("confirmation_outcomes_opened") is not False
    ):
        raise ValueError("Stage-A result did not select the direct-Prior axis")
    stage_a_path = _assert_file(frozen["stage_a_completion"], "Stage-A completion")
    stage_a = _load_json(stage_a_path)
    if stage_a.get("status") != STAGE_A_STATUS:
        raise ValueError("Stage-A checkpoint set did not pass")
    train_path = _assert_file(frozen["train_manifest"], "training manifest")
    for cell, expected_weight in NEW_CELLS.items():
        config_path = _assert_file(frozen["configs"][cell], f"{cell} config")
        _validate_weight_config(config_path, expected_weight)
    endpoint_path = _assert_file(
        frozen["configs"]["direct_100"], "direct_100 endpoint config"
    )
    _validate_weight_config(endpoint_path, 1.0)
    stage_runs = {
        (str(run["cell"]), int(run["seed"])): run for run in stage_a["runs"]
    }
    for seed in SEEDS:
        expected = payload["reused_endpoint_checkpoints"][f"direct_100/seed-{seed}"]
        observed = stage_runs[("full_025", seed)]
        if (
            expected["path"] != observed["checkpoint_path"]
            or expected["file_sha256"] != observed["checkpoint_file_sha256"]
        ):
            raise ValueError(f"direct_100 endpoint drift: seed-{seed}")
        _assert_file(expected, f"direct_100 checkpoint seed-{seed}")
    for name, specification in payload["implementation"].items():
        _assert_file(specification, f"implementation {name}")
    training = payload["training"]
    dataset = CLIRTrajectoryDataset(train_path)
    summary = supervision_summary(dataset, list(range(len(dataset))))
    if (
        training.get("seeds") != list(SEEDS)
        or training.get("new_cells") != list(NEW_CELLS)
        or int(training.get("new_runs", -1)) != 6
        or int(training.get("epochs", -1)) != 3
        or int(training.get("train_rows", -1)) != len(dataset)
        or training.get("supervision_per_epoch") != summary
    ):
        raise ValueError("direct-weight-grid training contract drift")
    return payload


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
    dataset = CLIRTrajectoryDataset(train_path)
    summary = supervision_summary(dataset, list(range(len(dataset))))
    prior_indices = _representative_indices(dataset)["prior"]
    raw_batch = clir_collate([dataset[index] for index in prior_indices])
    reports: dict[str, Any] = {}
    for cell, expected_weight in NEW_CELLS.items():
        config_path = _project_path(
            authorization["frozen_inputs"]["configs"][cell]["path"]
        )
        model_config, training = _validate_weight_config(config_path, expected_weight)
        validate_feature_contract(dataset, model_config, cell)
        validate_supervision_coverage(summary, model_config)
        set_seed(int(training["seed"]))
        model = ConsistencyLocalizedReward(model_config).to(device)
        model.train()
        report = _run_preflight_batch(
            model, raw_batch, device, str(training["amp_dtype"]), "prior_total"
        )
        for key in ("prior_key", "prior_complete", "prior_gate", "prior_total"):
            if report["losses"].get(key, 0.0) <= 0.0:
                raise ValueError(f"{cell}: preflight did not exercise {key}")
        gradients = report["objective_gradient_norms"] or {}
        for key in (
            "feature_encoder",
            "key_prior_head",
            "complete_prior_head",
            "token_reward_head",
        ):
            if gradients.get(key, 0.0) <= 0.0:
                raise ValueError(f"{cell}: preflight did not train {key}")
        reports[cell] = {
            "direct_weight": expected_weight,
            "gate_weight": 0.25,
            **report,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-direct-grid-preflight",
        "status": PREFLIGHT_STATUS,
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "supervision_per_epoch": summary,
        "cells": reports,
        "direct_axis_only": True,
        "gate_prior_weight_fixed": 0.25,
        "confirmation_outcomes_opened": False,
        "training_allowed": True,
    }
    target = _project_path(authorization["runtime"]["preflight_report"])
    if target.exists():
        raise FileExistsError(f"weight-grid preflight exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_run(
    *,
    authorization: Mapping[str, Any],
    git: Mapping[str, Any],
    cell: str,
    weight: float,
    seed: int,
) -> dict[str, Any]:
    config_path = _project_path(
        authorization["frozen_inputs"]["configs"][cell]["path"]
    )
    model_config, configured_training = _validate_weight_config(config_path, weight)
    run_root = _project_path(authorization["runtime"]["training_root"]) / cell / f"seed-{seed}"
    checkpoint_path = run_root / "checkpoint.pt"
    metrics_path = run_root / "metrics.jsonl"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete weight-grid run {cell}/seed-{seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("completed_epoch", -1)) != 3:
        raise ValueError(f"{cell}/seed-{seed}: incomplete epoch budget")
    if checkpoint.get("model_config") != model_config.__dict__:
        raise ValueError(f"{cell}/seed-{seed}: model config drift")
    if not _all_state_tensors_finite(checkpoint.get("state_dict", {})) or not _all_state_tensors_finite(
        checkpoint.get("optimizer_state_dict", {})
    ):
        raise FloatingPointError(f"{cell}/seed-{seed}: non-finite training state")
    expected_contract = dict(configured_training)
    expected_contract["seed"] = seed
    expected_contract.pop("epochs")
    if checkpoint.get("training_contract") != expected_contract:
        raise ValueError(f"{cell}/seed-{seed}: training contract drift")
    data = checkpoint["data_state"]
    frozen_train = authorization["frozen_inputs"]["train_manifest"]
    if (
        data.get("train_sha256") != frozen_train["file_sha256"]
        or int(data.get("train_rows", -1)) != 5_370
        or int(data.get("train_queries", -1)) != 1_493
        or data.get("train_supervision_per_epoch")
        != authorization["training"]["supervision_per_epoch"]
        or data.get("val_sha256") is not None
    ):
        raise ValueError(f"{cell}/seed-{seed}: data-state drift")
    provenance = checkpoint["run_provenance"]
    if (
        provenance.get("code", {}).get("commit") != git["commit"]
        or provenance.get("code", {}).get("dirty") is not False
        or provenance.get("config", {}).get("sha256") != file_sha256(config_path)
    ):
        raise ValueError(f"{cell}/seed-{seed}: run provenance drift")
    metrics = read_jsonl(metrics_path)
    if len(metrics) != 3:
        raise ValueError(f"{cell}/seed-{seed}: metric epoch-count drift")
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
            raise ValueError(f"{cell}/seed-{seed}: enabled loss missing")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise FloatingPointError(f"{cell}/seed-{seed}: non-finite metric")
        if float(values["prior_gate"]) <= 0.0:
            raise ValueError(f"{cell}/seed-{seed}: Gate loss was not exercised")
    return {
        "cell": cell,
        "factors": [1, 1, 1],
        "seed": seed,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "metrics_file_sha256": file_sha256(metrics_path),
        "completed_epoch": 3,
        "final_train_total": float(metrics[-1]["train"]["total"]),
        "mechanism": {
            "direct_key_weight": weight,
            "direct_complete_weight": weight,
            "gate_prior_weight": 0.25,
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
        or preflight.get("authorization_file_sha256") != file_sha256(authorization_path)
        or preflight.get("training_allowed") is not True
    ):
        raise ValueError("missing or stale direct-weight-grid preflight")
    new_runs = [
        _validate_run(
            authorization=authorization,
            git=git,
            cell=cell,
            weight=weight,
            seed=seed,
        )
        for cell, weight in NEW_CELLS.items()
        for seed in SEEDS
    ]
    stage_a_path = _project_path(
        authorization["frozen_inputs"]["stage_a_completion"]["path"]
    )
    stage_a = _load_json(stage_a_path)
    endpoint = {
        int(run["seed"]): dict(run)
        for run in stage_a["runs"]
        if run["cell"] == "full_025"
    }
    runs: list[dict[str, Any]] = []
    for cell, weight in ALL_CELLS.items():
        for seed in SEEDS:
            if cell in NEW_CELLS:
                run = next(
                    item
                    for item in new_runs
                    if item["cell"] == cell and item["seed"] == seed
                )
            else:
                run = dict(endpoint[seed])
                run["source_parent_cell"] = "full_025"
                run["cell"] = cell
                run["mechanism"] = {
                    "direct_key_weight": 1.0,
                    "direct_complete_weight": 1.0,
                    "gate_prior_weight": 0.25,
                }
            runs.append(run)
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-direct-grid-completion",
        "status": COMPLETION_STATUS,
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "preflight_file_sha256": file_sha256(preflight_path),
        "attribution_file_sha256": authorization["frozen_inputs"]["attribution"][
            "file_sha256"
        ],
        "runs": runs,
        "cells": list(ALL_CELLS),
        "weights": list(ALL_CELLS.values()),
        "seeds": list(SEEDS),
        "new_training_runs": 6,
        "reused_training_runs": 3,
        "selected_axis": "direct_prior",
        "gate_prior_weight_fixed": 0.25,
        "all_checkpoints_load_and_are_finite": True,
        "confirmation_outcomes_opened": False,
        "tuning_scoring_allowed": False,
        "next_gate": "separate_hash_bound_direct_weight_grid_scoring_authorization",
    }
    target = _project_path(authorization["runtime"]["completion_report"])
    if target.exists():
        raise FileExistsError(f"weight-grid completion exists: {target}")
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
