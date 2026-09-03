#!/usr/bin/env python
"""Preflight, train, and audit the matched Prior-ablation-v2 checkpoint grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from prepare_clir_prior_ablation_v2 import load_protocol
from prepare_clir_prior_v16_training import _representative_batches, _run_batch
from src.clir_data import CLIRTrajectoryDataset
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from src.consistency_localized_reward import ConsistencyLocalizedReward
from train_clir import (
    apply_training_overrides,
    load_config,
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2"
RUNTIME_AMENDMENT = (
    PROJECT_ROOT / "configs/prior_ablation_v2/runtime_amendment_v1.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _validate_runtime_amendment(
    *,
    plan: Mapping[str, Any],
    protocol_path: Path,
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    if state["commit"] == plan["code_commit"]:
        return None
    amendment = json.loads(RUNTIME_AMENDMENT.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version")
        != "clir-prior-ablation-v2-runtime-amendment-v1"
        or amendment.get("status")
        != "AUTHORIZED_POST_ROLLOUT_IMPLEMENTATION_REPAIR_BEFORE_CLIR_TRAINING"
        or amendment.get("base_code_commit") != plan["code_commit"]
        or amendment.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("Prior-ablation runtime amendment is missing or stale")
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "diff",
            "--name-only",
            f"{plan['code_commit']}..{state['commit']}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(changed) != set(amendment["allowed_changed_paths"]):
        raise ValueError("runtime amendment changed-path allowlist mismatch")
    for relative, expected_hash in amendment["runtime_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError(f"runtime amendment file hash drift: {relative}")
    return {
        "path": str(RUNTIME_AMENDMENT),
        "file_sha256": file_sha256(RUNTIME_AMENDMENT),
        "base_code_commit": plan["code_commit"],
        "runtime_code_commit": state["commit"],
    }


def _training_plan(protocol_path: Path, root: Path):
    protocol = load_protocol(protocol_path)
    plan_path = root / "training_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registry = json.loads(
        (root / "pre_rollout/manifest_registry.json").read_text(encoding="utf-8")
    )
    if (
        plan.get("schema_version") != "clir-prior-ablation-v2-training-plan"
        or plan.get("protocol_file_sha256") != file_sha256(protocol_path)
        or plan.get("code_commit") != registry.get("code_commit")
    ):
        raise ValueError("training plan is missing or stale")
    state = _git_state()
    if state["dirty"] or state["branch"] != "clir-clean-integration":
        raise RuntimeError("v2 training requires a clean clir-clean-integration commit")
    runtime_amendment = _validate_runtime_amendment(
        plan=plan,
        protocol_path=protocol_path,
        state=state,
    )
    for cell, record in plan["configs"].items():
        path = Path(record["path"])
        if file_sha256(path) != record["file_sha256"]:
            raise ValueError(f"training config hash drift: {cell}")
    train_spec = protocol["frozen_parents"]["training_manifest"]
    train_path = PROJECT_ROOT / train_spec["path"]
    rows = read_jsonl(train_path)
    if (
        file_sha256(train_path) != train_spec["file_sha256"]
        or len(rows) != int(train_spec["rows"])
        or len({str(row["query_id"]) for row in rows}) != int(train_spec["queries"])
    ):
        raise ValueError("training manifest drift")
    return protocol, plan, train_path, runtime_amendment


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _checkpoint_path(
    protocol: Mapping[str, Any], plan: Mapping[str, Any], root: Path, cell: str, seed: int
) -> tuple[Path, str | None]:
    anchors = protocol["immutable_anchor_checkpoints"]
    if cell in anchors:
        path, expected = anchors[cell][str(seed)]
        return PROJECT_ROOT / path, str(expected)
    jobs = [
        job
        for job in plan["jobs"]
        if job["cell"] == cell and int(job["seed"]) == seed
    ]
    if len(jobs) != 1:
        raise ValueError(f"missing training job: {cell}/{seed}")
    return Path(jobs[0]["checkpoint_path"]), None


def _audit_checkpoint(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    root: Path,
    cell: str,
    seed: int,
) -> dict[str, Any]:
    path, immutable_hash = _checkpoint_path(protocol, plan, root, cell, seed)
    observed_hash = file_sha256(path)
    if immutable_hash is not None and observed_hash != immutable_hash:
        raise ValueError(f"immutable checkpoint hash drift: {cell}/{seed}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = protocol["training"]
    if (
        int(checkpoint.get("completed_epoch", -1)) != int(training["epochs"])
        or len(checkpoint.get("metrics", [])) != int(training["epochs"])
        or not _all_finite(checkpoint.get("metrics", []))
    ):
        raise ValueError(f"incomplete or non-finite checkpoint: {cell}/{seed}")
    contract = checkpoint.get("training_contract", {})
    expected_contract = {
        "seed": seed,
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "amp_dtype": str(training["amp_dtype"]),
        "prior_phase_mode": str(training["prior_phase_mode"]),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError(f"training contract drift: {cell}/{seed}")
    train = protocol["frozen_parents"]["training_manifest"]
    data = checkpoint.get("data_state", {})
    if (
        data.get("train_sha256") != train["file_sha256"]
        or int(data.get("train_rows", -1)) != int(train["rows"])
        or int(data.get("train_queries", -1)) != int(train["queries"])
    ):
        raise ValueError(f"checkpoint training data drift: {cell}/{seed}")
    config_record = plan["configs"][cell]
    config = json.loads(Path(config_record["path"]).read_text(encoding="utf-8"))
    if checkpoint.get("model_config") != config["model"]:
        raise ValueError(f"checkpoint model config drift: {cell}/{seed}")
    provenance = checkpoint.get("run_provenance", {})
    if (
        provenance.get("config", {}).get("sha256") != config_record["file_sha256"]
        or provenance.get("code", {}).get("branch") != "clir-clean-integration"
        or provenance.get("code", {}).get("dirty") is not False
    ):
        raise ValueError(f"checkpoint provenance drift: {cell}/{seed}")
    bad = [
        name
        for name, tensor in checkpoint.get("state_dict", {}).items()
        if not bool(torch.isfinite(tensor).all())
    ]
    if bad:
        raise FloatingPointError(f"non-finite checkpoint tensors: {cell}/{seed} {bad[:3]}")
    return {
        "cell": cell,
        "seed": seed,
        "factors": list(protocol["cells"][cell]),
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": observed_hash,
        "config_path": config_record["path"],
        "config_file_sha256": config_record["file_sha256"],
        "completed_epoch": int(checkpoint["completed_epoch"]),
    }


def command_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    protocol, plan, train_path, runtime_amendment = _training_plan(
        protocol_path, root
    )
    target = root / "training/preflight.json"
    if target.exists():
        raise FileExistsError("training preflight already exists")
    dataset = CLIRTrajectoryDataset(train_path)
    summary = supervision_summary(dataset, list(range(len(dataset))))
    batches = _representative_batches(dataset, stage=2)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-width training preflight requires one visible CUDA GPU")
    free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
    if free_bytes < 70_000 * 1024**2:
        raise RuntimeError("visible GPU is not idle enough for the frozen preflight")
    reports: dict[str, Any] = {}
    for cell in ("k", "complete", "ch_kcmg"):
        config_path = Path(plan["configs"][cell]["path"])
        model_config, configured = load_config(config_path)
        training = apply_training_overrides(
            configured,
            argparse.Namespace(
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
            ),
        )
        validate_feature_contract(dataset, model_config, f"{cell}-preflight")
        validate_supervision_coverage(summary, model_config)
        set_seed(int(training["seed"]))
        model = ConsistencyLocalizedReward(model_config).to(device)
        cell_reports = {
            name: _run_batch(model, batch, device, str(training["amp_dtype"]))
            for name, batch in batches.items()
        }
        prior_losses = cell_reports["prior"]["losses"]
        if cell == "k" and not (
            prior_losses["prior_key"] > 0.0
            and prior_losses["prior_complete"] == 0.0
            and cell_reports["prior"]["gradient_norms"]["key_prior_head"] > 0.0
            and cell_reports["prior"]["gradient_norms"]["complete_prior_head"] == 0.0
        ):
            raise ValueError("Key-only preflight routed the wrong head")
        if cell == "complete" and not (
            prior_losses["prior_complete"] > 0.0
            and prior_losses["prior_key"] == 0.0
            and cell_reports["prior"]["gradient_norms"]["complete_prior_head"] > 0.0
            and cell_reports["prior"]["gradient_norms"]["key_prior_head"] == 0.0
        ):
            raise ValueError("Complete-only preflight routed the wrong head")
        if cell == "ch_kcmg" and not all(
            prior_losses[name] > 0.0
            for name in (
                "prior_key",
                "prior_complete",
                "prior_distill",
                "prior_gate",
            )
        ):
            raise ValueError("full mutual+Gate preflight missed a Prior loss")
        reports[cell] = {"batches": cell_reports, "passed": True}
        del model
        torch.cuda.empty_cache()
    report = {
        "schema_version": "clir-prior-ablation-v2-training-preflight",
        "status": "PASS_PRIOR_ABLATION_V2_FULL_WIDTH_TRAINING_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_plan_file_sha256": file_sha256(root / "training_plan.json"),
        "training_manifest_file_sha256": file_sha256(train_path),
        "runtime_amendment": runtime_amendment,
        "supervision_per_epoch": summary,
        "cells": reports,
    }
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    protocol, plan, train_path, _ = _training_plan(protocol_path, root)
    preflight = json.loads((root / "training/preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_PRIOR_ABLATION_V2_FULL_WIDTH_TRAINING_PREFLIGHT":
        raise ValueError("training preflight did not pass")
    worker = int(args.worker_index)
    if not 0 <= worker < 8:
        raise ValueError("worker index outside 0..7")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each training worker must see exactly one GPU")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 70_000 * 1024**2:
        raise RuntimeError("visible GPU is not idle enough to start training")
    jobs = [job for job in plan["jobs"] if int(job["worker_index"]) == worker]
    for number, job in enumerate(jobs, start=1):
        cell, seed = str(job["cell"]), int(job["seed"])
        checkpoint_path = Path(job["checkpoint_path"])
        if checkpoint_path.exists():
            _audit_checkpoint(protocol, plan, root, cell, seed)
            print(f"worker {worker}: verified existing {cell}/seed-{seed}", flush=True)
            continue
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = checkpoint_path.parent / "train.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_clir.py"),
            "--train_jsonl",
            str(train_path),
            "--config",
            str(job["config_path"]),
            "--output_model",
            str(checkpoint_path),
            "--device",
            "cuda",
            "--seed",
            str(seed),
        ]
        print(
            f"worker {worker}: training {number}/{len(jobs)} {cell}/seed-{seed}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        _audit_checkpoint(protocol, plan, root, cell, seed)
        print(f"worker {worker}: completed {cell}/seed-{seed}", flush=True)
    marker = {
        "schema_version": "clir-prior-ablation-v2-training-worker",
        "status": "PASS_PRIOR_ABLATION_V2_TRAINING_WORKER",
        "worker_index": worker,
        "jobs": len(jobs),
        "completed_at_utc": _utc_now(),
    }
    atomic_write_json(root / f"training/worker-{worker:03d}.json", marker)
    print(json.dumps(marker, ensure_ascii=False, indent=2))


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    protocol, plan, _, runtime_amendment = _training_plan(protocol_path, root)
    target = root / "training/completion.json"
    if target.exists():
        raise FileExistsError("training completion already exists")
    for worker in range(8):
        marker = json.loads(
            (root / f"training/worker-{worker:03d}.json").read_text(encoding="utf-8")
        )
        if marker.get("status") != "PASS_PRIOR_ABLATION_V2_TRAINING_WORKER":
            raise ValueError(f"training worker {worker} is incomplete")
    runs = [
        _audit_checkpoint(protocol, plan, root, cell, int(seed))
        for cell in protocol["cells"]
        for seed in protocol["training"]["seeds"]
    ]
    report = {
        "schema_version": "clir-prior-ablation-v2-training-completion",
        "status": "PASS_PRIOR_ABLATION_V2_MATCHED_TRAINING_GRID",
        "completed_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_plan_file_sha256": file_sha256(root / "training_plan.json"),
        "runtime_amendment": runtime_amendment,
        "cells": list(protocol["cells"]),
        "seeds": list(protocol["training"]["seeds"]),
        "run_count": len(runs),
        "new_run_count": len(plan["jobs"]),
        "reused_run_count": len(runs) - len(plan["jobs"]),
        "runs": runs,
    }
    atomic_write_json(target, report)
    print(json.dumps({**report, "runs": f"{len(runs)} audited runs"}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", default="cuda:0")
    preflight.set_defaults(func=command_preflight)
    worker = sub.add_parser("train-worker")
    worker.add_argument("--worker-index", required=True, type=int)
    worker.set_defaults(func=command_worker)
    sub.add_parser("finalize").set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
