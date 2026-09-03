#!/usr/bin/env python
"""Preflight, train, and audit the frozen official-SWIFT baseline checkpoints.

The baseline is a single ``nn.Linear(101376, 2)`` gate/reward head over directly
concatenated all-layer hidden states, trained only on the final answer
correct/incorrect BCE.  The training budget is matched to the frozen U0
reference cell (3 epochs, batch 4, no validation split, no checkpoint
selection) so the comparison isolates model structure rather than epochs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.clir_data import EpochRandomSampler
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from src.swift_official_baseline import (
    UPSTREAM_COMMIT,
    SwiftFeatureDataset,
    SwiftLinearRewardModel,
    swift_collate,
)
from train_clir import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/swift_official_baseline_v1/protocol.json"
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/swift_official_baseline_v1"
CELL = "swift_official"
REQUIRED_BRANCH = "clir-clean-integration"
IDLE_FREE_BYTES = 70_000 * 1024**2
PREFLIGHT_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_TRAINING_PREFLIGHT"
WORKER_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_TRAINING_WORKER"
COMPLETION_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_MATCHED_TRAINING_GRID"


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


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the frozen SWIFT baseline protocol."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a JSON object")
    if payload.get("schema_version") != "swift-official-baseline-v1":
        raise ValueError("unexpected protocol schema version")
    if list(payload["cells"]) != [CELL]:
        raise ValueError("protocol must declare exactly the swift_official cell")
    training = payload["training"]
    if int(training["run_count"]) != len(training["seeds"]):
        raise ValueError("protocol run count differs from the declared seeds")
    if training["validation_split"] is not None or training["early_stopping"] is not None:
        raise ValueError("this frozen budget has no validation split or early stopping")
    if not training["fixed_epoch_no_checkpoint_selection"]:
        raise ValueError("the frozen budget must fix the epoch count")
    if payload["upstream"]["commit"] != UPSTREAM_COMMIT:
        raise ValueError("protocol upstream commit differs from the adapter")
    if payload["evaluation"]["primary_contrasts"] != ["u0_minus_swift_official"]:
        raise ValueError("protocol primary contrast drift")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _bound_training_manifest(protocol: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    spec = protocol["frozen_parents"]["training_manifest"]
    path = _resolve(spec["path"])
    rows = read_jsonl(path)
    labels = [int(row["correctness"]) for row in rows]
    if (
        file_sha256(path) != spec["file_sha256"]
        or len(rows) != int(spec["rows"])
        or len({str(row["query_id"]) for row in rows}) != int(spec["queries"])
        or sum(labels) != int(spec["correct_rows"])
        or len(labels) - sum(labels) != int(spec["incorrect_rows"])
    ):
        raise ValueError("training manifest drift")
    return path, rows


def _require_clean_commit() -> dict[str, Any]:
    state = _git_state()
    if state["dirty"] or state["branch"] != REQUIRED_BRANCH:
        raise RuntimeError(
            f"SWIFT baseline training requires a clean {REQUIRED_BRANCH} commit"
        )
    return state


def _require_idle_gpu(index: int) -> dict[str, Any]:
    """Refuse to start unless the visible GPU is genuinely idle.

    The protocol forbids preempting or competing with the current user
    workload, so a busy GPU is a hard stop rather than a warning.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("SWIFT baseline training requires a visible CUDA GPU")
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    if free_bytes < IDLE_FREE_BYTES:
        raise RuntimeError(
            "visible GPU is not idle: "
            f"{free_bytes // 1024**2} MiB free, need {IDLE_FREE_BYTES // 1024**2} MiB"
        )
    return {
        "free_mib": int(free_bytes // 1024**2),
        "total_mib": int(total_bytes // 1024**2),
        "name": torch.cuda.get_device_name(index),
    }


def _autocast(device: torch.device, amp_dtype: str):
    if amp_dtype == "bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def _model_config(protocol: Mapping[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    return {
        "feature_dim": int(model["feature_dim"]),
        "num_feature_layers": int(model["num_feature_layers"]),
        "per_layer_dim": int(model["per_layer_dim"]),
        "disable_gate": bool(model["disable_gate"]),
    }


def _training_contract(protocol: Mapping[str, Any], seed: int) -> dict[str, Any]:
    training = protocol["training"]
    return {
        "seed": int(seed),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "amp_dtype": str(training["amp_dtype"]),
        "optimizer": str(training["optimizer"]),
        "loss": str(training["loss"]),
        "validation_split": None,
        "early_stopping": None,
    }


def _checkpoint_path(root: Path, seed: int) -> Path:
    return root / f"training/{CELL}/seed-{seed}/checkpoint.pt"


def _train_one_seed(
    *,
    protocol: Mapping[str, Any],
    train_path: Path,
    train_rows: Sequence[Mapping[str, Any]],
    seed: int,
    device: torch.device,
    target: Path,
    git_state: Mapping[str, Any],
    protocol_sha: str,
) -> dict[str, Any]:
    training = protocol["training"]
    epochs = int(training["epochs"])
    amp_dtype = str(training["amp_dtype"])
    config = _model_config(protocol)

    dataset = SwiftFeatureDataset(train_path)
    if len(dataset) != len(train_rows):
        raise ValueError("dataset/manifest row-count drift")
    for row in dataset.rows:
        if int(row["feature_dim"]) != config["feature_dim"]:
            raise ValueError(f"{row['id']}: training feature width differs from protocol")
        if int(row["num_feature_layers"]) != config["num_feature_layers"]:
            raise ValueError(f"{row['id']}: training layer count differs from protocol")

    set_seed(seed)
    model = SwiftLinearRewardModel(
        config["feature_dim"], disable_gate=config["disable_gate"]
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if parameters != int(protocol["model"]["trainable_parameters"]):
        raise ValueError(
            f"trainable parameter drift: {parameters} != "
            f"{protocol['model']['trainable_parameters']}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    sampler = EpochRandomSampler(list(range(len(dataset))), seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        collate_fn=swift_collate,
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=int(training["num_workers"]) > 0,
        generator=torch.Generator().manual_seed(seed + 1_000_003),
    )

    metrics: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch - 1)
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            hidden_states = batch["hidden_states"].to(device).float()
            labels = batch["correctness"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp_dtype):
                scores = model(hidden_states, batch["lengths"])
            loss = criterion(scores.float(), labels)
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_rows += int(labels.numel())
        if total_rows != len(dataset):
            raise ValueError(f"epoch {epoch} consumed {total_rows} of {len(dataset)} rows")
        metrics.append(
            {
                "epoch": epoch,
                "correctness_bce": total_loss / total_rows,
                "rows": total_rows,
            }
        )
        print(
            f"seed {seed} epoch {epoch}/{epochs} bce={metrics[-1]['correctness_bce']:.6f}",
            flush=True,
        )

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    bad = [name for name, tensor in state_dict.items() if not bool(torch.isfinite(tensor).all())]
    if bad:
        raise FloatingPointError(f"non-finite trained tensors: {bad}")
    checkpoint = {
        "schema_version": "swift-official-baseline-v1-checkpoint",
        "cell": CELL,
        "seed": int(seed),
        "completed_epoch": epochs,
        "metrics": metrics,
        "state_dict": state_dict,
        "model_config": config,
        "trainable_parameters": parameters,
        "training_contract": _training_contract(protocol, seed),
        "data_state": {
            "train_path": str(train_path),
            "train_sha256": file_sha256(train_path),
            "train_rows": len(dataset),
            "train_queries": len({str(row["query_id"]) for row in dataset.rows}),
        },
        "run_provenance": {
            "protocol": {"path": str(DEFAULT_PROTOCOL), "sha256": protocol_sha},
            "upstream": {
                "repository": protocol["upstream"]["repository"],
                "commit": UPSTREAM_COMMIT,
                "parity_target": protocol["upstream"]["parity_target"],
            },
            "code": dict(git_state),
            "created_at_utc": _utc_now(),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return checkpoint


def _audit_checkpoint(
    protocol: Mapping[str, Any], root: Path, seed: int, protocol_sha: str
) -> dict[str, Any]:
    path = _checkpoint_path(root, seed)
    observed_hash = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = protocol["training"]
    epochs = int(training["epochs"])
    if (
        int(checkpoint.get("completed_epoch", -1)) != epochs
        or len(checkpoint.get("metrics", [])) != epochs
        or not all(
            math.isfinite(float(entry["correctness_bce"]))
            for entry in checkpoint.get("metrics", [])
        )
    ):
        raise ValueError(f"incomplete or non-finite checkpoint: seed-{seed}")
    if checkpoint.get("training_contract") != _training_contract(protocol, seed):
        raise ValueError(f"training contract drift: seed-{seed}")
    if checkpoint.get("model_config") != _model_config(protocol):
        raise ValueError(f"checkpoint model config drift: seed-{seed}")
    if int(checkpoint.get("trainable_parameters", -1)) != int(
        protocol["model"]["trainable_parameters"]
    ):
        raise ValueError(f"checkpoint parameter count drift: seed-{seed}")
    spec = protocol["frozen_parents"]["training_manifest"]
    data = checkpoint.get("data_state", {})
    if (
        data.get("train_sha256") != spec["file_sha256"]
        or int(data.get("train_rows", -1)) != int(spec["rows"])
        or int(data.get("train_queries", -1)) != int(spec["queries"])
    ):
        raise ValueError(f"checkpoint training data drift: seed-{seed}")
    provenance = checkpoint.get("run_provenance", {})
    if (
        provenance.get("protocol", {}).get("sha256") != protocol_sha
        or provenance.get("upstream", {}).get("commit") != UPSTREAM_COMMIT
        or provenance.get("code", {}).get("branch") != REQUIRED_BRANCH
        or provenance.get("code", {}).get("dirty") is not False
    ):
        raise ValueError(f"checkpoint provenance drift: seed-{seed}")
    bad = [
        name
        for name, tensor in checkpoint.get("state_dict", {}).items()
        if not bool(torch.isfinite(tensor).all())
    ]
    if bad:
        raise FloatingPointError(f"non-finite checkpoint tensors: seed-{seed} {bad[:3]}")
    return {
        "cell": CELL,
        "seed": int(seed),
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": observed_hash,
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "trainable_parameters": int(checkpoint["trainable_parameters"]),
        "final_correctness_bce": float(checkpoint["metrics"][-1]["correctness_bce"]),
    }


def command_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    target = root / "training/preflight.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError("training preflight already exists")
    git_state = _require_clean_commit()
    train_path, train_rows = _bound_training_manifest(protocol)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("full-width training preflight requires one visible CUDA GPU")
    gpu = _require_idle_gpu(device.index or 0)

    config = _model_config(protocol)
    set_seed(int(protocol["training"]["seeds"][0]))
    model = SwiftLinearRewardModel(
        config["feature_dim"], disable_gate=config["disable_gate"]
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if parameters != int(protocol["model"]["trainable_parameters"]):
        raise ValueError("preflight trainable parameter drift")

    dataset = SwiftFeatureDataset(train_path)
    batch = swift_collate([dataset[index] for index in range(int(protocol["training"]["batch_size"]))])
    hidden_states = batch["hidden_states"].to(device).float()
    labels = batch["correctness"].to(device)
    with _autocast(device, str(protocol["training"]["amp_dtype"])):
        scores = model(hidden_states, batch["lengths"])
    loss = nn.BCEWithLogitsLoss()(scores.float(), labels)
    loss.backward()
    gradients = {
        name: float(value.grad.detach().float().norm())
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    if not gradients or any(not math.isfinite(value) or value == 0.0 for value in gradients.values()):
        raise ValueError("preflight produced no finite non-zero gradient")

    report = {
        "schema_version": "swift-official-baseline-v1-training-preflight",
        "status": PREFLIGHT_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_manifest_file_sha256": file_sha256(train_path),
        "training_rows": len(train_rows),
        "code": git_state,
        "gpu": gpu,
        "upstream_commit": UPSTREAM_COMMIT,
        "trainable_parameters": parameters,
        "probe": {
            "batch_size": int(batch["correctness"].numel()),
            "lengths": [int(value) for value in batch["lengths"]],
            "scores": [float(value) for value in scores.detach().float().cpu()],
            "correctness_bce": float(loss.detach()),
            "gradient_norms": gradients,
        },
        "conditions_are_never_loaded": True,
    }
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    root = Path(args.output_root).resolve()
    preflight_path = root / "training/preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != PREFLIGHT_STATUS:
        raise ValueError("training preflight did not pass")
    if preflight.get("protocol_file_sha256") != protocol_sha:
        raise ValueError("training preflight is bound to another protocol revision")

    seeds = [int(value) for value in protocol["training"]["seeds"]]
    worker = int(args.worker_index)
    if not 0 <= worker < len(seeds):
        raise ValueError(f"worker index outside 0..{len(seeds) - 1}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each training worker must see exactly one GPU")
    git_state = _require_clean_commit()
    _require_idle_gpu(0)
    train_path, train_rows = _bound_training_manifest(protocol)

    seed = seeds[worker]
    target = _checkpoint_path(root, seed)
    device = torch.device("cuda")
    if target.exists():
        _audit_checkpoint(protocol, root, seed, protocol_sha)
        print(f"worker {worker}: verified existing seed-{seed}", flush=True)
    else:
        print(f"worker {worker}: training seed-{seed}", flush=True)
        _train_one_seed(
            protocol=protocol,
            train_path=train_path,
            train_rows=train_rows,
            seed=seed,
            device=device,
            target=target,
            git_state=git_state,
            protocol_sha=protocol_sha,
        )
        _audit_checkpoint(protocol, root, seed, protocol_sha)
        print(f"worker {worker}: completed seed-{seed}", flush=True)

    marker = {
        "schema_version": "swift-official-baseline-v1-training-worker",
        "status": WORKER_STATUS,
        "worker_index": worker,
        "seed": seed,
        "completed_at_utc": _utc_now(),
    }
    atomic_write_json(root / f"training/worker-{worker:03d}.json", marker)
    print(json.dumps(marker, ensure_ascii=False, indent=2))


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    root = Path(args.output_root).resolve()
    target = root / "training/completion.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError("training completion already exists")
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    for worker in range(len(seeds)):
        marker = json.loads(
            (root / f"training/worker-{worker:03d}.json").read_text(encoding="utf-8")
        )
        if marker.get("status") != WORKER_STATUS or int(marker["seed"]) != seeds[worker]:
            raise ValueError(f"training worker {worker} is incomplete or misbound")
    runs = [_audit_checkpoint(protocol, root, seed, protocol_sha) for seed in seeds]
    if len({run["checkpoint_file_sha256"] for run in runs}) != len(runs):
        raise ValueError("distinct seeds produced identical checkpoints")
    report = {
        "schema_version": "swift-official-baseline-v1-training-completion",
        "status": COMPLETION_STATUS,
        "completed_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "cells": [CELL],
        "seeds": seeds,
        "run_count": len(runs),
        "runs": runs,
    }
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--overwrite", action="store_true")
    preflight.set_defaults(func=command_preflight)
    worker = sub.add_parser("train-worker")
    worker.add_argument("--worker-index", required=True, type=int)
    worker.set_defaults(func=command_worker)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--overwrite", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
