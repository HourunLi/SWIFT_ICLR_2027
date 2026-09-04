#!/usr/bin/env python
"""Freeze-check, train, and audit the two missing sampler-factorial cells.

The completed 2x2 is::

                       EpochRandomSampler   SemanticGroupBatchSampler
    plain SWIFT        immutable anchor     new ``swift_grouped``
    U0 CLIR structure  new ``u0_random``    immutable anchor

Only the two missing cells are trained.  The manifest, optimizer budget, seeds,
and epoch-3 primary checkpoint are fixed by the protocol.  Epoch 1 and 2 are
saved solely so the optimization trajectory remains inspectable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.clir_data import (
    CLIRTrajectoryDataset,
    EpochRandomSampler,
    SemanticGroupBatchSampler,
    clir_collate,
    first_present,
    move_batch_to_device,
)
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig
from src.swift_official_baseline import (
    SwiftFeatureDataset,
    SwiftLinearRewardModel,
    swift_collate,
)
from train_clir import (
    atomic_torch_save,
    capture_rng_state,
    load_config,
    restore_rng_state,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/swift_u0_sampler_factorial_v1/protocol.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/swift_u0_sampler_factorial_v1"
REQUIRED_SCHEMA = "swift-u0-sampler-factorial-v1"
REQUIRED_STATUS = "AUTHORIZED_FROZEN_TWO_MISSING_CELLS"
REQUIRED_BRANCH = "clir-clean-integration"
NEW_CELLS = ("u0_random", "swift_grouped")
PREFLIGHT_STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_PREFLIGHT"
WORKER_STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_WORKER"
COMPLETION_STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_TRAINING"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


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


def _require_clean_commit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    state = _git_state()
    required = str(protocol["runtime"]["required_branch"])
    if state["dirty"] or state["branch"] != required:
        raise RuntimeError(f"run requires a clean committed {required} checkout")
    return state


def _assert_bound_file(spec: Mapping[str, Any], label: str) -> Path:
    path = _resolve(str(spec["path"]))
    if file_sha256(path) != spec["file_sha256"]:
        raise ValueError(f"{label} hash drift: {path}")
    return path


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _load_json(path)
    if protocol.get("schema_version") != REQUIRED_SCHEMA:
        raise ValueError("unexpected sampler-factorial protocol schema")
    if protocol.get("status") != REQUIRED_STATUS:
        raise ValueError("sampler-factorial protocol is not authorized")
    factorial = protocol["factorial"]
    if tuple(factorial["new_cells"]) != NEW_CELLS:
        raise ValueError("new-cell grid drift")
    training = protocol["training"]
    if (
        int(training["new_run_count"])
        != len(NEW_CELLS) * len(training["seeds"])
        or list(training["saved_epochs"]) != [1, 2, 3]
        or int(training["primary_epoch"]) != 3
        or protocol["evaluation"]["ranking_scored_epochs"] != [3]
    ):
        raise ValueError("training/primary-epoch contract drift")
    if protocol["evidence_boundary"]["math_hard_eval_v1_remains_sealed"] is not True:
        raise ValueError("hard-MATH seal was not preserved")

    parents = protocol["frozen_parents"]
    # Keep the structural loader usable in a code-only checkout.  Large ignored
    # artifacts are checked by preflight/scoring, while tracked configs are
    # safe to bind here.
    for key in ("u0_grouped_config", "u0_random_config", "hard_math_protocol"):
        _assert_bound_file(parents[key], key)

    grouped = _load_json(_resolve(parents["u0_grouped_config"]["path"]))
    random = _load_json(_resolve(parents["u0_random_config"]["path"]))
    expected = json.loads(json.dumps(grouped))
    expected["training"]["group_by_semantic_id"] = False
    if random != expected:
        raise ValueError("u0_random config differs from grouped U0 by more than sampler")
    return protocol


def _audit_ignored_parent_reports(protocol: Mapping[str, Any]) -> dict[str, str]:
    parents = protocol["frozen_parents"]
    keys = (
        "prior_ablation_training_completion",
        "prior_ablation_score_merge",
        "swift_training_completion",
        "swift_score_merge",
    )
    return {
        key: file_sha256(_assert_bound_file(parents[key], key)) for key in keys
    }


def _bound_training_rows(protocol: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    spec = protocol["frozen_parents"]["training_manifest"]
    path = _assert_bound_file(spec, "training manifest")
    rows = read_jsonl(path)
    labels = [int(row["correctness"]) for row in rows]
    sources = Counter(str(row["source"]) for row in rows)
    if (
        len(rows) != int(spec["rows"])
        or len({str(row["query_id"]) for row in rows}) != int(spec["queries"])
        or sum(labels) != int(spec["correct_rows"])
        or len(labels) - sum(labels) != int(spec["incorrect_rows"])
        or dict(sources) != {str(key): int(value) for key, value in spec["source_rows"].items()}
    ):
        raise ValueError("training-manifest inventory drift")
    return path, rows


def _audit_anchors(protocol: Mapping[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for cell, cell_spec in protocol["immutable_anchors"].items():
        report[cell] = {"checkpoints": {}, "scores": {}}
        for kind in ("checkpoints", "scores"):
            for seed, (raw_path, expected_hash) in cell_spec[kind].items():
                path = _resolve(raw_path)
                observed = file_sha256(path)
                if observed != expected_hash:
                    raise ValueError(f"immutable {cell} {kind} drift for seed {seed}")
                report[cell][kind][seed] = observed
    return report


def _semantic_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        semantic = first_present(
            row,
            (
                "semantic_id",
                "semantic_ids",
                "augmentation_group",
                "augmentation_group_id",
                "group_id",
            ),
        )
        if semantic is not None:
            groups[repr(semantic)].append(index)
    return dict(groups)


def _batch_audit(
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    batch_size = int(protocol["training"]["batch_size"])
    grouped_sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    grouped_sampler.set_epoch(0)
    grouped_batches = list(grouped_sampler)
    random_sampler = EpochRandomSampler(list(range(len(rows))), seed=seed)
    random_sampler.set_epoch(0)
    random_order = list(random_sampler)
    random_batches = [
        random_order[start : start + batch_size]
        for start in range(0, len(random_order), batch_size)
    ]
    expected_indices = list(range(len(rows)))
    for name, batches in (("grouped", grouped_batches), ("random", random_batches)):
        flattened = [index for batch in batches for index in batch]
        if sorted(flattened) != expected_indices or len(flattened) != len(set(flattened)):
            raise ValueError(f"{name} sampler does not consume every row exactly once")
        if len(batches) != int(protocol["sampler_contract"]["expected_batches_per_epoch"]):
            raise ValueError(f"{name} sampler step-count drift")

    semantic_groups = _semantic_groups(rows)
    expected = protocol["sampler_contract"]["manifest_semantic_inventory"]
    size_histogram = Counter(len(indices) for indices in semantic_groups.values())
    semantic_rows = sum(size_histogram[size] * size for size in size_histogram)
    if (
        len(semantic_groups) != int(expected["semantic_groups"])
        or semantic_rows != int(expected["semantic_rows"])
        or {str(key): value for key, value in size_histogram.items()}
        != {str(key): int(value) for key, value in expected["group_size_histogram"].items()}
        or len(rows) - semantic_rows != int(expected["singleton_rows_without_semantic_id"])
    ):
        raise ValueError("semantic-group inventory drift")

    def diagnostics(batches: Sequence[Sequence[int]]) -> dict[str, Any]:
        locations = {
            index: batch_index
            for batch_index, batch in enumerate(batches)
            for index in batch
        }
        colocated = sum(
            len({locations[index] for index in members}) == 1
            for members in semantic_groups.values()
        )
        positive_histogram = Counter(
            sum(int(rows[index]["correctness"]) for index in batch)
            for batch in batches
        )
        return {
            "batches": len(batches),
            "rows": sum(map(len, batches)),
            "semantic_pairs_colocated": colocated,
            "positive_count_per_batch_histogram": {
                str(key): value for key, value in sorted(positive_histogram.items())
            },
            "ordered_batch_indices_sha256": _canonical_sha256(batches),
        }

    grouped_report = diagnostics(grouped_batches)
    random_report = diagnostics(random_batches)
    if grouped_report["semantic_pairs_colocated"] != int(expected["semantic_groups"]):
        raise ValueError("grouped sampler did not colocate every semantic pair")
    return {
        "seed": seed,
        "semantic_groups": len(semantic_groups),
        "semantic_rows": semantic_rows,
        "group_size_histogram": {
            str(key): value for key, value in sorted(size_histogram.items())
        },
        "grouped": grouped_report,
        "random": random_report,
    }


def _canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gpu_inventory(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    minimum = int(
        protocol["runtime"]["launch_only_when_all_eight_gpus_have_at_least_mib_free"]
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in result.stdout.splitlines():
        index, name, free, total = (part.strip() for part in line.split(",", 3))
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "free_mib": int(free),
                "total_mib": int(total),
            }
        )
    if len(inventory) != 8:
        raise RuntimeError(f"expected eight GPUs, observed {len(inventory)}")
    busy = [gpu for gpu in inventory if gpu["free_mib"] < minimum]
    if busy:
        raise RuntimeError(f"GPUs are not idle enough for launch: {busy}")
    return inventory


def _u0_probe(train_path: Path, protocol: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    config_path = _resolve(protocol["frozen_parents"]["u0_random_config"]["path"])
    model_config, _ = load_config(config_path)
    dataset = CLIRTrajectoryDataset(train_path)
    sampler = EpochRandomSampler(list(range(len(dataset))), seed=int(protocol["training"]["seeds"][0]))
    sampler.set_epoch(0)
    indices = list(iter(sampler))[: int(protocol["training"]["batch_size"])]
    raw = clir_collate([dataset[index] for index in indices])
    batch = move_batch_to_device(raw, device)
    set_seed(int(protocol["training"]["seeds"][0]))
    model = ConsistencyLocalizedReward(model_config).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, losses = model.training_step(batch, prior_phase="joint")
        loss = losses["total"]
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or not all(math.isfinite(value) for value in gradients.values()):
        raise ValueError("U0 full-width probe produced invalid gradients")
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if parameters != int(protocol["models"]["u0_clir_structure"]["trainable_parameters"]):
        raise ValueError("U0 parameter-count drift")
    report = {
        "indices": indices,
        "loss": float(loss.detach().float().cpu()),
        "nonzero_gradient_tensors": sum(value > 0.0 for value in gradients.values()),
        "gradient_tensors": len(gradients),
        "trainable_parameters": parameters,
    }
    del model, batch, raw, loss
    torch.cuda.empty_cache()
    return report


def _swift_probe(train_path: Path, protocol: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    dataset = SwiftFeatureDataset(train_path)
    sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=int(protocol["training"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        seed=int(protocol["training"]["seeds"][0]),
    )
    sampler.set_epoch(0)
    indices = next(iter(sampler))
    batch = swift_collate([dataset[index] for index in indices])
    set_seed(int(protocol["training"]["seeds"][0]))
    model = SwiftLinearRewardModel(feature_dim=101376, disable_gate=False).to(device)
    hidden = batch["hidden_states"].to(device).float()
    labels = batch["correctness"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = model(hidden, batch["lengths"])
    loss = nn.BCEWithLogitsLoss()(scores.float(), labels)
    loss.backward()
    gradients = {
        name: float(parameter.grad.detach().float().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or not all(math.isfinite(value) and value > 0.0 for value in gradients.values()):
        raise ValueError("SWIFT full-width probe produced invalid gradients")
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if parameters != int(protocol["models"]["plain_swift"]["trainable_parameters"]):
        raise ValueError("SWIFT parameter-count drift")
    report = {
        "indices": indices,
        "lengths": batch["lengths"],
        "loss": float(loss.detach().cpu()),
        "gradient_norms": gradients,
        "trainable_parameters": parameters,
    }
    del model, hidden, labels, scores, loss
    torch.cuda.empty_cache()
    return report


def command_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    target = root / "training/preflight.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"preflight exists: {target}")
    state = _require_clean_commit(protocol)
    train_path, rows = _bound_training_rows(protocol)
    anchors = _audit_anchors(protocol)
    ignored_parent_reports = _audit_ignored_parent_reports(protocol)
    gpu = _gpu_inventory(protocol)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-width preflight requires CUDA")
    sampler = _batch_audit(
        CLIRTrajectoryDataset(train_path),
        rows,
        protocol,
        seed=int(protocol["training"]["seeds"][0]),
    )
    report = {
        "schema_version": "swift-u0-sampler-factorial-v1-preflight",
        "status": PREFLIGHT_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "code": state,
        "training_manifest_file_sha256": file_sha256(train_path),
        "immutable_anchors": anchors,
        "ignored_parent_reports": ignored_parent_reports,
        "gpu_inventory": gpu,
        "sampler_audit": sampler,
        "full_width_probes": {
            "u0_random": _u0_probe(train_path, protocol, device),
            "swift_grouped": _swift_probe(train_path, protocol, device),
        },
        "hard_math_opened": False,
    }
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _checkpoint_path(root: Path, cell: str, seed: int, epoch: int) -> Path:
    return root / f"training/{cell}/seed-{seed}/epoch-{epoch}.pt"


def _metrics_path(root: Path, cell: str, seed: int, epoch: int) -> Path:
    return root / f"training/{cell}/seed-{seed}/epoch-{epoch}.metrics.jsonl"


def _is_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _expected_contract(protocol: Mapping[str, Any], seed: int, sampler: str) -> dict[str, Any]:
    training = protocol["training"]
    return {
        "seed": int(seed),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "amp_dtype": str(training["amp_dtype"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": bool(training["pin_memory"]),
        "sampler": sampler,
        "validation_split": None,
        "early_stopping": None,
    }


def _audit_u0_checkpoint(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    seed: int,
    epoch: int,
    code_commit: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config_spec = protocol["frozen_parents"]["u0_random_config"]
    model_config, _ = load_config(_resolve(config_spec["path"]))
    contract = checkpoint.get("training_contract", {})
    expected = _expected_contract(protocol, seed, "EpochRandomSampler")
    observed = {
        "seed": contract.get("seed"),
        "batch_size": contract.get("batch_size"),
        "learning_rate": contract.get("learning_rate"),
        "weight_decay": contract.get("weight_decay"),
        "max_grad_norm": contract.get("max_grad_norm"),
        "amp_dtype": contract.get("amp_dtype"),
        "num_workers": contract.get("num_workers"),
        "pin_memory": contract.get("pin_memory"),
        "sampler": "EpochRandomSampler" if contract.get("group_by_semantic_id") is False else None,
        "validation_split": None if checkpoint["data_state"].get("val_rows") == 0 else "present",
        "early_stopping": None,
    }
    provenance_records = [checkpoint.get("run_provenance", {})] + list(
        checkpoint.get("resume_provenance", [])
    )
    if (
        int(checkpoint.get("completed_epoch", -1)) != epoch
        or len(checkpoint.get("metrics", [])) != epoch
        or checkpoint.get("model_config") != dict(model_config.__dict__)
        or observed != expected
        or checkpoint.get("data_state", {}).get("train_sha256")
        != protocol["frozen_parents"]["training_manifest"]["file_sha256"]
        or int(checkpoint.get("data_state", {}).get("train_rows", -1))
        != int(protocol["frozen_parents"]["training_manifest"]["rows"])
        or not _is_finite(checkpoint.get("metrics", []))
        or not all(
            item.get("code", {}).get("commit") == code_commit
            and item.get("code", {}).get("branch") == REQUIRED_BRANCH
            and item.get("code", {}).get("dirty") is False
            and item.get("config", {}).get("sha256") == config_spec["file_sha256"]
            for item in provenance_records
        )
    ):
        raise ValueError(f"U0 checkpoint audit failed: seed={seed} epoch={epoch}")
    bad = [
        name
        for name, tensor in checkpoint["state_dict"].items()
        if not bool(torch.isfinite(tensor).all())
    ]
    if bad:
        raise FloatingPointError(f"non-finite U0 checkpoint tensors: {bad[:3]}")
    return {
        "cell": "u0_random",
        "seed": seed,
        "epoch": epoch,
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "correctness_bce": float(checkpoint["metrics"][-1]["train"]["final"]),
        "trainable_parameters": int(protocol["models"]["u0_clir_structure"]["trainable_parameters"]),
    }


def _train_u0(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    root: Path,
    train_path: Path,
    seed: int,
    code_commit: str,
) -> list[dict[str, Any]]:
    del protocol_path
    config = _resolve(protocol["frozen_parents"]["u0_random_config"]["path"])
    snapshots: list[dict[str, Any]] = []
    previous: Path | None = None
    for epoch in protocol["training"]["saved_epochs"]:
        epoch = int(epoch)
        target = _checkpoint_path(root, "u0_random", seed, epoch)
        metrics = _metrics_path(root, "u0_random", seed, epoch)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            snapshots.append(
                _audit_u0_checkpoint(
                    target,
                    protocol=protocol,
                    seed=seed,
                    epoch=epoch,
                    code_commit=code_commit,
                )
            )
            previous = target
            continue
        if epoch > 1 and (previous is None or not previous.exists()):
            raise FileNotFoundError(f"missing resume checkpoint before epoch {epoch}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_clir.py"),
            "--train_jsonl",
            str(train_path),
            "--config",
            str(config),
            "--output_model",
            str(target),
            "--metrics_jsonl",
            str(metrics),
            "--device",
            "cuda",
            "--seed",
            str(seed),
            "--epochs",
            str(epoch),
        ]
        if previous is not None:
            command.extend(["--resume_from", str(previous)])
        log = target.parent / f"epoch-{epoch}.train.log"
        print(f"u0_random seed {seed}: training through epoch {epoch}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
        snapshots.append(
            _audit_u0_checkpoint(
                target,
                protocol=protocol,
                seed=seed,
                epoch=epoch,
                code_commit=code_commit,
            )
        )
        previous = target
    return snapshots


def _swift_contract(protocol: Mapping[str, Any], seed: int) -> dict[str, Any]:
    contract = _expected_contract(protocol, seed, "SemanticGroupBatchSampler")
    contract.update(
        {
            "budget_epochs": int(protocol["training"]["epochs"]),
            "optimizer": str(protocol["training"]["optimizer"]),
            "loss": str(protocol["training"]["loss"]),
        }
    )
    return contract


def _audit_swift_checkpoint(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    seed: int,
    epoch: int,
    code_commit: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "swift-u0-sampler-factorial-v1-checkpoint"
        or checkpoint.get("cell") != "swift_grouped"
        or int(checkpoint.get("seed", -1)) != seed
        or int(checkpoint.get("completed_epoch", -1)) != epoch
        or len(checkpoint.get("metrics", [])) != epoch
        or checkpoint.get("training_contract") != _swift_contract(protocol, seed)
        or checkpoint.get("run_provenance", {}).get("protocol_sha256") != protocol_sha
        or checkpoint.get("run_provenance", {}).get("code", {}).get("commit") != code_commit
        or checkpoint.get("run_provenance", {}).get("code", {}).get("dirty") is not False
        or checkpoint.get("data_state", {}).get("train_sha256")
        != protocol["frozen_parents"]["training_manifest"]["file_sha256"]
        or not _is_finite(checkpoint.get("metrics", []))
        or not _is_finite(checkpoint.get("state_dict", {}))
    ):
        raise ValueError(f"grouped SWIFT checkpoint audit failed: seed={seed} epoch={epoch}")
    return {
        "cell": "swift_grouped",
        "seed": seed,
        "epoch": epoch,
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "correctness_bce": float(checkpoint["metrics"][-1]["correctness_bce"]),
        "trainable_parameters": int(protocol["models"]["plain_swift"]["trainable_parameters"]),
    }


def _train_swift(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    root: Path,
    train_path: Path,
    seed: int,
    code_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protocol_sha = file_sha256(protocol_path)
    dataset = SwiftFeatureDataset(train_path)
    training = protocol["training"]
    device = torch.device("cuda")
    set_seed(seed)
    model = SwiftLinearRewardModel(feature_dim=101376, disable_gate=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    sampler = SemanticGroupBatchSampler(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=swift_collate,
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=int(training["num_workers"]) > 0,
        generator=torch.Generator().manual_seed(seed + 1_000_003),
    )
    metrics: list[dict[str, Any]] = []
    start_epoch = 1
    existing = [
        epoch
        for epoch in training["saved_epochs"]
        if _checkpoint_path(root, "swift_grouped", seed, int(epoch)).exists()
    ]
    if existing:
        contiguous = list(range(1, max(map(int, existing)) + 1))
        if sorted(map(int, existing)) != contiguous:
            raise ValueError(f"non-contiguous grouped SWIFT snapshots for seed {seed}")
        latest = max(map(int, existing))
        latest_path = _checkpoint_path(root, "swift_grouped", seed, latest)
        for epoch in contiguous:
            _audit_swift_checkpoint(
                _checkpoint_path(root, "swift_grouped", seed, epoch),
                protocol=protocol,
                protocol_sha=protocol_sha,
                seed=seed,
                epoch=epoch,
                code_commit=str(code_state["commit"]),
            )
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        metrics = list(checkpoint["metrics"])
        start_epoch = latest + 1

    model.train()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        sampler.set_epoch(epoch - 1)
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            hidden = batch["hidden_states"].to(device).float()
            labels = batch["correctness"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(hidden, batch["lengths"])
            loss = criterion(scores.float(), labels)
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(f"non-finite grouped SWIFT loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            rows = int(labels.numel())
            total_loss += float(loss.detach()) * rows
            total_rows += rows
        if total_rows != len(dataset):
            raise ValueError(f"epoch {epoch} consumed {total_rows}/{len(dataset)} rows")
        metrics.append(
            {
                "epoch": epoch,
                "correctness_bce": total_loss / total_rows,
                "rows": total_rows,
            }
        )
        target = _checkpoint_path(root, "swift_grouped", seed, epoch)
        checkpoint = {
            "schema_version": "swift-u0-sampler-factorial-v1-checkpoint",
            "cell": "swift_grouped",
            "seed": seed,
            "completed_epoch": epoch,
            "metrics": metrics,
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "model_config": {"feature_dim": 101376, "disable_gate": False},
            "trainable_parameters": int(
                protocol["models"]["plain_swift"]["trainable_parameters"]
            ),
            "training_contract": _swift_contract(protocol, seed),
            "data_state": {
                "train_path": str(train_path),
                "train_sha256": file_sha256(train_path),
                "train_rows": len(dataset),
                "train_queries": len({str(row["query_id"]) for row in dataset.rows}),
            },
            "run_provenance": {
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "code": dict(code_state),
                "created_at_utc": _utc_now(),
            },
        }
        atomic_torch_save(checkpoint, target)
        print(
            f"swift_grouped seed {seed}: epoch {epoch}/3 "
            f"bce={metrics[-1]['correctness_bce']:.6f}",
            flush=True,
        )

    return [
        _audit_swift_checkpoint(
            _checkpoint_path(root, "swift_grouped", seed, int(epoch)),
            protocol=protocol,
            protocol_sha=protocol_sha,
            seed=seed,
            epoch=int(epoch),
            code_commit=str(code_state["commit"]),
        )
        for epoch in training["saved_epochs"]
    ]


def _jobs(protocol: Mapping[str, Any]) -> list[tuple[str, int]]:
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    return [(cell, seed) for cell in NEW_CELLS for seed in seeds]


def command_worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    preflight_path = root / "training/preflight.json"
    preflight = _load_json(preflight_path)
    if (
        preflight.get("status") != PREFLIGHT_STATUS
        or preflight.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("sampler-factorial preflight is missing or stale")
    state = _require_clean_commit(protocol)
    if preflight.get("code", {}).get("commit") != state["commit"]:
        raise ValueError("training code differs from the preflight commit")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each training worker must see exactly one GPU")
    minimum = int(
        protocol["runtime"]["launch_only_when_all_eight_gpus_have_at_least_mib_free"]
    )
    free, _ = torch.cuda.mem_get_info(0)
    if free // 1024**2 < minimum:
        raise RuntimeError("visible GPU is not idle enough to start a worker")
    jobs = _jobs(protocol)
    if not 0 <= int(args.worker_index) < len(jobs):
        raise ValueError(f"worker index outside 0..{len(jobs)-1}")
    train_path, _ = _bound_training_rows(protocol)
    cell, seed = jobs[int(args.worker_index)]
    print(f"worker {args.worker_index}: {cell}/seed-{seed}", flush=True)
    if cell == "u0_random":
        snapshots = _train_u0(
            protocol=protocol,
            protocol_path=protocol_path,
            root=root,
            train_path=train_path,
            seed=seed,
            code_commit=str(state["commit"]),
        )
    else:
        snapshots = _train_swift(
            protocol=protocol,
            protocol_path=protocol_path,
            root=root,
            train_path=train_path,
            seed=seed,
            code_state=state,
        )
    marker = {
        "schema_version": "swift-u0-sampler-factorial-v1-worker",
        "status": WORKER_STATUS,
        "completed_at_utc": _utc_now(),
        "worker_index": int(args.worker_index),
        "cell": cell,
        "seed": seed,
        "snapshots": snapshots,
    }
    atomic_write_json(root / f"training/worker-{int(args.worker_index):03d}.json", marker)
    print(json.dumps({**marker, "snapshots": len(snapshots)}, ensure_ascii=False, indent=2))


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    target = root / "training/completion.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"completion exists: {target}")
    state = _require_clean_commit(protocol)
    preflight = _load_json(root / "training/preflight.json")
    if preflight.get("code", {}).get("commit") != state["commit"]:
        raise ValueError("completion code differs from preflight")
    snapshots: list[dict[str, Any]] = []
    primary_runs: list[dict[str, Any]] = []
    for worker, (cell, seed) in enumerate(_jobs(protocol)):
        marker = _load_json(root / f"training/worker-{worker:03d}.json")
        if (
            marker.get("status") != WORKER_STATUS
            or marker.get("cell") != cell
            or int(marker.get("seed", -1)) != seed
        ):
            raise ValueError(f"worker {worker} is incomplete or misbound")
        if cell == "u0_random":
            audited = [
                _audit_u0_checkpoint(
                    _checkpoint_path(root, cell, seed, int(epoch)),
                    protocol=protocol,
                    seed=seed,
                    epoch=int(epoch),
                    code_commit=str(state["commit"]),
                )
                for epoch in protocol["training"]["saved_epochs"]
            ]
        else:
            audited = [
                _audit_swift_checkpoint(
                    _checkpoint_path(root, cell, seed, int(epoch)),
                    protocol=protocol,
                    protocol_sha=file_sha256(protocol_path),
                    seed=seed,
                    epoch=int(epoch),
                    code_commit=str(state["commit"]),
                )
                for epoch in protocol["training"]["saved_epochs"]
            ]
        snapshots.extend(audited)
        primary = dict(audited[-1])
        primary["architecture"] = protocol["factorial"]["cells"][cell]["architecture"]
        primary["sampler"] = protocol["factorial"]["cells"][cell]["sampler"]
        primary_runs.append(primary)
    hashes = [record["checkpoint_file_sha256"] for record in snapshots]
    if len(hashes) != len(set(hashes)):
        raise ValueError("two saved snapshots have identical checkpoint hashes")
    report = {
        "schema_version": "swift-u0-sampler-factorial-v1-training-completion",
        "status": COMPLETION_STATUS,
        "completed_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "code": state,
        "new_cells": list(NEW_CELLS),
        "seeds": [int(value) for value in protocol["training"]["seeds"]],
        "saved_epochs": [int(value) for value in protocol["training"]["saved_epochs"]],
        "snapshot_count": len(snapshots),
        "primary_run_count": len(primary_runs),
        "snapshots": snapshots,
        "runs": primary_runs,
        "immutable_anchors": _audit_anchors(protocol),
        "hard_math_opened": False,
    }
    atomic_write_json(target, report)
    print(json.dumps({**report, "snapshots": len(snapshots)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--overwrite", action="store_true")
    preflight.set_defaults(func=command_preflight)
    worker = subparsers.add_parser("train-worker")
    worker.add_argument("--worker-index", type=int, required=True)
    worker.set_defaults(func=command_worker)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--overwrite", action="store_true")
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
