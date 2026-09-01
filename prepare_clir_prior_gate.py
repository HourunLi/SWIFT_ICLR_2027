#!/usr/bin/env python
"""Preflight and validate the fixed-.25 Prior-to-reward Gate replication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from src.clir_data import CLIRTrajectoryDataset, clir_collate, read_jsonl
from src.clir_smoke import atomic_write_json, file_sha256
from src.consistency_localized_reward import ConsistencyLocalizedReward
from train_clir import (
    apply_training_overrides,
    autocast_context,
    load_config,
    prepare_batch,
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v12/posthoc_v1/gate_v1/protocol.json"
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


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def _git_state(protocol: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ValueError("Gate preflight/training validation requires a clean commit")
    minimum = str(protocol["implementation"]["minimum_parent_commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", minimum, commit],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode:
        raise ValueError("current commit does not descend from the frozen parent")
    return {"commit": commit, "dirty": False, "minimum_parent_commit": minimum}


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "clir-prior-v12-posthoc-gate-v1"
        or protocol.get("status") != "AUTHORIZED_FIXED_025_P0_PG0_REPLICATION"
        or protocol.get("evidence_tier")
        != "posthoc_exploratory_silver_no_human_verification"
    ):
        raise ValueError("unsupported, inactive, or claim-drifting Gate protocol")
    terminal = protocol["terminal_statuses_preserved"]
    if terminal != {
        "prior_v12": "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE",
        "prior_v13": "FAIL_PRIOR_V13_SCHEMA",
    }:
        raise ValueError("original v12/v13 terminal status drift")
    for name, specification in protocol["frozen_inputs"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
        if "row_count" in specification:
            if len(read_jsonl(source)) != int(specification["row_count"]):
                raise ValueError(f"{name} row-count drift")
    verify_config_pair(protocol)
    return protocol


def verify_config_pair(protocol: Mapping[str, Any]) -> dict[str, Any]:
    cells = protocol["cells"]
    if set(cells) != {"p0", "pg0"}:
        raise ValueError("Gate grid must contain exactly P0 and PG0")
    payloads: dict[str, dict[str, Any]] = {}
    observed: dict[str, Any] = {}
    for cell in ("p0", "pg0"):
        specification = cells[cell]
        path = _project_path(specification["config"])
        _assert_hash(path, specification["file_sha256"], f"{cell} config")
        payload = json.loads(path.read_text(encoding="utf-8"))
        weight = float(payload["model"].get("gate_prior_weight", -1))
        if weight != float(specification["gate_prior_weight"]):
            raise ValueError(f"{cell} Gate weight drift")
        if float(payload["model"]["prior_weight"]) != 1.0:
            raise ValueError(f"{cell} must retain direct Prior supervision")
        for disabled in (
            "consistency_weight",
            "hallucination_weight",
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "progress_weight",
            "prior_distill_weight",
            "reconstruction_weight",
        ):
            if float(payload["model"][disabled]) != 0.0:
                raise ValueError(f"{cell} unexpectedly enables {disabled}")
        payloads[cell] = payload
        observed[cell] = {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
    normalized = []
    for cell in ("p0", "pg0"):
        clone = json.loads(json.dumps(payloads[cell]))
        clone["model"].pop("gate_prior_weight")
        normalized.append(clone)
    if normalized[0] != normalized[1]:
        raise ValueError("P0/PG0 configs differ outside gate_prior_weight")
    if (
        float(payloads["p0"]["model"]["gate_prior_weight"]) != 0.0
        or float(payloads["pg0"]["model"]["gate_prior_weight"]) != 0.25
    ):
        raise ValueError("P0/PG0 must freeze Gate weights at 0/.25")
    return observed


def _gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            value = parameter.grad.detach().float()
            squared += float(torch.sum(value * value).cpu())
    return math.sqrt(squared)


def preflight(
    protocol_path: Path, device_name: str, output_json: Path | None
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    git = _git_state(protocol)
    config_path = _project_path(protocol["cells"]["pg0"]["config"])
    model_config, configured_training = load_config(config_path)
    training = apply_training_overrides(
        configured_training,
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
    set_seed(int(training["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.empty(1, device=device)
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    else:
        cuda_index = None

    train_path = _project_path(protocol["frozen_inputs"]["shared_train"]["path"])
    dev_path = _project_path(protocol["frozen_inputs"]["prior_dev"]["path"])
    dataset = CLIRTrajectoryDataset(train_path)
    validate_feature_contract(dataset, model_config, "prior-gate-v1-train")
    summary = supervision_summary(dataset, list(range(len(dataset))))
    validate_supervision_coverage(summary, model_config)
    expected = protocol["training"]
    required = {
        "rows": int(expected["train_rows"]),
        "correctness_rows": int(expected["train_rows"]),
        "paired_prior_rows": int(expected["prior_rows"]),
        "key_prior_tokens": int(expected["prior_target_tokens"]),
        "complete_prior_tokens": int(expected["prior_target_tokens"]),
    }
    for field, value in required.items():
        if int(summary[field]) != value:
            raise ValueError(f"supervision {field} drift: {summary[field]} != {value}")
    train_queries = {str(row["query_id"]) for row in dataset.rows}
    dev_rows = read_jsonl(dev_path)
    if train_queries & {str(row["query_id"]) for row in dev_rows}:
        raise ValueError("Prior train/dev query overlap")

    prior_indices = [
        index
        for index, row in enumerate(dataset.rows)
        if row.get("key_prior_target") is not None
        and row.get("complete_prior_target") is not None
    ]
    prior_indices.sort(
        key=lambda index: len(dataset.rows[index]["output_token_ids"]), reverse=True
    )
    if len(prior_indices) < 4:
        raise ValueError("need four paired-Prior rows for Gate preflight")
    raw_batch = clir_collate([dataset[index] for index in prior_indices[:4]])
    batch = prepare_batch(raw_batch, device, str(training["amp_dtype"]))
    model = ConsistencyLocalizedReward(model_config).to(device)
    model.train()
    with autocast_context(device, str(training["amp_dtype"])):
        outputs, losses = model.training_step(batch, prior_phase="joint")
    total = losses["total"]
    if not torch.isfinite(total):
        raise FloatingPointError("Gate preflight total is non-finite")
    total.backward()
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite:
        raise FloatingPointError("non-finite gradients: " + ", ".join(nonfinite[:8]))
    loss_values = {
        key: float(value.detach().float().cpu()) for key, value in losses.items()
    }
    if loss_values.get("prior_gate", 0.0) <= 0.0:
        raise ValueError("fixed-.25 PG0 preflight did not execute a positive Gate loss")
    gradient_norms = {
        "feature_encoder": _gradient_norm(model, "feature_encoder."),
        "token_reward_head": _gradient_norm(model, "token_reward_head."),
        "key_prior_head": _gradient_norm(model, "key_prior_head."),
        "complete_prior_head": _gradient_norm(model, "complete_prior_head."),
        "final_score_head": _gradient_norm(model, "final_score_head."),
    }
    if any(value <= 0.0 for value in gradient_norms.values()):
        raise ValueError(f"expected PG0 gradients are missing: {gradient_norms}")
    finite_outputs = {
        field: bool(torch.isfinite(outputs[field]).all().detach().cpu())
        for field in ("scores", "gates", "fused_prior", "key_prior", "complete_prior")
    }
    if not all(finite_outputs.values()):
        raise FloatingPointError("PG0 preflight outputs are non-finite")

    report = {
        "schema_version": "clir-prior-v12-posthoc-gate-preflight-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_FIXED_025_GATE_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "git": git,
        "protocol_file_sha256": file_sha256(protocol_path),
        "config": verify_config_pair(protocol)["pg0"],
        "supervision_per_epoch": summary,
        "row_ids": list(raw_batch["ids"]),
        "hidden_shape": list(raw_batch["hidden_states"].shape),
        "condition_shape": list(raw_batch["condition_states"].shape),
        "losses": loss_values,
        "gradient_norms": gradient_norms,
        "finite_outputs": finite_outputs,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None
        ),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None
            else None
        ),
        "training_allowed": True,
    }
    output_root = _project_path(protocol["runtime"]["output_root"])
    target = output_json or output_root / "training_preflight/preflight.json"
    if target.exists():
        raise FileExistsError(f"preflight output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def _all_state_tensors_finite(state: Any) -> bool:
    if torch.is_tensor(state):
        return bool(torch.isfinite(state).all())
    if isinstance(state, Mapping):
        return all(_all_state_tensors_finite(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return all(_all_state_tensors_finite(value) for value in state)
    return True


def validate_training(protocol_path: Path, output_json: Path | None) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    git = _git_state(protocol)
    output_root = _project_path(protocol["runtime"]["output_root"])
    preflight_path = output_root / "training_preflight/preflight.json"
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight_report.get("status")
        != "PASS_PRIOR_V12_POSTHOC_FIXED_025_GATE_PREFLIGHT"
        or preflight_report.get("git", {}).get("commit") != git["commit"]
        or preflight_report.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("missing or stale PG0 preflight")

    config_path = _project_path(protocol["cells"]["pg0"]["config"])
    config_hash = file_sha256(config_path)
    expected_train_hash = protocol["frozen_inputs"]["shared_train"]["file_sha256"]
    expected_dev_hash = protocol["frozen_inputs"]["prior_dev"]["file_sha256"]
    runs = []
    for seed in protocol["training"]["seeds"]:
        run_root = output_root / f"training/pg0/seed-{seed}"
        checkpoint_path = run_root / "checkpoint.pt"
        metrics_path = run_root / "metrics.jsonl"
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"incomplete PG0 seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("completed_epoch", -1)) != int(protocol["training"]["epochs"]):
            raise ValueError(f"PG0 seed {seed} did not complete the frozen epochs")
        if not _all_state_tensors_finite(checkpoint.get("state_dict", {})):
            raise FloatingPointError(f"PG0 seed {seed} has non-finite model state")
        model_config = checkpoint["model_config"]
        if float(model_config.get("gate_prior_weight", -1)) != 0.25:
            raise ValueError(f"PG0 seed {seed} Gate weight drift")
        if float(model_config.get("prior_weight", -1)) != 1.0:
            raise ValueError(f"PG0 seed {seed} direct Prior drift")
        data = checkpoint["data_state"]
        if (
            data.get("train_sha256") != expected_train_hash
            or data.get("val_sha256") != expected_dev_hash
            or int(data.get("train_rows", -1)) != int(protocol["training"]["train_rows"])
            or int(data.get("val_rows", -1)) != int(protocol["training"]["dev_rows"])
            or int(data["train_supervision_per_epoch"].get("paired_prior_rows", -1))
            != int(protocol["training"]["prior_rows"])
        ):
            raise ValueError(f"PG0 seed {seed} data/supervision drift")
        provenance = checkpoint["run_provenance"]
        if (
            provenance["code"].get("commit") != git["commit"]
            or provenance["code"].get("dirty") is not False
            or provenance["config"].get("sha256") != config_hash
        ):
            raise ValueError(f"PG0 seed {seed} provenance drift")
        metric_rows = read_jsonl(metrics_path)
        if len(metric_rows) != int(protocol["training"]["epochs"]):
            raise ValueError(f"PG0 seed {seed} metric epoch count drift")
        for row in metric_rows:
            for split in ("train", "validation"):
                values = row[split]
                if any(not math.isfinite(float(value)) for value in values.values()):
                    raise FloatingPointError(f"PG0 seed {seed} non-finite {split} metric")
                if float(values.get("prior_gate", 0.0)) <= 0.0:
                    raise ValueError(f"PG0 seed {seed} did not execute Gate loss")
        last = metric_rows[-1]
        runs.append(
            {
                "seed": int(seed),
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_file_sha256": file_sha256(checkpoint_path),
                "metrics_file_sha256": file_sha256(metrics_path),
                "completed_epoch": int(checkpoint["completed_epoch"]),
                "train_total": float(last["train"]["total"]),
                "train_prior_gate": float(last["train"]["prior_gate"]),
                "dev_total": float(last["validation"]["total"]),
                "dev_prior_gate": float(last["validation"]["prior_gate"]),
                "all_state_tensors_finite": True,
            }
        )
    report = {
        "schema_version": "clir-prior-v12-posthoc-gate-training-completion-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_PG0_THREE_RUNS",
        "created_at_utc": _utc_now(),
        "git": git,
        "protocol_file_sha256": file_sha256(protocol_path),
        "preflight_file_sha256": file_sha256(preflight_path),
        "reused_p0_checkpoints": protocol["cells"]["p0"]["checkpoint_sha256_by_seed"],
        "runs": runs,
        "same_data_all_runs": True,
        "all_checkpoints_load_and_finite": True,
        "dev_scoring_allowed": True,
        "ranking_scoring_allowed": False,
    }
    target = output_json or output_root / "training/completion_report.json"
    if target.exists():
        raise FileExistsError(f"training completion output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--device", default="cuda")
    preflight_parser.add_argument("--output-json", default=None)
    validate_parser = subparsers.add_parser("validate-training")
    validate_parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_path = Path(args.protocol).resolve()
    if args.command == "preflight":
        report = preflight(
            protocol_path,
            args.device,
            Path(args.output_json).resolve() if args.output_json else None,
        )
    elif args.command == "validate-training":
        report = validate_training(
            protocol_path,
            Path(args.output_json).resolve() if args.output_json else None,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
