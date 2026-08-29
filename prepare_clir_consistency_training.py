#!/usr/bin/env python
"""Prepare, verify, and full-width preflight the Consistency v6.1 C0/C1 run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from src.clir_consistency_scale_training import (
    construct_manifests,
    file_identity,
    load_authorization,
    verify_authorized_files,
)
from src.clir_data import CLIRTrajectoryDataset, clir_collate, read_jsonl
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
)
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
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/data_expansion_scale_v6/consistency_training_v6_1/authorization.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_scale_v6/consistency_training_v6_1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and preflight the hash-bound C0/C1 Consistency run."
    )
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    subparsers.add_parser("verify")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--cell", required=True, choices=["c0", "c1"])
    preflight.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    preflight.add_argument("--output-json", default=None)
    return parser


def _authorization_paths(
    authorization: Mapping[str, Any], root: Path
) -> dict[str, Path]:
    return {
        name: root / str(specification["path"])
        for name, specification in authorization["frozen_inputs"].items()
    }


def _reconstruct(
    authorization: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    paths = _authorization_paths(authorization, project_root)
    return construct_manifests(
        read_jsonl(paths["historical_correctness_train"]),
        read_jsonl(paths["extracted_selected_features"]),
        read_jsonl(paths["train_positive_relations"]),
        read_jsonl(paths["heldout_positive_relations"]),
        read_jsonl(paths["heldout_hard_negative_relations"]),
        historical_manifest_parent=paths["historical_correctness_train"].parent,
        feature_manifest_parent=paths["extracted_selected_features"].parent,
    )


def _config_pair_gate(authorization: Mapping[str, Any], project_root: Path) -> dict:
    config_paths = {
        name: project_root / str(specification["path"])
        for name, specification in authorization["execution_configs"].items()
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in config_paths.items()
    }
    if set(payloads) != {"c0", "c1"}:
        raise ValueError("Execution configs must be exactly c0 and c1")
    left = json.loads(json.dumps(payloads["c0"]))
    right = json.loads(json.dumps(payloads["c1"]))
    c0_weight = left["model"].pop("consistency_weight")
    c1_weight = right["model"].pop("consistency_weight")
    if c0_weight != 0.0 or c1_weight != 1.0 or left != right:
        raise ValueError(
            "C0/C1 configs must differ only in consistency_weight 0.0 versus 1.0"
        )
    return {
        name: {"path": str(path.resolve()), "file_sha256": file_sha256(path)}
        for name, path in config_paths.items()
    }


def materialize(authorization_path: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists; use a fresh path: {output_root}"
        )
    authorization = load_authorization(authorization_path)
    verified_inputs = verify_authorized_files(authorization, PROJECT_ROOT)
    configs = _config_pair_gate(authorization, PROJECT_ROOT)
    constructed = _reconstruct(authorization, PROJECT_ROOT)
    output_root.mkdir(parents=True)
    data_root = output_root / "data"
    manifests = {
        "train": publish_manifest(
            data_root / "train_c0_c1.jsonl",
            constructed["train_rows"],
            schema_version="clir-consistency-scale-c0-c1-train-manifest-v6.1",
            metadata={
                "shared_by_cells": ["c0", "c1"],
                "only_config_difference": "consistency_weight",
            },
        ),
        "heldout_positive_view": publish_manifest(
            data_root / "heldout_positive_training_view.jsonl",
            constructed["validation_rows"],
            schema_version=(
                "clir-consistency-scale-heldout-positive-training-view-v6.1"
            ),
            metadata={"evaluation_only": True, "relation_labels": "positive_only"},
        ),
        "heldout_endpoint_features": publish_manifest(
            data_root / "heldout_endpoint_features.jsonl",
            constructed["evaluation_rows"],
            schema_version="clir-consistency-scale-heldout-endpoints-v6.1",
            metadata={
                "evaluation_only": True,
                "relations_are_kept_in_separate_hash_bound_manifests": True,
            },
        ),
    }
    report = {
        "schema_version": "clir-consistency-scale-training-data-report-v6.1",
        "status": "PASS_C0_C1_DATA_MATERIALIZATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {
            "path": str(authorization_path.resolve()),
            "file_sha256": file_sha256(authorization_path),
        },
        "verified_inputs": verified_inputs,
        "execution_configs": configs,
        "manifests": manifests,
        "statistics": constructed["statistics"],
        "canonical_hashes": constructed["canonical_hashes"],
        "claim_boundary": (
            "deterministic_training_view_materialization_only_not_learnability_or_ranking"
        ),
    }
    atomic_write_json(output_root / "materialization_report.json", report)
    print(json.dumps(report, indent=2))


def _assert_manifest(
    path: Path, expected_rows: list[dict[str, Any]], expected_canonical: str
) -> dict[str, Any]:
    rows = read_jsonl(path)
    if canonical_sha256(rows) != expected_canonical or rows != expected_rows:
        raise ValueError(f"Materialized manifest content drift: {path}")
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "row_count": len(rows),
        "ordered_rows_sha256": expected_canonical,
        "file_sha256": file_sha256(path),
    }
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Materialized manifest sidecar drift: {sidecar_path}")
    return {**file_identity(path), "row_count": len(rows), **expected}


def verify(
    authorization_path: Path, output_root: Path, *, write_report: bool = True
) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    verified_inputs = verify_authorized_files(authorization, PROJECT_ROOT)
    configs = _config_pair_gate(authorization, PROJECT_ROOT)
    reconstructed = _reconstruct(authorization, PROJECT_ROOT)
    paths = {
        "train": output_root / "data/train_c0_c1.jsonl",
        "heldout_positive_view": (
            output_root / "data/heldout_positive_training_view.jsonl"
        ),
        "heldout_endpoint_features": (
            output_root / "data/heldout_endpoint_features.jsonl"
        ),
    }
    verified_manifests = {
        "train": _assert_manifest(
            paths["train"],
            reconstructed["train_rows"],
            reconstructed["canonical_hashes"]["train_rows"],
        ),
        "heldout_positive_view": _assert_manifest(
            paths["heldout_positive_view"],
            reconstructed["validation_rows"],
            reconstructed["canonical_hashes"]["validation_rows"],
        ),
        "heldout_endpoint_features": _assert_manifest(
            paths["heldout_endpoint_features"],
            reconstructed["evaluation_rows"],
            reconstructed["canonical_hashes"]["evaluation_rows"],
        ),
    }
    materialization_report = output_root / "materialization_report.json"
    materialized = json.loads(materialization_report.read_text(encoding="utf-8"))
    if materialized.get("statistics") != reconstructed["statistics"]:
        raise ValueError("Materialization statistics drift")
    report = {
        "schema_version": "clir-consistency-scale-training-data-verifier-v6.1",
        "status": "PASS_C0_C1_DATA_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_file_sha256": file_sha256(authorization_path),
        "materialization_report_file_sha256": file_sha256(materialization_report),
        "verified_inputs": verified_inputs,
        "execution_configs": configs,
        "verified_manifests": verified_manifests,
        "statistics": reconstructed["statistics"],
    }
    if write_report:
        atomic_write_json(output_root / "independent_data_verification.json", report)
    return report


def _git_state() -> dict[str, Any]:
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
        raise ValueError("Full-width preflight requires a clean implementation commit")
    return {"commit": commit, "dirty": False}


def _representative_consistency_indices(dataset: CLIRTrajectoryDataset) -> list[int]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        relation_id = row.get("consistency_relation_id")
        if relation_id is not None:
            grouped.setdefault(str(relation_id), []).append(index)
    usable = [indices for indices in grouped.values() if len(indices) == 2]
    if len(usable) < 2:
        raise ValueError("Preflight requires at least two two-view relations")
    indices = usable[0] + usable[1]
    styles = [dataset.rows[index].get("style_id") for index in indices]
    if Counter(styles) != Counter({"relative_compact": 2, "relative_expanded": 2}):
        raise ValueError("Representative preflight batch has invalid styles")
    return indices


def preflight(
    authorization_path: Path,
    output_root: Path,
    cell: str,
    device_name: str,
    output_json: Path | None,
) -> dict[str, Any]:
    verification = verify(authorization_path, output_root, write_report=False)
    verification_path = output_root / "independent_data_verification.json"
    if not verification_path.is_file():
        raise ValueError(
            "Run the independent data verifier before full-width preflight"
        )
    published_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    for key in ("status", "statistics", "verified_manifests"):
        if published_verification.get(key) != verification.get(key):
            raise ValueError("Published independent data verification has drifted")
    git_state = _git_state()
    authorization = load_authorization(authorization_path)
    config_spec = authorization["execution_configs"][cell]
    config_path = PROJECT_ROOT / str(config_spec["path"])
    model_config, configured_training = load_config(config_path)
    args = argparse.Namespace(
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
    training = apply_training_overrides(configured_training, args)
    set_seed(int(training["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.empty(1, device=device)
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_index)
    train_path = output_root / "data/train_c0_c1.jsonl"
    dataset = CLIRTrajectoryDataset(train_path)
    validate_feature_contract(dataset, model_config, "preflight-train")
    summary = supervision_summary(dataset, list(range(len(dataset))))
    validate_supervision_coverage(summary, model_config)
    indices = _representative_consistency_indices(dataset)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=4,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=0,
    )
    raw_batch = next(iter(loader))
    batch = prepare_batch(raw_batch, device, str(training["amp_dtype"]))
    model = ConsistencyLocalizedReward(model_config).to(device)
    model.train()
    with autocast_context(device, str(training["amp_dtype"])):
        outputs, losses = model.training_step(batch, prior_phase="joint")
        total = losses["total"]
    if not torch.isfinite(total):
        raise FloatingPointError("Preflight total loss is non-finite")
    total.backward()
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise FloatingPointError(
            "Preflight non-finite gradients: " + ", ".join(nonfinite_gradients[:8])
        )
    finite_outputs = {
        key: bool(torch.isfinite(outputs[key]).all().detach().cpu())
        for key in ("scores", "representations", "token_rewards", "gates")
    }
    if not all(finite_outputs.values()):
        raise FloatingPointError("Preflight model outputs are non-finite")
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    loss_values = {
        key: float(value.detach().float().cpu()) for key, value in losses.items()
    }
    if any(not math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError("Preflight reported a non-finite component loss")
    if cell == "c1" and "consistency_total" not in loss_values:
        raise ValueError("C1 preflight did not execute the Consistency objective")
    if cell == "c0" and any(key.startswith("consistency_") for key in loss_values):
        raise ValueError("C0 preflight unexpectedly executed Consistency")
    report = {
        "schema_version": "clir-consistency-scale-full-width-preflight-v6.1",
        "status": "PASS_FULL_WIDTH_FORWARD_BACKWARD",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell": cell,
        "git": git_state,
        "authorization_file_sha256": file_sha256(authorization_path),
        "data_verification_file_sha256": file_sha256(
            output_root / "independent_data_verification.json"
        ),
        "config": file_identity(config_path),
        "train_manifest": file_identity(train_path),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_index) if device.type == "cuda" else None
        ),
        "batch": {
            "row_ids": list(raw_batch["ids"]),
            "shape": list(raw_batch["hidden_states"].shape),
            "condition_shape": list(raw_batch["condition_states"].shape),
            "semantic_ids": raw_batch["semantic_ids"].tolist(),
            "style_ids": raw_batch["style_ids"].tolist(),
        },
        "supervision_per_epoch": summary,
        "losses": loss_values,
        "finite_outputs": finite_outputs,
        "parameters_with_gradients": sum(
            parameter.grad is not None for parameter in model.parameters()
        ),
        "peak_cuda_memory_allocated_bytes": peak_bytes,
    }
    target = output_json or output_root / "preflight" / f"{cell}.json"
    if target.exists():
        raise FileExistsError(f"Preflight report already exists: {target}")
    atomic_write_json(target, report)
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    authorization_path = Path(args.authorization)
    output_root = Path(args.output_root)
    if args.command == "materialize":
        materialize(authorization_path, output_root)
    elif args.command == "verify":
        report = verify(authorization_path, output_root)
        print(json.dumps(report, indent=2))
    elif args.command == "preflight":
        report = preflight(
            authorization_path,
            output_root,
            args.cell,
            args.device,
            Path(args.output_json) if args.output_json else None,
        )
        print(json.dumps(report, indent=2))
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
