#!/usr/bin/env python
"""Freeze and materialize the expanded three-module CLIR factorial data."""

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
from src.clir_smoke import (
    atomic_write_json,
    file_sha256,
    publish_manifest,
)
from src.clir_three_module import build_unified_data
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
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/three_module_expansion_v1/protocol.json"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/three_module_expansion_v1/training_authorization.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/three_module_expansion_v1"


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


def _git_state(minimum_parent: str = "34c5a68") -> dict[str, Any]:
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
        raise ValueError("three-module execution requires a clean commit")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", minimum_parent, commit],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode:
        raise ValueError("runtime commit does not descend from the frozen parent")
    return {
        "commit": commit,
        "dirty": False,
        "minimum_parent_commit": minimum_parent,
    }


def verify_factorial_configs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    cells = protocol["factorial_grid"]["cells"]
    expected_cells = {"u0", "c", "h", "p", "ch", "cp", "hp", "full"}
    if set(cells) != expected_cells:
        raise ValueError("three-module protocol must contain the complete 2x2x2 grid")
    payloads: dict[str, dict[str, Any]] = {}
    observed: dict[str, Any] = {}
    for cell, specification in cells.items():
        path = _project_path(specification["config"])
        _assert_hash(path, specification["file_sha256"], f"{cell} config")
        payload = json.loads(path.read_text(encoding="utf-8"))
        factors = tuple(int(value) for value in specification["factors"])
        if len(factors) != 3 or any(value not in (0, 1) for value in factors):
            raise ValueError(f"{cell}: invalid C/H/P factor tuple")
        model = payload["model"]
        observed_factors = (
            int(float(model["consistency_weight"]) == 1.0),
            int(float(model["hallucination_weight"]) == 1.0),
            int(float(model["prior_weight"]) == 1.0),
        )
        if observed_factors != factors:
            raise ValueError(f"{cell}: config does not match its C/H/P tuple")
        expected_gate = 0.25 if factors[2] else 0.0
        if float(model["gate_prior_weight"]) != expected_gate:
            raise ValueError(f"{cell}: P factor must carry fixed Gate {expected_gate}")
        for disabled in (
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "progress_weight",
            "prior_distill_weight",
            "reconstruction_weight",
        ):
            if float(model[disabled]) != 0.0:
                raise ValueError(f"{cell}: unexpectedly enables {disabled}")
        payloads[cell] = payload
        observed[cell] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "factors": list(factors),
        }

    normalized: list[dict[str, Any]] = []
    for cell in sorted(cells):
        payload = json.loads(json.dumps(payloads[cell]))
        for field in (
            "consistency_weight",
            "hallucination_weight",
            "prior_weight",
            "gate_prior_weight",
        ):
            payload["model"].pop(field)
        normalized.append(payload)
    if any(payload != normalized[0] for payload in normalized[1:]):
        raise ValueError("factorial configs differ outside C/H/P factor weights")
    return observed


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "clir-three-module-expansion-v1"
        or protocol.get("status")
        != "AUTHORIZED_MATERIALIZATION_ONLY_PENDING_EXACT_MANIFEST_FREEZE"
        or protocol.get("evidence_tier")
        != "posthoc_exploratory_silver_no_human_verification"
    ):
        raise ValueError("unsupported or claim-drifting three-module protocol")
    for name, specification in protocol["parent_results"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
    for name, specification in protocol["frozen_inputs"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
        if len(read_jsonl(source)) != int(specification["row_count"]):
            raise ValueError(f"{name} row-count drift")
    verify_factorial_configs(protocol)
    return protocol


def load_training_authorization(path: str | Path) -> dict[str, Any]:
    """Load and verify the separately frozen 24-run training authorization."""

    authorization_path = Path(path).resolve()
    payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "clir-three-module-expansion-v1-training-authorization"
        or payload.get("status")
        != "AUTHORIZED_POSTHOC_EXPLORATORY_COMPLETE_2X2X2_TRAINING"
        or payload.get("evidence_tier")
        != "posthoc_exploratory_silver_no_human_verification"
    ):
        raise ValueError("unsupported, inactive, or claim-drifting authorization")
    if payload.get("terminal_statuses_preserved") != {
        "hallucination_v7": "FAIL_H0_V7_RESERVE",
        "prior_v12": "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE",
        "prior_v13": "FAIL_PRIOR_V13_SCHEMA",
    }:
        raise ValueError("original terminal-status history drift")

    protocol_path = _project_path(payload["frozen_inputs"]["protocol"]["path"])
    _assert_hash(
        protocol_path,
        payload["frozen_inputs"]["protocol"]["file_sha256"],
        "three-module materialization protocol",
    )
    protocol = load_protocol(protocol_path)
    for name, specification in payload["frozen_inputs"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
        if "row_count" in specification:
            if len(read_jsonl(source)) != int(specification["row_count"]):
                raise ValueError(f"{name} row-count drift")
        if "sidecar_path" in specification:
            sidecar = _project_path(specification["sidecar_path"])
            _assert_hash(
                sidecar,
                specification["sidecar_file_sha256"],
                f"{name} sidecar",
            )

    materialization = json.loads(
        _project_path(
            payload["frozen_inputs"]["materialization_report"]["path"]
        ).read_text(encoding="utf-8")
    )
    verification = json.loads(
        _project_path(
            payload["frozen_inputs"]["materialization_verification"]["path"]
        ).read_text(encoding="utf-8")
    )
    if (
        materialization.get("status")
        != "PASS_THREE_MODULE_UNIFIED_DATA_MATERIALIZATION"
        or verification.get("status")
        != "PASS_THREE_MODULE_UNIFIED_DATA_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("the frozen unified-data gates did not pass")

    observed_configs = verify_factorial_configs(protocol)
    if set(payload["cells"]) != set(observed_configs):
        raise ValueError("authorization does not contain the complete 2x2x2 grid")
    for cell, specification in payload["cells"].items():
        if (
            specification["config"]
            != protocol["factorial_grid"]["cells"][cell]["config"]
            or specification["file_sha256"] != observed_configs[cell]["file_sha256"]
            or list(specification["factors"])
            != list(protocol["factorial_grid"]["cells"][cell]["factors"])
        ):
            raise ValueError(f"{cell}: authorization/config factor drift")

    for name, specification in payload["implementation"][
        "frozen_training_sources"
    ].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)

    training = payload["training"]
    if (
        list(training["seeds"]) != [42, 43, 44]
        or int(training["runs"]) != 24
        or int(training["epochs"]) != 3
        or int(training["batch_size"]) != 4
        or float(training["learning_rate"]) != 0.0001
        or training.get("validation_during_optimization") is not False
    ):
        raise ValueError("frozen factorial training schedule drift")
    return payload


def _publish(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = publish_manifest(path, rows, schema_version=schema, metadata=metadata)
    manifest["sidecar_file_sha256"] = file_sha256(
        path.with_suffix(path.suffix + ".manifest.json")
    )
    return manifest


def _query_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["query_id"]) for row in rows}


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    report_path = output_root / "materialization_report.json"
    if report_path.exists():
        raise FileExistsError(f"materialization report already exists: {report_path}")
    protocol = load_protocol(protocol_path)
    git = _git_state()
    frozen = protocol["frozen_inputs"]
    h_train_path = _project_path(frozen["consistency_h0_train"]["path"])
    prior_train_path = _project_path(frozen["prior_train"]["path"])
    h_dev_path = _project_path(frozen["h_dev"]["path"])
    prior_dev_path = _project_path(frozen["prior_dev"]["path"])
    data_root = output_root / "data"
    built = build_unified_data(
        consistency_h0_train=read_jsonl(h_train_path),
        prior_train=read_jsonl(prior_train_path),
        h_dev=read_jsonl(h_dev_path),
        prior_dev=read_jsonl(prior_dev_path),
        consistency_h0_parent=h_train_path.parent,
        prior_parent=prior_train_path.parent,
        h_dev_parent=h_dev_path.parent,
        prior_dev_parent=prior_dev_path.parent,
        target_parent=data_root,
    )
    train_queries = _query_ids(built["train"])
    endpoint_path = _project_path(frozen["consistency_heldout_endpoints"]["path"])
    ranking_path = _project_path(frozen["ranking"]["path"])
    consistency_overlap = train_queries & _query_ids(read_jsonl(endpoint_path))
    ranking_overlap = train_queries & _query_ids(read_jsonl(ranking_path))
    if consistency_overlap or ranking_overlap:
        raise ValueError("unified train overlaps frozen held-out Consistency/ranking")

    manifests = {
        "train": _publish(
            data_root / "train_factorial.jsonl",
            built["train"],
            "clir-three-module-expansion-v1-train-manifest",
            {
                "shared_by_cells": list(protocol["factorial_grid"]["cells"]),
                "posthoc_exploratory": True,
            },
        ),
        "h_dev": _publish(
            data_root / "h_dev_query_disjoint.jsonl",
            built["h_dev"],
            "clir-three-module-expansion-v1-h-dev-manifest",
            {"evaluation_only": True, "cross_module_query_disjoint": True},
        ),
        "prior_dev": _publish(
            data_root / "prior_dev_query_disjoint.jsonl",
            built["prior_dev"],
            "clir-three-module-expansion-v1-prior-dev-manifest",
            {"evaluation_only": True, "cross_module_query_disjoint": True},
        ),
    }
    report = {
        "schema_version": "clir-three-module-expansion-v1-materialization",
        "status": "PASS_THREE_MODULE_UNIFIED_DATA_MATERIALIZATION",
        "completed_at_utc": _utc_now(),
        "code_commit": git["commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "evidence_tier": protocol["evidence_tier"],
        "terminal_statuses_preserved": protocol["terminal_statuses_preserved"],
        "inventory": built["report"],
        "evaluation_query_overlap": {
            "consistency_heldout": 0,
            "ranking": 0,
            "clean_h_dev": 0,
            "clean_prior_dev": 0,
        },
        "manifests": manifests,
        "training_allowed": False,
        "next_gate": "SEPARATE_HASH_BOUND_FACTORIAL_TRAINING_AUTHORIZATION",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root).resolve()
    report_path = output_root / "materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_THREE_MODULE_UNIFIED_DATA_MATERIALIZATION":
        raise ValueError("three-module materialization did not pass")
    if report.get("protocol_file_sha256") != file_sha256(args.protocol):
        raise ValueError("materialization protocol hash drift")
    expected_rows = {
        "train": int(protocol["merge_contract"]["expected_train_rows"]),
        "h_dev": int(
            protocol["evaluation_split_contract"]["expected_clean_h_dev_rows"]
        ),
        "prior_dev": int(
            protocol["evaluation_split_contract"]["expected_clean_prior_dev_rows"]
        ),
    }
    observed: dict[str, Any] = {}
    for name, expected in expected_rows.items():
        manifest = report["manifests"][name]
        path = Path(manifest["path"])
        _assert_hash(path, manifest["file_sha256"], f"published {name}")
        if len(read_jsonl(path)) != expected:
            raise ValueError(f"published {name} row-count drift")
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        _assert_hash(sidecar, manifest["sidecar_file_sha256"], f"{name} sidecar")
        observed[name] = {
            "path": str(path),
            "row_count": expected,
            "file_sha256": manifest["file_sha256"],
        }
    verification = {
        "schema_version": "clir-three-module-expansion-v1-verification",
        "status": "PASS_THREE_MODULE_UNIFIED_DATA_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(args.protocol),
        "materialization_report_sha256": file_sha256(report_path),
        "manifests": observed,
        "training_allowed": False,
        "next_gate": "SEPARATE_HASH_BOUND_FACTORIAL_TRAINING_AUTHORIZATION",
    }
    verification_path = output_root / "materialization_verification.json"
    if verification_path.exists():
        raise FileExistsError(f"verification already exists: {verification_path}")
    atomic_write_json(verification_path, verification)
    print(json.dumps(verification, indent=2))


def _gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not name.startswith(prefix):
            continue
        value = parameter.grad.detach().float()
        squared += float(torch.sum(value * value).cpu())
    return math.sqrt(squared)


def _gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    return {
        prefix.rstrip("."): _gradient_norm(model, prefix)
        for prefix in (
            "feature_encoder.",
            "projector.",
            "hallucination_head.",
            "token_reward_head.",
            "key_prior_head.",
            "complete_prior_head.",
            "final_score_head.",
        )
    }


def _representative_indices(
    dataset: CLIRTrajectoryDataset,
) -> dict[str, list[int]]:
    relations: dict[str, list[int]] = {}
    positives: list[int] = []
    clean: list[int] = []
    prior: list[int] = []
    for index, row in enumerate(dataset.rows):
        semantic = row.get("semantic_id")
        style = row.get("style_id")
        if semantic is not None and style is not None:
            relations.setdefault(str(semantic), []).append(index)
        if row.get("hallucination_onset") is not None:
            onset = int(row["hallucination_onset"])
            (clean if onset == -1 else positives).append(index)
        if (
            row.get("key_prior_target") is not None
            and row.get("complete_prior_target") is not None
        ):
            prior.append(index)

    usable_relations: list[list[int]] = []
    for indices in relations.values():
        if len(indices) != 2:
            continue
        styles = Counter(str(dataset.rows[index]["style_id"]) for index in indices)
        if styles == Counter({"relative_compact": 1, "relative_expanded": 1}):
            usable_relations.append(indices)
    usable_relations.sort(
        key=lambda indices: max(
            len(dataset.rows[index]["output_token_ids"]) for index in indices
        )
    )
    positives.sort(key=lambda index: len(dataset.rows[index]["output_token_ids"]))
    clean.sort(key=lambda index: len(dataset.rows[index]["output_token_ids"]))
    prior.sort(key=lambda index: len(dataset.rows[index]["output_token_ids"]))
    if len(usable_relations) < 2 or len(positives) < 2 or len(clean) < 2:
        raise ValueError("full-width preflight supervision rows are missing")
    if len(prior) < 4:
        raise ValueError("full-width preflight requires four paired-Prior rows")
    batches = {
        "consistency": usable_relations[0] + usable_relations[1],
        "hallucination": clean[:2] + positives[:2],
        "prior": prior[:4],
    }
    for name, indices in batches.items():
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError(f"invalid representative {name} batch")
    return batches


def _run_preflight_batch(
    model: ConsistencyLocalizedReward,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    amp_dtype: str,
    objective_key: str | None,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    batch = prepare_batch(dict(raw_batch), device, amp_dtype)
    with autocast_context(device, amp_dtype):
        outputs, losses = model.training_step(batch, prior_phase="joint")
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("three-module preflight total is non-finite")
    loss_values = {
        key: float(value.detach().float().cpu()) for key, value in losses.items()
    }
    if any(not math.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError("three-module preflight loss is non-finite")

    objective_gradients: dict[str, float] | None = None
    if objective_key is not None and objective_key in losses:
        objective = losses[objective_key]
        if objective.requires_grad:
            objective.backward(retain_graph=True)
            objective_gradients = _gradient_norms(model)
            model.zero_grad(set_to_none=True)

    losses["total"].backward()
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite:
        raise FloatingPointError(
            "three-module preflight non-finite gradients: " + ", ".join(nonfinite[:8])
        )
    total_gradients = _gradient_norms(model)
    finite_outputs = {
        key: bool(torch.isfinite(outputs[key]).all().detach().cpu())
        for key in (
            "scores",
            "representations",
            "hallucination_logits",
            "token_rewards",
            "gates",
            "key_prior",
            "complete_prior",
            "fused_prior",
        )
    }
    if not all(finite_outputs.values()):
        raise FloatingPointError("three-module preflight output is non-finite")
    condition = raw_batch.get("condition_states")
    return {
        "row_ids": list(raw_batch["ids"]),
        "hidden_shape": list(raw_batch["hidden_states"].shape),
        "condition_shape": list(condition.shape) if condition is not None else None,
        "losses": loss_values,
        "objective_key": objective_key,
        "objective_gradient_norms": objective_gradients,
        "total_gradient_norms": total_gradients,
        "finite_outputs": finite_outputs,
    }


def _assert_cell_preflight(
    cell: str,
    factors: Sequence[int],
    batches: Mapping[str, Mapping[str, Any]],
) -> None:
    c_enabled, h_enabled, p_enabled = (bool(value) for value in factors)
    c_batch = batches["consistency"]
    h_batch = batches["hallucination"]
    p_batch = batches["prior"]
    if ("consistency_total" in c_batch["losses"]) != c_enabled:
        raise ValueError(f"{cell}: Consistency loss routing drift")
    if ("localization_token_bce" in h_batch["losses"]) != h_enabled:
        raise ValueError(f"{cell}: H0 loss routing drift")
    if ("prior_total" in p_batch["losses"]) != p_enabled:
        raise ValueError(f"{cell}: Prior loss routing drift")

    if c_enabled:
        gradients = c_batch["objective_gradient_norms"] or {}
        if gradients.get("projector", 0.0) <= 0.0:
            raise ValueError(f"{cell}: Consistency did not train the projector")
        for key in ("consistency_positive", "consistency_negative"):
            if c_batch["losses"].get(key, 0.0) <= 0.0:
                raise ValueError(f"{cell}: representative batch did not exercise {key}")
    if h_enabled:
        gradients = h_batch["objective_gradient_norms"] or {}
        if gradients.get("hallucination_head", 0.0) <= 0.0:
            raise ValueError(f"{cell}: H0 did not train the hallucination head")
        if h_batch["losses"].get("localization_token_bce", 0.0) <= 0.0:
            raise ValueError(f"{cell}: representative batch did not exercise H0 BCE")
    if p_enabled:
        gradients = p_batch["objective_gradient_norms"] or {}
        for prefix in (
            "key_prior_head",
            "complete_prior_head",
            "token_reward_head",
        ):
            if gradients.get(prefix, 0.0) <= 0.0:
                raise ValueError(f"{cell}: Prior/Gate did not train {prefix}")
        for key in ("prior_key", "prior_complete", "prior_gate"):
            if p_batch["losses"].get(key, 0.0) <= 0.0:
                raise ValueError(f"{cell}: representative batch did not exercise {key}")
    for name, batch in batches.items():
        gradients = batch["total_gradient_norms"]
        if gradients["feature_encoder"] <= 0.0 or gradients["final_score_head"] <= 0.0:
            raise ValueError(f"{cell}/{name}: base reward path lost gradients")


def command_preflight(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    authorization = load_training_authorization(authorization_path)
    git = _git_state(authorization["implementation"]["minimum_parent_commit"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    cuda_index: int | None = None
    if device.type == "cuda":
        torch.empty(1, device=device)
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_index)

    train_spec = authorization["frozen_inputs"]["train_manifest"]
    train_path = _project_path(train_spec["path"])
    dataset = CLIRTrajectoryDataset(train_path)
    summary = supervision_summary(dataset, list(range(len(dataset))))
    expected_summary = authorization["training"]["supervision_per_epoch"]
    if summary != expected_summary:
        raise ValueError(f"unified supervision inventory drift: {summary}")
    if len(query_ids(dataset)) != int(authorization["training"]["train_queries"]):
        raise ValueError("unified train query-count drift")

    train_queries = query_ids(dataset)
    for name in (
        "h_dev_manifest",
        "prior_dev_manifest",
        "consistency_heldout_endpoints",
        "ranking_manifest",
    ):
        rows = read_jsonl(_project_path(authorization["frozen_inputs"][name]["path"]))
        if train_queries & {str(row["query_id"]) for row in rows}:
            raise ValueError(f"unified train/{name} query overlap")

    indices = _representative_indices(dataset)
    raw_batches = {
        name: clir_collate([dataset[index] for index in selected])
        for name, selected in indices.items()
    }
    reports: dict[str, Any] = {}
    for cell in authorization["cell_order"]:
        specification = authorization["cells"][cell]
        config_path = _project_path(specification["config"])
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
        validate_feature_contract(dataset, model_config, f"three-module-{cell}")
        validate_supervision_coverage(summary, model_config)
        set_seed(int(training["seed"]))
        model = ConsistencyLocalizedReward(model_config).to(device)
        model.train()
        batch_reports = {
            "consistency": _run_preflight_batch(
                model,
                raw_batches["consistency"],
                device,
                str(training["amp_dtype"]),
                "consistency_total" if specification["factors"][0] else None,
            ),
            "hallucination": _run_preflight_batch(
                model,
                raw_batches["hallucination"],
                device,
                str(training["amp_dtype"]),
                "localization_token_bce" if specification["factors"][1] else None,
            ),
            "prior": _run_preflight_batch(
                model,
                raw_batches["prior"],
                device,
                str(training["amp_dtype"]),
                "prior_total" if specification["factors"][2] else None,
            ),
        }
        _assert_cell_preflight(cell, specification["factors"], batch_reports)
        reports[cell] = {
            "factors": list(specification["factors"]),
            "config_file_sha256": file_sha256(config_path),
            "batches": batch_reports,
            "passed": True,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": "clir-three-module-expansion-v1-full-width-preflight",
        "status": "PASS_THREE_MODULE_COMPLETE_2X2X2_FULL_WIDTH_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "evidence_tier": authorization["evidence_tier"],
        "supervision_per_epoch": summary,
        "cells": reports,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None
        ),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_index))
            if cuda_index is not None
            else None
        ),
        "training_runs_authorized": int(authorization["training"]["runs"]),
        "training_allowed": True,
    }
    target = (
        Path(args.output_json).resolve()
        if args.output_json
        else _project_path(authorization["runtime"]["preflight_report"])
    )
    if target.exists():
        raise FileExistsError(f"preflight report already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2))


def _all_state_tensors_finite(state: Any) -> bool:
    if torch.is_tensor(state):
        return bool(torch.isfinite(state).all())
    if isinstance(state, Mapping):
        return all(_all_state_tensors_finite(value) for value in state.values())
    if isinstance(state, (list, tuple)):
        return all(_all_state_tensors_finite(value) for value in state)
    return True


def command_validate_training(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    authorization = load_training_authorization(authorization_path)
    git = _git_state(authorization["implementation"]["minimum_parent_commit"])
    preflight_path = _project_path(authorization["runtime"]["preflight_report"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status")
        != "PASS_THREE_MODULE_COMPLETE_2X2X2_FULL_WIDTH_PREFLIGHT"
        or preflight.get("git", {}).get("commit") != git["commit"]
        or preflight.get("authorization_file_sha256") != file_sha256(authorization_path)
        or preflight.get("training_allowed") is not True
    ):
        raise ValueError("missing or stale three-module full-width preflight")

    expected_data = authorization["frozen_inputs"]["train_manifest"]
    output_root = _project_path(authorization["runtime"]["output_root"])
    runs: list[dict[str, Any]] = []
    for cell in authorization["cell_order"]:
        specification = authorization["cells"][cell]
        config_path = _project_path(specification["config"])
        model_config, _ = load_config(config_path)
        for seed in authorization["training"]["seeds"]:
            run_root = output_root / f"training/{cell}/seed-{seed}"
            checkpoint_path = run_root / "checkpoint.pt"
            metrics_path = run_root / "metrics.jsonl"
            if not checkpoint_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f"incomplete three-module run {cell}/{seed}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if int(checkpoint.get("completed_epoch", -1)) != int(
                authorization["training"]["epochs"]
            ):
                raise ValueError(f"{cell}/{seed}: incomplete epoch budget")
            if checkpoint.get("model_config") != model_config.__dict__:
                raise ValueError(f"{cell}/{seed}: model config drift")
            if not _all_state_tensors_finite(checkpoint.get("state_dict", {})):
                raise FloatingPointError(f"{cell}/{seed}: non-finite model state")
            if not _all_state_tensors_finite(
                checkpoint.get("optimizer_state_dict", {})
            ):
                raise FloatingPointError(f"{cell}/{seed}: non-finite optimizer state")
            contract = checkpoint["training_contract"]
            if (
                int(contract["seed"]) != int(seed)
                or int(contract["batch_size"])
                != int(authorization["training"]["batch_size"])
                or float(contract["learning_rate"])
                != float(authorization["training"]["learning_rate"])
            ):
                raise ValueError(f"{cell}/{seed}: training contract drift")
            data = checkpoint["data_state"]
            if (
                data.get("train_sha256") != expected_data["file_sha256"]
                or int(data.get("train_rows", -1))
                != int(authorization["training"]["train_rows"])
                or int(data.get("train_queries", -1))
                != int(authorization["training"]["train_queries"])
                or data.get("train_supervision_per_epoch")
                != authorization["training"]["supervision_per_epoch"]
                or data.get("val_sha256") is not None
                or int(data.get("val_rows", -1)) != 0
            ):
                raise ValueError(f"{cell}/{seed}: data-state drift")
            provenance = checkpoint["run_provenance"]
            if (
                provenance["code"].get("commit") != git["commit"]
                or provenance["code"].get("dirty") is not False
                or provenance["config"].get("sha256") != specification["file_sha256"]
            ):
                raise ValueError(f"{cell}/{seed}: run provenance drift")

            metrics = read_jsonl(metrics_path)
            if len(metrics) != int(authorization["training"]["epochs"]):
                raise ValueError(f"{cell}/{seed}: metric epoch-count drift")
            required_losses = {"final", "total"}
            c_enabled, h_enabled, p_enabled = specification["factors"]
            if c_enabled:
                required_losses.add("consistency_total")
            if h_enabled:
                required_losses.add("localization_token_bce")
            if p_enabled:
                required_losses.update(
                    {"prior_key", "prior_complete", "prior_gate", "prior_total"}
                )
            for row in metrics:
                values = row["train"]
                if not required_losses <= set(values):
                    raise ValueError(f"{cell}/{seed}: enabled loss was not executed")
                if any(not math.isfinite(float(value)) for value in values.values()):
                    raise FloatingPointError(f"{cell}/{seed}: non-finite train metric")
            last = metrics[-1]["train"]
            runs.append(
                {
                    "cell": cell,
                    "factors": list(specification["factors"]),
                    "seed": int(seed),
                    "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                    "checkpoint_file_sha256": file_sha256(checkpoint_path),
                    "metrics_file_sha256": file_sha256(metrics_path),
                    "completed_epoch": int(checkpoint["completed_epoch"]),
                    "final_train_total": float(last["total"]),
                    "all_state_tensors_finite": True,
                }
            )
    if len(runs) != int(authorization["training"]["runs"]):
        raise ValueError("three-module run-count drift")
    report = {
        "schema_version": "clir-three-module-expansion-v1-training-completion",
        "status": "PASS_THREE_MODULE_COMPLETE_2X2X2_24_RUN_TRAINING",
        "created_at_utc": _utc_now(),
        "git": git,
        "authorization_file_sha256": file_sha256(authorization_path),
        "preflight_file_sha256": file_sha256(preflight_path),
        "runs": runs,
        "same_manifest_all_runs": True,
        "all_checkpoints_load_and_are_finite": True,
        "mechanism_evaluation_allowed": True,
        "ranking_scoring_allowed": False,
    }
    target = (
        Path(args.output_json).resolve()
        if args.output_json
        else output_root / "training/completion_report.json"
    )
    if target.exists():
        raise FileExistsError(f"training completion report exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    subparsers.add_parser("verify")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--output-json", default=None)
    validate = subparsers.add_parser("validate-training")
    validate.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "materialize":
        command_materialize(args)
    elif args.command == "verify":
        command_verify(args)
    elif args.command == "preflight":
        command_preflight(args)
    elif args.command == "validate-training":
        command_validate_training(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
