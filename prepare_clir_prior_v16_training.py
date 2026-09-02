#!/usr/bin/env python
"""Prepare and verify the staged Prior-v16 R0/P0 -> CH/Full experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from src.clir_data import CLIRTrajectoryDataset, clir_collate, read_jsonl
from src.clir_prior_v16_posthoc_binary import validate_posthoc_silver_rows
from src.clir_scale_features import validate_tensor_file
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
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
    set_seed,
    supervision_summary,
    validate_feature_contract,
    validate_supervision_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_training_v1/protocol.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_prior_v16_posthoc_training_v1"
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


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_commit(minimum_parent: str) -> str:
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Prior-v16 training commands require a clean worktree")
    head = _git_head()
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "merge-base",
            "--is-ancestor",
            minimum_parent,
            head,
        ],
        check=False,
    )
    if ancestor.returncode:
        raise ValueError("minimum parent commit is not an ancestor of HEAD")
    return head


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def _published_identity(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "sidecar_file_sha256": file_sha256(sidecar),
        "row_count": len(rows),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def _publish(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    publish_manifest(path, rows, schema_version=schema, metadata=metadata)
    return _published_identity(path, rows)


def _verify_configs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_weights = {
        "r0": (0.0, 0.0, 0.0, 0.0),
        "p0": (0.0, 0.0, 1.0, 0.0),
        "ch": (1.0, 1.0, 0.0, 0.0),
        "full": (1.0, 1.0, 1.0, 0.25),
    }
    payloads: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    for cell in ("r0", "p0", "ch", "full"):
        specification = protocol["cells"][cell]
        path = _project_path(specification["config"])
        _assert_hash(path, specification["file_sha256"], f"{cell} config")
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload["model"]
        observed = (
            float(model["consistency_weight"]),
            float(model["hallucination_weight"]),
            float(model["prior_weight"]),
            float(model["gate_prior_weight"]),
        )
        if observed != expected_weights[cell]:
            raise ValueError(f"{cell}: factor weights drift: {observed}")
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
                raise ValueError(f"{cell}: forbidden objective {disabled} is enabled")
        clone = json.loads(json.dumps(payload))
        for key in (
            "consistency_weight",
            "hallucination_weight",
            "prior_weight",
            "gate_prior_weight",
        ):
            clone["model"].pop(key)
        normalized.append(clone)
        payloads[cell] = {
            "config": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "weights": list(observed),
        }
    if any(payload != normalized[0] for payload in normalized[1:]):
        raise ValueError("experiment configs differ outside the four frozen factors")
    return payloads


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-prior-v16-posthoc-training-protocol-v1"
        or protocol.get("status") != "AUTHORIZED_STAGED_R0_P0_CH_FULL_TRAINING"
    ):
        raise ValueError("unsupported or inactive Prior-v16 training protocol")
    boundary = protocol["evidence_boundary"]
    if (
        boundary.get("original_terminal_statuses_are_unchanged") is not True
        or boundary.get("gold_human_verified_confirmatory_or_protected_test")
        is not False
        or boundary.get("fresh_query_cluster_ranking_confirmation_required_later")
        is not True
    ):
        raise ValueError("Prior-v16 evidence boundary drift")
    for name, specification in protocol["frozen_inputs"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
        if "sidecar_file_sha256" in specification:
            sidecar = source.with_suffix(source.suffix + ".manifest.json")
            _assert_hash(sidecar, specification["sidecar_file_sha256"], f"{name} sidecar")
    _verify_configs(protocol)
    return protocol


def _output_root(protocol: Mapping[str, Any], requested: str | Path) -> Path:
    expected = _project_path(protocol["runtime"]["output_root"]).resolve()
    observed = Path(requested).resolve()
    if observed != expected:
        raise ValueError(f"output-root drift: {observed} != {expected}")
    return observed


def _load_silver(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    specification = protocol["frozen_inputs"]["prior_silver"]
    rows = read_jsonl(_project_path(specification["path"]))
    validation = validate_posthoc_silver_rows(rows)
    if (
        validation["rows"] != int(specification["rows"])
        or validation["train_rows"] != int(specification["train_rows"])
        or validation["dev_rows"] != int(specification["dev_rows"])
    ):
        raise ValueError("Prior-v16 Silver population drift")
    expected = protocol["feature_contract"]["expected"]
    observed = {
        "rows": len(rows),
        "queries": len({str(row["query_id"]) for row in rows}),
        "clusters": len({str(row["cluster_id"]) for row in rows}),
        "output_tokens": sum(len(row["output_token_ids"]) for row in rows),
        "prompt_tokens": sum(len(row["prompt_token_ids"]) for row in rows),
    }
    observed["total_tokens"] = observed["output_tokens"] + observed["prompt_tokens"]
    observed["raw_tensor_bytes"] = (
        observed["total_tokens"] * int(protocol["feature_contract"]["feature_dim"]) * 2
    )
    if observed != {key: int(value) for key, value in expected.items()}:
        raise ValueError(f"Prior-v16 feature inventory drift: {observed}")
    return rows


def _balanced_shards(
    rows: Sequence[Mapping[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -(len(row["prompt_token_ids"]) + len(row["output_token_ids"])),
            int(row["feature_inventory_index"]),
        ),
    )
    for row in ordered:
        index = min(range(shard_count), key=lambda value: (totals[value], value))
        shards[index].append(row)
        totals[index] += len(row["prompt_token_ids"]) + len(row["output_token_ids"])
    for shard in shards:
        shard.sort(key=lambda row: int(row["feature_inventory_index"]))
    if sorted(row["feature_inventory_index"] for shard in shards for row in shard) != list(
        range(len(rows))
    ):
        raise AssertionError("feature shard partition drift")
    return shards


def command_prepare(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    commit = _require_clean_commit(protocol["minimum_parent_commit"])
    output_root = _output_root(protocol, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    rows = _load_silver(protocol)
    inventory = [
        {**row, "feature_inventory_index": index}
        for index, row in enumerate(rows)
    ]
    plan_root = output_root / "plan"
    shard_root = plan_root / "shards"
    shard_root.mkdir(parents=True)
    inventory_path = plan_root / "feature_inventory.jsonl"
    inventory_identity = _publish(
        inventory_path,
        inventory,
        "clir-prior-v16-posthoc-training-feature-inventory-v1",
        {"selected_only": True, "decode_or_retokenize": False},
    )
    shards = _balanced_shards(
        inventory, int(protocol["feature_contract"]["worker_shards"])
    )
    shard_identities = {}
    shard_stats = {}
    for index, shard in enumerate(shards):
        path = shard_root / f"worker-{index:03d}.jsonl"
        shard_identities[str(index)] = _publish(
            path,
            shard,
            "clir-prior-v16-posthoc-training-feature-shard-v1",
            {"worker_index": index},
        )
        shard_stats[str(index)] = {
            "rows": len(shard),
            "tokens": sum(
                len(row["prompt_token_ids"]) + len(row["output_token_ids"])
                for row in shard
            ),
        }
    largest = max(
        inventory,
        key=lambda row: (
            len(row["prompt_token_ids"]) + len(row["output_token_ids"]),
            str(row["id"]),
        ),
    )
    preflight_path = plan_root / "preflight.jsonl"
    preflight_identity = _publish(
        preflight_path,
        [largest],
        "clir-prior-v16-posthoc-training-feature-preflight-v1",
        {"selection": "largest_prompt_plus_output_token_count"},
    )
    report = {
        "schema_version": "clir-prior-v16-posthoc-training-feature-plan-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_TRAINING_FEATURE_PLAN",
        "created_at_utc": _utc_now(),
        "code_commit": commit,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "config_gate": _verify_configs(protocol),
        "feature_inventory": inventory_identity,
        "feature_shards": shard_identities,
        "feature_shard_statistics": shard_stats,
        "preflight": preflight_identity,
        "feature_extraction_allowed": True,
        "training_allowed": False,
    }
    target = plan_root / "plan_report.json"
    atomic_write_json(target, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(target)}, indent=2))


def _load_plan(
    protocol_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    _output_root(protocol, output_root)
    plan_path = output_root / "plan/plan_report.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_PRIOR_V16_POSTHOC_TRAINING_FEATURE_PLAN":
        raise ValueError("feature plan did not pass")
    if plan.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("protocol changed after feature planning")
    if plan.get("code_commit") != _require_clean_commit(protocol["minimum_parent_commit"]):
        raise ValueError("code commit changed after feature planning")
    inventory_path = output_root / "plan/feature_inventory.jsonl"
    inventory = read_jsonl(inventory_path)
    if _published_identity(inventory_path, inventory) != plan["feature_inventory"]:
        raise ValueError("feature inventory drift after planning")
    return protocol, plan


def _safe_feature_path(parent: Path, raw: Any, output_root: Path) -> Path:
    value = Path(str(raw))
    path = value.resolve() if value.is_absolute() else (parent / value).resolve()
    if not path.is_relative_to(output_root.resolve()):
        raise ValueError(f"feature path escapes output root: {path}")
    return path


def _verify_extracted_rows(
    source_rows: Sequence[Mapping[str, Any]],
    extracted_path: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
    target_parent: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    extracted = read_jsonl(extracted_path)
    if len(extracted) != len(source_rows):
        raise ValueError(f"extracted row-count drift: {extracted_path}")
    contract = protocol["feature_contract"]
    output: list[dict[str, Any]] = []
    raw_bytes = serialized_bytes = 0
    conditions: set[tuple[str, str]] = set()
    for source, observed in zip(source_rows, extracted, strict=True):
        for key, value in source.items():
            if observed.get(key) != value:
                raise ValueError(f"{source['id']}: extracted source field drift: {key}")
        if (
            observed.get("feature_model") != contract["model_id"]
            or observed.get("feature_revision") != contract["model_revision"]
            or observed.get("feature_dtype") != contract["dtype"]
            or observed.get("feature_attention_implementation")
            != contract["attention_implementation"]
            or int(observed.get("num_feature_layers", -1))
            != int(contract["num_feature_layers"])
            or int(observed.get("per_layer_dim", -1))
            != int(contract["per_layer_dim"])
            or int(observed.get("feature_dim", -1)) != int(contract["feature_dim"])
        ):
            raise ValueError(f"{source['id']}: feature contract drift")
        hidden_path = _safe_feature_path(
            extracted_path.parent, observed["hidden_states_path"], output_root
        )
        hidden = validate_tensor_file(
            hidden_path,
            expected_shape=[len(source["output_token_ids"]), int(contract["feature_dim"])],
            expected_dtype=torch.bfloat16,
            expected_sha256=str(observed["hidden_states_sha256"]),
        )
        condition_path = _safe_feature_path(
            extracted_path.parent, observed["condition_states_path"], output_root
        )
        condition = validate_tensor_file(
            condition_path,
            expected_shape=[len(source["prompt_token_ids"]), int(contract["feature_dim"])],
            expected_dtype=torch.bfloat16,
            expected_sha256=str(observed["condition_states_sha256"]),
        )
        identity = (str(condition_path), str(observed["condition_states_sha256"]))
        if identity in conditions:
            raise ValueError("Prior-v16 feature inventory unexpectedly reused a condition")
        conditions.add(identity)
        raw_bytes += int(hidden["raw_tensor_bytes"]) + int(condition["raw_tensor_bytes"])
        serialized_bytes += int(hidden["serialized_bytes"]) + int(
            condition["serialized_bytes"]
        )
        row = dict(observed)
        row["hidden_states_path"] = os.path.relpath(hidden_path, target_parent)
        row["condition_states_path"] = os.path.relpath(condition_path, target_parent)
        output.append(row)
    return output, raw_bytes, serialized_bytes


def command_verify_feature_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    source = read_jsonl(output_root / "plan/preflight.jsonl")
    rows, raw_bytes, serialized_bytes = _verify_extracted_rows(
        source,
        output_root / "preflight/extracted.jsonl",
        output_root,
        protocol,
        output_root / "preflight",
    )
    report = {
        "schema_version": "clir-prior-v16-posthoc-feature-preflight-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_FULL_WIDTH_FEATURE_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "rows": len(rows),
        "raw_tensor_bytes": raw_bytes,
        "serialized_tensor_bytes": serialized_bytes,
        "training_allowed": False,
    }
    target = output_root / "preflight/verification.json"
    if target.exists():
        raise FileExistsError(f"feature preflight verification exists: {target}")
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2))


def _rebase_rows(
    rows: Sequence[Mapping[str, Any]], source_parent: Path, target_parent: Path
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for field in ("hidden_states_path", "condition_states_path"):
            raw = Path(str(row[field]))
            absolute = raw if raw.is_absolute() else (source_parent / raw).resolve()
            row[field] = os.path.relpath(absolute, target_parent)
        output.append(row)
    return output


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    preflight = json.loads(
        (output_root / "preflight/verification.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS_PRIOR_V16_POSTHOC_FULL_WIDTH_FEATURE_PREFLIGHT":
        raise ValueError("full-width feature preflight did not pass")
    feature_parent = output_root / "features"
    verified_rows: list[dict[str, Any]] = []
    raw_bytes = serialized_bytes = 0
    worker_count = int(protocol["feature_contract"]["worker_shards"])
    for index in range(worker_count):
        source_path = output_root / f"plan/shards/worker-{index:03d}.jsonl"
        extracted_path = output_root / f"features/worker-{index:03d}/extracted.jsonl"
        source = read_jsonl(source_path)
        if _published_identity(source_path, source) != plan["feature_shards"][str(index)]:
            raise ValueError(f"worker {index} source shard drift")
        rows, worker_raw, worker_serialized = _verify_extracted_rows(
            source, extracted_path, output_root, protocol, feature_parent
        )
        verified_rows.extend(rows)
        raw_bytes += worker_raw
        serialized_bytes += worker_serialized
    verified_rows.sort(key=lambda row: int(row["feature_inventory_index"]))
    if [int(row["feature_inventory_index"]) for row in verified_rows] != list(
        range(len(verified_rows))
    ):
        raise ValueError("merged feature order drift")
    expected = protocol["feature_contract"]["expected"]
    if len(verified_rows) != int(expected["rows"]) or raw_bytes != int(
        expected["raw_tensor_bytes"]
    ):
        raise ValueError("verified feature totals drift")

    verified_path = feature_parent / "verified_features.jsonl"
    verified_identity = _publish(
        verified_path,
        verified_rows,
        "clir-prior-v16-posthoc-verified-features-v1",
        {
            "all_shapes_dtypes_finiteness_and_checksums_verified": True,
            "posthoc_silver_no_human_verification": True,
        },
    )
    data_root = output_root / "data"
    prior_rows = _rebase_rows(verified_rows, feature_parent, data_root)
    for row in prior_rows:
        row["schema_version"] = "clir-prior-v16-posthoc-training-row-v1"
        row["experiment_population"] = "prior_v16_posthoc_binary_v1"
    prior_train = [row for row in prior_rows if row["prior_label_split"] == "train"]
    prior_dev = [row for row in prior_rows if row["prior_label_split"] == "dev"]
    historical_path = _project_path(
        protocol["frozen_inputs"]["historical_correctness_train"]["path"]
    )
    historical = read_jsonl(historical_path)
    historical = [
        {**row, "experiment_population": "historical_correctness_v1"}
        for row in historical
    ]
    direct_train = historical + prior_train
    contract = protocol["data_contract"]
    if (
        len(direct_train) != int(contract["direct_train_rows"])
        or len({str(row["query_id"]) for row in direct_train})
        != int(contract["direct_train_queries"])
        or len(prior_dev) != int(contract["prior_dev_rows"])
    ):
        raise ValueError("direct R0/P0 data inventory drift")
    direct_identity = _publish(
        data_root / "train_r0_p0.jsonl",
        direct_train,
        "clir-prior-v16-posthoc-direct-training-v1",
        {"shared_by_cells": ["r0", "p0"]},
    )
    prior_dev_identity = _publish(
        data_root / "prior_dev.jsonl",
        prior_dev,
        "clir-prior-v16-posthoc-prior-dev-v1",
        {"evaluation_only": True, "posthoc_silver_no_human_verification": True},
    )

    consistency_h0_path = _project_path(
        protocol["frozen_inputs"]["consistency_h0_train"]["path"]
    )
    h_dev_path = _project_path(protocol["frozen_inputs"]["h_dev"]["path"])
    built = build_unified_data(
        consistency_h0_train=read_jsonl(consistency_h0_path),
        prior_train=direct_train,
        h_dev=read_jsonl(h_dev_path),
        prior_dev=prior_dev,
        consistency_h0_parent=consistency_h0_path.parent,
        prior_parent=data_root,
        h_dev_parent=h_dev_path.parent,
        prior_dev_parent=data_root,
        target_parent=data_root,
        expected={
            "shared_historical_rows": int(contract["shared_historical_rows"]),
            "legacy_prior_rows": int(contract["legacy_prior_rows"]),
            "new_prior_rows": int(contract["new_prior_train_rows"]),
            "train_rows": int(contract["combined_train_rows"]),
            "train_queries": int(contract["combined_train_queries"]),
            "consistency_endpoint_rows": int(contract["consistency_endpoint_rows"]),
            "consistency_relations": int(contract["consistency_relations"]),
            "h_rows": int(contract["h_rows"]),
            "h_positive_rows": int(contract["h_positive_rows"]),
            "h_clean_rows": int(contract["h_clean_rows"]),
            "prior_rows": int(contract["combined_prior_rows"]),
            "clean_h_dev_rows": int(contract["clean_h_dev_rows"]),
            "clean_prior_dev_rows": int(contract["prior_dev_rows"]),
        },
        row_schema="clir-three-module-v16-posthoc-row-v1",
        experiment_population="three_module_v16_posthoc_v1",
        appended_prior_origin="v16_posthoc_appended_row",
    )
    if built["report"]["removed_h_dev_queries"] != contract[
        "removed_h_dev_queries"
    ] or built["report"]["removed_prior_dev_queries"] != contract[
        "removed_prior_dev_queries"
    ]:
        raise ValueError("cross-module dev removal identity drift")
    manifests = {
        "verified_features": verified_identity,
        "direct_train": direct_identity,
        "prior_dev": prior_dev_identity,
        "combined_train": _publish(
            data_root / "train_ch_full.jsonl",
            built["train"],
            "clir-three-module-v16-posthoc-training-v1",
            {"shared_by_cells": ["ch", "full"]},
        ),
        "h_dev": _publish(
            data_root / "h_dev_query_disjoint.jsonl",
            built["h_dev"],
            "clir-three-module-v16-posthoc-h-dev-v1",
            {"evaluation_only": True},
        ),
        "combined_prior_dev": _publish(
            data_root / "prior_dev_query_disjoint.jsonl",
            built["prior_dev"],
            "clir-three-module-v16-posthoc-prior-dev-v1",
            {"evaluation_only": True},
        ),
    }
    report = {
        "schema_version": "clir-prior-v16-posthoc-training-data-finalization-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_FEATURES_AND_STAGED_DATA",
        "created_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "raw_tensor_bytes": raw_bytes,
        "serialized_tensor_bytes": serialized_bytes,
        "manifests": manifests,
        "combined_report": built["report"],
        "training_allowed_after_independent_verification": False,
    }
    target = output_root / "finalization_report.json"
    if target.exists():
        raise FileExistsError(f"finalization report exists: {target}")
    atomic_write_json(target, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(target)}, indent=2))


def command_verify_final(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    report_path = output_root / "finalization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS_PRIOR_V16_POSTHOC_FEATURES_AND_STAGED_DATA"
        or report.get("code_commit") != plan["code_commit"]
        or report.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("stale or failed finalization report")
    observed: dict[str, Any] = {}
    for name, specification in report["manifests"].items():
        path = Path(specification["path"])
        rows = read_jsonl(path)
        identity = _published_identity(path, rows)
        if identity != specification:
            raise ValueError(f"final manifest drift: {name}")
        observed[name] = identity
    direct = read_jsonl(Path(observed["direct_train"]["path"]))
    combined = read_jsonl(Path(observed["combined_train"]["path"]))
    prior_dev = read_jsonl(Path(observed["prior_dev"]["path"]))
    h_dev = read_jsonl(Path(observed["h_dev"]["path"]))
    contract = protocol["data_contract"]
    if (
        len(direct) != int(contract["direct_train_rows"])
        or len(combined) != int(contract["combined_train_rows"])
        or len(prior_dev) != int(contract["prior_dev_rows"])
        or len(h_dev) != int(contract["clean_h_dev_rows"])
    ):
        raise ValueError("independent final data count drift")
    if {str(row["query_id"]) for row in combined} & {
        str(row["query_id"]) for row in prior_dev + h_dev
    }:
        raise ValueError("combined training/dev query leakage")
    verification = {
        "schema_version": "clir-prior-v16-posthoc-training-data-verification-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_STAGED_DATA_INDEPENDENT_RECOMPUTE",
        "created_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "finalization_report_file_sha256": file_sha256(report_path),
        "manifests": observed,
        "training_allowed": True,
        "claim_boundary": protocol["evidence_boundary"],
    }
    target = output_root / "finalization_verification.json"
    if target.exists():
        raise FileExistsError(f"final verification exists: {target}")
    atomic_write_json(target, verification)
    print(json.dumps(verification, indent=2))


def _gradient_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not name.startswith(prefix):
            continue
        values = parameter.grad.detach().float()
        total += float(torch.sum(values * values).cpu())
    return math.sqrt(total)


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
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError("preflight total loss is non-finite")
    losses["total"].backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("preflight gradient is non-finite")
    output_keys = (
        "scores",
        "representations",
        "hallucination_logits",
        "token_rewards",
        "gates",
        "key_prior",
        "complete_prior",
        "fused_prior",
    )
    if not all(bool(torch.isfinite(outputs[key]).all()) for key in output_keys):
        raise FloatingPointError("preflight model output is non-finite")
    return {
        "row_ids": list(raw_batch["ids"]),
        "hidden_shape": list(raw_batch["hidden_states"].shape),
        "losses": {
            key: float(value.detach().float().cpu()) for key, value in losses.items()
        },
        "gradient_norms": {
            name: _gradient_norm(model, f"{name}.")
            for name in (
                "feature_encoder",
                "projector",
                "hallucination_head",
                "token_reward_head",
                "key_prior_head",
                "complete_prior_head",
                "final_score_head",
            )
        },
    }


def _shortest_indices(rows: Sequence[Mapping[str, Any]], predicate, count: int) -> list[int]:
    indices = [index for index, row in enumerate(rows) if predicate(row)]
    indices.sort(key=lambda index: (len(rows[index]["output_token_ids"]), str(rows[index]["id"])))
    if len(indices) < count:
        raise ValueError("not enough representative preflight rows")
    return indices[:count]


def _representative_batches(dataset: CLIRTrajectoryDataset, stage: int) -> dict[str, Any]:
    rows = dataset.rows
    batches: dict[str, Any] = {}
    prior = _shortest_indices(
        rows,
        lambda row: row.get("key_prior_target") is not None
        and row.get("complete_prior_target") is not None,
        4,
    )
    batches["prior"] = clir_collate([dataset[index] for index in prior])
    if stage == 1:
        return batches
    relation_members: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("consistency_supervision") is True:
            relation_members.setdefault(str(row.get("semantic_id")), []).append(index)
    relations = sorted(
        (indices for indices in relation_members.values() if len(indices) == 2),
        key=lambda indices: (
            max(len(rows[index]["output_token_ids"]) for index in indices),
            str(rows[indices[0]]["semantic_id"]),
        ),
    )
    if len(relations) < 2:
        raise ValueError("not enough Consistency relations for preflight")
    consistency = relations[0] + relations[1]
    positive = _shortest_indices(
        rows, lambda row: row.get("hallucination_onset", -1) not in (-1, None), 2
    )
    clean = _shortest_indices(
        rows,
        lambda row: row.get("hallucination_onset") == -1
        and "path_hallucinated" in row,
        2,
    )
    batches["consistency"] = clir_collate(
        [dataset[index] for index in consistency]
    )
    batches["hallucination"] = clir_collate(
        [dataset[index] for index in clean + positive]
    )
    return batches


def command_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    verification = json.loads(
        (output_root / "finalization_verification.json").read_text(encoding="utf-8")
    )
    if verification.get("status") != (
        "PASS_PRIOR_V16_POSTHOC_STAGED_DATA_INDEPENDENT_RECOMPUTE"
    ) or verification.get("code_commit") != plan["code_commit"]:
        raise ValueError("missing or stale final data verification")
    stage = int(args.stage)
    cells = protocol["training"][f"stage_{stage}_cells"]
    manifest_name = "direct_train" if stage == 1 else "combined_train"
    manifest_path = Path(verification["manifests"][manifest_name]["path"])
    dataset = CLIRTrajectoryDataset(manifest_path)
    summary = supervision_summary(dataset, list(range(len(dataset))))
    batches = _representative_batches(dataset, stage)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but unavailable")
    reports: dict[str, Any] = {}
    for cell in cells:
        config_path = _project_path(protocol["cells"][cell]["config"])
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
        validate_feature_contract(dataset, model_config, f"{cell}-preflight")
        validate_supervision_coverage(summary, model_config)
        set_seed(int(training["seed"]))
        model = ConsistencyLocalizedReward(model_config).to(device)
        cell_batches = {
            name: _run_batch(model, batch, device, str(training["amp_dtype"]))
            for name, batch in batches.items()
        }
        factors = protocol["cells"][cell]["factors"]
        if factors[2]:
            required = {"prior_key", "prior_complete", "prior_total"}
            if not required <= set(cell_batches["prior"]["losses"]):
                raise ValueError(f"{cell}: direct Prior losses were not exercised")
            for head in ("key_prior_head", "complete_prior_head"):
                if cell_batches["prior"]["gradient_norms"][head] <= 0.0:
                    raise ValueError(f"{cell}: {head} did not receive a gradient")
        elif "prior_total" in cell_batches["prior"]["losses"]:
            raise ValueError(f"{cell}: disabled Prior objective was routed")
        if stage == 2:
            if factors[0] and "consistency_total" not in cell_batches["consistency"]["losses"]:
                raise ValueError(f"{cell}: Consistency loss was not exercised")
            if factors[1] and "localization_token_bce" not in cell_batches["hallucination"]["losses"]:
                raise ValueError(f"{cell}: H0 loss was not exercised")
        if any(
            batch["gradient_norms"]["feature_encoder"] <= 0.0
            or batch["gradient_norms"]["final_score_head"] <= 0.0
            for batch in cell_batches.values()
        ):
            raise ValueError(f"{cell}: base reward path lost gradients")
        reports[cell] = {
            "config_file_sha256": file_sha256(config_path),
            "factors": factors,
            "batches": cell_batches,
            "passed": True,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": "clir-prior-v16-posthoc-training-preflight-v1",
        "status": f"PASS_PRIOR_V16_POSTHOC_STAGE_{stage}_FULL_WIDTH_BATCH_PREFLIGHT",
        "created_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "stage": stage,
        "train_manifest": verification["manifests"][manifest_name],
        "supervision_per_epoch": summary,
        "cells": reports,
        "device": str(device),
        "training_allowed": True,
    }
    target = output_root / f"training_preflight/stage-{stage}.json"
    if target.exists():
        raise FileExistsError(f"training preflight exists: {target}")
    atomic_write_json(target, report)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("verify-feature-preflight").set_defaults(
        func=command_verify_feature_preflight
    )
    subparsers.add_parser("finalize").set_defaults(func=command_finalize)
    subparsers.add_parser("verify-final").set_defaults(func=command_verify_final)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--stage", type=int, required=True, choices=[1, 2])
    preflight.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    preflight.set_defaults(func=command_preflight)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
