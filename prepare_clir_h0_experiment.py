#!/usr/bin/env python
"""Prepare, verify, and materialize the exploratory H0 v7.4 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from src.clir_data import read_jsonl
from src.clir_h0_experiment import (
    assign_feature_workers,
    build_feature_inventory,
    inventory_statistics,
    rebase_feature_paths,
    select_fully_labeled_ranking,
    validate_h_partition,
)
from src.clir_scale_features import validate_tensor_file
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/h0_experiment_v7_4/protocol.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/h0_experiment_v7_4"
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
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("H0 v7.4 experiment commands require a clean worktree")
    return _git_head()


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
    )
    if result.returncode:
        raise ValueError("frozen parent commit is not an ancestor of current HEAD")


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"pinned {label} hash drift: {observed} != {expected}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-h0-v7.4-posthoc-exploratory-experiment-protocol"
        or protocol.get("status")
        != "AUTHORIZED_H0_V7_4_POSTHOC_EXPLORATORY_MATCHED_EXPERIMENT"
    ):
        raise ValueError("unsupported or inactive H0 v7.4 experiment protocol")
    if protocol["original_evidence_status"] != {
        "ranking_v7_h0_status": "FAIL_H0_V7_RESERVE",
        "failure_is_preserved": True,
        "salvage_evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "salvage_may_be_called_gold_or_confirmatory": False,
    }:
        raise ValueError("original v7 failure or claim boundary drift")
    for name, specification in protocol["frozen_inputs"].items():
        source_path = _project_path(specification["path"])
        _assert_hash(source_path, specification["file_sha256"], name)
        if "sidecar_file_sha256" in specification:
            sidecar = source_path.with_suffix(source_path.suffix + ".manifest.json")
            _assert_hash(sidecar, specification["sidecar_file_sha256"], f"{name} sidecar")
    for name, specification in protocol["execution_configs"].items():
        _assert_hash(
            _project_path(specification["path"]),
            specification["file_sha256"],
            f"{name} config",
        )
    return protocol


def _authorized_output_root(protocol: Mapping[str, Any], requested: str | Path) -> Path:
    expected = _project_path(protocol["runtime_contract"]["output_root"]).resolve()
    observed = Path(requested).resolve()
    if observed != expected:
        raise ValueError(f"output root drift: {observed} != {expected}")
    return observed


def _config_gate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payloads = {
        name: json.loads(_project_path(spec["path"]).read_text(encoding="utf-8"))
        for name, spec in protocol["execution_configs"].items()
    }
    expected_weights = {
        "C0": (0.0, 0.0),
        "C1": (1.0, 0.0),
        "H0": (0.0, 1.0),
        "CH0": (1.0, 1.0),
    }
    normalized = []
    observed: dict[str, Any] = {}
    for cell, payload in payloads.items():
        clone = json.loads(json.dumps(payload))
        consistency = clone["model"].pop("consistency_weight")
        hallucination = clone["model"].pop("hallucination_weight")
        if (consistency, hallucination) != expected_weights[cell]:
            raise ValueError(f"{cell}: objective-weight drift")
        for forbidden in (
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "prior_weight",
            "prior_distill_weight",
            "gate_prior_weight",
        ):
            if float(clone["model"][forbidden]) != 0.0:
                raise ValueError(f"{cell}: forbidden objective {forbidden} is enabled")
        normalized.append(clone)
        observed[cell] = {
            "consistency_weight": consistency,
            "hallucination_weight": hallucination,
            "file_sha256": file_sha256(
                _project_path(protocol["execution_configs"][cell]["path"])
            ),
        }
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError("cell configs differ outside consistency/H0 weights")
    return observed


def _source_key(row: Mapping[str, Any]) -> tuple[str, str]:
    value = row.get("source_record_id", row.get("source_index"))
    if value is None:
        value = row.get("query_id")
    return str(row.get("source", "")).lower(), str(value)


def _overlap_report(
    base: Sequence[Mapping[str, Any]],
    h_train: Sequence[Mapping[str, Any]],
    h_dev: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    populations = {
        "base": base,
        "h_train": h_train,
        "h_dev": h_dev,
        "ranking": ranking,
    }
    result: dict[str, Any] = {}
    names = list(populations)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_rows, right_rows = populations[left], populations[right]
            query_overlap = {str(row["query_id"]) for row in left_rows} & {
                str(row["query_id"]) for row in right_rows
            }
            source_overlap = {_source_key(row) for row in left_rows} & {
                _source_key(row) for row in right_rows
            }
            cluster_overlap = {
                str(row["cluster_id"])
                for row in left_rows
                if row.get("cluster_id")
            } & {
                str(row["cluster_id"])
                for row in right_rows
                if row.get("cluster_id")
            }
            key = f"{left}__{right}"
            result[key] = {
                "query_id": len(query_overlap),
                "source_record": len(source_overlap),
                "cluster_id": len(cluster_overlap),
            }
            if query_overlap or source_overlap or cluster_overlap:
                raise ValueError(f"population overlap at {key}: {result[key]}")
    return result


def _construct(protocol: Mapping[str, Any]) -> dict[str, Any]:
    inputs = protocol["frozen_inputs"]
    base = read_jsonl(_project_path(inputs["base_c0_c1_train"]["path"]))
    h_train = read_jsonl(_project_path(inputs["h_train"]["path"]))
    h_dev = read_jsonl(_project_path(inputs["h_dev"]["path"]))
    ranking_all = read_jsonl(_project_path(inputs["ranking_checked"]["path"]))
    if len(base) != int(inputs["base_c0_c1_train"]["row_count"]):
        raise ValueError("base training row count drift")
    h_statistics = {
        "train": validate_h_partition(
            h_train, split="train", expected_clean=200, expected_positive=200
        ),
        "dev": validate_h_partition(
            h_dev, split="dev", expected_clean=100, expected_positive=100
        ),
    }
    ranking, ranking_statistics = select_fully_labeled_ranking(ranking_all)
    expected_ranking = protocol["ranking_evaluation_subset"]
    for field, expected_field in (
        ("selected_queries", "selected_queries"),
        ("selected_rows", "selected_trajectories"),
        ("informative_queries_with_both_labels", "informative_queries_with_at_least_one_correct_and_one_incorrect"),
    ):
        if ranking_statistics[field] != int(expected_ranking[expected_field]):
            raise ValueError(f"ranking subset {field} drift")
    overlaps = _overlap_report(base, h_train, h_dev, ranking)
    inventory = build_feature_inventory(h_train, h_dev, ranking)
    inventory, worker_statistics = assign_feature_workers(
        inventory, int(protocol["feature_contract"]["worker_count"])
    )
    statistics = inventory_statistics(inventory)
    expected_inventory = protocol["feature_contract"]["expected"]
    for field, expected in expected_inventory.items():
        if field == "raw_feature_bytes":
            continue
        if int(statistics[field]) != int(expected):
            raise ValueError(f"feature inventory {field} drift")
    raw_bytes = statistics["total_feature_token_count"] * int(
        protocol["feature_contract"]["bytes_per_feature_token"]
    )
    if raw_bytes != int(expected_inventory["raw_feature_bytes"]):
        raise ValueError("raw feature byte budget drift")
    return {
        "base": base,
        "h_train": h_train,
        "h_dev": h_dev,
        "ranking": ranking,
        "inventory": inventory,
        "h_statistics": h_statistics,
        "ranking_statistics": ranking_statistics,
        "inventory_statistics": statistics,
        "worker_statistics": worker_statistics,
        "overlaps": overlaps,
        "raw_feature_bytes": raw_bytes,
    }


def _manifest_identity(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "sidecar_file_sha256": file_sha256(sidecar),
        "row_count": len(rows),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def command_prepare(args: argparse.Namespace) -> None:
    code_commit = _require_clean_commit()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    _require_ancestor(protocol["frozen_parent_commit"], code_commit)
    output_root = _authorized_output_root(protocol, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"H0 v7.4 output root is not empty: {output_root}")
    constructed = _construct(protocol)
    config_gate = _config_gate(protocol)
    plan_root = output_root / "plan"
    plan_root.mkdir(parents=True)
    inventory_path = plan_root / "selected_feature_inventory.jsonl"
    publish_manifest(
        inventory_path,
        constructed["inventory"],
        schema_version="clir-h0-v7.4-selected-feature-inventory",
        metadata={"posthoc_exploratory": True, "original_v7_status": "FAIL_H0_V7_RESERVE"},
    )
    workers: list[dict[str, Any]] = []
    worker_count = int(protocol["feature_contract"]["worker_count"])
    for worker_index in range(worker_count):
        rows = [
            row
            for row in constructed["inventory"]
            if int(row["feature_worker_index"]) == worker_index
        ]
        shard_path = plan_root / f"worker-{worker_index:03d}.jsonl"
        publish_manifest(
            shard_path,
            rows,
            schema_version="clir-h0-v7.4-feature-worker-input",
            metadata={"worker_index": worker_index, "worker_count": worker_count},
        )
        workers.append(_manifest_identity(shard_path, rows))
    largest = max(
        constructed["inventory"],
        key=lambda row: (
            int(row["prompt_token_count"]) + int(row["output_token_count"]),
            str(row["id"]),
        ),
    )
    preflight_path = plan_root / "preflight.jsonl"
    publish_manifest(
        preflight_path,
        [largest],
        schema_version="clir-h0-v7.4-feature-preflight-input",
        metadata={"selection": "largest_prompt_plus_output_token_count"},
    )
    report = {
        "schema_version": "clir-h0-v7.4-experiment-plan",
        "status": "PASS_H0_V7_4_EXPERIMENT_PLAN",
        "planned_at_utc": _utc_now(),
        "code_commit": code_commit,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "config_gate": config_gate,
        "h_statistics": constructed["h_statistics"],
        "ranking_statistics": constructed["ranking_statistics"],
        "overlap_report": constructed["overlaps"],
        "inventory_statistics": constructed["inventory_statistics"],
        "raw_feature_bytes": constructed["raw_feature_bytes"],
        "inventory": _manifest_identity(inventory_path, constructed["inventory"]),
        "workers": workers,
        "worker_statistics": constructed["worker_statistics"],
        "preflight": _manifest_identity(preflight_path, [largest]),
        "feature_extraction_allowed": True,
        "training_allowed": False,
    }
    report_path = plan_root / "plan_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(report_path)}, indent=2))


def _load_plan(protocol_path: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    _authorized_output_root(protocol, output_root)
    plan_path = output_root / "plan/plan_report.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_H0_V7_4_EXPERIMENT_PLAN":
        raise ValueError("H0 v7.4 plan did not pass")
    if plan.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("protocol changed after planning")
    if plan.get("code_commit") != _require_clean_commit():
        raise ValueError("code commit changed after H0 v7.4 planning")
    return protocol, plan


def _assert_published(path: Path, expected: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    if _manifest_identity(path, rows) != dict(expected):
        raise ValueError(f"published manifest drift: {path}")


def command_verify_plan(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    constructed = _construct(protocol)
    inventory_path = output_root / "plan/selected_feature_inventory.jsonl"
    _assert_published(inventory_path, plan["inventory"], constructed["inventory"])
    for worker_index, expected in enumerate(plan["workers"]):
        rows = [
            row
            for row in constructed["inventory"]
            if int(row["feature_worker_index"]) == worker_index
        ]
        _assert_published(output_root / f"plan/worker-{worker_index:03d}.jsonl", expected, rows)
    largest = max(
        constructed["inventory"],
        key=lambda row: (
            int(row["prompt_token_count"]) + int(row["output_token_count"]),
            str(row["id"]),
        ),
    )
    _assert_published(output_root / "plan/preflight.jsonl", plan["preflight"], [largest])
    report = {
        "schema_version": "clir-h0-v7.4-experiment-plan-verification",
        "status": "PASS_H0_V7_4_EXPERIMENT_PLAN_RECOMPUTATION",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "plan_report_file_sha256": file_sha256(output_root / "plan/plan_report.json"),
        "inventory_ordered_rows_sha256": canonical_sha256(constructed["inventory"]),
        "ranking_selected_rows": len(constructed["ranking"]),
        "training_allowed": False,
    }
    path = output_root / "plan/independent_verification.json"
    if path.exists():
        raise FileExistsError(f"plan verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


def _safe_feature_path(parent: Path, value: Any, output_root: Path) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (parent / raw).resolve()
    if not path.is_relative_to(output_root.resolve()):
        raise ValueError(f"feature path escapes output root: {path}")
    return path


def _verify_extracted(
    source_rows: Sequence[Mapping[str, Any]],
    extracted_path: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    extracted = read_jsonl(extracted_path)
    if len(extracted) != len(source_rows):
        raise ValueError("extracted row count drift")
    contract = protocol["feature_contract"]
    conditions: dict[str, dict[str, Any]] = {}
    raw_bytes = 0
    serialized_bytes = 0
    for source, row in zip(source_rows, extracted, strict=True):
        for field in (
            "id",
            "trajectory_id",
            "query_id",
            "candidate_index",
            "prompt_token_ids",
            "output_token_ids",
            "feature_role",
            "feature_worker_index",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"{source['id']}: extracted field drift: {field}")
        if (
            row.get("feature_model") != contract["model_id"]
            or row.get("feature_revision") != contract["model_revision"]
            or row.get("feature_dtype") != contract["dtype"]
            or row.get("feature_attention_implementation")
            != contract["attention_implementation"]
            or int(row.get("feature_dim", -1)) != int(contract["feature_dim"])
            or int(row.get("num_feature_layers", -1))
            != int(contract["num_feature_layers"])
            or int(row.get("per_layer_dim", -1)) != int(contract["per_layer_dim"])
        ):
            raise ValueError(f"{source['id']}: feature contract drift")
        hidden_path = _safe_feature_path(
            extracted_path.parent, row["hidden_states_path"], output_root
        )
        hidden = validate_tensor_file(
            hidden_path,
            expected_shape=[len(source["output_token_ids"]), int(contract["feature_dim"])],
            expected_dtype=torch.bfloat16,
            expected_sha256=str(row["hidden_states_sha256"]),
        )
        raw_bytes += int(hidden["raw_tensor_bytes"])
        serialized_bytes += int(hidden["serialized_bytes"])
        query_id = str(source["query_id"])
        condition_path = _safe_feature_path(
            extracted_path.parent, row["condition_states_path"], output_root
        )
        if query_id not in conditions:
            condition = validate_tensor_file(
                condition_path,
                expected_shape=[len(source["prompt_token_ids"]), int(contract["feature_dim"])],
                expected_dtype=torch.bfloat16,
                expected_sha256=str(row["condition_states_sha256"]),
            )
            conditions[query_id] = condition
            raw_bytes += int(condition["raw_tensor_bytes"])
            serialized_bytes += int(condition["serialized_bytes"])
        elif (
            conditions[query_id]["path"] != str(condition_path)
            or conditions[query_id]["sha256"] != row["condition_states_sha256"]
        ):
            raise ValueError(f"{query_id}: condition feature drift within query")
    return {
        "rows": extracted,
        "trajectory_count": len(extracted),
        "condition_count": len(conditions),
        "raw_tensor_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
    }


def command_verify_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    result = _verify_extracted(
        read_jsonl(output_root / "plan/preflight.jsonl"),
        output_root / "preflight/extracted.jsonl",
        output_root,
        protocol,
    )
    report = {
        "schema_version": "clir-h0-v7.4-feature-preflight-verification",
        "status": "PASS_H0_V7_4_FULL_WIDTH_FEATURE_PREFLIGHT",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "trajectory_count": result["trajectory_count"],
        "condition_count": result["condition_count"],
        "raw_tensor_bytes": result["raw_tensor_bytes"],
        "training_allowed": False,
    }
    path = output_root / "preflight/verification.json"
    if path.exists():
        raise FileExistsError(f"preflight verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


def command_verify_worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    preflight = json.loads((output_root / "preflight/verification.json").read_text())
    if preflight.get("status") != "PASS_H0_V7_4_FULL_WIDTH_FEATURE_PREFLIGHT":
        raise ValueError("feature preflight did not pass")
    worker_index = int(args.worker_index)
    worker_count = int(protocol["feature_contract"]["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker index is outside the frozen range")
    source_path = output_root / f"plan/worker-{worker_index:03d}.jsonl"
    extracted_path = output_root / f"features/worker-{worker_index:03d}.jsonl"
    result = _verify_extracted(
        read_jsonl(source_path), extracted_path, output_root, protocol
    )
    expected = plan["worker_statistics"][worker_index]
    if (
        result["trajectory_count"] != int(expected["trajectory_count"])
        or result["condition_count"] != int(expected["query_count"])
        or result["raw_tensor_bytes"]
        != int(expected["feature_token_count"])
        * int(protocol["feature_contract"]["bytes_per_feature_token"])
    ):
        raise ValueError("worker feature totals drift")
    report = {
        "schema_version": "clir-h0-v7.4-feature-worker-verification",
        "status": "PASS_H0_V7_4_FEATURE_WORKER_VERIFICATION",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "worker_index": worker_index,
        "source_file_sha256": file_sha256(source_path),
        "extracted_file_sha256": file_sha256(extracted_path),
        "trajectory_count": result["trajectory_count"],
        "condition_count": result["condition_count"],
        "raw_tensor_bytes": result["raw_tensor_bytes"],
        "serialized_bytes": result["serialized_bytes"],
        "all_shapes_dtypes_finiteness_and_checksums_verified": True,
        "training_allowed": False,
    }
    path = output_root / f"verification/worker-{worker_index:03d}.json"
    if path.exists():
        raise FileExistsError(f"worker verification already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


def _publish(path: Path, rows: Sequence[Mapping[str, Any]], schema: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    manifest = publish_manifest(path, rows, schema_version=schema, metadata=metadata)
    manifest["sidecar_file_sha256"] = file_sha256(
        path.with_suffix(path.suffix + ".manifest.json")
    )
    return manifest


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    constructed = _construct(protocol)
    worker_count = int(protocol["feature_contract"]["worker_count"])
    extracted: list[dict[str, Any]] = []
    worker_reports: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        report_path = output_root / f"verification/worker-{worker_index:03d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS_H0_V7_4_FEATURE_WORKER_VERIFICATION":
            raise ValueError(f"worker {worker_index} verification did not pass")
        extracted.extend(read_jsonl(output_root / f"features/worker-{worker_index:03d}.jsonl"))
        worker_reports.append(
            {
                "worker_index": worker_index,
                "report_file_sha256": file_sha256(report_path),
                "raw_tensor_bytes": report["raw_tensor_bytes"],
            }
        )
    extracted_by_id = {str(row["id"]): row for row in extracted}
    if len(extracted_by_id) != len(extracted):
        raise ValueError("duplicate extracted trajectory id")
    ordered = [extracted_by_id[str(row["id"])] for row in constructed["inventory"]]
    if len(ordered) != int(protocol["feature_contract"]["expected"]["trajectory_count"]):
        raise ValueError("final extracted feature count drift")
    features_root = output_root / "features"
    extracted_path = features_root / "extracted_features.jsonl"
    extracted_manifest = _publish(
        extracted_path,
        ordered,
        "clir-h0-v7.4-extracted-feature-manifest",
        {
            "independently_verified_workers": worker_count,
            "posthoc_exploratory": True,
            "original_v7_status": "FAIL_H0_V7_RESERVE",
        },
    )

    data_root = output_root / "data"
    base_parent = _project_path(
        protocol["frozen_inputs"]["base_c0_c1_train"]["path"]
    ).parent
    feature_parent = extracted_path.parent
    base_rows = [
        {
            **rebase_feature_paths(row, source_parent=base_parent, target_parent=data_root),
            "experiment_population": "base_c0_c1_v6_1",
        }
        for row in constructed["base"]
    ]
    h_train_rows = [
        {
            **rebase_feature_paths(row, source_parent=feature_parent, target_parent=data_root),
            "schema_version": "clir-h0-v7.4-matched-training-row",
            "experiment_population": "h0_v7_4_posthoc_train",
        }
        for row in ordered
        if row["feature_role"] == "h_train"
    ]
    h_dev_rows = [
        {
            **rebase_feature_paths(row, source_parent=feature_parent, target_parent=data_root),
            "schema_version": "clir-h0-v7.4-h-dev-row",
            "experiment_population": "h0_v7_4_posthoc_dev",
        }
        for row in ordered
        if row["feature_role"] == "h_dev"
    ]
    ranking_rows = [
        {
            **rebase_feature_paths(row, source_parent=feature_parent, target_parent=data_root),
            "schema_version": "clir-h0-v7.4-ranking-evaluation-row",
            "experiment_population": "ranking_v7_fully_labeled_evaluation",
        }
        for row in ordered
        if row["feature_role"] == "ranking_evaluation"
    ]
    train_rows = base_rows + h_train_rows
    if (len(train_rows), len(h_dev_rows), len(ranking_rows)) != (5168, 200, 14272):
        raise ValueError("final training/dev/ranking row counts drift")
    manifests = {
        "train": _publish(
            data_root / "train_c0_c1_h0_ch0.jsonl",
            train_rows,
            "clir-h0-v7.4-shared-matched-training-manifest",
            {"shared_by_cells": ["C0", "C1", "H0", "CH0"]},
        ),
        "h_dev": _publish(
            data_root / "h_dev.jsonl",
            h_dev_rows,
            "clir-h0-v7.4-h-dev-manifest",
            {"evaluation_only": True, "posthoc_exploratory": True},
        ),
        "ranking": _publish(
            data_root / "ranking_evaluation.jsonl",
            ranking_rows,
            "clir-h0-v7.4-fully-labeled-ranking-manifest",
            {"evaluation_only": True, "queries": 892, "candidate_count": 16},
        ),
    }
    report = {
        "schema_version": "clir-h0-v7.4-feature-and-data-finalization",
        "status": "PASS_H0_V7_4_FEATURES_AND_MATCHED_DATA",
        "completed_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "inventory_statistics": constructed["inventory_statistics"],
        "worker_reports": worker_reports,
        "raw_tensor_bytes": sum(int(report["raw_tensor_bytes"]) for report in worker_reports),
        "extracted_features": extracted_manifest,
        "manifests": manifests,
        "same_train_manifest_all_cells": True,
        "H1_prior_and_full_disabled": True,
        "feature_extraction_completed": True,
        "training_allowed": True,
        "next_gate": "FULL_WIDTH_FOUR_CELL_TRAINING_PREFLIGHT_THEN_THREE_SEEDS",
    }
    if report["raw_tensor_bytes"] != int(
        protocol["feature_contract"]["expected"]["raw_feature_bytes"]
    ):
        raise ValueError("final raw tensor bytes drift")
    report_path = output_root / "finalization_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(report_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("verify-plan").set_defaults(func=command_verify_plan)
    subparsers.add_parser("verify-preflight").set_defaults(func=command_verify_preflight)
    verify_worker = subparsers.add_parser("verify-worker")
    verify_worker.add_argument("--worker-index", type=int, required=True)
    verify_worker.set_defaults(func=command_verify_worker)
    subparsers.add_parser("finalize").set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
