#!/usr/bin/env python
"""Extract and independently verify the selected CLIR scale-v6.1 features.

The command is intentionally separate from the generic extractor.  It binds
the published v6.1 inventory, shards by original query, resumes only from
atomic payloads/query markers, and keeps training disabled after extraction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from extract_hidden_states import atomic_torch_save, extract_row
from src.clir_scale_features import (
    AUTHORIZATION_SCHEMA,
    EXTRACTED_ROW_SCHEMA,
    PLAN_SCHEMA,
    QUERY_MARKER_SCHEMA,
    SELECTED_INPUT_SCHEMA,
    VERIFIER_REPORT_SCHEMA,
    WORKER_REPORT_SCHEMA,
    assign_workers,
    build_selected_inputs,
    condition_relative_path,
    expected_payload_records,
    payload_record_digest,
    query_marker_relative_path,
    rows_for_worker,
    selected_statistics,
    trajectory_relative_path,
    validate_tensor_file,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_scale_v6/features_v6_1_run3"
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


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _require_clean_commit() -> str:
    if _git_dirty():
        raise ValueError("scale-v6.1 feature commands require a clean worktree")
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
    if result.returncode != 0:
        raise ValueError(
            f"authorized parent commit {ancestor} is not an ancestor of {descendant}"
        )


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"pinned {label} is missing: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"pinned {label} hash drift: {observed} != {expected}")


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization_path = Path(path).resolve()
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported feature-extraction authorization schema")
    if authorization.get("status") != "AUTHORIZED_SELECTED_INVENTORY_EXTRACTION_ONLY":
        raise ValueError("feature-extraction authorization is not active")
    scope = authorization["authorized_scope"]
    required_true = (
        "selected_inventory_materialization",
        "full_width_preflight",
        "inventory_only_feature_extraction",
        "independent_payload_verification",
        "publish_extracted_feature_manifest",
    )
    required_false = (
        "all_16000_rollout_feature_extraction",
        "new_rollout",
        "new_ai_annotation_or_label_change",
        "relation_or_threshold_change",
        "training",
    )
    if any(scope.get(field) is not True for field in required_true):
        raise ValueError("authorization is missing a required extraction capability")
    if any(scope.get(field) is not False for field in required_false):
        raise ValueError("authorization unexpectedly expands extraction scope")

    for field, spec in authorization["frozen_parent"]["files"].items():
        _assert_hash(_project_path(spec["path"]), spec["file_sha256"], field)
    return authorization


def _authorized_output_root(
    authorization: Mapping[str, Any], requested: str | Path
) -> Path:
    expected = _project_path(authorization["runtime_contract"]["output_root"]).resolve()
    observed = Path(requested).resolve()
    if observed != expected:
        raise ValueError(f"output root drift: {observed} != {expected}")
    return observed


def _read_published(path: Path, expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    _assert_hash(path, expected["file_sha256"], path.name)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    _assert_hash(sidecar, expected["sidecar_file_sha256"], sidecar.name)
    rows = read_jsonl(path)
    if len(rows) != expected["row_count"]:
        raise ValueError(f"published row count drift: {path}")
    if canonical_sha256(rows) != expected["ordered_rows_sha256"]:
        raise ValueError(f"published ordered-row hash drift: {path}")
    return rows


def _relation_endpoint_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    endpoints: set[str] = set()
    for row in rows:
        endpoints.add(str(row["left_id"]))
        endpoints.add(str(row["right_id"]))
    return endpoints


def command_prepare(args: argparse.Namespace) -> None:
    code_commit = _require_clean_commit()
    authorization_path = Path(args.authorization).resolve()
    authorization = load_authorization(authorization_path)
    _require_ancestor(
        authorization["frozen_parent"]["authorized_from_clean_parent_commit"],
        code_commit,
    )
    output_root = _authorized_output_root(authorization, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"feature output root is not empty: {output_root}")

    frozen = authorization["frozen_parent"]
    files = frozen["files"]
    inventory_spec = files["selected_feature_inventory"]
    inventory_path = _project_path(inventory_spec["path"])
    inventory = _read_published(inventory_path, inventory_spec)
    materialized_path = _project_path(files["materialized_rows"]["path"])
    materialized = read_jsonl(materialized_path)
    selected = build_selected_inputs(inventory, materialized)
    del materialized
    gc.collect()

    relation_endpoints: set[str] = set()
    relation_counts: dict[str, int] = {}
    for label in (
        "train_positive_relations",
        "heldout_positive_relations",
        "heldout_hard_negative_relations",
    ):
        spec = files[label]
        rows = _read_published(_project_path(spec["path"]), spec)
        relation_counts[label] = len(rows)
        relation_endpoints.update(_relation_endpoint_ids(rows))
    inventory_ids = {str(row["trajectory_id"]) for row in selected}
    if relation_endpoints != inventory_ids:
        raise ValueError(
            "selected inventory does not exactly equal the relation endpoint union"
        )

    worker_count = int(authorization["runtime_contract"]["worker_count"])
    selected, worker_stats = assign_workers(selected, worker_count)
    stats = selected_statistics(selected)
    expected = authorization["expected_inventory"]
    for field in (
        "trajectory_count",
        "query_count",
        "condition_count",
        "output_token_count",
        "prompt_token_count",
        "total_feature_token_count",
    ):
        if stats[field] != expected[field]:
            raise ValueError(f"selected inventory {field} drift")
    raw_bytes = stats["total_feature_token_count"] * int(
        authorization["feature_contract"]["bytes_per_feature_token"]
    )
    if raw_bytes != expected["raw_feature_bytes"]:
        raise ValueError("selected inventory raw feature byte count drift")

    plan_dir = output_root / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    selected_path = plan_dir / "selected_rows.jsonl"
    selected_manifest = publish_manifest(
        selected_path,
        selected,
        schema_version=("clir-consistency-scale-selected-feature-input-manifest-v6.1"),
        metadata={"source_inventory_file_sha256": inventory_spec["file_sha256"]},
    )
    selected_manifest["sidecar_file_sha256"] = file_sha256(
        selected_path.with_suffix(selected_path.suffix + ".manifest.json")
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "PASS_SELECTED_INVENTORY_EXTRACTION_PLAN_V6_1",
        "planned_at_utc": _utc_now(),
        "code_commit": code_commit,
        "authorization_path": str(authorization_path),
        "authorization_file_sha256": file_sha256(authorization_path),
        "output_root": str(output_root),
        "selected_rows": selected_manifest,
        "inventory_statistics": stats,
        "raw_feature_bytes": raw_bytes,
        "relation_counts": relation_counts,
        "relation_endpoint_count": len(relation_endpoints),
        "worker_count": worker_count,
        "worker_assignment": (
            "deterministic_largest_first_feature_token_balanced_by_query_v1"
        ),
        "worker_statistics": worker_stats,
        "feature_contract": dict(authorization["feature_contract"]),
        "feature_extraction_allowed": True,
        "training_allowed": False,
    }
    plan_path = plan_dir / "extraction_plan.json"
    atomic_write_json(plan_path, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_path": str(plan_path),
                "plan_file_sha256": file_sha256(plan_path),
                "inventory_statistics": stats,
                "raw_feature_bytes": raw_bytes,
                "worker_statistics": worker_stats,
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_plan(
    *, authorization_path: Path, output_root: Path, require_current_commit: bool = True
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    authorization = load_authorization(authorization_path)
    _authorized_output_root(authorization, output_root)
    plan_path = output_root / "plan/extraction_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("feature extraction plan schema drift")
    if plan.get("status") != "PASS_SELECTED_INVENTORY_EXTRACTION_PLAN_V6_1":
        raise ValueError("selected inventory extraction plan did not pass")
    if plan.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("feature extraction authorization drift after planning")
    if require_current_commit and plan.get("code_commit") != _require_clean_commit():
        raise ValueError("feature extraction plan code commit drift")
    selected_spec = plan["selected_rows"]
    selected_path = Path(selected_spec["path"])
    selected = _read_published(selected_path, selected_spec)
    if any(row.get("schema_version") != SELECTED_INPUT_SCHEMA for row in selected):
        raise ValueError("selected feature input row schema drift")
    if selected_statistics(selected) != plan["inventory_statistics"]:
        raise ValueError("selected feature input statistics drift")
    return authorization, plan, selected, plan_path


def command_verify_plan(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    authorization, plan, selected, plan_path = _load_plan(
        authorization_path=Path(args.authorization).resolve(),
        output_root=output_root,
    )
    expected_records = expected_payload_records(output_root, selected)
    if len(expected_records) != (
        authorization["expected_inventory"]["trajectory_count"]
        + authorization["expected_inventory"]["condition_count"]
    ):
        raise ValueError("planned payload count drift")
    report = {
        "schema_version": "clir-consistency-scale-feature-plan-verification-v6.1",
        "status": "PASS_SELECTED_INVENTORY_EXTRACTION_PLAN_VERIFIED_V6_1",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "plan_file_sha256": file_sha256(plan_path),
        "selected_rows_file_sha256": plan["selected_rows"]["file_sha256"],
        "planned_payload_count": len(expected_records),
        "planned_raw_feature_bytes": plan["raw_feature_bytes"],
        "all_16000_rollouts_selected": False,
        "training_allowed": False,
    }
    path = output_root / "plan/independent_plan_verification.json"
    if path.exists():
        raise FileExistsError(f"plan verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_model(
    authorization: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, str | None]:
    from transformers import AutoModelForCausalLM

    contract = authorization["feature_contract"]
    runtime = authorization["runtime_contract"]
    dtype = {"bfloat16": torch.bfloat16}[contract["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        contract["model_id"],
        revision=contract["model_revision"],
        cache_dir=str(_project_path(runtime["cache_dir"])),
        torch_dtype=dtype,
        trust_remote_code=False,
        attn_implementation=contract["attention_implementation"],
    ).to(device)
    model.eval()
    config = model.config
    if int(config.hidden_size) != int(contract["per_layer_dim"]):
        raise ValueError("loaded model hidden width drift")
    if int(config.num_hidden_layers) + 1 != int(contract["num_feature_layers"]):
        raise ValueError("loaded model hidden layer count drift")
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision not in (None, contract["model_revision"]):
        raise ValueError("loaded model revision drift")
    return model, resolved_revision


def _record_new_tensor(value: torch.Tensor, path: Path) -> dict[str, Any]:
    if path.exists():
        return validate_tensor_file(
            path,
            expected_shape=list(value.shape),
            expected_dtype=value.dtype,
        )
    atomic_torch_save(value.contiguous(), path)
    return {
        "path": str(path),
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": file_sha256(path),
        "serialized_bytes": path.stat().st_size,
        "raw_tensor_bytes": value.numel() * value.element_size(),
    }


def _initialize_cuda_memory_stats(device: torch.device) -> int:
    """Initialize the CUDA context before using PyTorch 2.3 memory stats."""

    cuda_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_device(cuda_index)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(cuda_index)
    return cuda_index


def command_preflight(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    authorization, plan, selected, plan_path = _load_plan(
        authorization_path=Path(args.authorization).resolve(),
        output_root=output_root,
    )
    report_path = output_root / "preflight/preflight_report.json"
    if report_path.exists():
        raise FileExistsError(f"feature preflight already exists: {report_path}")
    row = max(
        selected,
        key=lambda value: (
            int(value["prompt_token_count"]) + int(value["output_token_count"]),
            str(value["trajectory_id"]),
        ),
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("full-width preflight requires a visible CUDA GPU")
    cuda_index = _initialize_cuda_memory_stats(device)
    load_started = time.monotonic()
    model, resolved_revision = _load_model(authorization, device)
    model_load_seconds = time.monotonic() - load_started
    extraction_started = time.monotonic()
    trajectory, condition, layers, width = extract_row(
        model,
        row["prompt_token_ids"],
        row["output_token_ids"],
        device,
    )
    extraction_seconds = time.monotonic() - extraction_started
    contract = authorization["feature_contract"]
    if (layers, width) != (
        contract["num_feature_layers"],
        contract["per_layer_dim"],
    ):
        raise ValueError("preflight feature contract drift")
    if trajectory.dtype != torch.bfloat16 or condition.dtype != torch.bfloat16:
        raise ValueError("preflight did not preserve BF16")
    preflight_root = output_root / "preflight"
    trajectory_record = _record_new_tensor(trajectory, preflight_root / "trajectory.pt")
    condition_record = _record_new_tensor(condition, preflight_root / "condition.pt")
    trajectory_reload = validate_tensor_file(
        trajectory_record["path"],
        expected_shape=trajectory_record["shape"],
        expected_dtype=torch.bfloat16,
        expected_sha256=trajectory_record["sha256"],
    )
    condition_reload = validate_tensor_file(
        condition_record["path"],
        expected_shape=condition_record["shape"],
        expected_dtype=torch.bfloat16,
        expected_sha256=condition_record["sha256"],
    )
    report = {
        "schema_version": "clir-consistency-scale-feature-preflight-v6.1",
        "status": "PASS_FULL_WIDTH_EXACT_ID_FEATURE_PREFLIGHT_V6_1",
        "completed_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": file_sha256(Path(args.authorization)),
        "plan_file_sha256": file_sha256(plan_path),
        "trajectory_id": row["trajectory_id"],
        "query_id": row["query_id"],
        "prompt_token_count": row["prompt_token_count"],
        "output_token_count": row["output_token_count"],
        "selection": "largest_saved_prompt_plus_output_token_count",
        "model_id": contract["model_id"],
        "requested_revision": contract["model_revision"],
        "resolved_revision": resolved_revision,
        "dtype": contract["dtype"],
        "attention_implementation": contract["attention_implementation"],
        "trajectory": trajectory_reload,
        "condition": condition_reload,
        "model_load_seconds": model_load_seconds,
        "extraction_seconds": extraction_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index),
        "training_allowed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _relative_record(record: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    result = dict(record)
    result["relative_path"] = os.path.relpath(Path(record["path"]), output_root)
    result.pop("path", None)
    return result


def _validate_marker_payloads(
    marker: Mapping[str, Any],
    output_root: Path,
    query_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contract_width = 101376
    condition = marker["condition"]
    records = [condition, *marker["trajectories"]]
    expected_ids = {str(row["trajectory_id"]) for row in query_rows}
    observed_ids = {str(record["id"]) for record in marker["trajectories"]}
    if observed_ids != expected_ids:
        raise ValueError(f"{marker['query_id']}: marker trajectory population drift")
    owner = next(row for row in query_rows if row["condition_feature_owner"])
    if condition["id"] != marker["query_id"]:
        raise ValueError("condition marker query id drift")
    if condition["shape"] != [owner["prompt_token_count"], contract_width]:
        raise ValueError("condition marker shape drift")
    by_id = {str(row["trajectory_id"]): row for row in query_rows}
    for record in marker["trajectories"]:
        row = by_id[str(record["id"])]
        if record["shape"] != [row["output_token_count"], contract_width]:
            raise ValueError("trajectory marker shape drift")
    for record in records:
        path = output_root / record["relative_path"]
        if not path.is_file() or path.stat().st_size != record["serialized_bytes"]:
            raise ValueError(f"marker payload missing or size drift: {path}")
    return [dict(record) for record in records]


def command_extract_worker(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    authorization_path = Path(args.authorization).resolve()
    authorization, plan, selected, plan_path = _load_plan(
        authorization_path=authorization_path,
        output_root=output_root,
    )
    worker_index = int(args.worker_index)
    worker_count = int(plan["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index is outside the frozen worker population")
    worker_report_path = output_root / f"extraction/worker-{worker_index:03d}.json"
    if worker_report_path.exists():
        raise FileExistsError(f"worker report already exists: {worker_report_path}")
    queries = rows_for_worker(selected, worker_index)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("feature extraction worker requires a visible CUDA GPU")
    cuda_index = _initialize_cuda_memory_stats(device)
    started = time.monotonic()
    model, resolved_revision = _load_model(authorization, device)
    contract = authorization["feature_contract"]
    plan_sha256 = file_sha256(plan_path)
    authorization_sha256 = file_sha256(authorization_path)
    all_records: list[dict[str, Any]] = []
    extracted_queries = 0
    resumed_queries = 0
    extracted_trajectories = 0
    reused_trajectories = 0

    for query_number, (query_id, query_rows) in enumerate(queries.items(), start=1):
        marker_path = output_root / query_marker_relative_path(query_id)
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("schema_version") != QUERY_MARKER_SCHEMA:
                raise ValueError(f"query marker schema drift: {marker_path}")
            if marker.get("code_commit") != plan["code_commit"]:
                raise ValueError(f"query marker code commit drift: {marker_path}")
            if marker.get("worker_index") != worker_index:
                raise ValueError(f"query marker worker drift: {marker_path}")
            all_records.extend(
                _validate_marker_payloads(marker, output_root, query_rows)
            )
            resumed_queries += 1
            print(
                f"worker {worker_index}: verified marker {query_number}/{len(queries)} "
                f"{query_id}",
                flush=True,
            )
            continue

        owner = next(row for row in query_rows if row["condition_feature_owner"])
        condition_path = output_root / condition_relative_path(query_id)
        condition_record: dict[str, Any] | None = None
        trajectory_records: list[dict[str, Any]] = []
        for row in query_rows:
            trajectory_path = output_root / trajectory_relative_path(
                str(row["trajectory_id"])
            )
            need_condition = row is owner and not condition_path.exists()
            if trajectory_path.exists() and not need_condition:
                trajectory_record = validate_tensor_file(
                    trajectory_path,
                    expected_shape=[row["output_token_count"], contract["feature_dim"]],
                    expected_dtype=torch.bfloat16,
                )
                reused_trajectories += 1
                if row is owner:
                    condition_record = validate_tensor_file(
                        condition_path,
                        expected_shape=[
                            row["prompt_token_count"],
                            contract["feature_dim"],
                        ],
                        expected_dtype=torch.bfloat16,
                    )
            else:
                trajectory, condition, layers, width = extract_row(
                    model,
                    row["prompt_token_ids"],
                    row["output_token_ids"],
                    device,
                )
                if (layers, width) != (
                    contract["num_feature_layers"],
                    contract["per_layer_dim"],
                ):
                    raise ValueError("worker feature contract drift")
                if trajectory.dtype != torch.bfloat16:
                    raise ValueError("worker trajectory dtype drift")
                trajectory_record = _record_new_tensor(trajectory, trajectory_path)
                extracted_trajectories += 1
                if row is owner:
                    if condition.dtype != torch.bfloat16:
                        raise ValueError("worker condition dtype drift")
                    condition_record = _record_new_tensor(condition, condition_path)
                del trajectory, condition
            trajectory_record = _relative_record(trajectory_record, output_root)
            trajectory_record["kind"] = "trajectory"
            trajectory_record["id"] = row["trajectory_id"]
            trajectory_records.append(trajectory_record)
        if condition_record is None:
            condition_record = validate_tensor_file(
                condition_path,
                expected_shape=[owner["prompt_token_count"], contract["feature_dim"]],
                expected_dtype=torch.bfloat16,
            )
        condition_record = _relative_record(condition_record, output_root)
        condition_record["kind"] = "condition"
        condition_record["id"] = query_id
        marker = {
            "schema_version": QUERY_MARKER_SCHEMA,
            "status": "COMPLETE_QUERY_FEATURES_V6_1",
            "completed_at_utc": _utc_now(),
            "query_id": query_id,
            "worker_index": worker_index,
            "code_commit": plan["code_commit"],
            "authorization_file_sha256": authorization_sha256,
            "plan_file_sha256": plan_sha256,
            "model_id": contract["model_id"],
            "requested_revision": contract["model_revision"],
            "resolved_revision": resolved_revision,
            "dtype": contract["dtype"],
            "attention_implementation": contract["attention_implementation"],
            "condition": condition_record,
            "trajectories": sorted(
                trajectory_records, key=lambda record: str(record["id"])
            ),
        }
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker_path, marker)
        all_records.extend([condition_record, *marker["trajectories"]])
        extracted_queries += 1
        print(
            f"worker {worker_index}: completed {query_number}/{len(queries)} {query_id}",
            flush=True,
        )
        gc.collect()

    all_records.sort(key=lambda record: (record["kind"], str(record["id"])))
    stats = selected_statistics(
        [row for row in selected if row["worker_index"] == worker_index]
    )
    report = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "status": "PASS_FEATURE_EXTRACTION_WORKER_V6_1",
        "completed_at_utc": _utc_now(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": authorization_sha256,
        "plan_file_sha256": plan_sha256,
        "model_id": contract["model_id"],
        "requested_revision": contract["model_revision"],
        "resolved_revision": resolved_revision,
        "dtype": contract["dtype"],
        "attention_implementation": contract["attention_implementation"],
        "inventory_statistics": stats,
        "payload_count": len(all_records),
        "payload_record_digest": payload_record_digest(all_records),
        "serialized_bytes": sum(record["serialized_bytes"] for record in all_records),
        "raw_tensor_bytes": sum(record["raw_tensor_bytes"] for record in all_records),
        "new_query_count": extracted_queries,
        "resumed_query_count": resumed_queries,
        "new_trajectory_count": extracted_trajectories,
        "reused_trajectory_count": reused_trajectories,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index),
        "training_allowed": False,
    }
    worker_report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(worker_report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_worker(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    authorization_path = Path(args.authorization).resolve()
    authorization, plan, selected, plan_path = _load_plan(
        authorization_path=authorization_path,
        output_root=output_root,
    )
    worker_index = int(args.worker_index)
    worker_count = int(plan["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index is outside the frozen worker population")
    extraction_report_path = output_root / f"extraction/worker-{worker_index:03d}.json"
    extraction_report = json.loads(extraction_report_path.read_text(encoding="utf-8"))
    if extraction_report.get("status") != "PASS_FEATURE_EXTRACTION_WORKER_V6_1":
        raise ValueError("extraction worker did not pass")
    verifier_path = output_root / f"verification/worker-{worker_index:03d}.json"
    if verifier_path.exists():
        raise FileExistsError(f"worker verification already exists: {verifier_path}")

    contract = authorization["feature_contract"]
    query_groups = rows_for_worker(selected, worker_index)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for query_number, (query_id, query_rows) in enumerate(
        query_groups.items(), start=1
    ):
        marker_path = output_root / query_marker_relative_path(query_id)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != QUERY_MARKER_SCHEMA:
            raise ValueError(f"query marker schema drift: {marker_path}")
        marker_records = _validate_marker_payloads(marker, output_root, query_rows)
        for marker_record in marker_records:
            observed = validate_tensor_file(
                output_root / marker_record["relative_path"],
                expected_shape=marker_record["shape"],
                expected_dtype=torch.bfloat16,
                expected_sha256=marker_record["sha256"],
            )
            observed = _relative_record(observed, output_root)
            observed["kind"] = marker_record["kind"]
            observed["id"] = marker_record["id"]
            records.append(observed)
        print(
            f"verifier {worker_index}: checked {query_number}/{len(query_groups)} "
            f"{query_id}",
            flush=True,
        )
    records.sort(key=lambda record: (record["kind"], str(record["id"])))
    digest = payload_record_digest(records)
    if digest != extraction_report["payload_record_digest"]:
        raise ValueError("independent payload record digest differs from extraction")
    stats = selected_statistics(
        [row for row in selected if row["worker_index"] == worker_index]
    )
    report = {
        "schema_version": VERIFIER_REPORT_SCHEMA,
        "status": "PASS_INDEPENDENT_FEATURE_WORKER_VERIFICATION_V6_1",
        "verified_at_utc": _utc_now(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": file_sha256(authorization_path),
        "plan_file_sha256": file_sha256(plan_path),
        "extraction_report_file_sha256": file_sha256(extraction_report_path),
        "model_id": contract["model_id"],
        "model_revision": contract["model_revision"],
        "dtype": contract["dtype"],
        "feature_dim": contract["feature_dim"],
        "inventory_statistics": stats,
        "payload_count": len(records),
        "payload_record_digest": digest,
        "serialized_bytes": sum(record["serialized_bytes"] for record in records),
        "raw_tensor_bytes": sum(record["raw_tensor_bytes"] for record in records),
        "all_shapes_dtypes_finiteness_and_checksums_verified": True,
        "elapsed_seconds": time.monotonic() - started,
        "training_allowed": False,
    }
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(verifier_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _marker_maps(
    output_root: Path, selected: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    conditions: dict[str, dict[str, Any]] = {}
    query_groups: dict[str, list[dict[str, Any]]] = {}
    for worker in sorted({int(row["worker_index"]) for row in selected}):
        query_groups.update(rows_for_worker(selected, worker))
    for query_id, query_rows in query_groups.items():
        marker_path = output_root / query_marker_relative_path(query_id)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        _validate_marker_payloads(marker, output_root, query_rows)
        conditions[query_id] = dict(marker["condition"])
        for record in marker["trajectories"]:
            trajectory_id = str(record["id"])
            if trajectory_id in trajectories:
                raise ValueError(
                    f"duplicate trajectory payload marker: {trajectory_id}"
                )
            trajectories[trajectory_id] = dict(record)
    return trajectories, conditions


def command_finalize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    authorization_path = Path(args.authorization).resolve()
    authorization, plan, selected, plan_path = _load_plan(
        authorization_path=authorization_path,
        output_root=output_root,
    )
    preflight_path = output_root / "preflight/preflight_report.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_FULL_WIDTH_EXACT_ID_FEATURE_PREFLIGHT_V6_1":
        raise ValueError("full-width feature preflight did not pass")
    final_root = output_root / "final"
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError(f"final feature directory is not empty: {final_root}")

    worker_count = int(plan["worker_count"])
    extraction_reports: list[dict[str, Any]] = []
    verifier_reports: list[dict[str, Any]] = []
    report_hashes: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        extraction_path = output_root / f"extraction/worker-{worker_index:03d}.json"
        verifier_path = output_root / f"verification/worker-{worker_index:03d}.json"
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
        if extraction.get("status") != "PASS_FEATURE_EXTRACTION_WORKER_V6_1":
            raise ValueError(f"extraction worker {worker_index} did not pass")
        if (
            verifier.get("status")
            != "PASS_INDEPENDENT_FEATURE_WORKER_VERIFICATION_V6_1"
        ):
            raise ValueError(f"verification worker {worker_index} did not pass")
        if extraction["payload_record_digest"] != verifier["payload_record_digest"]:
            raise ValueError(f"worker {worker_index} payload digest mismatch")
        if extraction["raw_tensor_bytes"] != verifier["raw_tensor_bytes"]:
            raise ValueError(f"worker {worker_index} raw byte mismatch")
        extraction_reports.append(extraction)
        verifier_reports.append(verifier)
        report_hashes.append(
            {
                "worker_index": worker_index,
                "extraction_report_file_sha256": file_sha256(extraction_path),
                "verification_report_file_sha256": file_sha256(verifier_path),
            }
        )

    trajectories, conditions = _marker_maps(output_root, selected)
    expected = authorization["expected_inventory"]
    if len(trajectories) != expected["trajectory_count"]:
        raise ValueError("final trajectory payload count drift")
    if len(conditions) != expected["condition_count"]:
        raise ValueError("final condition payload count drift")
    raw_bytes = sum(report["raw_tensor_bytes"] for report in verifier_reports)
    if raw_bytes != expected["raw_feature_bytes"]:
        raise ValueError("final raw feature byte count drift")

    final_root.mkdir(parents=True, exist_ok=True)
    manifest_path = final_root / "extracted_selected_features.jsonl"
    extracted_rows: list[dict[str, Any]] = []
    contract = authorization["feature_contract"]
    for source in selected:
        row = dict(source)
        trajectory = trajectories[str(row["trajectory_id"])]
        condition = conditions[str(row["query_id"])]
        row["schema_version"] = EXTRACTED_ROW_SCHEMA
        row["hidden_states_path"] = os.path.relpath(
            output_root / trajectory["relative_path"], manifest_path.parent
        )
        row["condition_states_path"] = os.path.relpath(
            output_root / condition["relative_path"], manifest_path.parent
        )
        row["hidden_states_sha256"] = trajectory["sha256"]
        row["condition_states_sha256"] = condition["sha256"]
        row["hidden_states_serialized_bytes"] = trajectory["serialized_bytes"]
        row["condition_states_serialized_bytes"] = condition["serialized_bytes"]
        row["feature_dim"] = contract["feature_dim"]
        row["num_feature_layers"] = contract["num_feature_layers"]
        row["per_layer_dim"] = contract["per_layer_dim"]
        row["feature_model"] = contract["model_id"]
        row["feature_revision"] = contract["model_revision"]
        row["feature_dtype"] = contract["dtype"]
        row["feature_attention_implementation"] = contract["attention_implementation"]
        row["feature_extraction_code_commit"] = plan["code_commit"]
        row["feature_extraction_authorization_sha256"] = file_sha256(authorization_path)
        extracted_rows.append(row)
    extracted_manifest = publish_manifest(
        manifest_path,
        extracted_rows,
        schema_version=("clir-consistency-scale-extracted-feature-manifest-v6.1"),
        metadata={
            "authorization_file_sha256": file_sha256(authorization_path),
            "plan_file_sha256": file_sha256(plan_path),
            "independent_verification": True,
        },
    )
    serialized_bytes = sum(report["serialized_bytes"] for report in verifier_reports)
    final_report = {
        "schema_version": "clir-consistency-scale-feature-final-report-v6.1",
        "status": "PASS_SELECTED_FEATURE_EXTRACTION_AND_VERIFICATION_V6_1",
        "completed_at_utc": _utc_now(),
        "evidence_tier": "pipeline_pilot_exact_feature_publication",
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": file_sha256(authorization_path),
        "plan_file_sha256": file_sha256(plan_path),
        "preflight_report_file_sha256": file_sha256(preflight_path),
        "worker_reports": report_hashes,
        "inventory_statistics": plan["inventory_statistics"],
        "trajectory_payload_count": len(trajectories),
        "condition_payload_count": len(conditions),
        "payload_count": len(trajectories) + len(conditions),
        "raw_tensor_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
        "feature_contract": contract,
        "all_payload_shapes_dtypes_finiteness_and_checksums_independently_verified": True,
        "extracted_manifest": extracted_manifest,
        "all_16000_rollouts_extracted": False,
        "new_rollout_or_ai_annotation_used": False,
        "relation_or_threshold_changed": False,
        "feature_extraction_completed": True,
        "training_allowed": False,
        "next_gate": "SEPARATE_C_ONLY_TRAINING_AUTHORIZATION",
    }
    final_report_path = final_root / "feature_extraction_report.json"
    atomic_write_json(final_report_path, final_report)
    print(
        json.dumps(
            {
                **final_report,
                "final_report_path": str(final_report_path),
                "final_report_file_sha256": file_sha256(final_report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="join the pinned v6.1 inventory to exact saved token IDs"
    )
    _common_arguments(prepare)
    prepare.set_defaults(func=command_prepare)
    verify_plan = subparsers.add_parser(
        "verify-plan", help="independently verify the selected-only extraction plan"
    )
    _common_arguments(verify_plan)
    verify_plan.set_defaults(func=command_verify_plan)
    preflight = subparsers.add_parser(
        "preflight", help="run and reload one largest full-width exact-ID sample"
    )
    _common_arguments(preflight)
    preflight.add_argument("--device", default="cuda:0")
    preflight.set_defaults(func=command_preflight)
    extract_worker = subparsers.add_parser(
        "extract-worker", help="extract one deterministic query-balanced GPU worker"
    )
    _common_arguments(extract_worker)
    extract_worker.add_argument("--worker-index", type=int, required=True)
    extract_worker.add_argument("--device", default="cuda:0")
    extract_worker.set_defaults(func=command_extract_worker)
    verify_worker = subparsers.add_parser(
        "verify-worker", help="reload and independently verify every worker payload"
    )
    _common_arguments(verify_worker)
    verify_worker.add_argument("--worker-index", type=int, required=True)
    verify_worker.set_defaults(func=command_verify_worker)
    finalize = subparsers.add_parser(
        "finalize", help="publish the extracted manifest only after all verifiers pass"
    )
    _common_arguments(finalize)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
