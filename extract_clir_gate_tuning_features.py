#!/usr/bin/env python
"""Extract selected-only exact-ID features for Prior/Gate tuning v1.

The GPU inventory contains token IDs and provenance but no correctness labels
or CLIR scores.  Tuning labels are rejoined only in the final tuning manifest;
confirmation labels are rejoined only into an explicitly sealed manifest.
This stage cannot train or score any checkpoint.
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
from src.clir_gate_tuning import build_selected_feature_inventory
from src.clir_scale_features import (
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
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/feature_extraction_authorization.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/features_v1"
AUTHORIZATION_SCHEMA = "clir-prior-gate-tuning-v1-feature-authorization"
AUTHORIZATION_STATUS = "AUTHORIZED_SELECTED_ONLY_FEATURE_EXTRACTION"
PLAN_SCHEMA = "clir-prior-gate-tuning-v1-feature-plan"
INVENTORY_SCHEMA = "clir-gate-tuning-v1-feature-inventory-row"
MARKER_SCHEMA = "clir-prior-gate-tuning-v1-feature-query-marker"
WORKER_STATUS = "PASS_GATE_TUNING_V1_FEATURE_WORKER"
VERIFIER_STATUS = "PASS_GATE_TUNING_V1_FEATURE_WORKER_VERIFICATION"


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


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _require_clean_commit() -> str:
    if _git_dirty():
        raise RuntimeError("Prior/Gate feature commands require a clean Git commit")
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
            f"authorized code parent {ancestor} is not an ancestor of HEAD"
        )


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"pinned {label} hash drift: {observed} != {expected}")


def _read_published(
    path: Path, specification: Mapping[str, Any]
) -> list[dict[str, Any]]:
    _assert_hash(path, str(specification["file_sha256"]), path.name)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    _assert_hash(sidecar, str(specification["sidecar_file_sha256"]), sidecar.name)
    rows = read_jsonl(path)
    if len(rows) != int(specification["row_count"]):
        raise ValueError(f"published row count drift: {path}")
    if canonical_sha256(rows) != specification["ordered_rows_sha256"]:
        raise ValueError(f"published ordered row hash drift: {path}")
    return rows


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization_path = Path(path).resolve()
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported Prior/Gate feature authorization")
    if authorization.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("Prior/Gate selected-only features are not authorized")
    expected_scope = {
        "selected_inventory_materialization": True,
        "full_width_preflight": True,
        "selected_only_feature_extraction": True,
        "independent_payload_verification": True,
        "publish_split_feature_manifests": True,
        "unselected_rollout_feature_extraction": False,
        "new_rollout_or_checker_change": False,
        "training": False,
        "tuning_scoring": False,
        "confirmation_scoring": False,
        "confirmation_outcome_disclosure": False,
    }
    if authorization.get("authorized_scope") != expected_scope:
        raise ValueError("Prior/Gate feature authorization scope drift")
    for label, specification in authorization["frozen_parent"]["files"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, str(specification["file_sha256"]), label)
        if "sidecar_file_sha256" in specification:
            _assert_hash(
                source.with_suffix(source.suffix + ".manifest.json"),
                str(specification["sidecar_file_sha256"]),
                f"{label} sidecar",
            )
    contract = authorization["feature_contract"]
    if (
        contract.get("input_axis_truth")
        != "saved_prompt_token_ids_plus_saved_output_token_ids"
        or contract.get("decode_or_retokenize_allowed") is not False
        or contract.get("dtype") != "bfloat16"
        or int(contract.get("feature_dim", -1)) != 101376
    ):
        raise ValueError("Prior/Gate full-width feature contract drift")
    return authorization


def _authorized_root(authorization: Mapping[str, Any], requested: str | Path) -> Path:
    expected = _project_path(authorization["runtime_contract"]["output_root"]).resolve()
    observed = Path(requested).resolve()
    if observed != expected:
        raise ValueError(f"feature output root drift: {observed} != {expected}")
    return observed


def _selected_sources(
    authorization: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = authorization["frozen_parent"]["files"]
    tuning = _read_published(
        _project_path(files["tuning_selected"]["path"]), files["tuning_selected"]
    )
    confirmation = _read_published(
        _project_path(files["confirmation_selected"]["path"]),
        files["confirmation_selected"],
    )
    if any(row.get("sealed_until_weight_lock") is not True for row in confirmation):
        raise ValueError("confirmation selected rows lost their sealed marker")
    return tuning, confirmation


def _build_inventory(
    authorization: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tuning, confirmation = _selected_sources(authorization)
    runtime = authorization["runtime_contract"]
    inventory, report = build_selected_feature_inventory(
        tuning,
        confirmation,
        candidate_count=int(runtime["candidate_count"]),
        worker_count=int(runtime["worker_count"]),
    )
    expected = authorization["expected_inventory"]
    for field in (
        "trajectory_count",
        "query_count",
        "condition_count",
        "output_token_count",
        "prompt_token_count",
        "total_feature_token_count",
    ):
        if int(report["total"][field]) != int(expected[field]):
            raise ValueError(f"selected feature inventory {field} drift")
    raw_bytes = int(report["total"]["total_feature_token_count"]) * int(
        authorization["feature_contract"]["bytes_per_feature_token"]
    )
    if raw_bytes != int(expected["raw_feature_bytes"]):
        raise ValueError("selected feature inventory raw byte total drift")
    return inventory, report


def command_prepare(args: argparse.Namespace) -> None:
    code_commit = _require_clean_commit()
    authorization_path = Path(args.authorization).resolve()
    authorization = load_authorization(authorization_path)
    _require_ancestor(
        str(authorization["frozen_parent"]["authorized_code_parent_commit"]),
        code_commit,
    )
    output_root = _authorized_root(authorization, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"feature output root is not empty: {output_root}")
    inventory, inventory_report = _build_inventory(authorization)
    plan_root = output_root / "plan"
    inventory_path = plan_root / "selected_only_inventory.jsonl"
    inventory_manifest = publish_manifest(
        inventory_path,
        inventory,
        schema_version="clir-prior-gate-tuning-v1-feature-inventory",
        metadata={
            "contains_correctness": False,
            "contains_clir_scores": False,
            "confirmation_rows_remain_sealed": True,
        },
    )
    inventory_manifest["sidecar_file_sha256"] = file_sha256(
        inventory_path.with_suffix(inventory_path.suffix + ".manifest.json")
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "PASS_GATE_TUNING_V1_SELECTED_FEATURE_PLAN",
        "planned_at_utc": _utc_now(),
        "code_commit": code_commit,
        "authorization_file_sha256": file_sha256(authorization_path),
        "output_root": str(output_root),
        "inventory": inventory_manifest,
        "inventory_report": inventory_report,
        "raw_feature_bytes": authorization["expected_inventory"]["raw_feature_bytes"],
        "worker_assignment": (
            "deterministic_largest_first_feature_token_balanced_by_query_v1"
        ),
        "feature_contract": dict(authorization["feature_contract"]),
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "scoring_allowed": False,
    }
    plan_path = plan_root / "extraction_plan.json"
    atomic_write_json(plan_path, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_file_sha256": file_sha256(plan_path),
                "inventory_statistics": inventory_report["total"],
                "worker_statistics": inventory_report["worker_statistics"],
                "raw_feature_bytes": plan["raw_feature_bytes"],
                "confirmation_outcomes_opened": False,
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_plan(
    *, authorization_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    authorization = load_authorization(authorization_path)
    _authorized_root(authorization, output_root)
    plan_path = output_root / "plan/extraction_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "PASS_GATE_TUNING_V1_SELECTED_FEATURE_PLAN"
    ):
        raise ValueError("Prior/Gate feature plan is not a PASS")
    if plan.get("authorization_file_sha256") != file_sha256(authorization_path):
        raise ValueError("feature authorization drift after planning")
    if plan.get("code_commit") != _require_clean_commit():
        raise ValueError("feature plan code commit drift")
    specification = plan["inventory"]
    inventory_path = Path(specification["path"])
    inventory = _read_published(inventory_path, specification)
    if any(row.get("schema_version") != INVENTORY_SCHEMA for row in inventory):
        raise ValueError("selected-only feature inventory row schema drift")
    if selected_statistics(inventory) != plan["inventory_report"]["total"]:
        raise ValueError("selected-only feature inventory statistics drift")
    return authorization, plan, inventory, plan_path


def command_verify_plan(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    authorization, plan, inventory, plan_path = _load_plan(
        authorization_path=authorization_path, output_root=output_root
    )
    recomputed, recomputed_report = _build_inventory(authorization)
    if canonical_sha256(recomputed) != canonical_sha256(inventory):
        raise ValueError("selected-only feature inventory recomputation drift")
    if recomputed_report != plan["inventory_report"]:
        raise ValueError("selected-only feature plan report recomputation drift")
    records = expected_payload_records(output_root, inventory)
    expected = authorization["expected_inventory"]
    if len(records) != int(expected["trajectory_count"]) + int(
        expected["condition_count"]
    ):
        raise ValueError("planned feature payload count drift")
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-feature-plan-verification",
        "status": "PASS_GATE_TUNING_V1_SELECTED_FEATURE_PLAN_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "plan_file_sha256": file_sha256(plan_path),
        "inventory_file_sha256": plan["inventory"]["file_sha256"],
        "planned_payload_count": len(records),
        "planned_raw_feature_bytes": plan["raw_feature_bytes"],
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "scoring_allowed": False,
    }
    path = output_root / "plan/independent_verification.json"
    if path.exists():
        raise FileExistsError(f"feature plan verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_model(
    authorization: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, str | None]:
    from transformers import AutoModelForCausalLM

    contract = authorization["feature_contract"]
    runtime = authorization["runtime_contract"]
    model = AutoModelForCausalLM.from_pretrained(
        contract["model_id"],
        revision=contract["model_revision"],
        cache_dir=str(_project_path(runtime["cache_dir"])),
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation=contract["attention_implementation"],
    ).to(device)
    model.eval()
    if int(model.config.hidden_size) != int(contract["per_layer_dim"]):
        raise ValueError("loaded feature model hidden width drift")
    if int(model.config.num_hidden_layers) + 1 != int(contract["num_feature_layers"]):
        raise ValueError("loaded feature model layer count drift")
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved not in (None, contract["model_revision"]):
        raise ValueError("loaded feature model revision drift")
    return model, resolved


def _record_tensor(value: torch.Tensor, path: Path) -> dict[str, Any]:
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


def _initialize_cuda(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(index)
    return index


def command_preflight(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    authorization, plan, inventory, plan_path = _load_plan(
        authorization_path=authorization_path, output_root=output_root
    )
    report_path = output_root / "preflight/report.json"
    if report_path.exists():
        raise FileExistsError(f"feature preflight already exists: {report_path}")
    row = max(
        inventory,
        key=lambda value: (
            int(value["prompt_token_count"]) + int(value["output_token_count"]),
            str(value["trajectory_id"]),
        ),
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("full-width feature preflight requires one visible GPU")
    cuda_index = _initialize_cuda(device)
    started = time.monotonic()
    model, resolved = _load_model(authorization, device)
    loaded = time.monotonic()
    trajectory, condition, layers, width = extract_row(
        model, row["prompt_token_ids"], row["output_token_ids"], device
    )
    extracted = time.monotonic()
    contract = authorization["feature_contract"]
    if (layers, width) != (
        int(contract["num_feature_layers"]),
        int(contract["per_layer_dim"]),
    ):
        raise ValueError("preflight feature contract drift")
    if trajectory.dtype != torch.bfloat16 or condition.dtype != torch.bfloat16:
        raise ValueError("preflight feature dtype drift")
    root = output_root / "preflight"
    trajectory_record = _record_tensor(trajectory, root / "trajectory.pt")
    condition_record = _record_tensor(condition, root / "condition.pt")
    trajectory_record = validate_tensor_file(
        trajectory_record["path"],
        expected_shape=trajectory_record["shape"],
        expected_dtype=torch.bfloat16,
        expected_sha256=trajectory_record["sha256"],
    )
    condition_record = validate_tensor_file(
        condition_record["path"],
        expected_shape=condition_record["shape"],
        expected_dtype=torch.bfloat16,
        expected_sha256=condition_record["sha256"],
    )
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-feature-preflight",
        "status": "PASS_GATE_TUNING_V1_FULL_WIDTH_FEATURE_PREFLIGHT",
        "completed_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": file_sha256(authorization_path),
        "plan_file_sha256": file_sha256(plan_path),
        "trajectory_id": row["trajectory_id"],
        "query_id": row["query_id"],
        "role": row["role"],
        "prompt_token_count": row["prompt_token_count"],
        "output_token_count": row["output_token_count"],
        "model_id": contract["model_id"],
        "requested_revision": contract["model_revision"],
        "resolved_revision": resolved,
        "dtype": contract["dtype"],
        "attention_implementation": contract["attention_implementation"],
        "trajectory": trajectory_record,
        "condition": condition_record,
        "model_load_seconds": loaded - started,
        "extraction_seconds": extracted - loaded,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index),
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "scoring_allowed": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _relative(record: Mapping[str, Any], root: Path) -> dict[str, Any]:
    output = dict(record)
    output["relative_path"] = os.path.relpath(Path(output.pop("path")), root)
    return output


def _validate_marker(
    marker: Mapping[str, Any],
    output_root: Path,
    query_rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    authorization_sha256: str,
) -> list[dict[str, Any]]:
    if marker.get("schema_version") != MARKER_SCHEMA:
        raise ValueError("feature query marker schema drift")
    query_id = str(query_rows[0]["query_id"])
    if marker.get("query_id") != query_id:
        raise ValueError(f"{query_id}: feature query marker id drift")
    if marker.get("code_commit") != plan["code_commit"]:
        raise ValueError(f"{query_id}: feature marker code commit drift")
    if marker.get("authorization_file_sha256") != authorization_sha256:
        raise ValueError(f"{query_id}: feature marker authorization drift")
    condition = marker["condition"]
    trajectories = marker["trajectories"]
    expected_ids = {str(row["trajectory_id"]) for row in query_rows}
    if {str(record["id"]) for record in trajectories} != expected_ids:
        raise ValueError(f"{query_id}: feature marker trajectory population drift")
    owner = next(row for row in query_rows if row["condition_feature_owner"])
    width = int(plan["feature_contract"]["feature_dim"])
    if condition.get("id") != query_id or condition.get("shape") != [
        int(owner["prompt_token_count"]),
        width,
    ]:
        raise ValueError(f"{query_id}: condition feature marker shape drift")
    by_id = {str(row["trajectory_id"]): row for row in query_rows}
    for record in trajectories:
        row = by_id[str(record["id"])]
        if record.get("shape") != [int(row["output_token_count"]), width]:
            raise ValueError(f"{record['id']}: trajectory feature marker shape drift")
    records = [condition, *trajectories]
    for record in records:
        path = output_root / record["relative_path"]
        if not path.is_file() or path.stat().st_size != int(record["serialized_bytes"]):
            raise ValueError(f"feature marker payload missing or size drift: {path}")
    return [dict(record) for record in records]


def command_extract_worker(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    authorization, plan, inventory, plan_path = _load_plan(
        authorization_path=authorization_path, output_root=output_root
    )
    preflight = json.loads((output_root / "preflight/report.json").read_text())
    if preflight.get("status") != "PASS_GATE_TUNING_V1_FULL_WIDTH_FEATURE_PREFLIGHT":
        raise ValueError("full-width feature preflight did not pass")
    worker_index = int(args.worker_index)
    worker_count = int(authorization["runtime_contract"]["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise ValueError("feature worker index is outside the frozen range")
    report_path = output_root / f"extraction/worker-{worker_index:03d}.json"
    if report_path.exists():
        raise FileExistsError(f"feature worker report already exists: {report_path}")
    queries = rows_for_worker(inventory, worker_index)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("feature worker requires one visible CUDA GPU")
    cuda_index = _initialize_cuda(device)
    started = time.monotonic()
    model, resolved = _load_model(authorization, device)
    contract = authorization["feature_contract"]
    authorization_sha = file_sha256(authorization_path)
    plan_sha = file_sha256(plan_path)
    records: list[dict[str, Any]] = []
    new_queries = 0
    resumed_queries = 0
    new_trajectories = 0
    reused_trajectories = 0
    for query_number, (query_id, query_rows) in enumerate(queries.items(), start=1):
        marker_path = output_root / query_marker_relative_path(query_id)
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            records.extend(
                _validate_marker(
                    marker,
                    output_root,
                    query_rows,
                    plan=plan,
                    authorization_sha256=authorization_sha,
                )
            )
            resumed_queries += 1
            print(
                f"worker {worker_index}: resumed {query_number}/{len(queries)} {query_id}",
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
                    expected_shape=[
                        int(row["output_token_count"]),
                        int(contract["feature_dim"]),
                    ],
                    expected_dtype=torch.bfloat16,
                )
                reused_trajectories += 1
                if row is owner:
                    condition_record = validate_tensor_file(
                        condition_path,
                        expected_shape=[
                            int(row["prompt_token_count"]),
                            int(contract["feature_dim"]),
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
                    int(contract["num_feature_layers"]),
                    int(contract["per_layer_dim"]),
                ):
                    raise ValueError("feature worker layer contract drift")
                if trajectory.dtype != torch.bfloat16:
                    raise ValueError("feature worker trajectory dtype drift")
                trajectory_record = _record_tensor(trajectory, trajectory_path)
                new_trajectories += 1
                if row is owner:
                    if condition.dtype != torch.bfloat16:
                        raise ValueError("feature worker condition dtype drift")
                    condition_record = _record_tensor(condition, condition_path)
                del trajectory, condition
            trajectory_record = _relative(trajectory_record, output_root)
            trajectory_record["kind"] = "trajectory"
            trajectory_record["id"] = row["trajectory_id"]
            trajectory_records.append(trajectory_record)
        if condition_record is None:
            condition_record = validate_tensor_file(
                condition_path,
                expected_shape=[
                    int(owner["prompt_token_count"]),
                    int(contract["feature_dim"]),
                ],
                expected_dtype=torch.bfloat16,
            )
        condition_record = _relative(condition_record, output_root)
        condition_record["kind"] = "condition"
        condition_record["id"] = query_id
        marker = {
            "schema_version": MARKER_SCHEMA,
            "status": "COMPLETE_GATE_TUNING_V1_QUERY_FEATURES",
            "completed_at_utc": _utc_now(),
            "query_id": query_id,
            "role": owner["role"],
            "sealed_until_weight_lock": owner["sealed_until_weight_lock"],
            "worker_index": worker_index,
            "code_commit": plan["code_commit"],
            "authorization_file_sha256": authorization_sha,
            "plan_file_sha256": plan_sha,
            "model_id": contract["model_id"],
            "requested_revision": contract["model_revision"],
            "resolved_revision": resolved,
            "dtype": contract["dtype"],
            "attention_implementation": contract["attention_implementation"],
            "condition": condition_record,
            "trajectories": sorted(
                trajectory_records, key=lambda record: str(record["id"])
            ),
        }
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker_path, marker)
        records.extend([condition_record, *marker["trajectories"]])
        new_queries += 1
        print(
            f"worker {worker_index}: completed {query_number}/{len(queries)} {query_id}",
            flush=True,
        )
        gc.collect()
    records.sort(key=lambda record: (record["kind"], str(record["id"])))
    worker_rows = [row for row in inventory if int(row["worker_index"]) == worker_index]
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-feature-worker-report",
        "status": WORKER_STATUS,
        "completed_at_utc": _utc_now(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": authorization_sha,
        "plan_file_sha256": plan_sha,
        "model_id": contract["model_id"],
        "requested_revision": contract["model_revision"],
        "resolved_revision": resolved,
        "dtype": contract["dtype"],
        "attention_implementation": contract["attention_implementation"],
        "inventory_statistics": selected_statistics(worker_rows),
        "payload_count": len(records),
        "payload_record_digest": payload_record_digest(records),
        "serialized_bytes": sum(int(record["serialized_bytes"]) for record in records),
        "raw_tensor_bytes": sum(int(record["raw_tensor_bytes"]) for record in records),
        "new_query_count": new_queries,
        "resumed_query_count": resumed_queries,
        "new_trajectory_count": new_trajectories,
        "reused_trajectory_count": reused_trajectories,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_index),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_index),
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "scoring_allowed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_worker(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    authorization, plan, inventory, plan_path = _load_plan(
        authorization_path=authorization_path, output_root=output_root
    )
    worker_index = int(args.worker_index)
    worker_count = int(authorization["runtime_contract"]["worker_count"])
    if not 0 <= worker_index < worker_count:
        raise ValueError("feature verifier worker index is outside the frozen range")
    extraction_path = output_root / f"extraction/worker-{worker_index:03d}.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction.get("status") != WORKER_STATUS:
        raise ValueError("feature extraction worker did not pass")
    verifier_path = output_root / f"verification/worker-{worker_index:03d}.json"
    if verifier_path.exists():
        raise FileExistsError(f"feature verification already exists: {verifier_path}")
    queries = rows_for_worker(inventory, worker_index)
    authorization_sha = file_sha256(authorization_path)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for query_number, (query_id, query_rows) in enumerate(queries.items(), start=1):
        marker = json.loads(
            (output_root / query_marker_relative_path(query_id)).read_text(
                encoding="utf-8"
            )
        )
        marker_records = _validate_marker(
            marker,
            output_root,
            query_rows,
            plan=plan,
            authorization_sha256=authorization_sha,
        )
        for record in marker_records:
            observed = validate_tensor_file(
                output_root / record["relative_path"],
                expected_shape=record["shape"],
                expected_dtype=torch.bfloat16,
                expected_sha256=str(record["sha256"]),
            )
            observed = _relative(observed, output_root)
            observed["kind"] = record["kind"]
            observed["id"] = record["id"]
            records.append(observed)
        print(
            f"verifier {worker_index}: checked {query_number}/{len(queries)} {query_id}",
            flush=True,
        )
    records.sort(key=lambda record: (record["kind"], str(record["id"])))
    digest = payload_record_digest(records)
    if digest != extraction["payload_record_digest"]:
        raise ValueError("independent feature payload digest differs from extraction")
    worker_rows = [row for row in inventory if int(row["worker_index"]) == worker_index]
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-feature-worker-verification",
        "status": VERIFIER_STATUS,
        "verified_at_utc": _utc_now(),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": authorization_sha,
        "plan_file_sha256": file_sha256(plan_path),
        "extraction_report_file_sha256": file_sha256(extraction_path),
        "inventory_statistics": selected_statistics(worker_rows),
        "payload_count": len(records),
        "payload_record_digest": digest,
        "serialized_bytes": sum(int(record["serialized_bytes"]) for record in records),
        "raw_tensor_bytes": sum(int(record["raw_tensor_bytes"]) for record in records),
        "all_shapes_dtypes_finiteness_and_checksums_verified": True,
        "elapsed_seconds": time.monotonic() - started,
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "scoring_allowed": False,
    }
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(verifier_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _payload_maps(
    output_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    authorization_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    trajectories: dict[str, dict[str, Any]] = {}
    conditions: dict[str, dict[str, Any]] = {}
    worker_indices = sorted({int(row["worker_index"]) for row in inventory})
    for worker_index in worker_indices:
        for query_id, query_rows in rows_for_worker(inventory, worker_index).items():
            marker = json.loads(
                (output_root / query_marker_relative_path(query_id)).read_text(
                    encoding="utf-8"
                )
            )
            _validate_marker(
                marker,
                output_root,
                query_rows,
                plan=plan,
                authorization_sha256=authorization_sha256,
            )
            conditions[query_id] = dict(marker["condition"])
            for record in marker["trajectories"]:
                trajectory_id = str(record["id"])
                if trajectory_id in trajectories:
                    raise ValueError(f"duplicate trajectory feature: {trajectory_id}")
                trajectories[trajectory_id] = dict(record)
    return trajectories, conditions


def _attach_features(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    inventory_by_id: Mapping[str, Mapping[str, Any]],
    trajectories: Mapping[str, Mapping[str, Any]],
    conditions: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    manifest_path: Path,
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization_sha256: str,
) -> list[dict[str, Any]]:
    contract = authorization["feature_contract"]
    output: list[dict[str, Any]] = []
    for source in source_rows:
        trajectory_id = str(source["id"])
        query_id = str(source["query_id"])
        inventory = inventory_by_id[trajectory_id]
        trajectory = trajectories[trajectory_id]
        condition = conditions[query_id]
        row = dict(source)
        row["schema_version"] = "clir-prior-gate-tuning-v1-feature-row"
        row["feature_worker_index"] = int(inventory["worker_index"])
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
        row["feature_dim"] = int(contract["feature_dim"])
        row["num_feature_layers"] = int(contract["num_feature_layers"])
        row["per_layer_dim"] = int(contract["per_layer_dim"])
        row["feature_model"] = contract["model_id"]
        row["feature_revision"] = contract["model_revision"]
        row["feature_dtype"] = contract["dtype"]
        row["feature_attention_implementation"] = contract["attention_implementation"]
        row["feature_extraction_code_commit"] = plan["code_commit"]
        row["feature_extraction_authorization_sha256"] = authorization_sha256
        output.append(row)
    return output


def command_finalize(args: argparse.Namespace) -> None:
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    authorization, plan, inventory, plan_path = _load_plan(
        authorization_path=authorization_path, output_root=output_root
    )
    preflight_path = output_root / "preflight/report.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_GATE_TUNING_V1_FULL_WIDTH_FEATURE_PREFLIGHT":
        raise ValueError("full-width feature preflight did not pass")
    final_root = output_root / "final"
    if final_root.exists() and any(final_root.iterdir()):
        raise FileExistsError(f"final feature directory is not empty: {final_root}")
    worker_count = int(authorization["runtime_contract"]["worker_count"])
    report_hashes: list[dict[str, Any]] = []
    raw_bytes = 0
    serialized_bytes = 0
    for worker_index in range(worker_count):
        extraction_path = output_root / f"extraction/worker-{worker_index:03d}.json"
        verifier_path = output_root / f"verification/worker-{worker_index:03d}.json"
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
        if extraction.get("status") != WORKER_STATUS:
            raise ValueError(f"feature worker {worker_index} did not pass")
        if verifier.get("status") != VERIFIER_STATUS:
            raise ValueError(f"feature verifier {worker_index} did not pass")
        if extraction["payload_record_digest"] != verifier[
            "payload_record_digest"
        ] or int(extraction["raw_tensor_bytes"]) != int(verifier["raw_tensor_bytes"]):
            raise ValueError(f"feature worker {worker_index} verification drift")
        raw_bytes += int(verifier["raw_tensor_bytes"])
        serialized_bytes += int(verifier["serialized_bytes"])
        report_hashes.append(
            {
                "worker_index": worker_index,
                "extraction_report_file_sha256": file_sha256(extraction_path),
                "verification_report_file_sha256": file_sha256(verifier_path),
            }
        )
    expected = authorization["expected_inventory"]
    if raw_bytes != int(expected["raw_feature_bytes"]):
        raise ValueError("verified feature raw byte total drift")
    authorization_sha = file_sha256(authorization_path)
    trajectories, conditions = _payload_maps(
        output_root,
        inventory,
        plan=plan,
        authorization_sha256=authorization_sha,
    )
    if len(trajectories) != int(expected["trajectory_count"]):
        raise ValueError("final trajectory feature population drift")
    if len(conditions) != int(expected["condition_count"]):
        raise ValueError("final condition feature population drift")
    tuning_source, confirmation_source = _selected_sources(authorization)
    inventory_by_id = {str(row["trajectory_id"]): row for row in inventory}
    final_root.mkdir(parents=True, exist_ok=True)
    tuning_path = final_root / "tuning_features.jsonl"
    confirmation_path = final_root / "confirmation_features.sealed.jsonl"
    tuning_rows = _attach_features(
        tuning_source,
        inventory_by_id=inventory_by_id,
        trajectories=trajectories,
        conditions=conditions,
        output_root=output_root,
        manifest_path=tuning_path,
        authorization=authorization,
        plan=plan,
        authorization_sha256=authorization_sha,
    )
    confirmation_rows = _attach_features(
        confirmation_source,
        inventory_by_id=inventory_by_id,
        trajectories=trajectories,
        conditions=conditions,
        output_root=output_root,
        manifest_path=confirmation_path,
        authorization=authorization,
        plan=plan,
        authorization_sha256=authorization_sha,
    )
    tuning_manifest = publish_manifest(
        tuning_path,
        tuning_rows,
        schema_version="clir-prior-gate-tuning-v1-tuning-features",
        metadata={
            "queries": len(tuning_rows)
            // int(authorization["runtime_contract"]["candidate_count"]),
            "sealed": False,
            "independently_verified": True,
        },
    )
    confirmation_manifest = publish_manifest(
        confirmation_path,
        confirmation_rows,
        schema_version="clir-prior-gate-tuning-v1-confirmation-features-sealed",
        metadata={
            "queries": len(confirmation_rows)
            // int(authorization["runtime_contract"]["candidate_count"]),
            "sealed": True,
            "checker_outcome_distribution_sealed": True,
            "independently_verified": True,
        },
    )
    report = {
        "schema_version": "clir-prior-gate-tuning-v1-feature-completion",
        "status": "PASS_GATE_TUNING_V1_SELECTED_FEATURES",
        "completed_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "authorization_file_sha256": authorization_sha,
        "plan_file_sha256": file_sha256(plan_path),
        "preflight_report_file_sha256": file_sha256(preflight_path),
        "worker_reports": report_hashes,
        "inventory_statistics": plan["inventory_report"]["total"],
        "trajectory_payload_count": len(trajectories),
        "condition_payload_count": len(conditions),
        "raw_tensor_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
        "feature_contract": authorization["feature_contract"],
        "tuning_manifest": tuning_manifest,
        "confirmation_manifest": confirmation_manifest,
        "all_payload_shapes_dtypes_finiteness_and_checksums_verified": True,
        "unselected_rollouts_extracted": False,
        "confirmation_outcomes_opened": False,
        "training_allowed": False,
        "tuning_scoring_allowed": False,
        "confirmation_scoring_allowed": False,
        "next_gate": "separate_hash_bound_fresh_tuning_training_and_scoring_authorization",
    }
    report_path = final_root / "completion.json"
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_file_sha256": file_sha256(report_path),
                "inventory_statistics": report["inventory_statistics"],
                "raw_tensor_bytes": raw_bytes,
                "serialized_bytes": serialized_bytes,
                "confirmation_outcomes_opened": False,
                "training_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _common(prepare)
    prepare.set_defaults(func=command_prepare)
    verify_plan = subparsers.add_parser("verify-plan")
    _common(verify_plan)
    verify_plan.set_defaults(func=command_verify_plan)
    preflight = subparsers.add_parser("preflight")
    _common(preflight)
    preflight.add_argument("--device", default="cuda:0")
    preflight.set_defaults(func=command_preflight)
    extract_worker = subparsers.add_parser("extract-worker")
    _common(extract_worker)
    extract_worker.add_argument("--worker-index", type=int, required=True)
    extract_worker.add_argument("--device", default="cuda:0")
    extract_worker.set_defaults(func=command_extract_worker)
    verify_worker = subparsers.add_parser("verify-worker")
    _common(verify_worker)
    verify_worker.add_argument("--worker-index", type=int, required=True)
    verify_worker.set_defaults(func=command_verify_worker)
    finalize = subparsers.add_parser("finalize")
    _common(finalize)
    finalize.set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
