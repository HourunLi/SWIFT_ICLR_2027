#!/usr/bin/env python
"""Execute the single frozen H0 v7.1 acquisition-yield rescue round."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from src.clir_h_yield_rescue import RESCUE_SCHEMA, build_rescue_plan
from src.clir_scale_pre_annotation import (
    materialize_scale_rows,
    validate_scale_materialized_rows,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
    stable_priority,
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"
DEFAULT_AMENDMENT = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/yield_rescue_amendment_v7_1.json"
)
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/yield_rescue_authorization_v7_1.json"
)
DEFAULT_PRE_ANNOTATION_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_annotation"
)
DEFAULT_RESCUE_ROOT = DEFAULT_PRE_ANNOTATION_ROOT / "yield_rescue"
DEFAULT_PRE_ROLLOUT = DEFAULT_RESCUE_ROOT / "pre_rollout"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _read_published_jsonl(
    path: Path, *, expected_schema: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"published JSONL or sidecar missing: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if expected_schema is not None and sidecar.get("schema_version") != expected_schema:
        raise ValueError(f"{path}: schema mismatch")
    if sidecar.get("file_sha256") != file_sha256(path):
        raise ValueError(f"{path}: file hash mismatch")
    rows = read_jsonl(path)
    if int(sidecar.get("row_count", -1)) != len(rows):
        raise ValueError(f"{path}: row count mismatch")
    if sidecar.get("ordered_rows_sha256") != canonical_sha256(rows):
        raise ValueError(f"{path}: ordered-row hash mismatch")
    return rows, sidecar


def load_amendment(
    path: Path, *, protocol_path: Path, pre_annotation_root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("schema_version") != RESCUE_SCHEMA:
        raise ValueError("unsupported yield-rescue amendment")
    if amendment.get("status") != "FROZEN_ONE_SHOT_RESCUE_NOT_STARTED":
        raise ValueError("yield-rescue amendment is not at its frozen start gate")
    h_path = pre_annotation_root / "materialized/h_materialized.jsonl"
    h_rows, h_sidecar = _read_published_jsonl(
        h_path, expected_schema="clir-ranking-v7-h-materialized"
    )
    parent = amendment["parent"]
    expected = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_annotation_authorization_file_sha256": file_sha256(
            PROJECT_ROOT
            / "configs/ranking_expansion_v7/pre_annotation_authorization.json"
        ),
        "materialization_report_file_sha256": file_sha256(
            pre_annotation_root / "materialized/materialization_report.json"
        ),
        "h_materialized_file_sha256": h_sidecar["file_sha256"],
        "h_materialized_sidecar_file_sha256": file_sha256(
            h_path.with_suffix(h_path.suffix + ".manifest.json")
        ),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"yield-rescue parent {key} mismatch")
    if len(h_rows) != int(parent["h_materialized_row_count"]):
        raise ValueError("yield-rescue parent H row count mismatch")
    return protocol, amendment, h_rows


def command_freeze(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    amendment_path = Path(args.amendment).resolve()
    pre_root = Path(args.pre_annotation_root).resolve()
    output = Path(args.pre_rollout_dir).resolve()
    if _git_dirty():
        raise RuntimeError("yield-rescue freeze requires a clean Git commit")
    if (output / "manifest_registry.json").exists():
        raise FileExistsError("yield-rescue pre-rollout registry already exists")
    protocol, amendment, h_rows = load_amendment(
        amendment_path,
        protocol_path=protocol_path,
        pre_annotation_root=pre_root,
    )
    queries, shards, plan = build_rescue_plan(h_rows, protocol, amendment)
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "rescue_queries.jsonl"
    query_manifest = publish_manifest(
        query_path,
        queries,
        schema_version="clir-h0-v7.1-yield-rescue-queries",
        metadata=plan,
    )
    shards_path = output / "rollout_shards.json"
    atomic_write_json(shards_path, shards)
    report = {
        "schema_version": "clir-h0-v7.1-yield-rescue-freeze-report",
        "status": "PASS_H0_V7_1_YIELD_RESCUE_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "code_dirty": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "amendment_file_sha256": file_sha256(amendment_path),
        "parent_h_materialized_file_sha256": file_sha256(
            pre_root / "materialized/h_materialized.jsonl"
        ),
        "query_file_sha256": query_manifest["file_sha256"],
        "query_sidecar_file_sha256": file_sha256(
            query_path.with_suffix(query_path.suffix + ".manifest.json")
        ),
        "query_ordered_rows_sha256": query_manifest["ordered_rows_sha256"],
        "query_count": len(queries),
        "shards_file_sha256": file_sha256(shards_path),
        "shards_canonical_sha256": canonical_sha256(shards),
        "shard_count": len(shards),
        "expected_candidate_rows": sum(
            int(shard["expected_candidate_rows"]) for shard in shards
        ),
        "plan": plan,
        "rollout_started": False,
        "next_gate": "hash_bound_one_round_rescue_authorization",
    }
    report_path = output / "freeze_report.json"
    atomic_write_json(report_path, report)
    registry = {
        "schema_version": "clir-h0-v7.1-yield-rescue-registry",
        "status": report["status"],
        "code_commit": report["code_commit"],
        "amendment_file_sha256": report["amendment_file_sha256"],
        "query_file_sha256": report["query_file_sha256"],
        "query_sidecar_file_sha256": report["query_sidecar_file_sha256"],
        "query_ordered_rows_sha256": report["query_ordered_rows_sha256"],
        "shards_file_sha256": report["shards_file_sha256"],
        "shards_canonical_sha256": report["shards_canonical_sha256"],
        "freeze_report_file_sha256": file_sha256(report_path),
    }
    atomic_write_json(output / "manifest_registry.json", registry)
    print(
        json.dumps(
            {
                **report,
                "manifest_registry_file_sha256": file_sha256(
                    output / "manifest_registry.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_authorization(
    path: Path,
    *,
    protocol_path: Path,
    amendment_path: Path,
    pre_annotation_root: Path,
    pre_rollout_dir: Path,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "clir-h0-v7.1-yield-rescue-authorization"
    ):
        raise ValueError("unsupported yield-rescue authorization")
    if authorization.get("status") != "AUTHORIZED_ONE_RESCUE_ROLLOUT_ROUND_ONLY":
        raise ValueError("yield-rescue rollout is not authorized")
    scope = authorization.get("authorized_scope", {})
    if scope.get("one_rescue_rollout_round") is not True or any(
        scope.get(name) is not False
        for name in (
            "second_rescue_round",
            "ai_annotation",
            "query_role_or_split_change",
            "threshold_or_quota_change",
            "feature_extraction",
            "training",
        )
    ):
        raise ValueError("yield-rescue authorization scope is invalid")
    _, amendment, _ = load_amendment(
        amendment_path,
        protocol_path=protocol_path,
        pre_annotation_root=pre_annotation_root,
    )
    registry_path = pre_rollout_dir / "manifest_registry.json"
    report_path = pre_rollout_dir / "freeze_report.json"
    parent = authorization["frozen_parent"]
    expected = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "amendment_file_sha256": file_sha256(amendment_path),
        "pre_rollout_registry_file_sha256": file_sha256(registry_path),
        "freeze_report_file_sha256": file_sha256(report_path),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"yield-rescue authorization {key} mismatch")
    if int(amendment["rollout_shards"]) != int(
        authorization["runtime_contract"]["maximum_concurrent_shards"]
    ):
        raise ValueError("yield-rescue concurrency differs from amendment")
    return authorization


def _load_contract(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    Path,
]:
    protocol_path = Path(args.protocol).resolve()
    amendment_path = Path(args.amendment).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_root = Path(args.pre_annotation_root).resolve()
    pre_rollout = Path(args.pre_rollout_dir).resolve()
    rescue_root = Path(args.rescue_root).resolve()
    protocol, amendment, _ = load_amendment(
        amendment_path,
        protocol_path=protocol_path,
        pre_annotation_root=pre_root,
    )
    authorization = load_authorization(
        authorization_path,
        protocol_path=protocol_path,
        amendment_path=amendment_path,
        pre_annotation_root=pre_root,
        pre_rollout_dir=pre_rollout,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if rescue_root != expected_root:
        raise ValueError("yield-rescue output root differs from authorization")
    queries, _ = _read_published_jsonl(
        pre_rollout / "rescue_queries.jsonl",
        expected_schema="clir-h0-v7.1-yield-rescue-queries",
    )
    shards = json.loads(
        (pre_rollout / "rollout_shards.json").read_text(encoding="utf-8")
    )
    query_by_id = {str(row["query_id"]): row for row in queries}
    if len(query_by_id) != int(amendment["rescue_query_count"]):
        raise ValueError("yield-rescue query manifest count mismatch")
    return protocol, amendment, authorization, shards, query_by_id, rescue_root


def _select_shard(shards: Sequence[Mapping[str, Any]], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected one rescue shard {shard_id}, found {len(matches)}")
    return matches[0]


def _shard_paths(root: Path, shard: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = root.parent / str(shard["output_path"])
    return (
        output,
        output.with_suffix(output.suffix + ".manifest.json"),
        output.with_suffix(".complete.json"),
    )


def _seed(amendment: Mapping[str, Any], query_id: str) -> int:
    namespace = str(amendment["generation"]["independent_seed_namespace"])
    return int(stable_priority(namespace, query_id)[:16], 16) % (2**31)


def _prompt(question: str, template: str) -> str:
    if "<QUESTION>" not in template:
        raise ValueError("prompt template lacks <QUESTION>")
    return template.replace("<QUESTION>", question)


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    protocol_path: Path,
    amendment_path: Path,
    authorization_path: Path,
    pre_rollout_dir: Path,
) -> dict[str, Any]:
    start = int(shard["candidate_index_start"])
    count = int(shard["candidate_count"])
    expected_query_ids = [str(value) for value in shard["query_ids"]]
    encountered: list[str] = []
    by_query: dict[str, list[int]] = {}
    finish: Counter[str] = Counter()
    output_lengths: list[int] = []
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered or encountered[-1] != query_id:
            encountered.append(query_id)
        by_query.setdefault(query_id, []).append(int(row["candidate_index"]))
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"{shard['shard_id']}: unknown query {query_id}")
        index = int(row["candidate_index"])
        if row.get("id") != f"{query_id}:cand:{index:03d}":
            raise ValueError(f"{query_id}: noncanonical rescue trajectory ID")
        for field in (
            "source",
            "question",
            "reference_answer",
            "cluster_id",
            "h_target_checker_status",
            "h_label_split",
        ):
            if row.get(field) != query.get(field):
                raise ValueError(f"{row['id']}: frozen field {field} drift")
        if row.get("shard_id") != shard["shard_id"]:
            raise ValueError(f"{row['id']}: rescue shard ID drift")
        if row.get("prompt_token_ids") != query.get("prompt_token_ids"):
            raise ValueError(f"{row['id']}: exact prompt token IDs drift")
        if int(row["sampling_seed"]) != _seed(amendment, query_id):
            raise ValueError(f"{row['id']}: rescue sampling seed drift")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{row['id']}: missing rescue provenance")
        expected_provenance = {
            "protocol_file_sha256": file_sha256(protocol_path),
            "amendment_file_sha256": file_sha256(amendment_path),
            "authorization_file_sha256": file_sha256(authorization_path),
            "pre_rollout_registry_file_sha256": file_sha256(
                pre_rollout_dir / "manifest_registry.json"
            ),
            "model_revision": protocol["generation"]["model_revision"],
            "tokenizer_revision": protocol["generation"]["tokenizer_revision"],
            "vllm_version": protocol["generation"]["backend_version"],
        }
        if any(
            str(provenance.get(key)) != str(value)
            for key, value in expected_provenance.items()
        ):
            raise ValueError(f"{row['id']}: rescue provenance drift")
        finish[str(row.get("finish_reason"))] += 1
        output_lengths.append(len(row["output_token_ids"]))
    if encountered != expected_query_ids:
        raise ValueError(f"{shard['shard_id']}: query order differs from freeze")
    if len(rows) != int(shard["expected_candidate_rows"]):
        raise ValueError(f"{shard['shard_id']}: row count mismatch")
    expected_indices = list(range(start, start + count))
    if any(sorted(values) != expected_indices for values in by_query.values()):
        raise ValueError(f"{shard['shard_id']}: rescue candidate index set mismatch")
    code_commits = {str(row["provenance"]["code_commit"]) for row in rows}
    if len(code_commits) != 1:
        raise ValueError(f"{shard['shard_id']}: mixed rescue code commits")
    return {
        "shard_id": shard["shard_id"],
        "queries": len(by_query),
        "rows": len(rows),
        "candidate_index_start": start,
        "candidate_index_end_exclusive": start + count,
        "finish_reason_counts": dict(sorted(finish.items())),
        "total_output_tokens": sum(output_lengths),
        "output_token_min": min(output_lengths),
        "output_token_max": max(output_lengths),
        "output_token_mean": sum(output_lengths) / len(output_lengths),
        "code_commit": next(iter(code_commits)),
    }


def verify_shard(
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    protocol_path: Path,
    amendment_path: Path,
    authorization_path: Path,
    pre_rollout_dir: Path,
    rescue_root: Path,
) -> dict[str, Any]:
    output, sidecar_path, completion_path = _shard_paths(rescue_root, shard)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE_VERIFIED_H0_YIELD_RESCUE_SHARD":
        raise ValueError(f"invalid rescue completion marker: {completion_path}")
    expected = {
        "shard_spec_sha256": canonical_sha256(shard),
        "authorization_file_sha256": file_sha256(authorization_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(f"rescue completion binding drift: {completion_path}")
    if (
        file_sha256(output) != completion["file_sha256"]
        or file_sha256(sidecar_path) != completion["sidecar_file_sha256"]
    ):
        raise ValueError(f"rescue shard file hash drift: {output}")
    rows = read_jsonl(output)
    if canonical_sha256(rows) != completion["ordered_rows_sha256"]:
        raise ValueError(f"rescue shard ordered rows drift: {output}")
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        amendment=amendment,
        protocol_path=protocol_path,
        amendment_path=amendment_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    if validation != completion["validation"]:
        raise ValueError(f"rescue shard validation drift: {completion_path}")
    return {
        "status": "PASS_H0_YIELD_RESCUE_SHARD_VERIFY",
        "shard_id": shard["shard_id"],
        "path": str(output),
        "file_sha256": completion["file_sha256"],
        "row_count": len(rows),
        "validation": validation,
        "runtime": completion["runtime"],
    }


def command_rollout(args: argparse.Namespace) -> None:
    protocol, amendment, authorization, shards, query_by_id, rescue_root = (
        _load_contract(args)
    )
    authorization_path = Path(args.authorization).resolve()
    protocol_path = Path(args.protocol).resolve()
    amendment_path = Path(args.amendment).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    shard = _select_shard(shards, args.shard_id)
    calibration_id = str(authorization["runtime_contract"]["first_calibration_shard"])
    if shard["shard_id"] != calibration_id:
        calibration = _select_shard(shards, calibration_id)
        _, _, marker = _shard_paths(rescue_root, calibration)
        if not marker.exists():
            raise RuntimeError(
                f"{calibration_id} must verify before other rescue shards"
            )
        verify_shard(
            shard=calibration,
            query_by_id=query_by_id,
            protocol=protocol,
            amendment=amendment,
            protocol_path=protocol_path,
            amendment_path=amendment_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            rescue_root=rescue_root,
        )
    output, sidecar_path, completion_path = _shard_paths(rescue_root, shard)
    if completion_path.exists():
        print(
            json.dumps(
                verify_shard(
                    shard=shard,
                    query_by_id=query_by_id,
                    protocol=protocol,
                    amendment=amendment,
                    protocol_path=protocol_path,
                    amendment_path=amendment_path,
                    authorization_path=authorization_path,
                    pre_rollout_dir=pre_rollout_dir,
                    rescue_root=rescue_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("incomplete rescue shard exists; refusing overwrite")
    if _git_dirty():
        raise RuntimeError("yield-rescue rollout requires a clean Git commit")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("yield-rescue rollout requires torch and vLLM") from exc
    generation = protocol["generation"]
    runtime = authorization["runtime_contract"]
    if _package_version("vllm") != generation["backend_version"]:
        raise ValueError("vLLM version differs from frozen parent protocol")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each rescue shard process must see exactly one GPU")
    if torch.cuda.mem_get_info(0)[0] < 40_000_000_000:
        raise RuntimeError("visible GPU has less than 40 GB free")
    query_rows = [query_by_id[str(value)] for value in shard["query_ids"]]
    started_at = _utc_now()
    started_clock = time.monotonic()
    llm = LLM(
        model=generation["model_id"],
        revision=generation["model_revision"],
        tokenizer_revision=generation["tokenizer_revision"],
        dtype=runtime["dtype"],
        tensor_parallel_size=1,
        max_model_len=int(generation["max_model_length"]),
        max_num_seqs=int(runtime["max_num_seqs"]),
        gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
        seed=int(generation["base_seed"]),
        download_dir=str(_project_path(runtime["cache_dir"])),
    )
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    expected_prompt_ids: list[list[int]] = []
    sampling: list[Any] = []
    candidate_count = int(shard["candidate_count"])
    for query in query_rows:
        user_prompt = _prompt(
            str(query["question"]), str(generation["prompt_template"])
        )
        messages = [{"role": "user", "content": user_prompt}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        prompt_ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        ]
        if len(prompt_ids) != int(query["prompt_token_count"]):
            raise ValueError(f"{query['query_id']}: rescue prompt token count drift")
        if prompt_ids != query["prompt_token_ids"]:
            raise ValueError(f"{query['query_id']}: frozen rescue prompt IDs drift")
        expected_prompt_ids.append(prompt_ids)
        sampling.append(
            SamplingParams(
                n=candidate_count,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_seed(amendment, str(query["query_id"])),
            )
        )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    if len(request_outputs) != len(query_rows):
        raise ValueError("vLLM returned a different rescue query count")
    finished_at = _utc_now()
    elapsed = time.monotonic() - started_clock
    provenance = {
        "protocol_file_sha256": file_sha256(Path(args.protocol)),
        "amendment_file_sha256": file_sha256(Path(args.amendment)),
        "authorization_file_sha256": file_sha256(authorization_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "code_commit": _git_head(),
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "tokenizer_revision": generation["tokenizer_revision"],
        "vllm_version": _package_version("vllm"),
        "transformers_version": _package_version("transformers"),
        "torch_version": torch.__version__,
        "gpu_model": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": elapsed,
    }
    start = int(shard["candidate_index_start"])
    rows: list[dict[str, Any]] = []
    for query, request_output, expected_ids in zip(
        query_rows, request_outputs, expected_prompt_ids
    ):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        if prompt_ids != expected_ids:
            raise ValueError(f"{query['query_id']}: vLLM rescue prompt IDs drift")
        candidates = sorted(request_output.outputs, key=lambda value: int(value.index))
        if [int(value.index) for value in candidates] != list(range(candidate_count)):
            raise ValueError("vLLM rescue candidate indices are not contiguous")
        for candidate in candidates:
            global_index = start + int(candidate.index)
            output_ids = [int(value) for value in candidate.token_ids]
            if not output_ids:
                raise ValueError(f"{query['query_id']}: empty rescue output")
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{global_index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": global_index,
                    "shard_id": shard["shard_id"],
                    "role": query["role"],
                    "cluster_id": query["cluster_id"],
                    "source": query["source"],
                    "source_record_id": query.get("source_record_id"),
                    "source_subject": query.get("source_subject"),
                    "source_level": query.get("source_level"),
                    "source_license": query.get("source_license"),
                    "h_target_checker_status": query["h_target_checker_status"],
                    "h_label_split": query["h_label_split"],
                    "rescue_cell": query["rescue_cell"],
                    "question": query["question"],
                    "reference_answer": query["reference_answer"],
                    "prompt": _prompt(
                        str(query["question"]), str(generation["prompt_template"])
                    ),
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": output_ids,
                    "response": response,
                    "backend_response_text": candidate.text,
                    "decode_matches_backend_text": response == candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "sampling_seed": _seed(amendment, str(query["query_id"])),
                    "provenance": provenance,
                }
            )
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        amendment=amendment,
        protocol_path=protocol_path,
        amendment_path=amendment_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-h0-v7.1-yield-rescue-raw-shard",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-h0-v7.1-yield-rescue-shard-completion",
        "status": "COMPLETE_VERIFIED_H0_YIELD_RESCUE_SHARD",
        "shard_id": shard["shard_id"],
        "shard_spec_sha256": canonical_sha256(shard),
        "authorization_file_sha256": file_sha256(authorization_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar_path),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "row_count": manifest["row_count"],
        "validation": validation,
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed,
            "rows_per_second": len(rows) / elapsed,
            "output_tokens_per_second": validation["total_output_tokens"] / elapsed,
        },
    }
    atomic_write_json(completion_path, completion)
    print(
        json.dumps(
            verify_shard(
                shard=shard,
                query_by_id=query_by_id,
                protocol=protocol,
                amendment=amendment,
                protocol_path=protocol_path,
                amendment_path=amendment_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                rescue_root=rescue_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify_rollouts(args: argparse.Namespace) -> None:
    protocol, amendment, _, shards, query_by_id, rescue_root = _load_contract(args)
    authorization_path = Path(args.authorization).resolve()
    protocol_path = Path(args.protocol).resolve()
    amendment_path = Path(args.amendment).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for shard in shards:
        _, _, marker = _shard_paths(rescue_root, shard)
        if not marker.exists():
            missing.append(str(shard["shard_id"]))
            continue
        reports.append(
            verify_shard(
                shard=shard,
                query_by_id=query_by_id,
                protocol=protocol,
                amendment=amendment,
                protocol_path=protocol_path,
                amendment_path=amendment_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                rescue_root=rescue_root,
            )
        )
    if args.require_complete and missing:
        raise ValueError(f"missing rescue shards: {missing}")
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ALL_H0_YIELD_RESCUE_SHARDS"
                    if not missing
                    else "PARTIAL_H0_YIELD_RESCUE_SHARDS"
                ),
                "verified_shards": len(reports),
                "verified_rows": sum(row["row_count"] for row in reports),
                "missing_shards": missing,
                "shards": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_merge(args: argparse.Namespace) -> None:
    protocol, amendment, _, shards, query_by_id, rescue_root = _load_contract(args)
    authorization_path = Path(args.authorization).resolve()
    protocol_path = Path(args.protocol).resolve()
    amendment_path = Path(args.amendment).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    output = rescue_root / "rollouts/combined_raw.jsonl"
    report_path = rescue_root / "rollout_completion_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("yield-rescue merged artifacts already exist")
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for shard in shards:
        report = verify_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            amendment=amendment,
            protocol_path=protocol_path,
            amendment_path=amendment_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            rescue_root=rescue_root,
        )
        reports.append(report)
        rows.extend(read_jsonl(Path(report["path"])))
    code_commits = {str(row["validation"]["code_commit"]) for row in reports}
    if len(code_commits) != 1:
        raise ValueError("yield-rescue shards use mixed code commits")
    start = int(amendment["candidate_index_start"])
    normalized = [
        {**row, "candidate_index": int(row["candidate_index"]) - start} for row in rows
    ]
    population = validate_rollout_population(
        normalized,
        candidate_count=int(amendment["additional_candidates_per_query"]),
    )
    if len(rows) != int(amendment["expected_additional_candidate_rows"]):
        raise ValueError("yield-rescue merged row count differs from amendment")
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-h0-v7.1-yield-rescue-combined-raw",
        metadata={
            "protocol_file_sha256": file_sha256(Path(args.protocol)),
            "amendment_file_sha256": file_sha256(Path(args.amendment)),
            "authorization_file_sha256": file_sha256(authorization_path),
            **population,
        },
    )
    report = {
        "schema_version": "clir-h0-v7.1-yield-rescue-rollout-report",
        "status": "PASS_ALL_4584_H0_YIELD_RESCUE_ROWS",
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "rows": len(rows),
        "population": population,
        "shards": reports,
        "code_commit": next(iter(code_commits)),
        "next_gate": "checker_and_exact_token_unitization_then_single_final_yield_gate",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_materialize(args: argparse.Namespace) -> None:
    protocol, amendment, _, _, _, rescue_root = _load_contract(args)
    if _git_dirty():
        raise RuntimeError("yield-rescue materialization requires a clean Git commit")
    raw_path = rescue_root / "rollouts/combined_raw.jsonl"
    raw_rows, raw_sidecar = _read_published_jsonl(
        raw_path, expected_schema="clir-h0-v7.1-yield-rescue-combined-raw"
    )
    output = rescue_root / "materialized/rescue_rows.jsonl"
    report_path = rescue_root / "materialized/materialization_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("yield-rescue materialization already exists")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("yield-rescue unitization requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    processed, health = materialize_scale_rows(
        raw_rows,
        tokenizer,
        checker_version=str(protocol["checker"]["checker_version"]),
        unitizer_version=str(protocol["checker"]["unitizer_version"]),
    )
    start = int(amendment["candidate_index_start"])
    normalized_raw = [
        {**row, "candidate_index": int(row["candidate_index"]) - start}
        for row in raw_rows
    ]
    normalized_processed = [
        {**row, "candidate_index": int(row["candidate_index"]) - start}
        for row in processed
    ]
    validation = validate_scale_materialized_rows(
        normalized_processed,
        raw_rows=normalized_raw,
        candidate_count=int(amendment["additional_candidates_per_query"]),
        checker_version=str(protocol["checker"]["checker_version"]),
        unitizer_version=str(protocol["checker"]["unitizer_version"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        processed,
        schema_version="clir-h0-v7.1-yield-rescue-materialized",
        metadata={
            "protocol_file_sha256": file_sha256(Path(args.protocol)),
            "amendment_file_sha256": file_sha256(Path(args.amendment)),
            "authorization_file_sha256": file_sha256(Path(args.authorization)),
            "raw_file_sha256": raw_sidecar["file_sha256"],
            "code_commit": _git_head(),
            **health,
        },
    )
    parent_h_path = (
        Path(args.pre_annotation_root).resolve() / "materialized/h_materialized.jsonl"
    )
    report = {
        "schema_version": "clir-h0-v7.1-yield-rescue-materialization-report",
        "status": "PASS_H0_V7_1_YIELD_RESCUE_MATERIALIZATION",
        "rows": len(processed),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "parent_h_materialized_file_sha256": file_sha256(parent_h_path),
        "health": health,
        "validation": validation,
        "second_rescue_round_allowed": False,
        "next_gate": "rerun_original_H_proposal_freeze_once",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    parser.add_argument("--pre-rollout-dir", default=str(DEFAULT_PRE_ROLLOUT))
    parser.add_argument("--rescue-root", default=str(DEFAULT_RESCUE_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.set_defaults(func=command_freeze)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify = subparsers.add_parser("verify-rollouts")
    verify.add_argument("--require-complete", action="store_true")
    verify.set_defaults(func=command_verify_rollouts)
    merge = subparsers.add_parser("merge-rollouts")
    merge.set_defaults(func=command_merge)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--cache-dir")
    materialize.set_defaults(func=command_materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
