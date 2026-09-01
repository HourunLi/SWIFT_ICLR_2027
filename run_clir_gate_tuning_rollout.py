#!/usr/bin/env python
"""Run and verify hash-authorized Prior/Gate tuning-v1 Phi rollouts.

This runner can only generate the frozen raw candidate population.  It cannot
run the numeric checker, select eligible queries, extract features, train CLIR,
or score the sealed confirmation split.
"""

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

from prepare_clir_gate_tuning import load_protocol
from src.clir_gate_tuning import CONFIRMATION_ROLE, TUNING_ROLE
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
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_gate_tuning_v1/protocol.json"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/rollout_authorization.json"
)
DEFAULT_PRE_ROLLOUT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/pre_rollout"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1"


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
        raise RuntimeError("Prior/Gate rollout requires a clean Git commit")
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
        raise ValueError(f"authorized parent {ancestor} is not an ancestor of HEAD")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _assert_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{label} hash drift: {observed} != {expected_sha256}")


def load_authorization(
    path: str | Path,
    *,
    protocol_path: Path,
    pre_rollout_dir: Path,
) -> dict[str, Any]:
    authorization_path = Path(path).resolve()
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "clir-prior-gate-tuning-v1-rollout-authorization"
    ):
        raise ValueError("unsupported Prior/Gate rollout authorization")
    if authorization.get("status") != "AUTHORIZED_RAW_ROLLOUT_ONLY":
        raise ValueError("Prior/Gate raw rollout is not authorized")
    expected_scope = {
        "raw_rollout": True,
        "raw_shard_verification_and_merge": True,
        "numeric_checker_or_eligibility_selection": False,
        "feature_extraction": False,
        "new_training": False,
        "tuning_scoring": False,
        "confirmation_scoring": False,
        "query_quota_threshold_or_generator_change": False,
        "adaptive_additional_sampling": False,
    }
    if authorization.get("authorized_scope") != expected_scope:
        raise ValueError("Prior/Gate rollout authorization scope drift")
    parent = authorization["frozen_parent"]
    expected_files = {
        "protocol_file_sha256": protocol_path,
        "manifest_registry_file_sha256": pre_rollout_dir / "manifest_registry.json",
        "freeze_report_file_sha256": pre_rollout_dir / "freeze_report.json",
        "independent_verification_file_sha256": (
            pre_rollout_dir / "independent_verification.json"
        ),
        "tuning_queries_file_sha256": pre_rollout_dir / "tuning_queries.jsonl",
        "confirmation_queries_file_sha256": (
            pre_rollout_dir / "confirmation_queries.jsonl"
        ),
        "rollout_shards_file_sha256": pre_rollout_dir / "rollout_shards.json",
    }
    for field, file_path in expected_files.items():
        if file_sha256(file_path) != parent.get(field):
            raise ValueError(f"rollout authorization {field} mismatch")
    verification = json.loads(
        (pre_rollout_dir / "independent_verification.json").read_text(encoding="utf-8")
    )
    if verification.get("status") != (
        "PASS_GATE_TUNING_V1_PRE_ROLLOUT_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("pre-rollout independent verification is not a PASS")
    runtime = authorization["runtime_contract"]
    if int(runtime["tensor_parallel_size"]) != 1:
        raise ValueError("Prior/Gate rollout requires tensor parallel size 1")
    if int(runtime["maximum_concurrent_shards"]) > 8:
        raise ValueError("Prior/Gate rollout concurrency exceeds 8 GPUs")
    return authorization


def _read_frozen_queries(
    pre_rollout_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tuning = read_jsonl(pre_rollout_dir / "tuning_queries.jsonl")
    confirmation = read_jsonl(pre_rollout_dir / "confirmation_queries.jsonl")
    if len(tuning) != 1300 or len(confirmation) != 1300:
        raise ValueError("frozen Prior/Gate query count drift")
    if any(row.get("role") != TUNING_ROLE for row in tuning):
        raise ValueError("tuning manifest role drift")
    if any(row.get("role") != CONFIRMATION_ROLE for row in confirmation):
        raise ValueError("confirmation manifest role drift")
    if any(row.get("sealed_until_weight_lock") is not True for row in confirmation):
        raise ValueError("confirmation manifest is not sealed")
    tuning_ids = {str(row["query_id"]) for row in tuning}
    confirmation_ids = {str(row["query_id"]) for row in confirmation}
    tuning_clusters = {str(row["cluster_id"]) for row in tuning}
    confirmation_clusters = {str(row["cluster_id"]) for row in confirmation}
    if tuning_ids & confirmation_ids or tuning_clusters & confirmation_clusters:
        raise ValueError("frozen tuning/confirmation population overlap")
    return tuning, confirmation


def _load_contract(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    Path,
    Path,
    Path,
]:
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, _ = load_protocol(protocol_path)
    authorization = load_authorization(
        authorization_path,
        protocol_path=protocol_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if output_root != expected_root:
        raise ValueError(f"output root drift: {output_root} != {expected_root}")
    tuning, confirmation = _read_frozen_queries(pre_rollout_dir)
    query_by_id = {str(row["query_id"]): row for row in [*tuning, *confirmation]}
    shards = json.loads(
        (pre_rollout_dir / "rollout_shards.json").read_text(encoding="utf-8")
    )
    if len(shards) != 52:
        raise ValueError("frozen rollout shard count drift")
    shard_ids = [str(query_id) for row in shards for query_id in row["query_ids"]]
    if len(shard_ids) != len(set(shard_ids)) or set(shard_ids) != set(query_by_id):
        raise ValueError("rollout shards do not exactly partition frozen queries")
    return (
        protocol,
        authorization,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    )


def _derive_query_seed(protocol: Mapping[str, Any], query_id: str) -> int:
    generation = protocol["generation"]
    return int(
        stable_priority(
            str(generation["seed_namespace"]),
            int(generation["base_seed"]),
            query_id,
        )[:16],
        16,
    ) % (2**31)


def _prompt_for(question: str, template: str) -> str:
    if "<QUESTION>" not in template:
        raise ValueError("generation prompt lacks <QUESTION>")
    return template.replace("<QUESTION>", question)


def _select_shard(shards: Sequence[Mapping[str, Any]], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one shard {shard_id}")
    return matches[0]


def _shard_paths(root: Path, shard: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = root / str(shard["output_path"])
    return (
        output,
        output.with_suffix(output.suffix + ".manifest.json"),
        output.with_suffix(".complete.json"),
    )


def _ordered_candidates(request_output: Any, expected_count: int) -> list[Any]:
    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(f"vLLM candidate axis drift: {sorted(indices)}")
    return sorted(candidates, key=lambda candidate: int(candidate.index))


def _integer_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    authorization_sha256: str,
    registry_sha256: str,
) -> dict[str, Any]:
    candidate_count = int(shard["candidate_count"])
    population = validate_rollout_population(rows, candidate_count=candidate_count)
    expected_query_ids = [str(value) for value in shard["query_ids"]]
    encountered: list[str] = []
    output_lengths: list[int] = []
    prompt_lengths: list[int] = []
    finish_reasons: Counter[str] = Counter()
    decode_mismatches = 0
    code_commits: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered or encountered[-1] != query_id:
            encountered.append(query_id)
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"unknown rollout query {query_id}")
        index = int(row["candidate_index"])
        if row.get("id") != f"{query_id}:cand:{index:03d}":
            raise ValueError(f"{query_id}: trajectory ID drift")
        for field in (
            "role",
            "evaluation_split",
            "sealed_until_weight_lock",
            "cluster_id",
            "source",
            "question",
            "reference_answer",
        ):
            if row.get(field) != query.get(field):
                raise ValueError(f"{query_id}: rollout field drift: {field}")
        if [int(value) for value in row["prompt_token_ids"]] != [
            int(value) for value in query["prompt_token_ids"]
        ]:
            raise ValueError(f"{query_id}: prompt token IDs drift")
        if not row.get("output_token_ids"):
            raise ValueError(f"{query_id}: empty output token IDs")
        provenance = row["provenance"]
        expected_provenance = {
            "protocol_file_sha256": protocol_sha256,
            "pre_rollout_registry_file_sha256": registry_sha256,
            "authorization_file_sha256": authorization_sha256,
            "model_revision": protocol["generation"]["model_revision"],
            "tokenizer_revision": protocol["generation"]["tokenizer_revision"],
            "vllm_version": protocol["generation"]["backend_version"],
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                raise ValueError(f"{query_id}: provenance drift: {field}")
        code_commits.add(str(provenance["code_commit"]))
        output_lengths.append(len(row["output_token_ids"]))
        prompt_lengths.append(len(row["prompt_token_ids"]))
        finish_reasons[str(row.get("finish_reason"))] += 1
        decode_mismatches += int(not bool(row.get("decode_matches_backend_text")))
    if encountered != expected_query_ids:
        raise ValueError("rollout query order differs from frozen shard")
    if len(rows) != int(shard["expected_candidate_rows"]):
        raise ValueError("rollout shard row count drift")
    if len(code_commits) != 1:
        raise ValueError("rollout shard mixes code commits")
    return {
        **population,
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "query_order_matches_freeze": True,
        "candidate_axis_matches_freeze": True,
        "prompt_token_ids_match_freeze": True,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "prompt_token_count": _integer_summary(prompt_lengths),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
        "decode_mismatch_count": decode_mismatches,
        "code_commit": next(iter(code_commits)),
    }


def verify_shard(
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    authorization_path: Path,
    pre_rollout_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    output, sidecar_path, completion_path = _shard_paths(output_root, shard)
    if not completion_path.is_file():
        raise FileNotFoundError(f"missing shard completion: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE_VERIFIED_GATE_TUNING_V1_ROLLOUT_SHARD":
        raise ValueError(f"invalid shard completion status: {completion_path}")
    if completion.get("shard_spec_sha256") != canonical_sha256(shard):
        raise ValueError("rollout shard spec hash drift")
    _assert_file(output, str(completion["file_sha256"]), "rollout shard")
    _assert_file(
        sidecar_path,
        str(completion["sidecar_file_sha256"]),
        "rollout shard sidecar",
    )
    rows = read_jsonl(output)
    if canonical_sha256(rows) != completion["ordered_rows_sha256"]:
        raise ValueError("rollout shard ordered rows drift")
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        protocol_sha256=file_sha256(protocol_path),
        authorization_sha256=file_sha256(authorization_path),
        registry_sha256=file_sha256(pre_rollout_dir / "manifest_registry.json"),
    )
    if validation != completion["validation"]:
        raise ValueError("rollout shard validation summary drift")
    return {
        "status": "PASS_GATE_TUNING_V1_ROLLOUT_SHARD_VERIFY",
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "path": str(output),
        "row_count": len(rows),
        "file_sha256": completion["file_sha256"],
        "ordered_rows_sha256": completion["ordered_rows_sha256"],
        "validation": validation,
        "runtime": completion["runtime"],
    }


def command_rollout(args: argparse.Namespace) -> None:
    (
        protocol,
        authorization,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    ) = _load_contract(args)
    shard = _select_shard(shards, args.shard_id)
    pre_rollout_dir = Path(args.pre_rollout).resolve()
    calibration_id = str(authorization["runtime_contract"]["first_calibration_shard"])
    if shard["shard_id"] != calibration_id:
        calibration = _select_shard(shards, calibration_id)
        _, _, marker = _shard_paths(output_root, calibration)
        if not marker.exists():
            raise RuntimeError(f"calibration shard {calibration_id} must run first")
        verify_shard(
            shard=calibration,
            query_by_id=query_by_id,
            protocol=protocol,
            protocol_path=protocol_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            output_root=output_root,
        )
    output, sidecar_path, completion_path = _shard_paths(output_root, shard)
    if completion_path.exists():
        print(
            json.dumps(
                verify_shard(
                    shard=shard,
                    query_by_id=query_by_id,
                    protocol=protocol,
                    protocol_path=protocol_path,
                    authorization_path=authorization_path,
                    pre_rollout_dir=pre_rollout_dir,
                    output_root=output_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("incomplete rollout shard artifacts exist")
    code_commit = _require_clean_commit()
    _require_ancestor(
        str(authorization["runtime_contract"]["authorized_code_parent_commit"]),
        code_commit,
    )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Prior/Gate rollout requires torch and vLLM") from exc
    generation = protocol["generation"]
    runtime = authorization["runtime_contract"]
    if _package_version("vllm") != generation["backend_version"]:
        raise ValueError("installed vLLM version differs from frozen protocol")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each rollout process must see exactly one GPU")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < int(runtime["minimum_free_gpu_bytes"]):
        raise RuntimeError("visible GPU has insufficient free memory")

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
    sampling: list[Any] = []
    expected_prompt_ids: list[list[int]] = []
    candidate_count = int(generation["candidate_count"])
    for query in query_rows:
        content = _prompt_for(
            str(query["question"]), str(generation["prompt_template"])
        )
        messages = [{"role": "user", "content": content}]
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
        if prompt_ids != [int(value) for value in query["prompt_token_ids"]]:
            raise ValueError(f"{query['query_id']}: tokenizer prompt IDs changed")
        expected_prompt_ids.append(prompt_ids)
        sampling.append(
            SamplingParams(
                n=candidate_count,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_derive_query_seed(protocol, str(query["query_id"])),
            )
        )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    if len(request_outputs) != len(query_rows):
        raise ValueError("vLLM query output count drift")
    finished_at = _utc_now()
    elapsed = time.monotonic() - started_clock
    provenance = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "shard_spec_sha256": canonical_sha256(shard),
        "code_commit": code_commit,
        "code_dirty": False,
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "tokenizer_revision": generation["tokenizer_revision"],
        "backend": generation["backend"],
        "vllm_version": _package_version("vllm"),
        "transformers_version": _package_version("transformers"),
        "torch_version": torch.__version__,
        "tensor_parallel_size": 1,
        "dtype": runtime["dtype"],
        "gpu_model": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "max_num_seqs": int(runtime["max_num_seqs"]),
        "gpu_memory_utilization": float(runtime["gpu_memory_utilization"]),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": elapsed,
    }
    rows: list[dict[str, Any]] = []
    for query, request_output, frozen_prompt_ids in zip(
        query_rows, request_outputs, expected_prompt_ids, strict=True
    ):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        if prompt_ids != frozen_prompt_ids:
            raise ValueError(f"{query['query_id']}: vLLM prompt IDs changed")
        for candidate in _ordered_candidates(request_output, candidate_count):
            output_ids = [int(value) for value in candidate.token_ids]
            if not output_ids:
                raise ValueError(f"{query['query_id']}: empty vLLM output")
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            index = int(candidate.index)
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": index,
                    "shard_id": shard["shard_id"],
                    "role": query["role"],
                    "evaluation_split": query["evaluation_split"],
                    "evaluation_only": True,
                    "sealed_until_weight_lock": query["sealed_until_weight_lock"],
                    "cluster_id": query["cluster_id"],
                    "source": query["source"],
                    "source_record_id": query.get("source_record_id"),
                    "source_subject": query.get("source_subject"),
                    "source_level": query.get("source_level"),
                    "source_license": query.get("source_license"),
                    "question": query["question"],
                    "reference_answer": query["reference_answer"],
                    "prompt": _prompt_for(
                        str(query["question"]), str(generation["prompt_template"])
                    ),
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": output_ids,
                    "response": response,
                    "backend_response_text": candidate.text,
                    "decode_matches_backend_text": response == candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "sampling_seed": _derive_query_seed(
                        protocol, str(query["query_id"])
                    ),
                    "provenance": provenance,
                }
            )
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        protocol_sha256=file_sha256(protocol_path),
        authorization_sha256=file_sha256(authorization_path),
        registry_sha256=file_sha256(pre_rollout_dir / "manifest_registry.json"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-gate-tuning-v1-raw-rollout-shard",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-gate-tuning-v1-rollout-shard-completion",
        "status": "COMPLETE_VERIFIED_GATE_TUNING_V1_ROLLOUT_SHARD",
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "shard_spec_sha256": canonical_sha256(shard),
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
                protocol_path=protocol_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                output_root=output_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify(args: argparse.Namespace) -> None:
    (
        protocol,
        _,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    ) = _load_contract(args)
    pre_rollout_dir = Path(args.pre_rollout).resolve()
    requested = set(args.shard_id or [])
    if requested:
        unknown = requested - {str(row["shard_id"]) for row in shards}
        if unknown:
            raise ValueError(f"unknown shard IDs: {sorted(unknown)}")
        shards = [row for row in shards if row["shard_id"] in requested]
    complete: list[dict[str, Any]] = []
    missing: list[str] = []
    for shard in shards:
        _, _, marker = _shard_paths(output_root, shard)
        if not marker.exists():
            missing.append(str(shard["shard_id"]))
            continue
        complete.append(
            verify_shard(
                shard=shard,
                query_by_id=query_by_id,
                protocol=protocol,
                protocol_path=protocol_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                output_root=output_root,
            )
        )
    if args.require_complete and missing:
        raise ValueError(f"missing rollout shards: {missing}")
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ALL_GATE_TUNING_V1_ROLLOUT_SHARDS"
                    if not missing
                    else "PARTIAL_GATE_TUNING_V1_ROLLOUT_SHARDS"
                ),
                "verified_shards": len(complete),
                "verified_rows": sum(row["row_count"] for row in complete),
                "missing_shards": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_merge(args: argparse.Namespace) -> None:
    (
        protocol,
        _,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    ) = _load_contract(args)
    pre_rollout_dir = Path(args.pre_rollout).resolve()
    report_path = output_root / "rollout_completion_report.json"
    output_paths = {
        TUNING_ROLE: output_root / "rollouts/tuning_combined_raw.jsonl",
        CONFIRMATION_ROLE: (
            output_root / "rollouts/confirmation_combined_raw.sealed.jsonl"
        ),
    }
    if report_path.exists() or any(path.exists() for path in output_paths.values()):
        raise FileExistsError("combined rollout artifacts already exist")
    rows_by_role: dict[str, list[dict[str, Any]]] = {
        TUNING_ROLE: [],
        CONFIRMATION_ROLE: [],
    }
    reports = []
    for shard in shards:
        report = verify_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            protocol_path=protocol_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            output_root=output_root,
        )
        reports.append(report)
        rows_by_role[str(shard["role"])].extend(read_jsonl(Path(report["path"])))
    all_ids = [str(row["id"]) for rows in rows_by_role.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("rollout shards overlap by trajectory ID")
    records: dict[str, Any] = {}
    for role, rows in rows_by_role.items():
        expected_queries = 1300
        population = validate_rollout_population(
            rows, candidate_count=int(protocol["generation"]["candidate_count"])
        )
        if int(population["queries"]) != expected_queries:
            raise ValueError(f"{role} combined query count drift")
        if role == CONFIRMATION_ROLE and any(
            row.get("sealed_until_weight_lock") is not True for row in rows
        ):
            raise ValueError("combined confirmation rollout lost seal marker")
        path = output_paths[role]
        manifest = publish_manifest(
            path,
            rows,
            schema_version=(
                "clir-gate-tuning-v1-confirmation-raw-rollouts-sealed"
                if role == CONFIRMATION_ROLE
                else "clir-gate-tuning-v1-tuning-raw-rollouts"
            ),
            metadata={
                **population,
                "numeric_checker_run": False,
                "clir_scoring_run": False,
                "sealed": role == CONFIRMATION_ROLE,
            },
        )
        records[role] = {
            "path": str(path),
            "file_sha256": manifest["file_sha256"],
            "sidecar_file_sha256": file_sha256(
                path.with_suffix(path.suffix + ".manifest.json")
            ),
            "row_count": manifest["row_count"],
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
            "query_count": population["queries"],
        }
    code_commits = {
        str(row["provenance"]["code_commit"])
        for rows in rows_by_role.values()
        for row in rows
    }
    if len(code_commits) != 1:
        raise ValueError("combined rollouts mix code commits")
    report = {
        "schema_version": "clir-gate-tuning-v1-rollout-completion",
        "status": "PASS_GATE_TUNING_V1_RAW_ROLLOUT_COMPLETE",
        "completed_at_utc": _utc_now(),
        "code_commit": next(iter(code_commits)),
        "protocol_file_sha256": file_sha256(protocol_path),
        "authorization_file_sha256": file_sha256(authorization_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "verified_shards": len(reports),
        "verified_rows": sum(row["row_count"] for row in reports),
        "records": records,
        "total_output_tokens": sum(
            int(row["validation"]["total_output_tokens"]) for row in reports
        ),
        "confirmation_sealed": True,
        "numeric_checker_run": False,
        "clir_scoring_run": False,
        "feature_extraction_allowed": False,
        "new_training_allowed": False,
        "next_gate": "separate_hash_bound_checker_and_checker_only_selection_authorization",
    }
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {**report, "report_file_sha256": file_sha256(report_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--pre-rollout", default=str(DEFAULT_PRE_ROLLOUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--shard-id", action="append")
    verify.add_argument("--require-complete", action="store_true")
    verify.set_defaults(func=command_verify)
    subparsers.add_parser("merge").set_defaults(func=command_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
