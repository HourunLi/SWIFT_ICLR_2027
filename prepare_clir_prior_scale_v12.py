#!/usr/bin/env python
"""Freeze and verify fresh CLIR Prior strict-consensus scale-v12 acquisition.

The initial commands are deliberately model-free.  They regenerate pinned
train-only GSM8K/MATH sources, propagate every historical/v6/v7/Prior-smoke
query through the frozen template clustering contract, and freeze 2,000 fresh
query/split identities plus 40 rollout shards.  Rollout requires a later
hash-bound authorization file and is not implied by package readiness.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from prepare_clir_h_rescue import _read_published_jsonl
from prepare_clir_ranking import (
    _combine_exclusions,
    _load_extended_sources,
    load_protocol as load_base_protocol,
)
from src.clir_prior_consensus_scale import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROTOCOL_SCHEMA,
    PROPOSAL_SCHEMA,
    QUERY_SCHEMA,
    build_acquisition_shards,
    build_prior_annotation_shards,
    evaluate_prior_v12_labels,
    select_acquisition_queries,
    select_prior_proposals,
)
from src.clir_scale import build_source_candidates, build_template_clusters
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
DEFAULT_BASE_PROTOCOL = PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v12/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12/pre_rollout"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/rollout_authorization.json"
)
DEFAULT_PRE_ANNOTATION_AUTHORIZATION = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/pre_annotation_authorization.json"
)
DEFAULT_ROLLOUT_ROOT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12"
DEFAULT_PRE_ANNOTATION_ROOT = DEFAULT_ROLLOUT_ROOT / "pre_annotation"
DEFAULT_ANNOTATION_PROMPT = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/annotation_prompt.md"
)
DEFAULT_LAUNCH_PROMPT_A = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/launch_prompt_a.txt"
)
DEFAULT_LAUNCH_PROMPT_B = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/launch_prompt_b.txt"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _prompt_for(question: str, template: str) -> str:
    if "<QUESTION>" not in template:
        raise ValueError("Prior v12 prompt template lacks <QUESTION>")
    return template.replace("<QUESTION>", question)


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


def _ordered_vllm_candidates(request_output: Any, expected_count: int) -> list[Any]:
    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "Prior v12 candidate indices must be unique and contiguous: "
            f"expected 0..{expected_count - 1}, got {sorted(indices)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate.index))


def _select_shard(shards: Sequence[Mapping[str, Any]], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected one Prior v12 shard {shard_id}")
    return matches[0]


def _shard_paths(root: Path, shard: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = root / str(shard["output_path"])
    return (
        output,
        output.with_suffix(output.suffix + ".manifest.json"),
        output.with_suffix(".complete.json"),
    )


def _integer_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(int(value) for value in values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "median": median,
    }


def _verify_file(path: str | Path, expected_sha256: str) -> Path:
    resolved = _project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"pinned file hash mismatch: {resolved}")
    return resolved


def load_scale_protocol(
    path: Path, *, base_protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_base_protocol(base_protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior v12 protocol")
    if protocol.get("status") != "FROZEN_PREPARATION_ROLLOUT_NOT_STARTED":
        raise ValueError("Prior v12 is not at its pre-rollout preparation gate")
    scope = protocol.get("execution_authorization", {})
    if (
        scope.get("source_audit_allowed") is not True
        or scope.get("pre_rollout_freeze_allowed") is not True
    ):
        raise ValueError("Prior v12 source audit/freeze is not authorized")
    for forbidden in (
        "rollout_allowed",
        "checker_unitizer_materialization_allowed",
        "ai_annotation_allowed",
        "feature_extraction_allowed",
        "training_allowed",
    ):
        if scope.get(forbidden) is not False:
            raise ValueError(f"Prior v12 pre-rollout must keep {forbidden}=false")

    parent = protocol["parent"]
    if parent["base_ranking_protocol_file_sha256"] != file_sha256(base_protocol_path):
        raise ValueError("Prior v12 base protocol hash drift")
    _verify_file(
        parent["base_pre_rollout_registry_path"],
        parent["base_pre_rollout_registry_file_sha256"],
    )
    _verify_file(
        parent["v7_ranking_queries_path"],
        parent["v7_ranking_queries_file_sha256"],
    )
    _verify_file(parent["v7_h_queries_path"], parent["v7_h_queries_file_sha256"])
    for record in parent["prior_smoke_proposals"]:
        _verify_file(record["path"], record["file_sha256"])
    terminal = _verify_file(
        parent["v11_terminal_report_path"],
        parent["v11_terminal_report_file_sha256"],
    )
    terminal_report = json.loads(terminal.read_text(encoding="utf-8"))
    if terminal_report.get("status") != parent["v11_terminal_status"]:
        raise ValueError("Prior v11 terminal status drift")
    if (
        parent.get("v8_v9_v10_v11_rows_are_prompt_development_only") is not True
        or parent.get("no_failed_smoke_label_is_trainable") is not True
    ):
        raise ValueError("Prior v12 failed-smoke exclusion policy drift")
    return base, protocol


def load_rollout_authorization(
    path: Path,
    *,
    protocol_path: Path,
    pre_rollout_dir: Path,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != "clir-prior-v12-rollout-authorization":
        raise ValueError("unsupported Prior v12 rollout authorization")
    if authorization.get("status") != "AUTHORIZED_ROLLOUT_ONLY":
        raise ValueError("Prior v12 rollout has not been authorized")
    expected_scope = {
        "rollout": True,
        "checker_and_unitizer_materialization": False,
        "annotation": False,
        "feature_extraction": False,
        "training": False,
        "adaptive_additional_sampling": False,
        "threshold_quota_query_or_split_change": False,
    }
    if authorization.get("authorized_scope") != expected_scope:
        raise ValueError("Prior v12 rollout authorization scope drift")
    parent = authorization["frozen_parent"]
    expected_files = {
        "protocol_file_sha256": protocol_path,
        "manifest_registry_file_sha256": pre_rollout_dir / "manifest_registry.json",
        "freeze_report_file_sha256": pre_rollout_dir / "freeze_report.json",
        "independent_verification_file_sha256": (
            pre_rollout_dir / "independent_verification.json"
        ),
        "query_file_sha256": pre_rollout_dir / "acquisition_queries.jsonl",
        "shards_file_sha256": pre_rollout_dir / "rollout_shards.json",
    }
    for key, file_path in expected_files.items():
        if file_sha256(file_path) != parent.get(key):
            raise ValueError(f"Prior v12 rollout authorization {key} mismatch")
    freeze = json.loads(
        (pre_rollout_dir / "freeze_report.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (pre_rollout_dir / "independent_verification.json").read_text(encoding="utf-8")
    )
    if freeze.get("status") != parent.get("pre_rollout_status"):
        raise ValueError("Prior v12 frozen status drift")
    if verification.get("status") != parent.get("independent_verification_status"):
        raise ValueError("Prior v12 independent verification status drift")
    runtime = authorization["runtime_contract"]
    if int(runtime["maximum_concurrent_shards"]) > 8:
        raise ValueError("Prior v12 rollout concurrency exceeds frozen maximum")
    if int(runtime["tensor_parallel_size"]) != 1:
        raise ValueError("Prior v12 rollout requires tensor parallel size 1")
    return authorization


def load_pre_annotation_authorization(
    path: Path,
    *,
    protocol_path: Path,
    rollout_authorization_path: Path,
    pre_rollout_dir: Path,
    rollout_root: Path,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "clir-prior-v12-pre-annotation-authorization"
    ):
        raise ValueError("unsupported Prior v12 pre-annotation authorization")
    if authorization.get("status") != "AUTHORIZED_PRE_ANNOTATION_ONLY":
        raise ValueError("Prior v12 pre-annotation has not been authorized")
    expected_scope = {
        "checker_and_unitizer_materialization": True,
        "natural_proposal_freeze": True,
        "blind_package_construction": True,
        "ai_annotation_or_provider_call": False,
        "label_selection_or_finalization": False,
        "feature_extraction": False,
        "training": False,
        "threshold_quota_query_or_split_change": False,
        "raw_rollout_overwrite_or_regeneration": False,
    }
    if authorization.get("authorized_scope") != expected_scope:
        raise ValueError("Prior v12 pre-annotation scope drift")
    parent = authorization["frozen_parent"]
    expected_files = {
        "protocol_file_sha256": protocol_path,
        "rollout_authorization_file_sha256": rollout_authorization_path,
        "pre_rollout_registry_file_sha256": pre_rollout_dir / "manifest_registry.json",
        "rollout_completion_report_file_sha256": rollout_root
        / "rollout_completion_report.json",
        "combined_raw_file_sha256": rollout_root / "rollouts/combined_raw.jsonl",
        "combined_raw_sidecar_file_sha256": rollout_root
        / "rollouts/combined_raw.jsonl.manifest.json",
    }
    for key, file_path in expected_files.items():
        if file_sha256(file_path) != parent.get(key):
            raise ValueError(f"Prior v12 pre-annotation {key} mismatch")
    rollout_report = json.loads(
        (rollout_root / "rollout_completion_report.json").read_text(encoding="utf-8")
    )
    if rollout_report.get("status") != parent.get("rollout_status"):
        raise ValueError("Prior v12 rollout status drift")
    if rollout_report.get("code_commit") != parent.get("rollout_code_commit"):
        raise ValueError("Prior v12 rollout code commit drift")
    if authorization["runtime_contract"].get("materialization_is_cpu_only") is not True:
        raise ValueError("Prior v12 materialization must remain CPU-only")
    return authorization


def _load_rollout_contract(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    Path,
]:
    base_path = Path(args.base_protocol).resolve()
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.output).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    _, protocol = load_scale_protocol(protocol_path, base_protocol_path=base_path)
    authorization = load_rollout_authorization(
        authorization_path,
        protocol_path=protocol_path,
        pre_rollout_dir=pre_rollout,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if rollout_root != expected_root:
        raise ValueError("Prior v12 rollout root differs from authorization")
    queries, _ = _read_published_jsonl(
        pre_rollout / "acquisition_queries.jsonl", expected_schema=QUERY_SCHEMA
    )
    shards = json.loads(
        (pre_rollout / "rollout_shards.json").read_text(encoding="utf-8")
    )
    query_by_id = {str(row["query_id"]): row for row in queries}
    if len(query_by_id) != int(protocol["query_pool"]["query_count"]):
        raise ValueError("Prior v12 frozen query count drift")
    return protocol, authorization, shards, query_by_id, rollout_root


def _load_pre_annotation_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    base_path = Path(args.base_protocol).resolve()
    protocol_path = Path(args.protocol).resolve()
    rollout_authorization_path = Path(args.authorization).resolve()
    pre_annotation_authorization_path = Path(
        args.pre_annotation_authorization
    ).resolve()
    pre_rollout = Path(args.output).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    _, protocol = load_scale_protocol(protocol_path, base_protocol_path=base_path)
    authorization = load_pre_annotation_authorization(
        pre_annotation_authorization_path,
        protocol_path=protocol_path,
        rollout_authorization_path=rollout_authorization_path,
        pre_rollout_dir=pre_rollout,
        rollout_root=rollout_root,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if pre_annotation_root != expected_root:
        raise ValueError("Prior v12 pre-annotation root differs from authorization")
    return protocol, authorization, rollout_root, pre_annotation_root


def _raw_rows_for_shared_materializer(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = dict(raw)
        split = str(row.get("prior_label_split", ""))
        if split not in {"train", "dev"}:
            raise ValueError(f"{row.get('id')}: invalid Prior v12 label split")
        row["acquisition_split"] = split
        output.append(row)
    return output


def _recompute_materialization(
    protocol: Mapping[str, Any],
    rollout_root: Path,
    *,
    cache_dir: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    raw_path = rollout_root / "rollouts/combined_raw.jsonl"
    raw_rows, raw_sidecar = _read_published_jsonl(
        raw_path, expected_schema="clir-prior-v12-combined-raw-rollouts"
    )
    aliased_raw = _raw_rows_for_shared_materializer(raw_rows)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Prior v12 materialization requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        use_fast=True,
        cache_dir=cache_dir,
    )
    checker = protocol["checker_unitizer"]
    processed, health = materialize_scale_rows(
        aliased_raw,
        tokenizer,
        checker_version=str(checker["checker_version"]),
        unitizer_version=str(checker["unitizer_version"]),
    )
    validation = validate_scale_materialized_rows(
        processed,
        raw_rows=aliased_raw,
        candidate_count=int(generation["candidate_count"]),
        checker_version=str(checker["checker_version"]),
        unitizer_version=str(checker["unitizer_version"]),
    )
    if any(
        row.get("acquisition_split") != row.get("prior_label_split")
        for row in processed
    ):
        raise ValueError("Prior v12 split alias drift after materialization")
    return processed, health, validation, raw_sidecar


def _compact_query(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "source",
        "query_id",
        "source_record_id",
        "question",
        "reference_answer",
        "source_license",
        "source_subject",
        "source_level",
        "reference_reasoning_word_count",
        "reference_calculation_marker_count",
        "reference_distinct_intermediate_numeric_count",
        "cluster_id",
        "cluster_split_priority",
        "query_priority",
        "prior_source_stratum",
        "role",
        "prior_label_split",
        "role_priority",
        "prompt_token_count",
        "prompt_token_ids",
    )
    return {key: row[key] for key in keys if key in row}


def build_plan(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> dict[str, Any]:
    expanded = deepcopy(base)
    expanded["sources"]["math"]["allowed_levels"] = list(
        protocol["sources"]["math"]["allowed_levels"]
    )
    expanded["sources"]["math"]["minimum_official_solution_words"] = int(
        protocol["sources"]["math"]["minimum_official_solution_words"]
    )
    sources, source_report = _load_extended_sources(expanded, cache_dir=cache_dir)
    candidates, source_filter_report = build_source_candidates(
        sources,
        expanded,
        required_schema=str(base["schema_version"]),
    )
    historical, historical_report = _combine_exclusions(expanded)
    parent = protocol["parent"]
    v7_ranking, _ = _read_published_jsonl(
        _project_path(parent["v7_ranking_queries_path"]),
        expected_schema="clir-ranking-v7-evaluation-queries",
    )
    v7_h, _ = _read_published_jsonl(
        _project_path(parent["v7_h_queries_path"]),
        expected_schema="clir-ranking-v7-h-acquisition-queries",
    )
    prior_rows = []
    for record in parent["prior_smoke_proposals"]:
        prior_rows.extend(read_jsonl(_project_path(record["path"])))
    excluded_rows = [*historical, *v7_ranking, *v7_h, *prior_rows]
    excluded_ids = {str(row["query_id"]) for row in excluded_rows}
    by_id = {str(row["query_id"]): row for row in sources}
    missing = sorted(excluded_ids - set(by_id))
    if missing:
        raise ValueError(f"Prior v12 lacks {len(missing)} exclusion anchors")
    anchors = [by_id[query_id] for query_id in sorted(excluded_ids)]
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
        namespace=str(protocol["sources"]["template_clustering"]["namespace"]),
    )
    selected, selection_report = select_acquisition_queries(selectable, protocol)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Prior v12 pre-rollout requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    prompt_counts: list[int] = []
    query_rows: list[dict[str, Any]] = []
    template = str(generation["prompt_template"])
    for raw in selected:
        row = dict(raw)
        content = template.replace("<QUESTION>", str(row["question"]))
        ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        if len(ids) > int(generation["maximum_prompt_tokens"]):
            raise ValueError(f"{row['query_id']}: Prior v12 prompt too long")
        row["prompt_token_ids"] = ids
        row["prompt_token_count"] = len(ids)
        prompt_counts.append(len(ids))
        query_rows.append(_compact_query(row))
    shards = build_acquisition_shards(query_rows, protocol)

    selected_ids = {str(row["query_id"]) for row in query_rows}
    selected_clusters = {str(row["cluster_id"]) for row in query_rows}
    excluded_cluster_ids = {
        str(row["cluster_id"])
        for row in clusters
        if row.get("excluded_by_prior_membership")
    }
    if selected_ids & excluded_ids or selected_clusters & excluded_cluster_ids:
        raise AssertionError("Prior v12 selected query/cluster overlaps an exclusion")
    return {
        "queries": query_rows,
        "shards": shards,
        "reports": {
            "source": source_report,
            "source_filter": source_filter_report,
            "historical_exclusions": historical_report,
            "historical_v6_v7_prior_exclusion_ids": len(excluded_ids),
            "clusters": cluster_report,
            "cluster_rows_recomputed": len(clusters),
            "selection": selection_report,
            "prompt_tokens": {
                "count": len(prompt_counts),
                "min": min(prompt_counts),
                "max": max(prompt_counts),
                "mean": sum(prompt_counts) / len(prompt_counts),
            },
            "query_overlap_with_exclusions": 0,
            "cluster_overlap_with_exclusions": 0,
            "test_files_read": False,
        },
    }


def command_audit(args: argparse.Namespace) -> None:
    base_path = Path(args.base_protocol).resolve()
    protocol_path = Path(args.protocol).resolve()
    base, protocol = load_scale_protocol(protocol_path, base_protocol_path=base_path)
    plan = build_plan(base, protocol, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "status": "PASS_PRIOR_V12_FRESH_SOURCE_CAPACITY_AUDIT",
                "base_protocol_file_sha256": file_sha256(base_path),
                "protocol_file_sha256": file_sha256(protocol_path),
                "query_count": len(plan["queries"]),
                "shard_count": len(plan["shards"]),
                "candidate_rows": sum(
                    int(row["expected_candidate_rows"]) for row in plan["shards"]
                ),
                **plan["reports"],
                "artifacts_written": False,
                "rollout_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_freeze(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v12 freeze requires a clean Git commit")
    base_path = Path(args.base_protocol).resolve()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    if (output / "manifest_registry.json").exists():
        raise FileExistsError("Prior v12 pre-rollout is already frozen")
    base, protocol = load_scale_protocol(protocol_path, base_protocol_path=base_path)
    plan = build_plan(base, protocol, cache_dir=args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "acquisition_queries.jsonl"
    query_manifest = publish_manifest(
        query_path,
        plan["queries"],
        schema_version=QUERY_SCHEMA,
        metadata=plan["reports"]["selection"],
    )
    shards_path = output / "rollout_shards.json"
    atomic_write_json(shards_path, plan["shards"])
    audit_path = output / "source_audit.json"
    atomic_write_json(audit_path, plan["reports"])
    report = {
        "schema_version": "clir-prior-v12-pre-rollout-freeze-report",
        "status": "PASS_PRIOR_V12_PRE_ROLLOUT_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "code_dirty": False,
        "base_protocol_file_sha256": file_sha256(base_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "query_file_sha256": query_manifest["file_sha256"],
        "query_sidecar_file_sha256": file_sha256(
            query_path.with_suffix(query_path.suffix + ".manifest.json")
        ),
        "query_ordered_rows_sha256": query_manifest["ordered_rows_sha256"],
        "query_count": len(plan["queries"]),
        "shards_file_sha256": file_sha256(shards_path),
        "shards_canonical_sha256": canonical_sha256(plan["shards"]),
        "shard_count": len(plan["shards"]),
        "source_audit_file_sha256": file_sha256(audit_path),
        "expected_candidate_rows": sum(
            int(row["expected_candidate_rows"]) for row in plan["shards"]
        ),
        "query_overlap_with_exclusions": 0,
        "cluster_overlap_with_exclusions": 0,
        "rollout_allowed": False,
        "next_gate": "independent_recompute_then_hash_bound_rollout_authorization",
    }
    report_path = output / "freeze_report.json"
    atomic_write_json(report_path, report)
    registry = {
        "schema_version": "clir-prior-v12-pre-rollout-registry",
        "status": report["status"],
        "code_commit": report["code_commit"],
        "protocol_file_sha256": report["protocol_file_sha256"],
        "query_file_sha256": report["query_file_sha256"],
        "query_sidecar_file_sha256": report["query_sidecar_file_sha256"],
        "query_ordered_rows_sha256": report["query_ordered_rows_sha256"],
        "shards_file_sha256": report["shards_file_sha256"],
        "shards_canonical_sha256": report["shards_canonical_sha256"],
        "source_audit_file_sha256": report["source_audit_file_sha256"],
        "freeze_report_file_sha256": file_sha256(report_path),
    }
    registry_path = output / "manifest_registry.json"
    atomic_write_json(registry_path, registry)
    print(
        json.dumps(
            {
                **report,
                "manifest_registry_file_sha256": file_sha256(registry_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify(args: argparse.Namespace) -> None:
    base_path = Path(args.base_protocol).resolve()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    base, protocol = load_scale_protocol(protocol_path, base_protocol_path=base_path)
    registry_path = output / "manifest_registry.json"
    report_path = output / "freeze_report.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("Prior v12 registry protocol hash drift")
    if registry.get("freeze_report_file_sha256") != file_sha256(report_path):
        raise ValueError("Prior v12 freeze report hash drift")
    frozen_queries, sidecar = _read_published_jsonl(
        output / "acquisition_queries.jsonl", expected_schema=QUERY_SCHEMA
    )
    frozen_shards = json.loads(
        (output / "rollout_shards.json").read_text(encoding="utf-8")
    )
    frozen_audit = json.loads((output / "source_audit.json").read_text())
    if sidecar["file_sha256"] != registry["query_file_sha256"]:
        raise ValueError("Prior v12 query manifest drift")
    if file_sha256(output / "rollout_shards.json") != registry["shards_file_sha256"]:
        raise ValueError("Prior v12 shard manifest drift")
    if (
        file_sha256(output / "source_audit.json")
        != registry["source_audit_file_sha256"]
    ):
        raise ValueError("Prior v12 source audit drift")
    recomputed = build_plan(base, protocol, cache_dir=args.cache_dir)
    if recomputed["queries"] != frozen_queries:
        raise ValueError("Prior v12 independent query recomputation drift")
    if recomputed["shards"] != frozen_shards:
        raise ValueError("Prior v12 independent shard recomputation drift")
    if recomputed["reports"] != frozen_audit:
        raise ValueError("Prior v12 independent source audit drift")
    verification = {
        "schema_version": "clir-prior-v12-pre-rollout-verification",
        "status": "PASS_PRIOR_V12_PRE_ROLLOUT_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "freeze_report_file_sha256": file_sha256(report_path),
        "query_count": len(frozen_queries),
        "shard_count": len(frozen_shards),
        "candidate_rows": sum(
            int(row["expected_candidate_rows"]) for row in frozen_shards
        ),
        "query_overlap_with_exclusions": 0,
        "cluster_overlap_with_exclusions": 0,
        "rollout_allowed": False,
        "next_gate": "hash_bound_rollout_authorization",
    }
    verification_path = output / "independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        stable = set(verification) - {"verified_at_utc"}
        if any(old.get(key) != verification[key] for key in stable):
            raise ValueError("Prior v12 existing independent verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(
        json.dumps(
            {
                **verification,
                "verification_file_sha256": file_sha256(verification_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_shard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    protocol_file_sha256: str,
    authorization_file_sha256: str,
    registry_file_sha256: str,
) -> dict[str, Any]:
    candidate_count = int(protocol["generation"]["candidate_count"])
    population = validate_rollout_population(rows, candidate_count=candidate_count)
    expected_query_ids = [str(value) for value in shard["query_ids"]]
    encountered: list[str] = []
    provenance_values: dict[str, set[str]] = {
        key: set()
        for key in (
            "protocol_file_sha256",
            "pre_rollout_registry_file_sha256",
            "authorization_file_sha256",
            "code_commit",
            "model_revision",
            "tokenizer_revision",
            "vllm_version",
        )
    }
    finish_reasons: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    decode_mismatches = 0
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered or encountered[-1] != query_id:
            encountered.append(query_id)
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"{shard['shard_id']}: unknown query {query_id}")
        candidate_index = int(row["candidate_index"])
        if row.get("id") != f"{query_id}:cand:{candidate_index:03d}":
            raise ValueError(f"{query_id}: noncanonical Prior v12 trajectory ID")
        for field in (
            "source",
            "question",
            "reference_answer",
            "cluster_id",
            "prior_label_split",
        ):
            if row.get(field) != query.get(field):
                raise ValueError(f"{row['id']}: frozen field {field} drift")
        if row.get("shard_id") != shard["shard_id"]:
            raise ValueError(f"{row['id']}: Prior v12 shard ID drift")
        if row.get("prompt_token_ids") != query.get("prompt_token_ids"):
            raise ValueError(f"{row['id']}: exact prompt token IDs drift")
        if int(row.get("sampling_seed", -1)) != _derive_query_seed(protocol, query_id):
            raise ValueError(f"{row['id']}: Prior v12 sampling seed drift")
        output_ids = row.get("output_token_ids")
        if not isinstance(output_ids, list) or not output_ids:
            raise ValueError(f"{row['id']}: empty Prior v12 output token IDs")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{row['id']}: missing Prior v12 provenance")
        for key in provenance_values:
            provenance_values[key].add(str(provenance.get(key)))
        finish_reasons[str(row.get("finish_reason"))] += 1
        prompt_lengths.append(len(row["prompt_token_ids"]))
        output_lengths.append(len(output_ids))
        decode_mismatches += row.get("decode_matches_backend_text") is not True
    if encountered != expected_query_ids:
        raise ValueError(f"{shard['shard_id']}: query order differs from freeze")
    if len(rows) != int(shard["expected_candidate_rows"]):
        raise ValueError(f"{shard['shard_id']}: row count mismatch")
    expected_provenance = {
        "protocol_file_sha256": protocol_file_sha256,
        "pre_rollout_registry_file_sha256": registry_file_sha256,
        "authorization_file_sha256": authorization_file_sha256,
        "model_revision": str(protocol["generation"]["model_revision"]),
        "tokenizer_revision": str(protocol["generation"]["tokenizer_revision"]),
        "vllm_version": str(protocol["generation"]["backend_version"]),
    }
    for key, expected in expected_provenance.items():
        if provenance_values[key] != {expected}:
            raise ValueError(
                f"{shard['shard_id']}: provenance {key} mismatch: "
                f"{sorted(provenance_values[key])}"
            )
    if len(provenance_values["code_commit"]) != 1:
        raise ValueError(f"{shard['shard_id']}: mixed code commits")
    return {
        **population,
        "shard_id": shard["shard_id"],
        "query_order_matches_freeze": True,
        "exact_prompt_token_ids_match_freeze": True,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "prompt_token_count": _integer_summary(prompt_lengths),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
        "decode_mismatch_count": decode_mismatches,
        "code_commit": next(iter(provenance_values["code_commit"])),
    }


def verify_rollout_shard(
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    authorization_path: Path,
    pre_rollout_dir: Path,
    rollout_root: Path,
) -> dict[str, Any]:
    output, sidecar_path, completion_path = _shard_paths(rollout_root, shard)
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"missing Prior v12 completion marker: {completion_path}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE_VERIFIED_PRIOR_V12_ROLLOUT_SHARD":
        raise ValueError(f"invalid Prior v12 completion status: {completion_path}")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected = {
        "shard_id": shard["shard_id"],
        "shard_spec_sha256": canonical_sha256(shard),
        "protocol_file_sha256": authorization["frozen_parent"]["protocol_file_sha256"],
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Prior v12 completion binding drift: {completion_path}")
    if not output.is_file() or file_sha256(output) != completion["file_sha256"]:
        raise ValueError(f"Prior v12 shard hash drift: {output}")
    if (
        not sidecar_path.is_file()
        or file_sha256(sidecar_path) != completion["sidecar_file_sha256"]
    ):
        raise ValueError(f"Prior v12 shard sidecar hash drift: {sidecar_path}")
    rows = read_jsonl(output)
    if canonical_sha256(rows) != completion["ordered_rows_sha256"]:
        raise ValueError(f"Prior v12 ordered rows drift: {output}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("file_sha256") != completion["file_sha256"]
        or sidecar.get("ordered_rows_sha256") != completion["ordered_rows_sha256"]
        or int(sidecar.get("row_count", -1)) != int(completion["row_count"])
    ):
        raise ValueError(f"Prior v12 sidecar contents drift: {sidecar_path}")
    validation = _validate_shard_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        protocol_file_sha256=str(completion["protocol_file_sha256"]),
        authorization_file_sha256=file_sha256(authorization_path),
        registry_file_sha256=file_sha256(pre_rollout_dir / "manifest_registry.json"),
    )
    if validation != completion["validation"]:
        raise ValueError(f"Prior v12 validation summary drift: {completion_path}")
    return {
        "status": "PASS_PRIOR_V12_ROLLOUT_SHARD_VERIFY",
        "shard_id": shard["shard_id"],
        "path": str(output),
        "file_sha256": completion["file_sha256"],
        "ordered_rows_sha256": completion["ordered_rows_sha256"],
        "row_count": len(rows),
        "validation": validation,
        "runtime": completion["runtime"],
    }


def command_rollout(args: argparse.Namespace) -> None:
    protocol, authorization, shards, query_by_id, rollout_root = _load_rollout_contract(
        args
    )
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.output).resolve()
    protocol_path = Path(args.protocol).resolve()
    shard = _select_shard(shards, args.shard_id)
    calibration_id = str(authorization["runtime_contract"]["first_calibration_shard"])
    if shard["shard_id"] != calibration_id:
        calibration = _select_shard(shards, calibration_id)
        _, _, marker = _shard_paths(rollout_root, calibration)
        if not marker.exists():
            raise RuntimeError(f"{calibration_id} must verify before other shards")
        verify_rollout_shard(
            shard=calibration,
            query_by_id=query_by_id,
            protocol=protocol,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout,
            rollout_root=rollout_root,
        )
    output, sidecar_path, completion_path = _shard_paths(rollout_root, shard)
    if completion_path.exists():
        print(
            json.dumps(
                verify_rollout_shard(
                    shard=shard,
                    query_by_id=query_by_id,
                    protocol=protocol,
                    authorization_path=authorization_path,
                    pre_rollout_dir=pre_rollout,
                    rollout_root=rollout_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output.exists() or sidecar_path.exists():
        raise FileExistsError(
            f"{shard['shard_id']} has incomplete artifacts; overwrite is forbidden"
        )
    if _git_dirty():
        raise RuntimeError("Prior v12 rollout requires a clean Git commit")
    runtime = authorization["runtime_contract"]
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Prior v12 rollout requires torch and vLLM") from exc
    generation = protocol["generation"]
    if _package_version("vllm") != generation["backend_version"]:
        raise ValueError("Prior v12 installed vLLM version drift")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each Prior v12 shard process must see exactly one GPU")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < int(runtime["minimum_free_gpu_bytes"]):
        raise RuntimeError("visible GPU has less free memory than the frozen minimum")

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
    rendered_prompts: list[str] = []
    expected_prompt_ids: list[list[int]] = []
    sampling: list[Any] = []
    candidate_count = int(generation["candidate_count"])
    for query in query_rows:
        user_prompt = _prompt_for(
            str(query["question"]), str(generation["prompt_template"])
        )
        messages = [{"role": "user", "content": user_prompt}]
        rendered_prompts.append(
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
        if prompt_ids != query["prompt_token_ids"]:
            raise ValueError(f"{query['query_id']}: frozen prompt IDs drift")
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
    request_outputs = llm.generate(rendered_prompts, sampling, use_tqdm=True)
    if len(request_outputs) != len(query_rows):
        raise ValueError("vLLM returned a different Prior v12 query count")
    finished_at = _utc_now()
    elapsed = time.monotonic() - started_clock
    provenance = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "shard_spec_sha256": canonical_sha256(shard),
        "code_commit": _git_head(),
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
    for query, request_output, expected_ids in zip(
        query_rows, request_outputs, expected_prompt_ids
    ):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        if prompt_ids != expected_ids:
            raise ValueError(f"{query['query_id']}: vLLM prompt IDs drift")
        for candidate in _ordered_vllm_candidates(request_output, candidate_count):
            output_ids = [int(value) for value in candidate.token_ids]
            if not output_ids:
                raise ValueError(f"{query['query_id']}: empty Prior v12 output")
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            candidate_index = int(candidate.index)
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{candidate_index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": candidate_index,
                    "shard_id": shard["shard_id"],
                    "role": query["role"],
                    "prior_label_split": query["prior_label_split"],
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
    validation = _validate_shard_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        protocol_file_sha256=file_sha256(protocol_path),
        authorization_file_sha256=file_sha256(authorization_path),
        registry_file_sha256=file_sha256(pre_rollout / "manifest_registry.json"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-prior-v12-raw-rollout-shard",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-prior-v12-rollout-shard-completion",
        "status": "COMPLETE_VERIFIED_PRIOR_V12_ROLLOUT_SHARD",
        "shard_id": shard["shard_id"],
        "shard_spec_sha256": canonical_sha256(shard),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar_path),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "row_count": manifest["row_count"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": _git_head(),
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
            verify_rollout_shard(
                shard=shard,
                query_by_id=query_by_id,
                protocol=protocol,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout,
                rollout_root=rollout_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify_rollouts(args: argparse.Namespace) -> None:
    protocol, _, shards, query_by_id, rollout_root = _load_rollout_contract(args)
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.output).resolve()
    requested = set(args.shard_id or [])
    if requested:
        unknown = requested - {str(row["shard_id"]) for row in shards}
        if unknown:
            raise ValueError(f"unknown Prior v12 shard IDs: {sorted(unknown)}")
        shards = [row for row in shards if row["shard_id"] in requested]
    completed: list[dict[str, Any]] = []
    missing: list[str] = []
    for shard in shards:
        _, _, marker = _shard_paths(rollout_root, shard)
        if not marker.exists():
            missing.append(str(shard["shard_id"]))
            continue
        completed.append(
            verify_rollout_shard(
                shard=shard,
                query_by_id=query_by_id,
                protocol=protocol,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout,
                rollout_root=rollout_root,
            )
        )
    if args.require_complete and missing:
        raise ValueError(f"missing {len(missing)} Prior v12 shards: {missing}")
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ALL_PRIOR_V12_ROLLOUT_SHARDS"
                    if not missing and len(completed) == len(shards)
                    else "PARTIAL_PRIOR_V12_ROLLOUT_SHARDS"
                ),
                "verified_shards": len(completed),
                "verified_rows": sum(row["row_count"] for row in completed),
                "missing_shards": missing,
                "shards": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_merge_rollouts(args: argparse.Namespace) -> None:
    protocol, _, shards, query_by_id, rollout_root = _load_rollout_contract(args)
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.output).resolve()
    protocol_path = Path(args.protocol).resolve()
    combined_path = rollout_root / "rollouts/combined_raw.jsonl"
    sidecar_path = combined_path.with_suffix(combined_path.suffix + ".manifest.json")
    report_path = rollout_root / "rollout_completion_report.json"
    if combined_path.exists() or sidecar_path.exists() or report_path.exists():
        raise FileExistsError("Prior v12 combined rollout artifacts already exist")
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for shard in shards:
        report = verify_rollout_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout,
            rollout_root=rollout_root,
        )
        reports.append(report)
        rows.extend(read_jsonl(Path(report["path"])))
    population = validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    if len(rows) != int(protocol["generation"]["expected_candidate_rows"]):
        raise ValueError("Prior v12 combined row count drift")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prior v12 shards overlap by trajectory ID")
    expected_queries = [
        str(query_id) for shard in shards for query_id in shard["query_ids"]
    ]
    encountered_queries: list[str] = []
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered_queries or encountered_queries[-1] != query_id:
            encountered_queries.append(query_id)
    if encountered_queries != expected_queries:
        raise ValueError("Prior v12 combined query order differs from freeze")
    commits = {str(row["provenance"]["code_commit"]) for row in rows}
    if len(commits) != 1:
        raise ValueError("Prior v12 shards use mixed code commits")
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    output_lengths = [len(row["output_token_ids"]) for row in rows]
    metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": next(iter(commits)),
        "shard_count": len(shards),
        "shard_file_sha256": {row["shard_id"]: row["file_sha256"] for row in reports},
        **population,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
    }
    manifest = publish_manifest(
        combined_path,
        rows,
        schema_version="clir-prior-v12-combined-raw-rollouts",
        metadata=metadata,
    )
    report = {
        "schema_version": "clir-prior-v12-rollout-completion-report",
        "status": "PASS_ALL_16000_PRIOR_V12_RAW_ROLLOUTS",
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": next(iter(commits)),
        "combined_file_sha256": manifest["file_sha256"],
        "combined_sidecar_file_sha256": file_sha256(sidecar_path),
        "combined_ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "population": population,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
        "shards": reports,
        "next_gate": "hash_bound_checker_and_unitizer_materialization_authorization",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_materialize(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v12 materialization requires a clean Git commit")
    protocol, _, rollout_root, pre_annotation_root = _load_pre_annotation_contract(args)
    output = pre_annotation_root / "materialized/all_rows.jsonl"
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    report_path = pre_annotation_root / "materialized/materialization_report.json"
    if output.exists() or sidecar.exists() or report_path.exists():
        raise FileExistsError("Prior v12 materialized artifacts already exist")
    processed, health, validation, raw_sidecar = _recompute_materialization(
        protocol,
        rollout_root,
        cache_dir=args.cache_dir,
    )
    protocol_path = Path(args.protocol).resolve()
    preauth_path = Path(args.pre_annotation_authorization).resolve()
    manifest = publish_manifest(
        output,
        processed,
        schema_version="clir-prior-v12-materialized-rollouts",
        metadata={
            "protocol_file_sha256": file_sha256(protocol_path),
            "pre_annotation_authorization_file_sha256": file_sha256(preauth_path),
            "raw_file_sha256": raw_sidecar["file_sha256"],
            "code_commit": _git_head(),
            "split_alias": "acquisition_split=prior_label_split",
            **health,
        },
    )
    report = {
        "schema_version": "clir-prior-v12-materialization-report",
        "status": "PASS_PRIOR_V12_CHECKER_UNITIZER_MATERIALIZATION",
        "rows": len(processed),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_annotation_authorization_file_sha256": file_sha256(preauth_path),
        "rollout_completion_report_file_sha256": file_sha256(
            rollout_root / "rollout_completion_report.json"
        ),
        "code_commit": _git_head(),
        "health": health,
        "validation": validation,
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "next_gate": "independent_recompute_then_freeze_exact_800_natural_proposals",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_materialized(args: argparse.Namespace) -> None:
    protocol, _, rollout_root, pre_annotation_root = _load_pre_annotation_contract(args)
    output = pre_annotation_root / "materialized/all_rows.jsonl"
    report_path = pre_annotation_root / "materialized/materialization_report.json"
    frozen, sidecar = _read_published_jsonl(
        output, expected_schema="clir-prior-v12-materialized-rollouts"
    )
    published = json.loads(report_path.read_text(encoding="utf-8"))
    recomputed, health, validation, _ = _recompute_materialization(
        protocol,
        rollout_root,
        cache_dir=args.cache_dir,
    )
    if canonical_sha256(recomputed) != canonical_sha256(frozen):
        raise ValueError("Prior v12 independent materialization recomputation drift")
    normalized_health = json.loads(json.dumps(health, ensure_ascii=False))
    normalized_validation = json.loads(json.dumps(validation, ensure_ascii=False))
    if canonical_sha256(normalized_health) != canonical_sha256(
        published.get("health")
    ) or canonical_sha256(normalized_validation) != canonical_sha256(
        published.get("validation")
    ):
        raise ValueError("Prior v12 materialization summary drift")
    verification = {
        "schema_version": "clir-prior-v12-materialization-verification",
        "status": "PASS_PRIOR_V12_MATERIALIZATION_INDEPENDENT_RECOMPUTE",
        "materialized_file_sha256": sidecar["file_sha256"],
        "materialized_sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "materialization_report_file_sha256": file_sha256(report_path),
        "rows": len(frozen),
        "health": normalized_health,
        "validation": normalized_validation,
        "next_gate": "freeze_exact_800_natural_proposals",
    }
    verification_path = (
        pre_annotation_root / "materialized/independent_verification.json"
    )
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        if old != verification:
            raise ValueError("Prior v12 materialization verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


def _recompute_proposals(
    protocol: Mapping[str, Any], pre_annotation_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    materialized_path = pre_annotation_root / "materialized/all_rows.jsonl"
    rows, sidecar = _read_published_jsonl(
        materialized_path, expected_schema="clir-prior-v12-materialized-rollouts"
    )
    verification_path = (
        pre_annotation_root / "materialized/independent_verification.json"
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != (
        "PASS_PRIOR_V12_MATERIALIZATION_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("Prior v12 materialization verification is not PASS")
    proposals, report = select_prior_proposals(rows, protocol)
    return proposals, report, sidecar


def command_select_proposals(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v12 proposal freeze requires a clean Git commit")
    protocol, _, _, pre_annotation_root = _load_pre_annotation_contract(args)
    output = pre_annotation_root / "proposals/prior_natural_800.jsonl"
    sidecar_path = output.with_suffix(output.suffix + ".manifest.json")
    report_path = pre_annotation_root / "proposals/proposal_report.json"
    if output.exists() or sidecar_path.exists() or report_path.exists():
        raise FileExistsError("Prior v12 proposal artifacts already exist")
    proposals, selection, materialized_sidecar = _recompute_proposals(
        protocol, pre_annotation_root
    )
    manifest = publish_manifest(
        output,
        proposals,
        schema_version=PROPOSAL_SCHEMA,
        metadata={
            "protocol_file_sha256": file_sha256(Path(args.protocol).resolve()),
            "pre_annotation_authorization_file_sha256": file_sha256(
                Path(args.pre_annotation_authorization).resolve()
            ),
            "materialized_file_sha256": materialized_sidecar["file_sha256"],
            "code_commit": _git_head(),
            **selection,
        },
    )
    report = {
        "schema_version": "clir-prior-v12-proposal-report",
        "status": "PASS_PRIOR_V12_EXACT_800_NATURAL_PROPOSALS",
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar_path),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "code_commit": _git_head(),
        "selection": selection,
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "next_gate": "independent_recompute_then_construct_blind_annotation_shards",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_proposals(args: argparse.Namespace) -> None:
    protocol, _, _, pre_annotation_root = _load_pre_annotation_contract(args)
    output = pre_annotation_root / "proposals/prior_natural_800.jsonl"
    frozen, sidecar = _read_published_jsonl(output, expected_schema=PROPOSAL_SCHEMA)
    recomputed, selection, _ = _recompute_proposals(protocol, pre_annotation_root)
    if canonical_sha256(recomputed) != canonical_sha256(frozen):
        raise ValueError("Prior v12 proposal independent recomputation drift")
    report_path = pre_annotation_root / "proposals/proposal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selection") != selection:
        raise ValueError("Prior v12 proposal report drift")
    verification = {
        "schema_version": "clir-prior-v12-proposal-verification",
        "status": "PASS_PRIOR_V12_PROPOSAL_INDEPENDENT_RECOMPUTE",
        "proposal_file_sha256": sidecar["file_sha256"],
        "proposal_sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "proposal_report_file_sha256": file_sha256(report_path),
        "natural_rows": len(frozen),
        "selection": selection,
        "next_gate": "construct_blind_annotation_shards",
    }
    verification_path = pre_annotation_root / "proposals/independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        if old != verification:
            raise ValueError("Prior v12 proposal verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


def _annotation_package_path(
    pre_annotation_root: Path, annotator: str, shard_index: int
) -> Path:
    return (
        pre_annotation_root
        / f"packages/annotator_{annotator}/prior_v12_{annotator}_{shard_index:02d}.jsonl"
    )


def _assert_no_annotation_labels(pre_annotation_root: Path) -> None:
    existing = []
    for directory_name in ("labels_a", "labels_b"):
        directory = pre_annotation_root / directory_name
        if directory.exists():
            existing.extend(path for path in directory.rglob("*") if path.is_file())
    if existing:
        raise ValueError(
            "Prior v12 package readiness requires zero annotation labels: "
            f"found {len(existing)} files"
        )


def _recompute_annotation_packages(
    protocol: Mapping[str, Any], pre_annotation_root: Path
) -> tuple[
    dict[str, list[list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    proposal_path = pre_annotation_root / "proposals/prior_natural_800.jsonl"
    proposals, proposal_sidecar = _read_published_jsonl(
        proposal_path, expected_schema=PROPOSAL_SCHEMA
    )
    verification_path = pre_annotation_root / "proposals/independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != ("PASS_PRIOR_V12_PROPOSAL_INDEPENDENT_RECOMPUTE"):
        raise ValueError("Prior v12 proposal verification is not PASS")
    if verification.get("proposal_file_sha256") != proposal_sidecar["file_sha256"]:
        raise ValueError("Prior v12 proposal verification binding drift")
    packages, private, construction = build_prior_annotation_shards(proposals, protocol)
    return packages, private, construction, proposal_sidecar


def command_build_packages(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v12 package construction requires a clean Git commit")
    protocol, _, _, pre_annotation_root = _load_pre_annotation_contract(args)
    _assert_no_annotation_labels(pre_annotation_root)
    package_root = pre_annotation_root / "packages"
    report_path = package_root / "package_report.json"
    verification_path = package_root / "independent_verification.json"
    private_path = package_root / "PRIVATE_package_index.jsonl"
    if package_root.exists() and any(package_root.rglob("*")):
        raise FileExistsError("Prior v12 annotation package artifacts already exist")
    for required in (
        DEFAULT_ANNOTATION_PROMPT,
        DEFAULT_LAUNCH_PROMPT_A,
        DEFAULT_LAUNCH_PROMPT_B,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    packages, private, construction, proposal_sidecar = _recompute_annotation_packages(
        protocol, pre_annotation_root
    )
    code_commit = _git_head()
    protocol_path = Path(args.protocol).resolve()
    metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "proposal_file_sha256": proposal_sidecar["file_sha256"],
        "proposal_ordered_rows_sha256": proposal_sidecar["ordered_rows_sha256"],
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_ANNOTATION_PROMPT),
        "code_commit": code_commit,
        "labels_are_gold": False,
        "human_verification": False,
    }
    manifests: dict[str, dict[str, Any]] = {}
    for annotator in ("a", "b"):
        for shard_index, rows in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            manifests[key] = publish_manifest(
                _annotation_package_path(pre_annotation_root, annotator, shard_index),
                rows,
                schema_version=PACKAGE_SCHEMA,
                metadata={**metadata, "annotation_shard_id": key},
            )
    private_manifest = publish_manifest(
        private_path,
        private,
        schema_version=PRIVATE_SCHEMA,
        metadata={
            **metadata,
            "visibility": "PRIVATE_NEVER_SEND_TO_ANNOTATORS",
        },
    )
    report = {
        "schema_version": "clir-prior-v12-package-report",
        "status": "PASS_PRIOR_V12_BLIND_ANNOTATION_SHARDS_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_annotation_authorization_file_sha256": file_sha256(
            Path(args.pre_annotation_authorization).resolve()
        ),
        "proposal_file_sha256": proposal_sidecar["file_sha256"],
        "proposal_verification_file_sha256": file_sha256(
            pre_annotation_root / "proposals/independent_verification.json"
        ),
        "annotation_prompt": {
            "path": str(DEFAULT_ANNOTATION_PROMPT.resolve()),
            "file_sha256": file_sha256(DEFAULT_ANNOTATION_PROMPT),
        },
        "launch_prompts": {
            "a_file_sha256": file_sha256(DEFAULT_LAUNCH_PROMPT_A),
            "b_file_sha256": file_sha256(DEFAULT_LAUNCH_PROMPT_B),
        },
        "code_commit": code_commit,
        "construction": construction,
        "public_shards": {
            key: {
                "path": manifest["path"],
                "row_count": manifest["row_count"],
                "file_sha256": manifest["file_sha256"],
                "sidecar_file_sha256": file_sha256(
                    Path(manifest["path"]).with_suffix(
                        Path(manifest["path"]).suffix + ".manifest.json"
                    )
                ),
                "ordered_rows_sha256": manifest["ordered_rows_sha256"],
            }
            for key, manifest in sorted(manifests.items())
        },
        "private_index": {
            "path": private_manifest["path"],
            "row_count": private_manifest["row_count"],
            "file_sha256": private_manifest["file_sha256"],
            "sidecar_file_sha256": file_sha256(
                private_path.with_suffix(private_path.suffix + ".manifest.json")
            ),
            "ordered_rows_sha256": private_manifest["ordered_rows_sha256"],
        },
        "natural_rows_per_annotator": int(protocol["proposal_pool"]["natural_count"]),
        "rows_per_annotator": sum(len(rows) for rows in packages["a"]),
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "claim_boundary": (
            "blind package readiness only; no label accuracy, Prior learnability, "
            "gate efficacy, Best-of-N, or Full evidence"
        ),
        "next_gate": "independent_package_recompute_then_user_launches_two_blind_annotators",
    }
    atomic_write_json(report_path, report)
    if verification_path.exists():
        raise FileExistsError("Prior v12 package verification unexpectedly exists")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_packages(args: argparse.Namespace) -> None:
    protocol, _, _, pre_annotation_root = _load_pre_annotation_contract(args)
    _assert_no_annotation_labels(pre_annotation_root)
    packages, private, construction, proposal_sidecar = _recompute_annotation_packages(
        protocol, pre_annotation_root
    )
    package_root = pre_annotation_root / "packages"
    report_path = package_root / "package_report.json"
    published = json.loads(report_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    expected_public_paths: set[Path] = set()
    public_hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        for shard_index, expected_rows in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            path = _annotation_package_path(pre_annotation_root, annotator, shard_index)
            expected_public_paths.add(path.resolve())
            actual_rows, sidecar = _read_published_jsonl(
                path, expected_schema=PACKAGE_SCHEMA
            )
            if canonical_sha256(actual_rows) != canonical_sha256(expected_rows):
                mismatches.append(key)
            if any(
                set(row)
                != {"schema_version", "item_id", "question", "response", "units"}
                for row in actual_rows
            ):
                mismatches.append(f"{key}:public_field_leak")
            public_hashes[key] = sidecar["file_sha256"]
    actual_public_paths = {
        path.resolve()
        for annotator in ("a", "b")
        for path in (package_root / f"annotator_{annotator}").glob("*.jsonl")
    }
    if actual_public_paths != expected_public_paths:
        mismatches.append("public_file_population")
    private_path = package_root / "PRIVATE_package_index.jsonl"
    actual_private, private_sidecar = _read_published_jsonl(
        private_path, expected_schema=PRIVATE_SCHEMA
    )
    if canonical_sha256(actual_private) != canonical_sha256(private):
        mismatches.append("private_index")
    expected_report_bindings = {
        "status": "PASS_PRIOR_V12_BLIND_ANNOTATION_SHARDS_READY",
        "protocol_file_sha256": file_sha256(Path(args.protocol).resolve()),
        "proposal_file_sha256": proposal_sidecar["file_sha256"],
        "construction": construction,
    }
    for key, expected in expected_report_bindings.items():
        if published.get(key) != expected:
            mismatches.append(f"report:{key}")
    if published.get("annotation_prompt", {}).get("file_sha256") != file_sha256(
        DEFAULT_ANNOTATION_PROMPT
    ):
        mismatches.append("annotation_prompt_hash")
    expected_launch_hashes = {
        "a_file_sha256": file_sha256(DEFAULT_LAUNCH_PROMPT_A),
        "b_file_sha256": file_sha256(DEFAULT_LAUNCH_PROMPT_B),
    }
    if published.get("launch_prompts") != expected_launch_hashes:
        mismatches.append("launch_prompt_hashes")
    for key, actual_hash in public_hashes.items():
        if (
            published.get("public_shards", {}).get(key, {}).get("file_sha256")
            != actual_hash
        ):
            mismatches.append(f"report:{key}:file_sha256")
    if (
        published.get("private_index", {}).get("file_sha256")
        != private_sidecar["file_sha256"]
    ):
        mismatches.append("report:private_index:file_sha256")
    verification = {
        "schema_version": "clir-prior-v12-package-verification",
        "status": (
            "PASS_PRIOR_V12_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V12_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(Path(args.protocol).resolve()),
        "proposal_file_sha256": proposal_sidecar["file_sha256"],
        "package_report_file_sha256": file_sha256(report_path),
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_ANNOTATION_PROMPT),
        "published_code_commit": published.get("code_commit"),
        "verified_at_code_commit": _git_head(),
        "public_shards": len(public_hashes),
        "rows_per_public_shard": int(construction["rows_per_shard"]),
        "public_rows_total": sum(
            len(rows) for annotator in ("a", "b") for rows in packages[annotator]
        ),
        "private_rows": len(private),
        "labels_present": False,
        "next_gate": "user_runs_annotator_a_and_b_in_separate_blind_contexts",
    }
    verification_path = package_root / "independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        if old != verification:
            raise ValueError("Prior v12 package verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def _annotation_label_path(
    pre_annotation_root: Path, annotator: str, shard_index: int
) -> Path:
    return (
        pre_annotation_root
        / f"labels_{annotator}/prior_v12_{annotator}_{shard_index:02d}.jsonl"
    )


def _load_label_evaluation_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, _, _, pre_annotation_root = _load_pre_annotation_contract(args)
    package_root = pre_annotation_root / "packages"
    package_report_path = package_root / "package_report.json"
    package_verification_path = package_root / "independent_verification.json"
    package_report = json.loads(package_report_path.read_text(encoding="utf-8"))
    package_verification = json.loads(
        package_verification_path.read_text(encoding="utf-8")
    )
    if package_report.get("status") != "PASS_PRIOR_V12_BLIND_ANNOTATION_SHARDS_READY":
        raise ValueError("Prior v12 package report is not PASS")
    if package_verification.get("status") != (
        "PASS_PRIOR_V12_PACKAGE_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("Prior v12 package verification is not PASS")
    if package_verification.get("package_report_file_sha256") != file_sha256(
        package_report_path
    ):
        raise ValueError("Prior v12 package report verification binding drift")

    proposal_path = pre_annotation_root / "proposals/prior_natural_800.jsonl"
    proposals, proposal_sidecar = _read_published_jsonl(
        proposal_path, expected_schema=PROPOSAL_SCHEMA
    )
    private_path = package_root / "PRIVATE_package_index.jsonl"
    private_index, private_sidecar = _read_published_jsonl(
        private_path, expected_schema=PRIVATE_SCHEMA
    )
    if package_report.get("proposal_file_sha256") != proposal_sidecar["file_sha256"]:
        raise ValueError("Prior v12 package/proposal hash binding drift")
    if (
        package_report.get("private_index", {}).get("file_sha256")
        != private_sidecar["file_sha256"]
    ):
        raise ValueError("Prior v12 package/private-index hash binding drift")

    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    bindings: dict[str, Any] = {
        "protocol_file_sha256": file_sha256(Path(args.protocol).resolve()),
        "package_report_file_sha256": file_sha256(package_report_path),
        "package_verification_file_sha256": file_sha256(package_verification_path),
        "proposal_file_sha256": proposal_sidecar["file_sha256"],
        "private_index_file_sha256": private_sidecar["file_sha256"],
        "package_shards": {},
        "label_shards": {},
    }
    shard_count = int(protocol["annotation"]["natural_shards_per_annotator"])
    expected_rows = (
        int(protocol["annotation"]["natural_rows_per_shard"])
        + 1
        + int(protocol["annotation"]["self_repeats_total_per_annotator"]) // shard_count
    )
    for annotator in ("a", "b"):
        label_directory = pre_annotation_root / f"labels_{annotator}"
        expected_label_paths = {
            _annotation_label_path(
                pre_annotation_root, annotator, shard_index
            ).resolve()
            for shard_index in range(shard_count)
        }
        actual_label_paths = {
            path.resolve() for path in label_directory.glob("*.jsonl")
        }
        if actual_label_paths != expected_label_paths:
            missing = sorted(
                str(path) for path in expected_label_paths - actual_label_paths
            )
            extra = sorted(
                str(path) for path in actual_label_paths - expected_label_paths
            )
            raise ValueError(
                f"Prior v12 labels {annotator} shard population differs: "
                f"missing={missing}, extra={extra}"
            )
        for shard_index in range(shard_count):
            shard_id = f"{annotator}-{shard_index:02d}"
            package_path = _annotation_package_path(
                pre_annotation_root, annotator, shard_index
            )
            package_rows, package_sidecar = _read_published_jsonl(
                package_path, expected_schema=PACKAGE_SCHEMA
            )
            expected_hash = (
                package_report.get("public_shards", {})
                .get(shard_id, {})
                .get("file_sha256")
            )
            if expected_hash != package_sidecar["file_sha256"]:
                raise ValueError(f"Prior v12 package shard hash drift: {shard_id}")
            label_path = _annotation_label_path(
                pre_annotation_root, annotator, shard_index
            )
            label_rows = read_jsonl(label_path)
            if len(package_rows) != expected_rows or len(label_rows) != expected_rows:
                raise ValueError(
                    f"Prior v12 shard {shard_id} row count drift: "
                    f"package={len(package_rows)}, labels={len(label_rows)}, "
                    f"expected={expected_rows}"
                )
            package_ids = [str(row.get("item_id")) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(
                package_ids
            ):
                raise ValueError(
                    f"Prior v12 label/package ID binding drift: {shard_id}"
                )
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
            bindings["package_shards"][shard_id] = {
                "rows": len(package_rows),
                "file_sha256": package_sidecar["file_sha256"],
                "ordered_rows_sha256": canonical_sha256(package_rows),
            }
            bindings["label_shards"][shard_id] = {
                "rows": len(label_rows),
                "file_sha256": file_sha256(label_path),
                "ordered_rows_sha256": canonical_sha256(label_rows),
            }
    bindings["label_population_ordered_rows_sha256"] = {
        annotator: canonical_sha256(labels[annotator]) for annotator in ("a", "b")
    }
    return {
        "proposals": proposals,
        "package_a": packages["a"],
        "package_b": packages["b"],
        "private_index": private_index,
        "labels_a": labels["a"],
        "labels_b": labels["b"],
        "protocol": protocol,
    }, bindings


def _build_label_evaluation_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs, bindings = _load_label_evaluation_inputs(args)
    report = evaluate_prior_v12_labels(**inputs)
    report["bindings"] = bindings
    report["annotators"] = {
        "a": "user_reported_gpt_5_6_sol_xhigh",
        "b": "user_reported_claude_opus_5_high",
        "exact_revisions_and_temperatures_verified": False,
    }
    report["evaluated_at_code_commit"] = _git_head()
    return report


def command_evaluate_labels(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError(
            "Prior v12 first label evaluation requires a clean committed evaluator"
        )
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    evaluation_root = pre_annotation_root / "evaluation"
    report_path = evaluation_root / "raw_gate_report.json"
    verification_path = evaluation_root / "independent_verification.json"
    if report_path.exists() or verification_path.exists():
        raise FileExistsError(
            "Prior v12 label evaluation already exists; run verify-label-evaluation"
        )
    report = _build_label_evaluation_report(args)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_label_evaluation(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v12 label evaluation verification requires clean Git")
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    evaluation_root = pre_annotation_root / "evaluation"
    report_path = evaluation_root / "raw_gate_report.json"
    verification_path = evaluation_root / "independent_verification.json"
    published = json.loads(report_path.read_text(encoding="utf-8"))
    recomputed = _build_label_evaluation_report(args)
    matches = canonical_sha256(published) == canonical_sha256(recomputed)
    verification = {
        "schema_version": "clir-prior-v12-label-evaluation-verification",
        "status": (
            "PASS_PRIOR_V12_LABEL_EVALUATION_INDEPENDENT_RECOMPUTE"
            if matches
            else "FAIL_PRIOR_V12_LABEL_EVALUATION_INDEPENDENT_RECOMPUTE"
        ),
        "published_report_file_sha256": file_sha256(report_path),
        "published_report_canonical_sha256": canonical_sha256(published),
        "recomputed_report_canonical_sha256": canonical_sha256(recomputed),
        "evaluated_at_code_commit": published.get("evaluated_at_code_commit"),
        "verified_at_code_commit": _git_head(),
        "label_shards": len(published.get("bindings", {}).get("label_shards", {})),
        "terminal_or_next_gate": published.get("next_gate"),
    }
    if verification_path.exists():
        previous = json.loads(verification_path.read_text(encoding="utf-8"))
        if previous != verification:
            raise ValueError("Prior v12 label evaluation verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if not matches:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", default=str(DEFAULT_BASE_PROTOCOL))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument(
        "--pre-annotation-authorization",
        default=str(DEFAULT_PRE_ANNOTATION_AUTHORIZATION),
    )
    parser.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--cache-dir", default="run_artifacts/model_cache")
    audit.set_defaults(func=command_audit)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--cache-dir", default="run_artifacts/model_cache")
    freeze.set_defaults(func=command_freeze)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--cache-dir", default="run_artifacts/model_cache")
    verify.set_defaults(func=command_verify)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify_rollouts = subparsers.add_parser("verify-rollouts")
    verify_rollouts.add_argument("--shard-id", action="append")
    verify_rollouts.add_argument("--require-complete", action="store_true")
    verify_rollouts.set_defaults(func=command_verify_rollouts)
    merge = subparsers.add_parser("merge-rollouts")
    merge.set_defaults(func=command_merge_rollouts)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--cache-dir", default="run_artifacts/model_cache")
    materialize.set_defaults(func=command_materialize)
    verify_materialized = subparsers.add_parser("verify-materialized")
    verify_materialized.add_argument("--cache-dir", default="run_artifacts/model_cache")
    verify_materialized.set_defaults(func=command_verify_materialized)
    proposals = subparsers.add_parser("select-proposals")
    proposals.set_defaults(func=command_select_proposals)
    verify_proposals = subparsers.add_parser("verify-proposals")
    verify_proposals.set_defaults(func=command_verify_proposals)
    packages = subparsers.add_parser("build-packages")
    packages.set_defaults(func=command_build_packages)
    verify_packages = subparsers.add_parser("verify-packages")
    verify_packages.set_defaults(func=command_verify_packages)
    evaluate_labels = subparsers.add_parser("evaluate-labels")
    evaluate_labels.set_defaults(func=command_evaluate_labels)
    verify_label_evaluation = subparsers.add_parser("verify-label-evaluation")
    verify_label_evaluation.set_defaults(func=command_verify_label_evaluation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
