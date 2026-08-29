#!/usr/bin/env python
"""Freeze and verify CLIR ranking/H expansion-v7 pre-rollout manifests.

This stage is intentionally model-free.  It reads pinned train-only sources,
extends the already audited numeric MATH inventory with three unused subjects,
propagates every historical/v6 exclusion through template clusters, and freezes
disjoint ranking-evaluation and H-acquisition query populations.  Rollout,
annotation, feature extraction, and training require later authorization files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from src.clir_h_expansion import (
    H_PACKAGE_SCHEMA,
    H_PROPOSAL_SCHEMA,
    build_h_annotation_packages,
    build_h_proposals,
    evaluate_h_package_labels,
    smoke_gate,
    split_smoke_and_reserve,
)
from src.clir_ranking_scale import (
    H_ROLE,
    RANKING_ROLE,
    RANKING_V7_SCHEMA,
    build_role_manifests,
    build_rollout_shards,
    compute_budget,
)
from src.clir_scale import (
    build_source_candidates,
    build_template_clusters,
)
from src.clir_smoke import (
    UNITIZER_VERSION,
    atomic_write_json,
    canonical_sha256,
    check_numeric_response,
    extract_math_numeric_reference,
    file_sha256,
    publish_manifest,
    read_jsonl,
    stable_priority,
    validate_rollout_population,
)
from src.clir_scale_pre_annotation import (
    materialize_scale_rows,
    validate_scale_materialized_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_rollout"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/rollout_authorization.json"
)
DEFAULT_ROLLOUT_ROOT = PROJECT_ROOT / "run_artifacts/ranking_expansion_v7"
DEFAULT_PRE_ANNOTATION_AUTHORIZATION = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/pre_annotation_authorization.json"
)
DEFAULT_PRE_ANNOTATION_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_annotation"
)
REQUIRED_FILES = (
    "source_inventory.json",
    "permanent_exclusions.jsonl",
    "template_clusters.jsonl",
    "ranking_queries.jsonl",
    "h_acquisition_queries.jsonl",
    "rollout_shards.json",
)


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


def _prompt_for(question: str, template: str) -> str:
    if "<QUESTION>" not in template:
        raise ValueError("generation prompt template lacks <QUESTION>")
    return template.replace("<QUESTION>", question)


def _derive_query_seed(base_seed: int, query_id: str) -> int:
    return int(
        stable_priority("clir-ranking-v7-vllm-query-seed", base_seed, query_id)[:16],
        16,
    ) % (2**31)


def _ordered_vllm_candidates(request_output: Any, expected_count: int) -> list[Any]:
    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "vLLM candidate indices must be unique and contiguous: "
            f"expected 0..{expected_count - 1}, got {sorted(indices)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate.index))


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


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != RANKING_V7_SCHEMA:
        raise ValueError("prepare_clir_ranking supports only ranking/H v7")
    if protocol.get("status") != "FROZEN_PREPARATION_ROLLOUT_NOT_STARTED":
        raise ValueError("ranking/H v7 is not at its pre-rollout gate")
    authorization = protocol.get("execution_authorization", {})
    if authorization.get("pre_rollout_freeze_allowed") is not True:
        raise ValueError("protocol does not permit pre-rollout freezing")
    for forbidden in (
        "rollout_allowed",
        "annotation_allowed",
        "feature_extraction_allowed",
        "training_allowed",
    ):
        if authorization.get(forbidden) is not False:
            raise ValueError(f"pre-rollout protocol must keep {forbidden}=false")
    return protocol


def _verify_jsonl_input(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _project_path(str(config["path"]))
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if file_sha256(path) != config["file_sha256"]:
        raise ValueError(f"pinned JSONL file hash mismatch: {path}")
    if file_sha256(sidecar) != config["sidecar_file_sha256"]:
        raise ValueError(f"pinned JSONL sidecar hash mismatch: {sidecar}")
    rows = read_jsonl(path)
    if len(rows) != int(config["row_count"]):
        raise ValueError(f"pinned JSONL row count mismatch: {path}")
    if canonical_sha256(rows) != config["ordered_rows_sha256"]:
        raise ValueError(f"pinned JSONL ordered-row hash mismatch: {path}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("row_count", "file_sha256", "ordered_rows_sha256"):
        if manifest.get(key) != config[key]:
            raise ValueError(f"pinned JSONL sidecar field mismatch: {path} {key}")
    return rows


def _download_math_file(
    protocol: Mapping[str, Any], subject: str, *, cache_dir: str | None
) -> tuple[Path, list[dict[str, Any]]]:
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(
            "ranking source audit requires huggingface_hub and pyarrow"
        ) from exc
    math = protocol["sources"]["math"]
    config = math["train_files"][subject]
    path = Path(
        hf_hub_download(
            repo_id=math["dataset_id"],
            repo_type="dataset",
            filename=config["path"],
            revision=math["revision"],
            cache_dir=cache_dir,
        )
    )
    if file_sha256(path) != config["sha256"]:
        raise ValueError(f"MATH train parquet hash mismatch for {subject}")
    rows = parquet.read_table(path).to_pylist()
    if len(rows) != int(config["row_count"]):
        raise ValueError(f"MATH train parquet row count mismatch for {subject}")
    return path, rows


def _materialize_math_train(
    protocol: Mapping[str, Any], *, cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    math = protocol["sources"]["math"]
    allowed_levels = {int(value) for value in math["allowed_levels"]}
    minimum_words = int(math["minimum_official_solution_words"])
    output: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    file_hashes: dict[str, str] = {}
    raw_counts: dict[str, int] = {}
    eligible_counts: Counter[str] = Counter()
    for subject in sorted(math["allowed_subjects"]):
        path, rows = _download_math_file(protocol, subject, cache_dir=cache_dir)
        file_hashes[subject] = file_sha256(path)
        raw_counts[subject] = len(rows)
        for index, raw in enumerate(rows):
            problem = str(raw["problem"]).strip()
            solution = str(raw["solution"]).strip()
            level_match = re.search(r"(\d+)", str(raw["level"]))
            if level_match is None or int(level_match.group(1)) not in allowed_levels:
                rejected[f"{subject}|level"] += 1
                continue
            level = int(level_match.group(1))
            if "[asy]" in problem or "begin{asy}" in problem:
                rejected[f"{subject}|asymptote"] += 1
                continue
            if len(solution.split()) < minimum_words:
                rejected[f"{subject}|short_solution"] += 1
                continue
            reference = extract_math_numeric_reference(solution)
            if reference is None:
                rejected[f"{subject}|unsupported_reference"] += 1
                continue
            output.append(
                {
                    "source": "math",
                    "query_id": f"math:train:{subject}:{index:05d}",
                    "source_record_id": f"{subject}/train/{index}",
                    "question": problem,
                    "reference_answer": reference,
                    "source_solution": solution,
                    "source_level": level,
                    "source_subject": subject,
                    "selection_stratum": f"{subject}|level_{level}",
                    "source_license": "MIT",
                }
            )
            eligible_counts[f"{subject}|level_{level}"] += 1
    output.sort(key=lambda row: str(row["query_id"]))
    return output, {
        "raw_train_rows_by_subject": raw_counts,
        "eligible_rows": len(output),
        "eligible_by_subject_level": dict(sorted(eligible_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "file_sha256": file_hashes,
        "test_files_downloaded_or_read": False,
    }


def _load_extended_sources(
    protocol: Mapping[str, Any], *, cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent = _verify_jsonl_input(protocol["pinned_inputs"]["v3_source_corpus"])
    regenerated_math, math_report = _materialize_math_train(
        protocol, cache_dir=cache_dir
    )
    parent_math = {
        str(row["query_id"]): row for row in parent if row["source"] == "math"
    }
    regenerated_by_id = {str(row["query_id"]): row for row in regenerated_math}
    missing_parent = sorted(set(parent_math) - set(regenerated_by_id))
    if missing_parent:
        raise ValueError("v7 MATH regeneration lost pinned v3 rows")
    for query_id, old in parent_math.items():
        new = regenerated_by_id[query_id]
        for field in (
            "question",
            "reference_answer",
            "source_solution",
            "source_level",
            "source_subject",
        ):
            if old.get(field) != new.get(field):
                raise ValueError(f"regenerated MATH row drift: {query_id} {field}")
    non_math = [dict(row) for row in parent if row["source"] != "math"]
    extended = [*non_math, *regenerated_math]
    ids = [str(row["query_id"]) for row in extended]
    if len(ids) != len(set(ids)):
        raise ValueError("extended source corpus has duplicate query IDs")
    extended.sort(key=lambda row: str(row["query_id"]))
    return extended, {
        "parent_row_count": len(parent),
        "parent_math_rows_reproduced": len(parent_math),
        "extended_row_count": len(extended),
        "extended_source_counts": dict(
            sorted(Counter(row["source"] for row in extended).items())
        ),
        "math": math_report,
    }


def _combine_exclusions(
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old = _verify_jsonl_input(protocol["pinned_inputs"]["v6_permanent_exclusions"])
    v6_train = _verify_jsonl_input(protocol["pinned_inputs"]["v6_train_queries"])
    v6_heldout = _verify_jsonl_input(protocol["pinned_inputs"]["v6_heldout_queries"])
    reasons: dict[str, set[str]] = {}
    sources: dict[str, str] = {}
    for row in old:
        query_id = str(row["query_id"])
        reasons.setdefault(query_id, set()).update(
            str(value) for value in row.get("reasons", [])
        )
        sources[query_id] = str(row.get("source", query_id.split(":", 1)[0]))
    for row in [*v6_train, *v6_heldout]:
        query_id = str(row["query_id"])
        reasons.setdefault(query_id, set()).add(
            "consistency_scale_v6_acquisition_or_relation_use"
        )
        sources[query_id] = str(row["source"])
    output = [
        {
            "query_id": query_id,
            "source": sources.get(query_id, query_id.split(":", 1)[0]),
            "reasons": sorted(values),
        }
        for query_id, values in sorted(reasons.items())
    ]
    return output, {
        "v6_parent_exclusions": len(old),
        "v6_acquisition_queries_added": len(v6_train) + len(v6_heldout),
        "combined_unique_exclusions": len(output),
    }


def _rendered_prompt_count(tokenizer: Any, question: str, template: str) -> int:
    content = template.replace("<QUESTION>", question)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)


def _attach_prompt_counts(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("ranking pre-rollout freeze requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    maximum = int(generation["maximum_prompt_tokens"])
    kept: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    counts: list[int] = []
    for raw in rows:
        row = dict(raw)
        count = _rendered_prompt_count(
            tokenizer, str(row["question"]), str(generation["prompt_template"])
        )
        row["prompt_token_count"] = count
        counts.append(count)
        if count > maximum:
            overflow.append(
                {
                    "query_id": row["query_id"],
                    "source": row["source"],
                    "prompt_token_count": count,
                }
            )
        else:
            kept.append(row)
    ordered = sorted(counts)
    return kept, {
        "input_count": len(rows),
        "kept_count": len(kept),
        "overflow_count": len(overflow),
        "maximum_allowed": maximum,
        "count_min": ordered[0] if ordered else None,
        "count_max": ordered[-1] if ordered else None,
        "count_mean": sum(ordered) / len(ordered) if ordered else None,
        "overflow_rows": sorted(overflow, key=lambda row: str(row["query_id"])),
    }


def _compact_role_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
        "selection_stratum",
        "cluster_id",
        "cluster_split_priority",
        "query_priority",
        "prompt_token_count",
        "question_sha256",
        "template_signature_v6",
        "role",
        "role_priority",
        "evaluation_only",
        "h_target_checker_status",
        "h_label_split",
    )
    return {key: row[key] for key in keys if key in row}


def build_plan(protocol: Mapping[str, Any], *, cache_dir: str | None) -> dict[str, Any]:
    sources, source_report = _load_extended_sources(protocol, cache_dir=cache_dir)
    candidates, filter_report = build_source_candidates(
        sources, protocol, required_schema=RANKING_V7_SCHEMA
    )
    exclusions, exclusion_report = _combine_exclusions(protocol)
    excluded_ids = {str(row["query_id"]) for row in exclusions}
    by_id = {str(row["query_id"]): row for row in sources}
    missing_anchors = sorted(excluded_ids - set(by_id))
    if missing_anchors:
        raise ValueError(f"{len(missing_anchors)} exclusions lack source anchors")
    anchors = [by_id[query_id] for query_id in sorted(excluded_ids)]
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
        namespace=str(protocol["template_clustering"]["namespace"]),
    )
    selectable, prompt_report = _attach_prompt_counts(
        selectable, protocol, cache_dir=cache_dir
    )
    ranking, h_rows, role_report = build_role_manifests(selectable, protocol)
    ranking = [_compact_role_row(row) for row in ranking]
    h_rows = [_compact_role_row(row) for row in h_rows]
    shards = build_rollout_shards(ranking, h_rows, protocol)
    budget = compute_budget(ranking, h_rows, protocol)
    return {
        "sources": sources,
        "exclusions": exclusions,
        "clusters": clusters,
        "ranking": ranking,
        "h": h_rows,
        "shards": shards,
        "reports": {
            "source": source_report,
            "source_filter": filter_report,
            "exclusions": exclusion_report,
            "clusters": cluster_report,
            "prompt_tokens": prompt_report,
            "roles": role_report,
            "budget": budget,
        },
    }


def _json_record(path: Path, payload: Any) -> dict[str, Any]:
    atomic_write_json(path, payload)
    return {
        "path": path.name,
        "format": "json",
        "file_sha256": file_sha256(path),
        "canonical_payload_sha256": canonical_sha256(payload),
        "row_count": len(payload) if isinstance(payload, list) else None,
    }


def _jsonl_record(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = publish_manifest(
        path, rows, schema_version=schema_version, metadata=metadata
    )
    return {
        "path": path.name,
        "format": "jsonl",
        "file_sha256": manifest["file_sha256"],
        "row_count": manifest["row_count"],
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "sidecar_path": path.name + ".manifest.json",
        "sidecar_file_sha256": file_sha256(
            path.with_suffix(path.suffix + ".manifest.json")
        ),
    }


def command_audit(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    plan = build_plan(protocol, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "status": "PASS_RANKING_V7_SOURCE_CAPACITY_AUDIT",
                "protocol_file_sha256": file_sha256(Path(args.protocol)),
                "candidate_counts": plan["reports"]["source_filter"]["counts"],
                "cluster_report": plan["reports"]["clusters"],
                "role_report": plan["reports"]["roles"],
                "prompt_token_report": plan["reports"]["prompt_tokens"],
                "budget": plan["reports"]["budget"],
                "artifacts_written": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_freeze(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if _git_dirty():
        raise RuntimeError("pre-rollout freeze requires a clean committed worktree")
    output = Path(args.output_dir).resolve()
    if (output / "manifest_registry.json").exists():
        raise FileExistsError("ranking v7 pre-rollout directory is already frozen")
    plan = build_plan(protocol, cache_dir=args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_inventory = {
        "schema_version": "clir-ranking-v7-source-inventory",
        "protocol_file_sha256": file_sha256(protocol_path),
        **plan["reports"],
    }
    records: dict[str, Any] = {}
    records["source_inventory.json"] = _json_record(
        output / "source_inventory.json", source_inventory
    )
    records["permanent_exclusions.jsonl"] = _jsonl_record(
        output / "permanent_exclusions.jsonl",
        plan["exclusions"],
        schema_version="clir-ranking-v7-permanent-exclusions",
        metadata=plan["reports"]["exclusions"],
    )
    records["template_clusters.jsonl"] = _jsonl_record(
        output / "template_clusters.jsonl",
        plan["clusters"],
        schema_version="clir-ranking-v7-template-clusters",
        metadata=plan["reports"]["clusters"],
    )
    records["ranking_queries.jsonl"] = _jsonl_record(
        output / "ranking_queries.jsonl",
        plan["ranking"],
        schema_version="clir-ranking-v7-evaluation-queries",
        metadata={
            "source_counts": dict(Counter(row["source"] for row in plan["ranking"]))
        },
    )
    records["h_acquisition_queries.jsonl"] = _jsonl_record(
        output / "h_acquisition_queries.jsonl",
        plan["h"],
        schema_version="clir-ranking-v7-h-acquisition-queries",
        metadata={
            "source_counts": dict(Counter(row["source"] for row in plan["h"])),
            "preassigned_cells": plan["reports"]["roles"]["h_preassigned_cells"],
        },
    )
    records["rollout_shards.json"] = _json_record(
        output / "rollout_shards.json", plan["shards"]
    )
    report = {
        "schema_version": "clir-ranking-v7-pre-rollout-freeze-report",
        "status": "PASS_RANKING_V7_PRE_ROLLOUT_MANIFEST_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "code_dirty": False,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "records": records,
        "ranking_query_count": len(plan["ranking"]),
        "h_query_count": len(plan["h"]),
        "rollout_shard_count": len(plan["shards"]),
        "candidate_rows_total": plan["reports"]["budget"]["candidate_rows_total"],
        "query_overlap": 0,
        "cluster_overlap": 0,
        "rollout_started": False,
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "next_gate": "INDEPENDENT_RECOMPUTE_THEN_HASH_BOUND_ROLLOUT_AUTHORIZATION",
    }
    atomic_write_json(output / "pre_rollout_report.json", report)
    registry = {
        "schema_version": "clir-ranking-v7-pre-rollout-registry",
        "status": report["status"],
        "protocol_file_sha256": report["protocol_file_sha256"],
        "code_commit": report["code_commit"],
        "records": records,
        "pre_rollout_report_file_sha256": file_sha256(
            output / "pre_rollout_report.json"
        ),
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


def _verify_record(output: Path, name: str, record: Mapping[str, Any]) -> Any:
    path = output / name
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"frozen record hash mismatch: {name}")
    if record["format"] == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(payload) != record["canonical_payload_sha256"]:
            raise ValueError(f"frozen JSON canonical hash mismatch: {name}")
        return payload
    rows = read_jsonl(path)
    if len(rows) != int(record["row_count"]):
        raise ValueError(f"frozen JSONL row count mismatch: {name}")
    if canonical_sha256(rows) != record["ordered_rows_sha256"]:
        raise ValueError(f"frozen JSONL ordered hash mismatch: {name}")
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if file_sha256(sidecar) != record["sidecar_file_sha256"]:
        raise ValueError(f"frozen JSONL sidecar hash mismatch: {name}")
    return rows


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    output = Path(args.output_dir).resolve()
    registry_path = output / "manifest_registry.json"
    report_path = output / "pre_rollout_report.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("frozen registry protocol hash mismatch")
    if registry.get("pre_rollout_report_file_sha256") != file_sha256(report_path):
        raise ValueError("frozen pre-rollout report hash mismatch")
    frozen: dict[str, Any] = {}
    for name in REQUIRED_FILES:
        if name not in registry["records"]:
            raise ValueError(f"registry lacks required record {name}")
        frozen[name] = _verify_record(output, name, registry["records"][name])

    recomputed = build_plan(protocol, cache_dir=args.cache_dir)
    expected = {
        "permanent_exclusions.jsonl": recomputed["exclusions"],
        "template_clusters.jsonl": recomputed["clusters"],
        "ranking_queries.jsonl": recomputed["ranking"],
        "h_acquisition_queries.jsonl": recomputed["h"],
        "rollout_shards.json": recomputed["shards"],
    }
    for name, rows in expected.items():
        if frozen[name] != rows:
            raise ValueError(f"independent recomputation drift: {name}")
    source_inventory = frozen["source_inventory.json"]
    if source_inventory.get("source_filter") != recomputed["reports"]["source_filter"]:
        raise ValueError("independent source-filter report drift")
    if source_inventory.get("roles") != recomputed["reports"]["roles"]:
        raise ValueError("independent role report drift")
    verification = {
        "schema_version": "clir-ranking-v7-pre-rollout-independent-verification",
        "status": "PASS_RANKING_V7_PRE_ROLLOUT_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "pre_rollout_report_file_sha256": file_sha256(report_path),
        "ranking_query_count": len(recomputed["ranking"]),
        "h_query_count": len(recomputed["h"]),
        "rollout_shard_count": len(recomputed["shards"]),
        "query_overlap": 0,
        "cluster_overlap": 0,
        "rollout_allowed": False,
        "next_gate": "HASH_BOUND_ROLLOUT_AUTHORIZATION",
    }
    verification_path = output / "independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        stable_keys = set(verification) - {"verified_at_utc"}
        if any(old.get(key) != verification.get(key) for key in stable_keys):
            raise ValueError("existing independent verification report drift")
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


def verify_pre_rollout(output: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    registry_path = output / "manifest_registry.json"
    report_path = output / "pre_rollout_report.json"
    verification_path = output / "independent_verification.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "PASS_RANKING_V7_PRE_ROLLOUT_MANIFEST_FREEZE":
        raise ValueError("pre-rollout registry does not record a v7 PASS")
    if registry.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("pre-rollout registry protocol hash mismatch")
    if registry.get("pre_rollout_report_file_sha256") != file_sha256(report_path):
        raise ValueError("pre-rollout report hash mismatch")
    payloads = {
        name: _verify_record(output, name, registry["records"][name])
        for name in REQUIRED_FILES
    }
    ranking = payloads["ranking_queries.jsonl"]
    h_rows = payloads["h_acquisition_queries.jsonl"]
    shards = payloads["rollout_shards.json"]
    if len(ranking) != 1500 or len(h_rows) != 1000 or len(shards) != 50:
        raise ValueError("frozen v7 population count mismatch")
    if Counter(row["source"] for row in ranking) != {"math": 700, "gsm8k": 800}:
        raise ValueError("frozen ranking source count mismatch")
    if Counter(row["source"] for row in h_rows) != {"math": 600, "gsm8k": 400}:
        raise ValueError("frozen H-acquisition source count mismatch")
    ranking_ids = {str(row["query_id"]) for row in ranking}
    h_ids = {str(row["query_id"]) for row in h_rows}
    ranking_clusters = {str(row["cluster_id"]) for row in ranking}
    h_clusters = {str(row["cluster_id"]) for row in h_rows}
    if ranking_ids & h_ids or ranking_clusters & h_clusters:
        raise ValueError("frozen ranking/H query or cluster leakage")
    shard_ids = [str(value) for shard in shards for value in shard["query_ids"]]
    if len(shard_ids) != 2500 or set(shard_ids) != ranking_ids | h_ids:
        raise ValueError("rollout shards are not an exact query partition")
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("a query occurs in more than one rollout shard")
    role_by_id = {str(row["query_id"]): str(row["role"]) for row in [*ranking, *h_rows]}
    for shard in shards:
        role = str(shard["role"])
        cfg = protocol["roles"][role]
        if int(shard["query_count"]) != int(cfg["queries_per_shard"]):
            raise ValueError(f"invalid query count in shard {shard['shard_id']}")
        if int(shard["candidate_count"]) != int(cfg["candidate_count"]):
            raise ValueError(f"invalid candidate count in shard {shard['shard_id']}")
        if int(shard["expected_candidate_rows"]) != int(shard["query_count"]) * int(
            shard["candidate_count"]
        ):
            raise ValueError(f"invalid expected rows in shard {shard['shard_id']}")
        if {role_by_id[query_id] for query_id in shard["query_ids"]} != {role}:
            raise ValueError(f"mixed roles in shard {shard['shard_id']}")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("status")
        != "PASS_RANKING_V7_PRE_ROLLOUT_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("independent pre-rollout verification is not a PASS")
    if verification.get("manifest_registry_file_sha256") != file_sha256(registry_path):
        raise ValueError("independent verification registry hash mismatch")
    return {
        "status": "PASS_RANKING_V7_PRE_ROLLOUT_VERIFY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "independent_verification_file_sha256": file_sha256(verification_path),
        "ranking_queries": len(ranking),
        "h_queries": len(h_rows),
        "rollout_shards": len(shards),
        "planned_raw_trajectories": sum(
            int(shard["expected_candidate_rows"]) for shard in shards
        ),
    }


def load_rollout_authorization(
    path: Path, *, protocol_path: Path, pre_rollout_dir: Path
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != "clir-ranking-v7-rollout-authorization":
        raise ValueError("unsupported ranking-v7 rollout authorization schema")
    if authorization.get("status") != "AUTHORIZED_ROLLOUT_ONLY":
        raise ValueError("ranking-v7 rollout has not received explicit authorization")
    scope = authorization.get("authorized_scope", {})
    if scope.get("rollout") is not True or any(
        scope.get(name) is not False
        for name in (
            "checker_and_unitizer_materialization",
            "annotation",
            "feature_extraction",
            "training",
            "threshold_or_query_manifest_change",
        )
    ):
        raise ValueError("authorization must permit rollout and no later stage")
    parent = authorization["frozen_parent"]
    expected = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "pre_rollout_report_file_sha256": file_sha256(
            pre_rollout_dir / "pre_rollout_report.json"
        ),
        "independent_verification_file_sha256": file_sha256(
            pre_rollout_dir / "independent_verification.json"
        ),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"rollout authorization {key} mismatch")
    verify_pre_rollout(pre_rollout_dir, protocol_path)
    return authorization


def _load_rollout_contract(
    *,
    protocol_path: Path,
    authorization_path: Path,
    pre_rollout_dir: Path,
    rollout_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    protocol = load_protocol(protocol_path)
    authorization = load_rollout_authorization(
        authorization_path,
        protocol_path=protocol_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if rollout_root != expected_root:
        raise ValueError(
            f"rollout root differs from authorization: {rollout_root} != {expected_root}"
        )
    ranking = read_jsonl(pre_rollout_dir / "ranking_queries.jsonl")
    h_rows = read_jsonl(pre_rollout_dir / "h_acquisition_queries.jsonl")
    shards = json.loads(
        (pre_rollout_dir / "rollout_shards.json").read_text(encoding="utf-8")
    )
    query_by_id = {str(row["query_id"]): row for row in [*ranking, *h_rows]}
    if len(query_by_id) != 2500:
        raise ValueError("frozen v7 manifests do not contain 2,500 unique queries")
    return protocol, authorization, shards, query_by_id


def _select_shard(shards: Sequence[Mapping[str, Any]], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one frozen shard named {shard_id}, found {len(matches)}"
        )
    return matches[0]


def _shard_paths(
    rollout_root: Path, shard: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    output = rollout_root / str(shard["output_path"])
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    completion = output.with_suffix(".complete.json")
    return output, sidecar, completion


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
    candidate_count = int(shard["candidate_count"])
    population = validate_rollout_population(rows, candidate_count=candidate_count)
    expected_query_ids = [str(value) for value in shard["query_ids"]]
    encountered: list[str] = []
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered or encountered[-1] != query_id:
            encountered.append(query_id)
    if encountered != expected_query_ids:
        raise ValueError(
            f"{shard['shard_id']}: rollout query order differs from freeze"
        )
    if len(rows) != int(shard["expected_candidate_rows"]):
        raise ValueError(f"{shard['shard_id']}: rollout row count mismatch")
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
    output_lengths: list[int] = []
    prompt_lengths: list[int] = []
    decode_mismatches = 0
    for row in rows:
        query_id = str(row["query_id"])
        query = query_by_id.get(query_id)
        if query is None:
            raise ValueError(f"{shard['shard_id']}: unknown query {query_id}")
        candidate_index = int(row["candidate_index"])
        if row.get("id") != f"{query_id}:cand:{candidate_index:03d}":
            raise ValueError(f"{query_id}: noncanonical trajectory ID")
        for field in ("source", "question", "reference_answer", "cluster_id", "role"):
            if row.get(field) != query.get(field):
                raise ValueError(f"{row['id']}: {field} differs from frozen query")
        if row.get("shard_id") != shard["shard_id"]:
            raise ValueError(f"{row['id']}: shard_id mismatch")
        if int(query["prompt_token_count"]) != len(row["prompt_token_ids"]):
            raise ValueError(f"{row['id']}: prompt token count differs from freeze")
        expected_seed = _derive_query_seed(
            int(protocol["generation"]["base_seed"]), query_id
        )
        if int(row.get("sampling_seed", -1)) != expected_seed:
            raise ValueError(f"{row['id']}: sampling seed mismatch")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{row['id']}: missing rollout provenance")
        for key in provenance_values:
            provenance_values[key].add(str(provenance.get(key)))
        finish_reasons[str(row.get("finish_reason"))] += 1
        output_lengths.append(len(row["output_token_ids"]))
        prompt_lengths.append(len(row["prompt_token_ids"]))
        decode_mismatches += row.get("decode_matches_backend_text") is not True
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
        raise ValueError(f"{shard['shard_id']}: mixed code commits within shard")
    return {
        **population,
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "query_order_matches_freeze": True,
        "prompt_token_count_matches_freeze": True,
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
        raise FileNotFoundError(f"missing completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE_VERIFIED_ROLLOUT_SHARD_V7":
        raise ValueError(f"invalid completion status: {completion_path}")
    if completion.get("shard_id") != shard["shard_id"]:
        raise ValueError(f"completion marker shard mismatch: {completion_path}")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_bindings = {
        "shard_spec_sha256": canonical_sha256(shard),
        "protocol_file_sha256": authorization["frozen_parent"]["protocol_file_sha256"],
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
    }
    for key, expected in expected_bindings.items():
        if completion.get(key) != expected:
            raise ValueError(f"completion marker {key} mismatch: {completion_path}")
    if not output.is_file() or file_sha256(output) != completion["file_sha256"]:
        raise ValueError(f"rollout shard file hash mismatch: {output}")
    if (
        not sidecar_path.is_file()
        or file_sha256(sidecar_path) != completion["sidecar_file_sha256"]
    ):
        raise ValueError(f"rollout shard sidecar hash mismatch: {sidecar_path}")
    rows = read_jsonl(output)
    if canonical_sha256(rows) != completion["ordered_rows_sha256"]:
        raise ValueError(f"rollout shard ordered-row hash mismatch: {output}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("file_sha256") != completion["file_sha256"]
        or sidecar.get("ordered_rows_sha256") != completion["ordered_rows_sha256"]
        or int(sidecar.get("row_count", -1)) != int(completion["row_count"])
    ):
        raise ValueError(f"rollout shard sidecar contents mismatch: {sidecar_path}")
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
        raise ValueError(f"rollout shard validation summary drift: {completion_path}")
    return {
        "status": "PASS_ROLLOUT_SHARD_VERIFY_V7",
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "path": str(output),
        "file_sha256": completion["file_sha256"],
        "ordered_rows_sha256": completion["ordered_rows_sha256"],
        "row_count": len(rows),
        "validation": validation,
        "runtime": completion["runtime"],
    }


def command_rollout(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    protocol, authorization, shards, query_by_id = _load_rollout_contract(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
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
            pre_rollout_dir=pre_rollout_dir,
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
                    pre_rollout_dir=pre_rollout_dir,
                    rollout_root=rollout_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output.exists() or sidecar_path.exists():
        raise FileExistsError(
            f"{shard['shard_id']} has incomplete artifacts; refusing to overwrite"
        )
    if _git_dirty():
        raise RuntimeError("ranking-v7 rollout requires a clean Git commit")
    runtime = authorization["runtime_contract"]
    if int(runtime["tensor_parallel_size"]) != 1:
        raise ValueError("ranking-v7 authorization requires TP=1")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("ranking-v7 rollout requires torch and vLLM") from exc
    generation = protocol["generation"]
    if _package_version("vllm") != generation["backend_version"]:
        raise ValueError("installed vLLM version differs from frozen protocol")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each shard process must see exactly one GPU; set CUDA_VISIBLE_DEVICES"
        )
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 40_000_000_000:
        raise RuntimeError("visible GPU has less than 40 GB free before model load")

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
    sampling: list[Any] = []
    expected_prompt_ids: list[list[int]] = []
    candidate_count = int(shard["candidate_count"])
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
        if len(prompt_ids) != int(query["prompt_token_count"]):
            raise ValueError(
                f"{query['query_id']}: tokenizer prompt length changed since freeze"
            )
        expected_prompt_ids.append(prompt_ids)
        sampling.append(
            SamplingParams(
                n=candidate_count,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_derive_query_seed(
                    int(generation["base_seed"]), str(query["query_id"])
                ),
            )
        )
    request_outputs = llm.generate(rendered_prompts, sampling, use_tqdm=True)
    if len(request_outputs) != len(query_rows):
        raise ValueError("vLLM returned a different number of query outputs")
    finished_at = _utc_now()
    elapsed_seconds = time.monotonic() - started_clock
    provenance = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
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
        "elapsed_seconds": elapsed_seconds,
    }
    rows: list[dict[str, Any]] = []
    for query, request_output, expected_ids in zip(
        query_rows, request_outputs, expected_prompt_ids
    ):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        if prompt_ids != expected_ids:
            raise ValueError(f"{query['query_id']}: vLLM prompt IDs differ from freeze")
        for candidate in _ordered_vllm_candidates(request_output, candidate_count):
            output_ids = [int(value) for value in candidate.token_ids]
            if not output_ids:
                raise ValueError(f"{query['query_id']}: vLLM returned an empty output")
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
                    "cluster_id": query["cluster_id"],
                    "source": query["source"],
                    "source_record_id": query.get("source_record_id"),
                    "source_subject": query.get("source_subject"),
                    "source_level": query.get("source_level"),
                    "source_license": query.get("source_license"),
                    "evaluation_only": query.get("evaluation_only"),
                    "h_target_checker_status": query.get("h_target_checker_status"),
                    "h_label_split": query.get("h_label_split"),
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
                        int(generation["base_seed"]), str(query["query_id"])
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
        registry_file_sha256=file_sha256(pre_rollout_dir / "manifest_registry.json"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-ranking-v7-raw-rollout-shard",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-ranking-v7-rollout-shard-completion",
        "status": "COMPLETE_VERIFIED_ROLLOUT_SHARD_V7",
        "shard_id": shard["shard_id"],
        "role": shard["role"],
        "shard_spec_sha256": canonical_sha256(shard),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar_path),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "row_count": manifest["row_count"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": _git_head(),
        "validation": validation,
        "runtime": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": len(rows) / elapsed_seconds,
            "output_tokens_per_second": validation["total_output_tokens"]
            / elapsed_seconds,
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
                pre_rollout_dir=pre_rollout_dir,
                rollout_root=rollout_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify_rollouts(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    protocol, _, shards, query_by_id = _load_rollout_contract(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    requested = set(args.shard_id or [])
    if requested:
        unknown = requested - {str(row["shard_id"]) for row in shards}
        if unknown:
            raise ValueError(f"unknown shard IDs: {sorted(unknown)}")
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
                pre_rollout_dir=pre_rollout_dir,
                rollout_root=rollout_root,
            )
        )
    if args.require_complete and missing:
        raise ValueError(f"missing {len(missing)} rollout shards: {missing}")
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ALL_ROLLOUT_SHARDS_VERIFY_V7"
                    if not missing and len(completed) == len(shards)
                    else "PARTIAL_ROLLOUT_SHARDS_VERIFY_V7"
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
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    protocol, _, shards, query_by_id = _load_rollout_contract(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    report_path = rollout_root / "rollout_completion_report.json"
    outputs = {
        RANKING_ROLE: rollout_root / "rollouts/ranking_combined_raw.jsonl",
        H_ROLE: rollout_root / "rollouts/h_combined_raw.jsonl",
    }
    if report_path.exists() or any(
        path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists()
        for path in outputs.values()
    ):
        raise FileExistsError(
            "combined rollout artifacts already exist; never overwrite"
        )
    reports: list[dict[str, Any]] = []
    rows_by_role: dict[str, list[dict[str, Any]]] = {
        RANKING_ROLE: [],
        H_ROLE: [],
    }
    for shard in shards:
        report = verify_rollout_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            rollout_root=rollout_root,
        )
        reports.append(report)
        rows_by_role[str(shard["role"])].extend(read_jsonl(Path(report["path"])))
    all_ids = [str(row["id"]) for rows in rows_by_role.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("rollout shards overlap by trajectory ID")
    code_commits = {
        str(row["provenance"]["code_commit"])
        for rows in rows_by_role.values()
        for row in rows
    }
    if len(code_commits) != 1:
        raise ValueError("rollout shards were produced by different code commits")
    role_records: dict[str, Any] = {}
    for role, rows in rows_by_role.items():
        candidate_count = int(protocol["roles"][role]["candidate_count"])
        population = validate_rollout_population(rows, candidate_count=candidate_count)
        expected_queries = [
            str(query_id)
            for shard in shards
            if shard["role"] == role
            for query_id in shard["query_ids"]
        ]
        encountered: list[str] = []
        for row in rows:
            query_id = str(row["query_id"])
            if not encountered or encountered[-1] != query_id:
                encountered.append(query_id)
        if encountered != expected_queries:
            raise ValueError(f"combined {role} query order differs from frozen shards")
        finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
        output_lengths = [len(row["output_token_ids"]) for row in rows]
        metadata = {
            "role": role,
            "protocol_file_sha256": file_sha256(protocol_path),
            "pre_rollout_registry_file_sha256": file_sha256(
                pre_rollout_dir / "manifest_registry.json"
            ),
            "authorization_file_sha256": file_sha256(authorization_path),
            "code_commit": next(iter(code_commits)),
            **population,
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "output_token_count": _integer_summary(output_lengths),
            "total_output_tokens": sum(output_lengths),
        }
        manifest = publish_manifest(
            outputs[role],
            rows,
            schema_version=f"clir-ranking-v7-{role}-combined-raw-rollouts",
            metadata=metadata,
        )
        role_records[role] = {
            **metadata,
            "path": str(outputs[role]),
            "file_sha256": manifest["file_sha256"],
            "sidecar_file_sha256": file_sha256(
                outputs[role].with_suffix(outputs[role].suffix + ".manifest.json")
            ),
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        }
    report = {
        "schema_version": "clir-ranking-v7-rollout-completion-report",
        "status": "PASS_ALL_32000_RAW_ROLLOUTS_VERIFIED_V7",
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": next(iter(code_commits)),
        "roles": role_records,
        "shards": reports,
        "next_gate": "checker_unitizer_and_h_proposal_freeze_before_annotation",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_published_jsonl(
    path: Path, *, expected_schema: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"published JSONL or sidecar is missing: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if expected_schema is not None and sidecar.get("schema_version") != expected_schema:
        raise ValueError(f"{path}: published schema differs from {expected_schema}")
    if sidecar.get("file_sha256") != file_sha256(path):
        raise ValueError(f"{path}: file hash differs from sidecar")
    rows = read_jsonl(path)
    if int(sidecar.get("row_count", -1)) != len(rows):
        raise ValueError(f"{path}: sidecar row count mismatch")
    if sidecar.get("ordered_rows_sha256") != canonical_sha256(rows):
        raise ValueError(f"{path}: sidecar ordered-row hash mismatch")
    return rows, sidecar


def _verify_rollout_merge_contract(
    *,
    protocol_path: Path,
    rollout_authorization_path: Path,
    pre_rollout_dir: Path,
    rollout_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol, authorization, _, _ = _load_rollout_contract(
        protocol_path=protocol_path,
        authorization_path=rollout_authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    report_path = rollout_root / "rollout_completion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_ALL_32000_RAW_ROLLOUTS_VERIFIED_V7":
        raise ValueError("ranking-v7 rollout completion report is not a PASS")
    if report.get("authorization_file_sha256") != file_sha256(
        rollout_authorization_path
    ):
        raise ValueError("rollout completion report authorization hash mismatch")
    ranking_path = rollout_root / "rollouts/ranking_combined_raw.jsonl"
    h_path = rollout_root / "rollouts/h_combined_raw.jsonl"
    ranking, ranking_sidecar = _read_published_jsonl(
        ranking_path,
        expected_schema="clir-ranking-v7-ranking_evaluation-combined-raw-rollouts",
    )
    h_rows, h_sidecar = _read_published_jsonl(
        h_path,
        expected_schema="clir-ranking-v7-hallucination_acquisition-combined-raw-rollouts",
    )
    expected = {
        RANKING_ROLE: (ranking, ranking_sidecar),
        H_ROLE: (h_rows, h_sidecar),
    }
    for role, (rows, sidecar) in expected.items():
        record = report["roles"][role]
        if (
            record.get("file_sha256") != sidecar["file_sha256"]
            or record.get("ordered_rows_sha256") != sidecar["ordered_rows_sha256"]
            or int(record.get("rows", -1)) != len(rows)
        ):
            raise ValueError(f"rollout completion report drift for {role}")
    return protocol, authorization, ranking, h_rows, report


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
        "clir-ranking-v7-pre-annotation-authorization"
    ):
        raise ValueError("unsupported ranking-v7 pre-annotation authorization")
    if authorization.get("status") != (
        "AUTHORIZED_CHECKER_UNITIZER_PROPOSAL_AND_SMOKE_PACKAGE_ONLY"
    ):
        raise ValueError("ranking-v7 pre-annotation stage is not authorized")
    scope = authorization.get("authorized_scope", {})
    required_true = {
        "ranking_checker_materialization",
        "h_checker_and_unitizer_materialization",
        "h_proposal_freeze",
        "smoke_blind_package_construction",
    }
    required_false = {
        "ai_annotation_or_provider_call",
        "reserve_package_construction_before_smoke_pass",
        "label_finalization",
        "feature_extraction",
        "training",
        "threshold_or_query_manifest_change",
    }
    if any(scope.get(name) is not True for name in required_true) or any(
        scope.get(name) is not False for name in required_false
    ):
        raise ValueError("pre-annotation authorization scope is invalid")
    protocol, _, ranking, h_rows, report = _verify_rollout_merge_contract(
        protocol_path=protocol_path,
        rollout_authorization_path=rollout_authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    parent = authorization["frozen_parent"]
    expected = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "rollout_authorization_file_sha256": file_sha256(rollout_authorization_path),
        "rollout_completion_report_file_sha256": file_sha256(
            rollout_root / "rollout_completion_report.json"
        ),
        "ranking_combined_file_sha256": file_sha256(
            rollout_root / "rollouts/ranking_combined_raw.jsonl"
        ),
        "ranking_combined_ordered_rows_sha256": canonical_sha256(ranking),
        "h_combined_file_sha256": file_sha256(
            rollout_root / "rollouts/h_combined_raw.jsonl"
        ),
        "h_combined_ordered_rows_sha256": canonical_sha256(h_rows),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"pre-annotation authorization {key} mismatch")
    if report.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("rollout completion protocol hash mismatch")
    if protocol["checker"]["unitizer_version"] != UNITIZER_VERSION:
        raise ValueError("protocol unitizer differs from installed frozen unitizer")
    return authorization


def _load_pre_annotation_contract(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
]:
    protocol_path = Path(args.protocol).resolve()
    rollout_authorization_path = Path(args.authorization).resolve()
    pre_authorization_path = Path(args.pre_annotation_authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    output_root = Path(args.pre_annotation_root).resolve()
    authorization = load_pre_annotation_authorization(
        pre_authorization_path,
        protocol_path=protocol_path,
        rollout_authorization_path=rollout_authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if output_root != expected_root:
        raise ValueError(
            f"pre-annotation root differs from authorization: {output_root} != {expected_root}"
        )
    protocol, _, ranking, h_rows, _ = _verify_rollout_merge_contract(
        protocol_path=protocol_path,
        rollout_authorization_path=rollout_authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    return protocol, authorization, ranking, h_rows, output_root


def _require_clean_execution(stage: str) -> str:
    if _git_dirty():
        raise RuntimeError(f"{stage} requires a clean Git commit")
    return _git_head()


def command_materialize_h(args: argparse.Namespace) -> None:
    protocol, _, ranking_raw, h_raw, output_root = _load_pre_annotation_contract(args)
    code_commit = _require_clean_execution("ranking-v7 materialization")
    ranking_path = output_root / "materialized/ranking_checked.jsonl"
    h_path = output_root / "materialized/h_materialized.jsonl"
    report_path = output_root / "materialized/materialization_report.json"
    if any(
        path.exists()
        for path in (
            ranking_path,
            ranking_path.with_suffix(ranking_path.suffix + ".manifest.json"),
            h_path,
            h_path.with_suffix(h_path.suffix + ".manifest.json"),
            report_path,
        )
    ):
        raise FileExistsError("ranking-v7 materialization artifacts already exist")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("ranking-v7 unitization requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("ranking-v7 unitizer requires a fast tokenizer")
    checker_version = str(protocol["checker"]["checker_version"])
    ranking_checked: list[dict[str, Any]] = []
    for raw in ranking_raw:
        row = dict(raw)
        row["raw_reference_answer"] = str(row["reference_answer"])
        checker = check_numeric_response(
            response=str(row["response"]),
            raw_reference=row["raw_reference_answer"],
            source=str(row["source"]),
            finish_reason=row.get("finish_reason"),
            checker_version=checker_version,
        )
        if checker.get("checker_status") == "parse_failed":
            checker["eligible_for_supervision"] = False
        row.update(checker)
        ranking_checked.append(row)
    h_materialized, h_health = materialize_scale_rows(
        h_raw,
        tokenizer,
        checker_version=checker_version,
        unitizer_version=str(protocol["checker"]["unitizer_version"]),
    )
    h_validation = validate_scale_materialized_rows(
        h_materialized,
        raw_rows=h_raw,
        candidate_count=int(protocol["roles"][H_ROLE]["candidate_count"]),
        checker_version=checker_version,
        unitizer_version=str(protocol["checker"]["unitizer_version"]),
    )
    ranking_population = validate_rollout_population(
        ranking_checked,
        candidate_count=int(protocol["roles"][RANKING_ROLE]["candidate_count"]),
    )
    ranking_statuses = dict(
        sorted(Counter(str(row["checker_status"]) for row in ranking_checked).items())
    )
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_manifest = publish_manifest(
        ranking_path,
        ranking_checked,
        schema_version="clir-ranking-v7-ranking-checked",
        metadata={
            "protocol_file_sha256": file_sha256(Path(args.protocol)),
            "pre_annotation_authorization_file_sha256": file_sha256(
                Path(args.pre_annotation_authorization)
            ),
            "code_commit": code_commit,
            "checker_version": checker_version,
            "checker_statuses": ranking_statuses,
            **ranking_population,
        },
    )
    h_manifest = publish_manifest(
        h_path,
        h_materialized,
        schema_version="clir-ranking-v7-h-materialized",
        metadata={
            "protocol_file_sha256": file_sha256(Path(args.protocol)),
            "pre_annotation_authorization_file_sha256": file_sha256(
                Path(args.pre_annotation_authorization)
            ),
            "code_commit": code_commit,
            **h_health,
        },
    )
    report = {
        "schema_version": "clir-ranking-v7-materialization-report",
        "status": "PASS_RANKING_V7_CHECKER_UNITIZER_MATERIALIZATION",
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "code_commit": code_commit,
        "protocol_file_sha256": file_sha256(Path(args.protocol)),
        "pre_annotation_authorization_file_sha256": file_sha256(
            Path(args.pre_annotation_authorization)
        ),
        "ranking": {
            "rows": len(ranking_checked),
            "file_sha256": ranking_manifest["file_sha256"],
            "sidecar_file_sha256": file_sha256(
                ranking_path.with_suffix(ranking_path.suffix + ".manifest.json")
            ),
            "ordered_rows_sha256": ranking_manifest["ordered_rows_sha256"],
            "checker_statuses": ranking_statuses,
        },
        "h": {
            "rows": len(h_materialized),
            "file_sha256": h_manifest["file_sha256"],
            "sidecar_file_sha256": file_sha256(
                h_path.with_suffix(h_path.suffix + ".manifest.json")
            ),
            "ordered_rows_sha256": h_manifest["ordered_rows_sha256"],
            "health": h_health,
            "validation": h_validation,
        },
        "next_gate": "deterministic_h_proposal_freeze",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _verify_materialized_h(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    Path,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol, _, ranking_raw, h_raw, output_root = _load_pre_annotation_contract(args)
    ranking_path = output_root / "materialized/ranking_checked.jsonl"
    h_path = output_root / "materialized/h_materialized.jsonl"
    ranking, ranking_sidecar = _read_published_jsonl(
        ranking_path, expected_schema="clir-ranking-v7-ranking-checked"
    )
    h_rows, h_sidecar = _read_published_jsonl(
        h_path, expected_schema="clir-ranking-v7-h-materialized"
    )
    if len(ranking) != len(ranking_raw) or len(h_rows) != len(h_raw):
        raise ValueError("materialized row counts differ from raw rollouts")
    checker_version = str(protocol["checker"]["checker_version"])
    h_validation = validate_scale_materialized_rows(
        h_rows,
        raw_rows=h_raw,
        candidate_count=int(protocol["roles"][H_ROLE]["candidate_count"]),
        checker_version=checker_version,
        unitizer_version=str(protocol["checker"]["unitizer_version"]),
    )
    report_path = output_root / "materialized/materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS_RANKING_V7_CHECKER_UNITIZER_MATERIALIZATION"
        or report["ranking"].get("file_sha256") != ranking_sidecar["file_sha256"]
        or report["h"].get("file_sha256") != h_sidecar["file_sha256"]
    ):
        raise ValueError("materialization report does not bind current rows")
    rescue_path = output_root / "yield_rescue/materialized/rescue_rows.jsonl"
    if rescue_path.exists():
        rescue_rows, rescue_sidecar = _read_published_jsonl(
            rescue_path,
            expected_schema="clir-h0-v7.1-yield-rescue-materialized",
        )
        rescue_report_path = (
            output_root / "yield_rescue/materialized/materialization_report.json"
        )
        rescue_report = json.loads(rescue_report_path.read_text(encoding="utf-8"))
        if (
            rescue_report.get("status")
            != "PASS_H0_V7_1_YIELD_RESCUE_MATERIALIZATION"
            or rescue_report.get("file_sha256") != rescue_sidecar["file_sha256"]
            or int(rescue_report.get("rows", -1)) != len(rescue_rows)
            or rescue_report.get("parent_h_materialized_file_sha256")
            != file_sha256(h_path)
        ):
            raise ValueError("yield-rescue materialization report drift")
        parent_query_ids = {str(row["query_id"]) for row in h_rows}
        if not {str(row["query_id"]) for row in rescue_rows}.issubset(
            parent_query_ids
        ):
            raise ValueError("yield-rescue rows contain a non-parent query")
        h_rows = [*h_rows, *rescue_rows]
    return protocol, output_root, ranking, h_rows, h_validation


def command_propose_h(args: argparse.Namespace) -> None:
    protocol, output_root, _, h_rows, h_validation = _verify_materialized_h(args)
    code_commit = _require_clean_execution("ranking-v7 H proposal freeze")
    proposal_dir = output_root / "proposals"
    paths = {
        "all": proposal_dir / "h_proposals_all.jsonl",
        "smoke": proposal_dir / "h_smoke_proposals.jsonl",
        "reserve": proposal_dir / "h_reserve_proposals.jsonl",
    }
    report_path = proposal_dir / "proposal_report.json"
    if report_path.exists() or any(
        path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists()
        for path in paths.values()
    ):
        raise FileExistsError("ranking-v7 H proposal artifacts already exist")
    proposals, proposal_report = build_h_proposals(h_rows, protocol)
    smoke, reserve, split_report = split_smoke_and_reserve(proposals, protocol)
    proposal_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        name: publish_manifest(
            paths[name],
            rows,
            schema_version=H_PROPOSAL_SCHEMA + f"-{name}",
            metadata={
                "protocol_file_sha256": file_sha256(Path(args.protocol)),
                "pre_annotation_authorization_file_sha256": file_sha256(
                    Path(args.pre_annotation_authorization)
                ),
                "code_commit": code_commit,
            },
        )
        for name, rows in (("all", proposals), ("smoke", smoke), ("reserve", reserve))
    }
    report = {
        "schema_version": "clir-h0-v7-proposal-freeze-report",
        "status": "PASS_H0_V7_PROPOSAL_FREEZE",
        "annotation_started": False,
        "reserve_opened": False,
        "feature_extraction_started": False,
        "training_started": False,
        "code_commit": code_commit,
        "protocol_file_sha256": file_sha256(Path(args.protocol)),
        "pre_annotation_authorization_file_sha256": file_sha256(
            Path(args.pre_annotation_authorization)
        ),
        "materialization_validation": h_validation,
        "proposal": proposal_report,
        "split": split_report,
        "files": {
            name: {
                "path": str(path),
                "file_sha256": manifests[name]["file_sha256"],
                "ordered_rows_sha256": manifests[name]["ordered_rows_sha256"],
                "row_count": manifests[name]["row_count"],
                "sidecar_file_sha256": file_sha256(
                    path.with_suffix(path.suffix + ".manifest.json")
                ),
            }
            for name, path in paths.items()
        },
        "next_gate": "construct_smoke_only_blind_packages",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _verify_h_proposals(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    protocol, output_root, _, _, _ = _verify_materialized_h(args)
    report_path = output_root / "proposals/proposal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_H0_V7_PROPOSAL_FREEZE":
        raise ValueError("H proposal report is not a PASS")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("all", "smoke", "reserve"):
        path = output_root / f"proposals/h_{name}_proposals.jsonl"
        if name == "all":
            path = output_root / "proposals/h_proposals_all.jsonl"
        payload, sidecar = _read_published_jsonl(
            path, expected_schema=H_PROPOSAL_SCHEMA + f"-{name}"
        )
        record = report["files"][name]
        if (
            record["file_sha256"] != sidecar["file_sha256"]
            or record["ordered_rows_sha256"] != sidecar["ordered_rows_sha256"]
            or int(record["row_count"]) != len(payload)
        ):
            raise ValueError(f"H {name} proposal record drift")
        rows[name] = payload
    if (
        len(rows["all"]) != 800
        or len(rows["smoke"]) != 80
        or len(rows["reserve"]) != 720
    ):
        raise ValueError("H proposal partition counts differ from protocol")
    return protocol, output_root, rows, report


def command_package_h(args: argparse.Namespace) -> None:
    protocol, output_root, proposals, proposal_report = _verify_h_proposals(args)
    code_commit = _require_clean_execution("ranking-v7 H package construction")
    stage = str(args.stage)
    if stage not in {"smoke", "reserve"}:
        raise ValueError("package stage must be smoke or reserve")
    if stage == "reserve":
        smoke_report_path = output_root / "evaluation/smoke_evaluation_report.json"
        if not smoke_report_path.exists():
            raise RuntimeError(
                "reserve package is sealed until the smoke report passes"
            )
        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        if smoke_report.get("status") != "PASS_H0_V7_SMOKE":
            raise RuntimeError("reserve package is sealed because smoke did not pass")
    package_dir = output_root / f"packages/{stage}"
    paths = {
        "a": package_dir / "annotator_a/hallucination.jsonl",
        "b": package_dir / "annotator_b/hallucination.jsonl",
    }
    private_path = package_dir / "PRIVATE_package_index.jsonl"
    report_path = package_dir / "package_report.json"
    if (
        report_path.exists()
        or private_path.exists()
        or any(
            path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists()
            for path in paths.values()
        )
    ):
        raise FileExistsError(f"H {stage} package artifacts already exist")
    packages, private_rows, package_report = build_h_annotation_packages(
        proposals[stage],
        stage=stage,
        repeat_fraction=float(
            protocol["h_acquisition"]["annotation"]["self_repeat_fraction"]
        ),
    )
    manifests: dict[str, Any] = {}
    for annotator, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        manifests[annotator] = publish_manifest(
            path,
            packages[annotator],
            schema_version=H_PACKAGE_SCHEMA,
            metadata={
                "stage": stage,
                "annotator": annotator,
                "protocol_file_sha256": file_sha256(Path(args.protocol)),
                "proposal_report_file_sha256": file_sha256(
                    output_root / "proposals/proposal_report.json"
                ),
                "code_commit": code_commit,
            },
        )
    private_manifest = publish_manifest(
        private_path,
        private_rows,
        schema_version="clir-h0-v7-private-package-index",
        metadata={"stage": stage, "code_commit": code_commit},
    )
    report = {
        "schema_version": "clir-h0-v7-package-report",
        "status": f"PASS_H0_V7_{stage.upper()}_PACKAGE",
        "stage": stage,
        "annotation_started": False,
        "code_commit": code_commit,
        "protocol_file_sha256": file_sha256(Path(args.protocol)),
        "proposal_report_file_sha256": file_sha256(
            output_root / "proposals/proposal_report.json"
        ),
        "proposal_file_sha256": proposal_report["files"][stage]["file_sha256"],
        "package": package_report,
        "public": {
            annotator: {
                "path": str(path),
                "file_sha256": manifests[annotator]["file_sha256"],
                "ordered_rows_sha256": manifests[annotator]["ordered_rows_sha256"],
                "row_count": manifests[annotator]["row_count"],
            }
            for annotator, path in paths.items()
        },
        "private": {
            "path": str(private_path),
            "file_sha256": private_manifest["file_sha256"],
            "ordered_rows_sha256": private_manifest["ordered_rows_sha256"],
            "row_count": private_manifest["row_count"],
        },
        "next_gate": (
            "dual_ai_smoke_annotation_then_raw_gate"
            if stage == "smoke"
            else "dual_ai_reserve_annotation_then_finalization"
        ),
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_evaluate_h_smoke(args: argparse.Namespace) -> None:
    protocol, output_root, _, _ = _verify_h_proposals(args)
    package_dir = output_root / "packages/smoke"
    package_report = json.loads(
        (package_dir / "package_report.json").read_text(encoding="utf-8")
    )
    if package_report.get("status") != "PASS_H0_V7_SMOKE_PACKAGE":
        raise ValueError("H smoke package report is not a PASS")
    public = {
        annotator: _read_published_jsonl(
            package_dir / f"annotator_{annotator}/hallucination.jsonl",
            expected_schema=H_PACKAGE_SCHEMA,
        )[0]
        for annotator in ("a", "b")
    }
    private_rows = _read_published_jsonl(
        package_dir / "PRIVATE_package_index.jsonl",
        expected_schema="clir-h0-v7-private-package-index",
    )[0]
    labels = {
        "a": read_jsonl(Path(args.labels_a)),
        "b": read_jsonl(Path(args.labels_b)),
    }
    _, evaluation = evaluate_h_package_labels(
        public_by_annotator=public,
        private_rows=private_rows,
        labels_by_annotator=labels,
    )
    gate = smoke_gate(evaluation, protocol)
    report = {
        "schema_version": "clir-h0-v7-smoke-evaluation-report",
        "status": gate["status"],
        "protocol_file_sha256": file_sha256(Path(args.protocol)),
        "package_report_file_sha256": file_sha256(package_dir / "package_report.json"),
        "labels_a_path": str(Path(args.labels_a).resolve()),
        "labels_a_file_sha256": file_sha256(Path(args.labels_a)),
        "labels_b_path": str(Path(args.labels_b).resolve()),
        "labels_b_file_sha256": file_sha256(Path(args.labels_b)),
        "evaluation": evaluation,
        "gate": gate,
        "reserve_annotation_allowed": gate["pass"],
        "feature_extraction_allowed": False,
        "training_allowed": False,
    }
    report_path = output_root / "evaluation/smoke_evaluation_report.json"
    if report_path.exists():
        raise FileExistsError("H smoke evaluation report already exists")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run a read-only source/capacity audit")
    audit.add_argument("--cache-dir")
    audit.set_defaults(func=command_audit)
    freeze = subparsers.add_parser(
        "freeze", help="freeze pre-rollout manifests from a clean commit"
    )
    freeze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    freeze.add_argument("--cache-dir")
    freeze.set_defaults(func=command_freeze)
    verify = subparsers.add_parser(
        "verify", help="hash-check and independently recompute the freeze"
    )
    verify.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    verify.add_argument("--cache-dir")
    verify.set_defaults(func=command_verify)
    rollout = subparsers.add_parser("rollout", help="generate one authorized GPU shard")
    rollout.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    rollout.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    rollout.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify_rollouts = subparsers.add_parser(
        "verify-rollouts", help="verify completed authorized rollout shards"
    )
    verify_rollouts.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    verify_rollouts.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    verify_rollouts.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    verify_rollouts.add_argument("--shard-id", action="append")
    verify_rollouts.add_argument("--require-complete", action="store_true")
    verify_rollouts.set_defaults(func=command_verify_rollouts)
    merge = subparsers.add_parser(
        "merge-rollouts", help="verify and merge all 32,000 raw trajectories"
    )
    merge.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    merge.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    merge.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    merge.set_defaults(func=command_merge_rollouts)
    materialize_h = subparsers.add_parser(
        "materialize-h",
        help="apply the authorized checker and exact-token H unitizer",
    )
    materialize_h.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    materialize_h.add_argument(
        "--pre-annotation-authorization",
        default=str(DEFAULT_PRE_ANNOTATION_AUTHORIZATION),
    )
    materialize_h.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    materialize_h.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    materialize_h.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    materialize_h.add_argument("--cache-dir")
    materialize_h.set_defaults(func=command_materialize_h)
    propose_h = subparsers.add_parser(
        "propose-h", help="freeze deterministic H0 proposals and smoke/reserve split"
    )
    propose_h.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    propose_h.add_argument(
        "--pre-annotation-authorization",
        default=str(DEFAULT_PRE_ANNOTATION_AUTHORIZATION),
    )
    propose_h.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    propose_h.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    propose_h.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    propose_h.set_defaults(func=command_propose_h)
    package_h = subparsers.add_parser(
        "package-h", help="construct a blind H0 smoke or authorized reserve package"
    )
    package_h.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    package_h.add_argument(
        "--pre-annotation-authorization",
        default=str(DEFAULT_PRE_ANNOTATION_AUTHORIZATION),
    )
    package_h.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    package_h.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    package_h.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    package_h.add_argument("--stage", choices=("smoke", "reserve"), required=True)
    package_h.set_defaults(func=command_package_h)
    evaluate_h_smoke = subparsers.add_parser(
        "evaluate-h-smoke",
        help="validate dual-AI H0 smoke labels and apply frozen gates",
    )
    evaluate_h_smoke.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    evaluate_h_smoke.add_argument(
        "--pre-annotation-authorization",
        default=str(DEFAULT_PRE_ANNOTATION_AUTHORIZATION),
    )
    evaluate_h_smoke.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    evaluate_h_smoke.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    evaluate_h_smoke.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    evaluate_h_smoke.add_argument("--labels-a", required=True)
    evaluate_h_smoke.add_argument("--labels-b", required=True)
    evaluate_h_smoke.set_defaults(func=command_evaluate_h_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
