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
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from prepare_clir_h_rescue import _read_published_jsonl
from prepare_clir_ranking import (
    _combine_exclusions,
    _load_extended_sources,
    load_protocol as load_base_protocol,
)
from src.clir_prior_consensus_scale import (
    PROTOCOL_SCHEMA,
    QUERY_SCHEMA,
    build_acquisition_shards,
    select_acquisition_queries,
)
from src.clir_scale import build_source_candidates, build_template_clusters
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_PROTOCOL = PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v12/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12/pre_rollout"


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
    if parent["base_ranking_protocol_file_sha256"] != file_sha256(
        base_protocol_path
    ):
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
    base, protocol = load_scale_protocol(
        protocol_path, base_protocol_path=base_path
    )
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
                    int(row["expected_candidate_rows"])
                    for row in plan["shards"]
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
    base, protocol = load_scale_protocol(
        protocol_path, base_protocol_path=base_path
    )
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
    base, protocol = load_scale_protocol(
        protocol_path, base_protocol_path=base_path
    )
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
    if file_sha256(output / "source_audit.json") != registry["source_audit_file_sha256"]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", default=str(DEFAULT_BASE_PROTOCOL))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
