#!/usr/bin/env python
"""Freeze and execute the fresh, pre-label H0 acquisition supplement v7.2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from prepare_clir_h_rescue import (
    _git_dirty,
    _git_head,
    _package_version,
    _project_path,
    _prompt,
    _read_published_jsonl,
    _seed,
    _shard_paths,
    _utc_now,
    _validate_rows,
)
from prepare_clir_ranking import (
    _combine_exclusions,
    _load_extended_sources,
    load_protocol,
)
from src.clir_h_supplement import (
    SUPPLEMENT_SCHEMA,
    build_supplement_shards,
    select_supplement_queries,
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
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_PROTOCOL = PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"
DEFAULT_SUPPLEMENT = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/supplement_protocol_v7_2.json"
)
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/supplement_authorization_v7_2.json"
)
DEFAULT_PRE_ANNOTATION_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_annotation"
)
DEFAULT_SUPPLEMENT_ROOT = DEFAULT_PRE_ANNOTATION_ROOT / "supplement_v7_2"
DEFAULT_PRE_ROLLOUT = DEFAULT_SUPPLEMENT_ROOT / "pre_rollout"
BASE_PRE_ROLLOUT = PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_rollout"


def _verify_parent_hashes(
    supplement: Mapping[str, Any], *, base_protocol_path: Path
) -> None:
    parent = supplement["parent"]
    expected = {
        "base_protocol_file_sha256": file_sha256(base_protocol_path),
        "base_pre_rollout_registry_file_sha256": file_sha256(
            BASE_PRE_ROLLOUT / "manifest_registry.json"
        ),
        "base_ranking_queries_file_sha256": file_sha256(
            BASE_PRE_ROLLOUT / "ranking_queries.jsonl"
        ),
        "base_h_queries_file_sha256": file_sha256(
            BASE_PRE_ROLLOUT / "h_acquisition_queries.jsonl"
        ),
        "base_h_materialized_file_sha256": file_sha256(
            DEFAULT_PRE_ANNOTATION_ROOT / "materialized/h_materialized.jsonl"
        ),
        "yield_rescue_amendment_file_sha256": file_sha256(
            PROJECT_ROOT
            / "configs/ranking_expansion_v7/yield_rescue_amendment_v7_1.json"
        ),
        "yield_rescue_authorization_file_sha256": file_sha256(
            PROJECT_ROOT
            / "configs/ranking_expansion_v7/yield_rescue_authorization_v7_1.json"
        ),
        "yield_rescue_rollout_report_file_sha256": file_sha256(
            DEFAULT_PRE_ANNOTATION_ROOT / "yield_rescue/rollout_completion_report.json"
        ),
        "yield_rescue_materialization_report_file_sha256": file_sha256(
            DEFAULT_PRE_ANNOTATION_ROOT
            / "yield_rescue/materialized/materialization_report.json"
        ),
        "yield_rescue_materialized_file_sha256": file_sha256(
            DEFAULT_PRE_ANNOTATION_ROOT / "yield_rescue/materialized/rescue_rows.jsonl"
        ),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"fresh supplement parent {key} mismatch")
    if parent.get("ai_labels_seen") is not False:
        raise ValueError("fresh supplement must be frozen before any AI labels")


def _survival_by_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    output: Counter[str] = Counter()
    for candidates in by_query.values():
        first = candidates[0]
        cell = "|".join(
            (
                str(first["h_target_checker_status"]),
                str(first["h_label_split"]),
                str(first["source"]),
            )
        )
        if any(
            row.get("unitization_status") == "ok"
            and row.get("checker_status") == row.get("h_target_checker_status")
            and row.get("finish_reason") != "length"
            and bool(row.get("eligible_for_supervision"))
            and int(row.get("material_claim_count", 0)) >= 5
            for row in candidates
        ):
            output[cell] += 1
    return dict(sorted(output.items()))


def load_supplement(
    path: Path, *, base_protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_protocol(base_protocol_path)
    supplement = json.loads(path.read_text(encoding="utf-8"))
    if supplement.get("schema_version") != SUPPLEMENT_SCHEMA:
        raise ValueError("unsupported fresh H0 supplement protocol")
    if supplement.get("status") != "FROZEN_PREPARATION_ROLLOUT_NOT_STARTED":
        raise ValueError("fresh H0 supplement is not at its preparation gate")
    scope = supplement.get("execution_authorization", {})
    if (
        scope.get("source_audit_allowed") is not True
        or scope.get("pre_rollout_freeze_allowed") is not True
    ):
        raise ValueError("fresh supplement source planning is not authorized")
    for forbidden in (
        "rollout_allowed",
        "ai_annotation_allowed",
        "feature_extraction_allowed",
        "training_allowed",
    ):
        if scope.get(forbidden) is not False:
            raise ValueError(f"fresh supplement must keep {forbidden}=false")
    _verify_parent_hashes(supplement, base_protocol_path=base_protocol_path)
    original, _ = _read_published_jsonl(
        DEFAULT_PRE_ANNOTATION_ROOT / "materialized/h_materialized.jsonl",
        expected_schema="clir-ranking-v7-h-materialized",
    )
    rescued, _ = _read_published_jsonl(
        DEFAULT_PRE_ANNOTATION_ROOT / "yield_rescue/materialized/rescue_rows.jsonl",
        expected_schema="clir-h0-v7.1-yield-rescue-materialized",
    )
    observed = supplement["observed_final_fail_yield"]
    combined = [*original, *rescued]
    if len(combined) != int(observed["combined_candidate_rows"]):
        raise ValueError("fresh supplement parent candidate-row count drift")
    if len({str(row["query_id"]) for row in combined}) != int(
        observed["combined_original_queries"]
    ):
        raise ValueError("fresh supplement parent query count drift")
    if _survival_by_cell(combined) != observed["surviving_queries_by_cell"]:
        raise ValueError("fresh supplement observed cell survival drift")
    targets = {
        "numeric_match|dev|math": 50,
        "numeric_match|train|math": 100,
        "numeric_mismatch|dev|gsm8k": 33,
    }
    shortages = {
        cell: target - int(observed["surviving_queries_by_cell"][cell])
        for cell, target in targets.items()
    }
    if shortages != observed["remaining_shortages"]:
        raise ValueError("fresh supplement shortage computation drift")
    return base, supplement


def _compact_query(row: Mapping[str, Any]) -> dict[str, Any]:
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
        "cluster_id",
        "cluster_split_priority",
        "query_priority",
        "supplement_stratum",
        "supplement_cell",
        "role",
        "role_priority",
        "h_target_checker_status",
        "h_label_split",
        "prompt_token_count",
        "prompt_token_ids",
    )
    return {key: row[key] for key in keys if key in row}


def build_plan(
    base: Mapping[str, Any],
    supplement: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> dict[str, Any]:
    expanded = deepcopy(base)
    expanded["sources"]["math"]["allowed_levels"] = [2, 3, 4, 5]
    expanded["sources"]["math"]["minimum_official_solution_words"] = int(
        supplement["fresh_source_pool"]["math"]["minimum_official_solution_words"]
    )
    sources, source_report = _load_extended_sources(expanded, cache_dir=cache_dir)
    candidates, filter_report = build_source_candidates(
        sources, expanded, required_schema="clir-ranking-h-expansion-v7"
    )
    historical, historical_report = _combine_exclusions(expanded)
    ranking, _ = _read_published_jsonl(
        BASE_PRE_ROLLOUT / "ranking_queries.jsonl",
        expected_schema="clir-ranking-v7-evaluation-queries",
    )
    h_queries, _ = _read_published_jsonl(
        BASE_PRE_ROLLOUT / "h_acquisition_queries.jsonl",
        expected_schema="clir-ranking-v7-h-acquisition-queries",
    )
    excluded_ids = {str(row["query_id"]) for row in historical}
    excluded_ids.update(str(row["query_id"]) for row in [*ranking, *h_queries])
    by_id = {str(row["query_id"]): row for row in sources}
    missing = sorted(excluded_ids - set(by_id))
    if missing:
        raise ValueError(f"fresh supplement lacks {len(missing)} exclusion anchors")
    anchors = [by_id[query_id] for query_id in sorted(excluded_ids)]
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
        namespace=str(supplement["fresh_source_pool"]["cluster_namespace"]),
    )
    selected, selection_report = select_supplement_queries(selectable, supplement)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("fresh supplement freeze requires transformers") from exc
    generation = base["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    prompt_counts: list[int] = []
    with_prompts: list[dict[str, Any]] = []
    for raw in selected:
        row = dict(raw)
        messages = [
            {
                "role": "user",
                "content": _prompt(
                    str(row["question"]), str(generation["prompt_template"])
                ),
            }
        ]
        ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        ]
        if len(ids) > int(generation["maximum_prompt_tokens"]):
            raise ValueError(f"{row['query_id']}: fresh supplement prompt too long")
        row["prompt_token_ids"] = ids
        row["prompt_token_count"] = len(ids)
        prompt_counts.append(len(ids))
        with_prompts.append(_compact_query(row))
    shards = build_supplement_shards(with_prompts, supplement)
    used_ids = {str(row["query_id"]) for row in [*ranking, *h_queries]}
    used_clusters = {str(row["cluster_id"]) for row in [*ranking, *h_queries]}
    if used_ids & {str(row["query_id"]) for row in with_prompts}:
        raise AssertionError("fresh supplement query overlaps v7")
    if used_clusters & {str(row["cluster_id"]) for row in with_prompts}:
        raise AssertionError("fresh supplement cluster overlaps v7")
    return {
        "queries": with_prompts,
        "shards": shards,
        "reports": {
            "source": source_report,
            "source_filter": filter_report,
            "historical_exclusions": historical_report,
            "historical_plus_v7_exclusion_ids": len(excluded_ids),
            "clusters": cluster_report,
            "cluster_rows_recomputed": len(clusters),
            "selection": selection_report,
            "prompt_tokens": {
                "count": len(prompt_counts),
                "min": min(prompt_counts),
                "max": max(prompt_counts),
                "mean": sum(prompt_counts) / len(prompt_counts),
            },
            "query_overlap_with_v7": 0,
            "cluster_overlap_with_v7": 0,
            "test_files_read": False,
        },
    }


def command_audit(args: argparse.Namespace) -> None:
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    base, supplement = load_supplement(supplement_path, base_protocol_path=base_path)
    plan = build_plan(base, supplement, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "status": "PASS_H0_V7_2_FRESH_SUPPLEMENT_SOURCE_AUDIT",
                "base_protocol_file_sha256": file_sha256(base_path),
                "supplement_file_sha256": file_sha256(supplement_path),
                "query_count": len(plan["queries"]),
                "candidate_rows": sum(
                    int(row["expected_candidate_rows"]) for row in plan["shards"]
                ),
                **plan["reports"],
                "artifacts_written": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_freeze(args: argparse.Namespace) -> None:
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    output = Path(args.pre_rollout_dir).resolve()
    if _git_dirty():
        raise RuntimeError("fresh supplement freeze requires a clean Git commit")
    if (output / "manifest_registry.json").exists():
        raise FileExistsError("fresh supplement is already frozen")
    base, supplement = load_supplement(supplement_path, base_protocol_path=base_path)
    plan = build_plan(base, supplement, cache_dir=args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "supplement_queries.jsonl"
    query_manifest = publish_manifest(
        query_path,
        plan["queries"],
        schema_version="clir-h0-v7.2-fresh-supplement-queries",
        metadata=plan["reports"]["selection"],
    )
    shards_path = output / "rollout_shards.json"
    atomic_write_json(shards_path, plan["shards"])
    audit_path = output / "source_audit.json"
    atomic_write_json(audit_path, plan["reports"])
    report = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-freeze-report",
        "status": "PASS_H0_V7_2_FRESH_SUPPLEMENT_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "code_dirty": False,
        "base_protocol_file_sha256": file_sha256(base_path),
        "supplement_file_sha256": file_sha256(supplement_path),
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
        "query_overlap_with_v7": 0,
        "cluster_overlap_with_v7": 0,
        "ai_annotation_started": False,
        "next_gate": "independent_recompute_then_hash_bound_rollout_authorization",
    }
    report_path = output / "freeze_report.json"
    atomic_write_json(report_path, report)
    registry = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-registry",
        "status": report["status"],
        "code_commit": report["code_commit"],
        "supplement_file_sha256": report["supplement_file_sha256"],
        "query_file_sha256": report["query_file_sha256"],
        "query_sidecar_file_sha256": report["query_sidecar_file_sha256"],
        "query_ordered_rows_sha256": report["query_ordered_rows_sha256"],
        "shards_file_sha256": report["shards_file_sha256"],
        "shards_canonical_sha256": report["shards_canonical_sha256"],
        "source_audit_file_sha256": report["source_audit_file_sha256"],
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


def command_verify(args: argparse.Namespace) -> None:
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    output = Path(args.pre_rollout_dir).resolve()
    base, supplement = load_supplement(supplement_path, base_protocol_path=base_path)
    registry_path = output / "manifest_registry.json"
    report_path = output / "freeze_report.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("supplement_file_sha256") != file_sha256(supplement_path):
        raise ValueError("fresh supplement registry protocol hash drift")
    if registry.get("freeze_report_file_sha256") != file_sha256(report_path):
        raise ValueError("fresh supplement freeze report hash drift")
    frozen_queries, sidecar = _read_published_jsonl(
        output / "supplement_queries.jsonl",
        expected_schema="clir-h0-v7.2-fresh-supplement-queries",
    )
    frozen_shards = json.loads(
        (output / "rollout_shards.json").read_text(encoding="utf-8")
    )
    frozen_audit = json.loads((output / "source_audit.json").read_text())
    if sidecar["file_sha256"] != registry["query_file_sha256"]:
        raise ValueError("fresh supplement query manifest drift")
    if file_sha256(output / "rollout_shards.json") != registry["shards_file_sha256"]:
        raise ValueError("fresh supplement shard manifest drift")
    if (
        file_sha256(output / "source_audit.json")
        != registry["source_audit_file_sha256"]
    ):
        raise ValueError("fresh supplement source audit drift")
    recomputed = build_plan(base, supplement, cache_dir=args.cache_dir)
    if recomputed["queries"] != frozen_queries:
        raise ValueError("fresh supplement independent query recomputation drift")
    if recomputed["shards"] != frozen_shards:
        raise ValueError("fresh supplement independent shard recomputation drift")
    if recomputed["reports"] != frozen_audit:
        raise ValueError("fresh supplement independent source audit drift")
    verification = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-verification",
        "status": "PASS_H0_V7_2_FRESH_SUPPLEMENT_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "supplement_file_sha256": file_sha256(supplement_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "freeze_report_file_sha256": file_sha256(report_path),
        "query_count": len(frozen_queries),
        "shard_count": len(frozen_shards),
        "candidate_rows": sum(
            int(row["expected_candidate_rows"]) for row in frozen_shards
        ),
        "query_overlap_with_v7": 0,
        "cluster_overlap_with_v7": 0,
        "rollout_allowed": False,
        "next_gate": "hash_bound_rollout_authorization",
    }
    verification_path = output / "independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        stable = set(verification) - {"verified_at_utc"}
        if any(old.get(key) != verification[key] for key in stable):
            raise ValueError("fresh supplement existing verification drift")
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


def load_authorization(
    path: Path,
    *,
    base_protocol_path: Path,
    supplement_path: Path,
    pre_rollout_dir: Path,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "clir-h0-v7.2-fresh-supplement-authorization"
    ):
        raise ValueError("unsupported fresh supplement authorization")
    if authorization.get("status") != ("AUTHORIZED_ONE_FRESH_SUPPLEMENT_ROLLOUT_ONLY"):
        raise ValueError("fresh supplement rollout is not authorized")
    scope = authorization.get("authorized_scope", {})
    if scope.get("one_fresh_supplement_rollout") is not True or any(
        scope.get(name) is not False
        for name in (
            "adaptive_additional_sampling",
            "ai_annotation",
            "query_role_or_split_change",
            "threshold_or_quota_change",
            "feature_extraction",
            "training",
        )
    ):
        raise ValueError("fresh supplement authorization scope drift")
    parent = authorization["frozen_parent"]
    expected = {
        "base_protocol_file_sha256": file_sha256(base_protocol_path),
        "supplement_file_sha256": file_sha256(supplement_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "freeze_report_file_sha256": file_sha256(
            pre_rollout_dir / "freeze_report.json"
        ),
        "independent_verification_file_sha256": file_sha256(
            pre_rollout_dir / "independent_verification.json"
        ),
    }
    for key, value in expected.items():
        if parent.get(key) != value:
            raise ValueError(f"fresh supplement authorization {key} mismatch")
    runtime = authorization["runtime_contract"]
    if int(runtime["maximum_concurrent_shards"]) != int(
        json.loads(supplement_path.read_text(encoding="utf-8"))["rollout_shards"]
    ):
        raise ValueError("fresh supplement authorization shard concurrency drift")
    if int(runtime["candidate_count"]) != int(
        json.loads(supplement_path.read_text(encoding="utf-8"))["candidate_count"]
    ):
        raise ValueError("fresh supplement authorization candidate count drift")
    return authorization


def _load_runtime(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    Path,
]:
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.pre_rollout_dir).resolve()
    supplement_root = Path(args.supplement_root).resolve()
    base, supplement = load_supplement(supplement_path, base_protocol_path=base_path)
    authorization = load_authorization(
        authorization_path,
        base_protocol_path=base_path,
        supplement_path=supplement_path,
        pre_rollout_dir=pre_rollout,
    )
    expected_root = _project_path(
        authorization["runtime_contract"]["output_root"]
    ).resolve()
    if supplement_root != expected_root:
        raise ValueError("fresh supplement output root differs from authorization")
    queries, _ = _read_published_jsonl(
        pre_rollout / "supplement_queries.jsonl",
        expected_schema="clir-h0-v7.2-fresh-supplement-queries",
    )
    shards = json.loads(
        (pre_rollout / "rollout_shards.json").read_text(encoding="utf-8")
    )
    query_by_id = {str(row["query_id"]): row for row in queries}
    if len(query_by_id) != int(supplement["query_count"]):
        raise ValueError("fresh supplement frozen query count drift")
    return base, supplement, authorization, shards, query_by_id, supplement_root


def _select_shard(shards: Sequence[Mapping[str, Any]], shard_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected one fresh supplement shard {shard_id}")
    return matches[0]


def verify_shard(
    *,
    shard: Mapping[str, Any],
    query_by_id: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Any],
    supplement: Mapping[str, Any],
    base_protocol_path: Path,
    supplement_path: Path,
    authorization_path: Path,
    pre_rollout_dir: Path,
    supplement_root: Path,
) -> dict[str, Any]:
    output, sidecar_path, completion_path = _shard_paths(supplement_root, shard)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE_VERIFIED_H0_V7_2_SUPPLEMENT_SHARD":
        raise ValueError(f"invalid fresh supplement marker: {completion_path}")
    expected = {
        "shard_spec_sha256": canonical_sha256(shard),
        "authorization_file_sha256": file_sha256(authorization_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise ValueError(f"fresh supplement completion binding drift: {output}")
    if (
        file_sha256(output) != completion["file_sha256"]
        or file_sha256(sidecar_path) != completion["sidecar_file_sha256"]
    ):
        raise ValueError(f"fresh supplement shard file hash drift: {output}")
    rows = read_jsonl(output)
    if canonical_sha256(rows) != completion["ordered_rows_sha256"]:
        raise ValueError(f"fresh supplement ordered rows drift: {output}")
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=base,
        amendment=supplement,
        protocol_path=base_protocol_path,
        amendment_path=supplement_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    if validation != completion["validation"]:
        raise ValueError(f"fresh supplement validation drift: {output}")
    return {
        "status": "PASS_H0_V7_2_SUPPLEMENT_SHARD_VERIFY",
        "shard_id": shard["shard_id"],
        "path": str(output),
        "file_sha256": completion["file_sha256"],
        "row_count": len(rows),
        "validation": validation,
        "runtime": completion["runtime"],
    }


def command_rollout(args: argparse.Namespace) -> None:
    base, supplement, authorization, shards, query_by_id, supplement_root = (
        _load_runtime(args)
    )
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    shard = _select_shard(shards, args.shard_id)
    calibration_id = str(authorization["runtime_contract"]["first_calibration_shard"])
    if shard["shard_id"] != calibration_id:
        calibration = _select_shard(shards, calibration_id)
        _, _, marker = _shard_paths(supplement_root, calibration)
        if not marker.exists():
            raise RuntimeError(f"{calibration_id} must verify before other shards")
        verify_shard(
            shard=calibration,
            query_by_id=query_by_id,
            base=base,
            supplement=supplement,
            base_protocol_path=base_path,
            supplement_path=supplement_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            supplement_root=supplement_root,
        )
    output, sidecar_path, completion_path = _shard_paths(supplement_root, shard)
    if completion_path.exists():
        print(
            json.dumps(
                verify_shard(
                    shard=shard,
                    query_by_id=query_by_id,
                    base=base,
                    supplement=supplement,
                    base_protocol_path=base_path,
                    supplement_path=supplement_path,
                    authorization_path=authorization_path,
                    pre_rollout_dir=pre_rollout_dir,
                    supplement_root=supplement_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("incomplete fresh supplement shard exists")
    if _git_dirty():
        raise RuntimeError("fresh supplement rollout requires a clean Git commit")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("fresh supplement rollout requires torch and vLLM") from exc
    generation = base["generation"]
    runtime = authorization["runtime_contract"]
    if _package_version("vllm") != generation["backend_version"]:
        raise ValueError("fresh supplement vLLM version drift")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each fresh supplement shard must see one GPU")
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
    sampling: list[Any] = []
    candidate_count = int(shard["candidate_count"])
    for query in query_rows:
        messages = [
            {
                "role": "user",
                "content": _prompt(
                    str(query["question"]), str(generation["prompt_template"])
                ),
            }
        ]
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
        if prompt_ids != query["prompt_token_ids"]:
            raise ValueError(f"{query['query_id']}: frozen prompt IDs drift")
        sampling.append(
            SamplingParams(
                n=candidate_count,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_seed(supplement, str(query["query_id"])),
            )
        )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    if len(request_outputs) != len(query_rows):
        raise ValueError("vLLM returned a different fresh supplement query count")
    finished_at = _utc_now()
    elapsed = time.monotonic() - started_clock
    provenance = {
        "protocol_file_sha256": file_sha256(base_path),
        "amendment_file_sha256": file_sha256(supplement_path),
        "supplement_protocol_file_sha256": file_sha256(supplement_path),
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
    rows: list[dict[str, Any]] = []
    start = int(shard["candidate_index_start"])
    for query, request_output in zip(query_rows, request_outputs):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        if prompt_ids != query["prompt_token_ids"]:
            raise ValueError(f"{query['query_id']}: vLLM prompt IDs drift")
        candidates = sorted(request_output.outputs, key=lambda value: int(value.index))
        if [int(value.index) for value in candidates] != list(range(candidate_count)):
            raise ValueError("fresh supplement candidate indices are not contiguous")
        for candidate in candidates:
            index = start + int(candidate.index)
            output_ids = [int(value) for value in candidate.token_ids]
            if not output_ids:
                raise ValueError(f"{query['query_id']}: empty fresh output")
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": index,
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
                    "supplement_cell": query["supplement_cell"],
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
                    "sampling_seed": _seed(supplement, str(query["query_id"])),
                    "provenance": provenance,
                }
            )
    validation = _validate_rows(
        rows,
        shard=shard,
        query_by_id=query_by_id,
        protocol=base,
        amendment=supplement,
        protocol_path=base_path,
        amendment_path=supplement_path,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-h0-v7.2-fresh-supplement-raw-shard",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-shard-completion",
        "status": "COMPLETE_VERIFIED_H0_V7_2_SUPPLEMENT_SHARD",
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
                base=base,
                supplement=supplement,
                base_protocol_path=base_path,
                supplement_path=supplement_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                supplement_root=supplement_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def command_verify_rollouts(args: argparse.Namespace) -> None:
    base, supplement, _, shards, query_by_id, supplement_root = _load_runtime(args)
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    reports: list[dict[str, Any]] = []
    missing: list[str] = []
    for shard in shards:
        _, _, marker = _shard_paths(supplement_root, shard)
        if not marker.exists():
            missing.append(str(shard["shard_id"]))
            continue
        reports.append(
            verify_shard(
                shard=shard,
                query_by_id=query_by_id,
                base=base,
                supplement=supplement,
                base_protocol_path=base_path,
                supplement_path=supplement_path,
                authorization_path=authorization_path,
                pre_rollout_dir=pre_rollout_dir,
                supplement_root=supplement_root,
            )
        )
    if args.require_complete and missing:
        raise ValueError(f"missing fresh supplement shards: {missing}")
    print(
        json.dumps(
            {
                "status": (
                    "PASS_ALL_H0_V7_2_SUPPLEMENT_SHARDS"
                    if not missing
                    else "PARTIAL_H0_V7_2_SUPPLEMENT_SHARDS"
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
    base, supplement, _, shards, query_by_id, supplement_root = _load_runtime(args)
    base_path = Path(args.base_protocol).resolve()
    supplement_path = Path(args.supplement).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout_dir).resolve()
    output = supplement_root / "rollouts/combined_raw.jsonl"
    report_path = supplement_root / "rollout_completion_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("fresh supplement merged artifacts already exist")
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for shard in shards:
        report = verify_shard(
            shard=shard,
            query_by_id=query_by_id,
            base=base,
            supplement=supplement,
            base_protocol_path=base_path,
            supplement_path=supplement_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            supplement_root=supplement_root,
        )
        reports.append(report)
        rows.extend(read_jsonl(Path(report["path"])))
    population = validate_rollout_population(
        rows, candidate_count=int(supplement["candidate_count"])
    )
    if len(rows) != int(supplement["expected_candidate_rows"]):
        raise ValueError("fresh supplement merged row count drift")
    commits = {str(row["validation"]["code_commit"]) for row in reports}
    if len(commits) != 1:
        raise ValueError("fresh supplement shards use mixed code commits")
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-h0-v7.2-fresh-supplement-combined-raw",
        metadata={
            "base_protocol_file_sha256": file_sha256(base_path),
            "supplement_file_sha256": file_sha256(supplement_path),
            "authorization_file_sha256": file_sha256(authorization_path),
            **population,
        },
    )
    report = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-rollout-report",
        "status": "PASS_ALL_2880_H0_V7_2_FRESH_SUPPLEMENT_ROWS",
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "rows": len(rows),
        "population": population,
        "code_commit": next(iter(commits)),
        "shards": reports,
        "next_gate": "checker_and_exact_token_unitization_then_final_800_proposal_gate",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_materialize(args: argparse.Namespace) -> None:
    base, supplement, _, _, _, supplement_root = _load_runtime(args)
    if _git_dirty():
        raise RuntimeError("fresh supplement materialization requires clean Git")
    raw_path = supplement_root / "rollouts/combined_raw.jsonl"
    raw_rows, raw_sidecar = _read_published_jsonl(
        raw_path,
        expected_schema="clir-h0-v7.2-fresh-supplement-combined-raw",
    )
    output = supplement_root / "materialized/supplement_rows.jsonl"
    report_path = supplement_root / "materialized/materialization_report.json"
    if output.exists() or report_path.exists():
        raise FileExistsError("fresh supplement materialization already exists")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("fresh supplement unitization requires transformers") from exc
    generation = base["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    processed, health = materialize_scale_rows(
        raw_rows,
        tokenizer,
        checker_version=str(base["checker"]["checker_version"]),
        unitizer_version=str(base["checker"]["unitizer_version"]),
    )
    validation = validate_scale_materialized_rows(
        processed,
        raw_rows=raw_rows,
        candidate_count=int(supplement["candidate_count"]),
        checker_version=str(base["checker"]["checker_version"]),
        unitizer_version=str(base["checker"]["unitizer_version"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = publish_manifest(
        output,
        processed,
        schema_version="clir-h0-v7.2-fresh-supplement-materialized",
        metadata={
            "base_protocol_file_sha256": file_sha256(Path(args.base_protocol)),
            "supplement_file_sha256": file_sha256(Path(args.supplement)),
            "authorization_file_sha256": file_sha256(Path(args.authorization)),
            "raw_file_sha256": raw_sidecar["file_sha256"],
            "code_commit": _git_head(),
            **health,
        },
    )
    original_path = DEFAULT_PRE_ANNOTATION_ROOT / "materialized/h_materialized.jsonl"
    rescue_path = (
        DEFAULT_PRE_ANNOTATION_ROOT / "yield_rescue/materialized/rescue_rows.jsonl"
    )
    report = {
        "schema_version": "clir-h0-v7.2-fresh-supplement-materialization-report",
        "status": "PASS_H0_V7_2_FRESH_SUPPLEMENT_MATERIALIZATION",
        "rows": len(processed),
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(
            output.with_suffix(output.suffix + ".manifest.json")
        ),
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "parent_h_materialized_file_sha256": file_sha256(original_path),
        "parent_rescue_materialized_file_sha256": file_sha256(rescue_path),
        "supplement_protocol_file_sha256": file_sha256(Path(args.supplement)),
        "pre_rollout_registry_file_sha256": file_sha256(
            Path(args.pre_rollout_dir).resolve() / "manifest_registry.json"
        ),
        "independent_verification_file_sha256": file_sha256(
            Path(args.pre_rollout_dir).resolve() / "independent_verification.json"
        ),
        "authorization_file_sha256": file_sha256(Path(args.authorization)),
        "query_overlap_with_v7": 0,
        "cluster_overlap_with_v7": 0,
        "health": health,
        "validation": validation,
        "adaptive_additional_sampling_allowed": False,
        "next_gate": "rerun_original_800_H_proposal_freeze_once",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", default=str(DEFAULT_BASE_PROTOCOL))
    parser.add_argument("--supplement", default=str(DEFAULT_SUPPLEMENT))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--pre-rollout-dir", default=str(DEFAULT_PRE_ROLLOUT))
    parser.add_argument("--supplement-root", default=str(DEFAULT_SUPPLEMENT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--cache-dir")
    audit.set_defaults(func=command_audit)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--cache-dir")
    freeze.set_defaults(func=command_freeze)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--cache-dir")
    verify.set_defaults(func=command_verify)
    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify_rollouts = subparsers.add_parser("verify-rollouts")
    verify_rollouts.add_argument("--require-complete", action="store_true")
    verify_rollouts.set_defaults(func=command_verify_rollouts)
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
