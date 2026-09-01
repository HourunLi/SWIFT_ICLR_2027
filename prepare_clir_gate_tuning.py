#!/usr/bin/env python
"""Freeze fresh CLIR Prior/Gate tuning and sealed-confirmation populations.

This preparation command performs no generation and uses no GPU.  It rebuilds
the pinned GSM8K/MATH train-source inventory, removes every recorded historical
query and its near-template cluster, and freezes two disjoint raw populations.
Later commands must be separately hash-authorized for rollout and scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from prepare_clir_ranking import (
    _load_extended_sources,
    load_protocol as load_base_protocol,
)
from src.clir_gate_tuning import (
    CONFIRMATION_ROLE,
    PROTOCOL_SCHEMA,
    TUNING_ROLE,
    build_query_manifests,
    build_rollout_shards,
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
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_gate_tuning_v1/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/pre_rollout"
REQUIRED_FILES = (
    "source_audit.json",
    "permanent_exclusions.jsonl",
    "template_clusters.jsonl",
    "tuning_queries.jsonl",
    "confirmation_queries.jsonl",
    "rollout_shards.json",
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


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _assert_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"pinned {label} hash drift: {observed} != {expected_sha256}")


def _load_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _assert_file(path, expected_sha256, label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pinned {label} must be a JSON object")
    return payload


def _verify_published_jsonl(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _project_path(str(spec["path"]))
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    _assert_file(path, str(spec["file_sha256"]), str(spec["label"]))
    _assert_file(
        sidecar,
        str(spec["sidecar_file_sha256"]),
        f"{spec['label']} sidecar",
    )
    rows = read_jsonl(path)
    if len(rows) != int(spec["row_count"]):
        raise ValueError(f"pinned {spec['label']} row count drift")
    if canonical_sha256(rows) != spec["ordered_rows_sha256"]:
        raise ValueError(f"pinned {spec['label']} ordered rows drift")
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    for field in ("row_count", "file_sha256", "ordered_rows_sha256"):
        if sidecar_payload.get(field) != spec[field]:
            raise ValueError(f"pinned {spec['label']} sidecar {field} drift")
    return rows


def _validate_stage_a_configs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    parent = protocol["parent"]
    ch = _load_json(
        _project_path(parent["ch_config_path"]),
        str(parent["ch_config_file_sha256"]),
        "CH config",
    )
    full = _load_json(
        _project_path(parent["full_config_path"]),
        str(parent["full_config_file_sha256"]),
        "Full config",
    )
    diagnostic_path = (
        PROJECT_ROOT / "configs/prior_gate_tuning_v1/ch_direct_prior_gate0.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if ch["training"] != full["training"] or ch["training"] != diagnostic["training"]:
        raise ValueError("Stage-A training configs differ outside model factors")
    expected = {
        "CH": (0.0, 0.0, 1.0, 1.0),
        "CH_direct_P_gate0": (1.0, 0.0, 1.0, 1.0),
        "Full_025": (1.0, 0.25, 1.0, 1.0),
    }
    configs = {"CH": ch, "CH_direct_P_gate0": diagnostic, "Full_025": full}
    observed: dict[str, Any] = {}
    ignored = {"prior_weight", "gate_prior_weight"}
    baseline_model = {
        key: value for key, value in ch["model"].items() if key not in ignored
    }
    for label, payload in configs.items():
        model = payload["model"]
        if {
            key: value for key, value in model.items() if key not in ignored
        } != baseline_model:
            raise ValueError(f"{label} differs outside Prior/Gate Stage-A factors")
        factor_tuple = (
            float(model["prior_weight"]),
            float(model["gate_prior_weight"]),
            float(model["consistency_weight"]),
            float(model["hallucination_weight"]),
        )
        if factor_tuple != expected[label]:
            raise ValueError(f"{label} Stage-A factor drift: {factor_tuple}")
        observed[label] = {
            "prior_weight": factor_tuple[0],
            "gate_prior_weight": factor_tuple[1],
            "consistency_weight": factor_tuple[2],
            "hallucination_weight": factor_tuple[3],
        }
    return {
        "diagnostic_config_path": str(diagnostic_path.relative_to(PROJECT_ROOT)),
        "diagnostic_config_file_sha256": file_sha256(diagnostic_path),
        "factors": observed,
        "nonfactor_fields_match": True,
    }


def load_protocol(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior/Gate tuning protocol")
    if protocol.get("status") != "FROZEN_PREPARATION_ROLLOUT_NOT_STARTED":
        raise ValueError("Prior/Gate tuning protocol is not at preparation gate")
    scope = protocol.get("execution_authorization", {})
    if (
        scope.get("source_audit_allowed") is not True
        or scope.get("pre_rollout_freeze_allowed") is not True
    ):
        raise ValueError("protocol does not authorize source audit/freeze")
    for forbidden in (
        "rollout_allowed",
        "checker_materialization_allowed",
        "feature_extraction_allowed",
        "training_allowed",
        "confirmation_scoring_allowed",
    ):
        if scope.get(forbidden) is not False:
            raise ValueError(f"preparation protocol must keep {forbidden}=false")

    parent = protocol["parent"]
    base_path = _project_path(parent["base_ranking_protocol_path"])
    _assert_file(
        base_path,
        str(parent["base_ranking_protocol_file_sha256"]),
        "base ranking protocol",
    )
    completion = _load_json(
        _project_path(parent["factorial_completion_path"]),
        str(parent["factorial_completion_file_sha256"]),
        "factorial completion",
    )
    if (
        completion.get("status")
        != "COMPLETE_THREE_MODULE_FACTORIAL_EXPLORATORY_EVALUATION"
    ):
        raise ValueError("three-module factorial parent is not complete")
    if int(completion["data"]["ranking_queries"]) != 892:
        raise ValueError("factorial completion ranking population drift")
    base = load_base_protocol(base_path)
    return protocol, base


def _expanded_base(
    protocol: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    expanded = deepcopy(base)
    math = protocol["sources"]["math"]
    expanded["sources"]["math"]["allowed_levels"] = list(math["allowed_levels"])
    expanded["sources"]["math"]["minimum_official_solution_words"] = int(
        math["minimum_official_solution_words"]
    )
    gsm = protocol["sources"]["gsm8k"]
    expanded["sources"]["gsm8k"]["long_chain_filter"] = {
        "minimum_reference_reasoning_words": int(
            gsm["minimum_reference_reasoning_words"]
        ),
        "minimum_reference_calculation_markers": int(
            gsm["minimum_reference_calculation_markers"]
        ),
        "minimum_distinct_intermediate_numeric_values": int(
            gsm["minimum_distinct_intermediate_numeric_values"]
        ),
    }
    return expanded


def _combine_exclusions(
    protocol: Mapping[str, Any], source_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reasons: dict[str, set[str]] = defaultdict(set)
    per_input: dict[str, Any] = {}
    for spec in protocol["historical_exclusion_inputs"]:
        rows = _verify_published_jsonl(spec)
        seen: set[str] = set()
        aliases = 0
        for row in rows:
            raw_query_id = str(row.get("query_id", ""))
            if not raw_query_id:
                raise ValueError(f"{spec['label']} row lacks query_id")
            query_id = raw_query_id
            if query_id not in source_by_id:
                legacy_gsm = re.fullmatch(r"gsm8k-train-(\d{5})", query_id)
                if legacy_gsm is not None:
                    query_id = f"gsm8k:train:{legacy_gsm.group(1)}"
                    aliases += 1
            if query_id not in source_by_id:
                raise ValueError(
                    f"{spec['label']} query lacks pinned source anchor: {raw_query_id}"
                )
            seen.add(query_id)
            reason = str(spec["label"])
            if raw_query_id != query_id:
                reason += "|legacy_query_id_alias_resolved"
            reasons[query_id].add(reason)
        per_input[str(spec["label"])] = {
            "input_rows": len(rows),
            "unique_query_ids": len(seen),
            "legacy_alias_rows_resolved": aliases,
        }
    output = [
        {
            "query_id": query_id,
            "source": str(source_by_id[query_id]["source"]),
            "reasons": sorted(values),
        }
        for query_id, values in sorted(reasons.items())
    ]
    return output, {
        "inputs": per_input,
        "combined_unique_query_ids": len(output),
        "source_counts": dict(sorted(Counter(row["source"] for row in output).items())),
    }


def _attach_prompt_ids(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("gate-tuning freeze requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    template = str(generation["prompt_template"])
    maximum = int(generation["maximum_prompt_tokens"])
    output: list[dict[str, Any]] = []
    counts: list[int] = []
    for raw in rows:
        row = dict(raw)
        content = template.replace("<QUESTION>", str(row["question"]))
        token_ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        if len(token_ids) > maximum:
            raise ValueError(f"{row['query_id']}: prompt exceeds frozen maximum")
        row["prompt_token_ids"] = token_ids
        row["prompt_token_count"] = len(token_ids)
        counts.append(len(token_ids))
        output.append(_compact_query(row))
    return output, {
        "count": len(counts),
        "minimum": min(counts),
        "maximum": max(counts),
        "mean": sum(counts) / len(counts),
        "overflow": 0,
    }


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
        "selection_stratum",
        "cluster_id",
        "cluster_split_priority",
        "query_priority",
        "question_sha256",
        "template_signature_v6",
        "role",
        "evaluation_split",
        "evaluation_only",
        "sealed_until_weight_lock",
        "role_priority",
        "prompt_token_count",
        "prompt_token_ids",
    )
    return {key: row[key] for key in keys if key in row}


def build_plan(
    protocol: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    cache_dir: str | None,
) -> dict[str, Any]:
    expanded = _expanded_base(protocol, base)
    # Regenerate every numeric MATH train row so historical rows that do not
    # pass this stage's minimum-length filter can still act as cluster anchors.
    # The actual candidate filter remains the stricter ``expanded`` contract.
    source_regeneration = deepcopy(expanded)
    source_regeneration["sources"]["math"]["allowed_levels"] = [1, 2, 3, 4, 5]
    source_regeneration["sources"]["math"]["minimum_official_solution_words"] = 0
    sources, source_report = _load_extended_sources(
        source_regeneration, cache_dir=cache_dir
    )
    candidates, source_filter_report = build_source_candidates(
        sources, expanded, required_schema=str(base["schema_version"])
    )
    source_by_id = {str(row["query_id"]): row for row in sources}
    exclusions, exclusion_report = _combine_exclusions(protocol, source_by_id)
    excluded_ids = {str(row["query_id"]) for row in exclusions}
    anchors = [source_by_id[query_id] for query_id in sorted(excluded_ids)]
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
        namespace=str(protocol["template_clustering"]["namespace"]),
    )
    tuning, confirmation, population_report = build_query_manifests(
        selectable, protocol
    )
    tuning, tuning_prompt_report = _attach_prompt_ids(
        tuning, protocol, cache_dir=cache_dir
    )
    confirmation, confirmation_prompt_report = _attach_prompt_ids(
        confirmation, protocol, cache_dir=cache_dir
    )
    shards = build_rollout_shards(tuning, confirmation, protocol)
    selected_ids = {str(row["query_id"]) for row in [*tuning, *confirmation]}
    selected_clusters = {str(row["cluster_id"]) for row in [*tuning, *confirmation]}
    excluded_clusters = {
        str(row["cluster_id"])
        for row in clusters
        if row.get("excluded_by_prior_membership")
    }
    if selected_ids & excluded_ids or selected_clusters & excluded_clusters:
        raise AssertionError("selected population overlaps historical query/cluster")
    stage_a = _validate_stage_a_configs(protocol)
    return {
        "exclusions": exclusions,
        "clusters": clusters,
        "tuning": tuning,
        "confirmation": confirmation,
        "shards": shards,
        "reports": {
            "source": source_report,
            "source_filter": source_filter_report,
            "exclusions": exclusion_report,
            "clusters": cluster_report,
            "population": population_report,
            "prompt_tokens": {
                TUNING_ROLE: tuning_prompt_report,
                CONFIRMATION_ROLE: confirmation_prompt_report,
            },
            "stage_a_config_gate": stage_a,
            "historical_query_overlap": 0,
            "historical_cluster_overlap": 0,
            "test_files_read": False,
            "planned_raw_trajectories": sum(
                int(row["expected_candidate_rows"]) for row in shards
            ),
        },
    }


def _json_record(path: Path, payload: Any) -> dict[str, Any]:
    atomic_write_json(path, payload)
    return {
        "format": "json",
        "path": path.name,
        "file_sha256": file_sha256(path),
        "canonical_payload_sha256": canonical_sha256(payload),
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
        "format": "jsonl",
        "path": path.name,
        "file_sha256": manifest["file_sha256"],
        "sidecar_file_sha256": file_sha256(
            path.with_suffix(path.suffix + ".manifest.json")
        ),
        "row_count": manifest["row_count"],
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
    }


def command_audit(args: argparse.Namespace) -> None:
    protocol, base = load_protocol(args.protocol)
    plan = build_plan(protocol, base, cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "status": "PASS_GATE_TUNING_V1_FRESH_CAPACITY_AUDIT",
                **plan["reports"],
                "tuning_queries": len(plan["tuning"]),
                "confirmation_queries": len(plan["confirmation"]),
                "rollout_shards": len(plan["shards"]),
                "artifacts_written": False,
                "gpu_used": False,
                "rollout_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_freeze(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior/Gate pre-rollout freeze requires a clean commit")
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    if (output / "manifest_registry.json").exists():
        raise FileExistsError("Prior/Gate pre-rollout population is already frozen")
    protocol, base = load_protocol(protocol_path)
    plan = build_plan(protocol, base, cache_dir=args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "source_audit.json": _json_record(
            output / "source_audit.json", plan["reports"]
        ),
        "permanent_exclusions.jsonl": _jsonl_record(
            output / "permanent_exclusions.jsonl",
            plan["exclusions"],
            schema_version="clir-gate-tuning-v1-permanent-exclusions",
            metadata=plan["reports"]["exclusions"],
        ),
        "template_clusters.jsonl": _jsonl_record(
            output / "template_clusters.jsonl",
            plan["clusters"],
            schema_version="clir-gate-tuning-v1-template-clusters",
            metadata=plan["reports"]["clusters"],
        ),
        "tuning_queries.jsonl": _jsonl_record(
            output / "tuning_queries.jsonl",
            plan["tuning"],
            schema_version="clir-gate-tuning-v1-tuning-queries",
            metadata=plan["reports"]["population"]["selected"],
        ),
        "confirmation_queries.jsonl": _jsonl_record(
            output / "confirmation_queries.jsonl",
            plan["confirmation"],
            schema_version="clir-gate-tuning-v1-confirmation-queries-sealed",
            metadata=plan["reports"]["population"]["selected"],
        ),
        "rollout_shards.json": _json_record(
            output / "rollout_shards.json", plan["shards"]
        ),
    }
    report = {
        "schema_version": "clir-gate-tuning-v1-pre-rollout-freeze-report",
        "status": "PASS_GATE_TUNING_V1_PRE_ROLLOUT_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "code_dirty": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "records": records,
        "tuning_query_count": len(plan["tuning"]),
        "confirmation_query_count": len(plan["confirmation"]),
        "raw_candidate_rows": plan["reports"]["planned_raw_trajectories"],
        "query_overlap": 0,
        "cluster_overlap": 0,
        "historical_query_overlap": 0,
        "historical_cluster_overlap": 0,
        "confirmation_sealed": True,
        "gpu_used": False,
        "rollout_allowed": False,
        "next_gate": "independent_recompute_then_hash_bound_rollout_authorization",
    }
    report_path = output / "freeze_report.json"
    atomic_write_json(report_path, report)
    registry = {
        "schema_version": "clir-gate-tuning-v1-pre-rollout-registry",
        "status": report["status"],
        "code_commit": report["code_commit"],
        "protocol_file_sha256": report["protocol_file_sha256"],
        "freeze_report_file_sha256": file_sha256(report_path),
        "records": records,
    }
    atomic_write_json(output / "manifest_registry.json", registry)
    print(
        json.dumps(
            {
                **report,
                "freeze_report_file_sha256": file_sha256(report_path),
                "manifest_registry_file_sha256": file_sha256(
                    output / "manifest_registry.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _verify_record(output: Path, name: str, spec: Mapping[str, Any]) -> Any:
    path = output / name
    _assert_file(path, str(spec["file_sha256"]), name)
    if spec["format"] == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(payload) != spec["canonical_payload_sha256"]:
            raise ValueError(f"frozen JSON canonical hash drift: {name}")
        return payload
    rows = read_jsonl(path)
    if len(rows) != int(spec["row_count"]):
        raise ValueError(f"frozen JSONL row count drift: {name}")
    if canonical_sha256(rows) != spec["ordered_rows_sha256"]:
        raise ValueError(f"frozen JSONL ordered rows drift: {name}")
    _assert_file(
        path.with_suffix(path.suffix + ".manifest.json"),
        str(spec["sidecar_file_sha256"]),
        f"{name} sidecar",
    )
    return rows


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    protocol, base = load_protocol(protocol_path)
    registry_path = output / "manifest_registry.json"
    freeze_path = output / "freeze_report.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("registry protocol hash drift")
    if registry.get("freeze_report_file_sha256") != file_sha256(freeze_path):
        raise ValueError("registry freeze report hash drift")
    frozen = {
        name: _verify_record(output, name, registry["records"][name])
        for name in REQUIRED_FILES
    }
    recomputed = build_plan(protocol, base, cache_dir=args.cache_dir)
    expected = {
        "source_audit.json": recomputed["reports"],
        "permanent_exclusions.jsonl": recomputed["exclusions"],
        "template_clusters.jsonl": recomputed["clusters"],
        "tuning_queries.jsonl": recomputed["tuning"],
        "confirmation_queries.jsonl": recomputed["confirmation"],
        "rollout_shards.json": recomputed["shards"],
    }
    for name, value in expected.items():
        if frozen[name] != value:
            raise ValueError(f"independent recomputation drift: {name}")
    verification = {
        "schema_version": "clir-gate-tuning-v1-pre-rollout-verification",
        "status": "PASS_GATE_TUNING_V1_PRE_ROLLOUT_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "code_commit": _git_head(),
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "freeze_report_file_sha256": file_sha256(freeze_path),
        "tuning_queries": len(recomputed["tuning"]),
        "confirmation_queries": len(recomputed["confirmation"]),
        "rollout_shards": len(recomputed["shards"]),
        "query_overlap": 0,
        "cluster_overlap": 0,
        "confirmation_sealed": True,
        "rollout_allowed": False,
        "next_gate": "hash_bound_rollout_authorization",
    }
    verification_path = output / "independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        stable = set(verification) - {"verified_at_utc"}
        if any(old.get(key) != verification.get(key) for key in stable):
            raise ValueError("existing verification report drift")
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
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(func=command_audit)
    subparsers.add_parser("freeze").set_defaults(func=command_freeze)
    subparsers.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
