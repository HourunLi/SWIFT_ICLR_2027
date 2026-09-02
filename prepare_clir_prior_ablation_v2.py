#!/usr/bin/env python
"""Freeze and independently verify the CLIR Prior ablation-v2 contract.

This entry point is CPU-only.  It freezes new GSM8K/ASDiv train-source
questions, a score-unseen MATH reserve, the rollout shards, and all generated
training configurations before any v2 generation, training, or scoring.
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

from prepare_clir_gate_tuning import (
    _combine_exclusions,
    _expanded_base,
    load_protocol as load_gate_protocol,
)
from prepare_clir_ranking import _load_extended_sources
from src.clir_prior_ablation import (
    EXPECTED_CELLS,
    config_factor_projection,
    derive_config,
    factor_map,
    select_query_rows,
    validate_protocol,
)
from src.clir_scale import build_template_clusters, gsm8k_long_chain_metrics
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
    stable_priority,
    validate_rollout_population,
    validate_source_row,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2/pre_rollout"
ANCHOR_CONFIGS = {
    "u0": "configs/three_module_expansion_v1/u0_correctness_only.json",
    "c": "configs/three_module_expansion_v1/c_consistency.json",
    "h": "configs/three_module_expansion_v1/h_h0_onset_bce.json",
    "ch": "configs/three_module_expansion_v1/ch_consistency_h0.json",
    "full": "configs/three_module_expansion_v1/full_consistency_h0_prior_gate.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": head, "branch": branch, "dirty": bool(status)}


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prior ablation protocol must be a JSON object")
    validate_protocol(payload)
    for label, spec in payload["frozen_parents"].items():
        if "file_sha256" not in spec:
            continue
        pinned = _project_path(spec["path"])
        if file_sha256(pinned) != spec["file_sha256"]:
            raise ValueError(f"pinned parent drift: {label}")
    return payload


def _canonical_query_id(query_id: str, source_by_id: Mapping[str, Any]) -> str:
    if query_id in source_by_id:
        return query_id
    legacy = re.fullmatch(r"gsm8k-train-(\d{5})", query_id)
    if legacy is not None:
        canonical = f"gsm8k:train:{legacy.group(1)}"
        if canonical in source_by_id:
            return canonical
    raise ValueError(f"historical query lacks a source anchor: {query_id}")


def _read_bound_jsonl(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _project_path(spec["path"])
    rows = read_jsonl(path)
    if file_sha256(path) != spec["file_sha256"] or len(rows) != int(spec["rows"]):
        raise ValueError(f"bound JSONL drift: {path}")
    if "ordered_rows_sha256" in spec and canonical_sha256(rows) != spec[
        "ordered_rows_sha256"
    ]:
        raise ValueError(f"bound JSONL ordered rows drift: {path}")
    return rows


def _gate_checker_inputs(protocol: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    completion_spec = protocol["frozen_parents"]["ranking_checker_completion"]
    completion_path = _project_path(completion_spec["path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_GATE_TUNING_V1_CHECKER_AND_FROZEN_SELECTION":
        raise ValueError("Prior/Gate checker parent is not complete")
    result: dict[str, list[dict[str, Any]]] = {}
    for role in ("tuning", "confirmation"):
        for kind in ("checked", "selected"):
            binding = completion[role][kind]
            result[f"{role}_{kind}"] = _read_bound_jsonl(binding)
    return result


def _math_reserve(protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = _gate_checker_inputs(protocol)
    checked = [*inputs["tuning_checked"], *inputs["confirmation_checked"]]
    selected_ids = {
        str(row["query_id"])
        for row in [*inputs["tuning_selected"], *inputs["confirmation_selected"]]
    }
    candidate_count = int(protocol["generation"]["candidate_count"])
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in checked:
        if raw["source"] == "math" and str(raw["query_id"]) not in selected_ids:
            by_query[str(raw["query_id"])].append(dict(raw))
    eligible: list[tuple[str, list[dict[str, Any]]]] = []
    binary = {"numeric_match", "numeric_mismatch"}
    for query_id, rows in by_query.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        if (
            len(rows) == candidate_count
            and [int(row["candidate_index"]) for row in rows]
            == list(range(candidate_count))
            and all(row.get("checker_status") in binary for row in rows)
        ):
            eligible.append((query_id, rows))
    namespace = "clir-prior-ablation-v2-math-reserve"
    eligible.sort(key=lambda item: stable_priority(namespace, item[0]))
    available = int(
        protocol["ranking_population"]["math_reserve"][
            "available_queries_at_design_audit"
        ]
    )
    target = int(protocol["ranking_population"]["math_reserve"]["select_queries"])
    if len(eligible) != available or len(eligible) < target:
        raise ValueError(f"MATH reserve drift: {len(eligible)} != {available}")
    output: list[dict[str, Any]] = []
    for query_index, (query_id, rows) in enumerate(eligible[:target]):
        for raw in rows:
            row = dict(raw)
            row.update(
                {
                    "role": "prior_ablation_v2_ranking",
                    "evaluation_split": "prior_ablation_v2",
                    "evaluation_only": True,
                    "sealed_until_weight_lock": False,
                    "prior_ablation_origin": "score_unseen_prior_gate_v1_math_reserve",
                    "prior_ablation_query_order": query_index,
                    "prior_ablation_selection_priority": stable_priority(
                        namespace, query_id
                    ),
                }
            )
            output.append(row)
    validate_rollout_population(output, candidate_count=candidate_count)
    return output, {
        "available_queries": len(eligible),
        "selected_queries": target,
        "selected_rows": len(output),
        "selected_query_ids_sha256": canonical_sha256(
            [query_id for query_id, _ in eligible[:target]]
        ),
        "previously_selected_or_scored_query_overlap": 0,
    }


def _attach_prompt_ids(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    template = str(generation["prompt_template"])
    maximum = int(generation["maximum_prompt_tokens"])
    output: list[dict[str, Any]] = []
    lengths: list[int] = []
    for source in rows:
        row = dict(source)
        content = template.replace("<QUESTION>", str(row["question"]))
        token_ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        if not token_ids or len(token_ids) > maximum:
            raise ValueError(f"{row['query_id']}: prompt token count invalid")
        row["prompt_token_ids"] = token_ids
        row["prompt_token_count"] = len(token_ids)
        lengths.append(len(token_ids))
        output.append(row)
    return output, {
        "count": len(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _source_plan(
    protocol: Mapping[str, Any], cache_dir: str | None
) -> dict[str, Any]:
    gate_path = _project_path(protocol["frozen_parents"]["ranking_source_protocol"]["path"])
    gate_protocol, base = load_gate_protocol(gate_path)
    expanded = _expanded_base(gate_protocol, base)
    regeneration = deepcopy(expanded)
    regeneration["sources"]["math"]["allowed_levels"] = [1, 2, 3, 4, 5]
    regeneration["sources"]["math"]["minimum_official_solution_words"] = 0
    sources, source_report = _load_extended_sources(regeneration, cache_dir=cache_dir)
    source_by_id = {str(row["query_id"]): row for row in sources}
    candidates: list[dict[str, Any]] = []
    filter_counts: Counter[str] = Counter()
    for raw in sources:
        if raw["source"] not in {"gsm8k", "asdiv-a"}:
            continue
        row = validate_source_row(raw)
        if raw["source"] == "gsm8k":
            metrics = gsm8k_long_chain_metrics(
                str(raw.get("reference_answer", "")),
                expanded["sources"]["gsm8k"]["long_chain_filter"],
            )
            row.update(metrics)
            row["selection_stratum"] = (
                "gsm8k_long_chain" if metrics["long_chain_filter_pass"] else "gsm8k_short_or_medium"
            )
            filter_counts[row["selection_stratum"]] += 1
        else:
            row["selection_stratum"] = "asdiv-a"
            filter_counts["asdiv-a"] += 1
        row["query_priority"] = stable_priority(
            f"clir-prior-ablation-v2-{raw['source']}", str(row["query_id"])
        )
        candidates.append(row)
    filter_report = {
        "policy": "all_unseen_gsm8k_and_asdiv_with_gsm_chain_length_reported_not_filtered",
        "counts": dict(sorted(filter_counts.items())),
    }

    base_exclusions, exclusion_report = _combine_exclusions(
        gate_protocol, source_by_id
    )
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in base_exclusions:
        reasons[str(row["query_id"])].update(str(value) for value in row["reasons"])
    for path in (
        "run_artifacts/prior_gate_tuning_v1/pre_rollout/tuning_queries.jsonl",
        "run_artifacts/prior_gate_tuning_v1/pre_rollout/confirmation_queries.jsonl",
    ):
        for row in read_jsonl(_project_path(path)):
            reasons[str(row["query_id"])].add("prior_gate_tuning_v1_raw_population")
    train_path = _project_path(protocol["frozen_parents"]["training_manifest"]["path"])
    for row in read_jsonl(train_path):
        query_id = _canonical_query_id(str(row["query_id"]), source_by_id)
        reasons[query_id].add("prior_v16_posthoc_training_manifest")
    missing = set(reasons) - set(source_by_id)
    if missing:
        raise ValueError(f"exclusion source anchors missing: {sorted(missing)[:5]}")
    exclusions = [
        {
            "query_id": query_id,
            "source": source_by_id[query_id]["source"],
            "reasons": sorted(values),
        }
        for query_id, values in sorted(reasons.items())
    ]
    excluded_ids = set(reasons)
    anchors = [source_by_id[query_id] for query_id in sorted(excluded_ids)]
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
        namespace="clir-prior-ablation-v2",
    )
    selectable_by_source = {
        source: [row for row in selectable if row["source"] == source]
        for source in ("gsm8k", "asdiv-a")
    }
    capacities: dict[str, int] = {}
    representatives_by_source: dict[str, list[dict[str, Any]]] = {}
    raw_targets = protocol["ranking_population"]["new_rollout_raw_queries"]
    for source in ("gsm8k", "asdiv-a"):
        namespace = f"clir-prior-ablation-v2-raw-{source}"
        # select_query_rows applies the one-per-cluster rule before the hash.
        representatives = select_query_rows(
            selectable_by_source[source],
            len({str(row["cluster_id"]) for row in selectable_by_source[source]}),
            namespace=namespace,
        )
        capacities[source] = len(representatives)
        representatives_by_source[source] = representatives
    selected: list[dict[str, Any]] = []
    used_clusters: set[str] = set()
    # ASDiv has much less spare capacity, so its frozen quota is filled first.
    # GSM8K is then selected after removing those cross-source template clusters.
    for source in ("asdiv-a", "gsm8k"):
        available = [
            row
            for row in representatives_by_source[source]
            if str(row["cluster_id"]) not in used_clusters
        ]
        target = int(raw_targets[source])
        if len(available) < target:
            raise ValueError(f"{source}: cross-source-disjoint capacity {len(available)} < {target}")
        chosen = available[:target]
        used_clusters.update(str(row["cluster_id"]) for row in chosen)
        for source_index, raw in enumerate(chosen):
            row = dict(raw)
            row.update(
                {
                    "role": "prior_ablation_v2_ranking",
                    "evaluation_split": "prior_ablation_v2",
                    "evaluation_only": True,
                    "sealed_until_weight_lock": False,
                    "prior_ablation_origin": "new_rollout",
                    "prior_ablation_source_order": source_index,
                    "prior_ablation_final_priority": stable_priority(
                        f"{protocol['checker']['new_source_selection_namespace']}-{source}",
                        str(row["query_id"]),
                    ),
                }
            )
            selected.append(row)
    if len(used_clusters) != len(selected):
        raise AssertionError("new rollout selected more than one query per cluster")
    expected_capacity = protocol["ranking_population"][
        "fresh_source_capacity_at_design_audit"
    ]
    for source, observed in capacities.items():
        if int(expected_capacity[source]) != observed:
            raise ValueError(
                f"{source}: frozen fresh capacity drift {observed} != "
                f"{expected_capacity[source]}"
            )
    selected.sort(
        key=lambda row: stable_priority(
            "clir-prior-ablation-v2-rollout-order", str(row["query_id"])
        )
    )
    selected, prompt_report = _attach_prompt_ids(selected, protocol, cache_dir)
    shards: list[dict[str, Any]] = []
    shard_size = int(protocol["generation"]["queries_per_shard"])
    for offset in range(0, len(selected), shard_size):
        members = selected[offset : offset + shard_size]
        shard_id = f"ranking-{offset // shard_size:03d}"
        shards.append(
            {
                "shard_id": shard_id,
                "role": "prior_ablation_v2_ranking",
                "query_ids": [str(row["query_id"]) for row in members],
                "query_count": len(members),
                "candidate_count": int(protocol["generation"]["candidate_count"]),
                "expected_candidate_rows": len(members)
                * int(protocol["generation"]["candidate_count"]),
                "output_path": f"rollouts/shards/{shard_id}.jsonl",
            }
        )
    if len(shards) != int(protocol["generation"]["rollout_shards"]):
        raise ValueError("rollout shard count drift")
    return {
        "sources": sources,
        "exclusions": exclusions,
        "clusters": clusters,
        "queries": selected,
        "shards": shards,
        "reports": {
            "source": source_report,
            "source_filter": filter_report,
            "base_exclusions": exclusion_report,
            "clusters": cluster_report,
            "fresh_capacity_one_per_cluster": capacities,
            "selected_source_counts": dict(
                sorted(Counter(row["source"] for row in selected).items())
            ),
            "prompt_tokens": prompt_report,
            "test_files_read": False,
        },
    }


def _config_plan(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    base_path = _project_path(protocol["frozen_parents"]["base_config"]["path"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    records: dict[str, Any] = {}
    for cell in EXPECTED_CELLS:
        derived = derive_config(protocol, base, cell)
        if cell in ANCHOR_CONFIGS:
            path = _project_path(ANCHOR_CONFIGS[cell])
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != derived:
                raise ValueError(f"immutable anchor config differs from derivation: {cell}")
            kind = "immutable_anchor"
        else:
            path = output_root.parent / "configs" / f"{cell}.json"
            kind = "generated"
        records[cell] = {
            "kind": kind,
            "path": str(path),
            "payload": derived,
            "factors": factor_map(protocol, cell),
            "model_projection": config_factor_projection(derived),
        }
    return records


def build_plan(
    protocol: Mapping[str, Any], *, output_root: Path, cache_dir: str | None
) -> dict[str, Any]:
    source = _source_plan(protocol, cache_dir)
    reserve, reserve_report = _math_reserve(protocol)
    return {
        "source": source,
        "math_reserve": reserve,
        "math_reserve_report": reserve_report,
        "configs": _config_plan(protocol, output_root),
    }


def _manifest_record(path: Path, rows: Sequence[Mapping[str, Any]], schema: str) -> dict[str, Any]:
    manifest = publish_manifest(path, rows, schema_version=schema)
    return {
        "path": str(path),
        "rows": len(rows),
        "file_sha256": manifest["file_sha256"],
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "sidecar_file_sha256": file_sha256(path.with_suffix(path.suffix + ".manifest.json")),
    }


def command_audit(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    plan = build_plan(protocol, output_root=Path(args.output).resolve(), cache_dir=args.cache_dir)
    print(
        json.dumps(
            {
                "status": "PASS_PRIOR_ABLATION_V2_DESIGN_AUDIT",
                "fresh_capacity": plan["source"]["reports"]["fresh_capacity_one_per_cluster"],
                "selected_new_queries": plan["source"]["reports"]["selected_source_counts"],
                "math_reserve": plan["math_reserve_report"],
                "cells": len(plan["configs"]),
                "new_training_runs": len(protocol["training"]["new_cells"])
                * len(protocol["training"]["seeds"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_freeze(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    state = _git_state()
    if state["dirty"] or state["branch"] != "clir-clean-integration":
        raise RuntimeError("freeze requires a clean clir-clean-integration commit")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"pre-rollout output is not empty: {output}")
    protocol = load_protocol(protocol_path)
    plan = build_plan(protocol, output_root=output, cache_dir=args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "exclusions": _manifest_record(
            output / "permanent_exclusions.jsonl",
            plan["source"]["exclusions"],
            "clir-prior-ablation-v2-exclusions",
        ),
        "clusters": _manifest_record(
            output / "template_clusters.jsonl",
            plan["source"]["clusters"],
            "clir-prior-ablation-v2-template-clusters",
        ),
        "fresh_queries": _manifest_record(
            output / "fresh_queries.jsonl",
            plan["source"]["queries"],
            "clir-prior-ablation-v2-fresh-rollout-queries",
        ),
        "math_reserve": _manifest_record(
            output / "math_reserve_checked.jsonl",
            plan["math_reserve"],
            "clir-prior-ablation-v2-score-unseen-math-reserve",
        ),
    }
    atomic_write_json(output / "rollout_shards.json", plan["source"]["shards"])
    records["rollout_shards"] = {
        "path": str(output / "rollout_shards.json"),
        "rows": len(plan["source"]["shards"]),
        "file_sha256": file_sha256(output / "rollout_shards.json"),
        "canonical_payload_sha256": canonical_sha256(plan["source"]["shards"]),
    }
    config_records: dict[str, Any] = {}
    for cell, record in plan["configs"].items():
        path = Path(record["path"])
        if record["kind"] == "generated":
            atomic_write_json(path, record["payload"])
        config_records[cell] = {
            key: value for key, value in record.items() if key != "payload"
        }
        config_records[cell]["file_sha256"] = file_sha256(path)
    training_jobs = []
    for cell in protocol["training"]["new_cells"]:
        for seed in protocol["training"]["seeds"]:
            training_jobs.append(
                {
                    "cell": cell,
                    "seed": int(seed),
                    "worker_index": len(training_jobs) % 8,
                    "config_path": config_records[cell]["path"],
                    "config_file_sha256": config_records[cell]["file_sha256"],
                    "checkpoint_path": str(
                        output.parent / "training" / cell / f"seed-{seed}" / "checkpoint.pt"
                    ),
                }
            )
    training_plan = {
        "schema_version": "clir-prior-ablation-v2-training-plan",
        "code_commit": state["commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "training_manifest": protocol["frozen_parents"]["training_manifest"],
        "configs": config_records,
        "jobs": training_jobs,
    }
    atomic_write_json(output.parent / "training_plan.json", training_plan)
    records["training_plan"] = {
        "path": str(output.parent / "training_plan.json"),
        "file_sha256": file_sha256(output.parent / "training_plan.json"),
        "canonical_payload_sha256": canonical_sha256(training_plan),
    }
    report = {
        "schema_version": "clir-prior-ablation-v2-freeze-report",
        "status": "PASS_PRIOR_ABLATION_V2_PRE_ROLLOUT_FREEZE",
        "frozen_at_utc": _utc_now(),
        "code": state,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "records": records,
        "source_reports": plan["source"]["reports"],
        "math_reserve_report": plan["math_reserve_report"],
        "planned_new_rollout_rows": sum(
            int(row["expected_candidate_rows"]) for row in plan["source"]["shards"]
        ),
        "planned_final_ranking_rows": int(
            protocol["ranking_population"]["selected_candidate_rows"]
        ),
        "test_files_read": False,
        "v2_clir_scores_opened": False,
    }
    atomic_write_json(output / "freeze_report.json", report)
    registry = {
        "schema_version": "clir-prior-ablation-v2-manifest-registry",
        "code_commit": state["commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "freeze_report_file_sha256": file_sha256(output / "freeze_report.json"),
        "records": records,
    }
    atomic_write_json(output / "manifest_registry.json", registry)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    registry = json.loads((output / "manifest_registry.json").read_text(encoding="utf-8"))
    if registry["protocol_file_sha256"] != file_sha256(protocol_path):
        raise ValueError("registry protocol hash drift")
    plan = build_plan(protocol, output_root=output, cache_dir=args.cache_dir)
    expected_rows = {
        "exclusions": plan["source"]["exclusions"],
        "clusters": plan["source"]["clusters"],
        "fresh_queries": plan["source"]["queries"],
        "math_reserve": plan["math_reserve"],
    }
    for label, rows in expected_rows.items():
        spec = registry["records"][label]
        path = Path(spec["path"])
        actual = read_jsonl(path)
        if (
            file_sha256(path) != spec["file_sha256"]
            or canonical_sha256(actual) != canonical_sha256(rows)
            or actual != rows
        ):
            raise ValueError(f"independent recomputation drift: {label}")
    shards = json.loads((output / "rollout_shards.json").read_text(encoding="utf-8"))
    if shards != plan["source"]["shards"]:
        raise ValueError("independent rollout shard recomputation drift")
    training_plan_path = output.parent / "training_plan.json"
    training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    for cell, record in plan["configs"].items():
        path = Path(record["path"])
        if json.loads(path.read_text(encoding="utf-8")) != record["payload"]:
            raise ValueError(f"generated config drift: {cell}")
        if training_plan["configs"][cell]["file_sha256"] != file_sha256(path):
            raise ValueError(f"training-plan config binding drift: {cell}")
    report = {
        "schema_version": "clir-prior-ablation-v2-freeze-verification",
        "status": "PASS_PRIOR_ABLATION_V2_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "code_commit": registry["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "registry_file_sha256": file_sha256(output / "manifest_registry.json"),
        "fresh_queries": len(plan["source"]["queries"]),
        "math_reserve_queries": len(plan["math_reserve"])
        // int(protocol["generation"]["candidate_count"]),
        "all_rows_byte_identical_to_recomputation": True,
        "test_files_read": False,
        "v2_clir_scores_opened": False,
    }
    atomic_write_json(output / "independent_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit").set_defaults(func=command_audit)
    sub.add_parser("freeze").set_defaults(func=command_freeze)
    sub.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
