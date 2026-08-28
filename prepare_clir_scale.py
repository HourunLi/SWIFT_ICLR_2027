#!/usr/bin/env python
"""Prepare and execute the authorized CLIR Consistency scale-v6 data stage.

``freeze`` and ``verify`` manage the immutable pre-rollout gate.  After a
separate hash-bound user authorization, ``rollout`` writes one atomic frozen
shard, ``verify-rollouts`` audits shard artifacts, and ``merge-rollouts``
publishes the complete raw population.  This entry point intentionally has no
annotation, feature-extraction, or training command.
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

from src.clir_scale import (
    SCALE_V6_SCHEMA,
    build_rollout_shards,
    build_source_candidates,
    build_template_clusters,
    combine_permanent_exclusions,
    compact_query_row,
    select_acquisition_queries,
    storage_and_gpu_budget,
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
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_scale_v6/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_scale_v6/pre_rollout"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "configs/data_expansion_scale_v6/rollout_authorization.json"
)
DEFAULT_ROLLOUT_ROOT = PROJECT_ROOT / "run_artifacts/data_expansion_scale_v6"
REQUIRED_FILES = (
    "source_inventory.json",
    "permanent_exclusions.jsonl",
    "template_clusters.jsonl",
    "train_acquisition_queries.jsonl",
    "heldout_acquisition_queries.jsonl",
    "rollout_shards.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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
        stable_priority("clir-vllm-query-seed-v2", base_seed, query_id)[:16], 16
    ) % (2**31)


def _ordered_vllm_candidates(
    request_output: Any, expected_count: int
) -> list[Any]:
    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "vLLM candidate indices must be unique and contiguous: "
            f"expected 0..{expected_count - 1}, got {sorted(indices)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate.index))


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


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCALE_V6_SCHEMA:
        raise ValueError("prepare_clir_scale supports only frozen scale v6")
    if protocol.get("status") != "FROZEN_PREPARATION_ROLLOUT_NOT_STARTED":
        raise ValueError("scale-v6 protocol is not at the pre-rollout freeze gate")
    if protocol["claim_boundary"].get("rollout_has_started") is not False:
        raise ValueError("scale-v6 claim boundary no longer permits pre-rollout freeze")
    if protocol["execution_authorization"].get("rollout_allowed") is not False:
        raise ValueError("this command must not execute a rollout-enabled protocol")
    return protocol


def _verify_jsonl_input(
    *,
    path: Path,
    expected_file_sha256: str,
    expected_ordered_rows_sha256: str,
    expected_row_count: int,
    sidecar_file_sha256: str,
) -> list[dict[str, Any]]:
    if file_sha256(path) != expected_file_sha256:
        raise ValueError(f"pinned input file hash mismatch: {path}")
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if file_sha256(sidecar) != sidecar_file_sha256:
        raise ValueError(f"pinned input sidecar hash mismatch: {sidecar}")
    rows = read_jsonl(path)
    if len(rows) != expected_row_count:
        raise ValueError(f"pinned input row count mismatch: {path}")
    if canonical_sha256(rows) != expected_ordered_rows_sha256:
        raise ValueError(f"pinned input ordered-row hash mismatch: {path}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = {
        "row_count": expected_row_count,
        "ordered_rows_sha256": expected_ordered_rows_sha256,
        "file_sha256": expected_file_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"pinned input sidecar contents mismatch: {sidecar}")
    return rows


def _load_pinned_inputs(
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parent = protocol["pre_rollout_implementation"]["source_parent"]
    parent_protocol = _project_path(parent["protocol_path"])
    if file_sha256(parent_protocol) != parent["protocol_file_sha256"]:
        raise ValueError("pinned v3 source protocol hash mismatch")
    sources = _verify_jsonl_input(
        path=_project_path(parent["source_corpus_path"]),
        expected_file_sha256=parent["source_corpus_file_sha256"],
        expected_ordered_rows_sha256=parent["source_corpus_ordered_rows_sha256"],
        expected_row_count=int(parent["source_corpus_row_count"]),
        sidecar_file_sha256=parent["source_corpus_sidecar_file_sha256"],
    )
    historical = _verify_jsonl_input(
        path=_project_path(parent["historical_exclusion_path"]),
        expected_file_sha256=parent["historical_exclusion_file_sha256"],
        expected_ordered_rows_sha256=parent[
            "historical_exclusion_ordered_rows_sha256"
        ],
        expected_row_count=int(parent["historical_exclusion_row_count"]),
        sidecar_file_sha256=parent["historical_exclusion_sidecar_file_sha256"],
    )
    smoke_queries = _verify_jsonl_input(
        path=_project_path(parent["smoke_query_manifest_path"]),
        expected_file_sha256=parent["smoke_query_manifest_file_sha256"],
        expected_ordered_rows_sha256=parent[
            "smoke_query_manifest_ordered_rows_sha256"
        ],
        expected_row_count=int(parent["smoke_query_manifest_row_count"]),
        sidecar_file_sha256=parent[
            "smoke_query_manifest_sidecar_file_sha256"
        ],
    )
    return sources, historical, smoke_queries


def _rendered_prompt_token_count(
    tokenizer: Any, *, question: str, prompt_template: str
) -> int:
    if "<QUESTION>" not in prompt_template:
        raise ValueError("generation prompt template lacks <QUESTION>")
    content = prompt_template.replace("<QUESTION>", question)
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def _load_tokenizer(protocol: Mapping[str, Any], cache_dir: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("scale-v6 freeze requires transformers") from exc
    generation = protocol["generation"]
    return AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )


def _attach_prompt_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation = protocol["generation"]
    maximum = int(
        protocol["pre_rollout_implementation"]["prompt_token_counting"][
            "maximum_prompt_tokens"
        ]
    )
    kept: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    counts: list[int] = []
    for raw in rows:
        row = dict(raw)
        count = _rendered_prompt_token_count(
            tokenizer,
            question=str(row["question"]),
            prompt_template=str(generation["prompt_template"]),
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
    summary = {
        "input_count": len(rows),
        "kept_count": len(kept),
        "overflow_count": len(overflow),
        "maximum_allowed": maximum,
        "count_min": ordered[0] if ordered else None,
        "count_max": ordered[-1] if ordered else None,
        "count_mean": sum(ordered) / len(ordered) if ordered else None,
        "overflow_rows": sorted(overflow, key=lambda row: row["query_id"]),
    }
    return kept, summary


def _json_record(path: Path, payload: Any) -> dict[str, Any]:
    atomic_write_json(path, payload)
    row_count = len(payload) if isinstance(payload, list) else None
    return {
        "path": path.name,
        "format": "json",
        "file_sha256": file_sha256(path),
        "row_count": row_count,
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
        path,
        rows,
        schema_version=schema_version,
        metadata=metadata,
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


def _source_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["source"]) for row in rows).items()))


def command_freeze(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if _git_dirty():
        raise RuntimeError(
            "PRE_ROLLOUT_MANIFEST_FREEZE requires a clean Git commit; "
            "commit the implementation and protocol first"
        )
    output = Path(args.output_dir).resolve()
    if (output / "manifest_registry.json").exists():
        raise FileExistsError(
            f"{output} is already frozen; use a new directory, never overwrite a gate"
        )

    source_rows, historical_rows, smoke_rows = _load_pinned_inputs(protocol)
    candidates, source_filter_report = build_source_candidates(source_rows, protocol)
    exclusions = combine_permanent_exclusions(historical_rows, smoke_rows)
    excluded_ids = {str(row["query_id"]) for row in exclusions}
    source_by_id = {str(row["query_id"]): row for row in source_rows}
    anchors = [source_by_id[value] for value in sorted(excluded_ids & set(source_by_id))]
    missing_anchor_ids = sorted(excluded_ids - set(source_by_id))

    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        excluded_ids,
    )
    tokenizer = _load_tokenizer(protocol, args.cache_dir)
    selectable, prompt_report = _attach_prompt_counts(
        selectable,
        tokenizer=tokenizer,
        protocol=protocol,
    )
    train_rows, heldout_rows, selection_report = select_acquisition_queries(
        selectable, protocol
    )
    train = [compact_query_row(row) for row in train_rows]
    heldout = [compact_query_row(row) for row in heldout_rows]
    selected = [*train, *heldout]
    shards = build_rollout_shards(train, heldout, protocol)
    budget = storage_and_gpu_budget(selected, protocol)

    source_inventory = {
        "schema_version": "clir-scale-source-inventory-v6",
        "protocol_file_sha256": file_sha256(protocol_path),
        "input_source_row_count": len(source_rows),
        "input_source_counts": _source_counts(source_rows),
        "licenses": {
            "math": protocol["sources"]["math"]["license"],
            "gsm8k": protocol["sources"]["gsm8k"]["license"],
            "asdiv_a_not_selected": "CC-BY-NC-4.0",
        },
        "protected_splits": protocol["sources"]["protected_splits"],
        "source_filter_report": source_filter_report,
        "permanent_exclusion_count": len(exclusions),
        "exclusion_anchor_rows_found": len(anchors),
        "exclusion_anchor_query_ids_missing_from_pinned_corpus": missing_anchor_ids,
        "template_cluster_report": cluster_report,
        "prompt_token_report": prompt_report,
        "selection_report": selection_report,
        "selected_source_counts": _source_counts(selected),
        "storage_and_gpu_budget": budget,
    }

    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    records["source_inventory.json"] = _json_record(
        output / "source_inventory.json", source_inventory
    )
    records["permanent_exclusions.jsonl"] = _jsonl_record(
        output / "permanent_exclusions.jsonl",
        exclusions,
        schema_version="clir-permanent-query-exclusions-v6",
    )
    records["template_clusters.jsonl"] = _jsonl_record(
        output / "template_clusters.jsonl",
        clusters,
        schema_version="clir-template-clusters-v6",
        metadata=cluster_report,
    )
    records["train_acquisition_queries.jsonl"] = _jsonl_record(
        output / "train_acquisition_queries.jsonl",
        train,
        schema_version="clir-train-acquisition-queries-v6",
        metadata={"source_counts": _source_counts(train)},
    )
    records["heldout_acquisition_queries.jsonl"] = _jsonl_record(
        output / "heldout_acquisition_queries.jsonl",
        heldout,
        schema_version="clir-heldout-acquisition-queries-v6",
        metadata={"source_counts": _source_counts(heldout)},
    )
    records["rollout_shards.json"] = _json_record(
        output / "rollout_shards.json", shards
    )

    report = {
        "schema_version": "clir-pre-rollout-freeze-report-v6",
        "status": "PASS_PRE_ROLLOUT_MANIFEST_FREEZE",
        "rollout_started": False,
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "required_manifest_records": records,
        "counts": {
            "train_queries": len(train),
            "heldout_queries": len(heldout),
            "total_queries": len(selected),
            "planned_raw_trajectories": len(selected)
            * int(protocol["generation"]["candidate_count"]),
            "rollout_shards": len(shards),
        },
        "source_counts": {
            "train": _source_counts(train),
            "heldout": _source_counts(heldout),
        },
        "query_ids_sha256": {
            "train": canonical_sha256([row["query_id"] for row in train]),
            "heldout": canonical_sha256([row["query_id"] for row in heldout]),
        },
        "cluster_ids_sha256": {
            "train": canonical_sha256([row["cluster_id"] for row in train]),
            "heldout": canonical_sha256([row["cluster_id"] for row in heldout]),
        },
        "next_action": (
            "obtain_explicit_user_confirmation_before_starting_16000_rollouts"
        ),
    }
    report_record = _json_record(output / "pre_rollout_report.json", report)
    registry = {
        "schema_version": "clir-pre-rollout-manifest-registry-v6",
        "status": "PASS_PRE_ROLLOUT_MANIFEST_FREEZE",
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "required_files": list(REQUIRED_FILES),
        "files": records,
        "report": report_record,
    }
    atomic_write_json(output / "manifest_registry.json", registry)
    verify_pre_rollout(output, protocol_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _verify_record(output: Path, name: str, record: Mapping[str, Any]) -> Any:
    path = output / name
    if not path.is_file() or file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"frozen manifest hash mismatch: {path}")
    if record["format"] == "jsonl":
        rows = read_jsonl(path)
        if len(rows) != int(record["row_count"]):
            raise ValueError(f"frozen manifest row count mismatch: {path}")
        if canonical_sha256(rows) != record["ordered_rows_sha256"]:
            raise ValueError(f"frozen manifest ordered-row hash mismatch: {path}")
        sidecar = output / str(record["sidecar_path"])
        if file_sha256(sidecar) != record["sidecar_file_sha256"]:
            raise ValueError(f"frozen manifest sidecar hash mismatch: {sidecar}")
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(payload) != record["canonical_payload_sha256"]:
        raise ValueError(f"frozen JSON canonical hash mismatch: {path}")
    return payload


def verify_pre_rollout(output: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    registry_path = output / "manifest_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "PASS_PRE_ROLLOUT_MANIFEST_FREEZE":
        raise ValueError("pre-rollout registry does not record a PASS")
    if registry.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("pre-rollout registry protocol hash mismatch")
    if tuple(registry.get("required_files", [])) != REQUIRED_FILES:
        raise ValueError("pre-rollout registry required-file list mismatch")
    payloads = {
        name: _verify_record(output, name, registry["files"][name])
        for name in REQUIRED_FILES
    }
    train = payloads["train_acquisition_queries.jsonl"]
    heldout = payloads["heldout_acquisition_queries.jsonl"]
    shards = payloads["rollout_shards.json"]
    if len(train) != 1500 or len(heldout) != 500 or len(shards) != 40:
        raise ValueError("frozen v6 population count mismatch")
    if _source_counts(train) != {"gsm8k": 450, "math": 1050}:
        raise ValueError("frozen train source count mismatch")
    if _source_counts(heldout) != {"gsm8k": 150, "math": 350}:
        raise ValueError("frozen heldout source count mismatch")
    train_ids = {str(row["query_id"]) for row in train}
    heldout_ids = {str(row["query_id"]) for row in heldout}
    train_clusters = {str(row["cluster_id"]) for row in train}
    heldout_clusters = {str(row["cluster_id"]) for row in heldout}
    if train_ids & heldout_ids or train_clusters & heldout_clusters:
        raise ValueError("frozen train/heldout query or cluster leakage")
    shard_ids = [query_id for shard in shards for query_id in shard["query_ids"]]
    if len(shard_ids) != 2000 or set(shard_ids) != train_ids | heldout_ids:
        raise ValueError("rollout shard membership is not an exact query partition")
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("a query occurs in more than one rollout shard")
    for shard in shards:
        if (
            int(shard["query_count"]) != 50
            or shard["source_counts"] != {"math": 35, "gsm8k": 15}
            or int(shard["expected_candidate_rows"])
            != 50 * int(protocol["generation"]["candidate_count"])
        ):
            raise ValueError(f"invalid rollout shard: {shard['shard_id']}")
    report_record = registry["report"]
    report = _verify_record(output, "pre_rollout_report.json", report_record)
    if report.get("rollout_started") is not False:
        raise ValueError("pre-rollout report incorrectly claims generation started")
    return {
        "status": "PASS_PRE_ROLLOUT_MANIFEST_VERIFY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "manifest_registry_file_sha256": file_sha256(registry_path),
        "train_queries": len(train),
        "heldout_queries": len(heldout),
        "rollout_shards": len(shards),
        "planned_raw_trajectories": 16000,
        "rollout_started": False,
    }


def load_rollout_authorization(
    path: Path, *, protocol_path: Path, pre_rollout_dir: Path
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != (
        "clir-consistency-scale-v6-rollout-authorization"
    ):
        raise ValueError("unsupported scale-v6 rollout authorization schema")
    if authorization.get("status") != "AUTHORIZED_ROLLOUT_ONLY":
        raise ValueError("scale-v6 rollout has not received explicit authorization")
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
        raise ValueError("authorization must permit rollout and no later data stage")
    parent = authorization["frozen_parent"]
    if file_sha256(protocol_path) != parent["protocol_file_sha256"]:
        raise ValueError("rollout authorization protocol hash mismatch")
    registry_path = pre_rollout_dir / "manifest_registry.json"
    report_path = pre_rollout_dir / "pre_rollout_report.json"
    if file_sha256(registry_path) != parent["manifest_registry_file_sha256"]:
        raise ValueError("rollout authorization registry hash mismatch")
    if file_sha256(report_path) != parent["pre_rollout_report_file_sha256"]:
        raise ValueError("rollout authorization pre-rollout report hash mismatch")
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
    train = read_jsonl(pre_rollout_dir / "train_acquisition_queries.jsonl")
    heldout = read_jsonl(pre_rollout_dir / "heldout_acquisition_queries.jsonl")
    shards = json.loads(
        (pre_rollout_dir / "rollout_shards.json").read_text(encoding="utf-8")
    )
    query_by_id = {str(row["query_id"]): row for row in [*train, *heldout]}
    if len(query_by_id) != 2000:
        raise ValueError("frozen acquisition manifests do not contain 2,000 unique IDs")
    return protocol, authorization, shards, query_by_id


def _select_shard(
    shards: Sequence[Mapping[str, Any]], shard_id: str
) -> dict[str, Any]:
    matches = [dict(row) for row in shards if row.get("shard_id") == shard_id]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen shard named {shard_id}, found {len(matches)}")
    return matches[0]


def _shard_paths(rollout_root: Path, shard: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    output = rollout_root / str(shard["output_path"])
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    completion = output.with_suffix(".complete.json")
    return output, sidecar, completion


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
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered or encountered[-1] != query_id:
            encountered.append(query_id)
    if encountered != expected_query_ids:
        raise ValueError(f"{shard['shard_id']}: rollout query order differs from freeze")
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
        for field in (
            "source",
            "question",
            "reference_answer",
            "cluster_id",
            "acquisition_split",
        ):
            if row.get(field) != query.get(field):
                raise ValueError(f"{row['id']}: {field} differs from frozen query")
        if row.get("shard_id") != shard["shard_id"]:
            raise ValueError(f"{row['id']}: shard_id mismatch")
        if int(query["prompt_token_count"]) != len(row["prompt_token_ids"]):
            raise ValueError(f"{row['id']}: prompt token count differs from freeze")
        expected_seed = _derive_query_seed(
            int(protocol["generation"]["seed"]), query_id
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
    if completion.get("status") != "COMPLETE_VERIFIED_ROLLOUT_SHARD_V6":
        raise ValueError(f"invalid completion status: {completion_path}")
    if completion.get("shard_id") != shard["shard_id"]:
        raise ValueError(f"completion marker shard mismatch: {completion_path}")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_bindings = {
        "shard_spec_sha256": canonical_sha256(shard),
        "protocol_file_sha256": authorization["frozen_parent"][
            "protocol_file_sha256"
        ],
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
    if not sidecar_path.is_file() or file_sha256(sidecar_path) != completion[
        "sidecar_file_sha256"
    ]:
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
        "status": "PASS_ROLLOUT_SHARD_VERIFY_V6",
        "shard_id": shard["shard_id"],
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
    calibration_id = str(
        authorization["runtime_contract"]["first_calibration_shard"]
    )
    if shard["shard_id"] != calibration_id:
        calibration = _select_shard(shards, calibration_id)
        _, _, calibration_completion = _shard_paths(rollout_root, calibration)
        if not calibration_completion.exists():
            raise RuntimeError(
                f"{calibration_id} must complete and verify before other shards"
            )
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
            f"{shard['shard_id']} has incomplete existing artifacts; "
            "authorization forbids automatic overwrite"
        )
    if _git_dirty():
        raise RuntimeError("scale-v6 rollout requires a clean Git commit")
    runtime_contract = authorization["runtime_contract"]
    if int(runtime_contract["tensor_parallel_size"]) != 1:
        raise ValueError("scale-v6 authorization requires TP=1")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("scale-v6 rollout requires torch and vLLM") from exc
    if _package_version("vllm") != protocol["generation"]["backend_version"]:
        raise ValueError("installed vLLM version differs from frozen protocol")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each shard process must see exactly one GPU; set CUDA_VISIBLE_DEVICES"
        )
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 40_000_000_000:
        raise RuntimeError("visible GPU has less than 40 GB free before model load")

    query_rows = [query_by_id[str(value)] for value in shard["query_ids"]]
    generation = protocol["generation"]
    started_at = _utc_now()
    started_clock = time.monotonic()
    llm = LLM(
        model=generation["model_id"],
        revision=generation["model_revision"],
        tokenizer_revision=generation["tokenizer_revision"],
        dtype=runtime_contract["dtype"],
        tensor_parallel_size=1,
        max_model_len=int(generation["max_model_length"]),
        max_num_seqs=int(runtime_contract["max_num_seqs"]),
        gpu_memory_utilization=float(runtime_contract["gpu_memory_utilization"]),
        seed=int(generation["seed"]),
        download_dir=str(_project_path(runtime_contract["cache_dir"])),
    )
    tokenizer = llm.get_tokenizer()
    rendered_prompts: list[str] = []
    sampling: list[Any] = []
    expected_prompt_ids: list[list[int]] = []
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
                n=int(generation["candidate_count"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_derive_query_seed(int(generation["seed"]), query["query_id"]),
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
        "dtype": runtime_contract["dtype"],
        "gpu_model": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "max_num_seqs": int(runtime_contract["max_num_seqs"]),
        "gpu_memory_utilization": float(runtime_contract["gpu_memory_utilization"]),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": elapsed_seconds,
    }
    rows: list[dict[str, Any]] = []
    candidate_count = int(generation["candidate_count"])
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
                    "acquisition_split": query["acquisition_split"],
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
                        int(generation["seed"]), str(query["query_id"])
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
        schema_version="clir-consistency-scale-raw-rollout-shard-v6",
        metadata={**provenance, **validation},
    )
    completion = {
        "schema_version": "clir-consistency-scale-rollout-shard-completion-v6",
        "status": "COMPLETE_VERIFIED_ROLLOUT_SHARD_V6",
        "shard_id": shard["shard_id"],
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
    result = verify_rollout_shard(
        shard=shard,
        query_by_id=query_by_id,
        protocol=protocol,
        authorization_path=authorization_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
        _, _, completion_path = _shard_paths(rollout_root, shard)
        if not completion_path.exists():
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
                    "PASS_ALL_ROLLOUT_SHARDS_VERIFY_V6"
                    if not missing and len(completed) == len(shards)
                    else "PARTIAL_ROLLOUT_SHARDS_VERIFY_V6"
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
    output = rollout_root / "rollouts/combined_raw.jsonl"
    sidecar_path = output.with_suffix(output.suffix + ".manifest.json")
    report_path = rollout_root / "rollout_completion_report.json"
    if output.exists() or sidecar_path.exists() or report_path.exists():
        raise FileExistsError("combined rollout artifacts already exist; never overwrite")
    shard_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for shard in shards:
        report = verify_rollout_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout_dir,
            rollout_root=rollout_root,
        )
        shard_reports.append(report)
        rows.extend(read_jsonl(Path(report["path"])))
    population = validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("rollout shards overlap by trajectory ID")
    expected_queries = [
        str(query_id) for shard in shards for query_id in shard["query_ids"]
    ]
    encountered_queries: list[str] = []
    for row in rows:
        query_id = str(row["query_id"])
        if not encountered_queries or encountered_queries[-1] != query_id:
            encountered_queries.append(query_id)
    if encountered_queries != expected_queries:
        raise ValueError("combined rollout query order differs from frozen shard order")
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    output_lengths = [len(row["output_token_ids"]) for row in rows]
    code_commits = {str(row["provenance"]["code_commit"]) for row in rows}
    if len(code_commits) != 1:
        raise ValueError("rollout shards were produced by different code commits")
    metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": next(iter(code_commits)),
        "shard_count": len(shards),
        "shard_file_sha256": {
            row["shard_id"]: row["file_sha256"] for row in shard_reports
        },
        **population,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
    }
    manifest = publish_manifest(
        output,
        rows,
        schema_version="clir-consistency-scale-combined-raw-rollouts-v6",
        metadata=metadata,
    )
    report = {
        "schema_version": "clir-consistency-scale-rollout-completion-report-v6",
        "status": "PASS_ALL_16000_RAW_ROLLOUTS_VERIFIED_V6",
        "annotation_started": False,
        "feature_extraction_started": False,
        "training_started": False,
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout_dir / "manifest_registry.json"
        ),
        "authorization_file_sha256": file_sha256(authorization_path),
        "code_commit": next(iter(code_commits)),
        "combined_file_sha256": manifest["file_sha256"],
        "combined_sidecar_file_sha256": file_sha256(sidecar_path),
        "combined_ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "population": population,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "output_token_count": _integer_summary(output_lengths),
        "total_output_tokens": sum(output_lengths),
        "shards": shard_reports,
        "next_gate": "report_rollout_health_before_checker_and_unitizer_materialization",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    result = verify_pre_rollout(
        Path(args.output_dir).resolve(), Path(args.protocol).resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="publish the six scale-v6 pre-rollout manifests"
    )
    freeze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    freeze.add_argument("--cache-dir")
    freeze.set_defaults(func=command_freeze)
    verify = subparsers.add_parser(
        "verify", help="verify an existing scale-v6 pre-rollout freeze"
    )
    verify.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    verify.set_defaults(func=command_verify)
    rollout = subparsers.add_parser(
        "rollout", help="generate one authorized, frozen 50-query rollout shard"
    )
    rollout.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    rollout.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    rollout.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=command_rollout)
    verify_rollouts = subparsers.add_parser(
        "verify-rollouts", help="verify completed scale-v6 rollout shards"
    )
    verify_rollouts.add_argument(
        "--authorization", default=str(DEFAULT_AUTHORIZATION)
    )
    verify_rollouts.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    verify_rollouts.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    verify_rollouts.add_argument("--shard-id", action="append")
    verify_rollouts.add_argument("--require-complete", action="store_true")
    verify_rollouts.set_defaults(func=command_verify_rollouts)
    merge = subparsers.add_parser(
        "merge-rollouts", help="verify and merge all 40 authorized rollout shards"
    )
    merge.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    merge.add_argument("--pre-rollout-dir", default=str(DEFAULT_OUTPUT))
    merge.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    merge.set_defaults(func=command_merge_rollouts)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
