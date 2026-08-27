#!/usr/bin/env python
"""Freeze and verify CLIR Consistency scale-v6 pre-rollout manifests.

This entry point intentionally has no rollout, annotation, extraction, or
training command.  ``freeze`` reads only pinned train-source artifacts and a
pinned tokenizer, then publishes the six manifests required by the v6 gate.
``verify`` rechecks those files without modifying them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
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
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_scale_v6/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_scale_v6/pre_rollout"
REQUIRED_FILES = (
    "source_inventory.json",
    "permanent_exclusions.jsonl",
    "template_clusters.jsonl",
    "train_acquisition_queries.jsonl",
    "heldout_acquisition_queries.jsonl",
    "rollout_shards.json",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
