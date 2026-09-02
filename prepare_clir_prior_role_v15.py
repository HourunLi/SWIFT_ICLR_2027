#!/usr/bin/env python
"""Prepare, verify, and evaluate the fresh CLIR Prior role-only-v15 smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.clir_prior_role_v15 import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    build_blind_shards_v15,
    evaluate_blind_labels_v15,
    select_fresh_natural_rows_v15,
    summarize_role_burden,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v15/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v15/pre_annotation"
DEFAULT_PROMPT = PROJECT_ROOT / "configs/data_expansion_prior_v15/annotation_prompt.md"
DEFAULT_LAUNCH_A = PROJECT_ROOT / "configs/data_expansion_prior_v15/launch_prompt_a.txt"
DEFAULT_LAUNCH_B = PROJECT_ROOT / "configs/data_expansion_prior_v15/launch_prompt_b.txt"
ROLE_SOURCE = PROJECT_ROOT / "src/clir_prior_role_v15.py"


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


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-prior-role-only-smoke-v15":
        raise ValueError("unsupported Prior v15 protocol schema")
    return protocol


def _verify_parent_files(protocol: Mapping[str, Any]) -> dict[str, str]:
    parent = protocol["parent"]
    names = (
        "v12_protocol",
        "v12_materialized",
        "v12_proposals",
        "v12_terminal_report",
        "v13_protocol",
        "v13_proposals",
        "v13_terminal_report",
        "v14_protocol",
        "v14_proposals",
        "v14_terminal_report",
        "v15_dev_report",
    )
    bindings: dict[str, str] = {}
    for name in names:
        path = _project_path(parent[f"{name}_path"])
        expected = str(parent[f"{name}_file_sha256"])
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Prior v15 parent hash drift: {name}")
        bindings[f"{name}_file_sha256"] = actual

    status_bindings = (
        ("v12_terminal_report_path", "v12_terminal_status"),
        ("v13_terminal_report_path", "v13_terminal_status"),
        ("v14_terminal_report_path", "v14_terminal_status"),
        ("v15_dev_report_path", "v15_dev_status"),
    )
    for path_key, status_key in status_bindings:
        report = json.loads(_project_path(parent[path_key]).read_text(encoding="utf-8"))
        if report.get("status") != parent[status_key]:
            raise ValueError(f"Prior v15 parent status drift: {status_key}")
    return bindings


def _recompute(
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    parent = protocol["parent"]
    all_rows = read_jsonl(_project_path(parent["v12_materialized_path"]))
    exclusions: list[dict[str, Any]] = []
    for key in ("v12_proposals_path", "v13_proposals_path", "v14_proposals_path"):
        exclusions.extend(read_jsonl(_project_path(parent[key])))
    excluded_queries = {str(row["query_id"]) for row in exclusions}
    excluded_clusters = {str(row["cluster_id"]) for row in exclusions}
    proposals, selection = select_fresh_natural_rows_v15(
        all_rows,
        excluded_query_ids=excluded_queries,
        excluded_cluster_ids=excluded_clusters,
        strata=protocol["fresh_selection"]["strata"],
        namespace=str(protocol["fresh_selection"]["selection_namespace"]),
    )
    packages, private, construction = build_blind_shards_v15(
        proposals,
        shard_count=int(protocol["annotation"]["shards_per_annotator"]),
        repeats_per_shard=int(protocol["annotation"]["self_repeats_per_shard"]),
        namespace=str(protocol["fresh_selection"]["selection_namespace"]),
    )
    burden = summarize_role_burden(proposals)
    return proposals, selection, packages, private, construction, burden


def _package_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / f"packages/annotator_{annotator}/prior_v15_{annotator}_{shard_index:02d}.jsonl"


def _label_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / f"labels_{annotator}/prior_v15_{annotator}_{shard_index:02d}.jsonl"


def _prompt_bindings() -> dict[str, str]:
    return {
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_PROMPT),
        "launch_prompt_a_file_sha256": file_sha256(DEFAULT_LAUNCH_A),
        "launch_prompt_b_file_sha256": file_sha256(DEFAULT_LAUNCH_B),
    }


def _code_bindings() -> dict[str, str]:
    return {
        "role_source_file_sha256": file_sha256(ROLE_SOURCE),
        "prepare_source_file_sha256": file_sha256(Path(__file__).resolve()),
    }


def command_prepare(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v15 package freeze requires a clean Git commit")
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    if root.exists() and any(root.rglob("*")):
        raise FileExistsError(f"Prior v15 output is not empty: {root}")
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction, burden = _recompute(protocol)
    common_metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "labels_are_gold": False,
        "human_verification": False,
        "v14_terminal_decision_unchanged": True,
        "v15_dev_projection_is_nontrainable": True,
        **parent_bindings,
        **_prompt_bindings(),
        **_code_bindings(),
    }
    proposal_path = root / "proposals/prior_role_natural_48.jsonl"
    proposal_manifest = publish_manifest(
        proposal_path,
        proposals,
        schema_version=PROPOSAL_SCHEMA,
        metadata={**common_metadata, **selection},
    )
    public_manifests: dict[str, dict[str, Any]] = {}
    for annotator in ("a", "b"):
        for shard_index, rows in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            public_manifests[key] = publish_manifest(
                _package_path(root, annotator, shard_index),
                rows,
                schema_version=PACKAGE_SCHEMA,
                metadata={**common_metadata, "annotation_shard_id": key},
            )
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    private_manifest = publish_manifest(
        private_path,
        private,
        schema_version=PRIVATE_SCHEMA,
        metadata={
            **common_metadata,
            "visibility": "PRIVATE_NEVER_SEND_TO_ANNOTATORS",
        },
    )
    report = {
        "schema_version": "clir-prior-role-only-package-report-v15",
        "status": "PASS_PRIOR_V15_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "parent_bindings": parent_bindings,
        "prompt_bindings": _prompt_bindings(),
        "code_bindings": _code_bindings(),
        "proposal": {
            "path": proposal_manifest["path"],
            "row_count": proposal_manifest["row_count"],
            "file_sha256": proposal_manifest["file_sha256"],
            "ordered_rows_sha256": proposal_manifest["ordered_rows_sha256"],
        },
        "selection": selection,
        "construction": construction,
        "role_burden": burden,
        "public_shards": {
            key: {
                "path": value["path"],
                "row_count": value["row_count"],
                "file_sha256": value["file_sha256"],
                "ordered_rows_sha256": value["ordered_rows_sha256"],
            }
            for key, value in sorted(public_manifests.items())
        },
        "private_index": {
            "path": private_manifest["path"],
            "row_count": private_manifest["row_count"],
            "file_sha256": private_manifest["file_sha256"],
            "ordered_rows_sha256": private_manifest["ordered_rows_sha256"],
        },
        "annotation_started": False,
        "training_started": False,
        "trainable_labels_published": False,
        "next_gate": "independent_package_recompute_before_dual_ai_annotation",
    }
    atomic_write_json(root / "packages/package_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction, burden = _recompute(protocol)
    mismatches: list[str] = []

    proposal_path = root / "proposals/prior_role_natural_48.jsonl"
    if canonical_sha256(read_jsonl(proposal_path)) != canonical_sha256(proposals):
        mismatches.append("proposal_recompute")
    public_hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        for shard_index, expected in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            path = _package_path(root, annotator, shard_index)
            actual = read_jsonl(path)
            if canonical_sha256(actual) != canonical_sha256(expected):
                mismatches.append(f"package:{key}")
            allowed = {"schema_version", "item_id", "question", "response", "units", "structure"}
            if any(set(row) != allowed for row in actual):
                mismatches.append(f"public_field_leak:{key}")
            if any(row.get("schema_version") != PACKAGE_SCHEMA for row in actual):
                mismatches.append(f"public_schema:{key}")
            if any("candidate_edges" in row["structure"] for row in actual):
                mismatches.append(f"dependency_edge_leak:{key}")
            public_hashes[key] = file_sha256(path)

    private_path = root / "packages/PRIVATE_package_index.jsonl"
    if canonical_sha256(read_jsonl(private_path)) != canonical_sha256(private):
        mismatches.append("private_index_recompute")
    report_path = root / "packages/package_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "status": "PASS_PRIOR_V15_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "parent_bindings": parent_bindings,
        "prompt_bindings": _prompt_bindings(),
        "code_bindings": _code_bindings(),
        "selection": selection,
        "construction": construction,
        "role_burden": burden,
        "code_dirty": False,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            mismatches.append(f"package_report:{key}")
    for key, actual_hash in public_hashes.items():
        if report.get("public_shards", {}).get(key, {}).get("file_sha256") != actual_hash:
            mismatches.append(f"package_report:public_hash:{key}")

    shard_count = int(protocol["annotation"]["shards_per_annotator"])
    verification = {
        "schema_version": "clir-prior-role-only-package-verification-v15",
        "status": (
            "PASS_PRIOR_V15_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V15_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(report_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "private_index_file_sha256": file_sha256(private_path),
        "public_shards": len(public_hashes),
        "public_rows_total": sum(len(rows) for side in packages.values() for rows in side),
        "labels_present": any(
            _label_path(root, annotator, shard_index).is_file()
            for annotator in ("a", "b")
            for shard_index in range(shard_count)
        ),
        "v14_terminal_decision_unchanged": True,
        "next_gate": "user_runs_two_independent_max_reasoning_annotators",
    }
    verification_path = root / "packages/independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        if old != verification:
            raise ValueError("Prior v15 package verification drift")
    else:
        atomic_write_json(verification_path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    verification_path = root / "packages/independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS_PRIOR_V15_PACKAGE_INDEPENDENT_RECOMPUTE":
        raise ValueError("Prior v15 package verification is not PASS")
    package_report_path = root / "packages/package_report.json"
    if verification.get("package_report_file_sha256") != file_sha256(package_report_path):
        raise ValueError("Prior v15 package report binding drift")

    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    label_hashes: dict[str, str] = {}
    shard_count = int(protocol["annotation"]["shards_per_annotator"])
    rows_per_shard = int(protocol["annotation"]["rows_per_shard"])
    for annotator in ("a", "b"):
        directory = root / f"labels_{annotator}"
        expected_paths = {
            _label_path(root, annotator, index).resolve() for index in range(shard_count)
        }
        actual_paths = {path.resolve() for path in directory.glob("*.jsonl")}
        if actual_paths != expected_paths:
            raise ValueError(
                f"Prior v15 labels_{annotator} population mismatch: "
                f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                f"extra={sorted(map(str, actual_paths - expected_paths))}"
            )
        for shard_index in range(shard_count):
            key = f"{annotator}-{shard_index:02d}"
            package_rows = read_jsonl(_package_path(root, annotator, shard_index))
            label_path = _label_path(root, annotator, shard_index)
            label_rows = read_jsonl(label_path)
            if len(package_rows) != rows_per_shard or len(label_rows) != rows_per_shard:
                raise ValueError(f"Prior v15 shard {key} must contain {rows_per_shard} rows")
            package_ids = [str(row["item_id"]) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(package_ids):
                raise ValueError(f"Prior v15 label/package ID mismatch: {key}")
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
            label_hashes[key] = file_sha256(label_path)

    private_path = root / "packages/PRIVATE_package_index.jsonl"
    report = evaluate_blind_labels_v15(
        packages=packages,
        private_index=read_jsonl(private_path),
        labels=labels,
        gates=protocol["gates"],
    )
    report["bindings"] = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(package_report_path),
        "package_verification_file_sha256": file_sha256(verification_path),
        "private_index_file_sha256": file_sha256(private_path),
        "label_file_sha256": dict(sorted(label_hashes.items())),
    }
    report["label_counts"] = dict(
        sorted(Counter(str(row.get("eligibility")) for side in labels.values() for row in side).items())
    )
    evaluation_path = root / "evaluation/raw_gate_report.json"
    if evaluation_path.exists():
        old = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if old != report:
            raise ValueError("Prior v15 evaluation report already exists with different content")
    else:
        atomic_write_json(evaluation_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not str(report["status"]).startswith("PASS"):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "verify":
        command_verify(args)
    else:
        command_evaluate(args)


if __name__ == "__main__":
    main()
