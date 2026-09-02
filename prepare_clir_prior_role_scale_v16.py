#!/usr/bin/env python
"""Prepare, verify, evaluate, and materialize CLIR Prior role scale-v16."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.clir_prior_role_scale_v16 import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    ROW_SCHEMA,
    build_blind_shards_v16,
    construct_silver_rows_v16,
    evaluate_role_scale_v16,
    select_scale_rows_v16,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v16/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v16/pre_annotation"
DEFAULT_PROMPT = PROJECT_ROOT / "configs/data_expansion_prior_v16/annotation_prompt.md"
DEFAULT_LAUNCH_A = PROJECT_ROOT / "configs/data_expansion_prior_v16/launch_prompt_a.txt"
DEFAULT_LAUNCH_B = PROJECT_ROOT / "configs/data_expansion_prior_v16/launch_prompt_b.txt"
ROLE_SOURCE = PROJECT_ROOT / "src/clir_prior_role_scale_v16.py"


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
    if protocol.get("schema_version") != "clir-prior-role-only-scale-v16":
        raise ValueError("unsupported Prior v16 protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_ANY_V16_LABEL":
        raise ValueError("Prior v16 protocol is not frozen before labels")
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
        "v15_protocol",
        "v15_proposals",
        "v15_terminal_report",
    )
    bindings: dict[str, str] = {}
    for name in names:
        path = _project_path(parent[f"{name}_path"])
        actual = file_sha256(path)
        if actual != str(parent[f"{name}_file_sha256"]):
            raise ValueError(f"Prior v16 parent hash drift: {name}")
        bindings[f"{name}_file_sha256"] = actual
    for version in ("v12", "v13", "v14", "v15"):
        report = json.loads(
            _project_path(parent[f"{version}_terminal_report_path"]).read_text(
                encoding="utf-8"
            )
        )
        if report.get("status") != parent[f"{version}_terminal_status"]:
            raise ValueError(f"Prior v16 parent status drift: {version}")
    if (
        parent.get("all_v12_v13_v14_v15_queries_and_clusters_excluded") is not True
        or parent.get("all_prior_terminal_decisions_are_immutable") is not True
    ):
        raise ValueError("Prior v16 historical exclusion policy drift")
    return bindings


def _recompute(protocol: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    parent = protocol["parent"]
    materialized = read_jsonl(_project_path(parent["v12_materialized_path"]))
    exclusions: list[dict[str, Any]] = []
    for version in ("v12", "v13", "v14", "v15"):
        exclusions.extend(read_jsonl(_project_path(parent[f"{version}_proposals_path"])))
    excluded_queries = {str(row["query_id"]) for row in exclusions}
    excluded_clusters = {str(row["cluster_id"]) for row in exclusions}
    pool = protocol["proposal_pool"]
    proposals, selection = select_scale_rows_v16(
        materialized,
        excluded_query_ids=excluded_queries,
        excluded_cluster_ids=excluded_clusters,
        strata=pool["strata"],
        minimum_material_claims=int(pool["minimum_material_claims"]),
        maximum_material_claims=int(pool["maximum_material_claims"]),
        namespace=str(pool["selection_namespace"]),
    )
    annotation = protocol["annotation"]
    packages, private, construction = build_blind_shards_v16(
        proposals,
        shard_count=int(annotation["shards_per_annotator"]),
        natural_per_shard=int(annotation["natural_rows_per_shard"]),
        repeats_per_shard=int(annotation["self_repeats_per_shard"]),
        namespace=str(pool["selection_namespace"]),
    )
    natural_packages = [row for shard in packages["a"] for row in shard]
    natural_ids = {row["item_id"] for row in proposals}
    block_counts = [
        int(row["structure"]["block_count"])
        for row in natural_packages
        if row["item_id"] in natural_ids
    ]
    burden = {
        "natural_rows": len(block_counts),
        "min_blocks_per_row": min(block_counts),
        "mean_blocks_per_row": sum(block_counts) / len(block_counts),
        "max_blocks_per_row": max(block_counts),
        "role_decisions_per_annotator": sum(block_counts),
        "edge_decisions_per_row": 0,
        "key_or_complete_set_decisions_per_row": 0,
    }
    return proposals, selection, packages, private, construction, burden


def _package_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / (
        f"packages/annotator_{annotator}/prior_v16_{annotator}_{shard_index:02d}.jsonl"
    )


def _label_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / f"labels_{annotator}/prior_v16_{annotator}_{shard_index:02d}.jsonl"


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
        raise RuntimeError("Prior v16 package freeze requires a clean Git commit")
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    if root.exists() and any(root.rglob("*")):
        raise FileExistsError(f"Prior v16 output is not empty: {root}")
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction, burden = _recompute(protocol)
    common = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "labels_are_gold": False,
        "human_verification": False,
        "v15_smoke_rows_trainable": False,
        **parent_bindings,
        **_prompt_bindings(),
        **_code_bindings(),
    }
    proposal_manifest = publish_manifest(
        root / "proposals/prior_role_natural_600.jsonl",
        proposals,
        schema_version=PROPOSAL_SCHEMA,
        metadata={**common, **selection},
    )
    public_manifests: dict[str, dict[str, Any]] = {}
    for annotator in ("a", "b"):
        for shard_index, rows in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            public_manifests[key] = publish_manifest(
                _package_path(root, annotator, shard_index),
                rows,
                schema_version=PACKAGE_SCHEMA,
                metadata={**common, "annotation_shard_id": key},
            )
    private_manifest = publish_manifest(
        root / "packages/PRIVATE_package_index.jsonl",
        private,
        schema_version=PRIVATE_SCHEMA,
        metadata={**common, "visibility": "PRIVATE_NEVER_SEND_TO_ANNOTATORS"},
    )
    report = {
        "schema_version": "clir-prior-role-only-package-report-v16",
        "status": "PASS_PRIOR_V16_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "parent_bindings": parent_bindings,
        "prompt_bindings": _prompt_bindings(),
        "code_bindings": _code_bindings(),
        "proposal": {
            key: proposal_manifest[key]
            for key in ("path", "row_count", "file_sha256", "ordered_rows_sha256")
        },
        "selection": selection,
        "construction": construction,
        "role_burden": burden,
        "public_shards": {
            key: {
                field: value[field]
                for field in ("path", "row_count", "file_sha256", "ordered_rows_sha256")
            }
            for key, value in sorted(public_manifests.items())
        },
        "private_index": {
            key: private_manifest[key]
            for key in ("path", "row_count", "file_sha256", "ordered_rows_sha256")
        },
        "annotation_started": False,
        "feature_extraction_started": False,
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
    proposal_path = root / "proposals/prior_role_natural_600.jsonl"
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
        "status": "PASS_PRIOR_V16_FRESH_BLIND_PACKAGES_READY",
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
    for key, actual in public_hashes.items():
        if report.get("public_shards", {}).get(key, {}).get("file_sha256") != actual:
            mismatches.append(f"package_report:public_hash:{key}")
    verification = {
        "schema_version": "clir-prior-role-only-package-verification-v16",
        "status": (
            "PASS_PRIOR_V16_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V16_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(report_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "private_index_file_sha256": file_sha256(private_path),
        "public_shards": len(public_hashes),
        "public_rows_total": sum(
            len(shard) for side in packages.values() for shard in side
        ),
        "labels_present_at_freeze_verification": False,
        "next_gate": "user_runs_two_independent_max_reasoning_annotators",
    }
    path = root / "packages/independent_verification.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != verification:
        raise ValueError("Prior v16 package verification drift")
    if not path.exists():
        atomic_write_json(path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def _load_annotation_inputs(
    protocol: Mapping[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, str],
]:
    shard_count = int(protocol["annotation"]["shards_per_annotator"])
    rows_per_shard = int(protocol["annotation"]["rows_per_shard"])
    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    label_hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        directory = root / f"labels_{annotator}"
        expected_paths = {
            _label_path(root, annotator, index).resolve()
            for index in range(shard_count)
        }
        actual_paths = {path.resolve() for path in directory.glob("*.jsonl")}
        if actual_paths != expected_paths:
            raise ValueError(
                f"Prior v16 labels_{annotator} population mismatch: "
                f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                f"extra={sorted(map(str, actual_paths - expected_paths))}"
            )
        for shard_index in range(shard_count):
            key = f"{annotator}-{shard_index:02d}"
            package_rows = read_jsonl(_package_path(root, annotator, shard_index))
            label_path = _label_path(root, annotator, shard_index)
            label_rows = read_jsonl(label_path)
            if len(package_rows) != rows_per_shard or len(label_rows) != rows_per_shard:
                raise ValueError(f"Prior v16 shard {key} must contain {rows_per_shard} rows")
            package_ids = [str(row["item_id"]) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(package_ids):
                raise ValueError(f"Prior v16 label/package ID mismatch: {key}")
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
            label_hashes[key] = file_sha256(label_path)
    proposals = read_jsonl(root / "proposals/prior_role_natural_600.jsonl")
    private = read_jsonl(root / "packages/PRIVATE_package_index.jsonl")
    return proposals, packages, private, labels, label_hashes


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    verification_path = root / "packages/independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS_PRIOR_V16_PACKAGE_INDEPENDENT_RECOMPUTE":
        raise ValueError("Prior v16 package verification is not PASS")
    package_report_path = root / "packages/package_report.json"
    if verification.get("package_report_file_sha256") != file_sha256(package_report_path):
        raise ValueError("Prior v16 package report binding drift")
    proposals, packages, private, labels, label_hashes = _load_annotation_inputs(
        protocol, root
    )
    report = evaluate_role_scale_v16(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
        final_strata=protocol["prospective_selection"]["final_strata"],
        gates=protocol["gates"],
    )
    report["bindings"] = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(package_report_path),
        "package_verification_file_sha256": file_sha256(verification_path),
        "private_index_file_sha256": file_sha256(
            root / "packages/PRIVATE_package_index.jsonl"
        ),
        "label_file_sha256": dict(sorted(label_hashes.items())),
    }
    report["raw_label_eligibility_counts"] = dict(
        sorted(
            Counter(
                str(row.get("eligibility"))
                for side in labels.values()
                for row in side
            ).items()
        )
    )
    path = root / "evaluation/raw_gate_report.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise ValueError("Prior v16 evaluation report already exists with different content")
    if not path.exists():
        atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not str(report["status"]).startswith("PASS"):
        raise SystemExit(1)


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    evaluation_path = root / "evaluation/raw_gate_report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    proposals, packages, private, labels, label_hashes = _load_annotation_inputs(
        protocol, root
    )
    bindings = evaluation.get("bindings", {})
    if bindings.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("Prior v16 evaluation/protocol binding drift")
    if bindings.get("label_file_sha256") != dict(sorted(label_hashes.items())):
        raise ValueError("Prior v16 evaluation/label binding drift")
    rows, report = construct_silver_rows_v16(
        proposals=proposals,
        materialized_rows=read_jsonl(
            _project_path(protocol["parent"]["v12_materialized_path"])
        ),
        packages=packages,
        private_index=private,
        labels=labels,
        evaluation_report=evaluation,
    )
    published = root / "published"
    if published.exists() and any(published.rglob("*")):
        raise FileExistsError("Prior v16 published output is not empty")
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    common = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "evaluation_report_file_sha256": file_sha256(evaluation_path),
        "labels_are_gold": False,
        "human_verification": False,
    }
    manifests = {
        "train": publish_manifest(
            published / "prior_role_silver_train_400.jsonl",
            train,
            schema_version=ROW_SCHEMA,
            metadata=common,
        ),
        "dev": publish_manifest(
            published / "prior_role_silver_dev_100.jsonl",
            dev,
            schema_version=ROW_SCHEMA,
            metadata=common,
        ),
    }
    report["bindings"] = common
    report["manifests"] = {
        name: {
            key: manifest[key]
            for key in ("path", "row_count", "file_sha256", "ordered_rows_sha256")
        }
        for name, manifest in manifests.items()
    }
    atomic_write_json(published / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify_materialized(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    evaluation_path = root / "evaluation/raw_gate_report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    proposals, packages, private, labels, _ = _load_annotation_inputs(protocol, root)
    expected, expected_report = construct_silver_rows_v16(
        proposals=proposals,
        materialized_rows=read_jsonl(
            _project_path(protocol["parent"]["v12_materialized_path"])
        ),
        packages=packages,
        private_index=private,
        labels=labels,
        evaluation_report=evaluation,
    )
    actual = read_jsonl(root / "published/prior_role_silver_train_400.jsonl") + read_jsonl(
        root / "published/prior_role_silver_dev_100.jsonl"
    )
    # Published files are split, so compare by the frozen selection index.
    actual.sort(key=lambda row: int(row["selection_index"]))
    mismatches = []
    if canonical_sha256(actual) != canonical_sha256(expected):
        mismatches.append("published_rows_recompute")
    report_path = root / "published/materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key, value in expected_report.items():
        if report.get(key) != value:
            mismatches.append(f"materialization_report:{key}")
    verification = {
        "schema_version": "clir-prior-role-only-materialization-verification-v16",
        "status": (
            "PASS_PRIOR_V16_SILVER_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V16_SILVER_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "rows": len(actual),
        "train_rows": sum(row["split"] == "train" for row in actual),
        "dev_rows": sum(row["split"] == "dev" for row in actual),
        "materialization_report_file_sha256": file_sha256(report_path),
    }
    path = root / "published/independent_verification.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != verification:
        raise ValueError("Prior v16 materialization verification drift")
    if not path.exists():
        atomic_write_json(path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "verify", "evaluate", "materialize", "verify-materialized"),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "verify":
        command_verify(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "materialize":
        command_materialize(args)
    else:
        command_verify_materialized(args)


if __name__ == "__main__":
    main()
