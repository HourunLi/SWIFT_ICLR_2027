#!/usr/bin/env python
"""Prepare, verify, and evaluate the non-trainable CLIR Prior smoke-v17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.clir_prior_binary_v17 import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    PROTOCOL_SCHEMA,
    build_blind_shards_v17,
    evaluate_binary_smoke_v17,
    select_fresh_rows_v17,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v17/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v17/pre_annotation"
DEFAULT_PROMPT = PROJECT_ROOT / "configs/data_expansion_prior_v17/annotation_prompt.md"
DEFAULT_LAUNCH_A = PROJECT_ROOT / "configs/data_expansion_prior_v17/launch_prompt_a.txt"
DEFAULT_LAUNCH_B = PROJECT_ROOT / "configs/data_expansion_prior_v17/launch_prompt_b.txt"
SOURCE_FILE = PROJECT_ROOT / "src/clir_prior_binary_v17.py"


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
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior v17 protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_ANY_V17_LABEL":
        raise ValueError("Prior v17 protocol is not frozen before labels")
    return protocol


def _verify_parent_files(protocol: Mapping[str, Any]) -> dict[str, str]:
    parent = protocol["parent"]
    names = ["v12_materialized"]
    for version in range(12, 17):
        names.extend([f"v{version}_protocol", f"v{version}_proposals", f"v{version}_terminal_report"])
    bindings: dict[str, str] = {}
    for name in names:
        path = _project_path(parent[f"{name}_path"])
        actual = file_sha256(path)
        if actual != str(parent[f"{name}_file_sha256"]):
            raise ValueError(f"Prior v17 parent hash drift: {name}")
        bindings[f"{name}_file_sha256"] = actual
    for version in range(12, 17):
        report = json.loads(
            _project_path(parent[f"v{version}_terminal_report_path"]).read_text(
                encoding="utf-8"
            )
        )
        if report.get("status") != parent[f"v{version}_terminal_status"]:
            raise ValueError(f"Prior v17 parent status drift: v{version}")
    if (
        parent.get("all_v12_through_v16_queries_and_clusters_excluded") is not True
        or parent.get("all_prior_terminal_decisions_are_immutable") is not True
    ):
        raise ValueError("Prior v17 historical exclusion policy drift")
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
    rows = read_jsonl(_project_path(parent["v12_materialized_path"]))
    exclusions: list[dict[str, Any]] = []
    for version in range(12, 17):
        exclusions.extend(
            read_jsonl(_project_path(parent[f"v{version}_proposals_path"]))
        )
    pool = protocol["proposal_pool"]
    proposals, selection = select_fresh_rows_v17(
        rows,
        excluded_query_ids={str(row["query_id"]) for row in exclusions},
        excluded_cluster_ids={str(row["cluster_id"]) for row in exclusions},
        strata=pool["strata"],
        namespace=str(pool["selection_namespace"]),
    )
    annotation = protocol["annotation"]
    packages, private, construction = build_blind_shards_v17(
        proposals,
        shard_count=int(annotation["shards_per_annotator"]),
        natural_per_shard=int(annotation["natural_rows_per_shard"]),
        controls_per_shard=int(annotation["hidden_controls_per_shard"]),
        repeats_per_shard=int(annotation["self_repeats_per_shard"]),
        namespace=str(pool["selection_namespace"]),
    )
    natural_ids = {row["item_id"] for row in proposals}
    natural_packages = [
        row for shard in packages["a"] for row in shard if row["item_id"] in natural_ids
    ]
    residual_counts = [len(row["structure"]["residual_block_ids"]) for row in natural_packages]
    burden = {
        "natural_rows": len(natural_packages),
        "min_residual_decisions_per_row": min(residual_counts),
        "mean_residual_decisions_per_row": sum(residual_counts) / len(residual_counts),
        "max_residual_decisions_per_row": max(residual_counts),
        "residual_decisions_per_annotator": sum(residual_counts),
        "ai_key_complete_path_role_or_edge_decisions": 0,
    }
    return proposals, selection, packages, private, construction, burden


def _package_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / (
        f"packages/annotator_{annotator}/prior_v17_{annotator}_{shard_index:02d}.jsonl"
    )


def _label_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / f"labels_{annotator}/prior_v17_{annotator}_{shard_index:02d}.jsonl"


def _prompt_bindings() -> dict[str, str]:
    return {
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_PROMPT),
        "launch_prompt_a_file_sha256": file_sha256(DEFAULT_LAUNCH_A),
        "launch_prompt_b_file_sha256": file_sha256(DEFAULT_LAUNCH_B),
    }


def _code_bindings() -> dict[str, str]:
    return {
        "binary_source_file_sha256": file_sha256(SOURCE_FILE),
        "prepare_source_file_sha256": file_sha256(Path(__file__).resolve()),
    }


def command_prepare(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v17 package freeze requires a clean Git commit")
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    if root.exists() and any(root.rglob("*")):
        raise FileExistsError(f"Prior v17 output is not empty: {root}")
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction, burden = _recompute(protocol)
    common = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "labels_are_gold": False,
        "human_verification": False,
        "v17_smoke_rows_trainable": False,
        **parent_bindings,
        **_prompt_bindings(),
        **_code_bindings(),
    }
    proposal_manifest = publish_manifest(
        root / "proposals/prior_binary_natural_96.jsonl",
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
        "schema_version": "clir-prior-mechanical-key-binary-package-report-v17",
        "status": "PASS_PRIOR_V17_FRESH_BLIND_PACKAGES_READY",
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
        "annotation_burden": burden,
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
    proposal_path = root / "proposals/prior_binary_natural_96.jsonl"
    if canonical_sha256(read_jsonl(proposal_path)) != canonical_sha256(proposals):
        mismatches.append("proposal_recompute")
    public_hashes: dict[str, str] = {}
    allowed = {"schema_version", "item_id", "question", "response", "units", "structure"}
    for annotator in ("a", "b"):
        for shard_index, expected in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            path = _package_path(root, annotator, shard_index)
            actual = read_jsonl(path)
            if canonical_sha256(actual) != canonical_sha256(expected):
                mismatches.append(f"package:{key}")
            if any(set(row) != allowed for row in actual):
                mismatches.append(f"public_field_leak:{key}")
            if any(row.get("schema_version") != PACKAGE_SCHEMA for row in actual):
                mismatches.append(f"public_schema:{key}")
            public_hashes[key] = file_sha256(path)
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    if canonical_sha256(read_jsonl(private_path)) != canonical_sha256(private):
        mismatches.append("private_index_recompute")
    report_path = root / "packages/package_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "status": "PASS_PRIOR_V17_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "parent_bindings": parent_bindings,
        "prompt_bindings": _prompt_bindings(),
        "code_bindings": _code_bindings(),
        "selection": selection,
        "construction": construction,
        "annotation_burden": burden,
        "code_dirty": False,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            mismatches.append(f"package_report:{key}")
    for key, actual in public_hashes.items():
        if report.get("public_shards", {}).get(key, {}).get("file_sha256") != actual:
            mismatches.append(f"package_report:public_hash:{key}")
    verification = {
        "schema_version": "clir-prior-mechanical-key-binary-verification-v17",
        "status": (
            "PASS_PRIOR_V17_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V17_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(report_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "private_index_file_sha256": file_sha256(private_path),
        "public_shards": len(public_hashes),
        "public_rows_total": sum(len(shard) for side in packages.values() for shard in side),
        "labels_present_at_freeze_verification": False,
        "next_gate": "user_runs_two_independent_max_reasoning_annotators",
    }
    path = root / "packages/independent_verification.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != verification:
        raise ValueError("Prior v17 package verification drift")
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
    hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        directory = root / f"labels_{annotator}"
        expected_paths = {
            _label_path(root, annotator, index).resolve() for index in range(shard_count)
        }
        actual_paths = {path.resolve() for path in directory.glob("*.jsonl")}
        if actual_paths != expected_paths:
            raise ValueError(
                f"Prior v17 labels_{annotator} population mismatch: "
                f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                f"extra={sorted(map(str, actual_paths - expected_paths))}"
            )
        for shard_index in range(shard_count):
            key = f"{annotator}-{shard_index:02d}"
            package_rows = read_jsonl(_package_path(root, annotator, shard_index))
            label_path = _label_path(root, annotator, shard_index)
            label_rows = read_jsonl(label_path)
            if len(package_rows) != rows_per_shard or len(label_rows) != rows_per_shard:
                raise ValueError(f"Prior v17 shard {key} must contain {rows_per_shard} rows")
            package_ids = [str(row["item_id"]) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(package_ids):
                raise ValueError(f"Prior v17 label/package ID mismatch: {key}")
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
            hashes[key] = file_sha256(label_path)
    proposals = read_jsonl(root / "proposals/prior_binary_natural_96.jsonl")
    private = read_jsonl(root / "packages/PRIVATE_package_index.jsonl")
    return proposals, packages, private, labels, hashes


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    verification_path = root / "packages/independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS_PRIOR_V17_PACKAGE_INDEPENDENT_RECOMPUTE":
        raise ValueError("Prior v17 package verification is not PASS")
    package_report_path = root / "packages/package_report.json"
    if verification.get("package_report_file_sha256") != file_sha256(package_report_path):
        raise ValueError("Prior v17 package report binding drift")
    proposals, packages, private, labels, label_hashes = _load_annotation_inputs(
        protocol, root
    )
    report = evaluate_binary_smoke_v17(
        proposals=proposals,
        packages=packages,
        private_index=private,
        labels=labels,
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
    path = root / "evaluation/raw_gate_report.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != report:
        raise ValueError("Prior v17 evaluation report already exists with different content")
    if not path.exists():
        atomic_write_json(path, report)
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
