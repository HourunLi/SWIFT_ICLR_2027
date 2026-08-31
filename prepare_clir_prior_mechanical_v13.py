#!/usr/bin/env python
"""Prepare, verify, and evaluate the fresh CLIR Prior mechanical-v13 smoke."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.clir_prior_mechanical import PACKAGE_SCHEMA
from src.clir_prior_mechanical_smoke import (
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    build_blind_shards,
    evaluate_blind_labels,
    select_fresh_natural_rows,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v13/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v13/pre_annotation"
DEFAULT_PROMPT = PROJECT_ROOT / "configs/data_expansion_prior_v13/annotation_prompt.md"
DEFAULT_LAUNCH_A = PROJECT_ROOT / "configs/data_expansion_prior_v13/launch_prompt_a.txt"
DEFAULT_LAUNCH_B = PROJECT_ROOT / "configs/data_expansion_prior_v13/launch_prompt_b.txt"


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
    if protocol.get("schema_version") != "clir-prior-mechanical-local-audit-smoke-v13":
        raise ValueError("unsupported Prior v13 protocol schema")
    return protocol


def _verify_parent_files(protocol: Mapping[str, Any]) -> dict[str, str]:
    parent = protocol["parent"]
    bindings: dict[str, str] = {}
    for name in (
        "v12_protocol",
        "v12_materialized",
        "v12_proposals",
        "v12_terminal_report",
    ):
        path = _project_path(parent[f"{name}_path"])
        expected = str(parent[f"{name}_file_sha256"])
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Prior v13 parent hash drift: {name}")
        bindings[f"{name}_file_sha256"] = actual
    terminal = json.loads(
        _project_path(parent["v12_terminal_report_path"]).read_text(encoding="utf-8")
    )
    if terminal.get("status") != parent["v12_terminal_status"]:
        raise ValueError("Prior v12 terminal status drift")
    return bindings


def _recompute(
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    parent = protocol["parent"]
    all_rows = read_jsonl(_project_path(parent["v12_materialized_path"]))
    v12_proposals = read_jsonl(_project_path(parent["v12_proposals_path"]))
    excluded_queries = {str(row["query_id"]) for row in v12_proposals}
    excluded_clusters = {str(row["cluster_id"]) for row in v12_proposals}
    proposals, selection = select_fresh_natural_rows(
        all_rows,
        excluded_query_ids=excluded_queries,
        excluded_cluster_ids=excluded_clusters,
        strata=protocol["fresh_selection"]["strata"],
        namespace=str(protocol["fresh_selection"]["selection_namespace"]),
    )
    packages, private, construction = build_blind_shards(
        proposals,
        shard_count=int(protocol["annotation"]["shards_per_annotator"]),
        repeats_per_shard=int(protocol["annotation"]["self_repeats_per_shard"]),
        namespace=str(protocol["fresh_selection"]["selection_namespace"]),
    )
    return proposals, selection, packages, private, construction


def _package_path(root: Path, annotator: str, shard_index: int) -> Path:
    return (
        root
        / f"packages/annotator_{annotator}/prior_v13_{annotator}_{shard_index:02d}.jsonl"
    )


def _label_path(root: Path, annotator: str, shard_index: int) -> Path:
    return root / f"labels_{annotator}/prior_v13_{annotator}_{shard_index:02d}.jsonl"


def command_prepare(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("Prior v13 package freeze requires a clean Git commit")
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    if root.exists() and any(root.rglob("*")):
        raise FileExistsError(f"Prior v13 output is not empty: {root}")
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction = _recompute(protocol)

    prompt_bindings = {
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_PROMPT),
        "launch_prompt_a_file_sha256": file_sha256(DEFAULT_LAUNCH_A),
        "launch_prompt_b_file_sha256": file_sha256(DEFAULT_LAUNCH_B),
    }
    common_metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "labels_are_gold": False,
        "human_verification": False,
        **parent_bindings,
        **prompt_bindings,
    }
    proposal_path = root / "proposals/prior_mechanical_natural_48.jsonl"
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
        "schema_version": "clir-prior-mechanical-package-report-v13",
        "status": "PASS_PRIOR_V13_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "parent_bindings": parent_bindings,
        "prompt_bindings": prompt_bindings,
        "proposal": {
            "path": proposal_manifest["path"],
            "row_count": proposal_manifest["row_count"],
            "file_sha256": proposal_manifest["file_sha256"],
            "ordered_rows_sha256": proposal_manifest["ordered_rows_sha256"],
        },
        "selection": selection,
        "construction": construction,
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
        "next_gate": "independent_package_recompute_before_dual_ai_annotation",
    }
    atomic_write_json(root / "packages/package_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    proposals, selection, packages, private, construction = _recompute(protocol)
    mismatches: list[str] = []

    proposal_path = root / "proposals/prior_mechanical_natural_48.jsonl"
    actual_proposals = read_jsonl(proposal_path)
    if canonical_sha256(actual_proposals) != canonical_sha256(proposals):
        mismatches.append("proposal_recompute")
    public_hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        for shard_index, expected in enumerate(packages[annotator]):
            key = f"{annotator}-{shard_index:02d}"
            path = _package_path(root, annotator, shard_index)
            actual = read_jsonl(path)
            if canonical_sha256(actual) != canonical_sha256(expected):
                mismatches.append(f"package:{key}")
            allowed = {
                "schema_version",
                "item_id",
                "question",
                "response",
                "units",
                "structure",
            }
            if any(set(row) != allowed for row in actual):
                mismatches.append(f"public_field_leak:{key}")
            public_hashes[key] = file_sha256(path)
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    if canonical_sha256(read_jsonl(private_path)) != canonical_sha256(private):
        mismatches.append("private_index_recompute")

    report_path = root / "packages/package_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "status": "PASS_PRIOR_V13_FRESH_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "parent_bindings": parent_bindings,
        "selection": selection,
        "construction": construction,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            mismatches.append(f"package_report:{key}")
    expected_prompt_bindings = {
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_PROMPT),
        "launch_prompt_a_file_sha256": file_sha256(DEFAULT_LAUNCH_A),
        "launch_prompt_b_file_sha256": file_sha256(DEFAULT_LAUNCH_B),
    }
    if report.get("prompt_bindings") != expected_prompt_bindings:
        mismatches.append("package_report:prompt_bindings")
    for key, actual_hash in public_hashes.items():
        if (
            report.get("public_shards", {}).get(key, {}).get("file_sha256")
            != actual_hash
        ):
            mismatches.append(f"package_report:public_hash:{key}")

    verification = {
        "schema_version": "clir-prior-mechanical-package-verification-v13",
        "status": (
            "PASS_PRIOR_V13_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V13_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(report_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "private_index_file_sha256": file_sha256(private_path),
        "public_shards": len(public_hashes),
        "public_rows_total": sum(
            len(rows) for side in packages.values() for rows in side
        ),
        "labels_present": any(
            path.is_file()
            for annotator in ("a", "b")
            for shard_index in range(4)
            for path in [_label_path(root, annotator, shard_index)]
        ),
        "next_gate": "user_runs_two_independent_blind_annotators",
    }
    verification_path = root / "packages/independent_verification.json"
    if verification_path.exists():
        old = json.loads(verification_path.read_text(encoding="utf-8"))
        if old != verification:
            raise ValueError("Prior v13 package verification drift")
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
    if verification.get("status") != "PASS_PRIOR_V13_PACKAGE_INDEPENDENT_RECOMPUTE":
        raise ValueError("Prior v13 package verification is not PASS")
    package_report_path = root / "packages/package_report.json"
    if verification.get("package_report_file_sha256") != file_sha256(
        package_report_path
    ):
        raise ValueError("Prior v13 package report binding drift")

    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    label_hashes: dict[str, str] = {}
    shard_count = int(protocol["annotation"]["shards_per_annotator"])
    rows_per_shard = int(protocol["annotation"]["rows_per_shard"])
    for annotator in ("a", "b"):
        directory = root / f"labels_{annotator}"
        expected_paths = {
            _label_path(root, annotator, index).resolve()
            for index in range(shard_count)
        }
        actual_paths = {path.resolve() for path in directory.glob("*.jsonl")}
        if actual_paths != expected_paths:
            raise ValueError(
                f"Prior v13 labels_{annotator} population mismatch: "
                f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                f"extra={sorted(map(str, actual_paths - expected_paths))}"
            )
        for shard_index in range(shard_count):
            key = f"{annotator}-{shard_index:02d}"
            package_rows = read_jsonl(_package_path(root, annotator, shard_index))
            label_path = _label_path(root, annotator, shard_index)
            label_rows = read_jsonl(label_path)
            if len(package_rows) != rows_per_shard or len(label_rows) != rows_per_shard:
                raise ValueError(
                    f"Prior v13 shard {key} must contain {rows_per_shard} rows"
                )
            package_ids = [str(row["item_id"]) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(
                package_ids
            ):
                raise ValueError(f"Prior v13 label/package ID mismatch: {key}")
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
            label_hashes[key] = file_sha256(label_path)

    private_path = root / "packages/PRIVATE_package_index.jsonl"
    report = evaluate_blind_labels(
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
        sorted(
            Counter(
                str(row.get("eligibility"))
                for annotator in ("a", "b")
                for row in labels[annotator]
            ).items()
        )
    )
    evaluation_path = root / "evaluation/raw_gate_report.json"
    if evaluation_path.exists():
        old = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if old != report:
            raise ValueError(
                "Prior v13 evaluation report already exists with different content"
            )
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
