#!/usr/bin/env python
"""Prepare, verify, and evaluate the verification-first CLIR Prior smoke v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from src.clir_prior_partial import select_partial_prior_smoke_rows
from src.clir_prior_verified import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    build_verified_blind_packages,
    evaluate_verified_prior_labels,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v11/protocol.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_prior_v11/pre_annotation"
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean(stage: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"{stage} requires a clean Git worktree")
    return _git_head()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-prior-verified-smoke-v11":
        raise ValueError("unsupported Prior v11 protocol schema")
    if protocol.get("status") != "FROZEN_USER_AUTHORIZED_ANNOTATION":
        raise ValueError("Prior v11 protocol is not frozen and user-authorized")
    if not protocol.get("execution_authorization", {}).get("annotation_allowed"):
        raise ValueError("Prior v11 annotation authorization is missing")
    return protocol


def _verify_parent_artifacts(protocol: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, record in protocol["parent_artifacts"].items():
        path = _resolve(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"missing Prior v11 parent artifact: {path}")
        if file_sha256(path) != record["file_sha256"]:
            raise ValueError(f"Prior v11 parent artifact drift: {name}")
        paths[name] = path
    return paths


def _sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {str(row["query_id"]) for row in rows},
        {str(row["cluster_id"]) for row in rows},
    )


def _build(
    protocol_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol = _load_protocol(protocol_path)
    paths = _verify_parent_artifacts(protocol)
    materialized = read_jsonl(paths["v6_materialized_rows"])

    exclusion_names = [
        "v6_consistency_selected_inventory",
        "v7_h_selected_rows",
        "v7_ranking_evaluation_rows",
        "v8_prior_smoke_proposals",
        "v9_prior_smoke_proposals",
        "v10_prior_smoke_proposals",
    ]
    exclusion_sets: dict[str, tuple[set[str], set[str]]] = {}
    excluded_queries: set[str] = set()
    excluded_clusters: set[str] = set()
    for name in exclusion_names:
        query_ids, cluster_ids = _sets(read_jsonl(paths[name]))
        exclusion_sets[name] = (query_ids, cluster_ids)
        excluded_queries.update(query_ids)
        excluded_clusters.update(cluster_ids)

    proposals, selection_report = select_partial_prior_smoke_rows(
        materialized_rows=materialized,
        excluded_query_ids=excluded_queries,
        excluded_cluster_ids=excluded_clusters,
        selection=protocol["selection"],
        namespace="clir-prior-v11",
        proposal_schema=PROPOSAL_SCHEMA,
        version_label="Prior v11",
    )
    selected_queries, selected_clusters = _sets(proposals)
    overlap_report = {}
    for name, (query_ids, cluster_ids) in exclusion_sets.items():
        overlap_report[f"selected_vs_{name}_query_overlap"] = len(
            selected_queries & query_ids
        )
        overlap_report[f"selected_vs_{name}_cluster_overlap"] = len(
            selected_clusters & cluster_ids
        )
    if any(overlap_report.values()):
        raise ValueError(f"Prior v11 overlap audit failed: {overlap_report}")

    package_a, package_b, private_index, package_report = (
        build_verified_blind_packages(
            proposals,
            repeat_count_a=int(protocol["packages"]["self_repeats_per_annotator"]),
            repeat_count_b=int(protocol["packages"]["self_repeats_per_annotator"]),
        )
    )
    report = {
        "status": "PASS_PRIOR_VERIFIED_SMOKE_V11_PACKAGES_READY",
        "schema_version": "clir-prior-verified-smoke-pre-annotation-v11",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_file_sha256": file_sha256(protocol_path),
        "selection": selection_report,
        "packages": package_report,
        "overlap_audit": overlap_report,
        "annotation_allowed": True,
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "claim_boundary": "pre-annotation package readiness only; no label, learnability, gate, or ranking evidence",
    }
    return protocol, proposals, package_a, package_b, private_index, report


def _output_paths(root: Path) -> dict[str, Path]:
    filename = "prior_verified_smoke.jsonl"
    return {
        "proposals": root / "proposals/prior_verified_smoke_natural.jsonl",
        "package_a": root / f"packages/annotator_a/{filename}",
        "package_b": root / f"packages/annotator_b/{filename}",
        "private": root / "packages/PRIVATE_package_index.jsonl",
        "report": root / "pre_annotation_report.json",
        "verification": root / "independent_verification.json",
        "labels_a": root / f"labels_a/{filename}",
        "labels_b": root / f"labels_b/{filename}",
        "gate": root / "evaluation/raw_gate_report.json",
    }


def command_prepare(args: argparse.Namespace) -> None:
    code_commit = _require_clean("prepare-smoke")
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    paths = _output_paths(output_root)
    protected = {"labels_a", "labels_b", "gate"}
    if (
        any(path.exists() for key, path in paths.items() if key not in protected)
        and not args.overwrite
    ):
        raise FileExistsError("Prior v11 smoke output already exists; use verify-smoke")
    protocol, proposals, package_a, package_b, private_index, report = _build(
        protocol_path
    )
    metadata = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": code_commit,
        "label_tier_if_future_gate_passes": protocol["claim_boundary"][
            "planned_label_name"
        ],
    }
    manifests = {
        "proposals": publish_manifest(
            paths["proposals"], proposals, schema_version=PROPOSAL_SCHEMA, metadata=metadata
        ),
        "package_a": publish_manifest(
            paths["package_a"], package_a, schema_version=PACKAGE_SCHEMA, metadata=metadata
        ),
        "package_b": publish_manifest(
            paths["package_b"], package_b, schema_version=PACKAGE_SCHEMA, metadata=metadata
        ),
        "private": publish_manifest(
            paths["private"], private_index, schema_version=PRIVATE_SCHEMA, metadata=metadata
        ),
    }
    report["code_commit"] = code_commit
    report["files"] = {
        key: {
            "path": manifest["path"],
            "row_count": manifest["row_count"],
            "file_sha256": manifest["file_sha256"],
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        }
        for key, manifest in manifests.items()
    }
    atomic_write_json(paths["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    paths = _output_paths(output_root)
    _, proposals, package_a, package_b, private_index, recomputed = _build(
        protocol_path
    )
    expected = {
        "proposals": proposals,
        "package_a": package_a,
        "package_b": package_b,
        "private": private_index,
    }
    mismatches = []
    for key, rows in expected.items():
        actual = read_jsonl(paths[key])
        if canonical_sha256(actual) != canonical_sha256(rows):
            mismatches.append(key)
    published = json.loads(paths["report"].read_text(encoding="utf-8"))
    if published.get("status") != recomputed["status"]:
        mismatches.append("report_status")
    verification = {
        "schema_version": "clir-prior-verified-smoke-independent-verification-v11",
        "status": (
            "PASS_PRIOR_VERIFIED_SMOKE_V11_RECOMPUTATION"
            if not mismatches
            else "FAIL_PRIOR_VERIFIED_SMOKE_V11_RECOMPUTATION"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "published_report_file_sha256": file_sha256(paths["report"]),
    }
    atomic_write_json(paths["verification"], verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    paths = _output_paths(output_root)
    protocol = _load_protocol(protocol_path)
    if not paths["verification"].is_file():
        raise FileNotFoundError("run verify-smoke before evaluating labels")
    verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
    if verification.get("status") != "PASS_PRIOR_VERIFIED_SMOKE_V11_RECOMPUTATION":
        raise ValueError("Prior v11 package verification is not PASS")
    labels_a = Path(args.labels_a).resolve()
    labels_b = Path(args.labels_b).resolve()
    report = evaluate_verified_prior_labels(
        package_a=read_jsonl(paths["package_a"]),
        package_b=read_jsonl(paths["package_b"]),
        private_index=read_jsonl(paths["private"]),
        labels_a=read_jsonl(labels_a),
        labels_b=read_jsonl(labels_b),
        gates=protocol["raw_gates"],
    )
    report["protocol_file_sha256"] = file_sha256(protocol_path)
    report["package_verification_file_sha256"] = file_sha256(paths["verification"])
    report["labels_a_file_sha256"] = file_sha256(labels_a)
    report["labels_b_file_sha256"] = file_sha256(labels_b)
    report["evaluated_at_code_commit"] = _git_head()
    atomic_write_json(paths["gate"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-smoke")
    prepare.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("verify-smoke")
    evaluate = subparsers.add_parser("evaluate-labels")
    evaluate.add_argument(
        "--labels-a", default=str(_output_paths(DEFAULT_OUTPUT_ROOT)["labels_a"])
    )
    evaluate.add_argument(
        "--labels-b", default=str(_output_paths(DEFAULT_OUTPUT_ROOT)["labels_b"])
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-smoke":
        command_prepare(args)
    elif args.command == "verify-smoke":
        command_verify(args)
    elif args.command == "evaluate-labels":
        command_evaluate(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
