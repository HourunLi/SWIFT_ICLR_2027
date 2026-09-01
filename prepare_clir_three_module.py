#!/usr/bin/env python
"""Freeze and materialize the expanded three-module CLIR factorial data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.clir_data import read_jsonl
from src.clir_smoke import (
    atomic_write_json,
    file_sha256,
    publish_manifest,
)
from src.clir_three_module import build_unified_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/three_module_expansion_v1/protocol.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/three_module_expansion_v1"


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


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("three-module materialization requires a clean commit")
    parent = "34c5a68"
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, commit],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode:
        raise ValueError("runtime commit does not descend from the Gate result")
    return {"commit": commit, "dirty": False, "minimum_parent_commit": parent}


def verify_factorial_configs(protocol: Mapping[str, Any]) -> dict[str, Any]:
    cells = protocol["factorial_grid"]["cells"]
    expected_cells = {"u0", "c", "h", "p", "ch", "cp", "hp", "full"}
    if set(cells) != expected_cells:
        raise ValueError("three-module protocol must contain the complete 2x2x2 grid")
    payloads: dict[str, dict[str, Any]] = {}
    observed: dict[str, Any] = {}
    for cell, specification in cells.items():
        path = _project_path(specification["config"])
        _assert_hash(path, specification["file_sha256"], f"{cell} config")
        payload = json.loads(path.read_text(encoding="utf-8"))
        factors = tuple(int(value) for value in specification["factors"])
        if len(factors) != 3 or any(value not in (0, 1) for value in factors):
            raise ValueError(f"{cell}: invalid C/H/P factor tuple")
        model = payload["model"]
        observed_factors = (
            int(float(model["consistency_weight"]) == 1.0),
            int(float(model["hallucination_weight"]) == 1.0),
            int(float(model["prior_weight"]) == 1.0),
        )
        if observed_factors != factors:
            raise ValueError(f"{cell}: config does not match its C/H/P tuple")
        expected_gate = 0.25 if factors[2] else 0.0
        if float(model["gate_prior_weight"]) != expected_gate:
            raise ValueError(f"{cell}: P factor must carry fixed Gate {expected_gate}")
        for disabled in (
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "progress_weight",
            "prior_distill_weight",
            "reconstruction_weight",
        ):
            if float(model[disabled]) != 0.0:
                raise ValueError(f"{cell}: unexpectedly enables {disabled}")
        payloads[cell] = payload
        observed[cell] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256(path),
            "factors": list(factors),
        }

    normalized: list[dict[str, Any]] = []
    for cell in sorted(cells):
        payload = json.loads(json.dumps(payloads[cell]))
        for field in (
            "consistency_weight",
            "hallucination_weight",
            "prior_weight",
            "gate_prior_weight",
        ):
            payload["model"].pop(field)
        normalized.append(payload)
    if any(payload != normalized[0] for payload in normalized[1:]):
        raise ValueError("factorial configs differ outside C/H/P factor weights")
    return observed


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "clir-three-module-expansion-v1"
        or protocol.get("status")
        != "AUTHORIZED_MATERIALIZATION_ONLY_PENDING_EXACT_MANIFEST_FREEZE"
        or protocol.get("evidence_tier")
        != "posthoc_exploratory_silver_no_human_verification"
    ):
        raise ValueError("unsupported or claim-drifting three-module protocol")
    for name, specification in protocol["parent_results"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
    for name, specification in protocol["frozen_inputs"].items():
        source = _project_path(specification["path"])
        _assert_hash(source, specification["file_sha256"], name)
        if len(read_jsonl(source)) != int(specification["row_count"]):
            raise ValueError(f"{name} row-count drift")
    verify_factorial_configs(protocol)
    return protocol


def _publish(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = publish_manifest(path, rows, schema_version=schema, metadata=metadata)
    manifest["sidecar_file_sha256"] = file_sha256(
        path.with_suffix(path.suffix + ".manifest.json")
    )
    return manifest


def _query_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["query_id"]) for row in rows}


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    report_path = output_root / "materialization_report.json"
    if report_path.exists():
        raise FileExistsError(f"materialization report already exists: {report_path}")
    protocol = load_protocol(protocol_path)
    git = _git_state()
    frozen = protocol["frozen_inputs"]
    h_train_path = _project_path(frozen["consistency_h0_train"]["path"])
    prior_train_path = _project_path(frozen["prior_train"]["path"])
    h_dev_path = _project_path(frozen["h_dev"]["path"])
    prior_dev_path = _project_path(frozen["prior_dev"]["path"])
    data_root = output_root / "data"
    built = build_unified_data(
        consistency_h0_train=read_jsonl(h_train_path),
        prior_train=read_jsonl(prior_train_path),
        h_dev=read_jsonl(h_dev_path),
        prior_dev=read_jsonl(prior_dev_path),
        consistency_h0_parent=h_train_path.parent,
        prior_parent=prior_train_path.parent,
        h_dev_parent=h_dev_path.parent,
        prior_dev_parent=prior_dev_path.parent,
        target_parent=data_root,
    )
    train_queries = _query_ids(built["train"])
    endpoint_path = _project_path(frozen["consistency_heldout_endpoints"]["path"])
    ranking_path = _project_path(frozen["ranking"]["path"])
    consistency_overlap = train_queries & _query_ids(read_jsonl(endpoint_path))
    ranking_overlap = train_queries & _query_ids(read_jsonl(ranking_path))
    if consistency_overlap or ranking_overlap:
        raise ValueError("unified train overlaps frozen held-out Consistency/ranking")

    manifests = {
        "train": _publish(
            data_root / "train_factorial.jsonl",
            built["train"],
            "clir-three-module-expansion-v1-train-manifest",
            {
                "shared_by_cells": list(protocol["factorial_grid"]["cells"]),
                "posthoc_exploratory": True,
            },
        ),
        "h_dev": _publish(
            data_root / "h_dev_query_disjoint.jsonl",
            built["h_dev"],
            "clir-three-module-expansion-v1-h-dev-manifest",
            {"evaluation_only": True, "cross_module_query_disjoint": True},
        ),
        "prior_dev": _publish(
            data_root / "prior_dev_query_disjoint.jsonl",
            built["prior_dev"],
            "clir-three-module-expansion-v1-prior-dev-manifest",
            {"evaluation_only": True, "cross_module_query_disjoint": True},
        ),
    }
    report = {
        "schema_version": "clir-three-module-expansion-v1-materialization",
        "status": "PASS_THREE_MODULE_UNIFIED_DATA_MATERIALIZATION",
        "completed_at_utc": _utc_now(),
        "code_commit": git["commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "evidence_tier": protocol["evidence_tier"],
        "terminal_statuses_preserved": protocol["terminal_statuses_preserved"],
        "inventory": built["report"],
        "evaluation_query_overlap": {
            "consistency_heldout": 0,
            "ranking": 0,
            "clean_h_dev": 0,
            "clean_prior_dev": 0,
        },
        "manifests": manifests,
        "training_allowed": False,
        "next_gate": "SEPARATE_HASH_BOUND_FACTORIAL_TRAINING_AUTHORIZATION",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root).resolve()
    report_path = output_root / "materialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_THREE_MODULE_UNIFIED_DATA_MATERIALIZATION":
        raise ValueError("three-module materialization did not pass")
    if report.get("protocol_file_sha256") != file_sha256(args.protocol):
        raise ValueError("materialization protocol hash drift")
    expected_rows = {
        "train": int(protocol["merge_contract"]["expected_train_rows"]),
        "h_dev": int(
            protocol["evaluation_split_contract"]["expected_clean_h_dev_rows"]
        ),
        "prior_dev": int(
            protocol["evaluation_split_contract"]["expected_clean_prior_dev_rows"]
        ),
    }
    observed: dict[str, Any] = {}
    for name, expected in expected_rows.items():
        manifest = report["manifests"][name]
        path = Path(manifest["path"])
        _assert_hash(path, manifest["file_sha256"], f"published {name}")
        if len(read_jsonl(path)) != expected:
            raise ValueError(f"published {name} row-count drift")
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        _assert_hash(sidecar, manifest["sidecar_file_sha256"], f"{name} sidecar")
        observed[name] = {
            "path": str(path),
            "row_count": expected,
            "file_sha256": manifest["file_sha256"],
        }
    verification = {
        "schema_version": "clir-three-module-expansion-v1-verification",
        "status": "PASS_THREE_MODULE_UNIFIED_DATA_INDEPENDENT_RECOMPUTE",
        "verified_at_utc": _utc_now(),
        "protocol_file_sha256": file_sha256(args.protocol),
        "materialization_report_sha256": file_sha256(report_path),
        "manifests": observed,
        "training_allowed": False,
        "next_gate": "SEPARATE_HASH_BOUND_FACTORIAL_TRAINING_AUTHORIZATION",
    }
    verification_path = output_root / "materialization_verification.json"
    if verification_path.exists():
        raise FileExistsError(f"verification already exists: {verification_path}")
    atomic_write_json(verification_path, verification)
    print(json.dumps(verification, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    subparsers.add_parser("verify")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "materialize":
        command_materialize(args)
    elif args.command == "verify":
        command_verify(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
