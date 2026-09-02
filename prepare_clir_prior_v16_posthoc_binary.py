#!/usr/bin/env python
"""Freeze, verify, evaluate, and materialize the isolated Prior-v16 post-hoc replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.clir_prior_v16_posthoc_binary import (
    PACKAGE_SCHEMA,
    PRIVATE_SCHEMA,
    PROPOSAL_SCHEMA,
    PROTOCOL_SCHEMA,
    ROW_SCHEMA,
    build_posthoc_shards,
    construct_posthoc_silver_rows,
    evaluate_posthoc_replay,
    select_v16_posthoc_rows,
    validate_posthoc_silver_rows,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/data_expansion_prior_v16/posthoc_binary_v1/protocol.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "run_artifacts/data_expansion_prior_v16_posthoc_binary_v1/pre_annotation"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_binary_v1/annotation_prompt.md"
)
DEFAULT_LAUNCH_A = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_binary_v1/launch_prompt_a.txt"
)
DEFAULT_LAUNCH_B = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_binary_v1/launch_prompt_b.txt"
)
SOURCE_FILE = PROJECT_ROOT / "src/clir_prior_v16_posthoc_binary.py"


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
        raise ValueError("unsupported Prior-v16 post-hoc protocol schema")
    if protocol.get("status") != "FROZEN_POSTHOC_BEFORE_ANY_REPLAY_LABEL":
        raise ValueError("Prior-v16 post-hoc protocol was not frozen before labels")
    return protocol


def _verify_parent_files(protocol: Mapping[str, Any]) -> dict[str, str]:
    parent = protocol["parent"]
    names = (
        "v12_materialized",
        "v16_protocol",
        "v16_proposals",
        "v16_terminal_report",
        "v17_protocol",
        "v17_source",
        "v17_terminal_report",
    )
    bindings: dict[str, str] = {}
    for name in names:
        path = _project_path(parent[f"{name}_path"])
        actual = file_sha256(path)
        if actual != str(parent[f"{name}_file_sha256"]):
            raise ValueError(f"Prior-v16 post-hoc parent hash drift: {name}")
        bindings[f"{name}_file_sha256"] = actual
    status_checks = (
        ("v16_terminal_report", "STOP_PRIOR_V16_ROLE_ONLY_SCALE"),
        ("v17_terminal_report", "STOP_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE"),
    )
    for name, expected in status_checks:
        report = json.loads(
            _project_path(parent[f"{name}_path"]).read_text(encoding="utf-8")
        )
        if report.get("status") != expected:
            raise ValueError(f"Prior-v16 post-hoc parent status drift: {name}")
    if not all(
        parent.get(field) is True
        for field in (
            "original_v16_and_v17_terminal_decisions_are_immutable",
            "replay_is_separately_named_posthoc_development",
            "future_ranking_confirmation_requires_fresh_query_clusters",
        )
    ):
        raise ValueError("Prior-v16 post-hoc evidence boundary drift")
    return bindings


def _recompute(protocol: Mapping[str, Any]):
    parent = protocol["parent"]
    original = read_jsonl(_project_path(parent["v16_proposals_path"]))
    materialized = read_jsonl(_project_path(parent["v12_materialized_path"]))
    population = protocol["replay_population"]
    proposals, selection = select_v16_posthoc_rows(
        original,
        materialized,
        namespace=str(population["selection_namespace"]),
    )
    expected_selection = {
        "input_v16_rows": int(population["original_v16_rows"]),
        "selected_rows": int(population["mechanically_compilable_rows"]),
        "selected_train_rows": int(population["train_rows_before_quality_exclusions"]),
        "selected_dev_rows": int(population["dev_rows_before_quality_exclusions"]),
        "selected_by_stratum": dict(population["selected_by_stratum"]),
        "residual_decisions_per_annotator": int(
            population["residual_decisions_per_annotator"]
        ),
        "ordered_item_ids_sha256": str(population["ordered_item_ids_sha256"]),
        "rejection_counts": dict(population["rejection_counts"]),
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"Prior-v16 post-hoc selection drift: {key}")
    annotation = protocol["annotation"]
    packages, private, construction = build_posthoc_shards(
        proposals,
        shard_count=int(annotation["shards_per_annotator"]),
        repeats_per_shard=int(annotation["self_repeats_per_shard"]),
        namespace=str(population["selection_namespace"]),
    )
    if construction["rows_per_shard"] != annotation["rows_per_shard"]:
        raise ValueError("Prior-v16 post-hoc shard-size drift")
    return original, materialized, proposals, selection, packages, private, construction


def _package_path(root: Path, side: str, shard: int) -> Path:
    return (
        root / f"packages/annotator_{side}/prior_v16_posthoc_{side}_{shard:02d}.jsonl"
    )


def _label_path(root: Path, side: str, shard: int) -> Path:
    return root / f"labels_{side}/prior_v16_posthoc_{side}_{shard:02d}.jsonl"


def _prompt_bindings() -> dict[str, str]:
    return {
        "annotation_prompt_file_sha256": file_sha256(DEFAULT_PROMPT),
        "launch_prompt_a_file_sha256": file_sha256(DEFAULT_LAUNCH_A),
        "launch_prompt_b_file_sha256": file_sha256(DEFAULT_LAUNCH_B),
    }


def _code_bindings() -> dict[str, str]:
    return {
        "posthoc_source_file_sha256": file_sha256(SOURCE_FILE),
        "prepare_source_file_sha256": file_sha256(Path(__file__).resolve()),
    }


def command_prepare(args: argparse.Namespace) -> None:
    if _git_dirty():
        raise RuntimeError("post-hoc package freeze requires a clean Git commit")
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    if root.exists() and any(root.rglob("*")):
        raise FileExistsError(f"post-hoc output is not empty: {root}")
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    _, _, proposals, selection, packages, private, construction = _recompute(protocol)
    common = {
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": _git_head(),
        "code_dirty": False,
        "posthoc": True,
        "labels_are_gold": False,
        "human_verification": False,
        "original_v16_and_v17_statuses_unchanged": True,
        **parent_bindings,
        **_prompt_bindings(),
        **_code_bindings(),
    }
    proposal_manifest = publish_manifest(
        root / "proposals/prior_v16_posthoc_natural_490.jsonl",
        proposals,
        schema_version=PROPOSAL_SCHEMA,
        metadata={**common, **selection},
    )
    public_manifests: dict[str, dict[str, Any]] = {}
    for side in ("a", "b"):
        for shard, rows in enumerate(packages[side]):
            key = f"{side}-{shard:02d}"
            public_manifests[key] = publish_manifest(
                _package_path(root, side, shard),
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
        "schema_version": "clir-prior-v16-posthoc-package-report-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_BLIND_PACKAGES_READY",
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
        "next_gate": "independent_package_recompute_before_dual_ai_replay",
    }
    atomic_write_json(root / "packages/package_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    parent_bindings = _verify_parent_files(protocol)
    _, _, proposals, selection, packages, private, construction = _recompute(protocol)
    mismatches: list[str] = []
    proposal_path = root / "proposals/prior_v16_posthoc_natural_490.jsonl"
    if canonical_sha256(read_jsonl(proposal_path)) != canonical_sha256(proposals):
        mismatches.append("proposal_recompute")
    public_hashes: dict[str, str] = {}
    allowed = {
        "schema_version",
        "item_id",
        "question",
        "response",
        "units",
        "structure",
    }
    for side in ("a", "b"):
        for shard, expected in enumerate(packages[side]):
            key = f"{side}-{shard:02d}"
            path = _package_path(root, side, shard)
            actual = read_jsonl(path)
            if canonical_sha256(actual) != canonical_sha256(expected):
                mismatches.append(f"package:{key}")
            if any(set(row) != allowed for row in actual):
                mismatches.append(f"public_field_leak:{key}")
            public_hashes[key] = file_sha256(path)
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    if canonical_sha256(read_jsonl(private_path)) != canonical_sha256(private):
        mismatches.append("private_index_recompute")
    report_path = root / "packages/package_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "status": "PASS_PRIOR_V16_POSTHOC_BLIND_PACKAGES_READY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "parent_bindings": parent_bindings,
        "prompt_bindings": _prompt_bindings(),
        "code_bindings": _code_bindings(),
        "selection": selection,
        "construction": construction,
        "code_dirty": False,
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            mismatches.append(f"package_report:{key}")
    for key, actual in public_hashes.items():
        if report.get("public_shards", {}).get(key, {}).get("file_sha256") != actual:
            mismatches.append(f"package_report:public_hash:{key}")
    verification = {
        "schema_version": "clir-prior-v16-posthoc-package-verification-v1",
        "status": (
            "PASS_PRIOR_V16_POSTHOC_PACKAGE_INDEPENDENT_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V16_POSTHOC_PACKAGE_INDEPENDENT_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "protocol_file_sha256": file_sha256(protocol_path),
        "package_report_file_sha256": file_sha256(report_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "private_index_file_sha256": file_sha256(private_path),
        "public_shards": len(public_hashes),
        "public_rows_total": sum(
            len(shard_rows)
            for side_rows in packages.values()
            for shard_rows in side_rows
        ),
        "labels_present_at_freeze_verification": False,
        "next_gate": "user_runs_two_independent_max_reasoning_annotators",
    }
    path = root / "packages/independent_verification.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != verification:
        raise ValueError("post-hoc package verification drift")
    if not path.exists():
        atomic_write_json(path, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


def _load_annotation_inputs(protocol: Mapping[str, Any], root: Path):
    shard_count = int(protocol["annotation"]["shards_per_annotator"])
    expected_sizes = list(map(int, protocol["annotation"]["rows_per_shard"]))
    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    hashes: dict[str, str] = {}
    for side in ("a", "b"):
        expected_paths = {
            _label_path(root, side, shard).resolve() for shard in range(shard_count)
        }
        directory = root / f"labels_{side}"
        actual_paths = {path.resolve() for path in directory.glob("*.jsonl")}
        if actual_paths != expected_paths:
            raise ValueError(
                f"post-hoc labels_{side} population mismatch: "
                f"missing={sorted(map(str, expected_paths - actual_paths))}, "
                f"extra={sorted(map(str, actual_paths - expected_paths))}"
            )
        for shard in range(shard_count):
            key = f"{side}-{shard:02d}"
            package_rows = read_jsonl(_package_path(root, side, shard))
            label_path = _label_path(root, side, shard)
            label_rows = read_jsonl(label_path)
            if (
                len(package_rows) != expected_sizes[shard]
                or len(label_rows) != expected_sizes[shard]
            ):
                raise ValueError(
                    f"post-hoc shard {key} must contain {expected_sizes[shard]} rows"
                )
            package_ids = [str(row["item_id"]) for row in package_rows]
            label_ids = [str(row.get("item_id")) for row in label_rows]
            if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(
                package_ids
            ):
                raise ValueError(f"post-hoc label/package ID mismatch: {key}")
            packages[side].extend(package_rows)
            labels[side].extend(label_rows)
            hashes[key] = file_sha256(label_path)
    proposals = read_jsonl(root / "proposals/prior_v16_posthoc_natural_490.jsonl")
    private = read_jsonl(root / "packages/PRIVATE_package_index.jsonl")
    return proposals, packages, private, labels, hashes


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    verification_path = root / "packages/independent_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("status")
        != "PASS_PRIOR_V16_POSTHOC_PACKAGE_INDEPENDENT_RECOMPUTE"
    ):
        raise ValueError("post-hoc package verification is not PASS")
    package_report_path = root / "packages/package_report.json"
    if verification.get("package_report_file_sha256") != file_sha256(
        package_report_path
    ):
        raise ValueError("post-hoc package report binding drift")
    proposals, packages, private, labels, label_hashes = _load_annotation_inputs(
        protocol, root
    )
    report = evaluate_posthoc_replay(
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
        raise ValueError("post-hoc evaluation already exists with different content")
    if not path.exists():
        atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not str(report["status"]).startswith("PASS"):
        raise SystemExit(1)


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    report_path = root / "evaluation/raw_gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_PRIOR_V16_POSTHOC_BINARY_REPLAY":
        raise ValueError("post-hoc replay did not pass; materialization is forbidden")
    proposals, packages, _, labels, label_hashes = _load_annotation_inputs(
        protocol, root
    )
    materialized_source = read_jsonl(
        _project_path(protocol["parent"]["v12_materialized_path"])
    )
    rows, materialization = construct_posthoc_silver_rows(
        proposals=proposals,
        materialized_rows=materialized_source,
        packages=packages,
        labels=labels,
        evaluation_report=report,
    )
    validation = validate_posthoc_silver_rows(
        rows,
        expected_item_ids_sha256=report["publishable_population"][
            "ordered_item_ids_sha256"
        ],
    )
    output_path = root / "materialized/prior_v16_posthoc_silver.jsonl"
    materialization_report_path = root / "materialized/materialization_report.json"
    if output_path.exists() or materialization_report_path.exists():
        raise FileExistsError("post-hoc materialized artifacts already exist")
    manifest = publish_manifest(
        output_path,
        rows,
        schema_version=ROW_SCHEMA,
        metadata={
            "evaluation_report_file_sha256": file_sha256(report_path),
            "protocol_file_sha256": file_sha256(protocol_path),
            "label_file_sha256": dict(sorted(label_hashes.items())),
            "posthoc": True,
            "human_verified": False,
        },
    )
    final = {
        **materialization,
        "validation": validation,
        "protocol_file_sha256": file_sha256(protocol_path),
        "evaluation_report_file_sha256": file_sha256(report_path),
        "silver_file_sha256": manifest["file_sha256"],
        "silver_ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "silver_sidecar_file_sha256": file_sha256(
            output_path.with_suffix(output_path.suffix + ".manifest.json")
        ),
        "label_file_sha256": dict(sorted(label_hashes.items())),
    }
    atomic_write_json(materialization_report_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def command_verify_materialized(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output).resolve()
    protocol = _load_protocol(protocol_path)
    report_path = root / "evaluation/raw_gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proposals, packages, _, labels, _ = _load_annotation_inputs(protocol, root)
    materialized_source = read_jsonl(
        _project_path(protocol["parent"]["v12_materialized_path"])
    )
    expected, expected_report = construct_posthoc_silver_rows(
        proposals=proposals,
        materialized_rows=materialized_source,
        packages=packages,
        labels=labels,
        evaluation_report=report,
    )
    output_path = root / "materialized/prior_v16_posthoc_silver.jsonl"
    actual = read_jsonl(output_path)
    validation = validate_posthoc_silver_rows(
        actual,
        expected_item_ids_sha256=report["publishable_population"][
            "ordered_item_ids_sha256"
        ],
    )
    mismatches = []
    if canonical_sha256(actual) != canonical_sha256(expected):
        mismatches.append("silver_recompute")
    frozen_report_path = root / "materialized/materialization_report.json"
    frozen_report = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    for key, value in expected_report.items():
        if frozen_report.get(key) != value:
            mismatches.append(f"materialization_report:{key}")
    if frozen_report.get("silver_file_sha256") != file_sha256(output_path):
        mismatches.append("materialization_report:silver_file_sha256")
    verification = {
        "schema_version": "clir-prior-v16-posthoc-materialization-verification-v1",
        "status": (
            "PASS_PRIOR_V16_POSTHOC_MATERIALIZATION_RECOMPUTE"
            if not mismatches
            else "FAIL_PRIOR_V16_POSTHOC_MATERIALIZATION_RECOMPUTE"
        ),
        "mismatches": mismatches,
        "validation": validation,
        "protocol_file_sha256": file_sha256(protocol_path),
        "evaluation_report_file_sha256": file_sha256(report_path),
        "materialization_report_file_sha256": file_sha256(frozen_report_path),
        "silver_file_sha256": file_sha256(output_path),
        "feature_extraction_allowed": not mismatches,
        "training_allowed": not mismatches,
        "claim_is_posthoc_silver_only": True,
    }
    path = root / "materialized/independent_verification.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != verification:
        raise ValueError("post-hoc materialization verification drift")
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
    {
        "prepare": command_prepare,
        "verify": command_verify,
        "evaluate": command_evaluate,
        "materialize": command_materialize,
        "verify-materialized": command_verify_materialized,
    }[args.command](args)


if __name__ == "__main__":
    main()
