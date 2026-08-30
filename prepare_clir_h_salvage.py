#!/usr/bin/env python
"""Publish and verify the post-hoc exploratory ranking-v7 H0 salvage subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from prepare_clir_ranking import (
    _evaluate_h_stage_labels,
    _read_published_jsonl,
    _verify_h_proposals,
)
from src.clir_h_salvage import (
    SALVAGE_LABEL_SCHEMA,
    build_h_salvage_rows,
    find_retry_self_repeat_failures,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AMENDMENT = (
    PROJECT_ROOT / "configs/ranking_expansion_v7/salvage_amendment_v7_4.json"
)
DEFAULT_PRE_ANNOTATION_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_annotation"
)
OUTPUT_SCHEMA_SUFFIXES = {
    "eligible": "eligible-pool",
    "all": "selected-all",
    "train": "selected-train",
    "dev": "selected-dev",
}


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


def _paths(pre_annotation_root: Path) -> dict[str, Path]:
    return {
        "protocol": PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json",
        "proposal_report": pre_annotation_root / "proposals/proposal_report.json",
        "all_proposals": pre_annotation_root / "proposals/h_proposals_all.jsonl",
        "reserve_private_index": pre_annotation_root
        / "packages/reserve/PRIVATE_package_index.jsonl",
        "smoke_attempt_a": pre_annotation_root
        / "annotation/smoke_a_gpt56sol.jsonl",
        "smoke_attempt_b": pre_annotation_root
        / "annotation/smoke_b_claude_opus5.jsonl",
        "reserve_attempt_1_b": pre_annotation_root
        / "annotation/reserve_b_claude_opus5.jsonl",
        "reserve_attempt_2_a": pre_annotation_root
        / "reannotation_v7_3/merged/reserve_a_gpt56sol_retry_v7_3.jsonl",
        "reserve_attempt_2_b": pre_annotation_root
        / "reannotation_v7_3/merged/reserve_b_claude_opus5_retry_v7_3.jsonl",
        "reserve_attempt_2_merge_report": pre_annotation_root
        / "reannotation_v7_3/merged/merge_report.json",
        "terminal_v7_3_report": pre_annotation_root
        / "final_v7_3/finalization_report.json",
    }


def _load_amendment(
    amendment_path: Path, pre_annotation_root: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version")
        != "clir-h0-v7.4-posthoc-salvage-amendment"
        or amendment.get("status")
        != "AUTHORIZED_POSTHOC_EXPLORATORY_SALVAGE"
        or not amendment.get("scientific_status", {}).get("salvage_is_posthoc")
        or amendment.get("scientific_status", {}).get("original_v7_status_remains")
        != "FAIL_H0_V7_RESERVE"
    ):
        raise ValueError("unsupported or unauthorized H0 v7.4 salvage amendment")
    paths = _paths(pre_annotation_root)
    parent = amendment["parent"]
    expected_keys = {
        "protocol": "protocol_file_sha256",
        "proposal_report": "proposal_report_file_sha256",
        "all_proposals": "all_proposals_file_sha256",
        "reserve_private_index": "reserve_private_index_file_sha256",
        "smoke_attempt_a": "smoke_attempt_a_file_sha256",
        "smoke_attempt_b": "smoke_attempt_b_file_sha256",
        "reserve_attempt_1_b": "reserve_attempt_1_b_file_sha256",
        "reserve_attempt_2_a": "reserve_attempt_2_a_file_sha256",
        "reserve_attempt_2_b": "reserve_attempt_2_b_file_sha256",
        "reserve_attempt_2_merge_report": (
            "reserve_attempt_2_merge_report_file_sha256"
        ),
        "terminal_v7_3_report": "terminal_v7_3_report_file_sha256",
    }
    for name, parent_key in expected_keys.items():
        if file_sha256(paths[name]) != parent[parent_key]:
            raise ValueError(f"H0 v7.4 parent artifact drift: {name}")
    terminal = json.loads(paths["terminal_v7_3_report"].read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "FAIL_H0_V7_RESERVE"
        or terminal.get("selection", {}).get("status")
        != "NOT_RUN_RESERVE_QUALITY_GATE_FAILED"
        or terminal.get("feature_extraction_allowed") is not False
        or terminal.get("training_allowed") is not False
        or terminal.get("gates", {})
        .get("smoke_recalculated", {})
        .get("pass")
        is not True
        or terminal.get("gates", {}).get("reserve_quality", {}).get("pass")
        is not False
    ):
        raise ValueError("H0 v7.4 terminal parent is not the frozen v7.3 failure")
    merge = json.loads(
        paths["reserve_attempt_2_merge_report"].read_text(encoding="utf-8")
    )
    if merge.get("status") != "PASS_H0_V7_3_REANNOTATION_LABEL_MERGE":
        raise ValueError("H0 v7.4 retry merge parent is not a PASS")
    return amendment, paths


def _ranking_args(pre_annotation_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        protocol=str(PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json"),
        authorization=str(
            PROJECT_ROOT / "configs/ranking_expansion_v7/rollout_authorization.json"
        ),
        pre_annotation_authorization=str(
            PROJECT_ROOT
            / "configs/ranking_expansion_v7/pre_annotation_authorization.json"
        ),
        pre_rollout_dir=str(
            PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_rollout"
        ),
        rollout_root=str(PROJECT_ROOT / "run_artifacts/ranking_expansion_v7"),
        pre_annotation_root=str(pre_annotation_root),
    )


def _build(
    *, amendment_path: Path, pre_annotation_root: Path
) -> tuple[dict[str, Any], dict[str, Path], list[dict], list[dict], dict[str, Any]]:
    amendment, paths = _load_amendment(amendment_path, pre_annotation_root)
    _, output_root, proposals, _ = _verify_h_proposals(
        _ranking_args(pre_annotation_root)
    )
    if output_root != pre_annotation_root:
        raise ValueError("H0 v7.4 verified pre-annotation root drift")

    smoke, _, _ = _evaluate_h_stage_labels(
        output_root=pre_annotation_root,
        stage="smoke",
        labels_a=paths["smoke_attempt_a"],
        labels_b=paths["smoke_attempt_b"],
    )
    reserve_attempt_1, _, _ = _evaluate_h_stage_labels(
        output_root=pre_annotation_root,
        stage="reserve",
        labels_a=paths["reserve_attempt_2_a"],
        labels_b=paths["reserve_attempt_1_b"],
    )
    reserve_attempt_2, _, _ = _evaluate_h_stage_labels(
        output_root=pre_annotation_root,
        stage="reserve",
        labels_a=paths["reserve_attempt_2_a"],
        labels_b=paths["reserve_attempt_2_b"],
    )
    private_rows = _read_published_jsonl(
        paths["reserve_private_index"],
        expected_schema="clir-h0-v7-private-package-index",
    )[0]
    retry_raw = {
        "a": {row["item_id"]: row for row in read_jsonl(paths["reserve_attempt_2_a"])},
        "b": {row["item_id"]: row for row in read_jsonl(paths["reserve_attempt_2_b"])},
    }
    repeat_failed, repeat_report = find_retry_self_repeat_failures(
        private_rows=private_rows,
        retry_labels_by_annotator=retry_raw,
    )
    targets = amendment["selection"]["targets"]
    eligible, selected, selection_report = build_h_salvage_rows(
        proposals=proposals["all"],
        smoke_labels_by_annotator=smoke,
        reserve_attempt_1_b=reserve_attempt_1["b"],
        reserve_attempt_2_by_annotator=reserve_attempt_2,
        repeat_failed_proposal_ids=repeat_failed,
        targets=targets,
        label_name=amendment["scientific_status"]["label_name"],
    )
    selection_report["retry_self_repeat"] = repeat_report
    return amendment, paths, eligible, selected, selection_report


def _output_paths(pre_annotation_root: Path) -> dict[str, Path]:
    root = pre_annotation_root / "salvage_v7_4"
    return {
        "eligible": root / "h_salvage_eligible.jsonl",
        "all": root / "h_salvage_selected_all.jsonl",
        "train": root / "h_salvage_selected_train.jsonl",
        "dev": root / "h_salvage_selected_dev.jsonl",
        "report": root / "selection_report.json",
        "verification": root / "verification_report.json",
    }


def _manifest_record(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    return {
        "path": str(path),
        "row_count": manifest["row_count"],
        "file_sha256": manifest["file_sha256"],
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "sidecar_file_sha256": file_sha256(sidecar),
    }


def command_publish(args: argparse.Namespace) -> None:
    amendment_path = Path(args.amendment).resolve()
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    amendment, parent_paths, eligible, selected, selection_report = _build(
        amendment_path=amendment_path,
        pre_annotation_root=pre_annotation_root,
    )
    code_commit = _require_clean("H0 v7.4 post-hoc salvage publication")
    if selection_report["status"] != "PASS_H0_V7_4_POSTHOC_SALVAGE_SELECTION":
        raise RuntimeError("H0 v7.4 salvage selection did not meet frozen targets")
    paths = _output_paths(pre_annotation_root)
    if any(
        path.exists()
        or (
            path.suffix == ".jsonl"
            and path.with_suffix(path.suffix + ".manifest.json").exists()
        )
        for path in paths.values()
    ):
        raise FileExistsError("H0 v7.4 salvage artifacts already exist")
    partitions = {
        "eligible": eligible,
        "all": selected,
        "train": [row for row in selected if row["h_label_split"] == "train"],
        "dev": [row for row in selected if row["h_label_split"] == "dev"],
    }
    if {name: len(rows) for name, rows in partitions.items()} != {
        "eligible": int(
            amendment["selection"]["expected_eligible_rows_from_frozen_posthoc_audit"]
        ),
        "all": 600,
        "train": 400,
        "dev": 200,
    }:
        raise AssertionError("H0 v7.4 salvage partition counts drifted")
    manifests = {
        name: publish_manifest(
            paths[name],
            rows,
            schema_version=f"{SALVAGE_LABEL_SCHEMA}-{OUTPUT_SCHEMA_SUFFIXES[name]}",
            metadata={
                "amendment_file_sha256": file_sha256(amendment_path),
                "code_commit": code_commit,
                "original_v7_status": "FAIL_H0_V7_RESERVE",
                "posthoc_exploratory": True,
                "label_name": amendment["scientific_status"]["label_name"],
            },
        )
        for name, rows in partitions.items()
    }
    report = {
        "schema_version": "clir-h0-v7.4-posthoc-salvage-selection-report",
        "status": selection_report["status"],
        "code_commit": code_commit,
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "posthoc_exploratory": True,
        "confirmatory_or_formal_claims_allowed": False,
        "amendment_file_sha256": file_sha256(amendment_path),
        "parent_files": {
            name: file_sha256(path) for name, path in parent_paths.items()
        },
        "selection": selection_report,
        "files": {
            name: _manifest_record(paths[name], manifests[name])
            for name in ("eligible", "all", "train", "dev")
        },
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "next_gate": amendment["execution"]["next_gate"],
    }
    atomic_write_json(paths["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    amendment_path = Path(args.amendment).resolve()
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    code_commit = _require_clean("H0 v7.4 post-hoc salvage verification")
    _, _, eligible, selected, selection_report = _build(
        amendment_path=amendment_path,
        pre_annotation_root=pre_annotation_root,
    )
    paths = _output_paths(pre_annotation_root)
    published = {
        name: _read_published_jsonl(
            paths[name],
            expected_schema=(
                f"{SALVAGE_LABEL_SCHEMA}-{OUTPUT_SCHEMA_SUFFIXES[name]}"
            ),
        )[0]
        for name in ("eligible", "all", "train", "dev")
    }
    expected = {
        "eligible": eligible,
        "all": selected,
        "train": [row for row in selected if row["h_label_split"] == "train"],
        "dev": [row for row in selected if row["h_label_split"] == "dev"],
    }
    for name in expected:
        if published[name] != expected[name]:
            raise ValueError(f"H0 v7.4 published {name} rows differ from recomputation")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != selection_report["status"]
        or report.get("original_v7_status") != "FAIL_H0_V7_RESERVE"
        or report.get("posthoc_exploratory") is not True
        or report.get("feature_extraction_allowed") is not False
        or report.get("training_allowed") is not False
    ):
        raise ValueError("H0 v7.4 selection report status or scope drift")
    for name, rows in published.items():
        record = report["files"][name]
        if (
            int(record["row_count"]) != len(rows)
            or record["file_sha256"] != file_sha256(paths[name])
            or record["ordered_rows_sha256"] != canonical_sha256(rows)
            or record["sidecar_file_sha256"]
            != file_sha256(paths[name].with_suffix(paths[name].suffix + ".manifest.json"))
        ):
            raise ValueError(f"H0 v7.4 report binding drift: {name}")
    verification = {
        "schema_version": "clir-h0-v7.4-posthoc-salvage-verification-report",
        "status": "PASS_H0_V7_4_POSTHOC_SALVAGE_RECOMPUTATION",
        "code_commit": code_commit,
        "amendment_file_sha256": file_sha256(amendment_path),
        "selection_report_file_sha256": file_sha256(paths["report"]),
        "recomputed_eligible_rows": len(eligible),
        "recomputed_selected_rows": len(selected),
        "selected_ids_sha256": selection_report["selected_ids_sha256"],
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "posthoc_exploratory": True,
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "next_gate": "freeze_selected_only_feature_inventory_and_extraction_authorization",
    }
    if paths["verification"].exists():
        existing = json.loads(paths["verification"].read_text(encoding="utf-8"))
        if existing != verification:
            raise ValueError("H0 v7.4 verification report drift")
    else:
        atomic_write_json(paths["verification"], verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    parser.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="publish the frozen salvage subset")
    publish.set_defaults(func=command_publish)
    verify = commands.add_parser("verify", help="recompute and verify the salvage subset")
    verify.set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
