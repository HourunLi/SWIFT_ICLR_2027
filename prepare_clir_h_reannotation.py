#!/usr/bin/env python
"""Freeze, verify, and merge the one-shot H0 v7.3 reserve reannotation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.clir_h_expansion import H_PACKAGE_SCHEMA
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
    validate_annotation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AMENDMENT = (
    PROJECT_ROOT
    / "configs/ranking_expansion_v7/reannotation_amendment_v7_3.json"
)
DEFAULT_PRE_ANNOTATION_ROOT = (
    PROJECT_ROOT / "run_artifacts/ranking_expansion_v7/pre_annotation"
)
SHARD_SCHEMA = "clir-h0-v7.3-reserve-reannotation-package-shard"
MERGED_LABEL_SCHEMA = "clir-h0-v7.3-reserve-reannotation-merged-labels"
LABEL_FIELDS = {
    "item_id",
    "status",
    "first_bad_unit_index",
    "confidence",
    "rationale",
}


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


def _require_clean(stage: str) -> str:
    if _git_dirty():
        raise RuntimeError(f"{stage} requires a clean Git worktree")
    return _git_head()


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _read_published(
    path: Path, *, expected_schema: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = _manifest_path(path)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"published JSONL or sidecar is missing: {path}")
    rows = read_jsonl(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != expected_schema
        or int(manifest.get("row_count", -1)) != len(rows)
        or manifest.get("file_sha256") != file_sha256(path)
        or manifest.get("ordered_rows_sha256") != canonical_sha256(rows)
    ):
        raise ValueError(f"published JSONL manifest drift: {path}")
    return rows, manifest


def _load_contract(
    amendment_path: Path, pre_annotation_root: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version")
        != "clir-h0-v7.3-full-reserve-reannotation-amendment"
        or amendment.get("status")
        != "AUTHORIZED_ONE_FULL_RESERVE_REANNOTATION_ATTEMPT"
    ):
        raise ValueError("unsupported or unauthorized H0 v7.3 amendment")
    paths = {
        "protocol": PROJECT_ROOT / "configs/ranking_expansion_v7/protocol.json",
        "package_a": pre_annotation_root
        / "packages/reserve/annotator_a/hallucination.jsonl",
        "package_b": pre_annotation_root
        / "packages/reserve/annotator_b/hallucination.jsonl",
        "attempt_1_a": pre_annotation_root
        / "annotation/reserve_a_gpt56sol.jsonl",
        "attempt_1_b": pre_annotation_root
        / "annotation/reserve_b_claude_opus5.jsonl",
        "failed_report": pre_annotation_root / "final/finalization_report.json",
    }
    parent = amendment["parent"]
    expected = {
        "protocol": parent["protocol_file_sha256"],
        "package_a": parent["reserve_package_a_file_sha256"],
        "package_b": parent["reserve_package_b_file_sha256"],
        "attempt_1_a": parent["reserve_attempt_1_a_file_sha256"],
        "attempt_1_b": parent["reserve_attempt_1_b_file_sha256"],
        "failed_report": parent["failed_finalization_report_file_sha256"],
    }
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"H0 v7.3 parent artifact drift: {name}")
    failed = json.loads(paths["failed_report"].read_text(encoding="utf-8"))
    if (
        failed.get("status") != "FAIL_H0_V7_RESERVE"
        or failed.get("code_commit") != parent["failed_finalization_code_commit"]
    ):
        raise ValueError("H0 v7.3 parent failure report is not the frozen failure")
    return amendment, paths


def _output_root(pre_annotation_root: Path) -> Path:
    return pre_annotation_root / "reannotation_v7_3"


def _registry_path(pre_annotation_root: Path) -> Path:
    return _output_root(pre_annotation_root) / "package_registry.json"


def command_freeze(args: argparse.Namespace) -> None:
    amendment_path = Path(args.amendment).resolve()
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    amendment, paths = _load_contract(amendment_path, pre_annotation_root)
    code_commit = _require_clean("H0 v7.3 reannotation package freeze")
    output_root = _output_root(pre_annotation_root)
    registry_path = _registry_path(pre_annotation_root)
    if registry_path.exists() or (output_root / "packages").exists():
        raise FileExistsError("H0 v7.3 reannotation packages already exist")
    shard_count = int(amendment["mechanical_resharding"]["shards_per_annotator"])
    rows_per_shard = int(amendment["mechanical_resharding"]["rows_per_shard"])
    registry_rows: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        rows, _ = _read_published(
            paths[f"package_{annotator}"], expected_schema=H_PACKAGE_SCHEMA
        )
        if len(rows) != shard_count * rows_per_shard:
            raise ValueError("H0 v7.3 package size differs from amendment")
        for index in range(shard_count):
            shard_id = f"shard-{index:03d}"
            shard_rows = rows[index * rows_per_shard : (index + 1) * rows_per_shard]
            path = output_root / f"packages/annotator_{annotator}/{shard_id}.jsonl"
            manifest = publish_manifest(
                path,
                shard_rows,
                schema_version=SHARD_SCHEMA,
                metadata={
                    "annotator": annotator,
                    "shard_id": shard_id,
                    "attempt": amendment["retry_scope"]["attempt_name"],
                    "source_package_file_sha256": file_sha256(
                        paths[f"package_{annotator}"]
                    ),
                    "amendment_file_sha256": file_sha256(amendment_path),
                    "code_commit": code_commit,
                },
            )
            registry_rows.append(
                {
                    "annotator": annotator,
                    "shard_id": shard_id,
                    "path": str(path),
                    "row_count": len(shard_rows),
                    "file_sha256": manifest["file_sha256"],
                    "ordered_rows_sha256": manifest["ordered_rows_sha256"],
                    "sidecar_file_sha256": file_sha256(_manifest_path(path)),
                    "ordered_item_ids_sha256": canonical_sha256(
                        [str(row["item_id"]) for row in shard_rows]
                    ),
                }
            )
    registry = {
        "schema_version": "clir-h0-v7.3-reannotation-package-registry",
        "status": "PASS_H0_V7_3_REANNOTATION_PACKAGE_FREEZE",
        "code_commit": code_commit,
        "amendment_file_sha256": file_sha256(amendment_path),
        "failed_finalization_report_file_sha256": file_sha256(
            paths["failed_report"]
        ),
        "shards_per_annotator": shard_count,
        "rows_per_shard": rows_per_shard,
        "rows_per_annotator": shard_count * rows_per_shard,
        "shards": registry_rows,
        "next_gate": "independent_package_verification_before_reannotation",
    }
    atomic_write_json(registry_path, registry)
    print(json.dumps(registry, ensure_ascii=False, indent=2))


def _verify_packages(
    *, amendment_path: Path, pre_annotation_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    amendment, paths = _load_contract(amendment_path, pre_annotation_root)
    registry_path = _registry_path(pre_annotation_root)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        registry.get("status") != "PASS_H0_V7_3_REANNOTATION_PACKAGE_FREEZE"
        or registry.get("amendment_file_sha256") != file_sha256(amendment_path)
    ):
        raise ValueError("H0 v7.3 reannotation registry is not a bound PASS")
    by_key = {
        (str(row["annotator"]), str(row["shard_id"])): row
        for row in registry["shards"]
    }
    expected_shards = int(
        amendment["mechanical_resharding"]["shards_per_annotator"]
    )
    expected_rows = int(amendment["mechanical_resharding"]["rows_per_shard"])
    for annotator in ("a", "b"):
        original, _ = _read_published(
            paths[f"package_{annotator}"], expected_schema=H_PACKAGE_SCHEMA
        )
        reconstructed: list[dict[str, Any]] = []
        for index in range(expected_shards):
            shard_id = f"shard-{index:03d}"
            record = by_key[(annotator, shard_id)]
            shard_path = Path(record["path"])
            shard_rows, manifest = _read_published(
                shard_path, expected_schema=SHARD_SCHEMA
            )
            if (
                len(shard_rows) != expected_rows
                or record["file_sha256"] != manifest["file_sha256"]
                or record["ordered_rows_sha256"]
                != manifest["ordered_rows_sha256"]
                or record["sidecar_file_sha256"]
                != file_sha256(_manifest_path(shard_path))
            ):
                raise ValueError(f"H0 v7.3 shard registry drift: {shard_path}")
            reconstructed.extend(shard_rows)
        if reconstructed != original:
            raise ValueError(
                f"H0 v7.3 annotator {annotator} shards do not reconstruct package"
            )
    report = {
        "schema_version": "clir-h0-v7.3-reannotation-package-verification",
        "status": "PASS_H0_V7_3_REANNOTATION_PACKAGE_VERIFICATION",
        "amendment_file_sha256": file_sha256(amendment_path),
        "registry_file_sha256": file_sha256(registry_path),
        "shards_verified": expected_shards * 2,
        "rows_verified": expected_shards * expected_rows * 2,
        "content_or_item_id_changes": 0,
        "next_gate": "two_new_independent_full_reannotation_sessions",
    }
    return registry, report


def command_verify_packages(args: argparse.Namespace) -> None:
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    _, report = _verify_packages(
        amendment_path=Path(args.amendment).resolve(),
        pre_annotation_root=pre_annotation_root,
    )
    path = _output_root(pre_annotation_root) / "package_verification.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError("H0 v7.3 package verification report drift")
    else:
        atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_label_shards(
    *,
    amendment: Mapping[str, Any],
    pre_annotation_root: Path,
    annotator: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_count = int(amendment["mechanical_resharding"]["shards_per_annotator"])
    rows_per_shard = int(amendment["mechanical_resharding"]["rows_per_shard"])
    merged: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    rationale_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for index in range(shard_count):
        shard_id = f"shard-{index:03d}"
        package_path = (
            _output_root(pre_annotation_root)
            / f"packages/annotator_{annotator}/{shard_id}.jsonl"
        )
        label_path = (
            _output_root(pre_annotation_root)
            / f"annotation/annotator_{annotator}/{shard_id}.labels.jsonl"
        )
        items, _ = _read_published(package_path, expected_schema=SHARD_SCHEMA)
        labels = read_jsonl(label_path)
        if len(labels) != rows_per_shard:
            raise ValueError(f"{label_path}: expected {rows_per_shard} labels")
        if any(set(label) != LABEL_FIELDS for label in labels):
            raise ValueError(f"{label_path}: label fields differ from strict schema")
        item_by_id = {str(item["item_id"]): item for item in items}
        label_ids = [str(label.get("item_id")) for label in labels]
        if len(set(label_ids)) != len(label_ids) or set(label_ids) != set(item_by_id):
            raise ValueError(f"{label_path}: label IDs differ from package shard")
        validated = [
            validate_annotation("hallucination", label, item_by_id[str(label["item_id"])])
            for label in labels
        ]
        validated_by_id = {str(label["item_id"]): label for label in validated}
        ordered = [validated_by_id[str(item["item_id"])] for item in items]
        for label in ordered:
            rationale_counts[str(label["rationale"])] += 1
            status_counts[str(label["status"])] += 1
        merged.extend(ordered)
        shard_reports.append(
            {
                "shard_id": shard_id,
                "package_file_sha256": file_sha256(package_path),
                "label_file_sha256": file_sha256(label_path),
                "rows": len(ordered),
                "ordered_item_ids_sha256": canonical_sha256(
                    [label["item_id"] for label in ordered]
                ),
            }
        )
    max_identical = max(rationale_counts.values(), default=0)
    max_allowed = int(
        amendment["execution_rules"][
            "maximum_identical_rationale_rows_per_annotator"
        ]
    )
    if max_identical > max_allowed:
        raise ValueError(
            f"annotator {annotator}: identical rationale appears {max_identical} "
            f"times, above frozen maximum {max_allowed}"
        )
    if len({str(label["item_id"]) for label in merged}) != len(merged):
        raise ValueError(f"annotator {annotator}: duplicate IDs across label shards")
    return merged, {
        "annotator": annotator,
        "rows": len(merged),
        "shards": shard_reports,
        "status_counts": dict(sorted(status_counts.items())),
        "maximum_identical_rationale_rows": max_identical,
        "maximum_identical_rationale_rows_allowed": max_allowed,
        "ordered_item_ids_sha256": canonical_sha256(
            [label["item_id"] for label in merged]
        ),
    }


def command_merge_labels(args: argparse.Namespace) -> None:
    amendment_path = Path(args.amendment).resolve()
    pre_annotation_root = Path(args.pre_annotation_root).resolve()
    amendment, _ = _load_contract(amendment_path, pre_annotation_root)
    _verify_packages(
        amendment_path=amendment_path,
        pre_annotation_root=pre_annotation_root,
    )
    code_commit = _require_clean("H0 v7.3 reannotation label merge")
    merged_root = _output_root(pre_annotation_root) / "merged"
    report_path = merged_root / "merge_report.json"
    outputs = {
        "a": merged_root / "reserve_a_gpt56sol_retry_v7_3.jsonl",
        "b": merged_root / "reserve_b_claude_opus5_retry_v7_3.jsonl",
    }
    if report_path.exists() or any(
        path.exists() or _manifest_path(path).exists() for path in outputs.values()
    ):
        raise FileExistsError("H0 v7.3 merged label artifacts already exist")
    merged: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, Any] = {}
    # Validate both annotators completely before publishing either merged file.
    # This keeps a missing or malformed later shard from leaving a one-sided
    # merged artifact that can be mistaken for a completed merge.
    for annotator in ("a", "b"):
        merged[annotator], validation[annotator] = _validate_label_shards(
            amendment=amendment,
            pre_annotation_root=pre_annotation_root,
            annotator=annotator,
        )

    manifests: dict[str, Any] = {}
    for annotator in ("a", "b"):
        manifests[annotator] = publish_manifest(
            outputs[annotator],
            merged[annotator],
            schema_version=MERGED_LABEL_SCHEMA,
            metadata={
                "annotator": annotator,
                "attempt": amendment["retry_scope"]["attempt_name"],
                "amendment_file_sha256": file_sha256(amendment_path),
                "code_commit": code_commit,
            },
        )
    report = {
        "schema_version": "clir-h0-v7.3-reannotation-label-merge-report",
        "status": "PASS_H0_V7_3_REANNOTATION_LABEL_MERGE",
        "code_commit": code_commit,
        "amendment_file_sha256": file_sha256(amendment_path),
        "validation": validation,
        "files": {
            annotator: {
                "path": str(outputs[annotator]),
                "file_sha256": manifests[annotator]["file_sha256"],
                "ordered_rows_sha256": manifests[annotator][
                    "ordered_rows_sha256"
                ],
                "row_count": manifests[annotator]["row_count"],
                "sidecar_file_sha256": file_sha256(
                    _manifest_path(outputs[annotator])
                ),
            }
            for annotator in ("a", "b")
        },
        "next_gate": "attempt_2_unchanged_annotation_quality_and_final_yield_gates",
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    parser.add_argument(
        "--pre-annotation-root", default=str(DEFAULT_PRE_ANNOTATION_ROOT)
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze 16x50 retry package shards")
    freeze.set_defaults(func=command_freeze)
    verify = commands.add_parser(
        "verify-packages", help="verify shards reconstruct the original packages"
    )
    verify.set_defaults(func=command_verify_packages)
    merge = commands.add_parser(
        "merge-labels", help="validate and merge all retry label shards"
    )
    merge.set_defaults(func=command_merge_labels)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
