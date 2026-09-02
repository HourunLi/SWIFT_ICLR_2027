#!/usr/bin/env python
"""Replay future Prior edge proposals on the non-trainable v13 max bridge.

The bridge labels are development evidence only.  This command never rewrites
the frozen v13 packages, labels, evaluator report, or terminal decision, and it
never publishes a training manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.clir_prior_edge_candidates_v14 import propose_dependency_edges_v14
from src.clir_prior_mechanical import validate_local_audit_annotation
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_prior_v13/pre_annotation"
)
DEFAULT_LABELS_A = DEFAULT_ROOT / "labels_a_max_bridge"
DEFAULT_LABELS_B = DEFAULT_ROOT / "labels_b_max_bridge"
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "evaluation_max_bridge/edge_candidate_v14_dev_report.json"
)
NATURAL_ROWS = 48
SHARD_COUNT = 4
ROWS_PER_SHARD = 18


def _package_path(root: Path, side: str, shard: int) -> Path:
    return root / f"packages/annotator_{side}/prior_v13_{side}_{shard:02d}.jsonl"


def _label_path(directory: Path, side: str, shard: int) -> Path:
    return directory / f"prior_v13_{side}_{shard:02d}.jsonl"


def _load_side(
    *,
    root: Path,
    label_directory: Path,
    side: str,
    private_kind: dict[tuple[str, str], str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    expected_labels = {
        _label_path(label_directory, side, shard).resolve()
        for shard in range(SHARD_COUNT)
    }
    actual_labels = {path.resolve() for path in label_directory.glob("*.jsonl")}
    if actual_labels != expected_labels:
        raise ValueError(
            f"bridge labels_{side} population mismatch: "
            f"missing={sorted(map(str, expected_labels - actual_labels))}, "
            f"extra={sorted(map(str, actual_labels - expected_labels))}"
        )

    packages: dict[str, dict[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for shard in range(SHARD_COUNT):
        package_path = _package_path(root, side, shard)
        label_path = _label_path(label_directory, side, shard)
        package_rows = read_jsonl(package_path)
        label_rows = read_jsonl(label_path)
        if len(package_rows) != ROWS_PER_SHARD or len(label_rows) != ROWS_PER_SHARD:
            raise ValueError(f"bridge shard {side}-{shard:02d} must have 18 rows")
        package_map = {str(row["item_id"]): row for row in package_rows}
        label_ids = [str(row.get("item_id")) for row in label_rows]
        if (
            len(package_map) != ROWS_PER_SHARD
            or len(label_ids) != len(set(label_ids))
            or set(package_map) != set(label_ids)
        ):
            raise ValueError(f"bridge shard {side}-{shard:02d} ID mismatch")
        for label in label_rows:
            item_id = str(label["item_id"])
            package = package_map[item_id]
            packages[item_id] = package
            normalized[item_id] = validate_local_audit_annotation(label, package)
        hashes[f"package_{side}_{shard:02d}"] = file_sha256(package_path)
        hashes[f"label_{side}_{shard:02d}"] = file_sha256(label_path)

    natural_ids = {
        item_id
        for item_id in packages
        if private_kind.get((side, item_id)) == "natural"
    }
    if len(natural_ids) != NATURAL_ROWS:
        raise ValueError(f"bridge side {side} must contain 48 natural rows")
    return (
        {item_id: packages[item_id] for item_id in sorted(natural_ids)},
        {item_id: normalized[item_id] for item_id in sorted(natural_ids)},
        hashes,
    )


def _side_metrics(
    packages: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    missing_total = 0
    covered_total = 0
    rows_with_missing = 0
    residual_rows = 0
    for item_id, package in packages.items():
        proposed = {
            (int(edge["parent_block_id"]), int(edge["child_block_id"]))
            for edge in propose_dependency_edges_v14(package)
        }
        missing = {tuple(map(int, edge)) for edge in labels[item_id]["missing_edges"]}
        missing_total += len(missing)
        covered_total += len(missing & proposed)
        rows_with_missing += bool(missing)
        residual_rows += bool(missing - proposed)
    return {
        "natural_rows": len(packages),
        "observed_missing_edges": missing_total,
        "covered_missing_edges": covered_total,
        "observed_missing_edge_recall": (
            covered_total / missing_total if missing_total else 1.0
        ),
        "rows_with_observed_missing_edges": rows_with_missing,
        "residual_missing_edge_rows": residual_rows,
        "residual_missing_edge_row_rate": residual_rows / max(1, len(packages)),
    }


def evaluate_bridge(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    private = read_jsonl(private_path)
    private_kind = {
        (str(row["annotator"]), str(row["item_id"])): str(row["kind"])
        for row in private
    }
    packages: dict[str, dict[str, dict[str, Any]]] = {}
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    bindings = {"private_index": file_sha256(private_path)}
    for side, directory in (
        ("a", Path(args.labels_a).resolve()),
        ("b", Path(args.labels_b).resolve()),
    ):
        packages[side], labels[side], hashes = _load_side(
            root=root,
            label_directory=directory,
            side=side,
            private_kind=private_kind,
        )
        bindings.update(hashes)

    if set(packages["a"]) != set(packages["b"]):
        raise ValueError("bridge A/B natural populations differ")
    for item_id in packages["a"]:
        if (
            packages["a"][item_id]["structure"]["source_sha256"]
            != packages["b"][item_id]["structure"]["source_sha256"]
        ):
            raise ValueError(f"bridge A/B source mismatch: {item_id}")

    side_metrics = {
        side: _side_metrics(packages[side], labels[side]) for side in ("a", "b")
    }
    shared_total = shared_covered = union_total = union_covered = 0
    old_counts: list[int] = []
    new_counts: list[int] = []
    per_child = Counter()
    for item_id, package in packages["a"].items():
        proposed_rows = propose_dependency_edges_v14(package)
        proposed = {
            (int(edge["parent_block_id"]), int(edge["child_block_id"]))
            for edge in proposed_rows
        }
        missing_a = {
            tuple(map(int, edge)) for edge in labels["a"][item_id]["missing_edges"]
        }
        missing_b = {
            tuple(map(int, edge)) for edge in labels["b"][item_id]["missing_edges"]
        }
        shared = missing_a & missing_b
        union = missing_a | missing_b
        shared_total += len(shared)
        shared_covered += len(shared & proposed)
        union_total += len(union)
        union_covered += len(union & proposed)
        old_counts.append(len(package["structure"]["candidate_edges"]))
        new_counts.append(len(proposed_rows))
        per_child.update(
            Counter(int(edge["child_block_id"]) for edge in proposed_rows).values()
        )

    shared_recall = shared_covered / shared_total if shared_total else 1.0
    union_recall = union_covered / union_total if union_total else 1.0
    inflation = sum(new_counts) / max(1, sum(old_counts))
    max_per_child = max(per_child, default=0)
    checks = {
        "shared_missing_edge_recall": shared_recall >= 0.95,
        "residual_missing_rows_a": side_metrics["a"][
            "residual_missing_edge_row_rate"
        ]
        <= 0.20,
        "residual_missing_rows_b": side_metrics["b"][
            "residual_missing_edge_row_rate"
        ]
        <= 0.20,
        "candidate_inflation": inflation <= 1.50,
        "candidate_parent_cap": max_per_child <= 6,
    }
    passed = all(checks.values())
    return {
        "schema_version": "clir-prior-edge-candidate-v14-dev-replay-v1",
        "status": (
            "PASS_POSTHOC_EDGE_CANDIDATE_V14_DEV_REPLAY"
            if passed
            else "STOP_POSTHOC_EDGE_CANDIDATE_V14_DEV_REPLAY"
        ),
        "evidence_tier": "posthoc_bridge_development_only",
        "annotators": {
            "a": str(args.annotator_a_model),
            "b": str(args.annotator_b_model),
            "exact_model_revisions_verified": False,
        },
        "side_metrics": side_metrics,
        "cross_annotator": {
            "shared_observed_missing_edges": shared_total,
            "covered_shared_observed_missing_edges": shared_covered,
            "shared_observed_missing_edge_recall": shared_recall,
            "union_observed_missing_edges": union_total,
            "covered_union_observed_missing_edges": union_covered,
            "union_observed_missing_edge_recall_diagnostic": union_recall,
        },
        "candidate_burden": {
            "old_mean_edges_per_row": mean(old_counts),
            "new_mean_edges_per_row": mean(new_counts),
            "old_total_edges": sum(old_counts),
            "new_total_edges": sum(new_counts),
            "edge_count_inflation_ratio": inflation,
            "new_min_edges_per_row": min(new_counts),
            "new_max_edges_per_row": max(new_counts),
            "max_parents_for_any_child": max_per_child,
            "parents_per_child_histogram": {
                str(key): value for key, value in sorted(per_child.items())
            },
        },
        "development_checks": checks,
        "bindings": dict(sorted(bindings.items())),
        "old_v13_terminal_decision_unchanged": True,
        "training_labels_published": False,
        "fresh_annotation_authorized_by_this_report": False,
        "next_step": (
            "freeze_a_separate_fresh_protocol_before_new_annotation"
            if passed
            else "revise_mechanical_proposal_hypothesis_before_fresh_annotation"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels-a", type=Path, default=DEFAULT_LABELS_A)
    parser.add_argument("--labels-b", type=Path, default=DEFAULT_LABELS_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotator-a-model", default="user_reported_gpt_5_6_sol_max"
    )
    parser.add_argument(
        "--annotator-b-model",
        default="user_reported_upgraded_claude_opus_max_exact_revision_unknown",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_bridge(args)
    output = Path(args.output).resolve()
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old != report:
            raise ValueError("edge-candidate development report drift")
    else:
        atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not str(report["status"]).startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
