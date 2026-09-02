#!/usr/bin/env python
"""Read-only v14 projection used to justify the distinct Prior-v15 target."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.clir_prior_mechanical import validate_local_audit_annotation
from src.clir_prior_role_v15 import public_package_item_v15
from src.clir_smoke import atomic_write_json, file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v14/pre_annotation"
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "evaluation/prior_role_v15_posthoc_dev_report.json"
)


def _set_f1(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _set_iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _package_path(root: Path, annotator: str, shard: int) -> Path:
    return root / f"packages/annotator_{annotator}/prior_v14_{annotator}_{shard:02d}.jsonl"


def _label_path(root: Path, annotator: str, shard: int) -> Path:
    return root / f"labels_{annotator}/prior_v14_{annotator}_{shard:02d}.jsonl"


def _project(label: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    if label["eligibility"] != "usable":
        return {
            "eligibility": label["eligibility"],
            "path_status": None,
            "roles": (),
            "final_block_id": None,
            "key_units": (),
            "complete_units": (),
        }
    roles = tuple((row["block_id"], row["role"]) for row in label["block_roles"])
    complete_blocks = [block_id for block_id, role in roles if role == "main_step"]
    structure = public_package_item_v15(item)["structure"]
    by_id = {int(row["block_id"]): row for row in structure["blocks"]}
    complete_units = sorted(
        unit
        for block_id in complete_blocks
        for unit in map(int, by_id[block_id]["unit_indices"])
    )
    final_block = int(label["final_block_id"])
    key_units = tuple(map(int, by_id[final_block]["unit_indices"]))
    return {
        "eligibility": "usable",
        "path_status": label["path_status"],
        "roles": roles,
        "final_block_id": final_block,
        "key_units": key_units,
        "complete_units": tuple(complete_units),
    }


def build_report(root: Path) -> dict[str, Any]:
    private_path = root / "packages/PRIVATE_package_index.jsonl"
    terminal_path = root / "evaluation/raw_gate_report.json"
    private = read_jsonl(private_path)
    metadata = {(str(row["annotator"]), str(row["item_id"])): row for row in private}
    labels: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    items: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    repeats: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {
        "a": [],
        "b": [],
    }
    label_hashes: dict[str, str] = {}
    for annotator in ("a", "b"):
        for shard in range(4):
            package_path = _package_path(root, annotator, shard)
            label_path = _label_path(root, annotator, shard)
            package_rows = read_jsonl(package_path)
            label_rows = read_jsonl(label_path)
            package_by_id = {str(row["item_id"]): row for row in package_rows}
            label_hashes[f"{annotator}-{shard:02d}"] = file_sha256(label_path)
            for raw in label_rows:
                item_id = str(raw["item_id"])
                normalized = validate_local_audit_annotation(
                    raw, package_by_id[item_id]
                )
                info = metadata[(annotator, item_id)]
                if info["kind"] == "natural":
                    natural_id = str(info["natural_item_id"])
                    labels[annotator][natural_id] = normalized
                    items[annotator][natural_id] = package_by_id[item_id]
                elif info["kind"] == "repeat":
                    repeats[annotator].append(
                        (
                            str(info["natural_item_id"]),
                            normalized,
                            package_by_id[item_id],
                        )
                    )

    natural_ids = sorted(labels["a"])
    if len(natural_ids) != 48 or set(natural_ids) != set(labels["b"]):
        raise ValueError("v14 natural populations are incomplete or differ")

    key_exact = 0
    key_f1: list[float] = []
    complete_f1: list[float] = []
    complete_iou: list[float] = []
    coverage: list[float] = []
    all_material = 0
    role_agree = role_total = 0
    old_key_disagreement = 0
    old_key_disagreement_paths: Counter[str] = Counter()
    for item_id in natural_ids:
        left_old, right_old = labels["a"][item_id], labels["b"][item_id]
        left = _project(left_old, items["a"][item_id])
        right = _project(right_old, items["b"][item_id])
        if left_old["key_unit_indices"] != right_old["key_unit_indices"]:
            old_key_disagreement += 1
            old_key_disagreement_paths[
                f"{left_old['path_status']}|{right_old['path_status']}"
            ] += 1
        key_exact += left["key_units"] == right["key_units"]
        key_f1.append(_set_f1(left["key_units"], right["key_units"]))
        complete_f1.append(
            _set_f1(left["complete_units"], right["complete_units"])
        )
        complete_iou.append(
            _set_iou(left["complete_units"], right["complete_units"])
        )
        size = int(items["a"][item_id]["structure"]["material_unit_count"])
        coverage.append(
            1.0
            - len(set(left["complete_units"]) ^ set(right["complete_units"]))
            / max(1, size)
        )
        all_material += (
            len(set(left["complete_units"]) | set(right["complete_units"])) == size
        )
        left_roles = dict(left["roles"])
        right_roles = dict(right["roles"])
        role_total += len(left_roles)
        role_agree += sum(left_roles[key] == right_roles[key] for key in left_roles)

    repeat_report: dict[str, Any] = {}
    for annotator in ("a", "b"):
        exact = 0
        for parent_id, repeated_label, repeated_item in repeats[annotator]:
            parent = _project(labels[annotator][parent_id], items[annotator][parent_id])
            repeated = _project(repeated_label, repeated_item)
            exact += parent == repeated
        repeat_report[annotator] = {
            "exact": exact,
            "total": len(repeats[annotator]),
            "rate": exact / max(1, len(repeats[annotator])),
        }

    metrics = {
        "natural_rows": len(natural_ids),
        "structural_key_exact_rate": key_exact / len(natural_ids),
        "structural_key_macro_f1": mean(key_f1),
        "role_derived_complete_macro_f1": mean(complete_f1),
        "role_derived_complete_macro_iou": mean(complete_iou),
        "role_derived_complete_coverage": mean(coverage),
        "role_decision_agreement": role_agree / role_total,
        "all_material_union_rate": all_material / len(natural_ids),
        "projected_repeat_target_exact": repeat_report,
        "old_key_disagreements": old_key_disagreement,
        "old_key_disagreement_path_pairs": dict(sorted(old_key_disagreement_paths.items())),
    }
    checks = {
        "structural_key": metrics["structural_key_macro_f1"] >= 0.90,
        "complete_f1": metrics["role_derived_complete_macro_f1"] >= 0.90,
        "complete_iou": metrics["role_derived_complete_macro_iou"] >= 0.80,
        "coverage": metrics["role_derived_complete_coverage"] >= 0.90,
        "roles": metrics["role_decision_agreement"] >= 0.85,
        "all_material": metrics["all_material_union_rate"] <= 0.25,
        "repeat_a": repeat_report["a"]["rate"] >= 0.9375,
        "repeat_b": repeat_report["b"]["rate"] >= 0.9375,
        "all_old_key_disagreements_are_flawed_flawed": (
            old_key_disagreement_paths == Counter({"flawed|flawed": 11})
        ),
    }
    return {
        "schema_version": "clir-prior-role-only-v15-posthoc-dev-replay",
        "status": (
            "PASS_POSTHOC_PRIOR_ROLE_V15_DEV_REPLAY"
            if all(checks.values())
            else "FAIL_POSTHOC_PRIOR_ROLE_V15_DEV_REPLAY"
        ),
        "v14_terminal_status": json.loads(terminal_path.read_text())["status"],
        "v14_terminal_report_file_sha256": file_sha256(terminal_path),
        "v14_label_file_sha256": dict(sorted(label_hashes.items())),
        "metrics": metrics,
        "checks": checks,
        "interpretation": (
            "v14 role labels already reproduce stable structural Key/Complete; "
            "all old Key disagreements came from asking Prior to localize flaws"
        ),
        "claim_boundary": (
            "posthoc target-design evidence only; it does not repair v14, test "
            "the v15 prompt or fresh controls, publish labels, extract features, "
            "authorize training, or establish ranking efficacy"
        ),
        "next_gate": (
            "freeze a fresh query/cluster-disjoint role-only v15 smoke with new "
            "controls before any annotation"
        ),
        "training_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.root.resolve())
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not str(report["status"]).startswith("PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
