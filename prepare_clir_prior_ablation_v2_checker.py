#!/usr/bin/env python
"""Apply the frozen numeric checker and select the 2,400-query v2 ranking set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from prepare_clir_prior_ablation_v2 import load_protocol
from src.clir_gate_tuning import (
    materialize_numeric_checker_rows,
    validate_numeric_checker_rows,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2"


def _select(
    checked: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    new_query_priorities: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_count = int(protocol["generation"]["candidate_count"])
    binary = {"numeric_match", "numeric_mismatch"}
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in checked:
        by_query[str(raw["query_id"])].append(dict(raw))
    if set(by_query) != set(new_query_priorities):
        raise ValueError("new rollout queries do not match the frozen priority map")
    eligible: dict[str, list[dict[str, Any]]] = {}
    failures: Counter[str] = Counter()
    for query_id, rows in by_query.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        if len(rows) != candidate_count or [
            int(row["candidate_index"]) for row in rows
        ] != list(range(candidate_count)):
            failures["candidate_axis"] += 1
        elif any(row.get("checker_status") not in binary for row in rows):
            failures["nonbinary_checker_status"] += 1
        elif any(int(row.get("correctness", -1)) not in {0, 1} for row in rows):
            failures["nonbinary_correctness"] += 1
        else:
            eligible[query_id] = rows
    quotas = protocol["ranking_population"]["source_query_counts"]
    selected_query_rows: list[tuple[str, list[dict[str, Any]]]] = []
    source_report: dict[str, Any] = {}
    for source in ("gsm8k", "asdiv-a"):
        candidates = [
            (query_id, rows)
            for query_id, rows in eligible.items()
            if rows[0]["source"] == source
        ]
        candidates.sort(
            key=lambda item: str(new_query_priorities[item[0]])
        )
        target = int(quotas[source])
        if len(candidates) < target:
            raise ValueError(f"FAIL_YIELD: {source} has {len(candidates)} < {target}")
        selected_query_rows.extend(candidates[:target])
        source_report[source] = {
            "raw_queries": sum(
                1 for rows in by_query.values() if rows[0]["source"] == source
            ),
            "eligible_queries": len(candidates),
            "selected_queries": target,
        }

    reserve_population = validate_rollout_population(
        reserve, candidate_count=candidate_count
    )
    reserve_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in reserve:
        reserve_by_query[str(raw["query_id"])].append(dict(raw))
    reserve_queries = sorted(
        reserve_by_query.items(),
        key=lambda item: str(item[1][0]["prior_ablation_selection_priority"]),
    )
    if len(reserve_queries) != int(quotas["math"]):
        raise ValueError("MATH reserve query count drift")
    selected_query_rows.extend(reserve_queries)
    source_report["math"] = {
        "raw_queries": int(reserve_population["queries"]),
        "eligible_queries": int(reserve_population["queries"]),
        "selected_queries": len(reserve_queries),
        "new_rollout": False,
    }

    query_ids = [query_id for query_id, _ in selected_query_rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("selected ranking queries overlap")
    output: list[dict[str, Any]] = []
    for query_order, (query_id, rows) in enumerate(selected_query_rows):
        rows.sort(key=lambda row: int(row["candidate_index"]))
        for raw in rows:
            row = dict(raw)
            row.update(
                {
                    "role": "prior_ablation_v2_ranking",
                    "evaluation_split": "prior_ablation_v2",
                    "evaluation_only": True,
                    "sealed_until_weight_lock": False,
                    "prior_ablation_query_order": query_order,
                }
            )
            if query_id in new_query_priorities:
                row["prior_ablation_final_priority"] = new_query_priorities[query_id]
            output.append(row)
    population = validate_rollout_population(output, candidate_count=candidate_count)
    if (
        len(output) != int(protocol["ranking_population"]["selected_candidate_rows"])
        or int(population["queries"])
        != int(protocol["ranking_population"]["total_queries"])
    ):
        raise ValueError("final ranking population arithmetic drift")
    return output, {
        "new_raw_queries": len(by_query),
        "new_eligible_queries": len(eligible),
        "new_ineligible_queries": len(by_query) - len(eligible),
        "new_ineligible_reasons": dict(sorted(failures.items())),
        "by_source": source_report,
        "selected_query_ids_sha256": canonical_sha256(query_ids),
        "selection_used_clir_scores": False,
    }


def _load_inputs(root: Path, protocol: Mapping[str, Any]):
    rollout_completion_path = root / "rollout_completion_report.json"
    rollout_completion = json.loads(rollout_completion_path.read_text(encoding="utf-8"))
    if rollout_completion.get("status") != "PASS_PRIOR_ABLATION_V2_NEW_RAW_ROLLOUTS":
        raise ValueError("new raw rollout is not complete")
    raw_path = Path(rollout_completion["combined"]["path"])
    if file_sha256(raw_path) != rollout_completion["combined"]["file_sha256"]:
        raise ValueError("new raw rollout hash drift")
    raw = read_jsonl(raw_path)
    reserve_path = root / "pre_rollout/math_reserve_checked.jsonl"
    reserve = read_jsonl(reserve_path)
    freeze = json.loads((root / "pre_rollout/freeze_report.json").read_text(encoding="utf-8"))
    if file_sha256(reserve_path) != freeze["records"]["math_reserve"]["file_sha256"]:
        raise ValueError("MATH reserve hash drift")
    fresh_queries_path = root / "pre_rollout/fresh_queries.jsonl"
    if (
        file_sha256(fresh_queries_path)
        != freeze["records"]["fresh_queries"]["file_sha256"]
    ):
        raise ValueError("frozen fresh-query manifest hash drift")
    fresh_queries = read_jsonl(fresh_queries_path)
    new_query_priorities = {
        str(row["query_id"]): str(row["prior_ablation_final_priority"])
        for row in fresh_queries
    }
    if len(new_query_priorities) != len(fresh_queries):
        raise ValueError("duplicate query IDs in frozen fresh-query manifest")
    return (
        raw_path,
        raw,
        reserve_path,
        reserve,
        fresh_queries_path,
        new_query_priorities,
    )


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    protocol = load_protocol(protocol_path)
    (
        raw_path,
        raw,
        reserve_path,
        reserve,
        fresh_queries_path,
        new_query_priorities,
    ) = _load_inputs(root, protocol)
    checker_root = root / "checker"
    checked_path = checker_root / "new_checked.jsonl"
    selected_path = checker_root / "ranking_selected.jsonl"
    completion_path = checker_root / "completion.json"
    if any(path.exists() for path in (checked_path, selected_path, completion_path)):
        raise FileExistsError("v2 checker outputs already exist")
    checker_version = str(protocol["checker"]["checker_version"])
    checked, health = materialize_numeric_checker_rows(
        raw, checker_version=checker_version
    )
    validation = validate_numeric_checker_rows(
        checked,
        raw_rows=raw,
        candidate_count=int(protocol["generation"]["candidate_count"]),
        checker_version=checker_version,
    )
    selected, selection = _select(
        checked,
        reserve,
        protocol,
        new_query_priorities=new_query_priorities,
    )
    checked_manifest = publish_manifest(
        checked_path,
        checked,
        schema_version="clir-prior-ablation-v2-new-checked",
        metadata={"health": health, "selection_used_clir_scores": False},
    )
    selected_manifest = publish_manifest(
        selected_path,
        selected,
        schema_version="clir-prior-ablation-v2-ranking-selected",
        metadata={
            "queries": int(protocol["ranking_population"]["total_queries"]),
            "candidates_per_query": int(protocol["generation"]["candidate_count"]),
            "source_query_counts": protocol["ranking_population"]["source_query_counts"],
            "selection_used_clir_scores": False,
        },
    )
    completion = {
        "schema_version": "clir-prior-ablation-v2-checker-completion",
        "status": "PASS_PRIOR_ABLATION_V2_CHECKER_AND_SELECTION",
        "protocol_file_sha256": file_sha256(protocol_path),
        "checker_implementation": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "raw_rollout": {"path": str(raw_path), "file_sha256": file_sha256(raw_path)},
        "math_reserve": {
            "path": str(reserve_path),
            "file_sha256": file_sha256(reserve_path),
        },
        "frozen_fresh_queries": {
            "path": str(fresh_queries_path),
            "file_sha256": file_sha256(fresh_queries_path),
        },
        "checker_version": checker_version,
        "new_checked": {
            "path": str(checked_path),
            "rows": len(checked),
            "file_sha256": checked_manifest["file_sha256"],
            "ordered_rows_sha256": checked_manifest["ordered_rows_sha256"],
            "sidecar_file_sha256": file_sha256(
                checked_path.with_suffix(checked_path.suffix + ".manifest.json")
            ),
        },
        "ranking_selected": {
            "path": str(selected_path),
            "rows": len(selected),
            "queries": int(protocol["ranking_population"]["total_queries"]),
            "file_sha256": selected_manifest["file_sha256"],
            "ordered_rows_sha256": selected_manifest["ordered_rows_sha256"],
            "sidecar_file_sha256": file_sha256(
                selected_path.with_suffix(selected_path.suffix + ".manifest.json")
            ),
        },
        "health": health,
        "validation": validation,
        "selection": selection,
        "clir_scores_opened": False,
    }
    atomic_write_json(completion_path, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    root = Path(args.output_root).resolve()
    protocol = load_protocol(protocol_path)
    (
        raw_path,
        raw,
        reserve_path,
        reserve,
        fresh_queries_path,
        new_query_priorities,
    ) = _load_inputs(root, protocol)
    completion_path = root / "checker/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_PRIOR_ABLATION_V2_CHECKER_AND_SELECTION":
        raise ValueError("checker completion is not a pass")
    if completion.get("checker_implementation", {}).get("file_sha256") != file_sha256(
        Path(__file__).resolve()
    ):
        raise ValueError("checker implementation hash drift")
    checked_path = Path(completion["new_checked"]["path"])
    selected_path = Path(completion["ranking_selected"]["path"])
    checked = read_jsonl(checked_path)
    selected = read_jsonl(selected_path)
    if (
        file_sha256(checked_path) != completion["new_checked"]["file_sha256"]
        or file_sha256(selected_path) != completion["ranking_selected"]["file_sha256"]
    ):
        raise ValueError("checker output hash drift")
    checker_version = str(protocol["checker"]["checker_version"])
    validate_numeric_checker_rows(
        checked,
        raw_rows=raw,
        candidate_count=int(protocol["generation"]["candidate_count"]),
        checker_version=checker_version,
    )
    if (
        completion.get("frozen_fresh_queries", {}).get("file_sha256")
        != file_sha256(fresh_queries_path)
    ):
        raise ValueError("checker completion fresh-query provenance drift")
    recomputed, selection = _select(
        checked,
        reserve,
        protocol,
        new_query_priorities=new_query_priorities,
    )
    if selected != recomputed or selection != completion["selection"]:
        raise ValueError("checker selection recomputation drift")
    report = {
        "schema_version": "clir-prior-ablation-v2-checker-verification",
        "status": "PASS_PRIOR_ABLATION_V2_CHECKER_INDEPENDENT_RECOMPUTE",
        "protocol_file_sha256": file_sha256(protocol_path),
        "checker_completion_file_sha256": file_sha256(completion_path),
        "checker_implementation_file_sha256": file_sha256(Path(__file__).resolve()),
        "raw_rollout_file_sha256": file_sha256(raw_path),
        "math_reserve_file_sha256": file_sha256(reserve_path),
        "frozen_fresh_queries_file_sha256": file_sha256(fresh_queries_path),
        "ranking_selected_file_sha256": file_sha256(selected_path),
        "rows": len(selected),
        "queries": len({str(row["query_id"]) for row in selected}),
        "selection_used_clir_scores": False,
    }
    target = root / "checker/independent_verification.json"
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize").set_defaults(func=command_materialize)
    sub.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
