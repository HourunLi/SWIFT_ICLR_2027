#!/usr/bin/env python
"""Extract and independently verify exact-token features for MATH-hard v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import extract_clir_gate_tuning_features as engine
from prepare_clir_math_hard_eval import load_protocol
from src.clir_scale_features import assign_workers, selected_statistics
from src.clir_smoke import file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/math_hard_eval_v1/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1/features_v1"


def _authorization(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    root = (PROJECT_ROOT / protocol["runtime"]["output_root"]).resolve()
    completion_path = root / "checker/completion.json"
    verification_path = root / "checker/independent_verification.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "PASS_MATH_HARD_EVAL_V1_SWIFT_CHECKER"
        or verification.get("status")
        != "PASS_MATH_HARD_EVAL_V1_CHECKER_INDEPENDENT_RECOMPUTE"
        or verification.get("checker_completion_file_sha256")
        != file_sha256(completion_path)
    ):
        raise ValueError("protected checker is missing or stale")
    source_path = Path(completion["checked"]["path"])
    if file_sha256(source_path) != completion["checked"]["file_sha256"]:
        raise ValueError("protected checked population hash drift")
    rows = read_jsonl(source_path)
    candidate_count = int(protocol["generation"]["candidate_count"])
    inventory: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for index, row in enumerate(rows):
        query_id = str(row["query_id"])
        candidate_index = int(row["candidate_index"])
        owner = query_id not in seen_queries
        if owner:
            seen_queries.add(query_id)
        if owner != (candidate_index == 0):
            raise ValueError(f"{query_id}: candidate zero must own condition features")
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        output_ids = [int(value) for value in row["output_token_ids"]]
        inventory.append(
            {
                "schema_version": engine.INVENTORY_SCHEMA,
                "inventory_index": index,
                "id": str(row["id"]),
                "trajectory_id": str(row["id"]),
                "query_id": query_id,
                "candidate_index": candidate_index,
                "role": "math_hard_eval_v1",
                "evaluation_split": "protected_math_test_level_4_5",
                "sealed_until_weight_lock": True,
                "source": "math",
                "cluster_id": str(row["cluster_id"]),
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "prompt_token_count": len(prompt_ids),
                "output_token_count": len(output_ids),
                "condition_feature_owner": owner,
            }
        )
    expected_rows = int(protocol["source"]["total_queries"]) * candidate_count
    if len(rows) != expected_rows or len(seen_queries) != int(
        protocol["source"]["total_queries"]
    ):
        raise ValueError("protected feature source arithmetic drift")
    assigned, worker_stats = assign_workers(
        inventory, int(protocol["feature_extraction"]["worker_count"])
    )
    total = selected_statistics(assigned)
    expected = {
        **total,
        "raw_feature_bytes": int(total["total_feature_token_count"])
        * int(protocol["feature_extraction"]["bytes_per_feature_token"]),
    }
    registry = json.loads(
        (root / "pre_rollout/manifest_registry.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "clir-math-hard-eval-v1-feature-authorization-adapter",
        "status": "AUTHORIZED_MATH_HARD_EVAL_V1_SELECTED_ONLY_FEATURES",
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "source_path": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "inventory_rows": assigned,
        "inventory_report": {
            "total": total,
            "worker_count": len(worker_stats),
            "worker_statistics": worker_stats,
        },
        "frozen_parent": {
            "authorized_code_parent_commit": registry["code_commit"],
            "files": {},
        },
        "feature_contract": dict(protocol["feature_extraction"]),
        "runtime_contract": {
            "output_root": str(root / "features_v1"),
            "cache_dir": protocol["runtime"]["model_cache"],
            "worker_count": int(protocol["feature_extraction"]["worker_count"]),
            "candidate_count": candidate_count,
        },
        "expected_inventory": expected,
    }


def _build_inventory(authorization: Mapping[str, Any]):
    return (
        [dict(row) for row in authorization["inventory_rows"]],
        dict(authorization["inventory_report"]),
    )


def _selected_sources(authorization: Mapping[str, Any]):
    path = Path(str(authorization["source_path"]))
    if file_sha256(path) != authorization["source_file_sha256"]:
        raise ValueError("protected feature finalization source hash drift")
    return read_jsonl(path), []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--authorization", default=str(DEFAULT_PROTOCOL))
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))

    prepare = sub.add_parser("prepare")
    common(prepare)
    prepare.set_defaults(func=engine.command_prepare)
    verify_plan = sub.add_parser("verify-plan")
    common(verify_plan)
    verify_plan.set_defaults(func=engine.command_verify_plan)
    preflight = sub.add_parser("preflight")
    common(preflight)
    preflight.add_argument("--device", default="cuda:0")
    preflight.set_defaults(func=engine.command_preflight)
    worker = sub.add_parser("extract-worker")
    common(worker)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--device", default="cuda:0")
    worker.set_defaults(func=engine.command_extract_worker)
    verifier = sub.add_parser("verify-worker")
    common(verifier)
    verifier.add_argument("--worker-index", type=int, required=True)
    verifier.set_defaults(func=engine.command_verify_worker)
    finalize = sub.add_parser("finalize")
    common(finalize)
    finalize.set_defaults(func=engine.command_finalize)
    return parser


def main() -> None:
    engine.load_authorization = _authorization
    engine._build_inventory = _build_inventory
    engine._selected_sources = _selected_sources
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
