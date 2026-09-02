#!/usr/bin/env python
"""Extract and verify selected-only v2 ranking features.

The heavy tensor implementation is shared with the already tested Prior/Gate
extractor.  This adapter supplies the v2 hash-bound inventory and keeps every
candidate from the frozen 2,400 x 16 ranking population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import extract_clir_gate_tuning_features as engine
from prepare_clir_prior_ablation_v2 import load_protocol
from src.clir_scale_features import assign_workers, selected_statistics
from src.clir_smoke import file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2/features_v2"


def _authorization(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = load_protocol(protocol_path)
    root = (PROJECT_ROOT / protocol["runtime"]["output_root"]).resolve()
    checker_completion_path = root / "checker/completion.json"
    checker_verification_path = root / "checker/independent_verification.json"
    completion = json.loads(checker_completion_path.read_text(encoding="utf-8"))
    verification = json.loads(checker_verification_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "PASS_PRIOR_ABLATION_V2_CHECKER_AND_SELECTION"
        or verification.get("status")
        != "PASS_PRIOR_ABLATION_V2_CHECKER_INDEPENDENT_RECOMPUTE"
        or verification.get("checker_completion_file_sha256")
        != file_sha256(checker_completion_path)
    ):
        raise ValueError("v2 checker selection is missing or stale")
    source_path = Path(completion["ranking_selected"]["path"])
    if file_sha256(source_path) != completion["ranking_selected"]["file_sha256"]:
        raise ValueError("v2 ranking selection hash drift")
    rows = read_jsonl(source_path)
    candidate_count = int(protocol["generation"]["candidate_count"])
    inventory: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for index, row in enumerate(rows):
        query_id = str(row["query_id"])
        candidate_index = int(row["candidate_index"])
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        output_ids = [int(value) for value in row["output_token_ids"]]
        owner = query_id not in seen_queries
        if owner:
            seen_queries.add(query_id)
        if owner != (candidate_index == 0):
            raise ValueError(f"{query_id}: candidate zero must own condition features")
        inventory.append(
            {
                "schema_version": engine.INVENTORY_SCHEMA,
                "inventory_index": index,
                "id": str(row["id"]),
                "trajectory_id": str(row["id"]),
                "query_id": query_id,
                "candidate_index": candidate_index,
                "role": "prior_ablation_v2_ranking",
                "evaluation_split": "prior_ablation_v2",
                "sealed_until_weight_lock": False,
                "source": str(row["source"]),
                "cluster_id": str(row["cluster_id"]),
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "prompt_token_count": len(prompt_ids),
                "output_token_count": len(output_ids),
                "condition_feature_owner": owner,
            }
        )
    if len(rows) != int(protocol["ranking_population"]["selected_candidate_rows"]):
        raise ValueError("feature source row count drift")
    if len(seen_queries) != int(protocol["ranking_population"]["total_queries"]):
        raise ValueError("feature source query count drift")
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
        "schema_version": "clir-prior-ablation-v2-feature-authorization-adapter",
        "status": "AUTHORIZED_PRIOR_ABLATION_V2_SELECTED_ONLY_FEATURES",
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
            "output_root": str(root / "features_v2"),
            "cache_dir": protocol["runtime"]["model_cache"],
            "worker_count": int(protocol["feature_extraction"]["worker_count"]),
            "candidate_count": candidate_count,
        },
        "expected_inventory": expected,
    }


def _build_inventory(authorization: Mapping[str, Any]):
    rows = [dict(row) for row in authorization["inventory_rows"]]
    return rows, dict(authorization["inventory_report"])


def _selected_sources(authorization: Mapping[str, Any]):
    path = Path(str(authorization["source_path"]))
    if file_sha256(path) != authorization["source_file_sha256"]:
        raise ValueError("feature finalization source hash drift")
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
