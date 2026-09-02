#!/usr/bin/env python
"""Run the frozen Prior-ablation-v2 Phi rollout shards and merge them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_clir_gate_tuning_rollout as engine
from prepare_clir_prior_ablation_v2 import load_protocol
from src.clir_smoke import (
    atomic_write_json,
    file_sha256,
    publish_manifest,
    read_jsonl,
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_ablation_v2/protocol.json"
DEFAULT_PRE_ROLLOUT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2/pre_rollout"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/prior_ablation_v2"


def _load_contract(args: argparse.Namespace):
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout = Path(args.pre_rollout).resolve()
    output_root = Path(args.output_root).resolve()
    if authorization_path != protocol_path:
        raise ValueError("v2 rollout uses the frozen protocol as its authorization")
    protocol = load_protocol(protocol_path)
    expected_root = (PROJECT_ROOT / protocol["runtime"]["output_root"]).resolve()
    if output_root != expected_root:
        raise ValueError("rollout output root drift")
    verification = json.loads(
        (pre_rollout / "independent_verification.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (pre_rollout / "manifest_registry.json").read_text(encoding="utf-8")
    )
    if (
        verification.get("status") != "PASS_PRIOR_ABLATION_V2_INDEPENDENT_RECOMPUTE"
        or verification.get("registry_file_sha256")
        != file_sha256(pre_rollout / "manifest_registry.json")
        or registry.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("v2 pre-rollout freeze is missing or stale")
    queries = read_jsonl(pre_rollout / "fresh_queries.jsonl")
    if len(queries) != sum(
        int(value)
        for value in protocol["ranking_population"]["new_rollout_raw_queries"].values()
    ):
        raise ValueError("fresh query count drift")
    query_by_id = {str(row["query_id"]): row for row in queries}
    if len(query_by_id) != len(queries):
        raise ValueError("duplicate fresh query IDs")
    shards = json.loads((pre_rollout / "rollout_shards.json").read_text(encoding="utf-8"))
    shard_ids = [str(value) for shard in shards for value in shard["query_ids"]]
    if len(shards) != int(protocol["generation"]["rollout_shards"]):
        raise ValueError("rollout shard count drift")
    if shard_ids != [str(row["query_id"]) for row in queries]:
        raise ValueError("rollout shards do not preserve the frozen query order")
    authorization: dict[str, Any] = {
        "runtime_contract": {
            "authorized_code_parent_commit": registry["code_commit"],
            "first_calibration_shard": "ranking-000",
            "dtype": "bfloat16",
            "max_num_seqs": int(protocol["runtime"]["rollout_max_num_seqs"]),
            "gpu_memory_utilization": float(
                protocol["runtime"]["rollout_gpu_memory_utilization"]
            ),
            "minimum_free_gpu_bytes": 40_000_000_000,
            "cache_dir": protocol["runtime"]["model_cache"],
            "output_root": protocol["runtime"]["output_root"],
        }
    }
    return (
        protocol,
        authorization,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    )


def command_merge(args: argparse.Namespace) -> None:
    (
        protocol,
        _,
        shards,
        query_by_id,
        protocol_path,
        authorization_path,
        output_root,
    ) = _load_contract(args)
    pre_rollout = Path(args.pre_rollout).resolve()
    target = output_root / "rollouts/new_combined_raw.jsonl"
    completion_path = output_root / "rollout_completion_report.json"
    if target.exists() or completion_path.exists():
        raise FileExistsError("combined v2 rollout already exists")
    rows: list[dict[str, Any]] = []
    reports = []
    for shard in shards:
        report = engine.verify_shard(
            shard=shard,
            query_by_id=query_by_id,
            protocol=protocol,
            protocol_path=protocol_path,
            authorization_path=authorization_path,
            pre_rollout_dir=pre_rollout,
            output_root=output_root,
        )
        reports.append(report)
        rows.extend(read_jsonl(Path(report["path"])))
    population = validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    if int(population["queries"]) != len(query_by_id):
        raise ValueError("merged query count drift")
    manifest = publish_manifest(
        target,
        rows,
        schema_version="clir-prior-ablation-v2-new-raw-rollouts",
        metadata={
            **population,
            "numeric_checker_run": False,
            "clir_scoring_run": False,
        },
    )
    completion = {
        "schema_version": "clir-prior-ablation-v2-rollout-completion",
        "status": "PASS_PRIOR_ABLATION_V2_NEW_RAW_ROLLOUTS",
        "protocol_file_sha256": file_sha256(protocol_path),
        "pre_rollout_registry_file_sha256": file_sha256(
            pre_rollout / "manifest_registry.json"
        ),
        "verified_shards": len(reports),
        "verified_rows": len(rows),
        "combined": {
            "path": str(target),
            "rows": len(rows),
            "queries": int(population["queries"]),
            "file_sha256": manifest["file_sha256"],
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
            "sidecar_file_sha256": file_sha256(
                target.with_suffix(target.suffix + ".manifest.json")
            ),
        },
        "total_output_tokens": sum(
            int(report["validation"]["total_output_tokens"]) for report in reports
        ),
        "numeric_checker_run": False,
        "clir_scoring_run": False,
    }
    atomic_write_json(completion_path, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--authorization", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--pre-rollout", default=str(DEFAULT_PRE_ROLLOUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    sub = parser.add_subparsers(dest="command", required=True)
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--shard-id", required=True)
    rollout.set_defaults(func=engine.command_rollout)
    verify = sub.add_parser("verify")
    verify.add_argument("--shard-id", action="append")
    verify.add_argument("--require-complete", action="store_true")
    verify.set_defaults(func=engine.command_verify)
    sub.add_parser("merge").set_defaults(func=command_merge)
    return parser


def main() -> None:
    engine._load_contract = _load_contract
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
