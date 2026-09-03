#!/usr/bin/env python
"""Score the frozen official-SWIFT checkpoints on the reused ranking population.

The 2,400-query / 38,400-row feature manifest is reused verbatim from
``run_artifacts/prior_ablation_v2``; nothing is re-extracted.  Every checkpoint
is scored in a single feature pass via ``stacked_swift_scores``, and the emitted
row schema matches the CLIR scalar-scoring schema so the shared summary
machinery reads both without modification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from run_swift_official_baseline_training import CELL, load_protocol
from score_clir import atomic_write_jsonl, file_sha256
from score_clir_factorial import (
    _add_global_selections,
    _base_scalar_row,
    _shard_name,
    _validate_merged_source_row,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    read_jsonl,
    validate_rollout_population,
)
from src.swift_official_baseline import (
    UPSTREAM_COMMIT,
    SwiftFeatureDataset,
    swift_collate,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/swift_official_baseline_v1/protocol.json"
TRAINING_COMPLETION_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_MATCHED_TRAINING_GRID"
SHARD_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_SCORING_SHARD"
MERGE_STATUS = "PASS_SWIFT_OFFICIAL_BASELINE_SCORING_MERGE"
SCORING_MODE = "scalar_only"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_key(run: Mapping[str, Any]) -> str:
    return f"{run['cell']}/seed-{int(run['seed'])}"


def _output_path(root: Path, run: Mapping[str, Any], shard_name: str) -> Path:
    return root / "shards" / str(run["cell"]) / f"seed-{int(run['seed'])}" / f"{shard_name}.jsonl"


def _load_bound_contract(
    *, protocol_path: Path, completion_path: Path, input_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    """Bind protocol, training completion, and the reused feature manifest by hash."""

    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    root = _resolve(protocol["runtime"]["output_root"])
    if completion_path.resolve() != (root / "training/completion.json").resolve():
        raise ValueError("SWIFT scorer training-completion path drift")

    feature_spec = protocol["frozen_parents"]["ranking_feature_manifest"]
    if input_path.resolve() != _resolve(feature_spec["path"]):
        raise ValueError("SWIFT scorer ranking-input path drift")
    input_sha = file_sha256(input_path)
    if input_sha != feature_spec["file_sha256"]:
        raise ValueError("reused ranking feature manifest hash drift")

    feature_completion_spec = protocol["frozen_parents"]["ranking_feature_completion"]
    feature_completion = _load_json(_resolve(feature_completion_spec["path"]))
    if feature_completion.get("status") != feature_completion_spec["expected_status"]:
        raise ValueError("reused feature extraction is incomplete")
    if feature_completion["tuning_manifest"]["file_sha256"] != input_sha:
        raise ValueError("reused feature completion does not bind this manifest")

    completion = _load_json(completion_path)
    completion_sha = file_sha256(completion_path)
    if (
        completion.get("status") != TRAINING_COMPLETION_STATUS
        or completion.get("protocol_file_sha256") != protocol_sha
        or completion.get("upstream_commit") != UPSTREAM_COMMIT
        or int(completion.get("run_count", -1)) != int(protocol["training"]["run_count"])
    ):
        raise ValueError("SWIFT training completion is missing or stale")
    runs = [dict(run) for run in completion["runs"]]
    expected_grid = {(CELL, int(seed)) for seed in protocol["training"]["seeds"]}
    if {(str(run["cell"]), int(run["seed"])) for run in runs} != expected_grid:
        raise ValueError("SWIFT checkpoint grid drift")

    source = read_jsonl(input_path)
    population = validate_rollout_population(
        source, candidate_count=int(protocol["ranking_population"]["candidates_per_query"])
    )
    ranking = protocol["ranking_population"]
    if (
        len(source) != int(ranking["selected_candidate_rows"])
        or int(population["queries"]) != int(ranking["total_queries"])
    ):
        raise ValueError("reused ranking population drift")
    return protocol, protocol_sha, completion, completion_sha, source, input_sha


def _load_state_dicts(
    runs: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    model_spec = protocol["model"]
    if bool(model_spec["disable_gate"]):
        raise ValueError("stacked scoring requires the gated SWIFT head")
    state_dicts: list[dict[str, torch.Tensor]] = []
    for run in runs:
        path = _resolve(str(run["checkpoint_path"]))
        if file_sha256(path) != run["checkpoint_file_sha256"]:
            raise ValueError(f"checkpoint hash drift for {_run_key(run)}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("completed_epoch", -1)) != int(
            protocol["training"]["epochs"]
        ):
            raise ValueError(f"checkpoint epoch drift for {_run_key(run)}")
        if int(checkpoint.get("seed", -1)) != int(run["seed"]):
            raise ValueError(f"checkpoint seed drift for {_run_key(run)}")
        state = checkpoint["state_dict"]
        weight = state["fused_layer.weight"]
        if tuple(weight.shape) != (2, int(model_spec["feature_dim"])):
            raise ValueError(f"checkpoint head shape drift for {_run_key(run)}")
        state_dicts.append(
            {
                "fused_layer.weight": weight.to(device=device, dtype=torch.float32),
                "fused_layer.bias": state["fused_layer.bias"].to(
                    device=device, dtype=torch.float32
                ),
            }
        )
    return state_dicts


def _stacked_scores(
    hidden_states: torch.Tensor,
    lengths: Sequence[int],
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
    amp_dtype: str,
) -> torch.Tensor:
    from src.swift_official_baseline import stacked_swift_scores

    device = hidden_states.device
    if amp_dtype == "bfloat16" and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = stacked_swift_scores(hidden_states, lengths, state_dicts)
    else:
        scores = stacked_swift_scores(hidden_states, lengths, state_dicts)
    return scores.float()


def command_worker(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    protocol_path = Path(args.protocol).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    (
        protocol,
        protocol_sha,
        completion,
        completion_sha,
        source,
        input_sha,
    ) = _load_bound_contract(
        protocol_path=protocol_path,
        completion_path=completion_path,
        input_path=input_path,
    )
    runtime = protocol["runtime"]
    if (
        args.num_shards != int(runtime["ranking_score_shards"])
        or args.batch_size != int(runtime["ranking_score_batch_size"])
        or args.num_workers != int(runtime["ranking_score_num_workers"])
        or args.amp_dtype != runtime["ranking_score_amp_dtype"]
    ):
        raise ValueError("SWIFT scorer runtime contract drift")
    root = _resolve(runtime["output_root"])
    output_root = Path(args.output_root).resolve()
    if output_root != (root / "ranking/scoring_shards").resolve():
        raise ValueError("SWIFT scorer shard output root drift")

    runs = sorted(completion["runs"], key=lambda run: int(run["seed"]))
    shard_name = _shard_name(args.shard_index, args.num_shards)
    manifest_path = output_root / "shards" / f"{shard_name}.manifest.json"
    targets = [_output_path(output_root, run, shard_name) for run in runs]
    existing = [path for path in (manifest_path, *targets) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"scoring shard output exists: {existing[0]}")

    dataset = SwiftFeatureDataset(input_path)
    if len(dataset) != len(source):
        raise ValueError("SWIFT scorer dataset/source row-count drift")
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    if not indices:
        raise ValueError("scoring shard contains no source rows")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=swift_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    state_dicts = _load_state_dicts(runs, protocol, device)

    buffers: dict[str, list[dict[str, Any]]] = {_run_key(run): [] for run in runs}
    for batch in loader:
        row_indices = [int(value) for value in batch["row_indices"]]
        hidden_states = batch["hidden_states"].to(device).float()
        scores = _stacked_scores(
            hidden_states, batch["lengths"], state_dicts, args.amp_dtype
        )
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("non-finite SWIFT scores")
        if tuple(scores.shape) != (len(row_indices), len(runs)):
            raise ValueError("stacked score shape drift")
        host = scores.detach().cpu()
        for position, run in enumerate(runs):
            key = _run_key(run)
            checkpoint_sha = str(run["checkpoint_file_sha256"])
            for offset, source_row_index in enumerate(row_indices):
                row = _base_scalar_row(
                    dataset.rows[source_row_index], source_row_index, checkpoint_sha
                )
                row["clir_score"] = float(host[offset, position])
                buffers[key].append(row)
        del hidden_states, scores, host

    outputs_manifest: dict[str, Any] = {}
    for run in runs:
        key = _run_key(run)
        rows = buffers[key]
        if [int(row["source_row_index"]) for row in rows] != indices:
            raise ValueError(f"source-order drift in {key}")
        target = _output_path(output_root, run, shard_name)
        atomic_write_jsonl(target, rows)
        outputs_manifest[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }
    report = {
        "schema_version": "swift-official-baseline-v1-scoring-shard",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "input_jsonl_sha256": input_sha,
        "input_rows": len(dataset),
        "completion_report_sha256": completion_sha,
        "scorer_sha256": file_sha256(Path(__file__)),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_rows": len(indices),
        "source_indices_sha256": canonical_sha256(indices),
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "batch_size": args.batch_size,
        "single_feature_pass_for_all_checkpoints": True,
        "outputs": outputs_manifest,
    }
    atomic_write_json(manifest_path, report)
    print(json.dumps({"status": SHARD_STATUS, "shard": shard_name, "rows": len(indices)}))


def command_merge(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    protocol_path = Path(args.protocol).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    (
        protocol,
        protocol_sha,
        completion,
        completion_sha,
        source,
        input_sha,
    ) = _load_bound_contract(
        protocol_path=protocol_path,
        completion_path=completion_path,
        input_path=input_path,
    )
    root = _resolve(protocol["runtime"]["output_root"])
    shard_root = Path(args.shard_root).resolve()
    output_root = Path(args.output_root).resolve()
    if (
        shard_root != (root / "ranking/scoring_shards").resolve()
        or output_root != (root / "ranking/scored").resolve()
        or args.num_shards != int(protocol["runtime"]["ranking_score_shards"])
    ):
        raise ValueError("SWIFT scorer merge runtime contract drift")

    runs = sorted(completion["runs"], key=lambda run: int(run["seed"]))
    shard_manifests: dict[str, Any] = {}
    for shard_index in range(args.num_shards):
        shard_name = _shard_name(shard_index, args.num_shards)
        manifest_path = shard_root / "shards" / f"{shard_name}.manifest.json"
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") != SHARD_STATUS
            or manifest.get("protocol_file_sha256") != protocol_sha
            or manifest.get("input_jsonl_sha256") != input_sha
            or manifest.get("completion_report_sha256") != completion_sha
            or int(manifest.get("num_shards", -1)) != args.num_shards
        ):
            raise ValueError(f"scoring shard manifest drift: {shard_name}")
        shard_manifests[shard_name] = file_sha256(manifest_path)

    merged_outputs: dict[str, Any] = {}
    for run in runs:
        key = _run_key(run)
        rows: list[dict[str, Any]] = []
        for shard_index in range(args.num_shards):
            shard_name = _shard_name(shard_index, args.num_shards)
            path = _output_path(shard_root, run, shard_name)
            expected = _load_json(
                shard_root / "shards" / f"{shard_name}.manifest.json"
            )["outputs"][key]
            if file_sha256(path) != expected["file_sha256"]:
                raise ValueError(f"shard payload hash drift: {key}/{shard_name}")
            rows.extend(read_jsonl(path))
        if len(rows) != len(source):
            raise ValueError(f"merged row count drift in {key}")
        rows.sort(key=lambda row: int(row["source_row_index"]))
        for index, (row, reference) in enumerate(zip(rows, source, strict=True)):
            _validate_merged_source_row(row, reference, SCORING_MODE, index)
            if row.get("clir_checkpoint_sha256") != run["checkpoint_file_sha256"]:
                raise ValueError(f"checkpoint identity drift in {key}")
            if row.get("clir_scoring_mode") != SCORING_MODE:
                raise ValueError(f"scoring mode drift in {key}")
            if not math.isfinite(float(row["clir_score"])):
                raise ValueError(f"non-finite merged score in {key} at {index}")
        _add_global_selections(rows)
        target = output_root / str(run["cell"]) / f"seed-{int(run['seed'])}" / "scored.jsonl"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"merged score exists: {target}")
        atomic_write_jsonl(target, rows)
        merged_outputs[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }
    report = {
        "schema_version": "swift-official-baseline-v1-scoring-merge",
        "status": MERGE_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "input_jsonl_sha256": input_sha,
        "input_rows": len(source),
        "completion_report_sha256": completion_sha,
        "scorer_sha256": file_sha256(Path(__file__)),
        "num_shards": args.num_shards,
        "shard_manifest_sha256": shard_manifests,
        "outputs": merged_outputs,
    }
    atomic_write_json(output_root / "merge_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    sub = parser.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("worker")
    worker.add_argument("--completion-report", required=True)
    worker.add_argument("--input-jsonl", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--shard-index", required=True, type=int)
    worker.add_argument("--num-shards", required=True, type=int)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--batch-size", required=True, type=int)
    worker.add_argument("--num-workers", required=True, type=int)
    worker.add_argument("--amp-dtype", required=True, choices=["none", "bfloat16"])
    worker.add_argument("--overwrite", action="store_true")
    worker.set_defaults(func=command_worker)

    merge = sub.add_parser("merge")
    merge.add_argument("--completion-report", required=True)
    merge.add_argument("--input-jsonl", required=True)
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--num-shards", required=True, type=int)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=command_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
