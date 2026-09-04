#!/usr/bin/env python
"""Score the two new epoch-3 sampler-factorial cells on the reused 2,400 queries."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from run_swift_u0_sampler_factorial import (
    COMPLETION_STATUS,
    NEW_CELLS,
    _git_state,
    _audit_anchors,
    _audit_ignored_parent_reports,
    _require_clean_commit,
    _resolve,
    load_protocol,
)
from score_clir import atomic_write_jsonl
from score_clir_factorial import (
    _add_global_selections,
    _base_scalar_row,
    _shard_name,
    _validate_merged_source_row,
)
from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    validate_rollout_population,
)
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig
from src.swift_official_baseline import stacked_swift_scores


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/swift_u0_sampler_factorial_v1/protocol.json"
)
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/swift_u0_sampler_factorial_v1"
SHARD_STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_SCORING_SHARD"
MERGE_STATUS = "PASS_SWIFT_U0_SAMPLER_FACTORIAL_SCORING_MERGE"
SCORING_MODE = "scalar_only"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_key(run: Mapping[str, Any]) -> str:
    return f"{run['cell']}/seed-{int(run['seed'])}"


def _output_path(root: Path, run: Mapping[str, Any], shard: str) -> Path:
    return root / "shards" / str(run["cell"]) / f"seed-{int(run['seed'])}" / f"{shard}.jsonl"


def _load_contract(
    protocol_path: Path,
    completion_path: Path,
    input_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    protocol = load_protocol(protocol_path)
    protocol_sha = file_sha256(protocol_path)
    state = _require_clean_commit(protocol)
    _audit_ignored_parent_reports(protocol)
    _audit_anchors(protocol)
    root = _resolve(protocol["runtime"]["output_root"])
    if completion_path.resolve() != (root / "training/completion.json").resolve():
        raise ValueError("training-completion path drift")
    completion = _load_json(completion_path)
    completion_sha = file_sha256(completion_path)
    if (
        completion.get("status") != COMPLETION_STATUS
        or completion.get("protocol_file_sha256") != protocol_sha
        or completion.get("code", {}).get("commit") != state["commit"]
        or int(completion.get("primary_run_count", -1)) != 6
    ):
        raise ValueError("training completion is missing, stale, or from another commit")
    runs = [dict(run) for run in completion["runs"]]
    expected = {
        (cell, int(seed))
        for cell in NEW_CELLS
        for seed in protocol["training"]["seeds"]
    }
    if (
        {(str(run["cell"]), int(run["seed"])) for run in runs} != expected
        or any(int(run["epoch"]) != int(protocol["training"]["primary_epoch"]) for run in runs)
    ):
        raise ValueError("primary checkpoint grid drift")

    feature_spec = protocol["frozen_parents"]["ranking_feature_manifest"]
    if input_path.resolve() != _resolve(feature_spec["path"]):
        raise ValueError("ranking input path drift")
    input_sha = file_sha256(input_path)
    if input_sha != feature_spec["file_sha256"]:
        raise ValueError("ranking input hash drift")
    source = read_jsonl(input_path)
    population = validate_rollout_population(
        source, candidate_count=int(feature_spec["candidates_per_query"])
    )
    if (
        len(source) != int(feature_spec["rows"])
        or int(population["queries"]) != int(feature_spec["queries"])
    ):
        raise ValueError("ranking population inventory drift")
    return protocol, protocol_sha, completion, completion_sha, source, input_sha


def _autocast(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _load_models(
    runs: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[
    list[tuple[dict[str, Any], ConsistencyLocalizedReward]],
    list[dict[str, Any]],
    list[dict[str, torch.Tensor]],
]:
    u0: list[tuple[dict[str, Any], ConsistencyLocalizedReward]] = []
    swift_runs: list[dict[str, Any]] = []
    swift_states: list[dict[str, torch.Tensor]] = []
    for raw_run in sorted(runs, key=lambda item: (str(item["cell"]), int(item["seed"]))):
        run = dict(raw_run)
        path = Path(str(run["checkpoint_path"])).resolve()
        if file_sha256(path) != run["checkpoint_file_sha256"]:
            raise ValueError(f"checkpoint hash drift: {_run_key(run)}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("completed_epoch", -1)) != int(run["epoch"]):
            raise ValueError(f"checkpoint epoch drift: {_run_key(run)}")
        if run["cell"] == "u0_random":
            config = checkpoint.get("model_config")
            if not isinstance(config, Mapping):
                raise ValueError(f"U0 checkpoint lacks model config: {_run_key(run)}")
            model = ConsistencyLocalizedReward(RewardConfig(**dict(config))).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            u0.append((run, model))
        elif run["cell"] == "swift_grouped":
            state = checkpoint["state_dict"]
            weight = state["fused_layer.weight"]
            bias = state["fused_layer.bias"]
            if tuple(weight.shape) != (2, 101376) or tuple(bias.shape) != (2,):
                raise ValueError(f"SWIFT head shape drift: {_run_key(run)}")
            swift_runs.append(run)
            swift_states.append(
                {
                    "fused_layer.weight": weight.to(device=device, dtype=torch.float32),
                    "fused_layer.bias": bias.to(device=device, dtype=torch.float32),
                }
            )
        else:
            raise ValueError(f"unexpected scoring cell: {run['cell']}")
    if len(u0) != 3 or len(swift_runs) != 3:
        raise ValueError("expected three U0 and three SWIFT checkpoints")
    return u0, swift_runs, swift_states


def _u0_scores(
    model: ConsistencyLocalizedReward,
    batch: Mapping[str, Any],
    amp_dtype: str,
) -> torch.Tensor:
    with _autocast(batch["hidden_states"].device, amp_dtype):
        outputs = model(
            batch["hidden_states"],
            mask=batch["mask"],
            condition_states=batch.get("condition_states"),
            condition_mask=batch.get("condition_mask"),
            condition_embedding=batch.get("condition_embedding"),
            condition_embedding_mask=batch.get("condition_embedding_mask"),
        )
    scores = outputs["scores"].float()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite U0 ranking scores")
    return scores


@torch.no_grad()
def command_worker(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    protocol, protocol_sha, completion, completion_sha, source, input_sha = _load_contract(
        protocol_path, completion_path, input_path
    )
    runtime = protocol["runtime"]
    if (
        int(args.num_shards) != int(runtime["scoring_shards"])
        or int(args.batch_size) != int(runtime["scoring_batch_size"])
        or int(args.num_workers) != int(runtime["scoring_num_workers"])
        or args.amp_dtype != runtime["scoring_amp_dtype"]
        or not 0 <= int(args.shard_index) < int(args.num_shards)
    ):
        raise ValueError("scoring runtime contract drift")
    root = _resolve(runtime["output_root"])
    output_root = Path(args.output_root).resolve()
    if output_root != (root / "ranking/scoring_shards").resolve():
        raise ValueError("scoring output-root drift")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each scoring worker must see exactly one GPU")
    minimum = int(runtime["launch_only_when_all_eight_gpus_have_at_least_mib_free"])
    free, _ = torch.cuda.mem_get_info(0)
    if free // 1024**2 < minimum:
        raise RuntimeError("visible GPU is not idle enough to start scoring")

    shard = _shard_name(args.shard_index, args.num_shards)
    manifest_path = output_root / "shards" / f"{shard}.manifest.json"
    targets = [_output_path(output_root, run, shard) for run in completion["runs"]]
    existing = [path for path in (manifest_path, *targets) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"scoring output exists: {existing[0]}")

    dataset = CLIRTrajectoryDataset(input_path)
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("full-width scoring requires CUDA")
    u0_models, swift_runs, swift_states = _load_models(completion["runs"], device)
    buffers = {_run_key(run): [] for run in completion["runs"]}

    for raw_batch in loader:
        row_indices = [int(value) for value in raw_batch["row_index"].tolist()]
        batch = move_batch_to_device(raw_batch, device)
        for run, model in u0_models:
            scores = _u0_scores(model, batch, args.amp_dtype)
            key = _run_key(run)
            for offset, source_index in enumerate(row_indices):
                row = _base_scalar_row(
                    dataset.rows[source_index],
                    source_index,
                    str(run["checkpoint_file_sha256"]),
                )
                row["clir_score"] = float(scores[offset].cpu())
                buffers[key].append(row)
            del scores
        lengths = [int(value) for value in batch["mask"].sum(dim=1).tolist()]
        with _autocast(device, args.amp_dtype):
            swift_scores = stacked_swift_scores(
                batch["hidden_states"], lengths, swift_states
            ).float()
        if (
            tuple(swift_scores.shape) != (len(row_indices), len(swift_runs))
            or not bool(torch.isfinite(swift_scores).all())
        ):
            raise FloatingPointError("invalid stacked SWIFT ranking scores")
        for position, run in enumerate(swift_runs):
            key = _run_key(run)
            for offset, source_index in enumerate(row_indices):
                row = _base_scalar_row(
                    dataset.rows[source_index],
                    source_index,
                    str(run["checkpoint_file_sha256"]),
                )
                row["clir_score"] = float(swift_scores[offset, position].cpu())
                buffers[key].append(row)
        del batch, swift_scores

    outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows = buffers[key]
        if [int(row["source_row_index"]) for row in rows] != indices:
            raise ValueError(f"source-order drift for {key}")
        target = _output_path(output_root, run, shard)
        atomic_write_jsonl(target, rows)
        outputs[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }
    report = {
        "schema_version": "swift-u0-sampler-factorial-v1-scoring-shard",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "training_completion_file_sha256": completion_sha,
        "ranking_input_file_sha256": input_sha,
        "scorer_file_sha256": file_sha256(Path(__file__)),
        "code": _git_state(),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_indices_sha256": canonical_sha256(indices),
        "source_rows": len(indices),
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "single_feature_pass_for_all_six_primary_checkpoints": True,
        "outputs": outputs,
        "hard_math_opened": False,
    }
    atomic_write_json(manifest_path, report)
    print(json.dumps({"status": SHARD_STATUS, "shard": shard, "rows": len(indices)}))


def command_merge(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    protocol, protocol_sha, completion, completion_sha, source, input_sha = _load_contract(
        protocol_path, completion_path, input_path
    )
    runtime = protocol["runtime"]
    root = _resolve(runtime["output_root"])
    shard_root = Path(args.shard_root).resolve()
    output_root = Path(args.output_root).resolve()
    if (
        shard_root != (root / "ranking/scoring_shards").resolve()
        or output_root != (root / "ranking/scored").resolve()
        or int(args.num_shards) != int(runtime["scoring_shards"])
    ):
        raise ValueError("merge runtime contract drift")
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for shard_index in range(args.num_shards):
        shard = _shard_name(shard_index, args.num_shards)
        path = shard_root / "shards" / f"{shard}.manifest.json"
        manifest = _load_json(path)
        if (
            manifest.get("status") != SHARD_STATUS
            or manifest.get("protocol_file_sha256") != protocol_sha
            or manifest.get("training_completion_file_sha256") != completion_sha
            or manifest.get("ranking_input_file_sha256") != input_sha
            or manifest.get("scorer_file_sha256") != file_sha256(Path(__file__))
            or int(manifest.get("shard_index", -1)) != shard_index
            or int(manifest.get("num_shards", -1)) != args.num_shards
        ):
            raise ValueError(f"scoring shard manifest drift: {shard}")
        manifests.append((path, manifest))

    merged_outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows: list[dict[str, Any]] = []
        for _, manifest in manifests:
            record = manifest["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"shard lacks output {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"scored shard hash drift: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: int(row["source_row_index"]))
        if len(rows) != len(source):
            raise ValueError(f"merged row-count drift: {key}")
        for index, (row, reference) in enumerate(zip(rows, source, strict=True)):
            _validate_merged_source_row(row, reference, SCORING_MODE, index)
            if (
                row.get("clir_checkpoint_sha256") != run["checkpoint_file_sha256"]
                or row.get("clir_scoring_mode") != SCORING_MODE
                or not math.isfinite(float(row["clir_score"]))
            ):
                raise ValueError(f"merged score drift: {key}/{index}")
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
        "schema_version": "swift-u0-sampler-factorial-v1-scoring-merge",
        "status": MERGE_STATUS,
        "created_at_utc": _utc_now(),
        "protocol_file_sha256": protocol_sha,
        "training_completion_file_sha256": completion_sha,
        "ranking_input_file_sha256": input_sha,
        "scorer_file_sha256": file_sha256(Path(__file__)),
        "num_shards": args.num_shards,
        "shard_manifest_sha256": {
            str(index): file_sha256(path)
            for index, (path, _) in enumerate(manifests)
        },
        "outputs": merged_outputs,
        "hard_math_opened": False,
    }
    atomic_write_json(output_root / "merge_report.json", report)
    print(json.dumps({"status": MERGE_STATUS, "runs": len(merged_outputs), "rows": len(source)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--completion-report", required=True)
    worker.add_argument("--input-jsonl", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--num-workers", type=int, required=True)
    worker.add_argument("--amp-dtype", choices=("none", "bfloat16"), required=True)
    worker.add_argument("--overwrite", action="store_true")
    worker.set_defaults(func=command_worker)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--completion-report", required=True)
    merge.add_argument("--input-jsonl", required=True)
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=command_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
