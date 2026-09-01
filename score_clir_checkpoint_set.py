#!/usr/bin/env python
"""Score one frozen CLIR ranking population with a hash-bound checkpoint set.

The source feature tensor is loaded once per worker and reused across every
checkpoint on that worker.  The separate authorization controls whether the
input is the open tuning split or the one-time sealed confirmation split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from score_clir import atomic_write_jsonl, file_sha256
from score_clir_factorial import (
    _add_global_selections,
    _forward,
    _load_json,
    _load_models,
    _materialize_batch,
    _output_path,
    _require_finite,
    _run_key,
    _shard_name,
    _validate_merged_source_row,
)
from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
    read_jsonl,
)
from src.clir_smoke import atomic_write_json, canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parent
AUTHORIZED_STATUSES = {
    "AUTHORIZED_PRIOR_GATE_TUNING_V1_STAGE_A_SCORING": False,
    "AUTHORIZED_PRIOR_GATE_TUNING_V1_WEIGHT_GRID_SCORING": False,
    "AUTHORIZED_PRIOR_GATE_TUNING_V1_CONFIRMATION_SCORING": True,
}
SHARD_STATUS = "PASS_PRIOR_GATE_TUNING_V1_CHECKPOINT_SET_SCORING_SHARD"
MERGE_STATUS = "PASS_PRIOR_GATE_TUNING_V1_CHECKPOINT_SET_SCORING_MERGE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_confirmation_lock(
    *,
    authorization: Mapping[str, Any],
    completion_path: Path,
    completion_sha: str,
    completion: Mapping[str, Any],
    input_path: Path,
    input_sha: str,
) -> None:
    lock_path = _project_path(authorization["weight_lock_path"])
    if completion_path.resolve() != lock_path:
        raise ValueError("confirmation completion is not the frozen weight lock")
    if completion_sha != authorization["weight_lock_sha256"]:
        raise ValueError("confirmation weight-lock hash drift")
    if (
        completion.get("schema_version")
        != "clir-prior-gate-tuning-v1-confirmation-weight-lock"
        or completion.get("confirmation_opening_allowed") is not True
        or completion.get("no_second_weight_selection_after_confirmation") is not True
    ):
        raise ValueError("confirmation weight lock is inactive or malformed")

    selection_spec = completion.get("weight_selection")
    if not isinstance(selection_spec, Mapping):
        raise ValueError("confirmation weight lock lacks the tuning selection")
    selection_path = _project_path(str(selection_spec["path"]))
    selection_sha = file_sha256(selection_path)
    if (
        selection_sha != selection_spec["file_sha256"]
        or selection_path
        != _project_path(authorization["weight_selection_path"])
        or selection_sha != authorization["weight_selection_sha256"]
    ):
        raise ValueError("confirmation tuning-selection hash drift")
    selection = _load_json(selection_path)
    locked = selection.get("selection", {})
    if (
        selection.get("status")
        != "COMPLETE_PRIOR_GATE_TUNING_V1_DIRECT_WEIGHT_SELECTION"
        or selection.get("confirmation_outcomes_opened") is not False
        or locked.get("selected_cell") != selection_spec.get("selected_cell")
        or float(locked.get("selected_direct_weight"))
        != float(selection_spec.get("selected_direct_weight"))
        or float(locked.get("gate_prior_weight"))
        != float(selection_spec.get("gate_prior_weight"))
    ):
        raise ValueError("confirmation lock does not match the tuning decision")

    sealed = completion.get("sealed_confirmation")
    if not isinstance(sealed, Mapping):
        raise ValueError("confirmation weight lock lacks the sealed population")
    if (
        input_path.resolve() != _project_path(str(sealed["path"]))
        or input_sha != sealed["file_sha256"]
        or int(sealed["rows"]) != int(authorization["ranking_rows"])
        or int(sealed["queries"]) != int(authorization["ranking_queries"])
        or int(sealed["candidates_per_query"])
        != int(authorization["candidates_per_query"])
    ):
        raise ValueError("confirmation input differs from the locked sealed population")
    if (
        completion.get("cells") != authorization["cells"]
        or completion.get("seeds") != authorization["seeds"]
        or int(completion.get("run_count", -1))
        != int(authorization["run_count"])
    ):
        raise ValueError("confirmation checkpoint grid differs from the weight lock")


def _load_bound_contract(
    *,
    authorization_path: Path,
    completion_path: Path,
    input_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    authorization = _load_json(authorization_path)
    status = str(authorization.get("status"))
    expected_confirmation = AUTHORIZED_STATUSES.get(status)
    if expected_confirmation is None:
        raise ValueError("checkpoint-set scoring authorization has not passed")
    if bool(authorization.get("confirmation_scoring_allowed")) != expected_confirmation:
        raise ValueError("checkpoint-set confirmation authorization drift")
    if authorization.get("scorer_sha256") != file_sha256(__file__):
        raise ValueError("checkpoint-set authorization binds another scorer")
    expected_completion = _project_path(authorization["training_completion_path"])
    expected_input = _project_path(authorization["ranking_input_path"])
    if completion_path.resolve() != expected_completion:
        raise ValueError("checkpoint-set completion path drift")
    if input_path.resolve() != expected_input:
        raise ValueError("checkpoint-set ranking input path drift")
    completion_sha = file_sha256(completion_path)
    input_sha = file_sha256(input_path)
    if completion_sha != authorization["training_completion_sha256"]:
        raise ValueError("checkpoint-set completion hash drift")
    if input_sha != authorization["ranking_input_sha256"]:
        raise ValueError("checkpoint-set ranking population hash drift")

    completion = _load_json(completion_path)
    if completion.get("status") != authorization["training_completion_status"]:
        raise ValueError("checkpoint-set completion status drift")
    if expected_confirmation:
        _validate_confirmation_lock(
            authorization=authorization,
            completion_path=completion_path,
            completion_sha=completion_sha,
            completion=completion,
            input_path=input_path,
            input_sha=input_sha,
        )
    runs = completion.get("runs")
    cells = tuple(str(value) for value in authorization["cells"])
    seeds = tuple(int(value) for value in authorization["seeds"])
    expected_identities = {(cell, seed) for cell in cells for seed in seeds}
    if (
        not isinstance(runs, list)
        or len(runs) != int(authorization["run_count"])
        or len(runs) != len(expected_identities)
    ):
        raise ValueError("checkpoint-set run-count drift")
    identities = {(str(run["cell"]), int(run["seed"])) for run in runs}
    if identities != expected_identities:
        raise ValueError("checkpoint-set cell/seed grid drift")

    source = read_jsonl(input_path)
    query_count = len({str(row["query_id"]) for row in source})
    if (
        len(source) != int(authorization["ranking_rows"])
        or query_count != int(authorization["ranking_queries"])
    ):
        raise ValueError("checkpoint-set ranking inventory drift")
    if any(
        bool(row.get("sealed_until_weight_lock")) != expected_confirmation
        for row in source
    ):
        raise ValueError("checkpoint-set input sealing marker drift")
    candidate_count = int(authorization["candidates_per_query"])
    counts: dict[str, int] = {}
    for row in source:
        query_id = str(row["query_id"])
        counts[query_id] = counts.get(query_id, 0) + 1
    if set(counts.values()) != {candidate_count}:
        raise ValueError("checkpoint-set candidate axis drift")
    return (
        authorization,
        file_sha256(authorization_path),
        completion,
        completion_sha,
        source,
        input_sha,
    )


@torch.no_grad()
def command_worker(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    authorization_path = Path(args.authorization).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    (
        authorization,
        authorization_sha,
        completion,
        completion_sha,
        source,
        input_sha,
    ) = _load_bound_contract(
        authorization_path=authorization_path,
        completion_path=completion_path,
        input_path=input_path,
    )
    output_root = Path(args.output_root).resolve()
    runtime = authorization["runtime"]
    expected_root = _project_path(runtime["shard_output_root"])
    if output_root != expected_root:
        raise ValueError("checkpoint-set shard output root drift")
    if (
        args.num_shards != int(runtime["num_shards"])
        or args.batch_size != int(runtime["batch_size"])
        or args.num_workers != int(runtime["num_workers"])
        or bool(args.pin_memory) != bool(runtime["pin_memory"])
        or args.amp_dtype != runtime["amp_dtype"]
        or args.feature_root is not None
    ):
        raise ValueError("checkpoint-set worker runtime contract drift")
    shard_name = _shard_name(args.shard_index, args.num_shards)
    manifest_path = output_root / "shards" / f"{shard_name}.manifest.json"
    targets = [
        _output_path(output_root, run, shard_name) for run in completion["runs"]
    ]
    existing = [path for path in (manifest_path, *targets) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"scoring shard output exists: {existing[0]}")

    dataset = CLIRTrajectoryDataset(input_path, feature_root=args.feature_root)
    if len(dataset) != len(source):
        raise ValueError("checkpoint-set dataset/source row-count drift")
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    if not indices:
        raise ValueError("scoring shard contains no source rows")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    loaded = _load_models(completion["runs"], device)
    buffers = {_run_key(run): [] for run, _ in loaded}
    for raw_batch in loader:
        row_indices = [int(value) for value in raw_batch["row_index"].tolist()]
        batch = move_batch_to_device(raw_batch, device)
        if args.amp_dtype == "none":
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].float()
        for run, model in loaded:
            outputs = _forward(model, batch, amp_dtype=args.amp_dtype)
            _require_finite(outputs, "scalar")
            buffers[_run_key(run)].extend(
                _materialize_batch(
                    mode="scalar",
                    dataset_rows=dataset.rows,
                    row_indices=row_indices,
                    batch=batch,
                    outputs=outputs,
                    checkpoint_sha256=str(run["checkpoint_file_sha256"]),
                    onset_threshold=0.5,
                )
            )
            del outputs

    outputs_manifest: dict[str, Any] = {}
    for run, _ in loaded:
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
        "schema_version": "clir-prior-gate-tuning-v1-checkpoint-set-shard",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "authorization_file_sha256": authorization_sha,
        "authorization_status": authorization["status"],
        "input_jsonl_sha256": input_sha,
        "input_rows": len(dataset),
        "completion_report_sha256": completion_sha,
        "scorer_sha256": file_sha256(__file__),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_rows": len(indices),
        "source_indices_sha256": canonical_sha256(indices),
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "batch_size": args.batch_size,
        "outputs": outputs_manifest,
        "confirmation_scoring": bool(
            authorization["confirmation_scoring_allowed"]
        ),
    }
    atomic_write_json(manifest_path, report)
    print(json.dumps({"status": SHARD_STATUS, "shard": shard_name, "rows": len(indices)}))


def command_merge(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    authorization_path = Path(args.authorization).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_path = Path(args.input_jsonl).resolve()
    (
        authorization,
        authorization_sha,
        completion,
        completion_sha,
        source_rows,
        input_sha,
    ) = _load_bound_contract(
        authorization_path=authorization_path,
        completion_path=completion_path,
        input_path=input_path,
    )
    shard_root = Path(args.shard_root).resolve()
    output_root = Path(args.output_root).resolve()
    runtime = authorization["runtime"]
    if (
        args.num_shards != int(runtime["num_shards"])
        or shard_root != _project_path(runtime["shard_output_root"])
        or output_root != _project_path(runtime["merged_output_root"])
    ):
        raise ValueError("checkpoint-set merge runtime contract drift")
    target_report = output_root / "merge_report.json"
    if target_report.exists() and not args.overwrite:
        raise FileExistsError(f"score merge report exists: {target_report}")
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for shard_index in range(args.num_shards):
        shard_name = _shard_name(shard_index, args.num_shards)
        path = shard_root / "shards" / f"{shard_name}.manifest.json"
        manifest = _load_json(path)
        if (
            manifest.get("status") != SHARD_STATUS
            or manifest.get("authorization_file_sha256") != authorization_sha
            or manifest.get("input_jsonl_sha256") != input_sha
            or manifest.get("completion_report_sha256") != completion_sha
            or manifest.get("scorer_sha256") != file_sha256(__file__)
            or int(manifest.get("shard_index", -1)) != shard_index
            or int(manifest.get("num_shards", -1)) != args.num_shards
        ):
            raise ValueError(f"stale or invalid score shard: {path}")
        manifests.append((path, manifest))

    merged_outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows: list[dict[str, Any]] = []
        for manifest_path, manifest in manifests:
            record = manifest["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"score shard lacks {key}: {manifest_path}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"scored shard hash drift: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: int(row["source_row_index"]))
        if [int(row["source_row_index"]) for row in rows] != list(
            range(len(source_rows))
        ):
            raise ValueError(f"merged shards do not cover source rows for {key}")
        for index, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
            _validate_merged_source_row(row, source, "scalar", index)
            if row.get("clir_checkpoint_sha256") != run["checkpoint_file_sha256"]:
                raise ValueError(f"checkpoint identity drift in {key}")
            if not math.isfinite(float(row["clir_score"])):
                raise FloatingPointError(f"non-finite score in {key}")
        _add_global_selections(rows)
        target = (
            output_root
            / str(run["cell"])
            / f"seed-{int(run['seed'])}"
            / "scored.jsonl"
        )
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
        "schema_version": "clir-prior-gate-tuning-v1-checkpoint-set-merge",
        "status": MERGE_STATUS,
        "created_at_utc": _utc_now(),
        "authorization_file_sha256": authorization_sha,
        "authorization_status": authorization["status"],
        "input_jsonl_sha256": input_sha,
        "input_rows": len(source_rows),
        "completion_report_sha256": completion_sha,
        "scorer_sha256": file_sha256(__file__),
        "num_shards": args.num_shards,
        "shard_manifest_sha256": {
            str(index): file_sha256(path)
            for index, (path, _) in enumerate(manifests)
        },
        "outputs": merged_outputs,
        "confirmation_scoring": bool(
            authorization["confirmation_scoring_allowed"]
        ),
    }
    atomic_write_json(target_report, report)
    print(json.dumps({"status": MERGE_STATUS, "runs": len(merged_outputs), "rows": len(source_rows)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--authorization", required=True)
    worker.add_argument("--input-jsonl", required=True)
    worker.add_argument("--completion-report", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--feature-root", default=None)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--batch-size", type=int, default=2)
    worker.add_argument("--num-workers", type=int, default=0)
    worker.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False
    )
    worker.add_argument("--amp-dtype", choices=("none", "bfloat16"), default="bfloat16")
    worker.add_argument("--overwrite", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--authorization", required=True)
    merge.add_argument("--input-jsonl", required=True)
    merge.add_argument("--completion-report", required=True)
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    merge.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "worker":
        command_worker(args)
    elif args.command == "merge":
        command_merge(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
