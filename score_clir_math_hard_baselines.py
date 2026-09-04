#!/usr/bin/env python
"""Score the frozen SWIFT/U0 sampler controls on MATH-hard v1.

The protected query list belongs to ``math_hard_eval_v1``.  A separate,
pre-rollout model-set addendum binds the extra checkpoints so the original
57-cell CLIR grid and the completed architecture-by-sampler controls can be
compared on exactly the same 500 x 16 candidate population.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from prepare_clir_math_hard_eval import load_protocol as load_base_protocol
from score_clir import atomic_write_jsonl
from score_clir_factorial import (
    _add_global_selections,
    _base_scalar_row,
    _shard_name,
    _validate_merged_source_row,
)
from src.clir_data import CLIRTrajectoryDataset, clir_collate, move_batch_to_device
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
DEFAULT_ADDENDUM = (
    PROJECT_ROOT / "configs/math_hard_eval_v1/model_set_addendum_v2.json"
)
DEFAULT_FEATURES = (
    PROJECT_ROOT
    / "run_artifacts/math_hard_eval_v1/features_v1/final/tuning_features.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "run_artifacts/math_hard_eval_v1/ranking/baseline_controls"
)
ADDENDUM_SCHEMA = "clir-math-hard-model-set-addendum-v2"
ADDENDUM_STATUS = "AUTHORIZED_MATH_HARD_MODEL_SET_V2_BEFORE_FIRST_ROLLOUT"
SHARD_STATUS = "PASS_MATH_HARD_BASELINE_CONTROLS_SCORING_SHARD"
MERGE_STATUS = "PASS_MATH_HARD_BASELINE_CONTROLS_SCORING_MERGE"
EXPECTED_CELLS = ("u0_random", "swift_random", "swift_grouped")
EXPECTED_SOURCE_CELLS = {
    "u0_random": "u0_random",
    "swift_random": "swift_official",
    "swift_grouped": "swift_grouped",
}
SCORING_MODE = "scalar_only"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_clean_branch(addendum: Mapping[str, Any]) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    state = {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }
    runtime = addendum["runtime"]
    if (
        runtime.get("require_clean_committed_code") is not True
        or state["dirty"]
        or state["branch"] != runtime["required_branch"]
    ):
        raise RuntimeError("protected baseline scoring requires a clean committed branch")
    return state


def _run_key(run: Mapping[str, Any]) -> str:
    return f"{run['cell']}/seed-{int(run['seed'])}"


def _output_path(root: Path, run: Mapping[str, Any], shard: str) -> Path:
    return (
        root
        / "shards"
        / str(run["cell"])
        / f"seed-{int(run['seed'])}"
        / f"{shard}.jsonl"
    )


def load_addendum(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Load and structurally validate the pre-result model-set addendum."""

    addendum_path = Path(path).resolve()
    addendum = _load_json(addendum_path)
    if addendum.get("schema_version") != ADDENDUM_SCHEMA:
        raise ValueError("unexpected MATH-hard model-set addendum schema")
    if addendum.get("status") != ADDENDUM_STATUS:
        raise ValueError("MATH-hard model-set addendum is not authorized")
    boundary = addendum.get("evidence_boundary", {})
    required_true = (
        "test_questions_already_accessed_for_deterministic_selection",
        "no_rollout_correctness_or_reward_scores_opened_before_this_addendum",
        "model_set_locked_before_first_rollout",
        "no_post_result_checkpoint_epoch_weight_subset_or_seed_selection",
    )
    if any(boundary.get(key) is not True for key in required_true):
        raise ValueError("MATH-hard model-set evidence boundary is incomplete")

    seeds = tuple(int(value) for value in addendum["model_set"]["seeds"])
    runs = addendum["model_set"].get("runs")
    expected = {(cell, seed) for cell in EXPECTED_CELLS for seed in seeds}
    if (
        seeds != (42, 43, 44)
        or not isinstance(runs, list)
        or len(runs) != len(expected)
        or {(str(run["cell"]), int(run["seed"])) for run in runs} != expected
        or int(addendum["model_set"].get("additional_checkpoint_count", -1))
        != len(expected)
        or int(addendum["model_set"].get("total_checkpoint_count", -1)) != 66
    ):
        raise ValueError("MATH-hard additional checkpoint grid drift")
    for run in runs:
        cell = str(run["cell"])
        expected_kind = "u0_clir" if cell == "u0_random" else "plain_swift"
        if (
            run.get("model_kind") != expected_kind
            or int(run.get("epoch", -1)) != 3
            or run.get("source_cell") != EXPECTED_SOURCE_CELLS[cell]
            or not isinstance(run.get("checkpoint_path"), str)
            or not isinstance(run.get("checkpoint_file_sha256"), str)
        ):
            raise ValueError(f"invalid bound model row: {_run_key(run)}")
        if verify_files and file_sha256(_resolve(run["checkpoint_path"])) != run[
            "checkpoint_file_sha256"
        ]:
            raise ValueError(f"checkpoint hash drift: {_run_key(run)}")

    base = addendum["frozen_parent"]["math_hard_protocol"]
    base_path = _resolve(base["path"])
    if verify_files and file_sha256(base_path) != base["file_sha256"]:
        raise ValueError("base MATH-hard protocol hash drift")
    if verify_files:
        base_protocol = load_base_protocol(base_path)
        if int(base_protocol["source"]["total_queries"]) != 500:
            raise ValueError("base MATH-hard query count drift")
        registry = addendum["frozen_parent"]["pre_rollout_registry"]
        if file_sha256(_resolve(registry["path"])) != registry["file_sha256"]:
            raise ValueError("pre-rollout registry hash drift")

        # Each embedded run must be present in its bound, immutable completion.
        source_runs: dict[str, Mapping[str, Any]] = {}
        for source in addendum["frozen_parent"]["training_completions"]:
            source_path = _resolve(source["path"])
            if file_sha256(source_path) != source["file_sha256"]:
                raise ValueError(f"training completion hash drift: {source_path}")
            completion = _load_json(source_path)
            if completion.get("status") != source["status"]:
                raise ValueError(f"training completion status drift: {source_path}")
            for raw in completion.get("runs", []):
                key = f"{raw.get('cell')}/seed-{int(raw.get('seed', -1))}"
                source_runs[key] = raw
        for run in runs:
            source_key = f"{run['source_cell']}/seed-{int(run['seed'])}"
            raw = source_runs.get(source_key)
            if not isinstance(raw, Mapping):
                raise ValueError(f"bound completion lacks {_run_key(run)}")
            if (
                int(raw.get("epoch", raw.get("completed_epoch", -1))) != 3
                or _resolve(str(raw["checkpoint_path"]))
                != _resolve(str(run["checkpoint_path"]))
                or raw.get("checkpoint_file_sha256")
                != run["checkpoint_file_sha256"]
            ):
                raise ValueError(f"embedded checkpoint differs from completion: {_run_key(run)}")
        for label, spec in addendum["implementation"].items():
            if file_sha256(_resolve(spec["path"])) != spec["file_sha256"]:
                raise ValueError(f"bound implementation hash drift: {label}")
    return addendum


def _load_contract(
    addendum_path: Path, input_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    addendum = load_addendum(addendum_path, verify_files=True)
    _require_clean_branch(addendum)
    addendum_sha = file_sha256(addendum_path)
    base_protocol = load_base_protocol(
        _resolve(addendum["frozen_parent"]["math_hard_protocol"]["path"])
    )
    expected_input = (
        _resolve(base_protocol["runtime"]["output_root"])
        / "features_v1/final/tuning_features.jsonl"
    )
    if input_path.resolve() != expected_input:
        raise ValueError("protected feature-manifest path drift")
    feature_completion_path = expected_input.parent / "completion.json"
    feature_completion = _load_json(feature_completion_path)
    if feature_completion.get("status") != "PASS_GATE_TUNING_V1_SELECTED_FEATURES":
        raise ValueError("protected feature extraction is incomplete")
    manifest = feature_completion.get("tuning_manifest", {})
    input_sha = file_sha256(input_path)
    if input_sha != manifest.get("file_sha256"):
        raise ValueError("protected feature-manifest hash drift")
    source = read_jsonl(input_path)
    population = validate_rollout_population(
        source, candidate_count=int(base_protocol["generation"]["candidate_count"])
    )
    if len(source) != 8000 or int(population["queries"]) != 500:
        raise ValueError("protected feature population drift")
    completion = {
        "status": addendum["status"],
        "runs": [dict(run) for run in addendum["model_set"]["runs"]],
    }
    return addendum, addendum_sha, completion, addendum_sha, source, input_sha


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
    for raw_run in sorted(runs, key=lambda row: (str(row["cell"]), int(row["seed"]))):
        run = dict(raw_run)
        checkpoint_path = _resolve(run["checkpoint_path"])
        if file_sha256(checkpoint_path) != run["checkpoint_file_sha256"]:
            raise ValueError(f"checkpoint hash drift: {_run_key(run)}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        completed_epoch = int(checkpoint.get("completed_epoch", run["epoch"]))
        if completed_epoch != int(run["epoch"]):
            raise ValueError(f"checkpoint epoch drift: {_run_key(run)}")
        if run["model_kind"] == "u0_clir":
            config = checkpoint.get("model_config")
            if not isinstance(config, Mapping):
                raise ValueError(f"U0 checkpoint lacks model config: {_run_key(run)}")
            model = ConsistencyLocalizedReward(RewardConfig(**dict(config))).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            u0.append((run, model))
        elif run["model_kind"] == "plain_swift":
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
            raise ValueError(f"unknown model kind: {run['model_kind']}")
    if len(u0) != 3 or len(swift_runs) != 6:
        raise ValueError("expected three U0-random and six plain-SWIFT checkpoints")
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
        raise FloatingPointError("non-finite U0 scores")
    return scores


@torch.no_grad()
def command_worker(args: argparse.Namespace) -> None:
    addendum_path = Path(args.addendum).resolve()
    input_path = Path(args.input_jsonl).resolve()
    addendum, addendum_sha, completion, completion_sha, source, input_sha = (
        _load_contract(addendum_path, input_path)
    )
    runtime = addendum["runtime"]
    if (
        int(args.num_shards) != int(runtime["scoring_shards"])
        or int(args.batch_size) != int(runtime["scoring_batch_size"])
        or int(args.num_workers) != int(runtime["scoring_num_workers"])
        or args.amp_dtype != runtime["scoring_amp_dtype"]
        or not 0 <= int(args.shard_index) < int(args.num_shards)
    ):
        raise ValueError("protected baseline scoring runtime drift")
    output_root = Path(args.output_root).resolve()
    if output_root != _resolve(runtime["shard_output_root"]):
        raise ValueError("protected baseline shard output-root drift")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each protected scoring worker must see exactly one GPU")

    shard = _shard_name(args.shard_index, args.num_shards)
    manifest_path = output_root / "shards" / f"{shard}.manifest.json"
    targets = [_output_path(output_root, run, shard) for run in completion["runs"]]
    existing = [path for path in (manifest_path, *targets) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"protected baseline output exists: {existing[0]}")

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
        raise RuntimeError("full-width protected scoring requires CUDA")
    u0_models, swift_runs, swift_states = _load_models(completion["runs"], device)
    buffers = {_run_key(run): [] for run in completion["runs"]}

    for raw_batch in loader:
        row_indices = [int(value) for value in raw_batch["row_index"].tolist()]
        batch = move_batch_to_device(raw_batch, device)
        for run, model in u0_models:
            scores = _u0_scores(model, batch, args.amp_dtype)
            for offset, source_index in enumerate(row_indices):
                row = _base_scalar_row(
                    dataset.rows[source_index],
                    source_index,
                    str(run["checkpoint_file_sha256"]),
                )
                row["clir_score"] = float(scores[offset].cpu())
                buffers[_run_key(run)].append(row)
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
            raise FloatingPointError("invalid protected stacked-SWIFT scores")
        for position, run in enumerate(swift_runs):
            for offset, source_index in enumerate(row_indices):
                row = _base_scalar_row(
                    dataset.rows[source_index],
                    source_index,
                    str(run["checkpoint_file_sha256"]),
                )
                row["clir_score"] = float(swift_scores[offset, position].cpu())
                buffers[_run_key(run)].append(row)
        del batch, swift_scores

    outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows = buffers[key]
        if [int(row["source_row_index"]) for row in rows] != indices:
            raise ValueError(f"protected source-order drift for {key}")
        target = _output_path(output_root, run, shard)
        atomic_write_jsonl(target, rows)
        outputs[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }
    report = {
        "schema_version": "clir-math-hard-baseline-controls-scoring-shard-v2",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "addendum_file_sha256": addendum_sha,
        "model_registry_file_sha256": completion_sha,
        "ranking_input_file_sha256": input_sha,
        "scorer_file_sha256": file_sha256(Path(__file__)),
        "code": _require_clean_branch(addendum),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_indices_sha256": canonical_sha256(indices),
        "source_rows": len(indices),
        "batch_size": args.batch_size,
        "amp_dtype": args.amp_dtype,
        "single_feature_pass_for_all_nine_checkpoints": True,
        "protected_rollout_correctness_and_scores_opened": True,
        "outputs": outputs,
    }
    atomic_write_json(manifest_path, report)
    print(json.dumps({"status": SHARD_STATUS, "shard": shard, "rows": len(indices)}))


def command_merge(args: argparse.Namespace) -> None:
    addendum_path = Path(args.addendum).resolve()
    input_path = Path(args.input_jsonl).resolve()
    addendum, addendum_sha, completion, completion_sha, source, input_sha = (
        _load_contract(addendum_path, input_path)
    )
    runtime = addendum["runtime"]
    shard_root = Path(args.shard_root).resolve()
    output_root = Path(args.output_root).resolve()
    if (
        shard_root != _resolve(runtime["shard_output_root"])
        or output_root != _resolve(runtime["merged_output_root"])
        or int(args.num_shards) != int(runtime["scoring_shards"])
    ):
        raise ValueError("protected baseline merge runtime drift")
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for shard_index in range(args.num_shards):
        shard = _shard_name(shard_index, args.num_shards)
        path = shard_root / "shards" / f"{shard}.manifest.json"
        manifest = _load_json(path)
        if (
            manifest.get("status") != SHARD_STATUS
            or manifest.get("addendum_file_sha256") != addendum_sha
            or manifest.get("model_registry_file_sha256") != completion_sha
            or manifest.get("ranking_input_file_sha256") != input_sha
            or manifest.get("scorer_file_sha256") != file_sha256(Path(__file__))
            or int(manifest.get("shard_index", -1)) != shard_index
            or int(manifest.get("num_shards", -1)) != args.num_shards
        ):
            raise ValueError(f"protected baseline shard manifest drift: {shard}")
        manifests.append((path, manifest))

    merged_outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows: list[dict[str, Any]] = []
        for _, manifest in manifests:
            record = manifest["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"protected shard lacks {key}")
            path = Path(str(record["path"]))
            if file_sha256(path) != record["file_sha256"]:
                raise ValueError(f"protected score-shard hash drift: {path}")
            rows.extend(read_jsonl(path))
        rows.sort(key=lambda row: int(row["source_row_index"]))
        if len(rows) != len(source):
            raise ValueError(f"protected merged row-count drift: {key}")
        for index, (row, reference) in enumerate(zip(rows, source, strict=True)):
            _validate_merged_source_row(row, reference, SCORING_MODE, index)
            if (
                row.get("clir_checkpoint_sha256")
                != run["checkpoint_file_sha256"]
                or row.get("clir_scoring_mode") != SCORING_MODE
                or not math.isfinite(float(row["clir_score"]))
            ):
                raise ValueError(f"protected merged score drift: {key}/{index}")
        _add_global_selections(rows)
        target = output_root / str(run["cell"]) / f"seed-{int(run['seed'])}" / "scored.jsonl"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"protected merged score exists: {target}")
        atomic_write_jsonl(target, rows)
        merged_outputs[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }
    report = {
        "schema_version": "clir-math-hard-baseline-controls-scoring-merge-v2",
        "status": MERGE_STATUS,
        "created_at_utc": _utc_now(),
        "addendum_file_sha256": addendum_sha,
        "model_registry_file_sha256": completion_sha,
        "ranking_input_file_sha256": input_sha,
        "scorer_file_sha256": file_sha256(Path(__file__)),
        "code": _require_clean_branch(addendum),
        "num_shards": args.num_shards,
        "shard_manifest_sha256": {
            str(index): file_sha256(path)
            for index, (path, _) in enumerate(manifests)
        },
        "outputs": merged_outputs,
        "protected_rollout_correctness_and_scores_opened": True,
    }
    atomic_write_json(output_root / "merge_report.json", report)
    print(
        json.dumps(
            {"status": MERGE_STATUS, "runs": len(merged_outputs), "rows": len(source)}
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", default=str(DEFAULT_ADDENDUM))
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--input-jsonl", default=str(DEFAULT_FEATURES))
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
    merge.add_argument("--input-jsonl", default=str(DEFAULT_FEATURES))
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
