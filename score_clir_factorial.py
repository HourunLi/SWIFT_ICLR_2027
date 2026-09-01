#!/usr/bin/env python
"""Score one frozen feature population with every checkpoint in a factorial.

The worker shards rows, not checkpoints.  Consequently each hidden-state payload
is read by exactly one worker and then reused across all checkpoints on that
worker's GPU.  The merge command restores frozen source order and, for ranking
scores, computes stable whole-population Best-of-N selections.
"""

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

from evaluate_clir import atomic_write_json
from score_clir import atomic_write_jsonl, file_sha256
from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
    read_jsonl,
)
from src.clir_smoke import canonical_sha256
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    infer_pseudo_onsets,
    path_hallucination_probability,
    path_no_hallucination_log_probability,
)


COMPLETION_STATUS = "PASS_THREE_MODULE_COMPLETE_2X2X2_24_RUN_TRAINING"
RANKING_AUTHORIZATION_STATUS = "AUTHORIZED_THREE_MODULE_FACTORIAL_RANKING_V1"
MODES = ("consistency", "full", "scalar")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(__file__).parent / path).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_source(
    input_jsonl: Path, expected_input_sha256: str | None
) -> str:
    observed = file_sha256(input_jsonl)
    if expected_input_sha256 and observed != expected_input_sha256:
        raise ValueError("input JSONL hash does not match the frozen population")
    return observed


def _load_completion(path: Path, *, mode: str) -> tuple[dict[str, Any], str]:
    completion = _load_json(path)
    if completion.get("status") != COMPLETION_STATUS:
        raise ValueError("the complete 24-run training validation has not passed")
    if mode != "scalar" and completion.get("mechanism_evaluation_allowed") is not True:
        raise ValueError("mechanism evaluation is not authorized")
    runs = completion.get("runs")
    if not isinstance(runs, list) or len(runs) != 24:
        raise ValueError("training completion must contain exactly 24 runs")
    identities = {(str(run["cell"]), int(run["seed"])) for run in runs}
    if len(identities) != 24:
        raise ValueError("training completion contains duplicate run identities")
    return completion, file_sha256(path)


def _validate_ranking_authorization(
    path: Path | None,
    *,
    completion_sha256: str,
    input_sha256: str,
) -> str | None:
    if path is None:
        raise ValueError("scalar ranking scoring requires --ranking-authorization")
    payload = _load_json(path)
    if payload.get("status") != RANKING_AUTHORIZATION_STATUS:
        raise ValueError("ranking authorization has not passed")
    if payload.get("training_completion_sha256") != completion_sha256:
        raise ValueError("ranking authorization binds a different training completion")
    if payload.get("ranking_input_sha256") != input_sha256:
        raise ValueError("ranking authorization binds a different ranking population")
    expected_scorer = payload.get("scorer_sha256")
    if expected_scorer != file_sha256(__file__):
        raise ValueError("ranking authorization binds a different scorer implementation")
    return file_sha256(path)


def _autocast(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 autocast is supported only on CPU or CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _load_models(
    runs: Sequence[Mapping[str, Any]], device: torch.device
) -> list[tuple[dict[str, Any], ConsistencyLocalizedReward]]:
    loaded: list[tuple[dict[str, Any], ConsistencyLocalizedReward]] = []
    for source in runs:
        run = dict(source)
        checkpoint_path = _project_path(str(run["checkpoint_path"]))
        if file_sha256(checkpoint_path) != run["checkpoint_file_sha256"]:
            raise ValueError(
                f"checkpoint hash drift for {run['cell']}/seed-{run['seed']}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("completed_epoch", -1)) != int(run["completed_epoch"]):
            raise ValueError("checkpoint epoch drift")
        model_values = checkpoint.get("model_config")
        if not isinstance(model_values, Mapping):
            raise ValueError("checkpoint lacks model_config")
        factors = tuple(int(value) for value in run["factors"])
        observed = (
            int(float(model_values["consistency_weight"]) == 1.0),
            int(float(model_values["hallucination_weight"]) == 1.0),
            int(float(model_values["prior_weight"]) == 1.0),
        )
        if observed != factors:
            raise ValueError(
                f"checkpoint factor drift for {run['cell']}/seed-{run['seed']}"
            )
        model = ConsistencyLocalizedReward(RewardConfig(**model_values)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loaded.append((run, model))
        del checkpoint
    return loaded


def _run_key(run: Mapping[str, Any]) -> str:
    return f"{run['cell']}/seed-{int(run['seed'])}"


def _shard_name(shard_index: int, num_shards: int) -> str:
    return f"shard-{shard_index:03d}-of-{num_shards:03d}"


def _output_path(root: Path, run: Mapping[str, Any], shard_name: str) -> Path:
    return root / str(run["cell"]) / f"seed-{int(run['seed'])}" / f"{shard_name}.jsonl"


def _forward(
    model: ConsistencyLocalizedReward,
    batch: Mapping[str, Any],
    *,
    amp_dtype: str,
) -> dict[str, torch.Tensor]:
    device = batch["hidden_states"].device
    with _autocast(device, amp_dtype):
        return model(
            batch["hidden_states"],
            mask=batch["mask"],
            condition_states=batch.get("condition_states"),
            condition_mask=batch.get("condition_mask"),
            condition_embedding=batch.get("condition_embedding"),
            condition_embedding_mask=batch.get("condition_embedding_mask"),
        )


def _require_finite(outputs: Mapping[str, torch.Tensor], mode: str) -> None:
    keys = (
        ("scores", "representations")
        if mode == "consistency"
        else (
            "scores",
            "token_rewards",
            "token_values",
            "gates",
            "hallucination_logits",
            "condition_relevance",
            "key_prior_logits",
            "complete_prior_logits",
            "key_prior",
            "complete_prior",
        )
    )
    bad = [key for key in keys if not torch.isfinite(outputs[key]).all()]
    if bad:
        raise FloatingPointError("non-finite scoring outputs: " + ", ".join(bad))


def _base_scalar_row(
    source: Mapping[str, Any], source_row_index: int, checkpoint_sha256: str
) -> dict[str, Any]:
    required = ("id", "query_id", "candidate_index", "correctness")
    if any(key not in source for key in required):
        raise ValueError("ranking source row lacks a required compact field")
    return {
        "source_row_index": source_row_index,
        "id": source["id"],
        "query_id": source["query_id"],
        "candidate_index": source["candidate_index"],
        "correctness": source["correctness"],
        "clir_checkpoint_sha256": checkpoint_sha256,
        "clir_scoring_mode": "scalar_only",
    }


def _materialize_batch(
    *,
    mode: str,
    dataset_rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    batch: Mapping[str, Any],
    outputs: Mapping[str, torch.Tensor],
    checkpoint_sha256: str,
    onset_threshold: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    path_probs = path_logs = pseudo_onsets = None
    if mode == "full":
        path_probs = path_hallucination_probability(
            outputs["hallucination_logits"], outputs["mask"]
        )
        path_logs = path_no_hallucination_log_probability(
            outputs["hallucination_logits"], outputs["mask"]
        )
        pseudo_onsets = infer_pseudo_onsets(
            outputs["hallucination_logits"],
            outputs["mask"],
            threshold=onset_threshold,
        )
    for offset, source_row_index in enumerate(row_indices):
        source = dataset_rows[source_row_index]
        if mode == "scalar":
            row = _base_scalar_row(source, source_row_index, checkpoint_sha256)
        elif mode == "consistency":
            row = {
                "source_row_index": source_row_index,
                "id": source["id"],
                "query_id": source["query_id"],
                "clir_checkpoint_sha256": checkpoint_sha256,
                "clir_scoring_mode": "consistency",
                "clir_representation": [
                    float(value)
                    for value in outputs["representations"][offset]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                ],
            }
        else:
            row = dict(source)
            row.update(
                {
                    "source_row_index": source_row_index,
                    "clir_checkpoint_sha256": checkpoint_sha256,
                    "clir_scoring_mode": "full",
                }
            )
        row["clir_score"] = float(outputs["scores"][offset].detach().float().cpu())
        if mode != "full":
            result.append(row)
            continue

        valid_length = int(batch["mask"][offset].sum().detach().cpu())
        gate_attention = outputs["gates"][offset] / outputs["gates"][offset].sum().clamp_min(1e-8)
        prior_alignment = torch.sum(gate_attention * outputs["fused_prior"][offset])
        prior_gate_squared_l2 = torch.sum(
            (gate_attention - outputs["fused_prior"][offset]).pow(2)
        )
        row.update(
            {
                "clir_path_hallucination_prob": float(path_probs[offset].detach().cpu()),
                "clir_path_no_hallucination_log_prob": float(path_logs[offset].detach().cpu()),
                "clir_pseudo_onset": int(pseudo_onsets[offset].detach().cpu()),
                "clir_mean_gate": float(
                    outputs["gates"][offset, :valid_length].mean().detach().cpu()
                ),
                "clir_prior_gate_alignment": float(prior_alignment.detach().cpu()),
                "clir_prior_gate_squared_l2": float(
                    prior_gate_squared_l2.detach().cpu()
                ),
                "clir_condition_relevance": [
                    float(value)
                    for value in outputs["condition_relevance"][offset, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_gate_attention": [
                    float(value)
                    for value in gate_attention[:valid_length].detach().cpu().tolist()
                ],
                "clir_key_prior": [
                    float(value)
                    for value in outputs["key_prior"][offset, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_complete_prior": [
                    float(value)
                    for value in outputs["complete_prior"][offset, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_hallucination_prob": [
                    float(value)
                    for value in torch.sigmoid(
                        outputs["hallucination_logits"][offset, :valid_length]
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_token_reward": [
                    float(value)
                    for value in outputs["token_rewards"][offset, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_token_value": [
                    float(value)
                    for value in outputs["token_values"][offset, :valid_length]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_key_prior_membership": [
                    float(value)
                    for value in torch.sigmoid(
                        outputs["key_prior_logits"][offset, :valid_length]
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "clir_complete_prior_membership": [
                    float(value)
                    for value in torch.sigmoid(
                        outputs["complete_prior_logits"][offset, :valid_length]
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ],
            }
        )
        result.append(row)
    return result


@torch.no_grad()
def command_worker(args: argparse.Namespace) -> None:
    if args.mode not in MODES:
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise ValueError("onset threshold must be in [0, 1]")

    input_path = Path(args.input_jsonl).resolve()
    completion_path = Path(args.completion_report).resolve()
    input_sha256 = _validate_source(input_path, args.expected_input_sha256)
    completion, completion_sha256 = _load_completion(completion_path, mode=args.mode)
    ranking_authorization_sha256 = None
    if args.mode == "scalar":
        ranking_authorization_sha256 = _validate_ranking_authorization(
            Path(args.ranking_authorization).resolve()
            if args.ranking_authorization
            else None,
            completion_sha256=completion_sha256,
            input_sha256=input_sha256,
        )

    output_root = Path(args.output_root).resolve()
    shard_name = _shard_name(args.shard_index, args.num_shards)
    manifest_path = output_root / "shards" / f"{shard_name}.manifest.json"
    targets = [
        _output_path(output_root, run, shard_name) for run in completion["runs"]
    ]
    existing = [path for path in [manifest_path, *targets] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"shard outputs already exist: {existing[0]}")

    dataset = CLIRTrajectoryDataset(input_path, feature_root=args.feature_root)
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    if not indices:
        raise ValueError("shard contains no source rows")
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
            _require_finite(outputs, args.mode)
            buffers[_run_key(run)].extend(
                _materialize_batch(
                    mode=args.mode,
                    dataset_rows=dataset.rows,
                    row_indices=row_indices,
                    batch=batch,
                    outputs=outputs,
                    checkpoint_sha256=str(run["checkpoint_file_sha256"]),
                    onset_threshold=args.onset_threshold,
                )
            )
            del outputs

    outputs_manifest: dict[str, Any] = {}
    for run, _ in loaded:
        key = _run_key(run)
        rows = buffers[key]
        observed_indices = [int(row["source_row_index"]) for row in rows]
        if observed_indices != indices:
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
        "schema_version": "clir-factorial-multi-checkpoint-shard-v1",
        "status": "PASS_FACTORIAL_SCORING_SHARD",
        "created_at_utc": _utc_now(),
        "mode": args.mode,
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": input_sha256,
        "input_rows": len(dataset),
        "completion_report": str(completion_path),
        "completion_report_sha256": completion_sha256,
        "ranking_authorization_sha256": ranking_authorization_sha256,
        "scorer_sha256": file_sha256(__file__),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_rows": len(indices),
        "source_indices_sha256": canonical_sha256(indices),
        "device": str(device),
        "amp_dtype": args.amp_dtype,
        "batch_size": args.batch_size,
        "outputs": outputs_manifest,
    }
    atomic_write_json(manifest_path, report)
    print(json.dumps({"status": report["status"], "shard": shard_name, "rows": len(indices)}))


def _add_global_selections(rows: list[dict[str, Any]]) -> None:
    by_query: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_query.setdefault(str(row["query_id"]), []).append(index)
    selected: set[int] = set()
    for indices in by_query.values():
        best = max(indices, key=lambda index: float(rows[index]["clir_score"]))
        selected.add(best)
    for index, row in enumerate(rows):
        row["clir_selected_best_of_n"] = index in selected


def _validate_merged_source_row(
    row: Mapping[str, Any], source: Mapping[str, Any], mode: str, index: int
) -> None:
    if int(row.get("source_row_index", -1)) != index:
        raise ValueError("merged source index drift")
    if row.get("id") != source.get("id") or row.get("query_id") != source.get("query_id"):
        raise ValueError("merged source identity drift")
    if mode == "scalar":
        for field in ("candidate_index", "correctness"):
            if row.get(field) != source.get(field):
                raise ValueError(f"merged ranking field drift: {field}")
    elif mode == "full":
        for field, value in source.items():
            if row.get(field) != value:
                raise ValueError(f"merged full-score source field drift: {field}")


def command_merge(args: argparse.Namespace) -> None:
    if args.mode not in MODES:
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    input_path = Path(args.input_jsonl).resolve()
    completion_path = Path(args.completion_report).resolve()
    source_rows = read_jsonl(input_path)
    input_sha256 = _validate_source(input_path, args.expected_input_sha256)
    completion, completion_sha256 = _load_completion(completion_path, mode=args.mode)
    ranking_authorization_sha256 = None
    if args.mode == "scalar":
        ranking_authorization_sha256 = _validate_ranking_authorization(
            Path(args.ranking_authorization).resolve()
            if args.ranking_authorization
            else None,
            completion_sha256=completion_sha256,
            input_sha256=input_sha256,
        )
    shard_root = Path(args.shard_root).resolve()
    output_root = Path(args.output_root).resolve()
    target_report = output_root / "merge_report.json"
    if target_report.exists() and not args.overwrite:
        raise FileExistsError(f"merge report already exists: {target_report}")

    manifests = []
    for shard_index in range(args.num_shards):
        shard_name = _shard_name(shard_index, args.num_shards)
        manifest_path = shard_root / "shards" / f"{shard_name}.manifest.json"
        manifest = _load_json(manifest_path)
        if (
            manifest.get("status") != "PASS_FACTORIAL_SCORING_SHARD"
            or manifest.get("mode") != args.mode
            or manifest.get("input_jsonl_sha256") != input_sha256
            or manifest.get("completion_report_sha256") != completion_sha256
            or int(manifest.get("shard_index", -1)) != shard_index
            or int(manifest.get("num_shards", -1)) != args.num_shards
        ):
            raise ValueError(f"stale or invalid shard manifest: {manifest_path}")
        if (
            args.mode == "scalar"
            and manifest.get("ranking_authorization_sha256")
            != ranking_authorization_sha256
        ):
            raise ValueError("ranking shard authorization drift")
        if manifest.get("scorer_sha256") != file_sha256(__file__):
            raise ValueError("shards were produced by a different scorer implementation")
        manifests.append((manifest_path, manifest))

    merged_outputs: dict[str, Any] = {}
    for run in completion["runs"]:
        key = _run_key(run)
        rows: list[dict[str, Any]] = []
        for manifest_path, manifest in manifests:
            record = manifest["outputs"].get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"shard lacks run {key}: {manifest_path}")
            shard_path = Path(str(record["path"]))
            if file_sha256(shard_path) != record["file_sha256"]:
                raise ValueError(f"scored shard hash drift: {shard_path}")
            rows.extend(read_jsonl(shard_path))
        rows.sort(key=lambda row: int(row["source_row_index"]))
        indices = [int(row["source_row_index"]) for row in rows]
        if indices != list(range(len(source_rows))):
            raise ValueError(f"merged shards do not exactly cover source rows for {key}")
        for index, (row, source) in enumerate(zip(rows, source_rows)):
            _validate_merged_source_row(row, source, args.mode, index)
            if row.get("clir_checkpoint_sha256") != run["checkpoint_file_sha256"]:
                raise ValueError(f"checkpoint identity drift in merged rows for {key}")
            if not math.isfinite(float(row["clir_score"])):
                raise FloatingPointError(f"non-finite merged score for {key}")
        if args.mode == "scalar":
            _add_global_selections(rows)
        target = output_root / str(run["cell"]) / f"seed-{int(run['seed'])}" / "scored.jsonl"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"merged output already exists: {target}")
        atomic_write_jsonl(target, rows)
        merged_outputs[key] = {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "rows": len(rows),
            "checkpoint_sha256": run["checkpoint_file_sha256"],
        }

    report = {
        "schema_version": "clir-factorial-multi-checkpoint-merge-v1",
        "status": "PASS_FACTORIAL_SCORING_MERGE",
        "created_at_utc": _utc_now(),
        "mode": args.mode,
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": input_sha256,
        "input_rows": len(source_rows),
        "completion_report_sha256": completion_sha256,
        "ranking_authorization_sha256": ranking_authorization_sha256,
        "scorer_sha256": file_sha256(__file__),
        "num_shards": args.num_shards,
        "shard_manifest_sha256": {
            str(index): file_sha256(path)
            for index, (path, _) in enumerate(manifests)
        },
        "outputs": merged_outputs,
    }
    atomic_write_json(target_report, report)
    print(json.dumps({"status": report["status"], "runs": len(merged_outputs), "rows": len(source_rows)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--input-jsonl", required=True)
    worker.add_argument("--completion-report", required=True)
    worker.add_argument("--output-root", required=True)
    worker.add_argument("--mode", choices=MODES, required=True)
    worker.add_argument("--expected-input-sha256", default=None)
    worker.add_argument("--ranking-authorization", default=None)
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
    worker.add_argument("--onset-threshold", type=float, default=0.5)
    worker.add_argument("--overwrite", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-jsonl", required=True)
    merge.add_argument("--completion-report", required=True)
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--mode", choices=MODES, required=True)
    merge.add_argument("--expected-input-sha256", default=None)
    merge.add_argument("--ranking-authorization", default=None)
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
