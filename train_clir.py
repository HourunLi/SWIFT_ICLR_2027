"""Train CLIR from one explicit configuration and auditable full-state checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.clir_data import (
    CLIRTrajectoryDataset,
    EpochRandomSampler,
    SemanticGroupBatchSampler,
    clir_collate,
    first_present,
    move_batch_to_device,
    resolve_feature_metadata,
)
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "best_current.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CLIR on pre-extracted, token-aligned hidden states."
    )
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--val_feature_root", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--metrics_jsonl", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to replace an existing output checkpoint/metrics pair.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )

    # Small, operational overrides. Loss weights deliberately live only in the
    # JSON config so the CLI cannot drift into a second undocumented method.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--amp_dtype", choices=["none", "bfloat16"], default=None)
    parser.add_argument(
        "--group_by_semantic_id",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--prior_phase_mode",
        choices=["joint", "alternate", "key", "complete"],
        default=None,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help=(
            "Development override. It switches the configured encoder to identity; "
            "real all-layer runs should use the config unchanged."
        ),
    )
    return parser.parse_args(argv)


def load_config(
    path: str | Path,
    hidden_dim_override: Optional[int] = None,
) -> Tuple[RewardConfig, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"model", "training"}:
        raise ValueError("CLIR config must contain exactly `model` and `training`")
    model_values = dict(payload["model"])
    if hidden_dim_override is not None:
        model_values.update(
            {
                "hidden_dim": hidden_dim_override,
                "encoder_type": "identity",
                "model_dim": hidden_dim_override,
                "num_feature_layers": 1,
                "per_layer_dim": hidden_dim_override,
            }
        )
    model_config = RewardConfig(**model_values)
    training = dict(payload["training"])
    return model_config, training


def apply_training_overrides(
    training: Mapping[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    resolved = dict(training)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp_dtype": args.amp_dtype,
        "group_by_semantic_id": args.group_by_semantic_id,
        "prior_phase_mode": args.prior_phase_mode,
    }
    resolved.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    required = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "seed",
        "num_workers",
        "pin_memory",
        "amp_dtype",
        "group_by_semantic_id",
        "prior_phase_mode",
    }
    missing = required - set(resolved)
    if missing:
        raise ValueError(f"Training config is missing: {sorted(missing)}")
    for key in ("epochs", "batch_size", "seed", "num_workers"):
        value = resolved[key]
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{key} must be an integer")
    if resolved["epochs"] <= 0 or resolved["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if resolved["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    for key in ("learning_rate", "weight_decay", "max_grad_norm"):
        value = resolved[key]
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite and numeric")
    if resolved["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be positive")
    if resolved["weight_decay"] < 0.0 or resolved["max_grad_norm"] < 0.0:
        raise ValueError("weight_decay and max_grad_norm must be non-negative")
    for key in ("pin_memory", "group_by_semantic_id"):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"{key} must be boolean")
    if resolved["amp_dtype"] not in {"none", "bfloat16"}:
        raise ValueError("amp_dtype must be `none` or `bfloat16`")
    if resolved["prior_phase_mode"] not in {"joint", "alternate", "key", "complete"}:
        raise ValueError("Invalid prior_phase_mode")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def row_query_id(row: Mapping[str, Any]) -> str:
    value = first_present(row, ("query_id", "candidate_group_id", "prompt_id"))
    if value is None:
        raise ValueError("Every training/validation row requires an explicit query_id")
    return str(value)


def query_ids(dataset: CLIRTrajectoryDataset) -> set[str]:
    return {row_query_id(row) for row in dataset.rows}


def supervision_summary(
    dataset: CLIRTrajectoryDataset, indices: Sequence[int]
) -> Dict[str, int]:
    """Count per-epoch applicable supervision without loading feature payloads."""

    rows = [dataset.rows[index] for index in indices]
    summary = {
        "rows": len(rows),
        "correctness_rows": 0,
        "consistency_rows": 0,
        "consistency_positive_pairs": 0,
        "consistency_negative_pairs": 0,
        "onset_rows": 0,
        "path_hallucination_rows": 0,
        "token_advantage_rows": 0,
        "progress_rows": 0,
        "key_prior_tokens": 0,
        "complete_prior_tokens": 0,
        "paired_prior_rows": 0,
        "reconstruction_rows": 0,
    }
    consistency_records: list[tuple[str, str]] = []
    for row in rows:
        if first_present(row, ("correctness", "label", "final_correct")) is not None:
            summary["correctness_rows"] += 1
        semantic = first_present(
            row,
            (
                "semantic_id",
                "semantic_ids",
                "augmentation_group",
                "augmentation_group_id",
                "group_id",
            ),
        )
        style = first_present(
            row,
            (
                "style_id",
                "style_ids",
                "augmentation_style",
                "rewrite_style",
                "domain_id",
                "domain",
                "style",
            ),
        )
        if semantic is not None and style is not None:
            consistency_records.append((repr(semantic), repr(style)))
        if (
            first_present(row, ("hallucination_onset", "hallucination_start", "onset"))
            is not None
        ):
            summary["onset_rows"] += 1
        if (
            first_present(row, ("path_hallucinated", "hallucinated", "hallucination"))
            is not None
        ):
            summary["path_hallucination_rows"] += 1
        if (
            first_present(row, ("token_advantage", "token_advantages", "advantages"))
            is not None
        ):
            summary["token_advantage_rows"] += 1
        if (
            first_present(row, ("progress_targets", "progress", "progress_target"))
            is not None
        ):
            summary["progress_rows"] += 1
        key_target = first_present(row, ("key_prior_target", "key_prior"))
        complete_target = first_present(
            row, ("complete_prior_target", "complete_prior")
        )
        if key_target is not None:
            summary["key_prior_tokens"] += len(key_target)
        if complete_target is not None:
            summary["complete_prior_tokens"] += len(complete_target)
        if key_target is not None and complete_target is not None:
            summary["paired_prior_rows"] += 1
        if (
            first_present(row, ("complete_reconstruction_target", "csr_target"))
            is not None
        ):
            summary["reconstruction_rows"] += 1

    summary["consistency_rows"] = len(consistency_records)

    def pair_count(counts: Counter[Any]) -> int:
        return sum(count * (count - 1) // 2 for count in counts.values())

    semantic_counts = Counter(semantic for semantic, _ in consistency_records)
    style_counts = Counter(style for _, style in consistency_records)
    joint_counts = Counter(consistency_records)
    same_semantic_and_style = pair_count(joint_counts)
    summary["consistency_positive_pairs"] = (
        pair_count(semantic_counts) - same_semantic_and_style
    )
    summary["consistency_negative_pairs"] = (
        pair_count(style_counts) - same_semantic_and_style
    )
    return summary


def validate_supervision_coverage(
    summary: Mapping[str, int], config: RewardConfig
) -> None:
    """Fail before training when an enabled objective has no applicable target."""

    missing: list[str] = []
    if config.final_weight > 0.0 and summary["correctness_rows"] == 0:
        missing.append("final correctness")
    if config.consistency_weight > 0.0:
        if summary["consistency_positive_pairs"] == 0:
            missing.append("consistency positive pairs")
        if (
            config.negative_consistency_weight > 0.0
            and summary["consistency_negative_pairs"] == 0
        ):
            missing.append("consistency negative pairs")
    if config.hallucination_weight > 0.0 and summary["onset_rows"] == 0:
        missing.append("hallucination onset labels")
    if (
        config.token_reward_weight > 0.0
        and summary["onset_rows"] + summary["token_advantage_rows"] == 0
    ):
        missing.append("token advantage or onset labels")
    if config.tail_weight > 0.0 and summary["onset_rows"] == 0:
        missing.append("negative-tail onset labels")
    if config.mil_weight > 0.0 and summary["path_hallucination_rows"] == 0:
        missing.append("path hallucination labels")
    if config.pseudo_tail_weight > 0.0 and summary["path_hallucination_rows"] == 0:
        missing.append("pseudo-tail path labels")
    if config.progress_weight > 0.0 and summary["progress_rows"] == 0:
        missing.append("progress targets")
    if config.prior_weight > 0.0:
        if config.key_prior_weight > 0.0 and summary["key_prior_tokens"] == 0:
            missing.append("key-prior targets")
        if config.complete_prior_weight > 0.0 and summary["complete_prior_tokens"] == 0:
            missing.append("complete-prior targets")
        if config.prior_distill_weight > 0.0 and summary["paired_prior_rows"] == 0:
            missing.append("paired key/complete targets for distillation")
        if config.gate_prior_weight > 0.0 and summary["paired_prior_rows"] == 0:
            missing.append("paired key/complete targets for gate alignment")
        if config.reconstruction_weight > 0.0 and summary["reconstruction_rows"] == 0:
            missing.append("external reconstruction targets")
    if missing:
        raise ValueError(
            "Enabled objectives have no training supervision: " + ", ".join(missing)
        )


def split_indices_by_query(
    dataset: CLIRTrajectoryDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[list[int], Optional[list[int]]]:
    if val_fraction <= 0.0:
        return list(range(len(dataset))), None
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    grouped: Dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        grouped.setdefault(row_query_id(row), []).append(index)
    if len(grouped) < 2:
        raise ValueError("query-disjoint splitting requires at least two query_ids")
    names = list(grouped)
    random.Random(seed).shuffle(names)
    val_query_count = min(len(names) - 1, max(1, round(len(names) * val_fraction)))
    val_names = set(names[:val_query_count])
    train = [
        index
        for name, indices in grouped.items()
        if name not in val_names
        for index in indices
    ]
    val = [
        index
        for name, indices in grouped.items()
        if name in val_names
        for index in indices
    ]
    return train, val


def validate_feature_contract(
    dataset: CLIRTrajectoryDataset,
    model_config: RewardConfig,
    split_name: str,
) -> None:
    if model_config.encoder_type == "layer_transformer":
        for index, row in enumerate(dataset.rows):
            metadata = resolve_feature_metadata(row)
            feature_dim = metadata["feature_dim"]
            layer_count = metadata["num_feature_layers"]
            per_layer_dim = metadata["per_layer_dim"]
            if feature_dim is None or layer_count is None or per_layer_dim is None:
                raise ValueError(
                    f"{split_name} row {index} lacks the all-layer feature contract"
                )
            observed = (int(feature_dim), int(layer_count), int(per_layer_dim))
            expected = (
                model_config.hidden_dim,
                model_config.num_feature_layers,
                model_config.per_layer_dim,
            )
            if observed != expected:
                raise ValueError(
                    f"{split_name} row {index} feature contract {observed} "
                    f"does not match config {expected}"
                )
    sample = dataset[0]
    width = int(sample["hidden_states"].shape[-1])
    if width != model_config.hidden_dim:
        raise ValueError(
            f"{split_name} feature width is {width}, config expects {model_config.hidden_dim}"
        )


def split_identity_sha256(
    dataset: CLIRTrajectoryDataset, indices: Sequence[int]
) -> str:
    records = [
        {
            "index": index,
            "id": str(dataset.rows[index].get("id", index)),
            "query_id": row_query_id(dataset.rows[index]),
        }
        for index in indices
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_reference_state(
    dataset: CLIRTrajectoryDataset, indices: Sequence[int]
) -> Dict[str, Any]:
    """Bind resume to resolved feature files without rehashing huge payloads."""

    records: list[Dict[str, Any]] = []
    for index in indices:
        row = dataset.rows[index]
        record: Dict[str, Any] = {
            "id": str(row.get("id", index)),
            "query_id": row_query_id(row),
        }
        for prefix in ("hidden_states", "condition_states", "condition_embedding"):
            path_key = f"{prefix}_path"
            checksum_key = f"{prefix}_sha256"
            if path_key in row and row[path_key] is not None:
                path = Path(row[path_key])
                if not path.is_absolute():
                    path = dataset.feature_root / path
                path = path.resolve()
                stat = path.stat()
                record[path_key] = str(path)
                record[f"{prefix}_size"] = stat.st_size
                record[f"{prefix}_mtime_ns"] = stat.st_mtime_ns
                checksum = first_present(row, (checksum_key,))
                if prefix == "hidden_states" and checksum is None:
                    checksum = first_present(row, ("feature_sha256",))
                if prefix == "condition_states" and checksum is None:
                    checksum = first_present(row, ("condition_sha256",))
                if checksum is not None:
                    record[checksum_key] = str(checksum)
            elif prefix in row:
                record[prefix] = "inline-in-manifest"
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "resolved_feature_root": str(dataset.feature_root.resolve()),
        "reference_count": len(records),
        "reference_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def make_loader(
    dataset: CLIRTrajectoryDataset,
    indices: Sequence[int],
    batch_size: int,
    training: bool,
    group_by_semantic_id: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, Optional[Any]]:
    common = {
        "collate_fn": clir_collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        # Worker/base-seed bookkeeping must not consume the model RNG stream;
        # otherwise persistent-worker resume can change dropout randomness.
        "generator": torch.Generator().manual_seed(seed + 1_000_003),
    }
    if training and group_by_semantic_id:
        sampler = SemanticGroupBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            seed=seed,
            indices=indices,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common), sampler
    if training:
        sampler = EpochRandomSampler(indices, seed=seed)
        return (
            DataLoader(dataset, batch_size=batch_size, sampler=sampler, **common),
            sampler,
        )
    subset = Subset(dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, **common), None


def prior_phase_for_epoch(mode: str, epoch: int) -> str:
    if mode == "alternate":
        return "key" if epoch % 2 == 1 else "complete"
    return mode


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 autocast is supported only on CPU or CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def prepare_batch(
    batch: Dict[str, Any], device: torch.device, amp_dtype: str
) -> Dict[str, Any]:
    batch = move_batch_to_device(batch, device)
    if amp_dtype == "none":
        for key in ("hidden_states", "condition_states", "condition_embedding"):
            if key in batch:
                batch[key] = batch[key].float()
    return batch


def run_epoch(
    model: ConsistencyLocalizedReward,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    prior_phase: str,
    amp_dtype: str,
    max_grad_norm: float,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: Dict[str, float] = {}
    rows = 0

    for raw_batch in loader:
        batch = prepare_batch(raw_batch, device, amp_dtype)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            _, losses = model.training_step(batch, prior_phase=prior_phase)
            total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite total loss: {float(total.detach().cpu())}"
            )
        if is_training:
            if not total.requires_grad:
                raise ValueError(
                    "Training batch has no active supervision; offending rows: "
                    + ", ".join(str(value) for value in batch.get("ids", [])[:8])
                )
            total.backward()
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                raise FloatingPointError(
                    f"Non-finite gradients in: {', '.join(bad_gradients[:8])}"
                )
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_rows = int(batch["hidden_states"].shape[0])
        rows += batch_rows
        for key, value in losses.items():
            totals[key] = (
                totals.get(key, 0.0) + float(value.detach().cpu()) * batch_rows
            )
    return {key: value / max(rows, 1) for key, value in totals.items()}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_metrics(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def training_contract(training: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in training.items() if key != "epochs"}


def validate_resume(
    checkpoint: Mapping[str, Any],
    model_config: RewardConfig,
    training: Mapping[str, Any],
    data_state: Mapping[str, Any],
) -> None:
    if checkpoint["model_config"] != model_config.__dict__:
        raise ValueError(
            "Resume checkpoint model config does not match the current config"
        )
    if checkpoint["training_contract"] != training_contract(training):
        raise ValueError("Resume checkpoint training settings do not match")
    if checkpoint["data_state"] != dict(data_state):
        raise ValueError("Resume checkpoint data files do not match")


def main() -> None:
    args = parse_args()
    if args.val_jsonl and args.val_fraction > 0.0:
        raise ValueError("Use either --val_jsonl or --val_fraction, not both")
    output = Path(args.output_model)
    metrics_path = (
        Path(args.metrics_jsonl)
        if args.metrics_jsonl
        else Path(f"{output}.metrics.jsonl")
    )
    protected_paths = [Path(args.train_jsonl), Path(args.config)]
    if args.val_jsonl:
        protected_paths.append(Path(args.val_jsonl))
    resolved_output = output.resolve()
    resolved_metrics = metrics_path.resolve()
    if resolved_output == resolved_metrics:
        raise ValueError("output_model and metrics_jsonl must be different files")
    if resolved_output in {path.resolve() for path in protected_paths}:
        raise ValueError("output_model must not overwrite an input/config manifest")
    if resolved_metrics in {path.resolve() for path in protected_paths}:
        raise ValueError("metrics_jsonl must not overwrite an input/config manifest")
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if output.exists() and resolved_output != resume_path:
            raise FileExistsError(
                f"Refusing to overwrite unrelated checkpoint: {output}"
            )
        if metrics_path.exists() and resolved_output != resume_path:
            raise FileExistsError(
                f"Refusing to overwrite unrelated metrics: {metrics_path}"
            )
    elif not args.overwrite:
        collisions = [path for path in (output, metrics_path) if path.exists()]
        if collisions:
            raise FileExistsError(
                "Fresh run output already exists; pass --overwrite to replace: "
                + ", ".join(str(path) for path in collisions)
            )
    model_config, configured_training = load_config(args.config, args.hidden_dim)
    training = apply_training_overrides(configured_training, args)
    set_seed(int(training["seed"]))
    device = resolve_device(args.device)

    train_dataset = CLIRTrajectoryDataset(
        args.train_jsonl, feature_root=args.feature_root
    )
    validate_feature_contract(train_dataset, model_config, "train")
    train_indices, fraction_val_indices = split_indices_by_query(
        train_dataset, args.val_fraction, int(training["seed"])
    )
    train_supervision = supervision_summary(train_dataset, train_indices)
    validate_supervision_coverage(train_supervision, model_config)

    val_dataset: Optional[CLIRTrajectoryDataset]
    val_indices: Optional[list[int]]
    if args.val_jsonl:
        val_dataset = CLIRTrajectoryDataset(
            args.val_jsonl, feature_root=args.val_feature_root
        )
        validate_feature_contract(val_dataset, model_config, "validation")
        overlap = query_ids(train_dataset) & query_ids(val_dataset)
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(f"Train/validation query_id overlap: {examples}")
        val_indices = list(range(len(val_dataset)))
    elif fraction_val_indices is not None:
        val_dataset = train_dataset
        val_indices = fraction_val_indices
    else:
        val_dataset = None
        val_indices = None

    loader_args = {
        "batch_size": int(training["batch_size"]),
        "seed": int(training["seed"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": bool(training["pin_memory"]),
    }
    train_loader, epoch_sampler = make_loader(
        train_dataset,
        train_indices,
        training=True,
        group_by_semantic_id=bool(training["group_by_semantic_id"]),
        **loader_args,
    )
    val_loader = (
        make_loader(
            val_dataset,
            val_indices,
            training=False,
            group_by_semantic_id=False,
            **loader_args,
        )[0]
        if val_dataset is not None and val_indices is not None
        else None
    )

    data_state = {
        "train_sha256": file_sha256(args.train_jsonl),
        "train_rows": len(train_indices),
        "train_queries": len(
            {row_query_id(train_dataset.rows[i]) for i in train_indices}
        ),
        "train_split_sha256": split_identity_sha256(train_dataset, train_indices),
        "train_features": feature_reference_state(train_dataset, train_indices),
        "train_supervision_per_epoch": train_supervision,
        "val_sha256": file_sha256(args.val_jsonl) if args.val_jsonl else None,
        "val_rows": len(val_indices) if val_indices is not None else 0,
        "val_queries": (
            len({row_query_id(val_dataset.rows[i]) for i in val_indices})
            if val_dataset is not None and val_indices is not None
            else 0
        ),
        "val_split_sha256": (
            split_identity_sha256(val_dataset, val_indices)
            if val_dataset is not None and val_indices is not None
            else None
        ),
        "val_features": (
            feature_reference_state(val_dataset, val_indices)
            if val_dataset is not None and val_indices is not None
            else None
        ),
        "val_fraction": args.val_fraction,
    }
    model = ConsistencyLocalizedReward(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    metrics: list[Dict[str, Any]] = []
    completed_epoch = 0

    if args.resume_from:
        # Full-state checkpoints contain Python/NumPy RNG tuples in addition to
        # tensors. Only resume checkpoints produced by this project.
        # Keep RNG tensors on CPU while loading.  Mapping the whole checkpoint to
        # CUDA also moves torch.get_rng_state() there, but torch.set_rng_state()
        # only accepts a CPU ByteTensor.  Model and optimizer state are moved to
        # their parameter devices by load_state_dict below.
        checkpoint = torch.load(
            args.resume_from, map_location="cpu", weights_only=False
        )
        validate_resume(checkpoint, model_config, training, data_state)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epoch = int(checkpoint["completed_epoch"])
        metrics = list(checkpoint.get("metrics", []))
        restore_rng_state(checkpoint["rng_state"])

    if completed_epoch >= int(training["epochs"]):
        atomic_write_metrics(metrics_path, metrics)
        print(f"checkpoint already completed epoch {completed_epoch}: {output}")
        return

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"device={device} train_rows={len(train_indices)} val_rows={len(val_indices or [])} "
        f"trainable_parameters={parameter_count}"
    )
    print("supervision_per_epoch=" + json.dumps(train_supervision, sort_keys=True))
    for epoch in range(completed_epoch + 1, int(training["epochs"]) + 1):
        if epoch_sampler is not None:
            epoch_sampler.set_epoch(epoch - 1)
        prior_phase = prior_phase_for_epoch(str(training["prior_phase_mode"]), epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            prior_phase,
            str(training["amp_dtype"]),
            float(training["max_grad_norm"]),
        )
        record: Dict[str, Any] = {
            "epoch": epoch,
            "prior_phase": prior_phase,
            "train": train_metrics,
        }
        if val_loader is not None:
            with torch.no_grad():
                record["validation"] = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    prior_phase="joint",
                    amp_dtype=str(training["amp_dtype"]),
                    max_grad_norm=0.0,
                )
        metrics.append(record)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": epoch,
            "rng_state": capture_rng_state(),
            "model_config": dict(model_config.__dict__),
            "training_contract": training_contract(training),
            "data_state": data_state,
            "metrics": metrics,
        }
        # The checkpoint is authoritative. Publish it before its readable sidecar.
        atomic_torch_save(checkpoint, output)
        atomic_write_metrics(metrics_path, metrics)
        message = f"epoch={epoch} train_total={train_metrics.get('total', 0.0):.4f}"
        if "validation" in record:
            message += f" val_total={record['validation'].get('total', 0.0):.4f}"
        print(message)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
