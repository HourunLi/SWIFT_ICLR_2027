"""Score trajectories with a trained CLIR reward model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.clir_data import (
    CLIRTrajectoryDataset,
    clir_collate,
    move_batch_to_device,
    write_jsonl,
)
from src.consistency_localized_reward import (
    ConsistencyLocalizedReward,
    RewardConfig,
    infer_pseudo_onsets,
    path_hallucination_probability,
    path_no_hallucination_log_probability,
    select_best_of_n,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CLIR trajectories.")
    parser.add_argument("--input_jsonl", required=True, help="JSONL file to score.")
    parser.add_argument(
        "--model", required=True, help="CLIR checkpoint from train_clir.py."
    )
    parser.add_argument(
        "--output_jsonl", required=True, help="Where to write scored rows."
    )
    parser.add_argument(
        "--feature_root",
        default=None,
        help="Base directory for relative feature paths.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Conservative default for 101376-wide all-layer features.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--onset_threshold", type=float, default=0.5)
    parser.add_argument("--amp_dtype", default="bfloat16", choices=["none", "bfloat16"])
    parser.add_argument(
        "--scalar_only",
        action="store_true",
        help=(
            "Write only checkpoint identity, scalar CLIR score, and Best-of-N "
            "selection fields. Intended for large ranking populations that do "
            "not need per-token diagnostics."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(path: str | Path, device: torch.device) -> ConsistencyLocalizedReward:
    # CLIR full-state checkpoints contain training/RNG metadata. Only load a
    # checkpoint from a trusted training run.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_values = checkpoint.get("model_config", checkpoint.get("config"))
    if config_values is None:
        raise ValueError("Checkpoint does not contain a CLIR model config")
    config = RewardConfig(**config_values)
    model = ConsistencyLocalizedReward(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("bfloat16 autocast is supported only on CPU or CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def atomic_write_jsonl(path: str | Path, rows: List[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(descriptor)
    try:
        write_jsonl(temporary, rows)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not 0.0 <= args.onset_threshold <= 1.0:
        raise ValueError("onset_threshold must be in [0, 1]")
    output_path = Path(args.output_jsonl)
    protected = {Path(args.input_jsonl).resolve(), Path(args.model).resolve()}
    if output_path.resolve() in protected:
        raise ValueError("output_jsonl must not overwrite the input or checkpoint")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_path}")
    device = resolve_device(args.device)
    dataset = CLIRTrajectoryDataset(args.input_jsonl, feature_root=args.feature_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clir_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    model = load_model(args.model, device)
    checkpoint_sha256 = file_sha256(args.model)

    rows: List[Dict] = [
        {
            **row,
            "clir_checkpoint_sha256": checkpoint_sha256,
            "clir_scoring_mode": "scalar_only" if args.scalar_only else "full",
        }
        for row in dataset.rows
    ]
    scored_row_indices: List[int] = []
    scored_scores: List[float] = []
    scored_query_ids: List[str] = []

    for batch in loader:
        row_indices = batch["row_index"].tolist()
        query_ids_raw = list(batch["query_ids_raw"])
        batch = move_batch_to_device(batch, device)
        if args.amp_dtype == "none":
            for key in ("hidden_states", "condition_states", "condition_embedding"):
                if key in batch:
                    batch[key] = batch[key].float()
        with autocast_context(device, args.amp_dtype):
            outputs = model(
                batch["hidden_states"],
                mask=batch["mask"],
                condition_states=batch.get("condition_states"),
                condition_mask=batch.get("condition_mask"),
                condition_embedding=batch.get("condition_embedding"),
                condition_embedding_mask=batch.get("condition_embedding_mask"),
            )
        diagnostic_keys = (
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
        non_finite = [
            key for key in diagnostic_keys if not torch.isfinite(outputs[key]).all()
        ]
        if non_finite:
            raise FloatingPointError(
                "Non-finite scoring outputs: " + ", ".join(non_finite)
            )
        if not args.scalar_only:
            path_probs = path_hallucination_probability(
                outputs["hallucination_logits"], outputs["mask"]
            )
            path_log_clean = path_no_hallucination_log_probability(
                outputs["hallucination_logits"], outputs["mask"]
            )
            pseudo_onsets = infer_pseudo_onsets(
                outputs["hallucination_logits"],
                outputs["mask"],
                threshold=args.onset_threshold,
            )

        for local_idx, row_index in enumerate(row_indices):
            row = rows[row_index]
            row["clir_score"] = float(outputs["scores"][local_idx].detach().cpu())
            if args.scalar_only:
                scored_row_indices.append(row_index)
                scored_scores.append(row["clir_score"])
                scored_query_ids.append(str(query_ids_raw[local_idx]))
                continue

            valid_length = int(batch["mask"][local_idx].sum().detach().cpu())
            row["clir_path_hallucination_prob"] = float(
                path_probs[local_idx].detach().cpu()
            )
            row["clir_path_no_hallucination_log_prob"] = float(
                path_log_clean[local_idx].detach().cpu()
            )
            row["clir_pseudo_onset"] = int(pseudo_onsets[local_idx].detach().cpu())
            row["clir_mean_gate"] = float(
                outputs["gates"][local_idx, :valid_length].mean().detach().cpu()
            )
            gate_attention = outputs["gates"][local_idx] / outputs["gates"][
                local_idx
            ].sum().clamp_min(1e-8)
            prior_alignment = torch.sum(
                gate_attention * outputs["fused_prior"][local_idx]
            )
            row["clir_prior_gate_alignment"] = float(prior_alignment.detach().cpu())
            prior_gate_squared_l2 = torch.sum(
                (gate_attention - outputs["fused_prior"][local_idx]).pow(2)
            )
            row["clir_prior_gate_squared_l2"] = float(
                prior_gate_squared_l2.detach().cpu()
            )
            row["clir_condition_relevance"] = [
                float(x)
                for x in outputs["condition_relevance"][local_idx, :valid_length]
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_gate_attention"] = [
                float(x) for x in gate_attention[:valid_length].detach().cpu().tolist()
            ]
            row["clir_key_prior"] = [
                float(x)
                for x in outputs["key_prior"][local_idx, :valid_length]
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_complete_prior"] = [
                float(x)
                for x in outputs["complete_prior"][local_idx, :valid_length]
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_hallucination_prob"] = [
                float(x)
                for x in torch.sigmoid(
                    outputs["hallucination_logits"][local_idx, :valid_length]
                )
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_token_reward"] = [
                float(x)
                for x in outputs["token_rewards"][local_idx, :valid_length]
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_token_value"] = [
                float(x)
                for x in outputs["token_values"][local_idx, :valid_length]
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_key_prior_membership"] = [
                float(x)
                for x in torch.sigmoid(
                    outputs["key_prior_logits"][local_idx, :valid_length]
                )
                .detach()
                .cpu()
                .tolist()
            ]
            row["clir_complete_prior_membership"] = [
                float(x)
                for x in torch.sigmoid(
                    outputs["complete_prior_logits"][local_idx, :valid_length]
                )
                .detach()
                .cpu()
                .tolist()
            ]
            scored_row_indices.append(row_index)
            scored_scores.append(row["clir_score"])
            scored_query_ids.append(str(query_ids_raw[local_idx]))

    query_to_int: Dict[str, int] = {}
    encoded_groups = []
    for query_id in scored_query_ids:
        query_to_int.setdefault(query_id, len(query_to_int))
        encoded_groups.append(query_to_int[query_id])

    best_local_indices = select_best_of_n(
        torch.tensor(scored_scores, dtype=torch.float32),
        torch.tensor(encoded_groups, dtype=torch.long),
    )
    selected_indices = {
        scored_row_indices[local_idx] for local_idx in best_local_indices.values()
    }
    for idx, row in enumerate(rows):
        row["clir_selected_best_of_n"] = idx in selected_indices

    atomic_write_jsonl(output_path, rows)
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
