#!/usr/bin/env python
"""Score the matched Prior-v16 U0/C/H0/CH decomposition on reused 892 queries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

import score_clir_checkpoint_set as engine
from score_clir import file_sha256
from score_clir_prior_v16_reused_ranking import (
    _inventory,
    _validate_ranking_population,
)
from src.clir_data import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
AUTHORIZATION_STATUS = "AUTHORIZED_MATCHED_U0_C_H_CH_DECOMPOSITION"
SHARD_STATUS = "PASS_PRIOR_V16_CH_DECOMPOSITION_SCORING_SHARD"
MERGE_STATUS = "PASS_PRIOR_V16_CH_DECOMPOSITION_SCORING_MERGE"
CELLS = ("u0", "c", "h", "ch")
SEEDS = (42, 43, 44)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _checkpoint_path(
    authorization: Mapping[str, Any], cell: str, seed: int
) -> Path:
    specification = authorization["cells"][cell]
    if cell == "ch":
        return _project_path(specification["checkpoints"][str(seed)]["path"])
    return _project_path(
        str(specification["checkpoint_pattern"]).format(seed=seed)
    )


def _audit_checkpoint(
    authorization: Mapping[str, Any], cell: str, seed: int
) -> dict[str, Any]:
    specification = authorization["cells"][cell]
    checkpoint_path = _checkpoint_path(authorization, cell, seed)
    checkpoint_sha = file_sha256(checkpoint_path)
    if cell == "ch" and checkpoint_sha != specification["checkpoints"][str(seed)][
        "file_sha256"
    ]:
        raise ValueError(f"immutable CH checkpoint drift: seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = authorization["training"]
    if (
        int(checkpoint.get("completed_epoch", -1)) != int(training["epochs"])
        or len(checkpoint.get("metrics", [])) != int(training["epochs"])
        or not _all_finite(checkpoint.get("metrics", []))
    ):
        raise ValueError(f"incomplete or non-finite checkpoint: {cell}/{seed}")
    contract = checkpoint.get("training_contract", {})
    expected_contract = {
        "seed": seed,
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "amp_dtype": str(training["amp_dtype"]),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError(f"training contract drift: {cell}/{seed}")
    data = checkpoint.get("data_state", {})
    train = authorization["training_input"]
    if (
        data.get("train_sha256") != train["file_sha256"]
        or int(data.get("train_rows", -1)) != int(train["rows"])
        or int(data.get("train_queries", -1)) != int(train["queries"])
    ):
        raise ValueError(f"training data drift: {cell}/{seed}")
    model = checkpoint.get("model_config", {})
    expected_factors = [float(value) for value in specification["factors"]]
    if [float(model.get("consistency_weight", -1)), float(model.get("hallucination_weight", -1))] != expected_factors:
        raise ValueError(f"C/H0 factor drift: {cell}/{seed}")
    for key in (
        "prior_weight",
        "gate_prior_weight",
        "token_reward_weight",
        "tail_weight",
        "mil_weight",
        "pseudo_tail_weight",
        "prior_distill_weight",
        "progress_weight",
        "reconstruction_weight",
    ):
        if float(model.get(key, -1.0)) != 0.0:
            raise ValueError(f"unexpected objective {key}: {cell}/{seed}")
    config_path = _project_path(specification["config"])
    if file_sha256(config_path) != specification["config_sha256"]:
        raise ValueError(f"config hash drift: {cell}")
    provenance = checkpoint.get("run_provenance", {})
    if (
        provenance.get("config", {}).get("sha256")
        != specification["config_sha256"]
        or provenance.get("code", {}).get("branch") != "clir-clean-integration"
        or provenance.get("code", {}).get("dirty") is not False
    ):
        raise ValueError(f"checkpoint provenance drift: {cell}/{seed}")
    bad_tensors = [
        name
        for name, tensor in checkpoint.get("state_dict", {}).items()
        if not torch.isfinite(tensor).all()
    ]
    if bad_tensors:
        raise FloatingPointError(f"non-finite tensors in {cell}/{seed}: {bad_tensors}")
    return {
        "cell": cell,
        "seed": seed,
        # The decomposition protocol records only the two factors that vary
        # here (C and H0), while the shared checkpoint-set scorer uses the
        # historical three-factor tuple (C, H0, Prior).  Prior is deliberately
        # disabled in every cell of this experiment.
        "factors": [*expected_factors, 0.0],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": checkpoint_sha,
        "completed_epoch": int(checkpoint["completed_epoch"]),
    }


def _load_bound_contract(
    *, authorization_path: Path, completion_path: Path, input_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    if completion_path.resolve() != authorization_path.resolve():
        raise ValueError("for this adapter, completion-report must equal authorization")
    raw = _load_json(authorization_path)
    if (
        raw.get("schema_version")
        != "clir-prior-v16-posthoc-ch-decomposition-v1"
        or raw.get("status") != AUTHORIZATION_STATUS
        or raw.get("evidence_boundary", {}).get("fresh_ranking_confirmation")
        is not False
        or raw.get("evidence_boundary", {}).get("no_retuning_after_results")
        is not True
    ):
        raise ValueError("inactive or malformed C/H0 decomposition authorization")
    if raw.get("design", {}).get("cells") != list(CELLS):
        raise ValueError("cell grid drift")
    if raw.get("training", {}).get("seeds") != list(SEEDS):
        raise ValueError("seed grid drift")

    train_spec = raw["training_input"]
    train_path = _project_path(train_spec["path"])
    if file_sha256(train_path) != train_spec["file_sha256"]:
        raise ValueError("training manifest hash drift")
    ranking_spec = raw["ranking_evaluation"]
    ranking_path = _project_path(ranking_spec["path"])
    if input_path.resolve() != ranking_path or file_sha256(input_path) != ranking_spec[
        "file_sha256"
    ]:
        raise ValueError("ranking input drift")
    source = read_jsonl(input_path)
    ranking_inventory = _validate_ranking_population(source, {
        "ranking_rows": ranking_spec["rows"],
        "ranking_queries": ranking_spec["queries"],
        "candidates_per_query": ranking_spec["candidates_per_query"],
        "source_query_counts": {"gsm8k": 731, "math": 161},
    })
    train_inventory = _inventory(read_jsonl(train_path))
    overlaps = {
        "canonical_query_ids": len(
            ranking_inventory["query_ids"] & train_inventory["query_ids"]
        ),
        "declared_cluster_ids": len(
            ranking_inventory["cluster_ids"] & train_inventory["cluster_ids"]
        ),
        "row_ids": len(ranking_inventory["row_ids"] & train_inventory["row_ids"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"training/ranking overlap: {overlaps}")

    runs = [
        _audit_checkpoint(raw, cell, seed) for cell in CELLS for seed in SEEDS
    ]
    output_root = _project_path(raw["runtime"]["output_root"])
    authorization = dict(raw)
    authorization["confirmation_scoring_allowed"] = False
    authorization["runtime"] = {
        **raw["runtime"],
        "shard_output_root": str(output_root / "ranking/scoring_shards"),
        "merged_output_root": str(output_root / "ranking/scored"),
        "num_shards": int(raw["runtime"]["ranking_shards"]),
        "batch_size": int(raw["runtime"]["ranking_batch_size"]),
        "amp_dtype": "bfloat16",
    }
    completion = {
        "status": "PASS_MATCHED_U0_C_H_CH_TRAINING_AUDIT",
        "runs": runs,
    }
    protocol_sha = file_sha256(authorization_path)
    return (
        authorization,
        protocol_sha,
        completion,
        protocol_sha,
        source,
        file_sha256(input_path),
    )


def main() -> None:
    engine._load_bound_contract = _load_bound_contract
    engine.SHARD_STATUS = SHARD_STATUS
    engine.MERGE_STATUS = MERGE_STATUS
    engine.main()


if __name__ == "__main__":
    main()
