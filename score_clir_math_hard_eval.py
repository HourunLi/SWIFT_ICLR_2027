#!/usr/bin/env python
"""Score all 57 frozen CLIR checkpoints on protected MATH-hard v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import score_clir_checkpoint_set as engine
from prepare_clir_math_hard_eval import load_protocol
from src.clir_smoke import file_sha256, read_jsonl, validate_rollout_population


PROJECT_ROOT = Path(__file__).resolve().parent
STATUS = "AUTHORIZED_MATH_HARD_EVAL_V1_FINAL_ALL_CELL_SCORING"
SHARD_STATUS = "PASS_MATH_HARD_EVAL_V1_SCORING_SHARD"
MERGE_STATUS = "PASS_MATH_HARD_EVAL_V1_SCORING_MERGE"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_bound_contract(
    *, authorization_path: Path, completion_path: Path, input_path: Path
):
    protocol = load_protocol(authorization_path)
    root = (PROJECT_ROOT / protocol["runtime"]["output_root"]).resolve()
    training_spec = protocol["frozen_models"]["training_completion"]
    expected_completion = (PROJECT_ROOT / training_spec["path"]).resolve()
    expected_input = root / "features_v1/final/tuning_features.jsonl"
    if completion_path.resolve() != expected_completion or input_path.resolve() != expected_input:
        raise ValueError("MATH-hard scorer path drift")
    if file_sha256(completion_path) != training_spec["file_sha256"]:
        raise ValueError("frozen training completion hash drift")
    completion = _load_json(completion_path)
    if (
        completion.get("status") != "PASS_PRIOR_ABLATION_V2_MATCHED_TRAINING_GRID"
        or int(completion.get("run_count", -1)) != 57
    ):
        raise ValueError("frozen 57-checkpoint grid is missing or stale")
    feature_completion_path = root / "features_v1/final/completion.json"
    feature_completion = _load_json(feature_completion_path)
    if feature_completion.get("status") != "PASS_GATE_TUNING_V1_SELECTED_FEATURES":
        raise ValueError("protected feature extraction is incomplete")
    manifest = feature_completion["tuning_manifest"]
    if file_sha256(input_path) != manifest["file_sha256"]:
        raise ValueError("protected feature manifest hash drift")
    source = read_jsonl(input_path)
    population = validate_rollout_population(
        source, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    if (
        len(source)
        != int(protocol["source"]["total_queries"])
        * int(protocol["generation"]["candidate_count"])
        or int(population["queries"]) != int(protocol["source"]["total_queries"])
    ):
        raise ValueError("protected feature population drift")
    runs = []
    for run in completion["runs"]:
        factors = [float(value) for value in run["factors"]]
        prior_on = int(any(factors[index] for index in (2, 3, 4, 5)))
        runs.append(
            {
                **run,
                "full_factors": factors,
                "factors": [int(factors[0]), int(factors[1]), prior_on],
            }
        )
    runtime = protocol["runtime"]
    authorization = {
        "status": STATUS,
        "confirmation_scoring_allowed": False,
        "runtime": {
            "shard_output_root": str(root / "ranking/scoring_shards"),
            "merged_output_root": str(root / "ranking/scored"),
            "num_shards": int(runtime["ranking_score_shards"]),
            "batch_size": int(runtime["ranking_score_batch_size"]),
            "num_workers": int(runtime["ranking_score_num_workers"]),
            "pin_memory": False,
            "amp_dtype": runtime["ranking_score_amp_dtype"],
        },
    }
    engine_completion = {"status": completion["status"], "runs": runs}
    return (
        authorization,
        file_sha256(authorization_path),
        engine_completion,
        file_sha256(completion_path),
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
