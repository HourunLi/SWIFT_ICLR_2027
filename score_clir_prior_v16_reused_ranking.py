#!/usr/bin/env python
"""Score the frozen prior-v16 checkpoint grid on the reused 892-query set.

This is a narrow contract adapter around ``score_clir_checkpoint_set.py``.
It keeps the efficient one-feature-load/many-checkpoint scoring engine while
binding the v16 staged-training report, the already-inspected v7.4 ranking
population, and exact train/evaluation non-overlap checks.  The resulting
evidence is exploratory; this script does not open a protected test set.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import score_clir_checkpoint_set as engine
from score_clir import file_sha256
from src.clir_data import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parent
AUTHORIZATION_STATUS = "AUTHORIZED_PRIOR_V16_POSTHOC_REUSED_892_RANKING_V1"
TRAINING_STATUS = (
    "COMPLETE_PRIOR_V16_POSTHOC_STAGED_TRAINING_AND_MECHANISM_EVALUATION"
)
SHARD_STATUS = "PASS_PRIOR_V16_POSTHOC_REUSED_892_SCORING_SHARD"
MERGE_STATUS = "PASS_PRIOR_V16_POSTHOC_REUSED_892_SCORING_MERGE"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_query_id(value: Any) -> str:
    query_id = str(value)
    match = re.fullmatch(r"gsm8k-train-(\d+)", query_id)
    if match:
        return f"gsm8k:train:{int(match.group(1)):05d}"
    match = re.fullmatch(r"gsm8k:train:(\d+)", query_id)
    if match:
        return f"gsm8k:train:{int(match.group(1)):05d}"
    return query_id


def _validate_bound_file(spec: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(spec["path"]))
    observed = file_sha256(path)
    if observed != spec["file_sha256"]:
        raise ValueError(f"{label} hash drift: {observed}")
    return path


def _inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_ids = {_canonical_query_id(row["query_id"]) for row in rows}
    cluster_ids = {
        str(row["cluster_id"])
        for row in rows
        if row.get("cluster_id") not in (None, "")
    }
    row_ids = {str(row["id"]) for row in rows}
    return {
        "query_ids": query_ids,
        "cluster_ids": cluster_ids,
        "row_ids": row_ids,
    }


def _validate_ranking_population(
    rows: list[dict[str, Any]], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    expected_rows = int(authorization["ranking_rows"])
    expected_queries = int(authorization["ranking_queries"])
    expected_candidates = int(authorization["candidates_per_query"])
    if len(rows) != expected_rows:
        raise ValueError("reused ranking row-count drift")
    inventory = _inventory(rows)
    if len(inventory["query_ids"]) != expected_queries:
        raise ValueError("reused ranking query-count drift")
    if len(inventory["row_ids"]) != len(rows):
        raise ValueError("duplicate row id in reused ranking population")

    by_query: dict[str, list[int]] = {}
    query_sources: dict[str, str] = {}
    for row in rows:
        query_id = _canonical_query_id(row["query_id"])
        candidate_index = row.get("candidate_index")
        correctness = row.get("correctness")
        if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
            raise ValueError("ranking candidate_index must be an integer")
        if correctness not in (0, 1, 0.0, 1.0):
            raise ValueError("ranking correctness must be binary")
        by_query.setdefault(query_id, []).append(candidate_index)
        source = str(row.get("source"))
        previous = query_sources.setdefault(query_id, source)
        if source != previous:
            raise ValueError("ranking query spans multiple sources")
        if (
            int(row.get("feature_dim", -1)) != 101376
            or int(row.get("num_feature_layers", -1)) != 33
            or int(row.get("per_layer_dim", -1)) != 3072
            or row.get("feature_dtype") != "bfloat16"
        ):
            raise ValueError("ranking row violates the frozen feature contract")
        if len(row.get("output_token_ids", [])) != int(row["output_token_count"]):
            raise ValueError("saved output-token axis drift")
        if len(row.get("prompt_token_ids", [])) != int(row["prompt_token_count"]):
            raise ValueError("saved prompt-token axis drift")
        if not row.get("hidden_states_path") or not row.get("condition_states_path"):
            raise ValueError("ranking row lacks a saved feature path")
    if any(
        sorted(indices) != list(range(expected_candidates))
        for indices in by_query.values()
    ):
        raise ValueError("ranking candidate axis is not exactly 0..15 per query")
    source_queries: dict[str, int] = {}
    for source in query_sources.values():
        source_queries[source] = source_queries.get(source, 0) + 1
    if source_queries != authorization["source_query_counts"]:
        raise ValueError("ranking source/query composition drift")
    return inventory


def _validate_training_nonoverlap(
    ranking: Mapping[str, Any], authorization: Mapping[str, Any]
) -> None:
    for name, spec in authorization["training_inputs"].items():
        path = _validate_bound_file(spec, f"training input {name}")
        rows = read_jsonl(path)
        inventory = _inventory(rows)
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"training input {name} row-count drift")
        if len(inventory["query_ids"]) != int(spec["queries"]):
            raise ValueError(f"training input {name} query-count drift")
        overlaps = {
            "canonical_query_ids": len(
                ranking["query_ids"] & inventory["query_ids"]
            ),
            "declared_cluster_ids": len(
                ranking["cluster_ids"] & inventory["cluster_ids"]
            ),
            "row_ids": len(ranking["row_ids"] & inventory["row_ids"]),
        }
        if any(overlaps.values()):
            raise ValueError(f"training/evaluation overlap for {name}: {overlaps}")


def _runs_from_summary(
    summary: Mapping[str, Any], authorization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_runs = summary.get("runs")
    if not isinstance(raw_runs, Mapping):
        raise ValueError("staged summary lacks its run mapping")
    cells = [str(value) for value in authorization["cells"]]
    seeds = [int(value) for value in authorization["seeds"]]
    expected = {f"{cell}/seed-{seed}" for cell in cells for seed in seeds}
    if set(raw_runs) != expected:
        raise ValueError("staged summary cell/seed grid drift")
    runs: list[dict[str, Any]] = []
    for cell in cells:
        factors = authorization["factors"][cell]
        for seed in seeds:
            key = f"{cell}/seed-{seed}"
            record = raw_runs[key]
            checkpoint = record.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"staged summary lacks checkpoint: {key}")
            expected_train = authorization["cell_train_sha256"][cell]
            if (
                checkpoint.get("status") != "PASS_CHECKPOINT_AUDIT"
                or checkpoint.get("train_jsonl_sha256") != expected_train
                or int(checkpoint.get("completed_epoch", -1))
                != int(authorization["completed_epoch"])
            ):
                raise ValueError(f"checkpoint audit drift: {key}")
            runs.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "factors": factors,
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_file_sha256": checkpoint["file_sha256"],
                    "completed_epoch": checkpoint["completed_epoch"],
                }
            )
    return runs


def _load_bound_contract(
    *, authorization_path: Path, completion_path: Path, input_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, list[dict[str, Any]], str]:
    authorization = _load_json(authorization_path)
    if authorization.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("reused-ranking authorization is inactive")
    if authorization.get("evidence_tier") != "posthoc_exploratory_reused_not_fresh":
        raise ValueError("reused-ranking evidence boundary drift")
    if authorization.get("contract_adapter_sha256") != file_sha256(__file__):
        raise ValueError("authorization binds another contract adapter")
    if authorization.get("scorer_sha256") != file_sha256(engine.__file__):
        raise ValueError("authorization binds another scoring engine")
    if authorization.get("confirmation_scoring_allowed") is not False:
        raise ValueError("reused population cannot be marked as confirmation")

    expected_completion = _project_path(authorization["training_completion_path"])
    expected_input = _project_path(authorization["ranking_input_path"])
    if completion_path.resolve() != expected_completion:
        raise ValueError("staged training summary path drift")
    if input_path.resolve() != expected_input:
        raise ValueError("reused ranking input path drift")
    completion_sha = file_sha256(completion_path)
    input_sha = file_sha256(input_path)
    if completion_sha != authorization["training_completion_sha256"]:
        raise ValueError("staged training summary hash drift")
    if input_sha != authorization["ranking_input_sha256"]:
        raise ValueError("reused ranking input hash drift")
    for name, spec in authorization["bound_reports"].items():
        _validate_bound_file(spec, f"bound report {name}")

    summary = _load_json(completion_path)
    if summary.get("status") != TRAINING_STATUS:
        raise ValueError("v16 staged training is not complete")
    runs = _runs_from_summary(summary, authorization)
    source = read_jsonl(input_path)
    ranking_inventory = _validate_ranking_population(source, authorization)
    _validate_training_nonoverlap(ranking_inventory, authorization)
    synthesized = {
        "status": TRAINING_STATUS,
        "runs": runs,
        "source_summary_sha256": completion_sha,
    }
    return (
        authorization,
        file_sha256(authorization_path),
        synthesized,
        completion_sha,
        source,
        input_sha,
    )


def main() -> None:
    engine._load_bound_contract = _load_bound_contract
    engine.SHARD_STATUS = SHARD_STATUS
    engine.MERGE_STATUS = MERGE_STATUS
    engine.main()


if __name__ == "__main__":
    main()
