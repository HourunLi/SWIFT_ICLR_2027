#!/usr/bin/env python
"""Freeze the fair SWIFT/U0 MATH-hard model-set addendum before rollout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from score_clir_math_hard_baselines import (
    ADDENDUM_SCHEMA,
    ADDENDUM_STATUS,
    EXPECTED_CELLS,
)
from src.clir_smoke import atomic_write_json, file_sha256
from summarize_clir_math_hard_v2 import EXPECTED_FACTORIAL_TERMS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/math_hard_eval_v1/model_set_addendum_v2.json"
BASE_PROTOCOL = PROJECT_ROOT / "configs/math_hard_eval_v1/protocol.json"
PRE_ROLLOUT_REGISTRY = (
    PROJECT_ROOT / "run_artifacts/math_hard_eval_v1/pre_rollout/manifest_registry.json"
)
PRE_ROLLOUT_FREEZE = (
    PROJECT_ROOT / "run_artifacts/math_hard_eval_v1/pre_rollout/freeze_report.json"
)
PRIOR_COMPLETION = (
    PROJECT_ROOT / "run_artifacts/prior_ablation_v2/training/completion.json"
)
SWIFT_COMPLETION = (
    PROJECT_ROOT / "run_artifacts/swift_official_baseline_v1/training/completion.json"
)
FACTORIAL_COMPLETION = (
    PROJECT_ROOT / "run_artifacts/swift_u0_sampler_factorial_v1/training/completion.json"
)
SCORER = PROJECT_ROOT / "score_clir_math_hard_baselines.py"
SUMMARIZER = PROJECT_ROOT / "summarize_clir_math_hard_v2.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_branch() -> dict[str, Any]:
    state = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain")),
    }
    if state["dirty"] or state["branch"] != "clir-clean-integration":
        raise RuntimeError(
            "model-set freeze requires a clean clir-clean-integration commit"
        )
    return state


def _relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _completion_spec(path: Path, expected_status: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("status") != expected_status:
        raise ValueError(f"completion status drift: {path}")
    return payload, {
        "path": _relative(path),
        "file_sha256": file_sha256(path),
        "status": expected_status,
    }


def _one_run(
    completion: Mapping[str, Any], source_cell: str, seed: int
) -> Mapping[str, Any]:
    matches = [
        run
        for run in completion["runs"]
        if str(run["cell"]) == source_cell and int(run["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"completion lacks exactly one {source_cell}/seed-{seed}")
    return matches[0]


def _run_record(
    raw: Mapping[str, Any], *, cell: str, source_cell: str, model_kind: str
) -> dict[str, Any]:
    epoch = int(raw.get("epoch", raw.get("completed_epoch", -1)))
    path = Path(str(raw["checkpoint_path"])).resolve()
    checksum = str(raw["checkpoint_file_sha256"])
    if epoch != 3 or file_sha256(path) != checksum:
        raise ValueError(f"checkpoint record drift: {source_cell}/seed-{raw['seed']}")
    return {
        "cell": cell,
        "source_cell": source_cell,
        "seed": int(raw["seed"]),
        "epoch": epoch,
        "model_kind": model_kind,
        "checkpoint_path": _relative(path),
        "checkpoint_file_sha256": checksum,
    }


def _assert_results_unopened() -> None:
    root = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1"
    forbidden = [
        root / "rollout_completion.json",
        root / "rollouts/combined_raw.jsonl",
        root / "checker/completion.json",
        root / "features_v1/final/completion.json",
        root / "ranking/scored/merge_report.json",
        root / "ranking/baseline_controls/scored/merge_report.json",
        root / "summary/final.json",
        root / "summary/final_v2.json",
    ]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "cannot freeze model-set addendum after protected outcomes exist: "
            + ", ".join(existing)
        )


def command_freeze(args: argparse.Namespace) -> None:
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"model-set addendum already exists: {target}")
    state = _require_clean_branch()
    _assert_results_unopened()
    for path in (
        BASE_PROTOCOL,
        PRE_ROLLOUT_REGISTRY,
        PRE_ROLLOUT_FREEZE,
        PRIOR_COMPLETION,
        SWIFT_COMPLETION,
        FACTORIAL_COMPLETION,
        SCORER,
        SUMMARIZER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = _load_json(BASE_PROTOCOL)
    registry = _load_json(PRE_ROLLOUT_REGISTRY)
    freeze = _load_json(PRE_ROLLOUT_FREEZE)
    if (
        base.get("status") != "AUTHORIZED_ONE_SHOT_PROTECTED_EVALUATION"
        or registry.get("status") != "PASS_MATH_HARD_EVAL_V1_MANIFEST_REGISTRY"
        or freeze.get("status") != "PASS_MATH_HARD_EVAL_V1_PRE_ROLLOUT_FREEZE"
        or registry.get("protocol_file_sha256") != file_sha256(BASE_PROTOCOL)
        or freeze.get("clir_scores_opened") is not False
        or freeze.get("first_test_access_at_utc") is None
    ):
        raise ValueError("base MATH-hard pre-rollout freeze is stale")

    prior, prior_spec = _completion_spec(
        PRIOR_COMPLETION, "PASS_PRIOR_ABLATION_V2_MATCHED_TRAINING_GRID"
    )
    swift, swift_spec = _completion_spec(
        SWIFT_COMPLETION, "PASS_SWIFT_OFFICIAL_BASELINE_MATCHED_TRAINING_GRID"
    )
    factorial, factorial_spec = _completion_spec(
        FACTORIAL_COMPLETION, "PASS_SWIFT_U0_SAMPLER_FACTORIAL_TRAINING"
    )
    if len(prior.get("runs", [])) != 57:
        raise ValueError("base CLIR checkpoint grid is not 57")

    runs: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        runs.extend(
            [
                _run_record(
                    _one_run(factorial, "u0_random", seed),
                    cell="u0_random",
                    source_cell="u0_random",
                    model_kind="u0_clir",
                ),
                _run_record(
                    _one_run(swift, "swift_official", seed),
                    cell="swift_random",
                    source_cell="swift_official",
                    model_kind="plain_swift",
                ),
                _run_record(
                    _one_run(factorial, "swift_grouped", seed),
                    cell="swift_grouped",
                    source_cell="swift_grouped",
                    model_kind="plain_swift",
                ),
            ]
        )
    runs.sort(key=lambda row: (str(row["cell"]), int(row["seed"])))
    if {(run["cell"], run["seed"]) for run in runs} != {
        (cell, seed) for cell in EXPECTED_CELLS for seed in (42, 43, 44)
    }:
        raise ValueError("additional model registry is incomplete")

    addendum = {
        "schema_version": ADDENDUM_SCHEMA,
        "status": ADDENDUM_STATUS,
        "frozen_at_utc": _utc_now(),
        "purpose": (
            "extend the already selected 500-query MATH-hard protocol with the "
            "complete architecture-by-sampler controls before any rollout, "
            "correctness label, or reward score is generated"
        ),
        "code": state,
        "evidence_boundary": {
            "test_questions_already_accessed_for_deterministic_selection": True,
            "first_test_question_access_at_utc": freeze["first_test_access_at_utc"],
            "no_rollout_correctness_or_reward_scores_opened_before_this_addendum": True,
            "model_set_locked_before_first_rollout": True,
            "no_post_result_checkpoint_epoch_weight_subset_or_seed_selection": True,
            "question_list_candidate_order_checker_generator_and_feature_contract_unchanged": True,
            "all_2400_query_sampler_factorial_results_may_have_been_seen": True,
            "those_diagnostic_results_cannot_change_the_hard_test_model_set": True,
            "published_swift_numbers_directly_comparable": False,
        },
        "frozen_parent": {
            "math_hard_protocol": {
                "path": _relative(BASE_PROTOCOL),
                "file_sha256": file_sha256(BASE_PROTOCOL),
            },
            "pre_rollout_registry": {
                "path": _relative(PRE_ROLLOUT_REGISTRY),
                "file_sha256": file_sha256(PRE_ROLLOUT_REGISTRY),
            },
            "pre_rollout_freeze_report": {
                "path": _relative(PRE_ROLLOUT_FREEZE),
                "file_sha256": file_sha256(PRE_ROLLOUT_FREEZE),
            },
            "training_completions": [prior_spec, swift_spec, factorial_spec],
        },
        "model_set": {
            "base_clir_cells": 19,
            "base_clir_checkpoint_count": 57,
            "additional_cells": list(EXPECTED_CELLS),
            "additional_checkpoint_count": 9,
            "total_checkpoint_count": 66,
            "seeds": [42, 43, 44],
            "primary_epoch": 3,
            "runs": runs,
        },
        "evaluation": {
            "k_values": [1, 2, 4, 8, 16],
            "primary_k": 16,
            "sampler_factorial_contrasts": EXPECTED_FACTORIAL_TERMS,
            "all_19_clir_cells_compared_to_sampler_matched_swift_grouped": True,
            "holm_families": [
                "original_clir_primary_6",
                "architecture_by_sampler_5",
                "all_grouped_clir_vs_grouped_swift_19",
            ],
            "paired_query_bootstrap_replicates": 10000,
            "paired_sign_flip_replicates": 10000,
            "level_and_subject_strata": True,
            "random_expected_and_oracle_required": True,
        },
        "implementation": {
            "baseline_scorer": {
                "path": _relative(SCORER),
                "file_sha256": file_sha256(SCORER),
            },
            "joint_summarizer": {
                "path": _relative(SUMMARIZER),
                "file_sha256": file_sha256(SUMMARIZER),
            },
        },
        "runtime": {
            "required_branch": "clir-clean-integration",
            "require_clean_committed_code": True,
            "math_hard_output_root": "run_artifacts/math_hard_eval_v1",
            "shard_output_root": (
                "run_artifacts/math_hard_eval_v1/ranking/"
                "baseline_controls/scoring_shards"
            ),
            "merged_output_root": (
                "run_artifacts/math_hard_eval_v1/ranking/"
                "baseline_controls/scored"
            ),
            "baseline_merge_report": (
                "run_artifacts/math_hard_eval_v1/ranking/"
                "baseline_controls/scored/merge_report.json"
            ),
            "scoring_shards": 8,
            "scoring_batch_size": 2,
            "scoring_num_workers": 0,
            "scoring_amp_dtype": "bfloat16",
        },
    }
    atomic_write_json(target, addendum)
    print(
        json.dumps(
            {
                "status": ADDENDUM_STATUS,
                "output": str(target),
                "file_sha256": file_sha256(target),
                "additional_runs": len(runs),
                "total_checkpoints": 66,
                "test_questions_first_accessed_at": freeze["first_test_access_at_utc"],
                "protected_outcomes_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze").set_defaults(func=command_freeze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
