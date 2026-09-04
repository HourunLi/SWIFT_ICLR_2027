#!/usr/bin/env python
"""Prepare, audit, and summarize the released official SWIFT reproduction.

GPU generation and reward extraction deliberately execute the pinned upstream
scripts.  This file freezes their inputs, creates deterministic score shards,
checks every row on merge, and produces a single auditable BoN report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from src.clir_smoke import atomic_write_json, file_sha256


PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = PROJECT_ROOT / "configs/swift_official_generalization_v1/protocol.json"
SCHEMA = "swift-official-generalization-reproduction-v1"
STATUS = "AUTHORIZED_BEFORE_REPRODUCTION_ROLLOUTS"
PREPARE_STATUS = "PASS_SWIFT_OFFICIAL_GENERALIZATION_INPUT_FREEZE"
ROLLOUT_STATUS = "PASS_SWIFT_OFFICIAL_GENERALIZATION_ROLLOUT"
SHARD_STATUS = "PASS_SWIFT_OFFICIAL_GENERALIZATION_SCORE_SHARDS"
MERGE_STATUS = "PASS_SWIFT_OFFICIAL_GENERALIZATION_REWARD_MERGE"
FINAL_STATUS = "COMPLETE_SWIFT_OFFICIAL_GENERALIZATION_REPRODUCTION"
DATASETS = ("math", "gsm8k", "aqua_rat")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    return (
        str(resolved.relative_to(PROJECT_ROOT))
        if resolved.is_relative_to(PROJECT_ROOT)
        else str(resolved)
    )


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_branch(protocol: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain")),
    }
    runtime = protocol["runtime"]
    if (
        runtime.get("require_clean_committed_code") is not True
        or state["dirty"]
        or state["branch"] != runtime["required_branch"]
    ):
        raise RuntimeError("official reproduction requires a clean committed branch")
    return state


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file/hash drift: {path}")


def load_protocol(*, verify_files: bool = True) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    if not isinstance(protocol, dict):
        raise ValueError("official reproduction protocol must be an object")
    if protocol.get("schema_version") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("official reproduction protocol is not authorized")
    if tuple(protocol["datasets"]) != DATASETS:
        raise ValueError("official reproduction dataset order drift")
    if protocol["evaluation"]["K"] != [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError("official reproduction K grid drift")
    if int(protocol["generation"]["candidates_per_query"]) != 64:
        raise ValueError("official reproduction candidate count drift")
    if not verify_files:
        return protocol

    expected_python = (
        Path(protocol["runtime"]["python_environment"]) / "bin/python"
    ).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError("official reproduction Python interpreter drift")
    for package, expected in protocol["runtime"]["package_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"official reproduction package drift: {package}")

    vendor = _resolve(protocol["official_sources"]["repository"]["local_checkout"])
    commit = subprocess.run(
        ["git", "-C", str(vendor), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != protocol["official_sources"]["repository"]["commit"]:
        raise ValueError("pinned upstream SWIFT checkout drift")
    for dataset, spec in protocol["datasets"].items():
        generator = vendor / spec["upstream_generator"]
        _check_hash(generator, spec["upstream_generator_sha256"], f"{dataset} generator")
    _check_hash(
        vendor / "preprocess/process_gsm8k.py",
        protocol["datasets"]["gsm8k"]["upstream_preprocessor_sha256"],
        "GSM8K preprocessor",
    )
    _check_hash(
        vendor / "preprocess/process_aqua_rat.py",
        protocol["datasets"]["aqua_rat"]["upstream_preprocessor_sha256"],
        "AQuA-RAT preprocessor",
    )
    _check_hash(
        vendor / "eval/get_rewards.py",
        protocol["reward_scoring"]["get_rewards_sha256"],
        "upstream reward scorer",
    )
    _check_hash(
        vendor / "eval/bon_eval.py",
        protocol["evaluation"]["bon_eval_sha256"],
        "upstream BoN evaluator",
    )
    _check_hash(
        vendor / "utils.py",
        protocol["reward_scoring"]["utils_sha256"],
        "upstream SWIFT utilities",
    )
    for relative, expected in protocol["correctness"]["upstream_files"].items():
        _check_hash(vendor / relative, expected, f"upstream correctness {relative}")
    checkpoint = protocol["official_sources"]["released_reward_checkpoint"]
    checkpoint_path = _resolve(checkpoint["local_path"])
    _check_hash(
        checkpoint_path, checkpoint["file_sha256"], "reward checkpoint"
    )
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        sorted(state_dict) != sorted(checkpoint["state_dict_keys"])
        or list(state_dict["fused_layer.weight"].shape) != checkpoint["weight_shape"]
        or list(state_dict["fused_layer.bias"].shape) != checkpoint["bias_shape"]
    ):
        raise ValueError("released SWIFT checkpoint state-dict drift")
    math_spec = protocol["datasets"]["math"]
    _check_hash(
        _resolve(math_spec["input_path"]), math_spec["input_file_sha256"], "MATH input"
    )
    model_root = _resolve(protocol["official_sources"]["base_model"]["local_path"])
    for relative in protocol["official_sources"]["base_model"][
        "required_runtime_files"
    ]:
        if not (model_root / relative).is_file():
            raise FileNotFoundError(model_root / relative)
    config = _load_json(model_root / "config.json")
    if (
        int(config.get("hidden_size", -1)) != 4096
        or int(config.get("num_hidden_layers", -1)) != 36
    ):
        raise ValueError("Ministral base-model configuration drift")
    index_path = model_root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    weight_map = _load_json(index_path).get("weight_map", {})
    shards = sorted({str(value) for value in weight_map.values()})
    if not shards or any(not (model_root / shard).is_file() for shard in shards):
        raise FileNotFoundError("Ministral weight snapshot is incomplete")
    return protocol


def _root(protocol: Mapping[str, Any]) -> Path:
    return _resolve(protocol["runtime"]["output_root"])


def _prepare_manifest_path(protocol: Mapping[str, Any]) -> Path:
    return _root(protocol) / "pre_rollout_manifest.json"


def _load_prepare_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = _prepare_manifest_path(protocol)
    payload = _load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("status") != PREPARE_STATUS
        or payload.get("protocol_file_sha256") != file_sha256(PROTOCOL_PATH)
        or payload.get("runner_file_sha256") != file_sha256(Path(__file__))
    ):
        raise ValueError("official reproduction pre-rollout manifest drift")
    for dataset, spec in payload["inputs"].items():
        path = _resolve(spec["path"])
        if file_sha256(path) != spec["file_sha256"] or int(spec["rows"]) != 500:
            raise ValueError(f"prepared input drift: {dataset}")
    model_root = _resolve(protocol["official_sources"]["base_model"]["local_path"])
    index_path = model_root / "model.safetensors.index.json"
    weight_map = _load_json(index_path)["weight_map"]
    expected_model_files = set(
        protocol["official_sources"]["base_model"]["required_runtime_files"]
    ) | {str(value) for value in weight_map.values()}
    observed_model_files = payload.get("base_model_files", {})
    if (
        not isinstance(observed_model_files, Mapping)
        or set(observed_model_files) != expected_model_files
    ):
        raise ValueError("prepared base-model file inventory drift")
    for relative, expected in observed_model_files.items():
        _check_hash(model_root / relative, expected, f"prepared base model {relative}")
    expected_reward = protocol["official_sources"]["released_reward_checkpoint"][
        "file_sha256"
    ]
    if payload.get("reward_checkpoint_file_sha256") != expected_reward:
        raise ValueError("prepared reward-checkpoint binding drift")
    return payload


def _assert_no_results(protocol: Mapping[str, Any]) -> None:
    root = _root(protocol)
    forbidden = [root / "rollouts", root / "score_shards", root / "rewards", root / "summary"]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError("cannot freeze inputs after reproduction outputs exist: " + ", ".join(existing))


def _gsm8k_answer(raw: str) -> str:
    match = re.search(r"####\s*(.*)", raw)
    return match.group(1).strip() if match else raw.strip()


def command_prepare(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    state = _require_clean_branch(protocol)
    _assert_no_results(protocol)
    target = _prepare_manifest_path(protocol)
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)

    from datasets import load_dataset

    inputs: dict[str, dict[str, Any]] = {}
    math_path = _resolve(protocol["datasets"]["math"]["input_path"])
    if sum(1 for _ in math_path.open("r", encoding="utf-8")) != 500:
        raise ValueError("pinned MATH input is not 500 rows")
    inputs["math"] = {
        "path": str(math_path.relative_to(PROJECT_ROOT)),
        "file_sha256": file_sha256(math_path),
        "rows": 500,
        "selection": protocol["datasets"]["math"]["selection"],
    }

    gsm_spec = protocol["datasets"]["gsm8k"]
    gsm = load_dataset(
        gsm_spec["dataset_id"], gsm_spec["config"], revision=gsm_spec["revision"]
    )["test"]
    gsm_rows = [
        {"question": str(row["question"]), "answer": _gsm8k_answer(str(row["answer"]))}
        for row in list(gsm)[:500]
    ]
    gsm_path = _resolve(gsm_spec["input_path"])
    _atomic_jsonl(gsm_path, gsm_rows)
    inputs["gsm8k"] = {
        "path": str(gsm_path.relative_to(PROJECT_ROOT)),
        "file_sha256": file_sha256(gsm_path),
        "rows": len(gsm_rows),
        "selection": gsm_spec["selection"],
    }

    aqua_spec = protocol["datasets"]["aqua_rat"]
    aqua = load_dataset(
        aqua_spec["dataset_id"], aqua_spec["config"], revision=aqua_spec["revision"]
    )
    merged = list(aqua["validation"]) + list(aqua["test"])
    aqua_rows = [
        {
            "question": str(row["question"]) + "\n" + " ".join(row["options"]),
            "answer": str(row["correct"]).strip(),
        }
        for row in merged[:500]
    ]
    aqua_path = _resolve(aqua_spec["input_path"])
    _atomic_jsonl(aqua_path, aqua_rows)
    inputs["aqua_rat"] = {
        "path": str(aqua_path.relative_to(PROJECT_ROOT)),
        "file_sha256": file_sha256(aqua_path),
        "rows": len(aqua_rows),
        "selection": aqua_spec["selection"],
    }
    if any(spec["rows"] != 500 for spec in inputs.values()):
        raise ValueError("official reproduction input yield failure")

    model_root = _resolve(protocol["official_sources"]["base_model"]["local_path"])
    index_path = model_root / "model.safetensors.index.json"
    weight_map = _load_json(index_path)["weight_map"]
    model_files = [
        model_root / relative
        for relative in protocol["official_sources"]["base_model"][
            "required_runtime_files"
        ]
    ]
    model_files.extend(model_root / name for name in sorted(set(weight_map.values())))
    report = {
        "schema_version": "swift-official-generalization-input-freeze-v1",
        "status": PREPARE_STATUS,
        "created_at_utc": _utc_now(),
        "code": state,
        "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        "runner_file_sha256": file_sha256(Path(__file__)),
        "inputs": inputs,
        "base_model_files": {
            str(path.relative_to(model_root)): file_sha256(path) for path in model_files
        },
        "reward_checkpoint_file_sha256": protocol["official_sources"][
            "released_reward_checkpoint"
        ]["file_sha256"],
        "reproduction_rollout_correctness_rewards_opened": False,
    }
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _rollout_path(protocol: Mapping[str, Any], dataset: str) -> Path:
    return _root(protocol) / "rollouts" / f"{dataset}.json"


def _verify_rollout(protocol: Mapping[str, Any], dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _rollout_path(protocol, dataset)
    rows = _load_json(path)
    if not isinstance(rows, list) or len(rows) != 32000:
        raise ValueError(f"{dataset} rollout must contain 32,000 rows")
    counts = Counter()
    expected_indices: list[int] = []
    correctness = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{dataset} rollout row is not an object: {row_index}")
        query_index = int(row.get("idx", -1))
        counts[query_index] += 1
        expected_indices.append(row_index // 64)
        if (
            not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("response"), str)
            or not isinstance(row.get("reference"), str)
            or not isinstance(row.get("steps"), list)
            or any(not isinstance(step, str) for step in row["steps"])
            or not isinstance(row.get("correctness"), (bool, int))
        ):
            raise ValueError(f"{dataset} rollout schema drift at row {row_index}")
        correctness += int(bool(row["correctness"]))
    if [int(row["idx"]) for row in rows] != expected_indices:
        raise ValueError(f"{dataset} candidate/query order drift")
    if counts != Counter({index: 64 for index in range(500)}):
        raise ValueError(f"{dataset} query/candidate population drift")
    return rows, {
        "path": _display_path(path),
        "file_sha256": file_sha256(path),
        "rows": len(rows),
        "queries": len(counts),
        "candidates_per_query": 64,
        "correct_candidates": correctness,
        "candidate_accuracy": correctness / len(rows),
    }


def command_verify_rollout(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    _require_clean_branch(protocol)
    _load_prepare_manifest(protocol)
    _, report = _verify_rollout(protocol, args.dataset)
    report.update(
        {
            "status": ROLLOUT_STATUS,
            "dataset": args.dataset,
            "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        }
    )
    target = _root(protocol) / "rollouts" / f"{args.dataset}.manifest.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _query_ranges(shards: int = 8) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, 500, shards + 1, dtype=int).tolist()
    return list(zip(boundaries[:-1], boundaries[1:]))


def command_make_score_shards(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    _require_clean_branch(protocol)
    _load_prepare_manifest(protocol)
    rows, rollout_spec = _verify_rollout(protocol, args.dataset)
    root = _root(protocol) / "score_shards" / args.dataset
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(root)
    outputs: dict[str, Any] = {}
    for shard_index, (start, end) in enumerate(_query_ranges()):
        name = f"shard-{shard_index:03d}-of-008"
        shard_rows = rows[start * 64 : end * 64]
        path = root / f"{name}.json"
        _atomic_json(path, shard_rows)
        outputs[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "file_sha256": file_sha256(path),
            "query_start_inclusive": start,
            "query_end_exclusive": end,
            "queries": end - start,
            "rows": len(shard_rows),
        }
    report = {
        "schema_version": "swift-official-generalization-score-shards-v1",
        "status": SHARD_STATUS,
        "created_at_utc": _utc_now(),
        "dataset": args.dataset,
        "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        "rollout": rollout_spec,
        "outputs": outputs,
    }
    atomic_write_json(root / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _score_shard_manifest(protocol: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    path = _root(protocol) / "score_shards" / dataset / "manifest.json"
    report = _load_json(path)
    if (
        report.get("status") != SHARD_STATUS
        or report.get("dataset") != dataset
        or report.get("protocol_file_sha256") != file_sha256(PROTOCOL_PATH)
        or len(report.get("outputs", {})) != 8
    ):
        raise ValueError(f"{dataset} score-shard manifest drift")
    return report


def command_merge_rewards(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    state = _require_clean_branch(protocol)
    _load_prepare_manifest(protocol)
    rollout_rows, rollout_spec = _verify_rollout(protocol, args.dataset)
    shard_manifest = _score_shard_manifest(protocol, args.dataset)
    merged: list[dict[str, Any]] = []
    for shard_index, (start, end) in enumerate(_query_ranges()):
        name = f"shard-{shard_index:03d}-of-008"
        input_spec = shard_manifest["outputs"][name]
        input_path = _resolve(input_spec["path"])
        if file_sha256(input_path) != input_spec["file_sha256"]:
            raise ValueError(f"{args.dataset} input score-shard drift: {name}")
        source = _load_json(input_path)
        reward_path = _root(protocol) / "rewards/shards" / args.dataset / f"{name}.json"
        rewards = _load_json(reward_path)
        if not isinstance(rewards, list) or len(rewards) != len(source):
            raise ValueError(f"{args.dataset} reward-shard row mismatch: {name}")
        for offset, (row, reward) in enumerate(zip(source, rewards, strict=True)):
            for field in ("idx", "prompt", "reference", "correctness"):
                if reward.get(field) != row.get(field):
                    raise ValueError(f"{args.dataset} reward/source drift: {name}/{offset}/{field}")
            value = float(reward.get("reward", float("nan")))
            if not math.isfinite(value):
                raise ValueError(f"{args.dataset} non-finite reward: {name}/{offset}")
            global_index = start * 64 + offset
            merged.append(
                {
                    "source_row_index": global_index,
                    "idx": int(row["idx"]),
                    "candidate_index": global_index % 64,
                    "prompt": row["prompt"],
                    "reference": row["reference"],
                    "correctness": int(bool(row["correctness"])),
                    "reward": value,
                }
            )
    if len(merged) != len(rollout_rows) or [row["source_row_index"] for row in merged] != list(
        range(32000)
    ):
        raise ValueError(f"{args.dataset} merged reward order drift")
    target = _root(protocol) / "rewards/merged" / f"{args.dataset}.jsonl"
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    _atomic_jsonl(target, merged)
    report = {
        "schema_version": "swift-official-generalization-reward-merge-v1",
        "status": MERGE_STATUS,
        "created_at_utc": _utc_now(),
        "code": state,
        "dataset": args.dataset,
        "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        "rollout_file_sha256": rollout_spec["file_sha256"],
        "score_shard_manifest_file_sha256": file_sha256(
            _root(protocol) / "score_shards" / args.dataset / "manifest.json"
        ),
        "output": {
            "path": str(target.relative_to(PROJECT_ROOT)),
            "file_sha256": file_sha256(target),
            "rows": len(merged),
        },
    }
    atomic_write_json(target.with_suffix(".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected object row: {path}")
                rows.append(row)
    return rows


def _bootstrap_interval(values: np.ndarray, seed: int, replicates: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _dataset_summary(
    protocol: Mapping[str, Any], dataset: str, seed: int
) -> dict[str, Any]:
    path = _root(protocol) / "rewards/merged" / f"{dataset}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") != MERGE_STATUS
        or manifest.get("protocol_file_sha256") != file_sha256(PROTOCOL_PATH)
        or file_sha256(path) != manifest["output"]["file_sha256"]
    ):
        raise ValueError(f"{dataset} reward merge manifest drift")
    rows = _read_jsonl(path)
    if len(rows) != 32000:
        raise ValueError(f"{dataset} merged reward row count drift")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["idx"])].append(row)
    if set(grouped) != set(range(500)) or any(len(group) != 64 for group in grouped.values()):
        raise ValueError(f"{dataset} merged reward population drift")
    k_values = [int(value) for value in protocol["evaluation"]["K"]]
    by_k: dict[str, Any] = {}
    selected_at_64: np.ndarray | None = None
    for k in k_values:
        selected = []
        random_expected = []
        oracle = []
        for query_index in range(500):
            candidates = grouped[query_index][:k]
            labels = [int(row["correctness"]) for row in candidates]
            best = max(candidates, key=lambda row: float(row["reward"]))
            selected.append(int(best["correctness"]))
            random_expected.append(float(np.mean(labels)))
            oracle.append(max(labels))
        selected_array = np.asarray(selected, dtype=np.float64)
        if k == 64:
            selected_at_64 = selected_array
        by_k[str(k)] = {
            "SWIFT_accuracy": float(selected_array.mean()),
            "SWIFT_accuracy_percent_one_decimal": round(100 * float(selected_array.mean()), 1),
            "random_expected_accuracy": float(np.mean(random_expected)),
            "oracle_accuracy": float(np.mean(oracle)),
        }
    if by_k["1"]["SWIFT_accuracy"] != by_k["1"]["random_expected_accuracy"]:
        raise ValueError(f"{dataset} BoN@1 health check failed")
    assert selected_at_64 is not None
    published = float(protocol["evaluation"]["published_targets_percent"][dataset])
    reproduced = 100 * float(selected_at_64.mean())
    return {
        "queries": 500,
        "candidates_per_query": 64,
        "reward_file_sha256": file_sha256(path),
        "by_k": by_k,
        "primary_at_64": {
            "reproduced_accuracy_percent": reproduced,
            "query_bootstrap_95_ci_percent": [
                100 * value for value in _bootstrap_interval(selected_at_64, seed)
            ],
            "published_accuracy_percent": published,
            "reproduced_minus_published_percentage_points": reproduced - published,
            "same_after_one_decimal_rounding": round(reproduced, 1) == published,
        },
    }


def command_summarize(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    state = _require_clean_branch(protocol)
    _load_prepare_manifest(protocol)
    report = {
        "schema_version": "swift-official-generalization-final-v1",
        "status": FINAL_STATUS,
        "created_at_utc": _utc_now(),
        "code": state,
        "protocol_file_sha256": file_sha256(PROTOCOL_PATH),
        "runner_file_sha256": file_sha256(Path(__file__)),
        "datasets": {
            dataset: _dataset_summary(protocol, dataset, 20260904 + index)
            for index, dataset in enumerate(DATASETS)
        },
        "claim_boundary": {
            "released_official_checkpoint_reproduction": True,
            "no_downstream_fine_tuning": True,
            "public_questions_and_published_targets_seen_before_run": True,
            "not_protected_or_blinded": True,
            "not_directly_comparable_to_Phi_CLIR_results": True,
            "candidate_identity_need_not_match_unreleased_author_rollouts": True,
            "no_post_result_tuning_or_subset_selection": True,
        },
    }
    target = _root(protocol) / "summary/final.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(target)
    atomic_write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_show_commands(args: argparse.Namespace) -> None:
    protocol = load_protocol(verify_files=True)
    manifest = _load_prepare_manifest(protocol)
    root = _root(protocol)
    vendor = _resolve(protocol["official_sources"]["repository"]["local_checkout"])
    model = _resolve(protocol["official_sources"]["base_model"]["local_path"])
    checkpoint = _resolve(
        protocol["official_sources"]["released_reward_checkpoint"]["local_path"]
    )
    python = Path(protocol["runtime"]["python_environment"]) / "bin/python"
    input_path = _resolve(manifest["inputs"][args.dataset]["path"])
    generator = vendor / protocol["datasets"][args.dataset]["upstream_generator"]
    rollout = _rollout_path(protocol, args.dataset)
    rollout.parent.mkdir(parents=True, exist_ok=True)
    generation = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 {python} {generator} "
        f"--model_name {model} --input_file {input_path} --output_file {rollout} "
        "--split test --n_rollouts 64 --temperature 1.0 --top_p 0.9 "
        "--max_new_tokens 1024 --max_model_len 4096 --batch_size 16 "
        "--gpu_memory_utilization 0.9 --seed 42"
    )
    score_commands = []
    for shard_index in range(8):
        name = f"shard-{shard_index:03d}-of-008"
        source = root / "score_shards" / args.dataset / f"{name}.json"
        target = root / "rewards/shards" / args.dataset / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        score_commands.append(
            f"CUDA_VISIBLE_DEVICES={shard_index} {python} {vendor / 'eval/get_rewards.py'} "
            f"--model_name {model} --dataset {args.dataset} --dataset_file {source} "
            f"--reward_model_load {checkpoint} --output_file {target} "
            "--seed 42"
        )
    print(json.dumps({"generation": generation, "score_workers": score_commands}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=command_prepare)
    verify = subparsers.add_parser("verify-rollout")
    verify.add_argument("--dataset", choices=DATASETS, required=True)
    verify.add_argument("--overwrite", action="store_true")
    verify.set_defaults(func=command_verify_rollout)
    shards = subparsers.add_parser("make-score-shards")
    shards.add_argument("--dataset", choices=DATASETS, required=True)
    shards.add_argument("--overwrite", action="store_true")
    shards.set_defaults(func=command_make_score_shards)
    merge = subparsers.add_parser("merge-rewards")
    merge.add_argument("--dataset", choices=DATASETS, required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=command_merge_rewards)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--overwrite", action="store_true")
    summarize.set_defaults(func=command_summarize)
    commands = subparsers.add_parser("show-commands")
    commands.add_argument("--dataset", choices=DATASETS, required=True)
    commands.set_defaults(func=command_show_commands)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
