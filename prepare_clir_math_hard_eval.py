#!/usr/bin/env python
"""Freeze and verify the one-shot protected MATH Level-4/5 evaluation.

The materialize command is the first operation allowed to read MATH test rows.
It may only run from a clean commit containing the frozen protocol and code.
Selection never observes generated candidates, correctness, or CLIR scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from src.clir_scale import build_template_clusters
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
    stable_priority,
    validate_source_row,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/math_hard_eval_v1/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1/pre_rollout"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "branch": branch, "dirty": dirty}


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "clir-protected-math-hard-eval-v1":
        raise ValueError("unsupported hard-evaluation protocol")
    if protocol.get("status") != "AUTHORIZED_ONE_SHOT_PROTECTED_EVALUATION":
        raise ValueError("hard evaluation is not authorized")
    source = protocol["source"]
    quotas = {int(key): int(value) for key, value in source["target_queries_per_level"].items()}
    if set(quotas) != {4, 5} or sum(quotas.values()) != int(source["total_queries"]):
        raise ValueError("hard-evaluation level quotas drift")
    generation = protocol["generation"]
    if (
        int(generation["candidate_count"]) != 16
        or int(generation["queries_per_shard"]) != 50
        or int(generation["rollout_shards"]) != 10
    ):
        raise ValueError("hard-evaluation rollout arithmetic drift")
    for section, label in (
        (protocol["train_leakage_audit"]["source_corpus"], "source corpus"),
        (protocol["frozen_models"]["prior_ablation_protocol"], "Prior protocol"),
        (protocol["frozen_models"]["training_completion"], "training completion"),
    ):
        bound = _project_path(section["path"])
        if file_sha256(bound) != section["file_sha256"]:
            raise ValueError(f"{label} hash drift")
    return protocol


def extract_last_boxed(solution: str) -> str | None:
    """Extract the final balanced ``\\boxed``/``\\fbox`` payload."""

    text = str(solution)
    starts = [(text.rfind("\\boxed"), "\\boxed"), (text.rfind("\\fbox"), "\\fbox")]
    start, _ = max(starts, key=lambda item: item[0])
    if start < 0:
        return None
    left = text.find("{", start)
    if left < 0:
        return None
    depth = 0
    for index in range(left, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                value = text[left + 1 : index].strip()
                return value or None
    return None


def _level(value: Any) -> int | None:
    match = re.search(r"([1-5])", str(value))
    return int(match.group(1)) if match is not None else None


def _load_math_split(
    protocol: Mapping[str, Any], split: str, *, cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("MATH protected evaluation requires datasets") from exc
    source = protocol["source"]
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for subject in source["subjects"]:
        dataset = load_dataset(
            source["dataset_id"],
            subject,
            split=split,
            revision=source["revision"],
            cache_dir=cache_dir,
        )
        counts[str(subject)] = len(dataset)
        for index, raw in enumerate(dataset):
            solution = str(raw["solution"]).strip()
            reference = extract_last_boxed(solution)
            rows.append(
                {
                    "source": "math",
                    "query_id": f"math:{split}:{subject}:{index:05d}",
                    "source_record_id": f"{subject}/{split}/{index}",
                    "question": str(raw["problem"]).strip(),
                    # Clustering only needs non-empty reference text for train
                    # anchors.  Protected test candidates are separately
                    # required to have an extractable boxed answer.
                    "reference_answer": reference or "__unparsed_reference__",
                    "source_solution": solution,
                    "source_level": _level(raw["level"]),
                    "source_subject": str(subject),
                    "source_license": "MIT",
                }
            )
    return rows, counts


def _candidate_test_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allowed = {int(value) for value in protocol["source"]["allowed_levels"]}
    exclude_asy = bool(protocol["source"]["exclude_asymptote_prompts"])
    rejected: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for raw in rows:
        level = raw.get("source_level")
        if level not in allowed:
            rejected["level"] += 1
            continue
        question = str(raw["question"])
        if exclude_asy and ("[asy]" in question or "begin{asy}" in question):
            rejected["asymptote"] += 1
            continue
        reference = extract_last_boxed(str(raw["source_solution"]))
        if reference is None:
            rejected["missing_boxed_reference"] += 1
            continue
        row = dict(raw)
        row["reference_answer"] = reference
        row["role"] = "math_hard_eval_v1"
        row["evaluation_split"] = "protected_math_test_level_4_5"
        row["evaluation_only"] = True
        row["sealed_until_final_scoring"] = True
        # Compatibility name consumed by the shared rollout engine.  For this
        # campaign the seal is lifted only by the final all-cell scorer, not by
        # a weight-selection stage.
        row["sealed_until_weight_lock"] = True
        output.append(validate_source_row(row))
    return output, dict(sorted(rejected.items()))


def _attach_prompt_ids(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("hard-evaluation freeze requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        cache_dir=cache_dir,
    )
    maximum = int(generation["maximum_prompt_tokens"])
    kept: list[dict[str, Any]] = []
    too_long: list[str] = []
    lengths: list[int] = []
    for raw in rows:
        row = dict(raw)
        content = str(generation["prompt_template"]).replace(
            "<QUESTION>", str(row["question"])
        )
        token_ids = [
            int(value)
            for value in tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
        if not token_ids or len(token_ids) > maximum:
            too_long.append(str(row["query_id"]))
            continue
        row["prompt_token_ids"] = token_ids
        row["prompt_token_count"] = len(token_ids)
        kept.append(row)
        lengths.append(len(token_ids))
    return kept, {
        "kept": len(kept),
        "too_long": len(too_long),
        "too_long_query_ids_sha256": canonical_sha256(too_long),
        "minimum": min(lengths) if lengths else None,
        "maximum": max(lengths) if lengths else None,
        "mean": sum(lengths) / len(lengths) if lengths else None,
    }


def select_protected_queries(
    selectable: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    namespace = str(protocol["source"]["selection_namespace"])
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in selectable:
        by_cluster[str(raw["cluster_id"])].append(dict(raw))
    representatives: list[dict[str, Any]] = []
    for cluster_id, rows in by_cluster.items():
        rows.sort(key=lambda row: stable_priority(namespace + "-cluster", row["query_id"]))
        representative = rows[0]
        representative["within_cluster_priority"] = stable_priority(
            namespace + "-cluster", representative["query_id"]
        )
        representative["selection_priority"] = stable_priority(
            namespace, representative["query_id"]
        )
        representatives.append(representative)
    quotas = {
        int(key): int(value)
        for key, value in protocol["source"]["target_queries_per_level"].items()
    }
    picked: dict[int, list[dict[str, Any]]] = {}
    available: dict[str, int] = {}
    for level, target in sorted(quotas.items()):
        pool = sorted(
            (row for row in representatives if int(row["source_level"]) == level),
            key=lambda row: str(row["selection_priority"]),
        )
        available[str(level)] = len(pool)
        if len(pool) < target:
            raise ValueError(f"FAIL_YIELD Level {level}: {len(pool)} < {target}")
        picked[level] = pool[:target]
    # Equal quotas are frozen, so alternating levels yields 25/25 per shard.
    selected: list[dict[str, Any]] = []
    for left, right in zip(picked[4], picked[5], strict=True):
        selected.extend((left, right))
    for order, row in enumerate(selected):
        row["hard_eval_query_order"] = order
    return selected, {
        "clusters_before_one_per_cluster": len(by_cluster),
        "representatives": len(representatives),
        "available_by_level": available,
        "selected_by_level": {str(key): len(value) for key, value in picked.items()},
        "selected_query_ids_sha256": canonical_sha256(
            [row["query_id"] for row in selected]
        ),
        "selection_used_generation_or_scores": False,
    }


def build_rollout_shards(
    selected: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    size = int(protocol["generation"]["queries_per_shard"])
    count = int(protocol["generation"]["candidate_count"])
    shards: list[dict[str, Any]] = []
    for start in range(0, len(selected), size):
        rows = selected[start : start + size]
        levels = Counter(int(row["source_level"]) for row in rows)
        if levels != {4: 25, 5: 25}:
            raise ValueError("each hard-evaluation shard must contain 25 L4 and 25 L5")
        index = start // size
        shards.append(
            {
                "shard_id": f"hard-{index:03d}",
                "role": "math_hard_eval_v1",
                "query_ids": [str(row["query_id"]) for row in rows],
                "query_count": len(rows),
                "candidate_count": count,
                "expected_candidate_rows": len(rows) * count,
                "level_counts": {str(key): value for key, value in sorted(levels.items())},
                "output_path": f"rollouts/shards/hard-{index:03d}.jsonl",
            }
        )
    if len(shards) != int(protocol["generation"]["rollout_shards"]):
        raise ValueError("hard-evaluation shard count drift")
    return shards


def _training_overlap(
    selected: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    prior_protocol = json.loads(
        _project_path(protocol["frozen_models"]["prior_ablation_protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    spec = prior_protocol["frozen_parents"]["training_manifest"]
    path = _project_path(spec["path"])
    if file_sha256(path) != spec["file_sha256"]:
        raise ValueError("training manifest hash drift during leakage audit")
    training = read_jsonl(path)
    selected_ids = {str(row["query_id"]) for row in selected}
    train_ids = {str(row["query_id"]) for row in training}
    selected_prompts = {tuple(int(value) for value in row["prompt_token_ids"]) for row in selected}
    train_prompts = {tuple(int(value) for value in row["prompt_token_ids"]) for row in training}
    historical = read_jsonl(
        PROJECT_ROOT / "run_artifacts/prior_ablation_v2/pre_rollout/permanent_exclusions.jsonl"
    )
    historical_ids = {str(row["query_id"]) for row in historical}
    return {
        "training_rows": len(training),
        "training_queries": len(train_ids),
        "query_id_overlap": len(selected_ids & train_ids),
        "prompt_token_ids_overlap": len(selected_prompts & train_prompts),
        "historical_exclusion_query_id_overlap": len(selected_ids & historical_ids),
        "train_anchor_cluster_overlap": 0,
    }


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    state = _git_state()
    if state["dirty"]:
        raise RuntimeError("first protected-test access requires a clean committed tree")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("protected hard-evaluation freeze already exists")

    # Train is loaded first; test loading below is the recorded first protected access.
    train_rows, train_counts = _load_math_split(protocol, "train", cache_dir=args.cache_dir)
    test_accessed_at = _utc_now()
    test_rows, test_counts = _load_math_split(protocol, "test", cache_dir=args.cache_dir)
    candidates, rejection = _candidate_test_rows(test_rows, protocol)

    source_spec = protocol["train_leakage_audit"]["source_corpus"]
    source_path = _project_path(source_spec["path"])
    source_rows = read_jsonl(source_path)
    if len(source_rows) != int(source_spec["rows"]):
        raise ValueError("pinned source corpus row count drift")
    anchor_by_id = {str(row["query_id"]): dict(row) for row in source_rows}
    for row in train_rows:
        anchor_by_id[str(row["query_id"])] = row
    anchors = list(anchor_by_id.values())
    anchor_ids = set(anchor_by_id)
    clusters, selectable, cluster_report = build_template_clusters(
        candidates,
        anchors,
        anchor_ids,
        namespace="clir-math-hard-eval-v1",
    )
    selectable, prompt_report = _attach_prompt_ids(
        selectable, protocol, args.cache_dir
    )
    selected, selection_report = select_protected_queries(selectable, protocol)
    shards = build_rollout_shards(selected, protocol)
    overlap = _training_overlap(selected, protocol)
    if any(
        overlap[key] != 0
        for key in (
            "query_id_overlap",
            "prompt_token_ids_overlap",
            "historical_exclusion_query_id_overlap",
            "train_anchor_cluster_overlap",
        )
    ):
        raise ValueError(f"protected evaluation overlaps training/history: {overlap}")

    output.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    for filename, rows, schema in (
        ("test_candidates.jsonl", candidates, "clir-math-hard-test-candidates-v1"),
        ("template_clusters.jsonl", clusters, "clir-math-hard-template-clusters-v1"),
        ("selected_queries.jsonl", selected, "clir-math-hard-selected-queries-v1"),
    ):
        path = output / filename
        manifest = publish_manifest(path, rows, schema_version=schema)
        records[filename] = {
            "path": str(path),
            "rows": len(rows),
            "file_sha256": manifest["file_sha256"],
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
            "sidecar_file_sha256": file_sha256(
                path.with_suffix(path.suffix + ".manifest.json")
            ),
        }
    shards_path = output / "rollout_shards.json"
    atomic_write_json(shards_path, shards)
    records["rollout_shards.json"] = {
        "path": str(shards_path),
        "rows": len(shards),
        "file_sha256": file_sha256(shards_path),
        "ordered_rows_sha256": canonical_sha256(shards),
    }
    freeze = {
        "schema_version": "clir-math-hard-eval-v1-freeze-report",
        "status": "PASS_MATH_HARD_EVAL_V1_PRE_ROLLOUT_FREEZE",
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": state["commit"],
        "code_branch": state["branch"],
        "code_dirty": False,
        "first_test_access_at_utc": test_accessed_at,
        "train_raw_counts": train_counts,
        "test_raw_counts": test_counts,
        "candidate_rejection": rejection,
        "cluster_report": cluster_report,
        "prompt_report": prompt_report,
        "selection_report": selection_report,
        "leakage_audit": overlap,
        "clir_scores_opened": False,
        "records": records,
    }
    freeze_path = output / "freeze_report.json"
    atomic_write_json(freeze_path, freeze)
    registry = {
        "schema_version": "clir-math-hard-eval-v1-manifest-registry",
        "status": "PASS_MATH_HARD_EVAL_V1_MANIFEST_REGISTRY",
        "protocol_file_sha256": file_sha256(protocol_path),
        "code_commit": state["commit"],
        "freeze_report_file_sha256": file_sha256(freeze_path),
        "records": records,
    }
    atomic_write_json(output / "manifest_registry.json", registry)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    protocol = load_protocol(protocol_path)
    registry_path = output / "manifest_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    freeze_path = output / "freeze_report.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        registry.get("status") != "PASS_MATH_HARD_EVAL_V1_MANIFEST_REGISTRY"
        or registry.get("protocol_file_sha256") != file_sha256(protocol_path)
        or registry.get("freeze_report_file_sha256") != file_sha256(freeze_path)
        or freeze.get("status") != "PASS_MATH_HARD_EVAL_V1_PRE_ROLLOUT_FREEZE"
    ):
        raise ValueError("protected hard-evaluation registry is stale")
    for name, record in registry["records"].items():
        path = Path(record["path"])
        if file_sha256(path) != record["file_sha256"]:
            raise ValueError(f"hard-evaluation artifact hash drift: {name}")
    selected = read_jsonl(output / "selected_queries.jsonl")
    if len(selected) != int(protocol["source"]["total_queries"]):
        raise ValueError("selected protected query count drift")
    if Counter(int(row["source_level"]) for row in selected) != {4: 250, 5: 250}:
        raise ValueError("selected level balance drift")
    if len({str(row["query_id"]) for row in selected}) != len(selected):
        raise ValueError("duplicate selected protected query")
    if len({str(row["cluster_id"]) for row in selected}) != len(selected):
        raise ValueError("selected protected queries share a template cluster")
    shards = json.loads((output / "rollout_shards.json").read_text(encoding="utf-8"))
    expected_ids = [str(row["query_id"]) for row in selected]
    observed_ids = [str(value) for shard in shards for value in shard["query_ids"]]
    if expected_ids != observed_ids:
        raise ValueError("rollout shards do not preserve protected selection order")
    report = {
        "schema_version": "clir-math-hard-eval-v1-independent-verification",
        "status": "PASS_MATH_HARD_EVAL_V1_INDEPENDENT_ARTIFACT_VERIFICATION",
        "protocol_file_sha256": file_sha256(protocol_path),
        "registry_file_sha256": file_sha256(registry_path),
        "queries": len(selected),
        "levels": {"4": 250, "5": 250},
        "clusters": len({str(row["cluster_id"]) for row in selected}),
        "shards": len(shards),
        "test_dataset_reopened": False,
        "clir_scores_opened": False,
    }
    atomic_write_json(output / "independent_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--cache-dir", default=str(PROJECT_ROOT / "run_artifacts/model_cache")
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("materialize").set_defaults(func=command_materialize)
    sub.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
