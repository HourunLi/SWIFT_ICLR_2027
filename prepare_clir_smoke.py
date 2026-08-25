#!/usr/bin/env python
"""Prepare and validate the frozen CLIR multi-source smoke data pipeline.

The command intentionally stops before hidden-state extraction.  Its outputs
are query/rollout/proposal/annotation manifests under a caller-provided
artifact directory; generated artifacts are not source-controlled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import re
import statistics
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.clir_smoke import (
    agreement_report,
    annotation_signature,
    atomic_write_json,
    build_consistency_proposals,
    build_h_prior_proposals,
    canonical_sha256,
    check_numeric_response,
    cohen_kappa,
    consistency_item,
    extract_math_numeric_reference,
    file_sha256,
    freeze_query_pool,
    load_asdiv_a_repository,
    materialize_h_label,
    materialize_prior_label,
    near_duplicate_candidates,
    public_unit_item,
    publish_manifest,
    read_jsonl,
    resolve_blind_labels,
    select_joint_h_prior_rows,
    stable_priority,
    tokenize_visible_response,
    unitize_exact_tokens,
    validate_annotation,
    validate_annotator_roster,
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs" / "data_expansion_smoke_v3" / "protocol.json"
)


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") not in {
        "clir-data-expansion-smoke-v2",
        "clir-data-expansion-smoke-v3",
    }:
        raise ValueError("only CLIR data-expansion smoke v2/v3 is executable")
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _git_head(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_id_file(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = read_jsonl(source)
        values = [row.get("query_id") for row in rows]
    elif source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else payload.get("query_ids", [])
    else:
        values = [
            line.strip() for line in source.read_text(encoding="utf-8").splitlines()
        ]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{source}: exclusion IDs must be non-empty strings")
    return list(values)


def _prompt_for(question: str, template: str) -> str:
    if "<QUESTION>" not in template:
        raise ValueError("generation prompt template lacks <QUESTION>")
    return template.replace("<QUESTION>", question)


def _derive_query_seed(base_seed: int, query_id: str) -> int:
    return int(
        stable_priority("clir-vllm-query-seed-v2", base_seed, query_id)[:16], 16
    ) % (2**31)


def _ordered_vllm_candidates(request_output: Any, expected_count: int) -> list[Any]:
    candidates = list(request_output.outputs)
    indices = [int(candidate.index) for candidate in candidates]
    if sorted(indices) != list(range(expected_count)):
        raise ValueError(
            "vLLM candidate indices must be unique and contiguous: "
            f"expected 0..{expected_count - 1}, got {sorted(indices)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate.index))


def _load_math_train_rows(
    math_cfg: Mapping[str, Any], *, cache_dir: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only pinned train parquet files from the split-preserving mirror."""

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit(
            "MATH source export requires huggingface_hub and pyarrow"
        ) from exc
    allowed_levels = {int(value) for value in math_cfg["allowed_levels"]}
    min_solution_words = int(math_cfg["minimum_official_solution_words"])
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    file_hashes: dict[str, str] = {}
    raw_train_rows = 0
    for subject, file_cfg in sorted(math_cfg["train_files"].items()):
        path = Path(
            hf_hub_download(
                repo_id=math_cfg["dataset_id"],
                repo_type="dataset",
                filename=file_cfg["path"],
                revision=math_cfg["revision"],
                cache_dir=cache_dir,
            )
        )
        actual_hash = file_sha256(path)
        if actual_hash != file_cfg["sha256"]:
            raise ValueError(
                f"MATH {subject} train parquet hash mismatch: "
                f"expected {file_cfg['sha256']}, found {actual_hash}"
            )
        source_rows = parquet.read_table(path).to_pylist()
        if len(source_rows) != int(file_cfg["row_count"]):
            raise ValueError(
                f"MATH {subject} train row count mismatch: "
                f"expected {file_cfg['row_count']}, found {len(source_rows)}"
            )
        file_hashes[subject] = actual_hash
        raw_train_rows += len(source_rows)
        for index, raw in enumerate(source_rows):
            problem = str(raw["problem"]).strip()
            solution = str(raw["solution"]).strip()
            level_match = re.search(r"(\d+)", str(raw["level"]))
            if level_match is None or int(level_match.group(1)) not in allowed_levels:
                rejected["level"] += 1
                continue
            level = int(level_match.group(1))
            if "[asy]" in problem or "begin{asy}" in problem:
                rejected["asymptote"] += 1
                continue
            if len(solution.split()) < min_solution_words:
                rejected["short_official_solution"] += 1
                continue
            reference = extract_math_numeric_reference(solution)
            if reference is None:
                rejected["non_scalar_or_unsupported_reference"] += 1
                continue
            stratum = f"{subject}|level_{level}"
            rows.append(
                {
                    "source": "math",
                    "query_id": f"math:train:{subject}:{index:05d}",
                    "source_record_id": f"{subject}/train/{index}",
                    "question": problem,
                    "reference_answer": reference,
                    "source_solution": solution,
                    "source_level": level,
                    "source_subject": subject,
                    "selection_stratum": stratum,
                    "source_license": "MIT",
                }
            )
            strata[stratum] += 1
    return rows, {
        "raw_train_rows": raw_train_rows,
        "strict_numeric_rows": len(rows),
        "rejected": dict(sorted(rejected.items())),
        "file_sha256": file_hashes,
        "strata": dict(sorted(strata.items())),
        "test_files_downloaded_or_read": False,
    }


def command_sources(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    source_cfg = protocol["sources"]
    asdiv_cfg = source_cfg["asdiv_a"]
    actual_commit = _git_head(args.asdiv_repository)
    if actual_commit != asdiv_cfg["commit"]:
        raise ValueError(
            f"ASDiv repository commit mismatch: expected {asdiv_cfg['commit']}, "
            f"found {actual_commit}"
        )
    asdiv_rows = load_asdiv_a_repository(
        args.asdiv_repository,
        expected_xml_sha256=asdiv_cfg["xml_sha256"],
        expected_subset_size=int(asdiv_cfg["arithmetic_subset_size"]),
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the pinned requirements before exporting GSM8K"
        ) from exc
    datasets_version = _package_version("datasets")
    expected_datasets = protocol["generation"]["datasets_version"]
    if datasets_version != expected_datasets:
        raise ValueError(
            f"datasets version mismatch: expected {expected_datasets}, "
            f"found {datasets_version}"
        )
    gsm_cfg = source_cfg["gsm8k"]
    dataset = load_dataset(
        gsm_cfg["dataset_id"],
        "main",
        split=gsm_cfg["allowed_split"].removesuffix("_only"),
        revision=gsm_cfg["revision"],
        cache_dir=args.cache_dir,
    )
    gsm_rows = [
        {
            "source": "gsm8k",
            "query_id": f"gsm8k:train:{index:05d}",
            "source_record_id": index,
            "question": str(row["question"]),
            "reference_answer": str(row["answer"]),
            "source_license": "MIT",
        }
        for index, row in enumerate(dataset)
    ]
    math_rows: list[dict[str, Any]] = []
    math_report: dict[str, Any] | None = None
    if "math" in source_cfg:
        math_rows, math_report = _load_math_train_rows(
            source_cfg["math"], cache_dir=args.cache_dir
        )
    output = Path(args.output)
    report = publish_manifest(
        output,
        [*gsm_rows, *asdiv_rows, *math_rows],
        schema_version=(
            "clir-smoke-source-corpus-v3"
            if "math" in source_cfg
            else "clir-smoke-source-corpus-v2"
        ),
        metadata={
            "gsm8k_revision": gsm_cfg["revision"],
            "asdiv_commit": actual_commit,
            "counts": {
                "gsm8k": len(gsm_rows),
                "asdiv-a": len(asdiv_rows),
                "math": len(math_rows),
            },
            "math": math_report,
        },
    )
    print(json.dumps(report, indent=2))


def _v3_dedup_scope(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Limit MATH dedup review to a frozen selection pool with backups."""

    if protocol["schema_version"] != "clir-data-expansion-smoke-v3":
        return [dict(row) for row in rows]
    per_stratum = int(
        protocol["sources"]["math"]["dedup_candidate_pool_per_subject_level_stratum"]
    )
    by_stratum: dict[str, list[Mapping[str, Any]]] = {}
    retained = [dict(row) for row in rows if row.get("source") != "math"]
    for row in rows:
        if row.get("source") == "math":
            by_stratum.setdefault(str(row["selection_stratum"]), []).append(row)
    for stratum, candidates in sorted(by_stratum.items()):
        ordered = sorted(
            candidates,
            key=lambda row: stable_priority("clir-smoke-v3", row["query_id"]),
        )
        if len(ordered) < per_stratum:
            raise ValueError(
                f"MATH dedup pool {stratum} needs {per_stratum}, found {len(ordered)}"
            )
        retained.extend(dict(row) for row in ordered[:per_stratum])
    return retained


def command_dedup_candidates(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    source_rows = read_jsonl(args.sources)
    rows = _v3_dedup_scope(source_rows, protocol)
    excluded = set(_read_id_file(args.excluded_query_ids))
    all_candidates = near_duplicate_candidates(rows, jaccard_threshold=args.threshold)
    candidates = [
        row
        for row in all_candidates
        if not (row["left_query_id"] in excluded and row["right_query_id"] in excluded)
    ]
    report = publish_manifest(
        args.output,
        candidates,
        schema_version="clir-near-duplicate-candidates-v2",
        metadata={
            "threshold": args.threshold,
            "source_rows_sha256": canonical_sha256(source_rows),
            "dedup_scope_rows": len(rows),
            "dedup_scope_rows_sha256": canonical_sha256(rows),
            "excluded_query_ids_sha256": canonical_sha256(sorted(excluded)),
            "all_candidate_count": len(all_candidates),
            "skipped_both_excluded_count": len(all_candidates) - len(candidates),
        },
    )
    print(json.dumps(report, indent=2))


def command_seed_v3_dedup(args: argparse.Namespace) -> None:
    """Carry v2 decisions and conservatively collapse new MATH/MATH near pairs."""

    protocol = load_protocol(args.protocol)
    if protocol["schema_version"] != "clir-data-expansion-smoke-v3":
        raise ValueError("seed-v3-dedup is defined only for smoke v3")
    candidates = read_jsonl(args.candidates)
    prior = {str(row["pair_id"]): row for row in read_jsonl(args.prior_decisions)}
    decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for candidate in candidates:
        pair_id = str(candidate["pair_id"])
        if pair_id in prior:
            decision = str(prior[pair_id]["decision"])
            label_source = "carry_forward_v2_dedup"
        elif {
            str(candidate.get("left_source")),
            str(candidate.get("right_source")),
        } == {"math"}:
            decision = "duplicate"
            label_source = "conservative_math_near_duplicate"
        else:
            unresolved.append(dict(candidate))
            continue
        decisions.append(
            {
                "pair_id": pair_id,
                "decision": decision,
                "label_source": label_source,
            }
        )
        source_counts[label_source] += 1
    output_dir = Path(args.output_dir)
    decision_manifest = publish_manifest(
        output_dir / "decisions.jsonl",
        decisions,
        schema_version="clir-near-duplicate-decisions-v3",
        metadata={"label_sources": dict(source_counts)},
    )
    unresolved_manifest = publish_manifest(
        output_dir / "unresolved_cross_source.jsonl",
        unresolved,
        schema_version="clir-near-duplicate-unresolved-cross-source-v3",
    )
    report = {
        "status": "READY_TO_FREEZE" if not unresolved else "NEEDS_DUAL_AI_DEDUP",
        "candidates": len(candidates),
        "decisions": len(decisions),
        "unresolved_cross_source": len(unresolved),
        "label_sources": dict(source_counts),
    }
    atomic_write_json(output_dir / "seed_report.json", report)
    print(
        json.dumps(
            {
                "report": report,
                "decisions": decision_manifest,
                "unresolved": unresolved_manifest,
            },
            indent=2,
        )
    )


def _validate_dedup_labels(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    candidate_ids = {str(row["pair_id"]) for row in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("frozen dedup candidates contain duplicate pair_id values")
    row_ids = [str(row.get("pair_id")) for row in rows]
    if (
        set(row_ids) != candidate_ids
        or len(row_ids) != len(candidate_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise ValueError("dedup label population differs from frozen candidates")
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("decision") not in {"duplicate", "distinct", "uncertain"}:
            raise ValueError("dedup decision must be duplicate/distinct/uncertain")
        if row.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("dedup confidence is invalid")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("dedup rationale must be non-empty")
        output.append(dict(row))
    return output


def _dedup_auto_agrees(
    annotation_a: Mapping[str, Any], annotation_b: Mapping[str, Any]
) -> bool:
    return bool(
        annotation_a["decision"] == annotation_b["decision"]
        and annotation_a["decision"] != "uncertain"
        and annotation_a["confidence"] != "low"
        and annotation_b["confidence"] != "low"
    )


def command_dedup_triage(args: argparse.Namespace) -> None:
    """Publish only unresolved pairs to a third model, without A/B answers."""

    candidates = read_jsonl(args.candidates)
    labels_a = _validate_dedup_labels(read_jsonl(args.labels_a), candidates)
    labels_b = _validate_dedup_labels(read_jsonl(args.labels_b), candidates)
    by_a = {str(row["pair_id"]): row for row in labels_a}
    by_b = {str(row["pair_id"]): row for row in labels_b}
    third_items = [
        {
            **candidate,
            "requires_independent_answer": True,
        }
        for candidate in candidates
        if not _dedup_auto_agrees(
            by_a[str(candidate["pair_id"])], by_b[str(candidate["pair_id"])]
        )
    ]
    report = publish_manifest(
        args.output,
        third_items,
        schema_version="clir-near-duplicate-third-independent-v2",
        metadata={
            "candidate_count": len(candidates),
            "third_independent_count": len(third_items),
            "candidates_sha256": canonical_sha256(candidates),
            "labels_a_sha256": canonical_sha256(labels_a),
            "labels_b_sha256": canonical_sha256(labels_b),
            "contains_primary_annotations": False,
        },
    )
    print(json.dumps(report, indent=2))


def command_resolve_dedup(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.candidates)
    labels_a = _validate_dedup_labels(read_jsonl(args.labels_a), candidates)
    labels_b = _validate_dedup_labels(read_jsonl(args.labels_b), candidates)
    by_a = {str(row["pair_id"]): row for row in labels_a}
    by_b = {str(row["pair_id"]): row for row in labels_b}
    third_item_ids = {
        str(candidate["pair_id"])
        for candidate in candidates
        if not _dedup_auto_agrees(
            by_a[str(candidate["pair_id"])], by_b[str(candidate["pair_id"])]
        )
    }

    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    primary = roster.get("primary_annotators", [])
    if not isinstance(primary, list) or len(primary) != 2:
        raise ValueError("dedup roster requires exactly two primary annotators")
    validate_annotator_roster(primary, generator_family="phi")

    adjudications = read_jsonl(args.adjudications) if args.adjudications else []
    third_ids = [str(row.get("pair_id")) for row in adjudications]
    if len(third_ids) != len(set(third_ids)):
        raise ValueError("dedup adjudications contain duplicate pair_id values")
    unexpected_third = set(third_ids) - third_item_ids
    if unexpected_third:
        raise ValueError("dedup adjudications include an auto-agreed or unknown pair")

    adjudicator = roster.get("adjudicator")
    if adjudications:
        if not isinstance(adjudicator, Mapping):
            raise ValueError("third-model dedup labels require an adjudicator roster")
        validate_annotator_roster([*primary, adjudicator], generator_family="phi")
        families = {
            str(row["model_family"]).casefold() for row in [*primary, adjudicator]
        }
        if len(families) != 3:
            raise ValueError("dedup A/B/third model families must all differ")
    by_third = {str(row.get("pair_id")): row for row in adjudications}
    decisions: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()
    for candidate in candidates:
        pair_id = str(candidate["pair_id"])
        a, b = by_a[pair_id], by_b[pair_id]
        if _dedup_auto_agrees(a, b):
            decision = str(a["decision"])
            label_source = "auto_agree"
        else:
            third = by_third.get(pair_id)
            if third is None or third.get("decision") not in {"duplicate", "distinct"}:
                decision = "duplicate"
                label_source = "unresolved_conservative_duplicate"
            else:
                if not third.get("independent_answer_completed", False):
                    raise ValueError(
                        "dedup adjudicator did not answer independently first"
                    )
                decision = str(third["decision"])
                label_source = "adjudicated"
        decisions.append(
            {
                "pair_id": pair_id,
                "decision": decision,
                "label_source": label_source,
            }
        )
        sources[label_source] += 1
    left = [str(by_a[str(row["pair_id"])]["decision"]) for row in candidates]
    right = [str(by_b[str(row["pair_id"])]["decision"]) for row in candidates]
    report = {
        "candidates": len(candidates),
        "raw_agree": sum(a == b for a, b in zip(left, right)),
        "raw_agreement": (
            sum(a == b for a, b in zip(left, right)) / len(left) if left else None
        ),
        "kappa": cohen_kappa(left, right),
        "decision_counts": dict(Counter(row["decision"] for row in decisions)),
        "label_sources": dict(sources),
        "third_model_item_count": len(third_item_ids),
        "third_model_label_count": len(adjudications),
        "unresolved_policy": "conservative_duplicate",
    }
    manifest = publish_manifest(
        args.output,
        decisions,
        schema_version="clir-near-duplicate-decisions-v2",
        metadata=report,
    )
    atomic_write_json(Path(args.output).with_suffix(".report.json"), report)
    print(json.dumps({"manifest": manifest, "report": report}, indent=2))


def _canonical_train_query_id(row: Mapping[str, Any]) -> str | None:
    candidates = [row.get("query_id"), row.get("source_query_id")]
    for value in candidates:
        if not isinstance(value, str):
            continue
        match = re.search(r"gsm8k(?::|-)train(?::|-)(\d{1,5})", value)
        if match:
            return f"gsm8k:train:{int(match.group(1)):05d}"
    source = str(row.get("source", row.get("dataset", ""))).casefold()
    split = str(row.get("split", row.get("source_split", ""))).casefold()
    source_index = row.get("source_index")
    if "gsm8k" in source and split == "train" and isinstance(source_index, int):
        return f"gsm8k:train:{source_index:05d}"
    return None


def command_collect_exclusions(args: argparse.Namespace) -> None:
    reasons: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    for raw_path in args.input:
        path = Path(raw_path)
        rows = read_jsonl(path)
        row_counts[str(path.resolve())] = len(rows)
        for row in rows:
            query_id = _canonical_train_query_id(row)
            if query_id is not None:
                reasons.setdefault(query_id, set()).add(str(path.resolve()))
    output_rows = [
        {"query_id": query_id, "reasons": sorted(paths)}
        for query_id, paths in sorted(reasons.items())
    ]
    manifest = publish_manifest(
        args.output,
        output_rows,
        schema_version="clir-prior-train-query-exclusions-v2",
        metadata={"input_row_counts": row_counts},
    )
    print(json.dumps(manifest, indent=2))


def command_freeze(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    rows = _v3_dedup_scope(read_jsonl(args.sources), protocol)
    decisions = (
        read_jsonl(args.near_duplicate_decisions)
        if args.near_duplicate_decisions
        else []
    )
    source_cfg = protocol["sources"]
    is_v3 = protocol["schema_version"] == "clir-data-expansion-smoke-v3"
    required_ids = _read_id_file(args.required_query_ids)
    source_counts = {
        "gsm8k": int(source_cfg["gsm8k"]["query_count"]),
        "asdiv-a": int(source_cfg["asdiv_a"]["query_count"]),
    }
    source_strata: dict[tuple[str, str], int] = {}
    if is_v3:
        math_cfg = source_cfg["math"]
        source_counts["math"] = int(math_cfg["primary_query_count"]) + int(
            math_cfg["reserve_query_count"]
        )
        per_stratum = int(math_cfg["primary_per_subject_level_stratum"]) + int(
            math_cfg["reserve_per_subject_level_stratum"]
        )
        source_strata = {
            ("math", f"{subject}|level_{level}"): per_stratum
            for subject in math_cfg["subjects"]
            for level in math_cfg["allowed_levels"]
        }
        if len(required_ids) != int(protocol["selection"]["incumbent_v2_query_count"]):
            raise ValueError(
                "v3 freeze requires the complete 100-query v2 incumbent set"
            )
        if (
            canonical_sha256(required_ids)
            != protocol["selection"]["incumbent_v2_query_ids_sha256"]
        ):
            raise ValueError("v3 incumbent query IDs/order do not match frozen v2")
    selected, report = freeze_query_pool(
        rows,
        source_counts=source_counts,
        excluded_query_ids=_read_id_file(args.excluded_query_ids),
        required_query_ids=required_ids,
        near_duplicate_decisions=decisions,
        jaccard_threshold=args.threshold,
        selection_namespace="clir-smoke-v3" if is_v3 else "clir-smoke-v2",
        membership="train_only_smoke_v3" if is_v3 else "train_only_smoke_v2",
        source_stratum_counts=source_strata,
    )
    if is_v3:
        primary_per = int(source_cfg["math"]["primary_per_subject_level_stratum"])
        by_stratum: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            if row["source"] == "math":
                by_stratum.setdefault(str(row["selection_stratum"]), []).append(row)
        for stratum_rows in by_stratum.values():
            stratum_rows.sort(key=lambda row: row["selection_priority"])
            for rank, row in enumerate(stratum_rows):
                row["acquisition_batch"] = (
                    "primary" if rank < primary_per else "reserve"
                )
        for row in selected:
            if row["source"] != "math":
                row["acquisition_batch"] = "reused_v2"
    output_dir = Path(args.output_dir)
    manifest = publish_manifest(
        output_dir / "query_manifest.jsonl",
        selected,
        schema_version=(
            "clir-smoke-query-manifest-v3" if is_v3 else "clir-smoke-query-manifest-v2"
        ),
        metadata={"protocol_sha256": canonical_sha256(protocol)},
    )
    if is_v3:
        primary = [row for row in selected if row["acquisition_batch"] != "reserve"]
        new_primary = [row for row in selected if row["acquisition_batch"] == "primary"]
        reserve = [row for row in selected if row["acquisition_batch"] == "reserve"]
        expected = protocol["generation"]
        if len(primary) != int(expected["primary_combined_query_count"]):
            raise AssertionError("v3 primary combined query count is wrong")
        if len(new_primary) != int(expected["primary_new_query_count"]):
            raise AssertionError("v3 primary MATH query count is wrong")
        if len(reserve) != int(expected["reserve_new_query_count"]):
            raise AssertionError("v3 reserve MATH query count is wrong")
        publish_manifest(
            output_dir / "query_manifest_primary.jsonl",
            primary,
            schema_version="clir-smoke-query-manifest-primary-v3",
        )
        publish_manifest(
            output_dir / "new_primary_queries.jsonl",
            new_primary,
            schema_version="clir-smoke-new-primary-queries-v3",
        )
        publish_manifest(
            output_dir / "reserve_queries.jsonl",
            reserve,
            schema_version="clir-smoke-reserve-queries-v3",
        )
    atomic_write_json(output_dir / "source_freeze_report.json", report)
    publish_manifest(
        output_dir / "permanent_train_only_exclusions.jsonl",
        [
            {
                "query_id": row["query_id"],
                "reason": "train_only_smoke_v3" if is_v3 else "train_only_smoke_v2",
            }
            for row in selected
        ],
        schema_version=(
            "clir-smoke-permanent-exclusions-v3"
            if is_v3
            else "clir-smoke-permanent-exclusions-v2"
        ),
    )
    print(
        json.dumps(
            {
                "query_manifest": manifest,
                "selected_counts": report.get("selected_counts"),
                "selected_stratum_counts": report.get("selected_stratum_counts"),
                "required_query_count": report.get("required_query_count"),
                "excluded_clusters": report.get("excluded_clusters"),
            },
            indent=2,
        )
    )


def command_rollout(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    generation = protocol["generation"]
    queries = read_jsonl(args.queries)
    if protocol["schema_version"] == "clir-data-expansion-smoke-v3":
        allowed_counts = {
            int(generation["primary_new_query_count"]),
            int(generation["reserve_new_query_count"]),
        }
        if len(queries) not in allowed_counts or any(
            row.get("source") != "math" for row in queries
        ):
            raise ValueError(
                "v3 rollout accepts only the frozen new-primary or reserve MATH batch"
            )
    elif len(queries) != int(generation["query_count"]):
        raise ValueError("query manifest count does not match the frozen protocol")
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Rollout requires torch and vLLM in the active environment"
        ) from exc
    vllm_version = _package_version("vllm")
    if vllm_version != generation["backend_version"]:
        raise ValueError(
            f"vLLM version mismatch: expected {generation['backend_version']}, "
            f"found {vllm_version}"
        )
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    if tensor_parallel_size <= 0:
        raise RuntimeError("vLLM rollout requires at least one visible CUDA device")
    llm = LLM(
        model=generation["model_id"],
        revision=generation["model_revision"],
        tokenizer_revision=generation["tokenizer_revision"],
        dtype=args.dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=int(generation["max_model_length"]),
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=int(generation["seed"]),
        download_dir=args.cache_dir,
    )
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    sampling: list[Any] = []
    for query in queries:
        user_prompt = _prompt_for(query["question"], generation["prompt_template"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        sampling.append(
            SamplingParams(
                n=int(generation["candidate_count"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_tokens=int(generation["max_new_tokens"]),
                seed=_derive_query_seed(int(generation["seed"]), query["query_id"]),
            )
        )
    request_outputs = llm.generate(prompts, sampling, use_tqdm=True)
    provenance = {
        "protocol_sha256": canonical_sha256(protocol),
        "query_manifest_sha256": canonical_sha256(queries),
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "tokenizer_revision": generation["tokenizer_revision"],
        "backend": "vllm",
        "vllm_version": vllm_version,
        "transformers_version": _package_version("transformers"),
        "torch_version": torch.__version__,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_model": torch.cuda.get_device_name(0),
        "max_num_seqs": args.max_num_seqs,
    }
    rows: list[dict[str, Any]] = []
    n = int(generation["candidate_count"])
    for query, request_output in zip(queries, request_outputs):
        prompt_ids = [int(value) for value in request_output.prompt_token_ids]
        for candidate in _ordered_vllm_candidates(request_output, n):
            output_ids = [int(value) for value in candidate.token_ids]
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            candidate_index = int(candidate.index)
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{candidate_index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": candidate_index,
                    "source": query["source"],
                    "question": query["question"],
                    "reference_answer": query["reference_answer"],
                    "prompt": _prompt_for(
                        query["question"], generation["prompt_template"]
                    ),
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": output_ids,
                    "response": response,
                    "backend_response_text": candidate.text,
                    "decode_matches_backend_text": response == candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "stop_reason": getattr(candidate, "stop_reason", None),
                    "sampling_seed": _derive_query_seed(
                        int(generation["seed"]), query["query_id"]
                    ),
                    "provenance": provenance,
                }
            )
    population = validate_rollout_population(rows, candidate_count=n)
    manifest = publish_manifest(
        args.output,
        rows,
        schema_version=(
            "clir-smoke-raw-rollouts-v3"
            if protocol["schema_version"] == "clir-data-expansion-smoke-v3"
            else "clir-smoke-raw-rollouts-v2"
        ),
        metadata={**provenance, **population},
    )
    print(json.dumps(manifest, indent=2))


def command_merge_rollouts(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    if protocol["schema_version"] != "clir-data-expansion-smoke-v3":
        raise ValueError("merge-rollouts is defined only for smoke v3")
    batches = [read_jsonl(path) for path in args.input]
    expected_batch_count = 3 if args.reserve_included else 2
    if len(batches) != expected_batch_count:
        raise ValueError(
            f"expected {expected_batch_count} rollout batches, found {len(batches)}"
        )
    manifests = [read_jsonl(path) for path in args.query_manifest]
    if len(manifests) != len(batches):
        raise ValueError("provide one frozen query manifest per rollout input")

    old_hash = str(
        protocol["selection"]["incumbent_v2_raw_rollout_ordered_rows_sha256"]
    )
    batch_hashes = [canonical_sha256(batch) for batch in batches]
    if batch_hashes[0] != old_hash or old_hash in batch_hashes[1:]:
        raise ValueError("the first input must be the sole frozen v2 incumbent rollout")
    protocol_hash = canonical_sha256(protocol)
    new_query_counts: list[int] = []
    for batch, manifest, batch_hash in zip(batches, manifests, batch_hashes):
        manifest_by_id = {str(row.get("query_id")): row for row in manifest}
        if len(manifest_by_id) != len(manifest):
            raise ValueError("query manifest contains duplicate query IDs")
        batch_query_ids = {str(row.get("query_id")) for row in batch}
        if batch_query_ids != set(manifest_by_id):
            raise ValueError("rollout batch query IDs do not match its frozen manifest")
        for row in batch:
            query = manifest_by_id[str(row["query_id"])]
            for field in ("source", "question", "reference_answer"):
                if row.get(field) != query.get(field):
                    raise ValueError(
                        f"{row['id']}: rollout {field} differs from frozen query"
                    )
        if batch_hash == old_hash:
            if (
                canonical_sha256(manifest)
                != protocol["selection"][
                    "incumbent_v2_query_manifest_ordered_rows_sha256"
                ]
            ):
                raise ValueError(
                    "v2 query manifest does not match the frozen incumbent"
                )
            continue
        if any(row.get("source") != "math" for row in batch):
            raise ValueError("all newly generated v3 rollout rows must come from MATH")
        provenance_hashes = {
            str(row.get("provenance", {}).get("protocol_sha256")) for row in batch
        }
        if provenance_hashes != {protocol_hash}:
            raise ValueError("new rollout provenance does not bind to this v3 protocol")
        manifest_hash = canonical_sha256(manifest)
        provenance_manifest_hashes = {
            str(row.get("provenance", {}).get("query_manifest_sha256")) for row in batch
        }
        if provenance_manifest_hashes != {manifest_hash}:
            raise ValueError(
                "new rollout provenance does not bind to its frozen query manifest"
            )
        new_query_counts.append(len(batch_query_ids))

    expected_new_counts = [int(protocol["generation"]["primary_new_query_count"])]
    if args.reserve_included:
        expected_new_counts.append(
            int(protocol["generation"]["reserve_new_query_count"])
        )
    if new_query_counts != expected_new_counts:
        raise ValueError(
            f"new rollout batch sizes must be {expected_new_counts}, "
            f"found {new_query_counts}"
        )
    rows = [row for batch in batches for row in batch]
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("rollout inputs overlap by trajectory ID")
    population = validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    expected = (
        int(protocol["generation"]["maximum_combined_query_count"])
        if args.reserve_included
        else int(protocol["generation"]["primary_combined_query_count"])
    )
    if population["queries"] != expected:
        raise ValueError(
            f"combined rollout query count mismatch: expected {expected}, "
            f"found {population['queries']}"
        )
    report = publish_manifest(
        args.output,
        rows,
        schema_version="clir-smoke-combined-raw-rollouts-v3",
        metadata={
            **population,
            "protocol_sha256": protocol_hash,
            "input_ordered_hashes": batch_hashes,
            "input_query_manifest_hashes": [
                canonical_sha256(manifest) for manifest in manifests
            ],
            "reserve_included": bool(args.reserve_included),
        },
    )
    print(json.dumps(report, indent=2))


def materialize_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any | None = None,
    *,
    checker_version: str = "clir_numeric_multisource_v3",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for raw in raw_rows:
        row = dict(raw)
        row.update(
            check_numeric_response(
                response=str(row["response"]),
                raw_reference=str(row["reference_answer"]),
                source=str(row["source"]),
                finish_reason=row.get("finish_reason"),
                checker_version=checker_version,
            )
        )
        try:
            if tokenizer is None:
                mapping = row.pop("_fixture_mapping")
                row["token_mapping_mode"] = "fixture_supplied_offsets"
            else:
                mapping = tokenize_visible_response(
                    tokenizer, str(row["response"]), row["output_token_ids"]
                )
                row["token_mapping_mode"] = mapping.pop("mapping_mode")
            unitization = unitize_exact_tokens(
                response=str(row["response"]),
                output_token_ids=row["output_token_ids"],
                **mapping,
            )
            row.update(unitization)
            row["unitization_status"] = "ok"
        except (KeyError, TypeError, ValueError) as exc:
            row["unitization_status"] = "failed"
            row["unitization_error"] = f"{type(exc).__name__}: {exc}"
            row["units"] = []
            row["material_claim_count"] = 0
            row["eligible_for_supervision"] = False
            failures[type(exc).__name__] += 1
        processed.append(row)
    report = {
        "rows": len(processed),
        "unitization_ok": sum(row["unitization_status"] == "ok" for row in processed),
        "unitization_failures": dict(failures),
        "token_mapping_modes": dict(
            Counter(row.get("token_mapping_mode", "failed") for row in processed)
        ),
        "checker_statuses": dict(Counter(row["checker_status"] for row in processed)),
        "material_claim_count_histogram": dict(
            sorted(Counter(row["material_claim_count"] for row in processed).items())
        ),
    }
    report["exact_contract_pass_rate"] = (
        report["unitization_ok"] / len(processed) if processed else 0.0
    )
    return processed, report


def command_materialize(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Materialization requires transformers") from exc
    generation = protocol["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        generation["model_id"],
        revision=generation["tokenizer_revision"],
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("unitizer v2 requires a fast tokenizer with offset mappings")
    raw_rows = read_jsonl(args.rollouts)
    processed, report = materialize_rows(
        raw_rows,
        tokenizer,
        checker_version=str(protocol["checker"]["version"]),
    )
    population = validate_rollout_population(
        processed, candidate_count=int(generation["candidate_count"])
    )
    manifest = publish_manifest(
        args.output,
        processed,
        schema_version=(
            "clir-smoke-materialized-rollouts-v3"
            if protocol["schema_version"] == "clir-data-expansion-smoke-v3"
            else "clir-smoke-materialized-rollouts-v2"
        ),
        metadata={
            **report,
            **population,
            "protocol_sha256": canonical_sha256(protocol),
        },
    )
    atomic_write_json(
        Path(args.output).with_suffix(".report.json"), {**report, **population}
    )
    print(json.dumps(manifest, indent=2))


def propose_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    consistency_count: int,
    hp_quotas: Mapping[tuple[str, int], int],
    numeric_mismatch_status_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    consistency = build_consistency_proposals(
        rows,
        proposal_count=consistency_count,
        min_length_ratio=1.25,
        max_length_ratio=3.0,
    )
    hp = build_h_prior_proposals(
        rows,
        quotas=hp_quotas,
        consistency_proposals=consistency,
        numeric_mismatch_status_only=numeric_mismatch_status_only,
    )
    return consistency, hp


def _hp_quotas_from_strata(
    strata: Mapping[str, Any],
) -> dict[tuple[str, int], int]:
    """Parse frozen ``<source>_numeric_<status>`` proposal quota names."""

    quotas: dict[tuple[str, int], int] = {}
    for name, count in strata.items():
        if name.endswith("_numeric_mismatch"):
            source_name = name.removesuffix("_numeric_mismatch")
            match = 0
        elif name.endswith("_numeric_match"):
            source_name = name.removesuffix("_numeric_match")
            match = 1
        else:
            raise ValueError(f"invalid H/P source stratum name {name!r}")
        source = {"asdiv_a": "asdiv-a"}.get(source_name, source_name)
        key = (source, match)
        if key in quotas:
            raise ValueError(f"duplicate normalized H/P source stratum {key!r}")
        quotas[key] = int(count)
    return quotas


def command_propose(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    rows = read_jsonl(args.processed)
    proposal_cfg = protocol["proposal_manifests"]
    c_cfg = proposal_cfg["consistency"]
    hp_cfg = proposal_cfg["hallucination_and_prior"]
    strata = hp_cfg["source_numeric_strata"]
    quotas = _hp_quotas_from_strata(strata)
    consistency, hp = propose_rows(
        rows,
        consistency_count=int(c_cfg["natural_proposals"]),
        hp_quotas=quotas,
        numeric_mismatch_status_only=not bool(
            hp_cfg.get("candidate_parse_failure_allowed", True)
        ),
    )
    output_dir = Path(args.output_dir)
    c_manifest = publish_manifest(
        output_dir / "consistency_proposals.jsonl",
        consistency,
        schema_version="clir-consistency-proposals-v2",
    )
    hp_manifest = publish_manifest(
        output_dir / "h_prior_proposals.jsonl",
        hp,
        schema_version="clir-h-prior-proposals-v2",
    )
    by_id = {str(row["id"]): row for row in rows}
    publish_manifest(
        output_dir / "annotation_consistency_natural.jsonl",
        [consistency_item(proposal, by_id) for proposal in consistency],
        schema_version="clir-consistency-annotation-items-v2",
    )
    unit_items = [public_unit_item(by_id[str(proposal["id"])]) for proposal in hp]
    publish_manifest(
        output_dir / "annotation_hallucination_natural.jsonl",
        unit_items,
        schema_version="clir-hallucination-annotation-items-v2",
    )
    publish_manifest(
        output_dir / "annotation_prior_natural.jsonl",
        unit_items,
        schema_version="clir-prior-annotation-items-v2",
    )
    print(json.dumps({"consistency": c_manifest, "h_prior": hp_manifest}, indent=2))


def command_readiness(args: argparse.Namespace) -> None:
    """Apply the frozen primary/reserve yield rule before any v3 annotation."""

    protocol = load_protocol(args.protocol)
    if protocol["schema_version"] != "clir-data-expansion-smoke-v3":
        raise ValueError("readiness is defined only for smoke v3")
    rows = read_jsonl(args.processed)
    expected_queries = int(
        protocol["generation"][
            (
                "maximum_combined_query_count"
                if args.reserve_included
                else "primary_combined_query_count"
            )
        ]
    )
    population = validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    if population["queries"] != expected_queries:
        raise ValueError(
            f"readiness expected {expected_queries} queries, "
            f"found {population['queries']}"
        )
    mismatch_queries = {
        str(row["query_id"])
        for row in rows
        if row.get("checker_status") == "numeric_mismatch"
        and row.get("eligible_for_supervision")
        and row.get("unitization_status") == "ok"
        and sum(unit.get("kind") == "material_claim" for unit in row.get("units", []))
        >= int(
            protocol["unitizer"][
                "minimum_material_claim_units_for_h_and_prior_proposal"
            ]
        )
    }
    proposal_error = None
    try:
        proposal_cfg = protocol["proposal_manifests"]
        strata = proposal_cfg["hallucination_and_prior"]["source_numeric_strata"]
        quotas = _hp_quotas_from_strata(strata)
        consistency, hp = propose_rows(
            rows,
            consistency_count=int(proposal_cfg["consistency"]["natural_proposals"]),
            hp_quotas=quotas,
            numeric_mismatch_status_only=True,
        )
        proposal_counts = {"consistency": len(consistency), "h_prior": len(hp)}
    except ValueError as exc:
        proposal_error = str(exc)
        proposal_counts = None
    mismatch_min = int(
        protocol["acquisition_ladder"][
            "combined_mechanism_eligible_mismatch_query_count_min"
        ]
    )
    unitization_failures = sum(row.get("unitization_status") != "ok" for row in rows)
    empty_responses = sum(not str(row.get("response", "")).strip() for row in rows)
    truncated = sum(row.get("finish_reason") == "length" for row in rows)
    generation_contract_failure_ids = {
        str(row["id"])
        for row in rows
        if row.get("unitization_status") != "ok"
        or not str(row.get("response", "")).strip()
        or row.get("finish_reason") == "length"
    }
    generation_contract_failure_fraction = len(generation_contract_failure_ids) / len(
        rows
    )
    maximum_generation_failure_fraction = float(
        protocol["generation"]["maximum_truncated_empty_or_illegal_fraction"]
    )
    exact_contract_pass = unitization_failures == 0
    generation_quality_pass = (
        generation_contract_failure_fraction <= maximum_generation_failure_fraction
    )
    ready = (
        len(mismatch_queries) >= mismatch_min
        and proposal_error is None
        and exact_contract_pass
        and generation_quality_pass
    )
    if ready:
        status = "READY_FOR_FROZEN_V3_PROPOSAL_AND_ANNOTATION"
    elif args.reserve_included:
        status = "FAIL_YIELD_AFTER_FROZEN_RESERVE"
    else:
        status = "ADD_FROZEN_RESERVE_BEFORE_ANNOTATION"
    report = {
        "schema_version": "clir-smoke-acquisition-readiness-v3",
        "status": status,
        "reserve_included": bool(args.reserve_included),
        "population": population,
        "mechanism_eligible_numeric_mismatch_queries": len(mismatch_queries),
        "mechanism_eligible_numeric_mismatch_query_ids_sha256": canonical_sha256(
            sorted(mismatch_queries)
        ),
        "required_mismatch_queries": mismatch_min,
        "exact_token_contract": {
            "pass": exact_contract_pass,
            "unitization_failures": unitization_failures,
            "required_pass_rate": protocol["preregistered_gates"][
                "data_and_exact_token_contract_pass_rate"
            ],
        },
        "generation_quality": {
            "pass": generation_quality_pass,
            "truncated": truncated,
            "empty_responses": empty_responses,
            "unitization_failures": unitization_failures,
            "unique_failure_rows": len(generation_contract_failure_ids),
            "failure_fraction": generation_contract_failure_fraction,
            "maximum_fraction": maximum_generation_failure_fraction,
            "checker_ineligible_rows_reported_separately": population[
                "ineligible_rows"
            ],
        },
        "proposal_counts": proposal_counts,
        "proposal_error": proposal_error,
        "annotation_allowed": ready,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


def _simple_units(texts: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    trajectory = "\n".join(texts)
    units: list[dict[str, Any]] = []
    cursor = 0
    for index, text in enumerate(texts):
        units.append({"unit_index": index, "kind": "material_claim", "text": text})
        cursor += len(text) + 1
    return trajectory, units


def _protocol_controls(
    task: str, count: int, *, control_version: str = "v3"
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Create balanced, synthetic comprehension controls that are never trained."""

    controls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index in range(count):
        item_id = stable_priority("clir-hidden-control-v2", task, index)
        if task == "consistency":
            if index % 2 == 0:
                left_text = "First take 2 apples. Then add 3 apples. The total is 5."
                right_text = (
                    "We start from 2 apples. Receiving 3 more means addition. "
                    "The calculation is 2+3=5, so the total is 5."
                )
                decision = "accept"
            else:
                # An exact copy violates the natural C near-copy rule even though
                # its mathematical claims match.
                left_text = right_text = "Add 2 and 3 to obtain 5."
                decision = "reject"
            item = {
                "item_id": item_id,
                "query_id": f"control:consistency:{index}",
                "problem": "Mina has 2 apples and receives 3 more. How many apples?",
                "left": {"id": f"{item_id}:left", "trajectory": left_text, "units": []},
                "right": {
                    "id": f"{item_id}:right",
                    "trajectory": right_text,
                    "units": [],
                },
            }
            expected = {
                "item_id": item_id,
                "decision": decision,
                "confidence": "high",
                "rationale": "hidden protocol comprehension control",
            }
        elif task == "hallucination":
            hallucinated = index % 2 == 1
            texts = [
                "Mina starts with 2 apples.",
                (
                    "The problem gives her 30 more apples."
                    if hallucinated
                    else "The problem gives her 3 more apples."
                ),
                "The requested operation is addition.",
                (
                    "The final total is 32 apples."
                    if hallucinated
                    else "The final total is 5 apples."
                ),
            ]
            trajectory, units = _simple_units(texts)
            item = {
                "item_id": item_id,
                "query_id": f"control:hallucination:{index}",
                "source": "synthetic-control",
                "problem": "Mina has 2 apples and receives 3 more. How many apples?",
                "trajectory": trajectory,
                "units": units,
                "output_token_ids_sha256": stable_priority(
                    "control-token-axis", item_id
                ),
            }
            expected = {
                "item_id": item_id,
                "status": "hallucinated" if hallucinated else "clean",
                "first_bad_unit_index": 1 if hallucinated else None,
                "confidence": "high",
                "rationale": "hidden protocol comprehension control",
            }
        elif task == "prior":
            usable = index % 2 == 0
            control_problem = "Mina has 2 apples and receives 3 more. How many apples?"
            if usable:
                eligibility = "usable"
                if control_version == "v2":
                    texts = [
                        "Mina starts with 2 apples.",
                        "She receives 3 more apples.",
                        "Adding gives 2+3=5.",
                        "Therefore the answer is 5 apples.",
                    ]
                    key, complete = [2, 3], [0, 1, 2, 3]
                elif control_version == "v3":
                    control_problem = (
                        "Mina has 2 apples, receives 3 more, and then gives "
                        "1 apple away. How many apples remain?"
                    )
                    texts = [
                        "Mina starts with 2 apples.",
                        "She receives 3 more apples and then gives 1 away.",
                        "The intermediate total is 2+3=5.",
                        "After giving one away, 5-1=4 apples remain.",
                        "Therefore the answer is 4 apples.",
                    ]
                    key, complete = [3], [2, 3]
                else:
                    raise ValueError(f"unsupported control_version {control_version!r}")
            else:
                texts = ["I cannot solve this problem."]
                eligibility = "no_auditable_reasoning"
                key, complete = [], []
            trajectory, units = _simple_units(texts)
            item = {
                "item_id": item_id,
                "query_id": f"control:prior:{index}",
                "source": "synthetic-control",
                "problem": control_problem,
                "trajectory": trajectory,
                "units": units,
                "output_token_ids_sha256": stable_priority(
                    "control-token-axis", item_id
                ),
            }
            expected = {
                "item_id": item_id,
                "eligibility": eligibility,
                "key_unit_indices": key,
                "complete_unit_indices": complete,
                "confidence": "high",
                "rationale": "hidden protocol comprehension control",
            }
        else:
            raise ValueError(f"unsupported task {task!r}")
        controls.append((item, validate_annotation(task, expected, item)))
    return controls


def build_annotation_packages(
    natural_items: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    control_version: str = "v3",
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    packages: dict[str, dict[str, list[dict[str, Any]]]] = {"a": {}, "b": {}}
    private: dict[str, Any] = {
        "schema_version": "clir-blind-annotation-package-v2",
        "warning": "PRIVATE: never send this file to annotators",
        "tasks": {},
    }
    for task in ("consistency", "hallucination", "prior"):
        natural = [dict(item) for item in natural_items[task]]
        natural_ids = [str(item["item_id"]) for item in natural]
        if len(natural_ids) != len(set(natural_ids)):
            raise ValueError(f"{task}: natural item IDs are not unique")
        control_count = max(1, math.ceil(len(natural) * 0.10))
        controls = _protocol_controls(
            task, control_count, control_version=control_version
        )
        control_items = [item for item, _ in controls]
        repeat_count = max(1, math.ceil(len(natural) * 0.20))
        repeat_sources = sorted(
            natural,
            key=lambda item: stable_priority(
                "clir-self-repeat-source-v2", task, item["item_id"]
            ),
        )[:repeat_count]
        repeats: list[dict[str, Any]] = []
        repeat_map: list[dict[str, str]] = []
        for item in repeat_sources:
            repeated = dict(item)
            repeat_id = stable_priority("clir-self-repeat-v2", task, item["item_id"])
            repeated["item_id"] = repeat_id
            repeats.append(repeated)
            repeat_map.append(
                {"original_item_id": str(item["item_id"]), "repeat_item_id": repeat_id}
            )
        package_a = [*natural, *control_items, *repeats]
        package_b = [*natural, *control_items]
        package_a.sort(
            key=lambda item: stable_priority("clir-package-a-v2", task, item["item_id"])
        )
        package_b.sort(
            key=lambda item: stable_priority("clir-package-b-v2", task, item["item_id"])
        )
        packages["a"][task] = package_a
        packages["b"][task] = package_b
        private["tasks"][task] = {
            "control_version": control_version,
            "natural_item_ids": natural_ids,
            "controls": [
                {"item_id": item["item_id"], "expected_annotation": expected}
                for item, expected in controls
            ],
            "self_repeats_a": repeat_map,
            "package_a_ids_sha256": canonical_sha256(
                [item["item_id"] for item in package_a]
            ),
            "package_b_ids_sha256": canonical_sha256(
                [item["item_id"] for item in package_b]
            ),
        }
    return packages, private


def command_package(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    items_dir = Path(args.items_dir)
    natural = {
        "consistency": read_jsonl(items_dir / "annotation_consistency_natural.jsonl"),
        "hallucination": read_jsonl(
            items_dir / "annotation_hallucination_natural.jsonl"
        ),
        "prior": read_jsonl(items_dir / "annotation_prior_natural.jsonl"),
    }
    packages, private = build_annotation_packages(
        natural,
        control_version=str(protocol.get("annotation_protocol_version", "v2")),
    )
    output_dir = Path(args.output_dir)
    for annotator in ("a", "b"):
        for task, rows in packages[annotator].items():
            publish_manifest(
                output_dir / f"annotator_{annotator}" / f"{task}.jsonl",
                rows,
                schema_version=f"clir-{task}-blind-package-v2",
                metadata={"annotator_slot": annotator},
            )
    atomic_write_json(output_dir / "PRIVATE_package_manifest.json", private)
    print(
        json.dumps(
            {
                "status": "packages_ready",
                "send_only": ["annotator_a/", "annotator_b/"],
                "never_send": "PRIVATE_package_manifest.json",
                "counts": {
                    slot: {task: len(rows) for task, rows in task_rows.items()}
                    for slot, task_rows in packages.items()
                },
            },
            indent=2,
        )
    )


def evaluate_package_reliability(
    *,
    task: str,
    private_task: Mapping[str, Any],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_a = {str(row["item_id"]): row for row in labels_a}
    by_b = {str(row["item_id"]): row for row in labels_b}
    controls = private_task["controls"]
    control_results: dict[str, Any] = {}
    for slot, labels in (("a", by_a), ("b", by_b)):
        correct = 0
        for control in controls:
            item_id = str(control["item_id"])
            if item_id in labels and annotation_signature(
                task, labels[item_id]
            ) == annotation_signature(task, control["expected_annotation"]):
                correct += 1
        control_results[slot] = {
            "correct": correct,
            "total": len(controls),
            "accuracy": correct / len(controls) if controls else None,
        }
    repeats = private_task["self_repeats_a"]
    self_matches = sum(
        mapping["original_item_id"] in by_a
        and mapping["repeat_item_id"] in by_a
        and annotation_signature(task, by_a[mapping["original_item_id"]])
        == annotation_signature(task, by_a[mapping["repeat_item_id"]])
        for mapping in repeats
    )
    return {
        "hidden_controls": control_results,
        "annotator_a_self_agreement": {
            "agree": self_matches,
            "total": len(repeats),
            "rate": self_matches / len(repeats) if repeats else None,
        },
    }


def evaluate_raw_annotation_gates(
    *,
    protocol: Mapping[str, Any],
    natural_items: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    task_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every pre-adjudication gate that is already decidable."""

    thresholds = protocol["preregistered_gates"]
    results: list[dict[str, Any]] = []

    def add(
        name: str,
        value: float | int | None,
        operator: str,
        threshold: float | int,
        **counts: Any,
    ) -> None:
        if value is None:
            status = "FAIL"
        elif operator == ">=":
            status = "PASS" if value >= threshold else "FAIL"
        elif operator == "<=":
            status = "PASS" if value <= threshold else "FAIL"
        else:
            raise ValueError(f"unsupported raw gate operator {operator}")
        results.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "status": status,
                **counts,
            }
        )

    for task in ("consistency", "hallucination", "prior"):
        reliability = task_reports[task]["package_reliability"]
        for slot in ("a", "b"):
            control = reliability["hidden_controls"][slot]
            add(
                f"{task}_hidden_control_accuracy_{slot}",
                control["accuracy"],
                ">=",
                thresholds["hidden_control_accuracy_per_annotator_per_task"],
                numerator=control["correct"],
                denominator=control["total"],
            )
        repeat = reliability["annotator_a_self_agreement"]
        add(
            f"{task}_annotator_a_self_agreement",
            repeat["rate"],
            ">=",
            thresholds["annotator_a_decision_self_agreement_min"],
            numerator=repeat["agree"],
            denominator=repeat["total"],
        )

    c_raw = task_reports["consistency"]["agreement"]
    add(
        "consistency_decision_agreement",
        c_raw["exact_target_agreement"],
        ">=",
        thresholds["consistency_decision_agreement_min"],
        numerator=c_raw["exact_target_agree"],
        denominator=c_raw["items"],
    )
    add(
        "consistency_required_adjudication_fraction_lower_bound",
        c_raw["pre_adjudication_fraction"],
        "<=",
        thresholds["consistency_adjudication_fraction_max"],
    )
    if _consistency_kappa_applicable(c_raw):
        add(
            "consistency_kappa",
            c_raw["decision_kappa"],
            ">=",
            thresholds["consistency_kappa_min_when_each_class_has_at_least_five_calls"],
        )
    else:
        results.append(
            {
                "name": "consistency_kappa",
                "value": c_raw["decision_kappa"],
                "status": "NOT_APPLICABLE",
                "reason": "one decision class has fewer than five calls",
            }
        )

    h_raw = task_reports["hallucination"]["agreement"]
    for name, threshold_name in (
        ("hallucination_path_agreement", "hallucination_path_agreement_min"),
        (
            "hallucination_positive_specific_agreement",
            "hallucination_positive_specific_agreement_min",
        ),
        (
            "hallucination_clean_specific_agreement",
            "hallucination_clean_specific_agreement_min",
        ),
    ):
        raw_field = {
            "hallucination_path_agreement": "raw_path_agreement",
            "hallucination_positive_specific_agreement": (
                "positive_specific_agreement"
            ),
            "hallucination_clean_specific_agreement": "clean_specific_agreement",
        }[name]
        add(name, h_raw[raw_field], ">=", thresholds[threshold_name])
    add(
        "hallucination_positive_rate_absolute_gap",
        h_raw["positive_rate_absolute_gap"],
        "<=",
        thresholds["hallucination_positive_rate_absolute_gap_max"],
    )
    add(
        "hallucination_kappa",
        h_raw["path_kappa"],
        ">=",
        thresholds["hallucination_kappa_min_when_both_classes_have_support"],
    )
    add(
        "hallucination_common_positive_count",
        h_raw["common_positive"],
        ">=",
        thresholds["hallucination_common_positive_count_min"],
    )
    h_items = {str(item["item_id"]): item for item in natural_items["hallucination"]}
    h_a = {str(row["item_id"]): row for row in labels["a"]["hallucination"]}
    h_b = {str(row["item_id"]): row for row in labels["b"]["hallucination"]}
    common_positive_five = [
        item_id
        for item_id, item in h_items.items()
        if sum(unit["kind"] == "material_claim" for unit in item["units"]) >= 5
        and h_a[item_id]["status"] == h_b[item_id]["status"] == "hallucinated"
    ]
    exact_five = sum(
        h_a[item_id]["first_bad_unit_index"] == h_b[item_id]["first_bad_unit_index"]
        for item_id in common_positive_five
    )
    plus_one_five = sum(
        abs(
            int(h_a[item_id]["first_bad_unit_index"])
            - int(h_b[item_id]["first_bad_unit_index"])
        )
        <= 1
        for item_id in common_positive_five
    )
    add(
        "hallucination_exact_onset_unit_agreement_on_five_plus_unit_rows",
        exact_five / len(common_positive_five) if common_positive_five else None,
        ">=",
        thresholds[
            "hallucination_exact_onset_unit_agreement_min_on_five_plus_unit_rows"
        ],
        numerator=exact_five,
        denominator=len(common_positive_five),
    )
    add(
        "hallucination_plus_minus_one_unit_agreement",
        plus_one_five / len(common_positive_five) if common_positive_five else None,
        ">=",
        thresholds["hallucination_plus_minus_one_unit_agreement_min"],
        numerator=plus_one_five,
        denominator=len(common_positive_five),
    )
    add(
        "hallucination_required_adjudication_fraction_lower_bound",
        h_raw["pre_adjudication_fraction"],
        "<=",
        thresholds["hallucination_adjudication_fraction_max"],
    )

    p_raw = task_reports["prior"]["agreement"]
    add(
        "prior_eligibility_agreement",
        p_raw["eligibility_agreement"],
        ">=",
        thresholds["prior_eligibility_agreement_min"],
        numerator=p_raw["eligibility_agree"],
        denominator=p_raw["items"],
    )
    add(
        "prior_joint_usable_overlap",
        p_raw["usable_overlap"],
        ">=",
        thresholds["prior_joint_usable_overlap_min"],
    )
    add(
        "key_macro_unit_f1",
        p_raw["key_macro_f1"],
        ">=",
        thresholds["key_macro_unit_f1_min"],
    )
    add(
        "complete_macro_unit_f1",
        p_raw["complete_macro_f1"],
        ">=",
        thresholds["complete_macro_unit_f1_min"],
    )
    p_items = {str(item["item_id"]): item for item in natural_items["prior"]}
    for slot in ("a", "b"):
        usable = [
            row for row in labels[slot]["prior"] if row["eligibility"] == "usable"
        ]
        all_selected = sum(
            set(row["complete_unit_indices"])
            == {
                unit["unit_index"]
                for unit in p_items[str(row["item_id"])]["units"]
                if unit["kind"] == "material_claim"
            }
            for row in usable
        )
        add(
            f"complete_equals_all_material_units_fraction_{slot}",
            all_selected / len(usable) if usable else None,
            "<=",
            thresholds["complete_equals_all_material_units_fraction_max_per_annotator"],
            numerator=all_selected,
            denominator=len(usable),
        )
    add(
        "prior_required_adjudication_fraction_lower_bound",
        p_raw["pre_adjudication_fraction"],
        "<=",
        thresholds["prior_adjudication_fraction_max"],
    )

    failed = [row["name"] for row in results if row["status"] == "FAIL"]
    return {
        "status": (
            "PASS_RAW_ANNOTATION_GATES" if not failed else "STOP_RAW_GATE_FAILURE"
        ),
        "failed_gate_names": failed,
        "third_model_send_allowed": not failed,
        "adjudication_cannot_rescue_raw_failure": True,
        "results": results,
    }


def command_triage(args: argparse.Namespace) -> None:
    """Freeze third-model independent audit/dispute items after A/B complete."""

    items_dir = Path(args.items_dir)
    package_dir = Path(args.package_dir)
    labels_a_dir = Path(args.labels_a_dir)
    labels_b_dir = Path(args.labels_b_dir)
    private_package = json.loads(
        (package_dir / "PRIVATE_package_manifest.json").read_text(encoding="utf-8")
    )
    item_names = {
        "consistency": "annotation_consistency_natural.jsonl",
        "hallucination": "annotation_hallucination_natural.jsonl",
        "prior": "annotation_prior_natural.jsonl",
    }
    output_dir = Path(args.output_dir)
    private_triage: dict[str, Any] = {
        "schema_version": "clir-third-model-triage-v2",
        "warning": "PRIVATE: do not send this manifest to the third model",
        "tasks": {},
    }
    public_counts: dict[str, Any] = {}
    public_raw_report: dict[str, Any] = {
        "schema_version": "clir-raw-annotation-report-v2",
        "interpretation": (
            "raw A/B agreement and package reliability before third-model "
            "audit or adjudication; agreement is not accuracy"
        ),
        "tasks": {},
    }
    natural_items_by_task: dict[str, list[dict[str, Any]]] = {}
    natural_labels: dict[str, dict[str, list[dict[str, Any]]]] = {
        "a": {},
        "b": {},
    }
    for task, item_name in item_names.items():
        natural_items = read_jsonl(items_dir / item_name)
        natural_items_by_task[task] = natural_items
        natural_by_id = {str(item["item_id"]): item for item in natural_items}
        package_a = read_jsonl(package_dir / "annotator_a" / f"{task}.jsonl")
        package_b = read_jsonl(package_dir / "annotator_b" / f"{task}.jsonl")
        all_a = _validate_label_file(task, labels_a_dir / f"{task}.jsonl", package_a)
        all_b = _validate_label_file(task, labels_b_dir / f"{task}.jsonl", package_b)
        natural_ids = set(private_package["tasks"][task]["natural_item_ids"])
        by_a = {
            str(row["item_id"]): row
            for row in all_a
            if str(row["item_id"]) in natural_ids
        }
        by_b = {
            str(row["item_id"]): row
            for row in all_b
            if str(row["item_id"]) in natural_ids
        }
        natural_labels["a"][task] = list(by_a.values())
        natural_labels["b"][task] = list(by_b.values())
        auto_agree = [
            item_id
            for item_id in natural_ids
            if annotation_signature(task, by_a[item_id])
            == annotation_signature(task, by_b[item_id])
            and by_a[item_id]["confidence"] != "low"
            and by_b[item_id]["confidence"] != "low"
        ]
        disputes = sorted(natural_ids - set(auto_agree))
        audit_count = math.ceil(len(auto_agree) * 0.15)
        audit_ids = sorted(
            auto_agree,
            key=lambda item_id: stable_priority("clir-third-audit-v2", task, item_id),
        )[:audit_count]
        third_ids = sorted(
            set(audit_ids) | set(disputes),
            key=lambda item_id: stable_priority("clir-third-package-v2", task, item_id),
        )
        third_items = [natural_by_id[item_id] for item_id in third_ids]
        publish_manifest(
            output_dir / "third_independent" / f"{task}.jsonl",
            third_items,
            schema_version=f"clir-{task}-third-independent-package-v2",
        )
        private_triage["tasks"][task] = {
            "natural_item_ids": sorted(natural_ids),
            "auto_agree_audit_item_ids": audit_ids,
            "dispute_item_ids": disputes,
            "third_package_item_ids": third_ids,
        }
        public_counts[task] = {
            "natural": len(natural_ids),
            "auto_agree": len(auto_agree),
            "independent_auto_agree_audit": len(audit_ids),
            "needs_blind_adjudication": len(disputes),
            "third_independent_total": len(third_ids),
        }
        raw = agreement_report(task, list(by_a.values()), list(by_b.values()))
        raw.update(
            {
                "auto_agree_nonlow": len(auto_agree),
                "pre_adjudication_fraction": (
                    len(disputes) / len(natural_ids) if natural_ids else None
                ),
                "low_confidence_a": sum(
                    row["confidence"] == "low" for row in by_a.values()
                ),
                "low_confidence_b": sum(
                    row["confidence"] == "low" for row in by_b.values()
                ),
            }
        )
        public_raw_report["tasks"][task] = {
            "agreement": raw,
            "package_reliability": evaluate_package_reliability(
                task=task,
                private_task=private_package["tasks"][task],
                labels_a=all_a,
                labels_b=all_b,
            ),
        }
    protocol = load_protocol(getattr(args, "protocol", DEFAULT_PROTOCOL))
    raw_gate_decision = evaluate_raw_annotation_gates(
        protocol=protocol,
        natural_items=natural_items_by_task,
        labels=natural_labels,
        task_reports=public_raw_report["tasks"],
    )
    public_raw_report["raw_gate_decision"] = raw_gate_decision
    atomic_write_json(output_dir / "PRIVATE_triage_manifest.json", private_triage)
    atomic_write_json(output_dir / "triage_counts.json", public_counts)
    atomic_write_json(output_dir / "raw_annotation_report.json", public_raw_report)
    atomic_write_json(output_dir / "raw_gate_decision.json", raw_gate_decision)
    send_allowed = raw_gate_decision["third_model_send_allowed"]
    print(
        json.dumps(
            {
                "status": (
                    "third_independent_packages_ready"
                    if send_allowed
                    else "STOP_RAW_GATE_FAILURE"
                ),
                "third_model_send_allowed": send_allowed,
                "send_only": "third_independent/" if send_allowed else None,
                "do_not_send_reason": (
                    None
                    if send_allowed
                    else "pre-registered raw annotation gates failed"
                ),
                "never_send": "PRIVATE_triage_manifest.json",
                "failed_gate_names": raw_gate_decision["failed_gate_names"],
                "counts": public_counts,
            },
            indent=2,
        )
    )


def command_adjudication_package(args: argparse.Namespace) -> None:
    """Build anonymous A/B comparison packets after the third model answers alone."""

    items_dir = Path(args.items_dir)
    package_dir = Path(args.package_dir)
    labels_a_dir = Path(args.labels_a_dir)
    labels_b_dir = Path(args.labels_b_dir)
    triage_dir = Path(args.triage_dir)
    third_labels_dir = Path(args.third_independent_labels_dir)
    output_dir = Path(args.output_dir)
    private_package = json.loads(
        (package_dir / "PRIVATE_package_manifest.json").read_text(encoding="utf-8")
    )
    private_triage = json.loads(
        (triage_dir / "PRIVATE_triage_manifest.json").read_text(encoding="utf-8")
    )
    item_names = {
        "consistency": "annotation_consistency_natural.jsonl",
        "hallucination": "annotation_hallucination_natural.jsonl",
        "prior": "annotation_prior_natural.jsonl",
    }
    private_options: dict[str, Any] = {
        "schema_version": "clir-anonymous-adjudication-options-v2",
        "warning": "PRIVATE: never send this option identity map to the adjudicator",
        "tasks": {},
    }
    counts: dict[str, int] = {}
    for task, item_name in item_names.items():
        natural_items = read_jsonl(items_dir / item_name)
        natural_by_id = {str(item["item_id"]): item for item in natural_items}
        package_a = read_jsonl(package_dir / "annotator_a" / f"{task}.jsonl")
        package_b = read_jsonl(package_dir / "annotator_b" / f"{task}.jsonl")
        all_a = _validate_label_file(task, labels_a_dir / f"{task}.jsonl", package_a)
        all_b = _validate_label_file(task, labels_b_dir / f"{task}.jsonl", package_b)
        natural_ids = set(private_package["tasks"][task]["natural_item_ids"])
        by_a = {
            str(row["item_id"]): row
            for row in all_a
            if str(row["item_id"]) in natural_ids
        }
        by_b = {
            str(row["item_id"]): row
            for row in all_b
            if str(row["item_id"]) in natural_ids
        }
        third_items = read_jsonl(triage_dir / "third_independent" / f"{task}.jsonl")
        third_labels = _validate_label_file(
            task, third_labels_dir / f"{task}.jsonl", third_items
        )
        third_by_id = {str(row["item_id"]): row for row in third_labels}
        disputes = private_triage["tasks"][task]["dispute_item_ids"]
        packets: list[dict[str, Any]] = []
        option_map: dict[str, Any] = {}
        for item_id in disputes:
            swap = (
                int(stable_priority("clir-adjudication-order-v2", task, item_id), 16)
                % 2
            )
            ordered_slots = ("b", "a") if swap else ("a", "b")
            slot_labels = {"a": by_a[item_id], "b": by_b[item_id]}
            packets.append(
                {
                    "item_id": item_id,
                    "task": task,
                    "item": natural_by_id[item_id],
                    "independent_annotation": third_by_id[item_id],
                    "independent_annotation_sha256": canonical_sha256(
                        third_by_id[item_id]
                    ),
                    "anonymous_primary_proposals": [
                        {
                            "option": "option_1",
                            "annotation": slot_labels[ordered_slots[0]],
                        },
                        {
                            "option": "option_2",
                            "annotation": slot_labels[ordered_slots[1]],
                        },
                    ],
                }
            )
            option_map[item_id] = {
                "option_1": ordered_slots[0],
                "option_2": ordered_slots[1],
            }
        packets.sort(
            key=lambda row: stable_priority(
                "clir-adjudication-package-v2", task, row["item_id"]
            )
        )
        publish_manifest(
            output_dir / "adjudicator" / f"{task}.jsonl",
            packets,
            schema_version=f"clir-{task}-anonymous-adjudication-package-v2",
        )
        private_options["tasks"][task] = {"option_identity": option_map}
        counts[task] = len(packets)
    atomic_write_json(
        output_dir / "PRIVATE_adjudication_option_map.json", private_options
    )
    print(
        json.dumps(
            {
                "status": "anonymous_adjudication_packages_ready",
                "send_only": "adjudicator/",
                "never_send": "PRIVATE_adjudication_option_map.json",
                "counts": counts,
            },
            indent=2,
        )
    )


def evaluate_preregistered_gates(
    *,
    protocol: Mapping[str, Any],
    processed: Sequence[Mapping[str, Any]],
    natural_items: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    resolved: Mapping[str, Sequence[Mapping[str, Any]]],
    reports: Mapping[str, Any],
    final_consistency: Sequence[Mapping[str, Any]],
    final_hp: Sequence[Mapping[str, Any]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    thresholds = protocol["preregistered_gates"]
    results: list[dict[str, Any]] = []

    def add(
        name: str,
        value: float | int | None,
        operator: str,
        threshold: float | int,
        **counts: Any,
    ) -> None:
        if value is None:
            status = "FAIL"
        elif operator == ">=":
            status = "PASS" if value >= threshold else "FAIL"
        elif operator == "<=":
            status = "PASS" if value <= threshold else "FAIL"
        elif operator == "==":
            status = "PASS" if value == threshold else "FAIL"
        else:
            raise ValueError(f"unsupported gate operator {operator}")
        results.append(
            {
                "name": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "status": status,
                **counts,
            }
        )

    exact_ok = sum(row.get("unitization_status") == "ok" for row in processed)
    add(
        "data_and_exact_token_contract_pass_rate",
        exact_ok / len(processed) if processed else None,
        ">=",
        thresholds["data_and_exact_token_contract_pass_rate"],
        numerator=exact_ok,
        denominator=len(processed),
    )
    for task in ("consistency", "hallucination", "prior"):
        reliability = reports[f"{task}_reliability"]
        for slot in ("a", "b"):
            control = reliability["hidden_controls"][slot]
            add(
                f"{task}_hidden_control_accuracy_{slot}",
                control["accuracy"],
                ">=",
                thresholds["hidden_control_accuracy_per_annotator_per_task"],
                numerator=control["correct"],
                denominator=control["total"],
            )
        repeat = reliability["annotator_a_self_agreement"]
        add(
            f"{task}_annotator_a_self_agreement",
            repeat["rate"],
            ">=",
            thresholds["annotator_a_decision_self_agreement_min"],
            numerator=repeat["agree"],
            denominator=repeat["total"],
        )

    c_raw = reports["consistency_raw"]
    c_resolution = reports["consistency_resolution"]
    add(
        "consistency_decision_agreement",
        c_raw["exact_target_agreement"],
        ">=",
        thresholds["consistency_decision_agreement_min"],
        numerator=c_raw["exact_target_agree"],
        denominator=c_raw["items"],
    )
    add(
        "consistency_adjudication_fraction",
        c_resolution["adjudication_fraction"],
        "<=",
        thresholds["consistency_adjudication_fraction_max"],
    )
    add(
        "consistency_final_accepts",
        len(final_consistency),
        ">=",
        thresholds["consistency_final_accepts_min"],
    )
    if _consistency_kappa_applicable(c_raw):
        add(
            "consistency_kappa",
            c_raw["decision_kappa"],
            ">=",
            thresholds["consistency_kappa_min_when_each_class_has_at_least_five_calls"],
        )
    else:
        results.append(
            {
                "name": "consistency_kappa",
                "value": c_raw["decision_kappa"],
                "status": "NOT_APPLICABLE",
                "reason": "one decision class has fewer than five calls",
            }
        )

    h_raw = reports["hallucination_raw"]
    h_resolution = reports["hallucination_resolution"]
    for name, value, threshold_name in (
        (
            "hallucination_path_agreement",
            h_raw["raw_path_agreement"],
            "hallucination_path_agreement_min",
        ),
        (
            "hallucination_positive_specific_agreement",
            h_raw["positive_specific_agreement"],
            "hallucination_positive_specific_agreement_min",
        ),
        (
            "hallucination_clean_specific_agreement",
            h_raw["clean_specific_agreement"],
            "hallucination_clean_specific_agreement_min",
        ),
    ):
        add(name, value, ">=", thresholds[threshold_name])
    add(
        "hallucination_positive_rate_absolute_gap",
        h_raw["positive_rate_absolute_gap"],
        "<=",
        thresholds["hallucination_positive_rate_absolute_gap_max"],
    )
    add(
        "hallucination_kappa",
        h_raw["path_kappa"],
        ">=",
        thresholds["hallucination_kappa_min_when_both_classes_have_support"],
    )
    add(
        "hallucination_common_positive_count",
        h_raw["common_positive"],
        ">=",
        thresholds["hallucination_common_positive_count_min"],
    )
    h_items = {str(item["item_id"]): item for item in natural_items["hallucination"]}
    h_a = {str(row["item_id"]): row for row in labels["a"]["hallucination"]}
    h_b = {str(row["item_id"]): row for row in labels["b"]["hallucination"]}
    common_positive_five = []
    for item_id, item in h_items.items():
        material_count = sum(unit["kind"] == "material_claim" for unit in item["units"])
        if (
            material_count >= 5
            and h_a[item_id]["status"] == h_b[item_id]["status"] == "hallucinated"
        ):
            common_positive_five.append(item_id)
    exact_five = sum(
        h_a[item_id]["first_bad_unit_index"] == h_b[item_id]["first_bad_unit_index"]
        for item_id in common_positive_five
    )
    plus_one_five = sum(
        abs(
            int(h_a[item_id]["first_bad_unit_index"])
            - int(h_b[item_id]["first_bad_unit_index"])
        )
        <= 1
        for item_id in common_positive_five
    )
    add(
        "hallucination_exact_onset_unit_agreement_on_five_plus_unit_rows",
        exact_five / len(common_positive_five) if common_positive_five else None,
        ">=",
        thresholds[
            "hallucination_exact_onset_unit_agreement_min_on_five_plus_unit_rows"
        ],
        numerator=exact_five,
        denominator=len(common_positive_five),
    )
    add(
        "hallucination_plus_minus_one_unit_agreement",
        plus_one_five / len(common_positive_five) if common_positive_five else None,
        ">=",
        thresholds["hallucination_plus_minus_one_unit_agreement_min"],
        numerator=plus_one_five,
        denominator=len(common_positive_five),
    )
    add(
        "hallucination_adjudication_fraction",
        h_resolution["adjudication_fraction"],
        "<=",
        thresholds["hallucination_adjudication_fraction_max"],
    )

    p_raw = reports["prior_raw"]
    p_resolution = reports["prior_resolution"]
    add(
        "prior_eligibility_agreement",
        p_raw["eligibility_agreement"],
        ">=",
        thresholds["prior_eligibility_agreement_min"],
        numerator=p_raw["eligibility_agree"],
        denominator=p_raw["items"],
    )
    add(
        "prior_joint_usable_overlap",
        p_raw["usable_overlap"],
        ">=",
        thresholds["prior_joint_usable_overlap_min"],
    )
    add(
        "key_macro_unit_f1",
        p_raw["key_macro_f1"],
        ">=",
        thresholds["key_macro_unit_f1_min"],
    )
    add(
        "complete_macro_unit_f1",
        p_raw["complete_macro_f1"],
        ">=",
        thresholds["complete_macro_unit_f1_min"],
    )
    p_items = {str(item["item_id"]): item for item in natural_items["prior"]}
    for slot in ("a", "b"):
        usable = [
            row for row in labels[slot]["prior"] if row["eligibility"] == "usable"
        ]
        all_selected = 0
        for row in usable:
            material = {
                unit["unit_index"]
                for unit in p_items[str(row["item_id"])]["units"]
                if unit["kind"] == "material_claim"
            }
            all_selected += set(row["complete_unit_indices"]) == material
        add(
            f"complete_equals_all_material_units_fraction_{slot}",
            all_selected / len(usable) if usable else None,
            "<=",
            thresholds["complete_equals_all_material_units_fraction_max_per_annotator"],
            numerator=all_selected,
            denominator=len(usable),
        )
    add(
        "prior_adjudication_fraction",
        p_resolution["adjudication_fraction"],
        "<=",
        thresholds["prior_adjudication_fraction_max"],
    )

    final_statuses = Counter(row["h_status"] for row in final_hp)
    add(
        "final_positive_onset",
        final_statuses["hallucinated"],
        "==",
        thresholds["final_positive_onset"],
    )
    add(
        "final_explicit_clean",
        final_statuses["clean"],
        "==",
        thresholds["final_explicit_clean"],
    )
    add(
        "final_consistency_accepts",
        len(final_consistency),
        "==",
        thresholds["final_consistency_accepts"],
    )
    hp_cfg = protocol["proposal_manifests"]["hallucination_and_prior"]
    source_minima = hp_cfg.get("minimum_source_by_class")
    if source_minima is None:
        minimum_source = int(thresholds["minimum_each_source_per_final_h_class"])
        source_minima = {
            status: {"gsm8k": minimum_source, "asdiv-a": minimum_source}
            for status in ("hallucinated", "clean")
        }
    for status in ("hallucinated", "clean"):
        for source, minimum_source in sorted(source_minima[status].items()):
            count = sum(
                row["h_status"] == status and row["source"] == source
                for row in final_hp
            )
            add(
                f"final_{status}_{source}",
                count,
                ">=",
                int(minimum_source),
            )
    final_unit_counts = [
        sum(
            unit["kind"] == "material_claim"
            for unit in rows_by_id[str(row["id"])]["units"]
        )
        for row in final_hp
    ]
    add(
        "final_h_prior_median_material_units",
        statistics.median(final_unit_counts) if final_unit_counts else None,
        ">=",
        protocol["unitizer"]["minimum_final_h_prior_median_material_units"],
    )
    resolved_h = {str(row["item_id"]): row for row in resolved["hallucination"]}
    edge_onsets = 0
    positive_rows = [row for row in final_hp if row["h_status"] == "hallucinated"]
    for proposal in positive_rows:
        material = [
            unit["unit_index"]
            for unit in rows_by_id[str(proposal["id"])]["units"]
            if unit["kind"] == "material_claim"
        ]
        onset = resolved_h[str(proposal["id"])]["first_bad_unit_index"]
        edge_onsets += onset in {material[0], material[-1]}
    add(
        "first_or_last_material_unit_onset_fraction",
        edge_onsets / len(positive_rows) if positive_rows else None,
        "<=",
        thresholds["first_or_last_material_unit_onset_fraction_max"],
        numerator=edge_onsets,
        denominator=len(positive_rows),
    )
    failed = [row["name"] for row in results if row["status"] == "FAIL"]
    return {
        "status": (
            "PASS_ALL_PREREGISTERED_GATES" if not failed else "FAIL_PIPELINE_GATES"
        ),
        "failed_gate_names": failed,
        "results": results,
    }


def _consistency_kappa_applicable(
    raw_report: Mapping[str, Any], *, minimum_per_class: int = 5
) -> bool:
    """Require both decision classes from both annotators before gating kappa."""

    return all(
        raw_report[f"{slot}_decisions"].get(decision, 0) >= minimum_per_class
        for slot in ("a", "b")
        for decision in ("accept", "reject")
    )


def _validate_label_file(
    task: str,
    path: str | Path,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_item = {str(item["item_id"]): item for item in items}
    labels = read_jsonl(path)
    label_ids = [str(label.get("item_id")) for label in labels]
    if (
        set(label_ids) != set(by_item)
        or len(label_ids) != len(by_item)
        or len(set(label_ids)) != len(label_ids)
    ):
        raise ValueError(f"{task}: label population differs from frozen natural items")
    return [
        validate_annotation(task, label, by_item[str(label["item_id"])])
        for label in labels
    ]


def _load_and_validate_adjudications(
    task: str,
    path: str | Path | None,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if path is None:
        return []
    by_item = {str(item["item_id"]): item for item in items}
    rows = read_jsonl(path)
    row_ids = [str(row.get("item_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{task}: adjudications contain duplicate item_id values")
    for row in rows:
        item_id = str(row.get("item_id"))
        if item_id not in by_item:
            raise ValueError(f"{task}: adjudication references an unknown item")
        if row.get("resolution") == "synthesize":
            row["annotation"] = validate_annotation(
                task, row.get("annotation", {}), by_item[item_id]
            )
    return rows


def command_finalize(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    annotators = roster.get("primary_annotators", [])
    validate_annotator_roster(annotators, generator_family="phi")
    adjudicator = roster.get("adjudicator")
    if not isinstance(adjudicator, Mapping):
        raise ValueError("roster requires a third adjudicator")
    validate_annotator_roster([*annotators[:2], adjudicator], generator_family="phi")
    families = {
        str(row["model_family"]).casefold() for row in [*annotators[:2], adjudicator]
    }
    if len(families) != 3:
        raise ValueError(
            "A, B, and adjudicator must use three different model families"
        )

    items_dir = Path(args.items_dir)
    labels_a_dir = Path(args.labels_a_dir)
    labels_b_dir = Path(args.labels_b_dir)
    adjudication_dir = Path(args.adjudication_dir) if args.adjudication_dir else None
    package_dir = Path(args.package_dir) if args.package_dir else None
    triage_dir = Path(args.triage_dir)
    third_labels_dir = Path(args.third_independent_labels_dir)
    adjudication_package_dir = Path(args.adjudication_package_dir)
    private_package = None
    if package_dir is not None:
        private_package = json.loads(
            (package_dir / "PRIVATE_package_manifest.json").read_text(encoding="utf-8")
        )
        if private_package.get("schema_version") != "clir-blind-annotation-package-v2":
            raise ValueError("blind package private manifest has the wrong schema")
    private_triage = json.loads(
        (triage_dir / "PRIVATE_triage_manifest.json").read_text(encoding="utf-8")
    )
    if private_triage.get("schema_version") != "clir-third-model-triage-v2":
        raise ValueError("third-model triage manifest has the wrong schema")
    private_options = json.loads(
        (adjudication_package_dir / "PRIVATE_adjudication_option_map.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        private_options.get("schema_version")
        != "clir-anonymous-adjudication-options-v2"
    ):
        raise ValueError("adjudication option map has the wrong schema")
    task_files = {
        "consistency": "annotation_consistency_natural.jsonl",
        "hallucination": "annotation_hallucination_natural.jsonl",
        "prior": "annotation_prior_natural.jsonl",
    }
    resolved: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    natural_items_by_task: dict[str, list[dict[str, Any]]] = {}
    natural_labels: dict[str, dict[str, list[dict[str, Any]]]] = {
        "a": {},
        "b": {},
    }
    for task, item_name in task_files.items():
        natural_items = read_jsonl(items_dir / item_name)
        natural_items_by_task[task] = natural_items
        if package_dir is None:
            labels_a = _validate_label_file(
                task, labels_a_dir / f"{task}.jsonl", natural_items
            )
            labels_b = _validate_label_file(
                task, labels_b_dir / f"{task}.jsonl", natural_items
            )
        else:
            package_items_a = read_jsonl(package_dir / "annotator_a" / f"{task}.jsonl")
            package_items_b = read_jsonl(package_dir / "annotator_b" / f"{task}.jsonl")
            all_labels_a = _validate_label_file(
                task, labels_a_dir / f"{task}.jsonl", package_items_a
            )
            all_labels_b = _validate_label_file(
                task, labels_b_dir / f"{task}.jsonl", package_items_b
            )
            task_private = private_package["tasks"][task]
            reports[f"{task}_reliability"] = evaluate_package_reliability(
                task=task,
                private_task=task_private,
                labels_a=all_labels_a,
                labels_b=all_labels_b,
            )
            natural_ids = set(task_private["natural_item_ids"])
            expected_ids = {str(item["item_id"]) for item in natural_items}
            if natural_ids != expected_ids:
                raise ValueError(
                    f"{task}: private natural IDs differ from frozen proposal items"
                )
            labels_a = [
                row for row in all_labels_a if str(row["item_id"]) in natural_ids
            ]
            labels_b = [
                row for row in all_labels_b if str(row["item_id"]) in natural_ids
            ]
        natural_labels["a"][task] = labels_a
        natural_labels["b"][task] = labels_b
        task_triage = private_triage["tasks"][task]
        third_items = read_jsonl(triage_dir / "third_independent" / f"{task}.jsonl")
        third_labels = _validate_label_file(
            task,
            third_labels_dir / f"{task}.jsonl",
            third_items,
        )
        third_by_id = {str(row["item_id"]): row for row in third_labels}
        expected_third_ids = set(task_triage["third_package_item_ids"])
        if set(third_by_id) != expected_third_ids:
            raise ValueError(
                f"{task}: third-model labels differ from frozen triage population"
            )
        by_a_natural = {str(row["item_id"]): row for row in labels_a}
        by_b_natural = {str(row["item_id"]): row for row in labels_b}
        audit_ids = task_triage["auto_agree_audit_item_ids"]
        stable = sum(
            annotation_signature(task, third_by_id[item_id])
            == annotation_signature(task, by_a_natural[item_id])
            == annotation_signature(task, by_b_natural[item_id])
            for item_id in audit_ids
        )
        reports[f"{task}_third_model"] = {
            "independent_labels": len(third_labels),
            "disputes_independently_answered": len(task_triage["dispute_item_ids"]),
            "auto_agree_audit_stable": stable,
            "auto_agree_audit_total": len(audit_ids),
            "auto_agree_audit_stability": (
                stable / len(audit_ids) if audit_ids else None
            ),
            "interpretation": "third-model stability, not natural-label accuracy",
        }
        adjudications = _load_and_validate_adjudications(
            task,
            adjudication_dir / f"{task}.jsonl" if adjudication_dir else None,
            natural_items,
        )
        dispute_ids = set(task_triage["dispute_item_ids"])
        for adjudication in adjudications:
            item_id = str(adjudication["item_id"])
            if item_id not in dispute_ids:
                raise ValueError(f"{task}: only frozen A/B disputes may be adjudicated")
            expected_independent_hash = canonical_sha256(third_by_id[item_id])
            if (
                adjudication.get("independent_annotation_sha256")
                != expected_independent_hash
            ):
                raise ValueError(
                    f"{task}: adjudication lacks the matching frozen independent-answer hash"
                )
            resolution = adjudication.get("resolution")
            if resolution in {"adopt_option_1", "adopt_option_2"}:
                option = str(resolution).removeprefix("adopt_")
                slot = private_options["tasks"][task]["option_identity"][item_id][
                    option
                ]
                adjudication["resolution"] = f"adopt_{slot}"
        reports[f"{task}_raw"] = agreement_report(task, labels_a, labels_b)
        resolved[task], reports[f"{task}_resolution"] = resolve_blind_labels(
            task=task,
            labels_a=labels_a,
            labels_b=labels_b,
            adjudications=adjudications,
            label_tier=str(protocol["annotation"]["label_tier"]),
        )

    processed = read_jsonl(args.processed)
    by_id = {str(row["id"]): dict(row) for row in processed}
    c_proposals = read_jsonl(items_dir / "consistency_proposals.jsonl")
    hp_proposals = read_jsonl(items_dir / "h_prior_proposals.jsonl")
    c_by_id = {str(row["item_id"]): row for row in resolved["consistency"]}
    accepted = [
        proposal
        for proposal in c_proposals
        if c_by_id.get(str(proposal["proposal_id"]), {}).get("decision") == "accept"
    ]
    final_c_count = int(protocol["preregistered_gates"]["final_consistency_accepts"])
    if len(accepted) < final_c_count:
        raise ValueError(
            f"FAIL_YIELD: only {len(accepted)} consistency proposals accepted"
        )
    final_c = accepted[:final_c_count]
    for proposal in final_c:
        for style, field in (("compact", "left_id"), ("expanded", "right_id")):
            row = by_id[str(proposal[field])]
            row["semantic_id"] = str(proposal["query_id"])
            row["style_id"] = style
            row["consistency_label_tier"] = str(protocol["annotation"]["label_tier"])

    hp_cfg = protocol["proposal_manifests"]["hallucination_and_prior"]
    final_hp = select_joint_h_prior_rows(
        proposals=hp_proposals,
        h_labels=resolved["hallucination"],
        prior_labels=resolved["prior"],
        per_class=int(hp_cfg["final_positive_onset"]),
        minimum_each_source_per_class=(
            int(hp_cfg["minimum_each_source_per_class"])
            if "minimum_each_source_per_class" in hp_cfg
            else None
        ),
        minimum_source_by_class=hp_cfg.get("minimum_source_by_class"),
    )
    h_by_id = {str(row["item_id"]): row for row in resolved["hallucination"]}
    p_by_id = {str(row["item_id"]): row for row in resolved["prior"]}
    for proposal in final_hp:
        row = by_id[str(proposal["id"])]
        row.update(
            materialize_h_label(
                h_by_id[str(proposal["id"])],
                row,
                label_tier=str(protocol["annotation"]["label_tier"]),
            )
        )
        row.update(
            materialize_prior_label(
                p_by_id[str(proposal["id"])],
                row,
                label_tier=str(protocol["annotation"]["label_tier"]),
            )
        )
    output_dir = Path(args.output_dir)
    reports["final_counts"] = {
        "consistency_pairs": len(final_c),
        "h_prior_rows": len(final_hp),
        "h_status": dict(Counter(row["h_status"] for row in final_hp)),
    }
    reports["roster"] = roster
    reports["gates"] = evaluate_preregistered_gates(
        protocol=protocol,
        processed=processed,
        natural_items=natural_items_by_task,
        labels=natural_labels,
        resolved=resolved,
        reports=reports,
        final_consistency=final_c,
        final_hp=final_hp,
        rows_by_id=by_id,
    )
    if reports["gates"]["status"] != "PASS_ALL_PREREGISTERED_GATES":
        atomic_write_json(output_dir / "failed_finalization_report.json", reports)
        raise ValueError(
            "FAIL_PIPELINE: preregistered gates failed: "
            + ", ".join(reports["gates"]["failed_gate_names"])
        )
    publish_manifest(
        output_dir / "final_consistency_pairs.jsonl",
        final_c,
        schema_version="clir-final-consistency-pairs-v2",
    )
    publish_manifest(
        output_dir / "final_h_prior_rows.jsonl",
        final_hp,
        schema_version="clir-final-h-prior-selection-v2",
    )
    pre_extraction = sorted(
        by_id.values(), key=lambda row: (row["query_id"], row["candidate_index"])
    )
    manifest = publish_manifest(
        output_dir / "pre_extraction.jsonl",
        pre_extraction,
        schema_version="clir-smoke-pre-extraction-v2",
        metadata={
            "deferred_stage": "hidden_state_extraction",
            "claims_allowed": ["pipeline_contract", "dual_ai_agreement"],
            "module_efficacy_claim_allowed": False,
        },
    )
    reports["pre_extraction_manifest"] = manifest
    atomic_write_json(output_dir / "finalization_report.json", reports)
    print(json.dumps(reports, indent=2))


def _fixture_response(answer: int, wrong: int, candidate_index: int) -> str:
    target = answer if candidate_index in {0, 1, 4, 6} else wrong
    if candidate_index in {1, 3, 6, 7}:
        return (
            "Step 1: Read the two quantities from the problem.\n"
            "Step 2: The requested operation is addition.\n"
            "Step 3: Substitute the first quantity into the expression.\n"
            "Step 4: Substitute the second quantity into the expression.\n"
            f"Step 5: Evaluating the expression gives {target}.\n"
            f"Answer: Therefore the final value is \\boxed{{{target}}}."
        )
    return (
        "Step 1: Read both quantities.\n"
        "Step 2: Add the quantities.\n"
        f"Step 3: The calculation gives {target}.\n"
        f"Answer: The final value is \\boxed{{{target}}}."
    )


def _fixture_raw_rows(query_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_offset, query in enumerate(query_rows):
        answer = 11 + query_offset
        for candidate_index in range(8):
            response = _fixture_response(answer, answer + 1, candidate_index)
            encoded = [1000 + ord(char) for char in response]
            output_ids = [*encoded, 2]
            rows.append(
                {
                    "id": f"{query['query_id']}:cand:{candidate_index:03d}",
                    "query_id": query["query_id"],
                    "candidate_index": candidate_index,
                    "source": query["source"],
                    "question": query["question"],
                    "reference_answer": str(answer),
                    "prompt_token_ids": [10, 20 + query_offset, 30],
                    "output_token_ids": output_ids,
                    "response": response,
                    "finish_reason": "stop",
                    "_fixture_mapping": {
                        "encoded_token_ids": encoded,
                        "offsets": [
                            [index, index + 1] for index in range(len(response))
                        ],
                        "trailing_token_decodes_to_empty": [True],
                    },
                }
            )
    return rows


def command_fixture(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sources = [
        {
            "source": "gsm8k" if index < 4 else "asdiv-a",
            "query_id": (
                f"gsm8k:train:{index:05d}"
                if index < 4
                else f"asdiv-a:fixture-{index:04d}"
            ),
            "question": f"Fixture problem {index}: combine {index + 3} objects with another group.",
            "reference_answer": str(11 + index),
        }
        for index in range(8)
    ]
    candidates = near_duplicate_candidates(sources)
    decisions = [{**row, "decision": "distinct"} for row in candidates]
    queries, freeze_report = freeze_query_pool(
        sources,
        source_counts={"gsm8k": 4, "asdiv-a": 4},
        near_duplicate_decisions=decisions,
    )
    publish_manifest(
        output_dir / "query_manifest.jsonl",
        queries,
        schema_version="clir-smoke-fixture-query-manifest-v2",
    )
    raw = _fixture_raw_rows(queries)
    processed, materialization_report = materialize_rows(raw)
    validate_rollout_population(processed, candidate_count=8)
    publish_manifest(
        output_dir / "processed.jsonl",
        processed,
        schema_version="clir-smoke-fixture-processed-v2",
    )
    consistency, hp = propose_rows(
        processed,
        consistency_count=2,
        hp_quotas={
            ("gsm8k", 1): 1,
            ("gsm8k", 0): 1,
            ("asdiv-a", 1): 1,
            ("asdiv-a", 0): 1,
        },
    )
    by_id = {str(row["id"]): row for row in processed}
    c_items = [consistency_item(proposal, by_id) for proposal in consistency]
    hp_items = [public_unit_item(by_id[str(proposal["id"])]) for proposal in hp]

    c_a: list[dict[str, Any]] = []
    c_b: list[dict[str, Any]] = []
    for item in c_items:
        label = validate_annotation(
            "consistency",
            {
                "item_id": item["item_id"],
                "decision": "accept",
                "confidence": "high",
                "rationale": "fixture pair preserves the arithmetic path",
            },
            item,
        )
        c_a.append(label)
        c_b.append(dict(label))

    h_a: list[dict[str, Any]] = []
    h_b: list[dict[str, Any]] = []
    p_a: list[dict[str, Any]] = []
    p_b: list[dict[str, Any]] = []
    for proposal, item in zip(hp, hp_items):
        material = [
            unit["unit_index"]
            for unit in item["units"]
            if unit["kind"] == "material_claim"
        ]
        hallucinated = int(proposal["numeric_stratum"]) == 0
        h_payload = {
            "item_id": item["item_id"],
            "status": "hallucinated" if hallucinated else "clean",
            "first_bad_unit_index": material[-2] if hallucinated else None,
            "confidence": "high",
            "rationale": (
                "fixture label with a known final arithmetic mismatch"
                if hallucinated
                else "fixture clean chain"
            ),
        }
        prior_payload = {
            "item_id": item["item_id"],
            "eligibility": "usable",
            "key_unit_indices": [material[-1]],
            "complete_unit_indices": [material[0], material[-2], material[-1]],
            "confidence": "high",
            "rationale": "fixture dependency set",
        }
        h_label = validate_annotation("hallucination", h_payload, item)
        p_label = validate_annotation("prior", prior_payload, item)
        h_a.append(h_label)
        h_b.append(dict(h_label))
        p_a.append(p_label)
        p_b.append(dict(p_label))

    # Exercise the third-model path once without changing the final fixture target.
    h_b[0] = validate_annotation(
        "hallucination",
        {
            **h_b[0],
            "status": "uncertain",
            "first_bad_unit_index": None,
            "rationale": "fixture disagreement to exercise adjudication",
        },
        hp_items[0],
    )
    natural_items = {
        "consistency": c_items,
        "hallucination": hp_items,
        "prior": hp_items,
    }
    packages, private_package = build_annotation_packages(natural_items)
    natural_labels_by_slot = {
        "a": {"consistency": c_a, "hallucination": h_a, "prior": p_a},
        "b": {"consistency": c_b, "hallucination": h_b, "prior": p_b},
    }
    package_labels: dict[str, dict[str, list[dict[str, Any]]]] = {"a": {}, "b": {}}
    reliability: dict[str, Any] = {}
    for slot in ("a", "b"):
        for task in ("consistency", "hallucination", "prior"):
            labels_by_id = {
                str(label["item_id"]): dict(label)
                for label in natural_labels_by_slot[slot][task]
            }
            task_private = private_package["tasks"][task]
            for control in task_private["controls"]:
                labels_by_id[str(control["item_id"])] = dict(
                    control["expected_annotation"]
                )
            if slot == "a":
                for mapping in task_private["self_repeats_a"]:
                    repeated = dict(labels_by_id[mapping["original_item_id"]])
                    repeated["item_id"] = mapping["repeat_item_id"]
                    labels_by_id[mapping["repeat_item_id"]] = repeated
            package_labels[slot][task] = [
                validate_annotation(task, labels_by_id[str(item["item_id"])], item)
                for item in packages[slot][task]
            ]
            publish_manifest(
                output_dir / "blind_packages" / f"annotator_{slot}" / f"{task}.jsonl",
                packages[slot][task],
                schema_version=f"clir-fixture-{task}-blind-package-v2",
            )
    atomic_write_json(
        output_dir / "blind_packages" / "PRIVATE_package_manifest.json",
        private_package,
    )
    for task in ("consistency", "hallucination", "prior"):
        reliability[task] = evaluate_package_reliability(
            task=task,
            private_task=private_package["tasks"][task],
            labels_a=package_labels["a"][task],
            labels_b=package_labels["b"][task],
        )
    adjudication = [
        {
            "item_id": h_a[0]["item_id"],
            "resolution": "adopt_a",
            "independent_answer_completed": True,
        }
    ]
    resolved_c, c_resolution = resolve_blind_labels(
        task="consistency", labels_a=c_a, labels_b=c_b
    )
    resolved_h, h_resolution = resolve_blind_labels(
        task="hallucination", labels_a=h_a, labels_b=h_b, adjudications=adjudication
    )
    resolved_p, p_resolution = resolve_blind_labels(
        task="prior", labels_a=p_a, labels_b=p_b
    )
    final_hp = select_joint_h_prior_rows(
        proposals=hp,
        h_labels=resolved_h,
        prior_labels=resolved_p,
        per_class=2,
        minimum_each_source_per_class=1,
    )
    h_by_id = {str(row["item_id"]): row for row in resolved_h}
    p_by_id = {str(row["item_id"]): row for row in resolved_p}
    final_rows = {str(row["id"]): dict(row) for row in processed}
    for proposal in final_hp:
        row = final_rows[str(proposal["id"])]
        row.update(materialize_h_label(h_by_id[str(proposal["id"])], row))
        row.update(materialize_prior_label(p_by_id[str(proposal["id"])], row))
    for proposal, label in zip(consistency, resolved_c):
        if label["decision"] != "accept":
            continue
        for style, field in (("compact", "left_id"), ("expanded", "right_id")):
            final_rows[str(proposal[field])]["semantic_id"] = proposal["query_id"]
            final_rows[str(proposal[field])]["style_id"] = style
    manifest = publish_manifest(
        output_dir / "pre_extraction.jsonl",
        sorted(
            final_rows.values(),
            key=lambda row: (row["query_id"], row["candidate_index"]),
        ),
        schema_version="clir-smoke-fixture-pre-extraction-v2",
        metadata={"fixture_only": True, "module_efficacy_claim_allowed": False},
    )
    report = {
        "status": "PASS_PIPELINE_FIXTURE",
        "evidence_boundary": "deterministic_pipeline_debug_only",
        "queries": len(queries),
        "raw_rows": len(raw),
        "freeze": freeze_report,
        "materialization": materialization_report,
        "proposal_counts": {"consistency": len(consistency), "h_prior": len(hp)},
        "raw_agreement": {
            "consistency": agreement_report("consistency", c_a, c_b),
            "hallucination": agreement_report("hallucination", h_a, h_b),
            "prior": agreement_report("prior", p_a, p_b),
        },
        "resolution": {
            "consistency": c_resolution,
            "hallucination": h_resolution,
            "prior": p_resolution,
        },
        "protocol_controls_and_self_agreement": reliability,
        "final_h_status": dict(Counter(row["h_status"] for row in final_hp)),
        "pre_extraction": manifest,
    }
    atomic_write_json(output_dir / "fixture_report.json", report)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)

    sources = commands.add_parser(
        "sources",
        help="export protocol-pinned GSM8K/ASDiv-A rows and optional MATH train rows",
    )
    sources.add_argument("--asdiv-repository", required=True)
    sources.add_argument("--output", required=True)
    sources.add_argument("--cache-dir")
    sources.set_defaults(func=command_sources)

    candidates = commands.add_parser(
        "dedup-candidates", help="publish near-duplicate candidates for blind decisions"
    )
    candidates.add_argument("--sources", required=True)
    candidates.add_argument("--excluded-query-ids")
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--threshold", type=float, default=0.82)
    candidates.set_defaults(func=command_dedup_candidates)

    seed_v3_dedup = commands.add_parser(
        "seed-v3-dedup",
        help="carry v2 dedup decisions and conservatively handle MATH near pairs",
    )
    seed_v3_dedup.add_argument("--candidates", required=True)
    seed_v3_dedup.add_argument("--prior-decisions", required=True)
    seed_v3_dedup.add_argument("--output-dir", required=True)
    seed_v3_dedup.set_defaults(func=command_seed_v3_dedup)

    resolve_dedup = commands.add_parser(
        "resolve-dedup", help="merge blind A/B and third-model duplicate decisions"
    )
    resolve_dedup.add_argument("--candidates", required=True)
    resolve_dedup.add_argument("--labels-a", required=True)
    resolve_dedup.add_argument("--labels-b", required=True)
    resolve_dedup.add_argument("--adjudications")
    resolve_dedup.add_argument("--roster", required=True)
    resolve_dedup.add_argument("--output", required=True)
    resolve_dedup.set_defaults(func=command_resolve_dedup)

    dedup_triage = commands.add_parser(
        "dedup-triage",
        help="publish unresolved duplicate pairs for a blind third model",
    )
    dedup_triage.add_argument("--candidates", required=True)
    dedup_triage.add_argument("--labels-a", required=True)
    dedup_triage.add_argument("--labels-b", required=True)
    dedup_triage.add_argument("--output", required=True)
    dedup_triage.set_defaults(func=command_dedup_triage)

    exclusions = commands.add_parser(
        "collect-exclusions",
        help="normalize prior GSM8K-train query populations into one audit manifest",
    )
    exclusions.add_argument("--input", action="append", required=True)
    exclusions.add_argument("--output", required=True)
    exclusions.set_defaults(func=command_collect_exclusions)

    freeze = commands.add_parser(
        "freeze", help="freeze the protocol-pinned train-only query manifest"
    )
    freeze.add_argument("--sources", required=True)
    freeze.add_argument("--near-duplicate-decisions")
    freeze.add_argument("--excluded-query-ids")
    freeze.add_argument(
        "--required-query-ids",
        help="v3 incumbent query manifest/ID file that must remain selected",
    )
    freeze.add_argument("--threshold", type=float, default=0.82)
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(func=command_freeze)

    rollout = commands.add_parser(
        "rollout", help="generate all eight exact-ID candidates per frozen query"
    )
    rollout.add_argument("--queries", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--cache-dir")
    rollout.add_argument("--tensor-parallel-size", type=int, default=0)
    rollout.add_argument("--dtype", default="bfloat16")
    rollout.add_argument("--max-num-seqs", type=int, default=32)
    rollout.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    rollout.set_defaults(func=command_rollout)

    merge_rollouts = commands.add_parser(
        "merge-rollouts", help="merge reused v2 and new v3 rollout batches"
    )
    merge_rollouts.add_argument("--input", action="append", required=True)
    merge_rollouts.add_argument(
        "--query-manifest",
        action="append",
        required=True,
        help="frozen query manifest corresponding positionally to each --input",
    )
    merge_rollouts.add_argument("--reserve-included", action="store_true")
    merge_rollouts.add_argument("--output", required=True)
    merge_rollouts.set_defaults(func=command_merge_rollouts)

    materialize = commands.add_parser(
        "materialize", help="run the protocol-pinned checker and exact-token unitizer"
    )
    materialize.add_argument("--rollouts", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--cache-dir")
    materialize.set_defaults(func=command_materialize)

    propose = commands.add_parser(
        "propose", help="freeze C and H/P natural annotation manifests"
    )
    propose.add_argument("--processed", required=True)
    propose.add_argument("--output-dir", required=True)
    propose.set_defaults(func=command_propose)

    readiness = commands.add_parser(
        "readiness", help="apply the frozen v3 primary/reserve yield gate"
    )
    readiness.add_argument("--processed", required=True)
    readiness.add_argument("--reserve-included", action="store_true")
    readiness.add_argument("--output", required=True)
    readiness.set_defaults(func=command_readiness)

    package = commands.add_parser(
        "package", help="add hidden controls and A-only self repeats to blind packages"
    )
    package.add_argument("--items-dir", required=True)
    package.add_argument("--output-dir", required=True)
    package.set_defaults(func=command_package)

    triage = commands.add_parser(
        "triage",
        help="freeze the third-model 15%% auto-agree audit and all A/B disputes",
    )
    triage.add_argument("--items-dir", required=True)
    triage.add_argument("--package-dir", required=True)
    triage.add_argument("--labels-a-dir", required=True)
    triage.add_argument("--labels-b-dir", required=True)
    triage.add_argument("--output-dir", required=True)
    triage.set_defaults(func=command_triage)

    adjudication_package = commands.add_parser(
        "adjudication-package",
        help="show the third model anonymous A/B options only after its independent pass",
    )
    adjudication_package.add_argument("--items-dir", required=True)
    adjudication_package.add_argument("--package-dir", required=True)
    adjudication_package.add_argument("--labels-a-dir", required=True)
    adjudication_package.add_argument("--labels-b-dir", required=True)
    adjudication_package.add_argument("--triage-dir", required=True)
    adjudication_package.add_argument("--third-independent-labels-dir", required=True)
    adjudication_package.add_argument("--output-dir", required=True)
    adjudication_package.set_defaults(func=command_adjudication_package)

    finalize = commands.add_parser(
        "finalize",
        help="validate A/B/third-model labels and create pre-extraction rows",
    )
    finalize.add_argument("--processed", required=True)
    finalize.add_argument("--items-dir", required=True)
    finalize.add_argument(
        "--package-dir",
        required=True,
        help="blind package root; enables hidden-control and A self-repeat validation",
    )
    finalize.add_argument("--labels-a-dir", required=True)
    finalize.add_argument("--labels-b-dir", required=True)
    finalize.add_argument("--adjudication-dir")
    finalize.add_argument("--adjudication-package-dir", required=True)
    finalize.add_argument("--triage-dir", required=True)
    finalize.add_argument("--third-independent-labels-dir", required=True)
    finalize.add_argument("--roster", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.set_defaults(func=command_finalize)

    fixture = commands.add_parser(
        "fixture", help="run a deterministic 8-query no-model end-to-end fixture"
    )
    fixture.add_argument("--output-dir", required=True)
    fixture.set_defaults(func=command_fixture)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
