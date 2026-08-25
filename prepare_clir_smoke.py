#!/usr/bin/env python
"""Prepare and validate the frozen CLIR multi-source smoke-v2 data pipeline.

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
    LABEL_TIER,
    agreement_report,
    annotation_signature,
    atomic_write_json,
    build_consistency_proposals,
    build_h_prior_proposals,
    canonical_sha256,
    check_numeric_response,
    cohen_kappa,
    consistency_item,
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
    PROJECT_ROOT / "configs" / "data_expansion_smoke_v2" / "protocol.json"
)


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "clir-data-expansion-smoke-v2":
        raise ValueError("only clir-data-expansion-smoke-v2 is executable")
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
    output = Path(args.output)
    report = publish_manifest(
        output,
        [*gsm_rows, *asdiv_rows],
        schema_version="clir-smoke-source-corpus-v2",
        metadata={
            "gsm8k_revision": gsm_cfg["revision"],
            "asdiv_commit": actual_commit,
            "counts": {"gsm8k": len(gsm_rows), "asdiv-a": len(asdiv_rows)},
        },
    )
    print(json.dumps(report, indent=2))


def command_dedup_candidates(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.sources)
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
            "source_rows_sha256": canonical_sha256(rows),
            "excluded_query_ids_sha256": canonical_sha256(sorted(excluded)),
            "all_candidate_count": len(all_candidates),
            "skipped_both_excluded_count": len(all_candidates) - len(candidates),
        },
    )
    print(json.dumps(report, indent=2))


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
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left)
        if left
        else None,
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
    rows = read_jsonl(args.sources)
    decisions = (
        read_jsonl(args.near_duplicate_decisions)
        if args.near_duplicate_decisions
        else []
    )
    source_cfg = protocol["sources"]
    selected, report = freeze_query_pool(
        rows,
        source_counts={
            "gsm8k": int(source_cfg["gsm8k"]["query_count"]),
            "asdiv-a": int(source_cfg["asdiv_a"]["query_count"]),
        },
        excluded_query_ids=_read_id_file(args.excluded_query_ids),
        near_duplicate_decisions=decisions,
        jaccard_threshold=args.threshold,
    )
    output_dir = Path(args.output_dir)
    manifest = publish_manifest(
        output_dir / "query_manifest.jsonl",
        selected,
        schema_version="clir-smoke-query-manifest-v2",
        metadata={"protocol_sha256": canonical_sha256(protocol)},
    )
    atomic_write_json(output_dir / "source_freeze_report.json", report)
    publish_manifest(
        output_dir / "permanent_train_only_exclusions.jsonl",
        [
            {"query_id": row["query_id"], "reason": "train_only_smoke_v2"}
            for row in selected
        ],
        schema_version="clir-smoke-permanent-exclusions-v2",
    )
    print(json.dumps({"query_manifest": manifest, "freeze": report}, indent=2))


def command_rollout(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    generation = protocol["generation"]
    queries = read_jsonl(args.queries)
    if len(queries) != int(generation["query_count"]):
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
        schema_version="clir-smoke-raw-rollouts-v2",
        metadata={**provenance, **population},
    )
    print(json.dumps(manifest, indent=2))


def materialize_rows(
    raw_rows: Sequence[Mapping[str, Any]], tokenizer: Any | None = None
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
            )
        )
        try:
            if tokenizer is None:
                mapping = row.pop("_fixture_mapping")
            else:
                mapping = tokenize_visible_response(
                    tokenizer, str(row["response"]), row["output_token_ids"]
                )
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
    processed, report = materialize_rows(raw_rows, tokenizer)
    population = validate_rollout_population(
        processed, candidate_count=int(generation["candidate_count"])
    )
    manifest = publish_manifest(
        args.output,
        processed,
        schema_version="clir-smoke-materialized-rollouts-v2",
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
    )
    return consistency, hp


def command_propose(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    rows = read_jsonl(args.processed)
    proposal_cfg = protocol["proposal_manifests"]
    c_cfg = proposal_cfg["consistency"]
    hp_cfg = proposal_cfg["hallucination_and_prior"]
    strata = hp_cfg["source_numeric_strata"]
    quotas = {
        ("gsm8k", 1): int(strata["gsm8k_numeric_match"]),
        ("gsm8k", 0): int(strata["gsm8k_numeric_mismatch"]),
        ("asdiv-a", 1): int(strata["asdiv_a_numeric_match"]),
        ("asdiv-a", 0): int(strata["asdiv_a_numeric_mismatch"]),
    }
    consistency, hp = propose_rows(
        rows,
        consistency_count=int(c_cfg["natural_proposals"]),
        hp_quotas=quotas,
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


def _simple_units(texts: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    trajectory = "\n".join(texts)
    units: list[dict[str, Any]] = []
    cursor = 0
    for index, text in enumerate(texts):
        units.append({"unit_index": index, "kind": "material_claim", "text": text})
        cursor += len(text) + 1
    return trajectory, units


def _protocol_controls(
    task: str, count: int
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
                "The problem gives her 30 more apples."
                if hallucinated
                else "The problem gives her 3 more apples.",
                "The requested operation is addition.",
                "The final total is 32 apples."
                if hallucinated
                else "The final total is 5 apples.",
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
            if usable:
                texts = [
                    "Mina starts with 2 apples.",
                    "She receives 3 more apples.",
                    "Adding gives 2+3=5.",
                    "Therefore the answer is 5 apples.",
                ]
                eligibility = "usable"
                key, complete = [2, 3], [0, 1, 2, 3]
            else:
                texts = ["I cannot solve this problem."]
                eligibility = "no_auditable_reasoning"
                key, complete = [], []
            trajectory, units = _simple_units(texts)
            item = {
                "item_id": item_id,
                "query_id": f"control:prior:{index}",
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
        controls = _protocol_controls(task, control_count)
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
    items_dir = Path(args.items_dir)
    natural = {
        "consistency": read_jsonl(items_dir / "annotation_consistency_natural.jsonl"),
        "hallucination": read_jsonl(
            items_dir / "annotation_hallucination_natural.jsonl"
        ),
        "prior": read_jsonl(items_dir / "annotation_prior_natural.jsonl"),
    }
    packages, private = build_annotation_packages(natural)
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
    atomic_write_json(output_dir / "PRIVATE_triage_manifest.json", private_triage)
    atomic_write_json(output_dir / "triage_counts.json", public_counts)
    print(
        json.dumps(
            {
                "status": "third_independent_packages_ready",
                "send_only": "third_independent/",
                "never_send": "PRIVATE_triage_manifest.json",
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
    if (
        min(c_raw["a_decisions"].values(), default=0) >= 5
        and min(c_raw["b_decisions"].values(), default=0) >= 5
    ):
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
    minimum_source = int(thresholds["minimum_each_source_per_final_h_class"])
    for status in ("hallucinated", "clean"):
        for source in ("gsm8k", "asdiv-a"):
            count = sum(
                row["h_status"] == status and row["source"] == source
                for row in final_hp
            )
            add(f"final_{status}_{source}", count, ">=", minimum_source)
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
        "status": "PASS_ALL_PREREGISTERED_GATES"
        if not failed
        else "FAIL_PIPELINE_GATES",
        "failed_gate_names": failed,
        "results": results,
    }


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
            "auto_agree_audit_stability": stable / len(audit_ids)
            if audit_ids
            else None,
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
            row["consistency_label_tier"] = LABEL_TIER

    final_hp = select_joint_h_prior_rows(
        proposals=hp_proposals,
        h_labels=resolved["hallucination"],
        prior_labels=resolved["prior"],
        per_class=int(
            protocol["proposal_manifests"]["hallucination_and_prior"][
                "final_positive_onset"
            ]
        ),
        minimum_each_source_per_class=int(
            protocol["proposal_manifests"]["hallucination_and_prior"][
                "minimum_each_source_per_class"
            ]
        ),
    )
    h_by_id = {str(row["item_id"]): row for row in resolved["hallucination"]}
    p_by_id = {str(row["item_id"]): row for row in resolved["prior"]}
    for proposal in final_hp:
        row = by_id[str(proposal["id"])]
        row.update(materialize_h_label(h_by_id[str(proposal["id"])], row))
        row.update(materialize_prior_label(p_by_id[str(proposal["id"])], row))
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
            "rationale": "fixture label with a known final arithmetic mismatch"
            if hallucinated
            else "fixture clean chain",
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
        "sources", help="export pinned GSM8K train and ASDiv-A source rows"
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
        "freeze", help="freeze the 60/40 train-only query manifest"
    )
    freeze.add_argument("--sources", required=True)
    freeze.add_argument("--near-duplicate-decisions")
    freeze.add_argument("--excluded-query-ids")
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

    materialize = commands.add_parser(
        "materialize", help="run checker v2 and exact-token unitizer v2"
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
