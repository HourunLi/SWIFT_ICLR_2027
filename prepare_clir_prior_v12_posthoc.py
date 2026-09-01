#!/usr/bin/env python
"""Prepare and verify the exploratory exact-consensus Prior-v12 subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from src.clir_data import read_jsonl
from src.clir_prior_v12_posthoc import (
    ORIGINAL_V12_STATUS,
    construct_posthoc_rows,
    feature_inventory,
    inventory_statistics,
    rebase_feature_paths,
)
from src.clir_scale_features import validate_tensor_file
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "configs/data_expansion_prior_v12/posthoc_v1/protocol.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12/posthoc_v1"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_commit() -> str:
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Prior-v12 post-hoc commands require a clean worktree")
    return _git_head()


def _require_ancestor(ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
    )
    if result.returncode:
        raise ValueError("frozen parent commit is not an ancestor of current HEAD")


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash drift: {observed} != {expected}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "clir-prior-v12-posthoc-exact-experiment-protocol-v1"
        or protocol.get("status")
        != "AUTHORIZED_PRIOR_V12_POSTHOC_EXACT_DIRECT_LEARNABILITY"
    ):
        raise ValueError("unsupported or inactive Prior-v12 post-hoc protocol")
    status = protocol["original_evidence_status"]
    if (
        status.get("prior_v12_status") != ORIGINAL_V12_STATUS
        or status.get("prior_v12_failure_is_preserved") is not True
        or status.get("prior_v13_status") != "FAIL_PRIOR_V13_SCHEMA"
        or status.get("prior_v13_failure_is_preserved") is not True
        or status.get("may_be_called_gold_confirmatory_or_v12_pass") is not False
    ):
        raise ValueError("original v12/v13 terminal status or claim boundary drift")
    for name, specification in protocol["frozen_inputs"].items():
        source_path = _project_path(specification["path"])
        _assert_hash(source_path, specification["file_sha256"], name)
        if "sidecar_file_sha256" in specification:
            sidecar = source_path.with_suffix(source_path.suffix + ".manifest.json")
            _assert_hash(sidecar, specification["sidecar_file_sha256"], f"{name} sidecar")
    for name, specification in protocol["execution_configs"].items():
        _assert_hash(
            _project_path(specification["path"]),
            specification["file_sha256"],
            f"{name} config",
        )
    verification = json.loads(
        _project_path(
            protocol["frozen_inputs"]["v12_terminal_verification"]["path"]
        ).read_text(encoding="utf-8")
    )
    if (
        verification.get("status")
        != "PASS_PRIOR_V12_LABEL_EVALUATION_INDEPENDENT_RECOMPUTE"
        or verification.get("terminal_or_next_gate")
        != "terminal_preserve_labels_no_relabel_no_adaptive_salvage"
    ):
        raise ValueError("v12 independent terminal verification semantics drift")
    v13_report = json.loads(
        _project_path(
            protocol["frozen_inputs"]["v13_terminal_report"]["path"]
        ).read_text(encoding="utf-8")
    )
    if v13_report.get("status") != "FAIL_PRIOR_V13_SCHEMA":
        raise ValueError("v13 terminal schema failure semantics drift")
    return protocol


def _authorized_output_root(
    protocol: Mapping[str, Any], requested: str | Path
) -> Path:
    expected = _project_path(protocol["runtime_contract"]["output_root"]).resolve()
    observed = Path(requested).resolve()
    if observed != expected:
        raise ValueError(f"output root drift: {observed} != {expected}")
    return observed


def _published_identity(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "sidecar_file_sha256": file_sha256(sidecar),
        "row_count": len(rows),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def _load_annotation_bundle(
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    inputs = protocol["frozen_inputs"]
    gate_report = json.loads(
        _project_path(inputs["v12_terminal_report"]["path"]).read_text(encoding="utf-8")
    )
    if (
        gate_report.get("status") != ORIGINAL_V12_STATUS
        or gate_report.get("failure_is_terminal") is not True
        or gate_report.get("target_publication_authorized") is not False
    ):
        raise ValueError("v12 terminal report semantics drift")
    bindings = gate_report["bindings"]
    if bindings["proposal_file_sha256"] != inputs["natural_proposals"]["file_sha256"]:
        raise ValueError("v12 proposal binding drift")
    if (
        bindings["private_index_file_sha256"]
        != inputs["private_package_index"]["file_sha256"]
    ):
        raise ValueError("v12 private-index binding drift")

    proposals = read_jsonl(_project_path(inputs["natural_proposals"]["path"]))
    private = read_jsonl(_project_path(inputs["private_package_index"]["path"]))
    materialized = read_jsonl(_project_path(inputs["materialized_rollouts"]["path"]))
    for name, rows in (
        ("natural_proposals", proposals),
        ("private_package_index", private),
        ("materialized_rollouts", materialized),
    ):
        if len(rows) != int(inputs[name]["row_count"]):
            raise ValueError(f"{name} row-count drift")
    if (
        canonical_sha256(materialized)
        != inputs["materialized_rollouts"]["ordered_rows_sha256"]
    ):
        raise ValueError("materialized rollout ordered-row hash drift")

    packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    labels: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    shard_count = int(
        protocol["annotation_bundle_contract"][
            "package_and_label_shard_count_per_annotator"
        ]
    )
    expected_rows = int(protocol["annotation_bundle_contract"]["rows_per_shard"])
    for annotator in ("a", "b"):
        for index in range(shard_count):
            shard_id = f"{annotator}-{index:02d}"
            package_path = (
                PROJECT_ROOT
                / "run_artifacts/data_expansion_prior_v12/pre_annotation/packages"
                / f"annotator_{annotator}/prior_v12_{annotator}_{index:02d}.jsonl"
            )
            label_path = (
                PROJECT_ROOT
                / f"run_artifacts/data_expansion_prior_v12/pre_annotation/labels_{annotator}"
                / f"prior_v12_{annotator}_{index:02d}.jsonl"
            )
            package_binding = bindings["package_shards"][shard_id]
            label_binding = bindings["label_shards"][shard_id]
            _assert_hash(package_path, package_binding["file_sha256"], f"{shard_id} package")
            _assert_hash(label_path, label_binding["file_sha256"], f"{shard_id} labels")
            package_rows = read_jsonl(package_path)
            label_rows = read_jsonl(label_path)
            if len(package_rows) != expected_rows or len(label_rows) != expected_rows:
                raise ValueError(f"{shard_id}: shard row-count drift")
            if canonical_sha256(package_rows) != package_binding["ordered_rows_sha256"]:
                raise ValueError(f"{shard_id}: package ordered-row hash drift")
            if canonical_sha256(label_rows) != label_binding["ordered_rows_sha256"]:
                raise ValueError(f"{shard_id}: label ordered-row hash drift")
            packages[annotator].extend(package_rows)
            labels[annotator].extend(label_rows)
    return proposals, materialized, private, packages, labels


def _config_gate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    payloads = {
        name: json.loads(_project_path(spec["path"]).read_text(encoding="utf-8"))
        for name, spec in protocol["execution_configs"].items()
    }
    if set(payloads) != {"R0", "P0"}:
        raise ValueError("direct learnability grid must contain exactly R0 and P0")
    normalized = []
    observed: dict[str, Any] = {}
    for cell in ("R0", "P0"):
        payload = json.loads(json.dumps(payloads[cell]))
        prior_weight = float(payload["model"].pop("prior_weight"))
        expected = float(protocol["execution_configs"][cell]["prior_weight"])
        if prior_weight != expected:
            raise ValueError(f"{cell}: prior weight drift")
        for forbidden in (
            "consistency_weight",
            "hallucination_weight",
            "token_reward_weight",
            "tail_weight",
            "mil_weight",
            "pseudo_tail_weight",
            "progress_weight",
            "prior_distill_weight",
            "gate_prior_weight",
            "reconstruction_weight",
        ):
            if float(payload["model"][forbidden]) != 0.0:
                raise ValueError(f"{cell}: forbidden objective {forbidden} is enabled")
        normalized.append(payload)
        observed[cell] = {
            "prior_weight": prior_weight,
            "file_sha256": protocol["execution_configs"][cell]["file_sha256"],
        }
    if normalized[0] != normalized[1]:
        raise ValueError("R0/P0 configs differ outside prior_weight")
    return observed


def _source_identity(row: Mapping[str, Any]) -> tuple[str, str] | None:
    source = str(row.get("source", "")).lower()
    value = row.get("source_record_id", row.get("source_index"))
    if not source or value is None:
        return None
    return source, str(value)


def _overlap_gate(
    selected: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    selected_queries = {str(row["query_id"]) for row in selected}
    selected_clusters = {str(row["cluster_id"]) for row in selected}
    selected_sources = {key for row in selected if (key := _source_identity(row))}
    result = {
        "selected_historical_query": len(
            selected_queries & {str(row["query_id"]) for row in historical}
        ),
        "selected_historical_source_record": len(
            selected_sources
            & {key for row in historical if (key := _source_identity(row))}
        ),
        "selected_ranking_query": len(
            selected_queries & {str(row["query_id"]) for row in ranking}
        ),
        "selected_ranking_cluster": len(
            selected_clusters
            & {str(row["cluster_id"]) for row in ranking if row.get("cluster_id")}
        ),
        "selected_ranking_source_record": len(
            selected_sources & {key for row in ranking if (key := _source_identity(row))}
        ),
    }
    if any(result.values()):
        raise ValueError(f"post-hoc population overlap: {result}")
    return result


def _construct(protocol: Mapping[str, Any]) -> dict[str, Any]:
    proposals, materialized, private, packages, labels = _load_annotation_bundle(protocol)
    rows, selection_report = construct_posthoc_rows(
        proposals=proposals,
        materialized_rows=materialized,
        private_index=private,
        packages=packages,
        labels=labels,
    )
    expected_selection = protocol["posthoc_selection"]["expected"]
    for field in (
        "selected_rows",
        "selected_train_rows",
        "selected_dev_rows",
        "selected_ordered_proposal_ids_sha256",
        "selected_strata",
    ):
        if selection_report[field] != expected_selection[field]:
            raise ValueError(f"post-hoc selection {field} drift")
    inventory = feature_inventory(rows)
    statistics = inventory_statistics(inventory)
    expected_features = protocol["feature_contract"]["expected"]
    for field in (
        "trajectory_count",
        "query_count",
        "condition_count",
        "output_token_count",
        "prompt_token_count",
        "total_feature_token_count",
    ):
        if int(statistics[field]) != int(expected_features[field]):
            raise ValueError(f"feature inventory {field} drift")
    raw_bytes = int(statistics["total_feature_token_count"]) * int(
        protocol["feature_contract"]["bytes_per_feature_token"]
    )
    if raw_bytes != int(expected_features["raw_feature_bytes"]):
        raise ValueError("feature raw-byte budget drift")

    inputs = protocol["frozen_inputs"]
    historical = read_jsonl(_project_path(inputs["historical_correctness_train"]["path"]))
    ranking = read_jsonl(_project_path(inputs["exploratory_ranking_evaluation"]["path"]))
    if len(historical) != int(inputs["historical_correctness_train"]["row_count"]):
        raise ValueError("historical correctness row-count drift")
    if len(ranking) != int(inputs["exploratory_ranking_evaluation"]["row_count"]):
        raise ValueError("ranking evaluation row-count drift")
    ranking_queries = {str(row["query_id"]) for row in ranking}
    if len(ranking_queries) != int(inputs["exploratory_ranking_evaluation"]["query_count"]):
        raise ValueError("ranking evaluation query-count drift")
    overlap = _overlap_gate(rows, historical, ranking)
    return {
        "rows": rows,
        "inventory": inventory,
        "selection_report": selection_report,
        "inventory_statistics": statistics,
        "raw_feature_bytes": raw_bytes,
        "historical": historical,
        "ranking": ranking,
        "overlap_report": overlap,
    }


def command_prepare(args: argparse.Namespace) -> None:
    code_commit = _require_clean_commit()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    _require_ancestor(protocol["frozen_parent_commit"], code_commit)
    output_root = _authorized_output_root(protocol, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Prior-v12 post-hoc output root is not empty: {output_root}")
    constructed = _construct(protocol)
    config_gate = _config_gate(protocol)
    plan_root = output_root / "plan"
    plan_root.mkdir(parents=True)
    selection_path = plan_root / "selected_prior_rows.jsonl"
    inventory_path = plan_root / "feature_inventory.jsonl"
    publish_manifest(
        selection_path,
        constructed["rows"],
        schema_version="clir-prior-v12-posthoc-selected-rows-v1",
        metadata={
            "posthoc_exploratory": True,
            "original_v12_status": ORIGINAL_V12_STATUS,
            "human_verified": False,
        },
    )
    publish_manifest(
        inventory_path,
        constructed["inventory"],
        schema_version="clir-prior-v12-posthoc-feature-inventory-v1",
        metadata={"selected_only": True, "decode_or_retokenize": False},
    )
    largest = max(
        constructed["inventory"],
        key=lambda row: (
            int(row["prompt_token_count"]) + int(row["output_token_count"]),
            str(row["id"]),
        ),
    )
    preflight_path = plan_root / "preflight.jsonl"
    publish_manifest(
        preflight_path,
        [largest],
        schema_version="clir-prior-v12-posthoc-feature-preflight-v1",
        metadata={"selection": "largest_prompt_plus_output_token_count"},
    )
    report = {
        "schema_version": "clir-prior-v12-posthoc-plan-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_PLAN",
        "planned_at_utc": _utc_now(),
        "code_commit": code_commit,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": file_sha256(protocol_path),
        "config_gate": config_gate,
        "selection_report": constructed["selection_report"],
        "inventory_statistics": constructed["inventory_statistics"],
        "raw_feature_bytes": constructed["raw_feature_bytes"],
        "overlap_report": constructed["overlap_report"],
        "selected_rows": _published_identity(selection_path, constructed["rows"]),
        "feature_inventory": _published_identity(
            inventory_path, constructed["inventory"]
        ),
        "preflight": _published_identity(preflight_path, [largest]),
        "feature_extraction_allowed": True,
        "training_allowed": False,
    }
    report_path = plan_root / "plan_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(report_path)}, indent=2))


def _load_plan(
    protocol_path: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    _authorized_output_root(protocol, output_root)
    plan_path = output_root / "plan/plan_report.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_PRIOR_V12_POSTHOC_PLAN":
        raise ValueError("Prior-v12 post-hoc plan did not pass")
    if plan.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("protocol changed after planning")
    if plan.get("code_commit") != _require_clean_commit():
        raise ValueError("code commit changed after planning")
    return protocol, plan


def _assert_published(
    path: Path,
    expected: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if _published_identity(path, rows) != dict(expected):
        raise ValueError(f"published manifest drift: {path}")


def command_verify_plan(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    constructed = _construct(protocol)
    selection_path = output_root / "plan/selected_prior_rows.jsonl"
    inventory_path = output_root / "plan/feature_inventory.jsonl"
    _assert_published(selection_path, plan["selected_rows"], constructed["rows"])
    _assert_published(
        inventory_path, plan["feature_inventory"], constructed["inventory"]
    )
    largest = max(
        constructed["inventory"],
        key=lambda row: (
            int(row["prompt_token_count"]) + int(row["output_token_count"]),
            str(row["id"]),
        ),
    )
    _assert_published(output_root / "plan/preflight.jsonl", plan["preflight"], [largest])
    report = {
        "schema_version": "clir-prior-v12-posthoc-plan-verification-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_PLAN_RECOMPUTATION",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "plan_report_file_sha256": file_sha256(
            output_root / "plan/plan_report.json"
        ),
        "selected_ordered_rows_sha256": canonical_sha256(constructed["rows"]),
        "feature_inventory_ordered_rows_sha256": canonical_sha256(
            constructed["inventory"]
        ),
        "feature_extraction_allowed": True,
        "training_allowed": False,
    }
    path = output_root / "plan/independent_verification.json"
    if path.exists():
        raise FileExistsError(f"plan verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


def _safe_feature_path(parent: Path, value: Any, output_root: Path) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (parent / raw).resolve()
    if not path.is_relative_to(output_root.resolve()):
        raise ValueError(f"feature path escapes output root: {path}")
    return path


def _verify_extracted(
    source_rows: Sequence[Mapping[str, Any]],
    extracted_path: Path,
    output_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    extracted = read_jsonl(extracted_path)
    if len(extracted) != len(source_rows):
        raise ValueError("extracted feature row-count drift")
    contract = protocol["feature_contract"]
    conditions: dict[str, tuple[str, str]] = {}
    raw_bytes = serialized_bytes = 0
    for source, row in zip(source_rows, extracted, strict=True):
        for field, value in source.items():
            if row.get(field) != value:
                raise ValueError(f"{source['id']}: extracted source field drift: {field}")
        if (
            row.get("feature_model") != contract["model_id"]
            or row.get("feature_revision") != contract["model_revision"]
            or row.get("feature_dtype") != contract["dtype"]
            or row.get("feature_attention_implementation")
            != contract["attention_implementation"]
            or int(row.get("feature_dim", -1)) != int(contract["feature_dim"])
            or int(row.get("num_feature_layers", -1))
            != int(contract["num_feature_layers"])
            or int(row.get("per_layer_dim", -1)) != int(contract["per_layer_dim"])
        ):
            raise ValueError(f"{source['id']}: extracted feature contract drift")
        hidden_path = _safe_feature_path(
            extracted_path.parent, row["hidden_states_path"], output_root
        )
        hidden = validate_tensor_file(
            hidden_path,
            expected_shape=[
                len(source["output_token_ids"]),
                int(contract["feature_dim"]),
            ],
            expected_dtype=torch.bfloat16,
            expected_sha256=str(row["hidden_states_sha256"]),
        )
        raw_bytes += int(hidden["raw_tensor_bytes"])
        serialized_bytes += int(hidden["serialized_bytes"])
        condition_path = _safe_feature_path(
            extracted_path.parent, row["condition_states_path"], output_root
        )
        query_id = str(row["query_id"])
        identity = (str(condition_path), str(row["condition_states_sha256"]))
        if query_id not in conditions:
            condition = validate_tensor_file(
                condition_path,
                expected_shape=[
                    len(source["prompt_token_ids"]),
                    int(contract["feature_dim"]),
                ],
                expected_dtype=torch.bfloat16,
                expected_sha256=str(row["condition_states_sha256"]),
            )
            conditions[query_id] = identity
            raw_bytes += int(condition["raw_tensor_bytes"])
            serialized_bytes += int(condition["serialized_bytes"])
        elif conditions[query_id] != identity:
            raise ValueError(f"{query_id}: condition feature drift within query")
    return {
        "rows": extracted,
        "trajectory_count": len(extracted),
        "condition_count": len(conditions),
        "raw_tensor_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
    }


def command_verify_preflight(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    result = _verify_extracted(
        read_jsonl(output_root / "plan/preflight.jsonl"),
        output_root / "preflight/extracted.jsonl",
        output_root,
        protocol,
    )
    report = {
        "schema_version": "clir-prior-v12-posthoc-feature-preflight-verification-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_FULL_WIDTH_FEATURE_PREFLIGHT",
        "verified_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "trajectory_count": result["trajectory_count"],
        "condition_count": result["condition_count"],
        "raw_tensor_bytes": result["raw_tensor_bytes"],
        "training_allowed": False,
    }
    path = output_root / "preflight/verification.json"
    if path.exists():
        raise FileExistsError(f"preflight verification already exists: {path}")
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


def _publish(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = publish_manifest(path, rows, schema_version=schema, metadata=metadata)
    manifest["sidecar_file_sha256"] = file_sha256(
        path.with_suffix(path.suffix + ".manifest.json")
    )
    return manifest


def command_finalize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, plan = _load_plan(protocol_path, output_root)
    preflight = json.loads(
        (output_root / "preflight/verification.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS_PRIOR_V12_POSTHOC_FULL_WIDTH_FEATURE_PREFLIGHT":
        raise ValueError("full-width feature preflight did not pass")
    source_rows = read_jsonl(output_root / "plan/feature_inventory.jsonl")
    extracted_path = output_root / "features/extracted.jsonl"
    verified = _verify_extracted(source_rows, extracted_path, output_root, protocol)
    expected = protocol["feature_contract"]["expected"]
    if (
        verified["trajectory_count"] != int(expected["trajectory_count"])
        or verified["condition_count"] != int(expected["condition_count"])
        or verified["raw_tensor_bytes"] != int(expected["raw_feature_bytes"])
    ):
        raise ValueError("verified feature totals drift")

    features_root = output_root / "features"
    verified_path = features_root / "verified_features.jsonl"
    verified_manifest = _publish(
        verified_path,
        verified["rows"],
        "clir-prior-v12-posthoc-verified-features-v1",
        {
            "posthoc_exploratory": True,
            "all_shapes_dtypes_finiteness_and_checksums_verified": True,
        },
    )
    data_root = output_root / "data"
    prior_rows = [
        {
            **rebase_feature_paths(
                row, source_parent=extracted_path.parent, target_parent=data_root
            ),
            "schema_version": "clir-prior-v12-posthoc-training-row-v1",
            "experiment_population": "prior_v12_posthoc_exact",
        }
        for row in verified["rows"]
    ]
    prior_train = [row for row in prior_rows if row["split"] == "train"]
    prior_dev = [row for row in prior_rows if row["split"] == "dev"]
    historical = read_jsonl(
        _project_path(
            protocol["frozen_inputs"]["historical_correctness_train"]["path"]
        )
    )
    historical = [
        {**row, "experiment_population": "historical_correctness_v1"}
        for row in historical
    ]
    matched_train = historical + prior_train
    training = protocol["training_contract"]
    if (
        len(prior_train) != int(training["new_prior_train_rows"])
        or len(prior_dev) != int(training["prior_dev_rows"])
        or len(matched_train) != int(training["total_train_rows"])
    ):
        raise ValueError("matched training/dev row-count drift")
    manifests = {
        "verified_features": verified_manifest,
        "matched_train": _publish(
            data_root / "train_r0_p0.jsonl",
            matched_train,
            "clir-prior-v12-posthoc-matched-training-manifest-v1",
            {"shared_by_cells": ["R0", "P0"]},
        ),
        "prior_dev": _publish(
            data_root / "prior_dev.jsonl",
            prior_dev,
            "clir-prior-v12-posthoc-dev-manifest-v1",
            {"evaluation_only": True, "posthoc_exploratory": True},
        ),
    }
    report = {
        "schema_version": "clir-prior-v12-posthoc-finalization-v1",
        "status": "PASS_PRIOR_V12_POSTHOC_FEATURES_AND_MATCHED_DATA",
        "completed_at_utc": _utc_now(),
        "code_commit": plan["code_commit"],
        "protocol_file_sha256": file_sha256(protocol_path),
        "original_v12_status": ORIGINAL_V12_STATUS,
        "evidence_tier": "posthoc_exploratory_silver_no_human_verification",
        "selection_report": plan["selection_report"],
        "inventory_statistics": plan["inventory_statistics"],
        "raw_tensor_bytes": verified["raw_tensor_bytes"],
        "serialized_tensor_bytes": verified["serialized_bytes"],
        "manifests": manifests,
        "same_train_manifest_R0_P0": True,
        "mutual_gate_and_full_disabled": True,
        "feature_extraction_completed": True,
        "training_allowed": True,
        "next_gate": "MATCHED_R0_P0_FULL_WIDTH_PREFLIGHT_THEN_THREE_SEEDS",
    }
    report_path = output_root / "finalization_report.json"
    if report_path.exists():
        raise FileExistsError(f"finalization report already exists: {report_path}")
    atomic_write_json(report_path, report)
    print(json.dumps({**report, "report_file_sha256": file_sha256(report_path)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("verify-plan").set_defaults(func=command_verify_plan)
    subparsers.add_parser("verify-preflight").set_defaults(func=command_verify_preflight)
    subparsers.add_parser("finalize").set_defaults(func=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
