#!/usr/bin/env python
"""Materialize the frozen Prior/Gate numeric checker and final populations.

This CPU-only stage can run the pre-registered numeric checker and select the
fixed 800-query tuning and 800-query confirmation populations.  It cannot
extract hidden states, train a model, tune a weight, or score CLIR.  The
confirmation checker rows remain in files whose names and manifests are
explicitly sealed until the weight decision is locked.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from prepare_clir_gate_tuning import load_protocol
from src.clir_gate_tuning import (
    CONFIRMATION_ROLE,
    TUNING_ROLE,
    materialize_numeric_checker_rows,
    select_checker_eligible_rows,
    validate_numeric_checker_rows,
)
from src.clir_smoke import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    publish_manifest,
    read_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/prior_gate_tuning_v1/protocol.json"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT / "configs/prior_gate_tuning_v1/checker_authorization.json"
)
DEFAULT_PRE_ROLLOUT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/pre_rollout"
DEFAULT_ROLLOUT_ROOT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/rollouts"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run_artifacts/prior_gate_tuning_v1/checker"
AUTHORIZATION_SCHEMA = "clir-prior-gate-tuning-v1-checker-authorization"
AUTHORIZATION_STATUS = "AUTHORIZED_CHECKER_AND_FROZEN_SELECTION_ONLY"
COMPLETION_STATUS = "PASS_GATE_TUNING_V1_CHECKER_AND_FROZEN_SELECTION"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _require_clean_commit() -> str:
    if _git_dirty():
        raise RuntimeError("Prior/Gate checker stage requires a clean Git commit")
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
    if result.returncode != 0:
        raise ValueError(
            f"authorized code parent {ancestor} is not an ancestor of HEAD"
        )


def _assert_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"{label} hash drift: {observed} != {expected_sha256}")


def _read_published_jsonl(
    path: Path, *, expected_schema: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sidecar_path = path.with_suffix(path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != expected_schema:
        raise ValueError(f"unexpected schema for {path}")
    _assert_file(path, str(sidecar["file_sha256"]), str(path))
    rows = read_jsonl(path)
    if len(rows) != int(sidecar["row_count"]):
        raise ValueError(f"row count drift for {path}")
    if canonical_sha256(rows) != sidecar["ordered_rows_sha256"]:
        raise ValueError(f"ordered row hash drift for {path}")
    return rows, sidecar


def _load_authorization(
    path: Path,
    *,
    protocol_path: Path,
    pre_rollout_dir: Path,
    rollout_root: Path,
) -> dict[str, Any]:
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported Prior/Gate checker authorization")
    if authorization.get("status") != AUTHORIZATION_STATUS:
        raise ValueError("Prior/Gate checker stage is not authorized")
    expected_scope = {
        "numeric_checker": True,
        "checker_eligibility_selection": True,
        "confirmation_checker_materialization_to_sealed_file": True,
        "confirmation_outcome_distribution_disclosure": False,
        "feature_extraction": False,
        "new_training": False,
        "tuning_scoring": False,
        "confirmation_scoring": False,
        "quota_or_checker_change": False,
        "adaptive_additional_sampling": False,
    }
    if authorization.get("authorized_scope") != expected_scope:
        raise ValueError("Prior/Gate checker authorization scope drift")
    parent = authorization["frozen_parent"]
    expected_files = {
        "protocol_file_sha256": protocol_path,
        "tuning_queries_file_sha256": pre_rollout_dir / "tuning_queries.jsonl",
        "confirmation_queries_file_sha256": (
            pre_rollout_dir / "confirmation_queries.jsonl"
        ),
        "tuning_raw_file_sha256": rollout_root / "tuning_combined_raw.jsonl",
        "tuning_raw_sidecar_file_sha256": (
            rollout_root / "tuning_combined_raw.jsonl.manifest.json"
        ),
        "confirmation_raw_file_sha256": (
            rollout_root / "confirmation_combined_raw.sealed.jsonl"
        ),
        "confirmation_raw_sidecar_file_sha256": (
            rollout_root / "confirmation_combined_raw.sealed.jsonl.manifest.json"
        ),
    }
    for field, file_path in expected_files.items():
        _assert_file(file_path, str(parent[field]), field)
    if int(parent["tuning_raw_rows"]) != 20800:
        raise ValueError("authorized tuning raw row count drift")
    if int(parent["confirmation_raw_rows"]) != 20800:
        raise ValueError("authorized confirmation raw row count drift")
    if authorization["sealing_contract"].get("confirmation_remains_sealed") is not True:
        raise ValueError("confirmation sealing contract drift")
    return authorization


def _load_contract(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
    Path,
]:
    protocol_path = Path(args.protocol).resolve()
    authorization_path = Path(args.authorization).resolve()
    pre_rollout_dir = Path(args.pre_rollout).resolve()
    rollout_root = Path(args.rollout_root).resolve()
    output_root = Path(args.output_root).resolve()
    protocol, _ = load_protocol(protocol_path)
    authorization = _load_authorization(
        authorization_path,
        protocol_path=protocol_path,
        pre_rollout_dir=pre_rollout_dir,
        rollout_root=rollout_root,
    )
    expected_output = (
        PROJECT_ROOT / authorization["runtime_contract"]["output_root"]
    ).resolve()
    if output_root != expected_output:
        raise ValueError(
            f"checker output root drift: {output_root} != {expected_output}"
        )
    tuning_queries = read_jsonl(pre_rollout_dir / "tuning_queries.jsonl")
    confirmation_queries = read_jsonl(pre_rollout_dir / "confirmation_queries.jsonl")
    tuning_raw, _ = _read_published_jsonl(
        rollout_root / "tuning_combined_raw.jsonl",
        expected_schema="clir-gate-tuning-v1-tuning-raw-rollouts",
    )
    confirmation_raw, _ = _read_published_jsonl(
        rollout_root / "confirmation_combined_raw.sealed.jsonl",
        expected_schema="clir-gate-tuning-v1-confirmation-raw-rollouts-sealed",
    )
    if len(tuning_queries) != 1300 or len(confirmation_queries) != 1300:
        raise ValueError("frozen query manifest count drift")
    if any(row.get("role") != TUNING_ROLE for row in tuning_queries):
        raise ValueError("tuning query role drift")
    if any(row.get("role") != CONFIRMATION_ROLE for row in confirmation_queries):
        raise ValueError("confirmation query role drift")
    if any(row.get("sealed_until_weight_lock") is not True for row in confirmation_raw):
        raise ValueError("confirmation raw row lost sealed marker")
    return (
        protocol,
        authorization,
        tuning_queries,
        confirmation_queries,
        tuning_raw,
        confirmation_raw,
        authorization_path,
        output_root,
    )


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "tuning_checked": output_root / "tuning_checked.jsonl",
        "confirmation_checked": output_root / "confirmation_checked.sealed.jsonl",
        "tuning_selected": output_root / "tuning_selected.jsonl",
        "confirmation_selected": output_root / "confirmation_selected.sealed.jsonl",
        "completion": output_root / "completion.json",
    }


def _sidecar_hash(path: Path) -> str:
    return file_sha256(path.with_suffix(path.suffix + ".manifest.json"))


def _manifest_binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "rows": int(manifest["row_count"]),
        "file_sha256": manifest["file_sha256"],
        "ordered_rows_sha256": manifest["ordered_rows_sha256"],
        "sidecar_file_sha256": _sidecar_hash(path),
    }


def _redacted_confirmation_health(
    health: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    role = selection["by_role_source"][CONFIRMATION_ROLE]
    eligible = sum(int(values["eligible_queries"]) for values in role.values())
    return {
        "rows": int(health["rows"]),
        "queries": int(health["queries"]),
        "eligible_queries": eligible,
        "ineligible_queries": int(health["queries"]) - eligible,
        "selected_queries": sum(
            int(values["selected_queries"]) for values in role.values()
        ),
        "checker_version": health["checker_version"],
        "checker_outcome_distribution_sealed": True,
        "clir_scoring_run": False,
    }


def _require_absent(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
            raise FileExistsError(f"checker artifact already exists: {path}")


def command_materialize_select(args: argparse.Namespace) -> None:
    (
        protocol,
        authorization,
        tuning_queries,
        confirmation_queries,
        tuning_raw,
        confirmation_raw,
        authorization_path,
        output_root,
    ) = _load_contract(args)
    code_commit = _require_clean_commit()
    _require_ancestor(
        str(authorization["runtime_contract"]["authorized_code_parent_commit"]),
        code_commit,
    )
    paths = _paths(output_root)
    _require_absent(list(paths.values()))
    checker_version = str(protocol["checker"]["checker_version"])
    candidate_count = int(protocol["generation"]["candidate_count"])

    tuning_checked, tuning_health = materialize_numeric_checker_rows(
        tuning_raw, checker_version=checker_version
    )
    confirmation_checked, confirmation_health = materialize_numeric_checker_rows(
        confirmation_raw, checker_version=checker_version
    )
    tuning_validation = validate_numeric_checker_rows(
        tuning_checked,
        raw_rows=tuning_raw,
        candidate_count=candidate_count,
        checker_version=checker_version,
    )
    confirmation_validation = validate_numeric_checker_rows(
        confirmation_checked,
        raw_rows=confirmation_raw,
        candidate_count=candidate_count,
        checker_version=checker_version,
    )
    selected_tuning, selected_confirmation, selection = select_checker_eligible_rows(
        [*tuning_checked, *confirmation_checked],
        [*tuning_queries, *confirmation_queries],
        protocol,
    )
    protocol_sha = file_sha256(Path(args.protocol))
    authorization_sha = file_sha256(authorization_path)
    common_metadata = {
        "protocol_file_sha256": protocol_sha,
        "checker_authorization_file_sha256": authorization_sha,
        "code_commit": code_commit,
        "checker_version": checker_version,
        "clir_scoring_run": False,
    }
    tuning_checked_manifest = publish_manifest(
        paths["tuning_checked"],
        tuning_checked,
        schema_version="clir-gate-tuning-v1-tuning-checked",
        metadata={**common_metadata, "sealed": False, "health": tuning_health},
    )
    confirmation_checked_manifest = publish_manifest(
        paths["confirmation_checked"],
        confirmation_checked,
        schema_version="clir-gate-tuning-v1-confirmation-checked-sealed",
        metadata={
            **common_metadata,
            "sealed": True,
            "checker_outcome_distribution_sealed": True,
            "rows": len(confirmation_checked),
            "queries": len(confirmation_queries),
        },
    )
    tuning_selected_manifest = publish_manifest(
        paths["tuning_selected"],
        selected_tuning,
        schema_version="clir-gate-tuning-v1-tuning-selected",
        metadata={
            **common_metadata,
            "sealed": False,
            "queries": len(selected_tuning) // candidate_count,
            "selection_report_sha256": canonical_sha256(selection),
        },
    )
    confirmation_selected_manifest = publish_manifest(
        paths["confirmation_selected"],
        selected_confirmation,
        schema_version="clir-gate-tuning-v1-confirmation-selected-sealed",
        metadata={
            **common_metadata,
            "sealed": True,
            "checker_outcome_distribution_sealed": True,
            "queries": len(selected_confirmation) // candidate_count,
            "selection_report_sha256": canonical_sha256(selection),
        },
    )
    completion = {
        "schema_version": "clir-prior-gate-tuning-v1-checker-completion",
        "status": COMPLETION_STATUS,
        "completed_at_utc": _utc_now(),
        "code_commit": code_commit,
        "protocol_file_sha256": protocol_sha,
        "checker_authorization_file_sha256": authorization_sha,
        "checker_version": checker_version,
        "tuning": {
            "checked": _manifest_binding(
                paths["tuning_checked"], tuning_checked_manifest
            ),
            "selected": _manifest_binding(
                paths["tuning_selected"], tuning_selected_manifest
            ),
            "health": tuning_health,
            "validation": tuning_validation,
        },
        "confirmation": {
            "checked": _manifest_binding(
                paths["confirmation_checked"], confirmation_checked_manifest
            ),
            "selected": _manifest_binding(
                paths["confirmation_selected"], confirmation_selected_manifest
            ),
            "health": _redacted_confirmation_health(confirmation_health, selection),
            "validation": {
                "rows": int(confirmation_validation["rows"]),
                "queries": int(confirmation_validation["queries"]),
                "checker_rows_recomputed": int(
                    confirmation_validation["checker_rows_recomputed"]
                ),
                "candidate_axis_verified": True,
                "checker_outcome_distribution_sealed": True,
                "clir_scoring_run": False,
            },
        },
        "selection": selection,
        "confirmation_correctness_opened_for_weight_choice": False,
        "feature_extraction_started": False,
        "training_started": False,
        "tuning_scoring_started": False,
        "confirmation_scoring_started": False,
        "next_gate": "hash_bind_selected_populations_then_separate_selected_only_feature_authorization",
    }
    atomic_write_json(paths["completion"], completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


def _verify_manifest_binding(
    path: Path, binding: Mapping[str, Any], *, expected_schema: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, sidecar = _read_published_jsonl(path, expected_schema=expected_schema)
    if (
        sidecar["file_sha256"] != binding["file_sha256"]
        or sidecar["ordered_rows_sha256"] != binding["ordered_rows_sha256"]
        or _sidecar_hash(path) != binding["sidecar_file_sha256"]
    ):
        raise ValueError(f"completion binding drift for {path}")
    return rows, sidecar


def command_verify(args: argparse.Namespace) -> None:
    (
        protocol,
        _,
        tuning_queries,
        confirmation_queries,
        tuning_raw,
        confirmation_raw,
        _,
        output_root,
    ) = _load_contract(args)
    paths = _paths(output_root)
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    if completion.get("status") != COMPLETION_STATUS:
        raise ValueError("checker completion status is not a PASS")
    tuning_checked, _ = _verify_manifest_binding(
        paths["tuning_checked"],
        completion["tuning"]["checked"],
        expected_schema="clir-gate-tuning-v1-tuning-checked",
    )
    confirmation_checked, _ = _verify_manifest_binding(
        paths["confirmation_checked"],
        completion["confirmation"]["checked"],
        expected_schema="clir-gate-tuning-v1-confirmation-checked-sealed",
    )
    tuning_selected, _ = _verify_manifest_binding(
        paths["tuning_selected"],
        completion["tuning"]["selected"],
        expected_schema="clir-gate-tuning-v1-tuning-selected",
    )
    confirmation_selected, _ = _verify_manifest_binding(
        paths["confirmation_selected"],
        completion["confirmation"]["selected"],
        expected_schema="clir-gate-tuning-v1-confirmation-selected-sealed",
    )
    checker_version = str(protocol["checker"]["checker_version"])
    candidate_count = int(protocol["generation"]["candidate_count"])
    validate_numeric_checker_rows(
        tuning_checked,
        raw_rows=tuning_raw,
        candidate_count=candidate_count,
        checker_version=checker_version,
    )
    validate_numeric_checker_rows(
        confirmation_checked,
        raw_rows=confirmation_raw,
        candidate_count=candidate_count,
        checker_version=checker_version,
    )
    expected_tuning, expected_confirmation, selection = select_checker_eligible_rows(
        [*tuning_checked, *confirmation_checked],
        [*tuning_queries, *confirmation_queries],
        protocol,
    )
    if canonical_sha256(tuning_selected) != canonical_sha256(expected_tuning):
        raise ValueError("tuning selected population differs from recomputation")
    if canonical_sha256(confirmation_selected) != canonical_sha256(
        expected_confirmation
    ):
        raise ValueError("confirmation selected population differs from recomputation")
    if completion.get("selection") != selection:
        raise ValueError("stored checker selection report drift")
    source_counts = Counter(
        str(row["source"]) for row in tuning_selected[::candidate_count]
    )
    report = {
        "status": "PASS_GATE_TUNING_V1_CHECKER_INDEPENDENT_RECOMPUTE",
        "tuning_rows": len(tuning_selected),
        "tuning_queries": len(tuning_selected) // candidate_count,
        "tuning_source_counts": dict(sorted(source_counts.items())),
        "confirmation_rows": len(confirmation_selected),
        "confirmation_queries": len(confirmation_selected) // candidate_count,
        "confirmation_sealed": True,
        "confirmation_outcome_distribution_disclosed": False,
        "clir_scoring_run": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--authorization", default=str(DEFAULT_AUTHORIZATION))
    parser.add_argument("--pre-rollout", default=str(DEFAULT_PRE_ROLLOUT))
    parser.add_argument("--rollout-root", default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize-select").set_defaults(
        func=command_materialize_select
    )
    subparsers.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
