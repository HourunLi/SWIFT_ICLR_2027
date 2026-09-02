#!/usr/bin/env python
"""Summarize the matched expanded-data U0/C/H0/CH decomposition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluate_clir import atomic_write_json, file_sha256
from evaluate_clir_three_module_factorial import h_metrics
from score_clir_prior_v16_ch_decomposition import (
    AUTHORIZATION_STATUS,
    CELLS,
    MERGE_STATUS,
    SEEDS,
    _audit_checkpoint,
    _project_path,
)
from src.clir_data import read_jsonl
from summarize_clir_h0_experiment import summarize as summarize_h0_factorial
from summarize_clir_prior_v16_reused_ranking import _load_run
from summarize_clir_prior_v16_training import _validate_scored_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "configs/data_expansion_prior_v16/posthoc_training_v1/ch_decomposition_v1.json"
)
CELL_ALIASES = {"u0": "c0", "c": "c1", "h": "h0", "ch": "ch0"}


def _evaluate_unbalanced_h_dev(
    rows: list[dict[str, Any]],
    *,
    onset_threshold: float,
    onset_window_tokens: int,
) -> dict[str, Any]:
    """Use all 197 cross-module-cleaned rows, not the legacy 100/100 guard."""
    report = h_metrics(
        rows,
        onset_threshold=onset_threshold,
        onset_window_tokens=onset_window_tokens,
    )
    checkpoint_hashes = {
        str(row.get("clir_checkpoint_sha256", "")) for row in rows
    }
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("H dev score file must bind exactly one checkpoint")
    report["checkpoint_sha256"] = next(iter(checkpoint_hashes))
    return report


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _consistency_report(
    path: Path,
    *,
    cell: str,
    seed: int,
    checkpoint_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load_json(path)
    expected_cell = "c1" if protocol["cells"][cell]["factors"][0] else "c0"
    if (
        report.get("status") != "PASS_HELDOUT_RELATION_EVALUATION"
        or report.get("cell") != expected_cell
        or int(report.get("seed", -1)) != seed
        or int(report.get("completed_epoch", -1))
        != int(protocol["training"]["epochs"])
        or report.get("inputs", {}).get("checkpoint", {}).get("file_sha256")
        != checkpoint_sha256
    ):
        raise ValueError(f"Consistency report identity drift: {path}")
    expected = {
        "endpoint_manifest": protocol["mechanism_evaluation"][
            "consistency_endpoints"
        ]["file_sha256"],
        "positive_relations": protocol["mechanism_evaluation"][
            "consistency_positive_relations"
        ]["file_sha256"],
        "negative_relations": protocol["mechanism_evaluation"][
            "consistency_hard_negative_relations"
        ]["file_sha256"],
    }
    if any(
        report.get("inputs", {}).get(key, {}).get("file_sha256") != value
        for key, value in expected.items()
    ):
        raise ValueError(f"Consistency input hash drift: {path}")
    return report


def _nested(payload: Mapping[str, Any], dotted: str) -> float:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def _aggregate_consistency(
    reports: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[str, Any]:
    paths = {
        "representation_separation": (
            "representation.mean_separation_positive_minus_negative"
        ),
        "representation_auroc": (
            "representation.relation_classification_auroc"
        ),
        "representation_average_precision": (
            "representation.relation_classification_average_precision"
        ),
        "score_gap_separation": "score.mean_gap_separation_negative_minus_positive",
    }
    return {
        metric: {
            cell: {
                "mean": float(
                    np.mean([_nested(reports[(cell, seed)], path) for seed in SEEDS])
                ),
                "by_seed": {
                    str(seed): _nested(reports[(cell, seed)], path)
                    for seed in SEEDS
                },
            }
            for cell in CELLS
        }
        for metric, path in paths.items()
    }


def summarize(protocol_path: Path, merge_path: Path) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema_version")
        != "clir-prior-v16-posthoc-ch-decomposition-v1"
        or protocol.get("status") != AUTHORIZATION_STATUS
        or protocol.get("design", {}).get("cells") != list(CELLS)
        or protocol.get("training", {}).get("seeds") != list(SEEDS)
    ):
        raise ValueError("inactive or malformed decomposition protocol")
    protocol_sha = file_sha256(protocol_path)
    ranking_spec = protocol["ranking_evaluation"]
    ranking_path = _project_path(ranking_spec["path"])
    h_spec = protocol["mechanism_evaluation"]["h_dev"]
    h_path = _project_path(h_spec["path"])
    if (
        file_sha256(ranking_path) != ranking_spec["file_sha256"]
        or file_sha256(h_path) != h_spec["file_sha256"]
    ):
        raise ValueError("evaluation input hash drift")
    ranking_reference = read_jsonl(ranking_path)
    h_reference = read_jsonl(h_path)

    checkpoints = {
        (cell, seed): _audit_checkpoint(protocol, cell, seed)
        for cell in CELLS
        for seed in SEEDS
    }
    merge = _load_json(merge_path)
    if (
        merge.get("status") != MERGE_STATUS
        or merge.get("authorization_file_sha256") != protocol_sha
        or merge.get("input_jsonl_sha256") != ranking_spec["file_sha256"]
        or merge.get("completion_report_sha256") != protocol_sha
        or int(merge.get("num_shards", -1))
        != int(protocol["runtime"]["ranking_shards"])
    ):
        raise ValueError("ranking merge report violates the protocol")

    scored: dict[tuple[str, int], dict[str, Any]] = {}
    consistency_reports: dict[tuple[str, int], dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    output_root = _project_path(protocol["runtime"]["output_root"])
    for cell in CELLS:
        for seed in SEEDS:
            key = f"{cell}/seed-{seed}"
            checkpoint_sha = checkpoints[(cell, seed)]["checkpoint_file_sha256"]
            ranking_record = merge.get("outputs", {}).get(key)
            if not isinstance(ranking_record, Mapping):
                raise ValueError(f"ranking merge lacks {key}")
            ranking_scored = Path(str(ranking_record["path"]))
            _load_run(
                path=ranking_scored,
                expected_file_sha256=str(ranking_record["file_sha256"]),
                expected_checkpoint_sha256=checkpoint_sha,
                reference_rows=ranking_reference,
                k_values=[int(value) for value in ranking_spec["k"]],
            )
            h_scored = (
                PROJECT_ROOT
                / "run_artifacts/data_expansion_prior_v16_posthoc_training_v1"
                / f"evaluation/h_dev_scored/ch_seed-{seed}.jsonl"
                if cell == "ch"
                else output_root / f"mechanism/h_dev_scored/{cell}_seed-{seed}.jsonl"
            )
            h_rows, h_artifact = _validate_scored_rows(
                h_path, h_scored, checkpoint_sha
            )
            consistency_path = (
                PROJECT_ROOT
                / "run_artifacts/data_expansion_prior_v16_posthoc_training_v1"
                / f"evaluation/consistency_reports/ch_seed-{seed}.json"
                if cell == "ch"
                else output_root
                / f"mechanism/consistency_reports/{cell}_seed-{seed}.json"
            )
            consistency = _consistency_report(
                consistency_path,
                cell=cell,
                seed=seed,
                checkpoint_sha256=checkpoint_sha,
                protocol=protocol,
            )
            consistency_reports[(cell, seed)] = consistency
            scored[(CELL_ALIASES[cell], seed)] = {
                "h_dev": h_rows,
                "ranking": read_jsonl(ranking_scored),
            }
            artifacts[key] = {
                "checkpoint": checkpoints[(cell, seed)],
                "ranking": {
                    "path": str(ranking_scored),
                    "file_sha256": str(ranking_record["file_sha256"]),
                },
                "h_dev": h_artifact,
                "consistency": {
                    "path": str(consistency_path),
                    "file_sha256": file_sha256(consistency_path),
                },
            }

    factorial = summarize_h0_factorial(
        scored,
        k_values=[int(value) for value in ranking_spec["k"]],
        bootstrap_replicates=int(ranking_spec["bootstrap_replicates"]),
        bootstrap_seed=int(ranking_spec["bootstrap_seed"]),
        onset_threshold=0.5,
        onset_window_tokens=5,
        h_evaluator=_evaluate_unbalanced_h_dev,
    )
    factorial["schema_version"] = "clir-prior-v16-posthoc-ch-decomposition-summary-v1"
    factorial["status"] = "COMPLETE_MATCHED_U0_C_H_CH_DECOMPOSITION"
    factorial["created_at_utc"] = _utc_now()
    factorial["evidence_tier"] = protocol["evidence_boundary"]["tier"]
    factorial["original_v16_v17_terminal_results_unchanged"] = True
    factorial["protocol"] = {
        "path": str(protocol_path),
        "file_sha256": protocol_sha,
    }
    factorial["ranking_merge_report"] = {
        "path": str(merge_path),
        "file_sha256": file_sha256(merge_path),
    }
    factorial["consistency_heldout"] = _aggregate_consistency(
        consistency_reports
    )
    factorial["run_artifacts"] = artifacts
    factorial["cell_name_mapping"] = {
        "c0": "u0",
        "c1": "c",
        "h0": "h",
        "ch0": "ch",
    }
    factorial["claim_boundary"] = (
        "Matched post-hoc dual-AI Silver C-by-H0 decomposition on one 5,552-row "
        "training manifest and the already-inspected 892-query ranking population. "
        "Not fresh, protected, confirmatory, Gold, or human-verified; no retuning."
    )
    return factorial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--merge-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    report = summarize(Path(args.protocol).resolve(), Path(args.merge_report).resolve())
    atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
