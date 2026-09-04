#!/usr/bin/env python
"""Apply the hash-pinned official SWIFT MATH checker to protected rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from prepare_clir_math_hard_eval import load_protocol
from src.clir_smoke import (
    atomic_write_json,
    file_sha256,
    publish_manifest,
    read_jsonl,
    validate_rollout_population,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/math_hard_eval_v1/protocol.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "run_artifacts/math_hard_eval_v1"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def ensure_swift_checkout(protocol: Mapping[str, Any]) -> Path:
    checker = protocol["checker"]
    checkout = _project_path(protocol["runtime"]["swift_checkout"])
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", checker["upstream_repository"], str(checkout)])
        _run(["git", "checkout", "--detach", checker["upstream_commit"]], cwd=checkout)
    if _run(["git", "rev-parse", "HEAD"], cwd=checkout) != checker["upstream_commit"]:
        raise ValueError("SWIFT checker checkout commit drift")
    if _run(["git", "status", "--porcelain"], cwd=checkout):
        raise ValueError("SWIFT checker checkout is dirty")
    for relative, expected in checker["upstream_files"].items():
        if file_sha256(checkout / relative) != expected:
            raise ValueError(f"SWIFT checker source hash drift: {relative}")
    return checkout


def _grade(payload: tuple[str, str, str]) -> tuple[bool, bool, str | None, str | None]:
    checkout, response, reference = payload
    if checkout not in sys.path:
        sys.path.insert(0, checkout)
    try:
        from generate.generate_utils import evaluate_math

        matched, correct, extracted = evaluate_math(response, reference)
        return (
            bool(matched),
            bool(correct),
            None if extracted is None else str(extracted),
            None,
        )
    except Exception as exc:  # Fail closed and retain the candidate.
        return False, False, None, f"{type(exc).__name__}: {exc}"


def _grade_rows(
    rows: Iterable[Mapping[str, Any]], checkout: Path, workers: int
) -> list[tuple[bool, bool, str | None, str | None]]:
    payloads = [
        (str(checkout), str(row.get("response", "")), str(row["reference_answer"]))
        for row in rows
    ]
    if workers <= 1:
        return [_grade(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_grade, payloads, chunksize=8))


def _source_contract(protocol: Mapping[str, Any], root: Path) -> tuple[Path, list[dict[str, Any]]]:
    completion_path = root / "rollout_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS_MATH_HARD_EVAL_V1_RAW_ROLLOUTS":
        raise ValueError("protected rollout is not complete")
    source_path = Path(completion["combined"]["path"])
    if file_sha256(source_path) != completion["combined"]["file_sha256"]:
        raise ValueError("protected rollout hash drift")
    rows = read_jsonl(source_path)
    validate_rollout_population(
        rows, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    if len(rows) != int(protocol["source"]["total_queries"]) * int(
        protocol["generation"]["candidate_count"]
    ):
        raise ValueError("protected rollout row count drift")
    return source_path, rows


def command_fetch(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    checkout = ensure_swift_checkout(protocol)
    print(json.dumps({"status": "PASS_SWIFT_CHECKER_FETCH", "path": str(checkout)}, indent=2))


def command_materialize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    source_path, rows = _source_contract(protocol, root)
    completion = json.loads((root / "rollout_completion.json").read_text(encoding="utf-8"))
    if completion.get("protocol_file_sha256") != file_sha256(protocol_path):
        raise ValueError("protected rollout protocol binding drift")
    target = root / "checker/checked.jsonl"
    completion_path = root / "checker/completion.json"
    if target.exists() or completion_path.exists():
        raise FileExistsError("protected checker output already exists")
    checkout = ensure_swift_checkout(protocol)
    grades = _grade_rows(rows, checkout, args.workers)
    output: list[dict[str, Any]] = []
    for raw, (matched, correct, extracted, error) in zip(rows, grades, strict=True):
        row = dict(raw)
        row.update(
            {
                "checker_version": protocol["checker"]["version"],
                "correctness_semantics": "official_SWIFT_MATH_expression_equivalence",
                "checker_status": "expression_match" if correct else "expression_mismatch",
                "correctness": int(correct),
                "parsed_answer": extracted,
                "checker_answer_marker_matched": matched,
                "checker_exception": error,
                "eligible_for_supervision": False,
                "evaluation_only": True,
            }
        )
        output.append(row)
    population = validate_rollout_population(
        output, candidate_count=int(protocol["generation"]["candidate_count"])
    )
    health = {
        "rows": len(output),
        "queries": int(population["queries"]),
        "correct_rows": sum(int(row["correctness"]) for row in output),
        "correct_rate": sum(int(row["correctness"]) for row in output) / len(output),
        "answer_marker_matched": sum(bool(row["checker_answer_marker_matched"]) for row in output),
        "checker_exceptions": sum(row["checker_exception"] is not None for row in output),
        "finish_reason_counts": dict(sorted(Counter(str(row.get("finish_reason")) for row in output).items())),
        "by_level": {
            str(level): {
                "rows": sum(int(row["source_level"]) == level for row in output),
                "correct": sum(
                    int(row["correctness"])
                    for row in output
                    if int(row["source_level"]) == level
                ),
            }
            for level in (4, 5)
        },
        "query_filtering_after_checker": 0,
    }
    manifest = publish_manifest(
        target,
        output,
        schema_version="clir-math-hard-eval-v1-checked",
        metadata={"checker": protocol["checker"], "health": health},
    )
    report = {
        "schema_version": "clir-math-hard-eval-v1-checker-completion",
        "status": "PASS_MATH_HARD_EVAL_V1_SWIFT_CHECKER",
        "protocol_file_sha256": file_sha256(protocol_path),
        "source": {"path": str(source_path), "file_sha256": file_sha256(source_path)},
        "swift_checkout": {
            "path": str(checkout),
            "commit": protocol["checker"]["upstream_commit"],
        },
        "checked": {
            "path": str(target),
            "rows": len(output),
            "file_sha256": manifest["file_sha256"],
            "ordered_rows_sha256": manifest["ordered_rows_sha256"],
            "sidecar_file_sha256": file_sha256(
                target.with_suffix(target.suffix + ".manifest.json")
            ),
        },
        "health": health,
        "query_filtering_after_checker": False,
        "clir_scores_opened": False,
    }
    atomic_write_json(completion_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_verify(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    root = Path(args.output_root).resolve()
    completion_path = root / "checker/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "PASS_MATH_HARD_EVAL_V1_SWIFT_CHECKER"
        or completion.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise ValueError("protected checker completion is stale")
    checked_path = Path(completion["checked"]["path"])
    if file_sha256(checked_path) != completion["checked"]["file_sha256"]:
        raise ValueError("protected checked manifest hash drift")
    rows = read_jsonl(checked_path)
    checkout = ensure_swift_checkout(protocol)
    recomputed = _grade_rows(rows, checkout, args.workers)
    mismatches = 0
    for row, (matched, correct, extracted, error) in zip(rows, recomputed, strict=True):
        mismatches += int(
            int(row["correctness"]) != int(correct)
            or bool(row["checker_answer_marker_matched"]) != matched
            or row["parsed_answer"] != extracted
            or row["checker_exception"] != error
        )
    if mismatches:
        raise ValueError(f"official SWIFT checker recompute mismatches: {mismatches}")
    report = {
        "schema_version": "clir-math-hard-eval-v1-checker-verification",
        "status": "PASS_MATH_HARD_EVAL_V1_CHECKER_INDEPENDENT_RECOMPUTE",
        "protocol_file_sha256": file_sha256(protocol_path),
        "checker_completion_file_sha256": file_sha256(completion_path),
        "rows": len(rows),
        "mismatches": 0,
        "query_filtering_after_checker": False,
        "clir_scores_opened": False,
    }
    atomic_write_json(root / "checker/independent_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    # The pinned upstream grader implements symbolic timeouts with process-local
    # SIGALRM timers.  Reusing those workers in ProcessPoolExecutor can leave a
    # timer firing between tasks and terminate the whole pool.  Sequential
    # grading is deterministic, fast enough for the 8,000-row frozen set, and
    # keeps each timeout inside ``_grade``'s fail-closed exception boundary.
    parser.add_argument("--workers", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch").set_defaults(func=command_fetch)
    sub.add_parser("materialize").set_defaults(func=command_materialize)
    sub.add_parser("verify").set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
