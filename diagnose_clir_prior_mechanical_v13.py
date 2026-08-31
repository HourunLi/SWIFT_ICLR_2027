#!/usr/bin/env python
"""Run a read-only mechanical block projection over terminal Prior v12 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.clir_prior_mechanical import diagnose_v12_mechanical_projection


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "run_artifacts/data_expansion_prior_v12/pre_annotation"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/data_expansion_prior_v12/protocol.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "evaluation/mechanical_v13_prototype_replay.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _read_shards(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL shards under {directory}")
    return [row for path in paths for row in _read_jsonl(path)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    strict = protocol["strict_consensus"]
    report = diagnose_v12_mechanical_projection(
        packages_a=_read_shards(args.root / "packages/annotator_a"),
        packages_b=_read_shards(args.root / "packages/annotator_b"),
        private_index=_read_jsonl(args.root / "packages/PRIVATE_package_index.jsonl"),
        labels_a=_read_shards(args.root / "labels_a"),
        labels_b=_read_shards(args.root / "labels_b"),
        proposals=_read_jsonl(args.root / "proposals/prior_natural_800.jsonl"),
        final_strata=strict["final_strata"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
