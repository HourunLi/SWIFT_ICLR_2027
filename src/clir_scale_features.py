"""Fail-closed contracts for CLIR Consistency scale-v6.1 features.

This module contains the deterministic, CPU-testable portion of the selected-
inventory feature pipeline.  Model loading and GPU execution live in the thin
``extract_clir_scale_features.py`` entry point.  The important invariant is
that only trajectories named by the published v6.1 inventory can reach the
extractor.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.clir_smoke import canonical_sha256, file_sha256


AUTHORIZATION_SCHEMA = "clir-consistency-scale-v6.1-feature-extraction-authorization"
SELECTED_INPUT_SCHEMA = "clir-consistency-scale-selected-feature-input-v6.1"
PLAN_SCHEMA = "clir-consistency-scale-feature-extraction-plan-v6.1"
QUERY_MARKER_SCHEMA = "clir-consistency-scale-feature-query-marker-v6.1"
WORKER_REPORT_SCHEMA = "clir-consistency-scale-feature-worker-report-v6.1"
VERIFIER_REPORT_SCHEMA = "clir-consistency-scale-feature-worker-verification-v6.1"
EXTRACTED_ROW_SCHEMA = "clir-consistency-scale-extracted-feature-row-v6.1"


def validate_exact_ids(values: Any, *, field: str, row_id: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be an integer sequence")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{row_id}: {field} contains a non-integer token ID")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{row_id}: {field} contains a negative token ID")
        output.append(integer)
    if not output:
        raise ValueError(f"{row_id}: {field} must not be empty")
    return output


def stable_name(namespace: str, identifier: str) -> str:
    payload = f"clir-scale-v6.1-feature|{namespace}|{identifier}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trajectory_relative_path(trajectory_id: str) -> str:
    digest = stable_name("trajectory", trajectory_id)
    return f"payloads/trajectories/{digest[:2]}/{digest}.pt"


def condition_relative_path(query_id: str) -> str:
    digest = stable_name("condition", query_id)
    return f"payloads/conditions/{digest[:2]}/{digest}.pt"


def query_marker_relative_path(query_id: str) -> str:
    digest = stable_name("query-marker", query_id)
    return f"query_markers/{digest[:2]}/{digest}.json"


def _materialized_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError("materialized row is missing id")
        if row_id in result:
            raise ValueError(f"duplicate materialized trajectory id: {row_id}")
        result[row_id] = dict(row)
    return result


def build_selected_inputs(
    inventory_rows: Sequence[Mapping[str, Any]],
    materialized_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the immutable inventory to exact saved IDs without retokenizing."""

    materialized = _materialized_index(materialized_rows)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    prompts: dict[str, tuple[int, ...]] = {}
    owners: Counter[str] = Counter()

    for inventory_index, inventory in enumerate(inventory_rows):
        trajectory_id = str(inventory.get("trajectory_id", ""))
        if not trajectory_id:
            raise ValueError("inventory row is missing trajectory_id")
        if trajectory_id in seen:
            raise ValueError(f"duplicate inventory trajectory id: {trajectory_id}")
        seen.add(trajectory_id)
        if trajectory_id not in materialized:
            raise ValueError(
                f"inventory trajectory is absent from materialized rows: {trajectory_id}"
            )
        source = materialized[trajectory_id]
        query_id = str(inventory.get("query_id", ""))
        if not query_id or str(source.get("query_id")) != query_id:
            raise ValueError(f"{trajectory_id}: query_id drift")
        candidate_index = source.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, Integral)
            or int(candidate_index) != inventory.get("candidate_index")
        ):
            raise ValueError(f"{trajectory_id}: candidate_index drift")
        prompt_ids = validate_exact_ids(
            source.get("prompt_token_ids"),
            field="prompt_token_ids",
            row_id=trajectory_id,
        )
        output_ids = validate_exact_ids(
            source.get("output_token_ids"),
            field="output_token_ids",
            row_id=trajectory_id,
        )
        if len(prompt_ids) != inventory.get("prompt_token_count"):
            raise ValueError(f"{trajectory_id}: prompt token count drift")
        if len(output_ids) != inventory.get("output_token_count"):
            raise ValueError(f"{trajectory_id}: output token count drift")
        prompt_key = tuple(prompt_ids)
        if query_id in prompts and prompts[query_id] != prompt_key:
            raise ValueError(f"{trajectory_id}: prompt IDs drift within query")
        prompts[query_id] = prompt_key
        owner = inventory.get("condition_feature_owner")
        if not isinstance(owner, bool):
            raise ValueError(f"{trajectory_id}: condition_feature_owner must be bool")
        if owner:
            owners[query_id] += 1

        selected.append(
            {
                "schema_version": SELECTED_INPUT_SCHEMA,
                "inventory_index": inventory_index,
                "id": trajectory_id,
                "trajectory_id": trajectory_id,
                "query_id": query_id,
                "cluster_id": inventory.get("cluster_id"),
                "acquisition_split": inventory.get("acquisition_split"),
                "source": inventory.get("source"),
                "source_subject": inventory.get("source_subject"),
                "source_level": inventory.get("source_level"),
                "candidate_index": int(candidate_index),
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "prompt_token_count": len(prompt_ids),
                "output_token_count": len(output_ids),
                "condition_feature_owner": owner,
                "uses": list(inventory.get("uses", [])),
                "relation_ids": list(inventory.get("relation_ids", [])),
            }
        )

    bad_owners = {
        query_id: owners[query_id]
        for query_id in sorted(prompts)
        if owners[query_id] != 1
    }
    if bad_owners:
        raise ValueError(
            f"each query must have exactly one condition owner: {bad_owners}"
        )
    return selected


def assign_workers(
    selected_rows: Sequence[Mapping[str, Any]], worker_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Apply deterministic largest-first token-balanced query bin packing."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_query[str(row["query_id"])].append(row)
    query_costs: dict[str, int] = {}
    for query_id, rows in by_query.items():
        prompt_counts = {int(row["prompt_token_count"]) for row in rows}
        if len(prompt_counts) != 1:
            raise ValueError(f"{query_id}: inconsistent prompt counts")
        query_costs[query_id] = next(iter(prompt_counts)) + sum(
            int(row["output_token_count"]) for row in rows
        )

    loads = [0] * worker_count
    queries = [0] * worker_count
    trajectories = [0] * worker_count
    assignment: dict[str, int] = {}
    ordered_queries = sorted(
        by_query,
        key=lambda query_id: (
            -query_costs[query_id],
            stable_name("worker-tiebreak", query_id),
        ),
    )
    for query_id in ordered_queries:
        worker = min(range(worker_count), key=lambda value: (loads[value], value))
        assignment[query_id] = worker
        loads[worker] += query_costs[query_id]
        queries[worker] += 1
        trajectories[worker] += len(by_query[query_id])

    assigned = []
    for source in selected_rows:
        row = dict(source)
        row["worker_index"] = assignment[str(row["query_id"])]
        assigned.append(row)
    stats = [
        {
            "worker_index": index,
            "query_count": queries[index],
            "trajectory_count": trajectories[index],
            "feature_token_count": loads[index],
        }
        for index in range(worker_count)
    ]
    return assigned, stats


def selected_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    output_tokens = sum(int(row["output_token_count"]) for row in rows)
    prompt_tokens = 0
    for query_id, query_rows in by_query.items():
        owners = [row for row in query_rows if row["condition_feature_owner"]]
        if len(owners) != 1:
            raise ValueError(f"{query_id}: expected one condition owner")
        prompt_tokens += int(owners[0]["prompt_token_count"])
    return {
        "trajectory_count": len(rows),
        "query_count": len(by_query),
        "condition_count": len(by_query),
        "output_token_count": output_tokens,
        "prompt_token_count": prompt_tokens,
        "total_feature_token_count": output_tokens + prompt_tokens,
    }


def rows_for_worker(
    rows: Sequence[Mapping[str, Any]], worker_index: int
) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        if int(source["worker_index"]) == worker_index:
            row = dict(source)
            by_query[str(row["query_id"])].append(row)
    for query_id, query_rows in by_query.items():
        query_rows.sort(
            key=lambda row: (
                not bool(row["condition_feature_owner"]),
                int(row["inventory_index"]),
            )
        )
        if sum(bool(row["condition_feature_owner"]) for row in query_rows) != 1:
            raise ValueError(f"{query_id}: expected one condition owner")
    return dict(sorted(by_query.items()))


def tensor_raw_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * element_size


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def validate_tensor_file(
    path: str | Path,
    *,
    expected_shape: Sequence[int],
    expected_dtype: torch.dtype,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    tensor_path = Path(path)
    if not tensor_path.is_file():
        raise FileNotFoundError(f"feature tensor is missing: {tensor_path}")
    observed_sha256 = file_sha256(tensor_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"feature tensor checksum drift: {tensor_path}")
    value = _safe_torch_load(tensor_path)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"feature payload is not a tensor: {tensor_path}")
    shape = [int(dimension) for dimension in value.shape]
    if shape != [int(dimension) for dimension in expected_shape]:
        raise ValueError(
            f"feature tensor shape drift at {tensor_path}: "
            f"{shape} != {list(expected_shape)}"
        )
    if value.dtype != expected_dtype:
        raise ValueError(
            f"feature tensor dtype drift at {tensor_path}: "
            f"{value.dtype} != {expected_dtype}"
        )
    if not value.is_contiguous():
        raise ValueError(f"feature tensor is not contiguous: {tensor_path}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"feature tensor is non-finite: {tensor_path}")
    return {
        "path": str(tensor_path),
        "shape": shape,
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": observed_sha256,
        "serialized_bytes": tensor_path.stat().st_size,
        "raw_tensor_bytes": tensor_raw_bytes(shape, value.dtype),
    }


def expected_payload_records(
    output_root: str | Path,
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(output_root)
    records: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    for row in selected_rows:
        query_id = str(row["query_id"])
        if query_id not in seen_conditions:
            seen_conditions.add(query_id)
            records.append(
                {
                    "kind": "condition",
                    "id": query_id,
                    "relative_path": condition_relative_path(query_id),
                    "path": str(root / condition_relative_path(query_id)),
                    "shape": [int(row["prompt_token_count"]), 101376],
                }
            )
        trajectory_id = str(row["trajectory_id"])
        records.append(
            {
                "kind": "trajectory",
                "id": trajectory_id,
                "relative_path": trajectory_relative_path(trajectory_id),
                "path": str(root / trajectory_relative_path(trajectory_id)),
                "shape": [int(row["output_token_count"]), 101376],
            }
        )
    return records


def payload_record_digest(records: Sequence[Mapping[str, Any]]) -> str:
    stable = [
        {
            "kind": record["kind"],
            "id": record["id"],
            "relative_path": record["relative_path"],
            "shape": list(record["shape"]),
            "sha256": record["sha256"],
            "serialized_bytes": int(record["serialized_bytes"]),
            "raw_tensor_bytes": int(record["raw_tensor_bytes"]),
        }
        for record in records
    ]
    return canonical_sha256(stable)
