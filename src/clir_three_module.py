"""Deterministic data merge for the expanded three-module CLIR factorial."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.clir_h0_experiment import rebase_feature_paths


CORE_PARITY_FIELDS = (
    "id",
    "query_id",
    "candidate_index",
    "correctness",
    "prompt_token_ids",
    "output_token_ids",
)

PRIOR_COPY_FIELDS = (
    "key_prior_target",
    "key_prior_mask",
    "complete_prior_target",
    "complete_prior_mask",
    "key_unit_indices",
    "complete_unit_indices",
    "prior_label_name",
    "prior_label_source",
    "prior_label_split",
    "prior_human_verified",
    "prior_original_v12_status",
    "prior_posthoc_exploratory",
    "annotator_a_confidence",
    "annotator_b_confidence",
    "available_self_repeat_status",
    "proposal_id",
    "selection_index",
    "selection_priority",
    "clir_supervision_provenance",
    "label_provenance",
)

PRODUCTION_EXPECTED = {
    "shared_historical_rows": 3968,
    "legacy_prior_rows": 48,
    "new_prior_rows": 202,
    "train_rows": 5370,
    "train_queries": 1493,
    "consistency_endpoint_rows": 800,
    "consistency_relations": 400,
    "h_rows": 400,
    "h_positive_rows": 200,
    "h_clean_rows": 200,
    "prior_rows": 250,
    "clean_h_dev_rows": 198,
    "clean_prior_dev_rows": 49,
}


def _unique_by_id(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError(f"{label} contains a row without id")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate id {row_id}")
        indexed[row_id] = row
    return indexed


def _validate_prior_target(row: Mapping[str, Any]) -> None:
    row_id = str(row["id"])
    token_count = len(row["output_token_ids"])
    for target in ("key_prior_target", "complete_prior_target"):
        values = row.get(target)
        if not isinstance(values, list) or len(values) != token_count:
            raise ValueError(f"{row_id}: {target} does not cover the output-token axis")
        if not values or any(value not in (0, 1) for value in values):
            raise ValueError(f"{row_id}: {target} must be a nonempty binary mask")
    for mask in ("key_prior_mask", "complete_prior_mask"):
        if mask not in row:
            continue
        values = row[mask]
        if not isinstance(values, list) or len(values) != token_count:
            raise ValueError(f"{row_id}: {mask} does not cover the output-token axis")
        if any(value not in (0, 1) for value in values):
            raise ValueError(f"{row_id}: {mask} must be binary")


def _validate_h_target(row: Mapping[str, Any]) -> None:
    row_id = str(row["id"])
    path = int(row["path_hallucinated"])
    onset = int(row["hallucination_onset"])
    token_count = len(row["output_token_ids"])
    if path == 0 and onset != -1:
        raise ValueError(f"{row_id}: clean H row must use onset -1")
    if path == 1 and not 0 <= onset < token_count:
        raise ValueError(f"{row_id}: positive H onset is outside the output-token axis")
    if path not in (0, 1):
        raise ValueError(f"{row_id}: path_hallucinated must be binary")


def _rebase(
    row: Mapping[str, Any],
    source_parent: Path,
    target_parent: Path,
    *,
    row_schema: str,
    experiment_population: str,
) -> dict[str, Any]:
    rebased = rebase_feature_paths(
        row, source_parent=source_parent, target_parent=target_parent
    )
    rebased["source_experiment_population"] = row.get("experiment_population")
    rebased["schema_version"] = row_schema
    rebased["experiment_population"] = experiment_population
    return rebased


def _query_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["query_id"]) for row in rows}


def build_unified_data(
    *,
    consistency_h0_train: Sequence[Mapping[str, Any]],
    prior_train: Sequence[Mapping[str, Any]],
    h_dev: Sequence[Mapping[str, Any]],
    prior_dev: Sequence[Mapping[str, Any]],
    consistency_h0_parent: Path,
    prior_parent: Path,
    h_dev_parent: Path,
    prior_dev_parent: Path,
    target_parent: Path,
    expected: Mapping[str, int] | None = None,
    row_schema: str = "clir-three-module-expansion-v1-row",
    experiment_population: str = "three_module_expansion_v1",
    appended_prior_origin: str = "v12_posthoc_appended_row",
) -> dict[str, Any]:
    """Merge shared historical rows, append new Prior rows, and clean dev leaks."""

    expected = dict(PRODUCTION_EXPECTED if expected is None else expected)
    h_by_id = _unique_by_id(consistency_h0_train, "consistency/H0 train")
    p_by_id = _unique_by_id(prior_train, "Prior train")
    shared_ids = set(h_by_id) & set(p_by_id)
    if len(shared_ids) != int(expected["shared_historical_rows"]):
        raise ValueError(f"shared historical id count drift: {len(shared_ids)}")
    h_shared_order = [
        str(row["id"]) for row in consistency_h0_train if str(row["id"]) in shared_ids
    ]
    p_shared_order = [
        str(row["id"]) for row in prior_train if str(row["id"]) in shared_ids
    ]
    if h_shared_order != p_shared_order:
        raise ValueError("shared historical row order drift")

    for row_id in h_shared_order:
        left = h_by_id[row_id]
        right = p_by_id[row_id]
        for field in CORE_PARITY_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"{row_id}: shared core field drift for {field}")

    prior_labeled = [
        row
        for row in prior_train
        if "key_prior_target" in row or "complete_prior_target" in row
    ]
    if any(
        "key_prior_target" not in row or "complete_prior_target" not in row
        for row in prior_labeled
    ):
        raise ValueError("Prior train contains a one-sided Key/Complete target")
    for row in prior_labeled:
        _validate_prior_target(row)

    legacy_prior_ids = {str(row["id"]) for row in prior_labeled} & shared_ids
    new_prior_rows = [row for row in prior_train if str(row["id"]) not in shared_ids]
    if len(legacy_prior_ids) != int(expected["legacy_prior_rows"]) or len(
        new_prior_rows
    ) != int(expected["new_prior_rows"]):
        raise ValueError("merged legacy or appended Prior row count drift")
    if any("key_prior_target" not in row for row in new_prior_rows):
        raise ValueError("every appended Prior row must carry direct supervision")

    unified_train: list[dict[str, Any]] = []
    for source_row in consistency_h0_train:
        row = _rebase(
            source_row,
            consistency_h0_parent,
            target_parent,
            row_schema=row_schema,
            experiment_population=experiment_population,
        )
        prior_row = p_by_id.get(str(source_row["id"]))
        if prior_row is not None and str(source_row["id"]) in legacy_prior_ids:
            for field in PRIOR_COPY_FIELDS:
                if field in prior_row:
                    row[field] = prior_row[field]
            row["prior_merge_origin"] = "legacy_shared_historical_row"
        unified_train.append(row)
    for source_row in new_prior_rows:
        row = _rebase(
            source_row,
            prior_parent,
            target_parent,
            row_schema=row_schema,
            experiment_population=experiment_population,
        )
        row["prior_merge_origin"] = appended_prior_origin
        unified_train.append(row)

    unified_by_id = _unique_by_id(unified_train, "unified train")
    train_queries = _query_set(unified_train)
    consistency_rows = [
        row for row in unified_train if row.get("consistency_supervision") is True
    ]
    relation_counts = Counter(str(row.get("semantic_id")) for row in consistency_rows)
    if len(consistency_rows) != int(expected["consistency_endpoint_rows"]) or len(
        relation_counts
    ) != int(expected["consistency_relations"]):
        raise ValueError("Consistency supervision count drift")
    if relation_counts and set(relation_counts.values()) != {2}:
        raise ValueError("every Consistency relation must have exactly two endpoints")

    h_rows = [row for row in unified_train if "path_hallucinated" in row]
    for row in h_rows:
        _validate_h_target(row)
    h_counts = Counter(int(row["path_hallucinated"]) for row in h_rows)

    unified_prior_rows = [row for row in unified_train if "key_prior_target" in row]
    for row in unified_prior_rows:
        _validate_prior_target(row)

    if (
        len(unified_train) != int(expected["train_rows"])
        or len(unified_by_id) != int(expected["train_rows"])
        or len(train_queries) != int(expected["train_queries"])
        or len(h_rows) != int(expected["h_rows"])
        or h_counts
        != Counter(
            {
                0: int(expected["h_clean_rows"]),
                1: int(expected["h_positive_rows"]),
            }
        )
        or len(unified_prior_rows) != int(expected["prior_rows"])
    ):
        raise ValueError("unified training inventory drift")

    clean_h_dev_source = [
        row for row in h_dev if str(row["query_id"]) not in train_queries
    ]
    clean_prior_dev_source = [
        row for row in prior_dev if str(row["query_id"]) not in train_queries
    ]
    if len(clean_h_dev_source) != int(expected["clean_h_dev_rows"]) or len(
        clean_prior_dev_source
    ) != int(expected["clean_prior_dev_rows"]):
        raise ValueError("cross-module query-disjoint dev count drift")
    clean_h_dev = [
        _rebase(
            row,
            h_dev_parent,
            target_parent,
            row_schema=row_schema,
            experiment_population=experiment_population,
        )
        for row in clean_h_dev_source
    ]
    clean_prior_dev = [
        _rebase(
            row,
            prior_dev_parent,
            target_parent,
            row_schema=row_schema,
            experiment_population=experiment_population,
        )
        for row in clean_prior_dev_source
    ]
    if _query_set(clean_h_dev) & train_queries:
        raise ValueError("clean H dev still overlaps unified train")
    if _query_set(clean_prior_dev) & train_queries:
        raise ValueError("clean Prior dev still overlaps unified train")

    task_queries = {
        "consistency": _query_set(consistency_rows),
        "h": _query_set(h_rows),
        "prior": _query_set(unified_prior_rows),
    }
    query_overlap = {
        "consistency_h": len(task_queries["consistency"] & task_queries["h"]),
        "consistency_prior": len(task_queries["consistency"] & task_queries["prior"]),
        "h_prior": len(task_queries["h"] & task_queries["prior"]),
    }
    return {
        "train": unified_train,
        "h_dev": clean_h_dev,
        "prior_dev": clean_prior_dev,
        "report": {
            "train_rows": len(unified_train),
            "train_unique_ids": len(unified_by_id),
            "train_queries": len(train_queries),
            "shared_historical_rows": len(shared_ids),
            "legacy_prior_rows_merged": len(legacy_prior_ids),
            "new_prior_rows_appended": len(new_prior_rows),
            "consistency_endpoint_rows": len(consistency_rows),
            "consistency_relations": len(relation_counts),
            "h_rows": len(h_rows),
            "h_positive_rows": h_counts[1],
            "h_clean_rows": h_counts[0],
            "prior_rows": len(unified_prior_rows),
            "clean_h_dev_rows": len(clean_h_dev),
            "clean_prior_dev_rows": len(clean_prior_dev),
            "removed_h_dev_queries": sorted(_query_set(h_dev) & train_queries),
            "removed_prior_dev_queries": sorted(_query_set(prior_dev) & train_queries),
            "train_task_query_overlap": query_overlap,
        },
    }


__all__ = [
    "CORE_PARITY_FIELDS",
    "PRIOR_COPY_FIELDS",
    "PRODUCTION_EXPECTED",
    "build_unified_data",
]
