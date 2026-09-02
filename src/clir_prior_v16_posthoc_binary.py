"""Isolated post-hoc mechanical/binary replay over the old Prior-v16 pool.

This module does not amend the terminal v16 or v17 decisions.  It applies the
already-developed v17 mechanical-Key/residual-binary representation to the
600-row v16 population in a separately named, explicitly post-hoc data route.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Integral
from typing import Any, Mapping, Sequence

from src.clir_prior_binary_v17 import (
    PACKAGE_SCHEMA as V17_PACKAGE_SCHEMA,
    PRIVATE_SCHEMA as V17_PRIVATE_SCHEMA,
    STRUCTURE_SCHEMA as V17_STRUCTURE_SCHEMA,
    binary_target_signature,
    build_hidden_controls_v17,
    compile_binary_structure_v17,
    evaluate_binary_smoke_v17,
    public_package_item_v17,
    validate_binary_annotation_v17,
)
from src.clir_smoke import canonical_sha256, stable_priority


PROTOCOL_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-v1"
PROPOSAL_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-proposal-v1"
PACKAGE_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-package-v1"
STRUCTURE_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-structure-v1"
LABEL_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-label-v1"
PRIVATE_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-private-v1"
EVALUATION_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-evaluation-v1"
ROW_SCHEMA = "clir-prior-v16-posthoc-mechanical-binary-silver-row-v1"
DEFAULT_NAMESPACE = "clir-prior-v16-posthoc-mechanical-binary-v1"
LABEL_NAME = "posthoc_dual_ai_mechanical_key_binary_prior_v16_v1_no_human_verification"


def select_v16_posthoc_rows(
    original_proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen mechanical compiler to every original v16 row."""

    materialized = {str(row["id"]): row for row in materialized_rows}
    selected: list[dict[str, Any]] = []
    rejection: Counter[str] = Counter()
    for original in original_proposals:
        source_id = str(original["source_row_id"])
        if source_id not in materialized:
            raise ValueError(f"missing v16 materialized source: {source_id}")
        source = materialized[source_id]
        for field in (
            "query_id",
            "cluster_id",
            "source",
            "checker_status",
            "candidate_index",
        ):
            if source.get(field) != original.get(field):
                raise ValueError(f"{source_id}: v16 proposal/source {field} drift")
        candidate = {
            "schema_version": PROPOSAL_SCHEMA,
            "item_id": "prior-v16-posthoc-"
            + stable_priority(f"{namespace}:item", source_id)[:24],
            "original_v16_item_id": str(original["item_id"]),
            "source_row_id": source_id,
            "query_id": str(original["query_id"]),
            "cluster_id": str(original["cluster_id"]),
            "source": str(original["source"]),
            "checker_status": str(original["checker_status"]),
            "prior_label_split": str(original["prior_label_split"]),
            "candidate_index": int(original["candidate_index"]),
            "source_record_id": original.get("source_record_id"),
            "material_claim_count": int(original["material_claim_count"]),
            "selection_priority": str(original["selection_priority"]),
            "question": str(original["question"]),
            "response": str(original["response"]),
            "parsed_answer": str(source.get("parsed_answer", "")),
            "units": deepcopy(list(original["units"])),
        }
        try:
            structure = compile_binary_structure_v17(candidate)
        except ValueError as exc:
            rejection[str(exc)] += 1
            continue
        if len(structure["residual_block_ids"]) < 2:
            rejection["fewer than two residual blocks"] += 1
            continue
        candidate["mechanical_key_block_id"] = int(structure["key_block_id"])
        candidate["residual_block_count"] = len(structure["residual_block_ids"])
        candidate["fixed_non_main_block_count"] = len(
            structure["fixed_non_main_block_ids"]
        )
        selected.append(candidate)
    selected.sort(key=lambda row: (row["selection_priority"], row["item_id"]))
    if len({row["query_id"] for row in selected}) != len(selected) or len(
        {row["cluster_id"] for row in selected}
    ) != len(selected):
        raise ValueError("post-hoc selected rows are not query/cluster distinct")
    strata = Counter(
        (row["source"], row["checker_status"], row["prior_label_split"])
        for row in selected
    )
    return selected, {
        "input_v16_rows": len(original_proposals),
        "selected_rows": len(selected),
        "distinct_queries": len({row["query_id"] for row in selected}),
        "distinct_clusters": len({row["cluster_id"] for row in selected}),
        "selected_train_rows": sum(
            row["prior_label_split"] == "train" for row in selected
        ),
        "selected_dev_rows": sum(row["prior_label_split"] == "dev" for row in selected),
        "selected_by_stratum": {
            "|".join(key): value for key, value in sorted(strata.items())
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "mean_residual_blocks": sum(row["residual_block_count"] for row in selected)
        / max(1, len(selected)),
        "residual_decisions_per_annotator": sum(
            row["residual_block_count"] for row in selected
        ),
        "ordered_item_ids_sha256": canonical_sha256(
            [row["item_id"] for row in selected]
        ),
        "posthoc_population": True,
    }


def public_posthoc_item(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    base = public_package_item_v17(source, item_id=item_id)
    base["schema_version"] = PACKAGE_SCHEMA
    base["structure"]["schema_version"] = STRUCTURE_SCHEMA
    return base


def _as_v17_item(item: Mapping[str, Any]) -> dict[str, Any]:
    clone = deepcopy(dict(item))
    clone["schema_version"] = V17_PACKAGE_SCHEMA
    clone["structure"]["schema_version"] = V17_STRUCTURE_SCHEMA
    return clone


def validate_posthoc_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    if item.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("post-hoc package schema drift")
    normalized = validate_binary_annotation_v17(annotation, _as_v17_item(item))
    normalized["schema_version"] = LABEL_SCHEMA
    return normalized


_CONTROL_OVERRIDES: dict[str, dict[int, str]] = {
    "unused_early_guess": {0: "not_used", 1: "not_used", 2: "not_used"},
    "unused_alternative": {0: "not_used", 1: "not_used", 2: "not_used"},
    "used_case_choice": {0: "not_used", 1: "used", 2: "used"},
    "duplicate_and_plan_removed": {1: "not_used", 2: "not_used"},
}


def build_posthoc_controls(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Reuse v17 examples with deletion-rule-consistent post-hoc answers."""

    output = []
    for index, (old_item, old_expected, name) in enumerate(
        build_hidden_controls_v17(annotator)
    ):
        item = deepcopy(old_item)
        item["item_id"] = f"prior-v16-posthoc-control-{annotator}-{index:02d}"
        item["schema_version"] = PACKAGE_SCHEMA
        item["structure"]["schema_version"] = STRUCTURE_SCHEMA
        decisions = {
            int(row["block_id"]): str(row["decision"])
            for row in old_expected["residual_decisions"]
        }
        decisions.update(_CONTROL_OVERRIDES.get(name, {}))
        expected = validate_posthoc_annotation(
            {
                "item_id": item["item_id"],
                "residual_decisions": [
                    {"block_id": block_id, "decision": decisions[block_id]}
                    for block_id in item["structure"]["residual_block_ids"]
                ],
                "confidence": "high",
                "rationale": f"post-hoc deletion-rule control: {name}",
            },
            item,
        )
        output.append((item, expected, name))
    return output


def build_posthoc_shards(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 10,
    repeats_per_shard: int = 5,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    if len(proposals) != 490 or shard_count != 10 or repeats_per_shard != 5:
        raise ValueError("v16 post-hoc v1 freezes 490 natural rows in ten shards")
    ordered = sorted(
        (dict(row) for row in proposals),
        key=lambda row: stable_priority(f"{namespace}:shard", row["item_id"]),
    )
    natural_shards = [
        [public_posthoc_item(row) for row in ordered[index::shard_count]]
        for index in range(shard_count)
    ]
    if any(len(rows) != 49 for rows in natural_shards):
        raise AssertionError("post-hoc natural shard balance drift")
    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        shards = [list(rows) for rows in natural_shards]
        controls = build_posthoc_controls(annotator)
        for control_index, (item, expected, name) in enumerate(controls):
            shard_index = control_index % shard_count
            shards[shard_index].append(item)
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "shard_index": shard_index,
                    "kind": "control",
                    "item_id": item["item_id"],
                    "natural_item_id": None,
                    "control_name": name,
                    "expected_label": expected,
                }
            )
        for parent_shard, parents in enumerate(natural_shards):
            destination = (parent_shard + 1) % shard_count
            for local_index, parent in enumerate(parents[:repeats_per_shard]):
                repeat_id = (
                    f"{parent['item_id']}:repeat:{annotator}:"
                    f"{parent_shard:02d}:{local_index:02d}"
                )
                repeated = deepcopy(parent)
                repeated["item_id"] = repeat_id
                shards[destination].append(repeated)
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": destination,
                        "kind": "repeat",
                        "item_id": repeat_id,
                        "natural_item_id": parent["item_id"],
                        "control_name": None,
                        "expected_label": None,
                    }
                )
        for shard_index, rows in enumerate(shards):
            for natural in natural_shards[shard_index]:
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": shard_index,
                        "kind": "natural",
                        "item_id": natural["item_id"],
                        "natural_item_id": natural["item_id"],
                        "control_name": None,
                        "expected_label": None,
                    }
                )
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}:package:{annotator}:{shard_index}", row["item_id"]
                )
            )
        packages[annotator] = shards
    private.sort(key=lambda row: (row["annotator"], row["shard_index"], row["item_id"]))
    return (
        packages,
        private,
        {
            "shards_per_annotator": shard_count,
            "natural_rows_per_shard": 49,
            "controls_per_annotator": 12,
            "repeats_per_shard": repeats_per_shard,
            "repeats_per_annotator": shard_count * repeats_per_shard,
            "rows_per_shard": [len(rows) for rows in packages["a"]],
            "rows_per_annotator": sum(len(rows) for rows in packages["a"]),
            "posthoc": True,
            "ai_task": "residual_used_or_not_used_only",
        },
    )


def _normalized_maps(
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    package_map = {str(row["item_id"]): dict(row) for row in packages["a"]}
    normalized: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for side in ("a", "b"):
        side_packages = {str(row["item_id"]): dict(row) for row in packages[side]}
        side_labels = {str(row.get("item_id")): dict(row) for row in labels[side]}
        if len(side_labels) != len(labels[side]) or set(side_labels) != set(
            side_packages
        ):
            raise ValueError(f"post-hoc {side} package/label population mismatch")
        for item_id, item in side_packages.items():
            normalized[side][item_id] = validate_posthoc_annotation(
                side_labels[item_id], item
            )
    return package_map, normalized


def evaluate_posthoc_replay(
    *,
    proposals: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the v17 metric implementation, then apply publishable-row gates."""

    converted_packages: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for side in ("a", "b"):
        converted_packages[side] = [_as_v17_item(row) for row in packages[side]]
    converted_private = []
    for raw in private_index:
        row = deepcopy(dict(raw))
        row["schema_version"] = V17_PRIVATE_SCHEMA
        converted_private.append(row)
    base = evaluate_binary_smoke_v17(
        proposals=proposals,
        packages=converted_packages,
        private_index=converted_private,
        labels=labels,
        gates=gates,
    )
    base["schema_version"] = EVALUATION_SCHEMA
    if base.get("schema_errors"):
        base["status"] = "FAIL_PRIOR_V16_POSTHOC_BINARY_SCHEMA"
        base["next_step"] = "stop_posthoc_replay_without_publication"
        return base

    _, normalized = _normalized_maps(packages, labels)
    repeat_failed: set[str] = set()
    for row in private_index:
        if row["kind"] != "repeat":
            continue
        side = str(row["annotator"])
        repeat = normalized[side][row["item_id"]]
        parent = normalized[side][row["natural_item_id"]]
        if repeat["confidence"] == "low" or binary_target_signature(
            repeat
        ) != binary_target_signature(parent):
            repeat_failed.add(str(row["natural_item_id"]))
    proposal_by_id = {str(row["item_id"]): row for row in proposals}
    eligible_ids = [
        item_id
        for item_id in sorted(
            proposal_by_id,
            key=lambda value: (proposal_by_id[value]["selection_priority"], value),
        )
        if item_id not in repeat_failed
        and normalized["a"][item_id]["confidence"] != "low"
        and normalized["b"][item_id]["confidence"] != "low"
    ]
    train_count = sum(
        proposal_by_id[item_id]["prior_label_split"] == "train"
        for item_id in eligible_ids
    )
    dev_count = len(eligible_ids) - train_count
    publishable = {
        "eligible_rows": len(eligible_ids),
        "train_rows": train_count,
        "dev_rows": dev_count,
        "repeat_failed_parent_count": len(repeat_failed),
        "repeat_failed_parent_ids": sorted(repeat_failed),
        "ordered_item_ids": eligible_ids,
        "ordered_item_ids_sha256": canonical_sha256(eligible_ids),
    }
    base["publishable_population"] = publishable
    base["gate_checks"]["publishable_train_rows"] = train_count >= int(
        gates["publishable_train_rows_min"]
    )
    base["gate_checks"]["publishable_dev_rows"] = dev_count >= int(
        gates["publishable_dev_rows_min"]
    )
    passed = all(base["gate_checks"].values())
    base["status"] = (
        "PASS_PRIOR_V16_POSTHOC_BINARY_REPLAY"
        if passed
        else "STOP_PRIOR_V16_POSTHOC_BINARY_REPLAY"
    )
    base["posthoc_claim_boundary"] = {
        "original_v16_status_unchanged": "STOP_PRIOR_V16_ROLE_ONLY_SCALE",
        "v17_status_unchanged": "STOP_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE",
        "population_was_used_for_prior_target_development": True,
        "labels_are_silver_not_gold": True,
        "human_verification": False,
        "future_ranking_confirmation_must_use_fresh_query_clusters": True,
    }
    base["trainable_labels_published"] = False
    base["next_step"] = (
        "materialize_only_publishable_population"
        if passed
        else "stop_posthoc_replay_without_subset_salvage"
    )
    return base


def _integer_tokens(value: Any, *, field: str, row_id: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be an integer array")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0:
            raise ValueError(f"{row_id}: {field} contains an invalid token ID")
        output.append(int(item))
    if not output:
        raise ValueError(f"{row_id}: {field} is empty")
    return output


def _unit_spans(
    row: Mapping[str, Any], output_count: int
) -> dict[int, tuple[int, int, str]]:
    spans: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    for raw in row.get("units", []):
        if not isinstance(raw, Mapping):
            raise ValueError("unit is not an object")
        index, start, end = (
            raw.get("unit_index"),
            raw.get("token_start"),
            raw.get("token_end"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (index, start, end)
        ):
            raise ValueError("unit index/span is invalid")
        index, start, end = int(index), int(start), int(end)
        if index in spans or start != cursor or not start <= end <= output_count:
            raise ValueError("unit partition is invalid")
        spans[index] = (start, end, str(raw.get("kind")))
        cursor = end
    if cursor != output_count:
        raise ValueError("units do not cover exact output-token axis")
    return spans


def construct_posthoc_silver_rows(
    *,
    proposals: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluation_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if evaluation_report.get("status") != "PASS_PRIOR_V16_POSTHOC_BINARY_REPLAY":
        raise ValueError("post-hoc materialization requires a passing report")
    package_map, normalized = _normalized_maps(packages, labels)
    proposal_by_id = {str(row["item_id"]): row for row in proposals}
    source_by_id = {str(row["id"]): row for row in materialized_rows}
    selected_ids = list(evaluation_report["publishable_population"]["ordered_item_ids"])
    if (
        canonical_sha256(selected_ids)
        != evaluation_report["publishable_population"]["ordered_item_ids_sha256"]
    ):
        raise ValueError("post-hoc selected-ID binding drift")
    output = []
    for selection_index, item_id in enumerate(selected_ids):
        proposal = proposal_by_id[item_id]
        source = source_by_id[str(proposal["source_row_id"])]
        prompt_ids = _integer_tokens(
            source["prompt_token_ids"], field="prompt_token_ids", row_id=source["id"]
        )
        output_ids = _integer_tokens(
            source["output_token_ids"], field="output_token_ids", row_id=source["id"]
        )
        spans = _unit_spans(source, len(output_ids))
        structure = package_map[item_id]["structure"]
        left = {
            row["block_id"]: row["decision"]
            for row in normalized["a"][item_id]["residual_decisions"]
        }
        right = {
            row["block_id"]: row["decision"]
            for row in normalized["b"][item_id]["residual_decisions"]
        }
        positive_blocks = [int(structure["key_block_id"])] + [
            block_id for block_id in left if left[block_id] == right[block_id] == "used"
        ]
        ambiguous_blocks = [
            block_id for block_id in left if left[block_id] != right[block_id]
        ]
        blocks = {int(row["block_id"]): row for row in structure["blocks"]}
        positive_units = sorted(
            unit
            for block_id in positive_blocks
            for unit in blocks[block_id]["unit_indices"]
        )
        ambiguous_units = sorted(
            unit
            for block_id in ambiguous_blocks
            for unit in blocks[block_id]["unit_indices"]
        )
        key_units = list(map(int, structure["key_unit_indices"]))
        key_target = [0] * len(output_ids)
        complete_target = [0] * len(output_ids)
        key_mask = [1] * len(output_ids)
        complete_mask = [1] * len(output_ids)
        for unit in key_units:
            start, end, kind = spans[unit]
            if kind != "material_claim":
                raise ValueError("Key unit is not material")
            key_target[start:end] = [1] * (end - start)
        for unit in positive_units:
            start, end, kind = spans[unit]
            if kind != "material_claim":
                raise ValueError("Complete-positive unit is not material")
            complete_target[start:end] = [1] * (end - start)
        for unit in ambiguous_units:
            start, end, kind = spans[unit]
            if kind != "material_claim":
                raise ValueError("Complete-ambiguous unit is not material")
            complete_mask[start:end] = [0] * (end - start)
        if any(
            key and not complete
            for key, complete in zip(key_target, complete_target, strict=True)
        ):
            raise AssertionError("post-hoc Key must be Complete-positive")
        split = str(proposal["prior_label_split"])
        output.append(
            {
                "schema_version": ROW_SCHEMA,
                "selection_index": selection_index,
                "id": str(source["id"]),
                "trajectory_id": str(source["id"]),
                "proposal_id": item_id,
                "original_v16_item_id": proposal["original_v16_item_id"],
                "query_id": str(proposal["query_id"]),
                "cluster_id": str(proposal["cluster_id"]),
                "candidate_index": int(proposal["candidate_index"]),
                "source": proposal["source"],
                "source_record_id": proposal.get("source_record_id"),
                "split": split,
                "prior_label_split": split,
                "checker_status": proposal["checker_status"],
                "correctness": int(source["correctness"]),
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "prompt_token_count": len(prompt_ids),
                "output_token_count": len(output_ids),
                "key_unit_indices": key_units,
                "complete_unit_indices": positive_units,
                "complete_ambiguous_unit_indices": ambiguous_units,
                "key_prior_target": key_target,
                "key_prior_mask": key_mask,
                "complete_prior_target": complete_target,
                "complete_prior_mask": complete_mask,
                "prior_label_name": LABEL_NAME,
                "prior_label_source": "posthoc_blind_dual_ai_mechanical_binary_consensus",
                "prior_human_verified": False,
                "feature_role": (
                    "prior_train" if split == "train" else "prior_dev_posthoc"
                ),
            }
        )
    return output, {
        "schema_version": "clir-prior-v16-posthoc-materialization-report-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_SILVER_MATERIALIZATION",
        "selected_rows": len(output),
        "train_rows": sum(row["split"] == "train" for row in output),
        "dev_rows": sum(row["split"] == "dev" for row in output),
        "selected_ordered_item_ids_sha256": canonical_sha256(selected_ids),
        "output_token_count": sum(row["output_token_count"] for row in output),
        "key_positive_token_count": sum(sum(row["key_prior_target"]) for row in output),
        "complete_positive_token_count": sum(
            sum(row["complete_prior_target"]) for row in output
        ),
        "complete_supervised_token_count": sum(
            sum(row["complete_prior_mask"]) for row in output
        ),
        "human_verified": False,
        "posthoc": True,
        "feature_extraction_started": False,
        "training_started": False,
    }


def validate_posthoc_silver_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_item_ids_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on the exact-token contract used by the clean trainer."""

    ids: set[str] = set()
    proposal_ids: list[str] = []
    query_ids: set[str] = set()
    cluster_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    ambiguous_rows = 0
    for raw in rows:
        row = dict(raw)
        row_id = str(row.get("id", ""))
        proposal_id = str(row.get("proposal_id", ""))
        query_id = str(row.get("query_id", ""))
        cluster_id = str(row.get("cluster_id", ""))
        if row.get("schema_version") != ROW_SCHEMA or not all(
            (row_id, proposal_id, query_id, cluster_id)
        ):
            raise ValueError("post-hoc Silver row identity/schema drift")
        if row_id in ids or query_id in query_ids or cluster_id in cluster_ids:
            raise ValueError(
                "post-hoc Silver rows are not trajectory/query/cluster distinct"
            )
        ids.add(row_id)
        query_ids.add(query_id)
        cluster_ids.add(cluster_id)
        proposal_ids.append(proposal_id)

        prompt_ids = _integer_tokens(
            row.get("prompt_token_ids"), field="prompt_token_ids", row_id=row_id
        )
        output_ids = _integer_tokens(
            row.get("output_token_ids"), field="output_token_ids", row_id=row_id
        )
        if row.get("prompt_token_count") != len(prompt_ids) or row.get(
            "output_token_count"
        ) != len(output_ids):
            raise ValueError(f"{row_id}: token count metadata drift")
        vectors = {}
        for field in (
            "key_prior_target",
            "key_prior_mask",
            "complete_prior_target",
            "complete_prior_mask",
        ):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != len(output_ids):
                raise ValueError(f"{row_id}: {field} length drift")
            if any(value not in (0, 1) or isinstance(value, bool) for value in values):
                raise ValueError(f"{row_id}: {field} must be a binary integer vector")
            vectors[field] = values
        if vectors["key_prior_mask"] != [1] * len(output_ids):
            raise ValueError(f"{row_id}: Key coverage must span the full output axis")
        if not any(vectors["key_prior_target"]):
            raise ValueError(f"{row_id}: Key has no positive output token")
        if any(
            key and not complete
            for key, complete in zip(
                vectors["key_prior_target"],
                vectors["complete_prior_target"],
                strict=True,
            )
        ):
            raise ValueError(f"{row_id}: Key is not contained in Complete")
        if any(
            not mask and target
            for mask, target in zip(
                vectors["complete_prior_mask"],
                vectors["complete_prior_target"],
                strict=True,
            )
        ):
            raise ValueError(f"{row_id}: masked Complete token is positive")
        if (
            row.get("prior_label_name") != LABEL_NAME
            or row.get("prior_human_verified") is not False
        ):
            raise ValueError(f"{row_id}: label provenance drift")
        split = str(row.get("split"))
        if split not in {"train", "dev"} or row.get("prior_label_split") != split:
            raise ValueError(f"{row_id}: post-hoc split drift")
        split_counts[split] += 1
        ambiguous_rows += bool(row.get("complete_ambiguous_unit_indices"))

    actual_ids_sha256 = canonical_sha256(proposal_ids)
    if (
        expected_item_ids_sha256 is not None
        and actual_ids_sha256 != expected_item_ids_sha256
    ):
        raise ValueError("post-hoc Silver proposal order/hash drift")
    return {
        "schema_version": "clir-prior-v16-posthoc-silver-validation-v1",
        "status": "PASS_PRIOR_V16_POSTHOC_SILVER_VALIDATION",
        "rows": len(rows),
        "train_rows": split_counts["train"],
        "dev_rows": split_counts["dev"],
        "distinct_trajectories": len(ids),
        "distinct_queries": len(query_ids),
        "distinct_clusters": len(cluster_ids),
        "rows_with_masked_residual_disagreement": ambiguous_rows,
        "ordered_proposal_ids_sha256": actual_ids_sha256,
        "labels_are_silver": True,
        "human_verified": False,
        "posthoc": True,
    }
