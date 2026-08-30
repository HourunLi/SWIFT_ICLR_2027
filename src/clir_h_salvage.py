"""Post-hoc exploratory H0 salvage for the terminal ranking-v7 labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from src.clir_smoke import FINAL_H_VALUES, canonical_sha256, materialize_h_label


SALVAGE_LABEL_SCHEMA = "clir-h0-v7.4-posthoc-exploratory-salvage"


def _target_signature(label: Mapping[str, Any]) -> tuple[str, int | None]:
    onset = label.get("first_bad_unit_index")
    return str(label.get("status")), None if onset is None else int(onset)


def find_retry_self_repeat_failures(
    *,
    private_rows: Sequence[Mapping[str, Any]],
    retry_labels_by_annotator: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[set[str], dict[str, Any]]:
    """Return natural proposal IDs whose retry repeat disagreed with its primary."""

    failed_ids: set[str] = set()
    reports: dict[str, Any] = {}
    for annotator in ("a", "b"):
        labels = retry_labels_by_annotator.get(annotator, {})
        repeats = [
            row
            for row in private_rows
            if row.get("annotator") == annotator
            and row.get("role") == "self_repeat"
        ]
        agree = 0
        annotator_failures: list[str] = []
        for record in repeats:
            repeat_id = str(record["item_id"])
            primary_id = str(record["repeat_of_item_id"])
            if repeat_id not in labels or primary_id not in labels:
                raise ValueError(
                    f"annotator {annotator}: retry repeat label is missing"
                )
            proposal_id = str(record["canonical_item_id"])
            if _target_signature(labels[repeat_id]) == _target_signature(
                labels[primary_id]
            ):
                agree += 1
            else:
                failed_ids.add(proposal_id)
                annotator_failures.append(proposal_id)
        reports[annotator] = {
            "repeat_rows": len(repeats),
            "exact_agree": agree,
            "exact_agreement": agree / len(repeats) if repeats else None,
            "failed_natural_rows": len(annotator_failures),
            "failed_natural_ids_sha256": canonical_sha256(
                sorted(annotator_failures)
            ),
        }
    reports["union"] = {
        "failed_natural_rows": len(failed_ids),
        "failed_natural_ids_sha256": canonical_sha256(sorted(failed_ids)),
    }
    return failed_ids, reports


def _materialize_salvage_row(
    *,
    proposal: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
    consensus_sources: Sequence[str],
    stage: str,
    label_name: str,
) -> dict[str, Any]:
    status, onset = _target_signature(labels[0])
    label_source = (
        "posthoc_smoke_dual_exact_non_low"
        if stage == "smoke"
        else "posthoc_reserve_a2_b2_b1_exact_non_low_repeat_failures_excluded"
    )
    materialized = materialize_h_label(
        {
            "status": status,
            "first_bad_unit_index": onset,
            "label_source": label_source,
        },
        proposal,
        label_tier=label_name,
    )
    return {
        **dict(proposal),
        **materialized,
        "h_status": status,
        "h_label_name": label_name,
        "h_label_selection": (
            "posthoc_exact_consensus_then_original_frozen_proposal_priority"
        ),
        "h_salvage_stage": stage,
        "h_posthoc_exploratory": True,
        "h_original_v7_status": "FAIL_H0_V7_RESERVE",
        "h_consensus_sources": list(consensus_sources),
        "h_consensus_confidences": [str(label["confidence"]) for label in labels],
    }


def build_h_salvage_rows(
    *,
    proposals: Sequence[Mapping[str, Any]],
    smoke_labels_by_annotator: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    reserve_attempt_1_b: Mapping[str, Mapping[str, Any]],
    reserve_attempt_2_by_annotator: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ],
    repeat_failed_proposal_ids: set[str],
    targets: Mapping[str, int],
    label_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the deterministic post-hoc pool and balanced selected subset."""

    proposal_by_id = {str(row["proposal_id"]): dict(row) for row in proposals}
    if len(proposal_by_id) != len(proposals):
        raise ValueError("H salvage proposals contain duplicate proposal IDs")
    smoke_ids = set(smoke_labels_by_annotator.get("a", {}))
    if smoke_ids != set(smoke_labels_by_annotator.get("b", {})):
        raise ValueError("H salvage smoke A/B populations differ")
    reserve_ids = set(reserve_attempt_2_by_annotator.get("a", {}))
    if reserve_ids != set(reserve_attempt_2_by_annotator.get("b", {})):
        raise ValueError("H salvage retry A/B populations differ")
    if reserve_ids != set(reserve_attempt_1_b):
        raise ValueError("H salvage reserve B1/B2 populations differ")
    if smoke_ids & reserve_ids or smoke_ids | reserve_ids != set(proposal_by_id):
        raise ValueError("H salvage smoke/reserve partition differs from proposals")

    eligible: list[dict[str, Any]] = []
    discarded: Counter[str] = Counter()
    for proposal_id in sorted(smoke_ids):
        labels = [
            smoke_labels_by_annotator["a"][proposal_id],
            smoke_labels_by_annotator["b"][proposal_id],
        ]
        if len({_target_signature(label) for label in labels}) != 1:
            discarded["smoke|target_disagreement"] += 1
            continue
        if str(labels[0]["status"]) not in FINAL_H_VALUES:
            discarded["smoke|non_final_status"] += 1
            continue
        if any(label["confidence"] == "low" for label in labels):
            discarded["smoke|low_confidence"] += 1
            continue
        eligible.append(
            _materialize_salvage_row(
                proposal=proposal_by_id[proposal_id],
                labels=labels,
                consensus_sources=("smoke_a", "smoke_b"),
                stage="smoke",
                label_name=label_name,
            )
        )

    for proposal_id in sorted(reserve_ids):
        labels = [
            reserve_attempt_2_by_annotator["a"][proposal_id],
            reserve_attempt_2_by_annotator["b"][proposal_id],
            reserve_attempt_1_b[proposal_id],
        ]
        if len({_target_signature(label) for label in labels}) != 1:
            discarded["reserve|triple_target_disagreement"] += 1
            continue
        if str(labels[0]["status"]) not in FINAL_H_VALUES:
            discarded["reserve|non_final_status"] += 1
            continue
        if any(label["confidence"] == "low" for label in labels):
            discarded["reserve|low_confidence"] += 1
            continue
        if proposal_id in repeat_failed_proposal_ids:
            discarded["reserve|retry_self_repeat_failure"] += 1
            continue
        eligible.append(
            _materialize_salvage_row(
                proposal=proposal_by_id[proposal_id],
                labels=labels,
                consensus_sources=(
                    "reserve_attempt_2_a",
                    "reserve_attempt_2_b",
                    "reserve_attempt_1_b",
                ),
                stage="reserve",
                label_name=label_name,
            )
        )

    eligible.sort(
        key=lambda row: (str(row["proposal_priority"]), str(row["proposal_id"]))
    )
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_cell[f"{row['h_label_split']}|{row['h_status']}"].append(row)

    quotas = {
        "dev|clean": int(targets["dev_clean"]),
        "dev|hallucinated": int(targets["dev_hallucinated"]),
        "train|clean": int(targets["train_clean"]),
        "train|hallucinated": int(targets["train_hallucinated"]),
    }
    shortages: dict[str, dict[str, int]] = {}
    selected: list[dict[str, Any]] = []
    for cell, target in sorted(quotas.items()):
        available = by_cell.get(cell, [])
        if len(available) < target:
            shortages[cell] = {"target": target, "available": len(available)}
            continue
        selected.extend(available[:target])
    if shortages:
        selected = []
    selected.sort(
        key=lambda row: (str(row["proposal_priority"]), str(row["proposal_id"]))
    )

    selected_query_ids = [str(row["query_id"]) for row in selected]
    if len(selected_query_ids) != len(set(selected_query_ids)):
        raise ValueError("H salvage selected more than one trajectory per query")
    train_queries = {
        str(row["query_id"])
        for row in selected
        if row["h_label_split"] == "train"
    }
    dev_queries = {
        str(row["query_id"])
        for row in selected
        if row["h_label_split"] == "dev"
    }
    if train_queries & dev_queries:
        raise ValueError("H salvage train/dev query overlap")

    selected_by_cell = Counter(
        f"{row['h_label_split']}|{row['h_status']}" for row in selected
    )
    status = (
        "PASS_H0_V7_4_POSTHOC_SALVAGE_SELECTION"
        if not shortages
        else "FAIL_H0_V7_4_POSTHOC_SALVAGE_YIELD"
    )
    return eligible, selected, {
        "status": status,
        "original_v7_status": "FAIL_H0_V7_RESERVE",
        "posthoc_exploratory": True,
        "label_name": label_name,
        "proposal_rows": len(proposals),
        "eligible_rows": len(eligible),
        "eligible_by_cell": {
            cell: len(rows) for cell, rows in sorted(by_cell.items())
        },
        "discarded_rows": sum(discarded.values()),
        "discarded_by_reason": dict(sorted(discarded.items())),
        "targets_by_cell": quotas,
        "shortages": shortages,
        "selected_rows": len(selected),
        "selected_queries": len(set(selected_query_ids)),
        "selected_by_cell": dict(sorted(selected_by_cell.items())),
        "selected_by_source": dict(
            sorted(Counter(str(row["source"]) for row in selected).items())
        ),
        "selected_by_checker_status": dict(
            sorted(
                Counter(str(row["checker_status"]) for row in selected).items()
            )
        ),
        "selected_ids_sha256": canonical_sha256(
            [str(row["proposal_id"]) for row in selected]
        ),
        "train_dev_query_overlap": len(train_queries & dev_queries),
    }


__all__ = [
    "SALVAGE_LABEL_SCHEMA",
    "build_h_salvage_rows",
    "find_retry_self_repeat_failures",
]
