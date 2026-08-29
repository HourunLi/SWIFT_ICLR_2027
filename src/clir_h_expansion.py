"""Deterministic H0 proposal, packaging, and dual-AI label gates for v7."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_smoke import (
    FINAL_H_VALUES,
    agreement_report,
    annotation_signature,
    canonical_sha256,
    stable_priority,
    validate_annotation,
)


H_PROPOSAL_SCHEMA = "clir-h0-v7-proposals"
H_PACKAGE_SCHEMA = "clir-h0-v7-blind-package"
H_LABEL_SCHEMA = "clir-h0-v7-dual-ai-silver"


def _cell(source: str, checker_status: str, label_split: str) -> str:
    return f"{checker_status}|{label_split}|{source}"


def _expected_proposal_quotas(protocol: Mapping[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for status_split, sources in protocol["h_acquisition"]["proposal_target"].items():
        checker_status, label_split = status_split.split("|", 1)
        for source, count in sources.items():
            output[_cell(str(source), checker_status, label_split)] = int(count)
    return dict(sorted(output.items()))


def build_h_proposals(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select at most one eligible trajectory per preassigned query, then quota."""

    minimum_units = int(protocol["h_acquisition"]["minimum_material_units"])
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_query[str(raw["query_id"])].append(dict(raw))
    query_candidates: list[dict[str, Any]] = []
    query_failures: Counter[str] = Counter()
    eligible_candidates_by_cell: Counter[str] = Counter()
    query_count_by_cell: Counter[str] = Counter()
    for query_id, candidates in sorted(by_query.items()):
        indices = sorted(int(row["candidate_index"]) for row in candidates)
        if indices not in (list(range(8)), list(range(32))):
            raise ValueError(
                f"{query_id}: expected original 0..7 or rescued 0..31 candidates"
            )
        frozen = {
            (
                str(row["source"]),
                str(row["h_target_checker_status"]),
                str(row["h_label_split"]),
            )
            for row in candidates
        }
        if len(frozen) != 1:
            raise ValueError(f"{query_id}: H preassignment fields drift within query")
        source, target_status, label_split = next(iter(frozen))
        cell = _cell(source, target_status, label_split)
        query_count_by_cell[cell] += 1
        eligible = [
            row
            for row in candidates
            if row.get("unitization_status") == "ok"
            and row.get("checker_status") == target_status
            and row.get("finish_reason") != "length"
            and bool(row.get("eligible_for_supervision"))
            and int(row.get("material_claim_count", 0)) >= minimum_units
        ]
        eligible_candidates_by_cell[cell] += len(eligible)
        if not eligible:
            statuses = Counter(str(row.get("checker_status")) for row in candidates)
            if statuses.get(target_status, 0) == 0:
                query_failures[f"{cell}|no_target_checker_candidate"] += 1
            elif all(
                row.get("finish_reason") == "length"
                for row in candidates
                if row.get("checker_status") == target_status
            ):
                query_failures[f"{cell}|target_candidates_truncated"] += 1
            else:
                query_failures[f"{cell}|unit_or_eligibility_failure"] += 1
            continue
        chosen = min(
            eligible,
            key=lambda row: stable_priority(
                "clir-H-v7-candidate-within-query", str(row["id"])
            ),
        )
        chosen["proposal_id"] = str(chosen["id"])
        chosen["proposal_cell"] = cell
        chosen["within_query_eligible_candidate_count"] = len(eligible)
        chosen["proposal_priority"] = stable_priority(
            "clir-H-v7-proposal-order", str(chosen["id"])
        )
        query_candidates.append(chosen)

    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_candidates:
        by_cell[str(row["proposal_cell"])].append(row)
    quotas = _expected_proposal_quotas(protocol)
    selected: list[dict[str, Any]] = []
    shortages: dict[str, dict[str, int]] = {}
    for cell, target in quotas.items():
        available = sorted(
            by_cell.get(cell, []), key=lambda row: row["proposal_priority"]
        )
        if len(available) < target:
            shortages[cell] = {"target": target, "available": len(available)}
        selected.extend(available[:target])
    if shortages:
        raise ValueError(f"FAIL-yield: H proposal cell shortages: {shortages}")
    selected.sort(key=lambda row: row["proposal_priority"])
    target_total = int(protocol["h_acquisition"]["proposal_target_total"])
    if len(selected) != target_total:
        raise AssertionError("H proposal total differs from protocol")
    query_ids = [str(row["query_id"]) for row in selected]
    if len(query_ids) != len(set(query_ids)):
        raise AssertionError("H proposals reuse a query")
    return selected, {
        "input_rows": len(rows),
        "input_queries": len(by_query),
        "minimum_material_units": minimum_units,
        "query_counts_by_cell": dict(sorted(query_count_by_cell.items())),
        "eligible_candidate_counts_by_cell": dict(
            sorted(eligible_candidates_by_cell.items())
        ),
        "query_failures": dict(sorted(query_failures.items())),
        "surviving_query_candidates_by_cell": dict(
            sorted(Counter(row["proposal_cell"] for row in query_candidates).items())
        ),
        "proposal_targets_by_cell": quotas,
        "selected_by_cell": dict(
            sorted(Counter(row["proposal_cell"] for row in selected).items())
        ),
        "selected_rows": len(selected),
        "selected_queries": len(set(query_ids)),
        "ordered_proposal_ids_sha256": canonical_sha256(
            [row["proposal_id"] for row in selected]
        ),
    }


def split_smoke_and_reserve(
    proposals: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    smoke_cfg = protocol["h_acquisition"]["smoke"]
    required = {
        _cell(source, checker_status, "train"): int(count)
        for checker_status in ("numeric_mismatch", "numeric_match")
        for source, count in smoke_cfg[checker_status].items()
    }
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in proposals:
        row = dict(raw)
        by_cell[str(row["proposal_cell"])].append(row)
    smoke: list[dict[str, Any]] = []
    for cell, count in sorted(required.items()):
        available = sorted(
            by_cell[cell],
            key=lambda row: stable_priority(
                "clir-H-v7-smoke-selection", str(row["proposal_id"])
            ),
        )
        if len(available) < count:
            raise ValueError(
                f"FAIL-yield: smoke cell {cell} has {len(available)} < {count}"
            )
        smoke.extend(available[:count])
    smoke_ids = {str(row["proposal_id"]) for row in smoke}
    reserve = [
        dict(row) for row in proposals if str(row["proposal_id"]) not in smoke_ids
    ]
    smoke.sort(
        key=lambda row: stable_priority(
            "clir-H-v7-smoke-package-order", str(row["proposal_id"])
        )
    )
    reserve.sort(
        key=lambda row: stable_priority(
            "clir-H-v7-reserve-order", str(row["proposal_id"])
        )
    )
    if len(smoke) != int(smoke_cfg["train_proposals"]):
        raise AssertionError("smoke proposal count differs from protocol")
    if len(smoke) + len(reserve) != len(proposals):
        raise AssertionError("smoke/reserve split is not a partition")
    return (
        smoke,
        reserve,
        {
            "smoke_rows": len(smoke),
            "reserve_rows": len(reserve),
            "smoke_by_cell": dict(
                sorted(Counter(row["proposal_cell"] for row in smoke).items())
            ),
            "smoke_ids_sha256": canonical_sha256([row["proposal_id"] for row in smoke]),
            "reserve_ids_sha256": canonical_sha256(
                [row["proposal_id"] for row in reserve]
            ),
        },
    )


def _control_items() -> list[dict[str, Any]]:
    specs = [
        (
            "c0",
            "Lina has 3 red marbles and 2 blue marbles. How many marbles?",
            [
                "Lina has 3 red marbles.",
                "She has 2 blue marbles.",
                "The total is 3 + 2.",
                "3 + 2 = 5.",
                "Therefore the answer is 5.",
            ],
            "clean",
            None,
        ),
        (
            "c1",
            "A box has 12 pencils and gives away 4. How many remain?",
            [
                "The box starts with 12 pencils.",
                "It gives away 4 pencils.",
                "The remaining count is 12 - 4.",
                "12 - 4 = 8.",
                "Therefore 8 pencils remain.",
            ],
            "clean",
            None,
        ),
        (
            "c2",
            "Four bags hold 6 apples each. How many apples?",
            [
                "There are 4 bags.",
                "Each bag holds 6 apples.",
                "The total is 4 times 6.",
                "4 times 6 equals 24.",
                "Therefore there are 24 apples.",
            ],
            "clean",
            None,
        ),
        (
            "c3",
            "Twenty cookies are shared equally by 5 children. How many each?",
            [
                "There are 20 cookies.",
                "There are 5 children.",
                "Equal sharing uses 20 divided by 5.",
                "20 divided by 5 equals 4.",
                "Each child gets 4 cookies.",
            ],
            "clean",
            None,
        ),
        (
            "h0",
            "Lina has 3 red marbles and 2 blue marbles. How many marbles?",
            [
                "Lina has 3 red marbles.",
                "She has 2 blue marbles.",
                "The total is 3 + 2.",
                "3 + 2 = 6.",
                "Therefore the answer is 6.",
            ],
            "hallucinated",
            3,
        ),
        (
            "h1",
            "A box has 12 pencils and gives away 4. How many remain?",
            [
                "The box starts with 12 pencils.",
                "It gives away 7 pencils.",
                "The remaining count is 12 - 7.",
                "12 - 7 = 5.",
                "Therefore 5 pencils remain.",
            ],
            "hallucinated",
            1,
        ),
        (
            "h2",
            "Four bags hold 6 apples each. How many apples?",
            [
                "There are 4 bags.",
                "Each bag holds 6 apples.",
                "The total is 4 + 6.",
                "4 + 6 equals 10.",
                "Therefore there are 10 apples.",
            ],
            "hallucinated",
            2,
        ),
        (
            "h3",
            "Twenty cookies are shared equally by 5 children. How many each?",
            [
                "There are 20 cookies.",
                "There are 5 children.",
                "Equal sharing uses 20 divided by 5.",
                "20 divided by 5 equals 4.",
                "Each child gets 5 cookies.",
            ],
            "hallucinated",
            4,
        ),
    ]
    output: list[dict[str, Any]] = []
    for control_id, question, texts, status, onset in specs:
        output.append(
            {
                "control_id": control_id,
                "problem": question,
                "trajectory": "\n".join(texts),
                "units": [
                    {"unit_index": index, "kind": "material_claim", "text": text}
                    for index, text in enumerate(texts)
                ],
                "expected_status": status,
                "expected_first_bad_unit_index": onset,
            }
        )
    return output


def _public_content(row: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "problem": str(row.get("question", row.get("problem"))),
        "trajectory": str(row.get("response", row.get("trajectory"))),
        "units": [
            {
                "unit_index": int(unit["unit_index"]),
                "kind": str(unit["kind"]),
                "text": str(unit["text"]),
            }
            for unit in row["units"]
        ],
    }


def build_h_annotation_packages(
    proposals: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    repeat_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    if stage not in {"smoke", "reserve"}:
        raise ValueError("H package stage must be smoke or reserve")
    if not 0.0 <= repeat_fraction <= 1.0:
        raise ValueError("repeat fraction must be in [0,1]")
    natural = [dict(row) for row in proposals]
    repeat_count = math.ceil(len(natural) * repeat_fraction)
    repeat_ids = {
        str(row["proposal_id"])
        for row in sorted(
            natural,
            key=lambda row: stable_priority(
                "clir-H-v7-self-repeat-selection", stage, str(row["proposal_id"])
            ),
        )[:repeat_count]
    }
    controls = _control_items()
    public: dict[str, list[dict[str, Any]]] = {}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        decorated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in natural:
            proposal_id = str(row["proposal_id"])
            item_id = stable_priority(
                "clir-H-v7-public-item", annotator, stage, "natural", proposal_id
            )
            item = _public_content(row, item_id)
            decorated.append(
                (
                    item,
                    {
                        "item_id": item_id,
                        "annotator": annotator,
                        "stage": stage,
                        "role": "natural",
                        "canonical_item_id": proposal_id,
                        "public_content_sha256": canonical_sha256(item),
                    },
                )
            )
            if proposal_id in repeat_ids:
                repeat_id = stable_priority(
                    "clir-H-v7-public-item", annotator, stage, "repeat", proposal_id
                )
                repeated = _public_content(row, repeat_id)
                decorated.append(
                    (
                        repeated,
                        {
                            "item_id": repeat_id,
                            "annotator": annotator,
                            "stage": stage,
                            "role": "self_repeat",
                            "canonical_item_id": proposal_id,
                            "repeat_of_item_id": item_id,
                            "public_content_without_id_sha256": canonical_sha256(
                                {
                                    key: value
                                    for key, value in repeated.items()
                                    if key != "item_id"
                                }
                            ),
                            "public_content_sha256": canonical_sha256(repeated),
                        },
                    )
                )
        for control in controls:
            control_id = str(control["control_id"])
            item_id = stable_priority(
                "clir-H-v7-public-item", annotator, stage, "control", control_id
            )
            item = _public_content(control, item_id)
            decorated.append(
                (
                    item,
                    {
                        "item_id": item_id,
                        "annotator": annotator,
                        "stage": stage,
                        "role": "control",
                        "canonical_item_id": control_id,
                        "expected_status": control["expected_status"],
                        "expected_first_bad_unit_index": control[
                            "expected_first_bad_unit_index"
                        ],
                        "public_content_sha256": canonical_sha256(item),
                    },
                )
            )
        decorated.sort(
            key=lambda pair: stable_priority(
                "clir-H-v7-public-package-order", annotator, stage, pair[0]["item_id"]
            )
        )
        public[annotator] = [item for item, _ in decorated]
        private.extend(record for _, record in decorated)
    return (
        public,
        private,
        {
            "stage": stage,
            "natural_items_per_annotator": len(natural),
            "self_repeats_per_annotator": repeat_count,
            "controls_per_annotator": len(controls),
            "public_rows_per_annotator": len(natural) + repeat_count + len(controls),
            "annotator_a_ordered_rows_sha256": canonical_sha256(public["a"]),
            "annotator_b_ordered_rows_sha256": canonical_sha256(public["b"]),
            "private_rows_sha256": canonical_sha256(private),
        },
    )


def evaluate_h_package_labels(
    *,
    public_by_annotator: Mapping[str, Sequence[Mapping[str, Any]]],
    private_rows: Sequence[Mapping[str, Any]],
    labels_by_annotator: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    private = {
        (str(row["annotator"]), str(row["item_id"])): dict(row) for row in private_rows
    }
    canonical_primary: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    annotator_reports: dict[str, Any] = {}
    for annotator in ("a", "b"):
        items = {
            str(row["item_id"]): dict(row) for row in public_by_annotator[annotator]
        }
        labels = list(labels_by_annotator[annotator])
        if len(labels) != len(items):
            raise ValueError(f"annotator {annotator}: label count differs from package")
        if len({str(row.get("item_id")) for row in labels}) != len(labels):
            raise ValueError(f"annotator {annotator}: duplicate label item_id")
        if {str(row.get("item_id")) for row in labels} != set(items):
            raise ValueError(f"annotator {annotator}: label IDs differ from package")
        validated: dict[str, dict[str, Any]] = {}
        for raw in labels:
            item_id = str(raw["item_id"])
            validated[item_id] = validate_annotation(
                "hallucination", raw, items[item_id]
            )
        controls_correct = 0
        control_count = 0
        repeat_correct = 0
        repeat_count = 0
        for item_id, label in validated.items():
            record = private[(annotator, item_id)]
            role = str(record["role"])
            canonical_id = str(record["canonical_item_id"])
            if role == "natural":
                normalized = {**label, "item_id": canonical_id}
                canonical_primary[annotator][canonical_id] = normalized
            elif role == "control":
                control_count += 1
                controls_correct += (
                    label["status"] == record["expected_status"]
                    and label.get("first_bad_unit_index")
                    == record["expected_first_bad_unit_index"]
                )
            elif role == "self_repeat":
                repeat_count += 1
                primary = validated[str(record["repeat_of_item_id"])]
                repeat_correct += annotation_signature(
                    "hallucination", label
                ) == annotation_signature("hallucination", primary)
            else:
                raise ValueError(f"unsupported package role {role}")
        annotator_reports[annotator] = {
            "package_rows": len(items),
            "natural_rows": len(canonical_primary[annotator]),
            "controls_correct": controls_correct,
            "controls_total": control_count,
            "control_accuracy": controls_correct / control_count
            if control_count
            else None,
            "self_repeats_agree": repeat_correct,
            "self_repeats_total": repeat_count,
            "self_repeat_agreement": repeat_correct / repeat_count
            if repeat_count
            else None,
        }
    if set(canonical_primary["a"]) != set(canonical_primary["b"]):
        raise ValueError("A/B natural proposal populations differ")
    ids = sorted(canonical_primary["a"])
    labels_a = [canonical_primary["a"][item_id] for item_id in ids]
    labels_b = [canonical_primary["b"][item_id] for item_id in ids]
    agreement = agreement_report("hallucination", labels_a, labels_b)
    accepted = [
        item_id
        for item_id in ids
        if annotation_signature("hallucination", canonical_primary["a"][item_id])
        == annotation_signature("hallucination", canonical_primary["b"][item_id])
        and canonical_primary["a"][item_id]["status"] in FINAL_H_VALUES
        and canonical_primary["a"][item_id]["confidence"] != "low"
        and canonical_primary["b"][item_id]["confidence"] != "low"
    ]
    accepted_statuses = Counter(
        canonical_primary["a"][item_id]["status"] for item_id in accepted
    )
    report = {
        "annotators": annotator_reports,
        "agreement": agreement,
        "common_exact_non_low_accepted": len(accepted),
        "common_exact_non_low_by_status": dict(sorted(accepted_statuses.items())),
        "common_exact_non_low_ids_sha256": canonical_sha256(accepted),
    }
    return canonical_primary, report


def smoke_gate(
    report: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    annotation = protocol["h_acquisition"]["annotation"]
    smoke = protocol["h_acquisition"]["smoke"]
    agreement = report["agreement"]
    checks = {
        "raw_path_agreement": {
            "value": agreement["raw_path_agreement"],
            "threshold": float(annotation["raw_path_agreement_min"]),
            "pass": agreement["raw_path_agreement"]
            >= float(annotation["raw_path_agreement_min"]),
        },
        "common_positive_support": {
            "value": agreement["common_positive"],
            "threshold": int(annotation["common_positive_minimum_support"]),
            "pass": agreement["common_positive"]
            >= int(annotation["common_positive_minimum_support"]),
        },
        "exact_onset_agreement": {
            "value": agreement["exact_onset_agreement"],
            "threshold": float(annotation["common_positive_exact_onset_agreement_min"]),
            "pass": agreement["exact_onset_agreement"] is not None
            and agreement["exact_onset_agreement"]
            >= float(annotation["common_positive_exact_onset_agreement_min"]),
        },
        "accepted_positive": {
            "value": report["common_exact_non_low_by_status"].get("hallucinated", 0),
            "threshold": int(smoke["minimum_final_positive"]),
            "pass": report["common_exact_non_low_by_status"].get("hallucinated", 0)
            >= int(smoke["minimum_final_positive"]),
        },
        "accepted_clean": {
            "value": report["common_exact_non_low_by_status"].get("clean", 0),
            "threshold": int(smoke["minimum_final_clean"]),
            "pass": report["common_exact_non_low_by_status"].get("clean", 0)
            >= int(smoke["minimum_final_clean"]),
        },
    }
    for annotator in ("a", "b"):
        stats = report["annotators"][annotator]
        checks[f"{annotator}_controls"] = {
            "value": stats["controls_correct"],
            "threshold": int(annotation["controls_required_correct"]),
            "pass": stats["controls_correct"]
            == int(annotation["controls_required_correct"])
            and stats["controls_total"] == int(annotation["controls_per_annotator"]),
        }
        checks[f"{annotator}_self_repeat"] = {
            "value": stats["self_repeat_agreement"],
            "threshold": float(annotation["self_repeat_agreement_min"]),
            "pass": stats["self_repeat_agreement"] is not None
            and stats["self_repeat_agreement"]
            >= float(annotation["self_repeat_agreement_min"]),
        }
    passed = all(value["pass"] for value in checks.values())
    return {
        "status": "PASS_H0_V7_SMOKE" if passed else "FAIL_H0_V7_SMOKE",
        "pass": passed,
        "checks": checks,
        "reserve_annotation_allowed": passed,
    }


__all__ = [
    "H_LABEL_SCHEMA",
    "H_PACKAGE_SCHEMA",
    "H_PROPOSAL_SCHEMA",
    "build_h_annotation_packages",
    "build_h_proposals",
    "evaluate_h_package_labels",
    "smoke_gate",
    "split_smoke_and_reserve",
]
