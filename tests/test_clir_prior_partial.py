from __future__ import annotations

import pytest

from src.clir_prior_partial import (
    build_blind_packages,
    derive_partial_consensus_unit_targets,
    evaluate_partial_prior_labels,
    select_partial_prior_smoke_rows,
    validate_partial_prior_annotation,
)


def _units(count: int = 6) -> list[dict]:
    return [
        {
            "unit_index": index * 2 + 1,
            "kind": "material_claim",
            "text": f"claim {index}",
        }
        for index in range(count)
    ]


def _row(index: int, source: str, status: str, *, cluster: str | None = None) -> dict:
    return {
        "id": f"trajectory-{index}",
        "query_id": f"query-{index}",
        "cluster_id": cluster or f"cluster-{index}",
        "source": source,
        "source_record_id": index,
        "checker_status": status,
        "acquisition_split": "train_acquisition",
        "eligible_for_supervision": True,
        "unitization_status": "ok",
        "finish_reason": "stop",
        "material_claim_count": 6,
        "candidate_index": 0,
        "question": f"question {index}",
        "response": f"response {index}",
        "output_token_count": 20,
        "units": _units(),
    }


def _label(
    item: dict,
    *,
    eligibility: str = "usable",
    key: list[int] | None = None,
    complete: list[int] | None = None,
    confidence: str = "high",
) -> dict:
    return {
        "item_id": item["item_id"],
        "eligibility": eligibility,
        "key_unit_indices": key if key is not None else [],
        "complete_unit_indices": complete if complete is not None else [],
        "confidence": confidence,
        "rationale": "test annotation",
    }


def test_validate_direct_sets_and_subset_contract() -> None:
    item = {"item_id": "x", "units": _units(3)}
    normalized = validate_partial_prior_annotation(
        _label(item, key=[5], complete=[1, 5]), item
    )
    assert normalized["key_unit_indices"] == [5]
    assert normalized["complete_unit_indices"] == [1, 5]

    with pytest.raises(ValueError, match="subset"):
        validate_partial_prior_annotation(_label(item, key=[3], complete=[1, 5]), item)
    with pytest.raises(ValueError, match="empty Key and Complete"):
        validate_partial_prior_annotation(
            _label(
                item,
                eligibility="no_auditable_reasoning",
                key=[1],
                complete=[1],
            ),
            item,
        )


def test_partial_consensus_masks_only_complete_disagreements() -> None:
    item = {"item_id": "x", "units": _units(4)}
    left = _label(item, key=[7], complete=[1, 3, 7])
    right = _label(item, key=[7], complete=[1, 5, 7])
    targets = derive_partial_consensus_unit_targets(left, right, item)

    assert targets["key_trainable"] is True
    assert targets["key_positive_units"] == [7]
    assert targets["key_covered_units"] == [1, 3, 5, 7]
    assert targets["complete_trainable"] is True
    assert targets["complete_positive_units"] == [1, 7]
    assert targets["complete_negative_units"] == []
    assert targets["complete_ambiguous_units"] == [3, 5]
    assert targets["complete_covered_units"] == [1, 7]

    right["key_unit_indices"] = [5]
    targets = derive_partial_consensus_unit_targets(left, right, item)
    assert targets["key_trainable"] is False
    assert targets["complete_trainable"] is True


def test_low_confidence_disables_all_partial_targets() -> None:
    item = {"item_id": "x", "units": _units(3)}
    left = _label(item, key=[5], complete=[1, 5], confidence="low")
    right = _label(item, key=[5], complete=[1, 5])
    targets = derive_partial_consensus_unit_targets(left, right, item)
    assert targets["common_nonlow_usable"] is False
    assert targets["key_trainable"] is False
    assert targets["complete_trainable"] is False


def test_selection_is_balanced_and_respects_all_exclusions() -> None:
    rows = [
        _row(0, "gsm8k", "numeric_match"),
        _row(1, "gsm8k", "numeric_mismatch"),
        _row(2, "math", "numeric_match"),
        _row(3, "math", "numeric_mismatch"),
        _row(4, "math", "numeric_match", cluster="excluded-cluster"),
    ]
    selection = {
        "natural_count": 4,
        "minimum_material_claims": 6,
        "maximum_material_claims": 40,
        "strata": [
            {"source": "gsm8k", "checker_status": "numeric_match", "count": 1},
            {
                "source": "gsm8k",
                "checker_status": "numeric_mismatch",
                "count": 1,
            },
            {"source": "math", "checker_status": "numeric_match", "count": 1},
            {"source": "math", "checker_status": "numeric_mismatch", "count": 1},
        ],
    }
    selected, report = select_partial_prior_smoke_rows(
        materialized_rows=rows,
        excluded_query_ids={"query-99"},
        excluded_cluster_ids={"excluded-cluster"},
        selection=selection,
    )
    assert len(selected) == 4
    assert report["unique_queries"] == 4
    assert report["unique_clusters"] == 4
    assert set(report["selected_by_stratum"].values()) == {1}


def _package_labels(
    package: list[dict],
    private_by_id: dict[str, dict],
    natural: dict[str, dict],
) -> list[dict]:
    output = []
    for item in package:
        record = private_by_id[item["item_id"]]
        if record["kind"] == "natural":
            label = dict(natural[item["item_id"]])
        elif record["kind"] == "repeat":
            label = dict(natural[record["natural_item_id"]])
            label["item_id"] = item["item_id"]
        else:
            expected = record["expected_signature"]
            label = _label(
                item,
                eligibility=expected[0],
                key=list(expected[1]),
                complete=list(expected[2]),
            )
        output.append(label)
    return output


def _gates(count: int) -> dict:
    return {
        "eligibility_agreement_min": 0.95,
        "common_usable_min": count,
        "common_nonlow_usable_min": count,
        "key_macro_f1_min": 0.90,
        "complete_macro_f1_min": 0.90,
        "key_exact_nonlow_rows_min": count,
        "complete_nonempty_consensus_rows_min": count,
        "partial_paired_trainable_rows_min": count,
        "complete_unit_decision_agreement_min": 0.90,
        "complete_ambiguous_unit_fraction_max": 0.10,
        "complete_positive_intersection_over_union_min": 0.80,
        "complete_row_mask_coverage_mean_min": 0.90,
        "self_repeat_min": 0.95,
        "complete_all_material_rate_max": 0.25,
    }


def test_packages_and_perfect_partial_gate() -> None:
    proposals = []
    for index in range(12):
        row = _row(index, "math", "numeric_match")
        row.update({"proposal_id": f"natural-{index:02d}", "trajectory_id": row["id"]})
        proposals.append(row)
    package_a, package_b, private, package_report = build_blind_packages(
        proposals, repeat_count_a=3
    )
    assert package_report["package_a_rows"] == 21
    assert package_report["package_b_rows"] == 18

    natural = {}
    for proposal in proposals:
        item = next(
            row for row in package_a if row["item_id"] == proposal["proposal_id"]
        )
        indices = [unit["unit_index"] for unit in item["units"]]
        natural[item["item_id"]] = _label(
            item, key=[indices[-1]], complete=[indices[0], indices[-1]]
        )
    private_a = {row["item_id"]: row for row in private if row["annotator"] == "a"}
    private_b = {row["item_id"]: row for row in private if row["annotator"] == "b"}
    report = evaluate_partial_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=_package_labels(package_a, private_a, natural),
        labels_b=_package_labels(package_b, private_b, natural),
        gates=_gates(12),
    )
    assert report["status"] == "PASS_PRIOR_PARTIAL_SMOKE_V9"
    assert report["metrics"]["partial_paired_trainable_rows"] == 12
    assert report["metrics"]["complete_ambiguous_unit_fraction"] == 0.0


def test_complete_disagreement_can_fail_frozen_partial_gate() -> None:
    proposals = []
    for index in range(2):
        row = _row(index, "math", "numeric_match")
        row.update({"proposal_id": f"natural-{index}", "trajectory_id": row["id"]})
        proposals.append(row)
    package_a, package_b, private, _ = build_blind_packages(proposals, repeat_count_a=1)
    natural_a = {}
    natural_b = {}
    for proposal in proposals:
        item = next(
            row for row in package_a if row["item_id"] == proposal["proposal_id"]
        )
        indices = [unit["unit_index"] for unit in item["units"]]
        natural_a[item["item_id"]] = _label(
            item, key=[indices[-1]], complete=[indices[0], indices[-1]]
        )
        natural_b[item["item_id"]] = _label(
            item, key=[indices[-1]], complete=indices[1:]
        )
    private_a = {row["item_id"]: row for row in private if row["annotator"] == "a"}
    private_b = {row["item_id"]: row for row in private if row["annotator"] == "b"}
    report = evaluate_partial_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=_package_labels(package_a, private_a, natural_a),
        labels_b=_package_labels(package_b, private_b, natural_b),
        gates=_gates(2),
    )
    assert report["status"] == "STOP_PRIOR_PARTIAL_SMOKE_V9_RAW_GATE_FAILURE"
    assert report["gates"]["complete_ambiguity"]["pass"] is False
