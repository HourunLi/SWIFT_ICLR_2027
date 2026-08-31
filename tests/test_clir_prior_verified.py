from __future__ import annotations

import pytest

from src.clir_prior_canonical import canonical_control_items
from src.clir_prior_verified import (
    build_verified_blind_packages,
    evaluate_verified_prior_labels,
    validate_verified_prior_annotation,
    verification_control_items,
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


def _proposal(index: int) -> dict:
    return {
        "proposal_id": f"v11-natural-{index:02d}",
        "trajectory_id": f"v11-trajectory-{index:02d}",
        "question": f"question {index}",
        "response": f"response {index}",
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
        "rationale": "verified test annotation",
    }


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


def _labels_for_package(
    package: list[dict], private: list[dict], natural: dict[str, dict], annotator: str
) -> list[dict]:
    private_by_id = {
        row["item_id"]: row for row in private if row["annotator"] == annotator
    }
    output = []
    for item in package:
        record = private_by_id[item["item_id"]]
        if record["kind"] == "natural":
            label = dict(natural[item["item_id"]])
        elif record["kind"] == "repeat":
            label = dict(natural[record["natural_item_id"]])
            label["item_id"] = item["item_id"]
        else:
            eligibility, key, complete = record["expected_signature"]
            label = _label(
                item,
                eligibility=eligibility,
                key=list(key),
                complete=list(complete),
            )
        output.append(label)
    return output


def _natural_labels(proposals: list[dict], package: list[dict]) -> dict[str, dict]:
    package_by_id = {row["item_id"]: row for row in package}
    output = {}
    for proposal in proposals:
        item = package_by_id[proposal["proposal_id"]]
        indices = [unit["unit_index"] for unit in item["units"]]
        output[item["item_id"]] = _label(
            item, key=[indices[-1]], complete=[indices[0], indices[-1]]
        )
    return output


def test_v11_requires_single_key_and_uses_fresh_controls() -> None:
    item = {"item_id": "x", "units": _units(4)}
    normalized = validate_verified_prior_annotation(
        _label(item, key=[7], complete=[1, 3, 7]), item
    )
    assert normalized["key_unit_indices"] == [7]

    with pytest.raises(ValueError, match="exactly one Key"):
        validate_verified_prior_annotation(
            _label(item, key=[5, 7], complete=[1, 5, 7]), item
        )

    controls = verification_control_items()
    assert len(controls) == 8
    assert {row["item_id"] for row in controls}.isdisjoint(
        {row["item_id"] for row in canonical_control_items()}
    )
    assert controls[0]["expected_signature"] == ("usable", (0,), (0, 1))
    assert controls[1]["expected_signature"] == ("usable", (1,), (0, 1))


def test_v11_packages_and_pass_status() -> None:
    proposals = [_proposal(index) for index in range(12)]
    package_a, package_b, private, package_report = build_verified_blind_packages(
        proposals, repeat_count_a=3, repeat_count_b=3
    )
    assert package_report["controls_per_annotator"] == 8
    assert package_report["a_repeats"] == 3
    assert package_report["b_repeats"] == 3
    assert package_report["package_a_rows"] == 23
    assert package_report["package_b_rows"] == 23

    natural = _natural_labels(proposals, package_a)
    report = evaluate_verified_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=_labels_for_package(package_a, private, natural, "a"),
        labels_b=_labels_for_package(package_b, private, natural, "b"),
        gates=_gates(12),
    )
    assert report["status"] == "PASS_PRIOR_VERIFIED_SMOKE_V11"
    assert report["metrics"]["self_repeat"]["a"]["rate"] == 1.0
    assert report["metrics"]["self_repeat"]["b"]["rate"] == 1.0
    assert report["scale_protocol_preparation_allowed"] is True
    assert report["training_allowed"] is False


def test_v11_control_failure_is_not_yield_only() -> None:
    proposals = [_proposal(index) for index in range(2)]
    package_a, package_b, private, _ = build_verified_blind_packages(
        proposals, repeat_count_a=1, repeat_count_b=1
    )
    natural = _natural_labels(proposals, package_a)
    labels_a = _labels_for_package(package_a, private, natural, "a")
    labels_b = _labels_for_package(package_b, private, natural, "b")

    control_id = next(
        row["item_id"]
        for row in private
        if row["annotator"] == "a" and row["kind"] == "control"
    )
    for label in labels_a:
        if label["item_id"] == control_id:
            label["key_unit_indices"] = [1]
            label["complete_unit_indices"] = [0, 1]
            break
    report = evaluate_verified_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=labels_a,
        labels_b=labels_b,
        gates=_gates(2),
    )
    assert report["status"] == "STOP_PRIOR_VERIFIED_SMOKE_V11_DEFINITION_FAILURE"
    assert report["oversampled_scale_protocol_preparation_allowed"] is False


def test_v11_yield_only_status_remains_separate() -> None:
    proposals = [_proposal(index) for index in range(2)]
    package_a, package_b, private, _ = build_verified_blind_packages(
        proposals, repeat_count_a=1, repeat_count_b=1
    )
    natural = _natural_labels(proposals, package_a)
    report = evaluate_verified_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=_labels_for_package(package_a, private, natural, "a"),
        labels_b=_labels_for_package(package_b, private, natural, "b"),
        gates=_gates(3),
    )
    assert report["status"] == "STOP_PRIOR_VERIFIED_SMOKE_V11_YIELD_ONLY"
    assert report["oversampled_scale_protocol_preparation_allowed"] is True
