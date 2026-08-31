from __future__ import annotations

import pytest

from src.clir_prior_scale import (
    build_blind_packages,
    derive_prior_targets,
    evaluate_prior_labels,
    select_prior_smoke_rows,
    validate_prior_annotation,
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


def test_dependency_closure_and_flaw_key() -> None:
    key, complete = derive_prior_targets(
        material_unit_indices=[1, 3, 5, 7],
        conclusion_unit_indices=[7],
        dependency_edges=[[1, 5], [3, 5], [5, 7]],
        path_status="supported",
        first_flaw_unit_index=None,
    )
    assert key == [7]
    assert complete == [1, 3, 5, 7]

    key, complete = derive_prior_targets(
        material_unit_indices=[1, 3, 5],
        conclusion_unit_indices=[5],
        dependency_edges=[[1, 3], [3, 5]],
        path_status="flawed",
        first_flaw_unit_index=1,
    )
    assert key == [1]
    assert complete == [1, 3, 5]

    with pytest.raises(ValueError, match="point forward"):
        derive_prior_targets(
            material_unit_indices=[1, 3],
            conclusion_unit_indices=[3],
            dependency_edges=[[3, 1]],
            path_status="supported",
            first_flaw_unit_index=None,
        )


def test_validate_annotation_derives_targets() -> None:
    item = {"item_id": "x", "units": _units(3)}
    normalized = validate_prior_annotation(
        {
            "item_id": "x",
            "eligibility": "usable",
            "path_status": "supported",
            "conclusion_unit_indices": [5],
            "dependency_edges": [[1, 3], [3, 5]],
            "first_flaw_unit_index": None,
            "confidence": "high",
            "rationale": "linear chain",
        },
        item,
    )
    assert normalized["key_unit_indices"] == [5]
    assert normalized["complete_unit_indices"] == [1, 3, 5]

    with pytest.raises(ValueError, match="empty graph"):
        validate_prior_annotation(
            {
                "item_id": "x",
                "eligibility": "no_auditable_reasoning",
                "path_status": None,
                "conclusion_unit_indices": [1],
                "dependency_edges": [],
                "first_flaw_unit_index": None,
                "confidence": "high",
                "rationale": "bad schema",
            },
            item,
        )


def test_selection_is_balanced_query_and_cluster_distinct() -> None:
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
            {"source": "gsm8k", "checker_status": "numeric_mismatch", "count": 1},
            {"source": "math", "checker_status": "numeric_match", "count": 1},
            {"source": "math", "checker_status": "numeric_mismatch", "count": 1},
        ],
    }
    selected, report = select_prior_smoke_rows(
        materialized_rows=rows,
        excluded_query_ids={"query-99"},
        excluded_cluster_ids={"excluded-cluster"},
        selection=selection,
    )
    assert len(selected) == 4
    assert report["unique_queries"] == 4
    assert report["unique_clusters"] == 4
    assert set(report["selected_by_stratum"].values()) == {1}


def _annotation_for(
    item: dict,
    *,
    status: str,
    key: list[int],
    complete: list[int],
) -> dict:
    if status == "supported":
        conclusions = list(key)
        target = conclusions[0]
        edges = [[index, target] for index in complete if index != target]
        flaw = None
    else:
        conclusions = [max(complete)]
        edges = [list(pair) for pair in zip(complete, complete[1:])]
        flaw = key[0]
    return {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": status,
        "conclusion_unit_indices": conclusions,
        "dependency_edges": edges,
        "first_flaw_unit_index": flaw,
        "confidence": "high",
        "rationale": "test annotation",
    }


def test_packages_and_perfect_gate() -> None:
    proposals = []
    for index in range(12):
        row = _row(index, "math", "numeric_match")
        row.update(
            {
                "proposal_id": f"natural-{index:02d}",
                "trajectory_id": row["id"],
            }
        )
        proposals.append(row)
    package_a, package_b, private, package_report = build_blind_packages(
        proposals, repeat_count_a=3
    )
    assert package_report["package_a_rows"] == 21
    assert package_report["package_b_rows"] == 18

    private_a = {row["item_id"]: row for row in private if row["annotator"] == "a"}
    private_b = {row["item_id"]: row for row in private if row["annotator"] == "b"}
    natural_labels: dict[str, dict] = {}
    for index, proposal in enumerate(proposals):
        item = next(row for row in package_a if row["item_id"] == proposal["proposal_id"])
        indices = [unit["unit_index"] for unit in item["units"]]
        if index < 6:
            natural_labels[item["item_id"]] = _annotation_for(
                item,
                status="flawed",
                key=[indices[0]],
                complete=[indices[0], indices[-1]],
            )
        else:
            natural_labels[item["item_id"]] = _annotation_for(
                item,
                status="supported",
                key=[indices[-1]],
                complete=[indices[-1]],
            )

    def build_labels(package: list[dict], private_index: dict[str, dict]) -> list[dict]:
        output = []
        for item in package:
            record = private_index[item["item_id"]]
            if record["kind"] == "natural":
                label = dict(natural_labels[item["item_id"]])
            elif record["kind"] == "repeat":
                label = dict(natural_labels[record["natural_item_id"]])
                label["item_id"] = item["item_id"]
            else:
                expected = record["expected_signature"]
                label = _annotation_for(
                    item,
                    status=expected[1],
                    key=list(expected[2]),
                    complete=list(expected[3]),
                )
            output.append(label)
        return output

    report = evaluate_prior_labels(
        package_a=package_a,
        package_b=package_b,
        private_index=private,
        labels_a=build_labels(package_a, private_a),
        labels_b=build_labels(package_b, private_b),
        gates={
            "eligibility_agreement_min": 0.95,
            "common_usable_min": 12,
            "path_agreement_min": 0.90,
            "key_macro_f1_min": 0.90,
            "complete_macro_f1_min": 0.90,
            "exact_training_consensus_min": 12,
            "exact_training_consensus_rate_min": 1.0,
            "minimum_adjudication_fraction_max": 0.0,
            "common_flawed_min": 6,
            "first_flaw_exact_rate_min": 1.0,
            "self_repeat_min": 0.95,
            "complete_all_material_rate_max": 0.25,
        },
    )
    assert report["status"] == "PASS_PRIOR_DEPENDENCY_SMOKE_V8"
    assert report["metrics"]["exact_training_consensus"] == 12
