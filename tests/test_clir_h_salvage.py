from __future__ import annotations

from src.clir_h_salvage import (
    build_h_salvage_rows,
    find_retry_self_repeat_failures,
)


def _label(item_id: str, status: str, onset: int | None) -> dict:
    return {
        "item_id": item_id,
        "status": status,
        "first_bad_unit_index": onset,
        "confidence": "high",
        "rationale": "test rationale",
    }


def _proposals() -> list[dict]:
    rows: list[dict] = []
    index = 0
    for stage in ("smoke", "reserve"):
        for split in ("train", "dev"):
            for status in ("clean", "hallucinated"):
                proposal_id = f"{stage}-{split}-{status}"
                rows.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_priority": f"{index:03d}",
                        "query_id": f"query-{index:03d}",
                        "source": "math" if index % 2 == 0 else "gsm8k",
                        "checker_status": "numeric_match",
                        "h_label_split": split,
                        "units": [
                            {
                                "unit_index": unit_index,
                                "kind": "material_claim",
                                "token_start": unit_index * 10,
                                "token_end": unit_index * 10 + 5,
                            }
                            for unit_index in range(3)
                        ],
                    }
                )
                index += 1
    return rows


def _canonical_labels(ids: list[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for item_id in ids:
        status = "hallucinated" if item_id.endswith("hallucinated") else "clean"
        output[item_id] = _label(item_id, status, 2 if status == "hallucinated" else None)
    return output


def test_retry_self_repeat_failures_are_union_by_natural_proposal() -> None:
    private = [
        {
            "annotator": "a",
            "role": "self_repeat",
            "item_id": "a-repeat",
            "repeat_of_item_id": "a-primary",
            "canonical_item_id": "proposal-1",
        },
        {
            "annotator": "b",
            "role": "self_repeat",
            "item_id": "b-repeat",
            "repeat_of_item_id": "b-primary",
            "canonical_item_id": "proposal-1",
        },
    ]
    labels = {
        "a": {
            "a-primary": _label("a-primary", "hallucinated", 1),
            "a-repeat": _label("a-repeat", "hallucinated", 2),
        },
        "b": {
            "b-primary": _label("b-primary", "clean", None),
            "b-repeat": _label("b-repeat", "clean", None),
        },
    }

    failed, report = find_retry_self_repeat_failures(
        private_rows=private,
        retry_labels_by_annotator=labels,
    )

    assert failed == {"proposal-1"}
    assert report["a"]["exact_agree"] == 0
    assert report["b"]["exact_agree"] == 1
    assert report["union"]["failed_natural_rows"] == 1


def test_posthoc_salvage_selects_original_balanced_cells() -> None:
    proposals = _proposals()
    smoke_ids = [row["proposal_id"] for row in proposals[:4]]
    reserve_ids = [row["proposal_id"] for row in proposals[4:]]
    smoke = _canonical_labels(smoke_ids)
    reserve = _canonical_labels(reserve_ids)

    eligible, selected, report = build_h_salvage_rows(
        proposals=proposals,
        smoke_labels_by_annotator={"a": smoke, "b": smoke},
        reserve_attempt_1_b=reserve,
        reserve_attempt_2_by_annotator={"a": reserve, "b": reserve},
        repeat_failed_proposal_ids=set(),
        targets={
            "train_hallucinated": 1,
            "train_clean": 1,
            "dev_hallucinated": 1,
            "dev_clean": 1,
        },
        label_name="silver_posthoc_test",
    )

    assert len(eligible) == 8
    assert len(selected) == 4
    assert report["status"] == "PASS_H0_V7_4_POSTHOC_SALVAGE_SELECTION"
    assert report["selected_by_cell"] == {
        "dev|clean": 1,
        "dev|hallucinated": 1,
        "train|clean": 1,
        "train|hallucinated": 1,
    }
    assert report["train_dev_query_overlap"] == 0
    for row in selected:
        assert row["h_posthoc_exploratory"] is True
        if row["h_status"] == "clean":
            assert row["hallucination_onset"] == -1
        else:
            assert row["hallucination_onset"] == 20


def test_posthoc_salvage_fails_closed_when_a_cell_is_removed() -> None:
    proposals = _proposals()
    smoke_ids = [row["proposal_id"] for row in proposals[:4]]
    reserve_ids = [row["proposal_id"] for row in proposals[4:]]
    smoke = _canonical_labels(smoke_ids)
    reserve = _canonical_labels(reserve_ids)
    removed_id = "reserve-dev-hallucinated"

    _, selected, report = build_h_salvage_rows(
        proposals=proposals,
        smoke_labels_by_annotator={"a": smoke, "b": smoke},
        reserve_attempt_1_b=reserve,
        reserve_attempt_2_by_annotator={"a": reserve, "b": reserve},
        repeat_failed_proposal_ids={removed_id},
        targets={
            "train_hallucinated": 1,
            "train_clean": 1,
            "dev_hallucinated": 2,
            "dev_clean": 1,
        },
        label_name="silver_posthoc_test",
    )

    assert selected == []
    assert report["status"] == "FAIL_H0_V7_4_POSTHOC_SALVAGE_YIELD"
    assert report["shortages"] == {
        "dev|hallucinated": {"target": 2, "available": 1}
    }
