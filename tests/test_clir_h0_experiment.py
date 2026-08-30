from __future__ import annotations

from copy import deepcopy

import pytest

from src.clir_h0_experiment import (
    H_LABEL_NAME,
    assign_feature_workers,
    build_feature_inventory,
    inventory_statistics,
    select_fully_labeled_ranking,
    validate_h_partition,
)


def _ranking_row(query: str, candidate: int, *, valid: bool = True) -> dict:
    return {
        "id": f"{query}:cand:{candidate:03d}",
        "query_id": query,
        "candidate_index": candidate,
        "source": "gsm8k",
        "source_record_id": query,
        "cluster_id": f"cluster:{query}",
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4, candidate + 5],
        "checker_status": "numeric_match" if valid else "truncated",
        "correctness": candidate % 2 if valid else None,
        "eligible_for_supervision": valid,
        "evaluation_only": True,
    }


def _h_row(row_id: str, split: str, status: str) -> dict:
    positive = status == "hallucinated"
    return {
        "id": row_id,
        "query_id": f"q:{row_id}",
        "candidate_index": 0,
        "source": "math",
        "source_record_id": row_id,
        "cluster_id": f"cluster:{row_id}",
        "prompt_token_ids": [1, 2],
        "output_token_ids": [3, 4, 5],
        "correctness": 0 if positive else 1,
        "h_label_name": H_LABEL_NAME,
        "h_label_split": split,
        "h_posthoc_exploratory": True,
        "h_original_v7_status": "FAIL_H0_V7_RESERVE",
        "h_status": status,
        "hallucination_onset": 1 if positive else -1,
        "path_hallucinated": int(positive),
        "hallucination_label_tier": "silver_posthoc",
    }


def test_ranking_selection_is_whole_query_and_binary_only() -> None:
    rows = [_ranking_row("q0", index) for index in range(4)]
    rows += [_ranking_row("q1", index, valid=index != 2) for index in range(4)]
    selected, report = select_fully_labeled_ranking(rows, candidate_count=4)
    assert [row["id"] for row in selected] == [
        f"q0:cand:{index:03d}" for index in range(4)
    ]
    assert report["selected_queries"] == 1
    assert report["informative_queries_with_both_labels"] == 1
    assert report["rejected_query_counts"] == {"non_binary_checker_label": 1}
    assert report["selection_uses_clir_scores"] is False


def test_h_partition_fails_closed_on_onset_status_mismatch() -> None:
    rows = [_h_row("clean", "train", "clean"), _h_row("positive", "train", "hallucinated")]
    report = validate_h_partition(
        rows, split="train", expected_clean=1, expected_positive=1
    )
    assert report["status_counts"] == {"clean": 1, "hallucinated": 1}
    broken = deepcopy(rows)
    broken[0]["hallucination_onset"] = 0
    with pytest.raises(ValueError, match="clean H row"):
        validate_h_partition(
            broken, split="train", expected_clean=1, expected_positive=1
        )


def test_inventory_worker_assignment_keeps_queries_atomic() -> None:
    h_train = [_h_row("htrain", "train", "clean")]
    h_dev = [_h_row("hdev", "dev", "hallucinated")]
    ranking = [_ranking_row("ranking", index) for index in range(4)]
    inventory = build_feature_inventory(h_train, h_dev, ranking)
    assigned, workers = assign_feature_workers(inventory, 2)
    query_workers: dict[str, set[int]] = {}
    for row in assigned:
        query_workers.setdefault(row["query_id"], set()).add(
            row["feature_worker_index"]
        )
    assert all(len(values) == 1 for values in query_workers.values())
    assert sum(worker["trajectory_count"] for worker in workers) == 6
    stats = inventory_statistics(assigned)
    assert stats["trajectory_count"] == 6
    assert stats["query_count"] == 3
    assert stats["output_token_count"] == 18
    assert stats["prompt_token_count"] == 6


def test_feature_inventory_rejects_cross_role_query_overlap() -> None:
    train = _h_row("train", "train", "clean")
    dev = _h_row("dev", "dev", "clean")
    dev["query_id"] = train["query_id"]
    with pytest.raises(ValueError, match="feature-role query overlap"):
        build_feature_inventory([train], [dev], [])
