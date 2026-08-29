from __future__ import annotations

from src.clir_h_expansion import (
    build_h_annotation_packages,
    build_h_proposals,
    evaluate_h_package_labels,
    smoke_gate,
    split_smoke_and_reserve,
)


def _protocol() -> dict:
    return {
        "h_acquisition": {
            "minimum_material_units": 5,
            "proposal_target": {
                checker_split: {"math": 1, "gsm8k": 1}
                for checker_split in (
                    "numeric_match|train",
                    "numeric_mismatch|train",
                    "numeric_match|dev",
                    "numeric_mismatch|dev",
                )
            },
            "proposal_target_total": 8,
            "smoke": {
                "train_proposals": 4,
                "numeric_mismatch": {"math": 1, "gsm8k": 1},
                "numeric_match": {"math": 1, "gsm8k": 1},
                "minimum_final_positive": 2,
                "minimum_final_clean": 2,
            },
            "annotation": {
                "raw_path_agreement_min": 0.9,
                "common_positive_minimum_support": 2,
                "common_positive_exact_onset_agreement_min": 0.75,
                "controls_per_annotator": 8,
                "controls_required_correct": 8,
                "self_repeat_agreement_min": 0.95,
            },
        }
    }


def _rows() -> list[dict]:
    output: list[dict] = []
    query_index = 0
    for source in ("math", "gsm8k"):
        for label_split in ("train", "dev"):
            for target in ("numeric_match", "numeric_mismatch"):
                query_id = f"{source}:train:{query_index:05d}"
                for candidate_index in range(8):
                    status = (
                        target
                        if candidate_index == 0
                        else (
                            "numeric_mismatch"
                            if target == "numeric_match"
                            else "numeric_match"
                        )
                    )
                    output.append(
                        {
                            "id": f"{query_id}:cand:{candidate_index:03d}",
                            "query_id": query_id,
                            "candidate_index": candidate_index,
                            "source": source,
                            "question": "What is 2 + 3?",
                            "response": "2 plus 3 equals 5.",
                            "h_target_checker_status": target,
                            "h_label_split": label_split,
                            "checker_status": status,
                            "eligible_for_supervision": True,
                            "unitization_status": "ok",
                            "finish_reason": "stop",
                            "material_claim_count": 5,
                            "units": [
                                {
                                    "unit_index": index,
                                    "kind": "material_claim",
                                    "text": f"claim {index}",
                                    "token_start": index,
                                    "token_end": index + 1,
                                }
                                for index in range(5)
                            ],
                        }
                    )
                query_index += 1
    return output


def test_proposals_are_one_per_query_and_smoke_is_frozen_partition() -> None:
    protocol = _protocol()
    proposals, report = build_h_proposals(_rows(), protocol)
    smoke, reserve, split = split_smoke_and_reserve(proposals, protocol)

    assert len(proposals) == 8
    assert len({row["query_id"] for row in proposals}) == 8
    assert all(row["candidate_index"] == 0 for row in proposals)
    assert report["selected_rows"] == 8
    assert len(smoke) == len(reserve) == 4
    assert not (
        {row["proposal_id"] for row in smoke} & {row["proposal_id"] for row in reserve}
    )
    assert split["smoke_rows"] == split["reserve_rows"] == 4


def test_proposals_accept_one_frozen_rescue_round() -> None:
    protocol = _protocol()
    rows = _rows()
    query_id = rows[0]["query_id"]
    target = rows[0]["h_target_checker_status"]
    for row in rows:
        if row["query_id"] == query_id:
            row["checker_status"] = "numeric_mismatch"
    parent = next(row for row in rows if row["query_id"] == query_id)
    for candidate_index in range(8, 32):
        rescued = dict(parent)
        rescued["id"] = f"{query_id}:cand:{candidate_index:03d}"
        rescued["candidate_index"] = candidate_index
        rescued["checker_status"] = (
            target if candidate_index == 8 else "numeric_mismatch"
        )
        rows.append(rescued)

    proposals, report = build_h_proposals(rows, protocol)
    chosen = next(row for row in proposals if row["query_id"] == query_id)

    assert chosen["candidate_index"] == 8
    assert report["input_rows"] == 8 * 8 + 24


def test_proposals_accept_fresh_sixteen_candidate_queries() -> None:
    protocol = _protocol()
    rows = _rows()
    query_id = rows[0]["query_id"]
    target = rows[0]["h_target_checker_status"]
    fresh = [row for row in rows if row["query_id"] != query_id]
    parent = next(row for row in rows if row["query_id"] == query_id)
    for candidate_index in range(16):
        candidate = dict(parent)
        candidate["id"] = f"{query_id}:cand:{candidate_index:03d}"
        candidate["candidate_index"] = candidate_index
        candidate["checker_status"] = (
            target if candidate_index == 12 else "numeric_mismatch"
        )
        fresh.append(candidate)

    proposals, _ = build_h_proposals(fresh, protocol)
    chosen = next(row for row in proposals if row["query_id"] == query_id)

    assert chosen["candidate_index"] == 12


def test_blind_package_controls_repeats_and_smoke_gate() -> None:
    protocol = _protocol()
    proposals, _ = build_h_proposals(_rows(), protocol)
    smoke, _, _ = split_smoke_and_reserve(proposals, protocol)
    public, private, package_report = build_h_annotation_packages(
        smoke, stage="smoke", repeat_fraction=0.1
    )

    assert package_report["natural_items_per_annotator"] == 4
    assert package_report["self_repeats_per_annotator"] == 1
    assert package_report["controls_per_annotator"] == 8
    assert len(public["a"]) == len(public["b"]) == 13

    private_by_key = {(row["annotator"], row["item_id"]): row for row in private}
    natural_order = sorted(row["proposal_id"] for row in smoke)
    expected_natural = {
        proposal_id: (("hallucinated", 2) if index < 2 else ("clean", None))
        for index, proposal_id in enumerate(natural_order)
    }
    labels: dict[str, list[dict]] = {"a": [], "b": []}
    for annotator in ("a", "b"):
        for item in public[annotator]:
            record = private_by_key[(annotator, item["item_id"])]
            if record["role"] == "natural":
                signature = expected_natural[record["canonical_item_id"]]
            elif record["role"] == "control":
                signature = (
                    record["expected_status"],
                    record["expected_first_bad_unit_index"],
                )
            else:
                signature = expected_natural[record["canonical_item_id"]]
            status, onset = signature
            labels[annotator].append(
                {
                    "item_id": item["item_id"],
                    "status": status,
                    "first_bad_unit_index": onset,
                    "confidence": "high",
                    "rationale": "deterministic test label",
                }
            )

    _, evaluation = evaluate_h_package_labels(
        public_by_annotator=public,
        private_rows=private,
        labels_by_annotator=labels,
    )
    gate = smoke_gate(evaluation, protocol)

    assert evaluation["agreement"]["raw_path_agreement"] == 1.0
    assert evaluation["agreement"]["exact_onset_agreement"] == 1.0
    assert evaluation["common_exact_non_low_by_status"] == {
        "clean": 2,
        "hallucinated": 2,
    }
    assert gate["pass"] is True
