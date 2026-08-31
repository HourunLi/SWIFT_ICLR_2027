from __future__ import annotations

from copy import deepcopy

from src.clir_prior_mechanical import validate_local_audit_annotation
from src.clir_prior_mechanical_smoke import (
    build_blind_shards,
    build_hidden_controls,
    evaluate_blind_labels,
    select_fresh_natural_rows,
)


def _rollout_row(
    index: int,
    *,
    source: str,
    checker_status: str,
    material_count: int,
) -> dict:
    return {
        "id": f"row-{index}",
        "query_id": f"query-{index}",
        "cluster_id": f"cluster-{index}",
        "source": source,
        "checker_status": checker_status,
        "acquisition_split": "train",
        "status": "ok",
        "unitization_status": "ok",
        "eligible_for_supervision": True,
        "finish_reason": "stop",
        "material_claim_count": material_count,
        "question": f"Compute the value for case {index}.",
        "response": "\n".join(
            f"v{unit} = {unit} + 1 = {unit + 1}" for unit in range(material_count)
        ),
        "units": [
            {
                "unit_index": 2 * unit,
                "kind": "material_claim",
                "text": f"v{unit} = {unit} + 1 = {unit + 1}",
            }
            for unit in range(material_count)
        ],
    }


def _protocol_strata() -> list[dict]:
    return [
        {
            "source": source,
            "checker_status": checker,
            "length_band": band,
            "count": 6,
        }
        for source in ("gsm8k", "math")
        for checker in ("numeric_match", "numeric_mismatch")
        for band in ("medium", "long")
    ]


def _mock_proposals() -> list[dict]:
    rows: list[dict] = []
    index = 0
    for source in ("gsm8k", "math"):
        for checker in ("numeric_match", "numeric_mismatch"):
            for count in (8, 19):
                for _ in range(6):
                    rows.append(
                        _rollout_row(
                            index,
                            source=source,
                            checker_status=checker,
                            material_count=count,
                        )
                    )
                    index += 1
    proposals, report = select_fresh_natural_rows(
        rows,
        excluded_query_ids=set(),
        excluded_cluster_ids=set(),
        strata=_protocol_strata(),
    )
    assert report["selected"] == 48
    return proposals


def test_fresh_selection_is_balanced_and_distinct() -> None:
    proposals = _mock_proposals()
    assert len(proposals) == 48
    assert len({row["query_id"] for row in proposals}) == 48
    assert len({row["cluster_id"] for row in proposals}) == 48
    assert {row["length_band"] for row in proposals} == {"medium", "long"}


def test_controls_and_blind_shards_are_mechanically_valid() -> None:
    assert len(build_hidden_controls("a")) == 8
    packages, private, construction = build_blind_shards(_mock_proposals())
    assert construction["rows_per_shard"] == 18
    assert all(len(shard) == 18 for side in packages.values() for shard in side)
    assert len(private) == 144
    for annotator in ("a", "b"):
        source_hashes = {
            row["item_id"]: row["structure"]["source_sha256"]
            for shard in packages[annotator]
            for row in shard
        }
        for row in private:
            if row["annotator"] == annotator and row["kind"] == "repeat":
                assert (
                    source_hashes[row["item_id"]]
                    == source_hashes[row["natural_item_id"]]
                )
                assert row["shard_index"] != row["parent_shard_index"]


def _perfect_natural_label(item: dict) -> dict:
    structure = item["structure"]
    final = int(structure["block_count"]) - 1
    key = int(structure["blocks"][final]["unit_indices"][-1])
    annotation = {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": "supported",
        "block_roles": [
            {"block_id": index, "role": "main_step"}
            for index in range(int(structure["block_count"]))
        ],
        "final_block_id": final,
        "edge_decisions": [
            {
                "parent_block_id": edge["parent_block_id"],
                "child_block_id": edge["child_block_id"],
                "decision": "keep",
            }
            for edge in structure["candidate_edges"]
        ],
        "missing_edges": [],
        "key_unit_index": key,
        "confidence": "high",
        "rationale": "the last block completes the mock answer",
    }
    return validate_local_audit_annotation(annotation, item)


def test_perfect_blind_labels_pass_evaluator() -> None:
    packages_by_shard, private, _ = build_blind_shards(_mock_proposals())
    packages = {
        annotator: [row for shard in packages_by_shard[annotator] for row in shard]
        for annotator in ("a", "b")
    }
    private_map = {(row["annotator"], row["item_id"]): row for row in private}
    labels: dict[str, list[dict]] = {"a": [], "b": []}
    for annotator in ("a", "b"):
        natural_labels: dict[str, dict] = {}
        for item in packages[annotator]:
            metadata = private_map[(annotator, item["item_id"])]
            if metadata["kind"] == "natural":
                natural_labels[item["item_id"]] = _perfect_natural_label(item)
        for item in packages[annotator]:
            metadata = private_map[(annotator, item["item_id"])]
            if metadata["kind"] == "control":
                label = deepcopy(metadata["expected_label"])
            elif metadata["kind"] == "repeat":
                label = deepcopy(natural_labels[metadata["natural_item_id"]])
                label["item_id"] = item["item_id"]
            else:
                label = natural_labels[item["item_id"]]
            labels[annotator].append(label)

    gates = {
        "controls_min_pass": 8,
        "self_repeat_target_exact_min": 1.0,
        "common_usable_nonlow_min": 48,
        "final_block_exact_min": 1.0,
        "key_exact_min": 1.0,
        "complete_macro_f1_min": 1.0,
        "complete_macro_iou_min": 1.0,
        "complete_mask_coverage_min": 1.0,
        "role_decision_agreement_min": 1.0,
        "edge_decision_agreement_min": 1.0,
        "all_material_union_rate_max": 1.0,
        "missing_edge_row_rate_max": 0.0,
    }
    report = evaluate_blind_labels(
        packages=packages,
        private_index=private,
        labels=labels,
        gates=gates,
    )
    assert report["status"] == "PASS_PRIOR_V13_MECHANICAL_SMOKE"
