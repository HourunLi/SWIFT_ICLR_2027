from __future__ import annotations

from copy import deepcopy

from src.clir_prior_edge_candidates_v14 import FROZEN_EDGE_PROPOSAL_SCHEMA
from src.clir_prior_mechanical import validate_local_audit_annotation
from src.clir_prior_mechanical_smoke_v14 import (
    PACKAGE_SCHEMA,
    PROPOSAL_SCHEMA,
    build_blind_shards_v14,
    build_hidden_controls_v14,
    evaluate_blind_labels_v14,
    public_package_item_v14,
    select_fresh_natural_rows_v14,
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
            f"v{unit} = {unit} + 1 = {unit + 1}"
            for unit in range(material_count)
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
    proposals, report = select_fresh_natural_rows_v14(
        rows,
        excluded_query_ids=set(),
        excluded_cluster_ids=set(),
        strata=_protocol_strata(),
    )
    assert report["selected"] == 48
    return proposals


def test_v14_selection_has_new_schema_ids_and_distinct_rows() -> None:
    proposals = _mock_proposals()
    assert len(proposals) == 48
    assert {row["schema_version"] for row in proposals} == {PROPOSAL_SCHEMA}
    assert all(row["item_id"].startswith("prior-v14-natural-") for row in proposals)
    assert len({row["query_id"] for row in proposals}) == 48
    assert len({row["cluster_id"] for row in proposals}) == 48


def test_v14_public_item_freezes_recall_first_edges_with_a_six_parent_cap() -> None:
    source = {
        "item_id": "multi",
        "question": "Compute three subtotals and add them.",
        "response": "2*3=6\n4*5=20\n6*7=42\n6+20+42=68",
        "units": [
            {"unit_index": 0, "kind": "material_claim", "text": "2 * 3 = 6"},
            {"unit_index": 2, "kind": "material_claim", "text": "4 * 5 = 20"},
            {"unit_index": 4, "kind": "material_claim", "text": "6 * 7 = 42"},
            {
                "unit_index": 6,
                "kind": "material_claim",
                "text": "6 + 20 + 42 = 68",
            },
        ],
    }
    item = public_package_item_v14(source)
    assert item["schema_version"] == PACKAGE_SCHEMA
    assert item["structure"]["candidate_edge_schema"] == FROZEN_EDGE_PROPOSAL_SCHEMA
    edges = item["structure"]["candidate_edges"]
    assert {(edge["parent_block_id"], edge["child_block_id"]) for edge in edges} >= {
        (0, 3),
        (1, 3),
        (2, 3),
    }
    per_child: dict[int, int] = {}
    for edge in edges:
        per_child[edge["child_block_id"]] = per_child.get(
            edge["child_block_id"], 0
        ) + 1
    assert max(per_child.values()) <= 6


def test_fresh_v14_controls_and_repeat_shards_are_valid() -> None:
    controls = build_hidden_controls_v14("a")
    assert len(controls) == 8
    assert {name for _, _, name in controls} >= {
        "three_numeric_producers",
        "comma_number_producer",
        "answer_only_empty_structure",
    }
    packages, private, construction = build_blind_shards_v14(_mock_proposals())
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
                assert source_hashes[row["item_id"]] == source_hashes[
                    row["natural_item_id"]
                ]
                assert row["shard_index"] != row["parent_shard_index"]


def _perfect_natural_label(item: dict) -> dict:
    structure = item["structure"]
    final = int(structure["block_count"]) - 1
    key = int(structure["blocks"][final]["unit_indices"][-1])
    return validate_local_audit_annotation(
        {
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
            "rationale": "the last mock block completes the answer",
        },
        item,
    )


def _perfect_population() -> tuple[dict[str, list[dict]], list[dict], dict[str, list[dict]]]:
    packages_by_shard, private, _ = build_blind_shards_v14(_mock_proposals())
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
    return packages, private, labels


def _perfect_gates() -> dict:
    return {
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


def test_perfect_v14_blind_population_passes() -> None:
    packages, private, labels = _perfect_population()
    report = evaluate_blind_labels_v14(
        packages=packages,
        private_index=private,
        labels=labels,
        gates=_perfect_gates(),
    )
    assert report["status"] == "PASS_PRIOR_V14_MECHANICAL_RECALL_SMOKE"
    assert report["v13_terminal_decision_unchanged"] is True
    assert report["trainable_labels_published"] is False


def test_v14_schema_failure_does_not_report_a_v13_status() -> None:
    packages, private, labels = _perfect_population()
    answer_only = next(
        row
        for row in private
        if row["annotator"] == "b"
        and row.get("control_name") == "answer_only_empty_structure"
    )
    label = next(
        row for row in labels["b"] if row["item_id"] == answer_only["item_id"]
    )
    label["block_roles"] = [{"block_id": 0, "role": "answer_wrapper"}]
    report = evaluate_blind_labels_v14(
        packages=packages,
        private_index=private,
        labels=labels,
        gates=_perfect_gates(),
    )
    assert report["status"] == "FAIL_PRIOR_V14_SCHEMA"
    assert report["v13_terminal_decision_unchanged"] is True
