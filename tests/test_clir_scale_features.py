import json
from pathlib import Path

import pytest
import torch

from extract_clir_scale_features import build_parser
from src.clir_scale_features import (
    AUTHORIZATION_SCHEMA,
    SELECTED_INPUT_SCHEMA,
    assign_workers,
    build_selected_inputs,
    condition_relative_path,
    expected_payload_records,
    query_marker_relative_path,
    rows_for_worker,
    selected_statistics,
    trajectory_relative_path,
    validate_tensor_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _materialized(trajectory_id, query_id, candidate_index, prompt, output):
    return {
        "id": trajectory_id,
        "query_id": query_id,
        "candidate_index": candidate_index,
        "prompt_token_ids": prompt,
        "output_token_ids": output,
    }


def _inventory(
    trajectory_id, query_id, candidate_index, prompt_count, output_count, owner
):
    return {
        "trajectory_id": trajectory_id,
        "query_id": query_id,
        "cluster_id": f"cluster-{query_id}",
        "acquisition_split": "train_acquisition",
        "source": "gsm8k",
        "source_subject": None,
        "source_level": None,
        "candidate_index": candidate_index,
        "prompt_token_count": prompt_count,
        "output_token_count": output_count,
        "condition_feature_owner": owner,
        "uses": ["train_positive"],
        "relation_ids": [f"relation-{query_id}"],
    }


def _selected_fixture():
    materialized = [
        _materialized("q0-c0", "q0", 0, [1, 2], [3, 4, 5]),
        _materialized("q0-c1", "q0", 1, [1, 2], [6]),
        _materialized("q1-c0", "q1", 0, [7], [8, 9]),
    ]
    inventory = [
        _inventory("q0-c0", "q0", 0, 2, 3, True),
        _inventory("q0-c1", "q0", 1, 2, 1, False),
        _inventory("q1-c0", "q1", 0, 1, 2, True),
    ]
    return build_selected_inputs(inventory, materialized)


def test_selected_inventory_join_preserves_exact_ids_and_owner_contract():
    selected = _selected_fixture()

    assert [row["schema_version"] for row in selected] == [SELECTED_INPUT_SCHEMA] * 3
    assert selected[0]["prompt_token_ids"] == [1, 2]
    assert selected[0]["output_token_ids"] == [3, 4, 5]
    assert selected_statistics(selected) == {
        "trajectory_count": 3,
        "query_count": 2,
        "condition_count": 2,
        "output_token_count": 6,
        "prompt_token_count": 3,
        "total_feature_token_count": 9,
    }


def test_selected_inventory_rejects_multiple_condition_owners():
    selected = _selected_fixture()
    materialized = [
        _materialized(
            row["trajectory_id"],
            row["query_id"],
            row["candidate_index"],
            row["prompt_token_ids"],
            row["output_token_ids"],
        )
        for row in selected
    ]
    inventory = [
        _inventory(
            row["trajectory_id"],
            row["query_id"],
            row["candidate_index"],
            row["prompt_token_count"],
            row["output_token_count"],
            row["query_id"] == "q0" or row["condition_feature_owner"],
        )
        for row in selected
    ]

    with pytest.raises(ValueError, match="exactly one condition owner"):
        build_selected_inputs(inventory, materialized)


def test_selected_inventory_rejects_token_count_drift():
    materialized = [_materialized("q-c", "q", 0, [1], [2, 3])]
    inventory = [_inventory("q-c", "q", 0, 1, 3, True)]

    with pytest.raises(ValueError, match="output token count drift"):
        build_selected_inputs(inventory, materialized)


def test_query_worker_assignment_is_deterministic_balanced_and_query_atomic():
    selected = _selected_fixture()
    first, first_stats = assign_workers(selected, 2)
    second, second_stats = assign_workers(selected, 2)

    assert first == second
    assert first_stats == second_stats
    workers_by_query = {}
    for row in first:
        workers_by_query.setdefault(row["query_id"], set()).add(row["worker_index"])
    assert all(len(workers) == 1 for workers in workers_by_query.values())
    assert sum(stat["trajectory_count"] for stat in first_stats) == 3
    assert sum(stat["feature_token_count"] for stat in first_stats) == 9
    for worker_index in range(2):
        groups = rows_for_worker(first, worker_index)
        for rows in groups.values():
            assert rows[0]["condition_feature_owner"] is True


def test_expected_payloads_have_one_condition_per_query(tmp_path: Path):
    selected, _ = assign_workers(_selected_fixture(), 2)
    records = expected_payload_records(tmp_path, selected)

    assert sum(record["kind"] == "trajectory" for record in records) == 3
    assert sum(record["kind"] == "condition" for record in records) == 2
    assert len({record["relative_path"] for record in records}) == 5
    assert trajectory_relative_path("q0-c0").endswith(".pt")
    assert condition_relative_path("q0").endswith(".pt")
    assert query_marker_relative_path("q0").endswith(".json")


def test_tensor_verifier_checks_shape_dtype_finiteness_and_checksum(tmp_path: Path):
    path = tmp_path / "feature.pt"
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).contiguous()
    torch.save(tensor, path)

    report = validate_tensor_file(
        path,
        expected_shape=[3, 4],
        expected_dtype=torch.bfloat16,
    )
    assert report["shape"] == [3, 4]
    assert report["dtype"] == "bfloat16"
    assert report["raw_tensor_bytes"] == 24
    assert len(report["sha256"]) == 64

    with pytest.raises(ValueError, match="shape drift"):
        validate_tensor_file(
            path,
            expected_shape=[2, 6],
            expected_dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="checksum drift"):
        validate_tensor_file(
            path,
            expected_shape=[3, 4],
            expected_dtype=torch.bfloat16,
            expected_sha256="0" * 64,
        )


def test_tensor_verifier_rejects_nonfinite_payload(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([[float("nan")]], dtype=torch.bfloat16), path)

    with pytest.raises(FloatingPointError, match="non-finite"):
        validate_tensor_file(
            path,
            expected_shape=[1, 1],
            expected_dtype=torch.bfloat16,
        )


def test_v6_1_feature_authorization_is_inventory_only_and_training_off():
    path = (
        PROJECT_ROOT
        / "configs/data_expansion_scale_v6/feature_extraction_authorization_v6_1.json"
    )
    authorization = json.loads(path.read_text(encoding="utf-8"))

    assert authorization["schema_version"] == AUTHORIZATION_SCHEMA
    assert authorization["expected_inventory"]["trajectory_count"] == 1357
    assert authorization["expected_inventory"]["condition_count"] == 612
    assert authorization["expected_inventory"]["raw_feature_bytes"] == 105451923456
    assert authorization["authorized_scope"]["inventory_only_feature_extraction"]
    assert not authorization["authorized_scope"]["all_16000_rollout_feature_extraction"]
    assert not authorization["authorized_scope"]["training"]


def test_scale_feature_cli_has_separate_prepare_extract_verify_and_finalize_steps():
    parser = build_parser()

    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["verify-plan"]).command == "verify-plan"
    assert parser.parse_args(["preflight"]).command == "preflight"
    assert (
        parser.parse_args(["extract-worker", "--worker-index", "0"]).worker_index == 0
    )
    assert parser.parse_args(["verify-worker", "--worker-index", "0"]).worker_index == 0
    assert parser.parse_args(["finalize"]).command == "finalize"
