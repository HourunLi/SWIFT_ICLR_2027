import json
from pathlib import Path

from prepare_clir_three_module import (
    build_parser,
    load_training_authorization,
    verify_factorial_configs,
)
from src.clir_three_module import build_unified_data


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/three_module_expansion_v1/protocol.json"
AUTHORIZATION = ROOT / "configs/three_module_expansion_v1/training_authorization.json"


def _row(row_id: str, query_id: str, **extra: object) -> dict:
    return {
        "id": row_id,
        "query_id": query_id,
        "candidate_index": 0,
        "correctness": 1,
        "prompt_token_ids": [1],
        "output_token_ids": [2, 3],
        "hidden_states_path": f"features/{row_id}.pt",
        "condition_states_path": f"features/{query_id}.pt",
        **extra,
    }


def _prior(row: dict) -> dict:
    return {
        **row,
        "key_prior_target": [1, 0],
        "complete_prior_target": [1, 1],
    }


def test_three_module_configs_form_complete_factorial() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    observed = verify_factorial_configs(protocol)
    assert set(observed) == {"u0", "c", "h", "p", "ch", "cp", "hp", "full"}
    assert observed["u0"]["factors"] == [0, 0, 0]
    assert observed["full"]["factors"] == [1, 1, 1]


def test_three_module_parser_exposes_frozen_training_gates() -> None:
    args = build_parser().parse_args(["materialize"])
    assert args.command == "materialize"
    assert args.protocol.endswith("configs/three_module_expansion_v1/protocol.json")
    preflight = build_parser().parse_args(["preflight", "--device", "cpu"])
    assert preflight.command == "preflight"
    assert preflight.authorization.endswith(
        "configs/three_module_expansion_v1/training_authorization.json"
    )


def test_three_module_training_authorization_binds_complete_grid() -> None:
    authorization = load_training_authorization(AUTHORIZATION)
    assert authorization["cell_order"] == [
        "u0",
        "c",
        "h",
        "p",
        "ch",
        "cp",
        "hp",
        "full",
    ]
    assert authorization["training"]["runs"] == 24
    assert authorization["cells"]["full"]["factors"] == [1, 1, 1]


def test_unified_merge_enriches_shared_prior_and_removes_cross_task_dev() -> None:
    shared0 = _row("base-0", "q-base-0")
    shared1 = _row("base-1", "q-base-1")
    consistency = [
        _row(
            "c-0",
            "q-c",
            semantic_id="relation-0",
            consistency_supervision=True,
        ),
        _row(
            "c-1",
            "q-c",
            semantic_id="relation-0",
            consistency_supervision=True,
        ),
    ]
    h_rows = [
        _row(
            "h-positive",
            "q-h-positive",
            path_hallucinated=1,
            hallucination_onset=1,
        ),
        _row(
            "h-clean",
            "q-h-clean",
            path_hallucinated=0,
            hallucination_onset=-1,
        ),
    ]
    h_train = [shared0, shared1, *consistency, *h_rows]
    prior_train = [
        _prior(shared0),
        shared1,
        _prior(_row("prior-new", "q-prior-new")),
    ]
    h_dev = [
        _row("hdev-keep", "q-hdev-keep", path_hallucinated=0, hallucination_onset=-1),
        _row("hdev-drop", "q-prior-new", path_hallucinated=0, hallucination_onset=-1),
    ]
    prior_dev = [
        _prior(_row("pdev-keep", "q-pdev-keep")),
        _prior(_row("pdev-drop", "q-h-positive")),
    ]
    expected = {
        "shared_historical_rows": 2,
        "legacy_prior_rows": 1,
        "new_prior_rows": 1,
        "train_rows": 7,
        "train_queries": 6,
        "consistency_endpoint_rows": 2,
        "consistency_relations": 1,
        "h_rows": 2,
        "h_positive_rows": 1,
        "h_clean_rows": 1,
        "prior_rows": 2,
        "clean_h_dev_rows": 1,
        "clean_prior_dev_rows": 1,
    }
    result = build_unified_data(
        consistency_h0_train=h_train,
        prior_train=prior_train,
        h_dev=h_dev,
        prior_dev=prior_dev,
        consistency_h0_parent=Path("/source/h"),
        prior_parent=Path("/source/p"),
        h_dev_parent=Path("/source/h"),
        prior_dev_parent=Path("/source/p"),
        target_parent=Path("/target/data"),
        expected=expected,
    )
    assert [row["id"] for row in result["train"]] == [
        "base-0",
        "base-1",
        "c-0",
        "c-1",
        "h-positive",
        "h-clean",
        "prior-new",
    ]
    assert result["train"][0]["key_prior_target"] == [1, 0]
    assert result["train"][0]["prior_merge_origin"] == ("legacy_shared_historical_row")
    assert [row["id"] for row in result["h_dev"]] == ["hdev-keep"]
    assert [row["id"] for row in result["prior_dev"]] == ["pdev-keep"]
    assert result["report"]["removed_h_dev_queries"] == ["q-prior-new"]
    assert result["report"]["removed_prior_dev_queries"] == ["q-h-positive"]
