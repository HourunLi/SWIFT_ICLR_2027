import json
from pathlib import Path

from prepare_clir_prior_v16_training import (
    _balanced_shards,
    _verify_configs,
    build_parser,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/data_expansion_prior_v16/posthoc_training_v1/protocol.json"
)


def _row(index: int, prompt: int, output: int) -> dict:
    return {
        "id": f"row-{index}",
        "feature_inventory_index": index,
        "prompt_token_ids": list(range(prompt)),
        "output_token_ids": list(range(output)),
    }


def test_protocol_binds_minimal_staged_grid_and_claim_boundary() -> None:
    protocol = load_protocol(PROTOCOL)
    assert protocol["training"]["stage_1_cells"] == ["r0", "p0"]
    assert protocol["training"]["stage_2_cells"] == ["ch", "full"]
    assert protocol["cells"]["full"]["factors"] == [1, 1, 1]
    assert protocol["cells"]["full"]["gate_prior_weight"] == 0.25
    assert protocol["evaluation"]["ranking"].startswith("deferred_until_fresh")
    observed = _verify_configs(protocol)
    assert observed["p0"]["weights"] == [0.0, 0.0, 1.0, 0.0]
    assert observed["full"]["weights"] == [1.0, 1.0, 1.0, 0.25]


def test_token_balanced_shards_are_deterministic_and_complete() -> None:
    rows = [_row(0, 1, 9), _row(1, 2, 7), _row(2, 3, 5), _row(3, 4, 3)]
    first = _balanced_shards(rows, 2)
    second = _balanced_shards(rows, 2)
    assert first == second
    assert sorted(row["feature_inventory_index"] for shard in first for row in shard) == [
        0,
        1,
        2,
        3,
    ]
    totals = [
        sum(len(row["prompt_token_ids"]) + len(row["output_token_ids"]) for row in shard)
        for shard in first
    ]
    assert max(totals) - min(totals) <= 2


def test_parser_exposes_feature_and_training_preflight_stages() -> None:
    assert build_parser().parse_args(["prepare"]).command == "prepare"
    parsed = build_parser().parse_args(["preflight", "--stage", "2"])
    assert parsed.command == "preflight"
    assert parsed.stage == 2


def test_protocol_json_is_strict_json() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "clir-prior-v16-posthoc-training-protocol-v1"
