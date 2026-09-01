import json
import sys

import torch

from score_clir import main, parse_args
from src.consistency_localized_reward import ConsistencyLocalizedReward, RewardConfig


def _required() -> list[str]:
    return [
        "--input_jsonl",
        "input.jsonl",
        "--model",
        "model.pt",
        "--output_jsonl",
        "output.jsonl",
    ]


def test_full_diagnostics_remain_the_default() -> None:
    args = parse_args(_required())
    assert args.scalar_only is False


def test_scalar_only_mode_is_explicit() -> None:
    args = parse_args([*_required(), "--scalar_only"])
    assert args.scalar_only is True


def test_scalar_only_writes_scores_without_token_diagnostics(
    tmp_path, monkeypatch
) -> None:
    config = RewardConfig(hidden_dim=4, model_dim=4, encoder_type="identity")
    model = ConsistencyLocalizedReward(config)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_config": dict(config.__dict__), "state_dict": model.state_dict()},
        checkpoint,
    )
    rows = [
        {
            "id": "q0-c0",
            "query_id": "q0",
            "hidden_states": [[0.1, 0.2, 0.3, 0.4]],
            "correctness": 0,
        },
        {
            "id": "q0-c1",
            "query_id": "q0",
            "hidden_states": [[0.4, 0.3, 0.2, 0.1]],
            "correctness": 1,
        },
        {
            "id": "q1-c0",
            "query_id": "q1",
            "hidden_states": [[0.0, 0.1, 0.0, 0.1]],
            "correctness": 1,
        },
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_path = tmp_path / "output.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_clir.py",
            "--input_jsonl",
            str(input_path),
            "--model",
            str(checkpoint),
            "--output_jsonl",
            str(output_path),
            "--device",
            "cpu",
            "--amp_dtype",
            "none",
            "--scalar_only",
        ],
    )
    main()
    scored = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(scored) == 3
    assert sum(row["clir_selected_best_of_n"] for row in scored) == 2
    assert all(row["clir_scoring_mode"] == "scalar_only" for row in scored)
    assert all(isinstance(row["clir_score"], float) for row in scored)
    assert all("clir_key_prior" not in row for row in scored)
    assert all("clir_token_value" not in row for row in scored)
