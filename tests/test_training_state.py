import json
from pathlib import Path
import subprocess
import sys

import torch

from src.clir_data import write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def nested_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def run_train(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_clir.py"), *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_resume_matches_uninterrupted_training(tmp_path: Path):
    rows = []
    for index in range(4):
        feature = tmp_path / f"{index}.pt"
        torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4) + index, feature)
        rows.append(
            {
                "id": str(index),
                "query_id": f"q{index // 2}",
                "hidden_states_path": str(feature),
                "correctness": index % 2,
            }
        )
    data = tmp_path / "train.jsonl"
    write_jsonl(data, rows)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "hidden_dim": 4,
                    "projection_dim": 4,
                    "condition_attention_dim": 4,
                    "consistency_weight": 0.0,
                    "hallucination_weight": 0.0,
                    "mil_weight": 0.0,
                    "token_reward_weight": 0.0,
                    "tail_weight": 0.0,
                    "pseudo_tail_weight": 0.0,
                    "progress_weight": 0.0,
                    "prior_weight": 0.0,
                },
                "training": {
                    "seed": 17,
                    "epochs": 2,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "amp_dtype": "none",
                    "num_workers": 0,
                    "pin_memory": False,
                    "group_by_semantic_id": False,
                    "prior_phase_mode": "joint",
                },
            }
        ),
        encoding="utf-8",
    )
    common = ("--train_jsonl", str(data), "--config", str(config))
    full = tmp_path / "full.pt"
    partial = tmp_path / "partial.pt"
    resumed = tmp_path / "resumed.pt"

    run_train(*common, "--output_model", str(full))
    run_train(*common, "--output_model", str(partial), "--epochs", "1")
    run_train(
        *common,
        "--output_model",
        str(resumed),
        "--resume_from",
        str(partial),
    )

    full_state = torch.load(full, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
    assert full_state["completed_epoch"] == resumed_state["completed_epoch"] == 2
    assert full_state["metrics"] == resumed_state["metrics"]
    assert nested_equal(full_state["state_dict"], resumed_state["state_dict"])
    assert nested_equal(
        full_state["optimizer_state_dict"], resumed_state["optimizer_state_dict"]
    )
