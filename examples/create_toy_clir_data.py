"""Create a tiny CLIR JSONL file with synthetic hidden states.

This is only for smoke testing the training/scoring pipeline. Real experiments
should replace these random vectors with frozen LLM hidden states extracted from
query/context/trajectory tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clir_data import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create toy CLIR data.")
    parser.add_argument("--output_jsonl", default="examples/toy_clir.jsonl")
    parser.add_argument("--feature_dir", default="examples/features")
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    styles = ["direct", "verbose"]
    for query_idx in range(3):
        for cand_idx, style in enumerate(styles):
            length = 5 + cand_idx
            hidden = torch.randn(length, args.hidden_dim)
            condition = torch.randn(3, args.hidden_dim)
            feature_path = feature_dir / f"q{query_idx}_c{cand_idx}.pt"
            condition_path = feature_dir / f"q{query_idx}_c{cand_idx}_condition.pt"
            torch.save(hidden, feature_path)
            torch.save(condition, condition_path)

            hallucination_onset = 3 if cand_idx == 1 else -1
            path_hallucinated = cand_idx == 1
            key_prior = [1.0 if i == 1 else 0.0 for i in range(length)]
            complete_prior = [1.0 if i < min(length, 4) else 0.0 for i in range(length)]
            token_advantage = [0.2 * (i + 1) for i in range(length)]
            if path_hallucinated:
                token_advantage[hallucination_onset:] = [-0.5] * (
                    length - hallucination_onset
                )

            rows.append(
                {
                    "id": f"q{query_idx}-cand{cand_idx}",
                    "query_id": f"q{query_idx}",
                    "hidden_states_path": str(feature_path),
                    "condition_states_path": str(condition_path),
                    "correctness": 1 if cand_idx == 0 else 0,
                    "semantic_id": f"q{query_idx}",
                    "style_id": style,
                    "hallucination_onset": hallucination_onset,
                    "path_hallucinated": path_hallucinated,
                    "token_advantage": token_advantage,
                    "progress_targets": token_advantage,
                    "key_prior_target": key_prior,
                    "complete_prior_target": complete_prior,
                }
            )

    write_jsonl(args.output_jsonl, rows)
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
