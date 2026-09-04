import copy
import json
from pathlib import Path

from check_clir_math_hard_eval import build_parser as build_checker_parser
from prepare_clir_math_hard_eval import (
    _candidate_test_rows,
    build_rollout_shards,
    extract_last_boxed,
    load_protocol,
    select_protected_queries,
)
from src.clir_smoke import validate_source_row


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "configs/math_hard_eval_v1/protocol.json").read_text(encoding="utf-8")
    )


def test_protocol_is_hash_bound_and_balanced() -> None:
    protocol = load_protocol(ROOT / "configs/math_hard_eval_v1/protocol.json")
    assert protocol["source"]["target_queries_per_level"] == {"4": 250, "5": 250}
    assert protocol["generation"]["candidate_count"] == 16
    assert protocol["evaluation"]["primary_k"] == 16


def test_checker_defaults_to_signal_safe_sequential_execution() -> None:
    args = build_checker_parser().parse_args(["materialize"])
    assert args.workers == 1


def test_math_test_namespace_is_explicitly_valid() -> None:
    row = validate_source_row(
        {
            "source": "math",
            "query_id": "math:test:algebra:00001",
            "question": "Find x.",
            "reference_answer": "2",
        }
    )
    assert row["query_id"].startswith("math:test:")


def test_extract_last_boxed_handles_nested_and_uses_last() -> None:
    solution = r"First \boxed{1}. Finally \boxed{\frac{3}{x+1}}."
    assert extract_last_boxed(solution) == r"\frac{3}{x+1}"
    assert extract_last_boxed("no answer marker") is None


def test_candidate_filter_is_level_and_reference_only() -> None:
    rows = [
        {
            "source": "math",
            "query_id": "math:test:algebra:00000",
            "source_record_id": "algebra/test/0",
            "question": "Hard problem",
            "reference_answer": "ignored",
            "source_solution": r"Thus \boxed{7}.",
            "source_level": 4,
            "source_subject": "algebra",
            "source_license": "MIT",
        },
        {
            "source": "math",
            "query_id": "math:test:algebra:00001",
            "source_record_id": "algebra/test/1",
            "question": "Easy problem",
            "reference_answer": "ignored",
            "source_solution": r"Thus \boxed{1}.",
            "source_level": 3,
            "source_subject": "algebra",
            "source_license": "MIT",
        },
    ]
    selected, rejected = _candidate_test_rows(rows, _protocol())
    assert [row["reference_answer"] for row in selected] == ["7"]
    assert rejected == {"level": 1}
    assert selected[0]["sealed_until_weight_lock"] is True


def test_selection_is_deterministic_and_one_per_cluster() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["source"]["target_queries_per_level"] = {"4": 2, "5": 2}
    protocol["source"]["total_queries"] = 4
    rows = []
    for level in (4, 5):
        for index in range(4):
            rows.append(
                {
                    "query_id": f"math:test:algebra:{level}{index:04d}",
                    "cluster_id": f"cluster-{level}-{index // 2}",
                    "source_level": level,
                }
            )
    first, report = select_protected_queries(rows, protocol)
    second, _ = select_protected_queries(list(reversed(rows)), protocol)
    assert [row["query_id"] for row in first] == [row["query_id"] for row in second]
    assert [row["source_level"] for row in first] == [4, 5, 4, 5]
    assert len({row["cluster_id"] for row in first}) == 4
    assert report["selection_used_generation_or_scores"] is False


def test_rollout_shards_preserve_frozen_level_balance() -> None:
    rows = []
    for index in range(250):
        for level in (4, 5):
            rows.append(
                {
                    "query_id": f"math:test:algebra:{level}-{index:04d}",
                    "source_level": level,
                }
            )
    shards = build_rollout_shards(rows, _protocol())
    assert len(shards) == 10
    assert all(shard["level_counts"] == {"4": 25, "5": 25} for shard in shards)
    assert all(shard["expected_candidate_rows"] == 800 for shard in shards)
