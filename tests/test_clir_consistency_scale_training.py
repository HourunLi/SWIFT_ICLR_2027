import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_clir_consistency import REPORT_SCHEMA, relation_metrics
from prepare_clir_consistency_training import build_parser
from src.clir_consistency_scale_training import (
    construct_manifests,
    load_authorization,
    relative_length_roles,
)
from summarize_clir_consistency import summarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _feature(
    trajectory_id: str,
    query_id: str,
    cluster_id: str,
    split: str,
    length: int,
    candidate_index: int = 0,
):
    return {
        "id": trajectory_id,
        "query_id": query_id,
        "candidate_index": candidate_index,
        "source": "gsm8k",
        "source_subject": None,
        "source_level": None,
        "cluster_id": cluster_id,
        "acquisition_split": split,
        "prompt_token_ids": [1, 2],
        "output_token_ids": list(range(length)),
        "prompt_token_count": 2,
        "output_token_count": length,
        "hidden_states_path": f"{trajectory_id}.pt",
        "condition_states_path": f"{query_id}.condition.pt",
        "hidden_states_sha256": "a" * 64,
        "condition_states_sha256": "b" * 64,
        "feature_dim": 8,
        "num_feature_layers": 2,
        "per_layer_dim": 4,
        "feature_dtype": "bfloat16",
    }


def _positive(relation_id, query_id, cluster_id, left, right, split):
    return {
        "relation_id": relation_id,
        "label": 1,
        "label_tier": "silver",
        "query_id": query_id,
        "cluster_id": cluster_id,
        "left_id": left,
        "right_id": right,
        "acquisition_split": split,
    }


def test_constructed_manifests_are_matched_and_split_safe(tmp_path: Path):
    historical = [
        {
            **_feature("old-0", "gsm8k-train-00001", "old-c", "historical", 3),
            "split": "train",
            "source_index": 1,
            "correctness": 0,
            "semantic_id": "old-consistency-must-be-stripped",
            "style_id": "old-style-must-be-stripped",
        }
    ]
    features = [
        _feature(
            "train-short", "gsm8k:train:00002", "train-c", "train_acquisition", 2, 0
        ),
        _feature(
            "train-long", "gsm8k:train:00002", "train-c", "train_acquisition", 5, 1
        ),
        _feature(
            "held-short", "gsm8k:train:00003", "held-c", "heldout_acquisition", 2, 0
        ),
        _feature(
            "held-long", "gsm8k:train:00003", "held-c", "heldout_acquisition", 4, 1
        ),
        _feature(
            "negative-other",
            "gsm8k:train:00004",
            "other-c",
            "heldout_acquisition",
            2,
            0,
        ),
    ]
    train_positive = [
        _positive(
            "train-r",
            "gsm8k:train:00002",
            "train-c",
            "train-short",
            "train-long",
            "train_acquisition",
        )
    ]
    held_positive = [
        _positive(
            "held-r",
            "gsm8k:train:00003",
            "held-c",
            "held-short",
            "held-long",
            "heldout_acquisition",
        )
    ]
    held_negative = [
        {
            "relation_id": "negative-r",
            "label": 0,
            "evaluation_only": True,
            "left_id": "held-short",
            "right_id": "negative-other",
        }
    ]

    result = construct_manifests(
        historical,
        features,
        train_positive,
        held_positive,
        held_negative,
        historical_manifest_parent=tmp_path,
        feature_manifest_parent=tmp_path,
    )

    assert len(result["train_rows"]) == 3
    assert len(result["validation_rows"]) == 2
    assert len(result["evaluation_rows"]) == 3
    assert result["statistics"]["train_heldout_query_overlap"] == 0
    assert result["statistics"]["heldout_endpoint_overlap_positive_negative"] == 1
    old = result["train_rows"][0]
    assert "semantic_id" not in old and "style_id" not in old
    new = result["train_rows"][1:]
    assert {row["style_id"] for row in new} == {
        "relative_compact",
        "relative_expanded",
    }
    assert {row["correctness"] for row in new} == {1}


def test_relative_length_role_has_deterministic_id_tiebreak():
    features = {
        "b": {"output_token_count": 3},
        "a": {"output_token_count": 3},
    }
    roles = relative_length_roles({"left_id": "b", "right_id": "a"}, features)
    assert roles == {"a": "relative_compact", "b": "relative_expanded"}


def test_relation_metrics_separate_positive_and_negative_pairs():
    representations = {
        "a": [1.0, 0.0],
        "b": [1.0, 0.0],
        "c": [0.0, 1.0],
        "d": [0.0, 1.0],
    }
    scores = {"a": 1.0, "b": 1.1, "c": -1.0, "d": -0.9}
    positive = [
        {"relation_id": "p1", "label": 1, "left_id": "a", "right_id": "b"},
        {"relation_id": "p2", "label": 1, "left_id": "c", "right_id": "d"},
    ]
    negative = [
        {"relation_id": "n1", "label": 0, "left_id": "a", "right_id": "c"},
        {"relation_id": "n2", "label": 0, "left_id": "b", "right_id": "d"},
    ]

    report = relation_metrics(representations, scores, positive, negative)

    assert report["representation"]["mean_separation_positive_minus_negative"] == 1.0
    assert report["representation"]["relation_classification_auroc"] == 1.0
    assert report["representation"]["relation_classification_average_precision"] == 1.0
    assert report["score"]["mean_gap_separation_negative_minus_positive"] > 1.0


def _heldout_report(cell: str, seed: int, positive: float, negative: float):
    relations = [
        {
            "relation_id": "p1",
            "label": 1,
            "left_id": "a",
            "right_id": "b",
            "cosine_similarity": positive,
            "absolute_score_gap": 0.1,
        },
        {
            "relation_id": "p2",
            "label": 1,
            "left_id": "c",
            "right_id": "d",
            "cosine_similarity": positive,
            "absolute_score_gap": 0.1,
        },
        {
            "relation_id": "n1",
            "label": 0,
            "left_id": "a",
            "right_id": "c",
            "cosine_similarity": negative,
            "absolute_score_gap": 0.9,
        },
        {
            "relation_id": "n2",
            "label": 0,
            "left_id": "b",
            "right_id": "d",
            "cosine_similarity": negative,
            "absolute_score_gap": 0.9,
        },
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_HELDOUT_RELATION_EVALUATION",
        "cell": cell,
        "seed": seed,
        "relation_signature_sha256": "frozen",
        "inputs": {
            key: {"file_sha256": key}
            for key in (
                "endpoint_manifest",
                "positive_relations",
                "negative_relations",
                "expected_train_manifest",
            )
        },
        "relations": relations,
        "representation": {
            "mean_separation_positive_minus_negative": positive - negative,
            "relation_classification_auroc": 1.0,
            "relation_classification_average_precision": 1.0,
            "positive_cosine": {"mean": positive},
            "negative_cosine": {"mean": negative},
        },
        "score": {
            "mean_gap_separation_negative_minus_positive": 0.8,
            "relation_classification_auroc_from_negative_gap": 1.0,
        },
    }


def test_paired_summary_uses_all_seeds_and_frozen_relations(tmp_path: Path):
    for seed in (42, 43):
        for cell, positive, negative in (
            ("c0", 0.5, 0.4),
            ("c1", 0.8, 0.2),
        ):
            target = tmp_path / f"seed_{seed}" / cell / "heldout_relations.json"
            target.parent.mkdir(parents=True)
            target.write_text(
                json.dumps(_heldout_report(cell, seed, positive, negative)),
                encoding="utf-8",
            )

    report = summarize(tmp_path, [42, 43], bootstrap_replicates=100, bootstrap_seed=7)

    contrast = report["contrast_c1_minus_c0"]
    assert np.isclose(contrast["mean_cosine_separation_delta"], 0.5)
    assert contrast["decision"] == "SUPPORTS_C1_HELDOUT_RELATION_SEPARATION"
    assert set(contrast["per_seed"]) == {"42", "43"}


def test_real_authorization_is_c_only_and_cli_is_staged():
    path = (
        PROJECT_ROOT
        / "configs/data_expansion_scale_v6/consistency_training_v6_1/authorization.json"
    )
    authorization = load_authorization(path)
    assert authorization["authorized_scope"]["c0_c1_three_seed_three_epoch_training"]
    assert not authorization["authorized_scope"]["hallucination_training"]
    assert not authorization["authorized_scope"]["dual_prior_training"]
    parser = build_parser()
    assert parser.parse_args(["materialize"]).command == "materialize"
    assert parser.parse_args(["verify"]).command == "verify"
    assert parser.parse_args(["preflight", "--cell", "c1"]).cell == "c1"


def test_constructed_manifest_rejects_train_heldout_leakage(tmp_path: Path):
    features = [
        _feature("a", "gsm8k:train:00002", "shared", "train_acquisition", 2, 0),
        _feature("b", "gsm8k:train:00002", "shared", "train_acquisition", 3, 1),
        _feature("c", "gsm8k:train:00002", "shared", "heldout_acquisition", 2, 2),
        _feature("d", "gsm8k:train:00002", "shared", "heldout_acquisition", 3, 3),
        _feature("e", "gsm8k:train:00003", "other", "heldout_acquisition", 2, 0),
    ]
    historical = [
        {
            **_feature("old", "gsm8k-train-00001", "old", "historical", 2),
            "split": "train",
            "source_index": 1,
            "correctness": 1,
        }
    ]
    with pytest.raises(ValueError, match="Train/heldout query overlap"):
        construct_manifests(
            historical,
            features,
            [
                _positive(
                    "r1", "gsm8k:train:00002", "shared", "a", "b", "train_acquisition"
                )
            ],
            [
                _positive(
                    "r2", "gsm8k:train:00002", "shared", "c", "d", "heldout_acquisition"
                )
            ],
            [
                {
                    "relation_id": "n",
                    "label": 0,
                    "evaluation_only": True,
                    "left_id": "c",
                    "right_id": "e",
                }
            ],
            historical_manifest_parent=tmp_path,
            feature_manifest_parent=tmp_path,
        )
