"""Contract tests for the released official SWIFT reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from run_swift_official_generalization import (
    DATASETS,
    _bootstrap_interval,
    _gsm8k_answer,
    _load_prepare_manifest,
    _query_ranges,
    _verify_rollout,
    load_protocol,
)


def test_frozen_protocol_has_official_three_dataset_targets() -> None:
    protocol = load_protocol(verify_files=False)
    assert tuple(protocol["datasets"]) == DATASETS
    assert protocol["evaluation"]["K"] == [1, 2, 4, 8, 16, 32, 64]
    assert protocol["evaluation"]["published_targets_percent"] == {
        "math": 62.8,
        "gsm8k": 93.6,
        "aqua_rat": 75.8,
    }
    assert protocol["evidence_boundary"]["not_a_protected_or_blinded_test"] is True
    assert "tokenizer_config.json" in protocol["official_sources"]["base_model"][
        "required_runtime_files"
    ]
    assert protocol["runtime"]["package_versions"]["vllm"] == "0.5.3.post1"
    assert protocol["generation"]["worker_multiprocessing_method"] == "spawn"


def test_query_ranges_are_contiguous_balanced_and_exhaustive() -> None:
    ranges = _query_ranges()
    assert len(ranges) == 8
    assert ranges[0][0] == 0 and ranges[-1][1] == 500
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert {end - start for start, end in ranges} == {62, 63}
    assert sum(end - start for start, end in ranges) == 500


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("work #### 72", "72"), ("#### 1/2 ", "1/2"), ("plain", "plain")],
)
def test_gsm8k_answer_matches_upstream_preprocessor(raw: str, expected: str) -> None:
    assert _gsm8k_answer(raw) == expected


def test_rollout_population_requires_exact_upstream_order(tmp_path: Path) -> None:
    protocol = {"runtime": {"output_root": str(tmp_path)}}
    rows = [
        {
            "idx": query,
            "prompt": f"q{query}",
            "response": f"r{candidate}",
            "reference": "1",
            "steps": ["one"],
            "correctness": candidate == 0,
        }
        for query in range(500)
        for candidate in range(64)
    ]
    path = tmp_path / "rollouts/math.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    _, report = _verify_rollout(protocol, "math")
    assert report["rows"] == 32000
    assert report["queries"] == 500
    assert report["correct_candidates"] == 500

    rows[64]["idx"] = 0
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="order drift"):
        _verify_rollout(protocol, "math")


def test_bootstrap_interval_is_deterministic_and_on_probability_scale() -> None:
    values = np.asarray([0.0, 1.0] * 250)
    first = _bootstrap_interval(values, seed=7, replicates=500)
    second = _bootstrap_interval(values, seed=7, replicates=500)
    assert first == second
    assert 0.0 <= first[0] <= 0.5 <= first[1] <= 1.0


def test_prepare_manifest_rejects_incomplete_model_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model-00001.safetensors").write_bytes(b"weight")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n" * 500, encoding="utf-8")
    manifest = {
        "status": "PASS_SWIFT_OFFICIAL_GENERALIZATION_INPUT_FREEZE",
        "protocol_file_sha256": "protocol",
        "runner_file_sha256": "runner",
        "inputs": {
            "math": {
                "path": str(input_path),
                "file_sha256": "input",
                "rows": 500,
            }
        },
        "base_model_files": {"config.json": "config"},
        "reward_checkpoint_file_sha256": "reward",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol = {
        "runtime": {"output_root": str(tmp_path)},
        "official_sources": {
            "base_model": {
                "local_path": str(model_root),
                "required_runtime_files": [
                    "config.json",
                    "model.safetensors.index.json",
                ],
            },
            "released_reward_checkpoint": {"file_sha256": "reward"},
        },
    }
    monkeypatch.setattr(
        "run_swift_official_generalization._prepare_manifest_path",
        lambda unused: manifest_path,
    )
    monkeypatch.setattr(
        "run_swift_official_generalization.file_sha256",
        lambda path: {
            str(manifest_path): "unused",
            str(input_path): "input",
        }.get(str(path), "unexpected"),
    )
    monkeypatch.setattr(
        "run_swift_official_generalization.PROTOCOL_PATH", tmp_path / "protocol.json"
    )
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "run_swift_official_generalization.__file__", str(tmp_path / "runner.py")
    )
    (tmp_path / "runner.py").write_text("", encoding="utf-8")
    manifest["protocol_file_sha256"] = "unexpected"
    manifest["runner_file_sha256"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory drift"):
        _load_prepare_manifest(protocol)
