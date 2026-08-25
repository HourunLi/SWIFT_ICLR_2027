from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prepare_clir_smoke import (
    _ordered_vllm_candidates,
    build_annotation_packages,
    command_adjudication_package,
    command_resolve_dedup,
    command_dedup_triage,
    command_fixture,
    command_triage,
    evaluate_package_reliability,
)
from src.clir_smoke import (
    annotation_signature,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    check_numeric_response,
    freeze_query_pool,
    load_asdiv_a_repository,
    material_claim_char_spans,
    near_duplicate_candidates,
    normalize_question,
    read_jsonl,
    stable_priority,
    template_signature,
    unitize_exact_tokens,
    validate_annotation,
)


def _char_tokenization(response: str) -> dict:
    encoded = [1000 + ord(char) for char in response]
    return {
        "output_token_ids": [*encoded, 2],
        "encoded_token_ids": encoded,
        "offsets": [[index, index + 1] for index in range(len(response))],
        "trailing_token_decodes_to_empty": [True],
    }


@pytest.mark.parametrize(
    ("source", "reference", "response"),
    [
        ("gsm8k", "work\n#### 1/2", "Answer: \\boxed{0.5}."),
        ("asdiv-a", "9 (apples)", "The answer is \\boxed{9} apples."),
        ("asdiv-a", "50 (%)", "Answer: \\boxed{50\\%}."),
        ("gsm8k", "#### 2 1/2", "Answer: \\boxed{2.5}."),
    ],
)
def test_numeric_checker_matches_common_multisource_forms(source, reference, response):
    checked = check_numeric_response(
        response=response,
        raw_reference=reference,
        source=source,
    )

    assert checked["numeric_value_match"] == 1
    assert checked["correctness"] == 1
    assert checked["eligible_for_supervision"] is True
    assert checked["correctness_semantics"] == "numeric_value_match_v2"


def test_numeric_checker_fails_closed_on_conflicting_boxes_and_truncation():
    conflicting = check_numeric_response(
        response="First \\boxed{4}, but finally \\boxed{5}.",
        raw_reference="#### 5",
        source="gsm8k",
    )
    truncated = check_numeric_response(
        response="Answer: 5",
        raw_reference="#### 5",
        source="gsm8k",
        finish_reason="length",
    )

    assert conflicting["checker_status"] == "ambiguous_multiple_answers"
    assert conflicting["numeric_value_match"] is None
    assert conflicting["eligible_for_supervision"] is False
    assert truncated["checker_status"] == "truncated"
    assert truncated["eligible_for_supervision"] is False


def test_unitizer_regression_text_partitions_full_saved_axis():
    response = (
        "Step 1: Mr. Li has $1.50.\n\n"
        "2. For example, e.g. half is 0.5.\n"
        "Step 3: \\frac{1}{2}+\\frac{1}{2}=1.\n"
        "Answer: Thus x≤1 and the result is \\boxed{1}."
    )
    result = unitize_exact_tokens(response=response, **_char_tokenization(response))

    assert result["status"] == "ok"
    assert result["units"][0]["token_start"] == 0
    assert result["units"][-1]["token_end"] == len(response) + 1
    assert result["units"][-1]["kind"] == "non_claim"
    assert result["material_claim_count"] >= 4
    assert material_claim_char_spans(response)
    for left, right in zip(result["units"], result["units"][1:]):
        assert left["token_end"] == right["token_start"]


def test_unitizer_rejects_one_token_that_fuses_two_claims():
    response = "First claim. Second claim."
    with pytest.raises(ValueError, match="fuses"):
        unitize_exact_tokens(
            response=response,
            output_token_ids=[77],
            encoded_token_ids=[77],
            offsets=[[0, len(response)]],
            trailing_token_decodes_to_empty=[],
        )


def test_asdiv_loader_uses_official_fold_membership_and_parses_unit(tmp_path: Path):
    dataset = tmp_path / "dataset"
    folds = dataset / "nfolds" / "asdiv-a"
    folds.mkdir(parents=True)
    problems = []
    for index in range(5):
        problem_id = f"nluds-{index:04d}"
        problems.append(
            f"<Problem ID='{problem_id}'><Body>Body {index}.</Body>"
            f"<Question>Question {index}?</Question><Solution-Type>Addition</Solution-Type>"
            f"<Answer>{index + 1} (apples)</Answer><Formula>{index}+1={index + 1}</Formula></Problem>"
        )
        (folds / f"fold{index}.txt").write_text(problem_id + "\n", encoding="utf-8")
    xml_path = dataset / "ASDiv.xml"
    xml_path.write_text(
        "<?xml version='1.0'?><Machine-Reading-Corpus-File><ProblemSet>"
        + "".join(problems)
        + "</ProblemSet></Machine-Reading-Corpus-File>",
        encoding="utf-8",
    )
    import hashlib

    expected_hash = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    rows = load_asdiv_a_repository(
        tmp_path,
        expected_xml_sha256=expected_hash,
        expected_subset_size=5,
    )

    assert len(rows) == 5
    assert rows[0]["query_id"] == "asdiv-a:nluds-0000"
    assert rows[0]["source_unit"] == "apples"
    assert rows[0]["reference_answer"] == "1 (apples)"


def test_freeze_requires_decisions_and_is_hash_deterministic():
    rows = [
        {
            "source": "gsm8k" if index < 2 else "asdiv-a",
            "query_id": f"gsm8k:train:{index:05d}"
            if index < 2
            else f"asdiv-a:q{index}",
            "question": f"A distinct fixture question number {index} asks for a total.",
            "reference_answer": str(index + 1),
        }
        for index in range(4)
    ]
    candidates = near_duplicate_candidates(rows)
    decisions = [{**candidate, "decision": "distinct"} for candidate in candidates]
    first, first_report = freeze_query_pool(
        rows,
        source_counts={"gsm8k": 1, "asdiv-a": 1},
        near_duplicate_decisions=decisions,
    )
    second, second_report = freeze_query_pool(
        list(reversed(rows)),
        source_counts={"gsm8k": 1, "asdiv-a": 1},
        near_duplicate_decisions=decisions,
    )

    assert [row["query_id"] for row in first] == [row["query_id"] for row in second]
    assert (
        first_report["selected_query_ids_sha256"]
        == second_report["selected_query_ids_sha256"]
    )


def test_prior_exclusion_removes_the_entire_duplicate_cluster():
    rows = [
        {
            "source": "gsm8k",
            "query_id": "gsm8k:train:00000",
            "question": "Mina has two apples. How many apples does Mina have?",
            "reference_answer": "#### 2",
        },
        {
            "source": "gsm8k",
            "query_id": "gsm8k:train:00001",
            "question": "Mina has two apples. How many apples does Mina have?",
            "reference_answer": "#### 2",
        },
        {
            "source": "gsm8k",
            "query_id": "gsm8k:train:00002",
            "question": "A train travels ten miles. What distance did it travel?",
            "reference_answer": "#### 10",
        },
        {
            "source": "asdiv-a",
            "query_id": "asdiv-a:q3",
            "question": "Three birds sit in one tree. How many birds are there?",
            "reference_answer": "3 (birds)",
        },
    ]
    _, discovery = freeze_query_pool(
        rows,
        source_counts={"gsm8k": 0, "asdiv-a": 0},
        near_duplicate_decisions=[],
    )
    duplicate_cluster = next(
        cluster
        for cluster in discovery["clusters"]
        if len(cluster["member_query_ids"]) == 2
    )
    excluded_non_survivor = next(
        query_id
        for query_id in duplicate_cluster["member_query_ids"]
        if query_id != duplicate_cluster["survivor_query_id"]
    )
    selected, report = freeze_query_pool(
        rows,
        source_counts={"gsm8k": 1, "asdiv-a": 1},
        excluded_query_ids=[excluded_non_survivor],
        near_duplicate_decisions=[],
    )

    selected_ids = {row["query_id"] for row in selected}
    assert not selected_ids.intersection(duplicate_cluster["member_query_ids"])
    assert report["excluded_clusters"] == 1


def test_near_duplicate_decision_is_skipped_only_when_both_ends_are_excluded():
    rows = [
        {
            "source": "gsm8k",
            "query_id": "gsm8k:train:00000",
            "question": (
                "Mina has ten red apples and buys two more apples. "
                "How many red apples does Mina have in total?"
            ),
            "reference_answer": "#### 12",
        },
        {
            "source": "gsm8k",
            "query_id": "gsm8k:train:00001",
            "question": (
                "Today Mina has eleven red apples and buys three more apples. "
                "How many red apples does Mina have in total?"
            ),
            "reference_answer": "#### 14",
        },
    ]
    candidates = near_duplicate_candidates(rows, jaccard_threshold=0.70)
    assert len(candidates) == 1

    with pytest.raises(ValueError, match="lack frozen decisions"):
        freeze_query_pool(
            rows,
            source_counts={"gsm8k": 0},
            excluded_query_ids=[rows[0]["query_id"]],
            near_duplicate_decisions=[],
            jaccard_threshold=0.70,
        )

    selected, report = freeze_query_pool(
        rows,
        source_counts={"gsm8k": 0},
        excluded_query_ids=[row["query_id"] for row in rows],
        near_duplicate_decisions=[],
        jaccard_threshold=0.70,
    )
    assert selected == []
    assert report["near_duplicate_candidates"] == []
    assert report["near_duplicate_candidates_skipped_both_excluded"] == [
        candidates[0]["pair_id"]
    ]


def test_dedup_triage_hides_primary_answers_and_only_sends_unresolved(tmp_path: Path):
    candidates = [
        {
            "pair_id": f"pair-{index}",
            "left_query_id": f"left-{index}",
            "right_query_id": f"right-{index}",
            "left_question": "Question A",
            "right_question": "Question B",
            "decision": None,
        }
        for index in range(2)
    ]
    labels_a = [
        {
            "pair_id": "pair-0",
            "decision": "distinct",
            "confidence": "high",
            "rationale": "Different relations.",
        },
        {
            "pair_id": "pair-1",
            "decision": "duplicate",
            "confidence": "high",
            "rationale": "Same template.",
        },
    ]
    labels_b = [
        {
            "pair_id": "pair-0",
            "decision": "distinct",
            "confidence": "medium",
            "rationale": "Different structures.",
        },
        {
            "pair_id": "pair-1",
            "decision": "distinct",
            "confidence": "high",
            "rationale": "Different operation.",
        },
    ]
    paths = {
        "candidates": tmp_path / "candidates.jsonl",
        "labels_a": tmp_path / "labels_a.jsonl",
        "labels_b": tmp_path / "labels_b.jsonl",
        "output": tmp_path / "third.jsonl",
    }
    atomic_write_jsonl(paths["candidates"], candidates)
    atomic_write_jsonl(paths["labels_a"], labels_a)
    atomic_write_jsonl(paths["labels_b"], labels_b)
    command_dedup_triage(SimpleNamespace(**paths))

    third = read_jsonl(paths["output"])
    assert [row["pair_id"] for row in third] == ["pair-1"]
    assert third[0]["requires_independent_answer"] is True
    assert "annotation_a" not in third[0]
    assert "annotation_b" not in third[0]


def test_dedup_resolve_needs_no_third_model_when_all_primary_labels_agree(
    tmp_path: Path,
):
    candidate = {
        "pair_id": "pair-0",
        "left_query_id": "left-0",
        "right_query_id": "right-0",
        "left_question": "Question A",
        "right_question": "Question B",
        "decision": None,
    }
    label_a = {
        "pair_id": "pair-0",
        "decision": "distinct",
        "confidence": "high",
        "rationale": "Different relations.",
    }
    label_b = {
        "pair_id": "pair-0",
        "decision": "distinct",
        "confidence": "medium",
        "rationale": "Different structures.",
    }
    paths = {
        "candidates": tmp_path / "candidates.jsonl",
        "labels_a": tmp_path / "labels_a.jsonl",
        "labels_b": tmp_path / "labels_b.jsonl",
        "roster": tmp_path / "roster.json",
        "output": tmp_path / "decisions.jsonl",
    }
    atomic_write_jsonl(paths["candidates"], [candidate])
    atomic_write_jsonl(paths["labels_a"], [label_a])
    atomic_write_jsonl(paths["labels_b"], [label_b])
    atomic_write_json(
        paths["roster"],
        {
            "primary_annotators": [
                {
                    "provider": "provider-a",
                    "model_id": "model-a",
                    "model_family": "family-a",
                    "revision": "revision-a",
                },
                {
                    "provider": "provider-b",
                    "model_id": "model-b",
                    "model_family": "family-b",
                    "revision": "revision-b",
                },
            ]
        },
    )
    command_resolve_dedup(
        SimpleNamespace(
            **paths,
            adjudications=None,
        )
    )

    assert read_jsonl(paths["output"]) == [
        {
            "pair_id": "pair-0",
            "decision": "distinct",
            "label_source": "auto_agree",
        }
    ]
    report = json.loads(paths["output"].with_suffix(".report.json").read_text())
    assert report["third_model_item_count"] == 0
    assert report["third_model_label_count"] == 0


def test_jaccard_prefix_index_matches_brute_force_candidate_set():
    rows = []
    base = [
        "mina",
        "has",
        "red",
        "green",
        "apples",
        "and",
        "buys",
        "more",
        "find",
        "total",
    ]
    for index in range(40):
        tokens = list(base)
        if index % 3 == 0:
            tokens[-1] = "sum"
        if index % 5 == 0:
            tokens[-2] = "calculate"
        if index % 7 == 0:
            tokens[2] = "blue"
        tokens.extend(
            [str(index), f"unique{index}"] if index % 11 == 0 else [str(index)]
        )
        rows.append(
            {
                "source": "gsm8k",
                "query_id": f"gsm8k:train:{index:05d}",
                "question": " ".join(tokens),
                "reference_answer": f"#### {index}",
            }
        )
    threshold = 0.82
    expected = set()
    for left_index, left in enumerate(rows):
        left_tokens = set(template_signature(left["question"]).split())
        for right in rows[left_index + 1 :]:
            if normalize_question(left["question"]) == normalize_question(
                right["question"]
            ):
                continue
            right_tokens = set(template_signature(right["question"]).split())
            similarity = len(left_tokens & right_tokens) / len(
                left_tokens | right_tokens
            )
            if similarity >= threshold:
                left_id, right_id = sorted((left["query_id"], right["query_id"]))
                expected.add(
                    stable_priority("clir-near-duplicate-v2", left_id, right_id)
                )

    actual = {
        row["pair_id"]
        for row in near_duplicate_candidates(rows, jaccard_threshold=threshold)
    }
    assert actual == expected


def _natural_items(count: int = 10) -> dict[str, list[dict]]:
    consistency = []
    units = []
    for index in range(count):
        item_id = f"natural-{index}"
        consistency.append(
            {
                "item_id": item_id,
                "query_id": f"q-{index}",
                "problem": "2+3?",
                "left": {"id": f"{item_id}-l", "trajectory": "2+3=5", "units": []},
                "right": {
                    "id": f"{item_id}-r",
                    "trajectory": "Add to get 5",
                    "units": [],
                },
            }
        )
        trajectory_units = [
            {"unit_index": unit, "kind": "material_claim", "text": f"claim {unit}"}
            for unit in range(4)
        ]
        units.append(
            {
                "item_id": item_id,
                "query_id": f"q-{index}",
                "source": "gsm8k",
                "problem": "2+3?",
                "trajectory": "claims",
                "units": trajectory_units,
                "output_token_ids_sha256": "x",
            }
        )
    return {"consistency": consistency, "hallucination": units, "prior": units}


def test_blind_packages_hide_controls_and_measure_self_repeat():
    natural = _natural_items()
    packages, private = build_annotation_packages(natural)

    assert len(packages["a"]["consistency"]) == 13  # 10 natural +1 control +2 repeats
    assert len(packages["b"]["consistency"]) == 11
    assert all(
        "expected_annotation" not in item for item in packages["a"]["consistency"]
    )
    task_private = private["tasks"]["consistency"]

    labels_a = []
    labels_b = []
    expected_controls = {
        row["item_id"]: row["expected_annotation"] for row in task_private["controls"]
    }
    repeats = {
        row["repeat_item_id"]: row["original_item_id"]
        for row in task_private["self_repeats_a"]
    }
    natural_targets = {
        item["item_id"]: {
            "item_id": item["item_id"],
            "decision": "accept",
            "confidence": "high",
            "rationale": "fixture",
        }
        for item in natural["consistency"]
    }
    for slot, package, output in (
        ("a", packages["a"]["consistency"], labels_a),
        ("b", packages["b"]["consistency"], labels_b),
    ):
        by_item = {item["item_id"]: item for item in package}
        for item_id in by_item:
            if item_id in expected_controls:
                payload = expected_controls[item_id]
            elif item_id in repeats:
                payload = {**natural_targets[repeats[item_id]], "item_id": item_id}
            else:
                payload = natural_targets[item_id]
            output.append(validate_annotation("consistency", payload, by_item[item_id]))
    report = evaluate_package_reliability(
        task="consistency",
        private_task=task_private,
        labels_a=labels_a,
        labels_b=labels_b,
    )

    assert report["hidden_controls"]["a"]["accuracy"] == 1.0
    assert report["hidden_controls"]["b"]["accuracy"] == 1.0
    assert report["annotator_a_self_agreement"]["rate"] == 1.0
    assert annotation_signature("consistency", labels_a[0])


def test_vllm_outputs_are_restored_by_completion_index():
    outputs = [
        SimpleNamespace(index=2),
        SimpleNamespace(index=0),
        SimpleNamespace(index=1),
    ]
    request = SimpleNamespace(outputs=outputs)

    ordered = _ordered_vllm_candidates(request, 3)

    assert [row.index for row in ordered] == [0, 1, 2]


def test_third_model_is_independent_before_anonymous_adjudication(
    tmp_path: Path, capsys
):
    natural = _natural_items(10)
    packages, private = build_annotation_packages(natural)
    items_dir = tmp_path / "items"
    package_dir = tmp_path / "packages"
    labels_dirs = {slot: tmp_path / f"labels_{slot}" for slot in ("a", "b")}
    names = {
        "consistency": "annotation_consistency_natural.jsonl",
        "hallucination": "annotation_hallucination_natural.jsonl",
        "prior": "annotation_prior_natural.jsonl",
    }
    for task, name in names.items():
        atomic_write_jsonl(items_dir / name, natural[task])
        atomic_write_jsonl(
            package_dir / "annotator_a" / f"{task}.jsonl", packages["a"][task]
        )
        atomic_write_jsonl(
            package_dir / "annotator_b" / f"{task}.jsonl", packages["b"][task]
        )
    atomic_write_json(package_dir / "PRIVATE_package_manifest.json", private)

    natural_targets: dict[str, dict[str, dict]] = {"a": {}, "b": {}}
    for slot in ("a", "b"):
        for task in names:
            target_by_id = {}
            for item in natural[task]:
                if task == "consistency":
                    payload = {
                        "item_id": item["item_id"],
                        "decision": "accept",
                        "confidence": "high",
                        "rationale": "natural fixture",
                    }
                    if slot == "b" and item["item_id"] == "natural-0":
                        payload["decision"] = "reject"
                elif task == "hallucination":
                    payload = {
                        "item_id": item["item_id"],
                        "status": "clean",
                        "first_bad_unit_index": None,
                        "confidence": "high",
                        "rationale": "natural fixture",
                    }
                    if slot == "b" and item["item_id"] == "natural-0":
                        payload["status"] = "uncertain"
                else:
                    payload = {
                        "item_id": item["item_id"],
                        "eligibility": "usable",
                        "key_unit_indices": [3],
                        "complete_unit_indices": [0, 2, 3],
                        "confidence": "high",
                        "rationale": "natural fixture",
                    }
                    if slot == "b" and item["item_id"] == "natural-0":
                        payload["key_unit_indices"] = [2]
                target_by_id[item["item_id"]] = validate_annotation(task, payload, item)
            natural_targets[slot][task] = target_by_id

            task_private = private["tasks"][task]
            controls = {
                row["item_id"]: row["expected_annotation"]
                for row in task_private["controls"]
            }
            repeats = {
                row["repeat_item_id"]: row["original_item_id"]
                for row in task_private["self_repeats_a"]
            }
            package_items = packages[slot][task]
            labels = []
            for package_item in package_items:
                item_id = package_item["item_id"]
                if item_id in controls:
                    payload = controls[item_id]
                elif item_id in repeats:
                    payload = {**target_by_id[repeats[item_id]], "item_id": item_id}
                else:
                    payload = target_by_id[item_id]
                labels.append(validate_annotation(task, payload, package_item))
            atomic_write_jsonl(labels_dirs[slot] / f"{task}.jsonl", labels)

    triage_dir = tmp_path / "triage"
    command_triage(
        SimpleNamespace(
            items_dir=str(items_dir),
            package_dir=str(package_dir),
            labels_a_dir=str(labels_dirs["a"]),
            labels_b_dir=str(labels_dirs["b"]),
            output_dir=str(triage_dir),
        )
    )
    capsys.readouterr()
    third_labels_dir = tmp_path / "third_labels"
    for task in names:
        third_items = read_jsonl(triage_dir / "third_independent" / f"{task}.jsonl")
        labels = [
            {**natural_targets["a"][task][item["item_id"]]} for item in third_items
        ]
        atomic_write_jsonl(third_labels_dir / f"{task}.jsonl", labels)

    adjudication_dir = tmp_path / "adjudication_package"
    command_adjudication_package(
        SimpleNamespace(
            items_dir=str(items_dir),
            package_dir=str(package_dir),
            labels_a_dir=str(labels_dirs["a"]),
            labels_b_dir=str(labels_dirs["b"]),
            triage_dir=str(triage_dir),
            third_independent_labels_dir=str(third_labels_dir),
            output_dir=str(adjudication_dir),
        )
    )
    capsys.readouterr()

    triage_private = json.loads(
        (triage_dir / "PRIVATE_triage_manifest.json").read_text(encoding="utf-8")
    )
    assert triage_private["tasks"]["consistency"]["dispute_item_ids"] == ["natural-0"]
    public_packet = read_jsonl(adjudication_dir / "adjudicator" / "consistency.jsonl")
    assert len(public_packet) == 1
    assert "independent_annotation" in public_packet[0]
    assert {
        row["option"] for row in public_packet[0]["anonymous_primary_proposals"]
    } == {
        "option_1",
        "option_2",
    }
    assert "option_identity" not in public_packet[0]


def test_eight_query_fixture_is_deterministic(tmp_path: Path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    command_fixture(SimpleNamespace(output_dir=str(left)))
    capsys.readouterr()
    command_fixture(SimpleNamespace(output_dir=str(right)))
    capsys.readouterr()

    left_rows = read_jsonl(left / "pre_extraction.jsonl")
    right_rows = read_jsonl(right / "pre_extraction.jsonl")
    report = json.loads((left / "fixture_report.json").read_text(encoding="utf-8"))
    assert len(left_rows) == 64
    assert canonical_sha256(left_rows) == canonical_sha256(right_rows)
    assert report["status"] == "PASS_PIPELINE_FIXTURE"
    assert report["materialization"]["exact_contract_pass_rate"] == 1.0
    assert report["final_h_status"] == {"clean": 2, "hallucinated": 2}
