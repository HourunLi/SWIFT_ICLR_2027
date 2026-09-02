"""Mechanical-Key plus residual-binary CLIR Prior smoke-v17 contracts.

V16 established that two strong annotators can usually locate the final
answer-producing region, but disagree systematically about whether headings,
premise restatements, and generic formulas belong to the main trajectory.
V17 removes those decisions from the annotation task:

* a conservative compiler fixes one explicit answer-producing calculation as
  Key;
* obvious plans, restatements, duplicates, wrappers, and every post-Key block
  are fixed Complete negatives;
* annotators only decide whether each remaining pre-Key block is actually used
  by the candidate's route to the fixed Key (``used`` or ``not_used``).

The smoke is prospective and non-trainable.  A pass only authorizes a new,
separately frozen scale protocol on fresh query/template clusters.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from numbers import Integral
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_mechanical import (
    compile_reasoning_structure,
    expand_block_indices_to_units,
)
from src.clir_smoke import canonical_sha256, stable_priority


PROTOCOL_SCHEMA = "clir-prior-mechanical-key-binary-smoke-v17"
PROPOSAL_SCHEMA = "clir-prior-mechanical-key-binary-proposal-v17"
PACKAGE_SCHEMA = "clir-prior-mechanical-key-binary-package-v17"
STRUCTURE_SCHEMA = "clir-prior-mechanical-key-binary-structure-v17"
LABEL_SCHEMA = "clir-prior-mechanical-key-binary-label-v17"
PRIVATE_SCHEMA = "clir-prior-mechanical-key-binary-private-v17"
EVALUATION_SCHEMA = "clir-prior-mechanical-key-binary-evaluation-v17"
DEFAULT_NAMESPACE = "clir-prior-mechanical-key-binary-smoke-v17"
LABEL_NAME = "silver_dual_ai_mechanical_key_binary_prior_v17_no_human_verification"

DECISIONS = {"used", "not_used"}
CONFIDENCE_VALUES = {"high", "medium", "low"}

_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?"
)
_LATEX_SIMPLE_FRACTION = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"
    r"\s*\{\s*(\d[\d,]*(?:\.\d+)?)\s*\}"
)
_COMPARATOR = re.compile(r"(?:\\equiv|\\approx|\\simeq|≈|(?<![<>])=(?!=))")
_ARITHMETIC = re.compile(r"(?:\d|\})\s*(?:[+\-*/×÷^]|\\(?:times|cdot|div))")
_WRAPPER = re.compile(
    r"^\s*(?:so|therefore|thus|hence|in conclusion|final answer|the answer|"
    r"answer|所以|因此|故|最终答案)\b",
    flags=re.IGNORECASE,
)
_PLAN = re.compile(
    r"^\s*(?:(?:step|part)\s*[0-9a-z]+\s*[:.)-]?\s*)?"
    r"(?:to\s+|we\s+(?:need|have)\s+to\s+|(?:first|next|now|then|finally)\b[^\n]{0,24})?"
    r"(?:calculate|find|determine|solve|compute|evaluate|simplify|rewrite|"
    r"substitute|plug|convert|compare|check|add|subtract|multiply|divide|sum)\b",
    flags=re.IGNORECASE,
)
_FORMULA_CUE = re.compile(
    r"\b(?:formula|identity|theorem states|general form|we can use|remember that)\b|"
    r"公式|恒等式|定理",
    flags=re.IGNORECASE,
)


def _replace_simple_latex_fractions(text: str) -> str:
    current = text
    while True:
        updated = _LATEX_SIMPLE_FRACTION.sub(
            lambda match: f"{match.group(1)}/{match.group(2)}", current
        )
        if updated == current:
            return current
        current = updated


def _numeric_values(text: str) -> set[str]:
    normalized = _replace_simple_latex_fractions(text).replace("$", "")
    output: set[str] = set()
    for match in _NUMBER.findall(normalized):
        value = match.lstrip("+").replace(",", "")
        output.add(value)
        if "/" in value:
            output.update(part for part in value.split("/") if part)
    return output


def _candidate_answer_values(item: Mapping[str, Any]) -> set[str]:
    parsed = item.get("parsed_answer")
    if isinstance(parsed, str) and parsed.strip():
        values = _numeric_values(parsed)
        if values:
            return values
    response = str(item.get("response", ""))
    boxed = list(re.finditer(r"\\boxed\s*\{([^{}]*)\}", response))
    return _numeric_values(boxed[-1].group(1)) if boxed else set()


def _length_band(material_claim_count: int) -> str | None:
    if 8 <= material_claim_count <= 18:
        return "medium"
    if 19 <= material_claim_count <= 40:
        return "long"
    return None


def _material_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = row.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ValueError("row units must be an array")
    output: list[dict[str, Any]] = []
    for raw in units:
        if not isinstance(raw, Mapping) or raw.get("kind") != "material_claim":
            continue
        index = raw.get("unit_index")
        text = raw.get("text")
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("invalid material unit")
        output.append(
            {"unit_index": int(index), "kind": "material_claim", "text": text}
        )
    output.sort(key=lambda unit: unit["unit_index"])
    if len({unit["unit_index"] for unit in output}) != len(output):
        raise ValueError("material unit indices are duplicated")
    return output


def _looks_like_answer_calculation(text: str, answer_values: set[str]) -> bool:
    values = _numeric_values(text)
    return bool(
        answer_values
        and answer_values <= values
        and _COMPARATOR.search(text)
        and not ("\\boxed" in text and not _ARITHMETIC.search(text))
    )


def _safe_answer_suffix(text: str, answer_values: set[str]) -> bool:
    """Whether a post-Key block cannot introduce another calculation."""

    values = _numeric_values(text)
    if _COMPARATOR.search(text) or _ARITHMETIC.search(text):
        return False
    if values and not values <= answer_values:
        return False
    return True


def _mechanical_non_main_reason(block: Mapping[str, Any]) -> str | None:
    hint = str(block.get("role_hint", ""))
    text = str(block.get("text", ""))
    if hint == "exact_duplicate":
        return "exact_duplicate"
    if hint == "possible_answer_wrapper":
        return "answer_wrapper"
    if hint == "possible_premise_restatement":
        return "premise_restatement_without_new_calculation"
    if hint == "possible_plan_or_heading":
        return "plan_or_heading_without_result"
    if not _COMPARATOR.search(text) and _PLAN.search(text):
        return "instruction_only_plan"
    if (
        hint == "possible_formula_only"
        and _FORMULA_CUE.search(text)
        and not _numeric_values(text)
    ):
        return "generic_formula_without_instantiation"
    return None


def compile_binary_structure_v17(item: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the fixed Key/non-main partition and residual audit list."""

    base = compile_reasoning_structure(item)
    blocks = list(base["blocks"])
    answer_values = _candidate_answer_values(item)
    if not answer_values:
        raise ValueError("candidate answer has no numeric value")

    answer_candidates = [
        int(block["block_id"])
        for block in blocks
        if block.get("role_hint") != "exact_duplicate"
        and _looks_like_answer_calculation(str(block["text"]), answer_values)
    ]
    if not answer_candidates:
        raise ValueError("no explicit answer-producing calculation")

    key_block_id: int | None = None
    for candidate in reversed(answer_candidates):
        text = str(blocks[candidate]["text"])
        earlier = [value for value in answer_candidates if value < candidate]
        if _WRAPPER.search(text) and earlier:
            continue
        if all(
            _mechanical_non_main_reason(block) is not None
            or _safe_answer_suffix(str(block["text"]), answer_values)
            for block in blocks[candidate + 1 :]
        ):
            key_block_id = candidate
            break
    if key_block_id is None:
        raise ValueError("answer-producing calculation has unsafe suffix")

    public_blocks: list[dict[str, Any]] = []
    fixed_non_main: list[int] = []
    residual: list[int] = []
    for block in blocks:
        block_id = int(block["block_id"])
        if block_id == key_block_id:
            status = "fixed_key"
            reason = "last_safe_explicit_calculation_containing_candidate_answer"
        elif block_id > key_block_id:
            status = "fixed_non_main"
            reason = "post_key_answer_suffix"
            fixed_non_main.append(block_id)
        else:
            non_main_reason = _mechanical_non_main_reason(block)
            if non_main_reason is not None:
                status = "fixed_non_main"
                reason = non_main_reason
                fixed_non_main.append(block_id)
            else:
                status = "residual_ai_decision"
                reason = "trace_backward_from_fixed_key"
                residual.append(block_id)
        public_blocks.append(
            {
                "block_id": block_id,
                "unit_indices": [int(value) for value in block["unit_indices"]],
                "text": str(block["text"]),
                "mechanical_status": status,
                "mechanical_reason": reason,
            }
        )

    if not residual:
        raise ValueError("no residual block")
    if not fixed_non_main:
        raise ValueError("no mechanically fixed Complete negative")
    partition = sorted([key_block_id, *fixed_non_main, *residual])
    if partition != list(range(len(blocks))):
        raise AssertionError("v17 block partition drift")
    return {
        "schema_version": STRUCTURE_SCHEMA,
        "material_unit_count": int(base["material_unit_count"]),
        "block_count": len(blocks),
        "blocks": public_blocks,
        "unit_to_block": deepcopy(base["unit_to_block"]),
        "key_block_id": key_block_id,
        "key_unit_indices": expand_block_indices_to_units(base, [key_block_id]),
        "fixed_non_main_block_ids": fixed_non_main,
        "residual_block_ids": residual,
        "target_derivation": {
            "key": "mechanical fixed_key block",
            "complete_positive": "fixed_key plus residual blocks both annotators mark used",
            "complete_negative": "fixed_non_main plus residual blocks both mark not_used",
            "complete_masked": "residual blocks with annotator disagreement",
        },
    }


def public_package_item_v17(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    output_id = str(item_id or source["item_id"])
    raw = {
        "item_id": output_id,
        "question": str(source["question"]),
        "response": str(source["response"]),
        "parsed_answer": str(source.get("parsed_answer", "")),
        "units": deepcopy(list(source["units"])),
    }
    return {
        "schema_version": PACKAGE_SCHEMA,
        "item_id": output_id,
        "question": raw["question"],
        "response": raw["response"],
        "units": raw["units"],
        "structure": compile_binary_structure_v17(raw),
    }


def _eligible_source(row: Mapping[str, Any]) -> bool:
    count = row.get("material_claim_count")
    return bool(
        isinstance(count, Integral)
        and not isinstance(count, bool)
        and _length_band(int(count)) is not None
        and row.get("acquisition_split") == "train"
        and row.get("source") in {"gsm8k", "math"}
        and row.get("checker_status") in {"numeric_match", "numeric_mismatch"}
        and row.get("eligible_for_supervision") is True
        and row.get("unitization_status") == "ok"
        and row.get("status") == "ok"
        and row.get("finish_reason") == "stop"
    )


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    count = int(row["material_claim_count"])
    band = str(row.get("length_band") or _length_band(count))
    return str(row["source"]), str(row["checker_status"]), band


def _stratum_name(value: Sequence[str]) -> str:
    return "|".join(value)


def select_fresh_rows_v17(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: Iterable[str],
    excluded_cluster_ids: Iterable[str],
    strata: Sequence[Mapping[str, Any]],
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one compilable, fresh trajectory per query and cluster."""

    quotas = {
        (str(row["source"]), str(row["checker_status"]), str(row["length_band"])): int(
            row["count"]
        )
        for row in strata
    }
    if len(quotas) != len(strata) or any(value <= 0 for value in quotas.values()):
        raise ValueError("v17 strata must be unique and positive")
    excluded_queries = {str(value) for value in excluded_query_ids}
    excluded_clusters = {str(value) for value in excluded_cluster_ids}
    rejection: Counter[str] = Counter()
    by_stratum_query: dict[
        tuple[str, str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))

    for raw in rows:
        if not _eligible_source(raw):
            rejection["base_eligibility"] += 1
            continue
        query_id = str(raw.get("query_id", ""))
        cluster_id = str(raw.get("cluster_id", ""))
        if not query_id or not cluster_id:
            rejection["missing_identity"] += 1
            continue
        if query_id in excluded_queries or cluster_id in excluded_clusters:
            rejection["historically_used_query_or_cluster"] += 1
            continue
        stratum = _stratum(raw)
        if stratum not in quotas:
            rejection["outside_frozen_strata"] += 1
            continue
        material = _material_units(raw)
        if len(material) != int(raw["material_claim_count"]):
            raise ValueError(f"{raw.get('id')}: material claim count drift")
        candidate = {
            "schema_version": PROPOSAL_SCHEMA,
            "item_id": "prior-v17-natural-"
            + stable_priority(f"{namespace}:item", str(raw["id"]))[:24],
            "source_row_id": str(raw["id"]),
            "query_id": query_id,
            "cluster_id": cluster_id,
            "source": str(raw["source"]),
            "checker_status": str(raw["checker_status"]),
            "length_band": stratum[2],
            "prior_label_split": str(raw.get("prior_label_split", "train")),
            "candidate_index": int(raw["candidate_index"]),
            "source_record_id": raw.get("source_record_id"),
            "material_claim_count": len(material),
            "selection_priority": stable_priority(
                f"{namespace}:selection", str(raw["id"])
            ),
            "question": str(raw["question"]),
            "response": str(raw["response"]),
            "parsed_answer": str(raw.get("parsed_answer", "")),
            "units": material,
        }
        try:
            structure = compile_binary_structure_v17(candidate)
        except ValueError as exc:
            rejection[f"compiler:{exc}"] += 1
            continue
        if len(structure["residual_block_ids"]) < 2:
            rejection["compiler:fewer than two residual blocks"] += 1
            continue
        candidate["mechanical_key_block_id"] = int(structure["key_block_id"])
        candidate["residual_block_count"] = len(structure["residual_block_ids"])
        candidate["fixed_non_main_block_count"] = len(
            structure["fixed_non_main_block_ids"]
        )
        by_stratum_query[stratum][query_id].append(candidate)

    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for stratum, by_query in by_stratum_query.items():
        representatives = [
            min(
                values,
                key=lambda row: stable_priority(
                    f"{namespace}:candidate", row["query_id"], row["source_row_id"]
                ),
            )
            for values in by_query.values()
        ]
        candidates[stratum] = sorted(
            representatives,
            key=lambda row: stable_priority(
                f"{namespace}:query", *stratum, row["query_id"], row["source_row_id"]
            ),
        )

    protocol_order = [
        (str(row["source"]), str(row["checker_status"]), str(row["length_band"]))
        for row in strata
    ]
    ordered_strata = sorted(
        quotas,
        key=lambda value: (len(candidates.get(value, [])), protocol_order.index(value)),
    )
    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    for stratum in ordered_strata:
        chosen = 0
        for row in candidates.get(stratum, []):
            if row["query_id"] in used_queries or row["cluster_id"] in used_clusters:
                continue
            selected.append(row)
            used_queries.add(row["query_id"])
            used_clusters.add(row["cluster_id"])
            chosen += 1
            if chosen == quotas[stratum]:
                break
        if chosen != quotas[stratum]:
            raise ValueError(
                f"insufficient v17 capacity for {_stratum_name(stratum)}: "
                f"{chosen}/{quotas[stratum]}"
            )

    selected.sort(key=lambda row: str(row["selection_priority"]))
    expected = sum(quotas.values())
    if len(selected) != expected or len(used_queries) != expected or len(used_clusters) != expected:
        raise AssertionError("v17 selection distinctness or count drift")
    selected_counts = Counter(_stratum(row) for row in selected)
    return selected, {
        "namespace": namespace,
        "selected": len(selected),
        "distinct_queries": len(used_queries),
        "distinct_clusters": len(used_clusters),
        "excluded_query_count": len(excluded_queries),
        "excluded_cluster_count": len(excluded_clusters),
        "available_query_counts": {
            _stratum_name(value): len(candidates.get(value, []))
            for value in sorted(quotas)
        },
        "selected_by_stratum": {
            _stratum_name(value): selected_counts[value] for value in sorted(quotas)
        },
        "mean_residual_blocks": mean(row["residual_block_count"] for row in selected),
        "mean_fixed_non_main_blocks": mean(
            row["fixed_non_main_block_count"] for row in selected
        ),
        "rejection_counts": dict(sorted(rejection.items())),
        "ordered_item_ids_sha256": canonical_sha256(
            [row["item_id"] for row in selected]
        ),
    }


def validate_binary_annotation_v17(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    expected_fields = {"item_id", "residual_decisions", "confidence", "rationale"}
    if set(annotation) != expected_fields:
        raise ValueError("v17 label fields differ from the strict schema")
    if annotation.get("item_id") != item.get("item_id"):
        raise ValueError("v17 annotation item_id does not match package item")
    if item.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("v17 package schema drift")
    structure = item.get("structure")
    if not isinstance(structure, Mapping) or structure.get("schema_version") != STRUCTURE_SCHEMA:
        raise ValueError("v17 structure schema drift")
    confidence = annotation.get("confidence")
    rationale = annotation.get("rationale")
    decisions = annotation.get("residual_decisions")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("invalid v17 confidence")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("v17 rationale must be non-empty")
    if not isinstance(decisions, list):
        raise ValueError("v17 residual_decisions must be an array")
    normalized: list[dict[str, Any]] = []
    for raw in decisions:
        if not isinstance(raw, Mapping) or set(raw) != {"block_id", "decision"}:
            raise ValueError("v17 residual decision fields are invalid")
        block_id = raw.get("block_id")
        decision = raw.get("decision")
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, Integral)
            or decision not in DECISIONS
        ):
            raise ValueError("v17 residual decision value is invalid")
        normalized.append({"block_id": int(block_id), "decision": str(decision)})
    if [row["block_id"] for row in normalized] != list(structure["residual_block_ids"]):
        raise ValueError("v17 must decide every residual block exactly once in order")
    used = [
        int(structure["key_block_id"]),
        *(row["block_id"] for row in normalized if row["decision"] == "used"),
    ]
    not_used = [
        *map(int, structure["fixed_non_main_block_ids"]),
        *(row["block_id"] for row in normalized if row["decision"] == "not_used"),
    ]
    return {
        "schema_version": LABEL_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "residual_decisions": normalized,
        "complete_block_indices": sorted(used),
        "complete_negative_block_indices": sorted(not_used),
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


def binary_target_signature(annotation: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    return tuple(
        (int(row["block_id"]), str(row["decision"]))
        for row in annotation["residual_decisions"]
    )


def _control_raw(
    annotator: str,
    index: int,
    *,
    question: str,
    texts: Sequence[str],
    parsed_answer: str,
) -> dict[str, Any]:
    return {
        "item_id": f"prior-v17-control-{annotator}-{index:02d}",
        "question": question,
        "response": "\n".join(texts),
        "parsed_answer": parsed_answer,
        "units": [
            {"unit_index": 2 * block_id, "kind": "material_claim", "text": text}
            for block_id, text in enumerate(texts)
        ],
    }


def build_hidden_controls_v17(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    if annotator not in {"a", "b"}:
        raise ValueError("annotator must be a or b")
    specs = [
        (
            "generic_formula_not_used",
            "A rectangle is 12 meters long and 5 meters wide. Find its area.",
            [
                "To calculate the area, multiply length by width.",
                "Area = length * width.",
                "12 * 5 = 60 square meters.",
                "The answer is \\boxed{60}.",
            ],
            "60",
            {1: "not_used"},
        ),
        (
            "used_variable_chain",
            "A number is three more than 8, then multiplied by 4. Find the result.",
            [
                "Let x = 8 + 3.",
                "x = 11.",
                "4 * x = 44.",
                "The answer is \\boxed{44}.",
            ],
            "44",
            {0: "used", 1: "used"},
        ),
        (
            "unused_side_branch",
            "Four packs hold six cards each and two packs hold five cards each. Find the total.",
            [
                "4 * 6 = 24 cards.",
                "2 * 5 = 10 cards.",
                "9 * 9 = 81.",
                "24 + 10 = 34 cards.",
                "Final answer: \\boxed{34}.",
            ],
            "34",
            {0: "used", 1: "used", 2: "not_used"},
        ),
        (
            "wrong_but_used",
            "Five boxes hold seven items each and three are added. Find the total.",
            [
                "5 * 7 = 35 items.",
                "35 + 3 = 39 items.",
                "The answer is \\boxed{39}.",
            ],
            "39",
            {0: "used"},
        ),
        (
            "unused_early_guess",
            "Mia has 18 beads and buys 7 more. How many beads now?",
            [
                "A quick guess is 30 beads.",
                "Mia starts with 18 beads.",
                "She buys 7 additional beads.",
                "18 + 7 = 25 beads.",
                "Therefore \\boxed{25}.",
            ],
            "25",
            {0: "not_used", 1: "used", 2: "used"},
        ),
        (
            "used_conversion_fact",
            "A movie lasts 2 hours. How many minutes is that?",
            [
                "One hour equals 60 minutes.",
                "The movie lasts 2 hours.",
                "2 * 60 = 120 minutes.",
                "The answer is \\boxed{120}.",
            ],
            "120",
            {0: "used"},
        ),
        (
            "two_branches_used",
            "A shop sells 3 pens at 4 dollars and 2 books at 5 dollars. Find total revenue.",
            [
                "3 * 4 = 12 dollars from pens.",
                "2 * 5 = 10 dollars from books.",
                "12 + 10 = 22 dollars.",
                "Final answer: \\boxed{22}.",
            ],
            "22",
            {0: "used", 1: "used"},
        ),
        (
            "unused_check",
            "Six groups have 8 students each. Find the total.",
            [
                "6 * 8 = 48 students.",
                "As an unrelated check, 5 * 5 = 25.",
                "48 + 0 = 48 students.",
                "The answer is \\boxed{48}.",
            ],
            "48",
            {0: "used", 1: "not_used"},
        ),
        (
            "used_equation_setup",
            "A number plus 7 equals 19. Find the number.",
            [
                "Let n be the unknown number.",
                "n + 7 = 19.",
                "n = 19 - 7.",
                "n = 12.",
                "Thus \\boxed{12}.",
            ],
            "12",
            {0: "used", 1: "used", 2: "used"},
        ),
        (
            "unused_alternative",
            "There are 9 rows with 3 chairs each. Find the chair count.",
            [
                "One possible unrelated sum is 9 + 3 = 12.",
                "There are 9 equal rows.",
                "Each row has 3 chairs.",
                "9 * 3 = 27 chairs.",
                "Answer: \\boxed{27}.",
            ],
            "27",
            {0: "not_used", 1: "used", 2: "used"},
        ),
        (
            "used_case_choice",
            "Choose the positive solution of x squared equals 49.",
            [
                "x^2 = 49.",
                "The two candidates are x = 7 and x = -7.",
                "The question requires the positive candidate.",
                "x = 7.",
                "Final answer: \\boxed{7}.",
            ],
            "7",
            {0: "used", 1: "used", 2: "used"},
        ),
        (
            "duplicate_and_plan_removed",
            "A box has 14 red and 6 blue balls. Find the total.",
            [
                "First calculate the total number of balls.",
                "There are 14 red balls.",
                "There are 6 blue balls.",
                "14 + 6 = 20 balls.",
                "14 + 6 = 20 balls.",
                "The answer is \\boxed{20}.",
            ],
            "20",
            {1: "used", 2: "used"},
        ),
    ]
    output: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (name, question, texts, answer, expected_by_block) in enumerate(specs):
        raw = _control_raw(
            annotator,
            index,
            question=question,
            texts=texts,
            parsed_answer=answer,
        )
        item = public_package_item_v17(raw)
        residual = list(item["structure"]["residual_block_ids"])
        if set(residual) != set(expected_by_block):
            raise AssertionError(
                f"v17 control {name} residual drift: {residual} != {sorted(expected_by_block)}"
            )
        expected = validate_binary_annotation_v17(
            {
                "item_id": item["item_id"],
                "residual_decisions": [
                    {"block_id": block_id, "decision": expected_by_block[block_id]}
                    for block_id in residual
                ],
                "confidence": "high",
                "rationale": f"hidden control target: {name}",
            },
            item,
        )
        output.append((item, expected, name))
    return output


def build_blind_shards_v17(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 6,
    natural_per_shard: int = 16,
    controls_per_shard: int = 2,
    repeats_per_shard: int = 4,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    expected_natural = shard_count * natural_per_shard
    if (
        shard_count != 6
        or natural_per_shard != 16
        or controls_per_shard != 2
        or repeats_per_shard != 4
        or len(proposals) != expected_natural
    ):
        raise ValueError("v17 freezes 96 natural rows in six 22-row shards")
    ordered = sorted(
        (dict(row) for row in proposals),
        key=lambda row: stable_priority(f"{namespace}:natural-shard", row["item_id"]),
    )
    natural_shards = [
        [public_package_item_v17(row) for row in ordered[index::shard_count]]
        for index in range(shard_count)
    ]
    if any(len(rows) != natural_per_shard for rows in natural_shards):
        raise AssertionError("v17 natural shard balance drift")

    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        shard_rows = [list(rows) for rows in natural_shards]
        controls = build_hidden_controls_v17(annotator)
        for control_index, (item, expected, name) in enumerate(controls):
            shard_index = control_index // controls_per_shard
            shard_rows[shard_index].append(item)
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "shard_index": shard_index,
                    "kind": "control",
                    "item_id": item["item_id"],
                    "natural_item_id": None,
                    "control_name": name,
                    "expected_label": expected,
                }
            )
        for parent_shard, parents in enumerate(natural_shards):
            destination = (parent_shard + 1) % shard_count
            for local_index, parent in enumerate(parents[:repeats_per_shard]):
                repeat_id = (
                    f"{parent['item_id']}:repeat:{annotator}:"
                    f"{parent_shard:02d}:{local_index:02d}"
                )
                repeated = deepcopy(parent)
                repeated["item_id"] = repeat_id
                shard_rows[destination].append(repeated)
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": destination,
                        "kind": "repeat",
                        "item_id": repeat_id,
                        "natural_item_id": parent["item_id"],
                        "control_name": None,
                        "expected_label": None,
                    }
                )
        for shard_index, rows in enumerate(shard_rows):
            for natural in natural_shards[shard_index]:
                private.append(
                    {
                        "schema_version": PRIVATE_SCHEMA,
                        "annotator": annotator,
                        "shard_index": shard_index,
                        "kind": "natural",
                        "item_id": natural["item_id"],
                        "natural_item_id": natural["item_id"],
                        "control_name": None,
                        "expected_label": None,
                    }
                )
            if len(rows) != natural_per_shard + controls_per_shard + repeats_per_shard:
                raise AssertionError("v17 public shard composition drift")
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}:package:{annotator}:{shard_index}", row["item_id"]
                )
            )
        packages[annotator] = shard_rows
    private.sort(key=lambda row: (row["annotator"], row["shard_index"], row["item_id"]))
    return packages, private, {
        "annotators": ["a", "b"],
        "shards_per_annotator": shard_count,
        "natural_rows_per_shard": natural_per_shard,
        "controls_per_shard": controls_per_shard,
        "repeats_per_shard": repeats_per_shard,
        "rows_per_shard": natural_per_shard + controls_per_shard + repeats_per_shard,
        "natural_rows_per_annotator": expected_natural,
        "controls_per_annotator": shard_count * controls_per_shard,
        "repeats_per_annotator": shard_count * repeats_per_shard,
        "rows_per_annotator": expected_natural
        + shard_count * (controls_per_shard + repeats_per_shard),
        "ai_decisions": "binary_used_or_not_used_only_on_residual_pre_key_blocks",
        "ai_outputs_key_complete_path_or_roles": False,
        "trainable": False,
    }


def _normalize_population(
    *,
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    normalized: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    package_maps: dict[str, dict[str, dict[str, Any]]] = {}
    private_maps: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for raw in private_index:
        row = dict(raw)
        annotator = str(row.get("annotator"))
        item_id = str(row.get("item_id", ""))
        if (
            annotator not in private_maps
            or row.get("schema_version") != PRIVATE_SCHEMA
            or not item_id
            or item_id in private_maps[annotator]
        ):
            raise ValueError("v17 private index schema, annotator, or ID drift")
        private_maps[annotator][item_id] = row
    for annotator in ("a", "b"):
        package_map: dict[str, dict[str, Any]] = {}
        for raw in packages[annotator]:
            row = dict(raw)
            item_id = str(row.get("item_id", ""))
            if (
                row.get("schema_version") != PACKAGE_SCHEMA
                or not item_id
                or item_id in package_map
            ):
                raise ValueError(f"v17 package {annotator} schema or ID drift")
            package_map[item_id] = row
        label_map = {str(row.get("item_id")): dict(row) for row in labels[annotator]}
        if len(label_map) != len(labels[annotator]) or set(label_map) != set(package_map):
            raise ValueError(f"v17 annotator {annotator} label/package mismatch")
        if set(private_maps[annotator]) != set(package_map):
            raise ValueError(f"v17 annotator {annotator} package/private mismatch")
        package_maps[annotator] = package_map
        for item_id, item in package_map.items():
            normalized[annotator][item_id] = validate_binary_annotation_v17(
                label_map[item_id], item
            )
    return normalized, package_maps, private_maps


def _set_iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _binary_kappa(counts: Counter[tuple[str, str]]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    observed = (counts[("used", "used")] + counts[("not_used", "not_used")]) / total
    left_used = (counts[("used", "used")] + counts[("used", "not_used")]) / total
    right_used = (counts[("used", "used")] + counts[("not_used", "used")]) / total
    expected = left_used * right_used + (1 - left_used) * (1 - right_used)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def evaluate_binary_smoke_v17(
    *,
    proposals: Sequence[Mapping[str, Any]],
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        normalized, package_maps, private_maps = _normalize_population(
            packages=packages, private_index=private_index, labels=labels
        )
    except ValueError as exc:
        return {
            "schema_version": EVALUATION_SCHEMA,
            "status": "FAIL_PRIOR_V17_BINARY_SCHEMA",
            "schema_errors": [str(exc)],
            "trainable_labels_published": False,
            "next_step": "stop_v17_without_relabel_or_subset_salvage",
        }

    proposal_ids = {str(row["item_id"]) for row in proposals}
    if len(proposal_ids) != len(proposals):
        raise ValueError("v17 proposal IDs are duplicated")
    natural_ids = {
        annotator: {
            item_id
            for item_id, row in private_maps[annotator].items()
            if row["kind"] == "natural"
        }
        for annotator in ("a", "b")
    }
    if natural_ids["a"] != natural_ids["b"] or natural_ids["a"] != proposal_ids:
        raise ValueError("v17 proposal/A/B natural populations differ")

    controls_report: dict[str, Any] = {}
    repeats_report: dict[str, Any] = {}
    for annotator in ("a", "b"):
        controls = []
        for row in private_maps[annotator].values():
            if row["kind"] != "control":
                continue
            passed = binary_target_signature(
                normalized[annotator][row["item_id"]]
            ) == binary_target_signature(row["expected_label"])
            controls.append(
                {"item_id": row["item_id"], "name": row["control_name"], "pass": passed}
            )
        controls_report[annotator] = {
            "passed": sum(row["pass"] for row in controls),
            "total": len(controls),
            "items": controls,
        }
        repeats = []
        for row in private_maps[annotator].values():
            if row["kind"] != "repeat":
                continue
            passed = binary_target_signature(
                normalized[annotator][row["item_id"]]
            ) == binary_target_signature(
                normalized[annotator][row["natural_item_id"]]
            )
            repeats.append(
                {
                    "item_id": row["item_id"],
                    "natural_item_id": row["natural_item_id"],
                    "pass": passed,
                }
            )
        repeats_report[annotator] = {
            "exact": sum(row["pass"] for row in repeats),
            "total": len(repeats),
            "exact_rate": sum(row["pass"] for row in repeats) / max(1, len(repeats)),
        }

    decision_counts: Counter[tuple[str, str]] = Counter()
    row_exact = 0
    unit_coverage: list[float] = []
    complete_iou: list[float] = []
    all_material_union = 0
    low_confidence = Counter()
    residual_total = 0
    for item_id in sorted(proposal_ids):
        item = package_maps["a"][item_id]
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        if left["confidence"] == "low":
            low_confidence["a"] += 1
        if right["confidence"] == "low":
            low_confidence["b"] += 1
        left_map = {row["block_id"]: row["decision"] for row in left["residual_decisions"]}
        right_map = {row["block_id"]: row["decision"] for row in right["residual_decisions"]}
        pairs = [(left_map[key], right_map[key]) for key in left_map]
        decision_counts.update(pairs)
        residual_total += len(pairs)
        row_exact += left_map == right_map

        structure = item["structure"]
        disagreement_blocks = [
            block_id for block_id in left_map if left_map[block_id] != right_map[block_id]
        ]
        disagreement_units = expand_block_indices_to_units(structure, disagreement_blocks)
        unit_coverage.append(
            1 - len(disagreement_units) / max(1, int(structure["material_unit_count"]))
        )
        left_complete = expand_block_indices_to_units(
            structure, left["complete_block_indices"]
        )
        right_complete = expand_block_indices_to_units(
            structure, right["complete_block_indices"]
        )
        complete_iou.append(_set_iou(left_complete, right_complete))
        all_units = {
            int(unit)
            for block in structure["blocks"]
            for unit in block["unit_indices"]
        }
        all_material_union += set(left_complete) | set(right_complete) == all_units

    agreement = (
        decision_counts[("used", "used")] + decision_counts[("not_used", "not_used")]
    ) / max(1, residual_total)
    cross = {
        "natural_rows": len(proposal_ids),
        "residual_decisions": residual_total,
        "residual_decision_counts": {
            f"a_{left}__b_{right}": decision_counts[(left, right)]
            for left in sorted(DECISIONS)
            for right in sorted(DECISIONS)
        },
        "residual_decision_agreement": agreement,
        "residual_decision_kappa": _binary_kappa(decision_counts),
        "row_exact": row_exact,
        "row_exact_rate": row_exact / max(1, len(proposal_ids)),
        "complete_macro_unit_iou": mean(complete_iou) if complete_iou else 0.0,
        "complete_unit_mask_coverage": mean(unit_coverage) if unit_coverage else 0.0,
        "all_material_union_rate": all_material_union / max(1, len(proposal_ids)),
        "low_confidence_rows": dict(low_confidence),
        "low_confidence_rate": {
            side: low_confidence[side] / max(1, len(proposal_ids)) for side in ("a", "b")
        },
        "both_used_rate": decision_counts[("used", "used")] / max(1, residual_total),
        "both_not_used_rate": decision_counts[("not_used", "not_used")]
        / max(1, residual_total),
        "disagreement_rate": 1 - agreement,
    }
    checks = {
        "controls_a": controls_report["a"]["passed"] >= int(gates["controls_min_pass"]),
        "controls_b": controls_report["b"]["passed"] >= int(gates["controls_min_pass"]),
        "self_repeat_a": repeats_report["a"]["exact_rate"]
        >= float(gates["self_repeat_exact_min"]),
        "self_repeat_b": repeats_report["b"]["exact_rate"]
        >= float(gates["self_repeat_exact_min"]),
        "natural_population": len(proposal_ids) == int(gates["natural_rows_required"]),
        "low_confidence_a": cross["low_confidence_rate"]["a"]
        <= float(gates["low_confidence_rate_max"]),
        "low_confidence_b": cross["low_confidence_rate"]["b"]
        <= float(gates["low_confidence_rate_max"]),
        "residual_agreement": agreement >= float(gates["residual_agreement_min"]),
        "residual_kappa": cross["residual_decision_kappa"]
        >= float(gates["residual_kappa_min"]),
        "row_exact": cross["row_exact_rate"] >= float(gates["row_exact_min"]),
        "complete_iou": cross["complete_macro_unit_iou"]
        >= float(gates["complete_unit_iou_min"]),
        "complete_coverage": cross["complete_unit_mask_coverage"]
        >= float(gates["complete_unit_mask_coverage_min"]),
        "both_used_support": cross["both_used_rate"]
        >= float(gates["both_used_rate_min"]),
        "both_not_used_support": cross["both_not_used_rate"]
        >= float(gates["both_not_used_rate_min"]),
        "nondegenerate_complete": cross["all_material_union_rate"]
        <= float(gates["all_material_union_rate_max"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": (
            "PASS_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE"
            if passed
            else "STOP_PRIOR_V17_MECHANICAL_KEY_BINARY_SMOKE"
        ),
        "schema_errors": [],
        "controls": controls_report,
        "self_repeats": repeats_report,
        "cross_annotator_natural": cross,
        "gates": dict(gates),
        "gate_checks": checks,
        "target_definition": {
            "key": "single mechanically fixed answer-producing calculation block",
            "complete_positive": "fixed Key plus residual blocks both annotators mark used",
            "complete_negative": "mechanically fixed non-main plus residual blocks both mark not_used",
            "complete_masked": "only residual used/not_used disagreements",
            "dependency_edges_used": False,
            "path_or_error_localization_used": False,
        },
        "trainable_labels_published": False,
        "next_step": (
            "freeze_fresh_scale_v18_before_any_new_label"
            if passed
            else "stop_v17_without_relabel_or_subset_salvage_then_consider_v16_posthoc_replay"
        ),
    }
