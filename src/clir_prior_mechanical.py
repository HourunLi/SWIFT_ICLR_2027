"""Mechanical reasoning-block prototype for a future CLIR Prior protocol.

This module deliberately does not publish labels.  It compiles the immutable
material-claim units into a smaller reasoning-block view, proposes local
dependency edges for later blind AI audit, and derives lower/upper Complete
closures from two edge audits.  Original unit indices remain the only bridge
back to exact output-token ranges.

The v12 replay helpers are diagnostic only: they project already-inspected v12
sets onto deterministic blocks to measure boundary fragmentation.  They must
not be used to rescue, relabel, extract, or train from v12.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from numbers import Integral
import re
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


STRUCTURE_SCHEMA = "clir-prior-mechanical-structure-v13-prototype"
DIAGNOSTIC_SCHEMA = "clir-prior-v12-mechanical-replay-diagnostic-v1"
PACKAGE_SCHEMA = "clir-prior-mechanical-local-audit-package-v13"
LABEL_SCHEMA = "clir-prior-mechanical-local-audit-label-v13"
EDGE_DECISIONS = {"keep", "drop", "uncertain"}
ELIGIBILITY_VALUES = {
    "usable",
    "no_auditable_reasoning",
    "insufficient_structure",
}
PATH_STATUS_VALUES = {"supported", "flawed"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
BLOCK_ROLE_VALUES = {
    "main_step",
    "premise_restatement",
    "plan_or_heading",
    "formula_only",
    "duplicate",
    "answer_wrapper",
    "unused_branch",
}

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_COMPARATOR = re.compile(r"(?:\\equiv|\\approx|≈|(?<![<>])=(?!=))")
_CONTINUATION_START = re.compile(r"^\s*(?:[/*×÷]|\\(?:times|cdot|div)\b|[)\]}])")
_RESULT_START = re.compile(r"^\s*(?:=|≈|\\(?:approx|simeq)\b)")
_TRAILING_OPERATOR = re.compile(r"(?:=|[+\-/*×÷]|\\(?:times|cdot|div))\s*$")
_DISCOURSE_START = re.compile(
    r"^\s*(?:using|substituting|plugging|from this|this gives|this yields|"
    r"therefore|thus|hence|so|then|consequently|由此|代入|因此|所以|故)",
    flags=re.IGNORECASE,
)
_PLAN_START = re.compile(
    r"^\s*(?:(?:step\s+\d+\s*:\s*)?"
    r"(?:calculate|find|determine|solve|understand|formulate|check|"
    r"to solve|we need to|now,?\s+we need to|first,?\s+let(?:'s| us)|"
    r"next,?\s+we|let(?:'s| us)\s+(?:calculate|find|determine|solve)))",
    flags=re.IGNORECASE,
)
_WRAPPER_START = re.compile(
    r"^\s*(?:so|therefore|thus|hence|in conclusion|final answer|"
    r"the answer|answer|所以|因此|故|最终答案)",
    flags=re.IGNORECASE,
)
_FORMULA_CUE = re.compile(
    r"\b(?:formula|identity|theorem states|we can use|remember that|"
    r"general form)\b|公式|恒等式|定理",
    flags=re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "calculate",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "then",
    "this",
    "to",
    "we",
    "with",
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _material_units(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = item.get("units")
    if not isinstance(units, list):
        raise ValueError("item units must be a list")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for unit in units:
        if not isinstance(unit, Mapping) or unit.get("kind") != "material_claim":
            continue
        index = unit.get("unit_index")
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("material unit index must be an integer")
        integer = int(index)
        if integer in seen:
            raise ValueError("material unit indices must be unique")
        text = unit.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("material unit text must be non-empty")
        seen.add(integer)
        output.append({"unit_index": integer, "text": text.strip()})
    output.sort(key=lambda row: row["unit_index"])
    if not output:
        raise ValueError("item has no material-claim units")
    return output


def _normalized_text(text: str) -> str:
    value = text.casefold().replace("\\left", "").replace("\\right", "")
    value = value.replace("\\times", "*").replace("\\cdot", "*")
    value = value.replace("×", "*").replace("÷", "/")
    return _SPACE.sub(" ", value).strip()


def _compact_signature(text: str) -> str:
    return re.sub(r"[^a-z0-9一-龥=<>+\-*/().]", "", _normalized_text(text))


def _delimiter_balance(text: str) -> int:
    return text.count("(") + text.count("[") - text.count(")") - text.count("]")


def _looks_mathematical(text: str) -> bool:
    return bool(
        _COMPARATOR.search(text)
        or re.search(r"\d\s*[+\-/*×÷^]", text)
        or re.search(r"[+\-/*×÷^]\s*\d", text)
        or "\\frac" in text
        or "\\binom" in text
    )


def _merge_reason(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> str | None:
    """Return a conservative mechanical merge reason, if any.

    Same-line factorial fragments are adjacent material-unit indices because
    unitizer-v2 split on ``!``.  Normal newline-separated claims usually have a
    non-claim unit between them, so they are two indices apart.  Result-only
    lines beginning with ``=`` are also safe to attach to the preceding
    instantiated expression.
    """

    previous_text = str(previous["text"]).strip()
    current_text = str(current["text"]).strip()
    gap = int(current["unit_index"]) - int(previous["unit_index"])
    if gap <= 0 or gap > 2:
        return None
    if gap == 1 and _CONTINUATION_START.search(current_text):
        return "same_line_operator_continuation"
    if _RESULT_START.search(current_text) and _looks_mathematical(previous_text):
        return "result_only_continuation"
    if _TRAILING_OPERATOR.search(previous_text) and _looks_mathematical(current_text):
        return "trailing_operator_continuation"
    if (
        gap == 1
        and _delimiter_balance(previous_text) > 0
        and _delimiter_balance(previous_text + current_text)
        < _delimiter_balance(previous_text)
    ):
        return "unclosed_delimiter_continuation"
    return None


def _word_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if token.casefold() not in _STOPWORDS
    }


def _numeric_tokens(text: str) -> set[str]:
    return {token.lstrip("+") for token in _NUMBER.findall(text)}


def _equation_features(text: str) -> dict[str, list[str]]:
    normalized = _normalized_text(text)
    match = _COMPARATOR.search(normalized)
    if match is None:
        names = _word_tokens(normalized)
        numbers = _numeric_tokens(normalized)
        return {
            "defined_names": [],
            "used_names": sorted(names),
            "derived_values": [],
            "mentioned_values": sorted(numbers),
        }

    left = normalized[: match.start()]
    right = normalized[match.end() :]
    left_names = _word_tokens(left)
    right_names = _word_tokens(right)
    right_values = _numeric_tokens(right)
    all_values = _numeric_tokens(normalized)
    compact_left = _compact_signature(left)
    defined = set(left_names)
    if 1 <= len(compact_left) <= 80:
        defined.add(f"lhs:{compact_left}")
    return {
        "defined_names": sorted(defined),
        "used_names": sorted(right_names),
        "derived_values": sorted(right_values),
        "mentioned_values": sorted(all_values),
    }


def _premise_overlap(text: str, question: str) -> float:
    claim = _word_tokens(text)
    if not claim:
        return 0.0
    problem = _word_tokens(question)
    return len(claim & problem) / len(claim)


def _classify_role(
    *, text: str, question: str, duplicate_of: int | None
) -> tuple[str, list[str]]:
    normalized = _normalized_text(text)
    reasons: list[str] = []
    if duplicate_of is not None:
        return "exact_duplicate", [f"same_signature_as_block_{duplicate_of}"]
    if _WRAPPER_START.search(normalized) and (
        "boxed" in normalized or "answer" in normalized or "答案" in normalized
    ):
        return "possible_answer_wrapper", ["answer_wrapper_language"]
    if _PLAN_START.search(normalized) and not _looks_mathematical(normalized):
        return "possible_plan_or_heading", ["planning_language_without_calculation"]
    if _FORMULA_CUE.search(normalized):
        reasons.append("generic_formula_language")
        return "possible_formula_only", reasons
    overlap = _premise_overlap(text, question)
    if overlap >= 0.82 and not _looks_mathematical(text):
        return "possible_premise_restatement", [f"question_word_overlap={overlap:.3f}"]
    return "substantive", ["default_substantive"]


def _candidate_edges(
    blocks: Sequence[Mapping[str, Any]], *, question: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Only suppress roles that are nearly mechanical.  The other hints are
    # deliberately fallible: a model may decide that a planning-looking or
    # premise-looking block is actually part of the trajectory.  Surfacing
    # local edges for those blocks lets the audit correct the hint instead of
    # forcing it to invent a large unconstrained graph afterwards.
    excluded_roles = {"exact_duplicate", "possible_answer_wrapper"}
    question_values = _numeric_tokens(question)
    for child_index, child in enumerate(blocks):
        if child["role_hint"] in excluded_roles:
            continue
        scored: dict[int, tuple[int, list[str]]] = {}
        child_text = _normalized_text(str(child["text"]))
        child_names = set(child["used_names"])
        child_values = set(child["mentioned_values"]) - question_values

        # Always surface the nearest substantive predecessor.  This is a
        # proposal, not an accepted edge: the annotator may drop it.  It keeps
        # implicit prose dependencies auditable without asking the model to
        # invent an unconstrained graph as v8 did.
        for parent_index in range(child_index - 1, -1, -1):
            if blocks[parent_index]["role_hint"] not in excluded_roles:
                scored[parent_index] = (
                    2,
                    ["nearest_substantive_predecessor"],
                )
                break

        # A local audit should not face every earlier occurrence of x or 12.
        # For each cue, propose only its nearest substantive producer.
        for name in sorted(child_names):
            for parent_index in range(child_index - 1, -1, -1):
                parent = blocks[parent_index]
                if parent["role_hint"] in excluded_roles:
                    continue
                if name in set(parent["defined_names"]):
                    score, evidence = scored.get(parent_index, (0, []))
                    scored[parent_index] = (
                        score + 3,
                        [*evidence, f"nearest_defined_symbol:{name}"],
                    )
                    break

        for value in sorted(child_values):
            for parent_index in range(child_index - 1, -1, -1):
                parent = blocks[parent_index]
                if parent["role_hint"] in excluded_roles:
                    continue
                if value in set(parent["derived_values"]):
                    score, evidence = scored.get(parent_index, (0, []))
                    scored[parent_index] = (
                        score + 1,
                        [*evidence, f"nearest_derived_value:{value}"],
                    )
                    break

        # Multi-word left-hand quantities such as ``total time`` are stronger
        # than coincidental single-word overlap.
        for parent_index in range(child_index - 1, -1, -1):
            parent = blocks[parent_index]
            if parent["role_hint"] in excluded_roles:
                continue
            for name in set(parent["defined_names"]):
                if name.startswith("lhs:"):
                    phrase = name[4:]
                    if len(phrase) >= 2 and phrase in _compact_signature(child_text):
                        score, evidence = scored.get(parent_index, (0, []))
                        scored[parent_index] = (
                            score + 4,
                            [*evidence, f"defined_phrase:{phrase}"],
                        )
                        break

        if _DISCOURSE_START.search(child_text):
            for parent_index in range(child_index - 1, -1, -1):
                if blocks[parent_index]["role_hint"] not in excluded_roles:
                    score, evidence = scored.get(parent_index, (0, []))
                    scored[parent_index] = (
                        score + 1,
                        [*evidence, "adjacent_substantive_discourse_reference"],
                    )
                    break

        ranked = [
            (score, parent_index, evidence)
            for parent_index, (score, evidence) in scored.items()
        ]
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        for score, parent_index, evidence in ranked[:2]:
            output.append(
                {
                    "parent_block_id": parent_index,
                    "child_block_id": child_index,
                    "strength": "high" if score >= 4 else "medium",
                    "evidence": evidence,
                }
            )
    return output


def compile_reasoning_structure(item: Mapping[str, Any]) -> dict[str, Any]:
    """Compile material units into deterministic AI-auditable blocks."""

    units = _material_units(item)
    grouped: list[dict[str, Any]] = []
    for unit in units:
        if grouped:
            reason = _merge_reason(grouped[-1]["last_unit"], unit)
        else:
            reason = None
        if reason is None:
            grouped.append(
                {
                    "units": [unit],
                    "last_unit": unit,
                    "merge_reasons": [],
                }
            )
        else:
            grouped[-1]["units"].append(unit)
            grouped[-1]["last_unit"] = unit
            grouped[-1]["merge_reasons"].append(reason)

    question = str(item.get("question", ""))
    signatures: dict[str, int] = {}
    blocks: list[dict[str, Any]] = []
    for block_id, group in enumerate(grouped):
        texts = [str(unit["text"]) for unit in group["units"]]
        text = "\n".join(texts)
        signature = _compact_signature(text)
        duplicate_of = signatures.get(signature) if signature else None
        role, role_reasons = _classify_role(
            text=text, question=question, duplicate_of=duplicate_of
        )
        if signature and duplicate_of is None:
            signatures[signature] = block_id
        features = _equation_features(text)
        blocks.append(
            {
                "block_id": block_id,
                "unit_indices": [int(unit["unit_index"]) for unit in group["units"]],
                "text": text,
                "merge_reasons": list(group["merge_reasons"]),
                "role_hint": role,
                "role_reasons": role_reasons,
                **features,
            }
        )

    edges = _candidate_edges(blocks, question=question)
    excluded_final_roles = {
        "exact_duplicate",
        "possible_answer_wrapper",
        "possible_plan_or_heading",
        "possible_formula_only",
        "possible_premise_restatement",
    }
    substantive_final_candidates = [
        int(block["block_id"])
        for block in blocks
        if block["role_hint"] not in excluded_final_roles
    ][-3:]
    fallback_final_candidates = [
        int(block["block_id"])
        for block in blocks[-5:]
        if block["role_hint"] != "exact_duplicate"
    ]
    final_candidates = sorted(
        set(substantive_final_candidates) | set(fallback_final_candidates)
    )
    if substantive_final_candidates:
        proposed_final = substantive_final_candidates[-1]
    else:
        proposed_final = int(blocks[-1]["block_id"])
    unit_to_block = {
        str(unit_index): int(block["block_id"])
        for block in blocks
        for unit_index in block["unit_indices"]
    }
    source = {
        "question": question,
        "response": str(item.get("response", "")),
        "units": units,
    }
    return {
        "schema_version": STRUCTURE_SCHEMA,
        "item_id": str(
            item.get("item_id", item.get("proposal_id", item.get("id", "")))
        ),
        "source_sha256": _canonical_sha256(source),
        "material_unit_count": len(units),
        "block_count": len(blocks),
        "blocks": blocks,
        "unit_to_block": unit_to_block,
        "candidate_edges": edges,
        "proposed_final_block_id": proposed_final,
        "final_block_candidates": final_candidates,
        "claim_boundary": (
            "prototype structure only; every role and edge must be audited on "
            "fresh samples before label publication"
        ),
    }


def project_unit_indices_to_blocks(
    structure: Mapping[str, Any], unit_indices: Iterable[int]
) -> list[int]:
    mapping = {
        int(key): int(value) for key, value in structure["unit_to_block"].items()
    }
    projected: set[int] = set()
    for unit_index in unit_indices:
        integer = int(unit_index)
        if integer not in mapping:
            raise ValueError(
                f"unit index {integer} is missing from mechanical structure"
            )
        projected.add(mapping[integer])
    return sorted(projected)


def expand_block_indices_to_units(
    structure: Mapping[str, Any], block_indices: Iterable[int]
) -> list[int]:
    blocks = {int(block["block_id"]): block for block in structure["blocks"]}
    output: set[int] = set()
    for block_index in block_indices:
        integer = int(block_index)
        if integer not in blocks:
            raise ValueError(f"block index {integer} is missing")
        output.update(int(index) for index in blocks[integer]["unit_indices"])
    return sorted(output)


def _backward_closure(
    *, final_block_id: int, edges: Iterable[tuple[int, int]], block_count: int
) -> set[int]:
    if not 0 <= final_block_id < block_count:
        raise ValueError("final block is out of range")
    parents: dict[int, set[int]] = defaultdict(set)
    for parent, child in edges:
        if not 0 <= parent < child < block_count:
            raise ValueError("dependency edges must point forward between valid blocks")
        parents[child].add(parent)
    closure = {final_block_id}
    frontier = [final_block_id]
    while frontier:
        child = frontier.pop()
        for parent in parents.get(child, set()):
            if parent not in closure:
                closure.add(parent)
                frontier.append(parent)
    return closure


def derive_complete_consensus(
    *,
    structure: Mapping[str, Any],
    final_block_id: int,
    decisions_a: Mapping[tuple[int, int], str],
    decisions_b: Mapping[tuple[int, int], str],
) -> dict[str, Any]:
    """Derive definite/possible Complete closures from two local edge audits."""

    candidates = {
        (int(row["parent_block_id"]), int(row["child_block_id"]))
        for row in structure["candidate_edges"]
    }
    if set(decisions_a) != candidates or set(decisions_b) != candidates:
        raise ValueError("each annotator must decide every proposed edge exactly once")
    for decision in [*decisions_a.values(), *decisions_b.values()]:
        if decision not in EDGE_DECISIONS:
            raise ValueError(f"invalid edge decision: {decision!r}")

    definite = {
        edge for edge in candidates if decisions_a[edge] == decisions_b[edge] == "keep"
    }
    possible = {
        edge
        for edge in candidates
        if not (decisions_a[edge] == decisions_b[edge] == "drop")
    }
    block_count = int(structure["block_count"])
    definite_closure = _backward_closure(
        final_block_id=final_block_id,
        edges=definite,
        block_count=block_count,
    )
    possible_closure = _backward_closure(
        final_block_id=final_block_id,
        edges=possible,
        block_count=block_count,
    )
    if not definite_closure <= possible_closure:
        raise AssertionError("definite closure must be a subset of possible closure")
    positive_units = expand_block_indices_to_units(structure, definite_closure)
    possible_units = set(expand_block_indices_to_units(structure, possible_closure))
    all_units = {
        int(unit_index)
        for block in structure["blocks"]
        for unit_index in block["unit_indices"]
    }
    return {
        "definite_edges": [list(edge) for edge in sorted(definite)],
        "possible_edges": [list(edge) for edge in sorted(possible)],
        "positive_block_indices": sorted(definite_closure),
        "masked_block_indices": sorted(possible_closure - definite_closure),
        "negative_block_indices": sorted(set(range(block_count)) - possible_closure),
        "positive_unit_indices": positive_units,
        "masked_unit_indices": sorted(possible_units - set(positive_units)),
        "negative_unit_indices": sorted(all_units - possible_units),
    }


def validate_local_audit_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one block-local audit and derive its raw-unit Complete set."""

    if annotation.get("item_id") != item.get("item_id"):
        raise ValueError("annotation item_id does not match package item")
    structure = item.get("structure")
    if (
        not isinstance(structure, Mapping)
        or structure.get("schema_version") != STRUCTURE_SCHEMA
    ):
        raise ValueError("package lacks a supported mechanical structure")
    eligibility = annotation.get("eligibility")
    if eligibility not in ELIGIBILITY_VALUES:
        raise ValueError("invalid eligibility")
    confidence = annotation.get("confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("invalid confidence")
    rationale = annotation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty")

    block_roles = annotation.get("block_roles")
    edge_decisions = annotation.get("edge_decisions")
    missing_edges = annotation.get("missing_edges")
    final_block = annotation.get("final_block_id")
    key_unit = annotation.get("key_unit_index")
    path_status = annotation.get("path_status")
    if not isinstance(block_roles, list):
        raise ValueError("block_roles must be an array")
    if not isinstance(edge_decisions, list):
        raise ValueError("edge_decisions must be an array")
    if not isinstance(missing_edges, list):
        raise ValueError("missing_edges must be an array")

    if eligibility != "usable":
        if (
            path_status is not None
            or final_block is not None
            or key_unit is not None
            or block_roles
            or edge_decisions
            or missing_edges
        ):
            raise ValueError("ineligible audit must leave all structure fields empty")
        return {
            "schema_version": LABEL_SCHEMA,
            "item_id": str(annotation["item_id"]),
            "eligibility": str(eligibility),
            "path_status": None,
            "block_roles": [],
            "final_block_id": None,
            "edge_decisions": [],
            "missing_edges": [],
            "key_unit_index": None,
            "key_unit_indices": [],
            "complete_block_indices": [],
            "complete_unit_indices": [],
            "confidence": str(confidence),
            "rationale": rationale.strip(),
        }

    if path_status not in PATH_STATUS_VALUES:
        raise ValueError("usable audit has invalid path_status")
    block_count = int(structure["block_count"])
    normalized_roles: list[dict[str, Any]] = []
    for row in block_roles:
        if not isinstance(row, Mapping):
            raise ValueError("each block role must be an object")
        block_id = row.get("block_id")
        if isinstance(block_id, bool) or not isinstance(block_id, Integral):
            raise ValueError("block_id must be an integer")
        integer = int(block_id)
        role = row.get("role")
        if not 0 <= integer < block_count or role not in BLOCK_ROLE_VALUES:
            raise ValueError("invalid block role entry")
        normalized_roles.append({"block_id": integer, "role": str(role)})
    if [row["block_id"] for row in normalized_roles] != list(range(block_count)):
        raise ValueError("block_roles must decide every block in order")
    role_by_id = {row["block_id"]: row["role"] for row in normalized_roles}

    candidate_edges = [
        (int(row["parent_block_id"]), int(row["child_block_id"]))
        for row in structure["candidate_edges"]
    ]
    normalized_decisions: list[dict[str, Any]] = []
    for row in edge_decisions:
        if not isinstance(row, Mapping):
            raise ValueError("each edge decision must be an object")
        parent = row.get("parent_block_id")
        child = row.get("child_block_id")
        decision = row.get("decision")
        if (
            isinstance(parent, bool)
            or isinstance(child, bool)
            or not isinstance(parent, Integral)
            or not isinstance(child, Integral)
            or decision not in EDGE_DECISIONS
        ):
            raise ValueError("invalid edge decision")
        normalized_decisions.append(
            {
                "parent_block_id": int(parent),
                "child_block_id": int(child),
                "decision": str(decision),
            }
        )
    decision_pairs = [
        (row["parent_block_id"], row["child_block_id"]) for row in normalized_decisions
    ]
    if decision_pairs != candidate_edges:
        raise ValueError("edge_decisions must decide every candidate edge in order")

    normalized_missing: list[list[int]] = []
    for edge in missing_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("each missing edge must be [parent, child]")
        parent, child = edge
        if (
            isinstance(parent, bool)
            or isinstance(child, bool)
            or not isinstance(parent, Integral)
            or not isinstance(child, Integral)
        ):
            raise ValueError("missing edge endpoints must be integers")
        pair = (int(parent), int(child))
        if not 0 <= pair[0] < pair[1] < block_count:
            raise ValueError("missing edge must point forward between valid blocks")
        if pair in candidate_edges:
            raise ValueError("missing_edges must not repeat a candidate edge")
        normalized_missing.append([pair[0], pair[1]])
    normalized_missing_tuples = [tuple(edge) for edge in normalized_missing]
    if normalized_missing_tuples != sorted(set(normalized_missing_tuples)):
        raise ValueError("missing_edges must be sorted and unique")
    if len(normalized_missing) > 2:
        raise ValueError("at most two genuinely missing edges may be added")

    if isinstance(final_block, bool) or not isinstance(final_block, Integral):
        raise ValueError("usable audit requires an integer final_block_id")
    final_integer = int(final_block)
    if not 0 <= final_integer < block_count:
        raise ValueError("final_block_id is out of range")
    if role_by_id[final_integer] != "main_step":
        raise ValueError("final block must have role main_step")

    kept_edges = {
        (row["parent_block_id"], row["child_block_id"])
        for row in normalized_decisions
        if row["decision"] == "keep"
    } | {tuple(edge) for edge in normalized_missing}
    for parent, child in kept_edges:
        if role_by_id[parent] != "main_step" or role_by_id[child] != "main_step":
            raise ValueError("kept edges may connect only main_step blocks")
    complete_blocks = _backward_closure(
        final_block_id=final_integer,
        edges=kept_edges,
        block_count=block_count,
    )
    if any(role_by_id[block_id] != "main_step" for block_id in complete_blocks):
        raise AssertionError("Complete closure contains a non-main block")
    complete_units = expand_block_indices_to_units(structure, complete_blocks)

    if isinstance(key_unit, bool) or not isinstance(key_unit, Integral):
        raise ValueError("usable audit requires one integer key_unit_index")
    key_integer = int(key_unit)
    if key_integer not in set(complete_units):
        raise ValueError("Key unit must lie in the derived Complete closure")
    if path_status == "supported":
        final_units = set(expand_block_indices_to_units(structure, [final_integer]))
        if key_integer not in final_units:
            raise ValueError("supported-path Key must lie in the final block")

    return {
        "schema_version": LABEL_SCHEMA,
        "item_id": str(annotation["item_id"]),
        "eligibility": "usable",
        "path_status": str(path_status),
        "block_roles": normalized_roles,
        "final_block_id": final_integer,
        "edge_decisions": normalized_decisions,
        "missing_edges": normalized_missing,
        "key_unit_index": key_integer,
        "key_unit_indices": [key_integer],
        "complete_block_indices": sorted(complete_blocks),
        "complete_unit_indices": complete_units,
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }


def local_audit_target_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation.get("eligibility"),
        annotation.get("path_status"),
        tuple(annotation.get("key_unit_indices", [])),
        annotation.get("final_block_id"),
        tuple(annotation.get("complete_block_indices", [])),
    )


def _set_f1(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _set_iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _coverage(left: Iterable[int], right: Iterable[int], universe_size: int) -> float:
    return 1.0 - len(set(left) ^ set(right)) / max(1, universe_size)


def _number_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    converted = [float(value) for value in values]
    return {
        "count": len(converted),
        "min": min(converted),
        "median": median(converted),
        "mean": mean(converted),
        "max": max(converted),
    }


def diagnose_v12_mechanical_projection(
    *,
    packages_a: Sequence[Mapping[str, Any]],
    packages_b: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    proposals: Sequence[Mapping[str, Any]],
    final_strata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay v12 sets on blocks without producing or authorizing labels."""

    packages = {
        "a": {str(row["item_id"]): dict(row) for row in packages_a},
        "b": {str(row["item_id"]): dict(row) for row in packages_b},
    }
    labels = {
        "a": {str(row["item_id"]): dict(row) for row in labels_a},
        "b": {str(row["item_id"]): dict(row) for row in labels_b},
    }
    private: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for row in private_index:
        annotator = str(row["annotator"])
        if annotator not in private:
            raise ValueError("unexpected annotator in private index")
        private[annotator][str(row["item_id"])] = dict(row)

    natural_ids = {
        str(row["natural_item_id"])
        for row in private_index
        if row.get("kind") == "natural"
    }
    if not natural_ids:
        raise ValueError("private index has no natural rows")
    structures: dict[str, dict[str, Any]] = {}
    for item_id in sorted(natural_ids):
        if item_id not in packages["a"] or item_id not in packages["b"]:
            raise ValueError(f"natural package row is missing: {item_id}")
        left = compile_reasoning_structure(packages["a"][item_id])
        right = compile_reasoning_structure(packages["b"][item_id])
        if left["source_sha256"] != right["source_sha256"]:
            raise ValueError(f"A/B natural package content differs: {item_id}")
        structures[item_id] = left

    role_counts: Counter[str] = Counter()
    merge_counts: Counter[str] = Counter()
    candidate_edge_counts: list[int] = []
    material_counts: list[int] = []
    block_counts: list[int] = []
    for structure in structures.values():
        material_counts.append(int(structure["material_unit_count"]))
        block_counts.append(int(structure["block_count"]))
        candidate_edge_counts.append(len(structure["candidate_edges"]))
        for block in structure["blocks"]:
            role_counts[str(block["role_hint"])] += 1
            merge_counts.update(str(reason) for reason in block["merge_reasons"])

    repeat_report: dict[str, dict[str, Any]] = {}
    for annotator in ("a", "b"):
        repeat_rows = [
            row for row in private[annotator].values() if row.get("kind") == "repeat"
        ]
        raw_complete_exact = 0
        block_complete_exact = 0
        rescued = 0
        remaining_role_counts: Counter[str] = Counter()
        for row in repeat_rows:
            repeat_id = str(row["item_id"])
            natural_id = str(row["natural_item_id"])
            natural = labels[annotator][natural_id]
            repeated = labels[annotator][repeat_id]
            raw_equal = (
                natural["complete_unit_indices"] == repeated["complete_unit_indices"]
            )
            raw_complete_exact += raw_equal
            structure = structures[natural_id]
            natural_blocks = project_unit_indices_to_blocks(
                structure, natural["complete_unit_indices"]
            )
            repeat_blocks = project_unit_indices_to_blocks(
                structure, repeated["complete_unit_indices"]
            )
            block_equal = natural_blocks == repeat_blocks
            block_complete_exact += block_equal
            rescued += block_equal and not raw_equal
            if not block_equal:
                differing = set(natural_blocks) ^ set(repeat_blocks)
                by_id = {int(block["block_id"]): block for block in structure["blocks"]}
                remaining_role_counts.update(
                    str(by_id[block_id]["role_hint"]) for block_id in differing
                )
        repeat_report[annotator] = {
            "total": len(repeat_rows),
            "raw_complete_exact": raw_complete_exact,
            "raw_complete_exact_rate": raw_complete_exact / max(1, len(repeat_rows)),
            "block_projected_complete_exact": block_complete_exact,
            "block_projected_complete_exact_rate": block_complete_exact
            / max(1, len(repeat_rows)),
            "fragmentation_only_rows_rescued": rescued,
            "remaining_differing_block_roles": dict(
                sorted(remaining_role_counts.items())
            ),
        }

    def pair_metrics(ids: Sequence[str]) -> dict[str, Any]:
        raw_f1: list[float] = []
        raw_iou: list[float] = []
        raw_coverage: list[float] = []
        block_f1: list[float] = []
        block_iou: list[float] = []
        block_coverage: list[float] = []
        raw_exact = 0
        block_exact = 0
        rescued = 0
        candidate_closure_coverage_a: list[float] = []
        candidate_closure_coverage_b: list[float] = []
        candidate_full_coverage_a = 0
        candidate_full_coverage_b = 0
        anchor_in_final_candidates_a = 0
        anchor_in_final_candidates_b = 0
        for item_id in ids:
            left = labels["a"][item_id]["complete_unit_indices"]
            right = labels["b"][item_id]["complete_unit_indices"]
            structure = structures[item_id]
            left_blocks = project_unit_indices_to_blocks(structure, left)
            right_blocks = project_unit_indices_to_blocks(structure, right)
            raw_equal = left == right
            block_equal = left_blocks == right_blocks
            raw_exact += raw_equal
            block_exact += block_equal
            rescued += block_equal and not raw_equal
            raw_f1.append(_set_f1(left, right))
            raw_iou.append(_set_iou(left, right))
            raw_coverage.append(
                _coverage(left, right, int(structure["material_unit_count"]))
            )
            block_f1.append(_set_f1(left_blocks, right_blocks))
            block_iou.append(_set_iou(left_blocks, right_blocks))
            block_coverage.append(
                _coverage(left_blocks, right_blocks, int(structure["block_count"]))
            )
            candidate_edges = {
                (int(row["parent_block_id"]), int(row["child_block_id"]))
                for row in structure["candidate_edges"]
            }
            for projected, coverage_values, side in (
                (left_blocks, candidate_closure_coverage_a, "a"),
                (right_blocks, candidate_closure_coverage_b, "b"),
            ):
                if not projected:
                    coverage_values.append(1.0)
                    continue
                anchor = max(projected)
                closure = _backward_closure(
                    final_block_id=anchor,
                    edges=candidate_edges,
                    block_count=int(structure["block_count"]),
                )
                coverage = len(set(projected) & closure) / len(projected)
                coverage_values.append(coverage)
                if set(projected) <= closure:
                    if side == "a":
                        candidate_full_coverage_a += 1
                    else:
                        candidate_full_coverage_b += 1
                if anchor in structure["final_block_candidates"]:
                    if side == "a":
                        anchor_in_final_candidates_a += 1
                    else:
                        anchor_in_final_candidates_b += 1
        return {
            "rows": len(ids),
            "raw_exact": raw_exact,
            "block_projected_exact": block_exact,
            "fragmentation_only_rows_rescued": rescued,
            "raw_macro_f1": mean(raw_f1) if raw_f1 else None,
            "block_projected_macro_f1": mean(block_f1) if block_f1 else None,
            "raw_iou_mean": mean(raw_iou) if raw_iou else None,
            "block_projected_iou_mean": mean(block_iou) if block_iou else None,
            "raw_coverage_mean": mean(raw_coverage) if raw_coverage else None,
            "block_projected_coverage_mean": (
                mean(block_coverage) if block_coverage else None
            ),
            "candidate_graph_diagnostic": {
                "annotator_a_selected_block_recall_mean": (
                    mean(candidate_closure_coverage_a)
                    if candidate_closure_coverage_a
                    else None
                ),
                "annotator_b_selected_block_recall_mean": (
                    mean(candidate_closure_coverage_b)
                    if candidate_closure_coverage_b
                    else None
                ),
                "annotator_a_rows_with_full_selected_block_recall": (
                    candidate_full_coverage_a
                ),
                "annotator_b_rows_with_full_selected_block_recall": (
                    candidate_full_coverage_b
                ),
                "annotator_a_anchor_in_final_candidates": anchor_in_final_candidates_a,
                "annotator_b_anchor_in_final_candidates": anchor_in_final_candidates_b,
            },
        }

    ordered_natural = sorted(natural_ids)
    proposal_by_id = {str(row["proposal_id"]): dict(row) for row in proposals}
    strict_ids = [
        item_id
        for item_id in ordered_natural
        if labels["a"][item_id]["eligibility"]
        == labels["b"][item_id]["eligibility"]
        == "usable"
        and labels["a"][item_id]["confidence"] != "low"
        and labels["b"][item_id]["confidence"] != "low"
        and len(labels["a"][item_id]["key_unit_indices"]) == 1
        and labels["a"][item_id]["key_unit_indices"]
        == labels["b"][item_id]["key_unit_indices"]
        and set(labels["a"][item_id]["complete_unit_indices"])
        & set(labels["b"][item_id]["complete_unit_indices"])
    ]
    eligible_by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item_id in strict_ids:
        proposal = proposal_by_id[item_id]
        stratum = (
            str(proposal["source"]),
            str(proposal["checker_status"]),
            str(proposal["prior_label_split"]),
        )
        eligible_by_stratum[stratum].append(item_id)
    for stratum in eligible_by_stratum:
        eligible_by_stratum[stratum].sort(
            key=lambda item_id: (
                str(proposal_by_id[item_id]["selection_priority"]),
                item_id,
            )
        )
    selected_ids: list[str] = []
    for row in final_strata:
        stratum = (str(row["source"]), str(row["checker_status"]), str(row["split"]))
        count = int(row["count"])
        if len(eligible_by_stratum.get(stratum, [])) < count:
            raise ValueError(f"v12 replay cannot reproduce final stratum {stratum}")
        selected_ids.extend(eligible_by_stratum[stratum][:count])

    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "DIAGNOSTIC_ONLY_V12_REMAINS_TERMINAL",
        "claim_boundary": (
            "block projection measures unit fragmentation only; it is not a new "
            "annotation, reliability pass, target publication, or training authorization"
        ),
        "population": {
            "natural_rows": len(ordered_natural),
            "strict_eligible_rows_reproduced": len(strict_ids),
            "selected_rows_reproduced": len(selected_ids),
            "selected_ordered_ids_sha256": _canonical_sha256(selected_ids),
        },
        "mechanical_structure": {
            "material_units": _number_summary(material_counts),
            "reasoning_blocks": _number_summary(block_counts),
            "merged_unit_count": sum(material_counts) - sum(block_counts),
            "merge_reasons": dict(sorted(merge_counts.items())),
            "role_hints": dict(sorted(role_counts.items())),
            "candidate_edges_per_row": _number_summary(candidate_edge_counts),
        },
        "self_repeat": repeat_report,
        "a_b_natural": pair_metrics(ordered_natural),
        "a_b_prefrozen_selected_500": pair_metrics(selected_ids),
        "next_gate": (
            "inspect whether fragmentation gains are material; then freeze a fresh "
            "local-edge-audit smoke before any target publication"
        ),
        "training_allowed": False,
    }


__all__ = [
    "DIAGNOSTIC_SCHEMA",
    "BLOCK_ROLE_VALUES",
    "CONFIDENCE_VALUES",
    "EDGE_DECISIONS",
    "ELIGIBILITY_VALUES",
    "LABEL_SCHEMA",
    "PACKAGE_SCHEMA",
    "PATH_STATUS_VALUES",
    "STRUCTURE_SCHEMA",
    "compile_reasoning_structure",
    "derive_complete_consensus",
    "diagnose_v12_mechanical_projection",
    "expand_block_indices_to_units",
    "local_audit_target_signature",
    "project_unit_indices_to_blocks",
    "validate_local_audit_annotation",
]
