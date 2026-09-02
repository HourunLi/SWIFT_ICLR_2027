"""Recall-first dependency-edge proposals for a future CLIR Prior version.

This module is deliberately separate from :mod:`clir_prior_mechanical` so the
frozen Prior-v13 packages, compiler, labels, and terminal report remain byte
stable.  It consumes only a public v13-style package item and proposes a wider
but bounded set of parent edges for later independent ``keep/drop`` audit.

The current rules are post-hoc development rules.  Replaying them against the
v13 max-reasoning bridge labels can measure proposal recall, but cannot turn
v13 into a pass or publish training labels.  A future use must freeze these
rules before selecting and annotating fresh query/cluster-disjoint samples.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

from src.clir_prior_mechanical import STRUCTURE_SCHEMA


EDGE_PROPOSAL_SCHEMA = "clir-prior-mechanical-edge-candidates-v14-dev-v1"
FROZEN_EDGE_PROPOSAL_SCHEMA = (
    "clir-prior-mechanical-edge-candidates-v14-frozen-v1"
)
DEFAULT_MIN_PARENTS = 2
DEFAULT_MAX_PARENTS = 6

_COMPARATOR = re.compile(r"(?:\\equiv|\\approx|≈|(?<![<>])=(?!=))")
_LATEX_SIMPLE_FRACTION = re.compile(
    r"\\(?:d?frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
    r"\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}"
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_MATH_CUE = re.compile(r"[+\-/*^=()]|\d")
_PLAN_ONLY = re.compile(
    r"^\s*(?:now,?\s+|next,?\s+|to\s+)?(?:calculate|determine|divide|"
    r"evaluate|express|find|multiply|rearrange|simplify|solve|substitute|"
    r"use)\b",
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
    "determine",
    "each",
    "find",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "number",
    "of",
    "on",
    "or",
    "per",
    "since",
    "so",
    "that",
    "the",
    "then",
    "there",
    "this",
    "to",
    "using",
    "we",
    "with",
}
_EXCLUDED_PARENT_HINTS = {"exact_duplicate", "possible_answer_wrapper"}
_LOW_PRIORITY_PARENT_HINTS = {
    "possible_plan_or_heading",
    "possible_premise_restatement",
    "possible_formula_only",
}


def _replace_simple_latex_fractions(text: str) -> str:
    """Make simple numeric LaTeX fractions visible to the number scanner."""

    current = text
    while True:
        updated = _LATEX_SIMPLE_FRACTION.sub(
            lambda match: f"{match.group(1)}/{match.group(2)}", current
        )
        if updated == current:
            return current
        current = updated


def _numeric_values(text: str) -> set[str]:
    """Return comma-normalized values plus fraction components.

    Keeping both ``2026/2`` and its atomic operands ``2026`` and ``2`` lets a
    divisor-producing step become a candidate parent of a later LaTeX
    fraction without treating the old spurious token ``000`` as a value.
    """

    normalized = _replace_simple_latex_fractions(text).replace("$", "")
    output: set[str] = set()
    for match in _NUMBER.findall(normalized):
        value = match.lstrip("+").replace(",", "")
        output.add(value)
        if "/" in value:
            output.update(component for component in value.split("/") if component)
    return output


def _content_words(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(text)
        if token.casefold() not in _STOPWORDS
    }


def _math_variables(text: str) -> set[str]:
    """Extract explicit one-letter variables only from math-bearing text."""

    without_commands = _LATEX_COMMAND.sub(" ", text)
    if _COMPARATOR.search(without_commands) is None and _MATH_CUE.search(
        without_commands
    ) is None:
        return set()
    return {token for token in _WORD.findall(without_commands) if len(token) == 1}


def _equation_sides(text: str) -> tuple[str | None, str]:
    matches = list(_COMPARATOR.finditer(text))
    if not matches:
        return None, text
    first = matches[0]
    return text[: first.start()], text[first.end() :]


def _lhs_phrase(text: str) -> str:
    left, _ = _equation_sides(text)
    if left is None:
        return ""
    return " ".join(
        token.casefold()
        for token in _WORD.findall(left)
        if token.casefold() not in _STOPWORDS
    )


def _output_values(text: str, question_values: set[str]) -> set[str]:
    comparators = list(_COMPARATOR.finditer(text))
    if comparators:
        return _numeric_values(text[comparators[-1].end() :])
    # A prose block can produce a newly derived fact (for example, "the GCD
    # is 2").  Numbers copied directly from the question are not a reason to
    # place a premise restatement into Complete.
    return _numeric_values(text) - question_values


def _atomic_output_values(text: str) -> set[str]:
    """Return a terminal numeric result when the final RHS is not a formula."""

    comparators = list(_COMPARATOR.finditer(text))
    if not comparators:
        return set()
    right = _replace_simple_latex_fractions(text[comparators[-1].end() :])
    values = _numeric_values(right)
    if not values:
        return set()
    # A single simple value may be followed by a unit or short explanatory
    # noun phrase.  Additional arithmetic means the value is merely an input
    # to this step, not the step's own terminal result.
    without_fraction_slashes = re.sub(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", "", right)
    if re.search(r"[+*^()]", without_fraction_slashes):
        return set()
    primary = [value for value in values if "/" in value]
    if primary:
        return set(primary)
    if len(values) == 1:
        return values
    return set()


def _add_score(
    scores: dict[int, int],
    evidence: dict[int, list[str]],
    parent: int,
    amount: int,
    reason: str,
) -> None:
    scores[parent] += amount
    if reason not in evidence[parent]:
        evidence[parent].append(reason)


def _block_features(
    blocks: Sequence[Mapping[str, Any]], *, question: str
) -> list[dict[str, Any]]:
    question_values = _numeric_values(question)
    output: list[dict[str, Any]] = []
    for block in blocks:
        text = str(block["text"])
        _, right = _equation_sides(text)
        plan_only = _PLAN_ONLY.search(text) is not None and _COMPARATOR.search(
            text
        ) is None
        output_values = _output_values(text, question_values)
        if plan_only:
            output_values = set()
        output.append(
            {
                "numbers": _numeric_values(text),
                "output_values": output_values,
                "atomic_output_values": _atomic_output_values(text),
                "words": _content_words(text),
                "variables": _math_variables(text),
                "lhs_phrase": _lhs_phrase(text),
                "right_words": _content_words(right),
                "has_equation": _COMPARATOR.search(text) is not None,
                "plan_only": plan_only,
            }
        )
    return output


def propose_dependency_edges_v14(
    item: Mapping[str, Any],
    *,
    min_parents: int = DEFAULT_MIN_PARENTS,
    max_parents: int = DEFAULT_MAX_PARENTS,
    proposal_schema: str = EDGE_PROPOSAL_SCHEMA,
) -> list[dict[str, Any]]:
    """Propose bounded, recall-first dependency edges for one public item.

    The proposer never reads an annotation.  It reserves slots for the nearest
    substantive predecessor and the nearest producer of each explicit numeric
    or symbolic operand, then fills remaining slots using deterministic lexical
    and legacy-candidate scores.  False positives remain harmless proposals:
    independent annotators must still decide every edge as ``keep``, ``drop``,
    or ``uncertain``.
    """

    if not 1 <= min_parents <= max_parents:
        raise ValueError("parent limits must satisfy 1 <= min <= max")
    if max_parents > 8:
        raise ValueError("max_parents above eight is an unsafe audit burden")
    if not isinstance(proposal_schema, str) or not proposal_schema.strip():
        raise ValueError("proposal_schema must be a non-empty string")
    structure = item.get("structure")
    if (
        not isinstance(structure, Mapping)
        or structure.get("schema_version") != STRUCTURE_SCHEMA
    ):
        raise ValueError("item lacks a supported mechanical structure")
    blocks = structure.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("mechanical structure must contain blocks")
    if [int(block["block_id"]) for block in blocks] != list(range(len(blocks))):
        raise ValueError("mechanical block ids must be contiguous and ordered")

    features = _block_features(blocks, question=str(item.get("question", "")))
    old_candidates = {
        (int(edge["parent_block_id"]), int(edge["child_block_id"]))
        for edge in structure.get("candidate_edges", [])
    }
    output: list[dict[str, Any]] = []

    for child in range(len(blocks)):
        if str(blocks[child]["role_hint"]) in _EXCLUDED_PARENT_HINTS:
            continue
        scores: dict[int, int] = defaultdict(int)
        evidence: dict[int, list[str]] = defaultdict(list)
        mandatory: list[int] = []

        for parent, old_child in old_candidates:
            if old_child == child:
                _add_score(scores, evidence, parent, 2, "v13_candidate_fallback")

        # A plan, formula heading, or premise-looking line must not steal the
        # only local slot from the previous actual calculation.
        for parent in range(child - 1, -1, -1):
            if str(blocks[parent]["role_hint"]) == "substantive":
                _add_score(
                    scores,
                    evidence,
                    parent,
                    6,
                    "reserved_nearest_substantive",
                )
                mandatory.append(parent)
                break

        # Every concrete operand gets its nearest preceding producer.  This is
        # what makes A+B+C and 450000-375000 receive all direct inputs instead
        # of an arbitrary top two.
        for value in sorted(features[child]["numbers"]):
            producer: int | None = None
            producer_kind = ""
            # Prefer the actual calculation/definition over a later sentence
            # that merely repeats its result.  This prevents "there are 21"
            # from hiding the earlier ``3/4 * 28 = 21`` step.
            for kind, field in (
                ("atomic", "atomic_output_values"),
                ("equation", "output_values"),
                ("prose", "output_values"),
            ):
                for parent in range(child - 1, -1, -1):
                    if str(blocks[parent]["role_hint"]) in _EXCLUDED_PARENT_HINTS:
                        continue
                    if features[parent]["plan_only"]:
                        continue
                    if kind == "equation" and not features[parent]["has_equation"]:
                        continue
                    if kind == "prose" and features[parent]["has_equation"]:
                        continue
                    if value in features[parent][field]:
                        producer = parent
                        producer_kind = kind
                        break
                if producer is not None:
                    break
            if producer is not None:
                _add_score(
                    scores,
                    evidence,
                    producer,
                    16 if producer_kind == "atomic" else 13,
                    f"reserved_{producer_kind}_operand_value:{value}",
                )
                mandatory.append(producer)

        # Algebraic rewrites often change every numeric token.  Reserve the
        # nearest earlier equation that carries each explicit variable.
        for variable in sorted(features[child]["variables"]):
            for parent in range(child - 1, -1, -1):
                if str(blocks[parent]["role_hint"]) in _EXCLUDED_PARENT_HINTS:
                    continue
                if (
                    not features[parent]["plan_only"]
                    and features[parent]["has_equation"]
                    and variable in features[parent]["variables"]
                ):
                    _add_score(
                        scores,
                        evidence,
                        parent,
                        13,
                        f"reserved_variable:{variable}",
                    )
                    mandatory.append(parent)
                    break

        for parent in range(child):
            if str(blocks[parent]["role_hint"]) in _EXCLUDED_PARENT_HINTS:
                continue
            shared_values = (
                features[parent]["output_values"] & features[child]["numbers"]
            )
            if shared_values:
                _add_score(
                    scores,
                    evidence,
                    parent,
                    10 + 3 * min(3, len(shared_values)),
                    "output_operand_overlap:" + ",".join(sorted(shared_values)),
                )

            phrase = str(features[parent]["lhs_phrase"])
            if phrase and len(phrase.replace(" ", "")) >= 2:
                phrase_words = set(phrase.split())
                if phrase_words <= set(features[child]["right_words"]):
                    _add_score(
                        scores,
                        evidence,
                        parent,
                        10,
                        f"lhs_quantity_used:{phrase}",
                    )
                if phrase == features[child]["lhs_phrase"]:
                    _add_score(
                        scores,
                        evidence,
                        parent,
                        7,
                        f"same_lhs_quantity:{phrase}",
                    )

            shared_words = features[parent]["words"] & features[child]["words"]
            if len(shared_words) >= 2:
                _add_score(
                    scores,
                    evidence,
                    parent,
                    min(6, 2 * len(shared_words)),
                    "quantity_word_overlap:"
                    + ",".join(sorted(shared_words)[:4]),
                )
            if (
                str(blocks[parent]["role_hint"]) in _LOW_PRIORITY_PARENT_HINTS
                and not shared_values
            ):
                _add_score(
                    scores,
                    evidence,
                    parent,
                    -7,
                    "low_priority_role_without_value",
                )

        mandatory = list(dict.fromkeys(mandatory))
        strong_count = sum(score >= 12 for score in scores.values())
        target_count = min(
            max_parents,
            max(min_parents, len(mandatory), min(4, strong_count)),
        )
        mandatory = sorted(
            mandatory, key=lambda parent: (-scores[parent], -parent)
        )[:target_count]
        chosen = list(mandatory)
        for parent in sorted(scores, key=lambda value: (-scores[value], -value)):
            if (
                parent not in chosen
                and scores[parent] > 0
                and len(chosen) < target_count
            ):
                chosen.append(parent)

        for parent in sorted(chosen):
            output.append(
                {
                    "schema_version": proposal_schema,
                    "parent_block_id": parent,
                    "child_block_id": child,
                    "strength": "high" if scores[parent] >= 12 else "medium",
                    "score": scores[parent],
                    "mandatory": parent in mandatory,
                    "evidence": evidence[parent],
                }
            )

    if any(
        not 0
        <= int(edge["parent_block_id"])
        < int(edge["child_block_id"])
        < len(blocks)
        for edge in output
    ):
        raise AssertionError("dependency proposals must point forward")
    counts: dict[int, int] = defaultdict(int)
    for edge in output:
        counts[int(edge["child_block_id"])] += 1
    if any(count > max_parents for count in counts.values()):
        raise AssertionError("dependency proposal cap was exceeded")
    return output
