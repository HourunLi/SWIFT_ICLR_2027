"""Deterministic data contracts for the CLIR multi-source smoke pipeline.

This module deliberately contains no model-provider client.  It freezes source
queries, checks numeric outcomes, materializes exact-token reasoning units,
builds pre-annotation proposals, validates blind AI labels, and turns accepted
unit labels into the token targets consumed by the clean trainer.  Large and
provider-specific artifacts stay under ``run_artifacts``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
import xml.etree.ElementTree as ET


SMOKE_SCHEMA = "clir-data-expansion-smoke-v2"
CHECKER_VERSION = "clir_numeric_multisource_v3"
SUPPORTED_CHECKER_VERSIONS = {
    "clir_numeric_multisource_v2",
    CHECKER_VERSION,
}
CORRECTNESS_SEMANTICS = "numeric_value_match_v2"
UNITIZER_VERSION = "clir_material_claim_unitizer_v2"
LABEL_TIER = "silver_dual_ai_v2"
CONFIDENCE_VALUES = {"high", "medium", "low"}
FINAL_H_VALUES = {"hallucinated", "clean"}
H_STATUS_VALUES = FINAL_H_VALUES | {
    "uncertain",
    "insufficient_unitization",
    "no_auditable_reasoning",
}
PRIOR_ELIGIBILITY_VALUES = {
    "usable",
    "insufficient_unitization",
    "no_auditable_reasoning",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} is not an object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_text(Path(path), payload)


def publish_manifest(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish ordered rows and a hash-binding sidecar report."""

    output = Path(path)
    normalized = [dict(row) for row in rows]
    atomic_write_jsonl(output, normalized)
    report = {
        "schema_version": schema_version,
        "row_count": len(normalized),
        "ordered_rows_sha256": canonical_sha256(normalized),
        "file_sha256": file_sha256(output),
        "path": str(output.resolve()),
        "metadata": dict(metadata or {}),
    }
    atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), report)
    return report


def stable_priority(namespace: str, *parts: Any) -> str:
    payload = "|".join([namespace, *(str(part) for part in parts)])
    return text_sha256(payload)


def validate_token_ids(values: Any, *, field: str, row_id: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be an integer sequence")
    output: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{row_id}: {field} contains a non-integer token ID")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{row_id}: {field} contains a negative token ID")
        output.append(integer)
    if not output:
        raise ValueError(f"{row_id}: {field} must not be empty")
    return output


# ---------------------------------------------------------------------------
# Source freezing and duplicate clusters
# ---------------------------------------------------------------------------


_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?")
_WORD = re.compile(r"[a-z]+|<num>")


def normalize_question(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("question must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    normalized = "".join(
        char if char.isalnum() or char.isspace() else " " for char in normalized
    )
    return " ".join(normalized.split())


def template_signature(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _NUMBER.sub(" <num> ", normalized)
    words = _WORD.findall(normalized)
    return " ".join(words)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            lo, hi = sorted((a, b))
            self.parent[hi] = lo


def validate_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    if source not in {"gsm8k", "asdiv-a", "math"}:
        raise ValueError("source must be gsm8k, asdiv-a, or math")
    query_id = row.get("query_id")
    question = row.get("question")
    reference = row.get("reference_answer")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id must be a non-empty string")
    expected_prefixes = {
        "gsm8k": ("gsm8k:train:",),
        "asdiv-a": ("asdiv-a:",),
        # MATH test rows are permitted here so the same deterministic
        # exact/template/near-duplicate machinery can audit a protected
        # evaluation population against train-side anchors.  The caller still
        # controls which split is authorized; this validator only enforces an
        # explicit namespace rather than silently treating test rows as train.
        "math": ("math:train:", "math:test:"),
    }[str(source)]
    if not query_id.startswith(expected_prefixes):
        raise ValueError(f"{query_id}: query_id does not match source namespace")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{query_id}: question must be non-empty")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{query_id}: reference_answer must be non-empty")
    normalized = dict(row)
    normalized["question"] = question.strip()
    normalized["reference_answer"] = reference.strip()
    normalized["normalized_question"] = normalize_question(question)
    normalized["template_signature"] = template_signature(question)
    normalized["question_sha256"] = text_sha256(question.strip())
    return normalized


def load_asdiv_a_repository(
    repository_root: str | Path,
    *,
    expected_xml_sha256: str,
    expected_subset_size: int,
) -> list[dict[str, Any]]:
    """Load the repository's official ASDiv-A fold IDs from its frozen XML.

    The arithmetic subset is defined by the union of the five IDs under
    ``dataset/nfolds/asdiv-a``.  We intentionally do not infer membership from
    ``Solution-Type`` because the repository already publishes the exact fold
    membership used for ASDiv-A.
    """

    root = Path(repository_root)
    xml_path = root / "dataset" / "ASDiv.xml"
    if not xml_path.is_file():
        raise ValueError(f"ASDiv XML is missing: {xml_path}")
    actual_sha256 = file_sha256(xml_path)
    if actual_sha256 != expected_xml_sha256:
        raise ValueError(
            "ASDiv XML SHA-256 mismatch: "
            f"expected {expected_xml_sha256}, found {actual_sha256}"
        )
    subset_ids: list[str] = []
    for fold_index in range(5):
        fold_path = root / "dataset" / "nfolds" / "asdiv-a" / f"fold{fold_index}.txt"
        if not fold_path.is_file():
            raise ValueError(f"ASDiv-A fold file is missing: {fold_path}")
        subset_ids.extend(
            line.strip()
            for line in fold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if len(subset_ids) != len(set(subset_ids)):
        raise ValueError("ASDiv-A fold files contain duplicate problem IDs")
    if len(subset_ids) != expected_subset_size:
        raise ValueError(
            "ASDiv-A subset size mismatch: "
            f"expected {expected_subset_size}, found {len(subset_ids)}"
        )

    document = ET.parse(xml_path)
    problems: dict[str, ET.Element] = {}
    for problem in document.findall(".//Problem"):
        problem_id = problem.get("ID")
        if problem_id:
            problems[problem_id] = problem
    missing = sorted(set(subset_ids) - set(problems))
    if missing:
        raise ValueError(f"ASDiv XML lacks {len(missing)} arithmetic fold IDs")

    rows: list[dict[str, Any]] = []
    for problem_id in subset_ids:
        problem = problems[problem_id]
        body = (problem.findtext("Body") or "").strip()
        question = (problem.findtext("Question") or "").strip()
        answer = (problem.findtext("Answer") or "").strip()
        if not body or not question or not answer:
            raise ValueError(f"ASDiv problem {problem_id} has an empty required field")
        rows.append(
            {
                "source": "asdiv-a",
                "query_id": f"asdiv-a:{problem_id}",
                "source_record_id": problem_id,
                "question": f"{body} {question}",
                "reference_answer": answer,
                "source_body": body,
                "source_question": question,
                "source_unit": (
                    re.search(r"\(([^()]*)\)\s*$", answer).group(1).strip()
                    if re.search(r"\(([^()]*)\)\s*$", answer)
                    else None
                ),
                "source_formula": (problem.findtext("Formula") or "").strip(),
                "source_solution_type": (
                    problem.findtext("Solution-Type") or ""
                ).strip(),
                "source_license": "CC-BY-NC-4.0",
            }
        )
    return rows


def near_duplicate_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    jaccard_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    """Return deterministic high-recall candidates; decisions remain external."""

    normalized = [validate_source_row(row) for row in rows]
    by_query_id = {str(row["query_id"]): row for row in normalized}
    token_sets = [set(str(row["template_signature"]).split()) for row in normalized]
    token_frequency = Counter(token for tokens in token_sets for token in tokens)
    ordered_tokens = [
        sorted(tokens, key=lambda token: (token_frequency[token], token))
        for tokens in token_sets
    ]
    # Exact Jaccard prefix filter: every pair at or above the threshold must
    # share at least one token in these globally ordered prefixes.  We still
    # compute full Jaccard before publishing, so the index changes runtime, not
    # candidate semantics.
    prefixes = [
        tokens[: max(0, len(tokens) - math.ceil(jaccard_threshold * len(tokens)) + 1)]
        for tokens in ordered_tokens
    ]
    postings: dict[str, list[int]] = defaultdict(list)
    pairs: list[dict[str, Any]] = []
    for right_index, right in enumerate(normalized):
        possible_left = {
            left_index
            for token in prefixes[right_index]
            for left_index in postings[token]
        }
        for left_index in possible_left:
            left = normalized[left_index]
            if left["normalized_question"] == right["normalized_question"]:
                continue
            left_tokens, right_tokens = token_sets[left_index], token_sets[right_index]
            if min(len(left_tokens), len(right_tokens)) < (
                jaccard_threshold * max(len(left_tokens), len(right_tokens))
            ):
                continue
            union_size = len(left_tokens | right_tokens)
            similarity = (
                len(left_tokens & right_tokens) / union_size if union_size else 1.0
            )
            if similarity < jaccard_threshold:
                continue
            left_id, right_id = sorted((left["query_id"], right["query_id"]))
            pairs.append(
                {
                    "pair_id": stable_priority(
                        "clir-near-duplicate-v2", left_id, right_id
                    ),
                    "left_query_id": left_id,
                    "right_query_id": right_id,
                    "left_source": by_query_id[left_id]["source"],
                    "right_source": by_query_id[right_id]["source"],
                    "left_question": by_query_id[left_id]["question"],
                    "right_question": by_query_id[right_id]["question"],
                    "template_jaccard": similarity,
                    "decision": None,
                }
            )
        for token in prefixes[right_index]:
            postings[token].append(right_index)
    return sorted(pairs, key=lambda row: row["pair_id"])


def freeze_query_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_counts: Mapping[str, int],
    excluded_query_ids: Iterable[str] = (),
    required_query_ids: Iterable[str] = (),
    near_duplicate_decisions: Sequence[Mapping[str, Any]] = (),
    jaccard_threshold: float = 0.82,
    selection_namespace: str = "clir-smoke-v2",
    membership: str = "train_only_smoke_v2",
    source_stratum_counts: Mapping[tuple[str, str], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze exact/near-duplicate clusters and select source quotas by hash."""

    normalized = [validate_source_row(row) for row in rows]
    by_id = {row["query_id"]: row for row in normalized}
    if len(by_id) != len(normalized):
        raise ValueError("source rows contain duplicate query_id values")
    excluded = set(excluded_query_ids)
    required = set(required_query_ids)
    unknown_exclusions = excluded - set(by_id)
    # Historical exclusions may refer to queries not present in this source slice.
    if any(not isinstance(value, str) or not value for value in excluded):
        raise ValueError("excluded_query_ids must contain non-empty strings")
    unknown_required = required - set(by_id)
    if unknown_required:
        raise ValueError(
            f"required_query_ids contain {len(unknown_required)} unknown values"
        )
    if required & excluded:
        raise ValueError("a required query is also excluded")

    all_candidates = near_duplicate_candidates(
        normalized, jaccard_threshold=jaccard_threshold
    )
    # A pair whose two endpoints are both already forbidden cannot affect the
    # new pool, so paying annotators to decide it adds no protection.  A pair
    # with exactly one forbidden endpoint remains actionable: if it is a
    # duplicate, the previously unseen endpoint must be removed with the
    # historical member's entire cluster.
    skipped_both_excluded = [
        row
        for row in all_candidates
        if row["left_query_id"] in excluded and row["right_query_id"] in excluded
    ]
    candidates = [
        row
        for row in all_candidates
        if not (row["left_query_id"] in excluded and row["right_query_id"] in excluded)
    ]
    decisions: dict[str, str] = {}
    for decision in near_duplicate_decisions:
        pair_id = decision.get("pair_id")
        verdict = decision.get("decision")
        if not isinstance(pair_id, str) or verdict not in {"duplicate", "distinct"}:
            raise ValueError(
                "near-duplicate decisions require pair_id and a valid decision"
            )
        if pair_id in decisions:
            raise ValueError(f"duplicate near-duplicate decision for {pair_id}")
        decisions[pair_id] = verdict
    missing = [row["pair_id"] for row in candidates if row["pair_id"] not in decisions]
    if missing:
        raise ValueError(
            f"{len(missing)} near-duplicate candidates lack frozen decisions"
        )

    union = _UnionFind(by_id)
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in normalized:
        exact_groups[row["normalized_question"]].append(row["query_id"])
    for group in exact_groups.values():
        for other in group[1:]:
            union.union(group[0], other)
    for candidate in candidates:
        candidate["decision"] = decisions[candidate["pair_id"]]
        if candidate["decision"] == "duplicate":
            union.union(candidate["left_query_id"], candidate["right_query_id"])

    clusters: dict[str, list[str]] = defaultdict(list)
    for query_id in by_id:
        clusters[union.find(query_id)].append(query_id)
    survivors: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    excluded_clusters = 0
    for members in clusters.values():
        required_members = sorted(set(members) & required)
        if len(required_members) > 1:
            raise ValueError(
                "a duplicate cluster contains multiple required incumbent queries"
            )
        ordered = sorted(
            members,
            key=lambda query_id: (
                query_id not in required,
                stable_priority("clir-dedup-v2", query_id),
            ),
        )
        survivor = ordered[0]
        cluster_id = stable_priority("clir-cluster-v2", *sorted(members))
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "member_query_ids": sorted(members),
                "survivor_query_id": survivor,
                "excluded_by_prior_membership": any(
                    query_id in excluded for query_id in members
                ),
            }
        )
        if any(query_id in excluded for query_id in members):
            excluded_clusters += 1
        else:
            row = dict(by_id[survivor])
            row["cluster_id"] = cluster_id
            row["membership"] = membership
            row["selection_priority"] = stable_priority(selection_namespace, survivor)
            survivors.append(row)

    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    stratum_counts = dict(source_stratum_counts or {})
    for source, requested in source_counts.items():
        if source not in {"gsm8k", "asdiv-a", "math"} or requested < 0:
            raise ValueError("source_counts contains an invalid source/count")
        pool = sorted(
            (row for row in survivors if row["source"] == source),
            key=lambda row: row["selection_priority"],
        )
        available_counts[source] = len(pool)
        if len(pool) < requested:
            raise ValueError(
                f"Not enough {source} rows after exclusions/dedup: "
                f"need {requested}, found {len(pool)}"
            )
        source_strata = {
            stratum: count
            for (stratum_source, stratum), count in stratum_counts.items()
            if stratum_source == source
        }
        if source_strata:
            if sum(source_strata.values()) != requested:
                raise ValueError(
                    f"source {source} stratum quotas do not sum to source quota"
                )
            for stratum, stratum_requested in sorted(source_strata.items()):
                stratum_pool = [
                    row for row in pool if row.get("selection_stratum") == stratum
                ]
                if len(stratum_pool) < stratum_requested:
                    raise ValueError(
                        f"Not enough {source}/{stratum} rows after exclusions/dedup: "
                        f"need {stratum_requested}, found {len(stratum_pool)}"
                    )
                incumbent = [row for row in stratum_pool if row["query_id"] in required]
                if len(incumbent) > stratum_requested:
                    raise ValueError(
                        f"source {source}/{stratum} has too many required queries"
                    )
                fill = [row for row in stratum_pool if row["query_id"] not in required]
                selected.extend(
                    [*incumbent, *fill[: stratum_requested - len(incumbent)]]
                )
        else:
            incumbent = [row for row in pool if row["query_id"] in required]
            if len(incumbent) > requested:
                raise ValueError(
                    f"source {source} has more required queries than its frozen quota"
                )
            fill = [row for row in pool if row["query_id"] not in required]
            selected.extend([*incumbent, *fill[: requested - len(incumbent)]])
    missing_required = required - {row["query_id"] for row in selected}
    if missing_required:
        raise ValueError(f"{len(missing_required)} required queries were not selected")
    selected.sort(key=lambda row: (row["source"], row["selection_priority"]))
    report = {
        "schema_version": "clir-smoke-source-freeze-v2",
        "input_rows": len(normalized),
        "excluded_present": len(excluded & set(by_id)),
        "excluded_not_in_input": len(unknown_exclusions),
        "required_query_count": len(required),
        "excluded_clusters": excluded_clusters,
        "exact_duplicate_clusters": sum(
            len(group) > 1 for group in exact_groups.values()
        ),
        "near_duplicate_candidates": candidates,
        "near_duplicate_candidates_skipped_both_excluded": [
            row["pair_id"] for row in skipped_both_excluded
        ],
        "clusters": sorted(cluster_rows, key=lambda row: row["cluster_id"]),
        "available_after_dedup": available_counts,
        "selected_counts": dict(Counter(row["source"] for row in selected)),
        "selected_stratum_counts": dict(
            sorted(
                Counter(
                    f"{row['source']}|{row.get('selection_stratum')}"
                    for row in selected
                    if row.get("selection_stratum") is not None
                ).items()
            )
        ),
        "selected_query_ids_sha256": canonical_sha256(
            [row["query_id"] for row in selected]
        ),
    }
    return selected, report


# ---------------------------------------------------------------------------
# Numeric checker
# ---------------------------------------------------------------------------


_LATEX_FRACTION = re.compile(
    r"\\(?:d?frac|tfrac)\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}"
)
_MIXED_LATEX = re.compile(
    r"([-+]?\d+)\s*\\(?:d?frac|tfrac)\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}"
)
_MIXED_PLAIN = re.compile(r"(?<![\d.])([-+]?\d+)\s+(\d+)\s*/\s*(\d+)")
_PLAIN_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*)?\s*%?"
)


def boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        cursor = start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append(text[start : cursor - 1].strip())
    return answers


def _boxed_placeholder(value: str) -> bool:
    normalized = value.strip()
    text_match = re.fullmatch(r"\\text\s*\{(.*)\}", normalized, flags=re.DOTALL)
    if text_match:
        normalized = text_match.group(1).strip()
    if not normalized:
        return True
    letters = re.sub(r"[^A-Za-z]+", "", normalized).casefold()
    return letters in {"answer", "youranswer"} and not re.search(r"\d", normalized)


def _fraction_from_literal(
    literal: str, *, percent_points: bool = False
) -> Fraction | None:
    value = literal.strip().replace("\\%", "%").replace("\\$", "")
    value = value.strip("$").replace(",", "")
    value = re.sub(r"\s+", "", value)
    if re.search(r"(?i)(?:nan|inf(?:inity)?)", value):
        return None
    percentage = value.endswith("%")
    if percentage:
        value = value[:-1]
    latex = _LATEX_FRACTION.fullmatch(value)
    try:
        if latex:
            result = Fraction(int(latex.group(1)), int(latex.group(2)))
        elif re.fullmatch(r"[-+]?\d+/[-+]?\d+", value):
            numerator, denominator = value.split("/", 1)
            result = Fraction(int(numerator), int(denominator))
        else:
            result = Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if percentage and not percent_points:
        return result / 100
    return result


def numeric_options(literal: str) -> set[Fraction]:
    literal = literal.strip()
    mixed = _MIXED_LATEX.fullmatch(literal) or _MIXED_PLAIN.fullmatch(literal)
    if mixed:
        whole = int(mixed.group(1))
        denominator = int(mixed.group(3))
        if denominator == 0:
            return set()
        remainder = Fraction(int(mixed.group(2)), denominator)
        return {Fraction(whole) + (-remainder if whole < 0 else remainder)}
    value = _fraction_from_literal(literal, percent_points=True)
    if value is None:
        return set()
    normalized = literal.replace("\\%", "%").strip()
    if not normalized.endswith("%"):
        return {value}
    conventional = value / 100
    return {conventional} if abs(value) <= 1 else {value, conventional}


def _numeric_expressions(text: str) -> list[tuple[int, int, str]]:
    scrubbed = re.sub(r"\^\s*(?:\{\s*[-+]?\d+\s*\}|[-+]?\d+)", "", text)
    candidates: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in (_MIXED_LATEX, _MIXED_PLAIN, _LATEX_FRACTION):
        for match in pattern.finditer(scrubbed):
            candidates.append((match.start(), match.end(), match.group(0)))
            occupied.append((match.start(), match.end()))
    for match in _PLAIN_NUMBER.finditer(scrubbed):
        if any(
            start <= match.start() and match.end() <= end for start, end in occupied
        ):
            continue
        candidates.append((match.start(), match.end(), match.group(0).strip()))
    return sorted(candidates, key=lambda item: (item[0], -(item[1] - item[0])))


def _answer_span(response: str) -> str | None:
    patterns = (
        r"(?is)(?:final\s+answer|answer)\s*(?:is|=|:)\s*([^\n]+)",
        r"(?is)therefore[, ]+([^\n]+)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, response))
        if matches:
            return matches[-1].group(1).strip().rstrip(". ")
    return None


def _first_compound_duration(text: str) -> str | None:
    """Normalize a leading ``H hours M minutes`` answer to one hour fraction."""

    duration = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(?:\\text\{\s*)?(?:hours?|hrs?)\b\s*\}?"
        r"\s*(?:and\s+)?(\d+(?:\.\d+)?)\s*"
        r"(?:\\text\{\s*)?(?:minutes?|mins?)\b\s*\}?",
        text,
        flags=re.IGNORECASE,
    )
    if duration is None:
        return None
    earlier = _numeric_expressions(text[: duration.start()])
    if earlier:
        return None
    try:
        hours = Fraction(Decimal(duration.group(1)))
        minutes = Fraction(Decimal(duration.group(2)))
    except InvalidOperation:
        return None
    value = hours + minutes / 60
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _governed_numeric_expression(text: str) -> str | None:
    """Extract the numeric value governed by boxed/final-answer prose.

    A direct numeric literal remains authoritative.  Otherwise equality spans
    use their right-hand side, compound durations stay a single value, and
    ordinary answer prose uses its first number so qualifiers such as
    ``$15 for 10 sprays`` cannot silently replace the answer with ``10``.
    """

    if numeric_options(text):
        return text
    governed_text = text.rsplit("=", 1)[1] if "=" in text else text
    duration = _first_compound_duration(governed_text)
    if duration is not None:
        return duration
    expressions = _numeric_expressions(governed_text)
    if not expressions:
        return None
    selected = expressions[-1] if "=" in text else expressions[0]
    return selected[2]


_MATH_NONSCALAR_WORDS = {
    "and",
    "infinity",
    "infty",
    "or",
    "pi",
    "pm",
    "sqrt",
    "to",
}


def extract_math_numeric_reference(solution: str) -> str | None:
    """Extract a deliberately strict scalar numeric target from a MATH solution.

    MATH contains tuples, intervals, symbolic expressions, radicals, and other
    answers that the CLIR numeric checker does not claim to understand.  The
    v3 smoke admits only the final boxed answer when its right-hand side has
    exactly one rational/decimal/percentage literal plus optional prose units.
    This is a source-admission filter, not a general MATH equivalence checker.
    """

    if not isinstance(solution, str) or not solution.strip():
        return None
    boxes = [
        value for value in boxed_answers(solution) if not _boxed_placeholder(value)
    ]
    if not boxes:
        return None
    value = boxes[-1].strip()
    if "=" in value:
        value = value.rsplit("=", 1)[1].strip()
    # These markers can make a single visible numeral denote a symbolic,
    # vector/set, interval, factorial, exponent, or compound answer.
    if re.search(
        r"(?:\\(?:sqrt|pi|pm|infty|cup|cap|mod|pmod)\b|\^|[\[\](),;<>|!])",
        value,
        flags=re.IGNORECASE,
    ):
        return None
    normalized = value.replace(r"\$", "$").replace(r"\%", "%")
    # Make ordinary unit prose auditable while leaving mathematical commands
    # such as fractions intact for the numeric parser.
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(
            r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}",
            r" \1 ",
            normalized,
        )
    expressions = _numeric_expressions(normalized)
    if len(expressions) != 1:
        return None
    start, end, literal = expressions[0]
    if not numeric_options(literal):
        return None
    remainder = normalized[:start] + " " + normalized[end:]
    remainder = re.sub(r"\\(?:,|;|!|quad|qquad)", " ", remainder)
    remainder = remainder.replace("$", " ").strip().rstrip(".")
    if re.search(r"[\\{}:+*/-]", remainder):
        return None
    words = [word.casefold() for word in re.findall(r"[A-Za-z]+", remainder)]
    if any(word in _MATH_NONSCALAR_WORDS or len(word) == 1 for word in words):
        return None
    if re.sub(r"[A-Za-z.\s°]+", "", remainder):
        return None
    return literal


def _extract_candidate_literals(
    response: str, *, normalize_boxed_prose: bool
) -> tuple[list[str], str, int]:
    boxes = boxed_answers(response)
    usable_boxes = [value for value in boxes if not _boxed_placeholder(value)]
    if usable_boxes:
        if not normalize_boxed_prose:
            return usable_boxes, "boxed", len(boxes)
        normalized_boxes = [
            _governed_numeric_expression(value) or value for value in usable_boxes
        ]
        route = (
            "boxed"
            if normalized_boxes == usable_boxes
            else "boxed_numeric_subexpression"
        )
        return normalized_boxes, route, len(boxes)
    span = _answer_span(response)
    if span is not None:
        if not normalize_boxed_prose:
            expressions = _numeric_expressions(span)
            if expressions:
                selected = expressions[-1] if "=" in span else expressions[0]
                return [selected[2]], "answer_cue", len(boxes)
            return [span], "answer_cue_non_numeric", len(boxes)
        governed = _governed_numeric_expression(span)
        if governed is not None:
            return [governed], "answer_cue", len(boxes)
        return [span], "answer_cue_non_numeric", len(boxes)
    expressions = _numeric_expressions(response)
    return ([expressions[-1][2]] if expressions else []), "last_numeric", len(boxes)


def _reference_literal(source: str, raw_reference: str) -> str:
    if source == "gsm8k":
        match = re.search(r"####\s*(.+?)\s*$", raw_reference, flags=re.DOTALL)
        return (match.group(1) if match else raw_reference).strip()
    if source == "asdiv-a":
        expressions = _numeric_expressions(raw_reference)
        return (expressions[0][2] if expressions else raw_reference).strip()
    if source == "math":
        # Source export already freezes the strict literal.  Re-run the basic
        # numeric parse here so a malformed edited manifest fails closed.
        return raw_reference.strip() if numeric_options(raw_reference) else ""
    raise ValueError(f"unsupported source {source!r}")


def check_numeric_response(
    *,
    response: str,
    raw_reference: str,
    source: str,
    finish_reason: str | None = None,
    checker_version: str = CHECKER_VERSION,
) -> dict[str, Any]:
    """Return auditable numeric-value-match semantics for GSM8K/ASDiv-A."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if checker_version not in SUPPORTED_CHECKER_VERSIONS:
        raise ValueError(
            f"unsupported checker_version {checker_version!r}; expected one of "
            f"{sorted(SUPPORTED_CHECKER_VERSIONS)}"
        )
    reference_literal = _reference_literal(source, raw_reference)
    reference_options = numeric_options(reference_literal)
    base = {
        "checker_version": checker_version,
        "correctness_semantics": CORRECTNESS_SEMANTICS,
        "reference_answer": reference_literal,
        "explicit_unit_status": "not_checked",
        "checker_dispute": False,
    }
    if not reference_options:
        return {
            **base,
            "numeric_value_match": None,
            "correctness": None,
            "checker_status": "invalid_reference",
            "checker_failure_reason": "reference_not_uniquely_numeric",
            "parsed_answer": None,
            "normalized_candidate_answer": None,
            "boxed_answer_count": len(boxed_answers(response)),
            "eligible_for_supervision": False,
        }
    if finish_reason == "length":
        return {
            **base,
            "numeric_value_match": None,
            "correctness": None,
            "checker_status": "truncated",
            "checker_failure_reason": "finish_reason_length",
            "parsed_answer": None,
            "normalized_candidate_answer": None,
            "boxed_answer_count": len(boxed_answers(response)),
            "eligible_for_supervision": False,
        }
    if not response.strip():
        return {
            **base,
            "numeric_value_match": None,
            "correctness": None,
            "checker_status": "empty_output",
            "checker_failure_reason": "empty_response",
            "parsed_answer": None,
            "normalized_candidate_answer": None,
            "boxed_answer_count": 0,
            "eligible_for_supervision": False,
        }

    literals, route, box_count = _extract_candidate_literals(
        response,
        normalize_boxed_prose=checker_version == CHECKER_VERSION,
    )
    parsed_options = [numeric_options(value) for value in literals]
    usable = [
        (value, options) for value, options in zip(literals, parsed_options) if options
    ]
    if box_count > 1 and usable:
        union_values = set().union(*(options for _, options in usable))
        pairwise_equal = all(options == usable[0][1] for _, options in usable[1:])
        if len(union_values) > 1 and not pairwise_equal:
            return {
                **base,
                "numeric_value_match": None,
                "correctness": None,
                "checker_status": "ambiguous_multiple_answers",
                "checker_failure_reason": "conflicting_boxed_numeric_values",
                "parsed_answer": literals[-1] if literals else None,
                "normalized_candidate_answer": None,
                "boxed_answer_count": box_count,
                "parse_route": route,
                "eligible_for_supervision": False,
            }
    if not usable:
        return {
            **base,
            "numeric_value_match": 0,
            "correctness": 0,
            "checker_status": "parse_failed",
            "checker_failure_reason": "candidate_not_numeric",
            "parsed_answer": literals[-1] if literals else None,
            "normalized_candidate_answer": None,
            "boxed_answer_count": box_count,
            "parse_route": route,
            "eligible_for_supervision": True,
        }
    parsed, candidate_options = usable[-1]
    matched = bool(candidate_options & reference_options)
    return {
        **base,
        "numeric_value_match": int(matched),
        "correctness": int(matched),
        "checker_status": "numeric_match" if matched else "numeric_mismatch",
        "checker_failure_reason": None,
        "parsed_answer": parsed,
        "normalized_candidate_answer": sorted(
            f"{value.numerator}/{value.denominator}" for value in candidate_options
        ),
        "normalized_reference_answer": sorted(
            f"{value.numerator}/{value.denominator}" for value in reference_options
        ),
        "boxed_answer_count": box_count,
        "parse_route": route,
        "eligible_for_supervision": True,
    }


# ---------------------------------------------------------------------------
# Exact-token unitization
# ---------------------------------------------------------------------------


_HEADER = re.compile(
    r"^(?P<header>(?:#{1,6}\s*)?(?:step\s+\d+|final\s+answer|answer|solution)\s*:)",
    flags=re.IGNORECASE,
)
_LIST_PREFIX = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_ABBREVIATIONS = {"e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs."}


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _protected_period(text: str, index: int, line_start: int) -> bool:
    if index > line_start and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return True
    prefix = text[max(line_start, index - 4) : index + 1].casefold()
    return any(prefix.endswith(value) for value in _ABBREVIATIONS)


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    brace_depth = 0
    index = start
    while index < end:
        char = text[index]
        if char == "{":
            brace_depth += 1
        elif char == "}" and brace_depth:
            brace_depth -= 1
        boundary = False
        if brace_depth == 0 and char in "!?;":
            boundary = True
        elif (
            brace_depth == 0
            and char == "."
            and not _protected_period(text, index, start)
        ):
            boundary = True
        if boundary and (index + 1 == end or text[index + 1].isspace()):
            span = _trim_span(text, cursor, index + 1)
            if span is not None:
                spans.append(span)
            cursor = index + 1
        index += 1
    span = _trim_span(text, cursor, end)
    if span is not None:
        spans.append(span)
    return spans


def material_claim_char_spans(response: str) -> list[tuple[int, int]]:
    """Split visible reasoning into deterministic candidate claim spans."""

    if not isinstance(response, str) or not response.strip():
        return []
    spans: list[tuple[int, int]] = []
    for line in re.finditer(r"[^\r\n]+", response):
        trimmed = _trim_span(response, line.start(), line.end())
        if trimmed is None:
            continue
        start, end = trimmed
        line_text = response[start:end]
        header = _HEADER.match(line_text)
        if header is not None:
            start += header.end("header")
            while start < end and response[start].isspace():
                start += 1
        else:
            prefix = _LIST_PREFIX.match(line_text)
            if prefix is not None:
                start += prefix.end()
        if start < end:
            spans.extend(_sentence_spans(response, start, end))
    previous = -1
    for start, end in spans:
        if not 0 <= start < end <= len(response) or start < previous:
            raise ValueError("material claim segmentation produced invalid spans")
        previous = end
    return spans


def validate_visible_token_mapping(
    *,
    response: str,
    output_token_ids: Sequence[int],
    encoded_token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    trailing_token_decodes_to_empty: Sequence[bool],
) -> list[tuple[int, int]]:
    frozen = validate_token_ids(
        output_token_ids, field="output_token_ids", row_id="unitizer"
    )
    encoded = validate_token_ids(
        encoded_token_ids, field="encoded_token_ids", row_id="unitizer"
    )
    if frozen[: len(encoded)] != encoded:
        raise ValueError("visible response re-tokenization differs from frozen IDs")
    trailing = frozen[len(encoded) :]
    if len(trailing) != len(trailing_token_decodes_to_empty):
        raise ValueError("trailing token audit length differs from frozen IDs")
    if not all(bool(value) for value in trailing_token_decodes_to_empty):
        raise ValueError("a trailing token decodes to visible content")
    normalized = [(int(pair[0]), int(pair[1])) for pair in offsets]
    if len(normalized) != len(encoded):
        raise ValueError("offset count differs from visible encoded token count")
    if encoded:
        if normalized[0][0] != 0 or normalized[-1][1] != len(response):
            raise ValueError("visible token offsets do not cover the saved response")
        if any(not 0 <= start <= end <= len(response) for start, end in normalized):
            raise ValueError("visible token offset is out of range")
    return normalized


def _char_to_token_span(
    char_span: tuple[int, int], offsets: Sequence[tuple[int, int]]
) -> tuple[int, int]:
    start, end = char_span
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not overlapping:
        raise ValueError("material claim does not overlap a visible token")
    return overlapping[0], overlapping[-1] + 1


def unitize_exact_tokens(
    *,
    response: str,
    output_token_ids: Sequence[int],
    encoded_token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    trailing_token_decodes_to_empty: Sequence[bool],
) -> dict[str, Any]:
    """Partition the complete saved output-token axis into claim/non-claim units."""

    frozen = validate_token_ids(
        output_token_ids, field="output_token_ids", row_id="unitizer"
    )
    visible_offsets = validate_visible_token_mapping(
        response=response,
        output_token_ids=frozen,
        encoded_token_ids=encoded_token_ids,
        offsets=offsets,
        trailing_token_decodes_to_empty=trailing_token_decodes_to_empty,
    )
    claim_chars = material_claim_char_spans(response)
    claim_tokens: list[tuple[int, int, tuple[int, int]]] = []
    for char_span in claim_chars:
        token_start, token_end = _char_to_token_span(char_span, visible_offsets)
        if claim_tokens and token_start < claim_tokens[-1][1]:
            raise ValueError(
                "token boundary fuses independently segmented material claims"
            )
        claim_tokens.append((token_start, token_end, char_span))

    units: list[dict[str, Any]] = []

    def add_unit(
        kind: str,
        token_start: int,
        token_end: int,
        char_span: tuple[int, int] | None,
    ) -> None:
        if token_start >= token_end:
            return
        if char_span is None:
            visible = [
                visible_offsets[index]
                for index in range(token_start, min(token_end, len(visible_offsets)))
                if visible_offsets[index][1] > visible_offsets[index][0]
            ]
            if visible:
                char_start = min(start for start, _ in visible)
                char_end = max(end for _, end in visible)
            else:
                char_start = char_end = len(response)
        else:
            char_start, char_end = char_span
        units.append(
            {
                "unit_index": len(units),
                "kind": kind,
                "text": response[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_start,
                "token_end": token_end,
            }
        )

    cursor = 0
    for token_start, token_end, char_span in claim_tokens:
        add_unit("non_claim", cursor, token_start, None)
        add_unit("material_claim", token_start, token_end, char_span)
        cursor = token_end
    add_unit("non_claim", cursor, len(frozen), None)
    if not units:
        add_unit("non_claim", 0, len(frozen), None)

    expected = 0
    visible_covered = [False] * len(response)
    for expected_index, unit in enumerate(units):
        if unit["unit_index"] != expected_index:
            raise AssertionError("unit indices are not contiguous")
        if unit["token_start"] != expected or unit["token_end"] <= expected:
            raise ValueError("unit token ranges do not form a contiguous partition")
        expected = unit["token_end"]
        for index in range(unit["char_start"], unit["char_end"]):
            visible_covered[index] = True
    if expected != len(frozen):
        raise ValueError("unit token ranges do not cover the full output axis")
    if any(
        not covered
        for index, covered in enumerate(visible_covered)
        if not response[index].isspace()
    ):
        raise ValueError(
            "unit char ranges do not cover all visible non-whitespace text"
        )
    return {
        "unitizer_version": UNITIZER_VERSION,
        "status": "ok",
        "output_token_count": len(frozen),
        "visible_token_count": len(encoded_token_ids),
        "trailing_invisible_token_count": len(frozen) - len(encoded_token_ids),
        "material_claim_count": sum(unit["kind"] == "material_claim" for unit in units),
        "units": units,
    }


def tokenize_visible_response(
    tokenizer: Any, response: str, output_token_ids: Sequence[int]
) -> dict[str, Any]:
    """Map visible characters onto the frozen IDs without replacing those IDs.

    Fast-tokenizer offsets are preferred.  A decoded string can, however, have
    more than one valid tokenization (for example ``BBB`` or punctuation next
    to LaTeX).  In that case canonical re-tokenization is not the generated
    token axis, so fall back to audited prefix decoding of the frozen IDs.
    """

    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    encoded_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    frozen = validate_token_ids(
        output_token_ids, field="output_token_ids", row_id="tokenizer"
    )
    trailing = frozen[len(encoded_ids) :]
    trailing_empty = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        == ""
        for token_id in trailing
    ]
    if frozen[: len(encoded_ids)] == encoded_ids and all(trailing_empty):
        return {
            "encoded_token_ids": encoded_ids,
            "offsets": offsets,
            "trailing_token_decodes_to_empty": trailing_empty,
            "mapping_mode": "canonical_fast_offsets",
        }

    def decode(token_ids: Sequence[int]) -> str:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    visible_end = len(frozen)
    while visible_end > 0 and decode([frozen[visible_end - 1]]) == "":
        visible_end -= 1
    visible_ids = frozen[:visible_end]
    trailing_ids = frozen[visible_end:]
    if decode(frozen) != response:
        raise ValueError("frozen IDs do not decode to the saved visible response")

    prefix_offsets: list[tuple[int, int]] = []
    previous = ""
    for token_end in range(1, visible_end + 1):
        current = decode(visible_ids[:token_end])
        if not current.startswith(previous) or not response.startswith(current):
            raise ValueError("frozen token prefixes do not map monotonically to text")
        if len(current) == len(previous):
            raise ValueError("an internal frozen token decodes to no visible content")
        prefix_offsets.append((len(previous), len(current)))
        previous = current
    if previous != response:
        raise ValueError("frozen visible token prefixes do not cover the response")
    return {
        "encoded_token_ids": visible_ids,
        "offsets": prefix_offsets,
        "trailing_token_decodes_to_empty": [
            decode([token_id]) == "" for token_id in trailing_ids
        ],
        "mapping_mode": "frozen_prefix_decode_fallback",
    }


# ---------------------------------------------------------------------------
# Rollout validation and deterministic proposals
# ---------------------------------------------------------------------------


def validate_rollout_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row_id = row.get("id")
    query_id = row.get("query_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("rollout id must be a non-empty string")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"{row_id}: query_id must be non-empty")
    candidate_index = row.get("candidate_index")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, Integral):
        raise ValueError(f"{row_id}: candidate_index must be an integer")
    prompt_ids = validate_token_ids(
        row.get("prompt_token_ids"), field="prompt_token_ids", row_id=row_id
    )
    output_ids = validate_token_ids(
        row.get("output_token_ids"), field="output_token_ids", row_id=row_id
    )
    if not isinstance(row.get("response"), str):
        raise ValueError(f"{row_id}: response must be a string")
    normalized = dict(row)
    normalized["candidate_index"] = int(candidate_index)
    normalized["prompt_token_ids"] = prompt_ids
    normalized["output_token_ids"] = output_ids
    return normalized


def validate_rollout_population(
    rows: Sequence[Mapping[str, Any]], *, candidate_count: int
) -> dict[str, Any]:
    normalized = [validate_rollout_row(row) for row in rows]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_query[row["query_id"]].append(row)
    for query_id, query_rows in by_query.items():
        indices = sorted(row["candidate_index"] for row in query_rows)
        if indices != list(range(candidate_count)):
            raise ValueError(
                f"{query_id}: candidate indices are not 0..{candidate_count - 1}"
            )
        prompt_hashes = {
            canonical_sha256(row["prompt_token_ids"]) for row in query_rows
        }
        if len(prompt_hashes) != 1:
            raise ValueError(f"{query_id}: candidates do not share exact prompt IDs")
    invalid = sum(not row.get("eligible_for_supervision", True) for row in normalized)
    return {
        "queries": len(by_query),
        "rows": len(normalized),
        "candidate_count": candidate_count,
        "ineligible_rows": invalid,
        "ineligible_fraction": invalid / len(normalized) if normalized else 0.0,
    }


def _material_count(row: Mapping[str, Any]) -> int:
    return sum(unit.get("kind") == "material_claim" for unit in row.get("units", []))


def _mechanism_eligible(row: Mapping[str, Any], *, min_material_units: int = 4) -> bool:
    return bool(
        row.get("eligible_for_supervision")
        and row.get("unitization_status", row.get("unitizer_status", "ok")) == "ok"
        and _material_count(row) >= min_material_units
    )


def _near_copy(left: str, right: str, threshold: float = 0.92) -> bool:
    left_norm = normalize_question(left)
    right_norm = normalize_question(right)
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= threshold


_CONSISTENCY_MATH_REGION = re.compile(r"\$[^$]*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")
_CONSISTENCY_MATH_TOKEN = re.compile(r"[a-z]+|[-+]?\d+(?:\.\d+)?|[=+*/^%()-]")
_CONSISTENCY_SURFACE_STOPWORDS = frozenset(
    """
    a an and answer are as be by calculate can determine final find first for
    fourth from has have in is it let lets need now of on second so step that
    the then therefore third this thus to total using we which will with
    """.split()
)


def _consistency_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _consistency_math_trace(text: str) -> list[str]:
    """Extract a conservative ordered math-token trace from equations/TeX."""

    normalized = (
        _consistency_normalize(text)
        .replace(r"\times", "*")
        .replace(r"\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
    )
    tokens: list[str] = []
    for line in normalized.splitlines():
        regions = _CONSISTENCY_MATH_REGION.findall(line)
        if "=" not in line and not regions:
            continue
        payload = " ".join(regions) if regions else line
        payload = re.sub(r"\\(?:boxed|dfrac|frac|left|right|text|tfrac)", " ", payload)
        tokens.extend(_CONSISTENCY_MATH_TOKEN.findall(payload))
    return tokens


def _consistency_numeric_trace(text: str) -> list[str]:
    return [
        re.sub(r"[\s,]", "", literal)
        for literal in _NUMBER.findall(_consistency_normalize(text))
    ]


def _consistency_surface_tokens(text: str) -> list[str]:
    normalized = _CONSISTENCY_MATH_REGION.sub(" ", _consistency_normalize(text))
    normalized = _NUMBER.sub(" ", normalized)
    normalized = re.sub(r"[^a-z]+", " ", normalized)
    return [
        word
        for word in normalized.split()
        if len(word) > 1 and word not in _CONSISTENCY_SURFACE_STOPWORDS
    ]


def _sequence_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _bigram_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_bigrams = {
        (left[index], left[index + 1]) for index in range(max(0, len(left) - 1))
    }
    right_bigrams = {
        (right[index], right[index + 1]) for index in range(max(0, len(right) - 1))
    }
    union = left_bigrams | right_bigrams
    return len(left_bigrams & right_bigrams) / len(union) if union else 1.0


def consistency_surface_bigrams(text: str) -> frozenset[tuple[str, str]]:
    """Return the exact frozen v5 non-math surface-bigram set.

    Scale-v6.1 uses this public helper only as a lossless candidate-retrieval
    index.  Final edge admission still calls ``consistency_mechanical_metrics``
    so the metric and thresholds remain unchanged.
    """

    tokens = _consistency_surface_tokens(text)
    return frozenset(
        (tokens[index], tokens[index + 1])
        for index in range(max(0, len(tokens) - 1))
    )


def consistency_mechanical_metrics(
    left_response: str,
    right_response: str,
    *,
    left_token_count: int,
    right_token_count: int,
) -> dict[str, Any]:
    """Return the frozen v5 math/path and non-math surface diagnostics."""

    if min(left_token_count, right_token_count) <= 0:
        raise ValueError("consistency token counts must be positive")
    left_math = _consistency_math_trace(left_response)
    right_math = _consistency_math_trace(right_response)
    left_numeric = _consistency_numeric_trace(left_response)
    right_numeric = _consistency_numeric_trace(right_response)
    left_surface = _consistency_surface_tokens(left_response)
    right_surface = _consistency_surface_tokens(right_response)
    return {
        "metric_version": "clir_consistency_mechanical_v1",
        "token_length_ratio": max(left_token_count, right_token_count)
        / min(left_token_count, right_token_count),
        "math_trace_similarity": _sequence_similarity(left_math, right_math),
        "numeric_trace_similarity": _sequence_similarity(left_numeric, right_numeric),
        "surface_bigram_jaccard": _bigram_jaccard(left_surface, right_surface),
        "left_math_token_count": len(left_math),
        "right_math_token_count": len(right_math),
        "left_numeric_token_count": len(left_numeric),
        "right_numeric_token_count": len(right_numeric),
        "left_surface_bigram_count": max(0, len(left_surface) - 1),
        "right_surface_bigram_count": max(0, len(right_surface) - 1),
    }


def build_mechanical_consistency_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    min_material_units: int,
    min_length_ratio: float,
    max_length_ratio: float,
    min_math_tokens: int,
    min_numeric_tokens: int,
    min_surface_bigrams: int,
    min_math_similarity: float,
    min_numeric_similarity: float,
    min_surface_jaccard: float,
    max_surface_jaccard: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one deterministic mechanically admitted C pair per query."""

    if not 0 <= min_surface_jaccard <= max_surface_jaccard <= 1:
        raise ValueError("surface Jaccard band must lie inside [0, 1]")
    if not 0 <= min_math_similarity <= 1:
        raise ValueError("math similarity threshold must lie inside [0, 1]")
    if not 0 <= min_numeric_similarity <= 1:
        raise ValueError("numeric similarity threshold must lie inside [0, 1]")
    if not 1 <= min_length_ratio <= max_length_ratio:
        raise ValueError("length-ratio band is invalid")

    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    mechanically_eligible_candidates = 0
    for row in rows:
        if row.get("source") != source:
            continue
        if not _mechanism_eligible(row, min_material_units=min_material_units):
            continue
        if row.get("numeric_value_match") != 1:
            continue
        mechanically_eligible_candidates += 1
        by_query[str(row["query_id"])].append(row)

    rejection_counts: Counter[str] = Counter()
    admitted_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    considered_pairs = 0
    for query_id, query_rows in by_query.items():
        ordered = sorted(query_rows, key=lambda row: int(row["candidate_index"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                considered_pairs += 1
                if left.get("normalized_candidate_answer") != right.get(
                    "normalized_candidate_answer"
                ):
                    rejection_counts["normalized_answer_mismatch"] += 1
                    continue
                metrics = consistency_mechanical_metrics(
                    str(left["response"]),
                    str(right["response"]),
                    left_token_count=len(left["output_token_ids"]),
                    right_token_count=len(right["output_token_ids"]),
                )
                checks = (
                    (
                        metrics["token_length_ratio"] >= min_length_ratio,
                        "length_ratio_below_min",
                    ),
                    (
                        metrics["token_length_ratio"] <= max_length_ratio,
                        "length_ratio_above_max",
                    ),
                    (
                        min(
                            metrics["left_math_token_count"],
                            metrics["right_math_token_count"],
                        )
                        >= min_math_tokens,
                        "insufficient_math_trace",
                    ),
                    (
                        min(
                            metrics["left_numeric_token_count"],
                            metrics["right_numeric_token_count"],
                        )
                        >= min_numeric_tokens,
                        "insufficient_numeric_trace",
                    ),
                    (
                        min(
                            metrics["left_surface_bigram_count"],
                            metrics["right_surface_bigram_count"],
                        )
                        >= min_surface_bigrams,
                        "insufficient_surface_trace",
                    ),
                    (
                        metrics["math_trace_similarity"] >= min_math_similarity,
                        "math_similarity_below_min",
                    ),
                    (
                        metrics["numeric_trace_similarity"] >= min_numeric_similarity,
                        "numeric_similarity_below_min",
                    ),
                    (
                        metrics["surface_bigram_jaccard"] >= min_surface_jaccard,
                        "surface_similarity_below_min",
                    ),
                    (
                        metrics["surface_bigram_jaccard"] <= max_surface_jaccard,
                        "surface_similarity_above_max",
                    ),
                )
                failed_reason = next(
                    (reason for passed, reason in checks if not passed), None
                )
                if failed_reason is not None:
                    rejection_counts[failed_reason] += 1
                    continue
                pair_priority = stable_priority(
                    "clir-C-mechanical-pair-v1",
                    query_id,
                    left["candidate_index"],
                    right["candidate_index"],
                )
                admitted_by_query[query_id].append(
                    {
                        "schema_version": "clir-consistency-proposal-v5",
                        "proposal_id": pair_priority,
                        "query_id": query_id,
                        "left_id": str(left["id"]),
                        "right_id": str(right["id"]),
                        "left_candidate_index": int(left["candidate_index"]),
                        "right_candidate_index": int(right["candidate_index"]),
                        "mechanical_metrics": metrics,
                        "selection_priority": pair_priority,
                    }
                )

    chosen = [
        min(pairs, key=lambda row: row["selection_priority"])
        for pairs in admitted_by_query.values()
    ]
    chosen.sort(
        key=lambda row: stable_priority("clir-C-mechanical-query-v1", row["query_id"])
    )
    report = {
        "metric_version": "clir_consistency_mechanical_v1",
        "input_rows": len(rows),
        "mechanically_eligible_candidates": mechanically_eligible_candidates,
        "queries_with_eligible_candidates": len(by_query),
        "considered_pairs": considered_pairs,
        "admitted_pairs_before_one_per_query": sum(
            len(pairs) for pairs in admitted_by_query.values()
        ),
        "admitted_query_distinct_pairs": len(chosen),
        "first_failure_counts": dict(sorted(rejection_counts.items())),
    }
    return chosen, report


def build_consistency_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    proposal_count: int,
    min_length_ratio: float = 1.25,
    max_length_ratio: float = 3.0,
) -> list[dict[str, Any]]:
    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source") == "gsm8k" and _mechanism_eligible(row):
            by_query[str(row["query_id"])].append(row)
    chosen: list[dict[str, Any]] = []
    for query_id, query_rows in by_query.items():
        pairs: list[dict[str, Any]] = []
        ordered = sorted(query_rows, key=lambda row: int(row["candidate_index"]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.get("numeric_value_match") != right.get("numeric_value_match"):
                    continue
                if left.get("normalized_candidate_answer") != right.get(
                    "normalized_candidate_answer"
                ):
                    continue
                lengths = sorted(
                    (len(left["output_token_ids"]), len(right["output_token_ids"]))
                )
                ratio = lengths[1] / max(1, lengths[0])
                if not min_length_ratio <= ratio <= max_length_ratio:
                    continue
                if _near_copy(str(left["response"]), str(right["response"])):
                    continue
                first, second = sorted(
                    (left, right), key=lambda row: int(row["candidate_index"])
                )
                pair_priority = stable_priority(
                    "clir-C-proposal-v2",
                    query_id,
                    first["candidate_index"],
                    second["candidate_index"],
                )
                pairs.append(
                    {
                        "schema_version": "clir-consistency-proposal-v2",
                        "proposal_id": pair_priority,
                        "query_id": query_id,
                        "left_id": first["id"],
                        "right_id": second["id"],
                        "left_candidate_index": first["candidate_index"],
                        "right_candidate_index": second["candidate_index"],
                        "token_length_ratio": ratio,
                        "selection_priority": pair_priority,
                    }
                )
        if pairs:
            chosen.append(min(pairs, key=lambda row: row["selection_priority"]))
    chosen.sort(key=lambda row: stable_priority("clir-C-query-v2", row["query_id"]))
    if len(chosen) < proposal_count:
        raise ValueError(
            f"FAIL_YIELD: need {proposal_count} consistency queries, found {len(chosen)}"
        )
    return chosen[:proposal_count]


def build_h_prior_proposals(
    rows: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[tuple[str, int], int],
    consistency_proposals: Sequence[Mapping[str, Any]],
    numeric_mismatch_status_only: bool = False,
) -> list[dict[str, Any]]:
    c_view_ids = {
        str(proposal[field])
        for proposal in consistency_proposals
        for field in ("left_id", "right_id")
    }
    by_stratum: dict[tuple[str, int], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("id") in c_view_ids or not _mechanism_eligible(row):
            continue
        match = row.get("numeric_value_match")
        if match not in {0, 1}:
            continue
        if (
            numeric_mismatch_status_only
            and match == 0
            and row.get("checker_status") != "numeric_mismatch"
        ):
            continue
        key = (str(row.get("source")), int(match))
        by_stratum[key][str(row["query_id"])].append(row)

    selected: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    for stratum in sorted(quotas):
        requested = int(quotas[stratum])
        candidates: list[Mapping[str, Any]] = []
        for query_id, query_rows in by_stratum[stratum].items():
            if query_id in used_queries:
                continue
            candidates.append(
                min(
                    query_rows,
                    key=lambda row: stable_priority(
                        "clir-HP-candidate-v2", row["query_id"], row["candidate_index"]
                    ),
                )
            )
        candidates.sort(
            key=lambda row: stable_priority("clir-HP-query-v2", row["query_id"])
        )
        if len(candidates) < requested:
            raise ValueError(
                f"FAIL_YIELD: stratum {stratum} needs {requested}, found {len(candidates)}"
            )
        for row in candidates[:requested]:
            used_queries.add(str(row["query_id"]))
            selected.append(
                {
                    "schema_version": "clir-h-prior-proposal-v2",
                    "proposal_id": stable_priority("clir-HP-proposal-v2", row["id"]),
                    "id": row["id"],
                    "query_id": row["query_id"],
                    "candidate_index": row["candidate_index"],
                    "source": row["source"],
                    "numeric_stratum": int(row["numeric_value_match"]),
                    "selection_priority": stable_priority(
                        "clir-HP-final-v2", row["query_id"], row["candidate_index"]
                    ),
                }
            )
    selected.sort(key=lambda row: row["selection_priority"])
    if len({row["query_id"] for row in selected}) != len(selected):
        raise AssertionError("H/P proposals are not query-distinct")
    return selected


# ---------------------------------------------------------------------------
# Blind annotations, agreement, adjudication, and token materialization
# ---------------------------------------------------------------------------


def public_unit_item(row: Mapping[str, Any]) -> dict[str, Any]:
    units = [
        {
            "unit_index": int(unit["unit_index"]),
            "kind": str(unit["kind"]),
            "text": str(unit["text"]),
        }
        for unit in row["units"]
    ]
    return {
        "item_id": str(row["id"]),
        "query_id": str(row["query_id"]),
        "source": str(row["source"]),
        "problem": str(row["question"]),
        "trajectory": str(row["response"]),
        "units": units,
        "output_token_ids_sha256": canonical_sha256(row["output_token_ids"]),
    }


def consistency_item(
    proposal: Mapping[str, Any], rows_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    left = rows_by_id[str(proposal["left_id"])]
    right = rows_by_id[str(proposal["right_id"])]
    return {
        "item_id": str(proposal["proposal_id"]),
        "query_id": str(proposal["query_id"]),
        "problem": str(left["question"]),
        "left": {
            "id": left["id"],
            "trajectory": left["response"],
            "units": public_unit_item(left)["units"],
        },
        "right": {
            "id": right["id"],
            "trajectory": right["response"],
            "units": public_unit_item(right)["units"],
        },
    }


def _index_list(
    value: Any, *, field: str, units: Sequence[Mapping[str, Any]]
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    valid = {
        int(unit["unit_index"]) for unit in units if unit["kind"] == "material_claim"
    }
    output: list[int] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, Integral):
            raise ValueError(f"{field} must contain only integers")
        integer = int(element)
        if integer not in valid:
            raise ValueError(f"{field} references a non-material or missing unit")
        output.append(integer)
    if output != sorted(set(output)):
        raise ValueError(f"{field} must be sorted and unique")
    return output


def validate_annotation(
    task: str, annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation must be an object")
    if annotation.get("item_id") != item.get("item_id"):
        raise ValueError("annotation item_id does not match item")
    confidence = annotation.get("confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("annotation confidence is invalid")
    rationale = annotation.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("annotation rationale must be non-empty")
    base = {
        "task": task,
        "item_id": str(annotation["item_id"]),
        "confidence": str(confidence),
        "rationale": rationale.strip(),
    }
    if task == "consistency":
        decision = annotation.get("decision")
        if decision not in {"accept", "reject", "review"}:
            raise ValueError("consistency decision is invalid")
        return {**base, "decision": str(decision)}
    units = item.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("unit annotation item has no units")
    if task == "hallucination":
        status = annotation.get("status")
        if status not in H_STATUS_VALUES:
            raise ValueError("hallucination status is invalid")
        onset = annotation.get("first_bad_unit_index")
        if status == "hallucinated":
            indices = _index_list([onset], field="first_bad_unit_index", units=units)
            onset = indices[0]
        elif onset is not None:
            raise ValueError("only hallucinated labels may set first_bad_unit_index")
        return {**base, "status": str(status), "first_bad_unit_index": onset}
    if task == "prior":
        eligibility = annotation.get("eligibility")
        if eligibility not in PRIOR_ELIGIBILITY_VALUES:
            raise ValueError("prior eligibility is invalid")
        key = _index_list(
            annotation.get("key_unit_indices"), field="key_unit_indices", units=units
        )
        complete = _index_list(
            annotation.get("complete_unit_indices"),
            field="complete_unit_indices",
            units=units,
        )
        if eligibility == "usable":
            if not key or not complete:
                raise ValueError("usable prior labels require non-empty sets")
            if not set(key).issubset(complete):
                raise ValueError("key must be a subset of complete")
        elif key or complete:
            raise ValueError("ineligible prior labels require empty sets")
        return {
            **base,
            "eligibility": str(eligibility),
            "key_unit_indices": key,
            "complete_unit_indices": complete,
        }
    raise ValueError(f"unsupported annotation task {task!r}")


def annotation_signature(task: str, annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    if task == "consistency":
        return (annotation["decision"],)
    if task == "hallucination":
        return (annotation["status"], annotation.get("first_bad_unit_index"))
    if task == "prior":
        return (
            annotation["eligibility"],
            tuple(annotation["key_unit_indices"]),
            tuple(annotation["complete_unit_indices"]),
        )
    raise ValueError(f"unsupported task {task!r}")


def validate_annotator_roster(
    annotators: Sequence[Mapping[str, Any]], *, generator_family: str
) -> None:
    if len(annotators) < 2:
        raise ValueError("at least two primary annotators are required")
    families = []
    for annotator in annotators:
        for field in ("provider", "model_id", "model_family", "revision"):
            if not isinstance(annotator.get(field), str) or not annotator[field]:
                raise ValueError(f"annotator roster is missing {field}")
        families.append(str(annotator["model_family"]).casefold())
    if len(set(families[:2])) != 2:
        raise ValueError("primary annotators must use different model families")
    if generator_family.casefold() in set(families):
        raise ValueError("annotator family overlaps the generator/backbone family")


def _set_f1(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_set)
    recall = overlap / len(right_set)
    return 2 * precision * recall / (precision + recall)


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def agreement_report(
    task: str,
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_a = {str(row["item_id"]): row for row in labels_a}
    by_b = {str(row["item_id"]): row for row in labels_b}
    if set(by_a) != set(by_b):
        raise ValueError("A/B item populations differ")
    item_ids = sorted(by_a)
    agreements = [
        annotation_signature(task, by_a[item_id])
        == annotation_signature(task, by_b[item_id])
        for item_id in item_ids
    ]
    report: dict[str, Any] = {
        "task": task,
        "items": len(item_ids),
        "exact_target_agree": sum(agreements),
        "exact_target_agreement": sum(agreements) / len(item_ids) if item_ids else None,
        "disagreement_item_ids": [
            item_id for item_id, agree in zip(item_ids, agreements) if not agree
        ],
    }
    if task == "consistency":
        left = [str(by_a[item_id]["decision"]) for item_id in item_ids]
        right = [str(by_b[item_id]["decision"]) for item_id in item_ids]
        report["decision_kappa"] = cohen_kappa(left, right)
        report["a_decisions"] = dict(Counter(left))
        report["b_decisions"] = dict(Counter(right))
    elif task == "hallucination":
        left = [str(by_a[item_id]["status"]) for item_id in item_ids]
        right = [str(by_b[item_id]["status"]) for item_id in item_ids]
        path_agree = [a == b and a in FINAL_H_VALUES for a, b in zip(left, right)]
        common_positive = [
            item_id
            for item_id in item_ids
            if by_a[item_id]["status"] == by_b[item_id]["status"] == "hallucinated"
        ]
        common_clean = [
            item_id
            for item_id in item_ids
            if by_a[item_id]["status"] == by_b[item_id]["status"] == "clean"
        ]
        a_positive = sum(value == "hallucinated" for value in left)
        b_positive = sum(value == "hallucinated" for value in right)
        a_clean = sum(value == "clean" for value in left)
        b_clean = sum(value == "clean" for value in right)
        exact_onset = sum(
            by_a[item_id]["first_bad_unit_index"]
            == by_b[item_id]["first_bad_unit_index"]
            for item_id in common_positive
        )
        plus_one = sum(
            abs(
                int(by_a[item_id]["first_bad_unit_index"])
                - int(by_b[item_id]["first_bad_unit_index"])
            )
            <= 1
            for item_id in common_positive
        )
        report.update(
            {
                "raw_path_agree": sum(path_agree),
                "raw_path_agreement": (
                    sum(path_agree) / len(item_ids) if item_ids else None
                ),
                "path_kappa": cohen_kappa(left, right),
                "common_positive": len(common_positive),
                "common_clean": len(common_clean),
                "positive_specific_agreement": len(common_positive)
                / max(1, a_positive, b_positive),
                "clean_specific_agreement": len(common_clean)
                / max(1, a_clean, b_clean),
                "positive_rate_absolute_gap": (
                    abs(a_positive - b_positive) / len(item_ids) if item_ids else None
                ),
                "exact_onset_agreement": (
                    exact_onset / len(common_positive) if common_positive else None
                ),
                "plus_minus_one_onset_agreement": (
                    plus_one / len(common_positive) if common_positive else None
                ),
                "a_statuses": dict(Counter(left)),
                "b_statuses": dict(Counter(right)),
            }
        )
    elif task == "prior":
        eligibility_agree = sum(
            by_a[item_id]["eligibility"] == by_b[item_id]["eligibility"]
            for item_id in item_ids
        )
        usable = [
            item_id
            for item_id in item_ids
            if by_a[item_id]["eligibility"] == by_b[item_id]["eligibility"] == "usable"
        ]
        report.update(
            {
                "eligibility_agree": eligibility_agree,
                "eligibility_agreement": (
                    eligibility_agree / len(item_ids) if item_ids else None
                ),
                "usable_overlap": len(usable),
                "key_macro_f1": (
                    sum(
                        _set_f1(
                            by_a[item_id]["key_unit_indices"],
                            by_b[item_id]["key_unit_indices"],
                        )
                        for item_id in usable
                    )
                    / len(usable)
                    if usable
                    else None
                ),
                "complete_macro_f1": (
                    sum(
                        _set_f1(
                            by_a[item_id]["complete_unit_indices"],
                            by_b[item_id]["complete_unit_indices"],
                        )
                        for item_id in usable
                    )
                    / len(usable)
                    if usable
                    else None
                ),
            }
        )
    return report


def resolve_blind_labels(
    *,
    task: str,
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]] = (),
    label_tier: str = LABEL_TIER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_a = {str(row["item_id"]): dict(row) for row in labels_a}
    by_b = {str(row["item_id"]): dict(row) for row in labels_b}
    if set(by_a) != set(by_b):
        raise ValueError("A/B label populations differ")
    by_adjudication = {str(row["item_id"]): dict(row) for row in adjudications}
    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    source_counts: Counter[str] = Counter()
    adjudication_resolutions: Counter[str] = Counter()
    for item_id in sorted(by_a):
        a, b = by_a[item_id], by_b[item_id]
        if (
            annotation_signature(task, a) == annotation_signature(task, b)
            and a["confidence"] != "low"
            and b["confidence"] != "low"
        ):
            final = dict(a)
            final["label_source"] = "auto_agree"
        else:
            adjudication = by_adjudication.get(item_id)
            if adjudication is None or adjudication.get("resolution") == "unresolved":
                unresolved.append(item_id)
                continue
            resolution = adjudication.get("resolution")
            if resolution == "adopt_a":
                final = dict(a)
            elif resolution == "adopt_b":
                final = dict(b)
            elif resolution == "synthesize":
                synthesized = adjudication.get("annotation")
                if not isinstance(synthesized, Mapping):
                    raise ValueError("synthesize adjudication lacks annotation")
                # It was already validated by the caller against the same item.
                final = dict(synthesized)
            else:
                raise ValueError(f"invalid adjudication resolution {resolution!r}")
            if not adjudication.get("independent_answer_completed", False):
                raise ValueError("adjudicator did not answer independently first")
            final["label_source"] = "adjudicated"
            final["adjudication_resolution"] = resolution
            adjudication_resolutions[str(resolution)] += 1
        final["label_tier"] = label_tier
        resolved.append(final)
        source_counts[final["label_source"]] += 1
    return resolved, {
        "task": task,
        "natural_denominator": len(by_a),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "unresolved_item_ids": unresolved,
        "label_sources": dict(source_counts),
        "adjudication_resolutions": dict(adjudication_resolutions),
        "adjudication_fraction": (
            (source_counts["adjudicated"] + len(unresolved)) / len(by_a)
            if by_a
            else None
        ),
    }


def materialize_h_label(
    label: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    label_tier: str = LABEL_TIER,
) -> dict[str, Any]:
    status = label.get("status")
    if status not in FINAL_H_VALUES:
        raise ValueError("only clean/hallucinated H labels can be materialized")
    output = {
        "path_hallucinated": int(status == "hallucinated"),
        "hallucination_target_name": "first_bad_unit",
        "hallucination_compatibility_token_name": "first_bad_unit_start_token",
        "hallucination_label_source": label["label_source"],
        "hallucination_label_tier": label_tier,
    }
    if status == "clean":
        output["hallucination_onset"] = -1
        output["first_bad_unit_index"] = None
    else:
        unit_index = int(label["first_bad_unit_index"])
        unit = next(
            (unit for unit in row["units"] if int(unit["unit_index"]) == unit_index),
            None,
        )
        if unit is None or unit["kind"] != "material_claim":
            raise ValueError("first_bad_unit_index does not identify a material unit")
        output["hallucination_onset"] = int(unit["token_start"])
        output["first_bad_unit_index"] = unit_index
    return output


def materialize_prior_label(
    label: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    label_tier: str = LABEL_TIER,
) -> dict[str, Any]:
    if label.get("eligibility") != "usable":
        raise ValueError("only usable prior labels can be materialized")
    length = len(row["output_token_ids"])
    key = [0] * length
    complete = [0] * length
    by_index = {int(unit["unit_index"]): unit for unit in row["units"]}
    for field, target in (
        ("key_unit_indices", key),
        ("complete_unit_indices", complete),
    ):
        for unit_index in label[field]:
            unit = by_index[int(unit_index)]
            if unit["kind"] != "material_claim":
                raise ValueError("prior label references a non-material unit")
            for token_index in range(int(unit["token_start"]), int(unit["token_end"])):
                target[token_index] = 1
    if any(key[index] and not complete[index] for index in range(length)):
        raise AssertionError("materialized key target is not nested in complete")
    return {
        "key_prior_target": key,
        "complete_prior_target": complete,
        "key_unit_indices": list(label["key_unit_indices"]),
        "complete_unit_indices": list(label["complete_unit_indices"]),
        "prior_label_source": label["label_source"],
        "prior_label_tier": label_tier,
    }


def select_joint_h_prior_rows(
    *,
    proposals: Sequence[Mapping[str, Any]],
    h_labels: Sequence[Mapping[str, Any]],
    prior_labels: Sequence[Mapping[str, Any]],
    per_class: int,
    minimum_each_source_per_class: int | None = None,
    minimum_source_by_class: Mapping[str, Mapping[str, int]] | None = None,
) -> list[dict[str, Any]]:
    by_h = {str(row["item_id"]): row for row in h_labels}
    by_prior = {str(row["item_id"]): row for row in prior_labels}
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        item_id = str(proposal["id"])
        h = by_h.get(item_id)
        prior = by_prior.get(item_id)
        if h is None or prior is None:
            continue
        if (
            h.get("status") not in FINAL_H_VALUES
            or prior.get("eligibility") != "usable"
        ):
            continue
        row = dict(proposal)
        row["h_status"] = h["status"]
        eligible[str(h["status"])].append(row)

    selected: list[dict[str, Any]] = []
    if minimum_source_by_class is None:
        if minimum_each_source_per_class is None:
            raise ValueError("a final source-quota rule is required")
        minimum_source_by_class = {
            status: {
                "gsm8k": int(minimum_each_source_per_class),
                "asdiv-a": int(minimum_each_source_per_class),
            }
            for status in ("hallucinated", "clean")
        }
    for status in ("hallucinated", "clean"):
        pool = eligible[status]
        status_selected: list[dict[str, Any]] = []
        used: set[str] = set()
        source_minima = minimum_source_by_class.get(status, {})
        for source, minimum in sorted(source_minima.items()):
            minimum = int(minimum)
            source_rows = sorted(
                (row for row in pool if row["source"] == source),
                key=lambda row: row["selection_priority"],
            )
            if len(source_rows) < minimum:
                raise ValueError(
                    f"FAIL_YIELD: {status}/{source} lacks required joint usable rows"
                )
            for row in source_rows[:minimum]:
                status_selected.append(row)
                used.add(str(row["id"]))
        remainder = sorted(
            (row for row in pool if str(row["id"]) not in used),
            key=lambda row: row["selection_priority"],
        )
        needed = per_class - len(status_selected)
        if needed < 0:
            raise ValueError("minimum source quota exceeds per-class target")
        if len(remainder) < needed:
            raise ValueError(
                f"FAIL_YIELD: {status} lacks {per_class} joint usable rows"
            )
        status_selected.extend(remainder[:needed])
        selected.extend(status_selected)
    return sorted(
        selected, key=lambda row: (row["h_status"], row["selection_priority"])
    )


__all__ = [
    "CHECKER_VERSION",
    "CORRECTNESS_SEMANTICS",
    "LABEL_TIER",
    "SMOKE_SCHEMA",
    "UNITIZER_VERSION",
    "agreement_report",
    "annotation_signature",
    "atomic_write_json",
    "atomic_write_jsonl",
    "boxed_answers",
    "build_mechanical_consistency_proposals",
    "build_consistency_proposals",
    "build_h_prior_proposals",
    "canonical_json",
    "canonical_sha256",
    "check_numeric_response",
    "cohen_kappa",
    "consistency_mechanical_metrics",
    "consistency_surface_bigrams",
    "consistency_item",
    "extract_math_numeric_reference",
    "file_sha256",
    "freeze_query_pool",
    "load_asdiv_a_repository",
    "material_claim_char_spans",
    "materialize_h_label",
    "materialize_prior_label",
    "near_duplicate_candidates",
    "normalize_question",
    "numeric_options",
    "public_unit_item",
    "publish_manifest",
    "read_jsonl",
    "resolve_blind_labels",
    "select_joint_h_prior_rows",
    "stable_priority",
    "template_signature",
    "text_sha256",
    "tokenize_visible_response",
    "unitize_exact_tokens",
    "validate_annotation",
    "validate_annotator_roster",
    "validate_rollout_population",
    "validate_rollout_row",
    "validate_token_ids",
    "validate_visible_token_mapping",
]
