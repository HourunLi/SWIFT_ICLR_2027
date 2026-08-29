"""Deterministic pre-rollout contracts for CLIR Consistency scale v6.

This module is intentionally model-free.  It filters pinned train-source rows,
propagates historical exclusions through exact/near-duplicate clusters, freezes
the train/held-out acquisition split, and builds fixed rollout shards.  It does
not generate responses, annotate relations, extract hidden states, or train.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from src.clir_smoke import (
    canonical_sha256,
    stable_priority,
    validate_source_row,
)


SCALE_V6_SCHEMA = "clir-data-expansion-scale-v6"
ENTITY_TEMPLATE_VERSION = "clir_entity_template_v1"
CLUSTER_VERSION = "clir_template_cluster_v1"
MINHASH_PERMUTATIONS = 64
MINHASH_BANDS = 16
MINHASH_ROWS_PER_BAND = MINHASH_PERMUTATIONS // MINHASH_BANDS
TOKEN_JACCARD_MIN = 0.82
TRIGRAM_JACCARD_MIN = 0.72
HELDOUT_HASH_FRACTION = 0.25
GSM_LENGTH_QUANTILES = 4

_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d+)?")
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z'’-]*\b")
_TEMPLATE_TOKEN = re.compile(r"<num>|<ent>|[a-z]+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_CALCULATION = re.compile(r"<<([^<>]+)>>")
_NUMBER_LITERAL = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?")

# Capitalized question/function words are not named entities.  Everything else
# capitalized is conservatively abstracted.  This is deliberately deterministic
# and avoids an unpinned NER model in the split contract.
_ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "calculate",
    "determine",
    "does",
    "each",
    "find",
    "for",
    "from",
    "given",
    "how",
    "if",
    "in",
    "is",
    "let",
    "of",
    "on",
    "solve",
    "suppose",
    "the",
    "there",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


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
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            lower, upper = sorted((left_root, right_root))
            self.parent[upper] = lower


def entity_template_signature(text: str) -> str:
    """Normalize numbers and deterministic capitalized-entity candidates."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("question must be a non-empty string")
    value = unicodedata.normalize("NFKC", text)

    def replace_entity(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token.casefold() in _ENTITY_STOPWORDS else " <ent> "

    value = _CAPITALIZED.sub(replace_entity, value)
    value = _NUMBER.sub(" <num> ", value.casefold())
    return " ".join(_TEMPLATE_TOKEN.findall(value))


def template_trigrams(signature: str) -> set[str]:
    tokens = signature.split()
    if not tokens:
        return set()
    if len(tokens) < 3:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _hash64(namespace: str, value: str) -> int:
    payload = f"{namespace}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _minhash_signature(values: set[str]) -> tuple[int, ...]:
    if not values:
        values = {"<empty>"}
    base = [_hash64("clir-C-v6-shingle", value) for value in values]
    mask = (1 << 64) - 1
    signature: list[int] = []
    for permutation in range(MINHASH_PERMUTATIONS):
        multiplier = _hash64("clir-C-v6-minhash-a", str(permutation)) | 1
        offset = _hash64("clir-C-v6-minhash-b", str(permutation))
        signature.append(min((multiplier * value + offset) & mask for value in base))
    return tuple(signature)


def _prefix_jaccard_candidates(
    token_sets: Sequence[set[str]], threshold: float
) -> set[tuple[int, int]]:
    """Exact prefix-filter candidates for set Jaccard at ``threshold``."""

    frequency = Counter(token for values in token_sets for token in values)
    ordered = [
        sorted(values, key=lambda token: (frequency[token], token))
        for values in token_sets
    ]
    prefixes = [
        values[: max(0, len(values) - math.ceil(threshold * len(values)) + 1)]
        for values in ordered
    ]
    postings: dict[str, list[int]] = defaultdict(list)
    output: set[tuple[int, int]] = set()
    for right, prefix in enumerate(prefixes):
        for token in prefix:
            for left in postings[token]:
                output.add((left, right))
        for token in prefix:
            postings[token].append(right)
    return output


def _minhash_candidates(trigram_sets: Sequence[set[str]]) -> set[tuple[int, int]]:
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    output: set[tuple[int, int]] = set()
    for index, values in enumerate(trigram_sets):
        signature = _minhash_signature(values)
        for band in range(MINHASH_BANDS):
            start = band * MINHASH_ROWS_PER_BAND
            key = (band, signature[start : start + MINHASH_ROWS_PER_BAND])
            for left in buckets[key]:
                output.add((left, index))
            buckets[key].append(index)
    return output


def gsm8k_long_chain_metrics(
    reference_answer: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    reasoning = str(reference_answer).split("####", 1)[0]
    word_count = len(_WORD.findall(reasoning))
    calculations = _CALCULATION.findall(reasoning)
    intermediate_values: set[str] = set()
    for calculation in calculations:
        if "=" not in calculation:
            continue
        right_hand_side = calculation.rsplit("=", 1)[1].replace(",", "")
        intermediate_values.update(_NUMBER_LITERAL.findall(right_hand_side))
    passed = bool(
        word_count >= int(config["minimum_reference_reasoning_words"])
        and len(calculations) >= int(config["minimum_reference_calculation_markers"])
        and len(intermediate_values)
        >= int(config["minimum_distinct_intermediate_numeric_values"])
    )
    return {
        "reference_reasoning_word_count": word_count,
        "reference_calculation_marker_count": len(calculations),
        "reference_distinct_intermediate_numeric_count": len(intermediate_values),
        "long_chain_filter_pass": passed,
    }


def build_source_candidates(
    source_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    required_schema: str = SCALE_V6_SCHEMA,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen MATH/GSM8K source filters without reading test data."""

    if protocol.get("schema_version") != required_schema:
        raise ValueError(f"scale source filtering requires protocol {required_schema}")
    math_cfg = protocol["sources"]["math"]
    gsm_cfg = protocol["sources"]["gsm8k"]
    allowed_subjects = set(math_cfg["allowed_subjects"])
    allowed_levels = {int(value) for value in math_cfg["allowed_levels"]}
    minimum_solution_words = int(math_cfg["minimum_official_solution_words"])
    candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    math_strata: Counter[str] = Counter()
    gsm_word_counts: list[int] = []
    for raw in source_rows:
        source = raw.get("source")
        if source == "math":
            counts["math_input"] += 1
            subject = str(raw.get("source_subject", ""))
            level = raw.get("source_level")
            solution = str(raw.get("source_solution", ""))
            if subject not in allowed_subjects or level not in allowed_levels:
                counts["math_reject_subject_or_level"] += 1
                continue
            if len(solution.split()) < minimum_solution_words:
                counts["math_reject_short_solution"] += 1
                continue
            row = validate_source_row(raw)
            row["selection_stratum"] = f"{subject}|level_{int(level)}"
            row["query_priority"] = stable_priority("clir-C-v6-query", row["query_id"])
            candidates.append(row)
            math_strata[row["selection_stratum"]] += 1
            counts["math_eligible"] += 1
        elif source == "gsm8k":
            counts["gsm8k_input"] += 1
            metrics = gsm8k_long_chain_metrics(
                str(raw.get("reference_answer", "")),
                gsm_cfg["long_chain_filter"],
            )
            if not metrics["long_chain_filter_pass"]:
                counts["gsm8k_reject_not_long_chain"] += 1
                continue
            row = validate_source_row(raw)
            row.update(metrics)
            row["query_priority"] = stable_priority("clir-C-v6-query", row["query_id"])
            candidates.append(row)
            gsm_word_counts.append(int(metrics["reference_reasoning_word_count"]))
            counts["gsm8k_eligible"] += 1
    return candidates, {
        "counts": dict(sorted(counts.items())),
        "math_strata": dict(sorted(math_strata.items())),
        "gsm8k_reasoning_word_count": _summary(gsm_word_counts),
    }


def combine_permanent_exclusions(
    historical_rows: Sequence[Mapping[str, Any]],
    smoke_query_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reasons: dict[str, set[str]] = defaultdict(set)
    source_by_id: dict[str, str] = {}
    for row in historical_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("historical exclusion lacks query_id")
        raw_reasons = row.get("reasons", ["historical_efficacy_or_mechanism_use"])
        if not isinstance(raw_reasons, Sequence) or isinstance(raw_reasons, str):
            raw_reasons = [str(raw_reasons)]
        reasons[query_id].update(str(reason) for reason in raw_reasons)
    for row in smoke_query_rows:
        query_id = row.get("query_id")
        source = row.get("source")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("smoke query exclusion lacks query_id")
        if isinstance(source, str):
            source_by_id[query_id] = source
        batch = str(row.get("acquisition_batch", "unknown"))
        if batch == "reserve":
            reasons[query_id].add("smoke_v3_reserve_and_consistency_v5")
        elif batch == "primary":
            reasons[query_id].add("smoke_v3_primary_annotation")
        else:
            reasons[query_id].add("smoke_v2_train_only_annotation")
    return [
        {
            "query_id": query_id,
            "source": source_by_id.get(query_id, query_id.split(":", 1)[0]),
            "reasons": sorted(values),
        }
        for query_id, values in sorted(reasons.items())
    ]


def build_template_clusters(
    candidate_rows: Sequence[Mapping[str, Any]],
    exclusion_anchor_rows: Sequence[Mapping[str, Any]],
    excluded_query_ids: Iterable[str],
    *,
    namespace: str = "clir-C-v6",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cluster candidates and exclusion anchors, propagating every exclusion."""

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("cluster namespace must be a non-empty string")

    by_id: dict[str, dict[str, Any]] = {}
    candidate_ids = {str(row["query_id"]) for row in candidate_rows}
    for raw in [*candidate_rows, *exclusion_anchor_rows]:
        row = validate_source_row(raw)
        query_id = str(row["query_id"])
        prior = by_id.get(query_id)
        if prior is not None and prior["question_sha256"] != row["question_sha256"]:
            raise ValueError(f"query {query_id} has conflicting question text")
        row["template_signature_v6"] = entity_template_signature(row["question"])
        by_id[query_id] = row
    excluded = set(excluded_query_ids)
    missing_candidate = candidate_ids - set(by_id)
    if missing_candidate:
        raise ValueError("candidate rows were lost before clustering")

    query_ids = sorted(by_id)
    rows = [by_id[query_id] for query_id in query_ids]
    union = _UnionFind(query_ids)
    exact_groups: dict[str, list[str]] = defaultdict(list)
    template_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        exact_groups[str(row["normalized_question"])].append(str(row["query_id"]))
        template_groups[str(row["template_signature_v6"])].append(str(row["query_id"]))

    exact_removed: set[str] = set()
    exact_edge_count = 0
    for members in exact_groups.values():
        ordered = sorted(
            members, key=lambda value: hashlib.sha256(value.encode()).hexdigest()
        )
        for query_id in ordered[1:]:
            exact_removed.add(query_id)
            union.union(ordered[0], query_id)
            exact_edge_count += 1

    template_edge_count = 0
    for members in template_groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for query_id in ordered[1:]:
            union.union(ordered[0], query_id)
            template_edge_count += 1

    token_sets = [set(str(row["template_signature_v6"]).split()) for row in rows]
    trigram_sets = [
        template_trigrams(str(row["template_signature_v6"])) for row in rows
    ]
    candidate_pairs = _prefix_jaccard_candidates(token_sets, TOKEN_JACCARD_MIN)
    candidate_pairs.update(_minhash_candidates(trigram_sets))
    near_edges: list[tuple[str, str, float, float]] = []
    for left_index, right_index in sorted(candidate_pairs):
        left_id, right_id = query_ids[left_index], query_ids[right_index]
        if (
            by_id[left_id]["normalized_question"]
            == by_id[right_id]["normalized_question"]
        ):
            continue
        token_similarity = _jaccard(token_sets[left_index], token_sets[right_index])
        trigram_similarity = _jaccard(
            trigram_sets[left_index], trigram_sets[right_index]
        )
        if (
            token_similarity < TOKEN_JACCARD_MIN
            and trigram_similarity < TRIGRAM_JACCARD_MIN
        ):
            continue
        union.union(left_id, right_id)
        near_edges.append((left_id, right_id, token_similarity, trigram_similarity))

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        members_by_root[union.find(query_id)].append(query_id)
    near_edge_counts: Counter[str] = Counter()
    for left, _, _, _ in near_edges:
        near_edge_counts[union.find(left)] += 1

    cluster_rows: list[dict[str, Any]] = []
    selectable: list[dict[str, Any]] = []
    excluded_cluster_count = 0
    for root, members in members_by_root.items():
        ordered_members = sorted(members)
        cluster_id = stable_priority(f"{namespace}-cluster", *ordered_members)
        split_priority = stable_priority(f"{namespace}-split", cluster_id)
        split = (
            "heldout_acquisition"
            if int(split_priority, 16) < (1 << 254)
            else "train_acquisition"
        )
        excluded_members = sorted(set(ordered_members) & excluded)
        excluded_cluster = bool(excluded_members)
        if excluded_cluster:
            excluded_cluster_count += 1
        eligible_members = sorted(set(ordered_members) & candidate_ids)
        selectable_members = [
            query_id
            for query_id in eligible_members
            if query_id not in exact_removed and not excluded_cluster
        ]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "member_query_ids": ordered_members,
                "eligible_candidate_query_ids": eligible_members,
                "selectable_query_ids": selectable_members,
                "source_counts": dict(
                    sorted(
                        Counter(
                            str(by_id[value]["source"]) for value in ordered_members
                        ).items()
                    )
                ),
                "excluded_by_prior_membership": excluded_cluster,
                "excluded_member_query_ids": excluded_members,
                "exact_duplicate_removed_query_ids": sorted(
                    set(ordered_members) & exact_removed
                ),
                "near_duplicate_edge_count": near_edge_counts[root],
                "split_priority": split_priority,
                "acquisition_split": split,
            }
        )
        for query_id in selectable_members:
            row = dict(by_id[query_id])
            row["cluster_id"] = cluster_id
            row["cluster_split_priority"] = split_priority
            row["acquisition_split"] = split
            row["query_priority"] = stable_priority(f"{namespace}-query", query_id)
            selectable.append(row)

    implementation = {
        "entity_template_version": ENTITY_TEMPLATE_VERSION,
        "cluster_version": CLUSTER_VERSION,
        "minhash_permutations": MINHASH_PERMUTATIONS,
        "minhash_bands": MINHASH_BANDS,
        "token_jaccard_min": TOKEN_JACCARD_MIN,
        "trigram_jaccard_min": TRIGRAM_JACCARD_MIN,
        "heldout_hash_fraction": HELDOUT_HASH_FRACTION,
    }
    if namespace != "clir-C-v6":
        implementation["namespace"] = namespace
    report = {
        "implementation": implementation,
        "rows_considered": len(rows),
        "candidate_rows": len(candidate_ids),
        "exclusion_anchor_rows": len(set(by_id) - candidate_ids),
        "clusters": len(cluster_rows),
        "excluded_clusters": excluded_cluster_count,
        "exact_duplicate_edges": exact_edge_count,
        "exact_duplicate_removed": len(exact_removed),
        "exact_template_edges": template_edge_count,
        "near_duplicate_candidate_pairs": len(candidate_pairs),
        "near_duplicate_edges": len(near_edges),
        "selectable_rows": len(selectable),
    }
    return (
        sorted(cluster_rows, key=lambda row: row["cluster_id"]),
        sorted(selectable, key=lambda row: (row["source"], row["query_priority"])),
        report,
    )


def _assign_gsm_quantiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm_rows = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            row["query_priority"],
        ),
    )
    for index, row in enumerate(gsm_rows):
        quantile = min(
            GSM_LENGTH_QUANTILES - 1, index * GSM_LENGTH_QUANTILES // len(gsm_rows)
        )
        row["selection_stratum"] = f"reasoning_length_q{quantile + 1}"
    return output


def _proportional_quotas(available: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot allocate target {target} from {total} available rows")
    if target == 0:
        return {key: 0 for key in sorted(available)}
    raw = {key: target * count / total for key, count in available.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        available,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            stable_priority("clir-C-v6-quota", key),
        ),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    if any(quotas[key] > available[key] for key in quotas):
        raise AssertionError("proportional allocation exceeded a stratum capacity")
    return dict(sorted(quotas.items()))


def select_acquisition_queries(
    selectable_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _assign_gsm_quantiles(selectable_rows)
    targets = {
        ("math", "train_acquisition"): int(
            protocol["sources"]["math"]["train_acquisition_queries"]
        ),
        ("math", "heldout_acquisition"): int(
            protocol["sources"]["math"]["heldout_acquisition_queries"]
        ),
        ("gsm8k", "train_acquisition"): int(
            protocol["sources"]["gsm8k"]["train_acquisition_queries"]
        ),
        ("gsm8k", "heldout_acquisition"): int(
            protocol["sources"]["gsm8k"]["heldout_acquisition_queries"]
        ),
    }
    selected: list[dict[str, Any]] = []
    availability: dict[str, Any] = {}
    quotas_report: dict[str, Any] = {}
    for (source, split), target in sorted(targets.items()):
        pool = [
            row
            for row in rows
            if row["source"] == source and row["acquisition_split"] == split
        ]
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_stratum[str(row["selection_stratum"])].append(row)
        counts = {key: len(value) for key, value in by_stratum.items()}
        quotas = _proportional_quotas(counts, target)
        for stratum, quota in quotas.items():
            ordered = sorted(by_stratum[stratum], key=lambda row: row["query_priority"])
            selected.extend(dict(row) for row in ordered[:quota])
        report_key = f"{source}|{split}"
        availability[report_key] = dict(sorted(counts.items()))
        quotas_report[report_key] = quotas

    train = sorted(
        (row for row in selected if row["acquisition_split"] == "train_acquisition"),
        key=lambda row: row["query_priority"],
    )
    heldout = sorted(
        (row for row in selected if row["acquisition_split"] == "heldout_acquisition"),
        key=lambda row: row["query_priority"],
    )
    train_ids = {str(row["query_id"]) for row in train}
    heldout_ids = {str(row["query_id"]) for row in heldout}
    train_clusters = {str(row["cluster_id"]) for row in train}
    heldout_clusters = {str(row["cluster_id"]) for row in heldout}
    if train_ids & heldout_ids or train_clusters & heldout_clusters:
        raise AssertionError(
            "train and heldout acquisition sets are not cluster-disjoint"
        )
    expected_train = int(protocol["consistency_scale"]["train_acquisition_queries"])
    expected_heldout = int(protocol["consistency_scale"]["heldout_acquisition_queries"])
    if len(train) != expected_train or len(heldout) != expected_heldout:
        raise AssertionError("selected acquisition counts do not match v6")
    return (
        train,
        heldout,
        {
            "available_by_source_split_stratum": availability,
            "selected_quotas_by_source_split_stratum": quotas_report,
            "train_source_counts": dict(
                sorted(Counter(row["source"] for row in train).items())
            ),
            "heldout_source_counts": dict(
                sorted(Counter(row["source"] for row in heldout).items())
            ),
            "train_cluster_count": len(train_clusters),
            "heldout_cluster_count": len(heldout_clusters),
            "cluster_overlap": 0,
            "query_overlap": 0,
        },
    )


def build_rollout_shards(
    train_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    shard_size = int(protocol["generation"]["rollout_shard_query_count"])
    if shard_size != 50:
        raise ValueError("v6 source-balanced sharding is frozen at 50 queries")
    shards: list[dict[str, Any]] = []
    for split, rows in (
        ("train_acquisition", train_rows),
        ("heldout_acquisition", heldout_rows),
    ):
        math_rows = sorted(
            (dict(row) for row in rows if row["source"] == "math"),
            key=lambda row: stable_priority(
                "clir-C-v6-shard-order", split, row["query_id"]
            ),
        )
        gsm_rows = sorted(
            (dict(row) for row in rows if row["source"] == "gsm8k"),
            key=lambda row: stable_priority(
                "clir-C-v6-shard-order", split, row["query_id"]
            ),
        )
        shard_count = len(rows) // shard_size
        if (
            len(rows) % shard_size
            or len(math_rows) != shard_count * 35
            or len(gsm_rows) != shard_count * 15
        ):
            raise ValueError("v6 shards require exactly 35 MATH + 15 GSM8K per shard")
        for index in range(shard_count):
            query_rows = [
                *math_rows[index * 35 : (index + 1) * 35],
                *gsm_rows[index * 15 : (index + 1) * 15],
            ]
            query_rows.sort(
                key=lambda row: stable_priority("clir-C-v6-shard-row", row["query_id"])
            )
            shard_id = (
                f"{'train' if split == 'train_acquisition' else 'heldout'}-{index:03d}"
            )
            shards.append(
                {
                    "shard_id": shard_id,
                    "acquisition_split": split,
                    "query_count": len(query_rows),
                    "source_counts": {"math": 35, "gsm8k": 15},
                    "query_ids": [row["query_id"] for row in query_rows],
                    "ordered_query_ids_sha256": canonical_sha256(
                        [row["query_id"] for row in query_rows]
                    ),
                    "expected_candidate_rows": len(query_rows)
                    * int(protocol["generation"]["candidate_count"]),
                    "output_path": f"rollouts/{shard_id}.jsonl",
                }
            )
    expected = int(protocol["generation"]["planned_rollout_shards"])
    if len(shards) != expected:
        raise AssertionError(f"expected {expected} rollout shards, found {len(shards)}")
    return shards


def storage_and_gpu_budget(
    selected_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    prompt_counts = [int(row["prompt_token_count"]) for row in selected_rows]
    source_counts = Counter(str(row["source"]) for row in selected_rows)
    assumptions = protocol["pre_rollout_implementation"]["budget_assumptions"]
    expected_output_by_source = assumptions["expected_output_tokens_per_candidate"]
    candidate_count = int(protocol["generation"]["candidate_count"])
    expected_output_tokens = sum(
        source_counts[source] * candidate_count * int(expected_output_by_source[source])
        for source in source_counts
    )
    conservative_output_tokens = (
        len(selected_rows)
        * candidate_count
        * int(assumptions["conservative_output_tokens_per_candidate"])
    )
    worst_case_output_tokens = (
        len(selected_rows)
        * candidate_count
        * int(protocol["generation"]["max_new_tokens"])
    )
    bytes_per_token = int(assumptions["full_feature_bytes_per_token"])
    selected_trajectories = int(
        protocol["publication_and_extraction"]["selected_trajectory_count"]
    )
    selected_prompts = int(
        protocol["publication_and_extraction"]["selected_unique_prompt_count"]
    )
    output_estimate = int(assumptions["conservative_output_tokens_per_candidate"])
    average_prompt = sum(prompt_counts) / len(prompt_counts)
    selected_feature_bytes = int(
        (selected_trajectories * output_estimate + selected_prompts * average_prompt)
        * bytes_per_token
    )
    all_feature_bytes = int(
        (
            len(selected_rows) * candidate_count * output_estimate
            + len(selected_rows) * average_prompt
        )
        * bytes_per_token
    )
    return {
        "selected_query_count": len(selected_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "prompt_token_count": _summary(prompt_counts),
        "expected_raw_output_tokens": expected_output_tokens,
        "conservative_raw_output_tokens": conservative_output_tokens,
        "worst_case_raw_output_tokens": worst_case_output_tokens,
        "selected_only_feature_storage_gb": selected_feature_bytes / 1_000_000_000,
        "forbidden_all_rollout_feature_storage_tb": all_feature_bytes
        / 1_000_000_000_000,
        "bytes_per_full_feature_token": bytes_per_token,
        "gpu_budget": {
            "gpu_model": assumptions["gpu_model"],
            "tensor_parallel_size": int(assumptions["tensor_parallel_size"]),
            "atomic_shard_jobs": int(protocol["generation"]["planned_rollout_shards"]),
            "queries_per_job": int(protocol["generation"]["rollout_shard_query_count"]),
            "candidates_per_job": int(
                protocol["generation"]["rollout_shard_query_count"]
            )
            * candidate_count,
            "maximum_concurrent_jobs": int(assumptions["maximum_concurrent_l20z_jobs"]),
            "minimum_waves_at_maximum_concurrency": math.ceil(
                int(protocol["generation"]["planned_rollout_shards"])
                / int(assumptions["maximum_concurrent_l20z_jobs"])
            ),
            "runtime_policy": "measure_shard_000_before_estimating_wall_clock_no_threshold_changes",
        },
    }


def compact_query_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "source",
        "query_id",
        "source_record_id",
        "question",
        "reference_answer",
        "source_license",
        "source_subject",
        "source_level",
        "reference_reasoning_word_count",
        "reference_calculation_marker_count",
        "reference_distinct_intermediate_numeric_count",
        "selection_stratum",
        "cluster_id",
        "cluster_split_priority",
        "acquisition_split",
        "query_priority",
        "prompt_token_count",
        "question_sha256",
        "template_signature_v6",
    ]
    return {key: row[key] for key in keys if key in row}


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "median": median,
    }


__all__ = [
    "CLUSTER_VERSION",
    "ENTITY_TEMPLATE_VERSION",
    "GSM_LENGTH_QUANTILES",
    "HELDOUT_HASH_FRACTION",
    "MINHASH_BANDS",
    "MINHASH_PERMUTATIONS",
    "SCALE_V6_SCHEMA",
    "TOKEN_JACCARD_MIN",
    "TRIGRAM_JACCARD_MIN",
    "build_rollout_shards",
    "build_source_candidates",
    "build_template_clusters",
    "combine_permanent_exclusions",
    "compact_query_row",
    "entity_template_signature",
    "gsm8k_long_chain_metrics",
    "select_acquisition_queries",
    "storage_and_gpu_budget",
    "template_trigrams",
]
