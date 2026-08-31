"""Deterministic planning for CLIR Prior strict-consensus scale v12.

V12 is a prospective acquisition protocol, not a repair or salvage of the
failed v8--v11 smoke rows.  It freezes fresh query/split identities before
rollout, then selects one checker/unitizer-valid trajectory per query for a
large blind dual-AI annotation pool.  A later stage may retain only exact
singleton-Key consensus with non-empty partial Complete consensus.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from src.clir_prior_partial import (
    _material_units,
    _set_f1,
    build_blind_packages,
    target_signature,
    validate_partial_prior_annotation,
)
from src.clir_smoke import canonical_sha256, stable_priority


PROTOCOL_SCHEMA = "clir-prior-strict-consensus-scale-v12"
QUERY_SCHEMA = "clir-prior-v12-acquisition-query"
PROPOSAL_SCHEMA = "clir-prior-v12-natural-proposal"
PACKAGE_SCHEMA = "clir-prior-v12-annotation-package"
PRIVATE_SCHEMA = "clir-prior-v12-private-index"
LABEL_SCHEMA = "clir-prior-v12-label"
REPORT_SCHEMA = "clir-prior-v12-strict-consensus-gate"


def prior_v12_control_items() -> list[dict[str, Any]]:
    """Return 16 fresh hidden controls for the scale-v12 annotation pass."""

    definitions = [
        (
            "early_arithmetic_error",
            "Nine bags hold seven oranges each, and four loose oranges are added. How many oranges are there?",
            [
                "Nine bags contain 9×7=61 oranges.",
                "Adding the loose oranges gives 61+4=65 oranges.",
                "Therefore there are 65 oranges.",
            ],
            "usable",
            [0],
            [0, 1],
        ),
        (
            "later_arithmetic_error",
            "Six racks hold eight mugs each, and five loose mugs are added. How many mugs are there?",
            [
                "The racks contain 6×8=48 mugs.",
                "Adding the loose mugs gives 48+5=54 mugs.",
                "Therefore there are 54 mugs.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "late_target_error",
            "A train covers 180 miles in 3 hours. What is its speed in miles per hour?",
            [
                "Dividing distance by time gives 180/3=60 miles per hour.",
                "Therefore the train travels 60 miles in total.",
                "The final answer is 60.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "late_unit_error",
            "A recipe needs 3 cups of water per batch. How much water is needed for 4 batches?",
            [
                "Four batches need 3×4=12 cups of water.",
                "Therefore the recipe needs 12 liters of water.",
                "The final answer is 12.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "late_object_error",
            "A jar contains 12 red marbles and 7 blue marbles. How many marbles are in the jar?",
            [
                "Adding the colors gives 12+7=19 marbles in total.",
                "Therefore the jar contains 19 red marbles.",
                "The final answer is 19.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "split_calculation",
            "Eight packets hold six stickers each, with five loose stickers. How many stickers are there?",
            [
                "The packet calculation is 8×6.",
                "Evaluating it gives 48 stickers in packets.",
                "The total calculation is 48+5.",
                "Evaluating it gives 53 stickers.",
                "Therefore the answer is 53 stickers.",
            ],
            "usable",
            [3],
            [0, 1, 2, 3],
        ),
        (
            "given_restatement",
            "Leo read 14 pages Monday and 9 Tuesday, but 4 Tuesday pages were rereads. How many different pages did he read?",
            [
                "The daily counts total 14+9=23 pages.",
                "The problem says that 4 pages were rereads.",
                "Subtracting the rereads gives 23-4=19 different pages.",
                "The answer is 19 pages.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "unused_branch",
            "Two notebooks cost 5 dollars each and three pens cost 2 dollars each. What is the total cost?",
            [
                "Three pens cost 3×2=6 dollars.",
                "The word notebook has eight letters.",
                "Two notebooks cost 2×5=10 dollars.",
                "Adding the costs gives 6+10=16 dollars.",
                "The final answer is 16 dollars.",
            ],
            "usable",
            [3],
            [0, 2, 3],
        ),
        (
            "duplicate_result",
            "A shelf has seven rows of four books and two loose books. How many books are there?",
            [
                "The rows contain 7×4=28 books.",
                "So the row count is 28 books.",
                "Adding the loose books gives 28+2=30 books.",
                "Thus the answer is 30 books.",
            ],
            "usable",
            [2],
            [0, 2],
        ),
        (
            "duplicate_equation",
            "Solve x+7=19 for x.",
            [
                "Subtracting 7 from both sides gives x=12.",
                "Equivalently, x=19-7=12.",
                "Therefore x=12.",
            ],
            "usable",
            [0],
            [0],
        ),
        (
            "self_contained_chain",
            "Five boxes hold eleven cards each, and three loose cards are added. How many cards are there?",
            [
                "Five boxes contain 5×11=55 cards.",
                "Adding the loose cards gives 55+3=58 cards.",
                "Therefore the answer is 58 cards.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "later_algebra_error",
            "Solve 3x+5=20 for x.",
            [
                "Subtracting 5 gives 3x=15.",
                "Dividing by 3 gives x=6.",
                "Therefore x=6.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "decimal_chain",
            "Four tickets cost 2.50 dollars each and there is a 1.25 dollar fee. What is the total?",
            [
                "The tickets cost 4×2.50=10.00 dollars.",
                "Adding the fee gives 10.00+1.25=11.25 dollars.",
                "The final answer is 11.25 dollars.",
            ],
            "usable",
            [1],
            [0, 1],
        ),
        (
            "unused_formula",
            "A rectangle is 8 meters long and 5 meters wide. What is its area?",
            [
                "The perimeter formula is 2(length+width).",
                "The area is 8×5=40 square meters.",
                "Therefore the answer is 40 square meters.",
            ],
            "usable",
            [1],
            [1],
        ),
        (
            "answer_only",
            "What is 18 plus 9?",
            ["27"],
            "no_auditable_reasoning",
            [],
            [],
        ),
        (
            "refusal_only",
            "What is 24 divided by 6?",
            ["I cannot solve this problem."],
            "no_auditable_reasoning",
            [],
            [],
        ),
    ]
    output = []
    for name, question, texts, eligibility, key, complete in definitions:
        output.append(
            {
                "schema_version": PACKAGE_SCHEMA,
                "item_id": stable_priority("clir-prior-v12-control", name),
                "question": question,
                "response": "\n".join(texts),
                "units": [
                    {
                        "unit_index": index,
                        "kind": "material_claim",
                        "text": text,
                    }
                    for index, text in enumerate(texts)
                ],
                "expected_signature": (
                    eligibility,
                    tuple(key),
                    tuple(complete),
                ),
            }
        )
    return output


def _proportional_quotas(
    available: Mapping[str, int], target: int, *, namespace: str
) -> dict[str, int]:
    total = sum(int(value) for value in available.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot select {target} rows from capacity {total}")
    if not available:
        if target:
            raise ValueError("cannot select from an empty stratum map")
        return {}
    raw = {key: target * count / total for key, count in available.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        available,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            stable_priority(namespace, key),
        ),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    if any(quotas[key] > available[key] for key in quotas):
        raise AssertionError("stratified quota exceeded capacity")
    return dict(sorted(quotas.items()))


def _one_query_per_cluster(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        for field in ("cluster_id", "query_id", "source"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"selectable row lacks {field}")
        by_cluster[str(row["cluster_id"])].append(row)
    output = []
    for cluster_id, members in sorted(by_cluster.items()):
        sources = {str(row["source"]) for row in members}
        if len(sources) != 1:
            raise ValueError(f"template cluster {cluster_id} mixes sources")
        output.append(
            min(
                members,
                key=lambda row: stable_priority(
                    f"{namespace}-cluster-representative", str(row["query_id"])
                ),
            )
        )
    return output


def _attach_source_strata(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    gsm = sorted(
        (row for row in output if row["source"] == "gsm8k"),
        key=lambda row: (
            int(row["reference_reasoning_word_count"]),
            stable_priority(f"{namespace}-gsm-length", str(row["query_id"])),
        ),
    )
    for index, row in enumerate(gsm):
        quantile = min(3, index * 4 // max(1, len(gsm)))
        row["prior_source_stratum"] = f"reasoning_length_q{quantile + 1}"
    for row in output:
        if row["source"] == "math":
            subject = str(row.get("source_subject", ""))
            level = int(row.get("source_level", 0))
            if not subject or level not in {2, 3, 4, 5}:
                raise ValueError("MATH acquisition row lacks an allowed subject/level")
            row["prior_source_stratum"] = f"{subject}|level_{level}"
        elif row["source"] != "gsm8k":
            raise ValueError(f"unsupported Prior v12 source: {row['source']}")
    return output


def _select_stratified(
    rows: Sequence[Mapping[str, Any]], target: int, *, namespace: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        by_stratum[str(raw["prior_source_stratum"])].append(dict(raw))
    available = {key: len(values) for key, values in by_stratum.items()}
    quotas = _proportional_quotas(available, target, namespace=f"{namespace}-quota")
    selected = []
    for stratum, count in quotas.items():
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: stable_priority(f"{namespace}-row", str(row["query_id"])),
        )
        selected.extend(ordered[:count])
    return selected, {
        "available_by_stratum": dict(sorted(available.items())),
        "selected_by_stratum": quotas,
    }


def select_acquisition_queries(
    selectable_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze fresh train/dev query identities before any v12 rollout."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior v12 protocol")
    namespace = str(protocol["query_pool"]["namespace"])
    representatives = _attach_source_strata(
        _one_query_per_cluster(selectable_rows, namespace=namespace),
        namespace=namespace,
    )
    remaining = {str(row["query_id"]): row for row in representatives}
    selected: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    source_split_counts = protocol["query_pool"]["source_split_counts"]
    # Dev is frozen first and can never be consumed by train selection.
    for source in ("gsm8k", "math"):
        source_report: dict[str, Any] = {}
        for split in ("dev", "train"):
            count = int(source_split_counts[source][split])
            pool = [row for row in remaining.values() if row["source"] == source]
            chosen, report = _select_stratified(
                pool,
                count,
                namespace=f"{namespace}-{source}-{split}",
            )
            for raw in chosen:
                row = dict(raw)
                row["schema_version"] = QUERY_SCHEMA
                row["role"] = "prior_acquisition"
                row["prior_label_split"] = split
                row["role_priority"] = stable_priority(
                    f"{namespace}-role", split, str(row["query_id"])
                )
                selected.append(row)
                del remaining[str(row["query_id"])]
            source_report[split] = report
        reports[source] = source_report

    selected.sort(key=lambda row: str(row["role_priority"]))
    expected = sum(
        int(count)
        for values in source_split_counts.values()
        for count in values.values()
    )
    query_ids = [str(row["query_id"]) for row in selected]
    cluster_ids = [str(row["cluster_id"]) for row in selected]
    if len(selected) != expected or len(set(query_ids)) != expected:
        raise AssertionError("Prior v12 acquisition query count/identity drift")
    if len(set(cluster_ids)) != expected:
        raise AssertionError("Prior v12 acquisition reuses a template cluster")
    return selected, {
        "selectable_rows": len(selectable_rows),
        "cluster_representatives": len(representatives),
        "selected_queries": len(selected),
        "selected_by_source": dict(
            sorted(Counter(str(row["source"]) for row in selected).items())
        ),
        "selected_by_split": dict(
            sorted(Counter(str(row["prior_label_split"]) for row in selected).items())
        ),
        "selected_by_source_split": dict(
            sorted(
                Counter(
                    f"{row['source']}|{row['prior_label_split']}" for row in selected
                ).items()
            )
        ),
        "source_strata": reports,
        "selected_query_ids_sha256": canonical_sha256(query_ids),
        "selected_cluster_ids_sha256": canonical_sha256(cluster_ids),
        "remaining_cluster_representatives": len(remaining),
    }


def build_acquisition_shards(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cfg = protocol["generation"]
    shard_count = int(cfg["rollout_shards"])
    candidate_count = int(cfg["candidate_count"])
    members: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: stable_priority(
            "clir-prior-v12-rollout-shard-order", str(row["query_id"])
        ),
    )
    for index, row in enumerate(ordered):
        members[index % shard_count].append(row)
    shards = []
    for index, shard_rows in enumerate(members):
        shard_id = f"prior-{index:03d}"
        query_ids = [str(row["query_id"]) for row in shard_rows]
        shards.append(
            {
                "shard_id": shard_id,
                "query_count": len(shard_rows),
                "candidate_count": candidate_count,
                "candidate_index_start": 0,
                "candidate_index_end_exclusive": candidate_count,
                "query_ids": query_ids,
                "ordered_query_ids_sha256": canonical_sha256(query_ids),
                "source_counts": dict(
                    sorted(Counter(str(row["source"]) for row in shard_rows).items())
                ),
                "split_counts": dict(
                    sorted(
                        Counter(
                            str(row["prior_label_split"]) for row in shard_rows
                        ).items()
                    )
                ),
                "expected_candidate_rows": len(shard_rows) * candidate_count,
                "output_path": f"rollouts/{shard_id}.jsonl",
            }
        )
    expected_queries = int(protocol["query_pool"]["query_count"])
    expected_rows = expected_queries * candidate_count
    if sum(int(row["query_count"]) for row in shards) != expected_queries:
        raise AssertionError("Prior v12 shard query count drift")
    if sum(int(row["expected_candidate_rows"]) for row in shards) != expected_rows:
        raise AssertionError("Prior v12 shard candidate-row count drift")
    return shards


def select_prior_proposals(
    materialized_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the frozen 800-row annotation population after materialization."""

    pool = protocol["proposal_pool"]
    quotas = {
        (
            str(entry["source"]),
            str(entry["checker_status"]),
            str(entry["split"]),
        ): int(entry["count"])
        for entry in pool["strata"]
    }
    if sum(quotas.values()) != int(pool["natural_count"]):
        raise ValueError("Prior v12 proposal strata do not sum to natural_count")
    minimum = int(pool["minimum_material_claims"])
    maximum = int(pool["maximum_material_claims"])
    by_stratum_query: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    rejection: Counter[str] = Counter()
    for row in materialized_rows:
        stratum = (
            str(row.get("source")),
            str(row.get("checker_status")),
            str(row.get("prior_label_split")),
        )
        if stratum not in quotas:
            rejection["outside_strata"] += 1
            continue
        if not row.get("eligible_for_supervision"):
            rejection["not_supervision_eligible"] += 1
            continue
        if row.get("unitization_status") != "ok":
            rejection["unitization"] += 1
            continue
        if row.get("finish_reason") != "stop":
            rejection["finish_reason"] += 1
            continue
        claims = int(row.get("material_claim_count", 0))
        if not minimum <= claims <= maximum:
            rejection["material_claim_count"] += 1
            continue
        _material_units(row)
        by_stratum_query[stratum][str(row["query_id"])].append(row)

    candidates: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    namespace = str(pool["selection_namespace"])
    for stratum, by_query in by_stratum_query.items():
        one_per_query = []
        for query_id, rows in by_query.items():
            one_per_query.append(
                min(
                    rows,
                    key=lambda row: stable_priority(
                        f"{namespace}-candidate", query_id, str(row["id"])
                    ),
                )
            )
        candidates[stratum] = sorted(
            one_per_query,
            key=lambda row: stable_priority(
                f"{namespace}-query", *stratum, str(row["query_id"]), str(row["id"])
            ),
        )

    protocol_order = [
        (str(row["source"]), str(row["checker_status"]), str(row["split"]))
        for row in pool["strata"]
    ]
    ordered_strata = sorted(
        quotas,
        key=lambda stratum: (
            len(candidates.get(stratum, [])),
            protocol_order.index(stratum),
        ),
    )
    selected = []
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    for stratum in ordered_strata:
        selected_here = 0
        for row in candidates.get(stratum, []):
            query_id = str(row["query_id"])
            cluster_id = str(row["cluster_id"])
            if query_id in used_queries or cluster_id in used_clusters:
                continue
            proposal_id = stable_priority(f"{namespace}-proposal", str(row["id"]))
            selected.append(
                {
                    "schema_version": PROPOSAL_SCHEMA,
                    "proposal_id": proposal_id,
                    "trajectory_id": str(row["id"]),
                    "query_id": query_id,
                    "cluster_id": cluster_id,
                    "source": str(row["source"]),
                    "source_record_id": row.get("source_record_id"),
                    "checker_status": str(row["checker_status"]),
                    "prior_label_split": str(row["prior_label_split"]),
                    "candidate_index": int(row["candidate_index"]),
                    "question": str(row["question"]),
                    "response": str(row["response"]),
                    "material_claim_count": int(row["material_claim_count"]),
                    "output_token_count": int(row["output_token_count"]),
                    "units": _material_units(row),
                    "selection_priority": stable_priority(
                        f"{namespace}-query",
                        *stratum,
                        query_id,
                        str(row["id"]),
                    ),
                }
            )
            used_queries.add(query_id)
            used_clusters.add(cluster_id)
            selected_here += 1
            if selected_here == quotas[stratum]:
                break
        if selected_here != quotas[stratum]:
            raise ValueError(
                f"insufficient Prior v12 proposal capacity for {stratum}: "
                f"{selected_here}/{quotas[stratum]}"
            )
    selected.sort(key=lambda row: str(row["proposal_id"]))
    counts = Counter(
        (row["source"], row["checker_status"], row["prior_label_split"])
        for row in selected
    )
    return selected, {
        "natural_selected": len(selected),
        "unique_queries": len(used_queries),
        "unique_clusters": len(used_clusters),
        "selected_by_stratum": {
            "|".join(stratum): count for stratum, count in sorted(counts.items())
        },
        "available_query_counts": {
            "|".join(stratum): len(candidates.get(stratum, []))
            for stratum in sorted(quotas)
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "ordered_rows_sha256": canonical_sha256(selected),
    }


def build_prior_annotation_shards(
    proposals: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[
    dict[str, list[list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Build 16 isolated A/B shards with 50 natural, 1 control, 5 repeats."""

    annotation = protocol["annotation"]
    shard_count = int(annotation["natural_shards_per_annotator"])
    natural_per_shard = int(annotation["natural_rows_per_shard"])
    controls = prior_v12_control_items()
    repeat_count = int(annotation["self_repeats_total_per_annotator"])
    if len(proposals) != shard_count * natural_per_shard:
        raise ValueError("Prior v12 proposal count does not fill annotation shards")
    if len(controls) != int(annotation["hidden_controls_total_per_annotator"]):
        raise ValueError("Prior v12 hidden-control count drift")
    if repeat_count % shard_count:
        raise ValueError("Prior v12 repeats do not divide evenly across shards")
    repeats_per_shard = repeat_count // shard_count
    namespace = "clir-prior-v12"
    package_a, package_b, private, base_report = build_blind_packages(
        proposals,
        repeat_count_a=repeat_count,
        repeat_count_b=repeat_count,
        namespace=namespace,
        package_schema=PACKAGE_SCHEMA,
        private_schema=PRIVATE_SCHEMA,
        control_items=controls,
    )
    private_lookup = {
        (str(row["annotator"]), str(row["item_id"])): row for row in private
    }
    if len(private_lookup) != len(private):
        raise ValueError("Prior v12 private item identities are not unique")
    natural_ids = sorted(
        (str(row["proposal_id"]) for row in proposals),
        key=lambda item_id: stable_priority(
            f"{namespace}-natural-shard-order", item_id
        ),
    )
    natural_shard = {
        item_id: index // natural_per_shard for index, item_id in enumerate(natural_ids)
    }
    control_ids = sorted(
        (str(row["item_id"]) for row in controls),
        key=lambda item_id: stable_priority(
            f"{namespace}-control-shard-order", item_id
        ),
    )
    control_shard = {item_id: index for index, item_id in enumerate(control_ids)}

    packages: dict[str, list[list[dict[str, Any]]]] = {}
    enriched_private: list[dict[str, Any]] = []
    for annotator, flat in (("a", package_a), ("b", package_b)):
        repeat_ids = sorted(
            (
                str(row["item_id"])
                for row in private
                if row["annotator"] == annotator and row["kind"] == "repeat"
            ),
            key=lambda item_id: stable_priority(
                f"{namespace}-{annotator}-repeat-shard-order", item_id
            ),
        )
        repeat_parent_shard = {
            item_id: natural_shard[
                str(private_lookup[(annotator, item_id)]["natural_item_id"])
            ]
            for item_id in repeat_ids
        }
        repeat_slots = [
            (shard_index, slot_index)
            for shard_index in range(shard_count)
            for slot_index in range(repeats_per_shard)
        ]
        slot_owner: dict[tuple[int, int], str] = {}

        def assign_repeat(item_id: str, seen: set[tuple[int, int]]) -> bool:
            ordered_slots = sorted(
                repeat_slots,
                key=lambda slot: stable_priority(
                    f"{namespace}-{annotator}-repeat-slot",
                    item_id,
                    slot[0],
                    slot[1],
                ),
            )
            for slot in ordered_slots:
                if slot[0] == repeat_parent_shard[item_id] or slot in seen:
                    continue
                seen.add(slot)
                previous = slot_owner.get(slot)
                if previous is None or assign_repeat(previous, seen):
                    slot_owner[slot] = item_id
                    return True
            return False

        for item_id in repeat_ids:
            if not assign_repeat(item_id, set()):
                raise AssertionError("Prior v12 could not isolate blind repeats")
        if len(slot_owner) != repeat_count:
            raise AssertionError("Prior v12 repeat-slot population drift")
        repeat_shard = {
            item_id: shard_index for (shard_index, _), item_id in slot_owner.items()
        }
        if any(
            repeat_shard[item_id] == repeat_parent_shard[item_id]
            for item_id in repeat_ids
        ):
            raise AssertionError("Prior v12 repeat shares its natural parent shard")
        shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
        for raw in flat:
            row = dict(raw)
            item_id = str(row["item_id"])
            hidden = private_lookup[(annotator, item_id)]
            kind = str(hidden["kind"])
            if kind == "natural":
                shard_index = natural_shard[item_id]
            elif kind == "control":
                shard_index = control_shard[item_id]
            elif kind == "repeat":
                shard_index = repeat_shard[item_id]
            else:
                raise ValueError(f"unsupported Prior v12 package kind: {kind}")
            shards[shard_index].append(row)
            private_row = dict(hidden)
            private_row["annotation_shard_id"] = f"{annotator}-{shard_index:02d}"
            enriched_private.append(private_row)
        for shard_index, rows in enumerate(shards):
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}-{annotator}-shard-{shard_index:02d}",
                    str(row["item_id"]),
                )
            )
            kinds = Counter(
                str(private_lookup[(annotator, str(row["item_id"]))]["kind"])
                for row in rows
            )
            expected = {
                "natural": natural_per_shard,
                "control": 1,
                "repeat": repeats_per_shard,
            }
            if dict(kinds) != expected:
                raise AssertionError(
                    f"Prior v12 shard {annotator}-{shard_index:02d} drift: {kinds}"
                )
        packages[annotator] = shards
    enriched_private.sort(key=lambda row: (str(row["annotator"]), str(row["item_id"])))
    report = {
        **base_report,
        "shards_per_annotator": shard_count,
        "natural_per_shard": natural_per_shard,
        "controls_per_shard": 1,
        "repeats_per_shard": repeats_per_shard,
        "rows_per_shard": natural_per_shard + 1 + repeats_per_shard,
        "annotator_shards": {
            annotator: [
                {
                    "shard_id": f"{annotator}-{index:02d}",
                    "rows": len(rows),
                    "ordered_rows_sha256": canonical_sha256(rows),
                }
                for index, rows in enumerate(shards)
            ]
            for annotator, shards in packages.items()
        },
        "private_index_ordered_rows_sha256": canonical_sha256(enriched_private),
    }
    return packages, enriched_private, report


def validate_prior_v12_annotation(
    annotation: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact public v12 output schema against one package row."""

    required = {
        "item_id",
        "eligibility",
        "key_unit_indices",
        "complete_unit_indices",
        "confidence",
        "rationale",
    }
    if not isinstance(annotation, Mapping):
        raise ValueError("Prior v12 annotation must be a JSON object")
    if set(annotation) != required:
        missing = sorted(required - set(annotation))
        extra = sorted(set(annotation) - required)
        raise ValueError(
            f"Prior v12 annotation field mismatch: missing={missing}, extra={extra}"
        )
    normalized = validate_partial_prior_annotation(annotation, item)
    if (
        normalized["eligibility"] == "usable"
        and len(normalized["key_unit_indices"]) != 1
    ):
        raise ValueError("Prior v12 usable annotation requires exactly one Key unit")
    normalized["schema_version"] = LABEL_SCHEMA
    return normalized


def _v12_stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source"]),
        str(row["checker_status"]),
        str(row["prior_label_split"]),
    )


def _v12_stratum_name(stratum: Sequence[str]) -> str:
    return "|".join(str(value) for value in stratum)


def _v12_number_summary(values: Sequence[float | int]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    midpoint = len(numbers) // 2
    if len(numbers) % 2:
        median = numbers[midpoint]
    else:
        median = (numbers[midpoint - 1] + numbers[midpoint]) / 2.0
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": median,
        "mean": sum(numbers) / len(numbers),
        "max": numbers[-1],
    }


def _v12_fraction_gate(value: str) -> tuple[int, int]:
    parts = str(value).split("/")
    if len(parts) != 2:
        raise ValueError(f"invalid Prior v12 fraction gate: {value}")
    passed, total = (int(part) for part in parts)
    if passed < 0 or total <= 0 or passed > total:
        raise ValueError(f"invalid Prior v12 fraction gate: {value}")
    return passed, total


def _v12_package_population(
    *,
    annotator: str,
    package: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    expected_natural_count: int,
    expected_control_count: int,
    expected_repeat_count: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    public_fields = {"schema_version", "item_id", "question", "response", "units"}
    package_by_id: dict[str, dict[str, Any]] = {}
    for raw in package:
        row = dict(raw)
        if set(row) != public_fields:
            raise ValueError(f"Prior v12 package {annotator} public field drift")
        if row.get("schema_version") != PACKAGE_SCHEMA:
            raise ValueError(f"Prior v12 package {annotator} schema drift")
        item_id = str(row.get("item_id"))
        if item_id in package_by_id:
            raise ValueError(f"Prior v12 package {annotator} duplicate item_id")
        _material_units(row)
        package_by_id[item_id] = row

    expected_rows = (
        expected_natural_count + expected_control_count + expected_repeat_count
    )
    if len(package_by_id) != expected_rows:
        raise ValueError(
            f"Prior v12 package {annotator} row count drift: "
            f"{len(package_by_id)}/{expected_rows}"
        )

    private_by_id: dict[str, dict[str, Any]] = {}
    for raw in private_index:
        if raw.get("annotator") != annotator:
            continue
        row = dict(raw)
        if row.get("schema_version") != PRIVATE_SCHEMA:
            raise ValueError(f"Prior v12 private index {annotator} schema drift")
        item_id = str(row.get("item_id"))
        if item_id in private_by_id:
            raise ValueError(f"Prior v12 private index {annotator} duplicate item_id")
        if row.get("kind") not in {"natural", "control", "repeat"}:
            raise ValueError(f"Prior v12 private index {annotator} has invalid kind")
        private_by_id[item_id] = row
    if set(private_by_id) != set(package_by_id):
        raise ValueError(f"Prior v12 package/private binding differs for {annotator}")
    kind_counts = Counter(str(row["kind"]) for row in private_by_id.values())
    expected_kinds = {
        "natural": expected_natural_count,
        "control": expected_control_count,
        "repeat": expected_repeat_count,
    }
    if dict(kind_counts) != expected_kinds:
        raise ValueError(
            f"Prior v12 private population drift for {annotator}: {dict(kind_counts)}"
        )

    for item_id, hidden in private_by_id.items():
        kind = str(hidden["kind"])
        if kind == "natural":
            if str(hidden.get("natural_item_id")) != item_id:
                raise ValueError("Prior v12 natural private binding drift")
        elif kind == "control":
            signature = hidden.get("expected_signature")
            if not isinstance(signature, (list, tuple)) or len(signature) != 3:
                raise ValueError("Prior v12 control expected signature is invalid")
        else:
            parent_id = str(hidden.get("natural_item_id"))
            if parent_id == item_id or parent_id not in private_by_id:
                raise ValueError("Prior v12 repeat parent binding is invalid")
            if private_by_id[parent_id].get("kind") != "natural":
                raise ValueError("Prior v12 repeat parent is not natural")
            repeat_payload = {
                key: package_by_id[item_id][key]
                for key in ("question", "response", "units")
            }
            parent_payload = {
                key: package_by_id[parent_id][key]
                for key in ("question", "response", "units")
            }
            if canonical_sha256(repeat_payload) != canonical_sha256(parent_payload):
                raise ValueError("Prior v12 repeat payload differs from its parent")

    normalized: dict[str, dict[str, Any]] = {}
    for raw in labels:
        item_id = str(raw.get("item_id"))
        if item_id in normalized:
            raise ValueError(f"Prior v12 labels {annotator} duplicate item_id")
        if item_id not in package_by_id:
            raise ValueError(f"Prior v12 labels {annotator} contain unknown item_id")
        normalized[item_id] = validate_prior_v12_annotation(raw, package_by_id[item_id])
    if set(normalized) != set(package_by_id):
        missing = sorted(set(package_by_id) - set(normalized))
        extra = sorted(set(normalized) - set(package_by_id))
        raise ValueError(
            f"Prior v12 labels {annotator} population differs: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return package_by_id, private_by_id, normalized


def _v12_complete_pair_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    material = {unit["unit_index"] for unit in _material_units(item)}
    left_complete = set(left["complete_unit_indices"])
    right_complete = set(right["complete_unit_indices"])
    intersection = left_complete & right_complete
    union = left_complete | right_complete
    symmetric_difference = left_complete ^ right_complete
    return {
        "material_count": len(material),
        "complete_iou": len(intersection) / max(1, len(union)),
        "complete_mask_coverage": (len(material) - len(symmetric_difference))
        / max(1, len(material)),
        "complete_intersection_count": len(intersection),
        "complete_union_count": len(union),
        "complete_ambiguous_count": len(symmetric_difference),
        "a_complete_count": len(left_complete),
        "b_complete_count": len(right_complete),
        "a_complete_fraction": len(left_complete) / max(1, len(material)),
        "b_complete_fraction": len(right_complete) / max(1, len(material)),
        "a_complete_is_all_material": left_complete == material,
        "b_complete_is_all_material": right_complete == material,
    }


def evaluate_prior_v12_labels(
    *,
    proposals: Sequence[Mapping[str, Any]],
    package_a: Sequence[Mapping[str, Any]],
    package_b: Sequence[Mapping[str, Any]],
    private_index: Sequence[Mapping[str, Any]],
    labels_a: Sequence[Mapping[str, Any]],
    labels_b: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen v12 strict-consensus gate without publishing targets."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported Prior v12 protocol schema")
    annotation = protocol["annotation"]
    strict = protocol["strict_consensus"]
    gates = protocol["gates"]
    natural_count = int(protocol["proposal_pool"]["natural_count"])
    control_count = int(annotation["hidden_controls_total_per_annotator"])
    repeat_count = int(annotation["self_repeats_total_per_annotator"])
    expected_private = 2 * (natural_count + control_count + repeat_count)
    if len(private_index) != expected_private:
        raise ValueError(
            f"Prior v12 private index row count drift: "
            f"{len(private_index)}/{expected_private}"
        )

    packages: dict[str, dict[str, dict[str, Any]]] = {}
    private: dict[str, dict[str, dict[str, Any]]] = {}
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for annotator, package, labels in (
        ("a", package_a, labels_a),
        ("b", package_b, labels_b),
    ):
        packages[annotator], private[annotator], normalized[annotator] = (
            _v12_package_population(
                annotator=annotator,
                package=package,
                labels=labels,
                private_index=private_index,
                expected_natural_count=natural_count,
                expected_control_count=control_count,
                expected_repeat_count=repeat_count,
            )
        )

    natural_ids_by_annotator = {
        annotator: {
            item_id
            for item_id, row in private[annotator].items()
            if row["kind"] == "natural"
        }
        for annotator in ("a", "b")
    }
    if natural_ids_by_annotator["a"] != natural_ids_by_annotator["b"]:
        raise ValueError("Prior v12 A/B natural item populations differ")
    natural_ids = natural_ids_by_annotator["a"]

    proposal_by_id: dict[str, dict[str, Any]] = {}
    for raw in proposals:
        row = dict(raw)
        if row.get("schema_version") != PROPOSAL_SCHEMA:
            raise ValueError("Prior v12 proposal schema drift")
        proposal_id = str(row.get("proposal_id"))
        if proposal_id in proposal_by_id:
            raise ValueError("Prior v12 proposal IDs are not unique")
        _material_units(row)
        proposal_by_id[proposal_id] = row
    if len(proposal_by_id) != natural_count or set(proposal_by_id) != natural_ids:
        raise ValueError("Prior v12 proposal/natural package population differs")
    if len({str(row["query_id"]) for row in proposal_by_id.values()}) != natural_count:
        raise ValueError("Prior v12 proposal query IDs are not unique")
    if (
        len({str(row["cluster_id"]) for row in proposal_by_id.values()})
        != natural_count
    ):
        raise ValueError("Prior v12 proposal cluster IDs are not unique")
    for item_id in natural_ids:
        proposal_payload = {
            key: proposal_by_id[item_id][key]
            for key in ("question", "response", "units")
        }
        for annotator in ("a", "b"):
            package_payload = {
                key: packages[annotator][item_id][key]
                for key in ("question", "response", "units")
            }
            if canonical_sha256(proposal_payload) != canonical_sha256(package_payload):
                raise ValueError("Prior v12 natural package/proposal payload drift")

    proposal_quotas = {
        (str(row["source"]), str(row["checker_status"]), str(row["split"])): int(
            row["count"]
        )
        for row in protocol["proposal_pool"]["strata"]
    }
    proposal_counts = Counter(_v12_stratum(row) for row in proposal_by_id.values())
    if dict(proposal_counts) != proposal_quotas:
        raise ValueError(
            "Prior v12 proposal stratum population differs from the frozen protocol"
        )

    control_minimum, control_gate_total = _v12_fraction_gate(
        str(gates["controls_min_per_annotator"])
    )
    if control_gate_total != control_count:
        raise ValueError("Prior v12 control gate denominator drift")
    control_metrics: dict[str, dict[str, Any]] = {}
    repeat_metrics: dict[str, dict[str, Any]] = {}
    for annotator in ("a", "b"):
        controls = [
            row for row in private[annotator].values() if row["kind"] == "control"
        ]
        controls_passed = 0
        for row in controls:
            expected = row["expected_signature"]
            expected_signature = (
                expected[0],
                tuple(expected[1]),
                tuple(expected[2]),
            )
            controls_passed += (
                target_signature(normalized[annotator][str(row["item_id"])])
                == expected_signature
            )
        control_metrics[annotator] = {
            "passed": controls_passed,
            "total": len(controls),
            "rate": controls_passed / max(1, len(controls)),
        }
        repeats = [
            row for row in private[annotator].values() if row["kind"] == "repeat"
        ]
        repeats_passed = sum(
            target_signature(normalized[annotator][str(row["item_id"])])
            == target_signature(normalized[annotator][str(row["natural_item_id"])])
            for row in repeats
        )
        repeat_metrics[annotator] = {
            "passed": repeats_passed,
            "total": len(repeats),
            "rate": repeats_passed / max(1, len(repeats)),
        }

    ordered_natural_ids = sorted(natural_ids)
    raw_eligibility_agreement = 0
    raw_common_usable = 0
    raw_common_nonlow = 0
    raw_exact_key = 0
    raw_nonempty_complete_intersection = 0
    strict_eligible: list[str] = []
    raw_pair_metrics: list[dict[str, Any]] = []
    raw_complete_f1: list[float] = []
    raw_complete_relations: Counter[str] = Counter()
    by_stratum: dict[tuple[str, str, str], dict[str, int]] = {
        stratum: {"raw": count, "eligible": 0}
        for stratum, count in proposal_counts.items()
    }
    for item_id in ordered_natural_ids:
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        if left["eligibility"] == right["eligibility"]:
            raw_eligibility_agreement += 1
        common_usable = left["eligibility"] == right["eligibility"] == "usable"
        if common_usable:
            raw_common_usable += 1
            raw_complete_f1.append(
                _set_f1(left["complete_unit_indices"], right["complete_unit_indices"])
            )
        common_nonlow = (
            common_usable
            and left["confidence"] != "low"
            and right["confidence"] != "low"
        )
        if common_nonlow:
            raw_common_nonlow += 1
            pair = _v12_complete_pair_metrics(left, right, packages["a"][item_id])
            raw_pair_metrics.append(pair)
            left_complete = set(left["complete_unit_indices"])
            right_complete = set(right["complete_unit_indices"])
            if left_complete == right_complete:
                raw_complete_relations["equal"] += 1
            elif left_complete < right_complete:
                raw_complete_relations["a_strict_subset_b"] += 1
            elif right_complete < left_complete:
                raw_complete_relations["b_strict_subset_a"] += 1
            else:
                raw_complete_relations["overlap_or_disjoint"] += 1
        exact_key = (
            common_nonlow
            and len(left["key_unit_indices"]) == 1
            and left["key_unit_indices"] == right["key_unit_indices"]
        )
        if exact_key:
            raw_exact_key += 1
        complete_intersection = set(left["complete_unit_indices"]) & set(
            right["complete_unit_indices"]
        )
        if exact_key and complete_intersection:
            raw_nonempty_complete_intersection += 1
            strict_eligible.append(item_id)
            by_stratum[_v12_stratum(proposal_by_id[item_id])]["eligible"] += 1

    final_quotas = {
        (str(row["source"]), str(row["checker_status"]), str(row["split"])): int(
            row["count"]
        )
        for row in strict["final_strata"]
    }
    if sum(final_quotas.values()) != int(strict["final_target_rows"]):
        raise ValueError("Prior v12 final strata do not sum to final_target_rows")
    if set(final_quotas) != set(proposal_quotas):
        raise ValueError("Prior v12 proposal/final stratum identities differ")
    eligible_by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item_id in strict_eligible:
        eligible_by_stratum[_v12_stratum(proposal_by_id[item_id])].append(item_id)
    for stratum in eligible_by_stratum:
        eligible_by_stratum[stratum].sort(
            key=lambda item_id: (
                str(proposal_by_id[item_id]["selection_priority"]),
                item_id,
            )
        )
    quota_feasible = all(
        len(eligible_by_stratum.get(stratum, [])) >= quota
        for stratum, quota in final_quotas.items()
    )
    selected_ids: list[str] = []
    if quota_feasible:
        for stratum in [
            (str(row["source"]), str(row["checker_status"]), str(row["split"]))
            for row in strict["final_strata"]
        ]:
            selected_ids.extend(eligible_by_stratum[stratum][: final_quotas[stratum]])

    selected_pair_metrics = [
        _v12_complete_pair_metrics(
            normalized["a"][item_id],
            normalized["b"][item_id],
            packages["a"][item_id],
        )
        for item_id in selected_ids
    ]
    selected_iou_mean = (
        sum(row["complete_iou"] for row in selected_pair_metrics)
        / len(selected_pair_metrics)
        if selected_pair_metrics
        else None
    )
    selected_coverage_mean = (
        sum(row["complete_mask_coverage"] for row in selected_pair_metrics)
        / len(selected_pair_metrics)
        if selected_pair_metrics
        else None
    )
    selected_all_material_rate = {
        annotator: (
            sum(
                row[f"{annotator}_complete_is_all_material"]
                for row in selected_pair_metrics
            )
            / len(selected_pair_metrics)
            if selected_pair_metrics
            else None
        )
        for annotator in ("a", "b")
    }

    def annotator_distribution(annotator: str, ids: Sequence[str]) -> dict[str, Any]:
        rows = [normalized[annotator][item_id] for item_id in ids]
        usable_ids = [
            item_id
            for item_id in ids
            if normalized[annotator][item_id]["eligibility"] == "usable"
        ]
        complete_counts = [
            len(normalized[annotator][item_id]["complete_unit_indices"])
            for item_id in usable_ids
        ]
        complete_fractions = [
            len(normalized[annotator][item_id]["complete_unit_indices"])
            / len(_material_units(packages[annotator][item_id]))
            for item_id in usable_ids
        ]
        all_material = sum(
            set(normalized[annotator][item_id]["complete_unit_indices"])
            == {
                unit["unit_index"]
                for unit in _material_units(packages[annotator][item_id])
            }
            for item_id in usable_ids
        )
        return {
            "rows": len(rows),
            "eligibility": dict(
                sorted(Counter(row["eligibility"] for row in rows).items())
            ),
            "confidence": dict(
                sorted(Counter(row["confidence"] for row in rows).items())
            ),
            "complete_unit_count": _v12_number_summary(complete_counts),
            "complete_material_fraction": _v12_number_summary(complete_fractions),
            "complete_all_material_rate": all_material / max(1, len(usable_ids)),
        }

    selected_strata = Counter(
        _v12_stratum(proposal_by_id[item_id]) for item_id in selected_ids
    )
    raw_metrics = {
        "natural_denominator": natural_count,
        "eligibility_agreement_count": raw_eligibility_agreement,
        "eligibility_agreement_rate": raw_eligibility_agreement / natural_count,
        "common_usable": raw_common_usable,
        "common_nonlow_usable": raw_common_nonlow,
        "exact_singleton_key_consensus": raw_exact_key,
        "nonempty_complete_intersection_after_key_consensus": (
            raw_nonempty_complete_intersection
        ),
        "strict_eligible_rows": len(strict_eligible),
        "strict_eligible_rate": len(strict_eligible) / natural_count,
        "strict_eligible_ordered_ids_sha256": canonical_sha256(sorted(strict_eligible)),
        "complete_macro_f1_common_usable": (
            sum(raw_complete_f1) / len(raw_complete_f1) if raw_complete_f1 else None
        ),
        "complete_iou_common_nonlow": _v12_number_summary(
            [row["complete_iou"] for row in raw_pair_metrics]
        ),
        "complete_mask_coverage_common_nonlow": _v12_number_summary(
            [row["complete_mask_coverage"] for row in raw_pair_metrics]
        ),
        "complete_set_relations_common_nonlow": dict(
            sorted(raw_complete_relations.items())
        ),
        "by_stratum": {
            _v12_stratum_name(stratum): {
                **counts,
                "eligible_rate": counts["eligible"] / counts["raw"],
                "final_quota": final_quotas[stratum],
            }
            for stratum, counts in sorted(by_stratum.items())
        },
        "annotator_a": annotator_distribution("a", ordered_natural_ids),
        "annotator_b": annotator_distribution("b", ordered_natural_ids),
    }
    selected_metrics = {
        "selection_computable": quota_feasible,
        "target_rows": int(strict["final_target_rows"]),
        "selected_rows": len(selected_ids),
        "selected_ordered_ids_sha256": (
            canonical_sha256(selected_ids) if selected_ids else None
        ),
        "selected_by_stratum": {
            _v12_stratum_name(stratum): selected_strata.get(stratum, 0)
            for stratum in final_quotas
        },
        "complete_positive_iou_mean": selected_iou_mean,
        "complete_positive_iou": _v12_number_summary(
            [row["complete_iou"] for row in selected_pair_metrics]
        ),
        "complete_mask_coverage_mean": selected_coverage_mean,
        "complete_mask_coverage": _v12_number_summary(
            [row["complete_mask_coverage"] for row in selected_pair_metrics]
        ),
        "complete_all_material_rate": selected_all_material_rate,
        "material_claim_count": _v12_number_summary(
            [len(_material_units(packages["a"][item_id])) for item_id in selected_ids]
        ),
        "annotator_a": annotator_distribution("a", selected_ids),
        "annotator_b": annotator_distribution("b", selected_ids),
    }

    gate_results = {
        "population_schema_id_package_binding": {
            "pass": True,
            "observed": {
                "natural": natural_count,
                "package_rows_per_annotator": len(package_a),
                "private_rows": len(private_index),
            },
            "required": True,
        },
        "controls_a": {
            "pass": control_metrics["a"]["passed"] >= control_minimum,
            "observed": control_metrics["a"],
            "required_minimum": control_minimum,
        },
        "controls_b": {
            "pass": control_metrics["b"]["passed"] >= control_minimum,
            "observed": control_metrics["b"],
            "required_minimum": control_minimum,
        },
        "self_repeat_a": {
            "pass": repeat_metrics["a"]["rate"] >= float(gates["self_repeat_min"]),
            "observed": repeat_metrics["a"],
            "required_minimum": float(gates["self_repeat_min"]),
        },
        "self_repeat_b": {
            "pass": repeat_metrics["b"]["rate"] >= float(gates["self_repeat_min"]),
            "observed": repeat_metrics["b"],
            "required_minimum": float(gates["self_repeat_min"]),
        },
        "every_final_stratum_quota": {
            "pass": quota_feasible,
            "observed_eligible": {
                _v12_stratum_name(stratum): len(eligible_by_stratum.get(stratum, []))
                for stratum in final_quotas
            },
            "required": {
                _v12_stratum_name(stratum): quota
                for stratum, quota in final_quotas.items()
            },
        },
        "selected_complete_positive_iou_mean": {
            "pass": selected_iou_mean is not None
            and selected_iou_mean
            >= float(gates["selected_complete_positive_iou_mean_min"]),
            "observed": selected_iou_mean,
            "required_minimum": float(gates["selected_complete_positive_iou_mean_min"]),
        },
        "selected_complete_mask_coverage_mean": {
            "pass": selected_coverage_mean is not None
            and selected_coverage_mean
            >= float(gates["selected_complete_mask_coverage_mean_min"]),
            "observed": selected_coverage_mean,
            "required_minimum": float(
                gates["selected_complete_mask_coverage_mean_min"]
            ),
        },
        "selected_complete_all_material_rate_a": {
            "pass": selected_all_material_rate["a"] is not None
            and selected_all_material_rate["a"]
            <= float(gates["selected_complete_all_material_rate_max"]),
            "observed": selected_all_material_rate["a"],
            "required_maximum": float(gates["selected_complete_all_material_rate_max"]),
        },
        "selected_complete_all_material_rate_b": {
            "pass": selected_all_material_rate["b"] is not None
            and selected_all_material_rate["b"]
            <= float(gates["selected_complete_all_material_rate_max"]),
            "observed": selected_all_material_rate["b"],
            "required_maximum": float(gates["selected_complete_all_material_rate_max"]),
        },
    }
    failed_gates = [name for name, result in gate_results.items() if not result["pass"]]
    passed = not failed_gates
    return {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "PASS_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE"
            if passed
            else "STOP_PRIOR_V12_STRICT_CONSENSUS_DATA_GATE_FAILURE"
        ),
        "failed_gates": failed_gates,
        "metrics": {
            "controls": control_metrics,
            "self_repeat": repeat_metrics,
            "raw_population": raw_metrics,
            "prospective_frozen_selection": selected_metrics,
        },
        "gates": gate_results,
        "strict_consensus_label_name": strict["label_name"],
        "prospective_selection_gate_passed": passed,
        "target_publication_authorized": False,
        "feature_extraction_allowed": False,
        "training_allowed": False,
        "failure_is_terminal": not passed,
        "next_gate": (
            "explicit_user_authorization_to_publish_500_silver_targets"
            if passed
            else "terminal_preserve_labels_no_relabel_no_adaptive_salvage"
        ),
        "claim_boundary": (
            "dual-AI strict-consensus Silver data operability only; no human "
            "verification, Gold status, natural-label accuracy, Prior learnability, "
            "mutual benefit, fixed-.25 gate efficacy, Best-of-N, or Full evidence"
        ),
    }


__all__ = [
    "LABEL_SCHEMA",
    "PACKAGE_SCHEMA",
    "PRIVATE_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PROTOCOL_SCHEMA",
    "QUERY_SCHEMA",
    "REPORT_SCHEMA",
    "build_acquisition_shards",
    "build_prior_annotation_shards",
    "evaluate_prior_v12_labels",
    "prior_v12_control_items",
    "select_acquisition_queries",
    "select_prior_proposals",
    "validate_prior_v12_annotation",
]
