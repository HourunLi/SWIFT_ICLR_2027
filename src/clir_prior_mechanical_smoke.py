"""Fresh v13 smoke construction and evaluation for mechanical CLIR Prior labels.

The smoke never rewrites or salvages terminal v12 labels.  It deterministically
selects previously unannotated train-side trajectories from the frozen v12
acquisition, compiles a local dependency-audit view, injects hidden controls and
self-repeats, and evaluates two blind AI annotation passes.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from numbers import Integral
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.clir_prior_mechanical import (
    PACKAGE_SCHEMA,
    compile_reasoning_structure,
    local_audit_target_signature,
    validate_local_audit_annotation,
)
from src.clir_smoke import stable_priority


PROPOSAL_SCHEMA = "clir-prior-mechanical-smoke-proposal-v13"
PRIVATE_SCHEMA = "clir-prior-mechanical-smoke-private-index-v13"
CONTROL_SCHEMA = "clir-prior-mechanical-control-v13"
DEFAULT_NAMESPACE = "clir-prior-mechanical-smoke-v13"


def _material_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for unit in row.get("units", []):
        if isinstance(unit, Mapping) and unit.get("kind") == "material_claim":
            output.append(
                {
                    "unit_index": int(unit["unit_index"]),
                    "kind": "material_claim",
                    "text": str(unit["text"]),
                }
            )
    output.sort(key=lambda item: item["unit_index"])
    return output


def _length_band(material_claim_count: int) -> str | None:
    if 8 <= material_claim_count <= 18:
        return "medium"
    if 19 <= material_claim_count <= 40:
        return "long"
    return None


def _eligible_natural(row: Mapping[str, Any]) -> bool:
    count = row.get("material_claim_count")
    if isinstance(count, bool) or not isinstance(count, Integral):
        return False
    return (
        row.get("acquisition_split") == "train"
        and row.get("source") in {"gsm8k", "math"}
        and row.get("checker_status") in {"numeric_match", "numeric_mismatch"}
        and row.get("status") == "ok"
        and row.get("unitization_status") == "ok"
        and row.get("eligible_for_supervision") is True
        and row.get("finish_reason") == "stop"
        and _length_band(int(count)) is not None
    )


def select_fresh_natural_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_query_ids: Iterable[str],
    excluded_cluster_ids: Iterable[str],
    strata: Sequence[Mapping[str, Any]],
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one fresh trajectory per query/cluster using only frozen fields."""

    excluded_queries = {str(value) for value in excluded_query_ids}
    excluded_clusters = {str(value) for value in excluded_cluster_ids}
    eligible = [
        row
        for row in rows
        if _eligible_natural(row)
        and str(row.get("query_id")) not in excluded_queries
        and str(row.get("cluster_id")) not in excluded_clusters
    ]
    used_queries: set[str] = set()
    used_clusters: set[str] = set()
    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}

    for stratum_index, stratum in enumerate(strata):
        source = str(stratum["source"])
        checker_status = str(stratum["checker_status"])
        length_band = str(stratum["length_band"])
        target = int(stratum["count"])
        key = f"{source}|{checker_status}|{length_band}"
        candidates = [
            row
            for row in eligible
            if row.get("source") == source
            and row.get("checker_status") == checker_status
            and _length_band(int(row["material_claim_count"])) == length_band
        ]
        candidates.sort(
            key=lambda row: stable_priority(
                f"{namespace}:stratum:{stratum_index}",
                row.get("query_id"),
                row.get("id"),
            )
        )
        available[key] = len(candidates)
        chosen = 0
        for row in candidates:
            query_id = str(row["query_id"])
            cluster_id = str(row["cluster_id"])
            if query_id in used_queries or cluster_id in used_clusters:
                continue
            material_units = _material_units(row)
            if len(material_units) != int(row["material_claim_count"]):
                raise ValueError(f"{row['id']}: material claim count drift")
            natural_id = (
                "prior-v13-natural-"
                + stable_priority(f"{namespace}:natural-id", row["id"])[:20]
            )
            proposal = {
                "schema_version": PROPOSAL_SCHEMA,
                "item_id": natural_id,
                "source_row_id": str(row["id"]),
                "query_id": query_id,
                "cluster_id": cluster_id,
                "source": source,
                "checker_status": checker_status,
                "length_band": length_band,
                "material_claim_count": len(material_units),
                "selection_priority": stable_priority(
                    f"{namespace}:selection", row["id"]
                ),
                "question": str(row["question"]),
                "response": str(row["response"]),
                "units": material_units,
            }
            selected.append(proposal)
            used_queries.add(query_id)
            used_clusters.add(cluster_id)
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise ValueError(f"v13 stratum {key} produced {chosen}/{target} fresh rows")

    expected = sum(int(row["count"]) for row in strata)
    if len(selected) != expected:
        raise AssertionError("v13 selected natural row count drift")
    if len(used_queries) != expected or len(used_clusters) != expected:
        raise AssertionError("v13 natural rows are not query/cluster distinct")
    report = {
        "namespace": namespace,
        "selected": len(selected),
        "distinct_queries": len(used_queries),
        "distinct_clusters": len(used_clusters),
        "excluded_query_count": len(excluded_queries),
        "excluded_cluster_count": len(excluded_clusters),
        "available_before_global_distinctness": available,
        "selected_strata": dict(
            sorted(
                Counter(
                    f"{row['source']}|{row['checker_status']}|{row['length_band']}"
                    for row in selected
                ).items()
            )
        ),
    }
    return selected, report


def public_package_item(
    source: Mapping[str, Any], *, item_id: str | None = None
) -> dict[str, Any]:
    output_id = str(item_id or source["item_id"])
    raw = {
        "item_id": output_id,
        "question": str(source["question"]),
        "response": str(source["response"]),
        "units": deepcopy(list(source["units"])),
    }
    structure = compile_reasoning_structure(raw)
    return {
        "schema_version": PACKAGE_SCHEMA,
        "item_id": output_id,
        "question": raw["question"],
        "response": raw["response"],
        "units": raw["units"],
        "structure": structure,
    }


def _raw_control(
    control_id: str,
    *,
    question: str,
    unit_texts: Sequence[str],
    adjacent_indices: bool = False,
) -> dict[str, Any]:
    indices = (
        list(range(len(unit_texts)))
        if adjacent_indices
        else [2 * i for i in range(len(unit_texts))]
    )
    return {
        "schema_version": CONTROL_SCHEMA,
        "item_id": control_id,
        "question": question,
        "response": "\n".join(unit_texts),
        "units": [
            {"unit_index": index, "kind": "material_claim", "text": text}
            for index, text in zip(indices, unit_texts)
        ],
    }


def _expected_audit(
    item: Mapping[str, Any],
    *,
    eligibility: str = "usable",
    path_status: str | None = "supported",
    roles: Sequence[str] = (),
    final_block_id: int | None = None,
    key_unit_index: int | None = None,
    kept_edges: Iterable[tuple[int, int]] = (),
    missing_edges: Sequence[Sequence[int]] = (),
    rationale: str,
) -> dict[str, Any]:
    if eligibility != "usable":
        annotation = {
            "item_id": item["item_id"],
            "eligibility": eligibility,
            "path_status": None,
            "block_roles": [],
            "final_block_id": None,
            "edge_decisions": [],
            "missing_edges": [],
            "key_unit_index": None,
            "confidence": "high",
            "rationale": rationale,
        }
        return validate_local_audit_annotation(annotation, item)

    structure = item["structure"]
    if len(roles) != int(structure["block_count"]):
        raise ValueError(f"{item['item_id']}: control role count drift")
    kept = {tuple(edge) for edge in kept_edges}
    annotation = {
        "item_id": item["item_id"],
        "eligibility": "usable",
        "path_status": path_status,
        "block_roles": [
            {"block_id": index, "role": role} for index, role in enumerate(roles)
        ],
        "final_block_id": final_block_id,
        "edge_decisions": [
            {
                "parent_block_id": int(edge["parent_block_id"]),
                "child_block_id": int(edge["child_block_id"]),
                "decision": (
                    "keep"
                    if (int(edge["parent_block_id"]), int(edge["child_block_id"]))
                    in kept
                    else "drop"
                ),
            }
            for edge in structure["candidate_edges"]
        ],
        "missing_edges": [list(edge) for edge in missing_edges],
        "key_unit_index": key_unit_index,
        "confidence": "high",
        "rationale": rationale,
    }
    return validate_local_audit_annotation(annotation, item)


def build_hidden_controls(
    annotator: str,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Return eight public controls plus private expected normalized labels."""

    if annotator not in {"a", "b"}:
        raise ValueError("annotator must be a or b")
    specs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    specs.append(
        (
            "split_calculation",
            _raw_control(
                "",
                question="Three groups of four items receive two more items. How many?",
                unit_texts=["3 * 4", "= 12", "12 + 2 = 14", "The answer is 14."],
                adjacent_indices=True,
            ),
            {
                "roles": ["main_step", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 2,
                "kept_edges": {(0, 1)},
                "rationale": "the split expression/result and final addition form the main path",
            },
        )
    )
    specs.append(
        (
            "premise_restatement",
            _raw_control(
                "",
                question="Nine pencils and six pencils were sold, then two were returned. How many remain sold?",
                unit_texts=[
                    "9 + 6 = 15",
                    "The problem states that 2 pencils were returned.",
                    "15 - 2 = 13",
                    "The answer is 13.",
                ],
            ),
            {
                "roles": [
                    "main_step",
                    "premise_restatement",
                    "main_step",
                    "answer_wrapper",
                ],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 2)},
                "rationale": "the premise restatement is excluded because the final subtraction writes 2 explicitly",
            },
        )
    )
    specs.append(
        (
            "early_error",
            _raw_control(
                "",
                question="Nine boxes contain four items each, plus two extras. How many items?",
                unit_texts=["9 * 4 = 35", "35 + 2 = 37", "The answer is 37."],
            ),
            {
                "path_status": "flawed",
                "roles": ["main_step", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 0,
                "kept_edges": {(0, 1)},
                "rationale": "9 times 4 is 36 rather than 35; the next step inherits that error",
            },
        )
    )
    specs.append(
        (
            "generic_formula",
            _raw_control(
                "",
                question="A vehicle travels at 12 km/h for 3 hours. How far?",
                unit_texts=[
                    "The general formula states distance = speed * time.",
                    "distance = 12 * 3 = 36 km",
                    "The answer is 36 km.",
                ],
            ),
            {
                "roles": ["formula_only", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 2,
                "kept_edges": set(),
                "rationale": "the instantiated equation is sufficient; the generic formula is not part of Complete",
            },
        )
    )
    specs.append(
        (
            "duplicate_result",
            _raw_control(
                "",
                question="Six groups contain eight items each, then two are added. How many?",
                unit_texts=[
                    "6 * 8 = 48",
                    "6 * 8 = 48",
                    "48 + 2 = 50",
                    "The answer is 50.",
                ],
            ),
            {
                "roles": ["main_step", "duplicate", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 2)},
                "rationale": "the second copy is redundant; the first result feeds the final addition",
            },
        )
    )
    specs.append(
        (
            "unused_branch",
            _raw_control(
                "",
                question="Three bags have five items each, then four items are added. How many?",
                unit_texts=[
                    "3 * 5 = 15",
                    "7 * 2 = 14",
                    "15 + 4 = 19",
                    "The answer is 19.",
                ],
            ),
            {
                "roles": ["main_step", "unused_branch", "main_step", "answer_wrapper"],
                "final_block_id": 2,
                "key_unit_index": 4,
                "kept_edges": {(0, 2)},
                "rationale": "7 times 2 is an unused branch and does not support the answer",
            },
        )
    )
    specs.append(
        (
            "final_condition",
            _raw_control(
                "",
                question="The candidates are -3 and 3. Which candidate is positive?",
                unit_texts=[
                    "The candidate values are -3 and 3.",
                    "Since 3 > 0, the positive candidate is 3.",
                    "The answer is 3.",
                ],
            ),
            {
                "roles": ["premise_restatement", "main_step", "answer_wrapper"],
                "final_block_id": 1,
                "key_unit_index": 2,
                "kept_edges": set(),
                "rationale": "the positivity test itself first selects the requested answer",
            },
        )
    )
    specs.append(
        (
            "answer_only",
            _raw_control(
                "",
                question="What is six times seven?",
                unit_texts=["The answer is \\boxed{42}."],
            ),
            {
                "eligibility": "no_auditable_reasoning",
                "rationale": "the response only states the answer and has no auditable reasoning",
            },
        )
    )

    output: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for index, (name, raw, expected_kwargs) in enumerate(specs):
        item_id = f"prior-v13-control-{annotator}-{index:02d}"
        raw["item_id"] = item_id
        public = public_package_item(raw)
        expected = _expected_audit(public, **expected_kwargs)
        output.append((public, expected, name))
    return output


def build_blind_shards(
    proposals: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = 4,
    repeats_per_shard: int = 4,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, list[list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    """Build A/B shards with 12 natural, 2 controls, and 4 repeats each."""

    if shard_count != 4 or len(proposals) != 48 or repeats_per_shard != 4:
        raise ValueError("v13 smoke freezes 48 natural rows in four 18-row shards")
    public_natural = [public_package_item(row) for row in proposals]
    natural_shards = [
        public_natural[index::shard_count] for index in range(shard_count)
    ]
    if any(len(rows) != 12 for rows in natural_shards):
        raise AssertionError("natural shard balance drift")

    packages: dict[str, list[list[dict[str, Any]]]] = {"a": [], "b": []}
    private: list[dict[str, Any]] = []
    for annotator in ("a", "b"):
        controls = build_hidden_controls(annotator)
        shard_rows: list[list[dict[str, Any]]] = [list(rows) for rows in natural_shards]
        for control_index, (public, expected, name) in enumerate(controls):
            destination = control_index % shard_count
            shard_rows[destination].append(public)
            private.append(
                {
                    "schema_version": PRIVATE_SCHEMA,
                    "annotator": annotator,
                    "shard_index": destination,
                    "kind": "control",
                    "item_id": public["item_id"],
                    "natural_item_id": None,
                    "control_name": name,
                    "expected_label": expected,
                }
            )

        # Repeat the first four natural rows from each shard in the next shard.
        # Parent and repeat are never visible in the same context.
        repeat_counter = 0
        for parent_shard, parent_rows in enumerate(natural_shards):
            destination = (parent_shard + 1) % shard_count
            for local_index, parent in enumerate(parent_rows[:repeats_per_shard]):
                repeat_id = (
                    f"{parent['item_id']}:repeat:{annotator}:{repeat_counter:02d}"
                )
                repeated = public_package_item(parent, item_id=repeat_id)
                if (
                    repeated["structure"]["source_sha256"]
                    != parent["structure"]["source_sha256"]
                ):
                    raise AssertionError("repeat source hash differs from parent")
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
                        "parent_shard_index": parent_shard,
                        "repeat_local_index": local_index,
                    }
                )
                repeat_counter += 1

        for shard_index, rows in enumerate(shard_rows):
            natural_ids = {row["item_id"] for row in natural_shards[shard_index]}
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
            if len(natural_ids) != 12 or len(rows) != 18:
                raise AssertionError("v13 public shard composition drift")
            rows.sort(
                key=lambda row: stable_priority(
                    f"{namespace}:package:{annotator}:{shard_index}",
                    row["item_id"],
                )
            )
        packages[annotator] = shard_rows

    private.sort(key=lambda row: (row["annotator"], row["shard_index"], row["item_id"]))
    construction = {
        "annotators": ["a", "b"],
        "shards_per_annotator": shard_count,
        "natural_rows_per_annotator": 48,
        "controls_per_annotator": 8,
        "repeats_per_annotator": 16,
        "rows_per_shard": 18,
        "rows_per_annotator": 72,
        "parent_repeat_same_shard": 0,
    }
    return packages, private, construction


def _set_f1(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _set_iou(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _coverage(left: Iterable[int], right: Iterable[int], size: int) -> float:
    return 1.0 - len(set(left) ^ set(right)) / max(1, size)


def _decision_map(label: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    return {
        (int(row["parent_block_id"]), int(row["child_block_id"])): str(row["decision"])
        for row in label["edge_decisions"]
    }


def evaluate_blind_labels(
    *,
    packages: Mapping[str, Sequence[Mapping[str, Any]]],
    private_index: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate packages/labels and report preregistered v13 smoke gates."""

    normalized: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    package_maps: dict[str, dict[str, dict[str, Any]]] = {}
    private_maps: dict[str, dict[str, dict[str, Any]]] = {"a": {}, "b": {}}
    for row in private_index:
        private_maps[str(row["annotator"])][str(row["item_id"])] = dict(row)

    schema_errors: list[str] = []
    for annotator in ("a", "b"):
        package_map = {str(row["item_id"]): dict(row) for row in packages[annotator]}
        package_maps[annotator] = package_map
        label_map = {str(row.get("item_id")): dict(row) for row in labels[annotator]}
        if len(label_map) != len(labels[annotator]) or set(label_map) != set(
            package_map
        ):
            raise ValueError(f"annotator {annotator} label/package population mismatch")
        for item_id, item in package_map.items():
            try:
                normalized[annotator][item_id] = validate_local_audit_annotation(
                    label_map[item_id], item
                )
            except ValueError as exc:
                schema_errors.append(f"{annotator}:{item_id}:{exc}")
    if schema_errors:
        return {
            "schema_version": "clir-prior-mechanical-smoke-evaluation-v13",
            "status": "FAIL_PRIOR_V13_SCHEMA",
            "schema_errors": schema_errors,
            "trainable_labels_published": False,
        }

    control_report: dict[str, Any] = {}
    repeat_report: dict[str, Any] = {}
    for annotator in ("a", "b"):
        controls = [
            row for row in private_maps[annotator].values() if row["kind"] == "control"
        ]
        control_passes: list[dict[str, Any]] = []
        for row in controls:
            actual = normalized[annotator][str(row["item_id"])]
            expected = row["expected_label"]
            passed = local_audit_target_signature(
                actual
            ) == local_audit_target_signature(expected)
            control_passes.append(
                {"name": row["control_name"], "item_id": row["item_id"], "pass": passed}
            )
        control_report[annotator] = {
            "passed": sum(row["pass"] for row in control_passes),
            "total": len(control_passes),
            "items": control_passes,
        }

        repeats = [
            row for row in private_maps[annotator].values() if row["kind"] == "repeat"
        ]
        signature_exact = 0
        complete_exact = 0
        key_exact = 0
        for row in repeats:
            parent = normalized[annotator][str(row["natural_item_id"])]
            repeated = normalized[annotator][str(row["item_id"])]
            signature_exact += local_audit_target_signature(
                parent
            ) == local_audit_target_signature(repeated)
            complete_exact += (
                parent["complete_unit_indices"] == repeated["complete_unit_indices"]
            )
            key_exact += parent["key_unit_indices"] == repeated["key_unit_indices"]
        repeat_report[annotator] = {
            "total": len(repeats),
            "target_signature_exact": signature_exact,
            "target_signature_exact_rate": signature_exact / max(1, len(repeats)),
            "complete_exact": complete_exact,
            "complete_exact_rate": complete_exact / max(1, len(repeats)),
            "key_exact": key_exact,
            "key_exact_rate": key_exact / max(1, len(repeats)),
        }

    natural_ids = sorted(
        str(row["item_id"])
        for row in private_maps["a"].values()
        if row["kind"] == "natural"
    )
    if set(natural_ids) != {
        str(row["item_id"])
        for row in private_maps["b"].values()
        if row["kind"] == "natural"
    }:
        raise ValueError("A/B natural populations differ")

    common_usable: list[str] = []
    eligibility_exact = 0
    final_exact = 0
    key_exact = 0
    path_exact = 0
    complete_f1: list[float] = []
    complete_iou: list[float] = []
    complete_coverage: list[float] = []
    role_agree = role_total = 0
    edge_agree = edge_total = 0
    all_material_rows = 0
    missing_edge_rows = {"a": 0, "b": 0}
    for item_id in natural_ids:
        left = normalized["a"][item_id]
        right = normalized["b"][item_id]
        eligibility_exact += left["eligibility"] == right["eligibility"]
        if not (
            left["eligibility"] == right["eligibility"] == "usable"
            and left["confidence"] != "low"
            and right["confidence"] != "low"
        ):
            continue
        common_usable.append(item_id)
        final_exact += left["final_block_id"] == right["final_block_id"]
        key_exact += left["key_unit_indices"] == right["key_unit_indices"]
        path_exact += left["path_status"] == right["path_status"]
        complete_f1.append(
            _set_f1(left["complete_unit_indices"], right["complete_unit_indices"])
        )
        complete_iou.append(
            _set_iou(left["complete_unit_indices"], right["complete_unit_indices"])
        )
        size = int(package_maps["a"][item_id]["structure"]["material_unit_count"])
        complete_coverage.append(
            _coverage(
                left["complete_unit_indices"], right["complete_unit_indices"], size
            )
        )
        all_material_rows += (
            len(
                set(left["complete_unit_indices"]) | set(right["complete_unit_indices"])
            )
            == size
        )
        left_roles = {int(row["block_id"]): row["role"] for row in left["block_roles"]}
        right_roles = {
            int(row["block_id"]): row["role"] for row in right["block_roles"]
        }
        role_total += len(left_roles)
        role_agree += sum(left_roles[key] == right_roles[key] for key in left_roles)
        left_edges = _decision_map(left)
        right_edges = _decision_map(right)
        edge_total += len(left_edges)
        edge_agree += sum(left_edges[key] == right_edges[key] for key in left_edges)
        for annotator, label in (("a", left), ("b", right)):
            missing_edge_rows[annotator] += bool(label["missing_edges"])

    common_count = len(common_usable)
    cross = {
        "natural_rows": len(natural_ids),
        "eligibility_exact": eligibility_exact,
        "eligibility_exact_rate": eligibility_exact / max(1, len(natural_ids)),
        "common_usable_nonlow": common_count,
        "path_exact_rate": path_exact / max(1, common_count),
        "final_block_exact_rate": final_exact / max(1, common_count),
        "key_exact_rate": key_exact / max(1, common_count),
        "complete_macro_f1": mean(complete_f1) if complete_f1 else 0.0,
        "complete_macro_iou": mean(complete_iou) if complete_iou else 0.0,
        "complete_mask_coverage": mean(complete_coverage) if complete_coverage else 0.0,
        "role_decision_agreement": role_agree / max(1, role_total),
        "candidate_edge_decision_agreement": edge_agree / max(1, edge_total),
        "all_material_union_rate": all_material_rows / max(1, common_count),
        "missing_edge_row_rate_a": missing_edge_rows["a"] / max(1, common_count),
        "missing_edge_row_rate_b": missing_edge_rows["b"] / max(1, common_count),
    }

    checks = {
        "controls_a": control_report["a"]["passed"] >= int(gates["controls_min_pass"]),
        "controls_b": control_report["b"]["passed"] >= int(gates["controls_min_pass"]),
        "self_repeat_a": repeat_report["a"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "self_repeat_b": repeat_report["b"]["target_signature_exact_rate"]
        >= float(gates["self_repeat_target_exact_min"]),
        "common_usable": common_count >= int(gates["common_usable_nonlow_min"]),
        "final_block": cross["final_block_exact_rate"]
        >= float(gates["final_block_exact_min"]),
        "key": cross["key_exact_rate"] >= float(gates["key_exact_min"]),
        "complete_f1": cross["complete_macro_f1"]
        >= float(gates["complete_macro_f1_min"]),
        "complete_iou": cross["complete_macro_iou"]
        >= float(gates["complete_macro_iou_min"]),
        "coverage": cross["complete_mask_coverage"]
        >= float(gates["complete_mask_coverage_min"]),
        "roles": cross["role_decision_agreement"]
        >= float(gates["role_decision_agreement_min"]),
        "edges": cross["candidate_edge_decision_agreement"]
        >= float(gates["edge_decision_agreement_min"]),
        "all_material": cross["all_material_union_rate"]
        <= float(gates["all_material_union_rate_max"]),
        "missing_edges_a": cross["missing_edge_row_rate_a"]
        <= float(gates["missing_edge_row_rate_max"]),
        "missing_edges_b": cross["missing_edge_row_rate_b"]
        <= float(gates["missing_edge_row_rate_max"]),
    }
    passed = all(checks.values())
    return {
        "schema_version": "clir-prior-mechanical-smoke-evaluation-v13",
        "status": (
            "PASS_PRIOR_V13_MECHANICAL_SMOKE"
            if passed
            else "STOP_PRIOR_V13_MECHANICAL_SMOKE"
        ),
        "schema_errors": [],
        "controls": control_report,
        "self_repeats": repeat_report,
        "cross_annotator_natural": cross,
        "gates": dict(gates),
        "gate_checks": checks,
        "trainable_labels_published": False,
        "next_step": (
            "freeze_separate_scale_protocol_v14"
            if passed
            else "stop_mechanical_prompt_iteration_and_consider_versioned_v12_exploratory_subset"
        ),
    }
