"""Hash-bound data construction for the CLIR Consistency v6.1 replication.

The expanded Consistency relations contain pair labels, while ``train_clir``
consumes trajectory rows.  This module performs the deterministic conversion
without changing any relation, feature, split, or correctness label.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.clir_data import first_present, resolve_feature_metadata
from src.clir_smoke import canonical_sha256, file_sha256


AUTHORIZATION_SCHEMA = "clir-consistency-scale-v6.1-c-only-training-authorization"
TRAIN_ROW_SCHEMA = "clir-consistency-scale-c-only-train-row-v6.1"
EVAL_ROW_SCHEMA = "clir-consistency-scale-heldout-endpoint-row-v6.1"


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, description: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Every {description} row requires non-empty {field}")
        if value in indexed:
            raise ValueError(f"Duplicate {description} {field}: {value}")
        indexed[value] = row
    return indexed


def _resolve_payload_path(value: Any, manifest_parent: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Feature row requires non-empty {field}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_parent / path
    return str(path.resolve())


def _feature_projection(
    row: Mapping[str, Any], manifest_parent: Path
) -> dict[str, Any]:
    metadata = resolve_feature_metadata(row)
    required_metadata = {
        "feature_dim": metadata["feature_dim"],
        "num_feature_layers": metadata["num_feature_layers"],
        "per_layer_dim": metadata["per_layer_dim"],
    }
    if any(value is None for value in required_metadata.values()):
        raise ValueError(f"Feature row {row.get('id')} lacks the all-layer contract")
    output_ids = row.get("output_token_ids")
    prompt_ids = row.get("prompt_token_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise ValueError(f"Feature row {row.get('id')} lacks output_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError(f"Feature row {row.get('id')} lacks prompt_token_ids")
    if row.get("output_token_count", len(output_ids)) != len(output_ids):
        raise ValueError(f"Feature row {row.get('id')} output token count drift")
    if row.get("prompt_token_count", len(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"Feature row {row.get('id')} prompt token count drift")
    hidden_sha = first_present(row, ("hidden_states_sha256", "feature_sha256"))
    condition_sha = first_present(row, ("condition_states_sha256", "condition_sha256"))
    if not isinstance(hidden_sha, str) or not isinstance(condition_sha, str):
        raise ValueError(f"Feature row {row.get('id')} lacks payload checksums")
    return {
        "id": str(row["id"]),
        "query_id": str(row["query_id"]),
        "candidate_index": int(row["candidate_index"]),
        "source": row.get("source"),
        "source_subject": row.get("source_subject"),
        "source_level": row.get("source_level"),
        "cluster_id": row.get("cluster_id"),
        "acquisition_split": row.get("acquisition_split"),
        "prompt_token_ids": list(prompt_ids),
        "output_token_ids": list(output_ids),
        "prompt_token_count": len(prompt_ids),
        "output_token_count": len(output_ids),
        "hidden_states_path": _resolve_payload_path(
            row.get("hidden_states_path"), manifest_parent, "hidden_states_path"
        ),
        "condition_states_path": _resolve_payload_path(
            row.get("condition_states_path"), manifest_parent, "condition_states_path"
        ),
        "hidden_states_sha256": hidden_sha,
        "condition_states_sha256": condition_sha,
        **required_metadata,
        "feature_model": row.get("feature_model"),
        "feature_revision": row.get("feature_revision"),
        "feature_dtype": row.get("feature_dtype"),
        "feature_attention_implementation": row.get("feature_attention_implementation"),
    }


def _historical_projection(
    row: Mapping[str, Any], manifest_parent: Path
) -> dict[str, Any]:
    projected = _feature_projection(row, manifest_parent)
    correctness = row.get("correctness")
    if correctness not in {0, 1, 0.0, 1.0}:
        raise ValueError(f"Historical row {row.get('id')} has invalid correctness")
    projected.update(
        {
            "schema_version": TRAIN_ROW_SCHEMA,
            "correctness": int(correctness),
            "split": row.get("split"),
            "source_index": row.get("source_index"),
            "training_population": "historical_correctness_v1",
            "consistency_supervision": False,
        }
    )
    return projected


def relative_length_roles(
    relation: Mapping[str, Any], features_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """Assign compact/expanded by saved output length with an ID tie-break."""

    left_id = str(relation["left_id"])
    right_id = str(relation["right_id"])
    try:
        left = features_by_id[left_id]
        right = features_by_id[right_id]
    except KeyError as exc:
        raise ValueError(f"Relation endpoint is absent from feature manifest: {exc}")
    left_key = (int(left["output_token_count"]), left_id)
    right_key = (int(right["output_token_count"]), right_id)
    compact, expanded = (
        (left_id, right_id) if left_key < right_key else (right_id, left_id)
    )
    return {compact: "relative_compact", expanded: "relative_expanded"}


def _validate_positive_relations(
    relations: Sequence[Mapping[str, Any]],
    features_by_id: Mapping[str, Mapping[str, Any]],
    expected_split: str,
) -> tuple[set[str], set[str], set[str]]:
    endpoint_ids: set[str] = set()
    query_ids: set[str] = set()
    cluster_ids: set[str] = set()
    relation_ids: set[str] = set()
    for relation in relations:
        relation_id = relation.get("relation_id")
        if not isinstance(relation_id, str) or not relation_id:
            raise ValueError("Positive relation lacks relation_id")
        if relation_id in relation_ids:
            raise ValueError(f"Duplicate positive relation_id: {relation_id}")
        relation_ids.add(relation_id)
        if relation.get("label") != 1:
            raise ValueError(f"Positive relation {relation_id} does not have label=1")
        if relation.get("acquisition_split") != expected_split:
            raise ValueError(f"Positive relation {relation_id} has split drift")
        left_id, right_id = str(relation["left_id"]), str(relation["right_id"])
        if left_id == right_id or left_id in endpoint_ids or right_id in endpoint_ids:
            raise ValueError(
                f"Positive endpoints must be distinct and relation-disjoint: {relation_id}"
            )
        if left_id not in features_by_id or right_id not in features_by_id:
            raise ValueError(f"Positive relation {relation_id} has missing feature")
        left, right = features_by_id[left_id], features_by_id[right_id]
        if left["query_id"] != right["query_id"]:
            raise ValueError(f"Positive relation {relation_id} crosses queries")
        if left.get("cluster_id") != right.get("cluster_id"):
            raise ValueError(f"Positive relation {relation_id} crosses clusters")
        if (
            left.get("acquisition_split") != expected_split
            or right.get("acquisition_split") != expected_split
        ):
            raise ValueError(f"Positive relation {relation_id} feature split drift")
        if str(relation.get("query_id")) != str(left["query_id"]):
            raise ValueError(f"Positive relation {relation_id} query identity drift")
        endpoint_ids.update((left_id, right_id))
        query_ids.add(str(left["query_id"]))
        cluster_ids.add(str(left.get("cluster_id")))
    return endpoint_ids, query_ids, cluster_ids


def _validate_negative_relations(
    relations: Sequence[Mapping[str, Any]],
    features_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    endpoint_ids: set[str] = set()
    query_ids: set[str] = set()
    cluster_ids: set[str] = set()
    relation_ids: set[str] = set()
    for relation in relations:
        relation_id = relation.get("relation_id")
        if not isinstance(relation_id, str) or not relation_id:
            raise ValueError("Negative relation lacks relation_id")
        if relation_id in relation_ids:
            raise ValueError(f"Duplicate negative relation_id: {relation_id}")
        relation_ids.add(relation_id)
        if relation.get("label") != 0 or relation.get("evaluation_only") is not True:
            raise ValueError(
                f"Hard negative {relation_id} must be label=0 and evaluation_only"
            )
        left_id, right_id = str(relation["left_id"]), str(relation["right_id"])
        if left_id == right_id:
            raise ValueError(f"Hard negative {relation_id} repeats one endpoint")
        if left_id in endpoint_ids or right_id in endpoint_ids:
            raise ValueError(
                f"Hard-negative endpoints must form a matching: {relation_id}"
            )
        if left_id not in features_by_id or right_id not in features_by_id:
            raise ValueError(f"Hard negative {relation_id} has missing feature")
        left, right = features_by_id[left_id], features_by_id[right_id]
        if left["query_id"] == right["query_id"]:
            raise ValueError(f"Hard negative {relation_id} is same-query")
        if left.get("cluster_id") == right.get("cluster_id"):
            raise ValueError(f"Hard negative {relation_id} is same-cluster")
        if (
            left.get("acquisition_split") != "heldout_acquisition"
            or right.get("acquisition_split") != "heldout_acquisition"
        ):
            raise ValueError(f"Hard negative {relation_id} is not held out")
        endpoint_ids.update((left_id, right_id))
        query_ids.update((str(left["query_id"]), str(right["query_id"])))
        cluster_ids.update((str(left.get("cluster_id")), str(right.get("cluster_id"))))
    return endpoint_ids, query_ids, cluster_ids


def construct_manifests(
    historical_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    train_positive_relations: Sequence[Mapping[str, Any]],
    heldout_positive_relations: Sequence[Mapping[str, Any]],
    heldout_negative_relations: Sequence[Mapping[str, Any]],
    *,
    historical_manifest_parent: str | Path,
    feature_manifest_parent: str | Path,
) -> dict[str, Any]:
    """Construct the matched train, positive-view validation, and eval manifests."""

    historical_parent = Path(historical_manifest_parent)
    feature_parent = Path(feature_manifest_parent)
    historical_by_id = _index_unique(historical_rows, "id", "historical")
    features_by_id = _index_unique(feature_rows, "id", "extracted feature")
    train_endpoints, train_queries, train_clusters = _validate_positive_relations(
        train_positive_relations, features_by_id, "train_acquisition"
    )
    heldout_positive_endpoints, heldout_positive_queries, heldout_positive_clusters = (
        _validate_positive_relations(
            heldout_positive_relations, features_by_id, "heldout_acquisition"
        )
    )
    heldout_negative_endpoints, heldout_negative_queries, heldout_negative_clusters = (
        _validate_negative_relations(heldout_negative_relations, features_by_id)
    )
    heldout_endpoints = heldout_positive_endpoints | heldout_negative_endpoints
    if set(features_by_id) != train_endpoints | heldout_endpoints:
        raise ValueError(
            "Extracted feature population is not the exact relation endpoint union"
        )
    heldout_queries = heldout_positive_queries | heldout_negative_queries
    heldout_clusters = heldout_positive_clusters | heldout_negative_clusters
    if train_queries & heldout_queries:
        raise ValueError("Train/heldout query overlap")
    if train_clusters & heldout_clusters:
        raise ValueError("Train/heldout cluster overlap")

    historical_queries = {str(row["query_id"]) for row in historical_rows}
    if historical_queries & (train_queries | heldout_queries):
        raise ValueError("Historical/new query ID overlap")
    historical_source_keys = {
        (str(row.get("source")), str(row.get("split")), int(row["source_index"]))
        for row in historical_rows
        if row.get("source_index") is not None
    }
    new_source_keys: set[tuple[str, str, int]] = set()
    for row in feature_rows:
        parts = str(row["query_id"]).split(":")
        if len(parts) >= 3 and parts[-1].isdigit():
            new_source_keys.add((parts[0], parts[1], int(parts[-1])))
    if historical_source_keys & new_source_keys:
        raise ValueError("Historical/new source identity overlap")

    train_rows = [
        _historical_projection(row, historical_parent) for row in historical_rows
    ]
    for relation in train_positive_relations:
        roles = relative_length_roles(relation, features_by_id)
        relation_id = str(relation["relation_id"])
        for endpoint_id in sorted(roles, key=lambda value: (roles[value], value)):
            projected = _feature_projection(features_by_id[endpoint_id], feature_parent)
            projected.update(
                {
                    "schema_version": TRAIN_ROW_SCHEMA,
                    "correctness": 1,
                    "semantic_id": f"v6.1:{relation_id}",
                    "style_id": roles[endpoint_id],
                    "training_population": "consistency_scale_v6_1_train_positive",
                    "consistency_supervision": True,
                    "consistency_relation_id": relation_id,
                    "consistency_label_tier": relation.get("label_tier"),
                }
            )
            train_rows.append(projected)

    validation_rows: list[dict[str, Any]] = []
    for relation in heldout_positive_relations:
        roles = relative_length_roles(relation, features_by_id)
        relation_id = str(relation["relation_id"])
        for endpoint_id in sorted(roles, key=lambda value: (roles[value], value)):
            projected = _feature_projection(features_by_id[endpoint_id], feature_parent)
            projected.update(
                {
                    "schema_version": TRAIN_ROW_SCHEMA,
                    "correctness": 1,
                    "semantic_id": f"v6.1-heldout:{relation_id}",
                    "style_id": roles[endpoint_id],
                    "training_population": "consistency_scale_v6_1_heldout_positive",
                    "consistency_supervision": True,
                    "consistency_relation_id": relation_id,
                    "consistency_label_tier": relation.get("label_tier"),
                }
            )
            validation_rows.append(projected)

    evaluation_rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        endpoint_id = str(feature["id"])
        if endpoint_id not in heldout_endpoints:
            continue
        projected = _feature_projection(feature, feature_parent)
        projected.update(
            {
                "schema_version": EVAL_ROW_SCHEMA,
                "evaluation_population": "consistency_scale_v6_1_heldout_endpoint_union",
            }
        )
        evaluation_rows.append(projected)

    train_ids = [str(row["id"]) for row in train_rows]
    validation_ids = [str(row["id"]) for row in validation_rows]
    evaluation_ids = [str(row["id"]) for row in evaluation_rows]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Constructed training rows contain duplicate trajectory IDs")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Constructed validation rows contain duplicate trajectory IDs")
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Constructed evaluation rows contain duplicate trajectory IDs")
    if set(train_ids) & set(evaluation_ids):
        raise ValueError("Constructed train/evaluation endpoint overlap")
    if set(evaluation_ids) != heldout_endpoints:
        raise ValueError("Constructed evaluation endpoint population drift")

    train_style_counts = Counter(
        row.get("style_id") for row in train_rows if row.get("consistency_supervision")
    )
    heldout_overlap = heldout_positive_endpoints & heldout_negative_endpoints
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "evaluation_rows": evaluation_rows,
        "statistics": {
            "historical_correctness_rows": len(historical_by_id),
            "train_consistency_relations": len(train_positive_relations),
            "train_consistency_endpoint_rows": len(train_endpoints),
            "train_rows": len(train_rows),
            "train_queries": len(historical_queries | train_queries),
            "train_style_counts": dict(sorted(train_style_counts.items())),
            "heldout_positive_relations": len(heldout_positive_relations),
            "heldout_negative_relations": len(heldout_negative_relations),
            "validation_rows": len(validation_rows),
            "evaluation_endpoint_rows": len(evaluation_rows),
            "heldout_endpoint_overlap_positive_negative": len(heldout_overlap),
            "heldout_query_overlap_positive_negative": len(
                heldout_positive_queries & heldout_negative_queries
            ),
            "heldout_cluster_overlap_positive_negative": len(
                heldout_positive_clusters & heldout_negative_clusters
            ),
            "train_heldout_query_overlap": len(train_queries & heldout_queries),
            "train_heldout_cluster_overlap": len(train_clusters & heldout_clusters),
            "historical_new_source_identity_overlap": len(
                historical_source_keys & new_source_keys
            ),
        },
        "canonical_hashes": {
            "train_rows": canonical_sha256(train_rows),
            "validation_rows": canonical_sha256(validation_rows),
            "evaluation_rows": canonical_sha256(evaluation_rows),
        },
    }


def load_authorization(path: str | Path) -> dict[str, Any]:
    authorization = json.loads(Path(path).read_text(encoding="utf-8"))
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("Unexpected C-only training authorization schema")
    if authorization.get("status") != "AUTHORIZED_C0_C1_MECHANISM_REPLICATION":
        raise ValueError("C-only training is not authorized")
    scope = authorization.get("authorized_scope", {})
    required_true = (
        "deterministic_manifest_construction",
        "full_width_preflight",
        "seed_42_one_epoch_pilot",
        "c0_c1_three_seed_three_epoch_training",
        "heldout_relation_evaluation",
    )
    if any(scope.get(key) is not True for key in required_true):
        raise ValueError("Authorization is missing required C-only scope")
    forbidden = (
        "hallucination_training",
        "dual_prior_training",
        "full_integration_training",
        "new_feature_extraction",
        "ranking_efficacy_claim",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise ValueError("Authorization improperly expands beyond C-only scope")
    return authorization


def verify_authorized_files(
    authorization: Mapping[str, Any], project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verified: dict[str, Any] = {}
    for name, specification in authorization["frozen_inputs"].items():
        path = root / str(specification["path"])
        actual = file_sha256(path)
        expected = str(specification["file_sha256"])
        if actual != expected:
            raise ValueError(
                f"Frozen input hash drift for {name}: {actual} != {expected}"
            )
        if "row_count" in specification:
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
                        count += 1
            if count != int(specification["row_count"]):
                raise ValueError(f"Frozen input row-count drift for {name}")
        verified[name] = {"path": str(path), "file_sha256": actual}
    for name, specification in authorization["execution_configs"].items():
        path = root / str(specification["path"])
        actual = file_sha256(path)
        if actual != str(specification["file_sha256"]):
            raise ValueError(f"Execution config hash drift for {name}")
        verified[f"config:{name}"] = {"path": str(path), "file_sha256": actual}
    return verified


def file_identity(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    return {
        "path": str(value.resolve()),
        "file_sha256": file_sha256(value),
        "serialized_bytes": value.stat().st_size,
    }


def relation_signature(relations: Sequence[Mapping[str, Any]]) -> str:
    records = [
        {
            "relation_id": str(row["relation_id"]),
            "label": int(row["label"]),
            "left_id": str(row["left_id"]),
            "right_id": str(row["right_id"]),
        }
        for row in relations
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "EVAL_ROW_SCHEMA",
    "TRAIN_ROW_SCHEMA",
    "construct_manifests",
    "file_identity",
    "load_authorization",
    "relation_signature",
    "relative_length_roles",
    "verify_authorized_files",
]
