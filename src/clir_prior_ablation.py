"""Pure contracts for the prospective CLIR Prior component ablation.

The GPU entry points deliberately depend on these small, CPU-testable helpers.
They keep the 19-cell factor table, generated training configurations, query
selection, and contrast algebra deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from src.clir_smoke import stable_priority


SCHEMA = "clir-prior-component-ablation-v2"
FACTOR_ORDER = (
    "consistency",
    "hallucination_h0",
    "key_direct",
    "complete_direct",
    "mutual",
    "gate",
    "fusion_alpha",
)
EXPECTED_CELLS = (
    "u0",
    "k",
    "complete",
    "kc",
    "kcm",
    "kcg",
    "c",
    "c_kc",
    "c_kcg",
    "h",
    "h_kc",
    "h_kcg",
    "ch",
    "ch_kc",
    "ch_kcm",
    "full",
    "ch_kcmg",
    "ch_kcg_key",
    "ch_kcg_complete",
)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Prior ablation protocol")
    if protocol.get("status") != "AUTHORIZED_FREEZE_AND_RUN_AFTER_ALL_GPUS_IDLE":
        raise ValueError("Prior ablation protocol is not authorized")
    if tuple(protocol.get("cell_factor_order", ())) != FACTOR_ORDER:
        raise ValueError("Prior ablation factor order drift")
    cells = protocol.get("cells")
    if not isinstance(cells, Mapping) or tuple(cells) != EXPECTED_CELLS:
        raise ValueError("Prior ablation cell order or membership drift")
    seen: set[tuple[float, ...]] = set()
    for cell, raw in cells.items():
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError(f"{cell}: factors must be a sequence")
        values = tuple(float(value) for value in raw)
        if len(values) != len(FACTOR_ORDER):
            raise ValueError(f"{cell}: factor width drift")
        if any(value not in {0.0, 1.0} for value in values[:6]):
            raise ValueError(f"{cell}: binary factor drift")
        if not 0.0 <= values[6] <= 1.0:
            raise ValueError(f"{cell}: fusion alpha outside [0, 1]")
        if (values[4] or values[5]) and not (values[2] and values[3]):
            raise ValueError(f"{cell}: mutual/Gate requires paired Key+Complete")
        if values in seen:
            raise ValueError(f"{cell}: duplicate factor vector")
        seen.add(values)
    training = protocol["training"]
    new_cells = tuple(training["new_cells"])
    reused = tuple(training["reused_cells"])
    if set(new_cells) & set(reused) or set(new_cells) | set(reused) != set(cells):
        raise ValueError("new/reused cell partition drift")
    if tuple(int(value) for value in training["seeds"]) != (42, 43, 44):
        raise ValueError("seed grid drift")
    if int(training["epochs"]) != 3 or training["prior_phase_mode"] != "joint":
        raise ValueError("training schedule drift")
    ranking = protocol["ranking_population"]
    source_total = sum(int(value) for value in ranking["source_query_counts"].values())
    if source_total != int(ranking["total_queries"]):
        raise ValueError("ranking source counts do not sum to total")
    if int(ranking["selected_candidate_rows"]) != source_total * int(
        protocol["generation"]["candidate_count"]
    ):
        raise ValueError("ranking candidate-row arithmetic drift")


def factor_map(protocol: Mapping[str, Any], cell: str) -> dict[str, float]:
    validate_protocol(protocol)
    if cell not in protocol["cells"]:
        raise KeyError(cell)
    return {
        key: float(value)
        for key, value in zip(
            FACTOR_ORDER, protocol["cells"][cell], strict=True
        )
    }


def derive_config(
    protocol: Mapping[str, Any], base: Mapping[str, Any], cell: str
) -> dict[str, Any]:
    """Derive one cell while changing only the predeclared objective factors."""

    factors = factor_map(protocol, cell)
    if set(base) != {"model", "training"}:
        raise ValueError("base config must contain model and training only")
    output = deepcopy(base)
    model = output["model"]
    constants = protocol["method_constants"]
    prior_on = float(
        factors["key_direct"]
        or factors["complete_direct"]
        or factors["mutual"]
        or factors["gate"]
    )
    model.update(
        {
            "consistency_weight": factors["consistency"]
            * float(constants["consistency_weight_when_on"]),
            "hallucination_weight": factors["hallucination_h0"]
            * float(constants["hallucination_h0_bce_weight_when_on"]),
            "prior_weight": prior_on,
            # The historical no-Prior anchors keep these nested coefficients at
            # one while their enclosing prior_weight is zero.  Preserve that
            # inert value so U0/C/H/CH remain byte-for-byte comparable.
            "key_prior_weight": (
                factors["key_direct"] * float(constants["direct_prior_weight"])
                if prior_on
                else float(model["key_prior_weight"])
            ),
            "complete_prior_weight": (
                factors["complete_direct"]
                * float(constants["direct_prior_weight"])
                if prior_on
                else float(model["complete_prior_weight"])
            ),
            "prior_distill_weight": factors["mutual"]
            * float(constants["mutual_distill_weight"]),
            "gate_prior_weight": factors["gate"]
            * float(constants["main_style_gate_weight"]),
            "prior_fusion_alpha": factors["fusion_alpha"],
            "token_reward_weight": float(constants["negative_tail_h1_weight"]),
            "tail_weight": float(constants["negative_tail_h1_weight"]),
            "mil_weight": float(constants["path_mil_weight"]),
            "pseudo_tail_weight": float(constants["pseudo_tail_weight"]),
            "progress_weight": float(constants["progress_weight"]),
            "reconstruction_weight": float(constants["reconstruction_weight"]),
        }
    )
    training = output["training"]
    frozen_training = protocol["training"]
    training.update(
        {
            "seed": int(frozen_training["seeds"][0]),
            "epochs": int(frozen_training["epochs"]),
            "batch_size": int(frozen_training["batch_size"]),
            "learning_rate": float(frozen_training["learning_rate"]),
            "weight_decay": float(frozen_training["weight_decay"]),
            "max_grad_norm": float(frozen_training["max_grad_norm"]),
            "amp_dtype": str(frozen_training["amp_dtype"]),
            "prior_phase_mode": str(frozen_training["prior_phase_mode"]),
        }
    )
    return output


def config_factor_projection(config: Mapping[str, Any]) -> dict[str, float]:
    model = config["model"]
    return {
        "consistency_weight": float(model["consistency_weight"]),
        "hallucination_weight": float(model["hallucination_weight"]),
        "prior_weight": float(model["prior_weight"]),
        "key_prior_weight": float(model["key_prior_weight"]),
        "complete_prior_weight": float(model["complete_prior_weight"]),
        "prior_distill_weight": float(model["prior_distill_weight"]),
        "gate_prior_weight": float(model["gate_prior_weight"]),
        "prior_fusion_alpha": float(model["prior_fusion_alpha"]),
        "token_reward_weight": float(model["token_reward_weight"]),
        "tail_weight": float(model["tail_weight"]),
        "mil_weight": float(model["mil_weight"]),
        "pseudo_tail_weight": float(model["pseudo_tail_weight"]),
        "progress_weight": float(model["progress_weight"]),
        "reconstruction_weight": float(model["reconstruction_weight"]),
    }


def one_per_cluster_then_hash(
    rows: Sequence[Mapping[str, Any]], *, namespace: str
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        cluster_id = str(row.get("cluster_id", ""))
        query_id = str(row.get("query_id", ""))
        if not cluster_id or not query_id:
            raise ValueError("query selection requires cluster_id and query_id")
        by_cluster[cluster_id].append(row)
    representatives: list[dict[str, Any]] = []
    for cluster_id, members in sorted(by_cluster.items()):
        members.sort(
            key=lambda row: stable_priority(
                f"{namespace}-within-cluster", str(row["query_id"])
            )
        )
        chosen = members[0]
        chosen["prior_ablation_cluster_member_count"] = len(members)
        representatives.append(chosen)
    representatives.sort(
        key=lambda row: stable_priority(
            f"{namespace}-representative-order", str(row["query_id"])
        )
    )
    return representatives


def select_query_rows(
    rows: Sequence[Mapping[str, Any]], target: int, *, namespace: str
) -> list[dict[str, Any]]:
    if target < 0:
        raise ValueError("target must be non-negative")
    representatives = one_per_cluster_then_hash(rows, namespace=namespace)
    if len(representatives) < target:
        raise ValueError(
            f"only {len(representatives)} cluster representatives for target {target}"
        )
    selected = representatives[:target]
    for index, row in enumerate(selected):
        row["prior_ablation_selection_index"] = index
        row["prior_ablation_selection_priority"] = stable_priority(
            f"{namespace}-representative-order", str(row["query_id"])
        )
    return selected


CONTRAST_TERMS: dict[str, dict[str, float]] = {
    "kc_minus_u0": {"kc": 1, "u0": -1},
    "kcg_minus_kc": {"kcg": 1, "kc": -1},
    "kcg_minus_u0": {"kcg": 1, "u0": -1},
    "full_minus_ch": {"full": 1, "ch": -1},
    "k_minus_u0": {"k": 1, "u0": -1},
    "complete_minus_u0": {"complete": 1, "u0": -1},
    "kc_minus_k_minus_complete_plus_u0": {
        "kc": 1,
        "k": -1,
        "complete": -1,
        "u0": 1,
    },
    "kcm_minus_kc": {"kcm": 1, "kc": -1},
    "c_kc_minus_c": {"c_kc": 1, "c": -1},
    "c_kcg_minus_c_kc": {"c_kcg": 1, "c_kc": -1},
    "h_kc_minus_h": {"h_kc": 1, "h": -1},
    "h_kcg_minus_h_kc": {"h_kcg": 1, "h_kc": -1},
    "ch_kc_minus_ch": {"ch_kc": 1, "ch": -1},
    "full_minus_ch_kc": {"full": 1, "ch_kc": -1},
    "ch_kcm_minus_ch_kc": {"ch_kcm": 1, "ch_kc": -1},
    "ch_kcmg_minus_full": {"ch_kcmg": 1, "full": -1},
    "mutual_by_gate_interaction_on_ch": {
        "ch_kcmg": 1,
        "full": -1,
        "ch_kcm": -1,
        "ch_kc": 1,
    },
    "direct_prior_by_consistency_interaction": {
        "c_kc": 1,
        "c": -1,
        "kc": -1,
        "u0": 1,
    },
    "direct_prior_by_h0_interaction": {
        "h_kc": 1,
        "h": -1,
        "kc": -1,
        "u0": 1,
    },
    "full_stack_by_consistency_interaction": {
        "full": 1,
        "h_kcg": -1,
        "ch": -1,
        "h": 1,
    },
    "full_stack_by_h0_interaction": {
        "full": 1,
        "c_kcg": -1,
        "ch": -1,
        "c": 1,
    },
    "ch_kcg_key_minus_full": {"ch_kcg_key": 1, "full": -1},
    "ch_kcg_complete_minus_full": {
        "ch_kcg_complete": 1,
        "full": -1,
    },
}


def contrast_vector(
    selections: Mapping[str, np.ndarray], name: str
) -> np.ndarray:
    terms = CONTRAST_TERMS.get(name)
    if terms is None:
        raise KeyError(name)
    missing = set(terms) - set(selections)
    if missing:
        raise ValueError(f"contrast {name} lacks cells: {sorted(missing)}")
    result: np.ndarray | None = None
    for cell, coefficient in terms.items():
        value = np.asarray(selections[cell], dtype=np.float64) * coefficient
        result = value if result is None else result + value
    assert result is not None
    return result


__all__ = [
    "CONTRAST_TERMS",
    "EXPECTED_CELLS",
    "FACTOR_ORDER",
    "SCHEMA",
    "config_factor_projection",
    "contrast_vector",
    "derive_config",
    "factor_map",
    "one_per_cluster_then_hash",
    "select_query_rows",
    "validate_protocol",
]
