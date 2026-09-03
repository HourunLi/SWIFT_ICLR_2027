"""Hash-auditable clean-room adapter for the official SWIFT linear reward head.

The implementation mirrors ``utils.LinearRewardModel`` from the official
SWIFT repository at commit ``41f7c9f7e13734267450870f977e5dd7d62ac23e``.
Only generated-token hidden states are consumed.  CLIR's feature encoder,
condition fusion, residual score head, and auxiliary heads are deliberately
absent so this module can serve as a genuinely plain SWIFT baseline.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.clir_smoke import read_jsonl, stable_priority


UPSTREAM_COMMIT = "41f7c9f7e13734267450870f977e5dd7d62ac23e"


class SwiftLinearRewardModel(nn.Module):
    """Official SWIFT token gate/reward head and weighted-mean aggregation."""

    def __init__(self, feature_dim: int, disable_gate: bool = False) -> None:
        super().__init__()
        self.disable_gate = bool(disable_gate)
        if not self.disable_gate:
            self.fused_layer = nn.Linear(int(feature_dim), 2)
        else:
            self.reward_layer = nn.Linear(int(feature_dim), 1)

    def forward(
        self,
        x: Tensor,
        lengths: Sequence[int] | Tensor,
        is_eval: bool = False,
        boundaries: Any = None,
        reward_mode: Any = None,
    ) -> Tensor:
        """Return one scalar per candidate, matching upstream SWIFT semantics."""

        del is_eval, boundaries, reward_mode
        batch_size, max_seq_len, _ = x.size()
        mask = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.float32, device=x.device
        )
        for index, length in enumerate(lengths):
            mask[index, : int(length)] = 1.0
        if not self.disable_gate:
            fused_output = self.fused_layer(x)
            gates = torch.sigmoid(fused_output[..., 0])
            rewards = fused_output[..., 1]
            weighted = torch.sum(gates * rewards * mask, dim=1)
            denominator = torch.sum(gates * mask, dim=1)
            return weighted / denominator.clamp(min=1e-8)
        rewards = self.reward_layer(x).squeeze(-1)
        return torch.sum(rewards * mask, dim=1) / torch.sum(mask, dim=1).clamp(
            min=1
        )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location="cpu")


def load_feature_tensor(row: Mapping[str, Any], manifest_parent: Path) -> Tensor:
    """Load one output-token feature tensor without loading CLIR conditions."""

    raw_path = row.get("hidden_states_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{row.get('id')}: hidden_states_path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_parent / path
    value = _torch_load(path.resolve())
    if isinstance(value, Mapping):
        for key in ("hidden_states", "features", "tensor", "states"):
            if key in value:
                value = value[key]
                break
    tensor = torch.as_tensor(value)
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{row.get('id')}: SWIFT feature must be [time, width]")
    expected_tokens = len(row.get("output_token_ids", ()))
    if expected_tokens and tensor.shape[0] != expected_tokens:
        raise ValueError(f"{row.get('id')}: output-token feature axis drift")
    expected_width = int(row.get("feature_dim", tensor.shape[1]))
    if tensor.shape[1] != expected_width:
        raise ValueError(f"{row.get('id')}: feature width drift")
    if not tensor.is_floating_point():
        raise ValueError(f"{row.get('id')}: hidden states are not floating point")
    return tensor


class SwiftFeatureDataset(Dataset):
    """Flattened candidate dataset backed by existing exact-token tensors."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest).resolve()
        self.parent = self.manifest.parent
        self.rows = read_jsonl(self.manifest)
        if not self.rows:
            raise ValueError("SWIFT feature manifest is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        label = row.get("correctness")
        if label not in (0, 1, False, True):
            raise ValueError(f"{row.get('id')}: correctness must be binary")
        feature = load_feature_tensor(row, self.parent)
        return {
            "row_index": index,
            "id": str(row["id"]),
            "query_id": str(row["query_id"]),
            "candidate_index": int(row["candidate_index"]),
            "hidden_states": feature,
            "length": int(feature.shape[0]),
            "correctness": float(label),
        }


def swift_collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad candidate features while preserving exact, unpadded lengths."""

    if not items:
        raise ValueError("cannot collate an empty SWIFT batch")
    widths = {int(item["hidden_states"].shape[1]) for item in items}
    if len(widths) != 1:
        raise ValueError("SWIFT batch contains mixed feature widths")
    tensors = [item["hidden_states"] for item in items]
    return {
        "hidden_states": pad_sequence(tensors, batch_first=True, padding_value=0),
        "lengths": [int(item["length"]) for item in items],
        "correctness": torch.tensor(
            [float(item["correctness"]) for item in items], dtype=torch.float32
        ),
        "row_indices": [int(item["row_index"]) for item in items],
        "ids": [str(item["id"]) for item in items],
        "query_ids": [str(item["query_id"]) for item in items],
        "candidate_indices": [int(item["candidate_index"]) for item in items],
    }


def query_disjoint_split(
    rows: Sequence[Mapping[str, Any]], *, validation_fraction: float, namespace: str
) -> dict[str, Any]:
    """Create a deterministic query-level split without inspecting labels."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["query_id"])].append(index)
    query_ids = sorted(
        grouped,
        key=lambda query_id: stable_priority(namespace, query_id),
    )
    train_count = int(len(query_ids) * (1.0 - validation_fraction))
    train_queries = set(query_ids[:train_count])
    validation_queries = set(query_ids[train_count:])
    if not train_queries or not validation_queries:
        raise ValueError("query-level split produced an empty side")
    train_indices = [
        index
        for query_id in query_ids
        if query_id in train_queries
        for index in grouped[query_id]
    ]
    validation_indices = [
        index
        for query_id in query_ids
        if query_id in validation_queries
        for index in grouped[query_id]
    ]
    if set(train_indices) & set(validation_indices):
        raise AssertionError("query-disjoint split has row overlap")
    return {
        "train_query_ids": [value for value in query_ids if value in train_queries],
        "validation_query_ids": [
            value for value in query_ids if value in validation_queries
        ],
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


def stacked_swift_scores(
    hidden_states: Tensor,
    lengths: Sequence[int] | Tensor,
    state_dicts: Sequence[Mapping[str, Tensor]],
) -> Tensor:
    """Score several gated SWIFT checkpoints in one feature pass.

    The returned tensor has shape ``[batch, checkpoint]``.  Stacking only
    combines identical linear operations; each checkpoint retains its own gate
    denominator exactly as in :class:`SwiftLinearRewardModel`.
    """

    if not state_dicts:
        raise ValueError("at least one SWIFT checkpoint is required")
    weights = torch.cat(
        [state["fused_layer.weight"].to(hidden_states.device) for state in state_dicts],
        dim=0,
    )
    biases = torch.cat(
        [state["fused_layer.bias"].to(hidden_states.device) for state in state_dicts],
        dim=0,
    )
    fused = torch.nn.functional.linear(hidden_states, weights, biases)
    batch, time, _ = fused.shape
    models = len(state_dicts)
    fused = fused.reshape(batch, time, models, 2)
    positions = torch.arange(time, device=hidden_states.device)[None, :]
    length_tensor = torch.as_tensor(lengths, device=hidden_states.device)
    mask = (positions < length_tensor[:, None]).to(torch.float32)
    gates = torch.sigmoid(fused[..., 0]) * mask[:, :, None]
    rewards = fused[..., 1]
    return (gates * rewards).sum(dim=1) / gates.sum(dim=1).clamp_min(1e-8)


__all__ = [
    "UPSTREAM_COMMIT",
    "SwiftFeatureDataset",
    "SwiftLinearRewardModel",
    "load_feature_tensor",
    "query_disjoint_split",
    "stacked_swift_scores",
    "swift_collate",
]
