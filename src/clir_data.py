"""Data utilities for CLIR JSONL training and scoring.

Each JSONL row represents one generated trajectory. Minimal fields:

{
  "id": "sample-0-cand-0",
  "query_id": "sample-0",
  "hidden_states_path": "features/sample-0-cand-0.pt",
  "correctness": 1,
  "semantic_id": "sample-0",
  "style_id": "direct"
}

Hidden states can be stored inline as `hidden_states`, or by path in
`hidden_states_path`. Optional `condition_states`, prior targets, hallucination
labels, and token advantage targets follow the names used by the model.
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path
import random
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset, Sampler


TEXT_ID_FIELDS = {
    "id",
    "query_id",
    "semantic_id",
    "style_id",
    "domain_id",
    "source",
    "prompt",
    "context",
    "trajectory",
    "answer",
}

TOKEN_SEQUENCE_FIELDS = {
    "token_advantage",
    "token_advantages",
    "advantages",
    "progress_targets",
    "key_prior_target",
    "complete_prior_target",
    "key_prior",
    "complete_prior",
}


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class CLIRTrajectoryDataset(Dataset):
    """JSONL dataset for pre-extracted hidden-state trajectories."""

    def __init__(
        self, jsonl_path: str | Path, feature_root: Optional[str | Path] = None
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.feature_root = (
            Path(feature_root) if feature_root is not None else self.jsonl_path.parent
        )
        self.rows = read_jsonl(self.jsonl_path)
        if not self.rows:
            raise ValueError(f"No rows found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.rows[index])
        hidden_states = load_tensor_field(
            row, "hidden_states", "hidden_states_path", self.feature_root
        )
        if hidden_states.ndim != 2:
            raise ValueError(
                "Each hidden state item must have shape [time, hidden_dim]"
            )
        if hidden_states.shape[0] == 0:
            raise ValueError("Each trajectory must contain at least one token")
        if (
            "output_token_ids" in row
            and len(row["output_token_ids"]) != hidden_states.shape[0]
        ):
            raise ValueError(
                "output_token_ids length must exactly match trajectory hidden states"
            )
        feature_contract = resolve_feature_metadata(row)
        feature_dim = feature_contract["feature_dim"]
        layer_count = feature_contract["num_feature_layers"]
        per_layer_dim = feature_contract["per_layer_dim"]
        if feature_dim is not None and feature_dim != hidden_states.shape[1]:
            raise ValueError("feature_dim metadata does not match hidden states")
        if layer_count is not None and per_layer_dim is not None:
            expected_width = layer_count * per_layer_dim
            if expected_width != hidden_states.shape[1]:
                raise ValueError("layer feature metadata does not match hidden states")

        if not hidden_states.is_floating_point():
            hidden_states = hidden_states.float()
        item: Dict[str, Any] = {
            "row_index": index,
            "id": row.get("id", str(index)),
            "query_id": row.get(
                "query_id",
                row.get("candidate_group_id", row.get("prompt_id", str(index))),
            ),
            "hidden_states": hidden_states,
        }

        condition_states = maybe_load_tensor_field(
            row, "condition_states", "condition_states_path", self.feature_root
        )
        if condition_states is not None:
            if condition_states.ndim != 2:
                raise ValueError(
                    "condition_states must have shape [condition_time, hidden_dim]"
                )
            if condition_states.shape[0] == 0:
                raise ValueError("condition_states must contain at least one token")
            if (
                "prompt_token_ids" in row
                and len(row["prompt_token_ids"]) != condition_states.shape[0]
            ):
                raise ValueError(
                    "prompt_token_ids length must exactly match condition hidden states"
                )
            if condition_states.shape[-1] != hidden_states.shape[-1]:
                raise ValueError("condition_states width must match hidden_states")
            item["condition_states"] = (
                condition_states
                if condition_states.is_floating_point()
                else condition_states.float()
            )

        condition_embedding = maybe_load_tensor_field(
            row,
            "condition_embedding",
            "condition_embedding_path",
            self.feature_root,
        )
        if condition_embedding is not None:
            if condition_embedding.ndim != 1:
                raise ValueError("condition_embedding must have shape [hidden_dim]")
            if condition_embedding.shape[0] != hidden_states.shape[-1]:
                raise ValueError("condition_embedding width must match hidden_states")
            item["condition_embedding"] = (
                condition_embedding
                if condition_embedding.is_floating_point()
                else condition_embedding.float()
            )

        item.update(extract_metadata(row, hidden_states.shape[0]))
        return item


def load_tensor_field(
    row: Dict[str, Any],
    inline_key: str,
    path_key: str,
    feature_root: Path,
) -> Tensor:
    tensor = maybe_load_tensor_field(row, inline_key, path_key, feature_root)
    if tensor is None:
        raise KeyError(f"Row must contain `{inline_key}` or `{path_key}`")
    return tensor


def maybe_load_tensor_field(
    row: Dict[str, Any],
    inline_key: str,
    path_key: str,
    feature_root: Path,
) -> Optional[Tensor]:
    if inline_key in row and row[inline_key] is not None:
        return torch.as_tensor(row[inline_key])
    if path_key not in row or row[path_key] is None:
        return None

    path = Path(row[path_key])
    if not path.is_absolute():
        path = feature_root / path
    if path.suffix == ".pt" or path.suffix == ".pth":
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("hidden_states", "features", "tensor", "states"):
                if key in value:
                    value = value[key]
                    break
        return torch.as_tensor(value)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix == ".json":
        return torch.as_tensor(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported tensor file suffix: {path}")


def extract_metadata(row: Dict[str, Any], time: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {}

    scalar_fields = {
        "correctness": ("correctness", "label", "final_correct"),
        "semantic_id": (
            "semantic_id",
            "semantic_ids",
            "augmentation_group",
            "augmentation_group_id",
            "group_id",
        ),
        "style_id": (
            "style_id",
            "style_ids",
            "augmentation_style",
            "rewrite_style",
            "domain_id",
            "domain",
            "style",
        ),
        "hallucination_onset": ("hallucination_onset", "hallucination_start", "onset"),
        "path_hallucinated": ("path_hallucinated", "hallucinated", "hallucination"),
    }
    for output_key, aliases in scalar_fields.items():
        value = first_present(row, aliases)
        if value is not None:
            if output_key in {"correctness", "path_hallucinated"}:
                if not isinstance(value, Real):
                    raise ValueError(f"{output_key} must be numeric binary 0/1")
                numeric = float(value)
                if numeric not in {0.0, 1.0}:
                    raise ValueError(f"{output_key} must be binary, got {value!r}")
                value = numeric
            if output_key == "hallucination_onset":
                if not isinstance(value, Integral) or isinstance(value, bool):
                    raise ValueError("hallucination_onset must be an integer")
                value = int(value)
                if value < -1 or value >= time:
                    raise ValueError(
                        f"hallucination_onset must be -1 or in [0, {time}), got {value}"
                    )
            item[output_key] = value

    sequence_aliases = {
        "token_advantage": ("token_advantage", "token_advantages", "advantages"),
        "progress_targets": ("progress_targets", "progress", "progress_target"),
        "key_prior_target": ("key_prior_target", "key_prior"),
        "complete_prior_target": ("complete_prior_target", "complete_prior"),
    }
    for output_key, aliases in sequence_aliases.items():
        value = first_present(row, aliases)
        if value is not None:
            tensor = exact_length_1d(value, time, output_key)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Token label `{output_key}` contains NaN or Inf")
            if output_key in {"key_prior_target", "complete_prior_target"}:
                if not ((tensor == 0.0) | (tensor == 1.0)).all():
                    raise ValueError(f"Token label `{output_key}` must be binary")
            item[output_key] = tensor

    reconstruction_value = first_present(
        row, ("complete_reconstruction_target", "csr_target")
    )
    if reconstruction_value is not None:
        reconstruction = torch.as_tensor(
            reconstruction_value, dtype=torch.float32
        ).flatten()
        if not torch.isfinite(reconstruction).all():
            raise ValueError("complete_reconstruction_target contains NaN or Inf")
        item["complete_reconstruction_target"] = reconstruction

    return item


def first_present(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in row and row[key] is not None:
            return row[key]
    return None


def resolve_feature_metadata(row: Mapping[str, Any]) -> Dict[str, Optional[int]]:
    """Resolve the clean and historical all-layer feature metadata schemas.

    New manifests may put the three dimensions at the row top level. Existing
    panzhixin manifests put them in ``feature_metadata``. When both are present
    they are treated as an integrity assertion and must agree.
    """

    nested = row.get("feature_metadata")
    if nested is None:
        nested = {}
    if not isinstance(nested, Mapping):
        raise ValueError("feature_metadata must be an object when present")

    aliases = {
        "feature_dim": ("feature_dim", "hidden_dim"),
        "num_feature_layers": ("num_feature_layers", "layer_count"),
        "per_layer_dim": ("per_layer_dim", "per_layer_hidden_size"),
    }
    resolved: Dict[str, Optional[int]] = {}
    for output_key, field_aliases in aliases.items():
        top_value = first_present(row, field_aliases)
        nested_value = first_present(nested, field_aliases)
        for value in (top_value, nested_value):
            if value is not None and (
                not isinstance(value, Integral) or isinstance(value, bool)
            ):
                raise ValueError(f"{output_key} must be an integer")
        if (
            top_value is not None
            and nested_value is not None
            and top_value != nested_value
        ):
            raise ValueError(
                f"Conflicting top-level and feature_metadata values for {output_key}"
            )
        value = top_value if top_value is not None else nested_value
        if value is not None and value <= 0:
            raise ValueError(f"{output_key} must be positive")
        resolved[output_key] = int(value) if value is not None else None
    return resolved


def exact_length_1d(values: Any, length: int, field: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    if tensor.numel() != length:
        raise ValueError(
            f"Token label `{field}` length mismatch: expected {length}, got {tensor.numel()}"
        )
    return tensor


def clir_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_time = max(item["hidden_states"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden_states"].shape[1]
    hidden_dtype = batch[0]["hidden_states"].dtype
    hidden_states = torch.zeros(len(batch), max_time, hidden_dim, dtype=hidden_dtype)
    mask = torch.zeros(len(batch), max_time, dtype=torch.float32)

    for row, item in enumerate(batch):
        states = item["hidden_states"]
        if states.shape[1] != hidden_dim:
            raise ValueError("All hidden_states in a batch must have the same width")
        if states.dtype != hidden_dtype:
            raise ValueError("All hidden_states in a batch must have the same dtype")
        length = states.shape[0]
        hidden_states[row, :length] = states
        mask[row, :length] = 1.0

    output: Dict[str, Any] = {
        "row_index": torch.tensor(
            [item["row_index"] for item in batch], dtype=torch.long
        ),
        "ids": [item["id"] for item in batch],
        "query_ids_raw": [item["query_id"] for item in batch],
        "hidden_states": hidden_states,
        "mask": mask,
    }

    if any("condition_states" in item for item in batch):
        max_condition_time = max(
            item.get("condition_states", torch.empty(0, hidden_dim)).shape[0]
            for item in batch
        )
        condition_states = torch.zeros(
            len(batch), max_condition_time, hidden_dim, dtype=hidden_dtype
        )
        condition_mask = torch.zeros(
            len(batch), max_condition_time, dtype=torch.float32
        )
        for row, item in enumerate(batch):
            states = item.get("condition_states")
            if states is None:
                continue
            length = states.shape[0]
            condition_states[row, :length] = states
            condition_mask[row, :length] = 1.0
        output["condition_states"] = condition_states
        output["condition_mask"] = condition_mask

    if any("condition_embedding" in item for item in batch):
        condition_embedding = torch.zeros(len(batch), hidden_dim, dtype=hidden_dtype)
        condition_embedding_mask = torch.zeros(len(batch), dtype=torch.float32)
        for row, item in enumerate(batch):
            if "condition_embedding" in item:
                condition_embedding[row] = item["condition_embedding"]
                condition_embedding_mask[row] = 1.0
        output["condition_embedding"] = condition_embedding
        output["condition_embedding_mask"] = condition_embedding_mask

    add_optional_float(output, batch, "correctness", mask_key="correctness_mask")
    add_encoded_ids(
        output, batch, "semantic_id", "semantic_ids", "consistency_mask_semantic"
    )
    add_encoded_ids(output, batch, "style_id", "style_ids", "consistency_mask_style")
    if "semantic_ids" in output and "style_ids" in output:
        output["consistency_mask"] = output.pop(
            "consistency_mask_semantic"
        ) & output.pop("consistency_mask_style")

    add_optional_onset(output, batch)
    add_optional_float(output, batch, "path_hallucinated", mask_key="path_label_mask")
    add_optional_sequence(
        output, batch, "token_advantage", max_time, mask_key="token_advantage_mask"
    )
    add_optional_sequence(
        output, batch, "progress_targets", max_time, mask_key="progress_mask"
    )
    add_optional_sequence(
        output, batch, "key_prior_target", max_time, mask_key="key_prior_mask"
    )
    add_optional_sequence(
        output, batch, "complete_prior_target", max_time, mask_key="complete_prior_mask"
    )
    add_optional_vector(output, batch, "complete_reconstruction_target")

    output["query_ids"] = torch.tensor(
        encode_raw_ids(output["query_ids_raw"])[0], dtype=torch.long
    )
    return output


def add_optional_float(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
    mask_key: Optional[str] = None,
) -> None:
    values: List[float] = []
    mask: List[bool] = []
    for item in batch:
        if key in item and item[key] is not None:
            values.append(float(item[key]))
            mask.append(True)
        else:
            values.append(0.0)
            mask.append(False)
    if any(mask):
        output[key] = torch.tensor(values, dtype=torch.float32)
        if mask_key is not None:
            output[mask_key] = torch.tensor(mask, dtype=torch.bool)


def add_optional_onset(output: Dict[str, Any], batch: Sequence[Dict[str, Any]]) -> None:
    values: List[int] = []
    mask: List[bool] = []
    for item in batch:
        if "hallucination_onset" in item and item["hallucination_onset"] is not None:
            values.append(int(item["hallucination_onset"]))
            mask.append(True)
        else:
            values.append(-1)
            mask.append(False)
    if any(mask):
        output["hallucination_onset"] = torch.tensor(values, dtype=torch.long)
        output["onset_label_mask"] = torch.tensor(mask, dtype=torch.bool)


def add_encoded_ids(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    input_key: str,
    output_key: str,
    mask_key: str,
) -> None:
    values = [item.get(input_key) for item in batch]
    encoded, mask = encode_raw_ids(values)
    if any(mask):
        output[output_key] = torch.tensor(encoded, dtype=torch.long)
        output[mask_key] = torch.tensor(mask, dtype=torch.bool)


def encode_raw_ids(values: Sequence[Any]) -> Tuple[List[int], List[bool]]:
    mapping: Dict[str, int] = {}
    encoded: List[int] = []
    mask: List[bool] = []
    for value in values:
        if value is None:
            encoded.append(0)
            mask.append(False)
            continue
        key = repr(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        encoded.append(mapping[key])
        mask.append(True)
    return encoded, mask


def add_optional_sequence(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
    max_time: int,
    mask_key: str,
) -> None:
    values = torch.zeros(len(batch), max_time, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_time, dtype=torch.bool)
    has_any = False
    for row, item in enumerate(batch):
        if key not in item or item[key] is None:
            continue
        tensor = item[key].float().flatten()
        length = int(output["mask"][row].sum().item())
        if tensor.numel() != length:
            raise ValueError(f"{key} must have length {length}, got {tensor.numel()}")
        if length > 0:
            values[row, :length] = tensor
            mask[row, :length] = True
            has_any = True
    if has_any:
        output[key] = values
        output[mask_key] = mask


def add_optional_vector(
    output: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
    key: str,
) -> None:
    widths = {int(item[key].numel()) for item in batch if key in item}
    if not widths:
        return
    if len(widths) != 1:
        raise ValueError(f"All {key} vectors in a batch must have the same width")
    width = widths.pop()
    values = torch.zeros(len(batch), width, dtype=torch.float32)
    mask = []
    for row, item in enumerate(batch):
        if key not in item:
            mask.append(False)
            continue
        tensor = item[key].float().flatten()
        if tensor.numel() != width:
            raise ValueError(f"{key} must have length {width}, got {tensor.numel()}")
        values[row] = tensor
        mask.append(True)
    if any(mask):
        output[key] = values
        output[f"{key}_mask"] = torch.tensor(mask, dtype=torch.bool)


class SemanticGroupBatchSampler(BatchSampler):
    """Batch sampler that keeps LLM rewrites from the same semantic id together.

    Each batch packs small chunks from multiple semantic groups when possible.
    This makes PRISM-style positive and negative pairs likely in real training,
    instead of relying on random shuffle to place augmentations together.
    """

    def __init__(
        self,
        dataset: CLIRTrajectoryDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("SemanticGroupBatchSampler requires batch_size >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.indices = (
            list(indices) if indices is not None else list(range(len(dataset.rows)))
        )
        self.groups: Dict[str, List[int]] = {}
        for row_index in self.indices:
            row = dataset.rows[row_index]
            semantic_id = first_present(
                row,
                (
                    "semantic_id",
                    "semantic_ids",
                    "augmentation_group",
                    "augmentation_group_id",
                    "group_id",
                ),
            )
            key = repr(semantic_id) if semantic_id is not None else f"__row_{row_index}"
            self.groups.setdefault(key, []).append(row_index)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic order for an epoch, including after resume."""

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        yield from self._build_batches(rng if self.shuffle else None)

    def __len__(self) -> int:
        return len(self._build_batches(None))

    def _build_batches(self, rng: Optional[random.Random]) -> List[List[int]]:
        group_items = [list(indices) for indices in self.groups.values()]
        if rng is not None:
            rng.shuffle(group_items)
            for indices in group_items:
                rng.shuffle(indices)

        chunks: List[List[int]] = []
        leftovers: List[int] = []
        max_group_chunk = max(2, self.batch_size // 2)
        for indices in group_items:
            start = 0
            while start < len(indices):
                remaining = len(indices) - start
                if remaining == 1:
                    leftovers.append(indices[start])
                    break
                chunk = indices[start : start + min(max_group_chunk, remaining)]
                start += len(chunk)
                if len(chunk) >= 2:
                    chunks.append(chunk)
                else:
                    leftovers.extend(chunk)

        chunks.sort(key=len, reverse=True)
        if rng is not None:
            rng.shuffle(leftovers)

        batches: List[List[int]] = []
        current: List[int] = []
        for chunk in chunks:
            if len(current) + len(chunk) > self.batch_size:
                while len(current) < self.batch_size and leftovers:
                    current.append(leftovers.pop())
                if len(current) == self.batch_size or (current and not self.drop_last):
                    batches.append(current)
                current = []
            if len(chunk) == self.batch_size:
                batches.append(chunk)
            else:
                current.extend(chunk)

        while len(current) < self.batch_size and leftovers:
            current.append(leftovers.pop())
        if len(current) == self.batch_size or (current and not self.drop_last):
            batches.append(current)

        while leftovers:
            batch = leftovers[: self.batch_size]
            leftovers = leftovers[self.batch_size :]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                batches.append(batch)

        if rng is not None:
            rng.shuffle(batches)
        return batches


class EpochRandomSampler(Sampler[int]):
    """Epoch-indexed random order that is reproducible across checkpoint resume."""

    def __init__(self, indices: Sequence[int], seed: int = 0) -> None:
        self.indices = list(indices)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        return iter([self.indices[position] for position in order])

    def __len__(self) -> int:
        return len(self.indices)


def move_batch_to_device(
    batch: Dict[str, Any], device: torch.device | str
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


__all__ = [
    "CLIRTrajectoryDataset",
    "EpochRandomSampler",
    "SemanticGroupBatchSampler",
    "clir_collate",
    "move_batch_to_device",
    "read_jsonl",
    "resolve_feature_metadata",
    "write_jsonl",
]
