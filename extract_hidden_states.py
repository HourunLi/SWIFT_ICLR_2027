"""Extract exact-token, all-layer hidden states for CLIR.

Input rows must already contain ``prompt_token_ids`` and ``output_token_ids``.
The script never decodes and re-tokenizes the response, which avoids the token
boundary drift that invalidated several early localization experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Sequence

import torch

from src.clir_data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract token-aligned CLIR features.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        default=None,
        help="Pinned model revision/commit used for reproducible extraction.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional pinned Hugging Face cache directory.",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Explicit attention backend used for feature parity across runs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--expected_num_feature_layers", type=int, default=None)
    parser.add_argument("--expected_per_layer_dim", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing manifest and feature files.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def validate_token_ids(values: Any, field: str, row_id: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{row_id}: {field} must be a sequence of integer token IDs")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in values
    ):
        raise ValueError(f"{row_id}: {field} must contain only integer token IDs")
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{row_id}: {field} must not be empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{row_id}: {field} contains a negative token ID")
    return result


def validate_token_labels(
    row: Mapping[str, Any], output_length: int, row_id: str
) -> None:
    for field in (
        "token_advantage",
        "token_advantages",
        "advantages",
        "progress_targets",
        "key_prior_target",
        "complete_prior_target",
        "key_prior",
        "complete_prior",
    ):
        if field in row and row[field] is not None:
            values = row[field]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{row_id}: {field} must be a token sequence")
            if len(values) != output_length:
                raise ValueError(
                    f"{row_id}: {field} has length {len(values)}, expected {output_length}"
                )
    if "hallucination_onset" in row and row["hallucination_onset"] is not None:
        raw_onset = row["hallucination_onset"]
        if not isinstance(raw_onset, Integral) or isinstance(raw_onset, bool):
            raise ValueError(f"{row_id}: hallucination_onset must be an integer")
        onset = int(raw_onset)
        if onset < -1 or onset >= output_length:
            raise ValueError(f"{row_id}: hallucination_onset is outside the output")


@torch.no_grad()
def extract_row(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int],
    output_token_ids: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    combined = list(prompt_token_ids) + list(output_token_ids)
    input_ids = torch.tensor([combined], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if result.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    layers = tuple(result.hidden_states)
    if not layers:
        raise RuntimeError("Model returned an empty hidden-state tuple")
    expected_sequence_length = len(combined)
    if (
        layers[0].ndim != 3
        or layers[0].shape[0] != 1
        or layers[0].shape[1] != expected_sequence_length
    ):
        raise ValueError(
            "Hidden states must have shape [1, prompt+output tokens, hidden_dim]"
        )
    per_layer_dim = int(layers[0].shape[-1])
    if any(layer.shape != layers[0].shape for layer in layers):
        raise ValueError("All returned hidden-state layers must have the same shape")
    all_layer_states = torch.cat([layer[0] for layer in layers], dim=-1)
    prompt_length = len(prompt_token_ids)
    trajectory = all_layer_states[prompt_length:].detach().cpu().contiguous()
    condition = all_layer_states[:prompt_length].detach().cpu().contiguous()
    if trajectory.shape[0] != len(output_token_ids):
        raise AssertionError(
            "Output feature length no longer matches exact output token IDs"
        )
    if not torch.isfinite(trajectory).all() or not torch.isfinite(condition).all():
        raise FloatingPointError("Model returned non-finite hidden-state features")
    return trajectory, condition, len(layers), per_layer_dim


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "extract_hidden_states.py requires the optional `transformers` dependency"
        ) from exc

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    load_kwargs: Dict[str, Any] = {
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **load_kwargs,
    ).to(device)
    model.eval()

    output_jsonl = Path(args.output_jsonl)
    input_jsonl = Path(args.input_jsonl)
    if output_jsonl.resolve() == input_jsonl.resolve():
        raise ValueError("output_jsonl must not overwrite input_jsonl")
    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output manifest already exists; pass --overwrite: {output_jsonl}"
        )
    feature_dir = Path(args.feature_dir)
    rows = read_jsonl(input_jsonl)
    output_rows: list[Dict[str, Any]] = []
    observed_contract: tuple[int, int] | None = None
    query_prompts: dict[str, tuple[int, ...]] = {}
    condition_cache: dict[tuple[int, ...], tuple[Path, str]] = {}
    resolved_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None) or args.revision
    )
    for index, source in enumerate(rows):
        row = dict(source)
        row_id = str(row.get("id", index))
        if "query_id" not in row or row["query_id"] is None:
            raise ValueError(f"{row_id}: query_id is required")
        query_id = str(row["query_id"])
        if "prompt_token_ids" not in row or "output_token_ids" not in row:
            raise ValueError(
                f"{row_id}: exact prompt_token_ids and output_token_ids are required"
            )
        prompt_ids = validate_token_ids(
            row["prompt_token_ids"], "prompt_token_ids", row_id
        )
        output_ids = validate_token_ids(
            row["output_token_ids"], "output_token_ids", row_id
        )
        prompt_key = tuple(prompt_ids)
        if query_id in query_prompts and query_prompts[query_id] != prompt_key:
            raise ValueError(
                f"{row_id}: query_id {query_id!r} has inconsistent prompt_token_ids"
            )
        query_prompts[query_id] = prompt_key
        validate_token_labels(row, len(output_ids), row_id)
        trajectory, condition, num_layers, per_layer_dim = extract_row(
            model, prompt_ids, output_ids, device
        )
        contract = (num_layers, per_layer_dim)
        if observed_contract is not None and observed_contract != contract:
            raise ValueError("The model returned an inconsistent feature contract")
        observed_contract = contract
        if args.expected_num_feature_layers not in (None, num_layers):
            raise ValueError(
                f"Expected {args.expected_num_feature_layers} layers, got {num_layers}"
            )
        if args.expected_per_layer_dim not in (None, per_layer_dim):
            raise ValueError(
                f"Expected per-layer width {args.expected_per_layer_dim}, got {per_layer_dim}"
            )

        hidden_path = feature_dir / f"{index:08d}.hidden.pt"
        if hidden_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Feature already exists; pass --overwrite: {hidden_path}"
            )
        atomic_torch_save(trajectory, hidden_path)
        hidden_sha256 = file_sha256(hidden_path)
        if prompt_key in condition_cache:
            condition_path, condition_sha256 = condition_cache[prompt_key]
        else:
            prompt_digest = hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            condition_path = feature_dir / f"condition-{prompt_digest}.pt"
            if condition_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Feature already exists; pass --overwrite: {condition_path}"
                )
            atomic_torch_save(condition, condition_path)
            condition_sha256 = file_sha256(condition_path)
            condition_cache[prompt_key] = (condition_path, condition_sha256)
        row["hidden_states_path"] = os.path.relpath(hidden_path, output_jsonl.parent)
        row["condition_states_path"] = os.path.relpath(
            condition_path, output_jsonl.parent
        )
        row["hidden_states_sha256"] = hidden_sha256
        row["condition_states_sha256"] = condition_sha256
        row["feature_dim"] = num_layers * per_layer_dim
        row["num_feature_layers"] = num_layers
        row["per_layer_dim"] = per_layer_dim
        row["feature_model"] = args.model
        row["feature_revision"] = resolved_revision
        row["feature_dtype"] = args.dtype
        row["feature_attention_implementation"] = args.attn_implementation
        output_rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_jsonl.name}.", dir=output_jsonl.parent
    )
    os.close(descriptor)
    try:
        write_jsonl(temporary, output_rows)
        os.replace(temporary, output_jsonl)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"wrote {len(output_rows)} rows to {output_jsonl}")


if __name__ == "__main__":
    main()
