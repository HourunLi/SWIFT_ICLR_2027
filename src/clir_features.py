"""Compact encoders for frozen, all-layer LLM token features.

The production Phi features used by CLIR concatenate 33 layers of width 3072
for every generated token.  Applying the reward heads directly at that width
is needlessly large, so the default real-data configuration first models the
layer axis and pools it to a compact token representation.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IdentityFeatureEncoder(nn.Module):
    """Pass through features that are already compact."""

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return hidden_states, None


class LayerAxisFeatureEncoder(nn.Module):
    """Model and pool the layer axis before applying CLIR reward heads.

    Raw features are reshaped from ``[B, T, L*D]`` to ``[B*T, L, D]``.  A
    shared projection, a small Transformer, and learned pooling queries produce
    one compact vector per generated token.  Returned diagnostics have shape
    ``[B, T, pool_queries, layers]``.
    """

    max_normalization_elements = 2**31 - 1

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.input_dim = int(config.hidden_dim)
        self.num_feature_layers = int(config.num_feature_layers)
        self.per_layer_dim = int(config.per_layer_dim)
        self.layer_encoder_dim = int(config.layer_encoder_dim)
        self.layer_pool_queries = int(config.layer_pool_queries)

        self.input_norm = nn.LayerNorm(self.per_layer_dim)
        self.input_projection = nn.Linear(self.per_layer_dim, self.layer_encoder_dim)
        self.layer_positions = nn.Parameter(
            torch.empty(1, self.num_feature_layers, self.layer_encoder_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.layer_encoder_dim,
            nhead=int(config.layer_encoder_heads),
            dim_feedforward=self.layer_encoder_dim * 4,
            dropout=float(config.encoder_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.layer_encoder_blocks),
            enable_nested_tensor=False,
        )
        self.pool_queries = nn.Parameter(
            torch.empty(1, self.layer_pool_queries, self.layer_encoder_dim)
        )
        self.layer_pool = nn.MultiheadAttention(
            embed_dim=self.layer_encoder_dim,
            num_heads=int(config.layer_encoder_heads),
            dropout=float(config.encoder_dropout),
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            self.layer_pool_queries * self.layer_encoder_dim,
            int(config.model_dim),
        )
        self.output_norm = nn.LayerNorm(int(config.model_dim))
        nn.init.normal_(self.layer_positions, mean=0.0, std=0.02)
        nn.init.normal_(self.pool_queries, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"layer_transformer expects [batch, time, {self.input_dim}], "
                f"got {tuple(hidden_states.shape)}"
            )
        batch, time, _ = hidden_states.shape
        layer_states = hidden_states.reshape(
            batch * time, self.num_feature_layers, self.per_layer_dim
        )
        elements_per_row = self.num_feature_layers * self.per_layer_dim
        rows_per_chunk = max(1, self.max_normalization_elements // elements_per_row)
        projected = torch.cat(
            [
                self.input_projection(self.input_norm(chunk))
                for chunk in layer_states.split(rows_per_chunk, dim=0)
            ],
            dim=0,
        )
        layer_states = self.layer_transformer(projected + self.layer_positions)

        queries = self.pool_queries.expand(batch * time, -1, -1)
        pooled, attention = self.layer_pool(
            queries,
            layer_states,
            layer_states,
            need_weights=True,
            average_attn_weights=True,
        )
        encoded = self.output_norm(F.gelu(self.output_projection(pooled.flatten(1))))
        return (
            encoded.reshape(batch, time, -1),
            attention.reshape(
                batch,
                time,
                self.layer_pool_queries,
                self.num_feature_layers,
            ),
        )


def build_feature_encoder(config: Any) -> nn.Module:
    """Build the encoder selected by ``RewardConfig.encoder_type``."""

    if config.encoder_type == "identity":
        return IdentityFeatureEncoder()
    if config.encoder_type == "layer_transformer":
        return LayerAxisFeatureEncoder(config)
    raise ValueError(f"Unknown encoder_type: {config.encoder_type}")


__all__ = [
    "IdentityFeatureEncoder",
    "LayerAxisFeatureEncoder",
    "build_feature_encoder",
]
