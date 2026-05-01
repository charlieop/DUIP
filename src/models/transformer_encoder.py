"""Transformer encoder over a session of item ids -> dynamic-intent state.

This module intentionally mirrors ``LSTMEncoder``'s public API so DUIP can
switch encoders with a config-only change. It owns the item embedding table,
supports the same warm-start tensor, and returns a single ``[B, hidden_dim]``
state for the soft-prompt projector.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.item_embed_dim = int(item_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim) if ff_dim is not None else self.hidden_dim * 4
        self.max_seq_len = int(max_seq_len)
        self.padding_idx = padding_idx

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "transformer hidden_dim must be divisible by num_heads "
                f"(got hidden_dim={self.hidden_dim}, num_heads={self.num_heads})"
            )
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        # Match LSTMEncoder: reserve an optional extra row for padding.
        self.pad_idx = self.num_items if padding_idx is None else padding_idx
        vocab = self.num_items + (0 if padding_idx is not None else 1)

        self.item_embedding = nn.Embedding(
            num_embeddings=vocab,
            embedding_dim=self.item_embed_dim,
            padding_idx=self.pad_idx,
        )
        self.input_projection = (
            nn.Identity()
            if self.item_embed_dim == self.hidden_dim
            else nn.Linear(self.item_embed_dim, self.hidden_dim)
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    @torch.no_grad()
    def warm_start_from(self, embeddings: torch.Tensor) -> None:
        """Initialize the item table from a [num_items, item_embed_dim] tensor."""
        if embeddings.shape != (self.num_items, self.item_embed_dim):
            raise ValueError(
                f"warm-start tensor must have shape "
                f"({self.num_items}, {self.item_embed_dim}), "
                f"got {tuple(embeddings.shape)}"
            )
        self.item_embedding.weight.data[: self.num_items].copy_(embeddings.to(
            self.item_embedding.weight.dtype
        ))
        self.item_embedding.weight.data[self.pad_idx].zero_()

    def forward(
        self, history_ids: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a dynamic-intent state of shape [B, hidden_dim].

        ``history_ids``: [B, L] long, padded with ``pad_idx``.
        ``history_mask``: [B, L] bool, True where the position is a real item.
        """
        B, L = history_ids.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"history length {L} exceeds transformer max_seq_len={self.max_seq_len}"
            )

        ids = history_ids.clone()
        ids[~history_mask] = self.pad_idx

        positions = torch.arange(L, device=history_ids.device).unsqueeze(0)
        x = self.item_embedding(ids)
        x = self.input_projection(x)
        x = x + self.position_embedding(positions)
        x = self.input_dropout(x)

        key_padding_mask = ~history_mask
        empty_rows = key_padding_mask.all(dim=1)
        if empty_rows.any():
            # PyTorch attention cannot receive rows where every key is masked.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_rows, 0] = False

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)

        lengths = history_mask.sum(dim=1).clamp(min=1)
        last_idx = (lengths - 1).to(history_ids.device)
        batch_idx = torch.arange(B, device=history_ids.device)
        return encoded[batch_idx, last_idx]
