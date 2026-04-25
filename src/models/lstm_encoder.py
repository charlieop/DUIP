"""LSTM encoder over a session of item ids -> dynamic-intent hidden state.

Implements Eq. 1-2 of the paper.

The item embedding table is small (`d_item`-dim) but is initialized from the
*projected* mean Qwen-token embedding of each item's title. This gives every
item a content-aware warm start so even unseen-during-training items have a
sensible prior (helps with the cold-start case discussed in the paper).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.item_embed_dim = int(item_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.padding_idx = padding_idx

        # An extra padding row at the end so we can use index = num_items
        # for "pad" without colliding with a real item id.
        self.pad_idx = self.num_items if padding_idx is None else padding_idx
        vocab = self.num_items + (0 if padding_idx is not None else 1)

        self.item_embedding = nn.Embedding(
            num_embeddings=vocab,
            embedding_dim=self.item_embed_dim,
            padding_idx=self.pad_idx,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.item_embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

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
        # zero out the pad row
        self.item_embedding.weight.data[self.pad_idx].zero_()

    def forward(
        self, history_ids: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return h_t (Eq. 2) of shape [B, hidden_dim].

        ``history_ids``: [B, L] long, padded with ``pad_idx``.
        ``history_mask``: [B, L] bool, True where the position is a real item.
        """
        # Replace padding positions with the pad index, in case caller
        # used 0 instead.
        ids = history_ids.clone()
        ids[~history_mask] = self.pad_idx

        emb = self.item_embedding(ids)              # [B, L, D]
        emb = self.input_dropout(emb)

        lengths = history_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)             # h_n: [num_layers, B, H]
        h_t = h_n[-1]                               # [B, H]
        return h_t
