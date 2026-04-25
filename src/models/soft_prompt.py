"""Soft-prompt projector f(h_t) -> K x d_llm pseudo-token embeddings.

Implements Eq. 3 of the paper. We use a small two-layer MLP and a
LayerNorm at the end so the pseudo-tokens have a similar scale to the
frozen Qwen input-embedding distribution (this matters because we splice
them directly into Qwen's embedding stream).
"""

from __future__ import annotations

import torch
from torch import nn


class SoftPromptProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        llm_hidden_dim: int,
        num_soft_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.llm_hidden_dim = int(llm_hidden_dim)
        self.num_soft_tokens = int(num_soft_tokens)

        intermediate = max(self.in_dim, self.llm_hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, self.num_soft_tokens * self.llm_hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(self.llm_hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, in_dim] -> soft prompts [B, K, llm_hidden_dim]."""
        B = h.shape[0]
        out = self.net(h)
        out = out.view(B, self.num_soft_tokens, self.llm_hidden_dim)
        out = self.layer_norm(out)
        return out
