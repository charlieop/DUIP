







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

        B = h.shape[0]
        out = self.net(h)
        out = out.view(B, self.num_soft_tokens, self.llm_hidden_dim)
        out = self.layer_norm(out)
        return out
