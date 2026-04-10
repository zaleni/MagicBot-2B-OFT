from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Future3DPerceiverFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Future3DPerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.memory_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, seq_len, self.heads, width // self.heads)
        return x.transpose(1, 2)

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(memory)
        queries = self.query_norm(queries)

        q = self.to_q(queries)
        k, v = self.to_kv(memory).chunk(2, dim=-1)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(attended)


class Future3DPerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_heads: int, output_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        dim_head = dim // num_heads
        self.attn = Future3DPerceiverAttention(dim=dim, dim_head=dim_head, heads=num_heads)
        self.ff = Future3DPerceiverFeedForward(dim=dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim)

    def forward(self, output_queries: torch.Tensor, messenger_tokens: torch.Tensor) -> torch.Tensor:
        latents = output_queries
        latents = latents + self.attn(messenger_tokens, latents)
        latents = latents + self.ff(latents)
        return self.output_proj(self.output_norm(latents))
