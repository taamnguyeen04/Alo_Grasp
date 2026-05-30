from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    def __init__(
        self,
        text_dim: int,
        feat_channels: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            self.gamma_proj = nn.Linear(text_dim, feat_channels)
            self.beta_proj = nn.Linear(text_dim, feat_channels)
        else:
            self.gamma_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feat_channels),
            )
            self.beta_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, feat_channels),
            )
        nn.init.zeros_(
            self.gamma_proj.weight
            if isinstance(self.gamma_proj, nn.Linear)
            else self.gamma_proj[-1].weight
        )
        nn.init.ones_(
            self.gamma_proj.bias
            if isinstance(self.gamma_proj, nn.Linear)
            else self.gamma_proj[-1].bias
        )
        nn.init.zeros_(
            self.beta_proj.weight
            if isinstance(self.beta_proj, nn.Linear)
            else self.beta_proj[-1].weight
        )
        nn.init.zeros_(
            self.beta_proj.bias
            if isinstance(self.beta_proj, nn.Linear)
            else self.beta_proj[-1].bias
        )

    def forward(self, feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(text_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(text_emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, text_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_text_proj = nn.Linear(text_dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        residual = x

        q = self.q_proj(self.norm_q(x))
        kv = self.kv_text_proj(text_emb).unsqueeze(1)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, -1, self.dim)
        out = self.out_proj(out)
        x = residual + out

        x = x + self.ffn(self.norm_ffn(x))

        return x.transpose(1, 2).reshape(b, c, h, w)
