from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CrossAttentionBlock, FiLMBlock


FusionMode = Literal["concat", "film_last", "film_multiscale"]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        text_dim: int,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock(2 * out_ch, out_ch)
        self.use_film = use_film
        if use_film:
            self.film = FiLMBlock(text_dim=text_dim, feat_channels=out_ch, hidden_dim=out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up(x)
        s = self.skip_proj(skip)
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        if self.use_film:
            x = self.film(x, text_emb)
        return x


class GraspDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        text_dim: int,
        output_size: int = 224,
        fusion_mode: FusionMode = "film_multiscale",
        use_cross_attention: bool = True,
        cross_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        assert len(encoder_channels) == 4
        assert len(decoder_channels) == 4
        assert fusion_mode in ("concat", "film_last", "film_multiscale"), \
            f"Unknown fusion_mode: {fusion_mode}"

        self.output_size = output_size
        self.fusion_mode = fusion_mode

        self.enc_deep_to_shallow = list(reversed(encoder_channels))

        bn_ch = decoder_channels[0]

        if fusion_mode == "concat":
            self.text_concat_proj = nn.Linear(text_dim, bn_ch)
            self.bottleneck_proj = nn.Sequential(
                nn.Conv2d(self.enc_deep_to_shallow[0] + bn_ch, bn_ch, kernel_size=1),
                nn.GroupNorm(min(32, bn_ch), bn_ch),
                nn.GELU(),
            )
        else:
            self.bottleneck_proj = nn.Sequential(
                nn.Conv2d(self.enc_deep_to_shallow[0], bn_ch, kernel_size=1),
                nn.GroupNorm(min(32, bn_ch), bn_ch),
                nn.GELU(),
            )

        self.use_ca = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossAttentionBlock(
                dim=bn_ch, text_dim=text_dim, num_heads=cross_attn_heads,
            )

        if fusion_mode in ("film_last", "film_multiscale"):
            self.bottleneck_film = FiLMBlock(
                text_dim=text_dim, feat_channels=bn_ch, hidden_dim=bn_ch,
            )
        else:
            self.bottleneck_film = None

        use_film_in_upblocks = (fusion_mode == "film_multiscale")
        self.up_blocks = nn.ModuleList([
            UpBlock(
                in_ch=decoder_channels[i],
                skip_ch=self.enc_deep_to_shallow[i + 1],
                out_ch=decoder_channels[i + 1],
                text_dim=text_dim,
                use_film=use_film_in_upblocks,
            )
            for i in range(3)
        ])

        head_in = decoder_channels[-1]
        self.head = nn.Sequential(
            ConvBlock(head_in, head_in),
            nn.Conv2d(head_in, 4, kernel_size=1),
        )

    def forward(
        self, features: List[torch.Tensor], text_emb: torch.Tensor,
    ) -> torch.Tensor:
        deep_to_shallow = list(reversed(features))

        deepest = deep_to_shallow[0]
        if self.fusion_mode == "concat":
            b, _, h, w = deepest.shape
            t = self.text_concat_proj(text_emb)
            t = t.unsqueeze(-1).unsqueeze(-1).expand(b, -1, h, w)
            deepest = torch.cat([deepest, t], dim=1)
        x = self.bottleneck_proj(deepest)

        if self.use_ca:
            x = self.cross_attn(x, text_emb)
        if self.bottleneck_film is not None:
            x = self.bottleneck_film(x, text_emb)

        for i, blk in enumerate(self.up_blocks):
            x = blk(x, deep_to_shallow[i + 1], text_emb)

        x = F.interpolate(
            x, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False,
        )
        return self.head(x)
