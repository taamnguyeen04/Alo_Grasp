from __future__ import annotations

from typing import Dict, List, Literal

import torch
import torch.nn as nn

from .backbones import CLIPTextEncoder, CLIPVisualBackbone, DINOv2Backbone
from .decoder import GraspDecoder


VisualBackbone = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
    "clip_ViT-B-32", "clip_ViT-B-16",
]
FusionMode = Literal["concat", "film_last", "film_multiscale"]


class GraspCLIPD(nn.Module):
    def __init__(
        self,
        visual_backbone: str = "dinov2_vitb14",
        text_encoder: str = "ViT-B-32",
        text_pretrained: str = "openai",
        projection_dim: int = 768,
        decoder_channels: List[int] | None = None,
        fusion_mode: str = "film_multiscale",
        use_cross_attention: bool = True,
        cross_attn_heads: int = 8,
        output_size: int = 224,
        freeze_visual: bool = True,
        freeze_text: bool = True,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        if visual_backbone.startswith("dinov2_"):
            self.visual = DINOv2Backbone(model_name=visual_backbone, freeze=freeze_visual)
        elif visual_backbone.startswith("clip_"):
            clip_name = visual_backbone[len("clip_"):]
            self.visual = CLIPVisualBackbone(
                model_name=clip_name,
                pretrained=text_pretrained,
                freeze=freeze_visual,
            )
        else:
            raise ValueError(
                f"Unknown visual_backbone: {visual_backbone}. "
                f"Use 'dinov2_*' or 'clip_*' (e.g., 'clip_ViT-B-32')."
            )

        self.text = CLIPTextEncoder(
            model_name=text_encoder, pretrained=text_pretrained, freeze=freeze_text,
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text.embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.decoder = GraspDecoder(
            encoder_channels    = self.visual.feature_channels,
            decoder_channels    = decoder_channels,
            text_dim            = projection_dim,
            output_size         = output_size,
            fusion_mode         = fusion_mode,
            use_cross_attention = use_cross_attention,
            cross_attn_heads    = cross_attn_heads,
        )

        self.fusion_mode = fusion_mode

    def forward(self, image: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        device = image.device

        features = self.visual(image)
        text_emb = self.text(prompts, device)
        text_emb = self.text_proj(text_emb)

        raw = self.decoder(features, text_emb)
        q_logit, cos2t_raw, sin2t_raw, w_logit = raw.split(1, dim=1)
        return {
            "quality":   torch.sigmoid(q_logit),
            "cos2theta": cos2t_raw.clamp(-1.0, 1.0),
            "sin2theta": sin2t_raw.clamp(-1.0, 1.0),
            "width":     torch.sigmoid(w_logit),
            "logits":    raw,
        }

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
