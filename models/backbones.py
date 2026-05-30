from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in self.DIMS:
            raise ValueError(f"Unknown DINOv2 variant: {model_name}")
        self.model_name = model_name
        self.embed_dim = self.DIMS[model_name]
        self.patch_size = 14
        self.out_indices = tuple(out_indices)

        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @property
    def feature_channels(self) -> List[int]:
        return [self.embed_dim] * len(self.out_indices)

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen even in train()
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"Input H,W ({h},{w}) must be a multiple of patch size {self.patch_size}."
        )
        gh, gw = h // self.patch_size, w // self.patch_size

        # DINOv2 normalisation (ImageNet mean/std).
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        intermediates = self.model.get_intermediate_layers(
            x, n=self.out_indices, reshape=False, return_class_token=False, norm=True
        )

        features: List[torch.Tensor] = []
        for tokens in intermediates:
            feat = tokens.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
            features.append(feat)
        return features

class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "open_clip_torch is required. Install with `pip install open_clip_torch`."
            ) from e

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = self.model.text_projection.shape[1]  # type: ignore[union-attr]

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return emb.float()

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        if self.freeze:
            return self.encode(prompts, device)
        tokens = self.tokenizer(prompts).to(device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return emb.float()


class CLIPVisualBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required for CLIPVisualBackbone."
            ) from e

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.embed_dim = self.model.visual.transformer.width
        self.patch_size = self.model.visual.patch_size[0] \
            if hasattr(self.model.visual.patch_size, "__getitem__") \
            else self.model.visual.patch_size
        self.out_indices = tuple(out_indices)
        self.target_grid = 16  # match DINOv2 ViT-B/14 grid for 224 input

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @property
    def feature_channels(self) -> List[int]:
        return [self.embed_dim] * len(self.out_indices)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b = x.shape[0]

        # CLIP normalisation (different mean/std from ImageNet)
        clip_mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std  = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        x = (x - clip_mean) / clip_std

        vt = self.model.visual

        x = vt.conv1(x)                          # (B, C, H/p, W/p)
        h, w = x.shape[-2:]
        x = x.reshape(b, x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)

        cls = vt.class_embedding.to(x.dtype) + torch.zeros(
            b, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + vt.positional_embedding.to(x.dtype)
        x = vt.ln_pre(x)
        x = x.permute(1, 0, 2)

        # Capture outputs at desired layers
        intermediates = []
        for i, block in enumerate(vt.transformer.resblocks):
            x = block(x)
            if i in self.out_indices:
                feat = x.permute(1, 0, 2)[:, 1:, :]  # strip CLS
                feat = feat.transpose(1, 2).reshape(b, self.embed_dim, h, w)
                if (h, w) != (self.target_grid, self.target_grid):
                    feat = torch.nn.functional.interpolate(
                        feat, size=(self.target_grid, self.target_grid),
                        mode="bilinear", align_corners=False,
                    )
                intermediates.append(feat)

        return intermediates
