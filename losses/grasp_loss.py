from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraspLossOutput:
    total: torch.Tensor
    loss_q: torch.Tensor
    loss_angle: torch.Tensor
    loss_width: torch.Tensor


class GraspLoss(nn.Module):

    def __init__(
        self,
        weight_quality: float = 1.0,
        weight_angle: float = 1.0,
        weight_width: float = 0.5,
        angle_mask_threshold: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.w_q = weight_quality
        self.w_a = weight_angle
        self.w_w = weight_width
        self.mask_thresh = angle_mask_threshold

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> GraspLossOutput:
        q_pred = preds["quality"].squeeze(1)
        c_pred = preds["cos2theta"].squeeze(1)
        s_pred = preds["sin2theta"].squeeze(1)
        w_pred = preds["width"].squeeze(1)

        q_tgt = targets["quality"]
        c_tgt = targets["cos2theta"]
        s_tgt = targets["sin2theta"]
        w_tgt = targets["width"]

        loss_q = F.mse_loss(q_pred, q_tgt)

        mask = (q_tgt > self.mask_thresh).float()
        denom = mask.sum().clamp(min=1.0)
        loss_cos = ((c_pred - c_tgt) ** 2 * mask).sum() / denom
        loss_sin = ((s_pred - s_tgt) ** 2 * mask).sum() / denom
        loss_angle = 0.5 * (loss_cos + loss_sin)

        loss_w_pix = F.smooth_l1_loss(w_pred, w_tgt, reduction="none", beta=0.1)
        loss_width = (loss_w_pix * mask).sum() / denom

        total = self.w_q * loss_q + self.w_a * loss_angle + self.w_w * loss_width
        return GraspLossOutput(
            total=total, loss_q=loss_q, loss_angle=loss_angle, loss_width=loss_width
        )
