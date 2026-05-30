from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


MAX_GRIPPER_WIDTH_PX = 150.0


@dataclass
class Grasp:
    x: float
    y: float
    w: float
    h: float
    theta: float

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.x, self.y, self.w, self.h, self.theta)

    def corners(self) -> np.ndarray:
        return rect_corners(self.x, self.y, self.w, self.h, self.theta)


def rect_corners(x: float, y: float, w: float, h: float, theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = w / 2.0, h / 2.0
    local = np.array(
        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]],
        dtype=np.float32,
    )
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def normalise_angle(theta: float) -> float:
    theta = theta % np.pi
    if theta >= np.pi / 2:
        theta -= np.pi
    return theta


def gaussian_2d(shape: Tuple[int, int], centre: Tuple[float, float], sigma: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = centre
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return g


def grasp_to_maps(
    grasps: List[Grasp],
    image_size: Tuple[int, int],
    sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_size
    q_map = np.zeros((h, w), dtype=np.float32)
    cos2t_map = np.zeros((h, w), dtype=np.float32)
    sin2t_map = np.zeros((h, w), dtype=np.float32)
    w_map = np.zeros((h, w), dtype=np.float32)

    for grasp in grasps:
        theta = normalise_angle(grasp.theta)
        c = np.cos(2 * theta)
        s = np.sin(2 * theta)
        width_norm = float(np.clip(grasp.w / MAX_GRIPPER_WIDTH_PX, 0.0, 1.0))

        g = gaussian_2d((h, w), (grasp.x, grasp.y), sigma)
        mask = g > q_map
        q_map = np.where(mask, g, q_map)
        cos2t_map = np.where(mask, c, cos2t_map)
        sin2t_map = np.where(mask, s, sin2t_map)
        w_map = np.where(mask, width_norm, w_map)

    return q_map, cos2t_map, sin2t_map, w_map


def decode_maps(
    q: torch.Tensor,
    cos2t: torch.Tensor,
    sin2t: torch.Tensor,
    width: torch.Tensor,
    topk: int = 1,
    nms_radius: int = 7,
    height_ratio: float = 0.5,
) -> List[List[Grasp]]:
    if q.ndim == 3:
        q = q.unsqueeze(1)
        cos2t = cos2t.unsqueeze(1)
        sin2t = sin2t.unsqueeze(1)
        width = width.unsqueeze(1)

    b, _, h, w = q.shape
    pooled = F.max_pool2d(q, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)
    peaks = (q == pooled).float() * q
    peaks_flat = peaks.view(b, -1)
    topk_vals, topk_idx = peaks_flat.topk(topk, dim=1)

    cos2t_flat = cos2t.view(b, -1)
    sin2t_flat = sin2t.view(b, -1)
    width_flat = width.view(b, -1)

    batch_grasps: List[List[Grasp]] = []
    for i in range(b):
        grasps_i: List[Grasp] = []
        for j in range(topk):
            idx = topk_idx[i, j].item()
            if topk_vals[i, j].item() < 1e-4:
                continue
            y = idx // w
            x = idx % w
            c = cos2t_flat[i, idx].item()
            s = sin2t_flat[i, idx].item()
            theta = 0.5 * np.arctan2(s, c)
            grip_w = float(width_flat[i, idx].item()) * MAX_GRIPPER_WIDTH_PX
            grip_h = grip_w * height_ratio
            grasps_i.append(Grasp(x=float(x), y=float(y), w=grip_w, h=grip_h, theta=theta))
        batch_grasps.append(grasps_i)

    return batch_grasps
