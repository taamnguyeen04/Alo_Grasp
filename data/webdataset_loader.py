from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import webdataset as wds

from utils.grasp import Grasp, grasp_to_maps

cv2.setNumThreads(0)

def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def decode_prompt(cls_bytes: bytes) -> str:
    return cls_bytes.decode("utf-8")


def decode_tensor(pth_bytes: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(pth_bytes), map_location="cpu", weights_only=False)


def tensor_to_grasps(tensor: torch.Tensor, min_score: float = 0.0) -> List[Grasp]:
    grasps = []
    for row in tensor.tolist():
        score, x, y, w, h, theta_deg = row
        if score < min_score:
            continue
        theta_rad = np.deg2rad(theta_deg) % np.pi
        if theta_rad >= np.pi / 2:
            theta_rad -= np.pi
        grasps.append(Grasp(x=x, y=y, w=w, h=h, theta=theta_rad))
    return grasps


class TransformPipeline:
    def __init__(self, image_size: int = 224, sigma: float = 5.0, augment: bool = False, min_score: float = 0.0):
        self.image_size = image_size
        self.sigma = sigma
        self.augment = augment
        self.min_score = min_score

    def __call__(self, sample: Dict) -> Optional[Dict]:
        try:
            image_np = decode_jpg(sample["jpg"])
            prompt   = decode_prompt(sample["cls"])
            tensor   = decode_tensor(sample["pth"])
        except Exception:
            return None

        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            h0, w0 = image_np.shape[:2]
            scale_x, scale_y = self.image_size / w0, self.image_size / h0
            image_np = cv2.resize(image_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        else:
            scale_x = scale_y = 1.0

        grasps = tensor_to_grasps(tensor, self.min_score)
        if not grasps:
            return None

        if scale_x != 1.0 or scale_y != 1.0:
            grasps = [
                Grasp(g.x * scale_x, g.y * scale_y,
                      g.w * scale_x, g.h * scale_y, g.theta)
                for g in grasps
            ]

        if self.augment:
            image_np, grasps = _augment(image_np, grasps, self.image_size)

        q, c2t, s2t, w = grasp_to_maps(grasps, (self.image_size, self.image_size), self.sigma)

        return {
            "image"     : torch.from_numpy(image_np.copy()).permute(2,0,1).float() / 255.0,
            "prompt"    : prompt,
            "quality"   : torch.from_numpy(q).float(),
            "cos2theta" : torch.from_numpy(c2t).float(),
            "sin2theta" : torch.from_numpy(s2t).float(),
            "width"     : torch.from_numpy(w).float(),
            "grasps"    : grasps,
        }


def _augment(image: np.ndarray, grasps: List[Grasp], s: int) -> tuple:
    import random
    angle_deg = random.uniform(-15.0, 15.0)
    
    if abs(angle_deg) > 0.5:
        center = (s / 2.0, s / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        image = cv2.warpAffine(image, M, (s, s), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        
        theta_rad = -np.deg2rad(angle_deg)
        cos_r, sin_r = np.cos(theta_rad), np.sin(theta_rad)
        cx, cy = center
        new_grasps = []
        for g in grasps:
            dx, dy = g.x - cx, g.y - cy
            new_grasps.append(Grasp(
                cos_r * dx - sin_r * dy + cx,
                sin_r * dx + cos_r * dy + cy,
                g.w, g.h, g.theta + theta_rad,
            ))
        grasps = new_grasps
        
    if random.random() < 0.5:
        img = image.astype(np.float32)
        img *= random.uniform(0.8, 1.2)
        mu  = img.mean()
        img = (img - mu) * random.uniform(0.8, 1.2) + mu
        image = np.clip(img, 0, 255).astype(np.uint8)
        
    return image, grasps


def grasp_collate(batch: List[Dict]) -> Dict:
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    tensor_keys = ["image", "quality", "cos2theta", "sin2theta", "width"]
    list_keys   = ["prompt", "grasps"]
    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in list_keys:
        out[k] = [b[k] for b in batch]
    return out


def is_not_none(x):
    return x is not None

def open_local(url, *args, **kw):
    path = url.replace("local://", "")
    return open(path, "rb")
wds.gopen_schemes["local"] = open_local

def build_webdataset_loader(
    shard_index: str,
    batch_size: int = 64,
    num_workers: int = 8,
    image_size: int = 224,
    sigma: float = 5.0,
    augment: bool = False,
    min_score: float = 0.0,
    shuffle_buffer: int = 5000,
) -> torch.utils.data.DataLoader:
    shard_dir = Path(shard_index).parent
    raw_paths = Path(shard_index).read_text().strip().splitlines()
    
    shard_paths = []
    for p in raw_paths:
        p = p.strip()
        if not p:
            continue
        filename = Path(p.replace('\\', '/')).name
        real_path = shard_dir / filename
        shard_paths.append(f"local://{real_path.as_posix()}")

    transform = TransformPipeline(image_size, sigma, augment, min_score)

    dataset = (
        wds.WebDataset(
            shard_paths,
            resampled=augment,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(shuffle_buffer if augment else 0)
        .map(transform, handler=wds.warn_and_continue)
        .select(is_not_none)          
        .batched(batch_size, collation_fn=grasp_collate, partial=not augment)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return loader
