"""Grasp-Anything + Grasp-Anything++ dataset loader."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.grasp import Grasp, grasp_to_maps


class GraspAnythingPlus(Dataset):
    """Language-driven grasp dataset (Grasp-Anything + Grasp-Anything++)."""

    def __init__(
        self,
        base_root: str,
        pp_root: str,
        split: str = "train",
        image_size: int = 224,
        max_samples: Optional[int] = None,
        augment: bool = False,
        sigma: float = 5.0,
        min_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_root  = Path(base_root)
        self.pp_root    = Path(pp_root)
        self.image_size = image_size
        self.augment    = augment
        self.sigma      = sigma
        self.min_score  = min_score

        self.image_dir  = self.base_root / "image"
        self.instr_dir  = self.pp_root / "grasp_instructions"
        self.label_dir  = self.pp_root / "grasp_label_positive"

        for d, name in [(self.image_dir, "image/"),
                        (self.instr_dir, "grasp_instructions/"),
                        (self.label_dir, "grasp_label_positive/")]:
            if not d.is_dir():
                raise FileNotFoundError(
                    f"Missing {name} at {d}\n"
                    "Check base_root and pp_root in config."
                )

        self.samples: List[Tuple[Path, Path]] = self._build_index(split)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}'.\n"
                f"  instr_dir = {self.instr_dir}\n"
                f"  label_dir = {self.label_dir}"
            )

        print(f"[GraspAnythingPlus] split='{split}'  samples={len(self.samples):,}")

    def _build_index(self, split: str) -> List[Tuple[Path, Path]]:
        all_pkls = sorted(self.instr_dir.glob("*.pkl"))
        if not all_pkls:
            raise FileNotFoundError(f"No .pkl files in {self.instr_dir}")

        valid: List[Tuple[Path, Path]] = []
        for pkl_path in all_pkls:
            stem = pkl_path.stem
            pt_path = self.label_dir / f"{stem}.pt"
            if not pt_path.exists():
                continue
            scene_id = self._scene_id_from_stem(stem)
            img_path = self.image_dir / f"{scene_id}.jpg"
            if not img_path.exists():
                continue
            valid.append((pkl_path, pt_path))

        if not valid:
            raise FileNotFoundError(
                "No valid (pkl, pt, jpg) pairs found.\n"
                "Check base_root / pp_root paths."
            )

        return self._apply_split(valid, split)

    @staticmethod
    def _scene_id_from_stem(stem: str) -> str:
        return stem[:64]

    def _apply_split(
        self, samples: List[Tuple[Path, Path]], split: str
    ) -> List[Tuple[Path, Path]]:
        if Path(split).is_file():
            allowed = set(Path(split).read_text().splitlines())
            return [(p, l) for p, l in samples
                    if self._scene_id_from_stem(p.stem) in allowed]

        if split == "all":
            return samples

        from collections import defaultdict
        scene_map = defaultdict(list)
        for pair in samples:
            sid = self._scene_id_from_stem(pair[0].stem)
            scene_map[sid].append(pair)

        scene_ids = sorted(scene_map.keys())
        n = len(scene_ids)
        cut = max(1, int(n * 0.9))

        if split == "train":
            chosen_scenes = set(scene_ids[:cut])
        elif split in ("val", "unseen"):
            chosen_scenes = set(scene_ids[cut:])
        else:
            print(f"Warning: unknown split='{split}', using 'all'.")
            return samples

        result = []
        for sid in chosen_scenes:
            result.extend(scene_map[sid])
        return result

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_prompt(self, pkl_path: Path) -> str:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def _load_grasps(self, pt_path: Path) -> List[Grasp]:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        if not isinstance(data, torch.Tensor):
            return []
        if data.ndim == 1:
            data = data.unsqueeze(0)
        if data.shape[1] != 6:
            return []

        grasps: List[Grasp] = []
        for row in data.tolist():
            score, x, y, w, h, theta_deg = row
            if score < self.min_score:
                continue
            theta_rad = np.deg2rad(theta_deg) % np.pi
            if theta_rad >= np.pi / 2:
                theta_rad -= np.pi
            grasps.append(Grasp(x=x, y=y, w=w, h=h, theta=theta_rad))

        return grasps

    def _load_image(self, stem: str) -> np.ndarray:
        scene_id = self._scene_id_from_stem(stem)
        path = self.image_dir / f"{scene_id}.jpg"
        return np.asarray(Image.open(path).convert("RGB"))

    # ── Augmentation ───────────────────────────────────────────────────────────

    def _augment(
        self, image: np.ndarray, grasps: List[Grasp]
    ) -> Tuple[np.ndarray, List[Grasp]]:
        s = self.image_size

        angle_deg = random.uniform(-15.0, 15.0)
        if abs(angle_deg) > 0.5:
            image_pil = Image.fromarray(image).rotate(
                angle_deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0)
            )
            image = np.asarray(image_pil)
            theta_rad = -np.deg2rad(angle_deg)
            cos_r, sin_r = np.cos(theta_rad), np.sin(theta_rad)
            cx, cy = s / 2.0, s / 2.0
            new_grasps: List[Grasp] = []
            for g in grasps:
                dx, dy = g.x - cx, g.y - cy
                new_grasps.append(Grasp(
                    x     = cos_r * dx - sin_r * dy + cx,
                    y     = sin_r * dx + cos_r * dy + cy,
                    w     = g.w, h = g.h,
                    theta = g.theta + theta_rad,
                ))
            grasps = new_grasps

        if random.random() < 0.5:
            img = image.astype(np.float32)
            img *= random.uniform(0.8, 1.2)
            mu  = img.mean()
            img = (img - mu) * random.uniform(0.8, 1.2) + mu
            image = np.clip(img, 0, 255).astype(np.uint8)

        return image, grasps

    # ── PyTorch API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        pkl_path, pt_path = self.samples[idx]
        stem = pkl_path.stem

        image_np = self._load_image(stem)
        h0, w0 = image_np.shape[:2]
        s = self.image_size
        scale_x, scale_y = s / w0, s / h0

        image_np = np.asarray(
            Image.fromarray(image_np).resize((s, s), Image.BILINEAR)
        )

        raw_grasps = self._load_grasps(pt_path)
        grasps = [
            Grasp(
                x     = g.x * scale_x,
                y     = g.y * scale_y,
                w     = g.w * scale_x,
                h     = g.h * scale_y,
                theta = g.theta,
            )
            for g in raw_grasps
        ]

        prompt = self._load_prompt(pkl_path)

        if self.augment:
            image_np, grasps = self._augment(image_np, grasps)

        q, c2t, s2t, w = grasp_to_maps(grasps, (s, s), sigma=self.sigma)

        img_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0

        return {
            "image"     : img_tensor,
            "prompt"    : prompt,
            "quality"   : torch.from_numpy(q).float(),
            "cos2theta" : torch.from_numpy(c2t).float(),
            "sin2theta" : torch.from_numpy(s2t).float(),
            "width"     : torch.from_numpy(w).float(),
            "grasps"    : grasps,
            "stem"      : stem,
        }

def grasp_collate(batch: List[Dict]) -> Dict:
    tensor_keys = ["image", "quality", "cos2theta", "sin2theta", "width"]
    list_keys   = ["prompt", "grasps", "stem"]
    out: Dict   = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in list_keys:
        out[k] = [b[k] for b in batch]
    return out
