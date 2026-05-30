from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import GraspAnythingPlus, grasp_collate
from models import GraspCLIPD
from utils import decode_maps, extract_label_from_prompt, split_base_new
from utils.grasp import Grasp, MAX_GRIPPER_WIDTH_PX


# ---------------------------------------------------------------------------
# Fast vectorised rectangle IoU on GPU/CPU
# ---------------------------------------------------------------------------

def _corners_batch(cx: torch.Tensor, cy: torch.Tensor,
                   w: torch.Tensor, h: torch.Tensor,
                   theta: torch.Tensor) -> torch.Tensor:
    """Return (N, 4, 2) corner coordinates for N oriented rectangles."""
    cos_t = torch.cos(theta)   # (N,)
    sin_t = torch.sin(theta)
    dx, dy = w / 2, h / 2
    # local corners: BL BR TR TL
    lx = torch.stack([-dx, dx, dx, -dx], dim=1)   # (N, 4)
    ly = torch.stack([-dy, -dy, dy, dy], dim=1)
    rx = cos_t.unsqueeze(1) * lx - sin_t.unsqueeze(1) * ly + cx.unsqueeze(1)
    ry = sin_t.unsqueeze(1) * lx + cos_t.unsqueeze(1) * ly + cy.unsqueeze(1)
    return torch.stack([rx, ry], dim=2)   # (N, 4, 2)


def _poly_area(corners: torch.Tensor) -> torch.Tensor:
    """Shoelace formula for (N, 4, 2) polygons → (N,)."""
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    n = x.shape[1]
    area = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(n):
        j = (i + 1) % n
        area += x[:, i] * y[:, j] - x[:, j] * y[:, i]
    return torch.abs(area) / 2.0


def fast_iou_single(pred: Grasp, gts: List[Grasp],
                    device: torch.device) -> torch.Tensor:
    """IoU between one predicted grasp and N ground-truth grasps.

    Uses Shapely for correctness (oriented polygons) but batches all GTs
    together to avoid redundant per-pair Python overhead.

    Returns (N,) float tensor of IoU values.
    """
    # Fallback to Shapely — vectorising intersection of arbitrary oriented
    # polygons in pure PyTorch is complex; for N_gt < 50 (typical) Shapely
    # is fast enough when called once per predicted grasp (not per pair).
    from shapely.geometry import Polygon
    pred_poly = Polygon(pred.corners())
    if not pred_poly.is_valid:
        return torch.zeros(len(gts))
    ious = []
    for gt in gts:
        gt_poly = Polygon(gt.corners())
        if not gt_poly.is_valid:
            ious.append(0.0)
            continue
        inter = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        ious.append(inter / union if union > 0 else 0.0)
    return torch.tensor(ious, dtype=torch.float32)


def batch_is_correct(
    pred_grasps: List[List[Grasp]],
    gt_grasps:   List[List[Grasp]],
    iou_thresh:  float = 0.25,
    angle_thresh_deg: float = 30.0,
    device: torch.device = torch.device("cpu"),
) -> List[bool]:
    """Check correctness for a whole batch.  Returns list of bool (one per sample).

    Matches the exact logic of ``calculate_iou_match`` in the Fsoft-AIC repo:
      - angle_diff = |pred_theta - gt_theta| % (pi/2)   (Cornell convention)
      - correct iff IoU > iou_thresh AND angle_diff < pi/6
    """
    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    results = []
    for preds, gts in zip(pred_grasps, gt_grasps):
        if not preds or not gts:
            results.append(False)
            continue
        pred = preds[0]   # top-1 only
        correct = False
        for gt in gts:
            # Angle check (mod π/2, Cornell convention)
            adiff = abs(pred.theta - gt.theta) % (np.pi / 2)
            if adiff >= angle_thresh_rad:
                continue
            # IoU check (Shapely oriented polygon)
            from shapely.geometry import Polygon
            pp = Polygon(pred.corners())
            gp = Polygon(gt.corners())
            if not pp.is_valid or not gp.is_valid:
                continue
            union = pp.union(gp).area
            if union <= 0:
                continue
            if pp.intersection(gp).area / union > iou_thresh:
                correct = True
                break
        results.append(correct)
    return results


# ---------------------------------------------------------------------------
# Label counting directly from dataset index (avoids second full scan)
# ---------------------------------------------------------------------------

def count_labels_from_index(dataset: GraspAnythingPlus) -> Dict[str, int]:
    """Read labels directly from the in-memory sample list — O(N), no I/O.

    Each sample has a pre-loaded prompt string in dataset.samples via the pkl
    path; we read the pkl only if prompt is not cached. Since prompts are
    small strings, this is much faster than running the full DataLoader.
    """
    import pickle
    counts: Counter = Counter()
    print("  Reading labels from dataset index (no model needed) ...")
    for pkl_path, _ in tqdm(dataset.samples, desc="  indexing labels", leave=False):
        # Fast path: read pkl directly (already on disk, small string)
        try:
            with open(pkl_path, "rb") as f:
                prompt = pickle.load(f)
            lab = extract_label_from_prompt(str(prompt))
            if lab:
                counts[lab] += 1
        except Exception:
            pass
    return dict(counts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraspCLIP-D")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        help="'train', 'val', or 'all'")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base-root", type=str, default=None,
                        help="Override cfg.data.base_root")
    parser.add_argument("--pp-root", type=str, default=None,
                        help="Override cfg.data.pp_root")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Larger batch = faster GPU utilisation")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-new", action="store_true",
                        help="Compute Base/New/H split (adds label scan step)")
    parser.add_argument("--base-fraction", type=float, default=0.70)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate on first N samples only (for quick checks)")
    parser.add_argument("--visualise", type=int, default=0,
                        help="Save N qualitative figures")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Mixed precision inference (default: on)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint & config
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.create(ckpt["config"])

    if args.base_root:
        cfg.data.base_root = args.base_root
    if args.pp_root:
        cfg.data.pp_root = args.pp_root

    # Build model
    model = GraspCLIPD(
        visual_backbone     = cfg.model.visual_backbone,
        text_encoder        = cfg.model.text_encoder,
        text_pretrained     = cfg.model.text_pretrained,
        projection_dim      = cfg.model.projection_dim,
        decoder_channels    = list(cfg.model.decoder_channels),
        use_cross_attention = cfg.model.use_cross_attention,
        cross_attn_heads    = cfg.model.cross_attn_heads,
        output_size         = cfg.data.output_size,
        freeze_visual       = cfg.model.freeze_visual,
        freeze_text         = cfg.model.freeze_text,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # AMP dtype
    amp_enabled = args.amp and device.type == "cuda"
    if amp_enabled:
        cap = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        print(f"AMP: enabled ({amp_dtype})")
    else:
        amp_dtype = torch.float32
        print("AMP: disabled")

    # Dataset
    dataset = GraspAnythingPlus(
        base_root  = cfg.data.base_root,
        pp_root    = cfg.data.pp_root,
        split      = args.split,
        image_size = cfg.data.image_size,
        augment    = False,
        max_samples= args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        prefetch_factor = 4 if args.num_workers > 0 else None,
        persistent_workers = args.num_workers > 0,
        collate_fn  = grasp_collate,
    )
    print(f"Eval samples: {len(dataset):,}  batches: {len(loader):,}")

    # Step 1 (optional): Base/New label split
    base_labels, new_labels = set(), set()
    if args.base_new:
        print("\n[Step 1] Building Base/New label split ...")
        # Read labels directly from the dataset index — NO model forward needed
        label_counts = count_labels_from_index(dataset)
        base_labels, new_labels = split_base_new(label_counts, args.base_fraction)
        print(f"  Unique labels: {len(label_counts)} "
              f"| Base ({int(args.base_fraction*100)}%): {len(base_labels)} "
              f"| New: {len(new_labels)}")

    # Step 2: Inference + metric
    print(f"\n[Step 2] Running inference ...")

    iou_thresh   = cfg.evaluation.iou_threshold
    angle_thresh = cfg.evaluation.angle_threshold_deg

    # Accumulators
    n_correct_overall = 0
    n_total_overall   = 0
    n_correct_base = n_total_base = 0
    n_correct_new  = n_total_new  = 0

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / "viz")
    n_saved = 0
    if args.visualise > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        from utils.visualisation import plot_prediction

    t_inf_start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            image   = batch["image"].to(device, non_blocking=True)
            prompts = batch["prompt"]

            # Forward pass (AMP)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                preds = model(image, prompts)

            # Decode maps -> grasps (on CPU; NMS is cheap)
            pred_grasps = decode_maps(
                preds["quality"].float(),
                preds["cos2theta"].float(),
                preds["sin2theta"].float(),
                preds["width"].float(),
                topk=1,
            )

            # Correctness check (matches calculate_iou_match exactly)
            correct_flags = batch_is_correct(
                pred_grasps, batch["grasps"],
                iou_thresh=iou_thresh,
                angle_thresh_deg=angle_thresh,
                device=device,
            )

            # Accumulate metrics
            for i, correct in enumerate(correct_flags):
                n_total_overall += 1
                if correct:
                    n_correct_overall += 1

                if args.base_new:
                    lab = extract_label_from_prompt(prompts[i])
                    if lab in base_labels:
                        n_total_base += 1
                        if correct:
                            n_correct_base += 1
                    elif lab in new_labels:
                        n_total_new += 1
                        if correct:
                            n_correct_new += 1

            # Visualise (only first N, then skip)
            if n_saved < args.visualise:
                for i in range(min(image.size(0), args.visualise - n_saved)):
                    img_np = (image[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    fig = plot_prediction(
                        image       = img_np,
                        prompt      = prompts[i],
                        pred_grasps = pred_grasps[i],
                        gt_grasps   = batch["grasps"][i],
                        save_path   = output_dir / f"sample_{n_saved:03d}.png",
                    )
                    plt.close(fig)   # fix figure leak
                    n_saved += 1

    t_inf = time.time() - t_inf_start
    t_total = time.time() - t_start

    # Report
    overall = n_correct_overall / max(1, n_total_overall)
    base    = n_correct_base / max(1, n_total_base)
    new_sr  = n_correct_new  / max(1, n_total_new)
    H       = 2 * base * new_sr / (base + new_sr) if (base + new_sr) > 0 else 0.0

    fps = n_total_overall / t_inf

    print("\n" + "=" * 62)
    print(f"  EVALUATION RESULTS")
    print(f"  Rectangle Metric: IoU > {iou_thresh},  angle < {angle_thresh}°")
    print(f"  (angle_diff = |θ_pred - θ_gt| mod π/2  — Cornell convention)")
    print("=" * 62)
    print(f"  Split          : {args.split}")
    print(f"  Samples        : {n_total_overall:,}")
    print(f"  Inference FPS  : {fps:.1f}  ({t_inf:.1f}s total inference)")
    print(f"  Total time     : {t_total:.1f}s")
    print("-" * 62)
    print(f"  Overall        : {overall * 100:.2f} %")
    if args.base_new:
        print(f"  Base ({n_total_base:,})   : {base * 100:.2f} %")
        print(f"  New  ({n_total_new:,})    : {new_sr * 100:.2f} %")
        print(f"  Harmonic H     : {H * 100:.2f} %")
    print("=" * 62)

    if args.visualise > 0 and n_saved > 0:
        print(f"  Figures saved  : {n_saved} → {output_dir}")


if __name__ == "__main__":
    main()
