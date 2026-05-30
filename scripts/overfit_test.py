from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import GraspAnythingPlus, grasp_collate  # noqa: E402
from losses import GraspLoss  # noqa: E402
from models import GraspCLIPD  # noqa: E402
from utils import GraspMetric, decode_maps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GraspAnythingPlus(
        base_root=cfg.data.base_root,
        pp_root=cfg.data.pp_root,
        split=cfg.data.splits.train,
        image_size=cfg.data.image_size,
        augment=False,
        max_samples=args.num_samples,
    )
    subset = Subset(ds, list(range(min(args.num_samples, len(ds)))))
    loader = DataLoader(subset, batch_size=args.num_samples, shuffle=False, collate_fn=grasp_collate)
    batch = next(iter(loader))

    image = batch["image"].to(device)
    prompts = batch["prompt"]
    targets = {
        "quality": batch["quality"].to(device),
        "cos2theta": batch["cos2theta"].to(device),
        "sin2theta": batch["sin2theta"].to(device),
        "width": batch["width"].to(device),
    }

    model = GraspCLIPD(
        visual_backbone=cfg.model.visual_backbone,
        text_encoder=cfg.model.text_encoder,
        text_pretrained=cfg.model.text_pretrained,
        projection_dim=cfg.model.projection_dim,
        decoder_channels=list(cfg.model.decoder_channels),
        use_cross_attention=cfg.model.use_cross_attention,
        cross_attn_heads=cfg.model.cross_attn_heads,
        output_size=cfg.data.output_size,
        freeze_visual=cfg.model.freeze_visual,
        freeze_text=cfg.model.freeze_text,
    ).to(device)

    criterion = GraspLoss(
        weight_quality=cfg.loss.weight_quality,
        weight_angle=cfg.loss.weight_angle,
        weight_width=cfg.loss.weight_width,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    print(f"Overfitting on {args.num_samples} samples for {args.epochs} epochs ...")
    losses = []
    for ep in range(args.epochs):
        model.train()
        optim.zero_grad()
        preds = model(image, prompts)
        loss = criterion(preds, targets)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        losses.append(loss.total.item())
        if (ep + 1) % 20 == 0 or ep == 0:
            print(
                f"  epoch {ep + 1:4d}/{args.epochs} | "
                f"total {loss.total.item():.5f} | "
                f"q {loss.loss_q.item():.5f} | "
                f"ang {loss.loss_angle.item():.5f} | "
                f"w {loss.loss_width.item():.5f}"
            )

    print(f"\nFinal loss: {losses[-1]:.6f}  (started at {losses[0]:.6f})")

    model.eval()
    with torch.no_grad():
        preds = model(image, prompts)
        pred_grasps = decode_maps(
            preds["quality"], preds["cos2theta"], preds["sin2theta"], preds["width"], topk=1
        )
    metric = GraspMetric(
        iou_threshold=cfg.evaluation.iou_threshold,
        angle_threshold_deg=cfg.evaluation.angle_threshold_deg,
    )
    metric.update(pred_grasps, batch["grasps"])
    print(f"Train-set success rate (should approach 1.0): {metric.compute():.4f}")

    if losses[-1] > 0.05:
        print(f"\n[WARNING] Final loss is high ({losses[-1]:.6f}). Check LR, augmentations, or frozen modules.")
    else:
        print("\n[OK] Loss converged.")


if __name__ == "__main__":
    main()
