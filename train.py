from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import GraspAnythingPlus, grasp_collate
from data.webdataset_loader import build_webdataset_loader
from losses import GraspLoss
from models import GraspCLIPD
from utils import GraspMetric, decode_maps


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_param_groups(model: GraspCLIPD, lr_decoder: float, lr_backbone: float, weight_decay: float):
    backbone_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("visual.") or name.startswith("text."):
            backbone_params.append(p)
        else:
            decoder_params.append(p)
    groups = [{"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    return groups


def pick_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    cap = torch.cuda.get_device_capability(device)
    if cap[0] >= 8:  # Ampere+
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model: GraspCLIPD, loader: DataLoader, device: torch.device, cfg) -> Dict[str, float]:
    model.eval()
    metric = GraspMetric(
        iou_threshold=cfg.evaluation.iou_threshold,
        angle_threshold_deg=cfg.evaluation.angle_threshold_deg,
    )
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        prompts = batch["prompt"]
        amp_dtype = pick_amp_dtype(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.train.amp):
            preds = model(image, prompts)

        pred_grasps = decode_maps(
            preds["quality"],
            preds["cos2theta"],
            preds["sin2theta"],
            preds["width"],
            topk=cfg.evaluation.topk,
        )
        metric.update(pred_grasps, batch["grasps"])

    success = metric.compute()
    return {"success_rate": success}



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint .ckpt file to resume training."
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style overrides, e.g. train.batch_size=32",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.overrides))

    set_seed(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    if cfg.data.get("use_webdataset", False):
        print(f"[INFO] Using WebDataset Loader...")
        train_loader = build_webdataset_loader(
            shard_index=cfg.data.shard_train_index,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=cfg.data.image_size,
            sigma=5.0, # default
            augment=True,
            min_score=0.0,
            shuffle_buffer=5000,
        )
        val_loader = build_webdataset_loader(
            shard_index=cfg.data.shard_val_index,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=cfg.data.image_size,
            sigma=5.0,
            augment=False,
            min_score=0.0,
            shuffle_buffer=0,
        )
        steps_per_epoch = 4000000 // cfg.train.batch_size
    else:
        print(f"[INFO] Using Standard DataLoader (FileDataset)...")
        train_ds = GraspAnythingPlus(
            base_root=cfg.data.base_root,
            pp_root=cfg.data.pp_root,
            split=cfg.data.splits.train,
            image_size=cfg.data.image_size,
            max_samples=cfg.data.max_train_samples,
            augment=True,
        )
        val_ds = GraspAnythingPlus(
            base_root=cfg.data.base_root,
            pp_root=cfg.data.pp_root,
            split=cfg.data.splits.val,
            image_size=cfg.data.image_size,
            max_samples=cfg.data.max_val_samples,
            augment=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=grasp_collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=grasp_collate,
        )
        steps_per_epoch = len(train_loader)

    # Model
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

    n_trainable = model.count_trainable_parameters()
    print(f"Trainable parameters: {n_trainable / 1e6:.2f} M")

    criterion = GraspLoss(
        weight_quality=cfg.loss.weight_quality,
        weight_angle=cfg.loss.weight_angle,
        weight_width=cfg.loss.weight_width,
    ).to(device)

    # Optimiser + schedule
    param_groups = build_param_groups(
        model,
        lr_decoder=cfg.optimizer.lr_decoder,
        lr_backbone=cfg.optimizer.lr_backbone,
        weight_decay=cfg.optimizer.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg.optimizer.betas))

    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.scheduler.warmup_epochs * steps_per_epoch

    amp_dtype = pick_amp_dtype(device)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and amp_dtype == torch.float16)

    # Logging / checkpoints
    output_dir = Path(cfg.experiment.output_dir) / f"{cfg.experiment.name}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    print(f"Saving to {output_dir}")

    wandb_run = None
    if cfg.experiment.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg.experiment.wandb_project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    best_success = -1.0
    global_step = 0
    start_epoch = 0

    # Resume Checkpoint
    if args.resume:
        if not os.path.isfile(args.resume):
            print(f"[WARNING] Checkpoint '{args.resume}' not found. Training from scratch.")
        else:
            print(f"[INFO] Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            
            model.load_state_dict(ckpt["model"])
            
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            
            if "scaler" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])
                
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "global_step" in ckpt:
                global_step = ckpt["global_step"]
            else:
                global_step = start_epoch * steps_per_epoch
                
            if "success_rate" in ckpt:
                best_success = ckpt["success_rate"]
                
            print(f"  -> Resumed at Epoch {start_epoch}, Step {global_step}, Best Success: {best_success:.4f}")

    # Training loop
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        epoch_start = time.time()
        running = {"total": 0.0, "q": 0.0, "ang": 0.0, "w": 0.0}

        import itertools
        pbar = tqdm(itertools.islice(train_loader, steps_per_epoch), total=steps_per_epoch, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            targets = {
                "quality": batch["quality"].to(device, non_blocking=True),
                "cos2theta": batch["cos2theta"].to(device, non_blocking=True),
                "sin2theta": batch["sin2theta"].to(device, non_blocking=True),
                "width": batch["width"].to(device, non_blocking=True),
            }
            prompts = batch["prompt"]

            # LR schedule per step.
            lr_dec = cosine_warmup_lr(
                global_step, total_steps, warmup_steps,
                cfg.optimizer.lr_decoder, cfg.scheduler.min_lr,
            )
            optimizer.param_groups[0]["lr"] = lr_dec
            if len(optimizer.param_groups) > 1:
                lr_bb = cosine_warmup_lr(
                    global_step, total_steps, warmup_steps,
                    cfg.optimizer.lr_backbone, cfg.scheduler.min_lr * 0.1,
                )
                optimizer.param_groups[1]["lr"] = lr_bb

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.train.amp):
                preds = model(image, prompts)
                loss = criterion(preds, targets)

            if scaler.is_enabled():
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.train.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.train.grad_clip,
                )
                optimizer.step()

            running["total"] += loss.total.item()
            running["q"] += loss.loss_q.item()
            running["ang"] += loss.loss_angle.item()
            running["w"] += loss.loss_width.item()

            global_step += 1
            if global_step % cfg.train.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.total.item():.4f}",
                    q=f"{loss.loss_q.item():.4f}",
                    ang=f"{loss.loss_angle.item():.4f}",
                    w=f"{loss.loss_width.item():.4f}",
                    lr=f"{lr_dec:.2e}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.total.item(),
                            "train/loss_q": loss.loss_q.item(),
                            "train/loss_angle": loss.loss_angle.item(),
                            "train/loss_width": loss.loss_width.item(),
                            "train/lr": lr_dec,
                            "step": global_step,
                        }
                    )

        # End of epoch.
        epoch_loss = {k: v / max(1, steps_per_epoch) for k, v in running.items()}
        print(
            f"epoch {epoch} | loss {epoch_loss['total']:.4f} "
            f"| q {epoch_loss['q']:.4f} | ang {epoch_loss['ang']:.4f} "
            f"| w {epoch_loss['w']:.4f} | time {time.time() - epoch_start:.1f}s"
        )

        if (epoch + 1) % cfg.train.eval_every == 0:
            metrics = evaluate(model, val_loader, device, cfg)
            print(f"  -> val success rate: {metrics['success_rate']:.4f}")
            if wandb_run is not None:
                wandb_run.log({"val/success_rate": metrics["success_rate"], "epoch": epoch})

            if metrics["success_rate"] > best_success:
                best_success = metrics["success_rate"]
                ckpt_path = output_dir / "best.ckpt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "epoch": epoch,
                        "global_step": global_step,
                        "success_rate": best_success,
                    },
                    ckpt_path,
                )
                print(f"  -> saved best to {ckpt_path}")

        # Always save last (with full state for resuming).
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch,
                "global_step": global_step,
                "success_rate": best_success,
            },
            output_dir / "last.ckpt",
        )

    print(f"Training complete. Best val success: {best_success:.4f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
