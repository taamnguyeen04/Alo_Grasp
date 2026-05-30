import modal
import os
import sys


data_vol = modal.Volume.from_name("grasp-dataset")
ckpt_vol = modal.Volume.from_name("grasp-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0", "torchvision>=0.16.0", "open-clip-torch>=2.24.0",
        "einops>=0.7.0", "numpy<2.0.0", "pillow>=10.0.0", "shapely>=2.0.0",
        "pyyaml>=6.0", "omegaconf>=2.3.0", "tqdm>=4.66.0", "matplotlib>=3.8.0",
        "opencv-python-headless>=4.8.0", "wandb>=0.16.0", "scikit-image>=0.22.0", "webdataset"
    )
    .add_local_dir("models", remote_path="/root/Alo_Grasp/models")
    .add_local_dir("data", remote_path="/root/Alo_Grasp/data")
    .add_local_dir("losses", remote_path="/root/Alo_Grasp/losses")
    .add_local_dir("utils", remote_path="/root/Alo_Grasp/utils")
    .add_local_dir("configs", remote_path="/root/Alo_Grasp/configs")
    .add_local_dir("scripts", remote_path="/root/Alo_Grasp/scripts")
    .add_local_dir("ablation", remote_path="/root/Alo_Grasp/ablation")
    .add_local_file("evaluate.py", remote_path="/root/Alo_Grasp/evaluate.py")
)

app = modal.App(name="grasp-evaluate", image=image)

@app.function(
    gpu="A100",
    volumes={
        "/data": data_vol,               
        "/mnt/checkpoints": ckpt_vol,    
    },
    timeout=18000,
    cpu=8.0,
    memory=32768,
    max_containers=5, 
)
def evaluate_single_variant(var: str, batch_size: int = 512):
    import os
    import sys
    import glob
    import time
    import tarfile
    from pathlib import Path
    from collections import Counter
    import torch
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm

    sys.path.insert(0, "/root/Alo_Grasp")
    os.chdir("/root/Alo_Grasp")

    import evaluate 
    from utils import decode_maps, extract_label_from_prompt, split_base_new
    from models import GraspCLIPD
    from data.webdataset_loader import build_webdataset_loader

    results_dir = "/mnt/checkpoints/logs/ablation"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/results_modal.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    amp_enabled = True
    cap = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    print(f"\n--- [Container] Evaluating Variant: {var} ---")

    search_paths = [
        f"/data/ablation/*{var}*/best.ckpt",
        f"/mnt/checkpoints/logs/ablation/*{var}*/best.ckpt",
        f"/root/Alo_Grasp/ablation/*{var}*/best.ckpt"
    ]
    
    ckpt_paths = []
    for path in search_paths:
        ckpt_paths.extend(glob.glob(path))
    
    if not ckpt_paths:
        msg = f"[SKIP] Could not find best.ckpt checkpoint for {var}"
        print(msg)
        return {"variant": var, "success": False, "msg": msg}
    
    ckpt_paths.sort(key=os.path.getmtime, reverse=True)
    latest_ckpt = ckpt_paths[0]
    print(f"[{var}] Found checkpoint: {latest_ckpt}")

    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    
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

    shard_val_index = "/data/shards/val.txt"
    
    print(f"[{var}] Scanning .tar files to calculate Base/New split...")
    counts = Counter()
    shard_dir = Path(shard_val_index).parent
    raw_paths = Path(shard_val_index).read_text().strip().splitlines()
    for p in raw_paths:
        p = p.strip()
        if not p: continue
        filename = Path(p.replace('\\', '/')).name
        real_path = shard_dir / filename
        try:
            with tarfile.open(real_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".cls"):
                        f = tar.extractfile(member)
                        if f is not None:
                            prompt = f.read().decode("utf-8")
                            lab = extract_label_from_prompt(prompt)
                            if lab:
                                counts[lab] += 1
        except Exception:
            pass
            
    base_labels, new_labels = split_base_new(dict(counts), 0.70)
    print(f"[{var}] Scanning completed! Base: {len(base_labels)} labels | New: {len(new_labels)} labels.")

    print(f"[{var}] Initializing DataLoader...")
    loader = build_webdataset_loader(
        shard_index=shard_val_index,
        batch_size=batch_size,
        num_workers=8,
        image_size=cfg.data.image_size,
        sigma=5.0,
        augment=False,
        min_score=0.0,
        shuffle_buffer=0,
    )

    iou_thresh   = cfg.evaluation.iou_threshold
    angle_thresh = cfg.evaluation.angle_threshold_deg

    n_correct_overall = n_total_overall = 0
    n_correct_base = n_total_base = 0
    n_correct_new  = n_total_new  = 0

    t_inf_start = time.time()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {var}", dynamic_ncols=True):
            image = batch["image"].to(device, non_blocking=True)
            prompts = batch["prompt"]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = model(image, prompts)

            pred_grasps = decode_maps(
                preds["quality"].float(),
                preds["cos2theta"].float(),
                preds["sin2theta"].float(),
                preds["width"].float(),
                topk=1,
            )

            correct_flags = evaluate.batch_is_correct(
                pred_grasps, batch["grasps"],
                iou_thresh=iou_thresh,
                angle_thresh_deg=angle_thresh,
                device=device,
            )

            for i, correct in enumerate(correct_flags):
                n_total_overall += 1
                if correct:
                    n_correct_overall += 1

                lab = extract_label_from_prompt(prompts[i])
                if lab in base_labels:
                    n_total_base += 1
                    if correct:
                        n_correct_base += 1
                elif lab in new_labels:
                    n_total_new += 1
                    if correct:
                        n_correct_new += 1

    t_inf = time.time() - t_inf_start
    overall = n_correct_overall / max(1, n_total_overall)
    base    = n_correct_base / max(1, n_total_base)
    new_sr  = n_correct_new  / max(1, n_total_new)
    H       = 2 * base * new_sr / (base + new_sr) if (base + new_sr) > 0 else 0.0
    fps     = n_total_overall / t_inf if t_inf > 0 else 0

    print(f"\n[RESULT {var}]:")
    print(f"  Samples  : {n_total_overall}")
    print(f"  Overall  : {overall * 100:.2f}%")
    print(f"  Base     : {base * 100:.2f}%")
    print(f"  New      : {new_sr * 100:.2f}%")
    print(f"  Harmonic : {H * 100:.2f}%")
    print(f"  FPS      : {fps:.1f}")
    
    with open(results_file, "a") as f:
        f.write(f"\nVariant: {var}\n")
        f.write(f"  Checkpoint: {latest_ckpt}\n")
        f.write(f"  Samples: {n_total_overall}\n")
        f.write(f"  Overall: {overall * 100:.2f}%\n")
        f.write(f"  Base: {base * 100:.2f}%\n")
        f.write(f"  New: {new_sr * 100:.2f}%\n")
        f.write(f"  Harmonic: {H * 100:.2f}%\n")
        f.write(f"  FPS: {fps:.1f}\n")
        f.write("-" * 50 + "\n")
        
    ckpt_vol.commit()
    
    return {
        "variant": var,
        "success": True,
        "overall": overall * 100,
        "base": base * 100,
        "new": new_sr * 100,
        "harmonic": H * 100,
        "fps": fps,
        "n_total": n_total_overall,
    }


@app.local_entrypoint()
def main(variant: str = "all", batch_size: int = 512):
    import time
    
    print(f"[LOCAL] Triggering PARALLEL ablation evaluation (Full Metrics)...")
    print(f"[LOCAL] Batch size: {batch_size}")
    
    all_variants = [
        "v1_clip_concat",
        "v2_clip_film_last",
        "v3_dinov2_film_last",
        "v4_dinov2_film_multiscale",
        "v5_full_method"
    ]

    if variant == "all":
        variants = all_variants
    else:
        variants = [v for v in all_variants if v.startswith(variant)]
        if not variants:
            return
            
    print(f"[LOCAL] Running {len(variants)} containers in parallel...")

    args = [(var, batch_size) for var in variants]
    t0 = time.time()
    
    results = []
    for res in evaluate_single_variant.starmap(args):
        results.append(res)
        
    t_total = time.time() - t0
    
    print(f"\n--- EVALUATION COMPLETED (Total time: {t_total:.1f}s) ---")
    
    for r in results:
        var = r["variant"]
        if r.get("success"):
            print(f"[OK] {var:<30}")
            print(f"   Overall: {r['overall']:.2f}% | Base: {r['base']:.2f}% | New: {r['new']:.2f}% | H: {r['harmonic']:.2f}% | FPS: {r['fps']:.1f}")
        else:
            print(f"[FAIL] {var:<30} | {r.get('msg', 'Error')}")
            
    print(f"\nDetailed logs written to '/mnt/checkpoints/logs/ablation/results_modal.txt'")
