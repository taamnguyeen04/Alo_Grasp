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
    .add_local_file("train.py", remote_path="/root/Alo_Grasp/train.py")
    .add_local_file("evaluate.py", remote_path="/root/Alo_Grasp/evaluate.py")
)

app = modal.App(name="grasp-training", image=image)



@app.function(
    gpu="H100",
    volumes={
        "/data": data_vol,
        "/mnt/checkpoints": ckpt_vol,
    },
    timeout=18000,
    cpu=8.0,
    memory=32768,
)
def run_training(
    config: str = "configs/default.yaml",
    batch_size: int = None,
    epochs: int = None,
    lr_decoder: float = None,
    amp: bool = True
):

    import signal
    import sys
    import os
    import torch
    
    sys.path.insert(0, "/root/Alo_Grasp")
    os.chdir("/root/Alo_Grasp")
    
    def _commit():
        ckpt_vol.commit()
        print(f"[MODAL] Committed results to 'grasp-checkpoints' volume.")
    def _sigterm(signum, frame):
        print(f"[MODAL] Received SIGTERM signal (interrupted) - saving everything...")
        _commit()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)
    _orig_save = torch.save
    def _patched_save(*args, **kwargs):
        _orig_save(*args, **kwargs)
        _commit()
    torch.save = _patched_save

    sys.argv = ["train.py", "--config", config]
    
    sys.argv.extend([
        "data.base_root=/data",
        "data.pp_root=/data",
        "data.shard_train_index=/data/shards/train.txt",
        "data.shard_val_index=/data/shards/val.txt",
        "data.use_webdataset=true",
        "experiment.output_dir=/mnt/checkpoints/logs"
    ])
    
    if batch_size is not None:
        sys.argv.append(f"train.batch_size={batch_size}")
    if epochs is not None:
        sys.argv.append(f"train.epochs={epochs}")
    if lr_decoder is not None:
        sys.argv.append(f"optimizer.lr_decoder={lr_decoder}")
    if not amp:
        sys.argv.append("train.amp=false")
        
    print(f"[MODAL] Simulated command: python {' '.join(sys.argv)}")
    print(f"[MODAL] Starting Training Process...")
    
    import train
    
    try:
        train.main()
    except Exception as e:
        print(f"[MODAL] Error during training: {e}")
        raise e
    finally:
        print(f"[MODAL] Training finished. Making final data commit...")
        _commit()
        print(f"[MODAL] Process completed on Modal GPU!")


@app.local_entrypoint()
def main(
    config: str = "configs/default.yaml",
    batch_size: int = 512,
    epochs: int = 20,
    lr_decoder: float = 0.0005,
    amp: bool = True
):
    print(f"[LOCAL] Using config: {config}")

    run_training.remote(
        config=config,
        batch_size=batch_size,
        epochs=epochs,
        lr_decoder=lr_decoder,
        amp=amp
    )


@app.function(
    gpu="H100",
    volumes={
        "/data": data_vol,
        "/mnt/checkpoints": ckpt_vol,
    },
    timeout=18000,
    cpu=8.0,
    memory=32768,
)
def run_ablation_remote(variant: str = "all", quick_mode: bool = False):
    import subprocess
    import os
    import glob
    import time
    
    os.chdir("/root/Alo_Grasp")

    def _commit():
        ckpt_vol.commit()
        print(f"[MODAL] Committed results to 'grasp-checkpoints' volume.")

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
            print(f"[ERROR] No variant matched: '{variant}'")
            print(f"Valid variants: {', '.join(all_variants)}")
            return
            
    results_dir = "/mnt/checkpoints/logs/ablation"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/results_modal.txt"
    
    with open(results_file, "a") as f:
        f.write(f"\n\n=== Ablation results (Modal) - {time.ctime()} ===\n")
        
    for var in variants:
        print(f"\n--- Variant: {var} ---")
        
        # 1. Train
        cmd_train = ["python", "train.py", "--config", f"configs/ablation/{var}.yaml"]
        cmd_train.extend([
            "data.base_root=/data",
            "data.pp_root=/data",
            "data.shard_train_index=/data/shards/train.txt",
            "data.shard_val_index=/data/shards/val.txt",
            "data.use_webdataset=true",
            "experiment.output_dir=/mnt/checkpoints/logs/ablation"
        ])
        if quick_mode:
            cmd_train.extend(["train.epochs=5", "data.max_train_samples=5000", "data.max_val_samples=2000"])
            
        print(f"[Training] {var} ...")
        subprocess.run(cmd_train, check=True)
        _commit()
        
        # 2. Evaluate
        print(f"[Evaluating] {var} ... Skipping on Modal!")
        
        with open(results_file, "a") as f:
            f.write(f"\n[Variant: {var}]\n")
            f.write(f"Training completed.\n")
            f.write("\n--- end ---\n")
            
    print(f"\n[DONE] Ablation completed ({variant})! Results saved at: {results_file}")


@app.local_entrypoint()
def run_ablation(variant: str = "all", quick: bool = False):
    print(f"[LOCAL] Triggering ablation ({variant}) on Modal...")
    if quick:
        print(f"[LOCAL] QUICK MODE enabled.")
    run_ablation_remote.remote(variant=variant, quick_mode=quick)
