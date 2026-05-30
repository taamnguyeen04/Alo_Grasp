import argparse
import io
import pickle
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def get_all_stems(pp_root: Path) -> List[str]:
    instr_dir = pp_root / "grasp_instructions"
    label_dir = pp_root / "grasp_label_positive"
    stems = []
    for pkl in sorted(instr_dir.glob("*.pkl")):
        if (label_dir / f"{pkl.stem}.pt").exists():
            stems.append(pkl.stem)
    return stems


def load_sample(stem: str, base_root: Path, pp_root: Path, image_size: int):
    scene_id = stem[:64]
    img_path  = base_root / "image" / f"{scene_id}.jpg"
    pkl_path  = pp_root / "grasp_instructions" / f"{stem}.pkl"
    pt_path   = pp_root / "grasp_label_positive" / f"{stem}.pt"

    img = Image.open(img_path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    jpg_bytes = buf.getvalue()

    with open(pkl_path, "rb") as f:
        prompt = pickle.load(f)

    tensor = torch.load(pt_path, map_location="cpu", weights_only=False)

    return stem, jpg_bytes, prompt, tensor


def write_shard(
    shard_path: Path,
    stems: List[str],
    base_root: Path,
    pp_root: Path,
    image_size: int,
    workers: int = 4,
) -> int:
    written = 0
    with tarfile.open(shard_path, "w") as tar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(load_sample, s, base_root, pp_root, image_size): s
                for s in stems
            }
            for fut in as_completed(futures):
                try:
                    stem, jpg_bytes, prompt, tensor = fut.result()
                except Exception as e:
                    print(f"  Warning: skip {futures[fut]}: {e}")
                    continue

                key = stem

                info = tarfile.TarInfo(name=f"{key}.jpg")
                info.size = len(jpg_bytes)
                tar.addfile(info, io.BytesIO(jpg_bytes))

                prompt_bytes = prompt.encode("utf-8")
                info = tarfile.TarInfo(name=f"{key}.cls")
                info.size = len(prompt_bytes)
                tar.addfile(info, io.BytesIO(prompt_bytes))

                tensor_buf = io.BytesIO()
                torch.save(tensor, tensor_buf)
                tensor_bytes = tensor_buf.getvalue()
                info = tarfile.TarInfo(name=f"{key}.pth")
                info.size = len(tensor_bytes)
                tar.addfile(info, io.BytesIO(tensor_bytes))

                written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--pp-root",   required=True)
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-shard", type=int, default=2000)
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    pp_root   = Path(args.pp_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset...")
    all_stems = get_all_stems(pp_root)
    print(f"  Found {len(all_stems):,} samples")

    scene_ids = sorted(set(s[:64] for s in all_stems))
    cut = max(1, int(len(scene_ids) * (1 - args.val_ratio)))
    train_scenes = set(scene_ids[:cut])
    val_scenes   = set(scene_ids[cut:])

    train_stems = [s for s in all_stems if s[:64] in train_scenes]
    val_stems   = [s for s in all_stems if s[:64] in val_scenes]
    print(f"  Train: {len(train_stems):,} | Val: {len(val_stems):,}")

    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        print(f"\nPacking {split_name}...")
        shard_idx = 0
        total_written = 0

        for start in tqdm(range(0, len(stems), args.samples_per_shard),
                          desc=split_name):
            chunk = stems[start : start + args.samples_per_shard]
            shard_path = out_dir / f"{split_name}-{shard_idx:06d}.tar"
            n = write_shard(shard_path, chunk, base_root, pp_root,
                            args.image_size, args.workers)
            total_written += n
            shard_idx += 1

        print(f"  → {shard_idx} shards, {total_written:,} samples")

    for split_name in ["train", "val"]:
        shards = sorted(out_dir.glob(f"{split_name}-*.tar"))
        index_path = out_dir / f"{split_name}.txt"
        index_path.write_text("\n".join(str(s) for s in shards))
        print(f"Shard index: {index_path}  ({len(shards)} shards)")

    print("\nDone!")


if __name__ == "__main__":
    main()
