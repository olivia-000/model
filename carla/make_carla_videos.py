"""
把 CARLA recorded_Town01 / recorded_Town03 的 GT semantic frames
轉成 synthesized 影片，輸出到 pix2pixHD/results/mp4/

Town03: 已有 GT inference 結果，直接拼影片
Town01: 轉 GT label → pix2pixHD inference → 拼影片

Usage:
  conda run -n carla_env python3 make_carla_videos.py
"""

import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
CARLA_DIR    = Path("/home/itriu100/carla")
PIX2PIX_DIR  = CARLA_DIR / "pix2pixHD"
V6_DATASET   = CARLA_DIR / "datasets/training_semantic_v6"
OUT_DIR      = PIX2PIX_DIR / "results/mp4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOWNS = {
    "Town01": {
        "sem_dir":   CARLA_DIR / "datasets/recorded/semantic",
        "gt_label":  V6_DATASET / "test_Town01_gt_label",
        "phase":     "test_Town01_gt",
        "synth_dir": PIX2PIX_DIR / "results_fullres/carla2real_semantic_v6/test_Town01_gt_latest/images",
        "output":    OUT_DIR / "Town01_synthesized.mp4",
    },
    "Town03": {
        "sem_dir":   CARLA_DIR / "datasets/recorded_Town03/semantic",
        "gt_label":  V6_DATASET / "test_Town03_gt_label",
        "phase":     "test_Town03_gt",
        "synth_dir": PIX2PIX_DIR / "results_fullres/carla2real_semantic_v6/test_Town03_gt_latest/images",
        "output":    OUT_DIR / "Town03_synthesized.mp4",
    },
}

FPS = 20

# ── CARLA semantic → trainId label ────────────────────────────────────────────
CARLA_BGR_FILE_TO_ID = {
    (0,0,0):0,(70,70,70):1,(40,40,100):2,(80,90,55):3,(60,20,220):4,
    (153,153,153):5,(50,234,157):6,(128,64,128):7,(232,35,244):8,
    (35,142,107):9,(142,0,0):10,(156,102,102):11,(0,220,220):12,
    (180,130,70):13,(81,0,81):14,(100,100,150):15,(140,150,230):16,
    (180,165,180):17,(30,170,250):18,(160,190,110):19,(50,120,170):20,
    (150,60,45):21,(100,170,145):22,(230,0,0):10,
}
CARLA_TO_CITY19 = {
    0:0,1:2,2:4,3:2,4:11,5:5,6:0,7:0,8:1,9:8,10:13,11:3,
    12:7,13:10,14:0,15:2,16:0,17:3,18:6,19:2,20:13,21:0,22:9,
}

def build_lut():
    lut = {}
    for color, cid in CARLA_BGR_FILE_TO_ID.items():
        lut[color] = CARLA_TO_CITY19.get(cid, 0)
    return lut

def convert_semantic(sem_path, lut):
    arr = np.array(Image.open(sem_path).convert("RGB"))
    pixels = arr.reshape(-1, 3)
    result = np.zeros(len(pixels), dtype=np.uint8)
    for color, city_id in lut.items():
        mask = np.all(pixels == np.array(color, dtype=np.uint8), axis=1)
        result[mask] = city_id
    return Image.fromarray(result.reshape(arr.shape[:2]), mode="L")

def make_gt_labels(sem_dir, label_dir):
    label_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(sem_dir.glob("*.png"))
    todo = [f for f in files if not (label_dir / f.name).exists()]
    if not todo:
        print(f"  Labels already exist ({len(files)} frames), skipping.")
        return
    print(f"  Converting {len(todo)} semantic frames → trainId labels ...")
    lut = build_lut()
    for p in tqdm(todo):
        convert_semantic(p, lut).save(label_dir / p.name)

# ── pix2pixHD inference ───────────────────────────────────────────────────────
def run_inference(phase, n_frames, results_dir):
    synth_dir = results_dir / "carla2real_semantic_v6" / f"{phase}_latest" / "images"
    done = len(list(synth_dir.glob("*synthesized*"))) if synth_dir.exists() else 0
    if done >= n_frames:
        print(f"  Inference already done ({done} frames), skipping.")
        return synth_dir

    print(f"  Running pix2pixHD v6 ({n_frames} frames) ...")
    subprocess.run([
        sys.executable, "test.py",
        "--name",           "carla2real_semantic_v6",
        "--dataroot",       str(V6_DATASET),
        "--label_nc",       "19",
        "--no_instance",
        "--loadSize",       "1024",
        "--resize_or_crop", "scale_width",
        "--phase",          phase,
        "--how_many",       str(n_frames),
        "--results_dir",    str(results_dir),
        "--gpu_ids",        "0",
    ], check=True, cwd=str(PIX2PIX_DIR))
    return synth_dir

# ── stitch frames → video ─────────────────────────────────────────────────────
def stitch_video(synth_dir, output, fps):
    frames = sorted(synth_dir.glob("*synthesized_image.jpg"))
    if not frames:
        raise RuntimeError(f"No frames in {synth_dir}")
    concat = synth_dir / "concat.txt"
    with open(concat, "w") as f:
        for fr in frames:
            f.write(f"file '{fr.resolve()}'\n")
            f.write(f"duration {1/fps:.6f}\n")
    print(f"  Stitching {len(frames)} frames → {output.name} ({fps}fps) ...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(output)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Done → {output}  ({output.stat().st_size/1e6:.1f} MB)")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results_fullres = PIX2PIX_DIR / "results_fullres"
    results_fullres.mkdir(exist_ok=True)

    for town, cfg in TOWNS.items():
        print(f"\n{'='*50}")
        print(f"  {town}")
        print(f"{'='*50}")

        # Step 1: GT label conversion
        print("[1/3] GT label conversion ...")
        make_gt_labels(cfg["sem_dir"], cfg["gt_label"])

        # Step 2: pix2pixHD inference
        print("[2/3] pix2pixHD inference ...")
        n = len(list(cfg["sem_dir"].glob("*.png")))
        synth_dir = run_inference(cfg["phase"], n, results_fullres)

        # Step 3: stitch video
        print("[3/3] Stitch video ...")
        stitch_video(synth_dir, cfg["output"], FPS)

    print(f"\nAll done! Results in {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.mp4")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
