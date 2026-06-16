"""
Video inference pipeline: RGB mp4 → semantic labels → pix2pixHD → synthesized mp4

Steps:
  1. Extract frames from input video (ffmpeg)
  2. Run Mask2Former to generate Cityscapes-19 trainId labels
  3. Run pix2pixHD inference on labels
  4. Combine synthesized frames into output video (ffmpeg)

Usage:
  conda run -n carla_env python3 run_video_inference.py \
    --input  datasets/test_mp4/07.mp4 \
    --output datasets/test_mp4/07_synthesized.mp4 \
    --model  v6

All intermediate files go to datasets/test_mp4/<stem>_work/
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

PIX2PIX_DIR = Path("/home/itriu100/carla/pix2pixHD")
M2F_MODEL   = "facebook/mask2former-swin-large-cityscapes-semantic"
M2F_BATCH   = 4
TARGET_W    = 1024   # TARGET_H computed from input aspect ratio at runtime

MODEL_CFG = {
    "v6": dict(
        name="carla2real_semantic_v6",
        netG="global", ngf=64, n_blocks_global=9,
        n_local_enhancers=1, n_blocks_local=3,
    ),
}


# ── Step 1: extract frames ────────────────────────────────────────────────────

def get_video_info(video: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video)
    ])
    streams = json.loads(out)["streams"]
    for s in streams:
        if s["codec_type"] == "video":
            num, den = s["r_frame_rate"].split("/")
            return {
                "fps":    float(num) / float(den),
                "width":  int(s["width"]),
                "height": int(s["height"]),
            }
    raise RuntimeError("No video stream found")


def extract_frames(video: Path, frames_dir: Path) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if existing:
        print(f"[1/4] Frames already extracted ({len(existing)} frames), skipping.")
        return len(existing)
    print(f"[1/4] Extracting frames from {video.name} ...")
    subprocess.run([
        "ffmpeg", "-i", str(video),
        "-q:v", "2",
        str(frames_dir / "%06d.jpg")
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = len(list(frames_dir.glob("*.jpg")))
    print(f"      {n} frames extracted → {frames_dir}")
    return n


# ── Step 2: Mask2Former ───────────────────────────────────────────────────────

def load_m2f():
    print(f"[2/4] Loading Mask2Former ({M2F_MODEL}) ...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        M2F_MODEL, ignore_index=255, do_resize=False
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_MODEL)
    model.eval().cuda()
    print("      Model loaded on CUDA")
    return processor, model


@torch.no_grad()
def predict_batch(images, processor, model, target_w, target_h):
    resized = [img.resize((target_w, target_h), Image.BILINEAR) for img in images]
    inputs = processor(images=resized, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    results = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(target_h, target_w)] * len(images)
    )
    return [r.cpu().numpy().astype(np.uint8) for r in results]


def run_mask2former(frames_dir: Path, labels_dir: Path, processor, model, target_w, target_h):
    labels_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.jpg"))
    todo = [f for f in frames if not (labels_dir / f.stem).with_suffix(".png").exists()]
    if not todo:
        print(f"[2/4] Labels already exist ({len(frames)} frames), skipping.")
        return
    print(f"[2/4] Running Mask2Former on {len(todo)} frames → {target_w}×{target_h} ...")
    for i in tqdm(range(0, len(todo), M2F_BATCH), unit="batch"):
        batch_paths = todo[i : i + M2F_BATCH]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        labels = predict_batch(images, processor, model, target_w, target_h)
        for path, label in zip(batch_paths, labels):
            Image.fromarray(label, mode="L").save(
                (labels_dir / path.stem).with_suffix(".png")
            )
    print(f"      Labels saved → {labels_dir}")


# ── Step 3: pix2pixHD inference ──────────────────────────────────────────────

def run_pix2pixhd(labels_dir: Path, work_dir: Path, cfg: dict, n_frames: int):
    # pix2pixHD expects dataroot/test_<phase>_label/
    # We symlink labels_dir as dataroot/test_video_label/
    dataroot = work_dir / "dataroot"
    dataroot.mkdir(exist_ok=True)
    link = dataroot / "test_video_label"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(labels_dir.resolve())

    results_dir = work_dir / "results"
    synth_dir = results_dir / cfg["name"] / "test_video_latest" / "images"

    already_done = len(list(synth_dir.glob("*_synthesized_image.jpg"))) if synth_dir.exists() else 0
    if already_done >= n_frames:
        print(f"[3/4] Synthesized frames already exist ({already_done}), skipping.")
        return synth_dir

    print(f"[3/4] Running pix2pixHD ({cfg['name']}) on {n_frames} frames ...")
    cmd = [
        sys.executable, "test.py",
        "--name",            cfg["name"],
        "--dataroot",        str(dataroot),
        "--label_nc",        "19",
        "--no_instance",
        "--netG",            cfg["netG"],
        "--ngf",             str(cfg["ngf"]),
        "--n_blocks_global", str(cfg["n_blocks_global"]),
        "--n_local_enhancers", str(cfg["n_local_enhancers"]),
        "--n_blocks_local",  str(cfg["n_blocks_local"]),
        "--loadSize",        str(TARGET_W),
        "--resize_or_crop",  "scale_width",
        "--phase",           "test_video",
        "--how_many",        str(n_frames),
        "--results_dir",     str(results_dir),
        "--gpu_ids",         "0",
    ]
    subprocess.run(cmd, check=True, cwd=str(PIX2PIX_DIR))
    print(f"      Synthesized frames → {synth_dir}")
    return synth_dir


# ── Step 4: combine frames → video ───────────────────────────────────────────

def combine_frames(synth_dir: Path, output: Path, fps: float, out_w: int, out_h: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    # Sort synthesized images by frame index
    frames = sorted(synth_dir.glob("*_synthesized_image.jpg"))
    if not frames:
        raise RuntimeError(f"No synthesized frames in {synth_dir}")

    # Write a file list for ffmpeg concat
    concat_file = synth_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for fr in frames:
            f.write(f"file '{fr.resolve()}'\n")
            f.write(f"duration {1/fps:.6f}\n")

    print(f"[4/4] Combining {len(frames)} frames → {output.name} ({fps} fps, upscale to {out_w}×{out_h}) ...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-vf", f"fps={fps},scale={out_w}:{out_h}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(output)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size_mb = output.stat().st_size / 1e6
    print(f"      Done! {output}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input mp4 path")
    parser.add_argument("--output", default="", help="Output mp4 path (default: <stem>_synthesized.mp4)")
    parser.add_argument("--model",  default="v6", choices=list(MODEL_CFG.keys()), help="Model version (currently only v6)")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else \
                  input_path.parent / f"{input_path.stem}_synthesized.mp4"
    work_dir    = input_path.parent / f"{input_path.stem}_work"
    frames_dir  = work_dir / "frames"
    labels_dir  = work_dir / "labels"
    cfg         = MODEL_CFG[args.model]

    info    = get_video_info(input_path)
    fps     = info["fps"]
    out_w, out_h = info["width"], info["height"]
    # Inference resolution: scale width to TARGET_W, keep aspect ratio (even height)
    infer_h = int(round(TARGET_W * out_h / out_w / 2) * 2)
    infer_w = TARGET_W

    print(f"\n{'='*55}")
    print(f"  Input : {input_path.name}  ({out_w}×{out_h}, {fps}fps)")
    print(f"  Model : {cfg['name']}")
    print(f"  Infer : {infer_w}×{infer_h}  →  upscale to {out_w}×{out_h}")
    print(f"  Output: {output_path.name}")
    print(f"{'='*55}\n")

    n         = extract_frames(input_path, frames_dir)
    proc, mdl = load_m2f()
    run_mask2former(frames_dir, labels_dir, proc, mdl, infer_w, infer_h)
    del mdl; torch.cuda.empty_cache()
    synth     = run_pix2pixhd(labels_dir, work_dir, cfg, n)
    combine_frames(synth, output_path, fps, out_w, out_h)


if __name__ == "__main__":
    main()
