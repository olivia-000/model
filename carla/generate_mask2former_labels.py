"""
對所有 NuRec 圖用 Mask2Former 產生 Cityscapes 19-class trainId label。

輸入：
  datasets/nurec/        (7722 張，現有)
  datasets/nurec_raw/    (23053 張，新抽)
輸出：
  datasets/nurec_labels/ (對應每張圖的 label PNG，灰階 trainId 0-18)

Usage:
  conda run -n carla_env python3 generate_mask2former_labels.py
  conda run -n carla_env python3 generate_mask2former_labels.py --test-only  # 只處理 test
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from tqdm import tqdm

MODEL_NAME = "facebook/mask2former-swin-large-cityscapes-semantic"
BATCH_SIZE = 4
TARGET_W, TARGET_H = 1024, 512

INPUT_DIRS = [
    Path("/home/itriu100/carla/datasets/nurec"),      # 7722 existing
    Path("/home/itriu100/carla/datasets/nurec_raw"),  # 23053 new
]
LABEL_DIR = Path("/home/itriu100/carla/datasets/nurec_labels")

TEST_DIRS = {
    "test_Town03": Path("/home/itriu100/carla/datasets/training_semantic_v2/test_Town03_label"),
    "test_Town01": Path("/home/itriu100/carla/datasets/training_semantic_v2/test_label"),
}
TEST_RGB_DIRS = {
    "test_Town03": Path("/home/itriu100/carla/datasets/recorded_Town03/rgb"),
    "test_Town01": Path("/home/itriu100/carla/datasets/recorded/rgb"),
}
TEST_OUT_DIR = Path("/home/itriu100/carla/datasets/nurec_test_labels")


def load_model():
    print(f"Loading {MODEL_NAME}...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_NAME, ignore_index=255, do_resize=False
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    return processor, model, device


@torch.no_grad()
def predict_batch(images: list, processor, model, device) -> list:
    """images: list of PIL Image. Returns list of numpy H×W uint8 trainId arrays."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Post-process: returns list of {segmentation: tensor H×W, segments_info: [...]}
    results = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(img.height, img.width) for img in images]
    )
    labels = []
    for seg in results:
        arr = seg.cpu().numpy().astype(np.uint8)
        labels.append(arr)
    return labels


def process_dir(img_dir: Path, out_dir: Path, processor, model, device,
                resize_output: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))

    # Skip already processed
    todo = [p for p in img_paths if not (out_dir / p.name).with_suffix(".png").exists()]
    print(f"\n{img_dir.name}: {len(img_paths)} images, {len(todo)} to process")

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc=img_dir.name):
        batch_paths = todo[i:i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        labels = predict_batch(images, processor, model, device)

        for path, label in zip(batch_paths, labels):
            out_path = (out_dir / path.stem).with_suffix(".png")
            if resize_output:
                lbl_img = Image.fromarray(label).resize(
                    (TARGET_W, TARGET_H), Image.NEAREST
                )
            else:
                lbl_img = Image.fromarray(label)
            lbl_img.save(str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true",
                        help="Only regenerate test labels (Town03/Town01)")
    args = parser.parse_args()

    processor, model, device = load_model()

    if not args.test_only:
        for img_dir in INPUT_DIRS:
            if not img_dir.exists():
                print(f"[SKIP] {img_dir} not found")
                continue
            process_dir(img_dir, LABEL_DIR, processor, model, device)

        total = len(list(LABEL_DIR.glob("*.png")))
        print(f"\nNuRec labels done: {total} files in {LABEL_DIR}")

    # Test labels: run on CARLA RGB (Town03 / Town01)
    print("\n--- Regenerating test labels with Mask2Former ---")
    for name, rgb_dir in TEST_RGB_DIRS.items():
        if not rgb_dir.exists():
            print(f"[SKIP] {rgb_dir} not found")
            continue
        out_dir = TEST_OUT_DIR / name
        process_dir(rgb_dir, out_dir, processor, model, device)
        print(f"{name}: {len(list(out_dir.glob('*.png')))} labels saved")

    print("\nAll done.")


if __name__ == "__main__":
    main()
