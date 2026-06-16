"""
對 Cityscapes 2975 張圖用 Mask2Former 產生 trainId label。

輸入：datasets/training_semantic_v3/train_img/
輸出：datasets/cityscapes_m2f_labels/

Usage:
  conda run -n carla_env python3 generate_cityscapes_labels.py
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from tqdm import tqdm

MODEL_NAME = "facebook/mask2former-swin-large-cityscapes-semantic"
BATCH_SIZE = 4

INPUT_DIR = Path("/home/itriu100/carla/datasets/training_semantic_v3/train_img")
OUTPUT_DIR = Path("/home/itriu100/carla/datasets/cityscapes_m2f_labels")


def load_model():
    print(f"Loading {MODEL_NAME}...")
    processor = Mask2FormerImageProcessor.from_pretrained(
        MODEL_NAME, ignore_index=255, do_resize=False
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model on {device}")
    return processor, model, device


@torch.no_grad()
def predict_batch(images, processor, model, device):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    results = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(img.height, img.width) for img in images]
    )
    return [seg.cpu().numpy().astype(np.uint8) for seg in results]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(INPUT_DIR.glob("*.png"))
    todo = [p for p in img_paths if not (OUTPUT_DIR / p.name).exists()]
    print(f"{INPUT_DIR.name}: {len(img_paths)} images, {len(todo)} to process")

    processor, model, device = load_model()

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Cityscapes M2F"):
        batch = todo[i:i + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch]
        labels = predict_batch(images, processor, model, device)
        for path, label in zip(batch, labels):
            # Save at original size (1024×512), same as input
            Image.fromarray(label).save(str(OUTPUT_DIR / path.name))

    total = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"Done: {total} labels → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
