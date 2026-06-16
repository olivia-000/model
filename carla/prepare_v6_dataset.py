"""
建立 v6 訓練資料集：Pure NuRec × Mask2Former labels
  train_img/   → symlink NuRec 圖（nurec/ + nurec_raw/）
  train_label/ → symlink nurec_labels/ 對應 label
  test_label/  → symlink nurec_test_labels/test_Town01/
  test_Town03_label/ → symlink nurec_test_labels/test_Town03/

只有 train_img 和 train_label 都存在的 pair 才納入。

Usage:
  conda run -n carla_env python3 prepare_v6_dataset.py
"""

from pathlib import Path

NUREC_DIRS = [
    Path("/home/itriu100/carla/datasets/nurec"),
    Path("/home/itriu100/carla/datasets/nurec_raw"),
]
LABEL_DIR = Path("/home/itriu100/carla/datasets/nurec_labels")

# Cityscapes with Mask2Former labels (urban/building diversity)
CITYSCAPES_IMG_DIR   = Path("/home/itriu100/carla/datasets/training_semantic_v3/train_img")
CITYSCAPES_LABEL_DIR = Path("/home/itriu100/carla/datasets/cityscapes_m2f_labels")

TEST_LABEL_DIRS = {
    "test_label": Path("/home/itriu100/carla/datasets/nurec_test_labels/test_Town01"),
    "test_Town03_label": Path("/home/itriu100/carla/datasets/nurec_test_labels/test_Town03"),
}
OUT_DIR = Path("/home/itriu100/carla/datasets/training_semantic_v6")


def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main():
    (OUT_DIR / "train_img").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train_label").mkdir(parents=True, exist_ok=True)

    pairs = 0

    # NuRec images + Mask2Former labels
    for img_dir in NUREC_DIRS:
        for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            label_path = (LABEL_DIR / img_path.stem).with_suffix(".png")
            if not label_path.exists():
                continue
            dst_img   = OUT_DIR / "train_img"   / img_path.name
            dst_label = OUT_DIR / "train_label" / label_path.name
            link(img_path.resolve(),   dst_img)
            link(label_path.resolve(), dst_label)
            pairs += 1
    print(f"NuRec pairs: {pairs}")

    # Cityscapes images + Mask2Former labels (prefix cs_ to avoid filename collision)
    cs_pairs = 0
    if CITYSCAPES_IMG_DIR.exists() and CITYSCAPES_LABEL_DIR.exists():
        for img_path in sorted(CITYSCAPES_IMG_DIR.glob("*.png")):
            label_path = CITYSCAPES_LABEL_DIR / img_path.name
            if not label_path.exists():
                continue
            dst_img   = OUT_DIR / "train_img"   / f"cs_{img_path.name}"
            dst_label = OUT_DIR / "train_label" / f"cs_{img_path.name}"
            link(img_path.resolve(),   dst_img)
            link(label_path.resolve(), dst_label)
            cs_pairs += 1
    print(f"Cityscapes pairs: {cs_pairs}")
    pairs += cs_pairs

    print(f"Total pairs: {pairs}")

    # Test label symlinks (directory-level)
    for subdir, src in TEST_LABEL_DIRS.items():
        dst = OUT_DIR / subdir
        if dst.exists() or dst.is_symlink():
            import shutil
            if dst.is_symlink():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if src.exists():
            dst.symlink_to(src.resolve())
            print(f"{subdir}: linked → {src} ({len(list(src.glob('*.png')))} files)")
        else:
            print(f"[SKIP] {subdir}: {src} not found yet")

    print(f"\nDataset ready: {OUT_DIR}")
    print(f"  train pairs : {pairs}")


if __name__ == "__main__":
    main()
