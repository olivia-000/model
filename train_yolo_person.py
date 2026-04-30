from pathlib import Path
import argparse
import random

from ultralytics import YOLO


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def count_labels(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def sample_check(image_dir: Path, label_dir: Path, n: int = 5) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not images:
        print(f"[警告] {image_dir} 沒有找到圖片")
        return

    samples = random.sample(images, min(n, len(images)))
    print(f"\n[抽樣檢查] {image_dir.name}")
    for img_path in samples:
        label_path = label_dir / f"{img_path.stem}.txt"
        print(f"  圖片: {img_path.name} -> 標註: {'存在' if label_path.exists() else '不存在'}")


def validate_dataset(dataset_root: Path) -> None:
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"找不到資料夾: {d}")

    print("\n=== 資料集檢查 ===")
    for split in ["train", "val", "test"]:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        n_img = count_images(image_dir)
        n_lab = count_labels(label_dir)
        print(f"{split:>5} | images: {n_img:<4} labels: {n_lab:<4}")

    sample_check(dataset_root / "images" / "train", dataset_root / "labels" / "train")
    sample_check(dataset_root / "images" / "val", dataset_root / "labels" / "val")
    sample_check(dataset_root / "images" / "test", dataset_root / "labels" / "test")


def build_yaml_text(dataset_root: Path) -> str:
    path_str = dataset_root.as_posix()
    return f"""path: {path_str}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""


def maybe_write_yaml(dataset_root: Path, yaml_path: Path) -> None:
    if yaml_path.exists():
        print(f"\n[資訊] 已存在 YAML，不覆寫: {yaml_path}")
        return

    yaml_text = build_yaml_text(dataset_root)
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\n[完成] 已建立 YAML: {yaml_path}")


def train(args) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    yaml_path = Path(args.yaml_path).resolve()

    print(f"資料集根目錄: {dataset_root}")
    print(f"YAML 路徑: {yaml_path}")

    validate_dataset(dataset_root)
    maybe_write_yaml(dataset_root, yaml_path)

    print("\n=== 開始訓練 ===")
    print(f"模型: {args.model}")
    print(f"epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print(f"device: {args.device}")
    print(f"project: {args.project}")
    print(f"name: {args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        workers=args.workers,
        patience=args.patience,
        exist_ok=True,
    )

    print("\n=== 訓練完成 ===")
    print("請到 runs 目錄查看結果，例如：")
    print(f"{args.project}/{args.name}/weights/best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="訓練 person detection 的 YOLO 腳本")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\person_dataset",
        help="資料集根目錄，裡面要有 images/train、labels/train 等結構",
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\person.yaml",
        help="YOLO 資料集 YAML 輸出位置",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="預訓練模型名稱或本地權重路徑",
    )
    parser.add_argument("--epochs", type=int, default=50, help="訓練輪數")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入影像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU 用 0，CPU 用 cpu")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument(
        "--project",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\runs",
        help="訓練輸出資料夾",
    )
    parser.add_argument("--name", type=str, default="person_yolo_train", help="本次訓練名稱")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
