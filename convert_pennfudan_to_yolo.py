import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ====== 你的 INRIA 路徑 ======
SRC_ROOT = Path(r"C:\Users\user\Desktop\data\INRIAPerson")
DST_ROOT = Path(r"C:\Users\user\Desktop\data\person_dataset_inria")

TRAIN_XML_DIR = SRC_ROOT / "Train" / "Annotations"
TRAIN_IMG_DIR = SRC_ROOT / "Train" / "JPEGImages"

TEST_XML_DIR = SRC_ROOT / "Test" / "Annotations"
TEST_IMG_DIR = SRC_ROOT / "Test" / "JPEGImages"

# 類別只保留 person
CLASS_ID = 0

# 把 Train 再切成 train / val
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

SEED = 42


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    box_w = xmax - xmin
    box_h = ymax - ymin
    x_center = xmin + box_w / 2
    y_center = ymin + box_h / 2

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def parse_voc_xml(xml_path):
    """
    讀 Pascal VOC XML
    回傳:
    - filename
    - width
    - height
    - boxes: [(xmin, ymin, xmax, ymax), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_tag = root.find("filename")
    filename = filename_tag.text.strip() if filename_tag is not None else None

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path} 沒有 <size> 標籤")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        obj_name = name_tag.text.strip().lower() if name_tag is not None else "person"

        # 只保留 person
        if obj_name != "person":
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append((xmin, ymin, xmax, ymax))

    return filename, width, height, boxes


def find_image_by_stem(img_dir, stem):
    """
    根據 XML 檔名去找對應圖片
    優先找 png，再找 jpg/jpeg
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = img_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def convert_one(xml_path, img_dir, split):
    filename, img_w, img_h, boxes = parse_voc_xml(xml_path)

    # 優先用 XML 裡的 filename；沒有的話就用 xml 檔名去找
    if filename:
        stem = Path(filename).stem
    else:
        stem = xml_path.stem

    img_path = find_image_by_stem(img_dir, stem)
    if img_path is None:
        print(f"[跳過] 找不到對應圖片: {xml_path.name}")
        return

    # 複製圖片
    dst_img_path = DST_ROOT / "images" / split / img_path.name
    shutil.copy2(img_path, dst_img_path)

    # 寫 label
    dst_label_path = DST_ROOT / "labels" / split / f"{img_path.stem}.txt"
    lines = []

    for xmin, ymin, xmax, ymax in boxes:
        x_c, y_c, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    dst_label_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()

    train_xmls = sorted(TRAIN_XML_DIR.glob("*.xml"))
    test_xmls = sorted(TEST_XML_DIR.glob("*.xml"))

    if not train_xmls:
        raise FileNotFoundError(f"在 {TRAIN_XML_DIR} 找不到 XML")
    if not test_xmls:
        raise FileNotFoundError(f"在 {TEST_XML_DIR} 找不到 XML")

    print("TRAIN_XML_DIR =", TRAIN_XML_DIR)
    print("Exists? ", TRAIN_XML_DIR.exists())
    print("TEST_XML_DIR  =", TEST_XML_DIR)
    print("Exists? ", TEST_XML_DIR.exists())

    # Train 再切 train / val
    random.seed(SEED)
    random.shuffle(train_xmls)

    n = len(train_xmls)
    n_train = int(n * TRAIN_RATIO)

    train_split = train_xmls[:n_train]
    val_split = train_xmls[n_train:]

    print(f"\nTrain XML 總數: {len(train_xmls)}")
    print(f"切成 train: {len(train_split)}")
    print(f"切成 val:   {len(val_split)}")
    print(f"Test XML 總數:  {len(test_xmls)}")

    # 處理 train
    for xml_path in train_split:
        convert_one(xml_path, TRAIN_IMG_DIR, "train")

    # 處理 val
    for xml_path in val_split:
        convert_one(xml_path, TRAIN_IMG_DIR, "val")

    # 處理 test
    for xml_path in test_xmls:
        convert_one(xml_path, TEST_IMG_DIR, "test")

    print("\nINRIA 轉換完成。")


if __name__ == "__main__":
    main()
