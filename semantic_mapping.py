"""
Cityscapes-19 -> ViPlanner 30 類 RGB 色彩編碼 轉換工具

用法：
    from semantic_mapping import convert_cityscapes_to_viplanner
    convert_cityscapes_to_viplanner("cityscapes_label.png", "viplanner_semantic.png")

輸入假設：cityscapes_label.png 是單通道 uint8，像素值是 Cityscapes trainId (0-18, 255=ignore)
輸出：3 通道 RGB PNG，顏色對齊 viplanner_sem_meta.py 的 VIPLANNER_SEM_META

⚠️ 這份 mapping 是初稿，務必對照你實際的分割網路輸出類別和
   /home/itriu100/viplanner/viplanner/config/viplanner_sem_meta.py
   的類別名稱，確認每一類的歸類方向是否符合你的場景 (尤其 terrain / road 的邊界)。
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/itriu100/viplanner")  # 依你實際的 repo 路徑調整
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

# Cityscapes-19 trainId -> 類別名稱 (標準順序)
CITYSCAPES_19_NAMES = [
    "road",  # 0
    "sidewalk",  # 1
    "building",  # 2
    "wall",  # 3
    "fence",  # 4
    "pole",  # 5
    "traffic_light",  # 6
    "traffic_sign",  # 7
    "vegetation",  # 8
    "terrain",  # 9
    "sky",  # 10
    "person",  # 11
    "rider",  # 12
    "car",  # 13
    "truck",  # 14
    "bus",  # 15
    "train",  # 16
    "motorcycle",  # 17
    "bicycle",  # 18
]

# Cityscapes 類別名稱 -> ViPlanner 類別名稱 (對照 viplanner_sem_meta.py 裡的 "name" 欄位)
CITYSCAPES_TO_VIPLANNER = {
    "road": "road",
    "sidewalk": "sidewalk",
    "building": "building",
    "wall": "wall",
    "fence": "fence",
    "pole": "pole",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
    "vegetation": "vegetation",
    "terrain": "terrain",
    "sky": "sky",
    "person": "person",
    "rider": "person",  # 騎士當人類障礙處理
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "train": "on_rails",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


def build_lookup_table() -> np.ndarray:
    """建立一個 (256, 3) 的查表，index 是 cityscapes trainId，值是 viplanner RGB"""
    meta_handler = VIPlannerSemMetaHandler()
    lut = np.zeros((256, 3), dtype=np.uint8)

    # 預設 (255 / 未知) 用 static (代價最高的未知類別)
    default_color = meta_handler.class_color["static"]
    lut[:] = default_color

    for train_id, cs_name in enumerate(CITYSCAPES_19_NAMES):
        vip_name = CITYSCAPES_TO_VIPLANNER.get(cs_name)
        if vip_name is None or vip_name not in meta_handler.class_color:
            print(f"[警告] Cityscapes 類別 '{cs_name}' 沒有對應的 ViPlanner 類別，使用預設 static")
            continue
        lut[train_id] = meta_handler.class_color[vip_name]

    return lut


def convert_cityscapes_to_viplanner(label_path: str, output_path: str):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f"讀不到分割標籤圖: {label_path}")

    lut = build_lookup_table()
    rgb = lut[label]  # (H, W, 3), RGB 順序

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)
    print(f"[輸出] ViPlanner 語義圖已存到: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cityscapes trainId 標籤圖路徑")
    parser.add_argument("--output", required=True, help="輸出的 ViPlanner 語義圖路徑")
    args = parser.parse_args()
    convert_cityscapes_to_viplanner(args.input, args.output)
