"""
全景分割獨立推論腳本（階段1+2：RGB → ViPlanner 34 類色碼語義圖）

在 mask2former_env（mmdetection 環境）裡執行，「不需要」安裝 viplanner 套件——
viplanner_sem_meta.py / coco_sem_meta.py 這兩個純 Python 設定檔會用 importlib
直接從檔案路徑載入，避免把整個 viplanner（torch/wandb 相依）拖進 mmdet 環境。

用法範例（單張）:
    conda activate mask2former_env
    python panoptic_inference.py \
        --input  ../test_scene2_obstacle/rgb_debug.png \
        --output ../test_scene2_obstacle/sem_predicted.png \
        --config     ./m2f_ckpt/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py \
        --checkpoint ./m2f_ckpt/mask2former_r50_8xb2-lsj-50e_coco-panoptic_xxxx.pth

用法範例（整個資料夾）:
    python panoptic_inference.py --input ./rgb_dir --output ./sem_dir --config ... --checkpoint ...

輸出格式:
    (H, W, 3) uint8 RGB PNG，色碼精確對應 viplanner_sem_meta.py 的 34 類定義，
    與 CARLA ground truth 流程產生的 sem_viplanner.png 完全同格式、可互換。
"""

import argparse
import glob
import importlib.util
import os
import time

import cv2
import numpy as np

# mmdetection（必須是 mmdet 3.x，2.x 沒有 mmdet.evaluation 子模組）
from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation import INSTANCE_OFFSET

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VIPLANNER_ROOT = os.path.dirname(_THIS_DIR)  # repo 根目錄


def _load_module_from_path(module_name: str, file_path: str):
    """直接從檔案路徑載入模組，繞過 viplanner/config/__init__.py（那會拖進其他相依）。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PanopticSegmenter:
    """把 ros/planner/src/m2f_inference.py 的核心邏輯去 ROS 化。

    原理：
      1. inference_detector 回傳的 pred_panoptic_seg.sem_seg 是 (1,H,W) 編碼圖，
         每像素 = category_id * INSTANCE_OFFSET + instance_id（mmdet 標準全景編碼）。
      2. ViPlanner 只需要語義類別，% INSTANCE_OFFSET 把 instance_id 丟掉。
      3. get_class_for_id_mmdet() 依「這顆模型自己的類別順序」建 COCO→ViPlanner 34 類映射
         （換 checkpoint / mmdet 版本時映射表會自動重建，這正是不能寫死常數表的原因）。
      4. 用 VIPlannerSemMetaHandler 的 34 類色碼上色成 RGB 圖。
    """

    def __init__(self, config_file: str, checkpoint_file: str, viplanner_root: str, device: str = "cuda:0"):
        cfg_dir = os.path.join(viplanner_root, "viplanner", "config")
        sem_meta_mod = _load_module_from_path("viplanner_sem_meta", os.path.join(cfg_dir, "viplanner_sem_meta.py"))
        coco_meta_mod = _load_module_from_path("coco_sem_meta", os.path.join(cfg_dir, "coco_sem_meta.py"))

        print(f"[載入] mmdet 模型: {config_file}")
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.num_classes = len(self.model.dataset_meta["classes"])

        viplanner_meta = sem_meta_mod.VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = coco_meta_mod.get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.static_color = viplanner_meta.class_color["static"]
        self.coco_viplanner_color_mapping = {
            coco_id: viplanner_meta.class_color[cls_name] for coco_id, cls_name in coco_viplanner_cls_mapping.items()
        }
        print(f"[映射] COCO {self.num_classes} 類 → ViPlanner 34 類，成功映射 {len(self.coco_viplanner_color_mapping)} 類")

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """輸入 BGR 影像（OpenCV 慣例，任意解析度），回傳 (H,W,3) uint8 RGB 34 類色碼圖。"""
        result = inference_detector(self.model, image_bgr)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]  # (H, W)

        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            curr_label = curr_sem_class % INSTANCE_OFFSET  # 丟棄 instance_id，只留 category_id
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_label]
            except KeyError:
                # category_id == num_classes 是 COCO 的 void/unlabeled，其餘才是真的沒映射到
                if curr_sem_class != self.num_classes:
                    print(f"[警告] COCO 類別 {curr_label} 不在映射表內，回退成 static")
                panoptic_mask[result == curr_sem_class] = self.static_color
        return panoptic_mask


def main():
    parser = argparse.ArgumentParser(description="RGB → ViPlanner 34 類色碼語義圖（mmdet Mask2Former 全景分割）")
    parser.add_argument("--input", required=True, help="RGB 影像路徑，或包含影像的資料夾")
    parser.add_argument("--output", required=True, help="輸出 PNG 路徑（input 是資料夾時，此處也要是資料夾）")
    parser.add_argument("--config", required=True, help="mmdet config .py（例如 mask2former_r50_8xb2-lsj-50e_coco-panoptic.py）")
    parser.add_argument("--checkpoint", required=True, help="mmdet checkpoint .pth")
    parser.add_argument("--viplanner_root", default=_DEFAULT_VIPLANNER_ROOT, help="viplanner repo 根目錄（讀取色碼設定檔用）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overlay", action="store_true", help="額外輸出半透明疊圖（*_overlay.png），方便肉眼檢查")
    args = parser.parse_args()

    segmenter = PanopticSegmenter(args.config, args.checkpoint, args.viplanner_root, args.device)

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        image_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg") for p in glob.glob(os.path.join(args.input, ext))
        )
        out_paths = [os.path.join(args.output, os.path.splitext(os.path.basename(p))[0] + "_sem.png") for p in image_paths]
    else:
        image_paths = [args.input]
        out_paths = [args.output]
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for img_path, out_path in zip(image_paths, out_paths):
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[跳過] 讀不到影像: {img_path}")
            continue
        t0 = time.time()
        sem_rgb = segmenter.predict(image_bgr)
        dt = time.time() - t0
        cv2.imwrite(out_path, cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))
        print(f"[完成] {img_path} → {out_path}  ({dt*1000:.0f} ms)")

        if args.overlay:
            overlay = cv2.addWeighted(image_bgr, 0.5, cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR), 0.5, 0)
            overlay_path = os.path.splitext(out_path)[0] + "_overlay.png"
            cv2.imwrite(overlay_path, overlay)
            print(f"[完成] 疊圖 → {overlay_path}")


if __name__ == "__main__":
    main()
