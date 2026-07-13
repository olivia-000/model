"""
比對「CARLA ground truth 語義圖」vs「全景分割模型預測語義圖」的工具。

在 viplanner env 執行（只需要 numpy/cv2/matplotlib，不需要 mmdet）。

量化目的：用真實分割模型取代 ground truth 之後，語義輸入的誤差有多大？
這個誤差就是之後接真實機器狗時必然會遇到的雜訊來源，先在 CARLA 裡量出來。

用法:
    python compare_semantics.py \
        --gt   ../test_scene2_obstacle/sem_viplanner.png \
        --pred ../test_scene2_obstacle/sem_predicted.png \
        --output compare_result.png

    # 選配：同場景兩條軌跡（分別用 GT/預測語義圖跑 infer_single.py --save_traj 存的 .npy）
    python compare_semantics.py --gt ... --pred ... \
        --traj_gt traj_gt.npy --traj_pred traj_pred.npy
"""

import argparse
import os
import sys

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # repo 根目錄，讓 viplanner 可直接 import

from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler  # noqa: E402


def color_to_class_id(sem_rgb: np.ndarray, meta: VIPlannerSemMetaHandler) -> np.ndarray:
    """把 (H,W,3) 色碼圖轉回 (H,W) 類別 id 圖；不在 34 類色碼內的像素標 -1。"""
    class_map = np.full(sem_rgb.shape[:2], -1, dtype=np.int32)
    for name, color in meta.class_color.items():
        mask = np.all(sem_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_map[mask] = meta.class_id[name]
    return class_map


def main():
    parser = argparse.ArgumentParser(description="GT vs 預測語義圖比對")
    parser.add_argument("--gt", required=True, help="ground truth 語義圖（34 類色碼 RGB PNG）")
    parser.add_argument("--pred", required=True, help="模型預測語義圖（同格式）")
    parser.add_argument("--output", default="compare_result.png", help="差異視覺化輸出路徑")
    parser.add_argument("--traj_gt", default=None, help="用 GT 語義圖跑出的軌跡 .npy（選配）")
    parser.add_argument("--traj_pred", default=None, help="用預測語義圖跑出的軌跡 .npy（選配）")
    args = parser.parse_args()

    gt_bgr = cv2.imread(args.gt, cv2.IMREAD_COLOR)
    pred_bgr = cv2.imread(args.pred, cv2.IMREAD_COLOR)
    assert gt_bgr is not None, f"讀不到 {args.gt}"
    assert pred_bgr is not None, f"讀不到 {args.pred}"

    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
    if gt_rgb.shape != pred_rgb.shape:
        # 預測圖解析度可能跟 GT 不同（分割模型輸出跟輸入同解析度），以 GT 為準做最近鄰縮放，
        # 用 INTER_NEAREST 避免把離散色碼內插出不存在的顏色
        pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    meta = VIPlannerSemMetaHandler()
    gt_ids = color_to_class_id(gt_rgb, meta)
    pred_ids = color_to_class_id(pred_rgb, meta)

    valid = gt_ids >= 0
    agree = (gt_ids == pred_ids) & valid
    pixel_acc = agree.sum() / valid.sum()
    print(f"[整體] 像素類別一致率: {pixel_acc*100:.2f}%  （有效像素 {valid.sum()}）")

    # 「loss 等級一致率」比類別一致率更貼近規劃行為：
    # 網路真正在意的是 cost（loss 權重），person 和 vehicle 都是 OBSTACLE_LOSS=2.0，
    # 彼此認錯對軌跡幾乎沒影響；sidewalk(0) 認成 road(1.5) 才會真的改變路徑。
    id_to_loss = {meta.class_id[name]: meta.class_loss[name] for name in meta.class_loss}
    loss_lut = np.zeros(max(id_to_loss) + 1, dtype=np.float32)
    for cid, loss in id_to_loss.items():
        loss_lut[cid] = loss
    gt_loss = np.where(valid, loss_lut[np.clip(gt_ids, 0, None)], np.nan)
    pred_loss = np.where(pred_ids >= 0, loss_lut[np.clip(pred_ids, 0, None)], np.nan)
    loss_agree = (gt_loss == pred_loss) & valid & (pred_ids >= 0)
    print(f"[整體] loss 等級一致率: {loss_agree.sum()/valid.sum()*100:.2f}%")

    # 逐類統計（GT 出現過的類別）
    id_to_name = {v: k for k, v in meta.class_id.items()}
    print(f"\n{'類別':<20}{'GT像素數':>10}{'一致率':>10}")
    for cid in sorted(np.unique(gt_ids[valid])):
        cls_mask = gt_ids == cid
        acc = (pred_ids[cls_mask] == cid).mean()
        print(f"{id_to_name.get(cid, str(cid)):<20}{cls_mask.sum():>10}{acc*100:>9.1f}%")

    # 視覺化：GT | 預測 | 差異熱圖
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("全景分割預測")
    diff_vis = np.zeros_like(gt_rgb)
    diff_vis[~agree & valid] = [255, 0, 0]
    diff_vis[agree] = [220, 220, 220]
    axes[2].imshow(diff_vis)
    axes[2].set_title(f"差異（紅=不一致）類別一致率 {pixel_acc*100:.1f}%")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=130)
    print(f"\n[輸出] 差異視覺化 → {args.output}")

    # 軌跡漂移比較（選配）
    if args.traj_gt and args.traj_pred:
        traj_gt = np.load(args.traj_gt)  # (N,3) x前/y左/z上
        traj_pred = np.load(args.traj_pred)
        n = min(len(traj_gt), len(traj_pred))
        dev = np.linalg.norm(traj_gt[:n, :2] - traj_pred[:n, :2], axis=1)
        print(f"[軌跡] 平均偏移 {dev.mean():.3f} m，最大偏移 {dev.max():.3f} m，終點偏移 {dev[-1]:.3f} m")


if __name__ == "__main__":
    main()
