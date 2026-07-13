"""
把 ViPlanner 算出的軌跡，用簡單的透視投影畫回原始 RGB 照片上，
並排列出 [原始照片 | 語義配色圖 | 疊圖後的軌跡] 三張圖方便比較。

用法：
    python overlay_on_image.py \
        --rgb ./test_data/rgb_debug.png \
        --semantic ./test_data/sem_viplanner.png \
        --traj ./test_data/traj.npy \
        --output ./test_data/compare.png \
        --fov 90

⚠️ 這是簡化版透視投影，只用水平 FOV 反推焦距，沒有處理相機俯仰角 (camera_tilt) 造成的
   微幅垂直偏移，畫出來的路徑會「大致準確」但不是像素級精準對齊，純粹是為了讓你
   有更直覺的感受：模型算的軌跡實際落在畫面裡的哪個位置。
"""

import argparse

import cv2
import numpy as np


def project_trajectory(
    traj: np.ndarray, width: int, height: int, fov_deg: float, camera_tilt_rad: float = 0.0
) -> np.ndarray:
    """
    把軌跡從「機器人座標系 (x前方, y左方, z上方)」投影到影像像素座標。

    camera_tilt_rad: 訓練時相機的俯仰角 (低頭為正值)，如果你抓圖時的相機是水平拍攝
        (沒有跟著低頭)，這裡填 model.yaml 的 camera_tilt 值 (預設訓練值約 0.15 rad)，
        會把軌跡點先旋轉回「水平相機」看到的座標，再做透視投影。

    Returns:
        (N, 2) 陣列，每一列是 (u, v) 像素座標；forward<=0 的點（在相機後方）會被過濾掉。
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = (width / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0

    x_fwd = traj[:, 0]
    y_left = traj[:, 1]
    z_up = traj[:, 2]

    if camera_tilt_rad != 0.0:
        # 訓練用相機是低頭看的，軌跡座標系是相對於那個「低頭」的光軸。
        # 如果你抓圖用的是水平相機，要把軌跡點繞左右軸(y軸)轉回水平座標系再投影。
        c, s = np.cos(camera_tilt_rad), np.sin(camera_tilt_rad)
        x_fwd, z_up = x_fwd * c - z_up * s, x_fwd * s + z_up * c

    valid = x_fwd > 0.1  # 避免除以零或除以負數（相機後方的點沒有意義）
    x_fwd, y_left, z_up = x_fwd[valid], y_left[valid], z_up[valid]

    # 進一步過濾掉超出鏡頭水平視野角度的點：
    # 如果某個點的橫向角度已經超過鏡頭半視角，代表這段路徑根本不在畫面裡，
    # 硬算出來的像素座標會離譜爆走（除以接近 0 的 x_fwd，或角度本身就超出 FOV）。
    half_fov_rad = fov_rad / 2.0
    angle = np.arctan2(np.abs(y_left), x_fwd)
    in_fov = angle < (half_fov_rad * 0.95)  # 留一點餘裕，避免貼著邊緣的點也被誤判
    x_fwd, y_left, z_up = x_fwd[in_fov], y_left[in_fov], z_up[in_fov]

    # 機器人座標 -> 相機/影像座標: 相機看出去 X右 Y下 Z前
    u = cx - fx * (y_left / x_fwd)  # y左方 -> 像素往左是負方向，所以用減號
    v = cy - fy * (z_up / x_fwd)  # z上方 -> 像素往上是負方向，所以用減號

    pts = np.stack([u, v], axis=1)
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--traj", required=True, help="infer_single.py --save_traj 存出來的 .npy")
    parser.add_argument("--output", default="compare.png")
    parser.add_argument("--fov", type=float, default=90.0, help="相機水平視角 (度)，要跟抓圖時的設定一致")
    parser.add_argument(
        "--camera_tilt",
        type=float,
        default=0.0,
        help="訓練相機的俯仰角(弧度)，如果抓圖相機是水平的，填 0.15 (model.yaml 的值) 做校正；"
        "如果抓圖時相機已經用 --pitch_deg 低頭對齊過，這裡保持 0 即可。",
    )
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    sem = cv2.imread(args.semantic, cv2.IMREAD_COLOR)
    traj = np.load(args.traj)

    height, width = rgb.shape[:2]
    pixel_pts = project_trajectory(traj, width, height, args.fov, args.camera_tilt)

    rgb_overlay = rgb.copy()
    pts_int = pixel_pts.astype(int)
    for i in range(len(pts_int) - 1):
        p1, p2 = tuple(pts_int[i]), tuple(pts_int[i + 1])
        cv2.line(rgb_overlay, p1, p2, (0, 255, 0), 6)  # 綠色軌跡線 (加粗)
    for p in pts_int:
        cv2.circle(rgb_overlay, tuple(p), 7, (0, 0, 255), -1)  # 紅色關鍵點 (加大)

    # 三張圖並排：原圖 | 語義配色圖 | 疊圖
    sem_resized = cv2.resize(sem, (width, height))
    combined = np.hstack([rgb, sem_resized, rgb_overlay])

    # 加文字標籤
    labels = ["1. 你看到的 (RGB)", "2. 模型看到的 (語義配色)", "3. 模型算出的軌跡 (疊圖)"]
    for i, label in enumerate(labels):
        x_offset = i * width + 10
        cv2.putText(combined, label, (x_offset, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, label, (x_offset, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imwrite(args.output, combined)
    print(f"[輸出] 三圖並排比較已存到: {args.output}")
    print(f"[提醒] 軌跡點超出畫面範圍是正常的，代表那段路徑在相機視野之外")


if __name__ == "__main__":
    main()
