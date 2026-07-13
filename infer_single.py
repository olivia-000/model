"""
ViPlanner 獨立推論腳本 (不依賴 Isaac Sim / carb / omni.isaac.lab)

用法範例:
    python infer_single.py \
        --model_dir ~/viplanner_models \
        --depth depth.png \
        --semantic semantic_viplanner_colored.png \
        --goal 5.0 0.0 0.0 \
        --output result.png

輸入假設:
  - depth.png: uint16 單通道 PNG,單位是「公釐 (mm)」，對應 model.yaml 裡的 depth_scale=1000.0
             (也支援 .npy 格式，此時假設已經是「公尺 (m)」浮點數)
  - semantic_viplanner_colored.png: 3 通道 RGB PNG，顏色必須是
             viplanner_sem_meta.py 裡定義的精確色碼 (不是你自己模型的原始調色盤!)
             如果你的分割結果還是 Cityscapes/自訂調色盤，先用 semantic_mapping.py 轉換過。
  - goal: (x, y, z) 公尺，定義在「相機/機器人座標系」下 (x 前方, y 左方, z 上方)
          這裡略過 world->camera 的旋轉，直接假設你是在幫單張測試影像手動指定目標點。
"""

import argparse
import os

import cv2
import numpy as np

# NumPy 2.0 移除了一批舊別名 (np.float_, np.complex_, np.int_ ...)，
# 但舊版 wandb (viplanner 依賴的訓練 log 套件，推論階段用不到) 內部大量使用這些別名，
# 會在 import 階段直接炸掉。在 import viplanner 相關模組之前，
# 一次把所有已知會被移除的別名都補回去，避免每次噴一個新的又要重跑一次。
_NUMPY_2_REMOVED_ALIASES = {
    "float_": np.float64,
    "complex_": np.complex128,
    "int_": np.int64,
    "uint": np.uint64,
    "object_": object,
    "bool8": np.bool_,
    "int0": np.intp,
    "uint0": np.uintp,
    "void0": np.void,
    "bytes0": np.bytes_,
    "str0": np.str_,
    "object0": object,
    "longfloat": np.longdouble,
    "singlecomplex": np.complex64,
    "cfloat": np.complex128,
    "longcomplex": np.clongdouble,
    "clongfloat": np.clongdouble,
    "string_": np.bytes_,
    "unicode_": np.str_,
}
for _name, _replacement in _NUMPY_2_REMOVED_ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _replacement)

import torch

from viplanner.config import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.traj_cost_opt.traj_opt import TrajOpt


class VIPlannerInferenceStandalone:
    """VIPlannerAlgo 的獨立版本，拿掉所有 Isaac Sim 相依性。"""

    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda"):
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "model.pt 不存在"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "model.yaml 不存在"

        self.fear_threshold = fear_threshold
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        if self.device == "cpu":
            print("[警告] 使用 CPU 推論，速度會慢很多")

        self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        print(
            f"[模型資訊] sem={self.train_config.sem}, rgb={self.train_config.rgb}, "
            f"knodes={self.train_config.knodes}, in_channel={self.train_config.in_channel}, "
            f"img_input_size={self.train_config.img_input_size}"
        )

        # data_cfg 可能是 list (多個訓練環境各自的設定)，取第一個即可，max_depth 都一樣
        data_cfg = self.train_config.data_cfg
        if isinstance(data_cfg, list):
            data_cfg = data_cfg[0]
        self.max_depth = data_cfg.max_depth
        self.depth_scale = data_cfg.depth_scale
        self.img_input_size = tuple(self.train_config.img_input_size)  # (H, W)

        # 建立網路
        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        try:
            state_dict, _ = torch.load(
                os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True
            )
        except ValueError:
            state_dict = torch.load(
                os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True
            )
        self.net.load_state_dict(state_dict)
        self.net.eval()

        if self.device == "cuda":
            self.net = self.net.cuda()

        self.traj_generate = TrajOpt()

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms as transforms

        resize = transforms.Resize(self.img_input_size, antialias=None)
        return resize(tensor)

    def _depth_input_transform(self, depth: torch.Tensor) -> torch.Tensor:
        """跟 VIPlannerAlgo.input_transformer 完全一致：
        resize -> 超過 max_depth 的設為 0 -> 非有限值 (nan/inf) 設為 0
        注意：這裡不會除以 max_depth 做正規化，官方就是這樣訓練的。"""
        depth = self._resize(depth)
        depth[depth > self.max_depth] = 0.0
        depth[~torch.isfinite(depth)] = 0.0
        return depth

    def load_depth(self, path: str) -> torch.Tensor:
        if path.endswith(".npy"):
            depth_m = np.load(path).astype(np.float32)
        else:
            depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"讀不到深度圖: {path}")
            depth_m = depth_raw.astype(np.float32) / self.depth_scale
        tensor = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor.to(self.device)

    def load_semantic(self, path: str) -> torch.Tensor:
        sem_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if sem_bgr is None:
            raise FileNotFoundError(f"讀不到語義圖: {path}")
        sem_rgb = cv2.cvtColor(sem_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(sem_rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        return tensor.to(self.device)

    def plan(self, depth: torch.Tensor, semantic: torch.Tensor, goal_robot_frame: torch.Tensor):
        """對應 VIPlannerAlgo.plan_dual"""
        semantic_resized = self._resize(semantic) / 255.0
        depth_transformed = self._depth_input_transform(depth)

        with torch.no_grad():
            keypoints, fear = self.net(depth_transformed, semantic_resized, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        return keypoints, traj, fear


def visualize(
    traj: torch.Tensor,
    fear: torch.Tensor,
    goal: torch.Tensor,
    fear_threshold: float,
    output_path: str,
    obstacle_pos: tuple = None,
):
    import matplotlib.pyplot as plt

    # 明確指定中文字型候選清單，避免系統沒裝中文字型時整段文字變成方框 (tofu)。
    # 如果都找不到，matplotlib 會退回預設字型，中文仍可能顯示異常，
    # 這時候到終端機跑: sudo apt install fonts-wqy-zenhei -y
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 避免負號也跟著變方框

    traj_np = traj[0].detach().cpu().numpy()  # (N, 3)
    goal_np = goal[0].detach().cpu().numpy()
    fear_val = float(fear[0].item())
    is_fear = fear_val > fear_threshold

    fig, ax = plt.subplots(figsize=(6, 6))
    color = "red" if is_fear else "green"
    ax.plot(traj_np[:, 1] * -1, traj_np[:, 0], "-o", color=color, markersize=3, label="軌跡")
    ax.scatter([goal_np[1] * -1], [goal_np[0]], marker="*", s=200, color="blue", label="目標點")
    ax.scatter([0], [0], marker="s", s=100, color="black", label="機器人/相機位置")

    if obstacle_pos is not None:
        obs_forward, obs_right = obstacle_pos
        # 軌跡畫圖時用的是 (-y_left, x_forward)；障礙物的「右方距離」換算成 y_left 是負值，
        # 所以畫圖座標的 x 分量 = -y_left = -(-obs_right) = obs_right，直接用 obs_right 即可。
        obs_x_plot = obs_right
        obs_y_plot = obs_forward
        # 用矩形示意大略車身輪廓，尺寸對齊障礙物實際使用的 vehicle.audi.a2
        # (官方大約車寬 1.67m、車長 3.83m)，避免尺寸亂猜導致「有沒有繞開」的判斷失真
        car_width, car_length = 1.67, 3.83
        rect = plt.Rectangle(
            (obs_x_plot - car_width / 2, obs_y_plot - car_length / 2),
            car_width,
            car_length,
            color="darkred",
            alpha=0.4,
            label="障礙物 (概略車身範圍)",
        )
        ax.add_patch(rect)
        ax.scatter([obs_x_plot], [obs_y_plot], marker="x", s=80, color="darkred")
    ax.set_xlabel("← 左  y (m)  右 →")
    ax.set_ylabel("前方 x (m)")
    ax.set_title(f"ViPlanner 軌跡 (俯視圖)\nfear = {fear_val:.3f} (閾值 {fear_threshold}) {'⚠️ 高風險' if is_fear else '✅ 安全'}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[輸出] 軌跡視覺化已存到: {output_path}")
    print(f"[結果] fear = {fear_val:.4f}  {'>>> 高碰撞風險，建議不要執行這條路徑 <<<' if is_fear else '路徑安全'}")


def main():
    parser = argparse.ArgumentParser(description="ViPlanner 單張影像推論")
    parser.add_argument("--model_dir", required=True, help="包含 model.pt 和 model.yaml 的資料夾")
    parser.add_argument("--depth", required=True, help="深度圖路徑 (.png 為 uint16 mm, .npy 為 float 公尺)")
    parser.add_argument("--semantic", required=True, help="ViPlanner 30 類配色的語義圖路徑 (RGB png)")
    parser.add_argument("--goal", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"), help="目標點 (公尺，相機座標系: x前方 y左方 z上方)")
    parser.add_argument("--output", default="viplanner_result.png", help="輸出視覺化圖片路徑")
    parser.add_argument("--save_traj", default=None, help="額外把軌跡原始座標存成 .npy，供疊圖腳本使用")
    parser.add_argument("--fear_threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true", help="強制使用 CPU")
    parser.add_argument(
        "--obstacle_pos",
        nargs=2,
        type=float,
        default=None,
        metavar=("FORWARD", "RIGHT"),
        help="障礙物位置，直接填抓圖時印出的『前方 X m, 右方 Y m』兩個數字，"
        "會自動換算成跟軌跡一致的座標系並畫在俯視圖上。",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    algo = VIPlannerInferenceStandalone(args.model_dir, fear_threshold=args.fear_threshold, device=device)

    depth = algo.load_depth(args.depth)
    semantic = algo.load_semantic(args.semantic)
    goal = torch.tensor([args.goal], dtype=torch.float32, device=algo.device)

    keypoints, traj, fear = algo.plan(depth, semantic, goal)

    print(f"[關鍵點] shape={keypoints.shape}")
    print(f"[軌跡]   shape={traj.shape}")

    visualize(traj, fear, goal, args.fear_threshold, args.output, obstacle_pos=args.obstacle_pos)

    if args.save_traj:
        np.save(args.save_traj, traj[0].detach().cpu().numpy())
        print(f"[輸出] 軌跡原始座標已存到: {args.save_traj} (shape={traj[0].shape}, 座標系: x前方/y左方/z上方)")


if __name__ == "__main__":
    main()
