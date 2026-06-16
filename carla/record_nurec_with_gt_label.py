"""
NuRec Replay + CARLA Semantic GT 雙軌同步錄製腳本

同時捕捉：
  - NuRec 逼真 RGB（gRPC render_rgb）
  - CARLA 語意 GT label（掛載 semantic_segmentation sensor）

輸出格式與 training_semantic_v2 相同（1024×512 PNG，Cityscapes 19 trainId）。

Usage (必須從 nurec 目錄執行):
  cd /home/itriu100/carla/simulator/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec
  conda run -n carla_env python3 /home/itriu100/carla/record_nurec_with_gt_label.py \
    --usdz /path/to/scene.usdz \
    --output /home/itriu100/carla/datasets/nurec_gt \
    --max-frames 600
"""

import argparse
import os
import sys
import threading
import logging
import numpy as np
from pathlib import Path
from PIL import Image

# NuRec modules use relative paths for JSON files — must run from nurec directory
NUREC_DIR = Path(__file__).parent / "simulator/CARLA_0.9.16/PythonAPI/examples/nvidia/nurec"
os.chdir(NUREC_DIR)
sys.path.insert(0, str(NUREC_DIR))

import carla
from nurec_integration import NurecScenario
from constants import EGO_TRACK_ID


def make_transform_matrix(rotation=None, translation=None):
    mat = np.eye(4, dtype=float)
    if rotation is not None:
        pitch_deg, yaw_deg, roll_deg = rotation
        yaw   = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        roll  = np.radians(roll_deg)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0,            0,            1]])
        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                        [0,             1, 0            ],
                        [-np.sin(pitch),0, np.cos(pitch)]])
        Rx = np.array([[1, 0,           0          ],
                        [0, np.cos(roll),-np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_unreal = Rz @ Ry @ Rx
        A = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
        mat[:3, :3] = R_unreal @ A
    if translation is not None:
        mat[:3, 3] = translation
    return mat

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("nurec_gt_recorder")

# CARLA semantic tag → Cityscapes trainId (255 → clamped to 0 = road)
CARLA_TAG_TO_TRAINID = np.array([
    0,    # 0  Unlabeled    → road (fallback)
    2,    # 1  Building     → building
    4,    # 2  Fence        → fence
    255,  # 3  Other        → ignore → 0
    11,   # 4  Pedestrian   → person
    5,    # 5  Pole         → pole
    0,    # 6  RoadLine     → road
    0,    # 7  Road         → road
    1,    # 8  SideWalk     → sidewalk
    8,    # 9  Vegetation   → vegetation
    13,   # 10 Vehicles     → car
    3,    # 11 Wall         → wall
    7,    # 12 TrafficSign  → traffic sign
    10,   # 13 Sky          → sky
    9,    # 14 Ground       → terrain
    2,    # 15 Bridge       → building
    255,  # 16 RailTrack    → ignore → 0
    4,    # 17 GuardRail    → fence
    6,    # 18 TrafficLight → traffic light
    255,  # 19 Static       → ignore → 0
    255,  # 20 Dynamic      → ignore → 0
    255,  # 21 Water        → ignore → 0
    9,    # 22 Terrain      → terrain
], dtype=np.uint8)

TARGET_W, TARGET_H = 1024, 512
NUREC_RESOLUTION_RATIO = 0.5   # 3848 × 0.5 ≈ 1924 px wide

# Front-center camera matching carla_example_camera_config.yaml
NUREC_CAM_PARAMS = {
    "camera_type": "ftheta",
    "logical_id": "camera_front_wide_120fov",
    "resolution_w": 3848,
    "resolution_h": 2168,
    "principal_point_x": 1927.038818,
    "principal_point_y": 1099.315796,
    "pixeldist_to_angle_poly": [
        0.0, 0.00053944590035826, 2.46685849525363e-09,
        2.6309770334576e-12, -2.44043479933324e-16,
    ],
    "reference_poly": 1,
    "angle_to_pixeldist_poly": [],
    "max_angle": np.pi,
}
NUREC_CAM_TRANSFORM = make_transform_matrix(
    rotation=[0, 0, 0],       # pitch, yaw, roll (degrees)
    translation=[0, 0, 1.47], # x, y, z (metres, forward-facing at 1.47m height)
)


def carla_semantic_to_trainid(image: carla.Image) -> np.ndarray:
    """CARLA semantic camera raw bytes → Cityscapes 19-class trainId (H×W uint8)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    tag = arr[:, :, 2]  # BGRA → Red channel = semantic tag
    tag = np.clip(tag, 0, len(CARLA_TAG_TO_TRAINID) - 1)
    trainid = CARLA_TAG_TO_TRAINID[tag]
    trainid[trainid == 255] = 0
    return trainid


def save_pair(rgb: np.ndarray, label: np.ndarray, out_img: Path, out_lbl: Path) -> None:
    Image.fromarray(rgb).resize((TARGET_W, TARGET_H), Image.LANCZOS).save(str(out_img))
    Image.fromarray(label).resize((TARGET_W, TARGET_H), Image.NEAREST).save(str(out_lbl))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--nurec-port", default=46435, type=int)
    parser.add_argument("-u", "--usdz", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--max-frames", default=600, type=int)
    parser.add_argument("--fps", default=10, type=int)
    args = parser.parse_args()

    out_img_dir = Path(args.output) / "train_img"
    out_lbl_dir = Path(args.output) / "train_label"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_img_dir.glob("*.png"))
    start_idx = int(existing[-1].stem) + 1 if existing else 0
    logger.info(f"Output: {args.output}  |  Start index: {start_idx:06d}")

    # Shared state between callbacks (accessed from different CARLA callback threads)
    lock = threading.Lock()
    latest_label = [None]   # list for mutability in closure
    frame_count = [0]
    saved_count = [0]
    done_flag = [False]

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    with NurecScenario(
        client,
        args.usdz,
        port=args.nurec_port,
        move_spectator=False,
        fps=args.fps,
    ) as scenario:

        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # --- CARLA semantic camera (same position as NuRec front camera) ---
        sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", "1920")
        sem_bp.set_attribute("image_size_y", "1080")
        sem_bp.set_attribute("fov", "90")

        ego_actor = scenario.actor_mapping[EGO_TRACK_ID].actor_inst
        sem_camera = world.spawn_actor(
            sem_bp,
            carla.Transform(carla.Location(x=0, y=0, z=1.47)),
            attach_to=ego_actor,
        )

        def on_semantic(image: carla.Image):
            trainid = carla_semantic_to_trainid(image)
            with lock:
                latest_label[0] = trainid

        sem_camera.listen(on_semantic)

        # --- NuRec camera callback ---
        def on_nurec_rgb(rgb: np.ndarray):
            with lock:
                label = latest_label[0]
                idx = start_idx + frame_count[0]
                frame_count[0] += 1

            if label is None or saved_count[0] >= args.max_frames:
                return

            save_pair(
                rgb, label,
                out_img_dir / f"{idx:06d}.png",
                out_lbl_dir / f"{idx:06d}.png",
            )

            with lock:
                saved_count[0] += 1
                if saved_count[0] % 50 == 0:
                    logger.info(f"  Saved {saved_count[0]} / {args.max_frames}")
                if saved_count[0] >= args.max_frames:
                    done_flag[0] = True

        scenario.add_camera(
            NUREC_CAM_PARAMS,
            on_nurec_rgb,
            transform=NUREC_CAM_TRANSFORM,
            framerate=args.fps,
            resolution_ratio=NUREC_RESOLUTION_RATIO,
        )

        # --- Run replay ---
        logger.info("Starting replay...")
        scenario.start_replay()

        try:
            while not scenario.is_done() and not done_flag[0]:
                scenario.tick()
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            sem_camera.stop()
            sem_camera.destroy()

        logger.info(f"Scene done. Saved {saved_count[0]} pairs.")

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
