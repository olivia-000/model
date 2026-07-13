"""
從 CARLA 抓取一組測試用的 depth + semantic 影像對，直接輸出成 ViPlanner 推論腳本能吃的格式。

v2 改動：不再把相機掛在一整台轎車上。之前為了不讓鏡頭拍到自己的引擎蓋，
被迫把相機往前推 2.4m 這種不自然的距離，導致近距離幾何失真、fear 值異常偏高。
這版直接把相機固定在世界座標中的一個「虛擬機器人位置」，不依賴任何車輛網格，
可以用貼近 ANYmal 四足機器人真實體型的偏移量 (相機離地 ~0.5m、前移量很小)。

前提：CARLA server 要先啟動 (./CarlaUE4.sh)，且已安裝 CARLA 的 Python API

用法：
    python carla_capture_test_pair.py --output_dir ./test_data
    python carla_capture_test_pair.py --output_dir ./test_data_obstacle \
        --spawn_obstacle --obstacle_distance 8.0

輸出：
    test_data/depth.npy               # float32，單位公尺，infer_single.py 可直接讀
    test_data/sem_viplanner.png       # 已經是 ViPlanner 30 類配色，可直接餵給 infer_single.py
    test_data/rgb_debug.png           # 順便存一張原始 RGB，方便你肉眼核對場景
"""

import argparse
import os
import time

import carla
import numpy as np
from PIL import Image

from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

CARLA_TAG_TO_VIPLANNER = {
    0: "static",
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    6: "pole",
    7: "traffic_light",
    8: "traffic_sign",
    9: "vegetation",
    10: "terrain",
    11: "sky",
    12: "person",
    13: "person",
    14: "vehicle",
    15: "vehicle",
    16: "vehicle",
    17: "on_rails",
    18: "motorcycle",
    19: "bicycle",
    20: "static",
    21: "dynamic",
    22: "static",
    23: "water_surface",
    24: "road",
    25: "terrain",
    26: "building",
    27: "on_rails",
    28: "fence",
    255: "static",
}


def build_carla_lut() -> np.ndarray:
    meta_handler = VIPlannerSemMetaHandler()
    lut = np.zeros((256, 3), dtype=np.uint8)
    default_color = meta_handler.class_color["static"]
    lut[:] = default_color
    for tag_id, vip_name in CARLA_TAG_TO_VIPLANNER.items():
        if vip_name in meta_handler.class_color:
            lut[tag_id] = meta_handler.class_color[vip_name]
    return lut


def decode_depth_meters(carla_image: "carla.Image") -> np.ndarray:
    raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((carla_image.height, carla_image.width, 4))
    b = raw[:, :, 0].astype(np.float32)
    g = raw[:, :, 1].astype(np.float32)
    r = raw[:, :, 2].astype(np.float32)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    return (normalized * 1000.0).astype(np.float32)


def decode_semantic_tags(carla_image: "carla.Image") -> np.ndarray:
    raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((carla_image.height, carla_image.width, 4))
    return raw[:, :, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output_dir", default="./test_data")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--spawn_index", type=int, default=0, help="使用地圖第幾個 spawn point 當虛擬機器人的參考位置")
    parser.add_argument(
        "--pitch_deg", type=float, default=-8.6,
        help="相機俯仰角(度)，負值=低頭看。對齊 model.yaml 的 camera_tilt=0.15 rad (~8.6 度)。",
    )
    parser.add_argument(
        "--camera_height", type=float, default=0.5,
        help="相機離地高度 (公尺)，貼近 ANYmal 四足機器人的實際高度。",
    )
    parser.add_argument(
        "--camera_forward_offset", type=float, default=0.3,
        help="相機相對『機器人參考點』的前移距離 (公尺)。不再依附車輛網格，用貼近真實機器人感測器安裝的小數值。",
    )
    parser.add_argument("--spawn_obstacle", action="store_true", help="在正前方生一台靜止車輛當障礙物")
    parser.add_argument("--obstacle_distance", type=float, default=5.0)
    parser.add_argument("--obstacle_lateral_offset", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 不再生成任何車輛當作載體，直接用地圖 spawn point 的座標+朝向當「虛擬機器人」參考位置。
    # 這是地圖固定資料，沒有物理同步延遲問題，也沒有車身網格會擋到鏡頭。
    spawn_points = world.get_map().get_spawn_points()
    ego_transform = spawn_points[args.spawn_index]
    forward_vec = ego_transform.get_forward_vector()
    right_vec = ego_transform.get_right_vector()
    up_vec = ego_transform.get_up_vector()

    print(f"[參考位置] 虛擬機器人位置: {ego_transform.location}, 朝向: {ego_transform.rotation}")

    captured = {"rgb": None, "depth": None, "semantic": None}
    obstacle = None

    if args.spawn_obstacle:
        obstacle_loc = (
            ego_transform.location
            + forward_vec * args.obstacle_distance
            + right_vec * (-args.obstacle_lateral_offset)
            + carla.Location(z=0.5)
        )
        obstacle_transform = carla.Transform(obstacle_loc, ego_transform.rotation)
        obstacle_bp = blueprint_library.filter("vehicle.audi.a2")[0]
        try:
            obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
            print(f"[障礙物] 已在正前方 {args.obstacle_distance}m 生成車輛，等待落到路面穩定...")
            for _ in range(30):
                if world.get_settings().synchronous_mode:
                    world.tick()
                else:
                    time.sleep(0.05)
            obstacle.set_simulate_physics(False)

            final_obstacle_loc = obstacle.get_location()
            delta = final_obstacle_loc - ego_transform.location
            dist_forward = delta.x * forward_vec.x + delta.y * forward_vec.y
            dist_right = delta.x * right_vec.x + delta.y * right_vec.y
            print(
                f"[除錯] 障礙物最終位置: {final_obstacle_loc}\n"
                f"[除錯] 障礙物相對機器人參考點: 前方 {dist_forward:.2f}m, 右方 {dist_right:.2f}m"
            )
        except RuntimeError as e:
            print(f"[警告] 障礙物生成失敗 (試試加大 --obstacle_distance): {e}")

    camera_location = (
        ego_transform.location + forward_vec * args.camera_forward_offset + up_vec * args.camera_height
    )
    camera_rotation = carla.Rotation(
        pitch=ego_transform.rotation.pitch + args.pitch_deg,
        yaw=ego_transform.rotation.yaw,
        roll=ego_transform.rotation.roll,
    )
    camera_transform = carla.Transform(camera_location, camera_rotation)
    print(f"[相機] 世界座標: {camera_location}, 俯仰角: {camera_rotation.pitch:.1f}度")

    def make_camera(bp_name):
        bp = blueprint_library.find(bp_name)
        bp.set_attribute("image_size_x", str(args.width))
        bp.set_attribute("image_size_y", str(args.height))
        bp.set_attribute("fov", "90")
        return world.spawn_actor(bp, camera_transform, attach_to=None)

    cam_rgb = make_camera("sensor.camera.rgb")
    cam_depth = make_camera("sensor.camera.depth")
    cam_sem = make_camera("sensor.camera.semantic_segmentation")

    cam_rgb.listen(lambda img: captured.__setitem__("rgb", img))
    cam_depth.listen(lambda img: captured.__setitem__("depth", img))
    cam_sem.listen(lambda img: captured.__setitem__("semantic", img))

    try:
        print("[等待] 讓場景穩定並收集第一幀影像...")
        for _ in range(20):
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                time.sleep(0.05)

        timeout_start = time.time()
        while captured["rgb"] is None or captured["depth"] is None or captured["semantic"] is None:
            if time.time() - timeout_start > 15:
                raise TimeoutError("等待感測器資料超過 15 秒，檢查 CARLA server 狀態")
            time.sleep(0.1)

        rgb_arr = np.frombuffer(captured["rgb"].raw_data, dtype=np.uint8).reshape(
            (captured["rgb"].height, captured["rgb"].width, 4)
        )[:, :, :3][:, :, ::-1]
        Image.fromarray(rgb_arr).save(os.path.join(args.output_dir, "rgb_debug.png"))

        depth_m = decode_depth_meters(captured["depth"])
        np.save(os.path.join(args.output_dir, "depth.npy"), depth_m)

        tags = decode_semantic_tags(captured["semantic"])
        lut = build_carla_lut()
        sem_rgb = lut[tags]
        Image.fromarray(sem_rgb).save(os.path.join(args.output_dir, "sem_viplanner.png"))

        print(f"[完成] 輸出存到 {args.output_dir}/")
        print(f"  depth.npy         (深度，公尺，shape={depth_m.shape})")
        print(f"  sem_viplanner.png (ViPlanner 配色語義圖)")
        print(f"  rgb_debug.png     (原始 RGB，方便核對場景)")

    finally:
        cam_rgb.destroy()
        cam_depth.destroy()
        cam_sem.destroy()
        if obstacle is not None:
            obstacle.destroy()


if __name__ == "__main__":
    main()
