#!/usr/bin/env python3
"""
機器狗即時部署・節點②：ViPlanner 規劃節點（在 viplanner env 執行）

訂閱 depth（對齊到彩色相機）＋ 語義圖（節點①的輸出）＋ 目標點 ＋ 里程計，
每個週期執行 DualAutoEncoder 推論，發布 nav_msgs/Path（odom 座標系）與 fear 分數。

啟動:
    source /opt/ros/noetic/setup.bash
    conda run -n viplanner python viplanner_dog_node.py \
        _model_dir:=/home/itriu100/viplanner/viplanner_models \
        _depth_topic:=/camera/aligned_depth_to_color/image_raw \
        _odom_topic:=/odom \
        _goal_topic:=/viplanner/goal

座標系原理（重要）:
  - ViPlanner 網路吃的 goal 和吐的軌跡都在「相機機器人慣例座標系」：x前方/y左方/z上方。
  - 目標點以 PointStamped 給定在 odom（世界）座標系。每個規劃週期用「最新的里程計位姿」
    把 goal 轉進 base 座標系，再扣掉相機安裝偏移轉進相機座標系（純 numpy 四元數運算，
    不依賴 tf2 —— tf2_py 是編譯套件，在 conda env 裡經常跟系統 python 打架）。
  - 網路輸出軌跡（相機座標系）反向轉回 odom 座標系再發布，路徑跟隨器就能在機器狗
    移動中持續追蹤同一條世界座標系路徑。
"""

import os
import sys
import time

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int16

# 重用 repo 根目錄的獨立推論類別（模型載入/前處理跟 infer_single.py 完全一致）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from infer_single import VIPlannerInferenceStandalone  # noqa: E402


# ---------- 四元數/座標轉換小工具（取代 tf2） ----------

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(x,y,z,w) 四元數 → 3x3 旋轉矩陣。"""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pitch_rot(pitch_rad: float) -> np.ndarray:
    """繞 y 軸（機器人左方）的俯仰旋轉，pitch>0 = 相機朝下。"""
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def depth_msg_to_meters(msg: Image, depth_scale: float) -> np.ndarray:
    """sensor_msgs/Image 深度圖 → 公尺 float32。支援 RealSense 的 16UC1(mm) 與 32FC1(m)。"""
    if msg.encoding == "16UC1":
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return raw.astype(np.float32) / depth_scale
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"不支援的深度編碼: {msg.encoding}")


class VIPlannerDogNode:
    def __init__(self):
        rospy.init_node("viplanner_dog", anonymous=False)

        # ---- 參數 ----
        model_dir = rospy.get_param("~model_dir")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        sem_topic = rospy.get_param("~sem_topic", "/viplanner/sem_image")
        goal_topic = rospy.get_param("~goal_topic", "/viplanner/goal")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        path_topic = rospy.get_param("~path_topic", "/viplanner/path")
        self.main_freq = rospy.get_param("~main_freq", 5.0)  # 規劃頻率 Hz
        self.fear_threshold = rospy.get_param("~fear_threshold", 0.5)
        self.fear_buffer_size = rospy.get_param("~fear_buffer", 3)  # 連續 N 幀高 fear 才停
        self.goal_reached_dist = rospy.get_param("~goal_reached_dist", 0.5)  # m
        self.max_goal_clip = rospy.get_param("~max_goal_clip", 10.0)  # goal 超過訓練分佈就先夾住
        # 相機安裝：base 座標系（x前/y左/z上）下的相機位置與俯仰角
        self.cam_offset = np.array(
            [
                rospy.get_param("~cam_offset_x", 0.30),
                rospy.get_param("~cam_offset_y", 0.0),
                rospy.get_param("~cam_offset_z", 0.20),
            ]
        )
        self.cam_pitch = np.deg2rad(rospy.get_param("~cam_pitch_deg", 0.0))
        self._R_cam_base = pitch_rot(self.cam_pitch)  # base→cam 的旋轉（僅俯仰）

        # ---- 模型（跟 infer_single.py 完全同一套載入與前處理）----
        self.algo = VIPlannerInferenceStandalone(model_dir, fear_threshold=self.fear_threshold)
        assert self.algo.train_config.sem, "此節點是全景分割路線，model.yaml 必須是 sem=True 的模型"

        # ---- 狀態 ----
        self.depth_m: np.ndarray = None
        self.sem_rgb: np.ndarray = None
        self.odom_pos: np.ndarray = None  # (3,) odom 座標系
        self.odom_rot: np.ndarray = None  # 3x3, base→odom
        self.goal_odom: np.ndarray = None  # (3,) odom 座標系
        self.fear_count = 0

        # ---- ROS I/O ----
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=1)
        self.fear_pub = rospy.Publisher("/viplanner/fear", Float32, queue_size=1)
        self.status_pub = rospy.Publisher("/viplanner/status", Int16, queue_size=1)
        self.timer_pub = rospy.Publisher("/viplanner/timer", Float32, queue_size=1)
        rospy.Subscriber(depth_topic, Image, self.depth_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(sem_topic, Image, self.sem_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(goal_topic, PointStamped, self.goal_cb, queue_size=1)

        rospy.loginfo(f"[viplanner_dog] 就緒。depth={depth_topic}  sem={sem_topic}  goal={goal_topic}")
        self.spin()

    # ---------- callbacks：只存最新資料，運算集中在 spin ----------

    def depth_cb(self, msg: Image):
        self.depth_m = depth_msg_to_meters(msg, self.algo.depth_scale)

    def sem_cb(self, msg: Image):
        self.sem_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_pos = np.array([p.x, p.y, p.z])
        self.odom_rot = quat_to_rot(np.array([q.x, q.y, q.z, q.w]))

    def goal_cb(self, msg: PointStamped):
        self.goal_odom = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.fear_count = 0
        rospy.loginfo(f"[viplanner_dog] 新目標點 (odom): {self.goal_odom.round(2)}")

    # ---------- 主迴圈 ----------

    def spin(self):
        rate = rospy.Rate(self.main_freq)
        while not rospy.is_shutdown():
            rate.sleep()
            if any(v is None for v in (self.depth_m, self.sem_rgb, self.odom_pos, self.goal_odom)):
                continue

            # 拷貝當下快照，避免 callback 中途覆寫
            depth_m = self.depth_m.copy()
            sem_rgb = self.sem_rgb.copy()
            odom_pos, odom_rot = self.odom_pos.copy(), self.odom_rot.copy()

            # 1) goal: odom → base → cam（x前/y左/z上），z 壓成 0（網路訓練時 goal 落在地平面）
            goal_base = odom_rot.T @ (self.goal_odom - odom_pos)
            if np.linalg.norm(goal_base[:2]) < self.goal_reached_dist:
                self.status_pub.publish(Int16(1))  # 1 = 到達目標
                self.path_pub.publish(self._empty_path())
                continue
            goal_cam = self._R_cam_base @ (goal_base - self.cam_offset)
            goal_cam[2] = 0.0
            # 超出訓練分佈（max_goal_distance=15m，常用 ≤10m）就夾到最大距離方向上
            dist = np.linalg.norm(goal_cam[:2])
            if dist > self.max_goal_clip:
                goal_cam[:2] *= self.max_goal_clip / dist

            # 2) 網路推論（前處理交給 VIPlannerInferenceStandalone，跟離線驗證完全一致）
            t0 = time.time()
            depth_t = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0).to(self.algo.device)
            sem_t = torch.from_numpy(sem_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.algo.device)
            goal_t = torch.tensor([goal_cam], dtype=torch.float32, device=self.algo.device)
            _, traj, fear = self.algo.plan(depth_t, sem_t, goal_t)
            self.timer_pub.publish(Float32((time.time() - t0) * 1000.0))

            fear_val = float(fear[0].item())
            self.fear_pub.publish(Float32(fear_val))

            # 3) fear 防抖：連續 N 幀超標才判定危險（單幀誤判不至於急停）
            self.fear_count = self.fear_count + 1 if fear_val > self.fear_threshold else 0
            if self.fear_count >= self.fear_buffer_size:
                rospy.logwarn_throttle(1.0, f"[viplanner_dog] fear={fear_val:.2f} 連續超標，發布空路徑（停止）")
                self.status_pub.publish(Int16(-1))  # -1 = 高風險急停
                self.path_pub.publish(self._empty_path())
                continue

            # 4) 軌跡: cam → base → odom，發布世界座標系 Path
            traj_cam = traj[0].detach().cpu().numpy()  # (N,3)
            traj_base = (self._R_cam_base.T @ traj_cam.T).T + self.cam_offset
            traj_odom = (odom_rot @ traj_base.T).T + odom_pos
            self.status_pub.publish(Int16(0))  # 0 = 正常規劃中
            self.path_pub.publish(self._make_path(traj_odom))

    # ---------- Path 訊息 ----------

    def _empty_path(self) -> Path:
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"
        return path

    def _make_path(self, traj_odom: np.ndarray) -> Path:
        path = self._empty_path()
        for pt in traj_odom:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pt
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path


if __name__ == "__main__":
    try:
        VIPlannerDogNode()
    except rospy.ROSInterruptException:
        pass
