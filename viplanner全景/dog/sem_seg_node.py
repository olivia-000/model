#!/usr/bin/env python3
"""
機器狗即時部署・節點①：全景分割節點（在 mask2former_env 執行）

訂閱 RGB 相機影像 → mmdet Mask2Former 全景分割 → 發布 ViPlanner 34 類色碼語義圖。

跟 planner 節點跑在「不同 conda env、不同行程」，中間用 ROS topic 當介面——
這正是離線 pipeline「兩個 env 用檔案交接」的即時版，天然解掉 mmcv/torch 版本衝突。

啟動（記得先 source ROS）:
    source /opt/ros/noetic/setup.bash
    conda run -n mask2former_env python sem_seg_node.py \
        _rgb_topic:=/camera/color/image_raw \
        _m2f_config:=/path/to/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py \
        _m2f_checkpoint:=/path/to/xxx.pth

注意：mask2former_env 內需 `pip install rospkg pyyaml`（rospy 本體來自
source /opt/ros/noetic/setup.bash 的 PYTHONPATH，python3.10 也能用）。
刻意不用 cv_bridge（它綁定系統 python，在 conda 環境常炸），影像用 numpy 手動編解碼。
"""

import os
import sys
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32

# 讓 panoptic_inference.py 的 PanopticSegmenter 可以直接重用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from panoptic_inference import PanopticSegmenter  # noqa: E402


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → BGR np.ndarray，不經過 cv_bridge。"""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ("bgr8",):
        return data[:, :, :3]
    if msg.encoding in ("rgb8",):
        return cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
    if msg.encoding in ("bgra8",):
        return data[:, :, :3]
    if msg.encoding in ("rgba8",):
        return cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"不支援的影像編碼: {msg.encoding}")


def rgb_to_imgmsg(rgb: np.ndarray, header) -> Image:
    """RGB np.ndarray → sensor_msgs/Image (rgb8)。"""
    msg = Image()
    msg.header = header
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg


class SemSegNode:
    def __init__(self):
        rospy.init_node("viplanner_sem_seg", anonymous=False)

        rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        compressed = rospy.get_param("~compressed", False)
        sem_topic = rospy.get_param("~sem_topic", "/viplanner/sem_image")
        m2f_config = rospy.get_param("~m2f_config")
        m2f_checkpoint = rospy.get_param("~m2f_checkpoint")
        viplanner_root = rospy.get_param(
            "~viplanner_root", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        self.segmenter = PanopticSegmenter(m2f_config, m2f_checkpoint, viplanner_root)

        self.sem_pub = rospy.Publisher(sem_topic, Image, queue_size=1)
        self.timer_pub = rospy.Publisher("/viplanner/m2f_timer", Float32, queue_size=1)

        # queue_size=1 + buff_size 放大：分割比相機慢，永遠只處理最新一幀，舊幀直接丟掉，
        # 避免 callback 佇列堆積造成語義圖延遲越來越大
        if compressed:
            rospy.Subscriber(rgb_topic, CompressedImage, self.compressed_cb, queue_size=1, buff_size=2**24)
        else:
            rospy.Subscriber(rgb_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[sem_seg_node] 就緒。訂閱 {rgb_topic}（compressed={compressed}）→ 發布 {sem_topic}")
        rospy.spin()

    def compressed_cb(self, msg: CompressedImage):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        self._process(bgr, msg.header)

    def image_cb(self, msg: Image):
        self._process(imgmsg_to_bgr(msg), msg.header)

    def _process(self, bgr: np.ndarray, header):
        t0 = time.time()
        sem_rgb = self.segmenter.predict(bgr)
        dt = time.time() - t0
        # header 沿用 RGB 原始時間戳：讓 planner 端知道這張語義圖對應哪個時刻的畫面
        self.sem_pub.publish(rgb_to_imgmsg(sem_rgb, header))
        self.timer_pub.publish(Float32(dt * 1000.0))


if __name__ == "__main__":
    try:
        SemSegNode()
    except rospy.ROSInterruptException:
        pass
