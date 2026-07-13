#!/usr/bin/env python3
"""
機器狗即時部署・節點③：簡易 Pure Pursuit 路徑跟隨器

訂閱 ViPlanner 發布的 Path（odom 座標系）＋ 里程計，輸出 geometry_msgs/Twist
到 cmd_vel —— 大多數四足機器人（Unitree Go1/Go2、宇樹 ROS 驅動等）都吃
cmd_vel 速度指令（linear.x 前進、angular.z 轉向），由狗自己的運動控制器
負責落腳步態，這個節點只管「沿路徑走」。

原理（Pure Pursuit）:
  1. 在路徑上找一個距離機器人 lookahead 公尺的「前視點」。
  2. 把前視點轉到 base 座標系，得到偏角 alpha = atan2(y, x)。
  3. 角速度 w = 2 * v * sin(alpha) / lookahead（純追蹤曲率公式），
     偏角太大時先原地轉向再前進。
  4. 收到空路徑（fear 急停 / 到達目標）→ 立刻發零速度。

啟動:
    source /opt/ros/noetic/setup.bash
    python3 simple_path_follower.py _cmd_vel_topic:=/cmd_vel _max_v:=0.5
（此節點不依賴 torch/mmdet，任何有 rospy 的 python 都能跑。）
"""

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


def yaw_from_quat(x, y, z, w) -> float:
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class SimplePathFollower:
    def __init__(self):
        rospy.init_node("simple_path_follower", anonymous=False)

        path_topic = rospy.get_param("~path_topic", "/viplanner/path")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.lookahead = rospy.get_param("~lookahead", 0.8)  # m
        self.max_v = rospy.get_param("~max_v", 0.5)  # m/s
        self.max_w = rospy.get_param("~max_w", 0.8)  # rad/s
        self.turn_in_place_angle = np.deg2rad(rospy.get_param("~turn_in_place_deg", 60.0))
        self.ctrl_freq = rospy.get_param("~ctrl_freq", 20.0)  # Hz
        self.path_timeout = rospy.get_param("~path_timeout", 1.5)  # 秒，路徑太舊就停

        self.path_xy: np.ndarray = None  # (N,2) odom 座標
        self.path_stamp = rospy.Time(0)
        self.pose = None  # (x, y, yaw)

        self.cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        rospy.Subscriber(path_topic, Path, self.path_cb, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odom_cb, queue_size=1)

        rospy.loginfo(f"[follower] 就緒。{path_topic} + {odom_topic} → {cmd_vel_topic}")
        self.spin()

    def path_cb(self, msg: Path):
        self.path_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        if len(msg.poses) == 0:
            self.path_xy = None  # 空路徑 = 停止指令（fear 急停或到達目標）
        else:
            self.path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w))

    def spin(self):
        rate = rospy.Rate(self.ctrl_freq)
        while not rospy.is_shutdown():
            rate.sleep()
            cmd = Twist()  # 預設零速度，任何異常都會落到「停」
            if (
                self.pose is not None
                and self.path_xy is not None
                and (rospy.Time.now() - self.path_stamp).to_sec() < self.path_timeout
            ):
                x, y, yaw = self.pose
                rel = self.path_xy - np.array([x, y])
                dist = np.linalg.norm(rel, axis=1)

                # 找第一個距離 >= lookahead 的路徑點當前視點；都太近就取最後一點
                ahead_idx = np.argmax(dist >= self.lookahead) if (dist >= self.lookahead).any() else len(dist) - 1
                target = rel[ahead_idx]

                # 轉進 base 座標系
                c, s = np.cos(-yaw), np.sin(-yaw)
                tx = c * target[0] - s * target[1]
                ty = s * target[0] + c * target[1]
                alpha = np.arctan2(ty, tx)

                if abs(alpha) > self.turn_in_place_angle:
                    # 偏角太大：原地轉向，避免大弧線掃到旁邊障礙物
                    cmd.angular.z = np.clip(2.0 * alpha, -self.max_w, self.max_w)
                else:
                    ld = max(np.hypot(tx, ty), 1e-3)
                    v = self.max_v * max(np.cos(alpha), 0.2)  # 偏角越大越減速
                    cmd.linear.x = np.clip(v, 0.0, self.max_v)
                    cmd.angular.z = np.clip(2.0 * cmd.linear.x * np.sin(alpha) / ld, -self.max_w, self.max_w)
            self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        SimplePathFollower()
    except rospy.ROSInterruptException:
        pass
