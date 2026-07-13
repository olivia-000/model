#!/usr/bin/env bash
# 一鍵啟動機器狗三節點（各自跑在正確的環境裡）。
# 前提：roscore 已在跑、相機驅動已啟動（例如 realsense2_camera 且 align_depth:=true）、
#       機器狗驅動已提供 /odom 與 /cmd_vel。
#
# 用法: ./run_dog.sh
# 停止: Ctrl+C（會一併殺掉三個子行程）

set -euo pipefail

# ===== 依你的實際硬體/路徑修改這一段 =====
SEG_ENV=mask2former_env
VIP_ENV=viplanner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

MODEL_DIR="$REPO_ROOT/viplanner_models"
M2F_CONFIG="$SCRIPT_DIR/../m2f_ckpt/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py"
M2F_CKPT=$(ls "$SCRIPT_DIR"/../m2f_ckpt/*.pth | head -1)

RGB_TOPIC=/camera/color/image_raw
DEPTH_TOPIC=/camera/aligned_depth_to_color/image_raw   # 一定要用「對齊到彩色」的深度
ODOM_TOPIC=/odom
CMD_VEL_TOPIC=/cmd_vel

CAM_OFFSET_X=0.30   # 相機在 base 座標系的位置（公尺，x前/y左/z上）——依實際安裝量測
CAM_OFFSET_Z=0.20
CAM_PITCH_DEG=0.0   # 相機俯仰角（>0 朝下）
MAX_V=0.5           # 跟隨器最大前進速度 m/s
# ========================================

source /opt/ros/noetic/setup.bash

trap 'kill 0' EXIT INT TERM

echo "[1/3] 全景分割節點（$SEG_ENV）"
conda run --no-capture-output -n "$SEG_ENV" python "$SCRIPT_DIR/sem_seg_node.py" \
    _rgb_topic:="$RGB_TOPIC" \
    _m2f_config:="$M2F_CONFIG" \
    _m2f_checkpoint:="$M2F_CKPT" &

echo "[2/3] ViPlanner 規劃節點（$VIP_ENV）"
conda run --no-capture-output -n "$VIP_ENV" python "$SCRIPT_DIR/viplanner_dog_node.py" \
    _model_dir:="$MODEL_DIR" \
    _depth_topic:="$DEPTH_TOPIC" \
    _odom_topic:="$ODOM_TOPIC" \
    _cam_offset_x:="$CAM_OFFSET_X" \
    _cam_offset_z:="$CAM_OFFSET_Z" \
    _cam_pitch_deg:="$CAM_PITCH_DEG" &

echo "[3/3] 路徑跟隨器（系統 python）"
python3 "$SCRIPT_DIR/simple_path_follower.py" \
    _odom_topic:="$ODOM_TOPIC" \
    _cmd_vel_topic:="$CMD_VEL_TOPIC" \
    _max_v:="$MAX_V" &

echo ""
echo "三節點已啟動。下目標點（odom 座標系，例：前方 5m）："
echo '  rostopic pub -1 /viplanner/goal geometry_msgs/PointStamped \'
echo '    "{header: {frame_id: odom}, point: {x: 5.0, y: 0.0, z: 0.0}}"'
wait
