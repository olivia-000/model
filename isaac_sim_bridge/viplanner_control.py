"""Isaac Sim control loop: HIMLoco locomotion policy + ViPlanner visual navigation, driving the
Go2 articulation at /World/go2 toward a world-space goal point, inside an already-running,
already-rendering Kit app (this is NOT a standalone SimulationApp script -- it's meant to be
awaited from another --exec script, e.g. open_scene.py, after the stage/dog/goal are set up).

Physics/observation timing is NOT an exact reproduction of the original Isaac Gym env's
sim_dt=0.005/decimation=4 (50Hz) control loop -- this drives control off the Kit app's own
render-tied update loop instead (since a live GUI process must let Kit own stepping, see
World.step()'s own docstring warning against manual render=True stepping in an extension/exec
context). Close enough for a live visual demo; not a numeric match to the training env.
"""

#目標點在 /home/itri/isaac-sim-docker/home/open_scene_stair_usd.py 第 149~154 行的 waypoints 列表：

#      waypoints = [
#         (3.4, FLIGHT_LEN + 1.25, LANDING_Z + 0.35),        # 1. 平台 1      (3.4, 4.49, 2.69)
#         (2.2, -1.32, SECOND_FLOOR_Z + 0.35),               # 2. 2F 中間     (2.2, -1.32, 5.03)
#         (3.4, FLIGHT_LEN + 1.25, 3 * LANDING_Z + 0.35),    # 3. 平台 2      (3.4, 4.49, 7.37)
#         (2.2, -1.0, THIRD_FLOOR_Z + 0.35),                 # 4. 3F 中間     (2.2, -1.0, 9.71)
#      ]

# 每一點是 (x, y, z)，單位是公尺。狗會照順序一個一個走過去，最後一個就是最終目標，場景裡只有最終目標會畫出藍色的球。

# 要改的話請注意這幾點：

# 1. z 要設在「那層地板高度 + 0.35」，0.35 大約是狗站著時身體的高度。各層地板的 z：1F = 0、平台 1 = 2.34、2F =
#    4.68、平台 2 = 7.02、3F = 9.36。程式裡已經有對應的常數可以直接用，例如 SECOND_FLOOR_Z + 0.35。
# 2. 只想改最終目標，就改最後一行。前面幾個點是讓狗在平台和 2F 轉彎用的。如果最終目標改到 2F，就把第 3、4 行刪掉。
# 3. 各層可以走的範圍（x, y）：
#    - 1F：x 0~4.4，y -2~5.2。y < -2 和 x > 4.4 那兩側沒有牆也沒有地板，太靠近會掉出場景。
#    - 2F：x 0~4.4，y -2.6~0
#    - 3F：x 0~4.4，y -2~0。y = -2 那一側也沒有牆。
# 4. 目標點不要放在離樓梯 1.5 m 以內。控制程式會把它當成「樓梯口目標」，走法會不一樣。
# 5. 平台上的轉彎點 x 要放 3.4（樓梯正前方），不要放 2.2（平台中間）。放中間的話，狗會斜切過去，卡在欄杆末端。

# 改完要重新啟動 Isaac Sim 才會生效：

# cd ~/isaac-sim-docker && ./run_stair_usd.sh

# 另外提醒兩件事：

# - 狗的出生點也在同一個檔案，第 39~40 行的 DOG_START_XY（位置）和 DOG_START_YAW_DEG（朝向）。
# - 現在正在收集訓練資料，用的是目前這組目標點。改檔案不會影響已經在跑的收集，但如果收集還沒結束，請不要另外啟動 Isaac
#   Sim。兩個 container 都叫 isaac-sim，名稱衝突，新的會啟動失敗。



import asyncio
import math
import sys
import time
from collections import deque

import numpy as np
import torch

CONTAINER_HOME = "/home/isaacsim"
VIPLANNER2_ROOT = f"{CONTAINER_HOME}/viplanner2"
# gait/locomotion policy weight -- sourced directly from the viplanner2 project, not borrowed
# from a separate HIMLoco checkout (the earlier layout kept the driver scripts under the Isaac
# Sim docker home and only reached into viplanner2 for this one file; everything driver-side now
# lives in viplanner2/isaac_sim_bridge instead, so this is a same-project reference, not a hop).
GAIT_POLICY_CKPT = f"{VIPLANNER2_ROOT}/0908/policy_0430_134538_6340.pt"
# Fine-tuned on this exact stairs scene (2026-09-11): 30 rollout episodes (21,732 samples) collected
# via viplanner_control.py's own collect_data_dir mode, cost map built from the scene's exact AABB
# geometry with wall-class collision cost patched from the shared default (2.0) to 20.0 per explicit
# request (walls -- and persons, tagged the same class here -- must never be an acceptable path).
# See training_data/{build_cloud,build_costmap,train_stairs}.py. Falls back to the original
# multi-scene checkpoint if the fine-tuned one isn't present (e.g. a fresh checkout without having
# run the training scripts yet).
# v2 (2026-09-15): re-fine-tuned from the original base checkpoint with two fixes the v1 run above
# never had -- TrajCost.robot_width set to Go2's real width (0.28m, was defaulting to ANYmal's
# 0.6m) so the training loss actually samples obstacle cost across the robot's true footprint, and
# both cost-map sigma_smooth values raised to 6.0 cells (~0.3m, matching Go2's own ~0.354m
# half-diagonal) so the wall clearance gradient the network learns from is sized for this robot
# instead of a generic default. Trained 80 epochs total (30 + a 50-epoch continuation), val loss
# 11.85 -> 8.86 -> 8.78, confirmed converged (last 4 epochs at the scheduler's lowest LR produced
# no further improvement). Kept (not deleted) per explicit user request, as a reference/fallback
# alongside the original base checkpoint.
# v3 (2026-09-16): retrained on data recollected AFTER fixing the exposed-stair-back-face-
# mislabeled-as-walkable bug in open_scene_stairs.py -- v2's entire pipeline (cloud.ply/aabbs.json/
# collected rollout episodes) predated that fix (captured 2026-09-11), so v2 never actually learned
# from the corrected semantics despite the live scene having it. Also fixed a real data-integrity
# bug found while rebuilding this: camera_extrinsic.txt is written append-only, so a killed/
# restarted collection run left stale rows misaligned against the (cleanly overwritten) depth/
# semantic files -- see viplanner_control.py's own collection-setup comment. Trained 60 epochs
# total (30 + a 30-epoch continuation), val loss 12.59 -> 10.48 -> 10.14, confirmed converged (last
# 9 epochs of the continuation produced no further improvement, a flatter plateau than v2 showed at
# the same decision point).
_FINETUNED_MODEL_DIR = f"{VIPLANNER2_ROOT}/training_data/models/plannernet_envstairs_scene_ep30_inputDepSem_costSem_optimSGD_stairs_finetune_v3"
import os as _os_check
# diagnostic toggle: VIPLANNER_USE_ORIGINAL_MODEL=1 forces the original multi-scene checkpoint even
# when the fine-tuned one is present -- for comparing reliability across many episodes without
# touching either checkpoint's files. Multi-episode testing (2026-09-11, against v1 of the
# fine-tune above) found the fine-tuned checkpoint completed 2/4 episodes cleanly but got
# completely stuck on waypoint 1 (13000+ steps, never even entering the stair corridor) the other
# 2/4 -- worth re-checking whether v2's robot_width/sigma_smooth fix changed that reliability
# picture before trusting it over the original checkpoint again.
_use_original = _os_check.environ.get("VIPLANNER_USE_ORIGINAL_MODEL") == "1"
VIPLANNER_MODEL_DIR = (
    f"{CONTAINER_HOME}/viplanner_ckpt" if _use_original
    else (_FINETUNED_MODEL_DIR if _os_check.path.isdir(_FINETUNED_MODEL_DIR) else f"{CONTAINER_HOME}/viplanner_ckpt")
)
# VIPLANNER_MODEL_NAME: a model directory name under training_data/models/ (e.g. the stair_usd_geo_v7
# run) to load instead of the default v3 -- for comparing checkpoints without editing this file
if _os_check.environ.get("VIPLANNER_MODEL_NAME"):
    VIPLANNER_MODEL_DIR = f"{VIPLANNER2_ROOT}/training_data/models/{_os_check.environ['VIPLANNER_MODEL_NAME']}"

if f"{VIPLANNER2_ROOT}/viplanner_core" not in sys.path:
    sys.path.insert(0, f"{VIPLANNER2_ROOT}/viplanner_core")

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT_ANGLES = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32
)
KP, KD = 40.0, 1.0
ACTION_SCALE = 0.25

# Explicit user request ("好，先做腿部里程計"): Go2's own real, fixed mechanical dimensions, read
# directly from its URDF (/home/itri/fire_dog/go2_new/go2.urdf) -- a real deployment hard-codes
# these exact same numbers too (they're the robot's physical geometry, not simulator-privileged
# state), used below for forward-kinematics-based leg odometry instead of base_pos[2].
_LEG_HIP_ORIGIN = {  # hip joint's own origin, in body frame (xyz, meters)
    "FL": np.array([0.1934, 0.0465, 0.0], dtype=np.float32),
    "FR": np.array([0.1934, -0.0465, 0.0], dtype=np.float32),
    "RL": np.array([-0.1934, 0.0465, 0.0], dtype=np.float32),
    "RR": np.array([-0.1934, -0.0465, 0.0], dtype=np.float32),
}
_LEG_THIGH_Y_SIGN = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}  # hip-to-thigh offset direction
_LEG_THIGH_OFFSET = 0.0955  # hip-to-thigh joint offset magnitude, meters
_LEG_LINK_LENGTH = 0.213  # thigh length == calf length for this robot, meters
_LEG_ORDER = ["FL", "FR", "RL", "RR"]  # matches JOINT_NAMES' own leg grouping order


def go2_foot_positions_body_frame(dof_pos):
    """Explicit user request: standard forward kinematics for a Unitree-style leg (hip rotates
    about the leg's own local +X / ab-ad, thigh and calf about local +Y / pitch -- confirmed from
    the URDF's own <axis> tags, not assumed), building the chain as explicit rotation matrices
    rather than a memorized closed-form formula to avoid a sign-convention mistake. dof_pos: (12,)
    in JOINT_NAMES order. Returns (4,3) foot positions in BODY frame, ordered per _LEG_ORDER --
    exactly what a real robot computes from its own joint encoders + known link lengths, no
    simulator-privileged world pose involved anywhere in this function."""
    def _rot_x(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)

    def _rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    feet = np.zeros((4, 3), dtype=np.float32)
    for i, leg in enumerate(_LEG_ORDER):
        q_hip, q_thigh, q_calf = dof_pos[i * 3], dof_pos[i * 3 + 1], dof_pos[i * 3 + 2]
        p_calf_to_foot = np.array([0.0, 0.0, -_LEG_LINK_LENGTH], dtype=np.float32)
        p_thigh_to_calf = np.array([0.0, 0.0, -_LEG_LINK_LENGTH], dtype=np.float32) + _rot_y(q_calf) @ p_calf_to_foot
        p_hip_to_thigh = (
            np.array([0.0, _LEG_THIGH_Y_SIGN[leg] * _LEG_THIGH_OFFSET, 0.0], dtype=np.float32)
            + _rot_y(q_thigh) @ p_thigh_to_calf
        )
        feet[i] = _LEG_HIP_ORIGIN[leg] + _rot_x(q_hip) @ p_hip_to_thigh
    return feet

LIN_VEL_SCALE, ANG_VEL_SCALE, DOF_VEL_SCALE = 2.0, 0.25, 0.05
COMMANDS_SCALE = np.array([LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE], dtype=np.float32)
OBS_HISTORY_LEN = 6

REPLAN_EVERY_N_STEPS = 8  # history (all per explicit user request): 5 -> 8 -> 4 -> 5 -> 4 -> back
# to 8. 8/8 (this setting, paired with COMMAND_HOLD_STEPS below) is the best-tested combination so
# far -- 6/7 success over a real multi-episode run, no FPS concerns (~53fps). Camera capture +
# semantic classify + ViPlanner inference is the single most expensive block each cycle; less
# frequent replanning is the highest-leverage knob for overall smoothness if a faster value is
# tried again later.
# 100 (~1.6s of holding the same heading) was tuned for the wide-open lab scene; in the stairs
# scene's narrow corridors (walls close on both sides), holding a heading that long at the higher
# MIN_VX_SCALE speed runs it straight into a wall before the next correction ever happens. Down
# to 20 (~0.33s) so it re-aims far more often -- steering reactivity, not speed, is what actually
# keeps it off the walls in a tight corridor.
# History (all per explicit user request): 20 -> 5 -> 10 -> 8 -> 5 -> 4 -> back to 8. Originally, a
# 20-step hold against REPLAN_EVERY_N_STEPS=8 meant the drawn path could visibly update up to 2.5x
# before the robot's actual command caught up to it -- watching the viewport, the plan visibly
# changed well before the dog reacted. Kept equal to REPLAN_EVERY_N_STEPS at every value tried
# since, so every replan tick immediately updates the command with no mismatch between the two
# cadences. 8/8 (this setting) is the best-tested combination so far -- 6/7 success over a real
# multi-episode run. Paired with a faster HEADING_SMOOTH_RATE below (still smoothed, not instant,
# to keep the noisy per-replan raw heading from directly jerking the gait).
COMMAND_HOLD_STEPS = 8
LOOKAHEAD_IDX = 15
MAX_SPEED = 0.6
# was 2.0 -- combined with noisy per-replan heading readings (confirmed via diagnostics: the same
# stretch of corridor could read -41 -> +126 -> +98 degrees across consecutive replans), this
# translated every small wobble in the raw heading estimate into a big, visible turn -- looked like
# the dog "kept looking left and right" while otherwise walking a reasonable average path. Lower
# gain = same noise produces a gentler turn command.
YAW_GAIN = 1.2
MAX_YAW_RATE = 0.9
YAW_DEADZONE_RAD = math.radians(15.0)
FEAR_THRESHOLD = 0.5
REACH_THRESHOLD = 0.5

CAMERA_RES = (256, 144)  # W, H -- was (320, 180), lowered again for smoothness. Captured smaller
# than the ViPlanner model's own img_input_size
# (360x640) on purpose: VIPlannerAlgo._resize() bilinear-upsamples to the model's expected size
# internally, and capturing+classifying 1/4 the pixels each replan step removes the periodic
# stutter that came from running semantic AABB classification over the full 640x360 image on
# every replan (every REPLAN_EVERY_N_STEPS physics steps).
CAMERA_HFOV_DEG = 87.0
# Explicit user request (2026-10-02, option "B"): the camera never set a clipping range, so it
# rendered with Isaac Sim's default 1.0m near plane -- everything closer than 1m came back pinned at
# ~1.0m (confirmed with a controlled flat-floor render), i.e. a 1m blind zone in front of the dog.
# 0.2m is close to a real RealSense-class depth camera's minimum range. Data collected before this
# (v3/v5/v6 training sets) was rendered with the 1.0m near plane.
CAMERA_NEAR_CLIP = 0.2
# geometric_semantic_v2: depth at/below this is invalid. Just above CAMERA_NEAR_CLIP: with the camera at
# the URDF front_camera mount (head front) nothing of the robot is in view, and 0.2-0.35m returns are
# real terrain -- the next tread right in front while climbing (0.35 here, needed for the old behind-
# the-head mount, blanked those out: 35% of frames had >20% of pixels below 0.35m).
GEO_V2_MIN_RANGE = 0.21
GHOST_BASE_HEIGHT = 0.30  # m, VIPLANNER_GHOST base height above the ground (Go2 standing, as measured)
GO2_FOOT_RADIUS = 0.022  # m, FK gives the foot sphere's center; the sole is this much lower
# Explicit user request (2026-10-02): moved to the Go2 URDF's own front_camera_joint origin
# (0908/go2.urdf: xyz 0.32715 -0.00003 0.04297 from base), keeping the 8.6 deg downward tilt.
# The old (0.15, 0, 0.15) mount sat BEHIND the head: once CAMERA_NEAR_CLIP dropped to 0.2m the
# head itself filled the bottom of the depth image at ~0.3m (9% of frames >40% occluded). Data /
# models from before this (v3..v6_cont) used the old mount.
CAMERA_LOCAL_POS = np.array([0.32715, 0.0, 0.04297], dtype=np.float32)
CHASE_DIST = 1.0  # meters behind the dog, third-person viewport follow camera -- was 1.8, closer per request
CHASE_HEIGHT = 0.55  # meters above the dog's base -- was 0.9, scaled down with CHASE_DIST
_HALF_TILT = 0.075  # CAMERA_TILT/2
CAMERA_LOCAL_QUAT_WXYZ = np.array([math.cos(_HALF_TILT), 0.0, math.sin(_HALF_TILT), 0.0], dtype=np.float32)

# --- robot profile -------------------------------------------------------------------------------
# VIPLANNER_ROBOT selects the body + gait policy. Everything above is the Go2 profile (the default,
# so every existing scene script is unchanged). Explicit user request (2026-10-13): "leo", from the
# user's own files in viplanner2/1013 -- leo_description.urdf (converted to USD by
# training_data/tools/urdf_to_usd.py), policy_3800.pt and its env_config.json:
#   - same HIMLoco layout as Go2: 45x6 = 270 obs (commands, ang vel, gravity, dof pos, dof vel,
#     actions), 12 actions, dof order FL, FR, RL, RR x (hip, thigh, calf) -- per the leo deployment
#     config (rl_sar policy/leo/himloco/config.yaml), so JOINT_NAMES is shared
#   - control.stiffness 970, damping 26.2162, action_scale 0.25 with hip_reduction 0.5 on the hip
#     joints, default_joint_angles as below (right-side thigh/calf axes are mirrored in this URDF)
#   - rewards.base_height_target 0.58 m
ROBOT = _os_check.environ.get("VIPLANNER_ROBOT") or "go2"
# VIPLANNER_GUIDANCE (explicit user request, 2026-10-13): how much of the stairs guidance steers.
# In Stair.usdc seeking_stairs is True for the whole run, so with "full" the heading never comes
# from ViPlanner's own path and swapping models changes almost nothing -- these modes exist to
# evaluate the model itself.
#   full          (default) unchanged: stairs sub-goal + heading straight at that sub-goal
#   planner_steer ViPlanner is still GIVEN the stairs sub-goal, but the heading is its own path's
#   off           no sub-goal either: ViPlanner gets the real goal and steers by its own path
# Wall repulsion, virtual edge strips, centerline assist, fear slowdown and the stall watchdog
# stay on in every mode.
GUIDANCE = _os_check.environ.get("VIPLANNER_GUIDANCE") or "full"
# "perception" (2026-10-14): the stairs sub-goal comes from stair_perception.py -- the semantic image's
# stairs pixels + depth -- instead of the scene's aabbs; steering goes to that sub-goal as in "full".
# Still read from aabbs in this mode: wall / edge repulsion and the stall watchdog (safety net only).
if GUIDANCE not in ("full", "planner_steer", "off", "perception"):
    raise ValueError(f"unknown VIPLANNER_GUIDANCE {GUIDANCE!r} (expected full / planner_steer / off / perception)")
CAMERA_FAR_CLIP = None                    # m; None keeps the camera prim's own far clipping distance
# Extra cameras used ONLY to find stairs (GUIDANCE=perception); ViPlanner itself still sees the main
# camera only. (name, local xyz, local rpy [rad, URDF order], horizontal FOV deg), 640 x 360 each.
EXTRA_CAMERAS = []
# the robot's URDF, read by leg_odometry.py for the leg chain; None = no URDF in the container (Go2
# keeps its old built-in estimate)
LEG_URDF = None
LEG_TORQUE_LIMIT = 120.0   # N m, joint effort limit (growin URDF) -- the drive cannot push harder
ROBOT_STAND_HEIGHT = 0.30               # m, base above the ground when standing
ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH = 0.325, 0.14
if ROBOT == "leo":
    GAIT_POLICY_CKPT = f"{VIPLANNER2_ROOT}/1013/leo/policy_3800.pt"  # moved into 1013/leo by the user (2026-10-13)
    LEG_URDF = f"{VIPLANNER2_ROOT}/1013/leo/urdf/leo_description.urdf"
    DEFAULT_ANGLES = np.array(
        [0.0, 0.6, -1.3, 0.0, -0.6, 1.3, 0.0, 0.7, -1.3, 0.0, -0.7, 1.3], dtype=np.float32
    )
    KP, KD = 970.0, 26.2162
    ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float32)
    ROBOT_STAND_HEIGHT = 0.58
    ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH = 0.55, 0.27   # body x[-0.33, 0.51] y[+-0.23] + legs
    # leo's camera, per the user (2026-10-13): mounted at (0.51, 0.0, -0.08) from base, PORTRAIT
    # 600 x 960 (W x H) image, range 0.2 - 20 m, level (no downward tilt), FOV 78 deg horizontal x
    # 105 deg vertical (first read as landscape 960 x 600 / 105 x 78; the user corrected it). The
    # two agree with square pixels: 600 / (2 tan 39 deg) = 370.5 px -> 104.7 deg over 960 rows.
    CAMERA_LOCAL_POS = np.array([0.51, 0.0, -0.08], dtype=np.float32)
    CAMERA_LOCAL_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    CAMERA_HFOV_DEG = 78.0
    CAMERA_RES = (600, 960)
    CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP = 0.2, 20.0
    # viewport chase camera (watching only): leo is over twice Go2's size and filled the view at
    # Go2's 1.0 m / 0.55 m (explicit user request: "視角要後退一些它太大了")
    CHASE_DIST, CHASE_HEIGHT = 2.0, 1.05   # was 2.5 / 1.3, user asked for a little closer
elif ROBOT == "growin":
    # Explicit user request (2026-10-13): second user-supplied robot, viplanner2/1013/growin_description
    # (urdf/growin.urdf -> USD by training_data/tools/urdf_to_usd.py; weight/policy_1001_111828_8940.pt
    # and its env_config.json). Same HIMLoco layout again (270 obs, 12 actions). The URDF lists the legs
    # FL, RL, FR, RR, but Isaac Gym orders sibling joints by name, so the policy's dof order is still
    # FL, FR, RL, RR = JOINT_NAMES (indices are looked up by name from the articulation anyway).
    #   - control.stiffness 1100, damping 25, action_scale 0.25 with hip_reduction 0.5
    #   - default_joint_angles: thigh -0.6 / +0.6 front, -0.7 / +0.7 rear (left / right: the thigh axes
    #     are mirrored), every calf -1.3
    #   - init_state.pos z 0.55, rewards.base_height_target 0.53
    GAIT_POLICY_CKPT = f"{VIPLANNER2_ROOT}/1013/growin_description/weight/policy_1001_111828_8940.pt"
    LEG_URDF = f"{VIPLANNER2_ROOT}/1013/growin_description/urdf/growin.urdf"
    DEFAULT_ANGLES = np.array(
        [0.0, -0.6, -1.3, 0.0, 0.6, -1.3, 0.0, -0.7, -1.3, 0.0, 0.7, -1.3], dtype=np.float32
    )
    KP, KD = 1100.0, 25.0
    ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float32)
    ROBOT_STAND_HEIGHT = 0.53
    ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH = 0.52, 0.30   # body x[-0.50, 0.50] y[+-0.14], legs out to y ~0.29
    # cam_front from the package's own xacro/cameras_real.xacro: xyz 0.49539 0 0.105, rpy 0 0 0,
    # 640 x 360, horizontal FOV 1.5184 rad (87 deg). Its min/max depth there are 0.52 / 6.0 m; NOT
    # applied yet -- a 0.52 m near plane makes anything closer transparent in the render (the
    # see-through-the-floor problem CAMERA_NEAR_CLIP=0.2 was set for), so this keeps 0.2 until the
    # user decides.
    CAMERA_LOCAL_POS = np.array([0.49539, 0.0, 0.105], dtype=np.float32)
    CAMERA_LOCAL_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    CAMERA_HFOV_DEG = 87.0
    CAMERA_RES = (640, 360)
    CHASE_DIST, CHASE_HEIGHT = 2.0, 1.05
    # the package's other four cameras (same xacro), user request 2026-10-14: the front camera alone
    # kept losing the next flight (it is usually beside or behind the robot in a switchback stairwell,
    # and a flight going DOWN is barely visible to a level camera)
    EXTRA_CAMERAS = [
        ("front_bottom", (0.485641, 0.0, 0.015), (0.0, 0.7854, 0.0), 87.0),
        ("left_bottom", (0.0, 0.121747, 0.015), (0.0, 0.7854, 1.5708), 87.0),
        ("right_bottom", (0.0, -0.121747, 0.015), (0.0, 0.7854, -1.5708), 87.0),
        ("rear", (-0.474, 0.0, 0.100), (0.0, 0.0, 3.1416), math.degrees(2.1013)),
    ]
elif ROBOT != "go2":
    raise ValueError(f"unknown VIPLANNER_ROBOT {ROBOT!r} (expected 'go2', 'leo' or 'growin')")
# The guidance logic's height tests (standing on a tread? topped out? climbable?) are all written for
# a base riding GHOST_BASE_HEIGHT (0.30 m, Go2) above what it stands on. A taller robot's base height
# is shifted down by this much before the logic sees it; the camera is put back at the true height.
ROBOT_LOGIC_Z_OFFSET = ROBOT_STAND_HEIGHT - 0.30


def quat_conj_wxyz(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_wxyz(q, v):
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float32)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (w * uv + uuv)


def quat_rotate_inverse_wxyz(q, v):
    return quat_rotate_wxyz(quat_conj_wxyz(q), v)


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def reproject_depth_to_camera_local(depth: np.ndarray, hfov_deg: float = 87.0) -> np.ndarray:
    """Explicit user request ("用這個方式好了" -- real frame-to-frame visual odometry instead of the
    simulator's privileged base_pos[2] for height judgment): same pinhole math as
    camera_semantics_lab.reproject_depth_to_world, but stops BEFORE applying any camera pose --
    returns (H,W,3) points in the camera's own local frame (+X forward, +Y left, +Z up, matching
    that function's convention). This is exactly what a real depth sensor hands you directly, with
    no world-frame pose assumed at all -- the thing a real robot actually has."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    hfov = np.radians(hfov_deg)
    fx = W / (2 * np.tan(hfov / 2))
    fy = fx
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    with np.errstate(invalid="ignore"):
        x = (us - cx) * depth / fx
        y = (vs - cy) * depth / fy
        p_cam = np.stack([depth, -x, -y], axis=-1)
    return p_cam


def icp_point_to_point(source, target, max_iterations=15, tol=1e-5, trim_fraction=0.3):
    """Pure-numpy point-to-point ICP (Kabsch/SVD solve each iteration), with trimmed-ICP style
    outlier rejection -- after each nearest-neighbor correspondence pass, only the closest
    `1-trim_fraction` fraction of matches are used to solve for the rigid transform. This matters
    specifically because smoke-degraded depth (_apply_smoke_degradation) injects spurious random-
    depth dropout points that would otherwise corrupt the alignment; without trimming, a single bad
    correspondence pass can drag the whole solved transform off. source/target: (N,3)/(M,3) arrays,
    already filtered to finite values. Returns (R (3,3), t (3,)) such that for a source point p,
    R @ p + t lands at its corresponding point in target's own frame -- i.e. the rigid transform
    mapping the source cloud's frame into the target cloud's frame (used here as: source = this
    tick's camera-local points, target = last tick's -- so t is the camera's own displacement,
    expressed in last tick's camera-local frame)."""
    src = source.copy()
    R_total = np.eye(3, dtype=np.float32)
    t_total = np.zeros(3, dtype=np.float32)
    prev_error = None
    n_keep = max(10, int(len(src) * (1.0 - trim_fraction)))
    for _ in range(max_iterations):
        dists = np.linalg.norm(src[:, None, :] - target[None, :, :], axis=-1)
        nn_idx = np.argmin(dists, axis=1)
        nn_dist = dists[np.arange(len(src)), nn_idx]
        keep = np.argsort(nn_dist)[:n_keep]
        matched = target[nn_idx[keep]]
        src_keep = src[keep]
        mean_error = float(nn_dist[keep].mean())
        src_centroid = src_keep.mean(axis=0)
        tgt_centroid = matched.mean(axis=0)
        src_c = src_keep - src_centroid
        tgt_c = matched - tgt_centroid
        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, float(d)]).astype(np.float32)
        R = (Vt.T @ D @ U.T).astype(np.float32)
        t = (tgt_centroid - R @ src_centroid).astype(np.float32)
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = (R @ t_total) + t
        if prev_error is not None and abs(prev_error - mean_error) < tol:
            break
        prev_error = mean_error
    return R_total, t_total


def make_pointcloud2(points_xyz, frame_id="world", stamp=None):
    """points_xyz: (N,3) float array, world frame. Returns a sensor_msgs/PointCloud2 message."""
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    pts = np.asarray(points_xyz, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    msg = PointCloud2()
    msg.header = Header()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def make_image_msg(rgb_hwc, frame_id="camera", stamp=None):
    """rgb_hwc: (H,W,3) uint8. Returns a sensor_msgs/Image (rgb8). Downsampled 2x2 to stay well
    under this network's CycloneDDS MaxMessageSize=65500B (320x180 full-size would be 172800B)."""
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    img = np.asarray(rgb_hwc, dtype=np.uint8)[::2, ::2]
    msg = Image()
    msg.header = Header()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = img.shape[1] * 3
    msg.data = img.tobytes()
    return msg


def make_depth_image_msg(depth, near, far, frame_id="camera", stride=2):
    """depth: (H,W) float metres. Returns a grayscale (explicit user request) rgb8 sensor_msgs/Image:
    near = white ... far = dark grey, invalid (non-finite / outside near..far) = black. `stride`
    subsamples for a lighter message."""
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    valid = np.isfinite(d) & (d >= near) & (d <= far)
    t = np.clip((np.where(valid, d, far) - near) / max(far - near, 1e-6), 0.0, 1.0) ** 0.5  # sqrt: more contrast up close
    g = (255.0 - 215.0 * t).astype(np.uint8)  # 255 at near .. 40 at far, so far stays distinct from invalid
    g[~valid] = 0
    rgb = np.repeat(g[..., None], 3, axis=2)
    return make_image_msg(np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1), frame_id=frame_id)


def _route_marker(ns, mid, mtype, scale, rgba=(1.0, 1.0, 1.0, 1.0), points=None, colors=None, pos=None, action=0):
    """One visualization_msgs/Marker in the world frame (LINE_STRIP=4, LINE_LIST=5, SPHERE=2; action 3 = DELETEALL)."""
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker

    m = Marker()
    m.header.frame_id = "world"
    m.ns, m.id, m.type, m.action = ns, int(mid), int(mtype), int(action)
    m.pose.orientation.w = 1.0
    if pos is not None:
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(pos[0]), float(pos[1]), float(pos[2])
    sx, sy, sz = scale if isinstance(scale, (tuple, list)) else (scale, scale, scale)
    m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
    m.color = ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))
    if points is not None:
        m.points = [Point(x=float(q[0]), y=float(q[1]), z=float(q[2])) for q in points]
    if colors is not None:
        m.colors = [ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=1.0) for c in colors]
    return m


_ROUTE_Z_ANCHORS = np.array([[0.27, 0.0, 0.33], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55], [0.37, 0.79, 0.38], [0.99, 0.91, 0.14]])


def _route_height_colors(z, z_lo, z_hi):
    """Same idea as plot_ghost_paths.py: path colored by height (purple = low ... yellow = high)."""
    t = np.clip((np.asarray(z, dtype=np.float64) - z_lo) / max(z_hi - z_lo, 1e-6), 0.0, 1.0) * (len(_ROUTE_Z_ANCHORS) - 1)
    i0 = np.minimum(t.astype(np.int64), len(_ROUTE_Z_ANCHORS) - 2)
    return _ROUTE_Z_ANCHORS[i0] + (_ROUTE_Z_ANCHORS[i0 + 1] - _ROUTE_Z_ANCHORS[i0]) * (t - i0)[:, None]


async def run_walk(stage, app, dog_prim_path, goal_pos, obstacle_boxes, floor_box, max_steps=13000, precomputed_aabbs=None,
                    guardrail_boxes=None, collect_data_dir=None, collect_episodes=0, smoke_zone_z_min=None,
                    block_first_flight_after_wp=3):
    """block_first_flight_after_wp: waypoint index (0-based) from which "Stair A" -- the first
    stairs flight, identified as every stairs box sharing the first box's y-center -- is relabelled
    WALL so the robot doesn't head back to it (open_scene_stairs.py's two-flight layout, where Stair
    A runs along X). None disables that relabelling (and the matching _stair_a_cleared latch)
    entirely. Explicit user request (2026-10-02): Stair.usdc's flights all run along Y, so the y-center
    match instead picked the first tread of Flight1/Flight3 and the top tread of Flight2/Flight4 and,
    on the last leg (wp_idx 3), turned Flight4's top step into a wall in both ViPlanner's semantic
    input and the stairs logic -- open_scene_stair_usd.py passes None.

    collect_data_dir/collect_episodes: when collect_data_dir is set, saves every replan tick's
    depth+semantic image and camera pose in the exact directory layout viplanner_core's own
    PlannerDataGenerator expects (depth/N.npy, semantics/N.png, camera_extrinsic.txt,
    intrinsics.txt), for offline fine-tuning on this scene. Stops after collect_episodes full
    episodes (goal reached or timeout either way -- a timed-out episode's early/mid-route samples
    are still valid training data, just not a full successful trajectory).

    smoke_zone_z_min: explicit user request to simulate smoke on the destination floor and see it
    actually affect navigation, not just look hazy in a render. Real IR depth sensors (this
    project's own camera, and the physical Go2's) get scattered by smoke/fog particles in the air:
    range readings get noisier the further the beam travels through the haze, and some fraction of
    pixels return a short, spurious distance instead of the true one (backscatter off nearby
    particles rather than the real surface). When set (a world Z height), once the robot's own base
    z reaches or exceeds it, every replan's depth frame gets this same kind of degradation applied
    BEFORE it reaches ViPlanner -- see _apply_smoke_degradation and its call site below -- so the
    network is actually reasoning from corrupted depth, the same as it would need to in real smoke,
    rather than a cosmetic-only particle effect that never reaches the input the robot decides
    from (this project's model uses depth+semantic, not RGB, so a purely visual effect would
    change nothing about its behavior)."""
    from pxr import UsdPhysics
    import omni.timeline
    from isaacsim.core.prims import Articulation
    from isaacsim.sensors.camera import Camera
    from isaacsim.util.debug_draw import _debug_draw
    from isaacsim.core.utils.viewports import set_camera_view
    draw_iface = _debug_draw.acquire_debug_draw_interface()

    # VIPLANNER_FREE_CAMERA=1 skips the chase-cam viewport override below, so a user can freely
    # orbit/pan the viewport themselves while the control loop runs. Off by default to keep the
    # existing chase-view behavior for callers that rely on it (e.g. the stairs scene pipeline).
    _free_camera = _os_check.environ.get("VIPLANNER_FREE_CAMERA") == "1"

    # IMPORTANT: PhysX must claim its own CUDA/GPU context before PyTorch touches CUDA at all --
    # creating a CUDA tensor (even a bare torch.zeros(..., device="cuda")) before timeline.play()
    # makes the very first physics step hang indefinitely (confirmed by isolated testing this
    # session: reproducible 100%-CPU, zero-log-output hang). So: physics scene setup + play +
    # warm-up happens FIRST, and only THEN do we import/touch anything that initializes CUDA via
    # torch (the policy checkpoint, ViPlanner model, or even torch.cuda.is_available()).

    # --- physics scene ---
    has_scene = any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse())
    if not has_scene:
        print("[viplanner] no PhysicsScene found -> creating /World/PhysicsScene")
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    else:
        print("[viplanner] PhysicsScene already present in stage")
    # VIPLANNER_PHYSICS_HZ: physics steps per second. Unset = PhysX default 60 Hz, one step per control
    # step -- the gait was trained at 500 Hz (dt 0.002, decimation 10), and at 60 Hz the feet of
    # growin slid so much that only ~1.2 of 4 were ever still (found by leg odometry, 2026-10-15)
    _physics_hz = _os_check.environ.get("VIPLANNER_PHYSICS_HZ")
    try:
        from pxr import PhysxSchema
        for _p in stage.Traverse():
            if _p.IsA(UsdPhysics.Scene):
                if _physics_hz:
                    PhysxSchema.PhysxSceneAPI.Apply(_p).CreateTimeStepsPerSecondAttr().Set(int(_physics_hz))
                _a = PhysxSchema.PhysxSceneAPI(_p).GetTimeStepsPerSecondAttr()
                print(f"[viplanner] physics scene {_p.GetPath()}: timeStepsPerSecond="
                      f"{_a.Get() if _a and _a.HasAuthoredValue() else 'default'}")
    except Exception as _e:
        print("[viplanner] physics rate check failed:", repr(_e))

    # VIPLANNER_CONTROL_HZ (default 50): the gait runs once per app update, and each update advanced
    # the sim 1/60 s (Kit's default) -- the gait was trained at 50 Hz (user: "步態改回50Hz",
    # 2026-10-16). Same calls as SimulationContext.set_simulation_dt(rendering_dt=...): each update
    # now advances exactly 1/CONTROL_HZ s (10 physics steps at PHYSICS_HZ=500). "0" keeps Kit's own.
    # Step-count timings (REPLAN_EVERY_N_STEPS, COMMAND_HOLD_STEPS, stair-map ticks) stretch by 60/50.
    _control_hz = float(_os_check.environ.get("VIPLANNER_CONTROL_HZ", "50") or 0)
    if _control_hz > 0:
        try:
            import carb
            import omni.kit.loop._loop as _omni_loop
            from pxr import Usd
            _settings = carb.settings.get_settings()
            _tl = omni.timeline.get_timeline_interface()
            if _settings.get("/app/runLoops/main/rateLimitEnabled"):
                _settings.set("/app/runLoops/main/rateLimitFrequency", int(_control_hz))
                _tl.set_target_framerate(_control_hz)
            with Usd.EditContext(stage, stage.GetRootLayer()):
                stage.SetTimeCodesPerSecond(_control_hz)
            _tl.set_time_codes_per_second(_control_hz)
            _loop = _omni_loop.acquire_loop_interface()
            _loop.set_manual_step_size(1.0 / _control_hz)
            _loop.set_manual_mode(True)
            print(f"[viplanner] control rate: each app update advances {1.0 / _control_hz:.4f} s ({_control_hz:.0f} Hz)")
        except Exception as _e:
            print("[viplanner] control rate setup failed:", repr(_e))

    # Explicit user request ("好" -> real contact sensing for leg odometry, then "改用更底層的
    # PhysX 接觸 API 再試一次" after FOUR straight failures on isaacsim.core.prims.RigidPrim's
    # tensor-based contact view -- every combination tried (construction timing relative to
    # timeline.play(), schema-application timing, empty vs non-empty contact_filter_prim_paths_expr,
    # max_contact_count=0 vs 4) hit the exact same internal AttributeError (`sensor_count` on a None
    # physics_view), pointing at something about this API/build combination rather than any of our
    # own parameters. Switched to the older, prim-based `isaacsim.sensors.physics.ContactSensor` --
    # creates one real IsaacContactSensor prim per foot (via the same
    # "IsaacSensorCreateContactSensor" kit command countless existing Isaac Sim robot examples use),
    # queried per-step via get_current_frame()["in_contact"]/["force"] instead of a batched tensor
    # view -- structurally simpler, and isaacsim.sensors.physics is already a loaded extension in
    # this app (isaacsim.exp.base.kit's own dependency list), so no extra extension-enabling risk.
    # Created before timeline.play(), same schema-before-PhysX-parse lesson as everything else in
    # this function.
    from isaacsim.sensors.physics import ContactSensor
    foot_contact_sensors = {}
    for _leg in _LEG_ORDER:
        foot_contact_sensors[_leg] = ContactSensor(
            prim_path=f"{dog_prim_path}/{_leg}_foot/Contact_Sensor",
            name=f"{_leg}_contact_sensor",
            min_threshold=0,
            max_threshold=1000000,
            radius=-1,  # -1: whole collision shape, not a sub-region
        )
    print("[viplanner] created 4 foot ContactSensor prims (pre-play)")

    # isaacsim.exp.full.kit starts with async rendering on (throttling enable_async = true) and the
    # throttling extension turns it off from inside the PLAY event. With growin in the box-built L / U
    # scenes that set spun forever in libomni.usd during the first app update (py-spy, 2026-10-21;
    # every render thread asleep). Turn it off here, timeline still stopped, so PLAY has nothing to switch.
    try:
        import carb
        _settings = carb.settings.get_settings()
        _settings.set("/exts/isaacsim.core.throttling/enable_async", False)
        _settings.set("/app/asyncRendering", False)
        _settings.set("/app/asyncRenderingLowLatency", False)
        for _ in range(2):
            await app.next_update_async()
        print("[viplanner] async rendering off before play")
    except Exception as _e:
        print("[viplanner] async rendering setup failed:", repr(_e))

    print("[viplanner] calling timeline.play() ...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[viplanner] timeline.play() returned, doing 5 app updates ...")
    for i in range(5):
        await app.next_update_async()
        print("[viplanner]   app update", i + 1, "/5 done")

    print("[viplanner] constructing Articulation view ...")
    dog_art = Articulation(prim_paths_expr=dog_prim_path, name="go2_articulation")
    print("[viplanner] calling dog_art.initialize() ...")
    dog_art.initialize()
    print("[viplanner] articulation dof_names:", dog_art.dof_names)
    # DOF indices of JOINT_NAMES, taken from dof_names. Passing joint_names= instead resolves indices
    # over ALL joints, fixed ones included: fine for go2.usd, but the leo USD keeps its fixed joints
    # (feet, base_joint, imu) and some revolute joints then got an index >= 12 ("index 12 is out of
    # bounds for axis 1 with size 12").
    _JOINT_IDX = np.array([list(dog_art.dof_names).index(n) for n in JOINT_NAMES], dtype=np.int64)

    for _leg, _cs in foot_contact_sensors.items():
        _cs.initialize()
    print("[viplanner] foot ContactSensor prims initialized")
    # true foot world positions, recorded only with VIPLANNER_LEGODOM_DUMP to check the URDF leg
    # kinematics against the simulated robot
    _foot_view = None
    if _os_check.environ.get("VIPLANNER_LEGODOM_DUMP"):
        try:
            from isaacsim.core.prims import XFormPrim
            _foot_view = XFormPrim([f"{dog_prim_path}/{_leg}_foot" for _leg in _LEG_ORDER], name="feet_truth")
            print("[viplanner] foot truth view:", _foot_view.count)
        except Exception as _e:
            print("[viplanner] WARNING: foot truth view unavailable:", repr(_e))

    dog_art.set_joint_positions(DEFAULT_ANGLES[None, :], joint_indices=_JOINT_IDX)
    dog_art.set_joint_position_targets(DEFAULT_ANGLES[None, :], joint_indices=_JOINT_IDX)
    dog_art.set_gains(
        kps=np.full((1, 12), KP, dtype=np.float32),
        kds=np.full((1, 12), KD, dtype=np.float32),
        joint_indices=_JOINT_IDX,
    )
    for _ in range(10):
        await app.next_update_async()

    # --- camera ---
    camera = Camera(prim_path="/World/DogCamera", name="dog_camera", resolution=CAMERA_RES)
    camera.initialize()
    horizontal_aperture = 20.955
    focal_length = horizontal_aperture / (2 * math.tan(math.radians(CAMERA_HFOV_DEG) / 2))
    camera.set_horizontal_aperture(horizontal_aperture)
    camera.set_focal_length(focal_length)
    camera.set_clipping_range(near_distance=CAMERA_NEAR_CLIP, far_distance=CAMERA_FAR_CLIP)
    camera.add_distance_to_image_plane_to_frame()
    # extra stair-finding cameras (see EXTRA_CAMERAS); VIPLANNER_EXTRA_CAMS=0 turns them off
    _extra_cams = []
    if GUIDANCE == "perception" and EXTRA_CAMERAS and _os_check.environ.get("VIPLANNER_EXTRA_CAMS", "1") != "0":
        for _name, _xyz, _rpy, _hfov in EXTRA_CAMERAS:
            _cam = Camera(prim_path=f"/World/DogCam_{_name}", name=f"dog_cam_{_name}", resolution=(640, 360))
            _cam.initialize()
            _cam.set_horizontal_aperture(horizontal_aperture)
            _cam.set_focal_length(horizontal_aperture / (2 * math.tan(math.radians(_hfov) / 2)))
            _cam.set_clipping_range(near_distance=CAMERA_NEAR_CLIP, far_distance=CAMERA_FAR_CLIP or 20.0)
            _cam.add_distance_to_image_plane_to_frame()
            _r, _p, _y = _rpy  # URDF rpy -> wxyz, R = Rz(y) Ry(p) Rx(r)
            _q = quat_mul_wxyz(quat_mul_wxyz(
                np.array([math.cos(_y / 2), 0.0, 0.0, math.sin(_y / 2)], dtype=np.float32),
                np.array([math.cos(_p / 2), 0.0, math.sin(_p / 2), 0.0], dtype=np.float32)),
                np.array([math.cos(_r / 2), math.sin(_r / 2), 0.0, 0.0], dtype=np.float32))
            _extra_cams.append({"name": _name, "cam": _cam, "pos": np.array(_xyz, dtype=np.float32),
                                "quat": np.asarray(_q, dtype=np.float32), "hfov": _hfov})
        print(f"[viplanner] extra stair cameras: {[c['name'] for c in _extra_cams]}")
    _extra_obs = []  # latest (sem, world_xyz) per extra camera, flattened + subsampled
    for _ in range(10):
        await app.next_update_async()

    # --- data collection setup (for offline fine-tuning on this scene) ---
    collecting = collect_data_dir is not None and collect_episodes > 0
    if collecting:
        import os as _os
        from PIL import Image as _PILImage
        _depth_dir = _os.path.join(collect_data_dir, "depth")
        _sem_dir = _os.path.join(collect_data_dir, "semantics")
        _os.makedirs(_depth_dir, exist_ok=True)
        _os.makedirs(_sem_dir, exist_ok=True)
        # VIPLANNER_SAVE_RGB=1 (SAVE_RGB=1 in run_stair_usd.sh): also save the camera's colour image as
        # rgb/N.png (same index as depth/semantics) -- input for testing Mask2Former offline.
        _save_rgb = _os.environ.get("VIPLANNER_SAVE_RGB") == "1"
        _rgb_dir = _os.path.join(collect_data_dir, "rgb")
        if _save_rgb:
            _os.makedirs(_rgb_dir, exist_ok=True)
        _extrinsic_path = _os.path.join(collect_data_dir, "camera_extrinsic.txt")
        # Truncate up front instead of relying on the per-sample write below being append-only onto
        # an already-clean file. Confirmed via a real run: a killed/restarted collection run (e.g.
        # after fixing an unrelated bug mid-collection) leaves stale rows at the top of this file
        # from the aborted attempt, while depth/N.npy and semantics/N.png get cleanly overwritten
        # by the restart (same filenames, same _sample_idx starting back at 0) -- silently
        # misaligning every extrinsic row against the wrong image from then on, surfacing much
        # later as an opaque IndexError deep in PlannerDataGenerator during training, not here.
        open(_extrinsic_path, "w").close()
        _sample_idx = 0
        # fx,fy,cx,cy match reproject_depth_to_world's own pinhole model exactly, so the training
        # pipeline's geometry (loaded from this file) is consistent with what generated the
        # semantic images in the first place. Two identical rows: depth_suffix=sem_suffix="" means
        # PlannerDataGenerator reads intrinsics for both from the same file, but it still expects
        # two rows (one per camera, even when they're the same physical camera here).
        _cx, _cy = CAMERA_RES[0] / 2.0, CAMERA_RES[1] / 2.0
        _fx = CAMERA_RES[0] / (2 * math.tan(math.radians(CAMERA_HFOV_DEG) / 2))
        _fy = _fx
        _p_row = f"{_fx},0,{_cx},0,0,{_fy},{_cy},0,0,0,1,0"
        with open(_os.path.join(collect_data_dir, "intrinsics.txt"), "w") as f:
            f.write(_p_row + "\n" + _p_row + "\n")
        print(f"[viplanner] DATA COLLECTION mode: saving to {collect_data_dir}, {collect_episodes} episodes")

    # --- NOW it's safe to touch CUDA/torch: PhysX already has the GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[viplanner] device:", device)

    from viplanner_bridge_isaacsim import VIPlannerAlgo
    from camera_semantics_lab import (
        build_lab_aabbs, semantic_color_image_with_points, reproject_depth_to_world, depth_from_aabbs,
        SEG_ID_WALL, SEG_ID_STAIRS, SEG_ID_FLOOR, SEG_ID_RAILING,
    )
    # Real-hardware feasibility test, per explicit user request ("我要它實機也可以" -> "好試試看"):
    # this is viplanner_core's OWN real-deployment path (see that module's docstring) -- a pure
    # depth-only floor/obstacle classifier with no scene ground truth and no learned model, meant
    # to drop straight into VIPlannerAlgo.plan_dual in place of the ground-truth semantic_camera
    # this sim otherwise uses. Toggled by env var so the existing ground-truth path (which all the
    # other control-loop logic below -- wall-repulsion, virtual edge walls, seeking_stairs -- still
    # uses regardless of this flag, since this test is only about whether the NETWORK can navigate
    # from this input, not yet about porting every safety net) stays the default.
    _USE_GEOMETRIC_SEMANTIC = _os_check.environ.get("VIPLANNER_USE_GEOMETRIC_SEMANTIC") == "1"
    # Phase 1/2 of the real-robot plan (explicit user request, 2026-10-02): depth-only floor / stairs /
    # obstacle semantics from geometric_semantic_v2.py (no aabbs), referenced to the feet via leg
    # forward kinematics. Replaces `sem` (ViPlanner's input + what data collection saves) only; the
    # control-loop guidance / safety logic still reads aabbs until phase 3.
    _USE_GEO_V2 = _os_check.environ.get("VIPLANNER_GEOMETRIC_SEMANTIC_V2") == "1"
    # VIPLANNER_M2F=1 (M2F=1 in run_stair_usd.sh): `sem` comes from Mask2Former on the camera's colour
    # image, via training_data/tools/m2f_server.py running on the host. Like GEO_V2 it replaces
    # ViPlanner's input (+ saved data + /go2/semantic_view) only; guidance / safety still read aabbs.
    _USE_M2F = _os_check.environ.get("VIPLANNER_M2F") == "1"
    _m2f_url = _os_check.environ.get("VIPLANNER_M2F_URL") or "http://127.0.0.1:8765/seg"
    _m2f_stat = {"ok": 0, "fail": 0, "ms": 0.0}

    def _m2f_segment(fallback, cam=None):
        """Mask2Former semantic colour image for the current camera frame; `fallback` if unavailable."""
        import time as _time
        import urllib.request as _url
        try:
            _rgba = (cam or camera).get_rgba()
            if _rgba is None or getattr(_rgba, "ndim", 0) != 3:
                raise RuntimeError("no rgb frame yet")
            _rgb = np.ascontiguousarray(np.asarray(_rgba)[..., :3].astype(np.uint8))
            _h, _w = _rgb.shape[:2]
            _t0 = _time.time()
            _req = _url.Request(_m2f_url, data=_rgb.tobytes(), headers={"X-H": str(_h), "X-W": str(_w)})
            with _url.build_opener(_url.ProxyHandler({})).open(_req, timeout=5.0) as _r:
                _out = np.frombuffer(_r.read(), dtype=np.uint8).reshape(_h, _w, 3).copy()
            _m2f_stat["ok"] += 1
            _m2f_stat["ms"] += (_time.time() - _t0) * 1000.0
            if _m2f_stat["ok"] % 200 == 0:
                print(f"[viplanner] m2f: {_m2f_stat['ok']} frames, mean {_m2f_stat['ms'] / _m2f_stat['ok']:.0f} ms "
                      f"round trip, {_m2f_stat['fail']} failed")
            return _out
        except Exception as e:
            _m2f_stat["fail"] += 1
            if _m2f_stat["fail"] <= 3 or _m2f_stat["fail"] % 200 == 0:
                print(f"[viplanner] WARNING: m2f request failed ({_m2f_stat['fail']}x), using ground-truth "
                      f"semantics for this frame:", repr(e))
            return fallback

    if _USE_M2F:
        print("[viplanner] semantics from Mask2Former server:", _m2f_url)
    _leg_odom = None
    if LEG_URDF:
        try:
            from leg_odometry import LegHeightOdometry
            _leg_odom = LegHeightOdometry(LEG_URDF)
            print("[viplanner] leg odometry from", LEG_URDF)
        except Exception as _e:
            print("[viplanner] WARNING: leg odometry unavailable:", repr(_e))
    # VIPLANNER_HEIGHT=odom: GUIDANCE=perception takes the feet height from leg odometry instead of
    # the simulator (what the real robot has to do); default "truth" until the estimate is trusted
    _HEIGHT_FROM_ODOM = _os_check.environ.get("VIPLANNER_HEIGHT", "truth") == "odom"
    _legodom_dump_path = None
    if _os_check.environ.get("VIPLANNER_LEGODOM_DUMP"):
        _legodom_dump_path = f"{CONTAINER_HOME}/training_data/legodom/{_os_check.environ['VIPLANNER_LEGODOM_DUMP']}"
        _os_check.makedirs(_os_check.path.dirname(_legodom_dump_path), exist_ok=True)
    if _HEIGHT_FROM_ODOM and _leg_odom is None:
        raise RuntimeError("VIPLANNER_HEIGHT=odom needs a leg URDF for this robot (LEG_URDF)")
    if GUIDANCE == "perception":
        # stair_map.StairMap (2026-10-14, user: "改成樓梯地圖"): every flight seen is kept in a map and
        # the next one is picked from it; stair_perception.StairSeeker (frame-by-frame) is the old one
        from stair_map import StairMap
        _stair_seeker = StairMap()
        _percep_state = "none"
        _PERCEP_CLIMB_GAP = 0.8  # m, goal this far above the feet = on another level -> look for stairs
        _PERCEP_ODOM_REL_ERR = 0.10  # leg odometry error per metre of height travelled (HEIGHT=odom)
        # VIPLANNER_PERCEP_DUMP=<dir name under training_data> saves frames for offline debugging
        _percep_dump_dir = _os_check.environ.get("VIPLANNER_PERCEP_DUMP")
        if _percep_dump_dir:
            _percep_dump_dir = f"{CONTAINER_HOME}/training_data/{_percep_dump_dir}"
            _os_check.makedirs(_percep_dump_dir, exist_ok=True)
        _PERCEP_DESCEND_MAX_VX = 0.3  # m/s cap while heading down a flight (gait threshold is ~0.24)
        _PERCEP_LEVEL_TICKS = 12  # replan ticks of level body needed before trusting "arrived on the goal's level"
    # VIPLANNER_PROXY_DEPTH=1 (2026-10-06, combine_without_BlendersWall scene): the visible geometry is a
    # NuRec Gaussian splat, which Isaac Sim 5.1 renders but leaves out of the depth output -- so depth is
    # ray-cast from the scene's collision-proxy aabbs instead of read from the camera.
    _USE_PROXY_DEPTH = _os_check.environ.get("VIPLANNER_PROXY_DEPTH") == "1"

    def _get_depth(cam_pos, cam_quat_wxyz):
        if _USE_PROXY_DEPTH:
            return depth_from_aabbs(CAMERA_RES[0], CAMERA_RES[1], cam_pos, wxyz_to_xyzw(cam_quat_wxyz), aabbs,
                                    CAMERA_HFOV_DEG, CAMERA_NEAR_CLIP)
        return camera.get_depth()
    # Explicit user request (2026-10-02): simulate real-robot sensor/compute rates instead of the
    # lock-step "grab depth, build semantics, plan" every REPLAN_EVERY_N_STEPS. VIPLANNER_RATES="d,s,p"
    # (Hz, SIMULATION time): depth frames at d, semantics at s (from the latest depth frame and the
    # pose it was captured at), ViPlanner at p using the latest depth + latest -- possibly older --
    # semantics, the way an asynchronous sensor / segmentation / planner pipeline sees them.
    # Unset -> unchanged lock-step behaviour. Not meant for data collection (a sample's semantics
    # would not match its depth).
    # Explicit user request (2026-10-02): "ghost dog" -- judge ViPlanner's paths without the gait as a
    # variable. Everything up to the final smoothed (vx, wz) command runs unchanged (ViPlanner, the
    # planning_goal guidance, wall push, stall recovery); instead of the gait policy executing that
    # command, the base is moved kinematically: yaw += wz*dt, xy += vx*dt along the heading, body kept
    # level, height = ground under it (physics raycast) + GHOST_BASE_HEIGHT, legs held in the default
    # stance. Actual path + every planned path are saved per episode to VIPLANNER_GHOST_LOG.
    _GHOST = _os_check.environ.get("VIPLANNER_GHOST") == "1"
    # explicit user request ("幽靈狗我們飄高一些"): float height above the ground is configurable
    _ghost_h = float(_os_check.environ.get("VIPLANNER_GHOST_HEIGHT") or ROBOT_STAND_HEIGHT)
    # how far the true base rides above the 0.30 m the guidance logic assumes (see ROBOT_LOGIC_Z_OFFSET)
    _logic_z_off = (_ghost_h if _GHOST else ROBOT_STAND_HEIGHT) - GHOST_BASE_HEIGHT
    _ghost_log_dir = None
    if _GHOST:
        import time as _time_mod
        import carb
        from omni.physx import get_physx_scene_query_interface
        _ghost_sq = get_physx_scene_query_interface()
        _ghost_log_dir = _os_check.environ.get("VIPLANNER_GHOST_LOG") or \
            f"/home/isaacsim/training_data/ghost_logs/{_time_mod.strftime('%Y%m%d_%H%M%S')}"
        _os_check.makedirs(_ghost_log_dir, exist_ok=True)
        print(f"[viplanner] VIPLANNER_GHOST=1 -- kinematic base {_ghost_h:.2f}m above ground, no gait; "
              f"paths -> {_ghost_log_dir}")

    def _ghost_ground_z(x, y, z_from):
        """Highest surface under (x, y) below z_from, ignoring the dog's own collision shapes."""
        best = [None]

        def _hit(h):
            if not str(h.collision).startswith(dog_prim_path) and (best[0] is None or h.distance < best[0]):
                best[0] = h.distance
            return True
        _ghost_sq.raycast_all(carb.Float3(float(x), float(y), float(z_from)), carb.Float3(0.0, 0.0, -1.0), 5.0, _hit)
        return None if best[0] is None else z_from - best[0]

    _RATES = None
    if _os_check.environ.get("VIPLANNER_RATES"):
        _RATES = tuple(float(v) for v in _os_check.environ["VIPLANNER_RATES"].split(","))
        print(f"[viplanner] VIPLANNER_RATES: depth {_RATES[0]} Hz, semantics {_RATES[1]} Hz, "
              f"planner {_RATES[2]} Hz (simulation time)")
    if _USE_GEO_V2:
        import geometric_semantic_v2 as _geo_v2
        print("[viplanner] VIPLANNER_GEOMETRIC_SEMANTIC_V2=1 -- ViPlanner semantics from depth only "
              "(geometric_semantic_v2), feet height from leg FK")
    # Explicit user request, for a Z-axis-only isolation test where the goal's X/Y deliberately
    # matches the robot's own start (only Z differs, and Z isn't reachable exactly at that X/Y once
    # actually on the destination floor -- e.g. Floor_2 doesn't extend back to the entrance's own Y):
    # the normal 3D dist_to_goal < REACH_THRESHOLD check would never fire, since XY distance stays
    # large even once the height gap is fully closed. VIPLANNER_STOP_ON_Z_MATCH=1 adds an
    # ADDITIONAL, Z-only reach condition -- stop as soon as height matches the goal's, regardless of
    # XY -- without changing the normal 3D-distance behavior for every other scenario. Off by
    # default.
    _STOP_ON_Z_MATCH = _os_check.environ.get("VIPLANNER_STOP_ON_Z_MATCH") == "1"
    _Z_MATCH_THRESHOLD = 0.15  # meters

    # Explicit user request: simulate smoke on the destination floor in a way that actually affects
    # navigation. VIPLANNER_SMOKE_SEVERITY in [0,1] (0 = off, default 0) scales both effects below;
    # a scene script opts a robot into this by passing smoke_zone_z_min to run_walk (see that
    # parameter's own docstring for why this degrades depth rather than just adding a render effect).
    _SMOKE_SEVERITY = float(_os_check.environ.get("VIPLANNER_SMOKE_SEVERITY", "0.0"))

    def _apply_smoke_degradation(depth_np, severity):
        """Two effects, both scaled by `severity` (0..1), mirroring how smoke/fog actually degrades
        a real IR depth sensor: (1) range noise that grows with the reading itself -- a beam
        travelling further through the haze picks up more scattering -- and (2) a fraction of
        pixels replaced with a short, spurious return (backscatter off nearby particles instead of
        the true, farther surface), which is the specific failure mode that makes smoke dangerous
        for depth-based navigation: real obstacles can read as clear while nearby haze reads as a
        wall. Applied to the raw depth array BEFORE it reaches ViPlanner's own processing, so every
        downstream consumer (the network's own plan, the published point cloud, fear estimation)
        sees exactly what a real depth sensor would report in smoke -- not a separate, cosmetic
        overlay that never reaches the actual decision-making input."""
        if severity <= 0.0:
            return depth_np
        noisy = depth_np + np.random.normal(0.0, severity * 0.15, size=depth_np.shape).astype(np.float32) * depth_np
        dropout_mask = np.random.random(depth_np.shape) < (severity * 0.35)
        spurious = np.random.uniform(0.3, 1.2, size=depth_np.shape).astype(np.float32)
        noisy = np.where(dropout_mask, spurious, noisy)
        return np.clip(noisy, 0.05, None).astype(np.float32)

    if _USE_GEOMETRIC_SEMANTIC:
        from viplanner.utils.geometric_semantic import compute_geometric_semantic
        print("[viplanner] VIPLANNER_USE_GEOMETRIC_SEMANTIC=1 -- feeding ViPlanner depth-only "
              "geometric floor/obstacle classification instead of ground-truth AABB semantics "
              "(control-loop safety nets below still use ground truth for this first test)")

    # --- semantic AABBs for this scene ---
    if precomputed_aabbs is not None:
        aabbs = precomputed_aabbs
        print("[viplanner] using", len(aabbs), "precomputed semantic AABBs (caller-provided)")
    else:
        aabbs = build_lab_aabbs(obstacle_boxes, floor_box)
        print("[viplanner] built", len(aabbs), "semantic AABBs (walls+floor) for the lab scene")

    # --- wall-proximity safety margin: independent of ViPlanner's own (much slower-to-update)
    # avoidance, this checks EVERY control step how close the robot's XY position is to the
    # nearest wall-class AABB's footprint, and directly overrides the steering command to push
    # away once inside SAFE_WALL_DIST -- a hard, fast safety net instead of relying on dense
    # intermediate waypoints to keep it off the walls.
    # Keep z bounds too -- see _nearest_wall below for why (a flat XY-only check falsely matches
    # the 2F room's walls/door-frame, which sit almost directly above the Stair A ground-floor
    # entrance in this scene's XY layout, as "right next to" the robot while it's still on the
    # ground floor 3 meters below).
    # railings are just as non-walkable as walls, so they get the same proximity safety margin
    wall_boxes_xy = [(a[0], a[1], a[2], a[3], a[4], a[5]) for a in aabbs if a[6] in (SEG_ID_WALL, SEG_ID_RAILING)]
    print(f"[viplanner] wall-proximity safety margin active against {len(wall_boxes_xy)} wall footprints")
    # Thin stair/landing side-rails (from the scene's own guardrail_out list) get the cheaper,
    # heading-dependent margin below instead of the conservative diagonal one -- see that margin's
    # own comment for why a UNIFORM conservative margin caused a real, user-observed regression
    # (the robot visibly hesitating to turn into the stairs entrance at all, since the corridor's
    # clear center shrinks to almost nothing once every nearby rail claims a flat ~0.5m radius).
    _guardrail_xy_set = {
        (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6)) for b in (guardrail_boxes or [])
    }
    print(f"[viplanner] {len(_guardrail_xy_set)} of those are thin guardrails (soft steering override skipped)")
    for _gb in sorted(_guardrail_xy_set):
        print(f"[viplanner] DIAG guardrail box: {_gb}")
    # Body-footprint-aware margin, take 3: `wall_dist` is measured from base_pos (the body
    # *center*), so a flat SAFE_WALL_DIST treats the robot as a point -- correct nose-on, but lets
    # its flank clip a wall when moving nearly parallel to it. BODY_CLEARANCE is the buffer beyond
    # the body itself; GO2_HALF_LENGTH/WIDTH are the robot's own half-extents (~0.65m long x 0.28m
    # wide). First attempt at this projected the body's extent toward the wall using the robot's
    # CURRENT heading only (hx*|cos|+hy*|sin|) -- correct for straight-line travel, but wrong for a
    # turn: the whole point of a turn is that heading keeps changing through it, so a margin sized
    # for the heading at the instant of the check underestimates how far a corner can swing out
    # mid-turn, letting it clip a wall it was "clear" of a step earlier. Fixed: use the body's own
    # half-diagonal (its farthest corner from the center, at ANY orientation) instead of a
    # heading-dependent projection -- this is the one number that's still correct no matter how the
    # robot is rotated at the moment contact would happen, including mid-turn. Slightly more
    # conservative than the directional version during straight travel (worst case instead of
    # current case), but that's the right tradeoff for a number meant to bound "can the body reach
    # this wall," not just "is it reaching this wall right now."
    BODY_CLEARANCE = 0.15  # meters, beyond the body itself
    GO2_HALF_LENGTH = ROBOT_HALF_LENGTH  # meters (name kept; value follows VIPLANNER_ROBOT)
    GO2_HALF_WIDTH = ROBOT_HALF_WIDTH  # meters
    GO2_HALF_DIAGONAL = math.hypot(GO2_HALF_LENGTH, GO2_HALF_WIDTH)  # meters, orientation-independent
    SAFE_WALL_DIST = 0.35  # meters -- still used as-is for the stall/retreat "near a wall" gate

    # Real run diagnosis: a wall stall right at the Stair A entrance (ground floor, z~0.3) kept
    # matching wall_box=(-5.06,-4.84,-5.03,0.43) -- x-width 0.22m, matching FRAME_DEPTH exactly --
    # the 2F room's door-frame post, at z~3.0+. stair_b_exit_x works out equal to FIRST_START_X in
    # this scene's coordinate scheme, so the 2F room sits almost directly above the Stair A
    # entrance in XY. An XY-only nearest-wall search has no way to tell "3m overhead, irrelevant"
    # from "right here" -- so it kept triggering the stall-retreat/wall-repulsion safety net
    # against a wall the robot could never actually touch, dragging it off toward whatever
    # direction that overhead box's (irrelevant) long axis pointed.
    # Was 1.0m ("generous, not a tight match against the robot's own tiny height, since ground-
    # level wall segments here can span a good height range themselves"). Tightened to 0.15m after
    # a real run regressed: Stair_A_BackWall/Stair_B_BackWall (see open_scene_stairs.py, the fix
    # for the exposed-back-face-mislabeled-as-stairs bug) sit only ~0.3m below the height the robot
    # actually crosses at the top of each flight -- with a 1.0m margin those new walls counted as
    # "nearby" even while the robot stood exactly where it should (on top of the last step,
    # crossing onto the landing), and the safety net kept shoving it back down, oscillating at the
    # top of every flight. 0.15m is comfortably under that ~0.3m real gap while still nowhere near
    # tight enough to accidentally re-match the ~2m+-distant 2F-door-frame case above.
    _WALL_Z_MARGIN = 0.15  # meters

    def _nearest_wall(x, y, z):
        best_d, best_dx, best_dy, best_box = 1e9, 0.0, 0.0, None
        for xmin, xmax, ymin, ymax, zmin, zmax in wall_boxes_xy:
            if z < zmin - _WALL_Z_MARGIN or z > zmax + _WALL_Z_MARGIN:
                continue
            cx = min(max(x, xmin), xmax)
            cy = min(max(y, ymin), ymax)
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d, best_dx, best_dy, best_box = d, dx, dy, (xmin, xmax, ymin, ymax)
        return best_d, best_dx, best_dy, best_box

    # --- stair-centerline assist: a single goal point makes ViPlanner cut a diagonal toward it
    # the whole time it's still confined to a narrow flight (observed as "wants to drift toward
    # the left wall before reaching the platform"). Rather than moving the waypoint target itself
    # (tried that -- it stops the diagonal pull but then unreliably finds the next flight
    # afterward, see open_scene_stairs.py), this reads the SAME per-step semantic AABBs used for
    # perception: every stair step in a flight shares one exact y-span (all of Stair_A_00..09 are
    # tagged at y=FIRST_Y +/- STAIR_WIDTH/2, all of Stair_B_00..09 at y=SECOND_Y +/- STAIR_WIDTH/2)
    # regardless of which x-step it is, so "which flight am I standing over, and what's its exact
    # centerline y" is just a lookup by the robot's own x against the stair AABBs' x-ranges -- no
    # new scene-side parameters needed.
    stair_boxes_xy = [(a[0], a[1], (a[2] + a[3]) / 2.0) for a in aabbs if a[6] == SEG_ID_STAIRS]

    # Explicit user request: re-enable centerline assist, generalized to work on ANY flight
    # orientation (not just an X-running first flight) -- this scene's second flight turns 90
    # degrees and runs along Y, which the OLD Stair-A-only version (further below, matching by X
    # alone) can't handle at all. Clusters SEG_ID_STAIRS boxes by adjacency (same test used
    # elsewhere in this file for the virtual edge-safety strips), then for each cluster picks the
    # correction axis from whichever extent is SHORTER (the flight's own width, perpendicular to
    # its run direction) -- same "long axis = run direction" logic as those strips.
    def _cluster_stair_flights(aabb_list):
        _stairs = [a for a in aabb_list if a[6] == SEG_ID_STAIRS]

        def _adjacent(a, b, eps=0.02):
            ax0, ax1, ay0, ay1 = a[0], a[1], a[2], a[3]
            bx0, bx1, by0, by1 = b[0], b[1], b[2], b[3]
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            if ox < -eps or oy < -eps:
                return False
            mwx = min(ax1 - ax0, bx1 - bx0)
            mwy = min(ay1 - ay0, by1 - by0)
            return ox >= 0.5 * mwx or oy >= 0.5 * mwy

        n = len(_stairs)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if _adjacent(_stairs[i], _stairs[j]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        clusters = {}
        for i, a in enumerate(_stairs):
            clusters.setdefault(find(i), []).append(a)
        flights = []
        for c in clusters.values():
            fxmin = min(a[0] for a in c)
            fxmax = max(a[1] for a in c)
            fymin = min(a[2] for a in c)
            fymax = max(a[3] for a in c)
            fzmax = max(a[5] for a in c)
            if (fxmax - fxmin) >= (fymax - fymin):
                flights.append({"xmin": fxmin, "xmax": fxmax, "ymin": fymin, "ymax": fymax,
                                 "zmax": fzmax, "axis": "y", "centerline": (fymin + fymax) / 2.0})
            else:
                flights.append({"xmin": fxmin, "xmax": fxmax, "ymin": fymin, "ymax": fymax,
                                 "zmax": fzmax, "axis": "x", "centerline": (fxmin + fxmax) / 2.0})
        return flights

    _centerline_flights = _cluster_stair_flights(aabbs)
    print(f"[viplanner] stair-centerline assist (generalized) active against {len(_centerline_flights)} detected flight(s)")

    def _stair_centerline_correction(x, y, z):
        """(axis, centerline, current) for whichever flight currently contains (x,y), gated by z
        not yet within _CENTERLINE_TOP_CLEARANCE of that flight's own top -- releases control before
        reaching the top so the natural turn toward the next flight/landing isn't fought (this is
        exactly the documented reason the old Stair-A-only version was disabled: it kept target_wz
        locked against ViPlanner's own legitimate turn near the top of a flight)."""
        _CENTERLINE_TOP_CLEARANCE = 0.15  # meters
        if GUIDANCE == "perception":  # (the centreline comes from the perceived flight's sub-goal instead)
            return None, None, None
        for f in _centerline_flights:
            if f["xmin"] <= x <= f["xmax"] and f["ymin"] <= y <= f["ymax"] and z <= f["zmax"] - _CENTERLINE_TOP_CLEARANCE:
                return (("y", f["centerline"], y) if f["axis"] == "y" else ("x", f["centerline"], x))
        return None, None, None

    # Only assist on the FIRST flight (Stair A): confirmed via a real run that this nudge stacked
    # as a third term (alongside the goal heading and the wall-repulsion escape) into target_wz on
    # Stair B and caused an 8000+ step crawl that never reached the top (y pinned near 0.79 vs a
    # 1.20 target for thousands of steps). It's harmless -- even helpful -- on Stair A, where
    # there's no such fight. _build_scene() creates Stair_A_00..09 before Stair_B_00..09, so
    # stair_boxes_xy's first entry (list order == aabbs' append order) is always Stair A's
    # centerline y -- no new scene-side parameter needed to tell the two apart.
    # Gated by z (Stair A's own top height), not just x-range matching -- more robust than
    # comparing floating-point y-centers, and naturally also excludes the 2F floor once past the
    # top of Stair A (x could still fall inside some stair-adjacent range there).
    _stair_a_centerline_y = stair_boxes_xy[0][2] if stair_boxes_xy else None
    _stair_a_max_z = max(
        (a[5] for a in aabbs if a[6] == SEG_ID_STAIRS and _stair_a_centerline_y is not None
         and abs((a[2] + a[3]) / 2.0 - _stair_a_centerline_y) < 0.01),
        default=None,
    )
    # For the Stair-A-cleared latch below: Stair A's own X extent (only meaningful for an X-running
    # first flight, same assumption _stair_a_centerline_y/_stair_a_max_z already make).
    _stair_a_min_x = min(
        (a[0] for a in aabbs if a[6] == SEG_ID_STAIRS and _stair_a_centerline_y is not None
         and abs((a[2] + a[3]) / 2.0 - _stair_a_centerline_y) < 0.01),
        default=None,
    )
    _stair_a_max_x = max(
        (a[1] for a in aabbs if a[6] == SEG_ID_STAIRS and _stair_a_centerline_y is not None
         and abs((a[2] + a[3]) / 2.0 - _stair_a_centerline_y) < 0.01),
        default=None,
    )
    # One-way latch: once the robot's X has ever cleared Stair A's own far edge, keep Stair A
    # blocked from then on regardless of height -- a pure height check can't distinguish "still on
    # Stair A's own last tread" from "now on the landing" when the two are flush (this scene's
    # landing sits at exactly Stair A's own top height), so this needs its own signal. Explicit user
    # request: in L_SHAPED_SINGLE_GOAL mode there is no waypoint index to gate on at all (always a
    # single waypoint, wp_idx never advances), so the original `wp_idx >= 3` gate below never fires
    # and Stair A stayed seekable forever -- confirmed via a real run that after successfully
    # climbing Stair A once, the robot later got pulled back toward Stair A's own entrance again and
    # fell. This latch fires independently of waypoint count/index.
    # Gated on having genuinely been ON Stair A first (`_stair_a_ever_climbed`): confirmed via a real
    # run that spawning the robot EAST of Stair A's own extent (as L_SHAPED_SINGLE_GOAL's new
    # spawn-under-the-2F-goal point does) put base_pos[0] > _stair_a_max_x true from step 1, before
    # the robot had ever set foot on Stair A -- the latch fired immediately and Stair A was
    # permanently reclassified as a wall from spawn, wiping it from the semantic image the whole
    # episode. Requiring real prior use of the flight (its own XY footprint at a meaningfully
    # elevated height, not just standing at its own ground-level entrance) before the X-crossing
    # check is allowed to matter fixes that without reintroducing the old "go back down" risk.
    _stair_a_cleared = False
    _stair_a_ever_climbed = False

    # --- virtual edge "walls" along each stairs flight -- safety net for an explicit user request
    # ("它在樓梯上時要保持中線不要掉下樓梯") made right after temporarily disabling the real
    # guardrails (DISABLE_STAIR_SIDE_WALLS in open_scene_stairs.py) to inspect the bare scene. With
    # no physical collider at the tread edges any more, nothing stops the robot drifting past the
    # tread's own width and falling off the side. Reuses the EXACT SAME wall-repulsion safety net
    # that already protects against real walls (severity-based blend + hard escape fallback, see
    # below) instead of writing a new mechanism: synthesize a thin strip immediately outside each
    # flight's own actual tread extent (read from the real AABBs, so this tracks whatever the
    # tread width actually is without a hardcoded constant) and feed it into `wall_boxes_xy` --
    # NOT into `aabbs`, so this is purely a control-loop safety net and never alters what the
    # network itself perceives in the semantic image. Deliberately NOT added to `_guardrail_xy_set`
    # below -- falling off the side is exactly the real hazard the conservative half-diagonal
    # margin (not the relaxed guardrail treatment) exists for.
    # Grouping-by-shared-Y-center (the original approach) silently assumed every flight runs along
    # X -- true for both flights in the original two-flight scene, but broke completely on an
    # L-shaped scene whose second flight turns 90 degrees and climbs along +Y instead: each of that
    # flight's steps then has its own distinct Y-center, so the old code split one real flight into
    # N separate one-step "flights" and produced edge strips in the wrong axis entirely (confirmed
    # via a real run: the robot fell off that flight's actual left/right side, completely
    # unprotected, ~2/3 of the way up). Fixed generally instead of special-casing that one scene:
    # cluster stair AABBs by actual XY adjacency (works for any flight layout/orientation), then for
    # each cluster pick the safety-strip axis from whichever extent is LONGER -- the flight's own
    # run direction is always the long axis (many step-widths long), its side-to-side width the
    # short one, regardless of whether that run happens to be along X or Y.
    _VIRTUAL_EDGE_THICKNESS = 0.10  # meters
    _stair_step_aabbs = [a for a in aabbs if a[6] == SEG_ID_STAIRS]

    def _aabbs_adjacent(a, b, eps=0.02):
        """True only for boxes that share most of a full tread-width edge, not merely any overlap
        at all -- confirmed via a real run that a plain "AABBs overlap" test wrongly merged Stair A
        and Stair B into one cluster: their corner step boxes' own _AABB_MARGIN padding alone was
        enough to overlap by ~0.06m in both axes near the landing, even though the two flights don't
        actually touch. Real same-flight neighbors share their FULL width (e.g. two Stair A steps
        share the entire STAIR_WIDTH-wide Y-span, only touching by the margin along the depth/X
        axis) -- so require the overlap along at least one axis to cover most (>=50%) of the
        narrower box's extent there, not just be non-negative."""
        ax0, ax1, ay0, ay1 = a[0], a[1], a[2], a[3]
        bx0, bx1, by0, by1 = b[0], b[1], b[2], b[3]
        overlap_x = min(ax1, bx1) - max(ax0, bx0)
        overlap_y = min(ay1, by1) - max(ay0, by0)
        if overlap_x < -eps or overlap_y < -eps:
            return False
        min_width_x = min(ax1 - ax0, bx1 - bx0)
        min_width_y = min(ay1 - ay0, by1 - by0)
        return overlap_x >= 0.5 * min_width_x or overlap_y >= 0.5 * min_width_y

    _parent = list(range(len(_stair_step_aabbs)))

    def _find(i):
        while _parent[i] != i:
            _parent[i] = _parent[_parent[i]]
            i = _parent[i]
        return i

    for _i in range(len(_stair_step_aabbs)):
        for _j in range(_i + 1, len(_stair_step_aabbs)):
            if _aabbs_adjacent(_stair_step_aabbs[_i], _stair_step_aabbs[_j]):
                _ri, _rj = _find(_i), _find(_j)
                if _ri != _rj:
                    _parent[_ri] = _rj

    _flight_clusters = {}
    for _i, _a in enumerate(_stair_step_aabbs):
        _flight_clusters.setdefault(_find(_i), []).append(_a)

    # Explicit user request ("它還是為了避開所以從二樓跌落"): tracked separately from
    # `_guardrail_xy_set` (below) so the wall-proximity override can tell these apart from every
    # other entry in `wall_boxes_xy` -- see that override's own comment for why it matters.
    _edge_strip_xy_set = set()

    for _flight_steps in _flight_clusters.values():
        _f_xmin = min(a[0] for a in _flight_steps)
        _f_xmax = max(a[1] for a in _flight_steps)
        _f_ymin = min(a[2] for a in _flight_steps)
        _f_ymax = max(a[3] for a in _flight_steps)
        _f_zmax = max(a[5] for a in _flight_steps) + 0.5  # headroom above this flight's own top
        if (_f_xmax - _f_xmin) >= (_f_ymax - _f_ymin):
            # flight runs along X -- its sides are the Y bounds
            _strip_a = (_f_xmin, _f_xmax, _f_ymax, _f_ymax + _VIRTUAL_EDGE_THICKNESS, -0.5, _f_zmax)
            _strip_b = (_f_xmin, _f_xmax, _f_ymin - _VIRTUAL_EDGE_THICKNESS, _f_ymin, -0.5, _f_zmax)
        else:
            # flight runs along Y -- its sides are the X bounds
            _strip_a = (_f_xmax, _f_xmax + _VIRTUAL_EDGE_THICKNESS, _f_ymin, _f_ymax, -0.5, _f_zmax)
            _strip_b = (_f_xmin - _VIRTUAL_EDGE_THICKNESS, _f_xmin, _f_ymin, _f_ymax, -0.5, _f_zmax)
        wall_boxes_xy.append(_strip_a)
        wall_boxes_xy.append(_strip_b)
        _edge_strip_xy_set.add(tuple(round(v, 6) for v in _strip_a[:4]))
        _edge_strip_xy_set.add(tuple(round(v, 6) for v in _strip_b[:4]))
    print(f"[viplanner] added {2 * len(_flight_clusters)} virtual edge-safety wall strips along "
          f"{len(_flight_clusters)} detected stairs flight(s)")

    # --- explicit "seek the stairs" sub-goal, driven by remaining height (z) gap ---
    # The steering math everywhere else in this file only ever looks at horizontal (px, py) --
    # nothing here has ever told the network "you're still N meters below where you need to be, so
    # find a way up" beyond whatever it infers on its own from the semantic image. General version
    # (not hardcoded to one specific flight) lives in the replan block below: whenever the current
    # waypoint's z differs meaningfully from the robot's own z and it isn't already on some stairs,
    # plan toward the nearest stairs footprint (from `active_aabbs`) instead of the real, possibly
    # far away, goal.
    print(f"[viplanner] guidance mode: {GUIDANCE}")
    print(f"[viplanner] general stairs-seeking active against {len(stair_boxes_xy)} stair-step footprints")
    if smoke_zone_z_min is not None and _SMOKE_SEVERITY > 0.0:
        print(f"[viplanner] SMOKE simulation active: severity={_SMOKE_SEVERITY} once base z >= {smoke_zone_z_min}")
    # was DEADZONE=0.10/GAIN=1.2 -- that reached centering_strength=1.0 (fully overriding
    # ViPlanner's own path, not just nudging it) at just 0.83m of lateral offset, well inside the
    # 1.5m corridor and nowhere near the wall-repulsion margin (0.35m) that's supposed to be the
    # actual safety net. Observed as "the drawn planned path visibly goes left, but the dog goes
    # right anyway" -- the centerline pull was dominating ViPlanner's own (often perfectly
    # reasonable) guidance even when nowhere close to a real wall. Widened the deadzone and
    # softened the gain so it stays a gentle nudge across the corridor's safe width and only
    # becomes assertive near where wall-repulsion would take over anyway, instead of fighting
    # ViPlanner for the whole corridor.
    CENTERLINE_DEADZONE = 0.25  # meters
    CENTERLINE_GAIN = 0.5  # centering_strength now reaches 1.0 only at 2.0m offset -- never in
    # practice within this corridor, so it always leaves some weight to ViPlanner's own path

    # --- Stair A perception-blocking, once it's no longer needed: confirmed via a real run that
    # after reaching the landing, ViPlanner can wander back onto Stair A and descend it instead of
    # committing to Stair B -- both flights are tagged the same SEG_ID_STAIRS class (same green in
    # the semantic image), so to the network they're visually identical, and it has no notion of
    # "already visited, don't go back." Steering-level fixes (centerline assist, wall-repulsion,
    # the stall watchdog) can't fix this because they only ever decide *how to turn*, never *which
    # global path to commit to* -- that decision is ViPlanner's own, made from what it perceives.
    # So fix the perception directly: once past the landing / Stair-B junction, reclassify every
    # Stair-A-tagged AABB as SEG_ID_WALL for the rest of the episode, so the network's own semantic
    # input no longer shows Stair A as a walkable surface at all. (The actual wp_idx threshold that
    # gates this lives at the active_aabbs assignment below -- keep it in sync with whichever
    # waypoint index marks "done with Stair A" if the waypoint list changes again.)
    aabbs_stair_a_blocked = [
        (xmin, xmax, ymin, ymax, zmin, zmax,
         SEG_ID_WALL if (cid == SEG_ID_STAIRS and _stair_a_centerline_y is not None and abs((ymin + ymax) / 2.0 - _stair_a_centerline_y) < 0.01)
         else cid)
        for (xmin, xmax, ymin, ymax, zmin, zmax, cid) in aabbs
    ]

    # (Old Stair-A-only, X-matching-only `_stair_centerline_y` helper removed -- fully superseded by
    # the generalized `_stair_centerline_correction` above, which handles either flight orientation.)

    # --- load HIMLoco + ViPlanner ---
    print(f"[viplanner] robot profile: {ROBOT} (KP {KP}, KD {KD}, stand height {ROBOT_STAND_HEIGHT} m)")
    print("[viplanner] loading gait policy checkpoint:", GAIT_POLICY_CKPT)
    policy = torch.jit.load(GAIT_POLICY_CKPT, map_location=device)
    policy.eval()

    print("[viplanner] loading ViPlanner model dir:", VIPLANNER_MODEL_DIR)
    vip = VIPlannerAlgo(model_dir=VIPLANNER_MODEL_DIR, fear_threshold=FEAR_THRESHOLD, device=device)

    # --- ROS2 point cloud + semantic-image publishers (for rviz2) ---
    pc_pub = None
    sem_pub = None
    sem_truth_pub = None
    sem_extra_pub = None
    depth_pub = None
    route_pub = None
    try:
        import rclpy
        from sensor_msgs.msg import PointCloud2, PointField, Image
        from std_msgs.msg import Header

        if not rclpy.ok():
            rclpy.init()
        pc_node = rclpy.create_node("go2_pointcloud_publisher")
        pc_pub = pc_node.create_publisher(PointCloud2, "/go2/points", 1)
        sem_pub = pc_node.create_publisher(Image, "/go2/semantic_view", 1)
        # with M2F=1: the scene's perfect semantics next to Mask2Former's in rviz (ViPlanner still gets
        # Mask2Former's on /go2/semantic_view; user request 2026-10-15)
        sem_truth_pub = pc_node.create_publisher(Image, "/go2/semantic_truth", 1) if _USE_M2F else None
        # 2x2 mosaic of the extra stair cameras' Mask2Former output: left_bottom | front_bottom /
        # right_bottom | rear (GUIDANCE=perception with EXTRA_CAMERAS only)
        sem_extra_pub = pc_node.create_publisher(Image, "/go2/semantic_extra", 1)
        # Explicit user request (2026-10-13): live depth image + a live version of the ghost-test
        # route map (plot_ghost_paths.py) in rviz2. /go2/route carries several markers per tick
        # (walked path, ViPlanner plans, goals, scene outline), hence the deeper queue.
        from visualization_msgs.msg import Marker as _RouteMarkerMsg
        depth_pub = pc_node.create_publisher(Image, "/go2/depth_view", 1)
        route_pub = pc_node.create_publisher(_RouteMarkerMsg, "/go2/route", 200)
        print("[viplanner] publishing for rviz2: /go2/points, /go2/semantic_view, /go2/depth_view, /go2/route")
    except Exception as e:
        print("[viplanner] WARNING: could not set up ROS2 point cloud publisher:", repr(e))

    # goal_pos may be a single (x,y,z) or a list of them (sequential waypoints, e.g. stairs
    # landing -> 2F doorway -> 2F room center) -- normalize to a list either way.
    if len(goal_pos) > 0 and not np.isscalar(goal_pos[0]):
        waypoints = [np.array(g, dtype=np.float32) for g in goal_pos]
    else:
        waypoints = [np.array(goal_pos, dtype=np.float32)]
    print(f"[viplanner] {len(waypoints)} waypoint(s):", [w.tolist() for w in waypoints])

    # remember the starting pose so we can teleport back and run the walk again
    start_pos_arr, start_quat_arr = dog_art.get_world_poses()
    start_pos = np.asarray(start_pos_arr).copy()
    start_quat = np.asarray(start_quat_arr).copy()

    episode = 0
    while True:
        episode += 1
        if episode > 1:
            print(f"[viplanner] --- episode {episode}: resetting dog to start pose {start_pos[0]} ---")
            dog_art.set_world_poses(start_pos, start_quat)
            dog_art.set_joint_positions(DEFAULT_ANGLES[None, :], joint_indices=_JOINT_IDX)
            dog_art.set_joint_position_targets(DEFAULT_ANGLES[None, :], joint_indices=_JOINT_IDX)
            dog_art.set_joint_velocities(np.zeros((1, 12), dtype=np.float32), joint_indices=_JOINT_IDX)
            dog_art.set_velocities(np.zeros((1, 6), dtype=np.float32))
            draw_iface.clear_lines()
            draw_iface.clear_points()
            for _ in range(10):
                await app.next_update_async()

        # Explicit user request, confirmed via a real run: `_stair_a_cleared`/`_stair_a_ever_climbed`
        # (see their own comment near _stair_a_max_x, well above this loop) are declared ONCE before
        # this `while True:` episode loop, not reset per-episode -- so once episode 1 legitimately
        # climbed Stair A and latched `_stair_a_cleared=True`, that stayed true for every later
        # episode in the same run too, permanently blocking Stair A from step 1 of episode 2 onward,
        # before the robot had ever set foot on it that episode. Reset both here, at the top of every
        # episode (including the first -- harmless, they're already False there).
        _stair_a_cleared = False
        _stair_a_ever_climbed = False

        # Explicit user request ("用這個方式好了"): visual-odometry-based relative height estimate,
        # built from frame-to-frame depth ICP (reproject_depth_to_camera_local + icp_point_to_point
        # above) -- a real-hardware-feasible stand-in for base_pos[2], which is simulator-privileged
        # ground truth a real robot never has. Diagnostic-only for now: computed and logged
        # alongside the real base_pos[2] at every replan tick (see that block below) so its drift can
        # actually be measured against ground truth before anything is switched over to depend on
        # it -- not yet wired into any climbing/reach decision.
        _vo_height_estimate = 0.0
        _vo_prev_points = None
        _vo_prev_up = None
        _vo_start_base_z = None

        # Explicit user request ("好，先做腿部里程計"): leg/proprioceptive odometry, a second
        # real-hardware-feasible height estimate alongside the visual one above -- built from joint
        # encoders + known link geometry only (go2_foot_positions_body_frame above), no camera
        # involved at all, so immune to the smoke degradation that hurts VO. Also diagnostic-only for
        # now, logged alongside VO and the real base_pos[2] every 100 steps (see the per-step update
        # below) so both estimates' actual drift can be compared before either is trusted for a real
        # decision.
        _leg_odom_height_estimate = 0.0
        _leg_odom_prev_feet = None
        _leg_odom_prev_stance = None
        _leg_odom_start_base_z = None
        _leg_odom_targets = DEFAULT_ANGLES.copy()  # joint targets the drive is tracking (for torque)
        _leg_odom_prev_t = _leg_odom_prev_vz = _leg_odom_prev_z = None  # for the IMU acceleration
        _legodom_dump = {} if _legodom_dump_path else None

        obs_history = deque([np.zeros(45, dtype=np.float32) for _ in range(OBS_HISTORY_LEN)], maxlen=OBS_HISTORY_LEN)
        last_actions = np.zeros(12, dtype=np.float32)
        vx, wz = MAX_SPEED, 0.0  # smoothed commands actually sent to the policy every step
        # _viplanner_target_vx/wz: ViPlanner's own intent, re-decided only every COMMAND_HOLD_STEPS.
        # target_vx/target_wz: the per-step working value that centerline/wall-repulsion/stall
        # overrides actually modify. It used to just BE _viplanner_target_wz, overwritten in place
        # by those overrides -- since wall-repulsion (and the others) run every step but ViPlanner
        # only updates every COMMAND_HOLD_STEPS (20 steps), that meant every step's override was
        # blending against the *previous step's already-overridden* value, not ViPlanner's actual
        # current intent -- confirmed via a real run where ViPlanner steadily asked for wz=+0.66..
        # +0.90 (turn left) for 20+ consecutive steps while the applied command drifted the entire
        # opposite way, down to -0.90, because each step's wall-repulsion blend compounded onto the
        # last one instead of re-blending the same target. Now target_wz/target_vx are reset to the
        # pure ViPlanner values every step before any override touches them, so overrides always
        # blend against what ViPlanner is *currently* saying, not an accumulating chain.
        _viplanner_target_vx, _viplanner_target_wz = MAX_SPEED, 0.0
        target_vx, target_wz = MAX_SPEED, 0.0
        heading_error = 0.0
        # heading_error is tracked as an EMA over (cos, sin) instead of the raw angle -- ViPlanner's
        # raw per-replan heading estimate is noisy (observed swinging e.g. -41 -> +126 -> +98 -> -83
        # degrees across consecutive replan ticks, confirmed via diagnostics below: vx/wz deltas from
        # our own smoothing stayed small, but the raw policy action delta regularly spiked to 2-3 --
        # i.e. the gait itself was being forced into abrupt corrective steps by a target_wz that kept
        # flipping sign at the next COMMAND_HOLD_STEPS tick). Averaging unit vectors (not the angle
        # directly) is the correct way to smooth a wrapped quantity -- a naive scalar EMA breaks at
        # the +-180 degree boundary.
        _heading_cos, _heading_sin = 1.0, 0.0
        # was 0.35, then lowered to 0.2 alongside YAW_GAIN (see its own comment) to damp the raw
        # per-replan heading noise. Raised to 0.4 per explicit user request, paired with
        # COMMAND_HOLD_STEPS dropping 20->5 above -- together these were making the visualized
        # path visibly update well before the dog's actual heading caught up to it. Still an EMA
        # (not instant), so per-replan noise is damped, just less aggressively than before.
        HEADING_SMOOTH_RATE = 0.4
        fear_val = 0.0
        wp_idx = 0
        goal_np = waypoints[wp_idx]
        _route_pts, _route_tick = [], 0  # walked path for /go2/route, and a replan-tick counter
        if GUIDANCE == "perception":
            _stair_seeker.reset()
            _percep_climbing = False
            _percep_descend = False
            _percep_start_z, _percep_travel = None, 0.0
            _extra_obs = []
        if route_pub is not None:
            try:
                route_pub.publish(_route_marker("", 0, 0, 0.0, action=3))  # new episode: clear the old route
            except Exception as e:
                print("[viplanner] WARNING: route marker publish failed:", repr(e))

        # stall watchdog: heading_error, the centerline assist, and the wall-repulsion escape are
        # three independent terms all writing into target_wz now -- confirmed twice (once at a
        # door-frame corner, once on Stair B) that they can settle into a standoff where they net
        # to ~0 regardless of a large heading_error, freezing the robot in place indefinitely
        # (action_delta near-zero too -- a real "stuck", not just slow progress). Rather than
        # trying to enumerate and patch every possible cancellation between three blended terms,
        # this just checks real displacement periodically and forces a decisive turn if it hasn't
        # moved -- robust to whatever combination of terms caused the standoff.
        STALL_CHECK_INTERVAL = 200  # steps
        STALL_MOVE_THRESHOLD = 0.15  # meters of net displacement expected over that many steps
        STALL_RECOVERY_DURATION = 80  # steps of forced turning once triggered
        # retreat-and-realign: a turn-in-place recovery can't get past a wall that runs the whole
        # length of a corridor -- confirmed via a real run stuck for 1000+ steps just outside a
        # stair flight's own guard-rail wall (y drifted past the wall, onto the open floor beside
        # the stairs, with no gap in that wall except back at the flight's own entrance). When a
        # stall is detected AND the robot is genuinely close to a wall, retreat ALONG that specific
        # wall's own long axis toward whichever end is nearer (not a hardcoded world direction --
        # this file is scene-agnostic) instead of just turning in place, so it has a chance to clear
        # the wall's own length and re-approach the entrance correctly.
        STALL_BACKUP_MAX_STEPS = 400  # steps of forced retreat once triggered
        _stall_check_dist = None
        _stall_check_pos = None
        _tipped_steps = 0  # consecutive steps with the body tipped over, see FELL OVER check
        # VIPLANNER_RATES pipeline state: latest depth frame (+ its capture pose / time / feet height),
        # latest semantics (+ the capture time of the depth it was built from), next due times
        _lat_depth = _lat_depth_pos = _lat_depth_quat = _lat_depth_t = _lat_cam_h = None
        _lat_sem = _lat_sem_src_t = None
        _next_depth_t = _next_sem_t = _next_plan_t = timeline.get_current_time()
        _sem_age_log = []
        active_aabbs = aabbs
        if _GHOST:
            _gw, _gx, _gy, _gz = (float(v) for v in np.asarray(start_quat)[0])
            _ghost_yaw = math.atan2(2 * (_gw * _gz + _gx * _gy), 1 - 2 * (_gy * _gy + _gz * _gz))
            _ghost_pos = np.asarray(start_pos)[0].astype(np.float64).copy()
            _ghost_prev_t = timeline.get_current_time()
            _ghost_path, _ghost_plans = [], []
        _stall_recovery_left = 0
        _stall_recovery_sign = 1.0
        _stall_backup_left = 0
        _stall_backup_target = None
        # nested stuck-during-retreat check: the retreat's own heading calc depends on base_pos and
        # base_quat -- if the robot is genuinely physically wedged (not just blocked by a wall it
        # can route around), both stay frozen, so retreat_heading_error also freezes near whatever
        # it first computed (observed: stuck near 0 for 4000+ steps straight, producing a token
        # wz~-0.03 that never actually turns the robot). Same class of standoff the outer stall
        # watchdog above exists to catch, just one level down -- track real displacement during the
        # retreat itself and bail to the turn-in-place recovery (no dependence on translating, so it
        # has a real chance of breaking a physical wedge) if the retreat isn't actually moving it.
        STALL_BACKUP_CHECK_INTERVAL = 150  # steps
        STALL_BACKUP_MOVE_THRESHOLD = 0.1  # meters
        _stall_backup_check_pos = None
        _stall_backup_check_left = 0
        seeking_stairs = False  # set for real inside the replan block; safe default for status prints
        near_stairs_entrance = False  # same -- safe default for status prints before the first replan

        print(f"[viplanner] episode {episode}: starting control loop, waypoint 1/{len(waypoints)} =", goal_np)
        reached = False

        # --- diagnostics: is choppiness a render-frame-timing thing, is a "surge" a jump in the
        # vx/wz command itself (our control logic), or a jump in the raw policy action (the motor/
        # gait itself)? Tracked separately so the three can't be confused with each other.
        _diag_last_t = time.time()
        _diag_dts = []
        _diag_vx_deltas = []
        _diag_wz_deltas = []
        _diag_action_deltas = []
        _diag_fear_prev = None
        _diag_fear_deltas = []

        for step in range(1, max_steps + 1):
            await app.next_update_async()
            _diag_now = time.time()
            _diag_dts.append(_diag_now - _diag_last_t)
            _diag_last_t = _diag_now
            _prev_vx, _prev_wz = vx, wz

            # Confirmed via a real ~2 hour, 79-episode run: PhysX's articulation view can go
            # transiently unavailable for a single frame ("Physics Simulation View is not created
            # yet" warning, immediately followed by get_angular_velocities() returning a bare
            # 0-dimensional array instead of (1,3), so `[0]` throws IndexError) without the rest of
            # the simulation actually breaking -- get_world_poses() right above it succeeded the
            # same frame. Retry a couple of frames rather than let one transient glitch kill an
            # otherwise-fine multi-hour run.
            for _phys_read_attempt in range(3):
                try:
                    base_pos_arr, base_quat_arr = dog_art.get_world_poses()
                    base_pos = np.asarray(base_pos_arr)[0]
                    if _logic_z_off:
                        # (ghost float height, or a robot taller than Go2 -- see ROBOT_LOGIC_Z_OFFSET)
                        # The guidance logic's height tests (on a tread? topped out? climbable?) all
                        # assume the base rides ~0.30m above what it stands on. Floating higher
                        # (VIPLANNER_GHOST_HEIGHT) made "topped out" fire two treads early. Hand the
                        # logic a base at the real standing height; only the camera (below) and the
                        # visible body sit at the float height.
                        base_pos = base_pos.copy()
                        base_pos[2] -= _logic_z_off
                    base_quat = np.asarray(base_quat_arr)[0]  # wxyz
                    ang_vel_world = np.asarray(dog_art.get_angular_velocities())[0]
                    dof_pos = np.asarray(dog_art.get_joint_positions(joint_indices=_JOINT_IDX))[0]
                    dof_vel = np.asarray(dog_art.get_joint_velocities(joint_indices=_JOINT_IDX))[0]
                    # joint torque as the drive applied it (motor current on the real robot), for
                    # leg odometry's foot contact; None if this API is missing
                    try:
                        dof_effort = np.asarray(dog_art.get_measured_joint_efforts(joint_indices=_JOINT_IDX))[0]
                    except Exception:
                        dof_effort = None
                    break
                except IndexError as e:
                    print(f"[viplanner] WARNING: transient physics-view read failure at step {step} "
                          f"(attempt {_phys_read_attempt + 1}/3): {e!r} -- retrying next frame")
                    await app.next_update_async()
            else:
                print(f"[viplanner] WARNING: physics-view read still failing after 3 attempts at "
                      f"step {step} -- skipping this control step")
                continue

            # Explicit user request ("好，先做腿部里程計"): leg odometry update, every step (joint
            # encoders are read every step anyway, unlike the camera) -- see its own state-init
            # comment above for scope (diagnostic-only). Wrapped in try/except for the same reason as
            # the VO block: a diagnostic side-channel must never be able to take down real
            # navigation.
            try:
                if _leg_odom is not None:
                    # URDF-based estimate along gravity (leg_odometry.py, 2026-10-14); replaces the
                    # Go2-geometry one below, which ran the wrong way on growin
                    if _leg_odom_start_base_z is None:
                        _leg_odom_start_base_z = float(base_pos[2])
                        _leg_odom.reset(base_pos[2])   # the start height is the odometry origin
                    # IMU: gravity direction in the body frame, and vertical acceleration -- an
                    # accelerometer on the real robot; here the second difference of the body height
                    # over the last two control steps. (The articulation's velocity reading instead put
                    # -0.5 m of drift into a flat landing, 2026-10-15.)
                    _g_body = quat_rotate_inverse_wxyz(base_quat, np.array([0, 0, -1.0], dtype=np.float32))
                    _t_lo = timeline.get_current_time()
                    _dt_lo = (_t_lo - _leg_odom_prev_t) if _leg_odom_prev_t is not None else 0.0
                    _vz_lo = (float(base_pos[2]) - _leg_odom_prev_z) / _dt_lo if _dt_lo > 0 else 0.0
                    _acc_up = (_vz_lo - _leg_odom_prev_vz) / _dt_lo if (_dt_lo > 0 and _leg_odom_prev_vz is not None) else 0.0
                    _leg_odom_prev_t, _leg_odom_prev_z = _t_lo, float(base_pos[2])
                    _leg_odom_prev_vz = _vz_lo if _dt_lo > 0 else None
                    # joint torque: the drive's own (motor current on the real robot); the PD torque
                    # from last step's targets if the sim does not report it
                    _tau = dof_effort if dof_effort is not None else np.clip(
                        KP * (_leg_odom_targets - dof_pos) - KD * dof_vel, -LEG_TORQUE_LIMIT, LEG_TORQUE_LIMIT)
                    _leg_odom.update(dof_pos, _g_body, _tau, _acc_up, _dt_lo)
                    _contact = _leg_odom.planted
                    _sensor = [bool(foot_contact_sensors[leg].get_current_frame()["in_contact"]) for leg in _LEG_ORDER]
                    if _legodom_dump is not None:
                        # VIPLANNER_LEGODOM_DUMP: every step's proprioception + truth, for trying
                        # estimators offline (training_data/legodom/<name>_ep<N>.npz)
                        for _k, _v in (("step", step), ("t", timeline.get_current_time()),
                                       ("dof_pos", dof_pos), ("dof_vel", dof_vel),
                                       ("targets", _leg_odom_targets), ("g_body", _g_body), ("sensor", _sensor),
                                       ("effort", dof_effort if dof_effort is not None else np.full(12, np.nan)),
                                       ("foot_world", np.asarray(_foot_view.get_world_poses()[0]) if _foot_view is not None
                                        else np.full((4, 3), np.nan)),
                                       ("base_pos", base_pos), ("base_quat", base_quat)):
                            _legodom_dump.setdefault(_k, []).append(np.array(_v, dtype=np.float64))
                        if step % 500 == 0 or step in (50, 100, 150, 200, 300):   # early saves: falls happen in the first 200 steps
                            np.savez(f"{_legodom_dump_path}_ep{episode}.npz",
                                     **{k: np.stack(v) for k, v in _legodom_dump.items()})
                    if step % 100 == 1:
                        _cs = "".join("1" if c else "0" for c in _sensor)
                        print(f"[viplanner] DIAG leg-odometry height: base estimate={_leg_odom.base_z:+.3f}m "
                              f"true={base_pos[2]:+.3f}m drift={_leg_odom.base_z - base_pos[2]:+.3f}m "
                              f"planted={''.join('1' if c else '0' for c in _contact)} sensor={_cs} "
                              f"missed={_leg_odom.rejected} recovered={_leg_odom.recovered} (step={step} "
                              f"sim_t={_t_lo:.3f}s dt={_dt_lo:.4f}s)")
                else:
                    _leg_feet_now = go2_foot_positions_body_frame(dof_pos)
                    if _leg_odom_start_base_z is None:
                        _leg_odom_start_base_z = float(base_pos[2])
                    # Explicit user request ("好" -> real contact sensing instead of the height
                    # heuristic that failed, confirmed via a real run: estimate barely moved while the
                    # robot genuinely climbed 0.865m -- the "lowest of the 4" heuristic kept missing
                    # which foot was actually planted during the stairs gait's bigger, more irregular
                    # swing/stance pattern. ContactSensor.get_current_frame()["in_contact"] reads real
                    # PhysX contact state per foot directly as a bool -- a quadruped's own foot-force
                    # sensors give this for real, no simulator-privileged pose involved.
                    _stance_now = np.array(
                        [foot_contact_sensors[leg].get_current_frame()["in_contact"] for leg in _LEG_ORDER]
                    )
                    if _leg_odom_prev_feet is not None:
                        _stance_both = _stance_now & _leg_odom_prev_stance
                        if _stance_both.any():
                            _z_now = _leg_feet_now[:, 2]
                            _z_prev = _leg_odom_prev_feet[:, 2]
                            _leg_odom_height_estimate += float(-(_z_now[_stance_both] - _z_prev[_stance_both]).mean())
                    _leg_odom_prev_feet = _leg_feet_now
                    _leg_odom_prev_stance = _stance_now
                    if step % 100 == 1:
                        _leg_odom_true_delta = base_pos[2] - _leg_odom_start_base_z
                        print(f"[viplanner] DIAG leg-odometry height: estimate={_leg_odom_height_estimate:+.3f}m "
                              f"true_delta={_leg_odom_true_delta:+.3f}m "
                              f"drift={_leg_odom_height_estimate - _leg_odom_true_delta:+.3f}m (step={step})")
            except Exception as _leg_e:
                import traceback as _tb
                print(f"[viplanner] WARNING: leg-odometry diagnostic failed at step {step}, skipping this tick:", repr(_leg_e))
                if step <= 3:
                    _tb.print_exc()

            base_ang_vel_body = quat_rotate_inverse_wxyz(base_quat, ang_vel_world) * ANG_VEL_SCALE
            projected_gravity = quat_rotate_inverse_wxyz(base_quat, np.array([0, 0, -1.0], dtype=np.float32))
            commands_scaled = np.array([vx, 0.0, wz], dtype=np.float32) * COMMANDS_SCALE
            dof_pos_rel = dof_pos - DEFAULT_ANGLES
            dof_vel_scaled = dof_vel * DOF_VEL_SCALE

            obs_now = np.concatenate(
                [commands_scaled, base_ang_vel_body, projected_gravity, dof_pos_rel, dof_vel_scaled, last_actions]
            ).astype(np.float32)
            obs_history.appendleft(obs_now)
            full_obs = np.concatenate(list(obs_history))  # newest-first, 270-dim
            full_obs = np.clip(full_obs, -100.0, 100.0)

            obs_tensor = torch.from_numpy(full_obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor)[0].detach().cpu().numpy()
            action = np.clip(action, -100.0, 100.0)
            _diag_action_deltas.append(float(np.abs(action - last_actions).max()))
            last_actions = action.copy()

            joint_targets = DEFAULT_ANGLES + ACTION_SCALE * action
            if _GHOST:
                joint_targets = DEFAULT_ANGLES.copy()  # no gait: the base is moved kinematically below
            dog_art.set_joint_position_targets(joint_targets[None, :], joint_indices=_JOINT_IDX)
            _leg_odom_targets = joint_targets

            cam_world_pos = base_pos + quat_rotate_wxyz(base_quat, CAMERA_LOCAL_POS)
            if _logic_z_off:
                cam_world_pos = cam_world_pos + np.array([0.0, 0.0, _logic_z_off], dtype=np.float32)
            cam_world_quat = quat_mul_wxyz(base_quat, CAMERA_LOCAL_QUAT_WXYZ)
            camera.set_world_pose(cam_world_pos, cam_world_quat, camera_axes="world")
            for _ec in _extra_cams:
                _ec["world_pos"] = base_pos + quat_rotate_wxyz(base_quat, _ec["pos"])
                if _logic_z_off:
                    _ec["world_pos"] = _ec["world_pos"] + np.array([0.0, 0.0, _logic_z_off], dtype=np.float32)
                _ec["world_quat"] = quat_mul_wxyz(base_quat, _ec["quat"])
                _ec["cam"].set_world_pose(_ec["world_pos"], _ec["world_quat"], camera_axes="world")

            # third-person chase view: sits behind + above the dog, looking at it. Moves the
            # viewport's own perspective camera (not the ViPlanner sensor camera above), so this
            # is purely for watching -- has no effect on navigation/control.
            # Throttled to every 4th step (~15Hz, still smooth) instead of every step (~60Hz):
            # set_camera_view goes through omni.kit's undoable ViewportCameraState machinery, and
            # calling it at full control-loop rate for hundreds of steps appears to accumulate
            # cost over time (observed: CPU climbing continuously, base position frozen, on the
            # stairs scene once this was added -- not reproduced with the sensor camera's own
            # set_world_pose, which is a plain USD op with no command-stack involvement).
            if not _free_camera and step % 4 == 1:
                forward_dir = quat_rotate_wxyz(base_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
                chase_eye = base_pos - forward_dir * CHASE_DIST + np.array([0.0, 0.0, CHASE_HEIGHT], dtype=np.float32)
                chase_target = base_pos + np.array([0.0, 0.0, 0.25], dtype=np.float32)
                set_camera_view(chase_eye, chase_target)

            if _RATES is None:
                _do_plan = step % REPLAN_EVERY_N_STEPS == 1
            else:
                _t_now = timeline.get_current_time()
                if _t_now >= _next_depth_t:
                    _next_depth_t = max(_next_depth_t + 1.0 / _RATES[0], _t_now)
                    _d = _get_depth(cam_world_pos, cam_world_quat)
                    if _d is not None:
                        _lat_depth = np.asarray(_d, dtype=np.float32)
                        if smoke_zone_z_min is not None and base_pos[2] >= smoke_zone_z_min and _SMOKE_SEVERITY > 0.0:
                            _lat_depth = _apply_smoke_degradation(_lat_depth, _SMOKE_SEVERITY)
                        _lat_depth_pos = np.array(cam_world_pos, dtype=np.float32)
                        _lat_depth_quat = wxyz_to_xyzw(cam_world_quat)
                        _lat_depth_t = _t_now
                        _feet_w = np.array([quat_rotate_wxyz(base_quat, f)
                                            for f in go2_foot_positions_body_frame(dof_pos)])
                        _lat_cam_h = float(quat_rotate_wxyz(base_quat, CAMERA_LOCAL_POS)[2]
                                           - (_feet_w[:, 2].mean() - GO2_FOOT_RADIUS))
                        if _GHOST or ROBOT != "go2":  # (the leg FK above is Go2's geometry)
                            _lat_cam_h = float((_ghost_h if _GHOST else ROBOT_STAND_HEIGHT)
                                               + quat_rotate_wxyz(base_quat, CAMERA_LOCAL_POS)[2])
                if _t_now >= _next_sem_t and _lat_depth is not None:
                    _next_sem_t = max(_next_sem_t + 1.0 / _RATES[1], _t_now)
                    if _USE_GEO_V2:
                        _lat_sem = _geo_v2.to_color(_geo_v2.classify(
                            _lat_depth, _lat_depth_quat, _lat_cam_h, CAMERA_HFOV_DEG,
                            min_range=GEO_V2_MIN_RANGE)[0])
                    else:
                        _lat_sem = semantic_color_image_with_points(
                            _lat_depth, _lat_depth_pos, _lat_depth_quat, active_aabbs, CAMERA_HFOV_DEG)[0]
                    if _USE_M2F:
                        _lat_sem = _m2f_segment(_lat_sem)
                    _lat_sem_src_t = _lat_depth_t
                _do_plan = _t_now >= _next_plan_t and _lat_sem is not None
                if _do_plan:
                    _next_plan_t = max(_next_plan_t + 1.0 / _RATES[2], _t_now)
                    _sem_age_log.append(_lat_depth_t - _lat_sem_src_t)
                    if len(_sem_age_log) >= 50:
                        print(f"[viplanner] rates: semantics older than the depth it is paired with by "
                              f"mean {1000 * np.mean(_sem_age_log):.0f} ms, max {1000 * np.max(_sem_age_log):.0f} ms")
                        _sem_age_log = []

            if _do_plan:
                depth = _lat_depth if _RATES is not None else _get_depth(cam_world_pos, cam_world_quat)
                if depth is not None:
                    depth_np = np.asarray(depth, dtype=np.float32)
                    if (_RATES is None and smoke_zone_z_min is not None and base_pos[2] >= smoke_zone_z_min
                            and _SMOKE_SEVERITY > 0.0):
                        depth_np = _apply_smoke_degradation(depth_np, _SMOKE_SEVERITY)

                    # Explicit user request ("用這個方式好了"): frame-to-frame depth ICP visual
                    # odometry, diagnostic-only (see its own state-init comment above for why not
                    # yet wired into any decision). Uses the SAME smoke-degraded depth_np the network
                    # itself sees -- a real VO pipeline would be hurt by smoke too, so this is the
                    # honest worst-case test, not a cleaner synthetic one. Wrapped in one try/except
                    # spanning the whole block -- this is purely a diagnostic side-channel, so any
                    # unexpected failure here must degrade to "skip this tick's VO update," never
                    # crash the real navigation loop around it.
                    try:
                        if _vo_start_base_z is None:
                            _vo_start_base_z = float(base_pos[2])
                        _vo_cam_pts = reproject_depth_to_camera_local(depth_np, CAMERA_HFOV_DEG)
                        _vo_valid = np.isfinite(_vo_cam_pts).all(axis=-1) & (depth_np < 10.0)
                        _vo_valid_pts = _vo_cam_pts[_vo_valid]
                        # Capped at 800 (not more): brute-force ICP's per-iteration cost is O(n^2),
                        # and this whole block runs synchronously inside the replan tick, competing
                        # with ViPlanner's own inference for the same ~150ms budget -- 800 points x
                        # 15 iterations stays well under that without a noticeable extra stall.
                        if _vo_valid_pts.shape[0] > 800:
                            _vo_sel = np.random.default_rng(step).choice(_vo_valid_pts.shape[0], 800, replace=False)
                            _vo_valid_pts = _vo_valid_pts[_vo_sel]
                        # "up" in the camera's own local frame this tick, derived the same way
                        # HIMLoco's own gravity observation already is (quat_rotate_inverse of world
                        # +Z/-gravity) -- a real IMU gives you gravity-relative attitude for real, so
                        # this is a legitimate stand-in, unlike base_pos itself.
                        _vo_up_now = quat_rotate_inverse_wxyz(cam_world_quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
                        if _vo_prev_points is not None and _vo_valid_pts.shape[0] >= 20 and _vo_prev_points.shape[0] >= 20:
                            _, _vo_t = icp_point_to_point(_vo_valid_pts, _vo_prev_points)
                            _vo_height_estimate += float(np.dot(_vo_t, _vo_prev_up))
                        _vo_prev_points = _vo_valid_pts
                        _vo_prev_up = _vo_up_now
                        if step % 100 == 1:
                            _vo_true_delta = base_pos[2] - _vo_start_base_z
                            print(f"[viplanner] DIAG VO height: estimate={_vo_height_estimate:+.3f}m "
                                  f"true_delta={_vo_true_delta:+.3f}m drift={_vo_height_estimate - _vo_true_delta:+.3f}m "
                                  f"(step={step}, n_pts={_vo_valid_pts.shape[0]})")
                    except Exception as _vo_e:
                        print(f"[viplanner] WARNING: VO diagnostic failed at step {step}, skipping this tick:", repr(_vo_e))

                    cam_quat_xyzw = wxyz_to_xyzw(cam_world_quat)
                    # once past waypoint 3 (the landing / Stair-B junction -- index 2 in the current
                    # 5-waypoint layout, so wp_idx==3 once reached), Stair A is reclassified as a
                    # wall in what the network sees -- see aabbs_stair_a_blocked above. MUST stay
                    # ahead of the actual Stair-A climb: wp_idx>=1 (this used to be right, back when
                    # waypoint index 1 WAS the landing in an older, shorter waypoint list) blocked
                    # Stair A the instant the robot reached waypoint 1 -- the stair-A *entrance* in
                    # the current 5-waypoint layout -- wiping it from the semantic image before the
                    # robot had climbed a single step. Confirmed via a real run: the robot then had
                    # no SEG_ID_STAIRS geometry left nearby to seek except Stair B, and got pulled
                    # north into Stair B's corridor instead of ever climbing Stair A.
                    # `_stair_a_cleared` (see its own comment near _stair_a_max_x above): a
                    # waypoint-index-independent OR condition, latched permanently once true, so
                    # this also works correctly in L_SHAPED_SINGLE_GOAL mode where wp_idx never
                    # advances past 0. Gated on `_stair_a_ever_climbed` so a robot that spawns EAST
                    # of Stair A's own extent (never having climbed it) doesn't have this fire on
                    # step 1 purely from its own starting position.
                    # 0.45m: comfortably above the robot's own normal standing height on FLAT ground
                    # (observed ~0.28-0.35m throughout this file's own diagnostics), so this only
                    # fires on genuine climbing progress -- a plain fraction of _stair_a_max_z alone
                    # (e.g. 0.3 * 0.60 = 0.18) would sit BELOW that normal standing baseline and be
                    # trivially true just standing at Stair A's own ground-level entrance.
                    if (_stair_a_min_x is not None and _stair_a_max_x is not None
                            and _stair_a_min_x <= base_pos[0] <= _stair_a_max_x
                            and base_pos[2] > 0.45):
                        _stair_a_ever_climbed = True
                    if _stair_a_ever_climbed and _stair_a_max_x is not None and base_pos[0] > _stair_a_max_x:
                        _stair_a_cleared = True
                    # Explicit user request, confirmed via a real run: the latch above is one-way by
                    # design (correct for the L-shaped scene, where Stair A is genuinely done once
                    # past it and Stair B is the real next step) -- but catastrophic for a
                    # SINGLE-FLIGHT scene: the robot fell back down mid-climb (X briefly exceeded the
                    # flight's own exit boundary while still mid-air/mid-step, latching
                    # _stair_a_cleared=True, before it actually landed on the destination floor), and
                    # with the ONLY flight then permanently reclassified as a wall, it had nowhere
                    # left to go for the rest of the 13000-step budget (seeking_stairs stuck False,
                    # wandering with no target). Unlatch if height drops back near normal ground level
                    # (same 0.45m threshold used to detect genuine climbing progress) -- if it's
                    # fallen back down, it needs the flight again, whatever X it happens to be at.
                    if _stair_a_cleared and base_pos[2] < 0.45:
                        _stair_a_cleared = False
                        _stair_a_ever_climbed = False
                    if block_first_flight_after_wp is None:
                        active_aabbs = aabbs
                    else:
                        active_aabbs = (aabbs_stair_a_blocked
                                        if (wp_idx >= block_first_flight_after_wp or _stair_a_cleared) else aabbs)
                    sem, world_xyz = semantic_color_image_with_points(
                        depth_np, cam_world_pos, cam_quat_xyzw, active_aabbs, CAMERA_HFOV_DEG
                    )
                    if _USE_GEOMETRIC_SEMANTIC:
                        # Real-hardware feasibility test (see the import block's comment). Only
                        # `sem` -- what actually gets fed to VIPlannerAlgo.plan_dual below -- is
                        # replaced; `world_xyz` above still comes from the ground-truth path (only
                        # used for the rviz2 point-cloud publish, not navigation) and every other
                        # control-loop mechanism (wall-repulsion, virtual edge walls,
                        # seeking_stairs) still reads `aabbs`/`active_aabbs` regardless of this
                        # flag -- this test isolates ONE question: can the network itself navigate
                        # from depth-only floor/obstacle perception, before touching anything else.
                        #
                        # geometric_semantic.py's own default `ground_height=0.35` is an ABSOLUTE
                        # world-z cutoff -- correct for the flat single-level terrain it was
                        # designed for, but wrong here: once climbing, the tread the robot is
                        # actually standing on is itself well above 0.35m of absolute world z, so
                        # the unmodified default would misclassify the robot's own current step as
                        # an obstacle. Make it relative to the robot's CURRENT base height instead
                        # (base_pos[2] + a margin comfortably covering one riser's rise), so "floor"
                        # tracks wherever the robot currently stands, the same generalization this
                        # session's earlier `_CLIMB_RISE_LIMIT` control-logic fix already applied at
                        # the AABB-lookup level -- here it's applied at the raw-depth level instead.
                        _h, _w = depth_np.shape
                        _hfov = math.radians(CAMERA_HFOV_DEG)
                        _fx = _w / (2 * math.tan(_hfov / 2))
                        _depth_t = torch.from_numpy(depth_np).to(device)
                        _cam_pos_t = torch.tensor(cam_world_pos, dtype=torch.float32, device=device)
                        _cam_quat_t = torch.tensor(cam_world_quat, dtype=torch.float32, device=device)
                        _sem_t = compute_geometric_semantic(
                            _depth_t, _cam_pos_t, _cam_quat_t,
                            fx=_fx, fy=_fx, cx=_w / 2.0, cy=_h / 2.0,
                            ground_height=float(base_pos[2]) + 0.30,
                        )
                        sem = _sem_t.cpu().numpy()
                    if _USE_GEO_V2:
                        # camera height above the feet, gravity-aligned: leg FK (joint encoders) +
                        # base orientation (IMU on a real robot) + the fixed camera mount -- no
                        # world position involved
                        _feet_w = np.array([quat_rotate_wxyz(base_quat, f)
                                            for f in go2_foot_positions_body_frame(dof_pos)])
                        _cam_rel = quat_rotate_wxyz(base_quat, CAMERA_LOCAL_POS)
                        _cam_h = float(_cam_rel[2] - (_feet_w[:, 2].mean() - GO2_FOOT_RADIUS))
                        if _GHOST or ROBOT != "go2":  # floating, or not Go2's leg geometry
                            _cam_h = float((_ghost_h if _GHOST else ROBOT_STAND_HEIGHT) + _cam_rel[2])
                        _geo_cls, _ = _geo_v2.classify(depth_np, cam_quat_xyzw, _cam_h, CAMERA_HFOV_DEG,
                                                       min_range=GEO_V2_MIN_RANGE)
                        sem = _geo_v2.to_color(_geo_cls)
                        if step % 400 == 1:
                            print(f"[viplanner] geo_v2: cam height above feet {_cam_h:.3f}m "
                                  f"(sim truth cam z {cam_world_pos[2]:.2f})")

                    _sem_truth = sem   # before Mask2Former replaces it (for /go2/semantic_truth)
                    if _USE_M2F and _RATES is None:
                        sem = _m2f_segment(sem)
                    if _RATES is not None:
                        sem = _lat_sem  # asynchronous pipeline: whatever the semantics stage last produced

                    if collecting:
                        # .npy for depth (raw float32 meters, no lossy uint16/depth_scale quantization
                        # -- PlannerDataGenerator.load_images prefers .npy over .png when both exist)
                        # and .png for semantics (already an (H,W,3) uint8 RGB array in exactly the
                        # VIPlannerSemMetaHandler color scheme, per camera_semantics_lab.py's own
                        # SEG_CLASS_COLOR table -- no conversion needed). One appended row per sample
                        # in camera_extrinsic.txt, order matching the numeric filename index, exactly
                        # the pypose SE3 tensor layout PlannerDataGenerator.load_odom expects:
                        # tx,ty,tz,qx,qy,qz,qw.
                        np.save(_os.path.join(_depth_dir, f"{_sample_idx}.npy"), depth_np)
                        _PILImage.fromarray(sem).save(_os.path.join(_sem_dir, f"{_sample_idx}.png"))
                        if _save_rgb:
                            try:
                                _rgba = camera.get_rgba()
                                if _rgba is not None and getattr(_rgba, "ndim", 0) == 3:
                                    _PILImage.fromarray(np.asarray(_rgba)[..., :3].astype(np.uint8)).save(
                                        _os.path.join(_rgb_dir, f"{_sample_idx}.png"))
                            except Exception as e:
                                print("[viplanner] WARNING: rgb save failed:", repr(e))
                        with open(_extrinsic_path, "a") as _f:
                            _f.write(
                                f"{cam_world_pos[0]},{cam_world_pos[1]},{cam_world_pos[2]},"
                                f"{cam_quat_xyzw[0]},{cam_quat_xyzw[1]},{cam_quat_xyzw[2]},{cam_quat_xyzw[3]}\n"
                            )
                        _sample_idx += 1

                    if sem_pub is not None:
                        try:
                            sem_pub.publish(make_image_msg(sem, frame_id="go2_camera"))
                            if sem_truth_pub is not None and _RATES is None:
                                sem_truth_pub.publish(make_image_msg(_sem_truth, frame_id="go2_camera"))
                        except Exception as e:
                            print("[viplanner] WARNING: semantic image publish failed:", repr(e))

                    depth_t = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).to(device)
                    sem_t = torch.from_numpy(sem.copy()).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    cam_pos_t = torch.tensor([cam_world_pos if _RATES is None else _lat_depth_pos],
                                             dtype=torch.float, device=device)
                    cam_quat_t = torch.tensor([cam_quat_xyzw if _RATES is None else _lat_depth_quat],
                                              dtype=torch.float, device=device)

                    # General stairs-seeking: whenever the CURRENT waypoint's z differs meaningfully
                    # from the robot's own z, plan toward a stairs footprint instead of straight at
                    # the (possibly far away) real goal -- the network isn't given any explicit "go
                    # find a way up" signal otherwise (steering everywhere else in this file only
                    # ever looks at horizontal px,py). Stays active for the whole climb, not just
                    # the approach (see target_box selection below for why). Applies at every
                    # waypoint transition, not just one specific leg -- naturally resolves to
                    # whichever flight is relevant given the robot's current position, no need to
                    # hardcode which one.
                    # `active_aabbs` (not the raw `aabbs`) so a stairs flight already reclassified
                    # as a wall (see aabbs_stair_a_blocked above) is correctly excluded as a seek
                    # target too -- consistent with "don't go back" once past it.
                    z_gap = abs(goal_np[2] - base_pos[2])
                    # Explicit user request, confirmed via a real run in L_SHAPED_SINGLE_GOAL mode:
                    # this used to check ONLY XY containment, never the robot's own z -- harmless in
                    # the original two-flight scene, whose approach runs entirely through
                    # near_stairs_entrance's fixed-waypoint aim (bypassing target_box selection, and
                    # this check, altogether). Here, with no waypoint funnel at all, EVERY replan runs
                    # through target_box selection, and the spawn point sits right next to Stair B --
                    # while the robot wandered the ground floor near Stair B's own XY footprint (still
                    # at z~0.29, nowhere near actually climbing it), on_stairs read True purely from
                    # XY overlap, routing target_box into the "next step ahead" branch instead of the
                    # correct entrance-seeking one, and it aimed at essentially arbitrary nearby Stair
                    # B steps instead of Stair A's real entrance -- confirmed via added diagnostics
                    # showing planning_goal wandering near Stair B's own footprint instead of Stair
                    # A's fixed (-3.85,-1.50). Added a Z check per footprint (that footprint's own
                    # zmax, i.e. its top tread surface, +/- a margin -- not the fixed
                    # _WALL_Z_MARGIN=0.15 used for wall-proximity matching, since standing height
                    # above a tread varies more against a rising flight's own topmost step) so a
                    # footprint only counts as "the robot is on it" when the robot's height is
                    # actually consistent with standing on that specific tread.
                    on_stairs = any(
                        xmin <= base_pos[0] <= xmax and ymin <= base_pos[1] <= ymax
                        and zmax - 0.1 <= base_pos[2] <= zmax + 0.5
                        for (xmin, xmax, ymin, ymax, zmin, zmax, cid) in active_aabbs if cid == SEG_ID_STAIRS
                    )
                    # Real run: raw z_gap alone false-triggered stairs-seeking even standing right
                    # on the correct floor/landing -- waypoint markers each carry their own
                    # above-floor offset (e.g. wp2's is +0.6, wp3's is +0.35, a purely cosmetic
                    # placement choice, not a real height difference), so comparing the two z's
                    # directly reads that offset as "still need to climb" long after the robot has
                    # already arrived at the right level. What actually matters is reachability, not
                    # z equality: if the floor/landing the robot is currently standing on already
                    # spans the goal's z (with a generous margin for marker-offset noise), the goal
                    # is reachable by ordinary walking and no stairs are needed, however large the
                    # raw z_gap number looks.
                    _FLOOR_REACH_MARGIN = 0.8  # meters
                    # Explicit user request, confirmed via a real run: this used to check only the
                    # GOAL's z against the floor tag's own z-range, never the robot's OWN z -- fine
                    # as long as a floor's XY footprint never overlaps another floor's at a different
                    # height, which held for the old far-corner spawn but broke as soon as the robot
                    # could start (or wander) directly UNDER the 2F floor: its XY coincides with
                    # Floor_2's tagged footprint there even while standing on the GROUND FLOOR, 1.2m
                    # below, with a solid slab between them -- on_floor_reaching_goal read True purely
                    # from that XY coincidence, so seeking_stairs never engaged and the robot just
                    # wandered under the 2F floor instead of heading for the stairwell. Now also
                    # requires the robot's OWN z to fall within that same floor's z-range -- but with
                    # a MUCH tighter margin than the goal's own (0.8m is exactly what let the bad case
                    # above slip through: 0.29 is still inside Floor_2's zmin=1.05 minus 0.8m). The
                    # robot's own standing height above a surface only ever varies by its own body
                    # clearance (observed ~0.28-0.35m throughout this file's diagnostics), not by the
                    # better part of a meter, so 0.15m below zmin (AABB-padding/minor-sink slack) and
                    # 0.5m above zmax (covers standing height plus the SAME cosmetic per-waypoint
                    # marker-offset slack the goal check exists for) is correctly tight without
                    # reintroducing the false positives the goal check's own generous margin needs.
                    _ROBOT_ON_FLOOR_MARGIN_LO = 0.15
                    _ROBOT_ON_FLOOR_MARGIN_HI = 0.5
                    on_floor_reaching_goal = any(
                        xmin <= base_pos[0] <= xmax and ymin <= base_pos[1] <= ymax
                        and zmin - _FLOOR_REACH_MARGIN <= goal_np[2] <= zmax + _FLOOR_REACH_MARGIN
                        and zmin - _ROBOT_ON_FLOOR_MARGIN_LO <= base_pos[2] <= zmax + _ROBOT_ON_FLOOR_MARGIN_HI
                        for (xmin, xmax, ymin, ymax, zmin, zmax, cid) in active_aabbs if cid == SEG_ID_FLOOR
                    )
                    seeking_stairs = False
                    planning_goal = goal_np
                    # Which flight (if any) target_box below ends up aimed at, by footprint identity --
                    # populated right after target_box is picked, used by the height-aware "steer
                    # around too-tall stairs" nudge further down so it never treats the CURRENT
                    # target's own flight (further, higher-up steps of the very staircase being
                    # climbed one step at a time) as an obstacle to avoid. See that nudge's own
                    # comment for the real run this was diagnosed from.
                    _target_flight_boxes = set()
                    # Real run: live wall-match diagnostics ruled out the wall-proximity safety
                    # margin entirely (every wall matched near the Stair A entrance was correctly a
                    # guardrail with the override skipped) while the robot still took a huge detour
                    # ("big circle") instead of walking the ~1.5m straight to waypoint 1 (0.2m in
                    # front of the Stair A entrance, z=0.35 -- see open_scene_stairs.py's own
                    # waypoints list). Root cause: that waypoint has almost no height difference from
                    # ground level, so z_gap never exceeds 0.3 and the geometric override below never
                    # engaged for this leg -- control stayed 100% with the network's own raw
                    # trajectory prediction the whole approach, and that raw prediction avoids a
                    # direct line to the entrance on this scene. Extend the SAME trigger to also cover
                    # this case: if the CURRENT waypoint itself sits close to a stairs footprint (not
                    # the robot's own position -- the robot could be far away mid-detour when this
                    # first evaluates), take geometric control immediately rather than waiting for a
                    # height gap that this specific waypoint will never have.
                    # Deliberately NOT gated on `not on_floor_reaching_goal` -- that check exists to
                    # suppress the z_gap trigger's false positives from cosmetic per-waypoint marker
                    # offsets, but waypoint 1 here has no such offset to begin with (it's genuinely at
                    # ground level) and on_floor_reaching_goal is legitimately TRUE for it the whole
                    # approach (robot's on the right floor already). Confirmed via a real run that
                    # keeping that guard here silently left near_stairs_entrance permanently False and
                    # the whole fix inert -- this check's own 1.5m distance-to-a-stairs-footprint test
                    # is already the right filter for whether geometric guidance is warranted, no
                    # second gate needed.
                    # Real run: this ALSO used to gate on `not on_stairs`, and the exact same class
                    # of bug as the on_floor_reaching_goal one above showed up again, just further
                    # along the route -- on the landing (waypoint 2, platform center, only 0.72m from
                    # Stair A's own top step) the robot's XY footprint intermittently overlaps a
                    # stairs-tagged AABB (the top step it just climbed, or Stair B's not-yet-blocked
                    # footprint) even while genuinely standing on the flat landing, flipping
                    # `on_stairs` True for scattered replans and silently disabling this geometric aim
                    # each time -- control fell back to the raw network for those steps, which is what
                    # produced the long stuck-facing-a-wall spell on the landing a real run showed
                    # (confirmed via saved debug semantic frames: wall filling nearly the whole camera
                    # view for 2000+ steps while z stayed flat, i.e. not actually climbing, just
                    # facing a nearby wall on the landing). Like the earlier fix: this check's own
                    # distance-to-a-stairs-footprint test is already the right filter, on_stairs is
                    # irrelevant to it.
                    _ENTRANCE_SEEK_MARGIN = 1.5  # meters
                    near_stairs_entrance = False
                    for (sxmin, sxmax, symin, symax, _szmin, _szmax, scid) in active_aabbs:
                        if scid != SEG_ID_STAIRS:
                            continue
                        ecx = min(max(goal_np[0], sxmin), sxmax)
                        ecy = min(max(goal_np[1], symin), symax)
                        if math.hypot(goal_np[0] - ecx, goal_np[1] - ecy) < _ENTRANCE_SEEK_MARGIN:
                            near_stairs_entrance = True
                            break
                    # NOTE: this used to gate on `not on_stairs` -- guidance only got the robot
                    # lined up to *enter* the stairs, then handed 100% of control back to the
                    # network's own raw trajectory prediction for the entire climb. Confirmed via a
                    # real run (robot visibly followed the green raw-trajectory line back down mid-
                    # climb) that the raw prediction is NOT reliable once actually on the stairs on
                    # this custom scene -- so now stay geometrically in control for the whole climb,
                    # not just the approach.
                    # Explicit user request, confirmed via a real run: this branch was designed for a
                    # goal that sits right at the entrance with only a small height offset (e.g. the
                    # old waypoint 1, 0.2m in front, z=0.35) -- "close enough in XY, just aim straight
                    # at it" made sense there, back when the robot was still approaching from OUTSIDE
                    # the stairs (not on_stairs). It did NOT account for a goal that's ALSO far away in
                    # Z (e.g. a test goal with the entrance's own X/Y but raised to 2F height): once
                    # actually climbing and z_gap happened to shrink to <=0.3 near the top (the
                    # destination floor's own height being close to, but not yet confirmed reached),
                    # this branch fired anyway and aimed the raw network straight at the goal's literal
                    # X/Y -- which sits BEHIND/BELOW the robot at that point -- pulling it back down off
                    # the top of the flight it had almost finished climbing. Now also requires
                    # `not on_stairs`: once actually on a flight, the z_gap branch below (now gated on
                    # `not on_floor_reaching_goal` instead of a raw z_gap threshold, so it stays active
                    # through this same "z_gap shrank but not confirmed on the destination floor yet"
                    # window) takes over instead, which never aims at the goal's raw X/Y while there's
                    # still real climbing to verify.
                    if near_stairs_entrance and z_gap <= 0.3 and not on_stairs:
                        # Real run: routing this through the same target_box selection as the
                        # z_gap/on_stairs branch below caused a NEW oscillation right at the entrance
                        # -- as the robot's XY footprint straddled the boundary of the first step's
                        # AABB, on_stairs flipped True/False between replans, and each flip switched
                        # target_box between two different selection strategies (next-step-up vs.
                        # nearest-by-current-position), which could each resolve to a different stair
                        # step and yank the aim point around (dist_to_goal oscillating 1.26 <-> 2.3+
                        # for 3000+ steps, never committing). The waypoint itself is already placed
                        # right at the entrance (0.2m in front) -- just aim at it directly instead of
                        # re-deriving a target from the stairs geometry.
                        planning_goal = goal_np
                        seeking_stairs = True
                    elif not on_floor_reaching_goal:
                        # Dropped the old `z_gap > 0.3` requirement here (was the ONLY thing that used
                        # to gate this branch alongside on_floor_reaching_goal) -- confirmed via a real
                        # run that z_gap alone isn't the right signal for "is there still real climbing
                        # to do": on a flight whose own top height happens to land within 0.3m of the
                        # goal's Z (true here -- TOP_Z=1.00 vs goal Z=1.35), z_gap shrinks below that
                        # threshold before the robot has actually reached the destination floor,
                        # letting control fall through to the plain `planning_goal = goal_np` default
                        # below with nothing keeping it climbing. `on_floor_reaching_goal` (XY footprint
                        # AND z both confirmed matching some floor whose own range covers the goal) is
                        # the only reliable "actually done" signal already established elsewhere in
                        # this file -- reusing it here instead.
                        stairs_footprints = [
                            (xmin, xmax, ymin, ymax, zmin, zmax) for (xmin, xmax, ymin, ymax, zmin, zmax, cid)
                            in active_aabbs if cid == SEG_ID_STAIRS
                        ]
                        if stairs_footprints:
                            def _footprint_dist(box):
                                bxmin, bxmax, bymin, bymax = box[:4]
                                cx = min(max(base_pos[0], bxmin), bxmax)
                                cy = min(max(base_pos[1], bymin), bymax)
                                return math.hypot(base_pos[0] - cx, base_pos[1] - cy)

                            def _same_corridor(a, b, eps=0.02):
                                """True only for footprints sharing most of a full tread-width edge
                                (adjacency, not proximity) -- same test used elsewhere in this file
                                for clustering stairs AABBs into flights."""
                                ax0, ax1, ay0, ay1 = a[0], a[1], a[2], a[3]
                                bx0, bx1, by0, by1 = b[0], b[1], b[2], b[3]
                                ox = min(ax1, bx1) - max(ax0, bx0)
                                oy = min(ay1, by1) - max(ay0, by0)
                                if ox < -eps or oy < -eps:
                                    return False
                                mwx = min(ax1 - ax0, bx1 - bx0)
                                mwy = min(ay1 - ay0, by1 - by0)
                                return ox >= 0.5 * mwx or oy >= 0.5 * mwy

                            def _flood_flight(seed_boxes):
                                """Every footprint reachable from `seed_boxes` by adjacency -- the
                                WHOLE physical flight those seeds belong to, regardless of how far
                                any individual tread is from the robot's current position.
                                Adjacent treads must also be within one riser-ish of each other in
                                height: Stair.usdc's switchback stairwell stacks flights at the same
                                XY (Flight1 under Flight3), which XY adjacency alone merged into one
                                "flight" spanning 1F to 2F->3F."""
                                _flight = set(seed_boxes)
                                _to_visit = list(seed_boxes)
                                while _to_visit:
                                    _cur = _to_visit.pop()
                                    for _b in stairs_footprints:
                                        if (_b not in _flight and _same_corridor(_cur, _b)
                                                and abs(_b[5] - _cur[5]) <= 0.4):
                                            _flight.add(_b)
                                            _to_visit.append(_b)
                                return _flight

                            def _los_blocked(box, min_inside=0.1):
                                """True if the straight XY line from the robot to `box`'s center runs
                                through a wall/railing AABB (at the robot's height) for more than
                                `min_inside` meters. Explicit user request ("它現在卡在二樓要上去三樓的
                                欄杆"): at the top of Flight3 in Stair.usdc the next-flight pick was
                                Flight4's first tread, 0.4m away across the stairwell railing, and the
                                robot kept walking straight into the railing. min_inside tolerates the
                                robot itself standing just inside a padded railing box at a flight's
                                inner edge."""
                                x0, y0 = base_pos[0], base_pos[1]
                                dx = (box[0] + box[1]) / 2.0 - x0
                                dy = (box[2] + box[3]) / 2.0 - y0
                                seg_len = math.hypot(dx, dy)
                                if seg_len < 1e-6:
                                    return False
                                for wxmin, wxmax, wymin, wymax, wzmin, wzmax in wall_boxes_xy:
                                    if base_pos[2] < wzmin - _WALL_Z_MARGIN or base_pos[2] > wzmax + _WALL_Z_MARGIN:
                                        continue
                                    t0, t1 = 0.0, 1.0
                                    for p0, d, lo, hi in ((x0, dx, wxmin, wxmax), (y0, dy, wymin, wymax)):
                                        if abs(d) < 1e-9:
                                            if p0 < lo or p0 > hi:
                                                t0, t1 = 1.0, 0.0
                                                break
                                        else:
                                            ta, tb = sorted(((lo - p0) / d, (hi - p0) / d))
                                            t0, t1 = max(t0, ta), min(t1, tb)
                                    if t1 > t0 and (t1 - t0) * seg_len > min_inside:
                                        return True
                                return False

                            target_box = None
                            _entrance_approach = None
                            if on_stairs:
                                # already climbing: aim at the NEXT step up (smallest zmax that's
                                # still above the robot's current z) within the flight the robot is
                                # actually standing on, not "nearest" -- nearest is ambiguous
                                # (near-zero for every step whose x-range contains the robot) and
                                # can resolve to a step behind the robot, which is what was pulling
                                # it back down. Used to be "every footprint within 1.0m in XY", which
                                # in a switchback stairwell also caught the parallel flight across the
                                # railing (0.4m away) -- kept only as a fallback.
                                _standing_on = [
                                    b for b in stairs_footprints
                                    if b[0] <= base_pos[0] <= b[1] and b[2] <= base_pos[1] <= b[3]
                                    and b[5] - 0.1 <= base_pos[2] <= b[5] + 0.5
                                ]
                                same_flight = (list(_flood_flight(_standing_on)) if _standing_on
                                               else [b for b in stairs_footprints if _footprint_dist(b) < 1.0])
                                ahead = [b for b in same_flight if b[5] > base_pos[2] + 0.05 and not _los_blocked(b)]
                                if ahead:
                                    target_box = min(ahead, key=lambda b: b[5])
                            if target_box is None and not on_stairs:
                                # Explicit user request, extended further after a real run in
                                # L_SHAPED_SINGLE_GOAL mode (no waypoint funnel at all) still failed:
                                # plain "nearest climbable-by-absolute-height footprint" isn't enough
                                # on its own for a SHALLOW flight where every step's absolute top is
                                # within one _CLIMB_RISE_LIMIT of ground level (true here -- Stair A
                                # tops out at 0.60m) -- EVERY step in the flight passes that filter,
                                # so nearest-by-XY can still resolve to a MIDDLE step approached from
                                # the side/an angle other than the flight's own front. Confirmed: this
                                # pulled the robot into the flight's side, where the side-face wall
                                # patch pushed it back and it toppled. Fix: cluster the CURRENTLY
                                # ACTIVE stairs footprints into flights by adjacency (same test as the
                                # virtual edge-safety strips above, recomputed fresh here off
                                # `active_aabbs` rather than reused from setup, so an already-blocked
                                # flight like `aabbs_stair_a_blocked`'s Stair A is automatically
                                # excluded too), then aim at the LOWEST step -- the flight's own
                                # natural entrance -- of whichever flight is nearest. Never a middle
                                # step, regardless of which individual step happens to be closest in
                                # raw XY.
                                # `and not on_stairs` above: gated to ONLY the "not currently on any
                                # flight, need to go find one" case. Confirmed via a real run this
                                # entrance-seeking logic must NOT also run for on_stairs-but-ahead-
                                # empty (e.g. no step within 1.0m still higher than the robot, which
                                # can happen from an odd stance mid-climb) -- when already on a
                                # flight, that flight's own lowest step is very often the geometrically
                                # NEAREST entrance of any cluster, so this pulled the robot back down
                                # to its own entrance repeatedly instead of continuing up (observed as
                                # climbing partway, retreating to the bottom, climbing partway,
                                # retreating -- several full cycles before eventually breaking free).
                                # The on_stairs-but-ahead-empty case now falls through to the same
                                # local nearest-climbable fallback below instead, which stays close to
                                # the flight already committed to rather than jumping to "an entrance."
                                def _footprints_adjacent(a, b, eps=0.02):
                                    ax0, ax1, ay0, ay1 = a[0], a[1], a[2], a[3]
                                    bx0, bx1, by0, by1 = b[0], b[1], b[2], b[3]
                                    ox = min(ax1, bx1) - max(ax0, bx0)
                                    oy = min(ay1, by1) - max(ay0, by0)
                                    if ox < -eps or oy < -eps:
                                        return False
                                    mwx = min(ax1 - ax0, bx1 - bx0)
                                    mwy = min(ay1 - ay0, by1 - by0)
                                    return ox >= 0.5 * mwx or oy >= 0.5 * mwy

                                _nfp = len(stairs_footprints)
                                _fparent = list(range(_nfp))

                                def _ffind(i):
                                    while _fparent[i] != i:
                                        _fparent[i] = _fparent[_fparent[i]]
                                        i = _fparent[i]
                                    return i

                                for _fi in range(_nfp):
                                    for _fj in range(_fi + 1, _nfp):
                                        # height continuity too, same as _flood_flight: flights stacked
                                        # at the same XY (Stair.usdc switchback) are separate flights
                                        if (_footprints_adjacent(stairs_footprints[_fi], stairs_footprints[_fj])
                                                and abs(stairs_footprints[_fi][5] - stairs_footprints[_fj][5]) <= 0.4):
                                            _fri, _frj = _ffind(_fi), _ffind(_fj)
                                            if _fri != _frj:
                                                _fparent[_fri] = _frj

                                _fclusters = {}
                                for _fi, _fb in enumerate(stairs_footprints):
                                    _fclusters.setdefault(_ffind(_fi), []).append(_fb)

                                _entrance_steps = [min(_c, key=lambda b: b[5]) for _c in _fclusters.values()]
                                # Confirmed via a real run: without a climbability filter here, an
                                # UPPER flight's own entrance (e.g. Stair B's first step, top=0.70m --
                                # itself only reachable after climbing Stair A and crossing the
                                # landing) can be the geometrically NEAREST entrance-of-a-cluster from
                                # certain ground-level positions, even though it floats unreachably
                                # above the robot's actual current z -- the robot got pulled back and
                                # forth between Stair A's real (climbable) entrance and Stair B's
                                # inaccessible one, never committing to either, stuck the entire
                                # 13000-step budget. Same _CLIMB_RISE_LIMIT idea as the final fallback
                                # below, applied here first.
                                _ENTRANCE_CLIMB_RISE_LIMIT = 0.25  # meters, explicit user request (was 0.35)
                                # lower bound: an entrance a whole floor below (same XY, stacked flight)
                                # isn't reachable from here either
                                _climbable_entrances = [b for b in _entrance_steps
                                                        if base_pos[2] - 1.0 <= b[5] <= base_pos[2] + _ENTRANCE_CLIMB_RISE_LIMIT]
                                target_box = min(_climbable_entrances, key=_footprint_dist) if _climbable_entrances else None
                                # Explicit user request ("它現在掉下去了"): from Stair.usdc's landing, the
                                # direct line to Flight2's entrance cuts across the end of the stairwell
                                # railing; the robot wedged on it and eventually slipped through the
                                # 0.4m stairwell gap to 1F. If that direct line touches any wall/railing
                                # box, first walk to a point 0.8m out in front of the entrance (opposite
                                # the flight's ascent direction) and only then head in straight.
                                if target_box is not None and _los_blocked(target_box, min_inside=0.0):
                                    _ent_flight = sorted(_flood_flight({target_box}), key=lambda b: b[5])
                                    if len(_ent_flight) >= 2:
                                        _ecx = (target_box[0] + target_box[1]) / 2.0
                                        _ecy = (target_box[2] + target_box[3]) / 2.0
                                        _ux = (_ent_flight[1][0] + _ent_flight[1][1]) / 2.0 - _ecx
                                        _uy = (_ent_flight[1][2] + _ent_flight[1][3]) / 2.0 - _ecy
                                        _un = math.hypot(_ux, _uy)
                                        if _un > 1e-6:
                                            _ux, _uy = _ux / _un, _uy / _un
                                            # already out in front of the entrance -> go straight in
                                            _out = -((base_pos[0] - _ecx) * _ux + (base_pos[1] - _ecy) * _uy)
                                            if _out < 0.5:
                                                _entrance_approach = (_ecx - 0.8 * _ux, _ecy - 0.8 * _uy)

                            # Explicit user request: this whole block only ever runs when
                            # `not on_floor_reaching_goal` (the outer `elif` above requires it), so a
                            # nested `on_floor_reaching_goal` check here can never be true -- dead code
                            # that used to sit here (aiming at the raw goal_np once "done") never
                            # actually ran. Replaced with the intended simpler rule instead: while
                            # there's still a real height gap to close, XY doesn't matter at all yet --
                            # only ever aim at another climbable stairs footprint, never at the goal's
                            # raw XY. Once genuinely on the destination floor, `on_floor_reaching_goal`
                            # flips True, the outer `elif` stops matching, and control falls through to
                            # this function's own default (`planning_goal = goal_np`, set well above)
                            # automatically -- no special case needed here for that.
                            if target_box is None:
                                # Reached when either not on_stairs and the entrance-seeking block
                                # above found no cluster (defensive), or on_stairs with no "ahead" step
                                # within 1.0m (at/near the top of the CURRENT flight, still not on the
                                # destination floor -- e.g. just reaching the landing after Stair A).
                                # Confirmed via a real run: without excluding the flight just finished,
                                # the plain "nearest climbable-by-height" fallback kept re-selecting
                                # that SAME flight's own nearby steps (everything within reach becomes
                                # "climbable" at this height) instead of committing to the NEXT flight,
                                # so the robot climbed partway, drifted back to Stair A's own top,
                                # never reaching Stair B -- repeatedly, for the whole 13000-step budget.
                                _CLIMB_RISE_LIMIT = 0.25  # meters, explicit user request (was 0.35)
                                # Explicit user request ("它還是為了避開所以從二樓跌落"): confirmed via a real
                                # run that `same_flight`'s 1.0m proximity radius (fine for the LOCAL
                                # "ahead" step lookup above) is too short to cover a whole long flight
                                # once the robot is near one end of it -- treads more than 1m away
                                # along the SAME physical flight fell into `_other_footprints` and got
                                # picked as "climbable" (they're all far below current z once topped
                                # out), pulling the robot backward down its own flight (planning_goal
                                # landed on a middle Stair B tread while standing at the very top,
                                # z~3.1) and off the side once it walked through the virtual edge strip
                                # without stopping. Use the full adjacency-flood-filled flight instead
                                # of the proximity-only list, so every tread of the SAME physical
                                # flight is excluded regardless of distance from the robot right now.
                                _current_flight = _flood_flight(same_flight) if on_stairs else set()
                                _other_footprints = [b for b in stairs_footprints if b not in _current_flight]
                                _climbable_other = [b for b in _other_footprints
                                                    if base_pos[2] - 1.0 <= b[5] <= base_pos[2] + _CLIMB_RISE_LIMIT
                                                    and not _los_blocked(b)]
                                # Explicit user request ("它沒辦法走上濃煙的二樓該如何解"): confirmed via a
                                # real run this used to fall through to "nearest climbable footprint
                                # including the current flight" even when the robot had already
                                # genuinely topped out that flight (base z at/above its own highest
                                # tread, on_floor_reaching_goal not yet True only because the XY
                                # crossing onto the destination floor's own footprint hadn't happened
                                # yet -- the last ~0.1-0.3m). "Nearest by raw XY" then resolved to a
                                # MIDDLE tread of the SAME flight already climbed and passed, since the
                                # robot's XY sits almost on top of every tread's XY range near the very
                                # top -- pulling it backward down the flight it just finished, spinning
                                # in place at the top for the whole step budget (confirmed: heading_err
                                # ~170 degrees, wz oscillating +/-0.9, base z stuck at ~3.1-3.3 for
                                # 2000+ steps). Detected separately (topped out = no ahead step AND no
                                # other climbable flight AND z already >= this flight's own max height)
                                # so it's never confused with genuinely being mid-climb -- in that case
                                # there's nothing left to climb at all, so just walk toward the real
                                # goal's XY directly instead of hunting for another stair box.
                                _topped_out = (
                                    on_stairs and not _climbable_other and _current_flight
                                    and base_pos[2] >= max(b[5] for b in _current_flight) - 0.05
                                )
                                if _climbable_other:
                                    target_box = min(_climbable_other, key=_footprint_dist)
                                elif _topped_out:
                                    target_box = None
                                else:
                                    # Nothing else nearby and climbable (e.g. genuinely nowhere left to
                                    # go but back the way this flight came) -- fall back to whatever's
                                    # climbable including the current flight, same as before.
                                    _climbable = [b for b in stairs_footprints if b[5] <= base_pos[2] + _CLIMB_RISE_LIMIT]
                                    target_box = min(_climbable, key=_footprint_dist) if _climbable else min(
                                        stairs_footprints, key=_footprint_dist
                                    )
                            else:
                                _topped_out = False
                            if target_box is not None:
                                # target_box's own flight (see _flood_flight's own docstring) so the
                                # nudge below can exclude it -- every step further up THIS flight is,
                                # by definition, "too tall to climb directly from here," but that's
                                # not an obstacle to steer around, it's just the rest of the
                                # staircase target_box's own first step already commits to climbing
                                # one step at a time.
                                _target_flight_boxes = _flood_flight({target_box})
                            if _entrance_approach is not None:
                                planning_goal = np.array(
                                    [_entrance_approach[0], _entrance_approach[1], base_pos[2]], dtype=np.float32
                                )
                            elif target_box is not None:
                                nxmin, nxmax, nymin, nymax = target_box[:4]
                                planning_goal = np.array(
                                    [(nxmin + nxmax) / 2.0, (nymin + nymax) / 2.0, base_pos[2]], dtype=np.float32
                                )
                            elif _topped_out:
                                # Genuinely done climbing (see _topped_out's own comment above) --
                                # walk straight at the real goal's XY, keeping the robot's current z
                                # rather than the goal's (goal_np's z is the far final target; the
                                # actual remaining gap here is a small XY crossing, not more height).
                                planning_goal = np.array(
                                    [goal_np[0], goal_np[1], base_pos[2]], dtype=np.float32
                                )
                                # Explicit user request (2026-10-13, single 3F goal): "walk at the
                                # goal's XY" only works when the goal is the landing right ahead. With
                                # one far goal (3F, whose XY lies BEHIND a robot topping Flight1) it
                                # turned the robot around on the last treads, down two steps, up
                                # again, in circles. Step off the top of the flight instead: a point
                                # 0.8m past the top tread along the flight's own ascent direction;
                                # once on the landing (not on_stairs) the entrance logic takes over.
                                _fl = sorted(_current_flight, key=lambda b: b[5])
                                if len(_fl) >= 2:
                                    _tx, _ty = (_fl[-1][0] + _fl[-1][1]) / 2.0, (_fl[-1][2] + _fl[-1][3]) / 2.0
                                    _ax = _tx - (_fl[-2][0] + _fl[-2][1]) / 2.0
                                    _ay = _ty - (_fl[-2][2] + _fl[-2][3]) / 2.0
                                    _an = math.hypot(_ax, _ay)
                                    if _an > 1e-6:
                                        planning_goal = np.array(
                                            [_tx + 0.8 * _ax / _an, _ty + 0.8 * _ay / _an, base_pos[2]],
                                            dtype=np.float32)
                            seeking_stairs = True

                    # Explicit user request ("如果它前方標記是樓梯，但是高度很高他要繞過它"):
                    # whichever branch above picked planning_goal, if the CLOSEST stairs footprint to
                    # the robot's CURRENT position is too tall to climb from here (same idea as
                    # target_box's own _CLIMB_RISE_LIMIT filter above, just applied unconditionally
                    # instead of only inside the z_gap branch -- near_stairs_entrance's branch above
                    # deliberately skips that filter entirely, see its own comment on why merging the
                    # two selection strategies previously caused an oscillation regression), nudge
                    # planning_goal sideways, away from that footprint, instead of leaving the network
                    # to walk straight at/through it. Only ever ADDS a lateral offset on top of the
                    # existing aim point -- never replaces the aim-selection strategy itself, so this
                    # can't reproduce that same regression.
                    # Explicit user request, after a real run in L_SHAPED_SINGLE_GOAL mode got stuck
                    # for the ENTIRE 13000-step budget wandering near ground level, never once
                    # approaching Stair A's own (far away, correctly-selected) entrance: this nudge
                    # originally fired for ANY too-tall footprint near the robot's current position,
                    # with no check on whether it's actually in the way of planning_goal. The spawn
                    # point sits right next to Stair B, whose upper treads are always "too tall" from
                    # ground level -- so the nudge fired almost every replan, continuously shoving
                    # planning_goal away from Stair B even while the robot was correctly trying to
                    # head toward Stair A's entrance in an unrelated direction, fighting that pull
                    # the entire episode. Now also requires the footprint to be roughly ALONG the
                    # path to planning_goal (cosine of the angle between robot->footprint and
                    # robot->planning_goal above _NUDGE_ALIGN_MIN), not just nearby in any direction.
                    _NUDGE_CLIMB_RISE_LIMIT = 0.25  # meters, matches target_box's own threshold (explicit user request, was 0.35)
                    _NUDGE_SEEK_RADIUS = 1.2  # meters -- only a footprint actually in the way counts
                    _NUDGE_ALIGN_MIN = 0.5  # cos(60deg) -- footprint must be within ~60 degrees of
                    # the direction the robot is already trying to go
                    _gdx, _gdy = planning_goal[0] - base_pos[0], planning_goal[1] - base_pos[1]
                    _gd_norm = math.hypot(_gdx, _gdy)
                    _nearest_toofall, _nearest_toofall_d = None, 1e9
                    for (txmin, txmax, tymin, tymax, tzmin, tzmax, tcid) in active_aabbs:
                        if tcid != SEG_ID_STAIRS or tzmax <= base_pos[2] + _NUDGE_CLIMB_RISE_LIMIT:
                            continue
                        # a flight stacked overhead (Stair.usdc: Flight3 directly above Flight1)
                        # can't be in the way -- only footprints starting at/below the robot count
                        if tzmin > base_pos[2] + 0.5:
                            continue
                        if (txmin, txmax, tymin, tymax, tzmin, tzmax) in _target_flight_boxes:
                            # the current target's own flight -- climbing it one step at a time is
                            # the whole point, not an obstacle to nudge away from (see this block's
                            # setup comment above where _target_flight_boxes is populated)
                            continue
                        tcx = min(max(base_pos[0], txmin), txmax)
                        tcy = min(max(base_pos[1], tymin), tymax)
                        tdx, tdy = tcx - base_pos[0], tcy - base_pos[1]
                        td = math.hypot(tdx, tdy)
                        if td < 1e-3 or _gd_norm < 1e-3:
                            aligned = True  # standing on it, or no real target direction yet -- don't
                            # let the alignment check suppress a genuinely urgent case
                        else:
                            aligned = (tdx * _gdx + tdy * _gdy) / (td * _gd_norm) >= _NUDGE_ALIGN_MIN
                        if aligned and td < _nearest_toofall_d:
                            _nearest_toofall_d, _nearest_toofall = td, (tcx, tcy)
                    if _nearest_toofall is not None and _nearest_toofall_d < _NUDGE_SEEK_RADIUS:
                        _tcx, _tcy = _nearest_toofall
                        _away_x, _away_y = base_pos[0] - _tcx, base_pos[1] - _tcy
                        _away_norm = math.hypot(_away_x, _away_y)
                        if _away_norm < 1e-3:
                            # standing right on top of the footprint's XY projection -- push
                            # perpendicular to the current aim direction instead (arbitrary but
                            # consistent choice of side, +90 degrees)
                            _gdx, _gdy = planning_goal[0] - base_pos[0], planning_goal[1] - base_pos[1]
                            _away_x, _away_y = -_gdy, _gdx
                            _away_norm = math.hypot(_away_x, _away_y) or 1.0
                        _NUDGE_MAGNITUDE = 1.0  # meters
                        planning_goal = np.array([
                            planning_goal[0] + _NUDGE_MAGNITUDE * _away_x / _away_norm,
                            planning_goal[1] + _NUDGE_MAGNITUDE * _away_y / _away_norm,
                            planning_goal[2],
                        ], dtype=np.float32)
                        seeking_stairs = True

                    if GUIDANCE == "off":
                        planning_goal, seeking_stairs = goal_np, False
                    if GUIDANCE == "perception":
                        # "is the goal on another level?" -- goal height vs. the robot's own (sim truth for
                        # now; leg odometry on the real robot). base_pos rides 0.30 m above the feet here.
                        # VIPLANNER_HEIGHT=odom: leg odometry's base height instead (same 0.30 m convention:
                        # it starts from base_pos[2] and only integrates changes)
                        _feet_z = (_leg_odom.base_z if _HEIGHT_FROM_ODOM else float(base_pos[2])) - 0.30
                        planning_goal, seeking_stairs = goal_np, False
                        _fwd = quat_rotate_wxyz(base_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
                        _pitch_up = math.asin(max(-1.0, min(1.0, float(_fwd[2]))))
                        _level_ticks = _stair_seeker.track_level(_pitch_up, base_pos[:2])
                        # Leave climbing only once the height matches AND the body has been level for a
                        # while. Height alone switched 3 steps below the top (the goal marker floats above
                        # its floor), and the straight line to the goal from there cut the corner into the
                        # stairwell's centre wall -- a foot jammed there (user: "他的腳卡住", 2026-10-14).
                        # up or down is decided while the height gap is clear and then kept until the climb
                        # ends: the goal marker floats 0.35 m above its floor, so on the last steps down
                        # the feet dropped below it, "down" flipped to "up" and it turned back up the
                        # flight it was finishing (2026-10-14)
                        # With leg odometry the tolerance grows with the height travelled: it under-reads a
                        # climb by ~6% (foot balls rolling round the step edges), and at the 3F top after
                        # 9 m it still read the goal 0.88 m up, searched 3F for stairs and walked off the
                        # scene edge (2026-10-15). +10% of the travel stays well under one flight (2.3 m).
                        if _HEIGHT_FROM_ODOM:
                            _percep_travel = max(_percep_travel, abs(_feet_z - _percep_start_z)) if _percep_start_z is not None else 0.0
                            if _percep_start_z is None:
                                _percep_start_z = _feet_z
                        _gap_tol = _PERCEP_CLIMB_GAP + (_PERCEP_ODOM_REL_ERR * _percep_travel if _HEIGHT_FROM_ODOM else 0.0)
                        if abs(goal_np[2] - _feet_z) > _gap_tol:
                            _percep_descend = goal_np[2] - _feet_z < 0.0
                        _percep_climbing = (abs(goal_np[2] - _feet_z) > _gap_tol
                                            or (_percep_climbing and _level_ticks < _PERCEP_LEVEL_TICKS))
                        _dh, _dw = depth_np.shape
                        _patch = depth_np[_dh // 3: 2 * _dh // 3, 2 * _dw // 5: 3 * _dw // 5]
                        _patch = _patch[np.isfinite(_patch)]
                        _front_clear = float(np.percentile(_patch, 20)) if _patch.size else float("inf")
                        # extra cameras: every 2nd replan tick, each through Mask2Former, points kept at
                        # half resolution -- they only add stair points, ViPlanner never sees them
                        if _extra_cams and _percep_climbing and (step // REPLAN_EVERY_N_STEPS) % 2 == 0:
                            _extra_obs = []
                            for _ec in _extra_cams:
                                try:
                                    _ed = _ec["cam"].get_depth()
                                    if _ed is None or getattr(_ed, "ndim", 0) != 2:
                                        continue
                                    _ed = np.asarray(_ed, dtype=np.float32)
                                    if _USE_M2F:
                                        _es = _m2f_segment(None, _ec["cam"])
                                        if _es is None:
                                            continue
                                        _exyz = reproject_depth_to_world(_ed, _ec["world_pos"], wxyz_to_xyzw(_ec["world_quat"]),
                                                                         _ec["hfov"])
                                    else:
                                        # M2F=0: the same perfect (scene-box) semantics as the main camera
                                        # (user request 2026-10-15)
                                        _es, _exyz = semantic_color_image_with_points(
                                            _ed, _ec["world_pos"], wxyz_to_xyzw(_ec["world_quat"]), active_aabbs, _ec["hfov"])
                                    _extra_obs.append((_ec["name"], _es[::2, ::2], _exyz[::2, ::2]))
                                except Exception as e:
                                    print(f"[viplanner] WARNING: extra camera {_ec['name']} failed:", repr(e))
                            if sem_extra_pub is not None and _extra_obs:
                                try:
                                    _tiles = {n: t for n, t, _ in _extra_obs}
                                    _blank = np.zeros((180, 320, 3), dtype=np.uint8)
                                    _mos = np.concatenate([
                                        np.concatenate([_tiles.get("left_bottom", _blank), _tiles.get("front_bottom", _blank)], 1),
                                        np.concatenate([_tiles.get("right_bottom", _blank), _tiles.get("rear", _blank)], 1)], 0)
                                    sem_extra_pub.publish(make_image_msg(_mos, frame_id="go2_camera"))
                                except Exception as e:
                                    print("[viplanner] WARNING: extra semantic publish failed:", repr(e))
                        _sem_all, _xyz_all = sem, world_xyz
                        if _extra_obs:
                            _sem_all = np.concatenate([sem.reshape(-1, 3)] + [t.reshape(-1, 3) for _, t, _ in _extra_obs])
                            _xyz_all = np.concatenate([world_xyz.reshape(-1, 3)] + [x.reshape(-1, 3) for _, _, x in _extra_obs])
                        if _HEIGHT_FROM_ODOM:
                            # the points come from the TRUE camera pose; on the real robot they hang off the
                            # odometry pose, like the feet height. Mixing the two made a 1 m odometry error
                            # look like the next flight was out of reach (2026-10-15): shift them by the
                            # odometry's height error so both share one frame, as on the robot
                            _xyz_all = _xyz_all.copy()
                            _xyz_all[..., 2] += _leg_odom.base_z - float(base_pos[2])
                        if _percep_climbing:
                            _sub, _percep_state = _stair_seeker.update(
                                _sem_all, _xyz_all, base_pos[:2], math.atan2(_fwd[1], _fwd[0]), _feet_z,
                                pitch_up=_pitch_up, front_clear=_front_clear,
                                descend=_percep_descend)
                            planning_goal = np.array([_sub[0], _sub[1], base_pos[2]], dtype=np.float32)
                            # every 10th replan tick: dump the frame for offline debugging of the detector
                            if _percep_dump_dir and (step // REPLAN_EVERY_N_STEPS) % 10 == 0:
                                np.savez_compressed(
                                    _os_check.path.join(_percep_dump_dir, f"ep{episode:02d}_step{step:05d}_{_percep_state}.npz"),
                                    sem=sem, depth=depth_np, world_xyz=world_xyz.astype(np.float32),
                                    base_pos=base_pos, base_quat=base_quat, feet_z=_feet_z, goal=goal_np)
                            seeking_stairs = True
                        else:
                            _percep_state = "same_level"
                        if step % 200 < REPLAN_EVERY_N_STEPS:
                            print(f"[viplanner] perception step {step}: {_percep_state} target={_stair_seeker.target} "
                                  f"goal_above_feet={goal_np[2] - _feet_z:.2f} sub_goal=({planning_goal[0]:.2f},{planning_goal[1]:.2f}) "
                                  f"level_ticks={_level_ticks} front_clear={_front_clear:.2f} pitch={math.degrees(_pitch_up):+.0f} "
                                  f"seen_now={_stair_seeker.last_obs} | map: {_stair_seeker.summary()}")
                    goal_t = torch.tensor([planning_goal], dtype=torch.float, device=device)

                    goal_robot_frame = vip.goal_transformer(goal_t, cam_pos_t, cam_quat_t)
                    _, traj_cam, fear = vip.plan_dual(depth_t, sem_t, goal_robot_frame)
                    fear_val = fear.item()
                    if _diag_fear_prev is not None:
                        _diag_fear_deltas.append(abs(fear_val - _diag_fear_prev))
                    _diag_fear_prev = fear_val
                    idx = min(LOOKAHEAD_IDX, traj_cam.shape[1] - 1)
                    px, py = traj_cam[0, idx, 0].item(), traj_cam[0, idx, 1].item()
                    new_heading_error = math.atan2(py, px)

                    # while seeking stairs, don't trust the network's own trajectory prediction for
                    # heading -- confirmed via a real run that changing ViPlanner's *goal input* to
                    # the nearest stairs footprint wasn't enough on its own; it still relies on the
                    # network's own (often noisy) path output to actually get there. Compute the
                    # heading directly from geometry instead (same atan2-of-body-frame-direction
                    # approach as the wall-repulsion escape), bypassing the network's prediction
                    # entirely for this one signal -- fear_val above still comes from the network,
                    # so obstacle-based slowdown still applies while seeking.
                    if seeking_stairs and GUIDANCE in ("full", "perception"):
                        to_stairs = np.array(
                            [planning_goal[0] - base_pos[0], planning_goal[1] - base_pos[1], 0.0], dtype=np.float32
                        )
                        to_stairs_body = quat_rotate_inverse_wxyz(base_quat, to_stairs)
                        new_heading_error = math.atan2(to_stairs_body[1], to_stairs_body[0])

                    if wp_idx == 1 and step % 200 == 1:
                        gr = goal_robot_frame[0].tolist() if hasattr(goal_robot_frame, "shape") else goal_robot_frame
                        print(f"[viplanner] DIAG wp2 step {step}: base=({base_pos[0]:.2f},{base_pos[1]:.2f},{base_pos[2]:.2f}) "
                              f"goal_robot_frame={gr} lookahead(px,py)=({px:.2f},{py:.2f}) "
                              f"heading_err_deg_raw={math.degrees(new_heading_error):.1f} fear={fear_val:.2f}")

                    # defensive: a NaN/inf anywhere in this chain (goal transform, network output,
                    # atan2 on degenerate px/py) would otherwise silently propagate into wz/vx
                    # every COMMAND_HOLD_STEPS tick forever -- the robot then just stands frozen
                    # (PD controller holding current position as an effectively-NaN target) while
                    # this replan block keeps re-running every REPLAN_EVERY_N_STEPS at full CPU
                    # cost, which is exactly the "frozen position + climbing CPU" failure mode this
                    # is guarding against.
                    if math.isfinite(new_heading_error) and math.isfinite(fear_val):
                        _heading_cos += (math.cos(new_heading_error) - _heading_cos) * HEADING_SMOOTH_RATE
                        _heading_sin += (math.sin(new_heading_error) - _heading_sin) * HEADING_SMOOTH_RATE
                        heading_error = math.atan2(_heading_sin, _heading_cos)
                    else:
                        print(f"[viplanner] WARNING: non-finite planner output at step {step} "
                              f"(heading={new_heading_error}, fear={fear_val}, px={px}, py={py}, "
                              f"goal_robot_frame={goal_robot_frame.tolist()}) -- holding last heading_error")

                    if pc_pub is not None:
                        try:
                            # reuse the world_xyz already computed by semantic_color_image_with_points
                            # above instead of calling Isaac Sim's camera.get_pointcloud() a second
                            # time -- that call's 'annotator not available' fallback path is a slow
                            # per-pixel depth recompute, and was firing once per replan tick.
                            cloud_pts = world_xyz.reshape(-1, 3)
                            stride = max(1, cloud_pts.shape[0] // 5000)  # ~5000 pts stays under the 65500B MaxMessageSize limit (12B/point)
                            cloud_pts = cloud_pts[::stride]
                            pc_pub.publish(make_pointcloud2(cloud_pts, frame_id="world"))
                        except Exception as e:
                            print("[viplanner] WARNING: point cloud publish failed:", repr(e))

                    # draw the planned trajectory live in the viewport
                    traj_world = vip.path_transformer(traj_cam, cam_pos_t, cam_quat_t)
                    pts = traj_world[0].detach().cpu().numpy()

                    if depth_pub is not None:
                        try:
                            depth_pub.publish(make_depth_image_msg(
                                depth_np, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP or 20.0, frame_id="go2_camera",
                                stride=max(1, int(round(max(depth_np.shape) / 480.0)))))
                        except Exception as e:
                            print("[viplanner] WARNING: depth image publish failed:", repr(e))
                    if route_pub is not None:
                        try:
                            _route_pts.append((float(base_pos[0]), float(base_pos[1]), float(base_pos[2])))
                            if len(_route_pts) > 1200:  # keep the message small: thin out the older half
                                _route_pts = _route_pts[:600:2] + _route_pts[600:]
                            _rz_lo, _rz_hi = float(start_pos[0][2]), float(waypoints[-1][2])
                            _rp = np.asarray(_route_pts)
                            route_pub.publish(_route_marker(
                                "walked_path", 0, 4, 0.06, points=_rp, colors=_route_height_colors(_rp[:, 2], _rz_lo, _rz_hi)))
                            # ViPlanner's own plan: yellow = low fear ... red = high fear. "plan_now" is the
                            # current one (thick); every 5th is also kept as a thin line (last 40), as in the png.
                            _fc = (1.0, max(0.0, 1.0 - float(fear_val)), 0.1, 1.0)
                            route_pub.publish(_route_marker("plan_now", 0, 4, 0.05, rgba=_fc, points=pts))
                            if _route_tick % 5 == 0:
                                route_pub.publish(_route_marker(
                                    "plan_history", (_route_tick // 5) % 40, 4, 0.015, rgba=_fc[:3] + (0.7,), points=pts))
                            route_pub.publish(_route_marker(
                                "planning_goal", 0, 2, 0.18, rgba=(0.3, 0.5, 1.0, 1.0), pos=planning_goal))
                            if _route_tick % 25 == 0:  # static things, re-sent now and then for a late-started rviz
                                route_pub.publish(_route_marker("start", 0, 2, 0.25, rgba=(1.0, 1.0, 1.0, 1.0), pos=start_pos[0]))
                                for _wi, _w in enumerate(waypoints):
                                    route_pub.publish(_route_marker("waypoints", _wi, 2, 0.35, rgba=(1.0, 0.0, 0.0, 1.0), pos=_w))
                                for _ns, _cids, _col in (("scene_walls", (SEG_ID_WALL, SEG_ID_RAILING), (0.6, 0.6, 0.6, 0.8)),
                                                         ("scene_stairs", (SEG_ID_STAIRS,), (0.2, 0.8, 0.3, 0.8))):
                                    _seg = []
                                    for (_x0, _x1, _y0, _y1, _z0, _z1, _cid) in aabbs:
                                        if _cid in _cids:  # footprint rectangle at the box's top face
                                            _c4 = [(_x0, _y0, _z1), (_x1, _y0, _z1), (_x1, _y1, _z1), (_x0, _y1, _z1)]
                                            for _k in range(4):
                                                _seg += [_c4[_k], _c4[(_k + 1) % 4]]
                                    for _ci in range(0, len(_seg), 800):
                                        route_pub.publish(_route_marker(_ns, _ci // 800, 5, 0.02, rgba=_col, points=_seg[_ci:_ci + 800]))
                            _route_tick += 1
                        except Exception as e:
                            print("[viplanner] WARNING: route marker publish failed:", repr(e))
                    if _GHOST:
                        _ghost_plans.append((step, wp_idx, float(fear_val), pts.copy(),
                                             np.asarray(planning_goal, dtype=np.float32).copy()))
                    draw_iface.clear_lines()
                    if pts.shape[0] >= 2:
                        color = (1.0, 0.4, 0.1, 1.0) if fear_val > FEAR_THRESHOLD else (0.4, 1.0, 0.1, 1.0)
                        starts = [pts[i].tolist() for i in range(pts.shape[0] - 1)]
                        ends = [pts[i + 1].tolist() for i in range(pts.shape[0] - 1)]
                        colors = [color] * len(starts)
                        sizes = [4.0] * len(starts)
                        draw_iface.draw_lines(starts, ends, colors, sizes)
                else:
                    print("[viplanner] WARNING: camera.get_depth() returned None at step", step)

            # full 3D distance, not just XY -- an XY-only check let the robot "reach" an upper-floor
            # waypoint while still standing on the floor below it, whenever the two floors' XY
            # footprints overlap (which they do here: the 2F room sits directly above part of the
            # stairwell). Confirmed: it reported "reached waypoint 2/3" at z=0.30 (ground level)
            # while waypoint 2's actual target z was ~3.35 (2F), because it had wandered back under
            # the door's XY position at ground level after failing to find stair B.
            dist_to_goal = float(np.linalg.norm(base_pos - goal_np))
            # Explicit user request (U-shaped scene, single far-away 2F goal + VIPLANNER_STOP_ON_Z_MATCH):
            # the stall watchdog below was written assuming dist_to_goal (to goal_np, the CURRENT
            # waypoint) tracks whatever's actually being pursued right now -- true under the old
            # multi-waypoint design, where goal_np itself advanced to each near sub-goal in turn. With
            # a single far-away final goal and target_box/planning_goal picking a much closer dynamic
            # sub-target (the stairs entrance, the next step up, ...), goal_np no longer moves, so
            # climbing 4-5 steps up Stair A barely dents the huge remaining XY distance to the distant
            # final goal -- confirmed via a real run: the robot climbed to z=0.59 then the stall
            # watchdog (measuring against goal_np) read that as "no real progress" and forced a
            # recovery turn / wall-retreat, undoing the climb and leaving it oscillating at the
            # entrance indefinitely. Track progress against `planning_goal` (what's actually being
            # steered toward this step) instead -- goal_np is still used for the slowdown-near-goal
            # and REACH/Z-match checks below, where the real final goal is exactly what's wanted.
            stall_progress_dist = float(np.linalg.norm(base_pos - planning_goal))

            # VIPLANNER_RATES: latch the new command on the planner tick itself (plan ticks no longer
            # fall on step % COMMAND_HOLD_STEPS)
            if (_do_plan if _RATES is not None else step % COMMAND_HOLD_STEPS == 1):
                if abs(heading_error) < YAW_DEADZONE_RAD:
                    _viplanner_target_wz = 0.0
                else:
                    _viplanner_target_wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, YAW_GAIN * heading_error))
                _viplanner_target_vx = MAX_SPEED if abs(heading_error) < 1.0 else MAX_SPEED * 0.3

                # fear-based slowdown: the higher ViPlanner's fear (obstacle) signal, the slower we
                # go -- never fully frozen (their own original script deliberately avoids a hard
                # fear gate, since this checkpoint can report high fear constantly on out-of-
                # distribution scenes and a hard stop just freezes it forever), but a real
                # continuous braking response so it doesn't plow into walls at full speed.
                # MIN_VX_SCALE floor raised 0.15 -> 0.4 (was giving vx=0.09, which is BELOW the
                # threshold this HIMLoco checkpoint needs to produce a real walking gait from a
                # standing start -- it just stood in place instead, which also freezes the camera
                # view, so ViPlanner keeps recomputing the same fear/heading forever: a genuine
                # stuck deadlock, reproduced deterministically with a nearby first waypoint on the
                # stairs scene, not the CPU-climbing hang seen elsewhere this session).
                MIN_VX_SCALE = 0.4
                fear_scale = max(MIN_VX_SCALE, min(1.0, 1.0 - (fear_val - 0.2) / 0.3))
                _viplanner_target_vx *= fear_scale

                # slow down on approach to the goal so it doesn't collide with the goal marker
                if dist_to_goal < 1.0:
                    _viplanner_target_vx = min(_viplanner_target_vx, max(MIN_VX_SCALE * MAX_SPEED, MAX_SPEED * dist_to_goal))

                # going DOWN a flight (perception mode): walk it, don't run it. At MAX_SPEED it dropped
                # 3-4 steps at once off the top of a flight, ended nose-down 35 deg with the front legs
                # jammed and the gait frozen (user: "它下樓卡住了", 2026-10-14).
                if (GUIDANCE == "perception" and _percep_climbing
                        and _percep_descend and _stair_seeker.target is not None):
                    _viplanner_target_vx = min(_viplanner_target_vx, _PERCEP_DESCEND_MAX_VX)

            # every step (not just on a fresh COMMAND_HOLD_STEPS tick): start from ViPlanner's own
            # current intent, unmodified by any earlier step's override -- see the per-episode setup
            # comment for why this reset is here.
            target_wz, target_vx = _viplanner_target_wz, _viplanner_target_vx

                # NOTE: previously added a uniform floor here (target_vx = max(target_vx,
                # MIN_VX_SCALE*MAX_SPEED)) to cover the heading-based slowdown branch above
                # (MAX_SPEED*0.3=0.18, below the 0.24 floor used elsewhere). Reverted -- it fixed
                # the specific freeze it targeted but caused a *worse* regression: with vx forced
                # up to 0.24 even during a hard turn (wz near MAX_YAW_RATE), the robot no longer
                # has the option to slow down and pivot near-in-place, so instead it drives a
                # steady arc (radius ~ vx/wz) and got stuck in a stable circling loop for 4500+
                # steps rather than ever straightening out. The actual root cause of the freeze
                # this was meant to fix turned out to be the wall-repulsion block below cancelling
                # the heading-driven target_wz to ~0 (a blend, not a real fix) -- addressed there
                # instead of papering over it with a vx floor here.

            # wall-proximity safety margin -- checked every step (not just every COMMAND_HOLD_STEPS,
            # since "how close to a wall" changes faster than ViPlanner's own replan cadence), but
            # folded INTO target_vx/target_wz rather than hard-overriding vx/wz directly. It still
            # goes through the SMOOTH_RATE easing below like every other command change, so it can't
            # reintroduce the lurching/surging that COMMAND_HOLD_STEPS+SMOOTH_RATE were tuned to fix.
            #
            # First version of this hard-overrode wz using the sign of a cross product between the
            # dog's *current* heading and the escape direction -- unstable, because that sign flips
            # every step as the dog's own heading rotates, so it just spun in place near any wall
            # instead of committing to one escape direction. Replaced with a continuous body-frame
            # escape angle (same atan2/YAW_GAIN shape as the normal ViPlanner heading_error), which
            # is smooth in the dog's own heading and doesn't flip sign from noise alone.
            #
            # Also: the first version floored vx at MAX_SPEED*0.15 (~0.09 m/s) right at the wall --
            # this is BELOW the same MIN_VX_SCALE floor documented above that this HIMLoco checkpoint
            # needs to produce a real gait from standing, so it reproduced the exact "frozen position"
            # deadlock the MIN_VX_SCALE fix was for, just triggered by proximity instead of fear/goal-
            # approach. Floor kept consistent with MIN_VX_SCALE * MAX_SPEED here too.
            # Tried a full blend (target_wz = target_wz*(1-severity) + escape_wz*severity): can
            # cancel to ~0 when escape opposes the goal heading at moderate severity (0.9 vs -0.9
            # at severity=0.5 nets to 0) -- reproduced once as a 2000+ step deadlock at a door-frame
            # corner, heading_err~109 deg yet wz==0.00 the whole time, nothing ever perturbing it.
            # Tried a full hard-switch instead (escape always fully owns target_wz once inside the
            # margin): fixes that deadlock, but now a real run couldn't even commit to entering the
            # 1.5m-wide stair corridor at all -- any brief dip near either wall while lining up with
            # the entrance yanked steering fully away from the goal, so it just paced around outside
            # never climbing. Settled on: blend normally (gentle, lets it thread a corridor entrance
            # like the working baseline runs did), but only when *deep* inside the margin (severity
            # > 0.7 -- i.e. actually at risk of contact, not just a normal close pass) AND the blend
            # would net to a near-zero command (the specific cancellation pattern that caused the
            # door-frame deadlock), fall back to pure escape_wz to break that specific standoff.
            # stair-centerline assist -- RE-ENABLED per explicit user request, using the generalized
            # (any-flight-orientation) version from `_stair_centerline_correction` above instead of
            # the old Stair-A-only, X-matching-only one. The original was disabled because it kept
            # fighting ViPlanner's own legitimate guidance near the very top of a flight (locked
            # target_wz at -0.90 for 15+ steps while ViPlanner asked for +0.66..+0.90, nowhere near a
            # real wall) -- `_stair_centerline_correction`'s own `_CENTERLINE_TOP_CLEARANCE` gate
            # addresses that directly by releasing control before reaching each flight's own top,
            # instead of fighting the turn the way the old always-on version did.
            _c_axis, _c_center, _c_current = _stair_centerline_correction(base_pos[0], base_pos[1], base_pos[2])
            if _c_axis is not None:
                lateral_error = _c_current - _c_center
                if abs(lateral_error) > CENTERLINE_DEADZONE:
                    pull_world = (np.array([0.0, -lateral_error, 0.0], dtype=np.float32) if _c_axis == "y"
                                  else np.array([-lateral_error, 0.0, 0.0], dtype=np.float32))
                    pull_body = quat_rotate_inverse_wxyz(base_quat, pull_world)
                    pull_heading_error = math.atan2(pull_body[1], pull_body[0])
                    pull_wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, YAW_GAIN * pull_heading_error))
                    centering_strength = min(1.0, CENTERLINE_GAIN * abs(lateral_error))
                    target_wz = target_wz * (1.0 - centering_strength) + pull_wz * centering_strength

            wall_dist, wall_dx, wall_dy, _wall_box = _nearest_wall(base_pos[0], base_pos[1], base_pos[2])
            # push_body: world-frame away-from-wall vector (wall_dx,wall_dy), rotated into the body
            # frame -- still needed below for the escape heading itself, just not for the margin
            # width any more (see the half-diagonal comment above).
            push_body = quat_rotate_inverse_wxyz(base_quat, np.array([wall_dx, wall_dy, 0.0], dtype=np.float32))
            _is_guardrail = _wall_box is not None and (
                round(_wall_box[0], 6), round(_wall_box[1], 6), round(_wall_box[2], 6), round(_wall_box[3], 6)
            ) in _guardrail_xy_set
            _is_edge_strip = _wall_box is not None and (
                round(_wall_box[0], 6), round(_wall_box[1], 6), round(_wall_box[2], 6), round(_wall_box[3], 6)
            ) in _edge_strip_xy_set
            # Take 5, per explicit user request ("把對角線加回去好了") after the real root cause of
            # the stairs-entrance hesitation turned out to be unrelated to this margin at all (see
            # the near_stairs_entrance / on_floor_reaching_goal / on_stairs gating fixes elsewhere in
            # this function) -- take 4's guardrail skip (effective_wall_dist=0.0, no soft avoidance
            # at all for guardrails) is no longer needed for that problem and was its own downside:
            # zero soft avoidance meant the only thing stopping contact with a guardrail was PhysX's
            # hard collision, which reads as the robot visibly walking right up against/scraping the
            # rail. Now that the actual hesitation bug is fixed, go back to the same conservative
            # half-diagonal margin for guardrails as for real walls -- gives a gentle push-away
            # before actual contact instead of none.
            effective_wall_dist = max(0.2, min(0.6, BODY_CLEARANCE + GO2_HALF_DIAGONAL))
            if _wall_box is not None and wall_dist < 0.6 and not _is_guardrail and step % 10 == 0:
                print(f"[viplanner] DIAG wall match step={step} pos=({base_pos[0]:.2f},{base_pos[1]:.2f},{base_pos[2]:.2f}) "
                      f"wall_box={tuple(round(v,3) for v in _wall_box)} wall_dist={wall_dist:.2f} "
                      f"is_guardrail={_is_guardrail} is_edge_strip={_is_edge_strip} effective_wall_dist={effective_wall_dist:.2f}")
            if wall_dist < effective_wall_dist:
                escape_heading_error = math.atan2(push_body[1], push_body[0])
                escape_wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, YAW_GAIN * escape_heading_error))
                severity = 1.0 - max(wall_dist, 0.0) / effective_wall_dist  # 0 at the margin, 1 at the wall
                blended_wz = target_wz * (1.0 - severity) + escape_wz * severity
                if severity > 0.7 and abs(blended_wz) < 0.15 and abs(escape_wz) > 0.15:
                    target_wz = escape_wz
                else:
                    target_wz = blended_wz
                if _is_edge_strip and severity > 0.6:
                    # Explicit user request ("它還是為了避開所以從二樓跌落"): confirmed via a real run
                    # that the MIN_VX_SCALE floor below (kept for real walls/guardrails, where PhysX
                    # collision is the actual backstop and a brief slow approach while turning away is
                    # safe) let the robot keep walking FORWARD through a virtual edge-safety strip --
                    # no real collider exists there, so nothing physically stopped it -- while it spent
                    # 20+ steps spinning to resolve a large heading error at wall_dist~0, drifting all
                    # the way across the strip and off the actual tread edge. Edge strips mark a real
                    # fall hazard with no physical backstop, so let vx go to a near-stop instead of the
                    # usual floor once genuinely at one -- unlike the freeze-deadlock bug MIN_VX_SCALE
                    # was raised for (low vx for the WHOLE approach to a nearby waypoint), this only
                    # ever lasts the few steps needed to turn away from a confirmed edge.
                    target_vx = min(target_vx, MAX_SPEED * 0.05)
                else:
                    target_vx = min(target_vx, max(MIN_VX_SCALE * MAX_SPEED, MAX_SPEED * (0.75 - 0.35 * severity)))

            # stall watchdog: periodically check real progress, and if there isn't any, override
            # everything above with a forced, unambiguous turn -- see the per-episode setup comment
            # for why this exists instead of another blending patch.
            # Tracks dist_to_goal (3D distance to the CURRENT waypoint), not a fixed world axis --
            # tried X-only (every flight runs along X) and that mostly worked but is the wrong
            # signal once on the flat 2F room-crossing leg, where X isn't necessarily how it makes
            # progress. dist_to_goal auto-adapts: during a stair climb it's dominated by the z term
            # (real climbing progress, immune to lateral X/Y wobble from the centerline assist and
            # wall-repulsion), and during flat room navigation it's dominated by XY -- one signal
            # that means "getting closer to what actually matters" in every segment. Reset whenever
            # a waypoint advances so the distance jump to the next goal isn't misread as "no
            # progress" on the very next check.
            if _stall_check_dist is None:
                _stall_check_dist = stall_progress_dist
                _stall_check_pos = np.array(base_pos, dtype=np.float64)
            elif step % STALL_CHECK_INTERVAL == 0:
                progress = _stall_check_dist - stall_progress_dist  # positive == got closer to the goal
                # Explicit user request (2026-10-13, ghost-dog run): "closer to planning_goal" alone is
                # not a stall test. While climbing, planning_goal is the NEXT tread and moves up with
                # the robot, so the distance never shrinks -- a steadily climbing robot was flagged
                # every 200 steps (18 "stalls" in one clean 3F run) and each one forced an 80-step
                # max-rate turn the opposite way to ViPlanner: the left-right weaving, with or without
                # a gait. A stall also needs the robot itself to have gone nowhere: net 3D displacement
                # (z counts -- climbing is progress) under the same threshold.
                _stall_moved = float(np.linalg.norm(np.asarray(base_pos, dtype=np.float64) - _stall_check_pos))
                _stall_check_pos = np.array(base_pos, dtype=np.float64)
                if (progress < STALL_MOVE_THRESHOLD and _stall_moved < STALL_MOVE_THRESHOLD
                        and _stall_recovery_left <= 0 and _stall_backup_left <= 0):
                    if wall_dist < SAFE_WALL_DIST * 1.5 and _wall_box is not None:
                        # genuinely pinned near a wall, not just generically stuck -- a turn-in-place
                        # can't get past a wall that runs the whole length of a corridor, so retreat
                        # along that specific wall's own long axis toward whichever end is nearer,
                        # instead of spinning (see the per-episode setup comment for the real run
                        # this was diagnosed from).
                        wxmin, wxmax, wymin, wymax = _wall_box
                        print(f"[viplanner] DIAG stall wall match: base=({base_pos[0]:.2f},{base_pos[1]:.2f}) "
                              f"wall_box=({wxmin:.2f},{wxmax:.2f},{wymin:.2f},{wymax:.2f}) "
                              f"wall_dist={wall_dist:.2f} dx={wall_dx:.2f} dy={wall_dy:.2f}")
                        _wall_away_n = math.hypot(wall_dx, wall_dy)
                        if on_stairs and _wall_away_n > 1e-3:
                            # Explicit user request ("它現在卡在欄杆"): wedged SIDEWAYS against a
                            # railing running along the flight, the "back off past the wall's far
                            # end" retreat below means sliding along the railing back down the
                            # flight -- which a wedged robot can't do. On stairs, step straight
                            # away from the wall/railing instead.
                            _stall_backup_target = (base_pos[0] + 0.6 * wall_dx / _wall_away_n,
                                                    base_pos[1] + 0.6 * wall_dy / _wall_away_n)
                        elif (wxmax - wxmin) >= (wymax - wymin):
                            target_x = wxmin - 1.0 if (base_pos[0] - wxmin) <= (wxmax - base_pos[0]) else wxmax + 1.0
                            _stall_backup_target = (target_x, base_pos[1])
                        else:
                            target_y = wymin - 1.0 if (base_pos[1] - wymin) <= (wymax - base_pos[1]) else wymax + 1.0
                            _stall_backup_target = (base_pos[0], target_y)
                        _stall_backup_left = STALL_BACKUP_MAX_STEPS
                        _stall_backup_check_pos = (base_pos[0], base_pos[1])
                        _stall_backup_check_left = STALL_BACKUP_CHECK_INTERVAL
                        print(f"[viplanner] WARNING: stall detected at step {step} near a wall "
                              f"(dist={wall_dist:.2f}m) -- retreating toward {_stall_backup_target}")
                    else:
                        print(f"[viplanner] WARNING: stall detected at step {step} "
                              f"(only {progress:.3f}m closer to goal over last {STALL_CHECK_INTERVAL} steps) -- forcing recovery turn")
                        _stall_recovery_left = STALL_RECOVERY_DURATION
                        _stall_recovery_sign *= -1.0  # alternate side each time in case one direction re-traps it
                _stall_check_dist = stall_progress_dist

            if _stall_backup_left > 0:
                _stall_backup_check_left -= 1
                if _stall_backup_check_left <= 0:
                    moved = math.hypot(base_pos[0] - _stall_backup_check_pos[0], base_pos[1] - _stall_backup_check_pos[1])
                    if moved < STALL_BACKUP_MOVE_THRESHOLD:
                        print(f"[viplanner] WARNING: retreat itself made no progress at step {step} "
                              f"(moved {moved:.3f}m in {STALL_BACKUP_CHECK_INTERVAL} steps, likely "
                              f"physically wedged) -- abandoning retreat, forcing turn-in-place instead")
                        _stall_backup_left = 0
                        _stall_recovery_left = STALL_RECOVERY_DURATION
                        _stall_recovery_sign *= -1.0
                    else:
                        _stall_backup_check_pos = (base_pos[0], base_pos[1])
                        _stall_backup_check_left = STALL_BACKUP_CHECK_INTERVAL
            if _stall_backup_left > 0:
                to_target = np.array(
                    [_stall_backup_target[0] - base_pos[0], _stall_backup_target[1] - base_pos[1], 0.0],
                    dtype=np.float32,
                )
                if np.linalg.norm(to_target[:2]) < 0.3:
                    _stall_backup_left = 0  # reached the retreat point early -- hand back to normal logic
                else:
                    target_body = quat_rotate_inverse_wxyz(base_quat, to_target)
                    retreat_heading_error = math.atan2(target_body[1], target_body[0])
                    target_wz = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, YAW_GAIN * retreat_heading_error))
                    target_vx = MIN_VX_SCALE * MAX_SPEED
                    _stall_backup_left -= 1
            elif _stall_recovery_left > 0:
                target_wz = _stall_recovery_sign * MAX_YAW_RATE
                target_vx = MIN_VX_SCALE * MAX_SPEED
                _stall_recovery_left -= 1

            # smooth vx/wz toward their (possibly just-updated) target every single step, instead
            # of snapping instantly -- with COMMAND_HOLD_STEPS lowered to 20 for reactive steering
            # in the stairs scene's narrow corridors, instant snapping (e.g. 0.24 -> 0.60 m/s)
            # every 0.33s read as the robot "lurching"/surging rather than walking smoothly.
            # SMOOTH_RATE=0.15 reaches ~95% of a new target within ~20 steps (one hold interval).
            if (
                abs(_viplanner_target_wz) > 0.15
                and abs(target_wz) > 0.15
                and (_viplanner_target_wz > 0) != (target_wz > 0)
            ):
                print(f"[viplanner] DIAG sign-flip at step {step}: ViPlanner wz={_viplanner_target_wz:.2f} "
                      f"-> final target_wz={target_wz:.2f} (heading_err_deg={math.degrees(heading_error):.1f}, "
                      f"wall_dist={wall_dist:.2f}, centerline_active={_c_axis is not None} ({_c_axis})")

            SMOOTH_RATE = 0.15
            vx += (target_vx - vx) * SMOOTH_RATE
            wz += (target_wz - wz) * SMOOTH_RATE
            if _GHOST:
                _t_g = timeline.get_current_time()
                _dt_g = max(0.0, min(_t_g - _ghost_prev_t, 0.1))
                _ghost_prev_t = _t_g
                _ghost_yaw += wz * _dt_g
                _nx = _ghost_pos[0] + vx * math.cos(_ghost_yaw) * _dt_g
                _ny = _ghost_pos[1] + vx * math.sin(_ghost_yaw) * _dt_g
                # ground under the new spot, searched from just above where a step up could be
                _gz_new = _ghost_ground_z(_nx, _ny, _ghost_pos[2] - _ghost_h + 0.5)
                if _gz_new is not None:
                    _ghost_pos[:] = (_nx, _ny, _gz_new + _ghost_h)
                else:  # nothing below (off an edge): fall straight down so FELL OUT still triggers
                    _ghost_pos[:] = (_nx, _ny, _ghost_pos[2] - 9.81 * _dt_g)
                _q = np.array([[math.cos(_ghost_yaw / 2), 0.0, 0.0, math.sin(_ghost_yaw / 2)]], dtype=np.float32)
                dog_art.set_world_poses(_ghost_pos[None, :].astype(np.float32), _q)
                dog_art.set_velocities(np.zeros((1, 6), dtype=np.float32))
                dog_art.set_joint_positions(DEFAULT_ANGLES[None, :], joint_indices=_JOINT_IDX)
                dog_art.set_joint_velocities(np.zeros((1, 12), dtype=np.float32), joint_indices=_JOINT_IDX)
                _ghost_path.append((step, _t_g, *_ghost_pos, _ghost_yaw, vx, wz, wp_idx))
            _diag_vx_deltas.append(abs(vx - _prev_vx))
            _diag_wz_deltas.append(abs(wz - _prev_wz))

            if step % 100 == 0:
                print(
                    f"[viplanner] episode {episode} step {step}: waypoint {wp_idx + 1}/{len(waypoints)} "
                    f"pos=({base_pos[0]:.2f},{base_pos[1]:.2f},{base_pos[2]:.2f}) "
                    f"dist={dist_to_goal:.2f} fear={fear_val:.2f} vx={vx:.2f} wz={wz:.2f} "
                    f"heading_err_deg={math.degrees(heading_error):.1f} "
                    f"seeking_stairs={seeking_stairs} near_stairs_entrance={near_stairs_entrance} "
                    f"on_stairs={on_stairs} planning_goal=({planning_goal[0]:.2f},{planning_goal[1]:.2f})"
                )
                _dts = np.array(_diag_dts) if _diag_dts else np.zeros(1)
                _fps = 1.0 / max(_dts.mean(), 1e-6)
                print(
                    f"[viplanner]   diag: frame_dt ms mean={_dts.mean()*1000:.1f} "
                    f"max={_dts.max()*1000:.1f} p95={np.percentile(_dts, 95)*1000:.1f} (~{_fps:.1f} fps) | "
                    f"vx_delta max={max(_diag_vx_deltas, default=0):.3f} wz_delta max={max(_diag_wz_deltas, default=0):.3f} | "
                    f"action_delta max={max(_diag_action_deltas, default=0):.3f} | "
                    f"fear_delta max={max(_diag_fear_deltas, default=0):.3f} mean={np.mean(_diag_fear_deltas) if _diag_fear_deltas else 0:.3f}"
                )
                _diag_dts.clear()
                _diag_vx_deltas.clear()
                _diag_wz_deltas.clear()
                _diag_action_deltas.clear()
                _diag_fear_deltas.clear()
            _z_matched = _STOP_ON_Z_MATCH and abs(base_pos[2] - goal_np[2]) < _Z_MATCH_THRESHOLD
            # fell off an open edge (Stair.usdc's 1F has no floor/wall past y=-2): nothing to recover,
            # don't burn the rest of max_steps free-falling
            if base_pos[2] < -2.0:
                print(f"[viplanner] episode {episode}: FELL OUT of the scene at step {step} "
                      f"(base z={base_pos[2]:.1f}), stuck on waypoint {wp_idx + 1}/{len(waypoints)}")
                break
            # Explicit user request ("它跌倒了"): tipped over on the stairs -- the policy can't get
            # back up, so end the episode instead of idling to max_steps. Body z-axis's world-z
            # component from the wxyz quaternion: 1 upright, 0 on its side (~0.34 = 70 deg tilt).
            _qw, _qx, _qy, _qz = base_quat
            _body_up_z = 1.0 - 2.0 * (_qx * _qx + _qy * _qy)
            _tipped_steps = _tipped_steps + 1 if _body_up_z < 0.34 else 0
            if _tipped_steps >= 200:
                print(f"[viplanner] episode {episode}: FELL OVER at step {step} (body up_z={_body_up_z:.2f} "
                      f"at {base_pos[0]:.2f},{base_pos[1]:.2f},{base_pos[2]:.2f}), stuck on waypoint "
                      f"{wp_idx + 1}/{len(waypoints)}")
                break
            if dist_to_goal < REACH_THRESHOLD or _z_matched:
                if _z_matched and dist_to_goal >= REACH_THRESHOLD:
                    print(f"[viplanner] episode {episode}: Z MATCHED goal (base_z={base_pos[2]:.2f}, "
                          f"goal_z={goal_np[2]:.2f}) at step {step}, dist_to_goal={dist_to_goal:.2f} "
                          f"(XY not necessarily reached) -- VIPLANNER_STOP_ON_Z_MATCH=1")
                if wp_idx + 1 < len(waypoints):
                    print(f"[viplanner] episode {episode}: reached waypoint {wp_idx + 1}/{len(waypoints)} at step {step} -> advancing")
                    wp_idx += 1
                    goal_np = waypoints[wp_idx]
                    # force an immediate replan against the new goal instead of waiting for the
                    # next REPLAN_EVERY_N_STEPS tick -- also reset the smoothing vector so it
                    # doesn't keep pulling toward the old waypoint's stale heading for a moment
                    heading_error = 0.0
                    _heading_cos, _heading_sin = 1.0, 0.0
                    _stall_check_dist = None  # next check re-baselines against the new goal's distance
                else:
                    print(f"[viplanner] episode {episode}: REACHED FINAL GOAL at step {step}, dist={dist_to_goal:.2f}")
                    reached = True
                    break
        else:
            print(f"[viplanner] episode {episode}: stopped after max_steps={max_steps}, stuck on waypoint {wp_idx + 1}/{len(waypoints)}")

        if _GHOST:
            _gf = _os_check.path.join(_ghost_log_dir, f"episode_{episode:03d}.npz")  # (_os only exists when collecting)
            np.savez_compressed(
                _gf, path=np.asarray(_ghost_path, dtype=np.float64), reached=reached,
                waypoints=np.asarray(waypoints, dtype=np.float32),
                plan_steps=np.asarray([pl[0] for pl in _ghost_plans]),
                plan_wp=np.asarray([pl[1] for pl in _ghost_plans]),
                plan_fear=np.asarray([pl[2] for pl in _ghost_plans]),
                plan_pts=np.asarray([pl[3] for pl in _ghost_plans], dtype=np.float32),
                plan_goal=np.asarray([pl[4] for pl in _ghost_plans], dtype=np.float32))
            print(f"[viplanner] ghost: episode {episode} path ({len(_ghost_path)} steps, "
                  f"{len(_ghost_plans)} plans) -> {_gf}")

        if collecting:
            print(f"[viplanner] DATA COLLECTION: episode {episode}/{collect_episodes} done, "
                  f"{_sample_idx} samples saved so far to {collect_data_dir}")
            if episode >= collect_episodes:
                print(f"[viplanner] DATA COLLECTION complete: {_sample_idx} samples across {episode} episodes")
                return

        # brief pause at the goal so it's visually clear the walk finished, then loop again
        for _ in range(60):
            await app.next_update_async()
