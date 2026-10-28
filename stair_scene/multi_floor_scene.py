# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone two-floor building scene with a U-shaped (switchback) staircase -- two flights of 10
steps each, connected by a landing/turn platform, leading up to a 2F platform behind a walled
doorway, plus one box obstacle -- a quick static-geometry test scene, not wired into the ViPlanner
IsaacLab pipeline (no cost map / semantic labels / people / robot yet). Uses plain
omni.isaac.core.World + FixedCuboid, same pattern as an interactive Script Editor snippet, but
launched via AppLauncher so it can be viewed in a native Isaac Sim window (or headless + the Isaac
Sim WebRTC Streaming Client, if run with --headless --livestream 2).
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Two-floor building test scene with a staircase.")
parser.add_argument(
    "--screenshot_dir",
    default=None,
    type=str,
    help="If set, periodically capture the main viewport to <dir>/latest.png.",
)
parser.add_argument("--screenshot_interval", default=60, type=int, help="Capture every N render steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import UsdLux, Sdf, Gf

import omni.usd

if args_cli.screenshot_dir is not None:
    import omni.kit.viewport.utility as vp_util

world = World(stage_units_in_meters=1.0)

# the original script never added a light -- an unlit stage renders solid black, which is exactly
# the "screen is black" symptom reported after connecting. A distant light (sun-like, directional)
# plus a dim dome light (ambient fill so shadowed faces aren't pitch black) is enough for a plain
# geometry-only test scene like this.
stage = omni.usd.get_context().get_stage()
distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
distant_light.CreateIntensityAttr(3000)
distant_light.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 0.0))

dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
dome_light.CreateIntensityAttr(500)

# =========================
# 工具函式
# =========================


def create_box(name, position, scale, color=None):
    """建立固定方塊
    position = [x, y, z]
    scale = [x_size, y_size, z_size]
    color = [r, g, b], 0-1
    """
    if color is None:
        color = [0.6, 0.6, 0.6]
    box = FixedCuboid(
        prim_path=f"/World/{name}",
        name=name,
        position=np.array(position, dtype=np.float32),
        scale=np.array(scale, dtype=np.float32),
        color=np.array(color, dtype=np.float32),
    )
    world.scene.add(box)
    return box


def create_person(name, position, shirt_color):
    """簡單靜態人形 (block figure): legs, shoes, body, arms, head.
    position = 腳底中心位置 [x, y, z]
    """
    x, y, z = position

    skin_color = [0.86, 0.68, 0.52]
    pants_color = [0.18, 0.20, 0.25]
    shoe_color = [0.08, 0.08, 0.08]

    leg_height, leg_width, leg_depth = 0.75, 0.16, 0.20
    body_height, body_width, body_depth = 0.60, 0.42, 0.24
    head_height, head_width, head_depth = 0.25, 0.23, 0.23
    arm_height, arm_width, arm_depth = 0.55, 0.11, 0.13
    shoe_height = 0.08

    create_box(f"{name}_LeftLeg", position=[x, y + 0.11, z + shoe_height + leg_height / 2], scale=[leg_depth, leg_width, leg_height], color=pants_color)
    create_box(f"{name}_RightLeg", position=[x, y - 0.11, z + shoe_height + leg_height / 2], scale=[leg_depth, leg_width, leg_height], color=pants_color)
    create_box(f"{name}_LeftShoe", position=[x + 0.04, y + 0.11, z + shoe_height / 2], scale=[0.30, 0.17, shoe_height], color=shoe_color)
    create_box(f"{name}_RightShoe", position=[x + 0.04, y - 0.11, z + shoe_height / 2], scale=[0.30, 0.17, shoe_height], color=shoe_color)

    body_center_z = z + shoe_height + leg_height + body_height / 2
    create_box(f"{name}_Body", position=[x, y, body_center_z], scale=[body_depth, body_width, body_height], color=shirt_color)

    create_box(f"{name}_LeftArm", position=[x, y + body_width / 2 + arm_width / 2, body_center_z - 0.02], scale=[arm_depth, arm_width, arm_height], color=skin_color)
    create_box(f"{name}_RightArm", position=[x, y - body_width / 2 - arm_width / 2, body_center_z - 0.02], scale=[arm_depth, arm_width, arm_height], color=skin_color)

    head_center_z = z + shoe_height + leg_height + body_height + head_height / 2 + 0.05
    create_box(f"{name}_Head", position=[x, y, head_center_z], scale=[head_depth, head_width, head_height], color=skin_color)


# =========================
# 基本參數
# =========================

FLOOR_THICKNESS = 0.20
FLOOR_2_Z = 3.0

# 樓梯
STEP_HEIGHT = 0.15
STEP_DEPTH = 0.30
STAIR_WIDTH = 1.50
STEPS_PER_FLIGHT = 10

HALF_HEIGHT = STEP_HEIGHT * STEPS_PER_FLIGHT  # 10 * 0.15 = 1.5m
FLIGHT_LENGTH = STEP_DEPTH * STEPS_PER_FLIGHT  # 10 * 0.30 = 3m

# 轉角平台
LANDING_LENGTH = 1.50
LANDING_WIDTH = 4.20  # 平台 Y 方向比兩段樓梯稍微寬

# 平台矮牆
LOW_WALL_HEIGHT = 0.85
LOW_WALL_THICKNESS = 0.12

# 二樓
SECOND_FLOOR_LENGTH = 7.0
SECOND_FLOOR_WIDTH = 10.0

# 二樓門
DOOR_WIDTH = 1.60
DOOR_HEIGHT = 2.20
WALL_HEIGHT = 3.0
WALL_THICKNESS = 0.15

# =========================
# 顏色
# =========================

FLOOR_COLOR = [0.42, 0.42, 0.42]
STAIR_COLOR = [0.72, 0.72, 0.72]
LANDING_COLOR = [0.55, 0.60, 0.75]
LOW_WALL_COLOR = [0.82, 0.82, 0.82]
SECOND_FLOOR_COLOR = [0.38, 0.52, 0.68]
ROOM_WALL_COLOR = [0.78, 0.65, 0.48]
DOOR_FRAME_COLOR = [0.32, 0.22, 0.15]

# =========================
# 1F 地板
# =========================

create_box("Floor_1", position=[0.0, 0.0, -FLOOR_THICKNESS / 2], scale=[14.0, 10.0, FLOOR_THICKNESS], color=FLOOR_COLOR)

# =========================
# 第一段樓梯
#
# 往 +X 方向走
#
#                         Landing
#                            ↓
#       → → → → → → → → → →
#      █
#      ███
#      █████
# =========================

first_start_x = -4.0
first_y = -1.20

for i in range(STEPS_PER_FLIGHT):
    top_height = STEP_HEIGHT * (i + 1)
    x = first_start_x + STEP_DEPTH / 2 + i * STEP_DEPTH
    z = top_height / 2
    create_box(f"Stair_A_{i:02d}", position=[x, first_y, z], scale=[STEP_DEPTH, STAIR_WIDTH, top_height], color=STAIR_COLOR)

# =========================
# 第一段樓梯兩側實心矮牆
#
# 重點：每一段牆 bottom=Z 0, top=階梯高度+0.85m，所以牆會一路接到 1F 地面
# =========================

first_left_wall_y = first_y + STAIR_WIDTH / 2 + LOW_WALL_THICKNESS / 2
first_right_wall_y = first_y - STAIR_WIDTH / 2 - LOW_WALL_THICKNESS / 2

for i in range(STEPS_PER_FLIGHT):
    stair_top_z = STEP_HEIGHT * (i + 1)
    wall_total_height = stair_top_z + LOW_WALL_HEIGHT  # 整片牆從 Z=0 到 stair_top_z + LOW_WALL_HEIGHT
    wall_center_z = wall_total_height / 2
    x = first_start_x + STEP_DEPTH / 2 + i * STEP_DEPTH

    create_box(
        f"Stair_A_LeftWall_{i:02d}",
        position=[x, first_left_wall_y, wall_center_z],
        scale=[STEP_DEPTH, LOW_WALL_THICKNESS, wall_total_height],
        color=LOW_WALL_COLOR,
    )
    create_box(
        f"Stair_A_RightWall_{i:02d}",
        position=[x, first_right_wall_y, wall_center_z],
        scale=[STEP_DEPTH, LOW_WALL_THICKNESS, wall_total_height],
        color=LOW_WALL_COLOR,
    )

# 第一段樓梯結束的位置 (first_start_x=-4, FLIGHT_LENGTH=3 -> first_exit_x=-1)
first_exit_x = first_start_x + FLIGHT_LENGTH

# =========================
#
# 中間轉角平台
#
#                 +Y
#                  ↑
#        ┌───────────────────┐
#        │                   │
# Stair B│      Landing      │
# ←──────│                   │
#        │                   │
# Stair A│                   │
# ─────→ │                   │
#        └───────────────────┘
# =========================

landing_center_x = first_exit_x + LANDING_LENGTH / 2
landing_center_y = 0.0

create_box(
    "Landing",
    position=[landing_center_x, landing_center_y, HALF_HEIGHT - FLOOR_THICKNESS / 2],
    scale=[LANDING_LENGTH, LANDING_WIDTH, FLOOR_THICKNESS],
    color=LANDING_COLOR,
)

# =========================
#
# Landing 的矮牆 -- 左邊有兩個開口：下方=第一段樓梯進入, 上方=第二段樓梯離開
#
#             +Y
#              ↑
#       ┌─────────────────┐
#       │                 │
# B ←   │                 │
# ===== │                 │
#       │        ↻        │
#       │                 │
# ===== │                 │
# A →   │                 │
#       └─────────────────┘
# =========================

landing_x_min = first_exit_x
landing_x_max = first_exit_x + LANDING_LENGTH
landing_y_min = -LANDING_WIDTH / 2
landing_y_max = LANDING_WIDTH / 2

# 平台右側矮牆 (整條封起來)
create_box(
    "Landing_Wall_Right",
    position=[landing_x_max - LOW_WALL_THICKNESS / 2, 0.0, HALF_HEIGHT + LOW_WALL_HEIGHT / 2],
    scale=[LOW_WALL_THICKNESS, LANDING_WIDTH, LOW_WALL_HEIGHT],
    color=LOW_WALL_COLOR,
)

# 平台上側矮牆
create_box(
    "Landing_Wall_Top",
    position=[landing_center_x, landing_y_max - LOW_WALL_THICKNESS / 2, HALF_HEIGHT + LOW_WALL_HEIGHT / 2],
    scale=[LANDING_LENGTH, LOW_WALL_THICKNESS, LOW_WALL_HEIGHT],
    color=LOW_WALL_COLOR,
)

# 平台下側矮牆
create_box(
    "Landing_Wall_Bottom",
    position=[landing_center_x, landing_y_min + LOW_WALL_THICKNESS / 2, HALF_HEIGHT + LOW_WALL_HEIGHT / 2],
    scale=[LANDING_LENGTH, LOW_WALL_THICKNESS, LOW_WALL_HEIGHT],
    color=LOW_WALL_COLOR,
)

# =========================
#
# 平台左側中間矮牆 -- 這一段非常重要
#
#    第二段樓梯
#    ========
#    ┃
#    ┃ ← 中間矮牆
#    ┃
#    ========
#    第一段樓梯
# =========================

second_y = 1.20

first_opening_top = first_y + STAIR_WIDTH / 2
second_opening_bottom = second_y - STAIR_WIDTH / 2
middle_wall_length = second_opening_bottom - first_opening_top

if middle_wall_length > 0:
    middle_wall_y = first_opening_top + middle_wall_length / 2
    create_box(
        "Landing_Wall_Middle",
        position=[landing_x_min + LOW_WALL_THICKNESS / 2, middle_wall_y, HALF_HEIGHT + LOW_WALL_HEIGHT / 2],
        scale=[LOW_WALL_THICKNESS, middle_wall_length, LOW_WALL_HEIGHT],
        color=LOW_WALL_COLOR,
    )

# =========================
# 第二段樓梯
#
# 從 Landing 往 -X 方向
#
# 2F
# ← ← ← ← ←
# █████████
#      Landing
# =========================

second_start_x = first_exit_x

for i in range(STEPS_PER_FLIGHT):
    stair_height = STEP_HEIGHT * (i + 1)
    x = second_start_x - STEP_DEPTH / 2 - i * STEP_DEPTH
    z = HALF_HEIGHT + stair_height / 2
    create_box(f"Stair_B_{i:02d}", position=[x, second_y, z], scale=[STEP_DEPTH, STAIR_WIDTH, stair_height], color=STAIR_COLOR)

# =========================
# 第二段樓梯兩側實心矮牆
#
# 注意：第二段樓梯已經從 Z=1.5 開始，但牆體要垂直到 1F，所以牆的 bottom 一樣設為 Z=0。
# 例如最後一階：Stair top=3.0, Wall top=3.0+0.85=3.85m, Wall bottom=0
# =========================

second_left_wall_y = second_y + STAIR_WIDTH / 2 + LOW_WALL_THICKNESS / 2
second_right_wall_y = second_y - STAIR_WIDTH / 2 - LOW_WALL_THICKNESS / 2

for i in range(STEPS_PER_FLIGHT):
    stair_top_z = HALF_HEIGHT + STEP_HEIGHT * (i + 1)
    wall_total_height = stair_top_z + LOW_WALL_HEIGHT
    wall_center_z = wall_total_height / 2
    x = second_start_x - STEP_DEPTH / 2 - i * STEP_DEPTH

    create_box(
        f"Stair_B_LeftWall_{i:02d}",
        position=[x, second_left_wall_y, wall_center_z],
        scale=[STEP_DEPTH, LOW_WALL_THICKNESS, wall_total_height],
        color=LOW_WALL_COLOR,
    )
    create_box(
        f"Stair_B_RightWall_{i:02d}",
        position=[x, second_right_wall_y, wall_center_z],
        scale=[STEP_DEPTH, LOW_WALL_THICKNESS, wall_total_height],
        color=LOW_WALL_COLOR,
    )

# 第二段樓梯出口 (second_start_x=-1, FLIGHT_LENGTH=3 -> second_exit_x=-4)
second_exit_x = second_start_x - FLIGHT_LENGTH

# =========================
# 二樓 -- 從樓梯出口往 -X 延伸
# =========================

second_floor_center_x = second_exit_x - SECOND_FLOOR_LENGTH / 2

create_box(
    "Floor_2",
    position=[second_floor_center_x, 0.0, FLOOR_2_Z - FLOOR_THICKNESS / 2],
    scale=[SECOND_FLOOR_LENGTH, SECOND_FLOOR_WIDTH, FLOOR_THICKNESS],
    color=SECOND_FLOOR_COLOR,
)

# =========================
# 二樓門的位置 -- 樓梯 → 0.8m → 門
# =========================

DOOR_BUFFER = 0.80
door_wall_x = second_exit_x - DOOR_BUFFER

# 門一定正對第二段樓梯
door_center_y = second_y

# =========================
# 二樓房間真正使用的範圍 -- 門牆 = 房間前牆
# =========================

room_back_x = second_floor_center_x - SECOND_FLOOR_LENGTH / 2
room_front_x = door_wall_x
room_y_min = -SECOND_FLOOR_WIDTH / 2
room_y_max = SECOND_FLOOR_WIDTH / 2

# =========================
# 第 1 面牆：有門洞的前牆
# =========================

door_y_min = door_center_y - DOOR_WIDTH / 2
door_y_max = door_center_y + DOOR_WIDTH / 2

# 門洞下方牆
lower_wall_length = door_y_min - room_y_min
lower_wall_center_y = room_y_min + lower_wall_length / 2
create_box(
    "Room_FrontWall_Lower",
    position=[room_front_x, lower_wall_center_y, FLOOR_2_Z + WALL_HEIGHT / 2],
    scale=[WALL_THICKNESS, lower_wall_length, WALL_HEIGHT],
    color=ROOM_WALL_COLOR,
)

# 門洞上方牆
upper_wall_length = room_y_max - door_y_max
upper_wall_center_y = door_y_max + upper_wall_length / 2
create_box(
    "Room_FrontWall_Upper",
    position=[room_front_x, upper_wall_center_y, FLOOR_2_Z + WALL_HEIGHT / 2],
    scale=[WALL_THICKNESS, upper_wall_length, WALL_HEIGHT],
    color=ROOM_WALL_COLOR,
)

# 門上面的牆
door_top_height = WALL_HEIGHT - DOOR_HEIGHT
create_box(
    "Room_FrontWall_DoorTop",
    position=[room_front_x, door_center_y, FLOOR_2_Z + DOOR_HEIGHT + door_top_height / 2],
    scale=[WALL_THICKNESS, DOOR_WIDTH, door_top_height],
    color=ROOM_WALL_COLOR,
)

# =========================
# 第 2 面牆：二樓後牆
# =========================

create_box(
    "Room_BackWall",
    position=[room_back_x + WALL_THICKNESS / 2, 0.0, FLOOR_2_Z + WALL_HEIGHT / 2],
    scale=[WALL_THICKNESS, SECOND_FLOOR_WIDTH, WALL_HEIGHT],
    color=ROOM_WALL_COLOR,
)

# 二樓左右牆的長度 -- 從門牆一路到後牆
room_side_wall_length = room_front_x - room_back_x
room_side_wall_center_x = room_back_x + room_side_wall_length / 2

# =========================
# 第 3 面牆：二樓 +Y 側牆
# =========================

create_box(
    "Room_LeftWall",
    position=[room_side_wall_center_x, room_y_max - WALL_THICKNESS / 2, FLOOR_2_Z + WALL_HEIGHT / 2],
    scale=[room_side_wall_length, WALL_THICKNESS, WALL_HEIGHT],
    color=ROOM_WALL_COLOR,
)

# =========================
# 第 4 面牆：二樓 -Y 側牆
# =========================

create_box(
    "Room_RightWall",
    position=[room_side_wall_center_x, room_y_min + WALL_THICKNESS / 2, FLOOR_2_Z + WALL_HEIGHT / 2],
    scale=[room_side_wall_length, WALL_THICKNESS, WALL_HEIGHT],
    color=ROOM_WALL_COLOR,
)

# =========================
# 門框
# =========================

FRAME_THICKNESS = 0.08
FRAME_DEPTH = 0.22

create_box(
    "DoorFrame_Left",
    position=[room_front_x, door_y_min, FLOOR_2_Z + DOOR_HEIGHT / 2],
    scale=[FRAME_DEPTH, FRAME_THICKNESS, DOOR_HEIGHT],
    color=DOOR_FRAME_COLOR,
)

create_box(
    "DoorFrame_Right",
    position=[room_front_x, door_y_max, FLOOR_2_Z + DOOR_HEIGHT / 2],
    scale=[FRAME_DEPTH, FRAME_THICKNESS, DOOR_HEIGHT],
    color=DOOR_FRAME_COLOR,
)

create_box(
    "DoorFrame_Top",
    position=[room_front_x, door_center_y, FLOOR_2_Z + DOOR_HEIGHT],
    scale=[FRAME_DEPTH, DOOR_WIDTH, FRAME_THICKNESS],
    color=DOOR_FRAME_COLOR,
)

# =========================
# 二樓人物 1
# =========================

create_person("Person_01", position=[room_front_x - 2.2, -1.6, FLOOR_2_Z], shirt_color=[0.25, 0.45, 0.80])

# =========================
# 二樓人物 2
# =========================

create_person("Person_02", position=[room_front_x - 4.0, 2.0, FLOOR_2_Z], shirt_color=[0.75, 0.30, 0.35])

# =========================
# 二樓人物 3 -- 複製自 Person_01/02，位置取兩人中點
# =========================

create_person("Person_03", position=[room_front_x - 3.1, 0.2, FLOOR_2_Z], shirt_color=[0.30, 0.65, 0.35])

# =========================
# reset
# =========================

world.reset()

# this script never set a viewport camera, so it defaulted to Isaac Sim's stock perspective
# camera pose -- not framed for this building. Point it at a normal 3/4 overview angle instead.
set_camera_view(
    eye=np.array([-2.0, 0.001, 25.0]),
    target=np.array([-2.0, 0.0, 0.0]),
)

print("")
print("==========================================")
print("Multi-floor indoor environment created")
print("==========================================")
print("First stair:", first_start_x, "->", first_exit_x)
print("Landing height:", HALF_HEIGHT)
print("Second stair:", second_start_x, "->", second_exit_x)
print("Second floor Z:", FLOOR_2_Z)
print("Door center:", [room_front_x, door_center_y, FLOOR_2_Z])
print("Room X:", room_back_x, "to", room_front_x)
print("Room Y:", room_y_min, "to", room_y_max)
print("2F walls: COMPLETE")
print("People: 3")
print("==========================================")
print("")

# keep the app alive and rendering -- the original script exited immediately after building the
# scene, which would tear down the Kit process (and the WebRTC stream with it) before anyone could
# connect and look at it.
if args_cli.screenshot_dir is not None:
    os.makedirs(args_cli.screenshot_dir, exist_ok=True)
    viewport_api = vp_util.get_active_viewport()

step_count = 0
while simulation_app.is_running():
    world.step(render=True)
    step_count += 1
    if args_cli.screenshot_dir is not None and step_count % args_cli.screenshot_interval == 0:
        vp_util.capture_viewport_to_file(viewport_api, os.path.join(args_cli.screenshot_dir, "latest.png"))

simulation_app.close()

# EoF
