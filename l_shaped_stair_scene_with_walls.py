import asyncio
import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import FixedCuboid


def box(world, name, p, s, c):
    obj = FixedCuboid(
        prim_path=f"/World/{name}",
        name=name,
        position=np.array(p, dtype=np.float32),
        scale=np.array(s, dtype=np.float32),
        color=np.array(c, dtype=np.float32),
    )
    world.scene.add(obj)


async def main():
    if World.instance():
        World.instance().clear_instance()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    # ========================================================
    # Parameters
    # ========================================================
    FLOOR_T = 0.12

    STEP_H = 0.10
    STEP_D = 0.30
    WIDTH = 1.50
    N = 6

    # 中間平台改成和樓梯同寬：1.50 x 1.50 m
    LANDING = WIDTH

    HALF_Z = N * STEP_H
    TOP_Z = 2 * HALF_Z

    # 2F
    FLOOR2_X = 6.0
    FLOOR2_Y = 6.0

    # 2F 圍牆
    WALL_T = 0.12
    WALL_H = 1.20

    C_F = [0.42, 0.42, 0.42]
    C_S = [0.72, 0.72, 0.72]
    C_L = [0.55, 0.60, 0.75]
    C_2 = [0.38, 0.52, 0.68]
    C_W = [0.82, 0.82, 0.82]

    # ========================================================
    # 1F
    # ========================================================
    box(
        world,
        "Floor_1",
        [0.0, 0.0, -FLOOR_T / 2],
        [14.0, 12.0, FLOOR_T],
        C_F,
    )

    # ========================================================
    # Stair A：沿 +X
    # ========================================================
    x0 = -4.0
    y0 = -1.5

    for i in range(N):
        top = STEP_H * (i + 1)
        x = x0 + STEP_D / 2 + i * STEP_D

        box(
            world,
            f"Stair_A_{i:02d}",
            [x, y0, top / 2],
            [STEP_D, WIDTH, top],
            C_S,
        )

    exit_x = x0 + N * STEP_D

    # ========================================================
    # 中間小平台
    #
    # 關鍵修改：
    # LANDING = WIDTH = 1.50 m
    # 所以平台為 1.50 x 1.50 m
    # ========================================================
    lx = exit_x + LANDING / 2
    ly = y0

    box(
        world,
        "Landing",
        [lx, ly, HALF_Z - FLOOR_T / 2],
        [LANDING, WIDTH, FLOOR_T],
        C_L,
    )

    # ========================================================
    # Stair B：Landing 後左轉 90°，沿 +Y
    # ========================================================
    y1 = y0 + LANDING / 2

    for i in range(N):
        top = HALF_Z + STEP_H * (i + 1)
        y = y1 + STEP_D / 2 + i * STEP_D

        box(
            world,
            f"Stair_B_{i:02d}",
            [lx, y, top / 2],
            [WIDTH, STEP_D, top],
            C_S,
        )

    exit_y = y1 + N * STEP_D

    # ========================================================
    # 2F 平台
    #
    # Stair B 從 2F 的 -Y 邊中央進入。
    # ========================================================
    floor2_center_x = lx
    floor2_y_min = exit_y
    floor2_y_max = exit_y + FLOOR2_Y
    floor2_center_y = (floor2_y_min + floor2_y_max) / 2

    floor2_x_min = floor2_center_x - FLOOR2_X / 2
    floor2_x_max = floor2_center_x + FLOOR2_X / 2

    box(
        world,
        "Floor_2",
        [floor2_center_x, floor2_center_y, TOP_Z - FLOOR_T / 2],
        [FLOOR2_X, FLOOR2_Y, FLOOR_T],
        C_2,
    )

    # ========================================================
    # 2F 圍牆
    #
    # 三邊完整封住：
    #   左牆 / 右牆 / 後牆
    #
    # 樓梯入口所在的前牆 (-Y) 分成左右兩段，
    # 中央只留 WIDTH = 1.50 m 的樓梯口。
    # ========================================================

    wall_z = TOP_Z + WALL_H / 2

    # 左側牆 (-X)
    box(
        world,
        "Floor2_Wall_Left",
        [floor2_x_min + WALL_T / 2, floor2_center_y, wall_z],
        [WALL_T, FLOOR2_Y, WALL_H],
        C_W,
    )

    # 右側牆 (+X)
    box(
        world,
        "Floor2_Wall_Right",
        [floor2_x_max - WALL_T / 2, floor2_center_y, wall_z],
        [WALL_T, FLOOR2_Y, WALL_H],
        C_W,
    )

    # 後牆 (+Y)
    box(
        world,
        "Floor2_Wall_Back",
        [floor2_center_x, floor2_y_max - WALL_T / 2, wall_z],
        [FLOOR2_X, WALL_T, WALL_H],
        C_W,
    )

    # ========================================================
    # 前牆 (-Y)：中央留樓梯口
    # ========================================================
    opening_x_min = lx - WIDTH / 2
    opening_x_max = lx + WIDTH / 2

    # 左半段前牆
    front_left_len = opening_x_min - floor2_x_min

    if front_left_len > 0:
        front_left_center_x = floor2_x_min + front_left_len / 2

        box(
            world,
            "Floor2_Wall_Front_Left",
            [
                front_left_center_x,
                floor2_y_min + WALL_T / 2,
                wall_z,
            ],
            [
                front_left_len,
                WALL_T,
                WALL_H,
            ],
            C_W,
        )

    # 右半段前牆
    front_right_len = floor2_x_max - opening_x_max

    if front_right_len > 0:
        front_right_center_x = opening_x_max + front_right_len / 2

        box(
            world,
            "Floor2_Wall_Front_Right",
            [
                front_right_center_x,
                floor2_y_min + WALL_T / 2,
                wall_z,
            ],
            [
                front_right_len,
                WALL_T,
                WALL_H,
            ],
            C_W,
        )

    # ========================================================
    # Reset
    # ========================================================
    await world.reset_async()

    print("===================================================")
    print("L-shaped stair scene created")
    print("===================================================")
    print(f"Stair width        : {WIDTH:.2f} m")
    print(f"Landing size       : {LANDING:.2f} x {WIDTH:.2f} m")
    print(f"Landing height     : {HALF_Z:.2f} m")
    print(f"2F height          : {TOP_Z:.2f} m")
    print(f"2F size            : {FLOOR2_X:.2f} x {FLOOR2_Y:.2f} m")
    print(f"2F wall height     : {WALL_H:.2f} m")
    print(f"Stair opening width: {WIDTH:.2f} m")
    print("2F enclosed on all sides except stair opening.")
    print("===================================================")


asyncio.ensure_future(main())
