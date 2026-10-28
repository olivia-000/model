"""Isaac Sim entry script: straight_stair_scene.py's geometry (single straight staircase, no turn,
1F -> StraightStair -> 2F, originally isaacsim.core.api.World + FixedCuboid with no walls at all)
rebuilt on plain UsdGeom.Cube + UsdPhysics.CollisionAPI with per-box semantic tagging, then handed
off to the Go2 dog + ViPlanner/HIMLoco control loop (isaac_sim_bridge/viplanner_control.py) -- same
pattern as l_shaped_stair_scene_with_walls_viplanner.py, simplified for this single-flight layout
(no landing, no turn, no 2F room walls). Geometry/coordinate math is a direct port of
straight_stair_scene.py's own box-by-box layout, so it can't drift out of sync with that file.
"""
import asyncio
import os

import omni.usd
import omni.kit.app


# ============================================================
# geometry constants (from straight_stair_scene.py)
# ============================================================
FLOOR_T = 0.12
STEP_H = 0.10
STEP_D = 0.30
WIDTH = 1.50
N = 10
TOP_Z = N * STEP_H  # 1.00

FLOOR2_L = 7.0
FLOOR2_W = 7.0

WALL_T = 0.12  # 1F boundary wall thickness

C_F = (0.42, 0.42, 0.42)
C_S = (0.72, 0.72, 0.72)
C_2 = (0.38, 0.52, 0.68)
C_W = (0.82, 0.82, 0.82)
C_B = (0.78, 0.65, 0.48)  # 1F boundary walls -- same role/color as the L-shaped scene's

X0 = -4.0
Y0 = 0.0

# Same test setup as the L-shaped scene: spawn right at the stairs entrance.
DOG_START_XY = (X0 - 0.2, Y0)

_AABB_MARGIN = 0.03  # matches camera_semantics_lab's own padding convention


def _create_box(stage, name, position, scale, color, class_id=None, aabbs_out=None):
    from pxr import UsdGeom, UsdPhysics, Gf

    prim = UsdGeom.Cube.Define(stage, f"/World/{name}")
    prim.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddScaleOp().Set(Gf.Vec3f(*[max(s, 0.001) for s in scale]))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

    if class_id is not None and aabbs_out is not None:
        cx, cy, cz = position
        sx, sy, sz = scale
        m = _AABB_MARGIN
        aabbs_out.append((cx - sx / 2 - m, cx + sx / 2 + m, cy - sy / 2 - m, cy + sy / 2 + m,
                           cz - sz / 2 - m, cz + sz / 2 + m, class_id))
    return prim


def _tag_wall_region(aabbs_out, position, scale, class_id):
    """Semantic-only AABB, no USD prim / no collision -- see l_shaped_stair_scene_with_walls_viplanner.py's
    own comment on why (an overlapping real collider depenetrated the robot clean off the map in a
    real run there). Patches the last (tallest) step's exposed far face, which -- since every step
    here is a full wedge down to the ground -- would otherwise still read as walkable SEG_ID_STAIRS."""
    cx, cy, cz = position
    sx, sy, sz = scale
    m = _AABB_MARGIN
    aabbs_out.append((cx - sx / 2 - m, cx + sx / 2 + m, cy - sy / 2 - m, cy + sy / 2 + m,
                       cz - sz / 2 - m, cz + sz / 2 + m, class_id))


def _build_scene(stage, aabbs):
    from camera_semantics_lab import SEG_ID_FLOOR, SEG_ID_STAIRS, SEG_ID_WALL

    # --- 1F ---
    _create_box(stage, "Floor_1", [0.0, 0.0, -FLOOR_T / 2], [16.0, 10.0, FLOOR_T], C_F,
                SEG_ID_FLOOR, aabbs)

    # 1F boundary walls -- same fix as the L-shaped scene's own (a bare floor with no perimeter let
    # the robot wander off the edge into freefall in an earlier real run on a similar scene).
    # Floor_2 here (x up to 6.0) stays well within Floor_1's own x=[-8,8] extent, and Floor_2's own
    # y-range [-3.5,3.5] stays within Floor_1's y=[-5,5] too -- unlike the L-shaped scene, nothing
    # here needs a height-capped boundary to avoid blocking a legitimate 2F crossing, so these can
    # run the full height comfortably above TOP_Z.
    _BOUNDARY_WALL_H = 1.3
    _create_box(stage, "Floor_1_Wall_XMin", [-8.0 - WALL_T / 2, 0.0, _BOUNDARY_WALL_H / 2],
                [WALL_T, 10.0 + 2 * WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_XMax", [8.0 + WALL_T / 2, 0.0, _BOUNDARY_WALL_H / 2],
                [WALL_T, 10.0 + 2 * WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_YMin", [0.0, -5.0 - WALL_T / 2, _BOUNDARY_WALL_H / 2],
                [16.0, WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_YMax", [0.0, 5.0 + WALL_T / 2, _BOUNDARY_WALL_H / 2],
                [16.0, WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)

    # --- Straight stair: climbs +X, y=Y0 ---
    for i in range(N):
        top = STEP_H * (i + 1)
        x = X0 + STEP_D / 2 + i * STEP_D
        _create_box(stage, f"StraightStair_{i:02d}", [x, Y0, top / 2], [STEP_D, WIDTH, top], C_S,
                    SEG_ID_STAIRS, aabbs)

    exit_x = X0 + N * STEP_D

    # Last (tallest) step's far face (toward Floor_2) is fully exposed -- see
    # l_shaped_stair_scene_with_walls_viplanner.py's own comment for the full explanation. Full
    # height (no gap), per the established "unconditionally: too tall to climb from here = wall"
    # principle -- the crossing here (last tread -> Floor_2) happens at the same z either way (both
    # flush at TOP_Z), so this patch's top (TOP_Z) + 0.15m Z-margin still sits below the real
    # crossing height (TOP_Z + ~0.28m stance offset) -- no overlap risk.
    _stair_wall_h = TOP_Z
    _tag_wall_region(aabbs, [exit_x - 0.05, Y0, _stair_wall_h / 2], [0.10, WIDTH * 0.7, _stair_wall_h],
                      SEG_ID_WALL)  # narrowed to 70% width, same reasoning as the L-shaped scene's

    # Side faces: solid wedge boxes expose their full-height left/right sides to anyone approaching
    # off-axis, same as the L-shaped scene's Stair A/B side patches. Semantic-only, full length,
    # constant height (avoids the jagged-silhouette issue documented there).
    _side_t = 0.10
    _tag_wall_region(aabbs, [(X0 + exit_x) / 2, Y0 + WIDTH / 2 + _side_t / 2, TOP_Z / 2],
                      [exit_x - X0, _side_t, TOP_Z], SEG_ID_WALL)
    _tag_wall_region(aabbs, [(X0 + exit_x) / 2, Y0 - WIDTH / 2 - _side_t / 2, TOP_Z / 2],
                      [exit_x - X0, _side_t, TOP_Z], SEG_ID_WALL)

    # --- 2F floor ---
    floor2_center_x = exit_x + FLOOR2_L / 2
    _create_box(stage, "Floor_2", [floor2_center_x, Y0, TOP_Z - FLOOR_T / 2],
                [FLOOR2_L, FLOOR2_W, FLOOR_T], C_2, SEG_ID_FLOOR, aabbs)

    return {"floor2_center": (floor2_center_x, Y0, TOP_Z)}


async def _open():
    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    print("[straight_stair_viplanner] new stage created")

    from pxr import UsdGeom, UsdLux, Gf
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    import sys
    if "/home/isaacsim/viplanner2/isaac_sim_bridge" not in sys.path:
        sys.path.insert(0, "/home/isaacsim/viplanner2/isaac_sim_bridge")

    aabbs = []
    coords = _build_scene(stage, aabbs)
    print("[straight_stair_viplanner] scene built:", coords)
    print(f"[straight_stair_viplanner] tagged {len(aabbs)} semantic AABBs (floor/stairs/wall)")

    try:
        import json as _json
        _aabbs_json_path = "/home/isaacsim/training_data/straight_stairs_scene/aabbs.json"
        os.makedirs(os.path.dirname(_aabbs_json_path), exist_ok=True)
        with open(_aabbs_json_path, "w") as _f:
            _json.dump({"aabbs": aabbs}, _f)
        print(f"[straight_stair_viplanner] wrote {len(aabbs)} AABBs to {_aabbs_json_path}")
    except Exception as _e:
        print(f"[straight_stair_viplanner] WARNING: failed to write aabbs.json: {_e!r}")

    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight_bright")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.85, 0.9))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight_bright")
    key.CreateIntensityAttr(1500.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60, 30, 0))
    print("[straight_stair_viplanner] added DomeLight_bright + KeyLight_bright")

    bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"], useExtentsHint=False)

    # Explicit user request, same isolation test as the L-shaped scene: goal X/Y matches the dog's
    # own spawn point (the stairs entrance), Z raised to 2F height -- the only difference between
    # "here" and "the goal" is height. Paired with VIPLANNER_STOP_ON_Z_MATCH=1 (viplanner_control.py
    # env var) so the episode ends once height matches, regardless of X/Y (which isn't exactly
    # reachable at that Z here either -- Floor_2 doesn't extend back to Y0... it does actually, since
    # this scene's Floor_2 is centered on the same Y0, but X-wise the entrance's own X is unreachable
    # at 2F height, same idea as the L-shaped scene's test).
    waypoints = [(DOG_START_XY[0], DOG_START_XY[1], TOP_Z + 0.35)]
    _last_i = len(waypoints) - 1
    for i, wp in enumerate(waypoints):
        if i == _last_i:
            wp_prim = UsdGeom.Sphere.Define(stage, f"/World/ViPlanner_Goal_{i + 1}")
            wp_prim.CreateRadiusAttr(0.15)
            wp_prim.AddTranslateOp().Set(Gf.Vec3d(*wp))
            wp_prim.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.85, 1.0)])
        print(f"[straight_stair_viplanner] waypoint {i + 1}/{len(waypoints)}"
              f"{' (visible marker)' if i == _last_i else ''} at:", wp)

    # --- dog placement: probe-then-reposition, ground is z=0 ---
    dog_xform = UsdGeom.Xform.Define(stage, "/World/go2")
    translate_op = dog_xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(DOG_START_XY[0], DOG_START_XY[1], 0.0))
    # VIPLANNER_ROBOT (ROBOT= in run_stair_usd.sh), same bodies and spawn heights as open_scene_stairs.py
    _robot = os.environ.get("VIPLANNER_ROBOT") or "go2"
    _robot_usd = {"go2": "/home/isaacsim/fire_dog/go2.usd",
                  "leo": "/home/isaacsim/robots/leo/leo_description.usd",
                  "growin": "/home/isaacsim/robots/growin/growin.usd"}[_robot]
    dog_xform.GetPrim().GetReferences().AddReference(_robot_usd)
    print("[straight_stair_viplanner] added /World/go2 (probing pose), robot:", _robot, _robot_usd)

    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()

    probe_box = bbox_cache.ComputeWorldBound(dog_xform.GetPrim()).ComputeAlignedRange()
    foot_drop = probe_box.GetMin()[2]
    final_z = -foot_drop + 0.02
    if _robot == "leo":
        final_z = 0.60   # training init height; the zero-pose probe would fold its legs on landing
    translate_op.Set(Gf.Vec3d(DOG_START_XY[0], DOG_START_XY[1], final_z))
    print("[straight_stair_viplanner] probe bbox min z:", foot_drop, "-> final z placement:", final_z)

    # Fixed camera behind the dog's own start point -- see l_shaped_stair_scene_with_walls_viplanner.py's
    # own comment; requires launching with VIPLANNER_FREE_CAMERA=1 so viplanner_control.py's chase
    # camera doesn't overwrite this placement once the control loop starts.
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        _cam_eye = [DOG_START_XY[0] - 2.7, DOG_START_XY[1], 1.6]
        _cam_target = [DOG_START_XY[0], DOG_START_XY[1], 0.3]
        set_camera_view(_cam_eye, _cam_target)
        print(f"[straight_stair_viplanner] set static camera behind dog start: eye={_cam_eye} target={_cam_target}")
    except Exception as e:
        print("[straight_stair_viplanner] ERROR while setting static camera:", repr(e))

    # --- hand off to the ViPlanner + HIMLoco control loop ---
    from viplanner_control import run_walk

    _collect_dir = os.environ.get("VIPLANNER_COLLECT_DATA") or None
    _collect_episodes = int(os.environ.get("VIPLANNER_COLLECT_EPISODES", "30"))

    try:
        await run_walk(
            stage=stage,
            app=omni.kit.app.get_app(),
            dog_prim_path="/World/go2",
            goal_pos=waypoints,
            obstacle_boxes=None,
            floor_box=None,
            precomputed_aabbs=aabbs,
            guardrail_boxes=None,
            collect_data_dir=_collect_dir,
            collect_episodes=_collect_episodes,
            max_steps=int(os.environ.get("VIPLANNER_MAX_STEPS") or 13000),   # MAX_STEPS= in run_stair_usd.sh
            # Explicit user request: simulate smoke on 2F (see viplanner_control.py's own
            # smoke_zone_z_min docstring) -- actual degradation only happens if
            # VIPLANNER_SMOKE_SEVERITY > 0 is also set when launching, so this is harmless by
            # default. TOP_Z - 0.1 starts the effect slightly before the last tread, since the
            # last step's own top is flush with Floor_2's surface height.
            smoke_zone_z_min=TOP_Z - 0.1,
        )
    except Exception as e:
        import traceback
        print("[straight_stair_viplanner] ERROR in run_walk:", repr(e))
        traceback.print_exc()


asyncio.ensure_future(_open())
