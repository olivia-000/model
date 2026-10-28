"""Isaac Sim entry script: l_shaped_stair_scene_with_walls.py's geometry (single 90-degree-turn
staircase, 1F -> Stair A -> Landing -> Stair B -> walled-in 2F room), rebuilt on plain UsdGeom.Cube
prims with UsdPhysics.CollisionAPI + per-box semantic tagging (instead of isaacsim.core.api.World +
FixedCuboid), then handed off to the Go2 dog + ViPlanner/HIMLoco control loop
(isaac_sim_bridge/viplanner_control.py) -- same pattern as open_scene_stairs.py, adapted to this
simpler single-turn layout. Geometry/coordinate math is a direct port of
l_shaped_stair_scene_with_walls.py's own box-by-box layout, so it can't drift out of sync with that
file.
"""
import asyncio
import os

import omni.usd
import omni.kit.app


# ============================================================
# geometry constants (from l_shaped_stair_scene_with_walls.py)
# ============================================================
FLOOR_T = 0.12

STEP_H = 0.10
STEP_D = 0.30
WIDTH = 1.50
N = 6

LANDING = WIDTH  # 1.50 x 1.50 m, matches the stair width

HALF_Z = N * STEP_H   # 0.60
TOP_Z = 2 * HALF_Z     # 1.20

FLOOR2_X = 6.0
FLOOR2_Y = 6.0

WALL_T = 0.12
WALL_H = 1.20

C_F = (0.42, 0.42, 0.42)
C_S = (0.72, 0.72, 0.72)
C_L = (0.55, 0.60, 0.75)
C_2 = (0.38, 0.52, 0.68)
C_W = (0.82, 0.82, 0.82)
C_B = (0.78, 0.65, 0.48)  # 1F boundary walls -- same role as open_scene_stairs.py's, different color
                            # so the two wall "kinds" (2F room vs 1F boundary) stay visually distinct

X0 = -4.0
Y0 = -1.5

# Floor_1 is 14 x 12 centered at the origin -> x in [-7,7], y in [-6,6].

# Explicit user request: spawn right at the Stair A entrance (0.2m in front of it, same point as
# the old 4-waypoint mode's own waypoint 1) instead of under the 2F goal -- isolates the climb/
# landing-transition/turn behavior from the earlier find-the-entrance search phase, for testing
# those specifically without the search variability mixed in.
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
    """Semantic-only AABB, no USD prim / no collision -- see open_scene_stairs.py's own comment on
    why (an overlapping real collider here depenetrated the robot clean off the map in a real run).
    Used to patch the last (tallest) step of each flight, whose far face is fully exposed with
    nothing covering it and would otherwise still read as walkable SEG_ID_STAIRS."""
    cx, cy, cz = position
    sx, sy, sz = scale
    m = _AABB_MARGIN
    aabbs_out.append((cx - sx / 2 - m, cx + sx / 2 + m, cy - sy / 2 - m, cy + sy / 2 + m,
                       cz - sz / 2 - m, cz + sz / 2 + m, class_id))


def _build_scene(stage, aabbs):
    from camera_semantics_lab import SEG_ID_FLOOR, SEG_ID_STAIRS, SEG_ID_WALL

    # --- 1F ---
    _create_box(stage, "Floor_1", [0.0, 0.0, -FLOOR_T / 2], [14.0, 12.0, FLOOR_T], C_F,
                SEG_ID_FLOOR, aabbs)

    # 1F boundary walls -- same fix as open_scene_stairs.py's Floor_1_Wall_* (an earlier run there
    # let the robot wander clean off the floor's edge into freefall). Capped well below TOP_Z=1.20's
    # Floor_2 (2F floor bottom sits at TOP_Z-FLOOR_T/2=1.14): Floor_2 extends to y=7.05, past
    # Floor_1's own y=6 edge, so a full-height wall there would block the robot's legitimate 2F
    # crossing. 0.9m stops ground-level wandering while leaving 0.24m+ clearance under that crossing.
    _BOUNDARY_WALL_H = 0.9
    _create_box(stage, "Floor_1_Wall_XMin", [-7.0 - WALL_T / 2, 0.0, _BOUNDARY_WALL_H / 2],
                [WALL_T, 12.0 + 2 * WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_XMax", [7.0 + WALL_T / 2, 0.0, _BOUNDARY_WALL_H / 2],
                [WALL_T, 12.0 + 2 * WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_YMin", [0.0, -6.0 - WALL_T / 2, _BOUNDARY_WALL_H / 2],
                [14.0, WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor_1_Wall_YMax", [0.0, 6.0 + WALL_T / 2, _BOUNDARY_WALL_H / 2],
                [14.0, WALL_T, _BOUNDARY_WALL_H], C_B, SEG_ID_WALL, aabbs)

    # --- Stair A: climbs +X, y=Y0 ---
    for i in range(N):
        top = STEP_H * (i + 1)
        x = X0 + STEP_D / 2 + i * STEP_D
        _create_box(stage, f"Stair_A_{i:02d}", [x, Y0, top / 2], [STEP_D, WIDTH, top], C_S,
                    SEG_ID_STAIRS, aabbs)

    exit_x = X0 + N * STEP_D

    # Last (tallest) Stair A step's far face (toward the landing) is fully exposed -- see
    # open_scene_stairs.py's Stair_A_BackWall comment for the full explanation (confirmed there via
    # a real run: the robot got stuck here, reading this sheer face as flat walkable floor).
    # Explicit user request, repeated after watching RViz still show a sliver of green at the top of
    # this face: the principle is unconditional -- anywhere too tall to climb from here is wall, no
    # exception for a thin crossing margin. Now covers the FULL exposed height (0 to HALF_Z, matching
    # the real step box exactly), not stopping short at all. Safe against the top-of-flight
    # oscillation open_scene_stairs.py's own version of this patch was built to avoid (see its
    # comment): that risk is about the wall-proximity Z-margin reaching the robot's actual crossing
    # height, but here the crossing (last tread -> landing) happens at the SAME z either way (both
    # flush at HALF_Z), so the patch's top (HALF_Z) plus its 0.15m Z-margin (0.75) still sits below
    # the robot's real crossing height (~0.88, HALF_Z + its own ~0.28m stance offset) -- no overlap.
    # Explicit user request after a real run: with this patch AND both side patches below all at
    # full height, the exit end of the flight (where all three meet, right next to the landing/next
    # floor) got boxed in tightly enough that one episode never found a way through and timed out.
    # Narrowed to 70% of the tread's own width (centered) rather than the full WIDTH -- still marks
    # the exposed face as wall (unconditionally, for the part that matters most: straight ahead),
    # just leaves a strip of clearance on each side for the network to route through instead of
    # being pinched from both sides at once.
    _stair_a_wall_h = HALF_Z
    _stair_a_exposed_w = WIDTH * 0.7
    _tag_wall_region(aabbs, [exit_x - 0.05, Y0, _stair_a_wall_h / 2], [0.10, _stair_a_exposed_w, _stair_a_wall_h],
                      SEG_ID_WALL)

    # Explicit user request, confirmed by watching the live RViz semantic view: approached from the
    # SIDE (not the correct front entrance), each step's LEFT/RIGHT face is a solid vertical surface
    # up to HALF_Z tall -- since every Stair_A_i box is a full wedge down to the ground (not a thin
    # tread), not just that one step's own small riser -- but it was still rendering as plain
    # SEG_ID_STAIRS green, the same color as a genuinely climbable surface. The network has no way to
    # tell "this green means climbable from here" apart from "this green means climbable, just not
    # from this angle" -- it's visually identical. `viplanner_control.py`'s own virtual edge-safety
    # strips already stop the robot from physically walking off these sides, but those are
    # control-loop-only (deliberately never added to `aabbs`, so the network's own semantic
    # perception never sees them -- see that mechanism's own comment) -- purely a physical safety
    # net, not a perception fix. This patches the SAME side faces into the semantic layer itself
    # (still no collision, same as the exposed-face patches above -- geometry already collides
    # correctly via the real Stair_A_i boxes), so the network's own path prediction also reads them
    # as wall and is discouraged from approaching this flight from the side at all, favoring a wider
    # detour around to the real (front) entrance instead. Constant height (not tracking each step's
    # own rise) for the same jagged-silhouette reason as open_scene_stairs.py's guardrail fix.
    # Full length again (was briefly shortened at the exit end to relieve the "boxed in" feeling near
    # the exit, alongside the exposed-face narrowing above) -- confirmed via a real run that WAS the
    # wrong fix: it removed side protection exactly where a fall is most dangerous (near the exit,
    # higher up the flight), and the robot walked off the newly-unprotected side and fell (climbed to
    # z=0.73, dropped back to 0.29, drifted to y=-2.46 -- well outside the tread's own +/-0.75m
    # width). Reverted to full length; the exposed-face patch's own narrowing above is left in place
    # as the (safer, since it doesn't touch fall protection) attempt at relieving the boxed-in issue.
    _stair_a_side_t = 0.10
    _tag_wall_region(aabbs, [(X0 + exit_x) / 2, Y0 + WIDTH / 2 + _stair_a_side_t / 2, HALF_Z / 2],
                      [exit_x - X0, _stair_a_side_t, HALF_Z], SEG_ID_WALL)
    _tag_wall_region(aabbs, [(X0 + exit_x) / 2, Y0 - WIDTH / 2 - _stair_a_side_t / 2, HALF_Z / 2],
                      [exit_x - X0, _stair_a_side_t, HALF_Z], SEG_ID_WALL)

    # --- Landing (1.5 x 1.5 m) ---
    lx = exit_x + LANDING / 2
    ly = Y0
    _create_box(stage, "Landing", [lx, ly, HALF_Z - FLOOR_T / 2], [LANDING, WIDTH, FLOOR_T], C_L,
                SEG_ID_FLOOR, aabbs)
    # No PHYSICAL guard rail on the landing's two exposed edges (+X, away from Stair A; -Y, the side
    # facing away from Stair B's entrance) -- per explicit user request, the stairs/landing must stay
    # free of physical walls, matching a real staircase's own lack of guardrails. But confirmed via a
    # real run: the robot walked off one of these edges and fell the ~0.5m straight down to Floor_1
    # -- with no real geometry there at all, nothing in the semantic layer told it these edges are
    # any different from the rest of the landing's own walkable SEG_ID_FLOOR surface. Same principle
    # as every other patch in this file (unconditionally: too big a height drop to safely enter =
    # wall, not floor, regardless of whether real collision exists there) -- semantic-only WALL
    # strips along both exposed edges, still no collision (matches the side-face/exposed-face patches
    # above), so the network's own perception reads them as an edge to avoid and the existing
    # wall-proximity safety margin keeps the robot off them, without adding any physical rail.
    # Height needs to cover the robot's ACTUAL standing/walking height while on top of the landing
    # (observed ~0.87-0.90m base height there throughout this file's own diagnostics -- landing
    # surface HALF_Z=0.60 plus its own ~0.28-0.30m stance offset), not just the empty drop-space
    # below the surface -- the wall-proximity check gates on the robot's OWN z falling within
    # [zmin-0.15, zmax+0.15], so a patch that only spans 0..HALF_Z would sit entirely below where the
    # robot actually walks and never match at all. HALF_Z + 0.4 comfortably covers that with margin.
    # Confirmed via a real run: the robot fell at the landing's SOUTHWEST CORNER specifically, not
    # along either patched edge itself -- each patch below only spans its own edge's exact width
    # (matching the landing's own footprint exactly), so they meet EXACTLY at the corner with no
    # overlap. A robot drifting slightly past both boundaries at once (west of the landing's own x
    # AND south of its own y) lands in a diagonal gap neither patch covers, and neither does Stair
    # A's real geometry (which also only spans that same y-width) -- open air straight down, with
    # nothing in the semantic layer or real collision to warn it off. _CORNER_WRAP extends each
    # patch past the landing's own width/length so they overlap at both corners instead of meeting
    # edge-to-edge with a diagonal blind spot between them.
    _CORNER_WRAP = 0.3  # meters, past the landing's own edge on each side
    _landing_edge_t = 0.10
    _landing_edge_h = HALF_Z + 0.4
    _tag_wall_region(aabbs, [lx + LANDING / 2 + _landing_edge_t / 2, ly, _landing_edge_h / 2],
                      [_landing_edge_t, WIDTH + 2 * _CORNER_WRAP, _landing_edge_h], SEG_ID_WALL)
    _tag_wall_region(aabbs, [lx, ly - WIDTH / 2 - _landing_edge_t / 2, _landing_edge_h / 2],
                      [LANDING + 2 * _CORNER_WRAP, _landing_edge_t, _landing_edge_h], SEG_ID_WALL)

    # --- Stair B: turn 90 degrees at the landing, climbs +Y, x=lx ---
    y1 = Y0 + LANDING / 2
    for i in range(N):
        top = HALF_Z + STEP_H * (i + 1)
        y = y1 + STEP_D / 2 + i * STEP_D
        _create_box(stage, f"Stair_B_{i:02d}", [lx, y, top / 2], [WIDTH, STEP_D, top], C_S,
                    SEG_ID_STAIRS, aabbs)

    exit_y = y1 + N * STEP_D

    # Same fix, mirrored: Stair B's last (tallest) step's far face (toward Floor_2) is fully
    # exposed, spanning the ground all the way up to TOP_Z (this scene's Stair B boxes are full
    # height from the ground, unlike open_scene_stairs.py's -- verified against
    # l_shaped_stair_scene_with_walls.py's own box() call, which uses height=top, not top-HALF_Z).
    # Same full-height (no gap) change as Stair A's patch above, mirrored: crossing here (last
    # tread -> Floor_2) happens at the same z either way (both flush at TOP_Z), and the patch's top
    # (TOP_Z) + 0.15m Z-margin still sits below the real crossing height (~1.48, TOP_Z + ~0.28m
    # stance offset) -- no overlap risk.
    # Same exit-end narrowing as Stair A's exposed-face patch above -- confirmed via a real run
    # (single-goal mode) that the full WIDTH here, combined with both full-length side patches below,
    # boxed in the Stair-B/Floor_2 transition tightly enough that one episode timed out never finding
    # a way through.
    _stair_b_wall_h = TOP_Z
    _stair_b_exposed_w = WIDTH * 0.7
    _tag_wall_region(aabbs, [lx, exit_y + 0.05, _stair_b_wall_h / 2], [_stair_b_exposed_w, 0.10, _stair_b_wall_h],
                      SEG_ID_WALL)

    # Same side-face semantic patch as Stair A's above, mirrored for the turned flight: Stair B's
    # LEFT/RIGHT faces are at x = lx +/- WIDTH/2 here (its "sides" run in X since it climbs along Y),
    # full length. (Was briefly shortened at the exit end -- reverted, see Stair A's own patch above
    # for why: that removed fall protection exactly where it mattered most.)
    _stair_b_side_t = 0.10
    _tag_wall_region(aabbs, [lx + WIDTH / 2 + _stair_b_side_t / 2, (y1 + exit_y) / 2, TOP_Z / 2],
                      [_stair_b_side_t, exit_y - y1, TOP_Z], SEG_ID_WALL)
    _tag_wall_region(aabbs, [lx - WIDTH / 2 - _stair_b_side_t / 2, (y1 + exit_y) / 2, TOP_Z / 2],
                      [_stair_b_side_t, exit_y - y1, TOP_Z], SEG_ID_WALL)

    # No physical side rails on Stair B either, per the same explicit request -- side-fall
    # protection for the turned flight now comes entirely from viplanner_control.py's virtual
    # edge-safety mechanism, fixed to detect a flight's actual run direction (X or Y) from its own
    # step geometry instead of assuming every flight runs along X. See that file's own comment on
    # the fix, right above where it now builds `_flight_clusters`.

    # --- 2F floor ---
    floor2_center_x = lx
    floor2_y_min = exit_y
    floor2_y_max = exit_y + FLOOR2_Y
    floor2_center_y = (floor2_y_min + floor2_y_max) / 2
    floor2_x_min = floor2_center_x - FLOOR2_X / 2
    floor2_x_max = floor2_center_x + FLOOR2_X / 2

    _create_box(stage, "Floor_2", [floor2_center_x, floor2_center_y, TOP_Z - FLOOR_T / 2],
                [FLOOR2_X, FLOOR2_Y, FLOOR_T], C_2, SEG_ID_FLOOR, aabbs)

    # --- 2F walls: left/right/back solid, front split around the stair opening ---
    wall_z = TOP_Z + WALL_H / 2

    _create_box(stage, "Floor2_Wall_Left", [floor2_x_min + WALL_T / 2, floor2_center_y, wall_z],
                [WALL_T, FLOOR2_Y, WALL_H], C_W, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor2_Wall_Right", [floor2_x_max - WALL_T / 2, floor2_center_y, wall_z],
                [WALL_T, FLOOR2_Y, WALL_H], C_W, SEG_ID_WALL, aabbs)
    _create_box(stage, "Floor2_Wall_Back", [floor2_center_x, floor2_y_max - WALL_T / 2, wall_z],
                [FLOOR2_X, WALL_T, WALL_H], C_W, SEG_ID_WALL, aabbs)

    opening_x_min = lx - WIDTH / 2
    opening_x_max = lx + WIDTH / 2

    front_left_len = opening_x_min - floor2_x_min
    if front_left_len > 0:
        front_left_center_x = floor2_x_min + front_left_len / 2
        _create_box(stage, "Floor2_Wall_Front_Left",
                    [front_left_center_x, floor2_y_min + WALL_T / 2, wall_z],
                    [front_left_len, WALL_T, WALL_H], C_W, SEG_ID_WALL, aabbs)

    front_right_len = floor2_x_max - opening_x_max
    if front_right_len > 0:
        front_right_center_x = opening_x_max + front_right_len / 2
        _create_box(stage, "Floor2_Wall_Front_Right",
                    [front_right_center_x, floor2_y_min + WALL_T / 2, wall_z],
                    [front_right_len, WALL_T, WALL_H], C_W, SEG_ID_WALL, aabbs)

    return {
        "landing_center": (lx, ly, HALF_Z),
        "stair_b_exit": (lx, exit_y, TOP_Z),
        "floor2_center": (floor2_center_x, floor2_center_y, TOP_Z),
    }


async def _open():
    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    stage = ctx.get_stage()
    print("[l_shaped_walls_viplanner] new stage created")

    from pxr import UsdGeom, UsdLux, Gf
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    import sys
    if "/home/isaacsim/viplanner2/isaac_sim_bridge" not in sys.path:
        sys.path.insert(0, "/home/isaacsim/viplanner2/isaac_sim_bridge")

    aabbs = []
    coords = _build_scene(stage, aabbs)
    print("[l_shaped_walls_viplanner] scene built:", coords)
    print(f"[l_shaped_walls_viplanner] tagged {len(aabbs)} semantic AABBs (floor/stairs/wall)")

    # Own subdirectory -- NOT training_data/stairs_scene/aabbs.json, which is the two-flight scene's
    # own live training data (build_cloud.py/fine-tuning pipeline reads that); this is a different
    # scene's geometry and must not overwrite it.
    try:
        import json as _json
        _aabbs_json_path = "/home/isaacsim/training_data/l_shaped_stairs_scene/aabbs.json"
        os.makedirs(os.path.dirname(_aabbs_json_path), exist_ok=True)
        with open(_aabbs_json_path, "w") as _f:
            _json.dump({"aabbs": aabbs}, _f)
        print(f"[l_shaped_walls_viplanner] wrote {len(aabbs)} AABBs to {_aabbs_json_path}")
    except Exception as _e:
        print(f"[l_shaped_walls_viplanner] WARNING: failed to write aabbs.json: {_e!r}")

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
    print("[l_shaped_walls_viplanner] added DomeLight_bright + KeyLight_bright")

    bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"], useExtentsHint=False)

    # Explicit user request: isolate the pure Z-axis-gap judgment specifically -- goal keeps the same
    # X/Y as the dog's own spawn point (the Stair A entrance, DOG_START_XY) but with Z raised to 2F
    # height. Since X/Y match the robot's own starting position exactly, the ONLY difference between
    # "here" and "the goal" is height -- if it climbs at all, that can only be because it judged the
    # z gap and went looking for a way up, not because the goal pulled it sideways somewhere.
    # Expected per that request: it climbs to 2F (matching the goal's Z) and then stops -- though
    # note Floor_2 doesn't actually extend down to Y0 (the entrance's own Y), so the exact XY is
    # unreachable at that Z; REACH_THRESHOLD is a 3D distance, so whether it reads as "reached" once
    # up on Floor_2 depends on how far horizontally it ends up from (DOG_START_XY, TOP_Z+0.35) --
    # this is exactly what's being tested here, not assumed in advance.
    waypoints = [(DOG_START_XY[0], DOG_START_XY[1], TOP_Z + 0.40)]  # +0.05m higher, explicit user request
    _last_i = len(waypoints) - 1
    for i, wp in enumerate(waypoints):
        if i == _last_i:
            wp_prim = UsdGeom.Sphere.Define(stage, f"/World/ViPlanner_Goal_{i + 1}")
            wp_prim.CreateRadiusAttr(0.15)
            wp_prim.AddTranslateOp().Set(Gf.Vec3d(*wp))
            wp_prim.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.85, 1.0)])
        print(f"[l_shaped_walls_viplanner] waypoint {i + 1}/{len(waypoints)}"
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
    print("[l_shaped_walls_viplanner] added /World/go2 (probing pose), robot:", _robot, _robot_usd)

    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()

    probe_box = bbox_cache.ComputeWorldBound(dog_xform.GetPrim()).ComputeAlignedRange()
    foot_drop = probe_box.GetMin()[2]
    final_z = -foot_drop + 0.02
    if _robot == "leo":
        final_z = 0.60   # training init height; the zero-pose probe would fold its legs on landing
    translate_op.Set(Gf.Vec3d(DOG_START_XY[0], DOG_START_XY[1], final_z))
    print("[l_shaped_walls_viplanner] probe bbox min z:", foot_drop, "-> final z placement:", final_z)

    # Explicit user request: a fixed camera positioned behind the dog's own start point, NOT the
    # chase/follow camera (viplanner_control.py's own chase cam updates every 4 steps as the dog
    # moves -- skip it entirely by launching with VIPLANNER_FREE_CAMERA=1, which is what actually
    # keeps this placement from being overwritten once the control loop starts). "Behind" = further
    # back along -X from DOG_START_XY, since the dog spawns facing +X into the stairs.
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        # 2.7m back is close to the max that still stays inside Floor_1's own boundary wall (at
        # x=-7; DOG_START_XY is at x=-4.2, so >2.8m back would put the camera past the wall).
        _cam_eye = [DOG_START_XY[0] - 2.7, DOG_START_XY[1], 1.6]  # further back + higher, explicit user request
        _cam_target = [DOG_START_XY[0], DOG_START_XY[1], 0.3]
        set_camera_view(_cam_eye, _cam_target)
        print(f"[l_shaped_walls_viplanner] set static camera behind dog start: eye={_cam_eye} target={_cam_target}")
    except Exception as e:
        print("[l_shaped_walls_viplanner] ERROR while setting static camera:", repr(e))

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
            guardrail_boxes=None,  # no guardrail-type geometry in this scene -- every SEG_ID_WALL
                                    # AABB here (boundary walls, 2F room walls, exposed-step patches)
                                    # is a real wall / real drop, so the conservative margin is correct
            collect_data_dir=_collect_dir,
            collect_episodes=_collect_episodes,
            max_steps=int(os.environ.get("VIPLANNER_MAX_STEPS") or 13000),   # MAX_STEPS= in run_stair_usd.sh
            # Explicit user request: simulate smoke on 2F (see viplanner_control.py's own
            # smoke_zone_z_min docstring) -- only actually degrades depth if
            # VIPLANNER_SMOKE_SEVERITY > 0 is also set when launching, harmless by default.
            smoke_zone_z_min=TOP_Z - 0.1,
        )
    except Exception as e:
        import traceback
        print("[l_shaped_walls_viplanner] ERROR in run_walk:", repr(e))
        traceback.print_exc()


asyncio.ensure_future(_open())
