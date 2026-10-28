"""Depth-only floor / stairs / obstacle classification for ViPlanner's semantic input -- no per-scene
labels (no aabbs), nothing a real robot can't measure itself.

Phase 1 of the "real-robot-ready semantics" plan (explicit user request, 2026-10-02). Successor to
viplanner_core's compute_geometric_semantic(), which thresholds every point's absolute height
against one cutoff: anything below it (holes, the stairwell gap, an unwalled floor edge) reads as
walkable floor, and anything a step or more above the robot reads as obstacle, so stairs are never
recognized as climbable.

Per frame:
  1. Depth pixels -> 3D points, rotated by the camera's orientation only (only roll/pitch matter for
     heights -- an IMU's gravity vector on a real robot), origin at the camera.
  2. Height of every point relative to the robot's own foot plane (`cam_height_above_feet`: from leg
     forward kinematics + the camera mount on a real robot).
  3. Points -> robot-centric 2.5D height grid (CELL m cells, max surface height per cell).
  4. Flood fill from the cells around the robot's feet, stepping only between cells whose heights
     differ by <= STEP_MAX: reached cells are traversable, everything else is an obstacle -- walls
     are too tall to step onto, holes / drop-offs are too deep to step down into.
  5. Traversable cells with a one-step height change within STAIR_RADIUS -> stairs, else floor (a
     flat landing 2.34m up is floor, not "stairs" just because it's higher than the feet).
  6. Each pixel takes its cell's class.

Colors are ViPlanner's own (viplanner_sem_meta / camera_semantics_lab): floor, stairs, wall.
"""
from collections import deque

import numpy as np

FLOOR_COLOR = (0, 204, 0)      # VIPLANNER_SEM_META "floor"
STAIRS_COLOR = (0, 153, 0)     # VIPLANNER_SEM_META "stairs"
OBSTACLE_COLOR = (102, 0, 204) # VIPLANNER_SEM_META "wall" (same class v3/v5/v6 were trained on)

CLS_OBSTACLE, CLS_FLOOR, CLS_STAIRS = 0, 1, 2

CELL = 0.05           # m, height-grid resolution
STEP_MAX = 0.22       # m, largest height change between neighbouring walkable cells (Go2 + 0.18m risers)
STEP_MIN = 0.10       # m, smaller height changes than this are floor unevenness / depth noise, not a step
STAIR_RADIUS = 0.15   # m, a cell is "stairs" if a step edge is this close -- short, so depth noise and
                      # gentle slopes (which only add up over distance) don't read as steps
STAIR_SPREAD = 0.27   # m, stairs label spreads from a step edge across one tread depth
SEED_RADIUS = 1.5     # m, flood fill starts from standable cells this close to the camera
SEED_TOL = 0.08       # m, seed cells: within this of the layer closest to foot level
SEED_MAX_OFFSET = 0.6 # m, ... and that layer must be within this of the estimated foot level
GAP_CELLS = 2         # neighbour search radius in cells -- bridges small unobserved holes in the grid
MAX_RANGE = 8.0       # m, farther / invalid depth -> obstacle ("don't know" is never walkable)
PLANE_RADIUS = 2.5    # m, ground-plane fit uses near-foot-level points this close to the camera
PLANE_BAND = 0.15     # m, ... and within this height of the expected foot level
PLANE_MAX_TILT = 8.0  # deg, larger fitted tilts are rejected (wall/obstacle, not ground) -> no correction
PLANE_MIN_POINTS = 300
NARROW_RADIUS_CELLS = 2  # same-height neighbourhood for the "too narrow to stand on" test
NARROW_DH = 0.03          # m, "same height"
NARROW_MIN_CELLS = 4      # fewer same-height neighbours than this -> too sparse to judge
NARROW_MIN_STD = 0.6      # cells, minor-axis spread below this = a thin strip (ledge narrower than ~0.1m)
HORIZ_MIN = 0.9     # |normal_z| above this (surface tilted < ~26 deg) counts as standable
NORMAL_STEP = 2     # px, normals from neighbours this far apart (1px is too noisy on grazing floor)


def depth_to_points(depth, hfov_deg, depth_is_range=False):
    """(H,W) depth in meters -> (H,W,3) camera-frame points, +X forward / +Y left / +Z up, same pixel
    layout as camera_semantics_lab.reproject_depth_to_world.
    depth_is_range: Isaac Sim's Camera.get_depth() is distance_to_image_plane (planar), confirmed
    against a controlled flat-floor render (floor reprojects to +0.004..+0.010m). Kept as an option
    for depth sensors that report range along the ray."""
    H, W = depth.shape
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xn, yn = (us - W / 2.0) / fx, (vs - H / 2.0) / fx
    with np.errstate(invalid="ignore"):
        fwd = depth / np.sqrt(1.0 + xn * xn + yn * yn) if depth_is_range else depth
        return np.stack([fwd, -xn * fwd, -yn * fwd], axis=-1)


def fit_ground_plane(pts):
    """Least-squares plane z = a*x + b*y + c through near-foot-level points, with two rounds of
    outlier rejection. Returns (a, b, c) or None. Corrects the per-frame roll/pitch/height mismatch
    between the recorded camera pose and the instant the depth was rendered (gait sway; on a real
    robot: IMU/camera sync and mount errors) -- measured median 4.6 deg / varying direction per
    frame, so no fixed calibration offset can remove it."""
    d = np.hypot(pts[:, 0], pts[:, 1])
    m = (d < PLANE_RADIUS) & (np.abs(pts[:, 2]) < PLANE_BAND)
    for _ in range(3):
        if m.sum() < PLANE_MIN_POINTS:
            return None
        A = np.c_[pts[m, 0], pts[m, 1], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, pts[m, 2], rcond=None)
        res = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])
        m = m & (np.abs(res) < 0.05)
    if np.degrees(np.arctan(np.hypot(coef[0], coef[1]))) > PLANE_MAX_TILT:
        return None
    return coef


def rotate_xyzw(q, v):
    qx, qy, qz, qw = q
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    t = 2 * np.cross(qvec, v)
    return v + qw * t + np.cross(qvec, t)


def classify(depth, cam_quat_xyzw, cam_height_above_feet, hfov_deg=87.0, depth_is_range=False,
             ground_fit=True, min_range=0.0):
    """Returns (cls (H,W) int8 of CLS_*, rel_points (H,W,3)). rel_points: gravity-aligned, origin at
    the camera, z relative to the foot plane (ground-plane corrected when ground_fit).
    min_range: depth at or below this is invalid. viplanner_control.py's camera never sets a clipping
    range, so it renders with Isaac Sim's default near plane of 1.0m: everything closer comes back
    pinned at ~1.0m and reprojects to the wrong place (pass ~1.01 for that data)."""
    H, W = depth.shape
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth > max(min_range, 0.0)) & (depth <= MAX_RANGE)
    p = rotate_xyzw(cam_quat_xyzw, depth_to_points(np.where(valid, depth, 0.0), hfov_deg, depth_is_range))
    p[..., 2] += cam_height_above_feet
    if ground_fit:
        coef = fit_ground_plane(p[valid])
        if coef is not None:
            p[..., 2] -= coef[0] * p[..., 0] + coef[1] * p[..., 1] + coef[2]
    cls = np.full((H, W), CLS_OBSTACLE, dtype=np.int8)
    if not valid.any():
        return cls, p

    # --- surface orientation: only near-horizontal surfaces (treads, floors, landings) can be stood
    # on. Without this, a wall seen at an angle lands in a run of neighbouring cells whose max
    # heights rise in small increments, and the flood fill "climbs" it like a staircase. ---
    horiz = np.zeros((H, W), dtype=bool)
    k = NORMAL_STEP
    dxv, dyv = p[k:-k, 2 * k:] - p[k:-k, :-2 * k], p[2 * k:, k:-k] - p[:-2 * k, k:-k]
    n = np.cross(dxv, dyv)
    nn = np.linalg.norm(n, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.abs(n[..., 2]) / nn
    ok = valid[k:-k, 2 * k:] & valid[k:-k, :-2 * k] & valid[2 * k:, k:-k] & valid[:-2 * k, k:-k] & (nn > 0)
    horiz[k:-k, k:-k] = ok & (nz > HORIZ_MIN)
    top_px = valid & horiz
    steep_px = valid & ~horiz

    off = 4096  # cells are < MAX_RANGE/CELL = 160 from the origin
    def cell_key(q):
        ix = np.floor(q[..., 0] / CELL).astype(np.int64)
        iy = np.floor(q[..., 1] / CELL).astype(np.int64)
        return (ix + off) * (2 * off) + (iy + off)

    # walkable surface height per cell (horizontal points only)
    tkey = cell_key(p[top_px])
    ukeys, inv = np.unique(tkey, return_inverse=True)
    top = np.full(len(ukeys), -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, p[top_px][:, 2])
    cix, ciy = ukeys // (2 * off) - off, ukeys % (2 * off) - off
    cell_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cix, ciy))}
    # highest vertical surface per cell (walls, risers, railings)
    skey = cell_key(p[steep_px])
    sk, sinv = np.unique(skey, return_inverse=True)
    stop = np.full(len(sk), -np.inf, dtype=np.float32)
    np.maximum.at(stop, sinv, p[steep_px][:, 2])
    steep_top = {(int(k // (2 * off) - off), int(k % (2 * off) - off)): float(v) for k, v in zip(sk, stop)}

    # a cell whose own vertical surface rises more than a step above its walkable surface is the
    # foot of a wall -- don't stand there
    blocked = np.array([steep_top.get((int(a), int(b)), -np.inf) > h + STEP_MAX
                        for a, b, h in zip(cix, ciy, top)], dtype=bool)

    # A surface you can stand on has to be wide in BOTH horizontal directions. The top edge of a
    # skirting board / wall plinth / curb is a horizontal strip only a few cm deep, ~0.17m up: without
    # this the flood fill took it for a step up from the floor and labelled the wall base walkable
    # (explicit user request, 2026-10-02, frame 948 of stair_usd_eval_clip02_v2). Same-height cells
    # within NARROW_RADIUS: if they lie along a thin line (minor-axis spread < NARROW_MIN_STD cells),
    # the cell is a ledge, not a tread / floor. Too few same-height neighbours (sparse far-away
    # data) -> can't tell -> not marked narrow.
    # Only a strip backed by a TALL vertical surface (rising more than a step above it, i.e. a wall)
    # counts: a tread seen from below at a grazing angle is also just a 1-2 cell strip in the grid, but
    # what rises behind it is the next riser, one step high. Without this the filter also deleted
    # most stairs (stairs recall 90% -> 44% on stair_usd_eval_clip02_v2).
    rn = NARROW_RADIUS_CELLS
    narrow = np.zeros(len(ukeys), dtype=bool)
    for i in range(len(ukeys)):
        a, b, h = int(cix[i]), int(ciy[i]), top[i]
        offs = [(dx, dy) for dx in range(-rn, rn + 1) for dy in range(-rn, rn + 1)
                for j in [cell_of.get((a + dx, b + dy))] if j is not None and abs(top[j] - h) < NARROW_DH]
        if len(offs) < NARROW_MIN_CELLS:
            continue
        o = np.asarray(offs, dtype=np.float32)
        minor = np.sqrt(max(np.linalg.eigvalsh(np.cov(o.T))[0], 0.0))
        if minor < NARROW_MIN_STD and any(
                steep_top.get((a + dx, b + dy), -np.inf) > h + STEP_MAX
                for dx in range(-rn, rn + 1) for dy in range(-rn, rn + 1)):
            narrow[i] = True
    blocked |= narrow

    # --- flood fill over step-sized height changes, from foot-level cells near the robot. A jump
    # over a cell (GAP_CELLS=2) must not pass over a wall: the skipped cell's vertical surface may
    # rise to the higher side's height (a stair riser) but not beyond (a wall / railing). ---
    # Seeds: among the standable surfaces visible near the robot, the layer closest to foot level --
    # not a fixed |h| < tol cut. The ~0.5m in front of the feet is in the camera's blind spot, so on
    # a flight the nearest VISIBLE tread is already 1-3 risers up, and body pitch plus feet on
    # different treads put the fixed-mount foot-height estimate off by up to ~0.2m. (With the fixed
    # cut, half of all frames on the stairs / landings / upper floors had no seed at all, 2026-10-02.)
    dist = np.hypot((cix + 0.5) * CELL, (ciy + 0.5) * CELL)
    cand = np.where((dist < SEED_RADIUS) & ~blocked & (np.abs(top) < SEED_MAX_OFFSET))[0]
    if len(cand) == 0:
        seeds = cand
    else:
        h0 = top[cand[np.argmin(np.abs(top[cand]))]]
        seeds = cand[np.abs(top[cand] - h0) < SEED_TOL]
    reach = np.zeros(len(ukeys), dtype=bool)
    reach[seeds] = True
    queue = deque(seeds.tolist())
    nbrs = [(dx, dy) for dx in range(-GAP_CELLS, GAP_CELLS + 1) for dy in range(-GAP_CELLS, GAP_CELLS + 1)
            if (dx, dy) != (0, 0)]
    while queue:
        i = queue.popleft()
        a, b, h = int(cix[i]), int(ciy[i]), top[i]
        for dx, dy in nbrs:
            j = cell_of.get((a + dx, b + dy))
            if j is None or reach[j] or blocked[j] or abs(top[j] - h) > STEP_MAX:
                continue
            if max(abs(dx), abs(dy)) > 1:
                mid = steep_top.get((a + int(round(dx / 2)), b + int(round(dy / 2))), -np.inf)
                if mid > max(h, top[j]) + 0.05:
                    continue
            reach[j] = True
            queue.append(j)

    # --- floor vs stairs, two stages: (1) step EDGES -- a one-step height change within a short
    # STAIR_RADIUS, short so depth noise / gentle slopes (which only add up over distance) don't
    # read as steps; (2) spread the stairs label from those edges across each whole tread
    # (STAIR_SPREAD ~ one tread depth), only to traversable cells at a similar height, so a 0.27m
    # tread isn't left as "floor" in the middle with a thin "stairs" strip at its edges. ---
    cell_cls = np.where(reach, CLS_FLOOR, CLS_OBSTACLE).astype(np.int8)
    r = int(round(STAIR_RADIUS / CELL))
    edge = np.zeros(len(ukeys), dtype=bool)
    for i in np.where(reach)[0]:
        a, b, h = int(cix[i]), int(ciy[i]), top[i]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                j = cell_of.get((a + dx, b + dy))
                if j is not None and reach[j] and STEP_MIN <= abs(top[j] - h) <= STEP_MAX:
                    edge[i] = True
                    break
            if edge[i]:
                break
    cell_cls[edge] = CLS_STAIRS
    rs = int(round(STAIR_SPREAD / CELL))
    for i in np.where(edge)[0]:
        a, b, h = int(cix[i]), int(ciy[i]), top[i]
        for dx in range(-rs, rs + 1):
            for dy in range(-rs, rs + 1):
                if dx * dx + dy * dy > rs * rs:
                    continue
                j = cell_of.get((a + dx, b + dy))
                if j is not None and reach[j] and abs(top[j] - h) < STEP_MIN:
                    cell_cls[j] = CLS_STAIRS

    cls[top_px] = cell_cls[inv]

    # --- vertical pixels: a riser directly under a STAIRS tread less than a step above it takes the
    # stairs class; anything else vertical is an obstacle. Only stairs: a vertical face topped by
    # FLOOR is a curb / threshold / skirting board, not part of a staircase -- letting it inherit
    # "floor" painted wall bases walkable (frame 948, stair_usd_eval_clip02_v2). ---
    if steep_px.any():
        sp = p[steep_px]
        s_cls = np.full(len(sp), CLS_OBSTACLE, dtype=np.int8)
        for k_idx in range(len(sk)):
            a, b = int(sk[k_idx] // (2 * off) - off), int(sk[k_idx] % (2 * off) - off)
            tops = [(top[j], cell_cls[j]) for dx in range(-GAP_CELLS, GAP_CELLS + 1)
                    for dy in range(-GAP_CELLS, GAP_CELLS + 1)
                    for j in [cell_of.get((a + dx, b + dy))]
                    if j is not None and reach[j] and cell_cls[j] == CLS_STAIRS]
            if not tops:
                continue
            members = np.where(sinv == k_idx)[0]
            hz = sp[members, 2]
            for t, c in sorted(tops, key=lambda tc: tc[0]):
                hit = (s_cls[members] == CLS_OBSTACLE) & (hz >= t - STEP_MAX - 0.03) & (hz <= t + 0.03)
                s_cls[members[hit]] = c
        cls[steep_px] = s_cls
    return cls, p


def to_color(cls):
    lut = np.zeros((3, 3), dtype=np.uint8)
    lut[CLS_OBSTACLE], lut[CLS_FLOOR], lut[CLS_STAIRS] = OBSTACLE_COLOR, FLOOR_COLOR, STAIRS_COLOR
    return lut[cls]
