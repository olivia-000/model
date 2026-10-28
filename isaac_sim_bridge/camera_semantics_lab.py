"""Isaac Sim port of legged_gym/envs/go2_stairs/camera_semantics.py, adapted for the
Collected_lab_main scene instead of the stairs scene's ramp_cfg.yml.

Same geometric approach: reproject each depth pixel into world XYZ and classify it against a
list of axis-aligned obstacle boxes (built from the lab scene's own Wall/Cabinet/Floor prim
bounding boxes -- see build_lab_aabbs() -- instead of ramp_cfg.yml's stair blocks), producing the
same VIPlannerSemMetaHandler-colored (H,W,3) uint8 image the checkpoint's semantic encoder
(sem=True, rgb=False) expects. Camera frame convention: +X forward, +Y left, +Z up (same
robotics convention as the original, confirmed against this exact camera mount).
"""
import numpy as np

SEG_ID_FLOOR = 1
SEG_ID_STAIRS = 2
SEG_ID_WALL = 3
# railings/balusters: non-walkable like walls, mapped to ViPlanner's own "fence" class
SEG_ID_RAILING = 4

SEG_CLASS_COLOR = {
    0: (30, 30, 30),  # unlabeled/background
    SEG_ID_FLOOR: (0, 204, 0),  # VIPLANNER_SEM_META "floor"
    SEG_ID_STAIRS: (0, 153, 0),  # VIPLANNER_SEM_META "stairs"
    SEG_ID_WALL: (102, 0, 204),  # VIPLANNER_SEM_META "wall"
    SEG_ID_RAILING: (76, 0, 153),  # VIPLANNER_SEM_META "fence"
}

_MAX_SEG_ID = max(SEG_CLASS_COLOR.keys())
_COLOR_LUT = np.zeros((_MAX_SEG_ID + 1, 3), dtype=np.uint8)
for _seg_id, _color in SEG_CLASS_COLOR.items():
    _COLOR_LUT[_seg_id] = _color

_MARGIN = 0.03  # meters, AABB padding so points landing exactly on a shared face still classify


def build_lab_aabbs(wall_boxes_xy, floor_box_xyz):
    """wall_boxes_xy: list of (xmin,xmax,ymin,ymax) footprints (already includes any margin the
    caller wants -- no extra padding applied here beyond _MARGIN in Z).
    floor_box_xyz: (xmin,xmax,ymin,ymax,zmin,zmax) for the floor slab.
    Returns the same (xmin,xmax,ymin,ymax,zmin,zmax,class_id) tuple list classify_world_points
    expects, walls spanning floor to ceiling height."""
    aabbs = []
    floor_z = floor_box_xyz[4]
    ceiling_z = floor_box_xyz[5] + 3.0  # walls extend well above the room's own bbox top
    for xmin, xmax, ymin, ymax in wall_boxes_xy:
        aabbs.append((xmin, xmax, ymin, ymax, floor_z - _MARGIN, ceiling_z, SEG_ID_WALL))
    fx0, fx1, fy0, fy1, fz0, fz1 = floor_box_xyz
    aabbs.append((fx0 - _MARGIN, fx1 + _MARGIN, fy0 - _MARGIN, fy1 + _MARGIN, fz0 - _MARGIN, fz0 + 0.1, SEG_ID_FLOOR))
    return aabbs


def build_stairs_aabbs(ramp_cfg_path):
    """Reads a ramp_cfg.yml (multi_floor stairs scene) directly and returns the same
    (xmin,xmax,ymin,ymax,zmin,zmax,class_id) tuple list classify_world_points expects --
    stair_a_*/stair_b_* blocks -> SEG_ID_STAIRS, landing -> SEG_ID_FLOOR, everything else
    (floor_2_* blocks) -> SEG_ID_WALL. Mirrors legged_gym/envs/go2_stairs/camera_semantics.py's
    original _load_block_aabbs/_block_class_id exactly, just reading the yaml with PyYAML
    directly instead of loading it at import time from a fixed relative path."""
    import yaml

    def _class_id(key):
        if key.startswith("stair_a") or key.startswith("stair_b"):
            return SEG_ID_STAIRS
        if key == "landing":
            return SEG_ID_FLOOR
        return SEG_ID_WALL  # floor_2_* blocks

    with open(ramp_cfg_path, "r") as f:
        blocks = yaml.safe_load(f)

    margin = 0.03
    aabbs = []
    for key, block in blocks.items():
        cx, cy, cz = block["translation"]
        sx, sy, sz = block["size"]
        cid = _class_id(key)
        aabbs.append((
            cx - sx / 2 - margin, cx + sx / 2 + margin,
            cy - sy / 2 - margin, cy + sy / 2 + margin,
            cz - sz / 2 - margin, cz + sz / 2 + margin,
            cid,
        ))
    return aabbs


def reproject_depth_to_world(depth: np.ndarray, cam_pos, cam_quat, hfov_deg: float = 87.0) -> np.ndarray:
    """depth: (H,W) meters (positive == hit, inf/nan == no hit). Returns (H,W,3) world XYZ."""
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    hfov = np.radians(hfov_deg)
    fx = W / (2 * np.tan(hfov / 2))
    fy = fx

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # depth is inf/nan on no-hit pixels (by design, see docstring) -- (us-cx) or (vs-cy) is exactly
    # 0 on the center row/column, so 0*inf legitimately produces nan there. That nan is already
    # filtered out downstream by classify_world_points' `finite` mask, so this is expected, not a
    # bug -- silence the numpy RuntimeWarning it would otherwise print instead of "fixing" it away.
    with np.errstate(invalid="ignore"):
        x = (us - cx) * depth / fx
        y = (vs - cy) * depth / fy
        p_cam = np.stack([depth, -x, -y], axis=-1)  # +X fwd, +Y left, +Z up

        qx, qy, qz, qw = cam_quat
        qvec = np.array([qx, qy, qz], dtype=np.float32)
        t = 2 * np.cross(qvec, p_cam)
        p_world = p_cam + qw * t + np.cross(qvec, t)
        p_world = p_world + np.asarray(cam_pos, dtype=np.float32)
    return p_world


def depth_from_aabbs(W: int, H: int, cam_pos, cam_quat, aabbs, hfov_deg: float = 87.0,
                     near: float = 0.0) -> np.ndarray:
    """Synthetic (H,W) image-plane depth by ray-casting the AABB boxes -- the exact inverse of
    reproject_depth_to_world (same pinhole model, cam_quat xyzw). For scenes whose visible geometry
    isn't in the rendered depth (NuRec Gaussian splats in Isaac Sim 5.1): depth comes from the
    collision proxies instead. inf == no hit, like the rendered depth."""
    cx, cy = W / 2.0, H / 2.0
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # camera-frame ray with unit forward component, so the hit parameter t IS image-plane depth
    d_cam = np.stack([np.ones_like(us), -(us - cx) / fx, -(vs - cy) / fx], axis=-1).reshape(-1, 3)
    qx, qy, qz, qw = cam_quat
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    t = 2 * np.cross(qvec, d_cam)
    d = d_cam + qw * t + np.cross(qvec, t)                                   # (N,3) world

    boxes = np.asarray([a[:6] for a in aabbs], dtype=np.float32)            # (B,6)
    lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]                        # (B,3)
    o = np.asarray(cam_pos, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d                                                         # (N,3)
        t0 = (lo[None] - o) * inv[:, None]                                    # (N,B,3)
        t1 = (hi[None] - o) * inv[:, None]
    t0 = np.nan_to_num(t0, nan=-np.inf)
    t1 = np.nan_to_num(t1, nan=np.inf)
    tnear = np.minimum(t0, t1).max(axis=-1)                                   # (N,B)
    tfar = np.maximum(t0, t1).min(axis=-1)
    hit = (tnear <= tfar) & (tfar > near)
    t_hit = np.where(hit, np.maximum(tnear, near), np.inf).min(axis=-1)
    t_hit[t_hit <= near] = np.inf
    # +1mm: land just inside the box, so classify_world_points' inclusive bounds test isn't lost to
    # float error on points sitting exactly on a face
    return (t_hit + 1e-3).reshape(H, W).astype(np.float32)


def classify_world_points(world_xyz: np.ndarray, aabbs) -> np.ndarray:
    x, y, z = world_xyz[..., 0], world_xyz[..., 1], world_xyz[..., 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    class_ids = np.zeros(world_xyz.shape[:2], dtype=np.int64)

    for xmin, xmax, ymin, ymax, zmin, zmax, cid in aabbs:
        mask = finite & (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
        class_ids[mask] = cid

    unmatched_ground = finite & (class_ids == 0) & (z < 0.05)
    class_ids[unmatched_ground] = SEG_ID_FLOOR
    return class_ids


def semantic_color_image(depth: np.ndarray, cam_pos, cam_quat, aabbs, hfov_deg: float = 87.0) -> np.ndarray:
    world_xyz = reproject_depth_to_world(depth, cam_pos, cam_quat, hfov_deg)
    class_ids = classify_world_points(world_xyz, aabbs)
    return _COLOR_LUT[class_ids]


def semantic_color_image_with_points(depth: np.ndarray, cam_pos, cam_quat, aabbs, hfov_deg: float = 87.0):
    """Same as semantic_color_image, but also returns the (H,W,3) world-frame points it already
    computed along the way -- lets a caller that also wants a point cloud (e.g. for rviz2) reuse
    this reprojection instead of separately calling Isaac Sim's camera.get_pointcloud(), whose
    'annotator not available' fallback path is a per-pixel-loop depth-based recompute an order of
    magnitude slower than this vectorized numpy reprojection, and was the actual bottleneck behind
    the "still not smooth" complaint (996 fallback-warning log lines in one ~3-minute run, one per
    replan tick -- not the camera resolution or replan cadence, which had already been tuned down
    with no improvement)."""
    world_xyz = reproject_depth_to_world(depth, cam_pos, cam_quat, hfov_deg)
    class_ids = classify_world_points(world_xyz, aabbs)
    return _COLOR_LUT[class_ids], world_xyz
