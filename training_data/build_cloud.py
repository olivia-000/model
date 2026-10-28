"""Build a semantic point cloud (cloud.ply) for the stairs scene from its exact AABB geometry
(training_data/stairs_scene/aabbs.json, extracted from open_scene_stairs.py's own _build_scene()),
in the format viplanner_core's SemCostMap.pcd_init() expects: an Open3D point cloud with per-point
RGB colors matching VIPlannerSemMetaHandler's class colors (floor/stairs/wall), which is exactly
the same color scheme camera_semantics_lab.py already uses for the live semantic images -- so the
cost map this produces is consistent with what the network sees during both collection and
inference, not a separately-eyeballed approximation.

No real point-cloud scan needed: since we have exact box geometry, we sample points directly on
the relevant surfaces (top face for floor/stairs, all 4 side faces for walls) instead.
"""
import json
import sys

import numpy as np
import open3d as o3d

SEG_ID_FLOOR, SEG_ID_STAIRS, SEG_ID_WALL, SEG_ID_RAILING = 1, 2, 3, 4
CLASS_COLOR = {
    SEG_ID_FLOOR: np.array([0, 204, 0]) / 255.0,
    SEG_ID_STAIRS: np.array([0, 153, 0]) / 255.0,
    SEG_ID_WALL: np.array([102, 0, 204]) / 255.0,
    SEG_ID_RAILING: np.array([76, 0, 153]) / 255.0,  # VIPLANNER_SEM_META "fence"
}

SPACING = 0.06  # meters between sampled points -- finer than the cost map's own 0.05m grid
# resolution isn't needed; this just needs to be dense enough that no grid cell is ever empty


def sample_top_face(xmin, xmax, ymin, ymax, z):
    nx = max(2, int(round((xmax - xmin) / SPACING)) + 1)
    ny = max(2, int(round((ymax - ymin) / SPACING)) + 1)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def sample_wall_box(xmin, xmax, ymin, ymax, zmin, zmax):
    """All 4 vertical side faces (not the interior volume) -- what a sensor would actually see
    looking at this wall from any direction, and enough for the cost map's 2D XY projection to
    register the wall's full height range as an obstacle."""
    nz = max(2, int(round((zmax - zmin) / SPACING)) + 1)
    zs = np.linspace(zmin, zmax, nz)

    nx = max(2, int(round((xmax - xmin) / SPACING)) + 1)
    ny = max(2, int(round((ymax - ymin) / SPACING)) + 1)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    pts = []
    # y = ymin / y = ymax faces (vary x, z)
    xx, zz = np.meshgrid(xs, zs)
    pts.append(np.stack([xx.ravel(), np.full(xx.size, ymin), zz.ravel()], axis=-1))
    pts.append(np.stack([xx.ravel(), np.full(xx.size, ymax), zz.ravel()], axis=-1))
    # x = xmin / x = xmax faces (vary y, z)
    yy, zz2 = np.meshgrid(ys, zs)
    pts.append(np.stack([np.full(yy.size, xmin), yy.ravel(), zz2.ravel()], axis=-1))
    pts.append(np.stack([np.full(yy.size, xmax), yy.ravel(), zz2.ravel()], axis=-1))
    return np.concatenate(pts, axis=0)


def main(aabbs_json_path, out_ply_path):
    with open(aabbs_json_path) as f:
        data = json.load(f)
    aabbs = data["aabbs"]

    all_pts = []
    all_colors = []
    for xmin, xmax, ymin, ymax, zmin, zmax, cid in aabbs:
        if cid in (SEG_ID_FLOOR, SEG_ID_STAIRS):
            pts = sample_top_face(xmin, xmax, ymin, ymax, zmax)
        elif cid in (SEG_ID_WALL, SEG_ID_RAILING):
            pts = sample_wall_box(xmin, xmax, ymin, ymax, zmin, zmax)
        else:
            continue
        all_pts.append(pts)
        all_colors.append(np.tile(CLASS_COLOR[cid], (pts.shape[0], 1)))

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"[build_cloud] {points.shape[0]} points total "
          f"({sum(p.shape[0] for p, c in zip(all_pts, all_colors))} raw, deduping not applied)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    print(f"[build_cloud] wrote {out_ply_path}")


if __name__ == "__main__":
    aabbs_path = sys.argv[1] if len(sys.argv) > 1 else "/home/itri/Desktop/viplanner2/training_data/stairs_scene/aabbs.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/itri/Desktop/viplanner2/training_data/stairs_scene/cloud.ply"
    main(aabbs_path, out_path)
