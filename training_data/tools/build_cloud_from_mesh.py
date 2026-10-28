"""Labelled point cloud for the semantic cost map, from the scene's own collision geometry -- no
hand-written aabbs (phase 2 of the real-robot plan, explicit user request 2026-10-02). Same rules as
isaac_sim_bridge/geometric_semantic_v2.py, applied to mesh surface samples instead of a depth image:

  1. Sample every collision triangle uniformly (DENSITY points / m^2), keep each sample's face normal.
  2. Only UP-facing samples (n_z > HORIZ_MIN; Stair.usdc's winding is consistent: tops +z, bottoms -z)
     are candidate standing surfaces, and only with clearance: no DOWN-facing surface within
     CLEARANCE above (a cabinet's toe-kick recess top, the floor under a counter). Down-facing samples
     (slab / tread undersides) are dropped -- in a 2D cost map they'd land on top of the floor.
  3. Flood fill over standing surfaces (5cm cells, one node per distinct height in a column) from the
     robot's spawn point, stepping only between surfaces <= STEP_MAX apart in height and never across
     a wall (a vertical surface rising more than 5cm above both sides in the skipped cell).
     Reached = walkable, the rest (counter tops, unreachable ledges) = obstacle.
  4. Walkable with a one-step height change within STAIR_RADIUS -> stairs, spread over STAIR_SPREAD
     (one tread) to same-height walkable cells; other walkable -> floor.
  5. The semantic cost map is 2D (and drops everything above COSTMAP_CEIL), so anything hidden UNDER a
     walkable surface that is itself inside that band is DROPPED: risers, a flight's interior, and
     the floor under a solid flight -- otherwise they project onto the flight as obstacles. Covered
     standing surfaces (no clearance) are dropped too. Every other vertical sample -> obstacle.
Colors match build_cloud.py / camera_semantics_lab (floor, stairs, wall).

Usage: python build_cloud_from_mesh.py <collision_tris.npz> <out cloud.ply> <seed_x> <seed_y> <seed_z>
"""
import bisect
import sys
from collections import defaultdict, deque

import numpy as np
import open3d as o3d

DENSITY = 2500.0     # samples / m^2 (2cm spacing)
CELL = 0.05
HORIZ_MIN = 0.9
STEP_MAX = 0.22
STEP_MIN = 0.10
STAIR_RADIUS = 0.15
STAIR_SPREAD = 0.27
GAP_CELLS = 2
ZQ = 0.02            # height quantisation for nodes in a column
CLEARANCE = 0.30     # m, free height a standing surface needs above it
COSTMAP_CEIL = 2.1   # m, SemCostMapConfig robot_height * robot_height_factor: points above are ignored

FLOOR, STAIRS, WALL = np.array([0, 204, 0]) / 255.0, np.array([0, 153, 0]) / 255.0, np.array([102, 0, 204]) / 255.0


def sample(tris, owner, rng):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    keep = area > 1e-9
    a, b, c, n, area, owner = a[keep], b[keep], c[keep], n[keep], area[keep], owner[keep]
    nz = n[:, 2] / (2 * area)  # signed: + up-facing, - down-facing
    cnt = rng.poisson(area * DENSITY)
    idx = np.repeat(np.arange(len(area)), cnt)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    return pts, nz[idx], owner[idx]


def main(npz, out, seed):
    rng = np.random.default_rng(0)
    data = np.load(npz)
    pts, nz, own = sample(data["tris"].astype(np.float64), data["owner"], rng)
    up, down = nz > HORIZ_MIN, nz < -HORIZ_MIN
    horiz = up  # samples handled as standing-surface candidates below
    vert = ~up & ~down
    col = lambda p: (int(np.floor(p[0] / CELL)), int(np.floor(p[1] / CELL)))  # noqa: E731
    print(f"[mesh_cloud] {len(pts)} samples: {up.sum()} up-facing, {down.sum()} down-facing (dropped), {vert.sum()} vertical")

    # --- up-facing surfaces per column (unique heights), down-facing / vertical heights per column ---
    hcols, vcols, dcols = defaultdict(set), defaultdict(list), defaultdict(list)
    up_own = defaultdict(list)  # (column, mesh) -> up-facing heights, to recognise thin sheets
    for p, o in zip(pts[up], own[up]):
        hcols[col(p)].add(int(round(p[2] / ZQ)))
        up_own[(col(p), o)].append(p[2])
    for p in pts[vert]:
        vcols[col(p)].append(p[2])
    for p, o in zip(pts[down], own[down]):
        # a mat / floor-marking decal is a zero- or 1cm-thick sheet: its own top is right above its
        # bottom, and that bottom lies ON the floor -- it is not a ceiling over the floor
        if any(0 <= t - p[2] <= 0.03 for t in up_own.get((col(p), o), ())):
            continue
        dcols[col(p)].append(p[2])
    for k in vcols:
        vcols[k].sort()

    # --- clearance: drop standing surfaces covered by a down-facing surface within CLEARANCE ---
    tops = {}
    for k, zs in hcols.items():
        ceil = dcols.get(k, ())
        tops[k] = [z for z in sorted(zs) if not any(-0.01 <= c - z * ZQ <= CLEARANCE for c in ceil)]
    nodes = [(a, b, z) for (a, b), zs in tops.items() for z in zs]
    node_id = {n: i for i, n in enumerate(nodes)}
    by_col = defaultdict(list)
    for a, b, z in nodes:
        by_col[(a, b)].append(z)
    print(f"[mesh_cloud] {len(nodes)} standing-surface nodes")

    def wall_between(cell, h_lo, h_hi):
        zs = vcols.get(cell)
        if not zs:
            return False
        i = bisect.bisect_right(zs, h_hi + 0.05)
        return i < len(zs) and zs[i] < h_hi + 0.6  # vertical surface continuing well above both sides

    # --- flood fill from the seed ---
    sx, sy, sz = seed
    sc = (int(np.floor(sx / CELL)), int(np.floor(sy / CELL)))
    seeds = [node_id[(a, b, z)] for (a, b), zs in by_col.items() for z in zs
             if abs(a - sc[0]) <= 10 and abs(b - sc[1]) <= 10 and abs(z * ZQ - sz) < 0.08]
    reach = np.zeros(len(nodes), dtype=bool)
    reach[seeds] = True
    q = deque(seeds)
    nb = [(dx, dy) for dx in range(-GAP_CELLS, GAP_CELLS + 1) for dy in range(-GAP_CELLS, GAP_CELLS + 1) if (dx, dy) != (0, 0)]
    while q:
        a, b, z = nodes[q.popleft()]
        h = z * ZQ
        for dx, dy in nb:
            for z2 in by_col.get((a + dx, b + dy), ()):
                j = node_id[(a + dx, b + dy, z2)]
                h2 = z2 * ZQ
                if reach[j] or abs(h2 - h) > STEP_MAX:
                    continue
                if max(abs(dx), abs(dy)) > 1 and wall_between((a + int(round(dx / 2)), b + int(round(dy / 2))),
                                                              min(h, h2), max(h, h2)):
                    continue
                reach[j] = True
                q.append(j)
    print(f"[mesh_cloud] reached {reach.sum()} / {len(nodes)} nodes from seed {seed}")

    # --- floor vs stairs ---
    cls = np.where(reach, 1, 0)  # 0 obstacle, 1 floor, 2 stairs
    r, rs = int(round(STAIR_RADIUS / CELL)), int(round(STAIR_SPREAD / CELL))
    edge = np.zeros(len(nodes), dtype=bool)
    for i in np.where(reach)[0]:
        a, b, z = nodes[i]
        h = z * ZQ
        edge[i] = any(reach[node_id[(a + dx, b + dy, z2)]] and STEP_MIN <= abs(z2 * ZQ - h) <= STEP_MAX
                      for dx in range(-r, r + 1) for dy in range(-r, r + 1) for z2 in by_col.get((a + dx, b + dy), ()))
    cls[edge] = 2
    for i in np.where(edge)[0]:
        a, b, z = nodes[i]
        for dx in range(-rs, rs + 1):
            for dy in range(-rs, rs + 1):
                if dx * dx + dy * dy > rs * rs:
                    continue
                for z2 in by_col.get((a + dx, b + dy), ()):
                    j = node_id[(a + dx, b + dy, z2)]
                    if reach[j] and abs(z2 - z) * ZQ < STEP_MIN:
                        cls[j] = 2

    # --- label samples ---
    out_pts, out_col = [], []
    for p in pts[horiz]:
        k = col(p)
        z = int(round(p[2] / ZQ))
        if z not in tops.get(k, ()):
            continue  # covered (no clearance): inside / under something, not part of the 2D map
        c = cls[node_id[(k[0], k[1], z)]]
        out_pts.append(p)
        out_col.append(WALL if c == 0 else FLOOR if c == 1 else STAIRS)
    walk_tops = defaultdict(list)
    for i in np.where(reach)[0]:
        a, b, z = nodes[i]
        walk_tops[(a, b)].append(z * ZQ)
    for p in pts[vert]:
        k = col(p)
        hidden = any(p[2] - 0.03 <= t <= COSTMAP_CEIL and t >= p[2] - 0.03
                     for dx in range(-1, 2) for dy in range(-1, 2) for t in walk_tops.get((k[0] + dx, k[1] + dy), ()))
        if hidden:
            continue
        out_pts.append(p)
        out_col.append(WALL)
    out_pts, out_col = np.asarray(out_pts), np.asarray(out_col)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out_pts)
    pcd.colors = o3d.utility.Vector3dVector(out_col)
    o3d.io.write_point_cloud(out, pcd)
    n_f, n_s = (out_col == FLOOR).all(1).sum(), (out_col == STAIRS).all(1).sum()
    print(f"[mesh_cloud] wrote {len(out_pts)} points (floor {n_f}, stairs {n_s}, wall {len(out_pts) - n_f - n_s}) -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], tuple(float(v) for v in sys.argv[3:6]))
