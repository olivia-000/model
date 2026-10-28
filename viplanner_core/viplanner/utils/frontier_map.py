# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online occupancy mapping + frontier detection, and the "judge" that scores NoMaD's raw
diffusion samples against a chosen frontier instead of blindly averaging them.

See NOMAD_INTEGRATION.md for how this fits the overall SAR architecture. Unlike
viplanner/cost_maps/tsdf_cost_map.py (which builds a cost map from one fully pre-scanned
cloud.ply, offline), this builds its map incrementally, one depth frame at a time, live, because
frontier exploration is inherently about not knowing the map ahead of time -- the whole point is
deciding where to go next based on what's been seen SO FAR, in the field, not from a completed
scan. Uses a sparse dict-of-cells grid rather than the dense numpy arrays the offline pipeline
uses, since the explored extent isn't known in advance either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` (..., 3) by `quat` (..., 4, w-x-y-z). Same formula as nomad_bridge._quat_apply
    and victim_detection._quat_apply -- duplicated rather than imported, see those modules'
    docstrings for why (keeps each utility independently usable; it's five lines)."""
    q_w = quat[..., 0:1]
    q_xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer grid cells from (x0,y0) to (x1,y1) inclusive, via Bresenham's line algorithm --
    the standard way to enumerate exactly which cells a ray passes through without missing or
    double-stepping diagonally, needed so "mark everything between the sensor and the hit point
    as seen-free" doesn't leave gaps a frontier could hide behind."""
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return cells


@dataclass
class FrontierCluster:
    centroid_world: tuple[float, float]
    size: int  # number of frontier cells in this cluster -- rough proxy for information gain


class OnlineOccupancyGrid:
    """Incremental 2D free/occupied/unknown grid, built from a stream of depth frames + poses.

    Cell classification mirrors TsdfCostMapConfig's height-based convention (ground_height /
    robot_height*robot_height_factor) so behavior is consistent with the offline cost-map
    pipeline's notion of "floor" vs "obstacle", even though this builds its own map online rather
    than reusing that pipeline directly.
    """

    UNKNOWN = 0  # implicit: absent from self.cells
    FREE = 1
    OCCUPIED = 2

    def __init__(
        self,
        resolution: float = 0.1,
        ground_height: float = 0.35,
        robot_height: float = 0.70,
        robot_height_factor: float = 2.0,
    ):
        self.resolution = resolution
        self.ground_height = ground_height
        self.robot_height = robot_height
        self.robot_height_factor = robot_height_factor
        self.cells: dict[tuple[int, int], int] = {}

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self.resolution), math.floor(y / self.resolution))

    def _cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return ((ix + 0.5) * self.resolution, (iy + 0.5) * self.resolution)

    def update(
        self,
        depth_image: torch.Tensor,
        cam_pos_w: torch.Tensor,
        cam_quat_w: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        max_range: float = 8.0,
        pixel_stride: int = 8,
    ) -> None:
        """Fold one depth frame into the map. `pixel_stride` subsamples the image (matching the
        downsampling already used for the point-cloud publisher in viplanner_demo.py) -- frontier
        detection needs coverage, not every pixel, and raycasting every pixel every frame would
        be far more expensive than the coarse grid actually benefits from.

        Simple evidence rule, not a full Bayesian log-odds occupancy grid: a direct hit always
        marks its cell OCCUPIED (overwriting stale FREE evidence); a ray passing through a cell
        on its way to a farther hit marks that cell FREE only if it isn't already OCCUPIED
        (existing obstacle evidence wins over a single pass-through ray). Good enough for frontier
        detection's purposes -- revisit if noisy sensors cause visible flicker.
        """
        h, w = depth_image.shape
        vv, uu = torch.meshgrid(
            torch.arange(0, h, pixel_stride, device=depth_image.device),
            torch.arange(0, w, pixel_stride, device=depth_image.device),
            indexing="ij",
        )
        z = depth_image[vv, uu]
        valid = torch.isfinite(z) & (z > 0)
        if not torch.any(valid):
            return
        uu_v, vv_v, z_v = uu[valid].float(), vv[valid].float(), z[valid]

        x_cam = (uu_v - cx) * z_v / fx
        y_cam = (vv_v - cy) * z_v / fy
        points_cam = torch.stack([x_cam, y_cam, z_v], dim=-1)
        points_world = _quat_apply(cam_quat_w.unsqueeze(0).expand(points_cam.shape[0], -1), points_cam) + cam_pos_w

        cam_x, cam_y = float(cam_pos_w[0]), float(cam_pos_w[1])
        cam_cell = self._world_to_cell(cam_x, cam_y)

        for i in range(points_world.shape[0]):
            px, py, pz = float(points_world[i, 0]), float(points_world[i, 1]), float(points_world[i, 2])
            dist = float(z_v[i])
            beyond_range = dist > max_range
            if beyond_range:
                # clip the marked point to max_range along the ray -- we don't know what's out
                # there past sensor range, so don't record a (fabricated) hit, only the
                # confirmed-empty space up to the sensor's real limit
                scale = max_range / dist
                px = cam_x + (px - cam_x) * scale
                py = cam_y + (py - cam_y) * scale

            end_cell = self._world_to_cell(px, py)
            for cell in _bresenham_line(cam_cell[0], cam_cell[1], end_cell[0], end_cell[1])[:-1]:
                if self.cells.get(cell) != self.OCCUPIED:
                    self.cells[cell] = self.FREE

            if beyond_range:
                if self.cells.get(end_cell) != self.OCCUPIED:
                    self.cells[end_cell] = self.FREE
            elif pz < self.ground_height:
                self.cells[end_cell] = self.FREE
            elif pz < self.robot_height * self.robot_height_factor:
                self.cells[end_cell] = self.OCCUPIED
            # else: above robot height (ceiling/overhang) -- not floor, not a body-height
            # obstacle, deliberately not recorded either way

    def get_frontiers(self, min_cluster_cells: int = 3) -> list[FrontierCluster]:
        """FREE cells with at least one UNKNOWN 4-connected neighbor, clustered into contiguous
        regions. A cluster shrinking and eventually disappearing as more of it gets scanned IS the
        "already explored, don't revisit" signal -- no separate visited-cell tracking needed."""
        frontier_cells = set()
        for (ix, iy), state in self.cells.items():
            if state != self.FREE:
                continue
            if any((ix + dx, iy + dy) not in self.cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                frontier_cells.add((ix, iy))

        clusters: list[FrontierCluster] = []
        visited: set[tuple[int, int]] = set()
        for cell in frontier_cells:
            if cell in visited:
                continue
            component = []
            queue = [cell]
            visited.add(cell)
            while queue:
                cx, cy = queue.pop()
                component.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (nx, ny) in frontier_cells and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            if len(component) < min_cluster_cells:
                continue
            mean_ix = sum(c[0] for c in component) / len(component)
            mean_iy = sum(c[1] for c in component) / len(component)
            clusters.append(FrontierCluster(centroid_world=self._cell_to_world(mean_ix, mean_iy), size=len(component)))
        return clusters

    def select_frontier(
        self,
        robot_xy: tuple[float, float],
        frontiers: list[FrontierCluster] | None = None,
        heading_bias_rad: float | None = None,
        heading_weight: float = 1.0,
    ) -> FrontierCluster | None:
        """Pick the best frontier: bigger (more unexplored area behind it) and closer (cheaper to
        reach) both score higher. Returns None if nothing frontier-like exists yet (e.g. before
        the first depth frame, or the whole reachable area is already mapped).

        `heading_bias_rad` is an optional world-frame direction hint (e.g. from
        mission_vlm.VLMPolicy.next_semantic_subgoal) -- when given, frontiers whose direction from
        the robot aligns with it score a bonus on top of the base size/distance score. Left at its
        default (None) this reproduces the exact prior behavior: pure geometric scoring, no
        semantic layer involved. The alignment term is clamped to >=0 (`max(0.0, cos(...))`) so a
        wrong or low-confidence hint can only ever boost an aligned frontier, never actively
        penalize a good nearby one the hint didn't mention -- same conservative-extension style as
        the rest of this module.
        """
        if frontiers is None:
            frontiers = self.get_frontiers()
        if not frontiers:
            return None

        def score(f: FrontierCluster) -> float:
            dx, dy = f.centroid_world[0] - robot_xy[0], f.centroid_world[1] - robot_xy[1]
            dist = math.hypot(dx, dy)
            base = f.size / (1.0 + dist)
            if heading_bias_rad is None:
                return base
            alignment = max(0.0, math.cos(math.atan2(dy, dx) - heading_bias_rad))
            return base * (1.0 + heading_weight * alignment)

        return max(frontiers, key=score)


def select_best_trajectory(
    samples: torch.Tensor,
    frontier_target_w: tuple[float, float],
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    lookahead_distance: float = 3.0,
    max_goal_distance: float = 10.0,
) -> tuple[torch.Tensor, int]:
    """The frontier "judge": score NoMaD's raw (num_samples, H, 2) diffusion output against a
    chosen frontier and return the single best chunk instead of averaging all of them away (which
    is what nomad_inference.NomadPolicy.predict_waypoints(average=True) does by default -- call
    it with average=False to get the raw samples this function expects).

    Reuses nomad_bridge.nomad_output_to_world_goal for the actual per-sample world-frame
    projection (treating the `num_samples` candidate trajectories as if they were that many
    parallel envs sharing one robot pose) rather than duplicating the lookahead/rotation math.

    Returns (best_waypoints, best_index) -- the winning (H, 2) chunk plus its index into
    `samples`, in case the caller wants to log/visualize which of the diffusion draws won.
    """
    from viplanner.utils.nomad_bridge import nomad_output_to_world_goal

    num_samples = samples.shape[0]
    root_pos_batch = root_pos_w.unsqueeze(0).expand(num_samples, -1)
    root_quat_batch = root_quat_w.unsqueeze(0).expand(num_samples, -1)
    candidate_world_points = nomad_output_to_world_goal(
        samples, root_pos_batch, root_quat_batch, lookahead_distance, max_goal_distance
    )

    target = torch.tensor(frontier_target_w, dtype=candidate_world_points.dtype, device=candidate_world_points.device)
    dist = torch.norm(candidate_world_points[:, :2] - target, dim=-1)
    best_idx = int(torch.argmin(dist))
    return samples[best_idx], best_idx
