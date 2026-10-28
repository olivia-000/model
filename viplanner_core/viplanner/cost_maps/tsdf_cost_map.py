# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# python
import os

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# imperative-cost-map
from viplanner.config import GeneralCostMapConfig, TsdfCostMapConfig


def log_distance_transform(binary_grid: np.ndarray, sigma_smooth: float) -> np.ndarray:
    """Smoothed log-scaled Euclidean distance-to-nearest-zero-cell transform.

    This is the shared core of TsdfCostMap.CreateTSDFMap's cost recipe (distance_transform_edt +
    log(d+e) rescale + Gaussian smooth), factored out so viplanner/cost_maps/dynamic_cost_map.py
    can reuse the exact same numerical convention without duplicating it. `binary_grid` should be
    0 at cells "at/near" whatever reference set defines cost=minimum for the caller's use case
    (e.g. confirmed-free-space cells for the static map) and nonzero everywhere else -- the
    resulting value is high far from that reference set, low near/at it.
    """
    tsdf_array = ndimage.distance_transform_edt(binary_grid)
    tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
    tsdf_array = gaussian_filter(tsdf_array, sigma=sigma_smooth)
    return tsdf_array


class TsdfCostMap:
    """
    Cost Map based on geometric information
    """

    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        # set init flag
        self.is_map_ready = False
        # init point clouds
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        return

    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd = self.obs_pcd.voxel_down_sample(self._cfg_general.resolution)
            self.free_pcd = self.free_pcd.voxel_down_sample(self._cfg_general.resolution * 0.85)

        self.obs_points = np.asarray(self.obs_pcd.points)
        self.free_points = np.asarray(self.free_pcd.points)
        print("number of obs points: %d, free points: %d" % (self.obs_points.shape[0], self.free_points.shape[0]))

    def ReadPointFromFile(self):
        pcd_load = o3d.io.read_point_cloud(os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file))
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return

    def TerrainAnalysis(self, input_points):
        """Classify points into free (walkable) vs. obstacle.

        Beyond the flat floor (absolute height within +/-ground_height of 0), a point above
        ground_height*1.2 is ALSO free if it sits on a surface reachable from the floor through a
        chain of neighboring grid cells each stepping up by at most max_walkable_step (see
        _FloodFillWalkable) -- this is how a staircase/ramp registers as walkable ground no matter
        its total height, while an isolated raised obstacle (wall, furniture -- no graded
        transition from the floor) still doesn't. Points between ground_height and
        ground_height*1.2, or above the ceiling cutoff, are dropped from both sets (unchanged from
        the original height-only version).
        """
        z = input_points[:, 2] + self._cfg_tsdf.offset_z
        below_ceiling = z < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor
        points, z = input_points[below_ceiling], z[below_ceiling]

        ground_mask = (z < self._cfg_tsdf.ground_height) & (z > -self._cfg_tsdf.ground_height)
        elevated_mask = z > self._cfg_tsdf.ground_height * 1.2
        walkable_mask = self._FloodFillWalkable(points[:, :2], z, ground_mask, elevated_mask)

        free_mask = ground_mask | (elevated_mask & walkable_mask)
        obs_mask = elevated_mask & ~walkable_mask
        return points[obs_mask], points[free_mask]

    def _FloodFillWalkable(self, xy, z, ground_mask, elevated_mask):
        """2D grid flood-fill from the floor over elevated points.

        Builds TWO separate per-cell minimum-height grids -- one from ground points only, one
        from elevated points only -- then BFS-expands from ground cells into 4-connected elevated
        neighbors whose height differs from the *reaching* cell's own height by at most
        TsdfCostMapConfig.max_walkable_step, chaining onward through already-reached elevated
        cells. Returns a boolean mask (same length as xy/z) marking which points fall in a cell
        reached by that expansion.

        Keeping the ground and elevated height grids separate is required, not just tidier: an
        earlier version took one combined per-cell minimum over both point sets, which works for
        a stair tread (uniformly one height across its footprint) but silently broke for a real
        wall -- a wall's own base sits right at floor height, sharing the *same* XY footprint as
        its face reaching up several meters (it's a thin, nearly-vertical surface), so the
        combined minimum for that column was ~0m regardless of how tall the wall's elevated
        points actually were, and the flood-fill walked straight up it. Caught by a low obstacle
        count (98 points for a full warehouse pass) that a synthetic floor+staircase+isolated-wall
        test hadn't exercised; a floor-attached wall in the test suite catches it now. Also: a
        ground cell's own height is *not* assumed to be 0 -- e.g. a step already within
        ground_height counts as ground, but its actual elevation (not a hardcoded reference)
        is what the next step's rise is measured against, or a two-step rise that's individually
        climbable would be rejected as a single large jump from a fictitious flat baseline.

        Uses a *coarser* grid than the final cost map's own resolution (3x, see
        flood_fill_resolution_factor) purely for this connectivity search -- reconstructed point
        density thins out at oblique viewing angles, and a stair riser edge (looking down/forward
        at a shallow angle) is exactly where that happens. At the map's native resolution, a
        single missed cell at a step boundary silently severs the chain and the entire staircase
        above that gap falls back to being classified as one big obstacle (verified with a
        synthetic 8-step staircase during development: real floating-point jitter in tightly
        tiled synthetic tread points alone was enough to open a 1-cell gap, and 1 cell is all it
        takes). Aggregating into wider bins first is far cheaper than a larger BFS search radius
        and tolerates that sparsity without changing final map resolution at all.
        """
        if not np.any(elevated_mask):
            return np.zeros(len(z), dtype=bool)

        res = self._cfg_general.resolution * self._cfg_tsdf.flood_fill_resolution_factor
        considered = ground_mask | elevated_mask
        xy_c = xy[considered]
        min_xy = xy_c.min(axis=0)

        def cell_of(mask):
            return np.floor((xy[mask] - min_xy) / res).astype(int)

        grid_idx_all = np.floor((xy_c - min_xy) / res).astype(int)
        num_x, num_y = int(grid_idx_all[:, 0].max()) + 1, int(grid_idx_all[:, 1].max()) + 1

        def per_cell_min(mask):
            # vectorized (avoids a per-point Python loop over what can be millions of points)
            out = np.full(num_x * num_y, np.inf)
            gi = cell_of(mask)
            if len(gi) == 0:
                return out.reshape(num_x, num_y)
            flat_idx = gi[:, 0] * num_y + gi[:, 1]
            order = np.argsort(flat_idx)
            flat_sorted, z_sorted = flat_idx[order], z[mask][order]
            uniq_flat, start_pos = np.unique(flat_sorted, return_index=True)
            out[uniq_flat] = np.minimum.reduceat(z_sorted, start_pos)
            return out.reshape(num_x, num_y)

        ground_height = per_cell_min(ground_mask)
        elevated_height = per_cell_min(elevated_mask)
        elevated_occupied = np.isfinite(elevated_height)

        # a ground cell's own known height is its actual local floor elevation (which may
        # itself already be a low step, not always exactly 0)
        known_height = ground_height.copy()
        reached_elevated = np.zeros((num_x, num_y), dtype=bool)
        max_step = self._cfg_tsdf.max_walkable_step

        frontier = list(zip(*np.where(np.isfinite(ground_height))))
        while frontier:
            cx, cy = frontier.pop()
            h0 = known_height[cx, cy]
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    0 <= nx < num_x
                    and 0 <= ny < num_y
                    and elevated_occupied[nx, ny]
                    and not reached_elevated[nx, ny]
                    and abs(elevated_height[nx, ny] - h0) <= max_step
                ):
                    reached_elevated[nx, ny] = True
                    known_height[nx, ny] = elevated_height[nx, ny]
                    frontier.append((nx, ny))

        walkable = np.zeros(len(z), dtype=bool)
        elevated_idx = cell_of(elevated_mask)
        if len(elevated_idx) > 0:
            walkable[elevated_mask] = reached_elevated[elevated_idx[:, 0], elevated_idx[:, 1]]
        return walkable

    def UpdateMapParams(self):
        if self.obs_points.shape[0] == 0:
            print("No points received.")
            return
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist

        self.num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution

        print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    def CreateTSDFMap(self):
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        free_map = np.ones([self.num_x, self.num_y])
        obs_map = np.zeros([self.num_x, self.num_y])
        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)
        # create free place map
        # NOTE: the grid is sized from obs_points' own bounding box (see UpdateMapParams), but
        # free_points can legitimately extend beyond it (open floor reaching further than any
        # detected obstacle), and either can include noisy outliers -- indices from either set can
        # fall outside [0, num_x) x [0, num_y) and must be dropped, not just clamped on one side
        # (a negative index would otherwise silently wrap around via numpy's negative indexing).
        for i in obs_I:
            if 0 <= i[0] < self.num_x and 0 <= i[1] < self.num_y:
                obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=self._cfg_tsdf.sigma_expand)
        for i in free_I:
            if 0 <= i[0] < self.num_x and 0 <= i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)
        free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
        # assign obstacles
        free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0

        print("occupancy map generation completed.")
        # Distance Transform
        tsdf_array = log_distance_transform(free_map, self._cfg_general.sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        # TODO: Using true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0
        return [tsdf_array, viz_points, ground_array], [
            float(self.start_x),
            float(self.start_y),
        ]

    def IndexArrayOfPs(self, points):
        indexes = points[:, :2] - np.array([self.start_x, self.start_y])
        indexes = (np.round(indexes / self._cfg_general.resolution)).astype(int)
        return indexes

    def FilterCloud(self, points, outlier_filter=True):
        # crop points
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            points_x_idx_upper = (
                (points[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_x_idx_lower = (
                (points[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_upper = (
                (points[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_lower = (
                (points[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points = points[
                np.vstack(
                    (
                        points_x_idx_lower,
                        points_x_idx_upper,
                        points_y_idx_upper,
                        points_y_idx_lower,
                    )
                ).all(axis=0)
            ]

        if outlier_filter:
            # Filter outlier in points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self._cfg_tsdf.nb_neighbors,
                std_ratio=self._cfg_tsdf.std_ratio,
            )
            points = np.asarray(cl.points)

        return points

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])  # visualize point cloud


# EoF
