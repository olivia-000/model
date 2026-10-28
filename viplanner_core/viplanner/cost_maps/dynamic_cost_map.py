# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-rollout dynamic (moving-obstacle) cost volume builder.

Ground-truth counterpart to viplanner/cost_maps/tsdf_cost_map.py's static map, built from the
recorded positions of people animated by
omni.viplanner.utils.unreal_importer.UnRealImporter.update_people(dt) during a timed data-collection
rollout (see omniverse/standalone/data_collect_dynamic.py). Deliberately NOT the same "distance to
confirmed free space" trick TsdfCostMap.CreateTSDFMap uses for the static map -- there is no
free-space evidence for a future point in time, only the obstacle's (extrapolated) position, so
this computes a direct, correctly-signed "cost decays with distance from the obstacle" field
instead. Runs in the plain training conda env (no Isaac Sim / omni.* dependency), so
UnRealImporter.update_people's bounce/loop math is re-derived analytically here rather than
imported.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def person_position_at_time(
    start: np.ndarray,
    target: np.ndarray,
    speed: float,
    mode: str,
    pos_t0: np.ndarray,
    direction_t0: float,
    elapsed: float,
) -> np.ndarray:
    """Analytic (x, y) position of one UnRealImporter-animated person at `elapsed` seconds after a
    known state (pos_t0, direction_t0).

    Mirrors UnRealImporter.update_people's per-step bounce/loop stepping exactly, but evaluated in
    closed form via a reflected/wrapped-progress calculation instead of dt-stepping -- avoids
    accumulated integration error over a multi-second future horizon. "bounce" reflects at both
    ends (a particle bouncing between 0 and the segment length); "loop" always advances toward
    `target` and snaps back to `start` on arrival (matches update_people: loop mode never flips
    `direction`).
    """
    start_xy = np.asarray(start, dtype=float)[:2]
    target_xy = np.asarray(target, dtype=float)[:2]
    pos_xy = np.asarray(pos_t0, dtype=float)[:2]

    segment = target_xy - start_xy
    length = float(np.linalg.norm(segment))
    if length < 1e-6:
        return pos_xy.copy()
    unit = segment / length

    traveled_0 = float(np.clip(np.dot(pos_xy - start_xy, unit), 0.0, length))
    raw = traveled_0 + direction_t0 * speed * elapsed

    if mode == "loop":
        traveled = raw % length
    else:  # bounce
        period = 2.0 * length
        reduced = raw % period
        traveled = reduced if reduced <= length else period - reduced

    return start_xy + unit * traveled


def rasterize_obstacle_disks(
    positions_xy: np.ndarray,
    radius_m: float,
    num_x: int,
    num_y: int,
    start_x: float,
    start_y: float,
    resolution: float,
) -> np.ndarray:
    """Rasterize a set of circular obstacle footprints onto a (num_x, num_y) occupancy grid.

    Same (pos - [start_x, start_y]) / resolution grid-indexing convention as
    TsdfCostMap.IndexArrayOfPs, so the result aligns cell-for-cell with a static map built with the
    same start/resolution/extent.
    """
    grid = np.zeros((num_x, num_y), dtype=np.float64)
    radius_cells = radius_m / resolution
    for px, py in np.atleast_2d(positions_xy):
        cx = (px - start_x) / resolution
        cy = (py - start_y) / resolution
        x0 = max(0, int(np.floor(cx - radius_cells - 1)))
        x1 = min(num_x, int(np.ceil(cx + radius_cells + 2)))
        y0 = max(0, int(np.floor(cy - radius_cells - 1)))
        y1 = min(num_y, int(np.ceil(cy + radius_cells + 2)))
        if x0 >= x1 or y0 >= y1:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius_cells**2
        grid[x0:x1, y0:y1][mask] = 1.0
    return grid


def clearance_to_cost(clearance_m: np.ndarray, decay_scale_m: float) -> np.ndarray:
    """Monotonically decreasing cost from a distance-from-obstacle ("clearance") field.

    cost == 1.0 exactly at the obstacle (clearance == 0), decaying toward 0 as clearance grows,
    with `decay_scale_m` as the e-folding distance in meters. Deliberately not the static map's
    log(EDT+e) convention -- that one only reads as "cost" because its zero-reference is
    confirmed-free-space (nothing analogous exists for a single future obstacle position); this is
    the direct, explicitly-monotonic-in-the-correct-direction cost instead.
    """
    return np.exp(-clearance_m / decay_scale_m).astype(np.float32)


def obstacle_slice_cost(
    positions_xy: np.ndarray,
    radius_m: float,
    num_x: int,
    num_y: int,
    start_x: float,
    start_y: float,
    resolution: float,
    sigma_expand: float,
    decay_scale_m: float,
) -> np.ndarray:
    """Cost grid for one future time-slice: high at/near the (possibly multiple) obstacle
    positions, decaying with distance elsewhere."""
    occ = rasterize_obstacle_disks(positions_xy, radius_m, num_x, num_y, start_x, start_y, resolution)
    if sigma_expand > 0:
        occ = gaussian_filter(occ, sigma=sigma_expand)
    occupied = occ > 0.01
    if not occupied.any():
        return np.zeros((num_x, num_y), dtype=np.float32)
    clearance_cells = ndimage.distance_transform_edt(~occupied)
    clearance_m = clearance_cells * resolution
    return clearance_to_cost(clearance_m, decay_scale_m)


def build_dynamic_cost_volume(
    people: list[dict],
    num_x: int,
    num_y: int,
    start_x: float,
    start_y: float,
    resolution: float,
    future_horizon_s: float = 2.0,
    future_dt: float = 0.5,
    person_radius_m: float = 0.3,
    sigma_expand: float = 2.0,
    decay_scale_m: float = 0.5,
) -> np.ndarray:
    """Build a (T_future, num_x, num_y) ground-truth dynamic obstacle cost volume.

    Args:
        people: one dict per tracked person, each with keys "start", "target", "speed", "mode",
            "pos_t0" (position at the rollout's reference timestamp t0), "direction_t0".
        num_x, num_y, start_x, start_y, resolution: must match the rollout's underlying static
            scene cost map exactly (see CostMapConfig) so the dynamic volume aligns with it.
        future_horizon_s, future_dt: how far ahead / how finely to slice; T_future =
            round(future_horizon_s / future_dt).

    Returns:
        (T_future, num_x, num_y) float32 array; volume[t] is the cost grid at t0 + (t+1)*future_dt
        (i.e. slice 0 is the first future offset, NOT t0 itself).
    """
    t_future = round(future_horizon_s / future_dt)
    volume = np.zeros((t_future, num_x, num_y), dtype=np.float32)

    for t_idx in range(t_future):
        elapsed = (t_idx + 1) * future_dt
        positions = np.array(
            [
                person_position_at_time(
                    person["start"],
                    person["target"],
                    person["speed"],
                    person["mode"],
                    person["pos_t0"],
                    person["direction_t0"],
                    elapsed,
                )
                for person in people
            ]
        )
        volume[t_idx] = obstacle_slice_cost(
            positions,
            person_radius_m,
            num_x,
            num_y,
            start_x,
            start_y,
            resolution,
            sigma_expand,
            decay_scale_m,
        )

    return volume


# EoF
