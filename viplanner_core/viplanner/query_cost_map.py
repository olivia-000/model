# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Query the built static cost map at specific world (x, y) points.

Diagnostic for the multi_floor staircase not being climbed: checks whether the cost map treats the
stair centerline as high-cost (near-obstacle) compared to the side walls, which would point at the
cost map / semantic labeling rather than locomotion or goal/trajectory optimization as the root
cause. Runs entirely on CPU (gpu_id=None) so it doesn't contend with a GPU-bound training run.
"""

import argparse
import os

import numpy as np
import torch

from viplanner.cost_maps.cost_to_pcd import CostMapPCD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a built cost map at specific (x, y) points.")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g. 'multi_floor').")
    parser.add_argument("--map", type=str, default="cost_map_geom", help="Cost map name.")
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        action="append",
        required=True,
        metavar=("X", "Y"),
        help="World (x, y) point to query; repeat --point for multiple points.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_path = os.path.join(repo_root, "training_data", "data", args.env)

    cost_map = CostMapPCD.ReadTSDFMap(root_path, args.map, gpu_id=None)
    print(f"[INFO] Loaded {args.map} for {args.env}: cost_array shape {tuple(cost_map.cost_array.shape)}")
    print(f"[INFO] x_start={cost_map.cfg.x_start:.3f}, y_start={cost_map.cfg.y_start:.3f}, " f"resolution={cost_map.cfg.general.resolution}")

    # compute raw grid indices directly (matches CostMapPCD.Pos2Ind's own formula) instead of using
    # Pos2Ind itself, since it silently drops out-of-bounds points from its second return value --
    # that would desync the (point, index) pairing below for exactly the points we most want to see.
    start_xy = np.array([cost_map.cfg.x_start, cost_map.cfg.y_start])
    raw_inds = (np.array(args.point) - start_xy) / cost_map.cfg.general.resolution
    raw_inds = np.round(raw_inds).astype(int)

    for (x, y), (ix, iy) in zip(args.point, raw_inds.tolist()):
        in_bounds = 0 <= ix < cost_map.num_x and 0 <= iy < cost_map.num_y
        cost = cost_map.cost_array[ix, iy].item() if in_bounds else float("nan")
        ground = cost_map.ground_array[ix, iy].item() if in_bounds else float("nan")
        print(
            f"  ({x:+.2f}, {y:+.2f}) -> grid idx ({ix}, {iy}){'  [OUT OF BOUNDS]' if not in_bounds else ''}: "
            f"cost={cost:.4f}, ground_height={ground:.4f}"
        )

# EoF
