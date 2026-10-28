# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-time obstacle/traversable classification from depth alone -- no per-scene semantic labels
(no keyword_mapping.yml), no learned segmentation model, works in any unseen environment because
it never asks "what is this", only "does its height say I can walk here".

Reuses the exact ground_height/robot_height/robot_height_factor height convention
frontier_map.OnlineOccupancyGrid and cost_maps.tsdf_cost_map.TsdfCostMapConfig already use for the
same "floor vs obstacle" judgment call, just vectorized over every pixel of a full depth image
(torch, GPU) instead of raycasting a sparse sampled set of points into a dict -- so it's cheap
enough to run every single frame, unlike frontier_map.update()'s pure-Python per-point loop (see
sar_demo.py's --frontier_update_interval, added because that one *isn't* cheap every frame).

Outputs an RGB image using the exact two colors ViPlanner's own training taxonomy
(viplanner_sem_meta.VIPLANNER_SEM_META) already assigns: "floor" (traversable, loss=0) and
"static" -- literally documented there as "also everything unknown" (loss=OBSTACLE_LOSS). That
second class is the reason this drops in without retraining anything: the network was already
trained to treat unrecognized things as obstacles, which is exactly the safe default a detector
that "doesn't need to know what it is" wants. Runs directly on the depth_camera image ViPlanner
already consumes, so there's no cross-camera registration to get right (unlike the ground-truth
semantic_camera path this replaces, which is a different resolution/FOV -- see
victim_detection.py's docstring for that same, separate, known gap).
"""

import torch

# viplanner_sem_meta.py's exact RGB values -- keep these in sync if that file ever changes them
FLOOR_COLOR = (0, 204, 0)  # "floor", TRAVERSABLE_INTENDED_LOSS
OBSTACLE_COLOR = (0, 0, 0)  # "static" ("also everything unknown"), OBSTACLE_LOSS


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` (..., 3) by `quat` (broadcastable to (..., 4), w-x-y-z). Same formula as
    frontier_map._quat_apply / nomad_bridge._quat_apply / victim_detection._quat_apply --
    duplicated rather than imported, see those modules' docstrings for why (keeps each utility
    independently usable; it's five lines). Unlike those (one quat per one vec), here a single
    camera-pose quat is applied to a whole (H, W, 3) image of points, so quat is broadcast up to
    vec's shape first -- torch.cross requires matching, not just broadcastable, shapes."""
    quat = quat.reshape((1,) * (vec.dim() - 1) + (4,)).expand(*vec.shape[:-1], 4)
    q_w = quat[..., 0:1]
    q_xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def compute_geometric_semantic(
    depth_image: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    ground_height: float = 0.35,
    robot_height: float = 0.70,
    robot_height_factor: float = 2.0,
    max_range: float = 8.0,
) -> torch.Tensor:
    """depth_image: (H, W) in meters, same camera depth_camera's fx/fy/cx/cy describe.
    cam_pos_w: (3,), cam_quat_w: (4,, w-x-y-z) -- same fields viplanner_demo.py/sar_demo.py
    already read from obs["planner_transform"] every step.

    Returns (H, W, 3) uint8 RGB, ViPlanner's own color convention -- pass straight in as
    VIPlannerAlgo.plan_dual's sem_image (it does `self.transform(sem_image) / 255` on whatever it
    gets, no different from the ground-truth semantic_camera output it otherwise expects).

    Per-pixel rule, identical cutoffs to frontier_map.OnlineOccupancyGrid.update(): world_z below
    ground_height is FREE/floor; between ground_height and robot_height*robot_height_factor is an
    OCCUPIED obstacle; invalid depth, beyond max_range, or above the ceiling cutoff all fall back
    to the same obstacle color -- "don't know" must never be silently walkable.
    """
    h, w = depth_image.shape
    device = depth_image.device
    vv, uu = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    z = depth_image.float()
    valid = torch.isfinite(z) & (z > 0) & (z <= max_range)

    x_cam = (uu.float() - cx) * z / fx
    y_cam = (vv.float() - cy) * z / fy
    points_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (H, W, 3)

    points_world = _quat_apply(cam_quat_w, points_cam) + cam_pos_w
    world_z = points_world[..., 2]

    ground = valid & (world_z < ground_height)

    out = torch.empty((h, w, 3), dtype=torch.uint8, device=device)
    out[..., 0], out[..., 1], out[..., 2] = OBSTACLE_COLOR
    floor_color = torch.tensor(FLOOR_COLOR, dtype=torch.uint8, device=device)
    out[ground] = floor_color
    return out
