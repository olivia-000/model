# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge between a global/exploration policy (e.g. NoMaD) and ViPlanner's local goal interface.

See NOMAD_INTEGRATION.md at the repo root for the full architecture this fits into (Search &
Rescue mission -> frontier/victim goal -> NoMaD -> this bridge -> ViPlanner -> RL locomotion).

Kept dependency-free of Isaac Sim / omni.isaac.lab.utils.math (unlike viplanner_algo.py, which
runs inside the simulator) so it can be unit-tested and reused from a real-robot ROS node, not
just the Isaac Sim demo.
"""

from __future__ import annotations

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` (..., 3) by `quat` (..., 4, w-x-y-z order -- matches IsaacLab's convention).

    Standalone reimplementation of IsaacLab's omni.isaac.lab.utils.math.quat_apply (same formula,
    same convention) so this module has no Isaac Sim import dependency.
    """
    q_w = quat[..., 0:1]
    q_xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def nomad_output_to_world_goal(
    nomad_waypoints: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    lookahead_distance: float = 3.0,
    max_goal_distance: float = 10.0,
    goal_height: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Convert one NoMaD waypoint-chunk prediction into a world-frame goal point for ViPlanner.

    NoMaD predicts a chunk of H (dx, dy) waypoints in the robot's own base frame *at the moment
    of the observation it conditioned on* -- ground-plane-only (no z/pitch/roll), consistent with
    the wheeled-robot data the ViNT/NoMaD family is trained on. ViPlanner's own goal interface
    (VIPlannerAlgo.goal_transformer, see viplanner_algo.py) is unrelated to that: it wants a
    single WORLD-frame (x, y, z) point and does its own camera-frame transform internally
    (zeroing out the z difference before rotating -- goal height is a don't-care as long as it's
    finite). This function is the entire adapter between the two: no changes to ViPlanner's own
    code are needed, the output of this function is a drop-in replacement for the `goal_pos`
    world point already used in omniverse/standalone/viplanner_demo.py.

    Args:
        nomad_waypoints: (num_envs, H, 2) predicted (dx, dy) waypoints, robot base frame, meters,
            ordered near-to-far (index 0 is the nearest step in the prediction horizon).
        root_pos_w: (num_envs, 3) robot base world position, e.g. `robot.data.root_pos_w`.
        root_quat_w: (num_envs, 4) robot base world orientation, w-x-y-z, e.g.
            `robot.data.root_quat_w`. Same fields already published as the `/tf` "world"->"base"
            transform in viplanner_demo.py -- no separate pose source needed.
        lookahead_distance: how far along the predicted chunk (by cumulative arc length, not
            waypoint index) to place the subgoal. Deliberately short relative to what NoMaD
            actually predicts -- NoMaD's own chunk is the "big direction" signal, not a path to
            execute verbatim; ViPlanner re-does the actual fine-grained, obstacle-aware local
            planning between the robot and this point.
        max_goal_distance: hard clamp on the resulting goal's distance from the robot. ViPlanner's
            navigation network was trained with DataCfg.max_goal_distance=10.0 (see
            viplanner/config/learning_cfg.py); a goal further than that is out-of-distribution for
            the network, regardless of what NoMaD proposes.
        goal_height: world-frame z for the returned goal. ViPlanner discards this (see above), so
            any finite value works; defaults to the robot's own current height, matching the
            fixed-height convention already used for `goal_pos` in viplanner_demo.py.

    Returns:
        (num_envs, 3) world-frame goal points, ready to pass into VIPlannerAlgo.goal_transformer
        exactly like `goal_pos` in viplanner_demo.py.
    """
    num_envs, horizon, _ = nomad_waypoints.shape
    device = nomad_waypoints.device

    # cumulative arc length along the chunk, from the robot's current position (origin in base
    # frame) through each successive waypoint
    segment_start = torch.cat(
        [torch.zeros(num_envs, 1, 2, device=device), nomad_waypoints[:, :-1, :]], dim=1
    )
    segment_len = torch.norm(nomad_waypoints - segment_start, dim=-1)  # (num_envs, H)
    cum_len = torch.cumsum(segment_len, dim=-1)  # (num_envs, H)

    # for each env, find the lookahead point by linear interpolation along the chunk; if the
    # entire chunk is shorter than lookahead_distance, fall back to its last (furthest) waypoint
    # rather than extrapolating past what NoMaD actually predicted
    goal_local = torch.zeros(num_envs, 2, device=device)
    for env_idx in range(num_envs):
        lengths = cum_len[env_idx]
        beyond = (lengths >= lookahead_distance).nonzero(as_tuple=True)[0]
        if beyond.numel() == 0:
            goal_local[env_idx] = nomad_waypoints[env_idx, -1]
            continue
        end_idx = int(beyond[0])
        end_len = float(lengths[end_idx])
        start_pt = nomad_waypoints[env_idx, end_idx - 1] if end_idx > 0 else torch.zeros(2, device=device)
        start_len = float(lengths[end_idx - 1]) if end_idx > 0 else 0.0
        span = end_len - start_len
        # span==0 means a zero-length segment sits exactly at the lookahead boundary; take the
        # segment's end point rather than dividing by zero
        frac = 0.0 if span <= 1e-9 else (lookahead_distance - start_len) / span
        goal_local[env_idx] = start_pt + frac * (nomad_waypoints[env_idx, end_idx] - start_pt)

    # clamp to ViPlanner's trained goal-distance range without changing direction
    dist = torch.norm(goal_local, dim=-1, keepdim=True).clamp(min=1e-6)
    scale = torch.clamp(max_goal_distance / dist, max=1.0)
    goal_local = goal_local * scale

    # base frame (planar) -> world frame, via the robot's current pose
    goal_local_3d = torch.cat([goal_local, torch.zeros(num_envs, 1, device=device)], dim=-1)
    goal_world_xyz = _quat_apply(root_quat_w, goal_local_3d) + root_pos_w

    if goal_height is None:
        goal_world_xyz[:, 2] = root_pos_w[:, 2]
    else:
        goal_world_xyz[:, 2] = goal_height

    return goal_world_xyz


def should_replan_subgoal(
    root_pos_w: torch.Tensor,
    current_subgoal_w: torch.Tensor,
    replan_distance: float = 1.0,
) -> torch.Tensor:
    """Per-env bool: is the robot close enough to its current NoMaD-derived subgoal to fetch a new one?

    NoMaD is the "big direction" layer and shouldn't be re-queried every ViPlanner step (wasteful,
    and its own prediction horizon is short); re-querying only once the robot has nearly reached
    the current subgoal keeps it running at its own, slower cadence while ViPlanner keeps
    replanning locally every step as it already does.
    """
    dist = torch.norm(root_pos_w[:, :2] - current_subgoal_w[:, :2], dim=-1)
    return dist < replan_distance
