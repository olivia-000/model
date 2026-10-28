# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

from viplanner.cost_maps import CostMapPCD

# visual-imperative-planning
from .traj_opt import TrajOpt

try:
    import pypose as pp  # only used for training
    import wandb  # only used for training
except (ModuleNotFoundError, ImportError):  # eval in issac sim  # TODO: check if all can be installed in Isaac Sim
    print("[Warning] pypose or wandb not found, only use for evaluation")

    class _WandbStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

    wandb = _WandbStub()


class TrajCost:
    debug = False

    def __init__(
        self,
        gpu_id: Optional[int] = 0,
        log_data: bool = False,
        w_obs: float = 0.25,
        w_height: float = 1.0,
        w_motion: float = 1.5,
        w_goal: float = 2.0,
        w_dyn: float = 1.0,
        nominal_speed: float = 1.0,
        dyn_future_dt: float = 0.5,
        obstalce_thread: float = 0.75,
        robot_width: float = 0.6,
        robot_max_moving_distance: float = 0.15,
    ) -> None:
        # init map and optimizer
        self.gpu_id = gpu_id
        self.cost_map: CostMapPCD = None
        self.opt = TrajOpt()
        self.is_map = False
        self.neg_reward: torch.Tensor = None

        # loss weights
        self.w_obs = w_obs
        self.w_height = w_height
        self.w_motion = w_motion
        self.w_goal = w_goal

        # dynamic-obstacle loss (Phase 3): waypoint arrival-time estimation + weight
        self.w_dyn = w_dyn
        self.nominal_speed = nominal_speed
        self.dyn_future_dt = dyn_future_dt

        # fear label threshold value
        self.obstalce_thread = obstalce_thread

        # footprint radius
        self.robot_width = robot_width
        self.robot_max_moving_distance = robot_max_moving_distance

        # logging
        self.log_data = log_data
        return

    @staticmethod
    def TransformPoints(odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(
            batch_size,
            num_p,
            device=points.device,
            requires_grad=points.requires_grad,
        )
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps

    def SetMap(self, root_path, map_name):
        self.cost_map = CostMapPCD.ReadTSDFMap(root_path, map_name, self.gpu_id)
        self.is_map = True

        # get negative reward of cost-map
        self.neg_reward = torch.zeros(7, device=self.cost_map.device)
        if self.cost_map.cfg.semantics:
            self.neg_reward[2] = self.cost_map.cfg.sem_cost_map.negative_reward

        return

    def CostofTraj(
        self,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear: torch.Tensor,
        log_step: int,
        ahead_dist: float,
        dataset: str = "train",
        dyn_cost_volume: Optional[torch.Tensor] = None,
    ):
        batch_size, num_p, _ = waypoints.shape

        assert self.is_map, "Map has to be set for cost calculation"
        world_ps = self.TransformPoints(odom, waypoints).tensor()

        # Obstacle loss
        oloss_M = self._compute_oloss(world_ps, batch_size)
        oloss = torch.mean(torch.sum(oloss_M, axis=1))

        # Terrian Height loss
        norm_inds, _ = self.cost_map.Pos2Ind(world_ps)
        height_grid = self.cost_map.ground_array.T.expand(batch_size, 1, -1, -1)
        hloss_M = (
            F.grid_sample(
                height_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        hloss_M = torch.abs(world_ps[:, :, 2] - odom[:, None, 2] - hloss_M).to(
            torch.float32
        )  # world_ps - odom to have them on the ground to be comparable to the height map
        hloss_M = torch.sum(hloss_M, axis=1)
        hloss = torch.mean(hloss_M)

        # Goal Cost - Control Cost
        gloss_M = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        # gloss = torch.mean(gloss_M)
        gloss = torch.mean(torch.log(gloss_M + 1.0))

        # Moving Loss - punish staying
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0 / (num_p - 1))
        desired_ds = torch.norm(desired_wp[:, 1:num_p, :] - desired_wp[:, 0 : num_p - 1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:num_p, :] - waypoints[:, 0 : num_p - 1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, axis=1)
        mloss = torch.mean(mloss)

        # Complete Trajectory Loss
        trajectory_loss = self.w_obs * oloss + self.w_height * hloss + self.w_motion * mloss + self.w_goal * gloss

        # Dynamic Obstacle Loss (Phase 3) -- None (default) keeps every existing static-only
        # scene/dataset call site completely unaffected.
        dyn_oloss = None
        if dyn_cost_volume is not None:
            dyn_oloss_M = self._compute_dyn_oloss(world_ps, dyn_cost_volume, wp_ds)
            dyn_oloss = torch.mean(torch.sum(dyn_oloss_M, axis=1))
            trajectory_loss = trajectory_loss + self.w_dyn * dyn_oloss

        # Fear labels
        goal_dists = torch.cumsum(wp_ds, dim=1, dtype=wp_ds.dtype)
        goal_dists = torch.vstack([goal_dists] * 3)
        floss_M = torch.clone(oloss_M)
        floss_M[goal_dists > ahead_dist] = 0.0
        fear_labels = torch.max(floss_M, 1, keepdim=True)[0]
        # fear_labels = nn.Sigmoid()(fear_labels-obstalce_thread)
        fear_labels = fear_labels > self.obstalce_thread + self.neg_reward[2]
        fear_labels = torch.any(fear_labels.reshape(3, batch_size).T, dim=1, keepdim=True).to(torch.float32)
        # Fear loss
        collision_probabilty_loss = nn.BCELoss()(fear, fear_labels.float())

        # log
        if self.log_data:
            try:
                wandb.log(
                    {f"Height Loss {dataset}": self.w_height * hloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Obstacle Loss {dataset}": self.w_obs * oloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Goal Loss {dataset}": self.w_goal * gloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Motion Loss {dataset}": self.w_motion * mloss},
                    step=log_step,
                )
                wandb.log(
                    {f"Trajectory Loss {dataset}": trajectory_loss},
                    step=log_step,
                )
                wandb.log(
                    {f"Collision Loss {dataset}": collision_probabilty_loss},
                    step=log_step,
                )
                if dyn_oloss is not None:
                    wandb.log(
                        {f"Dynamic Obstacle Loss {dataset}": self.w_dyn * dyn_oloss},
                        step=log_step,
                    )
            except:  # noqa: E722
                print("wandb log failed")

        # TODO: kinodynamics cost
        return collision_probabilty_loss + trajectory_loss

    def obs_cost_eval(self, odom: torch.Tensor, waypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Obstacle Loss for eval_sim_static script!

        Args:
            odom (torch.Tensor): Current odometry
            waypoints (torch.Tensor): waypoints in camera frame

        Returns:
            tuple: mean obstacle loss for each trajectory, max obstacle loss for each trajectory
        """
        assert self.is_map, "Map has to be loaded for evaluation"

        # compute obstacle loss
        world_ps = self.TransformPoints(odom, waypoints).tensor()
        oloss_M = self._compute_oloss(world_ps, waypoints.shape[0])
        # account for negative reward
        oloss_M = oloss_M - self.neg_reward[2]
        oloss_M[oloss_M < 0] = 0.0
        oloss_M = oloss_M.reshape(-1, waypoints.shape[0], oloss_M.shape[1])
        return torch.mean(oloss_M, axis=[0, 2]), torch.amax(oloss_M, dim=[0, 2])

    def cost_of_recorded_path(
        self,
        waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Cost of recorded path - for evaluation only

        Args:
            waypoints (torch.Tensor): Path coordinates in world frame
        """
        assert self.is_map, "Map has to be loaded for evaluation"
        oloss_M = self._compute_oloss(waypoints.unsqueeze(0), 1)
        return torch.max(oloss_M)

    def _inflate_waypoints(self, world_ps):
        """Triple each waypoint into front_right/center/front_left copies offset by the robot's
        footprint width, so obstacle cost is checked across the robot's width, not just its
        centerline. Factored out of _compute_oloss so _compute_dyn_oloss (Phase 3) can reuse the
        exact same inflation without duplicating it."""
        if world_ps.shape[1] == 1:  # special case when evaluating cost of a recorded path
            return world_ps

        # include robot dimension as square
        tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]  # get tangent vector
        # near-zero-length segments (e.g. the network predicting a near-stationary path right
        # next to an obstacle) would otherwise divide by ~0 and poison the loss with NaN/Inf.
        tangent = tangent / torch.clamp(torch.norm(tangent, dim=2, keepdim=True), min=1e-6)  # normalize normals vector
        normals = tangent[:, :, [1, 0]] * torch.tensor(
            [-1, 1], dtype=torch.float32, device=world_ps.device
        )  # get normal vector
        world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)  # duplicate points
        world_ps_inflated[:, :, 0:2] = torch.vstack(
            [
                # movement corners
                world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,  # front_right
                world_ps[:, :-1, 0:2],  # center
                world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,  # front_left
            ]
        )
        return world_ps_inflated

    def _compute_oloss(self, world_ps, batch_size):
        world_ps_inflated = self._inflate_waypoints(world_ps)
        norm_inds, cost_idx = self.cost_map.Pos2Ind(world_ps_inflated)

        # Obstacle Cost
        cost_grid = self.cost_map.cost_array.T.expand(world_ps_inflated.shape[0], 1, -1, -1)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        oloss_M = oloss_M.to(torch.float32)

        if self.debug:
            # add negative reward for cost-map
            world_ps_inflated = world_ps_inflated + self.neg_reward

            import numpy as np

            # indexes in the cost map
            start_xy = torch.tensor(
                [self.cost_map.cfg.x_start, self.cost_map.cfg.y_start],
                dtype=torch.float64,
                device=world_ps_inflated.device,
            ).expand(1, 1, -1)
            H = (world_ps_inflated[:, :, 0:2] - start_xy) / self.cost_map.cfg.general.resolution
            cost_values = self.cost_map.cost_array[
                H[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy().astype(np.int64),
                H[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy().astype(np.int64),
            ]

            import matplotlib.pyplot as plt

            _, (ax1, ax2, ax3) = plt.subplots(1, 3)
            sc1 = ax1.scatter(
                world_ps_inflated[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy(),
                world_ps_inflated[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy(),
                c=oloss_M[[0, batch_size, batch_size * 2]].reshape(-1).detach().cpu().numpy(),
                cmap="rainbow",
                vmin=0,
                vmax=torch.max(cost_grid).item(),
            )
            ax1.set_aspect("equal", adjustable="box")
            ax2.scatter(
                H[[0, batch_size, batch_size * 2], :, 0].reshape(-1).detach().cpu().numpy(),
                H[[0, batch_size, batch_size * 2], :, 1].reshape(-1).detach().cpu().numpy(),
                c=cost_values.cpu().numpy(),
                cmap="rainbow",
                vmin=0,
                vmax=torch.max(cost_grid).item(),
            )
            ax2.set_aspect("equal", adjustable="box")
            cost_array = self.cost_map.cost_array.cpu().numpy()
            max_cost = torch.max(self.cost_map.cost_array).item()
            scale_factor = [1.4, 1.8]
            for idx, run_idx in enumerate([0, batch_size, batch_size * 2]):
                _, cost_idx = self.cost_map.Pos2Ind(world_ps_inflated[run_idx, :, :].unsqueeze(0))
                cost_array[
                    cost_idx.to(torch.int32).cpu().numpy()[:, 0],
                    cost_idx.to(torch.int32).cpu().numpy()[:, 1],
                ] = (
                    max_cost * scale_factor[idx]
                )
            ax3.imshow(cost_array)

            plt.figure()
            plt.title("cost_map")
            plt.imshow(cost_array)

            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                world_ps_inflated[[0, batch_size, batch_size * 2], :, :3].reshape(-1, 3).detach().cpu().numpy()
            )
            pcd.colors = o3d.utility.Vector3dVector(
                sc1.to_rgba(oloss_M[[0, batch_size, batch_size * 2]].reshape(-1).detach().cpu().numpy())[:, :3]
            )
            # pcd.colors = o3d.utility.Vector3dVector(sc2.to_rgba(cost_values[0].cpu().numpy())[:, :3])
            o3d.visualization.draw_geometries([self.cost_map.pcd_tsdf, pcd])

        return oloss_M

    def _compute_dyn_oloss(self, world_ps, dyn_cost_volume, wp_ds):
        """Dynamic-obstacle cost (Phase 3): samples a per-sample, per-future-time-slice cost
        volume (see viplanner/cost_maps/dynamic_cost_map.py) at each waypoint's WORLD (x, y)
        position and its ESTIMATED ARRIVAL TIME (from cumulative path distance / nominal_speed),
        using a 5D/trilinear grid_sample -- the same robot-width inflation and Pos2Ind grid-index
        convention _compute_oloss uses for the static map, extended with a third (time) axis.

        Args:
            world_ps: waypoints in world frame, (batch, num_p, 3) -- same input _compute_oloss takes.
            dyn_cost_volume: (batch, T_future, num_x, num_y) cost volume built at the SAME grid
                resolution/origin as self.cost_map (see dynamic_cost_map.build_dynamic_cost_volume).
                Slice t (0-indexed) corresponds to t0 + (t+1)*dyn_future_dt seconds ahead.
            wp_ds: (batch, num_p-1) waypoint-to-waypoint world-frame distances (same tensor
                CostofTraj already computes for the motion/fear-label calculations).

        Returns:
            (3*batch, num_p-1) per-waypoint dynamic obstacle cost, matching _compute_oloss's
            output shape/layout (front_right/center/front_left stacked along dim 0).
        """
        world_ps_inflated = self._inflate_waypoints(world_ps)
        norm_inds_xy, _ = self.cost_map.Pos2Ind(world_ps_inflated)  # (3*batch, num_p-1, 2)

        t_future = dyn_cost_volume.shape[1]
        t_arrival = torch.cumsum(wp_ds, dim=1) / self.nominal_speed  # (batch, num_p-1), seconds
        # continuous index into the T_future slices: slice i's time is (i+1)*dyn_future_dt
        slice_coord = t_arrival / self.dyn_future_dt - 1.0
        slice_coord = torch.vstack([slice_coord] * 3)  # (3*batch, num_p-1), matches the width-inflation stacking
        t_norm = (slice_coord - t_future / 2.0) / (t_future / 2.0)

        # grid_sample's 5D convention maps grid[...,0]->W_in, grid[...,1]->H_in, grid[...,2]->D_in;
        # transpose (x,y)->(y,x) to match cost_array.T's existing convention for the static map, so
        # grid[...,0]=x_norm hits W_in=num_x and grid[...,1]=y_norm hits H_in=num_y.
        grid = torch.cat([norm_inds_xy, t_norm[:, :, None]], dim=-1)  # (3*batch, num_p-1, 3) = (x, y, t)
        grid = grid.to(torch.float32)[:, None, None, :, :]  # (3*batch, D_out=1, H_out=1, W_out=num_p-1, 3)

        vol = dyn_cost_volume.transpose(-1, -2)  # (batch, T_future, num_y, num_x)
        vol = torch.vstack([vol] * 3).unsqueeze(1).to(torch.float32)  # (3*batch, 1, T_future, num_y, num_x)

        dyn_oloss_M = F.grid_sample(
            vol,
            grid,
            mode="bilinear",  # grid_sample has no 5D "bicubic"; bilinear is trilinear for 5D input
            padding_mode="border",
            align_corners=False,
        )
        return dyn_oloss_M.reshape(world_ps_inflated.shape[0], -1).to(torch.float32)

    def egocentric_occupancy_ground_truth(
        self,
        odom: torch.Tensor,
        dyn_cost_volume: torch.Tensor,
        grid_size: int,
        grid_extent_m: float,
    ) -> torch.Tensor:
        """Resample the world-frame GT dynamic cost volume into a fixed egocentric (robot-centered)
        grid at the current `odom` pose -- the supervision target for
        DynamicDualAutoEncoder.future_occupancy_head's prediction (see
        viplanner/plannernet/autoencoder.py), which predicts in this same robot-centered
        convention so it stays independent of world position at inference time.

        Args:
            odom: (batch, 7) current pose (translation + quaternion), same convention TransformPoints
                already uses elsewhere in this file.
            dyn_cost_volume: (batch, T_future, num_x, num_y), same per-sample GT volume
                _compute_dyn_oloss consumes.
            grid_size: side length (in cells) of the egocentric output grid.
            grid_extent_m: the egocentric grid spans [-grid_extent_m, +grid_extent_m] in both x
                and y, centered on the robot.

        Returns:
            (batch, T_future, grid_size, grid_size) ground-truth occupancy in the egocentric frame.
        """
        batch_size = odom.shape[0]
        t_future = dyn_cost_volume.shape[1]
        device = odom.device

        lin = torch.linspace(-grid_extent_m, grid_extent_m, grid_size, device=device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(lin, lin, indexing="ij")
        local_points = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())], dim=1)
        local_points = local_points[None].expand(batch_size, -1, -1)  # (batch, grid_size^2, 3)

        world_points = self.TransformPoints(odom, local_points).tensor()[:, :, 0:3]
        norm_inds_xy, _ = self.cost_map.Pos2Ind(world_points)  # (batch, grid_size^2, 2)
        grid = norm_inds_xy.to(torch.float32)[:, None, :, :]  # (batch, 1, grid_size^2, 2)

        vol = dyn_cost_volume.transpose(-1, -2)  # (batch, T_future, num_y, num_x), matches _compute_dyn_oloss

        gt = torch.zeros(batch_size, t_future, grid_size, grid_size, device=device, dtype=torch.float32)
        for t in range(t_future):
            sampled = F.grid_sample(vol[:, t : t + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)
            gt[:, t] = sampled.reshape(batch_size, grid_size, grid_size)
        return gt


# EoF
