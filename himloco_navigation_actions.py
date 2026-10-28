# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NavigationAction variant for a HIMLoco-trained low-level locomotion policy.

The stock NavigationAction (navigation_actions.py) calls its low-level policy with a single
current-frame observation (the "policy" observation group: lin_vel, ang_vel, gravity, commands,
joint_pos_rel, joint_vel_rel, previous actions, height_scan). A HIMLoco policy expects something
structurally different: a rolling HISTORY of 6 single-step observations (45 dims each, 270 total),
each step composed of ["commands", "ang_vel", "gravity_vec", "dof_pos", "dof_vel", "actions"] (no
lin_vel, no height_scan -- HIMLoco's own internal estimator infers velocity from the history
instead), in a fixed joint order (FL, FR, RL, RR -- each hip/thigh/calf) that generally does NOT
match whatever order the IsaacLab USD asset's Articulation happens to report joints in.

See /home/ashain/Desktop/config.yaml and /home/ashain/Desktop/env_config.json (the user's own
HIMLoco training/deployment config for this exact policy_3780_5.9.pt checkpoint) for where every
constant below comes from -- this is not guessed. The joint order is cross-referenced from
env_config.json's *named* default_joint_angles against config.yaml's positional default_dof_pos
array: default_dof_pos = [0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5] matches
FL(0.1,0.8,-1.5), FR(-0.1,0.8,-1.5), RL(0.1,1.0,-1.5), RR(-0.1,1.0,-1.5) exactly -- i.e. per-leg
contiguous blocks in FL/FR/RL/RR order. config.yaml's own `joint_mapping` array is NOT reused here
-- it was authored for a specific real-hardware/SDK deployment target's joint order, which has no
reason to match this IsaacLab USD asset's own order. Instead, the sim<->policy joint permutation
is built by matching joint NAMES at runtime (self._asset.joint_names), which is correct regardless
of either side's internal ordering.
"""

from __future__ import annotations

import os

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils import configclass

from .navigation_actions import NavigationAction, NavigationActionCfg

# Test G (2026-08-31 multi_floor stuck-at-spawn investigation): print the raw policy action /
# resulting joint targets each low-level decimation step. Env-var gated (not a cfg field) since
# this is the lowest layer of the stack -- convenient to flip on/off without touching env_cfg.
_DEBUG_PRINT_POLICY_ACTION = os.environ.get("VIPLANNER_DEBUG_PRINT_POLICY_ACTION", "0") == "1"

# the policy's own expected joint order -- confirmed against env_config.json's named
# default_joint_angles, see module docstring
HIMLOCO_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]  # fmt: skip

# from /home/ashain/Desktop/config.yaml's go2/himloco section
NUM_ONE_STEP_OBS = 45
HISTORY_LENGTH = 6  # observations_history: [0, 1, 2, 3, 4, 5], 0 = latest
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
COMMANDS_SCALE = (2.0, 2.0, 0.25)
CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0


class HimlocoNavigationAction(NavigationAction):
    """Identical to NavigationAction except for how the low-level policy is queried each
    low-level step: builds HIMLoco's 270-dim (6 x 45) observation history instead of using the
    generic single-frame "policy" observation group."""

    def __init__(self, cfg: HimlocoNavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        sim_joint_names = list(self._asset.joint_names)
        missing = set(HIMLOCO_JOINT_ORDER) - set(sim_joint_names)
        assert not missing, f"HIMLoco joint(s) not found on the sim asset: {missing} (asset has {sim_joint_names})"

        # policy_from_sim[i] = sim-order index of the joint that belongs at policy-order position
        # i -- i.e. sim_order_tensor[:, policy_from_sim] reorders a sim-order tensor into policy order
        self._policy_from_sim = torch.tensor(
            [sim_joint_names.index(name) for name in HIMLOCO_JOINT_ORDER], device=self.device, dtype=torch.long
        )
        # inverse permutation: sim_from_policy[i] = policy-order index of the joint at sim-order
        # position i -- policy_order_tensor[:, sim_from_policy] reorders back into sim order
        self._sim_from_policy = torch.tensor(
            [HIMLOCO_JOINT_ORDER.index(name) for name in sim_joint_names], device=self.device, dtype=torch.long
        )

        self._commands_scale = torch.tensor(COMMANDS_SCALE, device=self.device)
        self._obs_history = torch.zeros(self.num_envs, HISTORY_LENGTH, NUM_ONE_STEP_OBS, device=self.device)
        self._prev_policy_action = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._history_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._debug_print_counter = 0

        # Test G+ (2026-08-31 stair-climb A/B test): foot z-height/contact-force body index
        # mapping, precomputed once -- the contact sensor and the asset order bodies differently,
        # same cross-reference-by-name pattern terminations.illegal_obstacle_contact already uses.
        if _DEBUG_PRINT_POLICY_ACTION:
            contact_sensor = self._env.scene.sensors["contact_forces"]
            self._debug_foot_names = [n for n in contact_sensor.body_names if n.endswith("_foot")]
            self._debug_foot_sensor_idx = [contact_sensor.body_names.index(n) for n in self._debug_foot_names]
            self._debug_foot_asset_idx, _ = self._asset.find_bodies(self._debug_foot_names)

    def _compute_one_step_obs(self) -> torch.Tensor:
        commands = self._env.command_manager.get_command("vel_command") * self._commands_scale
        ang_vel = self._asset.data.root_ang_vel_b * ANG_VEL_SCALE
        gravity = self._asset.data.projected_gravity_b

        dof_pos_sim = self._asset.data.joint_pos - self._asset.data.default_joint_pos
        dof_pos = dof_pos_sim[:, self._policy_from_sim] * DOF_POS_SCALE

        dof_vel_sim = self._asset.data.joint_vel - self._asset.data.default_joint_vel
        dof_vel = dof_vel_sim[:, self._policy_from_sim] * DOF_VEL_SCALE

        obs = torch.cat([commands, ang_vel, gravity, dof_pos, dof_vel, self._prev_policy_action], dim=-1)
        return torch.clamp(obs, -CLIP_OBS, CLIP_OBS)

    def apply_actions(self):
        """Same decimation structure as NavigationAction.apply_actions, but queries the policy
        with a 270-dim observation history instead of the single-frame "policy" observation
        group."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._env.command_manager.compute(dt=self._low_level_step_dt)

            one_step_obs = self._compute_one_step_obs()
            # a brand-new episode has no real history yet -- fill every slot with the current
            # frame instead of leaving stale/zero history from a previous episode or all-zeros,
            # either of which would hand the estimator a nonsensical "nothing changed for 6
            # steps" or "robot was at rest with zero gravity" signal on its very first inference
            fresh = ~self._history_initialized
            if torch.any(fresh):
                self._obs_history[fresh] = one_step_obs[fresh].unsqueeze(1).expand(-1, HISTORY_LENGTH, -1)
                self._history_initialized[fresh] = True
            self._obs_history = torch.roll(self._obs_history, shifts=1, dims=1)
            self._obs_history[:, 0, :] = one_step_obs

            obs_history_flat = self._obs_history.reshape(self.num_envs, -1)
            policy_action = self.low_level_policy(obs_history_flat)
            policy_action = torch.clamp(policy_action, -CLIP_ACTIONS, CLIP_ACTIONS)
            self._prev_policy_action[:] = policy_action

            self._low_level_actions[:] = policy_action[:, self._sim_from_policy]
            self.low_level_action_term.process_actions(self._low_level_actions)

            if _DEBUG_PRINT_POLICY_ACTION and self._debug_print_counter % 20 == 0:
                joint_target = self.low_level_action_term.processed_actions[0]
                base_z = self._asset.data.root_pos_w[0, 2].item()
                foot_z = self._asset.data.body_pos_w[0, self._debug_foot_asset_idx, 2].tolist()
                contact_sensor = self._env.scene.sensors["contact_forces"]
                foot_force = torch.norm(contact_sensor.data.net_forces_w[0, self._debug_foot_sensor_idx], dim=-1).tolist()
                print(
                    f"[POLICY-DEBUG] commands={one_step_obs[0, :3].tolist()} base_z={base_z:.3f} "
                    f"foot_z{self._debug_foot_names}={[round(z, 3) for z in foot_z]} "
                    f"foot_force{self._debug_foot_names}={[round(f, 1) for f in foot_force]} "
                    f"policy_action(policy-order)={policy_action[0].tolist()} "
                    f"joint_target(sim-order, PD setpoint)={joint_target.tolist()}"
                )
                # 2026-09-08 obs-content sanity check (env var gated, separate from the print
                # above): dump every term's actual RAW value for env 0 -- checking magnitudes are
                # physically sane, not just that the term ORDER matches config.yaml. Slices match
                # _compute_one_step_obs' torch.cat order exactly: commands(3), ang_vel(3),
                # gravity(3), dof_pos(12), dof_vel(12), actions(12) = 45.
                if os.environ.get("VIPLANNER_DEBUG_PRINT_OBS_CONTENT", "0") == "1":
                    o = one_step_obs[0]
                    print(
                        f"[OBS-DEBUG] commands(scaled)={o[0:3].tolist()} "
                        f"ang_vel(scaled)={o[3:6].tolist()} "
                        f"gravity_vec={o[6:9].tolist()} "
                        f"dof_pos(policy-order,rel,scaled)={[round(v, 3) for v in o[9:21].tolist()]} "
                        f"dof_vel(policy-order,rel,scaled)={[round(v, 3) for v in o[21:33].tolist()]} "
                        f"prev_actions(policy-order)={[round(v, 3) for v in o[33:45].tolist()]} "
                        f"raw_root_ang_vel_b={self._asset.data.root_ang_vel_b[0].tolist()} "
                        f"raw_joint_pos(sim-order)={self._asset.data.joint_pos[0].tolist()} "
                        f"raw_default_joint_pos(sim-order)={self._asset.data.default_joint_pos[0].tolist()} "
                        f"sim_joint_names={list(self._asset.joint_names)}"
                    )
            self._debug_print_counter += 1

        self.low_level_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids=None):
        out = super().reset(env_ids) or {}
        if env_ids is None:
            self._history_initialized[:] = False
            self._prev_policy_action[:] = 0.0
        else:
            self._history_initialized[env_ids] = False
            self._prev_policy_action[env_ids] = 0.0
        return out


@configclass
class HimlocoNavigationActionCfg(NavigationActionCfg):
    class_type: type = HimlocoNavigationAction


# EoF
