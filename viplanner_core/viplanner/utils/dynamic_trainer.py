# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trainer subclass for Phase 3 (dynamic obstacle awareness). Overrides only the data/loss
plumbing needed for the extra depth-history/motion-cue input and dyn_cost_volume tensor --
train()/_load_model/_configure_optimizer/save_config are inherited from Trainer unchanged. Does
not modify Trainer itself, so every existing static-scene training path (train.py,
checkpoints/model.pt) is completely unaffected.

Does NOT support hierarchical training or the _test_epoch visualization path (TrajViz expects a
single-frame depth image, not a depth-history stack) -- both are out of scope for this
verification-run phase; a plain train/val loop with loss logging is all that's needed to confirm
the new loss terms decrease.
"""

from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data as Data
import tqdm

try:
    import wandb
except ImportError:

    class _WandbStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

    wandb = _WandbStub()

from viplanner.config import TrainCfg
from viplanner.plannernet.autoencoder import DynamicDualAutoEncoder
from viplanner.traj_cost_opt import TrajCost
from viplanner.utils.checkpoint_surgery import load_pretrained_into_dynamic
from viplanner.utils.dynamic_dataset import DynamicPlannerData
from viplanner.utils.trainer import Trainer

from .dataset import PlannerDataGenerator


class DynamicTrainer(Trainer):
    def __init__(
        self,
        cfg: TrainCfg,
        dynamic_envs: Set[str],
        num_depth_history: int = 4,
        t_future: int = 4,
        occupancy_grid_size: int = 32,
        occupancy_grid_extent_m: float = 5.0,
        motion_channels: int = 64,
        w_occ: float = 1.0,
    ) -> None:
        """Args:
            dynamic_envs: the subset of cfg.env_list that are rollout-based dynamic environments
                (have rollouts/ + rollout_manifest.json, e.g. "warehouse_dynamic"). Every other env
                in cfg.env_list is treated as an ordinary static anti-forgetting environment (e.g.
                "warehouse", "warehouse_official") -- see DynamicPlannerData's is_static mode.
        """
        super().__init__(cfg)
        self.dynamic_envs = dynamic_envs
        self.num_depth_history = num_depth_history
        self.t_future = t_future
        self.occupancy_grid_size = occupancy_grid_size
        self.occupancy_grid_extent_m = occupancy_grid_extent_m
        self.motion_channels = motion_channels
        self.w_occ = w_occ

        self.env_is_static: List[bool] = []
        self.rollouts_roots: List[Optional[str]] = []
        self.rollout_manifest_paths: List[Optional[str]] = []

    def load_pretrained_checkpoint(self, old_model_pt: str) -> None:
        """Fine-tune entrypoint: load a static DualAutoEncoder checkpoint's weights into
        self.net (must already be constructed via _load_model(resume=False))."""
        load_pretrained_into_dynamic(self.net, old_model_pt, motion_channels=self.motion_channels)

    # Helper function DATA (overrides Trainer._load_data: identical env/TrajCost/generator setup,
    # plus bookkeeping each env's rollouts_root/rollout_manifest_path for _get_dataloader)
    def _load_data(self, train: bool = True) -> None:
        if not isinstance(self._cfg.data_cfg, list):
            self._cfg.data_cfg = [self._cfg.data_cfg] * len(self._cfg.env_list)
        assert len(self._cfg.data_cfg) == len(
            self._cfg.env_list
        ), "Either single DataCfg or number matching number of environments must be provided"

        for idx, env_name in enumerate(self._cfg.env_list):
            if (train and idx == self._cfg.test_env_id) or (not train and idx != self._cfg.test_env_id):
                continue

            data_path = os.path.join(self._cfg.data_dir, env_name)

            traj_cost = TrajCost(
                self._cfg.gpu_id,
                log_data=train,
                w_obs=self._cfg.w_obs,
                w_height=self._cfg.w_height,
                w_goal=self._cfg.w_goal,
                w_motion=self._cfg.w_motion,
                w_dyn=getattr(self._cfg, "w_dyn", 1.0),
                nominal_speed=getattr(self._cfg, "nominal_speed", 1.0),
                dyn_future_dt=getattr(self._cfg, "dyn_future_dt", 0.5),
                obstalce_thread=self._cfg.obstacle_thread,
                robot_width=self._cfg.robot_width,
            )
            traj_cost.SetMap(data_path, self._cfg.cost_map_name)

            generator = PlannerDataGenerator(
                cfg=self._cfg.data_cfg[idx],
                root=data_path,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                cost_map=traj_cost.cost_map,
            )

            is_dynamic = env_name in self.dynamic_envs
            self.data_generators.append(generator)
            self.data_traj_cost.append(traj_cost)
            self.env_is_static.append(not is_dynamic)
            self.rollouts_roots.append(os.path.join(data_path, "rollouts") if is_dynamic else None)
            self.rollout_manifest_paths.append(
                os.path.join(data_path, "rollout_manifest.json") if is_dynamic else None
            )
            print(f"LOADED DATA FOR ENVIRONMENT: {env_name} ({'dynamic' if is_dynamic else 'static'})")

        print("[INFO] LOADED ALL DATA")
        return

    def _get_dataloader(
        self,
        train: bool = True,
        step: Optional[int] = None,
        allow_augmentation: bool = True,
    ):
        train_loader_list: List[Data.DataLoader] = []
        val_loader_list: List[Data.DataLoader] = []

        for i, generator in enumerate(self.data_generators):
            common_kwargs = dict(
                cfg=generator._cfg,
                transform=self.transform,
                rollouts_root=self.rollouts_roots[i],
                rollout_manifest_path=self.rollout_manifest_paths[i],
                img_input_size=self._cfg.img_input_size,
                num_depth_history=self.num_depth_history,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
                is_static=self.env_is_static[i],
                t_future=self.t_future,
            )
            val_data = DynamicPlannerData(**common_kwargs)
            train_data = DynamicPlannerData(**common_kwargs) if train else None

            generator.split_samples(
                train_dataset=train_data,
                test_dataset=val_data,
                generate_split=train,
                ratio_back_samples=self.back_ratio,
                ratio_front_samples=self.front_ratio,
                ratio_fov_samples=self.fov_ratio,
                allow_augmentation=allow_augmentation,
            )

            if self._cfg.load_in_ram:
                if train:
                    train_data.load_data_in_memory()
                val_data.load_data_in_memory()

            if train:
                train_loader_list.append(
                    Data.DataLoader(
                        dataset=train_data,
                        batch_size=self._cfg.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=self._cfg.num_workers,
                    )
                )
            val_loader_list.append(
                Data.DataLoader(
                    dataset=val_data,
                    batch_size=self._cfg.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=self._cfg.num_workers,
                )
            )

        return train_loader_list, val_loader_list

    def _train_step(self, inputs, env_id: int, log_step: int, dataset: str = "train") -> float:
        """One gradient-update step on a single batch from env `env_id`. Factored out of
        _train_epoch so the dynamic-centered replay loop (train_dynamic_centered_epoch) can drive
        BOTH the dynamic-env batches and the static-replay batches through the identical
        forward/backward/optimizer-step path, batch-by-batch, without either env needing its own
        full-pass loop."""
        odom = inputs[2].cuda(self._cfg.gpu_id)
        goal = inputs[3].cuda(self._cfg.gpu_id)
        dyn_cost_volume = inputs[5].cuda(self._cfg.gpu_id)
        self.optimizer.zero_grad()

        depth_history = inputs[0].cuda(self._cfg.gpu_id)
        sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
        preds, fear, predicted_occupancy = self.net(depth_history, sem_rgb_image, goal)

        preds_flip = torch.clone(preds)
        preds_flip[inputs[4], :, 1] = preds_flip[inputs[4], :, 1] * -1
        goal_flip = torch.clone(goal)
        goal_flip[inputs[4], 1] = goal_flip[inputs[4], 1] * -1

        loss, _ = self._loss(
            preds_flip,
            fear,
            predicted_occupancy,
            self.data_traj_cost[env_id],
            odom,
            goal_flip,
            dyn_cost_volume,
            log_step=log_step,
            dataset=dataset,
        )
        wandb.log({f"{dataset}_loss_step": loss}, step=log_step)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epoch(self, loader: Data.DataLoader, epoch: int, env_id: int) -> float:
        """Full pass over one env's entire train loader. Still used for pure single/uncapped-env
        training, but NOT for the dynamic+static mix (see train_dynamic_centered_epoch): looping
        every static env's full (much larger) loader every epoch would make gradient updates
        overwhelmingly dominated by static-scene batches, drowning out the dynamic learning signal
        that is the actual point of this fine-tune."""
        train_loss, batches = 0, len(loader)
        enumerater = tqdm.tqdm(enumerate(loader))

        for batch_idx, inputs in enumerater:
            log_step = batch_idx + epoch * batches
            loss = self._train_step(inputs, env_id, log_step)
            train_loss += loss
            enumerater.set_description(
                f"Epoch: {epoch} in Env: ({env_id+1}/{len(self._cfg.env_list)-1}) "
                f"- train loss:{round(train_loss/(batch_idx+1), 4)} on {batch_idx}/{batches}"
            )
        return train_loss / (batch_idx + 1)

    def train_dynamic_centered_epoch(
        self,
        dynamic_loader: Data.DataLoader,
        dynamic_env_id: int,
        static_loaders: List[Data.DataLoader],
        static_env_ids: List[int],
        replay_ratio: float,
        epoch: int,
    ) -> Tuple[float, float]:
        """Dynamic-centered balanced replay: epoch length is driven by the DYNAMIC env's own
        loader (its full pass, once), with `replay_ratio` static-env replay batches interleaved
        per dynamic batch (round-robining across static_env_ids so each replay batch stays
        entirely within one scene's own cost map -- CostofTraj/Pos2Ind need a single consistent
        TrajCost per batch, so replay batches are never a mix of samples from different scenes).

        replay_ratio may be fractional (e.g. 0.5 -> one replay batch every other dynamic batch);
        a running accumulator fires a replay step whenever it crosses 1.0, so both replay_ratio<1
        and >1 are handled smoothly without rounding bias.

        Returns (mean_dynamic_loss, mean_static_replay_loss) separately -- averaging them together
        would reintroduce the same "large dataset dominates the reported number" issue this method
        exists to avoid on the gradient side.
        """
        static_iters = [iter(loader) for loader in static_loaders]
        dynamic_loss_sum, dynamic_steps = 0.0, 0
        static_loss_sum, static_steps = 0.0, 0
        replay_accumulator = 0.0
        num_dynamic_batches = len(dynamic_loader)

        enumerater = tqdm.tqdm(enumerate(dynamic_loader))
        for batch_idx, dyn_inputs in enumerater:
            log_step = batch_idx + epoch * num_dynamic_batches
            dynamic_loss_sum += self._train_step(dyn_inputs, dynamic_env_id, log_step)
            dynamic_steps += 1

            replay_accumulator += replay_ratio
            while replay_accumulator >= 1.0 and static_loaders:
                replay_accumulator -= 1.0
                static_slot = static_steps % len(static_loaders)
                try:
                    static_inputs = next(static_iters[static_slot])
                except StopIteration:
                    static_iters[static_slot] = iter(static_loaders[static_slot])
                    static_inputs = next(static_iters[static_slot])
                static_loss_sum += self._train_step(static_inputs, static_env_ids[static_slot], log_step, dataset="static_replay")
                static_steps += 1

            enumerater.set_description(
                f"Epoch: {epoch} [dynamic-centered] "
                f"dyn_loss:{round(dynamic_loss_sum/dynamic_steps, 4)} "
                f"static_replay_loss:{round(static_loss_sum/max(static_steps,1), 4)} "
                f"({dynamic_steps} dyn / {static_steps} replay steps)"
            )

        mean_dynamic_loss = dynamic_loss_sum / max(dynamic_steps, 1)
        mean_static_loss = static_loss_sum / max(static_steps, 1)
        return mean_dynamic_loss, mean_static_loss

    def _test_epoch(
        self,
        loader,
        env_id: int,
        epoch: int = 0,
        is_visual: bool = False,
        fov_angle: float = 90.0,
        dataset: str = "val",
    ) -> float:
        test_loss = 0
        num_batches = len(loader)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(loader):
                odom = inputs[2].cuda(self._cfg.gpu_id)
                goal = inputs[3].cuda(self._cfg.gpu_id)
                dyn_cost_volume = inputs[5].cuda(self._cfg.gpu_id)

                depth_history = inputs[0].cuda(self._cfg.gpu_id)
                sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
                preds, fear, predicted_occupancy = self.net(depth_history, sem_rgb_image, goal)

                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1

                log_step = epoch * num_batches + batch_idx
                loss, _ = self._loss(
                    preds,
                    fear,
                    predicted_occupancy,
                    self.data_traj_cost[env_id],
                    odom,
                    goal,
                    dyn_cost_volume,
                    log_step=log_step,
                    dataset=dataset,
                )
                if dataset == "val":
                    wandb.log({f"{dataset}_loss_step": loss}, step=log_step)
                test_loss += loss.item()

        return test_loss / (batch_idx + 1)

    def _loss(
        self,
        preds: torch.Tensor,
        fear: torch.Tensor,
        predicted_occupancy: torch.Tensor,
        traj_cost: TrajCost,
        odom: torch.Tensor,
        goal: torch.Tensor,
        dyn_cost_volume: torch.Tensor,
        log_step: int,
        step: float = 0.1,
        dataset: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
        loss = traj_cost.CostofTraj(
            waypoints,
            odom,
            goal,
            fear,
            log_step,
            ahead_dist=self._cfg.fear_ahead_dist,
            dataset=dataset,
            dyn_cost_volume=dyn_cost_volume,
        )

        occupancy_gt = traj_cost.egocentric_occupancy_ground_truth(
            odom, dyn_cost_volume, self.occupancy_grid_size, self.occupancy_grid_extent_m
        )
        occupancy_loss = F.mse_loss(predicted_occupancy, occupancy_gt)
        loss = loss + self.w_occ * occupancy_loss

        try:
            wandb.log({f"Occupancy Prediction Loss {dataset}": self.w_occ * occupancy_loss}, step=log_step)
        except:  # noqa: E722
            print("wandb log failed")

        return loss, waypoints

    def _load_model(self, resume: bool = False) -> None:
        """Overrides Trainer._load_model to construct DynamicDualAutoEncoder instead of
        DualAutoEncoder/AutoEncoder. `resume` here means "resume a DYNAMIC checkpoint" (produced
        by a previous run of this class) -- loading the ORIGINAL static checkpoint for fine-tuning
        is a separate, explicit step via load_pretrained_checkpoint(), called after this method."""
        self.net = DynamicDualAutoEncoder(
            self._cfg,
            num_history=self.num_depth_history,
            t_future=self.t_future,
            occupancy_grid_size=self.occupancy_grid_size,
            motion_channels=self.motion_channels,
        )

        assert torch.cuda.is_available(), "Code requires GPU"
        self.net = self.net.cuda(self._cfg.gpu_id)
        print(f"[INFO] DYNAMIC MODEL LOADED")

        if resume:
            model_state_dict, self.best_loss = torch.load(self.model_path)
            self.net.load_state_dict(model_state_dict)
            print(f"Resume train from {self.model_path} with loss {self.best_loss}")

        return


# EoF
