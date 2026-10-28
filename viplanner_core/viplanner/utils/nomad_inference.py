# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inference wrapper around NoMaD (github.com/robodhruv/visualnav-transformer, vendored as a
submodule at viplanner/third_party/visualnav-transformer), producing the waypoint-chunk format
viplanner/utils/nomad_bridge.py's nomad_output_to_world_goal() expects.

See NOMAD_INTEGRATION.md for how this fits into the overall SAR architecture.

Deliberately mirrors the diffusion sampling loop in the reference repo's own
deployment/src/explore.py line-for-line (context queue -> vision encoder -> DDPM denoising ->
cumulative-waypoint decoding -> deployment-robot velocity rescale) rather than reimplementing
NoMaD from the paper description, so behavior matches the maintained reference implementation
exactly. The one deliberate departure: `predict_waypoints` averages across all sampled diffusion
trajectories instead of sending sample 0 like the live demo does -- there's no human in the loop
here to react to a single bad sample, so averaging gives a steadier estimate for an autonomous
consumer (nomad_bridge.py's lookahead-along-arclength logic already smooths over the chunk shape
either way).

Setup (see NOMAD_INTEGRATION.md for the full checkpoint-download story):
    pip install -e viplanner/third_party/visualnav-transformer/train/   # registers `vint_train`
    pip install diffusers einops efficientnet_pytorch vit_pytorch
`diffusion_policy` (for ConditionalUnet1D) is intentionally NOT pip-installed -- its own
setup.py pulls in a heavy, unrelated dependency chain (robomimic, robosuite, ...) for training
its own manipulation policies, none of which this wrapper needs. Only
`viplanner/third_party/diffusion_policy` is added to sys.path below, importing the one module
tree actually used.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

_VIPLANNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIFFUSION_POLICY_DIR = os.path.join(_VIPLANNER_DIR, "third_party", "diffusion_policy")
if _DIFFUSION_POLICY_DIR not in sys.path:
    sys.path.insert(0, _DIFFUSION_POLICY_DIR)

DEFAULT_CHECKPOINT_PATH = os.path.join(
    _VIPLANNER_DIR, "third_party", "visualnav-transformer", "deployment", "model_weights", "checkpoints", "nomad.pth"
)
DEFAULT_CONFIG_PATH = os.path.join(_VIPLANNER_DIR, "third_party", "visualnav-transformer", "train", "config", "nomad.yaml")

# Fixed dataset-wide per-step (dx, dy) delta range the model was trained to predict into, from
# vint_train/data/data_config.yaml's `action_stats` -- copied here rather than imported from
# vint_train.training.train_utils, which pulls in `wandb` (a training-only, not inference,
# dependency) just to read this one constant.
_ACTION_STATS_MIN = np.array([-2.5, -4.0])
_ACTION_STATS_MAX = np.array([5.0, 4.0])


def _get_cumulative_waypoints(diffusion_output: torch.Tensor) -> torch.Tensor:
    """Denoised diffusion output -> cumulative (dx, dy) waypoints, in the training data's own
    canonical (embodiment-agnostic) meters. Same math as the reference repo's
    vint_train.training.train_utils.get_action -- reproduced locally, see module docstring."""
    deltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2).detach().cpu().numpy()
    deltas = (deltas + 1) / 2 * (_ACTION_STATS_MAX - _ACTION_STATS_MIN) + _ACTION_STATS_MIN
    return torch.from_numpy(np.cumsum(deltas, axis=1)).to(diffusion_output.device)


class NomadPolicy:
    """Loads a NoMaD checkpoint and turns a rolling RGB context into a waypoint-chunk prediction.

    Undirected-exploration mode only (goal image masked out) -- goal-conditioned mode (steering
    toward a victim detection or topological subgoal image) is not implemented yet; see
    NOMAD_INTEGRATION.md's frontier-planning section for where that would plug in.
    """

    def __init__(
        self,
        max_v: float,
        frame_rate: float,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        device: str | None = None,
    ):
        """
        Args:
            max_v: deployment robot's max linear speed (m/s) -- Go2's, not the LoCoBot values in
                the reference repo's deployment/config/robot.yaml. Required, not defaulted: NoMaD's
                raw output is normalized relative to whatever robot it's deployed on (see
                `_rescale_to_robot` below), silently using the wrong value produces a
                plausible-looking but wrong-scale goal with no error to catch it.
            frame_rate: rate (Hz) `update_context` is called at.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        if self.config["model_type"] != "nomad":
            raise ValueError(f"{config_path} is not a NoMaD config (model_type={self.config['model_type']!r})")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.context_size: int = self.config["context_size"]
        self.len_traj_pred: int = self.config["len_traj_pred"]
        self.image_size: list[int] = self.config["image_size"]  # [width, height]
        self.max_v = max_v
        self.frame_rate = frame_rate
        self._normalize = self.config.get("normalize", True)

        self.model = self._build_model(checkpoint_path).to(self.device).eval()

        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.num_diffusion_iters: int = self.config["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.context_queue: list[Image.Image] = []

    def _build_model(self, checkpoint_path: str) -> nn.Module:
        # imported here, not at module level, so importing this file doesn't require vint_train /
        # diffusion_policy to be installed unless a NomadPolicy is actually constructed
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        cfg = self.config
        if cfg["vision_encoder"] != "nomad_vint":
            raise NotImplementedError(f"vision_encoder={cfg['vision_encoder']!r} not supported (only nomad_vint)")
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=cfg["encoding_size"],
            context_size=cfg["context_size"],
            mha_num_attention_heads=cfg["mha_num_attention_heads"],
            mha_num_attention_layers=cfg["mha_num_attention_layers"],
            mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=cfg["encoding_size"],
            down_dims=cfg["down_dims"],
            cond_predict_scale=cfg["cond_predict_scale"],
        )
        dist_pred_net = DenseNetwork(embedding_dim=cfg["encoding_size"])
        model = NoMaD(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net, dist_pred_net=dist_pred_net)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"NoMaD checkpoint not found at {checkpoint_path} -- see NOMAD_INTEGRATION.md for the download step"
            )
        # the released nomad.pth is a raw state_dict (unlike gnm/vint's checkpoints, which wrap it
        # in a {"model": ...} dict for DataParallel compatibility -- see the reference repo's
        # deployment/src/utils.py load_model for the same branch)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        return model

    def reset_context(self) -> None:
        """Call on episode reset/teleport -- the context queue holds recent frames that are only
        meaningful for a spatially-continuous trajectory."""
        self.context_queue = []

    def update_context(self, rgb_image: np.ndarray) -> None:
        """Push one new RGB frame (H, W, 3 uint8). Call at `frame_rate`, not every sim/ViPlanner
        step -- NoMaD is the slow, "big direction" layer (see NOMAD_INTEGRATION.md)."""
        self.context_queue.append(Image.fromarray(rgb_image))
        if len(self.context_queue) > self.context_size + 1:
            self.context_queue.pop(0)

    @property
    def ready(self) -> bool:
        """False until enough frames have been pushed for a prediction."""
        return len(self.context_queue) > self.context_size

    def predict_waypoints(self, num_samples: int = 8, average: bool = True) -> torch.Tensor:
        """Run diffusion sampling and return waypoint chunk(s) of cumulative (dx, dy) coordinates
        in meters, in the robot's own base frame at the current step.

        With `average=True` (default): returns one (len_traj_pred, 2) chunk, averaged across all
        `num_samples` diffusion draws -- feed straight into
        nomad_bridge.nomad_output_to_world_goal (as a (1, H, 2) batch of one). Reasonable default
        when nothing else is available to pick among the candidates.

        With `average=False`: returns all (num_samples, len_traj_pred, 2) raw candidate chunks
        instead of collapsing them -- feed into frontier_map.select_best_trajectory to pick the
        one whose endpoint is closest to a chosen frontier, rather than blindly averaging away
        the diffusion model's sample diversity.
        """
        if not self.ready:
            raise RuntimeError(
                f"need {self.context_size + 1} frames in context, have {len(self.context_queue)} "
                "-- call update_context() first"
            )

        # resize to the model's trained input size (96x96) -- direct resize, no aspect-preserving
        # crop, matching the reference repo's transform_images(..., center_crop=False)
        resized = [img.resize((self.image_size[0], self.image_size[1])) for img in self.context_queue]
        obs_images = torch.cat([self._transform(img) for img in resized]).unsqueeze(0).to(self.device)
        # undirected exploration: goal image is random noise, masked out (mask=1 -> ignored)
        fake_goal = torch.randn((1, 3, self.image_size[1], self.image_size[0]), device=self.device)
        goal_mask = torch.ones(1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            obs_cond = self.model("vision_encoder", obs_img=obs_images, goal_img=fake_goal, input_goal_mask=goal_mask)
            obs_cond = obs_cond.repeat(num_samples, 1) if obs_cond.dim() == 2 else obs_cond.repeat(num_samples, 1, 1)

            naction = torch.randn((num_samples, self.len_traj_pred, 2), device=self.device)
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

        waypoints = _get_cumulative_waypoints(naction)  # (num_samples, len_traj_pred, 2)
        if average:
            waypoints = waypoints.mean(dim=0)
        if self._normalize:
            # rescale from the training data's canonical/embodiment-agnostic units to this
            # deployment robot's real meters -- see __init__'s max_v docstring
            waypoints = waypoints * (self.max_v / self.frame_rate)
        return waypoints
