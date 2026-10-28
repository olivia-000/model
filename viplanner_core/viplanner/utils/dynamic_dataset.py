# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset extension for Phase 3 (dynamic obstacle awareness): returns a depth-history stack +
optical-flow motion cue + the sample's ground-truth dynamic cost volume, on top of everything
PlannerData already provides. Does not modify PlannerData or PlannerDataGenerator -- every
existing static-scene training path is completely unaffected; this class is only used for the new
"<scene>_dynamic" training environments.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from viplanner.config import DataCfg
from viplanner.utils.dataset import PlannerData


def _clean_raw_depth(depth_image: np.ndarray, max_depth: float) -> np.ndarray:
    """Same finite-filter/scale/clamp cleaning PlannerData._load_depth_img applies to a single
    frame, factored out so DynamicPlannerData can apply it identically to every history frame."""
    depth_image = depth_image.astype("float32")
    depth_image[~np.isfinite(depth_image)] = 0.0
    depth_image = depth_image / 1000.0
    depth_image[depth_image > max_depth] = 0.0
    return depth_image


def _load_raw_depth_frame(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        depth = np.load(path)
        # simulated depth cameras save a trailing singleton channel dim in .npy that the .png
        # path below doesn't have (PIL collapses it to 2D on save/load) -- see the matching note
        # in PlannerDataGenerator._load_depth_image (viplanner/utils/dataset.py)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth
    from PIL import Image

    return np.array(Image.open(path))


class DynamicPlannerData(PlannerData):
    """PlannerData subclass for the "<scene>_dynamic" rollout-based training environments.

    `update_buffers` (inherited, unchanged) is still populated the same way by
    PlannerDataGenerator's existing odom-goal pair sampling over the rollouts' representative
    frames (see omniverse/standalone/data_collect_dynamic.py's storage layout) -- this class only
    overrides __getitem__ to additionally load the depth history + motion cue + dynamic cost
    volume for whichever rollout each representative frame belongs to.
    """

    def __init__(
        self,
        cfg: DataCfg,
        transform,
        img_input_size: tuple[int, int],
        rollouts_root: Optional[str] = None,
        rollout_manifest_path: Optional[str] = None,
        num_depth_history: int = 4,
        semantics: bool = False,
        rgb: bool = False,
        pixel_mean: Optional[np.ndarray] = None,
        pixel_std: Optional[np.ndarray] = None,
        is_static: bool = False,
        t_future: int = 4,
    ) -> None:
        """Args:
            is_static: True for an ordinary static scene used only as an anti-forgetting training
                signal (e.g. "warehouse", "warehouse_official") -- no rollouts_root/manifest
                needed. Returns the single static frame repeated num_depth_history times plus an
                all-zero motion-cue frame (Farneback flow between identical frames is exactly zero
                anyway, so this is the mathematically consistent degenerate case, not an
                approximation), and an all-zero placeholder dyn_cost_volume -- egocentric_
                occupancy_ground_truth/grid_sample of an all-zero volume is always exactly zero
                regardless of its spatial resolution, so a small fixed placeholder shape is safe.
                This trains FutureOccupancyHead to (correctly) predict "nothing coming" for a
                genuinely static scene, and contributes zero to _compute_dyn_oloss, without any
                special-casing in TrajCost -- the static and dynamic branches share the same loss
                code path.
        """
        super().__init__(cfg, transform, semantics, rgb, pixel_mean, pixel_std)
        self.img_input_size = tuple(img_input_size)
        self.num_depth_history = num_depth_history
        self.is_static = is_static
        self.t_future = t_future
        self.rollouts_root = rollouts_root
        self.rollout_manifest: dict = {}
        self._volume_cache: dict[str, torch.Tensor] = {}
        if not is_static:
            assert rollouts_root and rollout_manifest_path, "rollouts_root/rollout_manifest_path required unless is_static=True"
            with open(rollout_manifest_path) as f:
                self.rollout_manifest = json.load(f)

    def _rollout_info(self, idx: int) -> tuple[str, int]:
        stem = os.path.splitext(os.path.basename(self.depth_filename[idx]))[0]
        global_idx = str(int(stem))
        entry = self.rollout_manifest[global_idx]
        return entry["rollout"], entry["ref_local_idx"]

    def _load_depth_history_stack(self, rollout: str, ref_local_idx: int) -> np.ndarray:
        """Returns (K, H, W) float32 array of cleaned raw depth frames, oldest first, clamped at
        the start of the rollout (repeats the first available frame if fewer than K exist)."""
        rollout_depth_dir = os.path.join(self.rollouts_root, rollout, "depth")
        ext = ".npy" if os.path.exists(os.path.join(rollout_depth_dir, "0000.npy")) else ".png"

        frames = []
        for k in range(self.num_depth_history):
            local_idx = max(0, ref_local_idx - self.num_depth_history + 1 + k)
            path = os.path.join(rollout_depth_dir, f"{local_idx:04d}{ext}")
            frames.append(_clean_raw_depth(_load_raw_depth_frame(path), self._cfg.max_depth))
        return np.stack(frames, axis=0)

    def _compute_motion_cue(self, depth_stack: np.ndarray) -> np.ndarray:
        """Dense optical flow magnitude between the last two history frames, as a proxy motion
        cue -- computed from the raw frames at data-loading time (no ground-truth dependency, so
        this works identically at real deployment inference time)."""
        prev, curr = depth_stack[-2], depth_stack[-1]

        def _to_uint8(d):
            d_max = d.max()
            if d_max < 1e-6:
                return np.zeros_like(d, dtype=np.uint8)
            return ((d / d_max) * 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            _to_uint8(prev), _to_uint8(curr), None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return magnitude.astype("float32")

    def _load_dyn_cost_volume(self, rollout: str) -> torch.Tensor:
        if rollout not in self._volume_cache:
            path = os.path.join(self.rollouts_root, rollout, "dyn_cost_volume.npy")
            self._volume_cache[rollout] = torch.from_numpy(np.load(path)).float()
        return self._volume_cache[rollout]

    def __getitem__(self, idx):
        if self.is_static:
            return self._getitem_static(idx)
        return self._getitem_dynamic(idx)

    def _getitem_static(self, idx):
        if self.load_ram:
            depth_image = self.depth_imgs[idx]
        else:
            depth_image = self._load_depth_img(idx)  # (1, H', W'), already resized/pair-augmented

        depth_stack_final = depth_image.repeat(self.num_depth_history, 1, 1)  # (K, H', W')
        zero_motion = torch.zeros((1, *depth_stack_final.shape[1:]), dtype=depth_stack_final.dtype)
        depth_stack_final = torch.cat([depth_stack_final, zero_motion], dim=0)  # (K+1, H', W')

        if self.load_ram:
            sem_rgb_image = self.sem_imgs[idx] if (self.semantics or self.rgb) else 0
        elif self.semantics or self.rgb:
            sem_rgb_image = self._load_sem_rgb_img(idx)
        else:
            sem_rgb_image = 0

        dyn_cost_volume = torch.zeros((self.t_future, 4, 4), dtype=torch.float32)

        return (
            depth_stack_final,
            sem_rgb_image,
            self.odom[idx],
            self.goal[idx],
            self.pair_augment[idx],
            dyn_cost_volume,
        )

    def _getitem_dynamic(self, idx):
        rollout, ref_local_idx = self._rollout_info(idx)
        depth_stack = self._load_depth_history_stack(rollout, ref_local_idx)  # (K, H, W) raw meters
        motion_cue = self._compute_motion_cue(depth_stack)  # (H, W)

        stacked = np.concatenate([depth_stack, motion_cue[None]], axis=0)  # (K+1, H, W)
        stacked_t = torch.from_numpy(stacked).unsqueeze(0)  # (1, K+1, H, W)
        resized = F.interpolate(
            stacked_t, size=self.img_input_size, mode="bilinear", align_corners=False
        ).squeeze(0)  # (K+1, H', W')

        if self.pair_augment[idx]:
            resized = self.flip_transform.forward(resized)

        # keep all K+1 channels (K real depth frames + 1 explicit optical-flow motion-cue "frame")
        # -- DynamicDualAutoEncoder processes all K+1 through encoder_depth and its Temporal
        # Encoder; slicing this down to just the first K would silently drop the motion cue this
        # method just computed (a real bug an earlier version of this file had).
        depth_stack_final = resized  # (K+1, H', W')

        if self.load_ram:
            sem_rgb_image = self.sem_imgs[idx] if (self.semantics or self.rgb) else 0
        elif self.semantics or self.rgb:
            sem_rgb_image = self._load_sem_rgb_img(idx)
        else:
            sem_rgb_image = 0

        dyn_cost_volume = self._load_dyn_cost_volume(rollout)

        return (
            depth_stack_final,
            sem_rgb_image,
            self.odom[idx],
            self.goal[idx],
            self.pair_augment[idx],
            dyn_cost_volume,
        )


# EoF
