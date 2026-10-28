# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

from viplanner.config import TrainCfg

# visual-imperative-planner
from .future_occupancy_head import FutureOccupancyHead
from .motion_encoder import MotionEncoder
from .PlannerNet import PlannerNet
from .rgb_encoder import PRE_TRAIN_POSSIBLE, RGBEncoder
from .temporal_encoder import TemporalEncoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder_channel=64, k=5):
        super().__init__()
        self.encoder = PlannerNet(layers=[2, 2, 2, 2])
        self.decoder = Decoder(512, encoder_channel, k)

    def forward(self, x: torch.Tensor, goal: torch.Tensor):
        x = x.expand(-1, 3, -1, -1)
        x = self.encoder(x)
        x, c = self.decoder(x, goal)
        return x, c


class DualAutoEncoder(nn.Module):
    def __init__(
        self,
        train_cfg: TrainCfg,
        m2f_cfg=None,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem and PRE_TRAIN_POSSIBLE:
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])

        if train_cfg.decoder_small:
            self.decoder = DecoderS(1024, train_cfg.in_channel, train_cfg.knodes)
        else:
            self.decoder = Decoder(1024, train_cfg.in_channel, train_cfg.knodes)
        return

    def forward(self, x_depth: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        # encode depth
        x_depth = x_depth.expand(-1, 3, -1, -1)
        x_depth = self.encoder_depth(x_depth)
        # encode sem
        x_sem = self.encoder_sem(x_sem)
        # concat
        x = torch.cat((x_depth, x_sem), dim=1)  # x.size = (N, 1024, 12, 20)
        # decode
        x, c = self.decoder(x, goal)
        return x, c


class DynamicDualAutoEncoder(nn.Module):
    """Predict-then-plan variant (Phase 3, dynamic obstacle awareness): a stack of past depth
    frames is temporally aggregated and used to explicitly PREDICT short-horizon future obstacle
    occupancy, which is then encoded and fused into the same trajectory/fear decoder the static
    DualAutoEncoder uses. Does not modify DualAutoEncoder or Decoder -- every static-scene
    training/inference path is completely unaffected.

    encoder_depth/encoder_sem/decoder's underlying weights are structurally compatible with a
    trained DualAutoEncoder checkpoint (encoder_depth is called once per history frame with
    shared weights instead of once; decoder's conv1 gains `motion_channels` extra input channels)
    -- see viplanner/utils/checkpoint_surgery.py for the exact weight-transfer recipe.
    """

    def __init__(
        self,
        train_cfg: TrainCfg,
        num_history: int = 4,
        t_future: int = 4,
        occupancy_grid_size: int = 32,
        motion_channels: int = 64,
        m2f_cfg=None,
        weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_history = num_history
        self.motion_channels = motion_channels

        self.encoder_depth = PlannerNet(layers=[2, 2, 2, 2])
        if train_cfg.rgb and train_cfg.pre_train_sem and PRE_TRAIN_POSSIBLE:
            self.encoder_sem = RGBEncoder(m2f_cfg, weight_path, freeze=train_cfg.pre_train_freeze)
        else:
            self.encoder_sem = PlannerNet(layers=[2, 2, 2, 2])

        # +1: the input carries num_history real depth frames PLUS one explicit optical-flow
        # motion-cue "frame" (see DynamicPlannerData.__getitem__) -- both go through the same
        # per-frame encoder_depth pass and get temporally aggregated together.
        self.temporal_encoder = TemporalEncoder(num_history + 1, channels=512)
        self.future_occupancy_head = FutureOccupancyHead(
            in_channels=512, t_future=t_future, grid_size=occupancy_grid_size
        )
        self.motion_encoder = MotionEncoder(t_future=t_future, motion_channels=motion_channels)

        decoder_cls = DecoderS if train_cfg.decoder_small else Decoder
        self.decoder = decoder_cls(1024 + motion_channels, train_cfg.in_channel, train_cfg.knodes)
        return

    def forward(self, x_depth_history: torch.Tensor, x_sem: torch.Tensor, goal: torch.Tensor):
        """Args:
            x_depth_history: (B, K+1, H, W) the last K raw depth frames (oldest first) PLUS one
                explicit optical-flow motion-cue map appended as the final "frame" (see
                DynamicPlannerData.__getitem__), K == self.num_history. Single-channel (not yet
                expanded to 3ch).
            x_sem: (B, 3, H, W) current-frame semantic image (unchanged from DualAutoEncoder).
            goal: (B, 3) goal position in camera frame (unchanged from DualAutoEncoder).

        Returns:
            keypoints: (B, k, 3), fear: (B, 1) -- identical shapes/semantics to DualAutoEncoder's
                outputs. predicted_occupancy: (B, t_future, grid_size, grid_size), supervised by
                TrainCfg's occupancy-prediction loss (see viplanner/traj_cost_opt/traj_cost.py).
        """
        batch_size, num_frames, height, width = x_depth_history.shape
        assert num_frames == self.num_history + 1, (
            f"DynamicDualAutoEncoder configured for {self.num_history} history frames + 1 motion "
            f"cue = {self.num_history + 1} input frames, got {num_frames}"
        )

        x_depth_flat = x_depth_history.reshape(batch_size * num_frames, 1, height, width).expand(-1, 3, -1, -1)
        feat_flat = self.encoder_depth(x_depth_flat)  # (B*(K+1), 512, Hf, Wf), shared weights across frames
        _, feat_channels, feat_h, feat_w = feat_flat.shape
        feat_stack = feat_flat.reshape(batch_size, num_frames, feat_channels, feat_h, feat_w)

        spatio_temporal_feature = self.temporal_encoder(feat_stack)  # (B, 512, Hf, Wf)
        predicted_occupancy = self.future_occupancy_head(spatio_temporal_feature)  # (B, t_future, Hg, Wg)
        f_motion = self.motion_encoder(predicted_occupancy)  # (B, motion_channels)

        # index num_history - 1, NOT -1: the last slot (index num_history) is the motion-cue
        # "frame", not a real depth image -- the skip connection must use the most recent REAL
        # depth frame's own feature.
        current_depth_feature = feat_stack[:, self.num_history - 1]
        feat_sem = self.encoder_sem(x_sem)
        current_visual_feature = torch.cat((current_depth_feature, feat_sem), dim=1)  # (B, 1024, Hf, Wf)

        f_motion_spatial = f_motion[:, :, None, None].expand(-1, -1, feat_h, feat_w)
        fused = torch.cat((current_visual_feature, f_motion_spatial), dim=1)  # (B, 1024+motion_channels, Hf, Wf)

        keypoints, fear = self.decoder(fused, goal)
        return keypoints, fear, predicted_occupancy


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels),
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, k * 3)

        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x = self.fc3(x)
        x = x.reshape(-1, self.k, 3)

        c = self.relu(self.frc1(f))
        c = self.sigmoid(self.frc2(c))

        return x, c


class DecoderS(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu = nn.ReLU(inplace=True)
        self.fg = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            (in_channels + goal_channels),
            512,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(64 * 48, 256)  # --> in that setting 33 million parameters
        self.fc2 = nn.Linear(256, k * 3)

        self.frc1 = nn.Linear(256, 1)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)  # x.size = (N, 1024+16, 12, 20)
        # compute x
        x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)  --> (N, 512, 10, 18),
        x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)  --> (N, 256, 8, 16)
        x = self.relu(self.conv3(x))  # size = (N, 512, x.H/90, x.W/90)  --> (N, 128, 6, 14)
        x = self.relu(self.conv4(x))  # size = (N, 512, x.H/120, x.W/120) --> (N, 64, 4, 12)
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.fc2(f)
        x = x.reshape(-1, self.k, 3)

        c = self.sigmoid(self.frc1(f))

        return x, c


# EoF
