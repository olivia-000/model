# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predicts short-horizon future obstacle occupancy from the spatio-temporal visual feature
(Phase 3, dynamic obstacle awareness). See DynamicDualAutoEncoder
(viplanner/plannernet/autoencoder.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureOccupancyHead(nn.Module):
    """Small transposed-conv decoder predicting T_future occupancy maps on a fixed-size egocentric
    (robot-centered) grid -- the same "robot-centered" coordinate convention depth/semantic images
    already use, so this stays independent of world-frame position at inference time. Output is
    sigmoid-bounded to [0, 1] to match the ground-truth dynamic cost volume's
    clearance_to_cost(...) range (see viplanner/cost_maps/dynamic_cost_map.py) that supervises it.
    """

    def __init__(self, in_channels: int = 512, t_future: int = 4, grid_size: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.t_future = t_future

        self.reduce = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(32, t_future, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, spatio_temporal_feature: torch.Tensor) -> torch.Tensor:
        """Args:
            spatio_temporal_feature: (B, in_channels, H, W), e.g. TemporalEncoder's output.
        Returns:
            (B, t_future, grid_size, grid_size) predicted occupancy, in [0, 1].
        """
        x = self.relu(self.reduce(spatio_temporal_feature))
        x = self.relu(self.up1(x))
        x = self.relu(self.up2(x))
        x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        return self.sigmoid(self.out_conv(x))


# EoF
