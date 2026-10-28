# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encodes the FutureOccupancyHead's own predicted occupancy maps into a compact feature vector
(Phase 3, dynamic obstacle awareness). See DynamicDualAutoEncoder
(viplanner/plannernet/autoencoder.py).

Deliberately encodes the head's OWN prediction (differentiable, gradients flow back through the
TemporalEncoder) rather than the ground truth -- this is what lets the decoder's trajectory
implicitly account for anticipated obstacle motion, instead of only being penalized for it after
the fact via the loss (see TrajCost._compute_dyn_oloss in viplanner/traj_cost_opt/traj_cost.py,
which is trained against ground truth for a stable signal, while this pathway threads the
network's own belief through to planning).
"""

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    def __init__(self, t_future: int = 4, motion_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(t_future, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, motion_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, predicted_occupancy: torch.Tensor) -> torch.Tensor:
        """Args:
            predicted_occupancy: (B, t_future, grid_size, grid_size), e.g.
                FutureOccupancyHead's output.
        Returns:
            (B, motion_channels) F_motion feature vector.
        """
        x = self.relu(self.conv1(predicted_occupancy))
        x = self.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.relu(self.fc(x))


# EoF
