# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporal aggregation over a stack of per-frame depth-encoder feature maps (Phase 3, dynamic
obstacle awareness). See DynamicDualAutoEncoder (viplanner/plannernet/autoencoder.py)."""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Collapses a (B, K, C, H, W) stack of per-frame visual features into one (B, C, H, W)
    "spatio-temporal feature" via a single Conv3d with a kernel spanning the full K history --
    the cheapest temporal-aggregation choice that's still genuinely convolutional over time (vs.
    e.g. a ConvLSTM or temporal attention block), a deliberate simplification for a single-GPU
    (RTX 2080 Ti) verification run. Channel count is preserved so the output can feed straight
    into the same downstream shape the un-aggregated per-frame features already have.
    """

    def __init__(self, num_history: int, channels: int = 512):
        super().__init__()
        self.num_history = num_history
        self.temporal_conv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(num_history, 1, 1),
            stride=1,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_stack: torch.Tensor) -> torch.Tensor:
        """Args:
            feat_stack: (B, K, C, H, W) per-frame visual-encoder features, oldest frame first.
        Returns:
            (B, C, H, W) spatio-temporal feature.
        """
        assert feat_stack.shape[1] == self.num_history, (
            f"TemporalEncoder configured for {self.num_history} history frames, got {feat_stack.shape[1]}"
        )
        x = feat_stack.permute(0, 2, 1, 3, 4)  # (B, C, K, H, W)
        x = self.relu(self.temporal_conv(x))  # (B, C, 1, H, W)
        return x.squeeze(2)  # (B, C, H, W)


# EoF
