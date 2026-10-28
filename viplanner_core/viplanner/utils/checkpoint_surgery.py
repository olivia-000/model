# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loads a pretrained (static) DualAutoEncoder checkpoint into a DynamicDualAutoEncoder for
fine-tuning (Phase 3, dynamic obstacle awareness).

Every layer except `decoder.conv1` is architecturally identical between the two models
(encoder_depth is called once per history frame with the SAME shared weights instead of once;
encoder_sem, decoder.fc1-3, decoder.frc1-2, decoder.fg are untouched) and loads directly.
`decoder.conv1`'s input channel count grows from `1024 + goal_channels` to
`1024 + motion_channels + goal_channels` (see DynamicDualAutoEncoder.forward's channel-ordering
comment in viplanner/plannernet/autoencoder.py: [visual(1024), motion(motion_channels)] fused by
the caller, then goal(goal_channels) appended internally by Decoder.forward) -- this module
copies the old weights into the matching visual/goal slices and zero-initializes only the
genuinely new motion-cue slice, so the network starts fine-tuning from a checkpoint at which
every reused pathway behaves EXACTLY as it did before (the new motion pathway starts as a no-op,
since its weights are zero, and the network must learn to use it).
"""

from __future__ import annotations

import torch

from viplanner.plannernet.autoencoder import DynamicDualAutoEncoder


def load_pretrained_into_dynamic(
    net: DynamicDualAutoEncoder,
    old_model_pt: str,
    motion_channels: int,
) -> None:
    """Load a static DualAutoEncoder checkpoint into `net` in place.

    Args:
        net: a freshly-constructed DynamicDualAutoEncoder (its `motion_channels` must match the
            `motion_channels` argument here).
        old_model_pt: path to the old checkpoint, saved as (state_dict, val_loss) per
            Trainer.train()'s save convention.
        motion_channels: must equal `net.motion_channels` -- passed explicitly rather than read
            off `net` so a mismatch is loud (an AssertionError) rather than silently wrong.
    """
    assert motion_channels == net.motion_channels, (
        f"motion_channels mismatch: got {motion_channels}, net was built with {net.motion_channels}"
    )

    old_state_dict, _val_loss = torch.load(old_model_pt, map_location="cpu", weights_only=False)
    old_state_dict = dict(old_state_dict)  # shallow copy, safe to pop from

    old_conv1_weight = old_state_dict.pop("decoder.conv1.weight")  # (512, 1024+goal_channels, 5, 5)
    old_conv1_bias = old_state_dict.pop("decoder.conv1.bias")  # (512,), output channels unchanged

    out_channels, old_in_channels, kh, kw = old_conv1_weight.shape
    goal_channels = old_in_channels - 1024
    new_in_channels = 1024 + motion_channels + goal_channels

    new_conv1_weight = torch.zeros(out_channels, new_in_channels, kh, kw, dtype=old_conv1_weight.dtype)
    new_conv1_weight[:, :1024] = old_conv1_weight[:, :1024]  # visual features, unchanged slice
    # motion_channels slice (1024:1024+motion_channels) stays exactly zero -- the new pathway
    # starts as a no-op contribution to conv1's output until the network learns to use it
    new_conv1_weight[:, 1024 + motion_channels :] = old_conv1_weight[:, 1024:]  # goal features, unchanged slice

    missing, unexpected = net.load_state_dict(old_state_dict, strict=False)
    expected_missing = {
        "decoder.conv1.weight",
        "decoder.conv1.bias",
        "temporal_encoder.temporal_conv.weight",
        "temporal_encoder.temporal_conv.bias",
        "future_occupancy_head.reduce.weight",
        "future_occupancy_head.reduce.bias",
        "future_occupancy_head.up1.weight",
        "future_occupancy_head.up1.bias",
        "future_occupancy_head.up2.weight",
        "future_occupancy_head.up2.bias",
        "future_occupancy_head.out_conv.weight",
        "future_occupancy_head.out_conv.bias",
        "motion_encoder.conv1.weight",
        "motion_encoder.conv1.bias",
        "motion_encoder.conv2.weight",
        "motion_encoder.conv2.bias",
        "motion_encoder.fc.weight",
        "motion_encoder.fc.bias",
    }
    assert set(missing) == expected_missing, f"unexpected set of missing keys: {set(missing) - expected_missing}"
    assert unexpected == [], f"unexpected leftover keys in old checkpoint: {unexpected}"

    with torch.no_grad():
        net.decoder.conv1.weight.copy_(new_conv1_weight)
        net.decoder.conv1.bias.copy_(old_conv1_bias)


# EoF
