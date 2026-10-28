# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Victim detection: RGB person-detection + depth back-projection to a world-frame goal point.

Sits above the NoMaD/ViPlanner stack (see NOMAD_INTEGRATION.md) as the "Victim detection / Goal"
box. Its output is the SAME world-frame point contract ViPlanner's own goal_transformer expects
(see nomad_bridge.py's docstring for how that was confirmed from viplanner_algo.py) -- so a
confirmed victim can be handed straight to ViPlanner as a goal without waiting on NoMaD, or used
to seed NoMaD's goal-image conditioning while still far away.

No COCO class is literally "victim"; this uses the "person" class as a stand-in, same as every
public SAR-robotics prototype that hasn't fine-tuned on a real casualty dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Detection:
    """One detected person, in pixel space of whichever image was passed to detect_persons."""

    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2), pixels
    confidence: float


_default_model = None


def detect_persons(rgb_image: np.ndarray, model=None, confidence_threshold: float = 0.5) -> list[Detection]:
    """Run YOLO person detection on a single RGB frame.

    Pass an already-loaded `model` (e.g. `ultralytics.YOLO("yolov8n.pt")`, loaded once at
    startup) to avoid reloading weights every call in a sim/ROS loop. Without one, lazily loads
    and caches a default yolov8n checkpoint on first call, so the function is usable
    standalone/in tests without every caller wiring up model loading.
    """
    global _default_model
    if model is None:
        if _default_model is None:
            from ultralytics import YOLO

            _default_model = YOLO("yolov8n.pt")
        model = _default_model

    # COCO class 0 = "person"; see module docstring for why this stands in for "victim"
    results = model.predict(rgb_image, classes=[0], conf=confidence_threshold, verbose=False)
    return [
        Detection(bbox_xyxy=tuple(box.xyxy[0].tolist()), confidence=float(box.conf[0]))
        for box in results[0].boxes
    ]


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` (..., 3) by `quat` (..., 4, w-x-y-z). Same formula as
    nomad_bridge._quat_apply -- duplicated rather than imported so this module stays
    independently usable; it's five lines, not worth a shared-utility coupling for.
    """
    q_w = quat[..., 0:1]
    q_xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def detection_to_world_point(
    detection: Detection,
    depth_image: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
    inner_fraction: float = 0.5,
) -> torch.Tensor | None:
    """Back-project a detected person's bounding box into a world-frame 3D point.

    `depth_image` must be on the same pixel grid (resolution/FOV) as the image `detection` was
    found in. In this repo's sim setup, the RGB and depth streams come from two separate cameras
    at different resolutions (see warehouse_cfg.py: depth_camera 848x480, semantic_camera
    1280x720) that happen to share the same extrinsic offset/rotation -- resampling one to the
    other's pixel grid is an integration choice for the caller, not something assumed here.

    `cam_pos_w`/`cam_quat_w` should be the CAMERA's own world pose, not the robot base -- in
    viplanner_demo.py these are already computed every step as
    `obs["planner_transform"]["cam_position"]` / `["cam_orientation"]`, the exact values already
    passed into VIPlannerAlgo.goal_transformer, so no new pose plumbing is needed to call this.

    Takes the median depth over the INNER `inner_fraction` of the bbox rather than a single center
    pixel or the full box: a person's silhouette isn't a filled rectangle (gaps between legs/arms
    can return background depth), and the box edges are the most likely place for detector slop to
    include a sliver of background -- median over a shrunk central region is robust to both
    without needing real segmentation.

    Returns None if every pixel in the (shrunk) box has invalid depth (0, inf, or nan) -- e.g. a
    detection right at the depth sensor's max range or seen through a window -- rather than
    silently returning a garbage point.
    """
    x1, y1, x2, y2 = detection.bbox_xyxy
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = w * (1 - inner_fraction) / 2, h * (1 - inner_fraction) / 2
    ix1, iy1 = int(x1 + margin_x), int(y1 + margin_y)
    ix2, iy2 = int(x2 - margin_x), int(y2 - margin_y)

    patch = depth_image[iy1:iy2, ix1:ix2]
    valid = torch.isfinite(patch) & (patch > 0)
    if not torch.any(valid):
        return None
    z = float(torch.median(patch[valid]))

    u = (ix1 + ix2) / 2.0
    v = (iy1 + iy2) / 2.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    point_cam = torch.tensor([x, y, z], dtype=torch.float32, device=cam_pos_w.device)

    point_world = _quat_apply(cam_quat_w.unsqueeze(0), point_cam.unsqueeze(0)) + cam_pos_w.unsqueeze(0)
    return point_world.squeeze(0)
