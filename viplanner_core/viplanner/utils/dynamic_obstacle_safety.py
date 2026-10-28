# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight reactive safety layer: proximity + closing-speed signal derived from the live
depth camera, used to scale down commanded speed near close and/or approaching obstacles without
touching the trained ViPlanner network. See declarative-chasing-tome plan, Phase 2."""

from __future__ import annotations

import torch


def _squeeze_depth(depth_image: torch.Tensor) -> torch.Tensor:
    depth = depth_image
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    elif depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
    if depth.dim() != 3:
        raise ValueError(f"expected a (num_envs, H, W)-like depth tensor, got shape {tuple(depth_image.shape)}")
    return depth


def _percentile_min(roi_flat: torch.Tensor, robust_percentile: float) -> torch.Tensor:
    """(num_envs,) robust-min over a flattened (num_envs, N) ROI -- +inf for an env with no valid
    (finite, positive) reading. Shared by min_depth_in_roi and min_depth_in_roi_split."""
    num_envs = roi_flat.shape[0]
    valid = torch.isfinite(roi_flat) & (roi_flat > 0.0)
    result = torch.full((num_envs,), float("inf"), device=roi_flat.device)
    for i in range(num_envs):
        vals = roi_flat[i][valid[i]]
        if vals.numel() > 0:
            result[i] = torch.quantile(vals.float(), robust_percentile)
    return result


def min_depth_in_roi(depth_image: torch.Tensor, roi_frac: float = 0.5, robust_percentile: float = 0.20) -> torch.Tensor:
    """Robust "closest thing in view" depth (meters) within a centered square region of interest.

    Args:
        depth_image: camera sensor output for "distance_to_image_plane", accepted as either the
            raw sensor shape (num_envs, H, W, 1), the VIPlannerAlgo-style (num_envs, 1, H, W), or
            plain (num_envs, H, W).
        roi_frac: fraction of the image height/width (centered) to consider.
        robust_percentile: use this percentile of valid ROI depths instead of the raw minimum.
            Root-cause fix for the 2026-08-31 multi_floor stuck-at-riser investigation: a raw
            min() picked up the floor at grazing incidence in the ROI's lower rows (camera mounted
            near ground level, pos=(0.51, 0.0, 0.015) relative to base) and reported a
            permanently-close "obstacle" even in an open corridor. Confirmed via two diagnostics
            (debug_roi_depth_stats, debug_roi_grid): the near-field cluster was column-uniform
            (rules out a corridor side-wall) and confined to roughly the bottom 10-20% of the ROI
            by depth-percentile rank, while the median stayed at the genuinely-open distance the
            whole time. A first attempt modeled this geometrically (camera height + intrinsics ->
            per-row floor-intercept distance) but the predicted floor distance didn't match the
            measured one closely enough to trust (the "distance_to_image_plane" convention may not
            be the simple Euclidean ray distance the model assumed, among other unverified
            assumptions) -- rather than keep guessing at camera geometry, this percentile approach
            sidesteps needing to know the exact physical cause: it's robust to any near-field
            cluster occupying a minority of the ROI, whatever it actually is, while still
            triggering normally once a real obstacle fills enough of the frame (confirmed in the
            same log: median converged to the raw min once the robot was genuinely close to the
            stairs). 0.20 has real margin above the ~10% contamination measured, without being so
            high it would miss a real hazard that only partially fills the ROI.

    Returns:
        (num_envs,) tensor. Envs with no valid (finite, positive) reading in the ROI get +inf,
        which downstream scaling treats as "nothing close" (no slowdown).
    """
    depth = _squeeze_depth(depth_image)
    num_envs, height, width = depth.shape
    roi_h = max(1, int(height * roi_frac))
    roi_w = max(1, int(width * roi_frac))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    roi = depth[:, y0 : y0 + roi_h, x0 : x0 + roi_w].reshape(num_envs, -1)
    return _percentile_min(roi, robust_percentile)


def min_depth_in_roi_split(
    depth_image: torch.Tensor, roi_frac: float = 0.5, center_frac: float = 0.4, robust_percentile: float = 0.20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like min_depth_in_roi, but splits the ROI into a centered vertical column band ("center",
    where a climbable stair riser directly ahead reads) and the two flanking column bands
    ("side", where a corridor's side walls read), returning (center_dist, side_dist) each
    (num_envs,).

    Purely geometric (2026-08-31, stair-aware safety follow-up to the percentile fix above) --
    NOT based on semantic segmentation of what's actually in view, per user's explicit direction:
    stair_mode itself must come from the existing navigation FSM/topological-waypoint state (see
    PathFollowerCommandGeneratorCfg.stair_center_frac and the command generator's `stair_mode`),
    not be inferred from this depth split alone. A semantic-segmentation-based version (reading
    the depth_camera's own class labels to tell a stair tread from a wall directly, generalizing
    beyond a fixed center/side split) is a real option for later but deliberately deferred --
    this geometric split ships first since it reuses the already-working percentile logic with no
    new sensor plumbing.

    Args:
        depth_image: see min_depth_in_roi.
        roi_frac: see min_depth_in_roi.
        center_frac: fraction of the ROI's width (centered) treated as "center"; the remainder
            (split evenly left/right) is "side".
        robust_percentile: see min_depth_in_roi.

    Returns:
        (center_dist, side_dist): each (num_envs,), +inf where that band has no valid reading.
    """
    depth = _squeeze_depth(depth_image)
    num_envs, height, width = depth.shape
    roi_h = max(1, int(height * roi_frac))
    roi_w = max(1, int(width * roi_frac))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    roi = depth[:, y0 : y0 + roi_h, x0 : x0 + roi_w]

    center_w = max(1, int(roi_w * center_frac))
    c0 = (roi_w - center_w) // 2
    center_roi = roi[:, :, c0 : c0 + center_w].reshape(num_envs, -1)
    side_roi = torch.cat([roi[:, :, :c0], roi[:, :, c0 + center_w :]], dim=2).reshape(num_envs, -1)

    center_dist = _percentile_min(center_roi, robust_percentile)
    side_dist = _percentile_min(side_roi, robust_percentile)
    return center_dist, side_dist


def debug_roi_grid(depth_image: torch.Tensor, roi_frac: float = 0.5, env_idx: int = 0, n_bands: int = 6) -> str:
    """One-shot diagnostic (2026-08-31): median depth in an n_bands x n_bands grid over the ROI,
    to localize a near-field cluster by ROW (floor at grazing incidence -> only bottom rows close,
    uniform across columns) vs. COLUMN (a corridor side-wall -> only left or right columns close,
    across many rows) -- min_depth_in_roi's own physically-derived floor model didn't cleanly match
    the empirical percentiles, so don't guess further; look at where it actually is."""
    depth = depth_image
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    elif depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    _, height, width = depth.shape
    roi_h = max(1, int(height * roi_frac))
    roi_w = max(1, int(width * roi_frac))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    roi = depth[env_idx, y0 : y0 + roi_h, x0 : x0 + roi_w]
    valid = torch.isfinite(roi) & (roi > 0.0)

    row_edges = torch.linspace(0, roi_h, n_bands + 1).long()
    col_edges = torch.linspace(0, roi_w, n_bands + 1).long()
    lines = []
    for i in range(n_bands):
        row_vals = []
        for j in range(n_bands):
            block = roi[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            block_valid = valid[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            v = block[block_valid]
            row_vals.append(f"{v.median().item():5.2f}" if v.numel() > 0 else "  nan")
        lines.append(f"row_band{i} (img rows {y0 + row_edges[i].item()}-{y0 + row_edges[i + 1].item()}): " + " ".join(row_vals))
    return "\n".join(lines)


def debug_roi_depth_stats(depth_image: torch.Tensor, roi_frac: float = 0.5, env_idx: int = 0) -> str:
    """Diagnostic only (2026-08-31 multi_floor safety-layer investigation) -- NOT used by the real
    safety scaling, only by debug_print_cmd. Percentile spread of valid depths inside the exact
    same ROI `min_depth_in_roi` reads, for one env. Distinguishes "a small cluster of near pixels
    (e.g. floor at grazing incidence) is dragging the min down" (min/p05/p10 all low, median high)
    from "the whole ROI is genuinely close" (min through median all low)."""
    depth = depth_image
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    elif depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    _, height, width = depth.shape
    roi_h = max(1, int(height * roi_frac))
    roi_w = max(1, int(width * roi_frac))
    y0 = (height - roi_h) // 2
    x0 = (width - roi_w) // 2
    roi = depth[env_idx, y0 : y0 + roi_h, x0 : x0 + roi_w]

    valid = torch.isfinite(roi) & (roi > 0.0)
    vals = roi[valid]
    if vals.numel() == 0:
        return "roi_stats=<no valid pixels>"
    q = torch.quantile(vals.float(), torch.tensor([0.05, 0.10, 0.50, 0.90], device=vals.device))
    return (
        f"roi_min={vals.min().item():.3f} p05={q[0].item():.3f} p10={q[1].item():.3f} "
        f"median={q[2].item():.3f} p90={q[3].item():.3f} n_valid={vals.numel()}"
    )


class ClosingSpeedTracker:
    """Per-env frame-to-frame differencing of a distance signal.

    Deliberately not a Kalman filter / object tracker -- a first-version reactive safety layer
    only needs "is the nearest thing in view getting closer faster than expected", not identity
    or trajectory prediction of individual obstacles.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self._prev_dist = torch.full((num_envs,), float("inf"), device=device)

    def update(self, curr_dist: torch.Tensor, dt: float) -> torch.Tensor:
        """Returns (num_envs,) closing speed in m/s; positive means getting closer."""
        closing_speed = torch.zeros_like(curr_dist)
        if dt > 0.0:
            valid = torch.isfinite(self._prev_dist) & torch.isfinite(curr_dist)
            closing_speed[valid] = (self._prev_dist[valid] - curr_dist[valid]) / dt
        self._prev_dist = curr_dist.clone()
        return closing_speed

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_dist[env_ids] = float("inf")
