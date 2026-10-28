# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline driver: builds the GT dynamic obstacle cost volume for every rollout collected by
omniverse/standalone/data_collect_dynamic.py.

Deliberately a SEPARATE script from data_collect_dynamic.py: this needs open3d/scipy (via
CostMapPCD/TsdfCostMap) which are training-side dependencies not installed in the Isaac Sim
container's python -- run this one in the plain training conda env instead, after collection
finishes inside the container.

Usage:
    python viplanner/cost_maps/build_dynamic_maps.py --scene warehouse
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from viplanner.cost_maps.cost_to_pcd import CostMapPCD
from viplanner.cost_maps.dynamic_cost_map import build_dynamic_cost_volume


def main():
    parser = argparse.ArgumentParser(description="Build GT dynamic cost volumes for a collected scene.")
    parser.add_argument("--scene", required=True, type=str, help="Scene name, e.g. 'warehouse'.")
    parser.add_argument("--future_horizon_s", type=float, default=2.0)
    parser.add_argument("--future_dt", type=float, default=0.5)
    parser.add_argument("--person_radius_m", type=float, default=0.3)
    parser.add_argument("--sigma_expand", type=float, default=2.0)
    parser.add_argument("--decay_scale_m", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_data_path = os.path.join(repo_root, "training_data", "data", args.scene)
    dynamic_data_path = os.path.join(repo_root, "training_data", "data", f"{args.scene}_dynamic")
    people_cfg_path = os.path.join(
        repo_root, "omniverse", "extension", "omni.viplanner", "data", args.scene, "people_cfg.yml"
    )

    assert os.path.isdir(
        os.path.join(static_data_path, "maps")
    ), f"No static cost map found at {static_data_path}/maps -- run data_collect.py + cost_builder.py for '{args.scene}' first."
    assert os.path.isfile(people_cfg_path), f"people_cfg.yml not found at {people_cfg_path}"

    with open(people_cfg_path) as f:
        people_cfg = yaml.safe_load(f)
    people_by_name = {v["prim_name"]: v for v in people_cfg.values()}
    # people_cfg.yml is keyed by e.g. "person_1" (matching UnRealImporter._people_state's "name"
    # field), NOT by "prim_name" ("Person_1") -- build both lookups to be robust to either.
    people_by_key = dict(people_cfg)

    def _lookup_person_cfg(name: str) -> dict:
        if name in people_by_key:
            return people_by_key[name]
        if name in people_by_name:
            return people_by_name[name]
        raise KeyError(f"Person '{name}' not found in {people_cfg_path}")

    cost_map = CostMapPCD.ReadTSDFMap(static_data_path, "cost_map_geom", gpu_id=None)

    manifest_path = os.path.join(dynamic_data_path, "rollout_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    rollout_names = sorted({entry["rollout"] for entry in manifest.values()})
    print(f"[INFO] Building dynamic cost volumes for {len(rollout_names)} rollouts...")

    for rollout_name in rollout_names:
        rollout_dir = os.path.join(dynamic_data_path, "rollouts", rollout_name)
        state = np.load(os.path.join(rollout_dir, "dynamic_state.npz"), allow_pickle=True)
        person_ids = [str(p) for p in state["person_ids"]]

        people = []
        for p_idx, name in enumerate(person_ids):
            cfg = _lookup_person_cfg(name)
            start = np.array(cfg["translation"], dtype=float)
            target = np.array(cfg["target"], dtype=float)
            speed = float(cfg.get("speed", 1.0))
            mode = cfg.get("mode", "bounce")

            pos_t0 = state["person_pos"][-1, p_idx]
            vel_t0 = state["person_vel"][-1, p_idx]
            # person_pos/person_vel are stored full 3D (matching UnRealImporter._people_state's
            # start/target/pos, themselves parsed directly from the 3-element translation/target
            # YAML fields) -- keep segment_dir 3D too so the dot product's shapes match.
            segment_dir = target - start
            direction_t0 = 1.0 if np.dot(vel_t0, segment_dir) >= 0 else -1.0

            people.append(dict(start=start, target=target, speed=speed, mode=mode, pos_t0=pos_t0, direction_t0=direction_t0))

        volume = build_dynamic_cost_volume(
            people,
            num_x=cost_map.num_x,
            num_y=cost_map.num_y,
            start_x=cost_map.cfg.x_start,
            start_y=cost_map.cfg.y_start,
            resolution=cost_map.cfg.general.resolution,
            future_horizon_s=args.future_horizon_s,
            future_dt=args.future_dt,
            person_radius_m=args.person_radius_m,
            sigma_expand=args.sigma_expand,
            decay_scale_m=args.decay_scale_m,
        )
        np.save(os.path.join(rollout_dir, "dyn_cost_volume.npy"), volume)

    print(f"[INFO] Done. Wrote dyn_cost_volume.npy for {len(rollout_names)} rollouts under {dynamic_data_path}/rollouts/")


if __name__ == "__main__":
    main()

# EoF
