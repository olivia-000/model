# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import argparse

import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViPlanner training.")
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="skip trainer.train() and only evaluate the already-saved checkpoint on the held-out env "
        "(trainer.test() loads it from disk regardless of whether train() ran in this process)",
    )
    cli_args = parser.parse_args()

    # "warehouse_official" stays held out entirely (never seen during training) and is only used
    # by trainer.test() to check generalization to a novel scene; every other entry is trained on.
    # warehouse_with_forklifts/warehouse_multiple_shelves added 2026-08-14 specifically for
    # obstacle-avoidance diversity -- the original "warehouse"-only model never saw a forklift or
    # a dense multi-shelf layout during training. full_warehouse deliberately left out: it crashes
    # a few sim steps in (see full_warehouse_cfg.py's KNOWN ISSUE, a PhysX GPU contact-buffer
    # sizing bug), so no data was ever collectable for it.
    env_list_warehouse = [
        "warehouse",
        "warehouse_official",
        "warehouse_with_forklifts",
        "warehouse_multiple_shelves",
    ]
    # depth_suffix/sem_suffix default to "_cam0"/"_cam1" for real robots where the depth and
    # semantic cameras are physically offset; data_collect.py's simulated cameras share the same
    # viewpoint and save plain "camera_extrinsic.txt"/"NNNN.png" with no suffix at all.
    data_cfg = DataCfg(max_goal_distance=10.0, depth_suffix="", sem_suffix="")
    warehouse: TrainCfg = TrainCfg(
        sem=True,
        epochs=150,
        # geometric (not semantic) cost map: the warehouse scene's semantic labels don't render
        # correctly (see costmap_cfg.py note), so the cost map is built from pure 3D occupancy instead
        cost_map_name="cost_map_geom",
        env_list=env_list_warehouse,
        test_env_id=1,
        # distinct from the original "warehouse"-only checkpoint (checkpoints/... isn't touched by
        # this rename -- this is the plannernet/cost-map training output, a separate model)
        file_name="warehouse_obstacles",
        data_cfg=data_cfg,
        n_visualize=15,
        wb_project="viplanner",
        # this container's /dev/shm is only 64MB (Docker default); multi-worker DataLoader IPC
        # for image batches needs much more and crashes workers with a bus error. Single-process
        # loading avoids shared memory entirely -- slower, but recreating the container to raise
        # --shm-size isn't worth it (past incidents lost the manually-installed ros_env/editable
        # packages, see go2_rviz_guide.md).
        num_workers=0,
    )
    trainer = Trainer(warehouse)
    if not cli_args.test_only:
        trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()
