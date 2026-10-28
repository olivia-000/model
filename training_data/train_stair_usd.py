"""Fine-tune v3 on the user's Stair.usdc scene (open_scene_stair_usd.py) with the new RAILING
semantic class (camera_semantics_lab.SEG_ID_RAILING -> ViPlanner "fence" color, same WALL_LOSS as
walls in build_costmap.py).

Starts from v3 (explicit user choice), not the base checkpoint: v3 already handles this project's
stair scenes; this run only has to add railings + this scene's layout. Pipeline before running:
  1. collect rollouts: VIPLANNER_COLLECT_DATA=/home/isaacsim/training_data/stair_usd_scene
  2. build_cloud.py stair_usd_scene/aabbs.json stair_usd_scene/cloud.ply
  3. build_costmap.py /home/itri/Desktop/viplanner2/training_data/stair_usd_scene
  4. data/stair_usd_scene -> ../stair_usd_scene symlink
Run with the viplanner_train conda env.
"""
import os
import shutil
import sys

sys.path.insert(0, "/home/itri/Desktop/viplanner2/viplanner_core")

from viplanner.config import DataCfg, TrainCfg  # noqa: E402
from viplanner.utils.trainer import Trainer  # noqa: E402

TRAINING_DATA_ROOT = "/home/itri/Desktop/viplanner2/training_data"
# START_CKPT_DIR (env): model dir to start from -- default v3; set to e.g. v6's dir for a continuation run
START_CKPT_DIR = os.environ.get("START_CKPT_DIR") or os.path.join(
    TRAINING_DATA_ROOT, "models",
    "plannernet_envstairs_scene_ep30_inputDepSem_costSem_optimSGD_stairs_finetune_v3")

cfg = TrainCfg.from_yaml(os.path.join(START_CKPT_DIR, "model.yaml"))

# TRAIN_ENVS: comma-separated scene dirs under data/ (e.g. two collection batches of the same scene)
cfg.env_list = os.environ.get("TRAIN_ENVS", "stair_usd_scene").split(",")
cfg.test_env_id = -1  # same reasoning as train_stairs.py -- one env, train on all of it
cfg.cost_map_name = "cost_map_sem"
cfg.file_path = TRAINING_DATA_ROOT
cfg.file_name = os.environ.get("TRAIN_FILE_NAME", "stair_usd_railing_v5")
cfg.resume = True
# v3's best_loss was measured on a different scene -- not comparable, would block every save.
# RESET_BEST_LOSS=0 for a continuation on the SAME data (e.g. v6 -> v6_cont), where it is comparable.
cfg.reset_best_loss_on_resume = os.environ.get("RESET_BEST_LOSS", "1") == "1"
cfg.epochs = int(os.environ.get("TRAIN_EPOCHS", "30"))
cfg.lr = 2e-4
cfg.num_workers = 0  # small /dev/shm crashes multi-worker DataLoader IPC
cfg.wb_project = "viplanner_stairs"
cfg.n_visualize = 15
cfg.robot_width = float(os.environ.get("TRAIN_ROBOT_WIDTH", "0.28"))  # Go2 footprint (train_stairs.py); leo: 0.54

base_dc = cfg.data_cfg[0] if isinstance(cfg.data_cfg, list) else cfg.data_cfg
cfg.data_cfg = DataCfg(
    depth_suffix="",
    sem_suffix="",
    max_goal_distance=10.0,
    min_goal_distance=base_dc.min_goal_distance,
    distance_scheme=base_dc.distance_scheme,
    obs_cost_height=base_dc.obs_cost_height,
    ratio=0.9,
)

os.makedirs(cfg.curr_model_dir, exist_ok=True)
dest = os.path.join(cfg.curr_model_dir, "model.pt")
if not os.path.exists(dest):
    shutil.copy(os.path.join(START_CKPT_DIR, "model.pt"), dest)
    print(f"[train_stair_usd] staged start checkpoint ({START_CKPT_DIR}) at {dest}")

print(f"[train_stair_usd] curr_model_dir = {cfg.curr_model_dir}")
print(f"[train_stair_usd] data_dir       = {cfg.data_dir}")

trainer = Trainer(cfg)
trainer.train()
trainer.save_config()
print("[train_stair_usd] DONE. Fine-tuned checkpoint + model.yaml at:", cfg.curr_model_dir)
