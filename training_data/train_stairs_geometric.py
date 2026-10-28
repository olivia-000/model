"""Fine-tune the ORIGINAL base ViPlanner checkpoint on stairs-scene rollout data collected using
viplanner_core's own real-hardware-ready geometric semantic classifier (compute_geometric_semantic,
see viplanner_core/viplanner/utils/geometric_semantic.py) instead of this project's ground-truth
AABB-lookup semantic image.

Why a separate script from train_stairs.py (the v3 line): v3's entire lineage -- and the base
checkpoint's own original training -- used the ground-truth 3-class (floor/stairs/wall) semantic
scheme. The geometric classifier only outputs 2 classes (floor/obstacle, no stairs-specific color --
see that module's docstring), a materially different input distribution the network has never seen.
Per explicit user request ("先深度分類器...如果效果不好，我們再加真實分割模型"), this starts fresh
from the pristine base checkpoint rather than continuing v3 (which is itself overfit to the 3-class
scheme) -- same reasoning the project has used for every prior "start a new lineage instead of
continuing the wrong-distribution one" decision (v2->v3 for the stale-geometry bug, this one for the
semantic-scheme swap).

Data was collected with the stairs' side guardrails and landing walls DISABLED
(DISABLE_STAIR_SIDE_WALLS=True in open_scene_stairs.py) -- deliberately, since this whole exercise is
about real-hardware viability and a real robot won't have guardrails either; training on wall-present
depth cues that won't exist on the real robot would defeat the point.

cloud.ply / maps/ (cost map) / aabbs.json were copied in from training_data/stairs_scene/ rather than
rebuilt -- they're purely scene-geometry-derived and don't depend on which semantic classifier
produced the collected rollout's depth/semantic image pairs, so build_cloud.py/build_costmap.py don't
need to run again for this.
"""
import os
import shutil
import sys

sys.path.insert(0, "/home/itri/Desktop/viplanner2/viplanner_core")

from viplanner.config import DataCfg, TrainCfg  # noqa: E402
from viplanner.utils.trainer import Trainer  # noqa: E402

BASE_CKPT_DIR = "/home/itri/Desktop/viplanner2/base_checkpoint"  # T7 drive's copy was inaccessible
# (drive not mounted, then mounted-but-checkpoint-missing when it was) -- re-downloaded fresh from
# ViPlanner's own official release links (README.md's "Model Download" section) instead.
TRAINING_DATA_ROOT = "/home/itri/Desktop/viplanner2/training_data"

cfg = TrainCfg.from_yaml(os.path.join(BASE_CKPT_DIR, "model.yaml"))

cfg.env_list = ["stairs_scene_geometric"]
cfg.test_env_id = -1  # same reasoning as train_stairs.py -- one env, train on all of it
cfg.cost_map_name = "cost_map_sem"  # matches build_costmap.py's cfg.map_name (copied file, same name)
cfg.file_path = TRAINING_DATA_ROOT
cfg.file_name = "stairs_finetune_v4_geometric"
cfg.resume = True
# IMPORTANT: despite "starting a fresh lineage from the pristine base checkpoint" (not continuing
# v3), `resume` here does NOT mean "continue a previous run of this file" -- Trainer._load_model
# only loads from `self.model_path` (== <curr_model_dir>/model.pt, staged below) when resume=True;
# resume=False builds a network with FRESH RANDOM WEIGHTS and never touches the staged checkpoint
# at all (see trainer.py's _load_model: the `if resume:` block is the only place model_state_dict
# is ever loaded). So resume=True is required here too, exactly like v3's own first run -- it's
# simply how this codebase loads ANY starting weights, whether that's a previous fine-tune's output
# or (as here) a fresh copy of the untouched base checkpoint. Caught this before running: an
# earlier draft of this script had resume=False, thinking it meant "don't continue v3" -- that
# would have silently trained from random initialization instead of fine-tuning at all.
cfg.reset_best_loss_on_resume = True  # the base checkpoint's own best_loss is a cross-env value
# from unrelated real-world training data -- not comparable to loss on this scene, so start fresh
# rather than comparing against it (matches what v3's own first run against the base checkpoint did).
cfg.epochs = 30  # same starting budget as v3's own first pass
cfg.lr = 2e-4  # same as v3 -- order of magnitude below the from-scratch default, appropriate for
# fine-tuning an already-converged checkpoint rather than training from nothing
cfg.num_workers = 0  # small /dev/shm crashes multi-worker DataLoader IPC (see train.py's own note)
cfg.wb_project = "viplanner_stairs"
cfg.n_visualize = 15
cfg.robot_width = 0.28  # Go2's real footprint width, not TrajCost's ANYmal-sized 0.6m default

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

# stage the base checkpoint's weights at the exact path _load_model(resume=True) reads from.
os.makedirs(cfg.curr_model_dir, exist_ok=True)
dest = os.path.join(cfg.curr_model_dir, "model.pt")
if not os.path.exists(dest):
    shutil.copy(os.path.join(BASE_CKPT_DIR, "model.pt"), dest)
    print(f"[train_stairs_geometric] staged base checkpoint at {dest}")

print(f"[train_stairs_geometric] curr_model_dir = {cfg.curr_model_dir}")
print(f"[train_stairs_geometric] data_dir       = {cfg.data_dir}")

trainer = Trainer(cfg)
trainer.train()
trainer.save_config()
print("[train_stairs_geometric] DONE. Fine-tuned checkpoint + model.yaml at:", cfg.curr_model_dir)
