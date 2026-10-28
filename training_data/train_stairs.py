"""Fine-tune the existing ViPlanner checkpoint on this stairs scene's collected rollout data.

Starts from the deployed checkpoint (/media/itri/T7/viplanner/checkpoints/model.pt + model.yaml)
rather than training from scratch -- loads its exact TrainCfg first (so network architecture params
like in_channel/knodes/img_input_size match the saved state_dict's shapes) and only overrides what
this run actually needs to change: which environment/data to train on, the cost map, output
location, and a lower learning rate appropriate for fine-tuning rather than training from scratch.

Per explicit user request, wall (and person, tagged as the same class in this scene) collision cost
is patched much higher than the shared library default before the cost map is even built -- see
build_costmap.py, which must be run before this script.
"""
import os
import shutil
import sys

sys.path.insert(0, "/home/itri/Desktop/viplanner2/viplanner_core")

from viplanner.config import DataCfg, TrainCfg  # noqa: E402
from viplanner.utils.trainer import Trainer  # noqa: E402

BASE_CKPT_DIR = "/media/itri/T7/viplanner/checkpoints"
TRAINING_DATA_ROOT = "/home/itri/Desktop/viplanner2/training_data"

cfg = TrainCfg.from_yaml(os.path.join(BASE_CKPT_DIR, "model.yaml"))

# --- what this fine-tune run actually changes vs. the base checkpoint's own training config ---
cfg.env_list = ["stairs_scene"]
# no held-out environment -- we only have this one scene and want it to learn this scene well, not
# validate generalization to something else. -1 never matches a real index, so _load_data's
# `idx == test_env_id` skip-during-train check never fires and this one env always loads for
# training; trainer.test() is deliberately never called below (see bottom) since it would try to
# load an env at idx==test_env_id and find nothing.
cfg.test_env_id = -1
cfg.cost_map_name = "cost_map_sem"  # matches build_costmap.py's cfg.map_name
cfg.file_path = TRAINING_DATA_ROOT  # -> data_dir=.../training_data/data, curr_model_dir=.../training_data/models/...
cfg.file_name = "stairs_finetune_v3"  # v3: retrained on data recollected AFTER fixing the
# exposed-stair-back-face-mislabeled-as-walkable bug in open_scene_stairs.py -- v2's entire
# training pipeline (cloud.ply/aabbs.json/collected rollout episodes) predated that fix (captured
# 2026-09-11), so v2 never actually learned from the corrected semantics even though the LIVE scene
# had it. v2's own directory is left untouched (per explicit user request, kept as a reference/
# fallback alongside the original base checkpoint) -- this writes to a new directory instead of
# resuming it, same reasoning "v2" itself used against the original "stairs_finetune" run.
cfg.resume = True
# CONTINUATION RUN: the first 30-epoch pass (val loss 12.59 -> 10.48) went flat for its last 5
# epochs (25-29, no improvement at all) right as the scheduler made its second LR cut (1e-4 ->
# 5e-5) -- a stronger plateau signal than what prompted v2's own 30->80 continuation, but only 1
# epoch has actually run at the new, lower LR, so confirming it holds needs a few more. Manually
# pre-staged this run's own model.pt/model.yaml into ep60's directory before launching (see the
# GOTCHA this file's history already has on curr_model_dir baking in cfg.epochs) -- confirm the
# log's "Resume train from ... with loss ..." reads ~10.48 (this run's own val_loss), not 8.098
# (the pristine base checkpoint's cross-env loss), before trusting this actually resumed.
cfg.reset_best_loss_on_resume = False  # this run's own best_loss (10.48) IS comparable now.
cfg.epochs = 30  # 30 (already done) + 30 (this run) = 60 total
cfg.lr = 2e-4  # base config used 2e-3 for training from scratch; an order of magnitude lower here
# to adapt to this scene without wrecking what the checkpoint already learned elsewhere
cfg.num_workers = 0  # per train.py's own note: small /dev/shm crashes multi-worker DataLoader IPC
cfg.wb_project = "viplanner_stairs"
cfg.n_visualize = 15
# TrajCost's own default (0.6m) is sized for ANYmal -- the base checkpoint's saved model.yaml
# inherits that, and nothing in this repo overrode it before, so every prior fine-tune (including
# the wall-weight-only one earlier this session) was sampling obstacle cost as if the robot were
# more than twice as wide as a Go2. Set to the real footprint width; the soft clearance gradient
# (see build_costmap.py's clearance-based wall cost) provides the extra margin beyond contact, so
# this should be the true body width, not body+buffer.
cfg.robot_width = 0.28

# our data_collect instrumentation in viplanner_control.py shares one camera viewpoint for both
# depth and semantics (see the DIAG comment there), unlike the base config's separate real-hardware
# _cam0/_cam1 cameras -- matches train.py's own note about data_collect.py's simulated cameras.
base_dc = cfg.data_cfg[0] if isinstance(cfg.data_cfg, list) else cfg.data_cfg
cfg.data_cfg = DataCfg(
    depth_suffix="",
    sem_suffix="",
    max_goal_distance=10.0,  # this scene's own route is ~14m total; base default (15.0) is fine
    # too, but goals sampled beyond the route's own extent would just clip to whatever's on the map
    min_goal_distance=base_dc.min_goal_distance,
    distance_scheme=base_dc.distance_scheme,
    obs_cost_height=base_dc.obs_cost_height,
    ratio=0.9,
)

# stage the base checkpoint's weights at the exact path _load_model(resume=True) reads from
# (<curr_model_dir>/model.pt, a (state_dict, best_loss) tuple -- same format the base checkpoint is
# already saved in, so a plain copy is enough).
os.makedirs(cfg.curr_model_dir, exist_ok=True)
dest = os.path.join(cfg.curr_model_dir, "model.pt")
if not os.path.exists(dest):
    shutil.copy(os.path.join(BASE_CKPT_DIR, "model.pt"), dest)
    print(f"[train_stairs] staged base checkpoint at {dest}")

print(f"[train_stairs] curr_model_dir = {cfg.curr_model_dir}")
print(f"[train_stairs] data_dir       = {cfg.data_dir}")

trainer = Trainer(cfg)
trainer.train()
trainer.save_config()
print("[train_stairs] DONE. Fine-tuned checkpoint + model.yaml at:", cfg.curr_model_dir)
