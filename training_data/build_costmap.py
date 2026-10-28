"""Build and save the semantic cost map for the stairs scene from cloud.ply (see build_cloud.py),
using viplanner_core's own cost_builder.py pipeline directly instead of reimplementing it.

Per explicit user request: wall collisions must be penalized far more heavily than the library's
shared default (OBSTACLE_LOSS=2.0, the same value used for every obstacle class). Patched here, in
this script's own process only -- NOT written back into viplanner_sem_meta.py, since that's shared
library code other scenes/classes also use "wall" from.
"""
import os
import sys

sys.path.insert(0, "/home/itri/Desktop/viplanner2/viplanner_core")

from viplanner.config import viplanner_sem_meta  # noqa: E402

WALL_LOSS = 20.0  # vs. the library default OBSTACLE_LOSS=2.0 -- walls (and persons, tagged as the
# same "wall" class in this scene, see camera_semantics_lab.py) must never be an acceptable path
# "fence" = railings (camera_semantics_lab.SEG_ID_RAILING): exactly as non-walkable as walls
for _entry in viplanner_sem_meta.VIPLANNER_SEM_META:
    if _entry["name"] in ("wall", "fence"):
        print(f"[build_costmap] patching '{_entry['name']}' class loss {_entry['loss']} -> {WALL_LOSS}")
        _entry["loss"] = WALL_LOSS

from viplanner.config import CostMapConfig  # noqa: E402
from viplanner import cost_builder  # noqa: E402

# optional argv[1]: scene directory (default: the original stairs_scene)
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/itri/Desktop/viplanner2/training_data/stairs_scene"

cfg = CostMapConfig()
cfg.semantics = True
cfg.geometry = False
cfg.map_name = "cost_map_sem"
cfg.general.root_path = ROOT
cfg.general.ply_file = "cloud.ply"
cfg.sem_cost_map.compute_height_map = False  # flat indoor floor (+ stairs as discrete steps, not
# continuous terrain) -- matches the matterport/warehouse convention this option is meant for

# Per explicit user request: the network should learn Go2-appropriate wall clearance from
# depth+semantics alone (no hand-coded footprint check at inference time), not just be told "walls
# are bad" via WALL_LOSS above. _dense_grid_loss (sem_cost_map.py) already builds a genuinely
# graded, EDT-based cost field -- not a hard binary wall/free split -- and these two sigma_smooth
# values are what actually control how far that graded cost extends from a wall into the walkable
# area (both are Gaussian-blur std devs, in GRID CELLS at cfg.general.resolution=0.05m/cell, i.e.
# 1 cell = 5cm). Defaults (2.5 and 3.0 cells = 12.5cm and 15cm) were never sized for this robot --
# Go2's own half-diagonal is ~0.354m (half-length 0.325m, half-width 0.14m, see
# GO2_HALF_DIAGONAL in isaac_sim_bridge/viplanner_control.py), well past where the default's
# elevated cost has already decayed to nothing. Raised to 6 cells (~0.3m std dev) here: still a
# SOFT gradient (a Gaussian's cost is nonzero for several std devs out, so this doesn't hard-block
# anything), but now centered on the robot's actual scale instead of a generic default -- expect
# the network to end up preferring more clearance where the map is open and accepting less where a
# corridor genuinely doesn't offer more, rather than a fixed inflation radius either blocking a
# tight-but-passable corridor outright or being too thin to matter in the open.
cfg.sem_cost_map.sigma_smooth = 6.0
cfg.general.sigma_smooth = 6.0

cfg.visualize = False

if __name__ == "__main__":
    cost_builder.main(cfg, final_viz=False)
