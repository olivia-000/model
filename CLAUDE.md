# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ViPlanner is a learning-based local path planner that takes depth + semantic (or RGB) camera images and a goal
point, and outputs a local trajectory plus a "fear" (collision risk) score. It is trained entirely in simulation
(Matterport3D indoor meshes, CARLA town meshes, NVIDIA Warehouse) and deployed either as a ROS Noetic node on real
legged robots (ANYmal) or as an NVIDIA Isaac Sim / IsaacLab extension. Repo: https://github.com/pascal-roth/viplanner.

## Install

```bash
pip install .                          # base install
pip install -e .[standard]             # editable, adds pypose (needed for training/data generation)
pip install -e .[standard,inference]   # adds mmcv/mmengine/mmdet for the Mask2Former-based ROS semantic segmentation node
pip install -e .[inference,jetson]     # Jetson target (pins torch==1.11, since mmdet needs torch.distributed built against it)
```

- Requires CUDA toolkit matching the version torch was compiled with (repo assumes 11.7); set `CUDA_HOME` if not
  auto-detected.
- The RGB-input training path (as opposed to the published semantic-input planner) needs the `third_party/mask2former`
  git submodule plus detectron2 and a compiled CUDA pixel-decoder op:
  ```bash
  pip install git+https://github.com/facebookresearch/detectron2.git
  git submodule update --init
  pip install -r third_party/mask2former/requirements.txt
  cd third_party/mask2former/mask2former/modeling/pixel_decoder/ops && sh make.sh
  ```
  This submodule is empty until `git submodule update --init` is run — do not assume it's populated.
- `mmcv` wheel builds often fail unless installed with an explicit CUDA/torch index, e.g.
  `pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html`. Prefer isolating
  mmdet/mmcv/detectron2 in their own environment rather than fighting version pins in the main training env — these
  are only needed for the optional real-RGB semantic-segmentation inference path, not for the core planner.

## Lint / format

```bash
./formatter.sh                # installs pre-commit if missing, runs `pre-commit run --all-files`
```

Hooks (see `.pre-commit-config.yaml`): black (line-length 120), flake8 (`.flake8`, max-complexity 30, docstring
convention google), isort (black profile), pyupgrade (py37+), codespell, license-header insertion
(`.github/LICENSE_HEADER.txt`). Run a single hook with `pre-commit run <hook-id> --files <path>`.

## Tests

There is no automated test suite in this repository (no `pytest`/test directory under version control). Validate
changes by running the relevant pipeline stage manually (cost-map build, `train.py`, or ROS node) against sample
data.

## Core architecture

The package is organized around a **three-stage pipeline**: build a differentiable cost map from labeled 3D data →
train a network to output trajectories that minimize that cost map's loss → deploy the trained network standalone
(depth+semantic in, keypoints+fear out).

### 1. Cost-map building (`viplanner/cost_maps/`, `viplanner/cost_builder.py`, `viplanner/depth_reconstruct.py`)
- Input is either a semantically-labeled pointcloud (real-world, e.g. from Open3D-Slam) or depth+semantic image pairs
  from simulation, which first go through `depth_reconstruct.py` (warps semantic image onto depth image accounting
  for differing camera frames, then voxelizes into 3D).
- `SemCostMap` / `TsdfCostMap` (`cost_maps/sem_cost_map.py`, `cost_maps/tsdf_cost_map.py`) turn the pointcloud into a
  differentiable cost grid; `CostMapPCD` (`cost_maps/cost_to_pcd.py`) is the shared storage/lookup format used by
  training. Config in `viplanner/config/costmap_cfg.py`. Per-class cost weights live in
  `viplanner/config/viplanner_sem_meta.py` (`OBSTACLE_LOSS`, `ROAD_LOSS`, `TERRAIN_LOSS`, etc.) — obstacle loss must
  stay above the general `obs_cost_height` used elsewhere, this invariant is asserted nowhere so don't break it.
- Output cost maps are written under `<env>/maps/{cloud,data,params}/` next to the environment's source data.

### 2. Data / training (`viplanner/utils/dataset.py`, `viplanner/train.py`, `viplanner/config/learning_cfg.py`)
- `PlannerDataGenerator` reads a per-environment directory (`camera_extrinsic{depth_suffix,sem_suffix}.txt`,
  `intrinsics.txt` as ROS P-matrices, `depth/`, `semantics/` or `rgb/`), builds a connectivity graph over collision-free
  odometry points using the cost map, and samples start/goal pairs according to a distance scheme
  (`DataCfg.distance_scheme`) split into "within FOV" / "in front" / "behind robot" categories. This is the expensive
  step — it does full odom-goal graph search and semantic image re-warping per environment.
- `PlannerData` is the actual `torch.utils.data.Dataset` consumed by training; depth images are always divided by
  `depth_scale` (mm→m) and clipped at `max_depth`; semantic/RGB images are resized and (for semantics) normalized to
  `[0,1]`. `semantics` and `rgb` inputs are mutually exclusive (`DataCfg`/`PlannerData` assert this).
- `TrainCfg` (`viplanner/config/learning_cfg.py`) is the single source of truth for a trained model — it's serialized
  to `model.yaml` alongside `model.pt` and must be loaded back to reconstruct the network at inference time (see
  `TrainCfg.from_yaml`, used by both the ROS node and `omniverse/.../viplanner_algo.py`). Never assume defaults;
  always read shape/scale parameters (`img_input_size`, `in_channel`, `knodes`, `depth_scale`, `max_depth`) from the
  model's own `model.yaml`.
- Training entrypoint: `python viplanner/train.py` (edit the `env_list_combi`/`TrainCfg(...)` block directly, there's
  no CLI). Uses `viplanner/utils/trainer.py::Trainer`. Directory layout expected under `TrainCfg.file_path` (or env
  var `EXPERIMENT_DIRECTORY`): `data/<env_name>/`, `models/<model_name>/`, `logs/<model_name>/`. Full structure
  documented in `TRAINING.md`.

### 3. Network (`viplanner/plannernet/`)
- `PlannerNet` (`PlannerNet.py`) is a ResNet18-style encoder (3-channel input) used for both the depth branch and,
  when not using a pretrained RGB backbone, the semantic branch.
- `AutoEncoder` (depth-only) and `DualAutoEncoder` (depth + semantic/RGB) in `autoencoder.py` are the actual planner
  networks. Depth is always `expand`ed to 3 channels before the ResNet encoder (it's not RGB, just channel-repeated).
  When `TrainCfg.rgb and pre_train_sem`, the semantic branch is instead `RGBEncoder` (`rgb_encoder.py`), a frozen
  ResNet50 backbone loaded from Mask2Former COCO-panoptic pretrained weights (requires detectron2 +
  `third_party/mask2former`, see Install above) — this only supplies feature weights, it is not doing segmentation
  inference itself.
- `Decoder`/`DecoderS` fuse the goal point (only x,y,z used) with encoder features via concatenation, then regress
  `knodes` 3D keypoints plus a sigmoid "fear" score. `viplanner/traj_cost_opt/traj_opt.py::TrajOpt` turns keypoints
  into a dense trajectory (cubic-spline-like `TrajGeneratorFromPFreeRot`); `traj_cost.py` implements the loss used
  during training against the cost map.

### 4. Deployment — two independent, non-shared code paths
- **ROS** (`ros/planner/src/`): `viplanner_node.py` is an orchestrator that owns **two separate models** and runs
  them in sequence per frame — they are not one integrated network, and each has its own checkpoint/config pair:
  - `VIPlannerInference` (`vip_inference.py`) — loads `model.pt`/`model.yaml` (`cfg.model_save`) and wraps
    `AutoEncoder`/`DualAutoEncoder`, exactly like `omniverse/.../viplanner_algo.py::VIPlannerAlgo` does for Isaac Sim.
    This *is* ViPlanner: `vip_algo.plan(...)` is the actual trajectory+fear inference call in `spin()`.
  - `Mask2FormerInference` (`m2f_inference.py`) — loads a completely separate checkpoint/config
    (`cfg.m2f_model_path`/`cfg.m2f_cfg_file`) and runs Mask2Former panoptic segmentation via **mmdetection**
    (`mmdet.apis`) on live RGB purely to produce the semantic *input image* for the step above. It is off-the-shelf
    perception, not part of the ViPlanner model — only instantiated when `train_cfg.sem` is true, called once per RGB
    frame from `imageCallback`/`semPrediction` before the depth+goal ever reach `vip_algo.plan`. Its output is decoded
    from COCO's 133-class panoptic encoding (`category_id = pixel % INSTANCE_OFFSET`), mapped to ViPlanner's 34
    classes via `viplanner/config/coco_sem_meta.py::get_class_for_id_mmdet`, then colorized via
    `viplanner_sem_meta.VIPlannerSemMetaHandler`. This is a different Mask2Former integration than the one in
    `rgb_encoder.py` (mmdetection here vs. detectron2 there) — don't conflate the two when changing either.
- **Isaac Sim / IsaacLab** (`omniverse/extension/omni.viplanner/`): `viplanner_algo.py::VIPlannerAlgo` is the
  IsaacLab-integrated equivalent of `vip_inference.py` (`load_model`, `plan`/`plan_dual`, `input_transformer`); it
  depends on `carb`/`omni.isaac.lab` and only runs inside Isaac Sim. `carla_cfg.py` defines a *separate* CARLA
  integration that imports a CARLA-exported town mesh as USD into Isaac Sim and uses IsaacLab's own
  semantic-segmentation camera annotator — this is unrelated to using the real CARLA simulator's Python API and its
  native `sensor.camera.semantic_segmentation` tag output directly (0–28 fixed class IDs in the R channel of the raw
  BGRA buffer). Know which one a given script is using before assuming a semantic label format.

## Semantic label format (applies everywhere `sem=True`)

The network never consumes raw class-ID label maps. The semantic *input image* is always an RGB PNG (or in-memory
array) colorized with the exact palette in `viplanner/config/viplanner_sem_meta.py::VIPLANNER_SEM_META` (34 classes,
each with a `color`, per-class `loss` weight used in cost-map/training, and a `ground` flag used during 3D
reconstruction). Any upstream segmentation source (CARLA native tags, Cityscapes trainIds, COCO-panoptic via
Mask2Former) must be remapped into this exact 34-class RGB palette before it can be used as planner input — see
`viplanner/config/coco_sem_meta.py` (`_COCO_MAPPING`) for the COCO→ViPlanner keyword-based mapping used by the ROS
node.

## Project direction: panoptic-segmentation semantic-input pipeline

**Settled decision, not open for re-litigation:** the semantic *input image* for this project must be produced via
**panoptic segmentation** (Mask2Former COCO-panoptic via mmdetection), never plain/pure semantic segmentation and
never CARLA ground truth at deployment time. Ground truth / hand-built Cityscapes LUTs remain fine for offline
dataset prep, but the live/deployed pipeline direction is panoptic. Full walkthrough kept at
`ViPlanner 走全景分割路線的完整架構.txt` (repo root); summary below.

Pipeline (see `### 4. Deployment` above for the code-path framing — this is the same `Mask2FormerInference` /
`m2f_inference.py` stage, expanded):

1. **RGB → panoptic inference** (`ros/planner/src/m2f_inference.py`): `init_detector()` + `inference_detector()` on
   a BGR `np.ndarray` of any resolution. Output is `result.pred_panoptic_seg.sem_seg`, shape `(1,H,W)` → `(H,W)`,
   where each pixel = `category_id * INSTANCE_OFFSET + instance_id` (mmdet's standard panoptic encoding).
   ViPlanner discards `instance_id` entirely (`curr_label = curr_sem_class % INSTANCE_OFFSET`) — only the semantic
   category is used, instances are irrelevant to the planner. `category_id == num_classes` (COCO's void/unlabeled)
   falls back to ViPlanner's `"static"` class.
2. **COCO 133-class → ViPlanner 34-class mapping** (`viplanner/config/coco_sem_meta.py`): `get_class_for_id_mmdet()`
   walks `model.dataset_meta["classes"]` (index order = model-specific) against the keyword table `_COCO_MAPPING` to
   build `coco_index -> viplanner_class_name`. **This table must be rebuilt whenever the mmdet checkpoint/version
   changes class ordering** — it is not a fixed constant. Result is colorized via
   `VIPlannerSemMetaHandler().class_color[...]` into a `(H,W,3)` uint8 RGB image — this is the actual semantic input
   the network consumes, format-identical to and interchangeable with a CARLA-ground-truth `sem_viplanner.png`.
3. **Depth** is independent of this route and unchanged (see Depth format below).
4. **Network input tensors** (unchanged from the ground-truth path — this route only changes how the semantic
   tensor's pixels are sourced):

   | input | source | shape | normalization |
   |---|---|---|---|
   | depth | stage 3 | `(1,1,H,W)` | none (raw meters), `expand`ed to 3ch before the ResNet encoder |
   | semantic | stage 1+2 colorized panoptic mask | `(1,3,H,W)` | resized, then `/255.0` |
   | goal | caller-supplied, camera frame (x forward, y left, z up) | `(1,3)` | only (x,y,z) used, meters |

5. **Network forward** is exactly `DualAutoEncoder` as documented in `### 3. Network` above — no architecture
   changes for this route. Output is `keypoints (N, knodes, 3)` + `fear (N, 1)` sigmoid, fed to
   `TrajOpt.TrajGeneratorFromPFreeRot(keypoints, step=0.1)` for the dense trajectory, same as always.

Do not conflate this with `rgb_encoder.py::RGBEncoder` (detectron2-based, frozen Mask2Former backbone used only as
a feature extractor when `train_cfg.rgb=True and pre_train_sem=True`) — separate integration, separate environment,
unaffected by this pipeline choice. See `### 3. Network` and `### 4. Deployment` above for that distinction.

## Depth format (applies everywhere)

Depth is a single-channel image, either `.png` (uint16, raw units — `DataCfg.depth_scale`, default `1000.0`, i.e.
millimeters) or `.npy` (float, assumed already in meters — preferred/more precise when both exist, per
`PlannerDataGenerator.load_images`). Loading always: replace non-finite values with 0, divide by `depth_scale`, zero
out anything beyond `DataCfg.max_depth` (default 15m). No further normalization is applied — raw metric depth values
are fed to the network. `real_world_data=True` triggers a 180° image rotation on load (platform-specific camera
mounting quirk) that does not apply to simulated data.
