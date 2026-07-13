# ViPlanner 完整架構總覽

本文整合 `CLAUDE.md`、`viplanner.txt`、`ViPlanner 走全景分割路線的完整架構.txt` 三份筆記，並對照原始碼（見各節標註路徑）逐項核實，作為單一、完整、由訓練到部署的架構參考。

---

## 全局資料流（一張圖）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 階段一：Cost Map 建置（離線，每個環境跑一次）                                │
│   語義點雲 / (depth+semantic 影像對) → 3D 點雲 → 可微分 cost grid           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │ maps/{cloud,data,params}/
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 階段二：資料生成 + 訓練（離線）                                             │
│   odom 圖搜尋取樣 start/goal → PlannerData → Trainer 用 cost map 算 loss   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │ model.pt + model.yaml
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 階段三：網路本體（訓練/推論共用）                                            │
│   depth + semantic(或rgb) + goal → DualAutoEncoder/AutoEncoder          │
│   → keypoints + fear → TrajOpt 內插成密集軌跡                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│ 部署 A：ROS（真實機器人 ANYmal）  │   │ 部署 B：Isaac Sim / IsaacLab extension│
│ viplanner_node.py 同時管兩顆模型│   │ VIPlannerAlgo（omniverse extension）│
│  ① VIPlannerInference = ViPlanner│  │ 語義輸入來自 IsaacLab 自己的       │
│  ② Mask2FormerInference(mmdet)  │   │ semantic_segmentation camera 標註器│
│    只負責產生語義輸入圖（全景分割）│  │ carla_cfg.py 是 CARLA USD 匯入場景 │
└───────────────────────────────┘   └───────────────────────────────────┘
```

**關鍵原則**：語義輸入圖永遠是「34 類 RGB 色碼 PNG」這個固定格式（`viplanner_sem_meta.py::VIPLANNER_SEM_META`）。不管來源是 CARLA ground truth、手刻 Cityscapes LUT，還是 Mask2Former 全景分割猜出來的，只要色碼對到這 34 類，對網路來說完全等價、可互換。**本專案的既定方向：正式部署走全景分割（mmdetection + Mask2Former COCO-panoptic），ground truth 只用於離線資料準備/驗證，不用於線上部署。**

---

## 階段一：Cost Map 建置

程式：`viplanner/cost_builder.py`（入口）、`viplanner/depth_reconstruct.py`、`viplanner/cost_maps/{sem_cost_map,tsdf_cost_map,cost_to_pcd}.py`，設定：`viplanner/config/costmap_cfg.py`。

### 1a. 點雲來源（`depth_reconstruct.py::DepthReconstruction`）
兩種輸入皆可匯聚成同一種輸出（`.ply` 點雲）：
- 真實世界：Open3D-Slam 產生的語義標註點雲（可直接跳過這一步)。
- 模擬環境：逐張 `depth/xxxx.png|.npy` + `semantics/xxxx.png` 影像對，搭配 `camera_extrinsic{depth_suffix,sem_suffix}.txt`（x,y,z,qx,qy,qz,qw）與 `intrinsics.txt`（ROS P-matrix）。

流程：depth 像素 → 相機內參反投影成方向向量（`_computePixelTensor`，轉成 x前/y左/z上機器人座標系）→ 乘上深度值、加上外參平移得到世界座標點 → 若有語義，把同一批 3D 點投影進語義相機座標系（`_get_semantic_image`：反轉語義相機外參、除以深度正規化、轉回相機 z-前方慣例、用語義內參投影、篩選視野內像素、丟棄「static/未分類」像素）→ 逐張累積進 `o3d.geometry.PointCloud`，每 `point_cloud_batch_size`（預設200）張做一次 `voxel_down_sample(voxel_size)`。輸出 `cloud.ply`。

`ReconstructionCfg` 關鍵欄位：`voxel_size`（Matterport 0.05 / CARLA 0.1）、`depth_scale=1000`、`max_images=1000`、`semantics=True`。

### 1b. Cost Grid 生成（`cost_builder.py::main`）
依 `CostMapConfig.semantics` / `.geometry` 二選一路線：

- **`SemCostMap`**（語義路線，實際使用的路線）：`pcd_init()` 讀 `.ply`、依 x/y 邊界裁切、`_set_map_parameters()` 算格點數（依 `resolution` 無條件進位到 10 的倍數）→ 選擇性建地面高度圖（KD-tree 補洞）→ `_pcd_filter()`（高度帶 + 統計離群點過濾）。`create_costmap()`：`_get_grid_loss()` 把每個點的顏色透過 `VIPlannerSemMetaHandler` 對回語義類別、指定該類的 loss 值，再做多進程 KD-tree 平滑；`_dense_grid_loss()` 對未分類格點用最近已分類鄰居補值（否則用 `OBSTACLE_LOSS`），高斯平滑後，依 loss 等級分段用歐氏距離轉換做梯度：可通行區域給負獎勵梯度（`negative_reward`，最後拉回 ≥0）、障礙區（`> obstacle_threshold * max_loss`）給對數縮放的距離代價、中間等級給線性內插梯度。
- **`TsdfCostMap`**（純幾何路線，備用/對照）：完全不看語義，只用 z 高度區分 `obs_points`（`ground_height*1.2` ~ `robot_height*robot_height_factor` 之間）與 `free_points`；`CreateTSDFMap()` 用高斯膨脹障礙物、二值化後做 `distance_transform_edt` + 對數縮放 + 高斯平滑。`ground_array` 目前是全零佔位（原始碼標註 TODO，未實作）。

兩者皆輸出 `[cost_array, viz_points, ground_array]` + 格點原點 `(x_start, y_start)`，包進 `CostMapPCD`（`cost_to_pcd.py`）：`Pos2Ind(points)` 把世界座標轉成 `[-1,1]` 正規化格點索引（供 `F.grid_sample` 查表用），`SaveTSDFMap()` 寫出：

```
<env>/maps/data/{map_name}_map.txt      # cost array
<env>/maps/data/{map_name}_ground.txt   # 地面高度
<env>/maps/cloud/{map_name}_cloud.txt   # 可視化點雲
<env>/maps/params/config_{map_name}.yaml  # 完整 CostMapConfig 存檔
```

`SemCostMapConfig.obstacle_threshold`（Matterport 0.5~0.6，CARLA 0.8）是「乘上最高類別 loss」的比例閾值——**這個閾值與 `viplanner_sem_meta.py` 裡各類別的 `loss` 權重是耦合的，改任一邊都要重新檢視另一邊，程式碼裡沒有斷言保護這個不變量**。

---

## 階段二：資料生成 + 訓練

程式：`viplanner/utils/dataset.py`、`viplanner/train.py`、`viplanner/utils/trainer.py`，設定：`viplanner/config/learning_cfg.py`。

### 2a. `PlannerDataGenerator`（取樣 start/goal pair，最耗時的一步）
每個環境目錄需要：`camera_extrinsic{depth_suffix}.txt`、`camera_extrinsic{sem_suffix}.txt`、`intrinsics.txt`、`depth/`、`semantics/`（或 `rgb/`）。流程：

1. `load_odom()` 讀里程計 → `filter_obs_inflation()` 用 cost map 的 `Pos2Ind`/`grid_sample` 濾掉落在障礙物膨脹區內的 odom 點（CARLA Town01 另外硬編碼排除幾個開闊區域框）。
2. `get_graph()`：KD-tree 找每個節點 3 個最近鄰（`num_connections=3`），每條邊內插 3 個中間點（`num_intermediate=3`），若中間點落在 tsdf 佔用格內就剔除這條邊——本質上是建一張「無碰撞可視圖」的 `networkx.Graph`。
3. `get_pairs()`：對每個 odom 點跑 Dijkstra，依路徑長度 cutoff（`max_goal_distance`，預設 15m）收集可達節點，用 `get_goal_categories()` 分成三類：
   - **within_fov**：`|atan2(y,x)| < fov/2 * fov_scale`
   - **front_of_robot**：`|angle| < π/2` 但不在視野內
   - **behind_robot**：其餘
   再依 `DataCfg.distance_scheme`（預設 `{1:0.2, 3:0.35, 5:0.25, 7.5:0.15, 10:0.05}`，key=距離桶、value=取樣比例）分桶取樣，並對每個 pair 做語義影像疊到深度影像上的處理（`compute_overlay`/`_get_overlay_img`，對應階段一同款 warp 邏輯）。
4. `reduce_pairs()` 每個 odom 點每個距離桶最多取 3 個樣本；`split_samples()` 依 `ratio_fov_samples/ratio_front_samples/ratio_back_samples` 及 `ratio`（train/val 分割，預設 0.9）組出最終 `PlannerData`。

### 2b. `PlannerData`（真正的 `torch.utils.data.Dataset`）
`__getitem__` 回傳 `(depth_image, sem_rgb_image, odom, goal, pair_augment)`：
- `depth_image`：`ToTensor + Resize(img_input_size)` 後的 `float32` tensor，shape `[1,H,W]`。載入時一律：非有限值設 0 → 除以 `depth_scale` → 超過 `max_depth` 的像素歸零，不做進一步正規化。
- `sem_rgb_image`：`semantics`/`rgb` 二擇一啟用時才是影像 tensor `[3,H,W]`（否則是 `0`），語義影像會正規化到 `[0,1]`。
- `odom`/`goal`：7 維（x,y,z,qx,qy,qz,qw）SE3 pose。
- `pair_augment`：是否做水平翻轉增強的旗標。

### 2c. `TrainCfg` / `DataCfg`（唯一的模型設定真相來源，序列化成 `model.yaml`）

`DataCfg` 重點欄位：`max_depth=15.0`、`depth_scale=1000.0`、`max_goal_distance=15.0`、`min_goal_distance=0.5`、`distance_scheme`（如上）、`fov_scale=1.0`、`ratio=0.9`、`pairs_per_image=4`，以及可選雜訊增強（`depth_salt_pepper`、`depth_random_polygons_nb`、`sem_rgb_pepper` 等）。

`TrainCfg` 重點欄位（訓練與推論都要靠 `model.yaml` 讀回這些值，**不可假設預設值**）：

| 類別 | 欄位 | 預設 |
|---|---|---|
| 輸入模式 | `sem` / `rgb`（互斥） | `True` / `False` |
| 網路形狀 | `img_input_size` | `[360,640]` |
| 網路形狀 | `in_channel`（goal 編碼維度） | `16` |
| 網路形狀 | `knodes` | `5` |
| RGB backbone | `pre_train_sem` / `pre_train_freeze` | `True` / `True` |
| RGB backbone | `pre_train_cfg` / `pre_train_weights` | Mask2Former COCO panoptic detectron2 config/pkl |
| Loss 權重 | `w_obs / w_height / w_motion / w_goal` | `0.25 / 1.0 / 1.5 / 4.0` |
| Loss 權重 | `fear_ahead_dist` / `obstacle_thread` | `2.5` / `1.2` |
| 優化器 | `optimizer` / `lr` / `momentum` / `w_decay` | `"sgd"` / `2e-3` / `0.1` / `1e-4` |
| 排程 | `factor` / `min_lr` / `patience` | `0.5` / `1e-5` / `3` |
| 訓練 | `epochs` / `batch_size` | `100` / `64` |
| 記錄 | `wb_project` / `wb_entity` | `"Matterport"` / `"viplanner"` |

注意：`fear_threshold`（推論時判斷高風險路徑的閾值，預設 0.5）**不在** `TrainCfg` 裡，是推論端（`VIPlannerAlgo`/ROS node）另外傳入的參數。

目錄慣例（`TrainCfg.file_path` 或環境變數 `EXPERIMENT_DIRECTORY`）：
```
data/<env_name>/         # 訓練資料 + cost map
models/<model_name>/     # model.pt + model.yaml
logs/<model_name>/       # wandb / 訓練紀錄
```

### 2d. 訓練迴圈（`Trainer`，`viplanner/utils/trainer.py`）
`_init_logging()`（wandb）→ `_load_model()`（`sem`或`rgb`為真則建 `DualAutoEncoder`，否則 `AutoEncoder`；強制 CUDA）→ `_configure_optimizer()`（SGD/Adam + `ReduceLROnPlateau`風格的 `EarlyStopScheduler`）→ `_load_data()`（每個環境各自的 `TrajCost` + `PlannerDataGenerator` + `TrajViz`）→ 逐 epoch 對每個環境的 dataloader 跑 `_train_epoch`/`_test_epoch`。

單步 loss：`preds → TrajOpt.TrajGeneratorFromPFreeRot(preds, step=0.1) → TrajCost.CostofTraj(...)`。每次驗證 loss 改善就 `torch.save((net.state_dict(), val_loss), model_path)`；`save_config()` 把 `{"config": vars(cfg), "loss": {...}}` 寫成 `model.yaml`。

---

## 階段三：網路本體

程式：`viplanner/plannernet/{PlannerNet,autoencoder,rgb_encoder}.py`，軌跡：`viplanner/traj_cost_opt/{traj_opt,traj_cost}.py`。

### 3a. Encoder / Decoder
- `PlannerNet(layers=[2,2,2,2])`：ResNet18 風格，`conv1=Conv2d(3,64,k=7,s=2,p=3)` + 4 個殘差階段（64/128/256/512 通道），**沒有**最後的 pooling/fc，直接回傳 feature map（輸入 360x640 時輸出約 `(N,512,12,20)`）。
- `AutoEncoder`（depth-only）：單一 `PlannerNet` 編碼器，depth 先 `.expand(-1,3,-1,-1)` 從 1 通道複製成 3 通道（**不是 RGB，只是通道複製**）。
- `DualAutoEncoder(train_cfg, m2f_cfg=None, weight_path=None)`：`encoder_depth` 固定是 `PlannerNet`；`encoder_sem` 依條件二選一——`rgb=True and pre_train_sem=True` 時是凍結的 `RGBEncoder`（detectron2 版 Mask2Former ResNet50 backbone，只取 `res5` 2048 通道特徵，`conv1: Conv2d(2048,512)` 降到 512 通道對齊 depth 特徵），否則也是一顆獨立的 `PlannerNet`（**這才是目前語義色碼路線實際用的分支**）。兩路特徵 `torch.cat(dim=1)` 得到 `(N,1024,12,20)`。
- `Decoder`/`DecoderS`：goal 只取 `(x,y,z)` → `Linear(3, goal_channels)` → 廣播到空間維度後 concat 進特徵圖 → 卷積+flatten+全連接 → `(N, knodes, 3)` 關鍵點；並行的 fear head 卷積/全連接 → sigmoid → `(N,1)`。`DecoderS` 是更小的變體（`decoder_small=True` 時使用）。

### 3b. 軌跡與 Loss（`traj_opt.py` / `traj_cost.py`）
- `TrajOpt.TrajGeneratorFromPFreeRot(preds, step=0.1)`：在 `preds` 前補一個原點，用自訂的 Hermite 基底三次樣條（`CubicSplineTorch`）在 `arange(0, num_p-1+step, step)` 上內插出密集軌跡點。訓練和推論都用同一支函式，只是訓練時內插的是「理想路徑」，推論時內插的是網路輸出的 keypoints。
- `TrajCost.CostofTraj(waypoints, odom, goal, fear, ...)` 總 loss = `collision_probability_loss(BCE) + w_obs*障礙loss + w_height*高度loss + w_motion*運動loss + w_goal*目標loss`：
  - 障礙 loss：把軌跡往左右各膨脹 `robot_width/2` 變成三條平行軌跡，用 `F.grid_sample`（bicubic）在 cost grid 上取值加總。
  - 高度 loss：取 cost map 的 `ground_array`，比對軌跡點高度與實際高度差（**注意 `TsdfCostMap` 路線的 `ground_array` 是全零佔位，這條 loss 在純幾何路線下不生效**）。
  - 運動 loss：比較實際軌跡逐步位移 vs. 一條指向 goal 的「理想直線路徑」，懲罰原地打轉/不前進。
  - 目標 loss：`log(||goal - 最終路徑點|| + 1)`。
  - fear 標籤：在 `ahead_dist` 範圍內對障礙 loss 做閾值化（`obstalce_thread + negative_reward`）產生二元標籤，跟網路輸出的 fear 算 BCE。

---

## 部署 A：ROS（真實機器人 / ANYmal，`ros/planner/src/`）

`viplanner_node.py::VIPlannerNode` 是編排者，**同時管兩顆彼此獨立、各自有自己 checkpoint/config 的模型**：

1. **`VIPlannerInference`**（`vip_inference.py`）——這才是 ViPlanner 本體：載入 `cfg.model_save` 下的 `model.pt`/`model.yaml`，依 `rgb or sem` 建 `DualAutoEncoder` 或 `AutoEncoder`。`plan(depth_image, sem_rgb_image, goal_robot_frame) -> (traj[N,3], fear)` 是 `spin()` 裡真正的推論呼叫；純 depth 模式用 `plan_depth(...)`。
2. **`Mask2FormerInference`**（`m2f_inference.py`）——**只負責產生上一步要吃的語義輸入圖**，跟 ViPlanner 本身的 PyTorch 推論完全解耦（一個吃 RGB 吐 PNG，一個吃 PNG+depth 吐軌跡）。用 **mmdetection** 的 `init_detector`/`inference_detector` 跑 COCO-panoptic Mask2Former；只在 `train_cfg.sem=True` 時才實例化。

### 節點內部順序（`spin()` / callback）
```
imageCallback(RGB) ──▶ semPrediction() ──▶ Mask2FormerInference.predict(bgr_image)
                                              │
                                              ▼ (H,W,3) uint8 RGB，34類色碼
depthCallback(depth) ──▶ 存最新 depth                │
goalCallback ──▶ 存最新 goal                          │
                                              │
spin() 主迴圈（cfg.main_freq）：確認 depth/sem/goal 都 ready
      → goalProjection()（世界座標→相機座標）
      → VIPlannerInference.plan(depth, sem_rgb, goal_cam_frame)
      → 相機座標→機器人座標 → fear>0.7 觸發 fearPathDetection
      → pubPath()（發布 nav_msgs/Path）
```

訂閱：`depth_topic`（`Image`）、`rgb_topic`（`Image`/`CompressedImage`，僅 `sem or rgb` 時訂閱）、`goal_topic`（`PointStamped`）、`/joy`、`{depth,rgb}_info_topic`。發布：`path_topic`（+`_viz`/`_fear`）、`/viplanner/status`、`/viplanner/timer`、`m2f_timer_topic`、`/viplanner/sem_image/compressed`。

### Mask2Former 全景分割解碼（`m2f_inference.py` + `viplanner/config/coco_sem_meta.py`）
```
result.pred_panoptic_seg.sem_seg  # (1,H,W) → (H,W)
每像素 = category_id * INSTANCE_OFFSET + instance_id   （mmdet 標準全景編碼）
curr_label = pixel % INSTANCE_OFFSET                    # 只要語義類別，丟棄 instance_id
category_id == 類別總數（COCO void/unlabeled）→ 回退成 ViPlanner "static" 類
get_class_for_id_mmdet(model.dataset_meta["classes"])   # 依 model 的類別順序建 coco_index → viplanner類名
  逐一比對 _COCO_MAPPING（關鍵字表，例如 "vehicle": ["car","bus","truck","boat"]）
VIPlannerSemMetaHandler().class_color[viplanner_class_name]  # 上色
→ (H,W,3) uint8 RGB，跟 CARLA ground truth 的 sem_viplanner.png 同格式，可互換
```

**注意事項**（部署時務必核對，非一次性設定）：
- `_COCO_MAPPING` / `get_class_for_id_mmdet` 依賴的是模型自己的 `dataset_meta["classes"]` **索引順序**，換 checkpoint 或換 mmdet 版本、類別順序不同時，**這張表必須重建**，不是固定常數。
- `from mmdet.evaluation import INSTANCE_OFFSET` 這條路徑只在 **mmdet 3.x** 存在（2.x 沒有 `mmdet.evaluation` 子模組），需確認裝的是 3.x。
- mmdetection 的 config 命名慣例（如 `mask2former_r50_8xb2-lsj-50e_coco-panoptic.py`）跟原版 detectron2 版 Mask2Former repo 不同，不要混用兩邊的 config 檔名。
- 建議把 mmdetection 隔離在獨立 conda env（例如 `mask2former_env`，torch 版本落在 mmcv 官方相容矩陣內，通常 2.1~2.4 / cu118~cu121），不要塞進主要的 `viplanner` 訓練/推論 env——這一步跟 ViPlanner 本身的 torch 推論是用檔案系統當介面的兩個獨立行程，本來就不需要活在同一個 Python 環境。

**容易搞混的兩套 Mask2Former，務必分清楚**：
| | 階段/位置 | 底層框架 | 用途 |
|---|---|---|---|
| 全景分割（本節主角） | ROS 部署，`m2f_inference.py` | **mmdetection** | 產生語義輸入圖，完整推論 |
| `rgb_encoder.py::RGBEncoder` | 網路本體，僅 `rgb=True and pre_train_sem=True` 時 | **detectron2** | 凍結 backbone 當特徵提取器，不做分割推論，只在你訓練/使用 RGB 模型時才會用到 |

走全景分割路線**只換掉「怎麼產生語義輸入圖」這一步**，下游的深度前處理、`DualAutoEncoder` 網路架構、輸出格式完全不變。

---

## 部署 B：Isaac Sim / IsaacLab extension（`omniverse/extension/omni.viplanner/`）

`viplanner_algo.py::VIPlannerAlgo(model_dir, fear_threshold=0.5, device="cuda")` 是 `vip_inference.py` 的 IsaacLab 版本對照：
- `load_model()`：讀 `model.pt`/`model.yaml`（`TrainCfg.from_yaml`），依 `sem` 建 `DualAutoEncoder`/`AutoEncoder`，`weights_only=True` 載入權重。
- `input_transformer(image)`：`Resize(img_input_size)` + 非有限值/`>max_depth` 歸零（跟 dataset.py 的深度前處理邏輯一致）。
- `plan(image, goal_robot_frame)`：depth-only；`plan_dual(dep_image, sem_image, goal_robot_frame)`：語義先 `/255` 正規化再進網路。兩者都接 `TrajGeneratorFromPFreeRot(keypoints, step=0.1)`。
- `goal_transformer`/`path_transformer`：用 IsaacLab 的四元數工具做相機座標↔世界座標轉換；`debug_draw` 用 `omni.isaac.debug_draw` 畫路徑（`fear > fear_threshold` 時畫紅色）。

依賴 `carb`/`omni.isaac.lab`，只能跑在 Isaac Sim 裡，跟 ROS 路徑完全獨立、不共用程式碼。

### `carla_cfg.py`：CARLA 場景匯入 IsaacLab（跟「用 CARLA 原生模擬器」是兩碼事）
`ViPlannerCarlaCfg` 把 CARLA 匯出的 `carla.usd` 地形透過 `UnRealImporterCfg` 匯入 IsaacLab 場景（搭配 `cw_multiply_cfg.yml`/`keyword_mapping.yml`/`people_cfg.yml`/`vehicle_cfg.yml` 等 CARLA 專屬映射檔），並用 **IsaacLab 自己的 `semantic_camera`**（`CameraCfg`，`data_types=["semantic_segmentation","rgb"]`，`colorize_semantic_segmentation=False`，即輸出原始 class-id 而非上色圖）做語義標註，**不是**呼叫 CARLA 模擬器原生 Python API 的 `sensor.camera.semantic_segmentation`（那個是 0~28 固定類別 ID，寫在 BGRA buffer 的 R channel）。兩者輸出的類別體系、取值方式都不同，改動前要先確認手上的腳本是走哪一條。

---

## 語義輸入格式（全域一致，`sem=True` 時任何路徑都適用）

`viplanner/config/viplanner_sem_meta.py::VIPLANNER_SEM_META`：34 類清單，每類一個 dict：
```python
{"name": "sidewalk", "loss": 0,   "color": [0, 255, 0],  "ground": True}
{"name": "road",     "loss": 1.5, "color": [255,128,0],  "ground": True}
{"name": "person",   "loss": 2.0, "color": [255,0,0],    "ground": False}
{"name": "static",   "loss": 2.0, "color": [0,0,0],       "ground": False}  # 未分類/catch-all
```
`OBSTACLE_LOSS=2.0`、`ROAD_LOSS=1.5`、`TERRAIN_LOSS=1.0`、`TRAVERSABLE_UNINTENDED_LOSS=0.5`、`TRAVERSABLE_INTENDED_LOSS=0`。`VIPlannerSemMetaHandler` 在初始化時建好 `class_loss`/`class_color`/`class_ground`/`class_id` 四張查表字典，`get_colors_for_names()` 是常用的輔助函式。

任何上游分割來源（CARLA 原生 tag、Cityscapes trainId、COCO-panoptic）都必須先轉換成這套固定的 34 類 RGB 色碼，才能餵給網路。

---

## 深度格式（全域一致）

單通道影像：`.png`（uint16，除以 `depth_scale`，預設 1000.0 即毫米）或 `.npy`（float，假設已是公尺——兩者都有時優先用 `.npy`，較精確）。載入一律：非有限值設 0 → 除以 `depth_scale` → 超過 `max_depth`（預設 15m）的像素歸零，不做進一步正規化，直接把公尺數值餵進網路。`real_world_data=True` 會多做一次 180° 影像旋轉（真實相機安裝方式的平台特性），模擬資料不適用。

---

## 各段程式碼速查表

| 階段 | 主要程式 | 設定 |
|---|---|---|
| Cost map 建置 | `cost_builder.py`, `depth_reconstruct.py`, `cost_maps/{sem_cost_map,tsdf_cost_map,cost_to_pcd}.py` | `config/costmap_cfg.py` |
| 資料生成 | `utils/dataset.py`（`PlannerDataGenerator`, `PlannerData`） | `config/learning_cfg.py::DataCfg` |
| 訓練 | `train.py`, `utils/trainer.py::Trainer` | `config/learning_cfg.py::TrainCfg` |
| 網路 | `plannernet/{PlannerNet,autoencoder,rgb_encoder}.py` | `TrainCfg`（`img_input_size`/`in_channel`/`knodes`/`sem`/`rgb`/`pre_train_*`） |
| 軌跡/Loss | `traj_cost_opt/{traj_opt,traj_cost}.py` | `TrainCfg`（`w_obs`/`w_height`/`w_motion`/`w_goal`/`fear_ahead_dist`） |
| ROS 部署 | `ros/planner/src/{viplanner_node,vip_inference,m2f_inference}.py` | ROS launch cfg + `model.yaml` |
| Isaac 部署 | `omniverse/extension/.../viplanner_algo.py`, `config/carla_cfg.py` | `model.yaml` + IsaacLab scene cfg |
| 語義色碼 | `config/viplanner_sem_meta.py`, `config/coco_sem_meta.py` | — |
