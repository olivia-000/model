# ViPlanner 全景分割路線 —— 實作與機器狗部署完整指南

本資料夾是「全景分割路線」的完整可執行實作：從離線驗證（CARLA）一路到四足機器人（機器狗）即時部署。
原理背景見 repo 根目錄的 `ARCHITECTURE.md` 與 `ViPlanner 走全景分割路線的完整架構.txt`。

```
viplanner全景/
├── README.md                  ← 本文件
├── panoptic_inference.py      # 階段1+2：RGB → 34類色碼語義圖（mask2former_env 執行）
├── compare_semantics.py       # GT vs 預測語義圖比對（viplanner env 執行）
├── run_pipeline.sh            # 離線 pipeline：兩個 env 自動串接
├── mask2former_env.yml        # 全景分割環境定義（conda env create -f 直接建；RTX 50 系列不適用，見 NEW_PC_SETUP.md 分支 B）
├── viplanner_env.yml          # viplanner 環境定義（從實機匯出，重建/搬機用）
├── NEW_PC_SETUP.md            # 新電腦交接檔：給 Claude Code 看的完整重建+驗證手冊（含 50 系列分支）
├── m2f_ckpt/                  # （執行 mim download 後產生）mmdet config + checkpoint
└── dog/                       # 機器狗即時部署
    ├── sem_seg_node.py        # 節點①：RGB topic → 語義圖 topic（mask2former_env）
    ├── viplanner_dog_node.py  # 節點②：depth+語義+goal+odom → Path+fear（viplanner env）
    ├── simple_path_follower.py# 節點③：Path → cmd_vel（Pure Pursuit，系統 python）
    └── run_dog.sh             # 一鍵啟動三節點
```

---

## 第 0 部分：原理 —— 為什麼是這個架構

### 0.1 全景分割在整條 pipeline 裡的角色

ViPlanner 網路（`DualAutoEncoder`）吃三個輸入：**深度圖（公尺）＋ 語義色碼圖 ＋ 目標點**，
吐出 **5 個路徑關鍵點 ＋ fear 碰撞風險分數**。它從頭到尾不做任何感知——語義理解全部
外包給上游。語義輸入的格式是死的：**一張 (H,W,3) uint8 RGB 圖，顏色必須精確等於
`viplanner/config/viplanner_sem_meta.py` 定義的 34 類色碼**。

所以「走全景分割路線」的意思只是：**把產生這張色碼圖的方法，從 CARLA ground truth
換成 Mask2Former 模型預測**。下游的深度處理、網路架構、輸出格式一個位元組都不變——
你已經在用的 `infer_single.py` 完全不用改。

### 0.2 全景分割的輸出編碼原理

mmdetection 的 Mask2Former（COCO-panoptic 版）輸出 `result.pred_panoptic_seg.sem_seg`，
shape `(1,H,W)`，每個像素是一個整數：

```
pixel = category_id * INSTANCE_OFFSET + instance_id     # INSTANCE_OFFSET 通常是 1000
```

這就是「全景（panoptic）」和「純語義（semantic）」分割的差別：同類別的不同物件
（兩台不同的車）會有不同 `instance_id`。**但 ViPlanner 只在乎「這一格是什麼類別」，
不在乎「是第幾台車」**，所以解碼時 `pixel % INSTANCE_OFFSET` 直接把 instance 資訊丟掉。
另外 `category_id == 133`（COCO 的 void/unlabeled）視為無效，回退成 ViPlanner 的
`static` 類（黑色，loss=2.0，當障礙物處理——保守但安全）。

> 那為什麼還要用全景模型而不是純語義模型？因為 COCO-panoptic 的 Mask2Former 是
> 現成、訓練充分、133 類同時涵蓋前景物件（person/car…）與背景區域（pavement/sky…）
> 的模型，原版 ViPlanner ROS 部署就是用它——類別覆蓋完整度才是重點，instance 只是順帶。

### 0.3 COCO 133 類 → ViPlanner 34 類的映射原理

`viplanner/config/coco_sem_meta.py::_COCO_MAPPING` 是一張手刻的**關鍵字**映射表
（例如 `"vehicle": ["car","bus","truck","boat"]`）。`get_class_for_id_mmdet()` 在
模型載入後，拿 `model.dataset_meta["classes"]`（**這顆 checkpoint 自己的類別順序**）
逐一比對關鍵字，動態建出 `coco_index → viplanner 類名` 字典，再查
`VIPlannerSemMetaHandler().class_color` 上色。

**為什麼是動態建表而不是寫死常數？** 因為 COCO 類別的「索引順序」是模型 metadata 的
一部分，不同 checkpoint / 不同 mmdet 版本可能不同。寫死就會在換模型時默默把 person
塗成 sky。這也是換 checkpoint 後第一件要做的驗證：跑一張圖、開 `--overlay` 肉眼看
顏色對不對。

### 0.4 為什麼要兩個 conda 環境（以及機器狗上怎麼解）

mmcv 的預編譯 wheel 落後新版 torch/CUDA 好幾個月；你的 `viplanner` env 是
torch 2.1x+cu128，硬塞 mmdet 進去幾乎必定觸發 mmcv 原始碼編譯地雷。而架構上，
全景分割和 ViPlanner 推論本來就是**兩個解耦的行程**（一個吃 RGB 吐語義圖，
一個吃語義圖+depth 吐軌跡），所以：

- **離線**：兩個 env，用**檔案系統**當介面（`run_pipeline.sh`）。
- **機器狗即時**：兩個 env，用 **ROS topic** 當介面（`dog/` 的節點①②）。
  這是同一個解耦思想的即時版——不需要把兩套 torch 塞進同一個行程。

### 0.5 座標系原理（部署時最容易錯的地方）

ViPlanner 全程使用**機器人慣例相機座標系：x 前方、y 左方、z 上方**（不是 ROS 光學
座標系的 z 前方/x 右方/y 下方！）。網路吃的 goal、吐的軌跡都在這個座標系。部署時的
轉換鏈：

```
goal (odom 世界座標)
  --[里程計位姿反變換]--> base 座標系（狗身，x前/y左/z上）
  --[扣相機安裝偏移 + 俯仰旋轉]--> 相機座標系 → 餵網路（z 強制設 0）
網路輸出軌跡（相機座標系）
  --[反向轉換]--> odom 座標系 → 發布 Path → 跟隨器追蹤
```

軌跡發布在 odom（世界）座標系的原因：狗在走的過程中路徑點不會跟著狗漂移，
跟隨器可以用最新里程計持續對同一條世界路徑做 Pure Pursuit。

### 0.6 深度輸入原理

深度永遠是**公尺級原始值**，不做正規化：非有限值→0、除以 `depth_scale`（uint16 mm
→ /1000）、超過 `max_depth`（15 m）→0。**深度圖必須和 RGB（→語義圖）像素對齊**——
訓練時兩台相機的視差是用逐像素 warp 補償的，部署時最簡單的等價做法是直接用
RealSense 的 `align_depth:=true`，讓深度圖對齊到彩色相機光心，語義圖和深度圖
天生逐像素對應。

---

## 第 1 部分：環境建置

兩個環境都有現成的 yml（本資料夾內），直接建即可：

```bash
conda env create -f mask2former_env.yml    # 全景分割環境（torch 2.1.2+cu121 + mmdet 3.3.0）
conda env create -f viplanner_env.yml      # viplanner 環境（從實機匯出；已有此環境者跳過）
```

### 1.1 mask2former_env（一次性）

用上面的 yml 建立（版本選擇原理寫在 yml 開頭註解：torch 刻意用 2.1.2+cu121 而非主環境的
2.11+cu128，確保 mmcv 抓得到官方預編譯 wheel 而不是觸發原始碼編譯）。建好後驗證：

```bash
conda activate mask2former_env
# 必須是 mmdet 3.x（2.x 沒有 mmdet.evaluation，會直接 import 失敗）
python -c "import mmdet; from mmdet.evaluation import INSTANCE_OFFSET; print(mmdet.__version__, INSTANCE_OFFSET)"
```

### 1.2 下載 Mask2Former COCO-panoptic 權重（一次性）

```bash
cd /home/itriu100/viplanner/viplanner全景
# 先確認 config 確切名稱（不要用猜的）
mim search mmdet --model mask2former
# 下載 config + checkpoint（R50 版約 200~300MB）
mim download mmdet --config mask2former_r50_8xb2-lsj-50e_coco-panoptic --dest ./m2f_ckpt
```

> 注意 `ros/planner/src/m2f_inference.py` 裡預設的
> `configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml` 是 detectron2
> 版命名慣例的佔位字串，**不能照抄**；mmdet 的 config 檔名長得像
> `mask2former_r50_8xb2-lsj-50e_coco-panoptic.py`。

### 1.3 viplanner env

現有環境維持現狀即可（你已能跑 `infer_single.py`）。模型放 `viplanner_models/model.pt` + `model.yaml`。

若要在新機器（例如機器狗的隨行運算機）上重建，用 `viplanner_env.yml`，**建完後必須
手動裝本地套件**（yml 裡刻意移除了 `viplanner==0.1.0`，因為它是本地程式碼不是 PyPI 套件）：

```bash
conda env create -f viplanner_env.yml
conda activate viplanner
cd /home/itriu100/viplanner && pip install -e .[standard]
```

---

## 第 2 部分：離線驗證（先在 CARLA 把誤差量出來，再上狗）

**原理**：真實部署時沒有 ground truth，全景分割猜錯就是猜錯。先在 CARLA 裡對同一個
畫面同時產生 GT 語義圖和模型預測語義圖，量化「換成真實分割模型後，語義輸入誤差多大、
軌跡/fear 漂移多少」——這個誤差就是上真狗後必然存在的雜訊底線。

### 步驟 2.1：產生測試資料（你已有的流程）

```bash
# CARLA 端：同時存 depth.npy + sem_viplanner.png(GT) + rgb_debug.png
python carla_capture_test_pair.py --output_dir ./test_scene_x ...
```

### 步驟 2.2：全景分割產生預測語義圖

```bash
conda activate mask2former_env
cd /home/itriu100/viplanner/viplanner全景
python panoptic_inference.py \
    --input  ../test_scene2_obstacle/rgb_debug.png \
    --output ../test_scene2_obstacle/sem_predicted.png \
    --config     ./m2f_ckpt/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py \
    --checkpoint ./m2f_ckpt/<下載到的>.pth \
    --overlay      # 產生半透明疊圖，先肉眼確認顏色/類別合理
```

### 步驟 2.3：兩張語義圖各跑一次 ViPlanner，比較

```bash
conda activate viplanner
cd /home/itriu100/viplanner

# GT 路線（基準）
python infer_single.py --model_dir ./viplanner_models \
    --depth ./test_scene2_obstacle/depth.npy \
    --semantic ./test_scene2_obstacle/sem_viplanner.png \
    --goal 13.0 0.0 0.0 --output r_gt.png --save_traj traj_gt.npy

# 全景分割路線
python infer_single.py --model_dir ./viplanner_models \
    --depth ./test_scene2_obstacle/depth.npy \
    --semantic ./test_scene2_obstacle/sem_predicted.png \
    --goal 13.0 0.0 0.0 --output r_pred.png --save_traj traj_pred.npy

# 量化比對：像素一致率、loss 等級一致率、逐類統計、軌跡漂移
python viplanner全景/compare_semantics.py \
    --gt ./test_scene2_obstacle/sem_viplanner.png \
    --pred ./test_scene2_obstacle/sem_predicted.png \
    --traj_gt traj_gt.npy --traj_pred traj_pred.npy \
    --output compare_scene2.png
```

`compare_semantics.py` 額外報一個「**loss 等級一致率**」：網路行為由類別的 loss 權重
決定（person 和 vehicle 都是 2.0，互認錯無所謂；sidewalk(0) 認成 road(1.5) 才會
改變路徑），這個指標比原始類別一致率更能預測軌跡漂移。

### 步驟 2.4（可選）：一條指令跑完整條 pipeline

```bash
cd /home/itriu100/viplanner/viplanner全景
./run_pipeline.sh ../test_scene2_obstacle/rgb_debug.png \
                  ../test_scene2_obstacle/depth.npy  13.0 0.0 0.0  ./out_scene2
```

**驗收標準建議**：多個場景（含障礙物、路口、行人）下軌跡終點偏移 < 0.5 m、
fear 判定（安全/危險）與 GT 路線一致，再進入機器狗部署。

---

## 第 3 部分：機器狗即時部署

### 3.1 硬體與前置需求

| 項目 | 需求 | 說明 |
|---|---|---|
| 相機 | RealSense D435/D455（或任何 RGB-D） | 建議掛狗頭/胸前，高度 0.3~0.5 m、pitch 0~15° 朝下——貼近訓練資料分佈（你 CARLA 測試也是 0.5 m） |
| 深度對齊 | `align_depth:=true` | `roslaunch realsense2_camera rs_camera.launch align_depth:=true`，深度 topic 用 `/camera/aligned_depth_to_color/image_raw`（16UC1，mm） |
| 里程計 | `/odom`（nav_msgs/Odometry） | 狗的驅動一般都有（腿式里程計或 VIO）；短程局部規劃允許慢漂移 |
| 速度介面 | `/cmd_vel`（geometry_msgs/Twist） | Unitree Go1/Go2 等的 ROS 驅動都支援 |
| 運算 | 一張 CUDA GPU | 桌機/工控機 OK；Jetson 見 3.6 |
| ROS | Noetic | 三個節點都是 ROS1 Python |

### 3.2 架構：三個節點、兩個環境、topic 當介面

```
/camera/color/image_raw ──▶ ①sem_seg_node.py（mask2former_env）
                                │  Mask2Former 全景分割 → 34類色碼
                                ▼
                          /viplanner/sem_image (Image rgb8)
                                │
/camera/aligned_depth_.. ──────▶│
/viplanner/goal (odom座標) ────▶ ②viplanner_dog_node.py（viplanner env）
/odom ─────────────────────────▶│  goal→相機座標、DualAutoEncoder 推論、
                                │  fear 防抖急停、軌跡→odom 座標
                                ▼
                          /viplanner/path (nav_msgs/Path, odom frame)
                          /viplanner/fear, /viplanner/status
                                │
/odom ─────────────────────────▶ ③simple_path_follower.py（系統 python）
                                │  Pure Pursuit
                                ▼
                          /cmd_vel ──▶ 機器狗運動控制器（步態由狗自己管）
```

設計要點（原理對應到程式）：

- **頻率解耦**：分割（節點①）大約 3~10 Hz，規劃（節點②）預設 5 Hz，跟隨（節點③）
  20 Hz。訂閱全部 `queue_size=1`，永遠只用最新一幀，慢的環節不會把延遲堆給快的環節。
  語義圖可以比深度圖「舊」個一兩百毫秒——語義描述的是場景結構，變化比深度慢，
  原版 ROS 節點也是這樣近似的。
- **fear 防抖**：`fear > fear_threshold`（0.5）**連續 3 幀**才發空路徑急停，單幀
  分割誤判不會讓狗急煞；空路徑同時是「到達目標」的訊號，跟隨器收到一律速度歸零。
- **不用 tf2 / cv_bridge**：兩者都是綁系統 python 的編譯套件，在 conda env 裡常出事。
  座標轉換用純 numpy 四元數（訂 `/odom` 自己算），影像用 `np.frombuffer` 手動編解碼。
- **goal 夾限**：goal 距離超過 `max_goal_clip`（10 m）會被夾到該方向的 10 m 處——
  訓練分佈 `max_goal_distance=15 m` 且大多樣本 ≤10 m，超出分佈的 goal 輸出不可信。
  遠目標的正確做法是外層再包一個全域規劃器逐段餵 goal。

#### 資料流完整走一遍：一筆資料從相機到狗腳

上面的圖是「誰接誰」，這裡按時間順序講「什麼東西、變成什麼、交給誰」。

**第 0 站：源頭。** 整條鏈有 4 個起點，全部以 ROS topic 送出：

| 來源 | Topic | 資料長相 |
|---|---|---|
| RealSense 彩色鏡頭 | `/camera/color/image_raw` | RGB 影像，~30 Hz |
| RealSense 深度（已對齊彩色） | `/camera/aligned_depth_to_color/image_raw` | 每像素一個距離值（mm），~30 Hz |
| 狗的定位系統 | `/odom` | 狗現在在世界的哪裡、朝哪個方向 |
| 人（或上層任務） | `/viplanner/goal` | 「走到世界座標的這個點」，只發一次 |

深度必須是 aligned to color——它跟 RGB 逐像素對齊，是後面深度和語義圖能配對使用的前提。

**第 1 站：RGB → 語義圖（節點①）。** 一張 RGB 照片進來，Mask2Former 全景分割出每個
像素的 COCO 類別（133 類）→ 查映射表換成 ViPlanner 34 類 → 每類塗上固定色碼，發布到
`/viplanner/sem_image`。這一站把「照片」翻譯成「規劃器看得懂的地面/障礙標記圖」，
原始照片到此丟棄，後面沒有人再看 RGB。它是全鏈最慢的一站，所以只追最新一幀、舊幀直接丟。

**第 2 站：四路匯合 → 一條軌跡（節點②）。** 唯一「多對一」的匯合點。四個 callback 只做
「存最新一筆」，運算集中在 5 Hz 主迴圈，每 0.2 秒醒來做四步：

1. **goal 世界座標 → 相機座標**：用**最新的** `/odom` 位姿換算（0.5 節的轉換鏈）。因為每
   週期都用最新位姿重算，狗越走近、換算出的 goal 自動越短——這就是「持續追同一個世界
   目標」的機制。距離 < 0.5 m 判定到達（發空路徑）；> 10 m 夾限（見上）。
2. **網路推論**：深度圖 `(1,1,H,W)`（「東西離我多遠」）＋ 語義圖 `(1,3,H,W)`（「哪些是路、
   哪些是障礙」）＋ goal `(1,3)`（「我想去哪」）餵進 `DualAutoEncoder`，吐出一串 3D 關鍵點
   （展開成密集路徑點）和一個 fear 分數。
3. **fear 安檢門**：連續 3 幀超標才急停（發空路徑 + `status=-1`），單幀誤判當雜訊放行。
4. **軌跡相機座標 → 世界座標**：用同一組轉換反向走回去，發布 `/viplanner/path`（odom frame）。
   發世界座標的原因：接下來 0.2 秒狗還在動，路徑釘在世界上，下游才能邊走邊修正。

**第 3 站：軌跡 → 速度指令（節點③）。** Pure Pursuit 以 20 Hz（比規劃快 4 倍）執行：
在路徑上找離狗 0.8 m 的「前視點」（像騎車時眼睛盯前方一點而不是盯輪子），算相對狗頭的
偏角 α；偏角 > 60° 先原地轉向，否則前進並按曲率公式配轉向、偏角越大越慢。**任何異常都
輸出零速度**：空路徑、路徑超過 1.5 s 沒更新（上游掛了）、沒有 odom——通通停。輸出
`/cmd_vel`（`linear.x` 前進速度 + `angular.z` 轉向角速度，就 2 個數字）。

**第 4 站：速度指令 → 腳步。** `/cmd_vel` 進到狗原廠的運動控制器，怎麼落腳、抬哪隻腿、
保持平衡全由狗自己處理——我們的節點到 `/cmd_vel` 為止，完全不管步態。

```
 RGB照片 ──①分割──▶ 語義色塊圖 ─┐
 深度圖 ─────────────────────┤
 goal(世界座標) ──用odom換算──▶ goal(相機座標) ─┤──②網路推論──▶ 軌跡(相機座標)+fear
 /odom ──────────────────────┘                      │
                                        fear連續超標？──是──▶ 空路徑(停)
                                              │否
                                   用odom反算 ▼
                              /viplanner/path (世界座標路徑)
                                              │
 /odom ──────────────▶ ③Pure Pursuit（20Hz 找前視點）
                                              │
                                        /cmd_vel (前進+轉向速度)
                                              │
                                   狗的運動控制器 → 腳步
```

貫穿全程的兩個觀念：

1. **資料形態一路「降維」**：照片（百萬像素）→ 語義圖（34 種顏色）→ 軌跡（幾十個點）→
   速度指令（2 個數字）→ 腳步。每站都把資訊濃縮成下一站剛好需要的形式，所以各站可以
   獨立替換（換分割模型、換跟隨器都不動其他站）。
2. **三種節奏靠「世界座標路徑」黏合**：分割盡力跑（幾 Hz）、規劃 5 Hz、控制 20 Hz。
   慢的環節不會卡住快的——規劃永遠拿當下最新的深度/語義；控制在兩次規劃之間，靠 20 Hz
   的最新 odom 對著釘在世界上的舊路徑持續修正。而「空路徑 + 1.5 s 逾時」保證任何一站
   斷掉，狗的預設行為都是停下來，不是暴走。

### 3.3 環境準備（在狗上/隨行運算機上）

```bash
# 兩個 conda env 內都要能 import rospy：
conda activate mask2former_env && pip install rospkg pyyaml catkin-pkg
conda activate viplanner       && pip install rospkg pyyaml catkin-pkg
# rospy 本體來自 source /opt/ros/noetic/setup.bash 後的 PYTHONPATH（純 python，py3.10 可用）
```

### 3.4 啟動

```bash
# 終端 A：相機
roslaunch realsense2_camera rs_camera.launch align_depth:=true

# 終端 B：狗的驅動（提供 /odom、吃 /cmd_vel）——依你的狗而定
# 例如 Unitree Go2: roslaunch go2_bringup bringup.launch

# 終端 C：三節點一鍵啟動（先改 run_dog.sh 頂部的相機安裝參數！）
cd /home/itriu100/viplanner/viplanner全景/dog
./run_dog.sh

# 終端 D：下目標點（odom 座標系，例：出發點前方 5 m）
rostopic pub -1 /viplanner/goal geometry_msgs/PointStamped \
  "{header: {frame_id: odom}, point: {x: 5.0, y: 0.0, z: 0.0}}"
```

`run_dog.sh` 頂部必改的參數：
- `CAM_OFFSET_X/Z`、`CAM_PITCH_DEG`：**實際量測**相機相對狗身中心的安裝位置與俯仰角，
  轉換鏈（0.5 節）全靠它，量錯 10 cm 軌跡就系統性偏 10 cm。
- `MAX_V`：第一次實測建議 0.3 m/s 以下。

### 3.5 分階段安全上機流程（強烈建議照順序）

1. **狗架起來離地**（或掛安全繩），跑完整 pipeline，用 rviz 看
   `/viplanner/path`、`/viplanner/sem_image`、`/viplanner/fear` 合不合理。
2. **只跑①②不跑③**（註解掉 follower），人推著/遙控狗走，確認路徑會繞開障礙物、
   fear 在正對牆時會飆高。
3. 接上③，空曠場地、`MAX_V=0.3`、目標 3~5 m，逐步加難度（障礙物、行人、窄道）。
4. 全程保留手動遙控最高優先權（多數狗的驅動有 mux；沒有就準備隨時拍急停）。

### 3.6 Jetson / 算力不足時

- 節點①是瓶頸（Mask2Former R50 在桌機 GPU 約 50~150 ms/幀，Jetson 上更慢）。
  對策：輸入先縮到 640×360 再分割（色碼圖之後反正會被 resize 到 `img_input_size`）、
  或換 mmdet 裡更輕的 panoptic 模型、或用 TensorRT 匯出。
- 節點②本體很輕（ResNet18×2，~10 ms），不會是瓶頸。
- 規劃頻率 2~3 Hz 對 0.5 m/s 的狗已足夠（局部路徑 5~10 m 長，重規劃間隔內狗只走幾十 cm）。

---

## 第 4 部分：常見地雷 checklist

- [ ] mmdet 必須 3.x（`from mmdet.evaluation import INSTANCE_OFFSET` 能過）。
- [ ] 換 Mask2Former checkpoint 後：跑一張圖開 `--overlay` 肉眼驗證映射
      （0.3 節——類別順序是 checkpoint 的 metadata，映射表是動態建的）。
- [ ] 深度 topic 用 **aligned**（`/camera/aligned_depth_to_color/...`），不是原始深度。
- [ ] RealSense 深度是 16UC1（mm），節點自動 /1000；若你的相機吐 32FC1 則已是公尺。
- [ ] goal 的 z 會被強制設 0；goal 定義在 odom 座標系，不是相機座標系。
- [ ] 相機安裝參數（offset/pitch）用量的，不用猜的。
- [ ] `model.yaml` 必須跟 `model.pt` 配對——所有 shape/scale 參數都從它讀，不要假設預設值。
- [ ] 語義色碼圖是 **RGB** 順序；用 OpenCV 存檔/讀檔時記得 `cvtColor`（本資料夾腳本都已處理）。
- [ ] 室內玻璃/強光下 RealSense 深度會破洞（值=0）——網路把 0 當「無資訊」，大面積破洞時
      fear 不可靠，避開逆光/玻璃多的場景或加裝遮光罩。
