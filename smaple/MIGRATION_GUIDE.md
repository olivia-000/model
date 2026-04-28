# AC 路面 + 橋樑偵測系統 — ROS1 移植指南

> **適用對象**：要把 `firetruck_detection/`（PyQt5 + 多執行緒）移植到 ROS1 多節點架構的工程師。
> **前提**：你不需要懂 ROS，這份文件會把該知道的 ROS 知識一次講清楚。實際撰寫程式時請用 Claude 搭配本文件逐段執行。

---

## 0. 任務一句話

把目前 Firetruck 專案的「橋樑 Stage1 + 兩個 Stage2 分類 + AC 路面 Stage1」整套偵測邏輯，**搬到 ROS1 上跑**，
照 `smaple/Yolov8_ros/` 的 ROS package 格式建立，**只做一個整合節點**（不是每路相機各一個），
**不要做 GUI、不要做 LLM、不要做 email/web_api**。

---

## ⚠ 0.1 兩條最高優先約束（先看這個）

### 約束 A：輸出格式必須與 sample 一致（**硬規定**）

對外發布的 **topic 名稱、msg 結構、欄位名稱**全部照 sample，**任何一個欄位都不准動**。

**為什麼**：下游團隊會接一個 GPS 節點，訂閱我們的 `BoundingBoxes` / `BoundingBoxesWithImage`，再加上座標資訊後重新發布。我們改任何欄位都會把下游打掛。

**具體要求**：
- `BoundingBox.msg`、`BoundingBoxes.msg`、`BoundingBoxesWithImage.msg` 三個檔案 **原封不動**
- topic 命名照 sample（見第 5 節對應表）
- 即使整合節點內部架構與 sample 不同，**對外介面 100% 相同**
- 想加新欄位？先跟下游確認，不要自己改

### 約束 B：本次不做 GUI

驗證一律用 `rqt_image_view` / `rostopic` 看，**不要寫任何視窗程式**。GUI 留給未來。

---

## 1. 你會用到的兩個來源

| 來源 | 路徑 | 用途 |
|---|---|---|
| **Sample 範本** | `smaple/Yolov8_ros/` | ROS package 結構、自訂 msg、ROS 寫法的範例 |
| **既有專案** | `firetruck_detection/` | 偵測邏輯、模型權重、per-channel 規則的真實出處 |

**閱讀建議**：先看 `smaple/Yolov8_ros/yolov8_ros/scripts/yolov8_multi_node_2_phase.py` 了解 ROS node 怎麼寫；
再看 `firetruck_detection/cfg/default_settings.yaml` 與 `firetruck_detection/utils/__init__.py` 了解我們要保留哪些邏輯。

---

## 2. ROS1 速成（沒碰過 ROS 的人請先看這節）

### 2.1 ROS 是什麼
ROS1（Noetic）是一套**多進程通訊框架**。每個 process 叫做 **node**，node 之間用 **topic**（pub/sub）或 **service**（call/response）溝通。
你寫一支 Python 腳本，加上 `rospy.init_node(...)` 與 publisher/subscriber，它就是個 node。

### 2.2 核心名詞對照

| ROS 名詞 | 對應的東西 |
|---|---|
| Node | 一個 process（你寫的 .py 程式） |
| Topic | 一個具名通道，例如 `/cam/front_120/raw` |
| Message (`.msg`) | Topic 上傳的資料結構（類似 dataclass） |
| Publisher | 寫資料到 topic |
| Subscriber | 收 topic 上的資料，觸發 callback |
| Package | 一個資料夾，含 `package.xml` + `CMakeLists.txt`，是 ROS 編譯的最小單位 |
| Workspace | 含 `src/` 的根目錄，`catkin_make` 編譯後會產生 `build/` 與 `devel/` |
| Launch file | `.launch` 是 XML，描述「同時啟動哪些 node、各自帶什麼參數」 |

### 2.3 自訂 Message 為什麼存在
ROS 內建的 message（如 `sensor_msgs/Image`、`std_msgs/Header`）不夠用。
我們要傳「一張影像 + 一堆 bbox」這種組合，就要自己定義 `.msg`。
ROS 用 `catkin_make` 把 `.msg` 自動編譯成 Python class，你 `from yolov8_ros_msgs.msg import BoundingBox` 就能用。

### 2.4 Workspace 結構（你要在電腦上建立的）
```
~/catkin_ws/
├── src/                          ← 把 smaple/Yolov8_ros/ 整個資料夾搬到這裡
│   └── Yolov8_ros/
│       ├── yolov8_ros/
│       ├── yolov8_ros_msgs/
│       └── yolov8_ros_box_image_msgs/
├── build/                        ← catkin_make 產生
└── devel/                        ← catkin_make 產生
```

### 2.5 必備指令（記下來）
```bash
# 編譯（catkin 工具用 python3.8，節點本身用 conda python3.10）
cd /catkin_bridge_ws && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8

# 每次新開 terminal 都要 source（讓系統找得到 package）
source /catkin_bridge_ws/devel/setup.bash

# 啟動偵測節點（LD_PRELOAD 修正 libffi 衝突，device:=cuda 使用 RTX 5090）
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 \
  roslaunch yolov8_ros yolo_v8_multi_camera_unified.launch device:=cuda

# 播 rosbag（-l 循環，-r 0.5 半速）
rosbag play -l -r 0.5 /workspace/your_recording.bag

# 看有哪些 topic
rostopic list

# 看某個 topic 的內容
rostopic echo /yolov8/front_120/BoundingBoxes
```

---

## 3. 架構決策（為什麼是「一個節點」而不是「每路一個」）

Sample 範本是「**每路相機各一個 node**」（5 個 bridge node + 2 個 road damage node = 7 個 process）。
**我們不照做**，原因如下：

| 維度 | Sample 風格（每路一節點） | **本專案方案：整合節點** |
|---|---|---|
| 進程數 | 7 | **1** |
| 模型載入次數 | bridge × 5 + road × 2 = 7 份重複 | bridge / road / 2 個 stage2 各 1 份 |
| 影像同步 | 各路非同步 | `ApproximateTimeSynchronizer` 5 路一起進 callback |
| 跨相機合併視窗 | 需要再寫一個 node 拼接 | 在同節點內直接拼 |
| 對應 firetruck 既有架構 | 拆掉 shared inference thread | **幾乎 1:1 對應** `Detection._inference_worker` |
| GPU 記憶體（車載） | 吃光 | 省一倍以上 |

**結論**：寫一個 `multi_camera_detection_node.py`，一次處理 5 路。
Sample 的兩支 node（`yolov8_multi_node_2_phase.py`、`yolov8_multi_node_road_damage.py`）**保留當參考範本**，不啟用。

---

## 4. 三個 ROS Package（這些 sample 已經寫好，照搬即可）

### 4.1 `yolov8_ros_msgs` — 基礎 bbox 訊息
**位置**：`smaple/Yolov8_ros/yolov8_ros_msgs/`
**完全保留，不要改**（見約束 A — 下游 GPS 節點靠這個欄位結構接）。

#### 訊息檔
- `msg/BoundingBox.msg` — 單個 bbox：
  ```
  float64 probability
  int64 xmin
  int64 ymin
  int64 xmax
  int64 ymax
  string Class
  ```
- `msg/BoundingBoxes.msg` — 一張影像的所有 bbox：
  ```
  Header header
  Header image_header
  BoundingBox[] bounding_boxes
  ```

#### `package.xml` 重點
- `<buildtool_depend>catkin</buildtool_depend>` — 編譯工具
- `<build_depend>std_msgs</build_depend>` — 因為 `Header` 來自 `std_msgs`
- `<exec_depend>std_msgs</exec_depend>` — 執行時也需要

#### `CMakeLists.txt` 重點（37 行的小檔案）
```cmake
find_package(catkin REQUIRED COMPONENTS std_msgs rospy message_generation)

add_message_files(
  DIRECTORY msg
  FILES
  BoundingBox.msg
  BoundingBoxes.msg
)

generate_messages(DEPENDENCIES std_msgs)
```
這三段是「**告訴 ROS 編譯系統去產生 Python class**」的標準寫法。

---

### 4.2 `yolov8_ros_box_image_msgs` — 影像 + bbox 組合訊息
**位置**：`smaple/Yolov8_ros/yolov8_ros_box_image_msgs/`
**完全保留，不要改**（見約束 A — 下游 GPS 節點靠這個欄位結構接）。

#### 訊息檔
- `msg/BoundingBoxesWithImage.msg`：
  ```
  Header header
  sensor_msgs/Image image
  yolov8_ros_msgs/BoundingBoxes bounding_boxes
  ```

這是「**過濾後的異常結果**」專用：發布時把標註過的影像 + 對應 bbox 一起送出，方便下游模組（例如儲存截圖、上傳）一次拿到。

#### 注意 `package.xml` 多了三行依賴
- `sensor_msgs`（因為含 `sensor_msgs/Image`）
- `yolov8_ros_msgs`（因為含 `BoundingBoxes`）
- `message_generation` / `message_runtime`

---

### 4.3 `yolov8_ros` — 主節點 package
**位置**：`smaple/Yolov8_ros/yolov8_ros/`
**這個 package 會新增檔案，但 sample 既有檔案保留當參考。**

#### 既有結構
```
yolov8_ros/
├── package.xml
├── CMakeLists.txt           ← 含模型權重自動下載邏輯，保留
├── scripts/
│   ├── yolov8_multi_node_2_phase.py       ← 參考用，不啟用
│   └── yolov8_multi_node_road_damage.py   ← 參考用，不啟用
├── launch/
│   ├── yolo_v8_multi_node_2_phase.launch          ← 參考用，不啟用
│   └── yolo_v8_multi_node_road_damage_2_cam.launch ← 參考用，不啟用
└── weights/                 ← .pt / .engine 檔案（CMakeLists 自動下載）
```

#### 你要新增的檔案
```
yolov8_ros/
├── scripts/
│   └── multi_camera_detection_node.py     ← 新增（本指南第 7 節）
├── launch/
│   └── yolo_v8_multi_camera_unified.launch ← 新增（本指南第 9 節）
└── config/
    └── channel_rules.yaml                  ← 新增（從 firetruck 移植 per-channel 規則）
```

---

## 5. Channel ↔ Camera ID ↔ 方向 對應表

這張表是後續所有檔案的單一真實來源（single source of truth）。

| firetruck `ch_key` | sample `camera_id` | 方向 | Camera Topic | 對應的 Stage2 |
|---|---|---|---|---|
| `ch1` | 0 | right | `/cam/right_120/raw` | height_cls |
| `ch2` | 1 | front | `/cam/front_120/raw` | （AC 路面） |
| `ch3` | 2 | left | `/cam/left_120/raw` | height_cls |
| `ch4` | 3 | back-down | `/cam/back_120_1/raw` | gap_cls（且只顯示 cls 6） |
| `ch5` | 4 | back | `/cam/back_120_2/raw` | （AC 路面） |

**規則**：`camera_id = N - 1`（ch 是 1-indexed，camera_id 是 0-indexed）。

每路發布的 topic（**名稱、型別、訊息欄位都不准改 — 下游 GPS 接這個**）：
- `/yolov8/{direction}/BoundingBoxes` — 全部 bbox（型別 `yolov8_ros_msgs/BoundingBoxes`）
- `/{direction}/detection_image` — 標註過的影像（型別 `sensor_msgs/Image`）
- `/{direction}/filter_image_bbox` — 只含異常類別（型別 `yolov8_ros_box_image_msgs/BoundingBoxesWithImage`）
- `/yolov8/{direction}/BoundingBoxes_road_damage` — 路面偵測 bbox（只 ch2/ch5）
- `/{direction}/detection_image_road_damage`（只 ch2/ch5）
- `/{direction}/filter_image_bbox_road_damage`（只 ch2/ch5）

---

## 6. 偵測邏輯（從 firetruck 移植過來的核心規則）

### 6.1 Bridge Stage1 模型（10 個類別）
`firetruck_detection/cfg/default_settings.yaml` 裡 `detection_modules.bridge` 用 `2026-04-01-yolov11s-2-phase`。

| cls | 名稱 | 異常? | 顯示色 |
|---|---|---|---|
| 0 | spalled concrete | 異常 | 紅 |
| 1 | corroded bolt | 異常 | 紅 |
| 2 | expansion joint | 正常 | 綠 |
| 3 | normal drainage hole | 正常 | 綠 |
| 4 | abnormal drainage hole | 異常 | 紅 |
| 5 | normal bolt | 正常 | 綠 |
| 6 | block expansion joint | 異常 | 紅 |
| 7 | rust steel pipe | 異常 | 紅 |
| 8 | barrier damage | 異常 | 紅 |
| 9 | blocked drainage hole | 異常 | 紅 |

**異常類別清單**：`[0, 1, 4, 6, 7, 8, 9]` ← `desired_class_ids`

### 6.2 Bridge Stage2（per-channel 條件式）
**觸發類別**：Stage1 偵測到 `cls in [2, 6]` 才跑 Stage2。

| 來源 channel | Stage2 模型 | 處理對象 | 判定 → Stage1 cls 改寫 |
|---|---|---|---|
| ch1 / ch3 | `2025-07-02-expansion_joint_height_cls` | **裁切後的 bbox** | `top1=0` (height_difference) → cls 改成 6 |
| ch4 | `2025-07-02-expansion_joint_gap_cls` | **整張影像** | `top1=0` (abnormal_expansion_joint) → cls 改成 6 |
| 其他 | （不跑 stage2） | — | — |

> 抄 `smaple/Yolov8_ros/yolov8_ros/scripts/yolov8_multi_node_2_phase.py:172-204` 的 `publish_results` 邏輯，按上表 per-channel 套。

### 6.3 ch4 額外規則
ch4 是 back-down 相機，**最終只顯示 cls 6**（block expansion joint）。
其他類別在 `filtered_detect_show` 與 `plot_bboxes` 都要過濾掉（見 sample script 第 262, 347 行）。

### 6.4 AC 路面 Stage1（4 類）
只跑在 **ch2 / ch5** 兩路。模型：`2026-03-03-yolov11-road-damage`。
類別 `[0, 1, 2, 3]` 全部視為「需通報」。

### 6.5 ch2 / ch5 路面額外過濾
從 firetruck `default_settings.yaml`：
```yaml
channel_rules:
  ch2:
    road_filter_bbox_y_ratio: 0.5
  ch5:
    road_filter_bbox_y_ratio: 0.5
```
**規則**：bbox 中心 y 若 `> 0.5 * image_height`（即偏下半部）就忽略。
（防止超大誤檢；見 firetruck `detector.py` 對應段落。）

> 注意：sample script 第 269-276 行有寫類似邏輯（針對 cls 4），但條件是「bbox_y > height/2 視為太高過濾」。
> 我們的需求是 **路面整體** 過濾，不限定 cls，請以 firetruck 的設定為準。

### 6.6 各 ch 實際 bbox 輸出規格（ROS 節點實作）

這是 `multi_camera_detection_node.py` 實際繪製的內容，為各 ch 的**單一真實來源**。

#### ch1（cam_id=0，right，右側鏡頭）
- **橋樑 Stage1**：畫出全部 10 類（綠色正常 cls 2/3/5，紅色異常 cls 0/1/4/6/7/8/9）
- **Stage2**：Stage1 偵測到 cls 2 或 6 時，裁切該 bbox 圖塊送入 `expansion_joint_height_cls`；`top1==0`（height_difference）→ 將該 box cls 改寫為 6（紅色）
- **額外文字疊加**：畫面左上角顯示 ch4 的 gap 分類結果文字，例如 `Gap: abnormal_expansion_joint`（紅）或 `Gap: normal_expansion_joint`（綠）
- **路面偵測**：無

#### ch2（cam_id=1，front，前方鏡頭）
- **橋樑 Stage1**：畫出全部 10 類
- **Stage2**：無（ch2 不在 height / gap 規則中）
- **路面損壞 Stage1**：`road-damage` 模型偵測 4 類（cls 0/1/2/3）；bbox 中心 y > 0.5 × 影像高度的框過濾掉

#### ch3（cam_id=2，left，左側鏡頭）
- **橋樑 Stage1**：畫出全部 10 類
- **Stage2**：同 ch1，對 cls 2/6 的裁切 bbox 跑高低差分類器，確認後改寫為 cls 6
- **額外文字疊加**：同 ch1，左上角疊加 ch4 gap 分類文字
- **路面偵測**：無

#### ch4（cam_id=3，back-down，後下方鏡頭）
- **橋樑 Stage1**：**只畫 cls 6**（block expansion joint，紅色）；其他 9 類全部不顯示
  - 根據 `channel_rules.ch4.bridge_allowed_class_ids: [6]` 設計，ch4 是俯拍伸縮縫的專用相機
- **Stage2**：Stage1 偵測到 cls 2 或 6 時，對**整張影像**（不裁切）送入 `expansion_joint_gap_cls`；`top1==0`（abnormal_expansion_joint）→ cls 改寫為 6
- **filter_image_bbox**：同樣只輸出 cls 6
- **路面偵測**：無

> **注意**：ch4 的 gap_cls 結果同時廣播給 ch1 / ch3 做文字疊加（在 `synced_callback` 提前跑一次，再傳入 `process_bridge_camera`）。

#### ch5（cam_id=4，back，後方鏡頭）
- **橋樑 Stage1**：畫出全部 10 類
- **Stage2**：無
- **路面損壞 Stage1**：同 ch2，4 類路面損壞，bbox 中心 y > 0.5H 過濾
- **filter_image_bbox**：只含異常類 {0, 1, 4, 6, 7, 8, 9}

---

## 7. 統一偵測節點設計（核心工作）

新檔案：`yolov8_ros/scripts/multi_camera_detection_node.py`

### 7.1 整體流程圖
```
                  ┌─ /cam/right_120/raw   (ch1)
camera publishers ├─ /cam/front_120/raw   (ch2)   ← 車上 driver / rosbag
(此節點不負責)    ├─ /cam/left_120/raw    (ch3)
                  ├─ /cam/back_120_1/raw  (ch4)
                  └─ /cam/back_120_2/raw  (ch5)
                                │
                                ▼
              message_filters.ApproximateTimeSynchronizer
                          (slop=0.05s)
                                │
              ┌─────────────────▼─────────────────────┐
              │  multi_camera_detection_node          │
              │                                        │
              │  [batch infer]                         │
              │   bridge_model([f1,f2,f3,f4,f5])      │
              │   road_model([f2,f5])                 │
              │                                        │
              │  [per-channel post-processing]        │
              │   ch1: filter bridge → height stage2  │
              │   ch2: bridge + road_damage(y filter) │
              │   ch3: filter bridge → height stage2  │
              │   ch4: bridge → gap stage2 → only cls6│
              │   ch5: bridge + road_damage(y filter) │
              │                                        │
              │  [stage2 異步: ThreadPoolExecutor]    │
              └─────────────────┬─────────────────────┘
                                │
              ┌─────────────────┴─────────────────────┐
              ▼                                       ▼
    per-camera publishers                  (選用) merged_view publisher
    /yolov8/{dir}/BoundingBoxes            /yolov8/merged_view
    /{dir}/detection_image
    /{dir}/filter_image_bbox
    /yolov8/{dir}/BoundingBoxes_road_damage  (只 ch2/ch5)
    /{dir}/detection_image_road_damage
    /{dir}/filter_image_bbox_road_damage
```

### 7.2 程式骨架（Claude 照這個結構填）

```python
#!/opt/conda/envs/yolov8_rtx5090/bin/python3.10
# -*- coding: utf-8 -*-
"""
Unified multi-camera detection node.

Subscribes to 5 camera image topics (time-synchronized via ApproximateTimeSynchronizer),
runs:
  - Bridge Stage1 batch inference on all 5 frames
  - AC road damage Stage1 on ch2/ch5
  - Per-channel Stage2 (height/gap classification) where applicable
Publishes per-camera detection results.
"""
import os
import copy
import yaml
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import rospy
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov8_ros_box_image_msgs.msg import BoundingBoxesWithImage


# ─── 常數（對應 firetruck utils/__init__.py） ──────────────────────────
NUM_CAMERAS = 5
CAMERA_DIRECTIONS = ['right', 'front', 'left', 'back-down', 'back']  # camera_id 0..4
TOPIC_NAMES = {
    0: 'right_120',
    1: 'front_120',
    2: 'left_120',
    3: 'back_120_1',
    4: 'back_120_2',
}

# Bridge cls map（見指南第 6.1 節）
BRIDGE_DESIRED_CLASS_IDS = [0, 1, 4, 6, 7, 8, 9]
BRIDGE_GREEN_CLASSES = {2, 3, 5}
BRIDGE_COLOR_MAP = {
    0: (0, 80, 247), 1: (0, 80, 247), 2: (133, 223, 2), 3: (133, 223, 2),
    4: (0, 80, 247), 5: (133, 223, 2), 6: (0, 80, 247), 7: (0, 80, 247),
    8: (0, 80, 247), 9: (0, 80, 247),
}

# Stage2 觸發
STAGE2_TRIGGER_CLS = [2, 6]
HEIGHT_CHANNELS = {0, 2}    # ch1, ch3
GAP_CHANNELS = {3}          # ch4
ROAD_CHANNELS = {1, 4}      # ch2, ch5
CH4_ALLOWED_CLS = {6}       # back-down 只顯示 cls 6


def main():
    rospy.init_node('multi_camera_detection_node', anonymous=False)

    # ─── 讀 ROS 參數 ─────────────────────────────────────────
    bridge_rt = rospy.get_param('~bridge_rt_weight')
    bridge_pt = rospy.get_param('~bridge_pt_weight')
    road_rt = rospy.get_param('~road_rt_weight')
    road_pt = rospy.get_param('~road_pt_weight')
    height_rt = rospy.get_param('~height_rt_weight')
    height_pt = rospy.get_param('~height_pt_weight')
    gap_rt = rospy.get_param('~gap_rt_weight')
    gap_pt = rospy.get_param('~gap_pt_weight')

    use_tensorrt = str(rospy.get_param('~tensorrt', 'False')).lower() in ('true', '1', 'yes')
    device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
    bridge_conf = rospy.get_param('~bridge_conf', 0.3)
    road_conf = rospy.get_param('~road_conf', 0.3)
    height_conf = rospy.get_param('~height_conf', 0.3)
    gap_conf = rospy.get_param('~gap_conf', 0.1)
    sync_slop = rospy.get_param('~sync_slop', 0.05)
    road_filter_y_ratio = rospy.get_param('~road_filter_y_ratio', 0.5)

    camera_topics = [rospy.get_param(f'~camera_topic_{i}') for i in range(NUM_CAMERAS)]

    # ─── 載入模型（每個只載一次） ────────────────────────────
    def load_model(rt_path, pt_path, task=None):
        path = rt_path if use_tensorrt else pt_path
        if not os.path.exists(path):
            rospy.logerr(f'Weight not found: {path}')
            raise FileNotFoundError(path)
        if use_tensorrt:
            return YOLO(path, task=task) if task else YOLO(path)
        m = YOLO(path).to(device)
        m.fuse()
        return m

    bridge_model = load_model(bridge_rt, bridge_pt)
    road_model = load_model(road_rt, road_pt)
    height_model = load_model(height_rt, height_pt, task='classify')
    gap_model = load_model(gap_rt, gap_pt, task='classify')

    # warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for m in (bridge_model, road_model, height_model, gap_model):
        _ = m(dummy, device=device, verbose=False)
    rospy.loginfo('All models loaded and warmed up.')

    # ─── Publisher（per-camera） ─────────────────────────────
    bridge_pubs = {}
    road_pubs = {}
    for cam_id in range(NUM_CAMERAS):
        topic = TOPIC_NAMES[cam_id]
        bridge_pubs[cam_id] = {
            'bbox': rospy.Publisher(f'/yolov8/{topic}/BoundingBoxes', BoundingBoxes, queue_size=1),
            'image': rospy.Publisher(f'/{topic}/detection_image', Image, queue_size=1),
            'filter': rospy.Publisher(f'/{topic}/filter_image_bbox', BoundingBoxesWithImage, queue_size=1),
        }
        if cam_id in ROAD_CHANNELS:
            road_pubs[cam_id] = {
                'bbox': rospy.Publisher(f'/yolov8/{topic}/BoundingBoxes_road_damage', BoundingBoxes, queue_size=1),
                'image': rospy.Publisher(f'/{topic}/detection_image_road_damage', Image, queue_size=1),
                'filter': rospy.Publisher(f'/{topic}/filter_image_bbox_road_damage', BoundingBoxesWithImage, queue_size=1),
            }

    bridge = CvBridge()
    executor = ThreadPoolExecutor(max_workers=8)  # bridge × 5 + stage2 × 3 + road × 2 同時可以丟

    # ─── Synchronized callback ──────────────────────────────
    def synced_callback(*image_msgs):
        """5 路 Image message 一起進來，做 batch inference 與 per-channel post-processing。"""
        try:
            frames = [bridge.imgmsg_to_cv2(m, desired_encoding='bgr8') for m in image_msgs]
        except Exception as e:
            rospy.logerr(f'cv_bridge error: {e}')
            return

        # ── Bridge Stage1: batch=5 ────────────────────────
        bridge_results = bridge_model(frames, conf=bridge_conf, device=device, verbose=False)
        # bridge_results 是 list，長度 = NUM_CAMERAS

        # ── Road Stage1: batch=2 (ch2, ch5) ───────────────
        road_input = [frames[1], frames[4]]
        road_results = road_model(road_input, conf=road_conf, device=device, verbose=False)
        road_result_map = {1: road_results[0], 4: road_results[1]}

        # ── Per-camera post-processing（丟到 executor） ──
        for cam_id in range(NUM_CAMERAS):
            executor.submit(
                process_bridge_camera,
                cam_id, frames[cam_id], copy.deepcopy(bridge_results[cam_id]),
                image_msgs[cam_id], height_model, gap_model, height_conf, gap_conf,
                bridge_pubs[cam_id], device,
            )
            if cam_id in ROAD_CHANNELS:
                executor.submit(
                    process_road_camera,
                    cam_id, frames[cam_id], copy.deepcopy(road_result_map[cam_id]),
                    image_msgs[cam_id], road_pubs[cam_id], road_filter_y_ratio,
                )

    # ─── Subscribers + ApproximateTimeSynchronizer ──────────
    subs = [message_filters.Subscriber(t, Image) for t in camera_topics]
    ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=sync_slop)
    ts.registerCallback(synced_callback)

    rospy.loginfo(f'Subscribing to: {camera_topics}')
    rospy.loginfo(f'sync slop: {sync_slop}s')
    rospy.spin()


def process_bridge_camera(cam_id, frame, result, image_msg, height_model, gap_model,
                           height_conf, gap_conf, pubs, device):
    """單路 bridge stage1 → 條件式 stage2 → 發布三個 topic"""
    # 1. Stage2（如果 cam_id 屬於 height/gap channel）
    if cam_id in HEIGHT_CHANNELS or cam_id in GAP_CHANNELS:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for i, cls in enumerate(classes):
            if int(cls) not in STAGE2_TRIGGER_CLS:
                continue
            if cam_id in HEIGHT_CHANNELS:
                x1, y1, x2, y2 = boxes[i]
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                sr = height_model(crop, device=device, verbose=False, conf=height_conf)
                if int(sr[0].probs.top1) == 0:  # height_difference
                    result.boxes.cls[i] = 6
            elif cam_id in GAP_CHANNELS:
                sr = gap_model(frame, device=device, verbose=False, conf=gap_conf)
                if int(sr[0].probs.top1) == 0:  # abnormal_expansion_joint
                    result.boxes.cls[i] = 6

    # 2. 發 BoundingBoxes + detection_image
    publish_bbox_and_image(cam_id, result, frame, image_msg, pubs,
                            allowed_cls=CH4_ALLOWED_CLS if cam_id == 3 else None,
                            color_map=BRIDGE_COLOR_MAP, green_classes=BRIDGE_GREEN_CLASSES)

    # 3. 發 filter_image_bbox（只含異常類別）
    publish_filtered(cam_id, result, frame, image_msg, pubs,
                     desired_cls=BRIDGE_DESIRED_CLASS_IDS,
                     allowed_cls=CH4_ALLOWED_CLS if cam_id == 3 else None,
                     color_map=BRIDGE_COLOR_MAP, green_classes=BRIDGE_GREEN_CLASSES)


def process_road_camera(cam_id, frame, result, image_msg, pubs, y_ratio):
    """ch2/ch5 路面 stage1 → bbox_y 過濾 → 發布"""
    # bbox_y > y_ratio * height 的過濾掉（見指南 6.5 節）
    h = frame.shape[0]
    keep_mask = result.boxes.xywh[:, 1].cpu().numpy() <= (y_ratio * h)
    # 套用 mask 到 result.boxes ...（依 ultralytics API 自行處理）

    publish_bbox_and_image(cam_id, result, frame, image_msg, pubs)
    publish_filtered(cam_id, result, frame, image_msg, pubs, desired_cls=[0, 1, 2, 3])


def publish_bbox_and_image(cam_id, result, frame, image_msg, pubs, **kwargs):
    """發 BoundingBoxes + 標註過的 Image。對照 sample script 第 210-242 行。"""
    # TODO: 實作（照 sample 的 detect_show）
    ...


def publish_filtered(cam_id, result, frame, image_msg, pubs, desired_cls, **kwargs):
    """發 BoundingBoxesWithImage（只含 desired_cls）。對照 sample script 第 245-313 行。"""
    # TODO: 實作（照 sample 的 filtered_detect_show）
    ...


if __name__ == '__main__':
    main()
```

> 上面是骨架，TODO 部分照 `smaple/Yolov8_ros/yolov8_ros/scripts/yolov8_multi_node_2_phase.py` 的 `detect_show` / `filtered_detect_show` / `plot_bboxes` 三個函式抄寫即可。

### 7.3 關鍵實作要點

1. **Batch inference**：`bridge_model([f1, f2, f3, f4, f5])` ultralytics 會自動 batch；TensorRT engine 必須是 batch=5 export 的（sample CMakeLists 已下載 `*-batch-5.engine`）。
2. **`copy.deepcopy(result)`**：因為要在 callback 後改 `result.boxes.cls`，避免動到原始 result。
3. **`ThreadPoolExecutor`**：照 sample 用，避免 callback 阻塞下一張影像進來。
4. **error handling**：`cv_bridge` / 模型推論都要 try/except，避免一次例外整個 node 掛掉。

---

## 8. 模型權重

從 `firetruck_detection/weights/` 複製這些到 `yolov8_ros/weights/`：

| 檔名 | 用途 |
|---|---|
| `2026-04-01-yolov11s-2-phase.pt` / `.engine` | bridge stage1 |
| `2026-03-03-yolov11-road-damage.pt` / `.engine` | road stage1 |
| `2025-07-02-expansion_joint_height_cls.pt` / `.engine` | stage2 height (ch1, ch3) |
| `2025-07-02-expansion_joint_gap_cls.pt` / `.engine` | stage2 gap (ch4) |

> sample 的 `CMakeLists.txt` 已寫好「自動從 itri 內網下載」邏輯（按 GPU 型號分流 RTX 4080/5080）。
> 如果可以連到 `dl.itriadv.co`，編譯時會自動把 weights 抓下來，不需手動複製。

---

## 9. Launch 檔（新增）

新增 `yolov8_ros/launch/yolo_v8_multi_camera_unified.launch`：

```xml
<?xml version="1.0" encoding="utf-8"?>
<launch>

  <!-- 模型權重路徑 -->
  <arg name="bridge_pt_weight" default="$(find yolov8_ros)/weights/2026-04-01-yolov11s-2-phase.pt" />
  <arg name="bridge_rt_weight" default="$(find yolov8_ros)/weights/2026-04-01-yolov11s-2-phase.engine" />
  <arg name="road_pt_weight"   default="$(find yolov8_ros)/weights/2026-03-03-yolov11-road-damage.pt" />
  <arg name="road_rt_weight"   default="$(find yolov8_ros)/weights/2026-03-03-yolov11-road-damage.engine" />
  <arg name="height_pt_weight" default="$(find yolov8_ros)/weights/2025-07-02-expansion_joint_height_cls.pt" />
  <arg name="height_rt_weight" default="$(find yolov8_ros)/weights/2025-07-02-expansion_joint_height_cls.engine" />
  <arg name="gap_pt_weight"    default="$(find yolov8_ros)/weights/2025-07-02-expansion_joint_gap_cls.pt" />
  <arg name="gap_rt_weight"    default="$(find yolov8_ros)/weights/2025-07-02-expansion_joint_gap_cls.engine" />

  <!-- 推論參數 -->
  <arg name="tensorrt"    default="False" />
  <arg name="device"      default="cuda" />
  <arg name="bridge_conf" default="0.3" />
  <arg name="road_conf"   default="0.3" />
  <arg name="height_conf" default="0.3" />
  <arg name="gap_conf"    default="0.1" />

  <!-- ApproximateTimeSynchronizer slop（秒） -->
  <arg name="sync_slop"   default="0.05" />

  <!-- Per-channel 規則 -->
  <arg name="road_filter_y_ratio" default="0.5" />

  <!-- 5 路 Camera Topic -->
  <arg name="camera_topic_0" default="/cam/right_120/raw" />
  <arg name="camera_topic_1" default="/cam/front_120/raw" />
  <arg name="camera_topic_2" default="/cam/left_120/raw" />
  <arg name="camera_topic_3" default="/cam/back_120_1/raw" />
  <arg name="camera_topic_4" default="/cam/back_120_2/raw" />

  <node pkg="yolov8_ros" type="multi_camera_detection_node.py"
        name="multi_camera_detection_node" output="screen">
    <param name="bridge_pt_weight" value="$(arg bridge_pt_weight)" />
    <param name="bridge_rt_weight" value="$(arg bridge_rt_weight)" />
    <param name="road_pt_weight"   value="$(arg road_pt_weight)" />
    <param name="road_rt_weight"   value="$(arg road_rt_weight)" />
    <param name="height_pt_weight" value="$(arg height_pt_weight)" />
    <param name="height_rt_weight" value="$(arg height_rt_weight)" />
    <param name="gap_pt_weight"    value="$(arg gap_pt_weight)" />
    <param name="gap_rt_weight"    value="$(arg gap_rt_weight)" />
    <param name="tensorrt"    value="$(arg tensorrt)" />
    <param name="device"      value="$(arg device)" />
    <param name="bridge_conf" value="$(arg bridge_conf)" />
    <param name="road_conf"   value="$(arg road_conf)" />
    <param name="height_conf" value="$(arg height_conf)" />
    <param name="gap_conf"    value="$(arg gap_conf)" />
    <param name="sync_slop"   value="$(arg sync_slop)" />
    <param name="road_filter_y_ratio" value="$(arg road_filter_y_ratio)" />
    <param name="camera_topic_0" value="$(arg camera_topic_0)" />
    <param name="camera_topic_1" value="$(arg camera_topic_1)" />
    <param name="camera_topic_2" value="$(arg camera_topic_2)" />
    <param name="camera_topic_3" value="$(arg camera_topic_3)" />
    <param name="camera_topic_4" value="$(arg camera_topic_4)" />
  </node>

</launch>
```

---

## 10. 建置 & 執行

### 10.1 系統需求（實際部署環境）

| 項目 | 實際值 |
|---|---|
| 作業系統 | Ubuntu 20.04（Docker container `ros_yolo_bridge`） |
| ROS | Noetic |
| Python（節點 shebang） | **`/opt/conda/envs/yolov8_rtx5090/bin/python3.10`** |
| PyTorch | 2.9.1 + CUDA 12.8（已在 conda 環境內） |
| GPU | NVIDIA RTX 5090（sm_120，Blackwell）|
| Workspace 路徑 | `/catkin_bridge_ws/`（container 內） |

> **重要**：RTX 5090（sm_120）需要 PyTorch 2.7+。系統 python3.8 的 PyTorch 最高支援 sm_90，**不能用**。
> 必須改用 conda `yolov8_rtx5090` 環境的 Python 3.10（已安裝 PyTorch 2.9.1）。

### 10.2 建立 workspace
```bash
# 在 container 內
mkdir -p /catkin_bridge_ws/src
cd /catkin_bridge_ws/src
cp -r /path/to/Yolov8_ros ./
```

### 10.3 Python 依賴（conda 環境）
```bash
# conda yolov8_rtx5090 環境已有：torch 2.9.1、ultralytics 8.3.56、rospy、cv_bridge
# 確認指令：
/opt/conda/envs/yolov8_rtx5090/bin/python3.10 -c "import torch, ultralytics, rospy, cv_bridge; print('OK')"
```

若有缺套件：
```bash
/opt/conda/envs/yolov8_rtx5090/bin/pip install ultralytics
```

### 10.4 編譯
```bash
cd /catkin_bridge_ws
# 必須指定 python3.8 給 catkin_make（catkin 工具本身用 3.8，不影響節點執行）
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8
source devel/setup.bash
```

> **第一次編譯可能會失敗**：`CMakeLists.txt` 裡有 `dl_from_http` 自動下載模型，需要連到 `dl.itriadv.co`（itri 內網）。
> 如果無法下載，把 `CMakeLists.txt` 裡的 `dl_from_http(...)` 全部註解，手動複製 `.pt` 權重到 `weights/`。

### 10.5 給 Python 腳本執行權限
```bash
chmod +x /catkin_bridge_ws/src/Yolov8_ros/yolov8_ros/scripts/multi_camera_detection_node.py
```

### 10.6 執行

```bash
# Terminal 1: 播 rosbag（loop，半速）
rosbag play -l -r 0.5 /workspace/your_recording.bag

# Terminal 2: 啟動偵測節點（每次新開 terminal 都要 source）
source /catkin_bridge_ws/devel/setup.bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 \
  roslaunch yolov8_ros yolo_v8_multi_camera_unified.launch device:=cuda
```

> **`LD_PRELOAD` 為必要**：conda 環境的 libffi 與系統 `libp11-kit` 有版本衝突，
> 不加此行 cv_bridge 會報 `undefined symbol: ffi_type_pointer`。

---

## 11. 驗證測試

### 11.1 編譯通過檢查
```bash
catkin_make 2>&1 | grep -i error
# 沒有 output 代表 OK
```

### 11.2 Topic 列表
```bash
rostopic list | grep yolov8
# 預期會看到：
#   /yolov8/right_120/BoundingBoxes
#   /yolov8/front_120/BoundingBoxes
#   /yolov8/front_120/BoundingBoxes_road_damage
#   /yolov8/left_120/BoundingBoxes
#   /yolov8/back_120_1/BoundingBoxes
#   /yolov8/back_120_2/BoundingBoxes
#   /yolov8/back_120_2/BoundingBoxes_road_damage
#   ... 加上對應的 detection_image / filter_image_bbox
```

### 11.3 看 bbox 內容
```bash
rostopic echo /yolov8/front_120/BoundingBoxes
```

### 11.4 看標註影像（用 rqt_image_view）
```bash
rosrun rqt_image_view rqt_image_view
# 在 GUI 選 /front_120/detection_image 看畫面
```

### 11.5 偵測延遲
```bash
rostopic hz /yolov8/front_120/BoundingBoxes
# 預期：與相機 FPS 相當（10~30 Hz）
```

---

## 12. 嚴禁移植的部分（**以下都不要做**）

> 對應約束 B：**本次不做 GUI**，驗證統一用 `rqt_image_view` / `rostopic`。

| firetruck 模組 | 為什麼不移植 |
|---|---|
| `view/` (PyQt5 GUI) | 用 rqt_image_view 觀察，不需自製 GUI |
| `controller/` | 沒有 GUI 就沒有 controller |
| `model/vlm_worker.py` | LLM 應用本次不在範圍 |
| `vlm:` YAML 區塊 | 跳過 |
| `email_list` / `mail_*` 設定 | 不發 email |
| `web_api_url` / `chX_camera_id` | 不串第三方 API |
| `model/post_processing.py` 的 email/web 部分 | 同上 |
| `model/video_stream.py` | ROS 版相機來源由外部 publisher 提供 |
| GUI log 存圖 / `detection_result_image_data/` | 不需要 |
| ROI 設定 / `roi_ch1.json` | 暫不移植，改成 launch 參數即可（之後可加） |
| `ToggleSwitch` / Stage2 vs LLM 互斥 | 沒 GUI 就沒這層 |

---

## 13. firetruck_detection 程式碼參考索引

照功能去翻：

| 想做的事 | 看 firetruck 的哪裡 |
|---|---|
| 5 路 channel/方向常數 | `firetruck_detection/utils/__init__.py:24` (`CAMERA_DIRECTIONS`) |
| Bridge cls map | `firetruck_detection/cfg/default_settings.yaml:20-23` 註解 |
| 預設信心值 | `firetruck_detection/cfg/default_settings.yaml:9, 17, 31, 33` |
| Stage2 觸發類別 | `firetruck_detection/cfg/default_settings.yaml:29` (`trigger_class_ids: [2, 6]`) |
| ch4 只顯示 cls 6 規則 | `firetruck_detection/cfg/default_settings.yaml:58-59` |
| 路面 bbox_y 過濾比例 | `firetruck_detection/cfg/default_settings.yaml:60-63` |
| Per-channel stage2 模型對應 | `firetruck_detection/cfg/default_settings.yaml:30-33` |
| Detector 工作流 | `firetruck_detection/model/detector.py` |
| Batch inference 主迴圈 | `firetruck_detection/model/detection.py:_inference_worker` |

---

## 14. 常見問題

### Q1: `ImportError: /usr/lib/.../libp11-kit.so.0: undefined symbol: ffi_type_pointer`
conda 環境的 libffi 與系統 libp11-kit 衝突。執行時必須加：
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```
或直接寫在 roslaunch 前面：
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 roslaunch yolov8_ros ...
```

### Q2: `RuntimeError: CUDA error: no kernel image is available for execution on the device`
RTX 5090（sm_120）需要 PyTorch 2.7+。系統 python3.8 的 PyTorch 只到 sm_90，必須改用 conda Python 3.10。
確認 shebang 第一行是：
```
#!/opt/conda/envs/yolov8_rtx5090/bin/python3.10
```
修改方式：
```bash
sed -i '1s|.*|#!/opt/conda/envs/yolov8_rtx5090/bin/python3.10|' \
  /catkin_bridge_ws/src/Yolov8_ros/yolov8_ros/scripts/multi_camera_detection_node.py
```

### Q3: `ImportError: No module named cv_bridge`
確認 conda 環境有安裝：
```bash
/opt/conda/envs/yolov8_rtx5090/bin/python3.10 -c "import cv_bridge; print('ok')"
```

### Q4: `ultralytics` 找不到
```bash
/opt/conda/envs/yolov8_rtx5090/bin/pip install ultralytics
```

### Q5: TensorRT engine 報錯 `batch size mismatch`
batch=5 export 才能餵 5 張影像。用 sample 提供的 `*-batch-5.engine` 或自己重新 export：
```python
from ultralytics import YOLO
m = YOLO('xxx.pt')
m.export(format='engine', batch=5)
```

### Q6: ApproximateTimeSynchronizer 一直沒觸發
- 檢查 5 路相機 topic 的 `header.stamp` 是否合理（不能是 0）
- `slop` 調大一點（試 0.1 或 0.2）
- `queue_size` 太小：`message_filters.Subscriber(t, Image, queue_size=10)` 要明確設

### Q7: callback 阻塞造成 frame drop
- `ThreadPoolExecutor` 的 `max_workers` 開大（建議 8 以上）
- stage2 一定要丟 executor，不能在 callback thread 跑

### Q8: 跑起來 GPU OOM
- `use_tensorrt=True` 改用 engine（量化過）
- 檢查是不是同一支 model 載了多次

---

## 15. Done 的判斷標準（驗收）

完成本次移植的標準：

- [ ] `catkin_make` 三個 package 全部編譯通過
- [ ] `rostopic list` 看得到 11 個 yolov8 相關 topic
- [ ] 播 rosbag 後，`rostopic hz /yolov8/front_120/BoundingBoxes` 有正常頻率
- [ ] `rqt_image_view` 看 `/front_120/detection_image` 看到 bbox 標註
- [ ] ch4 (`/back_120_1/detection_image`) 只看到 cls 6（block expansion joint）
- [ ] ch2/ch5 (`/front_120/detection_image_road_damage`) 看到路面類別且下半部沒有大誤檢
- [ ] 沒有任何 GUI 視窗、沒有 LLM 呼叫、沒有 email、沒有 web_api
- [ ] **`rostopic info /yolov8/front_120/BoundingBoxes` 顯示型別為 `yolov8_ros_msgs/BoundingBoxes`**（與 sample 完全一致 — 下游 GPS 節點要接）
- [ ] **`rosmsg show yolov8_ros_msgs/BoundingBox` 顯示六個欄位 `probability/xmin/ymin/xmax/ymax/Class`，順序與型別不變**

---

## 16. 接下來（不在本次範圍）

留給未來：
- **下游 GPS 節點**（不是我們做，由其他人接 `BoundingBoxes` / `BoundingBoxesWithImage` 加上座標）
- ROI 設定移植（目前 firetruck `roi_ch1.json`）
- LLM/VLM 串接（重新實作為一支 ROS subscriber node 即可）
- 異常 log / 截圖儲存（接 `BoundingBoxesWithImage` 寫一支 logger node）
- 跨相機合併視窗 publisher
- GUI（之後若有需要再用 rqt plugin 或獨立 PyQt 視窗）

這些都可以另外加 node，不影響本次架構 — 前提是**本次的對外 msg 格式不能改**。

---

**檔案結束**。
有疑問先讀指南第 13 節的 firetruck 索引；不確定處保留為 `# TODO` 註解，先讓主流程跑得起來。
