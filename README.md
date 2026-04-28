# 橋樑 & 路面異常偵測系統

基於 **兩階段 YOLO + 本地 LLM** 的橋樑與 AC 路面異常即時偵測系統，支援 5 路相機同步監控，PyQt5 GUI 操作介面，可在一般 x86 電腦執行（無需 ROS）。

![GUI 介面](smaple/gui_screenshot.png)

---

## 系統功能

### 偵測模組

| 模組 | 說明 |
|------|------|
| 橋樑偵測 Stage1 | 5 路相機全覆蓋，YOLO 偵測異常構件（剝落、鏽蝕螺栓、破損護欄等） |
| AC 路面偵測 | ch2/ch5 路面破損偵測 |
| 橋樑 Stage2 CLS | ch1/ch3 伸縮縫高低差、ch4 伸縮縫阻塞分類確認 |
| LLM 串接 | Stage2 偵測結果送本地 Qwen / Gemini / Claude 進行語意確認 |

### GUI 功能

- 5 路相機即時畫面（3×2 格局）
- 各模組啟用/停用開關
- 橋樑/路面/高低差/阻塞信心值滑桿（即時生效）
- LLM 模式切換（Provider / Model / 送件間隔）
- 每路相機獨立 ROI 設定
- 異常事件 log 欄位（點擊可檢視 annotated 圖片）

---

## 系統需求

- OS：Linux（Ubuntu）或 Windows
- Python：3.11（conda 環境）
- GPU：建議 8GB VRAM 以上（本地 LLM 需額外 ~7GB）
- CUDA：12.x 以上

---

## 安裝

### 1. 建立 conda 環境

```bash
conda create -n multi-task_detection python=3.11 -y
conda activate multi-task_detection
```

### 2. 安裝 PyTorch（CUDA 13.0）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
    --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
```

### 3. 安裝其他套件

```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install "numpy<2" "opencv-python-headless<4.9" \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 4. 放置模型權重

將 `.pt` 檔案放入 `firetruck_detection/weights/`：

- Stage1 橋樑偵測模型
- Stage2 伸縮縫分類模型（ch1/ch3 高低差、ch4 阻塞）
- AC 路面偵測模型

---

## 執行

```bash
conda activate multi-task_detection
bash run.sh
```

---

## 設定檔

主設定檔：`firetruck_detection/cfg/default_settings.yaml`

| 參數 | 說明 |
|------|------|
| `source` | `0` = 影片檔，`1` = RTSP 串流 |
| `test_video_dir` | 影片根目錄，需含 `ch1/`～`ch5/` 子資料夾 |
| `vlm.provider` | `qwen_local` / `gemini` / `claude` |
| `vlm.cooldown_sec` | 每路相機最短送件間隔（本地建議 5 秒） |

---

## 本地 LLM（Qwen）部署

需額外部署 vLLM 伺服器，請參考 `deploy-vllm.md`。

啟動流程：

```bash
# 終端機 1：啟動 vLLM server
bash start_vllm.sh

# 終端機 2：啟動偵測系統
bash run.sh
```

---

## 架構

MVC 設計模式：

```
firetruck_detection/
├── main.py              # 程式入口
├── model/
│   ├── detection.py     # 頂層 Model，管理所有 channel worker
│   ├── detector.py      # 單路偵測 worker
│   ├── post_processing.py  # Stage2 分類
│   └── vlm_worker.py    # LLM 串接（per-channel queue + thread）
├── view/                # PyQt5 GUI
├── controller/          # 橋接 Model ↔ View
├── cfg/                 # 設定檔
└── weights/             # 模型權重
```

---

## 相機配置

| Channel | 位置 | 用途 |
|---------|------|------|
| ch1 | Right | 橋樑右側、伸縮縫高低差 |
| ch2 | Front | AC 路面偵測 |
| ch3 | Left | 橋樑左側、伸縮縫高低差 |
| ch4 | Back-Down | 伸縮縫阻塞 |
| ch5 | Back | AC 路面偵測 |
