# Loitering Detection System
# 徘徊行為偵測系統

**EN** — A real-time surveillance system that detects people loitering in a defined area using **YOLOv11** object detection, **ByteTrack** multi-object tracking, and a **Flask** web dashboard. Supports both local video files and live **ESP32-CAM** streams.

**ZH** — 即時監控系統，使用 **YOLOv11** 物件偵測、**ByteTrack** 多目標追蹤與 **Flask** 網頁儀表板，偵測人員在指定區域內的徘徊行為。支援本地影片檔與即時 **ESP32-CAM** 串流。

---

## Features / 功能特色

- **EN** Real-time person detection with YOLOv11 + ByteTrack
  **ZH** YOLOv11 + ByteTrack 即時人物偵測與追蹤
- **EN** Configurable ROI (Region of Interest) polygon and loitering time threshold
  **ZH** 可自訂 ROI 多邊形區域與徘徊時間門檻
- **EN** Behavior classification: `stationary` / `wandering` / `circling` / `traversing`
  **ZH** 行為分類：`stationary`（靜止）/ `wandering`（遊走）/ `circling`（繞圈）/ `traversing`（穿越）
- **EN** Live web dashboard with three simultaneous MJPEG video feeds
  **ZH** 網頁儀表板同時顯示三路 MJPEG 即時畫面
- **EN** Heatmap overlay with temporal decay
  **ZH** 熱力圖疊加（含時間衰減效果）
- **EN** Alert log and REST API
  **ZH** 警報 log 與 REST API
- **EN** ESP32-CAM HTTP capture support
  **ZH** ESP32-CAM HTTP capture 支援

---

## System Architecture / 系統架構

```
ESP32-CAM (HTTP /capture)  ──or / 或──  Local file / webcam（本地影片 / 鏡頭）
         ↓
  esp32_source.py  (background thread, normalizes all video sources / 背景執行緒，統一所有影像來源)
         ↓
  YOLO.track()  (YOLOv11 + ByteTrack — track_id, bbox, confidence)
         ↓
  ROI polygon check → per-track dwell time accumulation
  （ROI 多邊形判斷 → 各目標停留時間累計）
         ↓
  Loitering alert when dwell > threshold（停留超過門檻 → 發出警報）
         ↓
  Dashboard display  +  REST API  +  in-memory alert log
  （儀表板顯示 + REST API + 記憶體警報 log）
```

---

## Project Structure / 專案結構

```
├── loitering_detect.py          # CLI detection tool / CLI 偵測工具
├── loitering_dashboard/
│   ├── app.py                   # Flask dashboard app / Flask 儀表板應用
│   └── templates/index.html     # Dashboard frontend / 儀表板前端
├── esp32_source.py              # Video source abstraction / 影像來源抽象層（本地 + ESP32-CAM）
├── train_yolo_person.py         # Model training script / 模型訓練腳本
├── convert_pennfudan_to_yolo.py # Dataset format conversion / 資料集格式轉換
├── test_model.py                # Quick inference test / 快速推論測試
├── CameraWebServer/             # ESP32-CAM Arduino firmware / ESP32-CAM 韌體
│   └── CameraWebServer.ino
├── runs/
│   └── person_yolo_train/
│       └── weights/
│           └── best.pt          # Trained model weights / 訓練完成的模型權重
├── yolo11n.pt                   # YOLOv11 nano base weights / YOLOv11 nano 基底權重
└── person.yaml                  # Dataset config (auto-generated) / 資料集設定（自動產生）
```

---

## Requirements / 環境需求

```bash
pip install ultralytics opencv-python numpy flask requests
```

**EN** — ESP32-CAM firmware is compiled separately via **Arduino IDE** (`CameraWebServer/CameraWebServer.ino`).

**ZH** — ESP32-CAM 韌體需另外使用 **Arduino IDE** 編譯燒錄（`CameraWebServer/CameraWebServer.ino`）。

---

## Usage / 使用方式

### 1. Train a Person Detection Model / 訓練人物偵測模型

**EN** — Prepare your dataset in the following structure:

**ZH** — 請先準備以下資料集目錄結構：

```
person_dataset/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

```bash
python train_yolo_person.py --dataset-root person_dataset --epochs 50 --batch 16 --device 0
```

**EN** — Trained weights are saved to `runs/person_yolo_train/weights/best.pt`.

**ZH** — 訓練完成的權重儲存於 `runs/person_yolo_train/weights/best.pt`。

---

### 2. CLI Loitering Detection / CLI 徘徊偵測（影片檔或鏡頭）

```bash
# EN: Run on a video file, display window + save output
# ZH: 對影片檔執行，顯示視窗並儲存結果
python loitering_detect.py --source test.mp4 --show --save

# EN: Run on a live ESP32-CAM stream
# ZH: 接 ESP32-CAM 即時串流
python loitering_detect.py --source http://172.20.10.2/capture --show

# EN: Run on webcam
# ZH: 接 Webcam
python loitering_detect.py --source 0 --show
```

| Flag | Default | EN Description | ZH 說明 |
|---|---|---|---|
| `--model` | `runs/.../best.pt` | Path to YOLO model weights | YOLO 模型權重路徑 |
| `--source` | `test.mp4` | Video path, webcam index, or HTTP URL | 影片路徑、鏡頭編號或 HTTP URL |
| `--conf` | `0.5` | Detection confidence threshold | 偵測信心值門檻 |
| `--loiter-seconds` | `10.0` | Seconds in ROI before alert | 觸發警報所需的 ROI 停留秒數 |
| `--grace-seconds` | `1.0` | Tolerance for brief missed detections | 短暫漏偵的容忍秒數 |
| `--show` | off | Show live preview window | 顯示即時預覽視窗 |
| `--save` | off | Save annotated output video | 儲存標註後的輸出影片 |

---

### 3. Live Web Dashboard / 即時網頁儀表板

**EN** — Requires an active ESP32-CAM or compatible video source.

**ZH** — 需有 ESP32-CAM 或相容的影像來源。

```bash
python loitering_dashboard/app.py
```

**EN** — Open `http://localhost:5000` in a browser. The dashboard provides three simultaneous live feeds:

**ZH** — 在瀏覽器開啟 `http://localhost:5000`，儀表板提供三路同步即時畫面：

| Feed / 畫面 | EN Description | ZH 說明 |
|---|---|---|
| Bounding Box | Detection boxes with track IDs and dwell time | 偵測框（含追蹤 ID 與停留時間） |
| Trajectory | Movement paths with behavior classification | 移動軌跡（含行為分類） |
| Heatmap | Dwell density map with temporal decay | 停留密度熱力圖（含時間衰減） |

---

### 4. REST API

| Endpoint | EN Description | ZH 說明 |
|---|---|---|
| `GET /api/status` | Camera, model status, FPS, active alerts | 相機、模型狀態、FPS、當前警報 |
| `GET /api/alerts` | Last 20 loitering events | 最近 20 筆徘徊事件 |
| `GET /api/hourly` | Alert counts per hour (0–23) | 每小時警報次數（0–23 時） |

---

## Configuration / 參數設定

**EN** — All configuration is hardcoded in the source files. Key values to adjust:

**ZH** — 所有設定以常數寫在原始碼中，主要可調整項目如下：

| Parameter / 參數 | File / 檔案 | Default / 預設值 |
|---|---|---|
| ESP32-CAM IP | `loitering_dashboard/app.py` | `172.20.10.2` |
| Loiter threshold (dashboard) / 徘徊門檻（儀表板） | `loitering_dashboard/app.py` | `5.0` 秒 |
| Loiter threshold (CLI) / 徘徊門檻（CLI） | `loitering_detect.py` via `--loiter-seconds` | `10.0` 秒 |
| ROI polygon points / ROI 頂點座標 | Both / 兩者 | Full frame `(0,0)→(1280,960)` |
| Model path / 模型路徑 | Both / 兩者 | `runs/person_yolo_train/weights/best.pt` |

**EN** — To change the ROI, edit the `roi_points` list:

**ZH** — 修改 ROI 區域請編輯 `roi_points`：

```python
# loitering_detect.py  or  loitering_dashboard/app.py
ROI_POINTS = [
    (100, 200),
    (800, 200),
    (800, 700),
    (100, 700),
]
```

---

## ESP32-CAM Setup / ESP32-CAM 設定

**EN:**
1. Open `CameraWebServer/CameraWebServer.ino` in Arduino IDE.
2. Set your Wi-Fi credentials in the sketch.
3. Flash to the ESP32-CAM board.
4. Note the assigned IP address and update `ESP32_IP` in `loitering_dashboard/app.py`.

**ZH:**
1. 以 Arduino IDE 開啟 `CameraWebServer/CameraWebServer.ino`。
2. 在程式碼中填入 Wi-Fi 帳號密碼。
3. 燒錄至 ESP32-CAM 開發板。
4. 記錄分配到的 IP，並更新 `loitering_dashboard/app.py` 中的 `ESP32_IP`。

**EN** — The system polls `http://<ESP32_IP>/capture` to fetch JPEG frames.

**ZH** — 系統以輪詢方式從 `http://<ESP32_IP>/capture` 抓取 JPEG 影像幀。

---

## Dataset Conversion / 資料集格式轉換

**EN** — To convert **PennFudan** or **INRIA** datasets to YOLO label format:

**ZH** — 將 **PennFudan** 或 **INRIA** 資料集轉換為 YOLO 標籤格式：

```bash
python convert_pennfudan_to_yolo.py
```

---

## Training Results / 訓練結果

**EN** — Training curves and validation images are stored under `runs/person_yolo_train/`:

**ZH** — 訓練曲線與驗證圖片儲存於 `runs/person_yolo_train/`：

| File / 檔案 | EN Description | ZH 說明 |
|---|---|---|
| `results.png` | Loss and mAP curves | Loss 與 mAP 曲線 |
| `confusion_matrix.png` | Detection confusion matrix | 偵測混淆矩陣 |
| `BoxPR_curve.png` | Precision-Recall curve | Precision-Recall 曲線 |
| `weights/best.pt` | Best checkpoint (used for inference) | 最佳檢查點（推論時使用） |
