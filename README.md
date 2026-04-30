# Loitering Detection System

A real-time surveillance system that detects people loitering in a defined area using **YOLOv11** object detection, **ByteTrack** multi-object tracking, and a **Flask** web dashboard. Supports both local video files and live **ESP32-CAM** streams.

---

## Features

- Real-time person detection with YOLOv11 + ByteTrack
- Configurable ROI (Region of Interest) polygon and loitering time threshold
- Behavior classification: `stationary` / `wandering` / `circling` / `traversing`
- Live web dashboard with three simultaneous MJPEG video feeds
- Heatmap overlay with temporal decay
- Alert log and REST API
- ESP32-CAM HTTP capture support

---

## System Architecture

```
ESP32-CAM (HTTP /capture)  ──or──  Local file / webcam
         ↓
  esp32_source.py  (background thread, normalizes all video sources)
         ↓
  YOLO.track()  (YOLOv11 + ByteTrack — track_id, bbox, confidence)
         ↓
  ROI polygon check → per-track dwell time accumulation
         ↓
  Loitering alert when dwell > threshold
         ↓
  Dashboard display  +  REST API  +  in-memory alert log
```

---

## Project Structure

```
├── loitering_detect.py          # CLI detection tool
├── loitering_dashboard/
│   ├── app.py                   # Flask dashboard app
│   └── templates/index.html     # Dashboard frontend
├── esp32_source.py              # Video source abstraction (local + ESP32-CAM)
├── train_yolo_person.py         # Model training script
├── convert_pennfudan_to_yolo.py # Dataset format conversion
├── test_model.py                # Quick inference test
├── CameraWebServer/             # ESP32-CAM Arduino firmware
│   └── CameraWebServer.ino
├── runs/
│   └── person_yolo_train/
│       └── weights/
│           └── best.pt          # Trained model weights
├── yolo11n.pt                   # YOLOv11 nano base weights
└── person.yaml                  # Dataset config (auto-generated)
```

---

## Requirements

```bash
pip install ultralytics opencv-python numpy flask requests
```

ESP32-CAM firmware is compiled separately via **Arduino IDE** (`CameraWebServer/CameraWebServer.ino`).

---

## Usage

### 1. Train a Person Detection Model

Prepare your dataset in the following structure:

```
person_dataset/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

Then run:

```bash
python train_yolo_person.py --dataset-root person_dataset --epochs 50 --batch 16 --device 0
```

Trained weights are saved to `runs/person_yolo_train/weights/best.pt`.

### 2. CLI Loitering Detection (video file or webcam)

```bash
# Run on a video file, display window + save output
python loitering_detect.py --source test.mp4 --show --save

# Run on a live ESP32-CAM stream
python loitering_detect.py --source http://172.20.10.2/capture --show

# Run on webcam
python loitering_detect.py --source 0 --show
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `runs/.../best.pt` | Path to YOLO model weights |
| `--source` | `test.mp4` | Video path, webcam index, or HTTP URL |
| `--conf` | `0.5` | Detection confidence threshold |
| `--loiter-seconds` | `10.0` | Seconds in ROI before alert |
| `--grace-seconds` | `1.0` | Tolerance for brief missed detections |
| `--show` | off | Show live preview window |
| `--save` | off | Save annotated output video |

### 3. Live Web Dashboard (requires ESP32-CAM)

```bash
python loitering_dashboard/app.py
```

Open `http://localhost:5000` in a browser.

The dashboard provides three simultaneous live feeds:

| Feed | Description |
|---|---|
| Bounding Box | Detection boxes with track IDs and dwell time |
| Trajectory | Movement paths with behavior classification |
| Heatmap | Dwell density map with temporal decay |

### 4. REST API

| Endpoint | Description |
|---|---|
| `GET /api/status` | Camera, model status, FPS, active alerts |
| `GET /api/alerts` | Last 20 loitering events |
| `GET /api/hourly` | Alert counts per hour (0–23) |

---

## Configuration

All configuration is hardcoded in the source files. Key values to adjust:

| Parameter | File | Default |
|---|---|---|
| ESP32-CAM IP | `loitering_dashboard/app.py` | `172.20.10.2` |
| Loiter threshold (dashboard) | `loitering_dashboard/app.py` | `5.0` seconds |
| Loiter threshold (CLI) | `loitering_detect.py` via `--loiter-seconds` | `10.0` seconds |
| ROI polygon points | Both files | Full frame `(0,0)→(1280,960)` |
| Model path | Both files | `runs/person_yolo_train/weights/best.pt` |

To change the ROI, edit the `roi_points` list:

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

## ESP32-CAM Setup

1. Open `CameraWebServer/CameraWebServer.ino` in Arduino IDE.
2. Set your Wi-Fi credentials in the sketch.
3. Flash to the ESP32-CAM board.
4. Note the assigned IP address and update `ESP32_IP` in `loitering_dashboard/app.py`.

The system polls `http://<ESP32_IP>/capture` to fetch JPEG frames.

---

## Dataset Conversion

To convert **PennFudan** or **INRIA** datasets to YOLO label format:

```bash
python convert_pennfudan_to_yolo.py
```

---

## Training Results

Training curves and validation images are stored under `runs/person_yolo_train/`:

- `results.png` — loss and mAP curves
- `confusion_matrix.png` — detection confusion matrix
- `BoxPR_curve.png` — Precision-Recall curve
- `weights/best.pt` — best checkpoint (used for inference)
