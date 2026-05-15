# olivia-000 / model

電腦視覺與機器學習研究專案集，涵蓋即時影像偵測、行為分析、犯罪熱點預測。

---

## 專案列表

| 分支 | 專案 | 技術棧 |
|------|------|--------|
| [AC_road_inference_ROS](#-橋樑--ac-路面異常偵測系統) | 橋樑 & AC 路面異常偵測系統 | YOLOv11 · PyQt5 · ROS1 · YOLO-CLS · VLM |
| [ESP32-cam](#-徘徊行為偵測系統) | 徘徊行為偵測系統 | YOLOv11 · ByteTrack · Flask · ESP32-CAM |
| [predict-crime](#-時空犯罪遷移學習) | 時空犯罪遷移學習 | DANN · XGBoost · SHAP · Folium |

---

## 橋樑 & AC 路面異常偵測系統

**分支**：[`AC_road_inference_ROS`](../../tree/AC_road_inference_ROS)

部署於消防車的 5 路相機即時偵測系統，整合橋樑結構異常偵測（兩階段 YOLO）與 AC 路面破損偵測。消防車為巡檢平台，偵測對象是橋樑結構與路面，而非車輛本身。

**核心功能：**
- 橋樑 Stage 1：YOLOv11 偵測 10 類構件異常（剝落混凝土、鏽蝕螺栓、破損護欄等），全 5 路相機
- AC 路面 Stage 1：ch2（前）/ ch5（後）兩路偵測 4 類路面破損（裂縫、坑洞）
- Stage 2 分類：ch1/ch3 伸縮縫高低差、ch4 伸縮縫阻塞（YOLO-CLS），Stage 1 觸發後裁切/整圖送入
- VLM 整合：支援本地 Qwen、Gemini、Claude 作為 Stage 2 的語意確認替代方案
- PyQt5 GUI：5 路即時畫面、信心值滑桿（即時生效）、ROI 設定、異常事件 log
- ROS1 版本：完整移植指南，單一整合節點，發布 BoundingBoxes / Image topics

**相機配置：**

| Channel | 位置 | 用途 |
|---------|------|------|
| ch1 | 右側 | 橋樑右側 + 伸縮縫高低差 |
| ch2 | 前方 | AC 路面偵測 |
| ch3 | 左側 | 橋樑左側 + 伸縮縫高低差 |
| ch4 | 後下方 | 伸縮縫俯拍 + 阻塞分類 |
| ch5 | 後方 | AC 路面偵測 |

**快速開始：**
```bash
conda activate multi-task_detection
bash run.sh
```

---

## 徘徊行為偵測系統

**分支**：[`ESP32-cam`](../../tree/ESP32-cam)

即時監控系統，偵測人員在指定區域內的徘徊行為，支援本地影片、Webcam 及 ESP32-CAM 串流。

**核心功能：**
- YOLOv11 人物偵測 + ByteTrack 多目標追蹤
- 可設定 ROI 多邊形與徘徊時間門檻
- 行為分類：`stationary` / `wandering` / `circling` / `traversing`
- Flask Web 儀表板（三路同時 MJPEG 畫面）
- 熱力圖疊加（含時間衰減）
- 警報 log + REST API
- ESP32-CAM HTTP capture 支援
- 自動 LED 控制：偵測到徘徊時亮燈，區域淨空後熄燈（`/led?state=on|off`）

**系統架構：**
```
ESP32-CAM (HTTP)  ──或──  影片檔 / Webcam
        ↓
   YOLOv11 + ByteTrack  →  bbox、track_id
        ↓
   ROI 判斷 + 停留時間累計
        ↓
   Flask 儀表板 + REST API + 警報 log
        ↓
   ESP32 LED on/off（HTTP /led?state=on|off）
```

---

## 時空犯罪遷移學習

**分支**：[`predict-crime`](../../tree/predict-crime)

跨城市犯罪熱點預測研究，以格子層級時空特徵、域對抗神經網路（DANN）與整合模型，探討從資料豐富城市（NYC、Chicago）遷移至其他城市的可行性。

**核心貢獻：**
- 將任務定義從事件層級改為**格子層級主導類別預測**
- 發現並修正格子切分的特徵洩漏，改採時間切分
- 零樣本遷移 NYC→Chicago 超越本地基準（精確率 +17.5pp）
- 量化同國 fine-tuning 的負遷移現象，以 SHAP 解釋根本原因
- 實作 DANN v1/v2（域對抗神經網路）進行深度領域自適應

**涵蓋城市：** New York City · Chicago · Los Angeles · London · Philadelphia · Washington DC · West Yorkshire（共 7 城市）

**技術棧：** Python · XGBoost · PyTorch (DANN) · SHAP · Folium · Pandas · scikit-learn
