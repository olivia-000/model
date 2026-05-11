# 架構介紹 — Spatiotemporal Crime Transfer Learning

## 專案目錄結構

```
spatiotemporal-crime-transfer-learning-main/
├── src/
│   ├── 01_download.py       # 資料下載腳本
│   └── 02_preprocess.py     # 資料清洗與合併
├── notebook/
│   ├── crime_classification_full_NYC.ipynb        # NYC 完整實驗（含 Transfer Learning）
│   ├── crime_classification_full_Chicage&final_step.ipynb
│   ├── crime_classification_full_LA copy.ipynb
│   └── crime_classification_full_Karachi .ipynb
├── data/
│   ├── raw/                 # 原始 CSV（nyc_raw.csv, chicago_raw.csv, ...）
│   └── processed/           # 清洗後資料（nyc_clean.csv, chicago_clean.csv, all_cities.csv）
├── outputs/
│   ├── models/              # 訓練好的模型與校正器
│   └── maps/                # Folium 互動地圖（map_nyc.html, map_chicago.html）
├── interduction/
│   ├── system_introduction.md
│   └── architecture_introduction.md
├── CLAUDE.md
└── requirements.txt
```

---

## 整體流程架構

```
原始資料 (API / Kaggle)
        │
        ▼
  01_download.py
  → data/raw/*.csv
        │
        ▼
  02_preprocess.py
  → 統一欄位格式
  → 犯罪類別對應 (violent / property / drug / public_order)
  → data/processed/all_cities.csv
        │
        ▼
  Notebook（各城市）
  ├── Grid 建立
  ├── 時間切分
  ├── 特徵工程
  ├── 模型訓練
  ├── 機率校正
  ├── Threshold 分析
  └── Transfer Learning
        │
        ▼
  outputs/models/   outputs/maps/
```

---

## 核心模組說明

### 1. 資料下載 (`src/01_download.py`)

- NYC、Chicago、LA：透過 Socrata Open Data API 逐年分批下載，存為 CSV
- Karachi：需手動從 Kaggle 下載合成資料集，放至 `data/raw/karachi_synthetic_raw.csv`
- 已存在的檔案自動跳過，不重複下載

### 2. 資料前處理 (`src/02_preprocess.py`)

每個城市有獨立的 `process_*()` 函式，統一輸出以下欄位：

| 欄位 | 說明 |
|------|------|
| `city` | 城市名稱 |
| `datetime` | 事件時間 |
| `hour`, `month`, `weekday` | 時間特徵 |
| `crime_category` | 對應後的統一類別 |
| `latitude`, `longitude` | 座標（含邊界過濾） |
| `district` | 行政區 |

Karachi 的合成資料有兩種格式，各有對應函式（`process_karachi` / `process_karachi_synthetic`）。

---

### 3. Grid 建立與時間切分（Notebook Cell 2–3）

**Grid 定義：**
- 空間解析度：0.01° × 0.01°（約 1 km²）
- 時間解析度：4 個時段（0–5, 6–11, 12–17, 18–23 時）
- 每個 grid-time 單位需至少 3 筆事件（`min_count=3`）

**預測目標：** 每個 grid-time 中最多的犯罪類別（dominant category）

**時間切分（無洩漏）：**
```
Train   全資料 - 最後 12 個月   (~74%)
Val     最後 12 個月前半段      (~13%)   ← 機率校正與 threshold 調參
Test    最後 6 個月            (~13%)   ← 最終評估，不可用於調參
```

> 重要：`hist_*` 特徵只能在 Train 期間計算，再 join 至 Val/Test，不得跨時間洩漏。

---

### 4. 特徵工程（Notebook Cell 4–5）

**27 個特徵分為 6 群：**

| 群組 | 特徵 | 說明 |
|------|------|------|
| 歷史組成 | `hist_violent`, `hist_property`, `hist_other` | 該 grid 在 Train 期間各類別佔比 |
| 空間延遲 | `lag_violent`, `lag_property`, `lag_other` | 鄰近 grid 的犯罪組成平均 |
| 相對百分位 | `violent_pct`, `density_pct`, `entropy_pct`, `dom_gap_pct` | 相對全城市的分位數 |
| 穩定性 | `top1_ratio`, `dominance_gap`, `entropy` | 犯罪分佈的集中程度 |
| 時間 | `time_slot`, `is_weekend`, sin/cos 編碼 | 時間週期性特徵 |
| 空間 | `lat_bin`, `lon_bin`, `lat_norm`, `lon_norm` | 格網座標與正規化座標 |

**特徵重要性（Ablation Study）：**

| 特徵組合 | Precision Macro |
|---------|----------------|
| `hist_*` 3 個特徵 | **0.649** |
| 全部 26 個特徵 | 0.567 |
| 無 `hist_*`（23 個）| 0.520 |

---

### 5. 模型訓練與校正（Notebook Cell 6–13）

**模型：**
- `CatBoostClassifier`：balanced class weights，內建 categorical 支援
- `LightGBMClassifier`：balanced class weights，速度較快

**機率校正：**
- Platt Scaling（Logistic Regression）
- Isotonic Regression
- 校正後繪製 Reliability Diagram 驗證

**Confidence Threshold：**
- 全域 threshold sweep：在 Val 集掃描最佳 precision-coverage 平衡點
- 類別別 threshold：各類別獨立調整，提升整體 precision

---

### 6. Transfer Learning（Notebook Cell 18–20）

**同國遷移：NYC → Chicago**

| 實驗 | 方法 |
|------|------|
| Zero-shot | 直接用 NYC 模型預測 Chicago 測試集 |
| Fine-tune 10/20/50% | 繼續訓練 NYC 模型，加入部分 Chicago 訓練資料 |
| Teacher-Student | NYC 為 Teacher，用 soft labels（T=3.0）訓練 Chicago Student |

**跨文化遷移：NYC/Chicago → Karachi**
- 兩個美國城市對 Karachi 的 zero-shot 表現完全相同（Precision 0.389）
- 確認障礙為文化/結構差異，而非城市特異性

**Domain Shift 量化：**
- 使用 Jensen-Shannon Divergence（JSD）計算各特徵的分佈差異
- `hist_*` 特徵在跨城市間 JSD 最高 → 解釋跨文化遷移失敗原因

---

### 7. 輸出（Notebook Cell 22）

| 檔案 | 說明 |
|------|------|
| `outputs/models/model_{city}_catboost.cbm` | CatBoost 模型 |
| `outputs/models/model_{city}_lgb.pkl` | LightGBM 模型 |
| `outputs/models/cal_platt_{city}.pkl` | Platt 校正器 |
| `outputs/models/label_encoder_{city}.pkl` | Label Encoder |
| `outputs/maps/map_{city}.html` | Folium 互動地圖（含信心分層顯示） |
