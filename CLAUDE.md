# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Last updated: 2026-05-08

---

## Environment Setup

```bash
conda create -n crime-tl python=3.10 -y
conda activate crime-tl
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 \
            catboost lightgbm xgboost==2.0.3 scipy joblib \
            folium==0.16.0 matplotlib==3.9.0 seaborn==0.13.2 \
            tqdm==4.66.4 jupyter torch
```

> `requirements.txt` is **incomplete** — it omits `catboost`, `lightgbm`, `scipy`, `joblib`, and `torch`, all of which are required by the notebooks and DANN scripts.

---

## Directory Structure

```
spatiotemporal-crime-transfer-learning-main/
├── src/
│   ├── 01_download.py              # 資料下載（NYC / Chicago / LA 自動，其餘手動）
│   └── 02_preprocess.py            # 清洗 & 統一欄位格式，輸出 data/processed/
├── notebook/
│   ├── crime_classification_full_NYC.ipynb          # NYC 完整實驗（含 Transfer Learning）
│   ├── crime_classification_full_NYC.py             # 同上的 .py 版本
│   ├── crime_classification_full_Chicage&final_step.ipynb
│   ├── crime_classification_full_Chicage_final_step.py
│   ├── crime_classification_full_LA copy.ipynb
│   ├── crime_classification_full_LA_copy.py
│   ├── crime_classification_full_London.py
│   ├── crime_classification_full_Philadelphia.py
│   ├── crime_classification_full_WestYorkshire.py
│   ├── crime_classification_full_DC.py
│   ├── clean_dc.py                 # DC 原始資料清洗腳本
│   ├── clean_philadelphia.py       # Philadelphia 原始資料清洗腳本
│   ├── clean_westyorkshire.py      # West Yorkshire 原始資料清洗腳本
│   ├── dann_crime.py               # DANN v1（基礎版域對抗神經網路）
│   └── dann_v2.py                  # DANN v2（4 種改善：RF初始化、特徵篩選、Ensemble Teacher、JSD λ調度）
├── data/
│   ├── raw/                        # 原始 CSV
│   │   ├── nyc_raw.csv
│   │   ├── chicago_raw.csv
│   │   ├── la_raw.csv
│   │   ├── karachi_raw.csv / karachi_synthetic_raw.csv / karachi_crime_dataset.csv
│   │   ├── london_clean.csv
│   │   ├── dc_raw/                 # DC 按年份分檔（Crime_Incidents_-_20XX.csv）
│   │   ├── philly_raw/             # Philadelphia 按年份分檔（incidents_part1_part2_20XX.csv）
│   │   └── westyorkshire_raw/      # West Yorkshire 按月份分資料夾（2023-04 ～ 2026-03）
│   └── processed/
│       ├── all_cities.csv          # 所有城市合併（主要訓練資料來源）
│       ├── nyc_clean.csv
│       ├── chicago_clean.csv
│       ├── la_clean.csv
│       ├── karachi_clean.csv / karachi_crime_dataset.csv
│       ├── london_clean.csv
│       ├── dc_clean.csv
│       ├── philadelphia_clean.csv
│       └── west_yorkshire_clean.csv
├── outputs/
│   ├── models/                     # 所有城市的模型、校正器、Grid 風險分數
│   ├── eda/                        # 混淆矩陣、特徵重要性、SHAP、Transfer 比較圖
│   └── maps/
│       ├── crime_map_v3.html       # 主要互動地圖（最新版，功能最完整）
│       ├── crime_map_v2.html
│       ├── crime_map.html
│       ├── map_nyc.html
│       └── map_chicago.html
├── interduction/
│   ├── system_introduction.md
│   └── architecture_introduction.md
├── CLAUDE.md
├── README.md
└── requirements.txt
```

---

## Running the Pipeline

### Step 1 — Download raw data
```bash
python src/01_download.py
```
- **自動下載（API）**：NYC、Chicago、LA
- **手動下載**：
  - Karachi → Kaggle 下載 `karachi_synthetic_raw.csv` → 放入 `data/raw/`
  - London → data.police.uk 手動下載，已存於 `data/raw/london_clean.csv`
  - DC → Open Data DC 下載年份 CSV → 放入 `data/raw/dc_raw/`
  - Philadelphia → OpenDataPhilly 下載年份 CSV → 放入 `data/raw/philly_raw/`
  - West Yorkshire → data.police.uk 月份 CSV → 放入 `data/raw/westyorkshire_raw/YYYY-MM/`

### Step 2 — Preprocess
```bash
python src/02_preprocess.py
```
輸出統一格式 CSV 至 `data/processed/`。`all_cities.csv` 為所有 notebook 共用的合併檔。

### Step 2b — 新增城市清洗（DC / Philadelphia / West Yorkshire）
```bash
python notebook/clean_dc.py
python notebook/clean_philadelphia.py
python notebook/clean_westyorkshire.py
```
這三個腳本各自產生對應的 `*_clean.csv` 到 `data/processed/`。

### Step 3 — Train & Experiment（Notebook）
```bash
jupyter notebook
# 開啟 notebook/crime_classification_full_NYC.ipynb
# Kernel → Restart & Run All
```
修改 Cell 3（Section 2）中的 `CITY = 'NYC'` 來切換城市。

| 城市 | Notebook 檔案 |
|------|--------------|
| NYC | `crime_classification_full_NYC.ipynb` |
| Chicago | `crime_classification_full_Chicage&final_step.ipynb` |
| LA | `crime_classification_full_LA copy.ipynb` |
| London | `crime_classification_full_London.py`（純 .py 版） |
| Philadelphia | `crime_classification_full_Philadelphia.py` |
| DC | `crime_classification_full_DC.py` |
| West Yorkshire | `crime_classification_full_WestYorkshire.py` |

### Step 4 — Domain Adversarial Transfer（可選）
```bash
# DANN v1（基礎版）
python notebook/dann_crime.py --source NYC --target Chicago
python notebook/dann_crime.py --all   # 跑所有 city pair

# DANN v2（改善版）
python notebook/dann_v2.py --source NYC --target Chicago --epochs 100 --top_k 12
python notebook/dann_v2.py --all --epochs 100
```

---

## Task Definition

- **空間粒度**：0.01° × 0.01° 格子（約 1 km²），每格至少 3 筆事件
- **時間粒度**：4 個時段（深夜 0–5, 早晨 6–11, 下午 12–17, 夜晚 18–23）
- **預測目標**：每個 grid-time 單位的**主導犯罪類別**（出現次數最多的類別）
- **為何用 Grid 層級**：事件層級預測精度上限約 0.35 precision，Grid 層級可達 0.65–0.82

---

## Data Split（時間切分，無洩漏）

```
Train  最早 ～ 倒數 12 個月前   (~74%)   ← hist_* 特徵只在此計算
Val    倒數 12 個月前半段        (~13%)   ← 機率校正、threshold 調參
Test   最後 6 個月              (~13%)   ← 最終評估，不可用於調參
```

**關鍵限制**：`hist_*` 特徵（grid 在 Train 期間的犯罪組成佔比）必須僅用 Train 資料計算，再 join 至 Val/Test。**絕不能**在切分前就計算全資料的 hist_*。

---

## Crime Category Mapping

所有城市統一映射至 4 大類：

| 類別 | 說明 | 範例 |
|------|------|------|
| `violent` | 暴力犯罪 | 攻擊、搶劫、殺人、強姦 |
| `property` | 財產犯罪 | 竊盜、入室盜竊、汽車竊盜、詐欺 |
| `drug` | 毒品犯罪 | 持有、販賣毒品 |
| `public_order` | 妨害公序 | 擾亂公共安寧、違反交通法規 |

> Notebook 中 `drug` 與 `public_order` 因 Grid 層級資料稀疏，合併為 `other`（`MERGE_MAP` 在 Cell 3），最終使用 **3 個有效類別**：`violent / property / other`。

---

## Feature Groups（27 個特徵）

| 群組 | 特徵 | 說明 |
|------|------|------|
| 歷史組成 | `hist_violent`, `hist_property`, `hist_other` | grid 在 Train 期間各類別佔比，**單獨 3 個就超越全部 26 個特徵** |
| 空間延遲 | `lag_violent`, `lag_property`, `lag_other` | 鄰近 grid 的犯罪組成平均 |
| 相對百分位 | `violent_pct`, `density_pct`, `entropy_pct`, `dom_gap_pct` | 相對全城市的分位數 |
| 穩定性 | `top1_ratio`, `dominance_gap`, `entropy` | 犯罪分佈的集中程度 |
| 時間 | `time_slot`, `is_weekend`, hour/month/weekday sin+cos | 時間週期性特徵 |
| 空間 | `lat_bin`, `lon_bin`, `lat_norm`, `lon_norm` | 格網座標與正規化座標 |

**Ablation Study 結論**（NYC）：

| 特徵組合 | Precision Macro |
|---------|----------------|
| `hist_*` 3 個特徵 | **0.649** |
| 全部 26 個特徵 | 0.567 |
| 無 `hist_*`（23 個）| 0.520 |

---

## Model Pipeline

1. **CatBoost** + **LightGBM**，balanced class weights
2. **Platt Scaling** / **Isotonic Regression** 機率校正（在 Val 集上校正）
3. **Confidence Threshold Sweep**：全域 & 類別別閾值，precision-coverage 權衡
4. **Two-Stage Model**：先判 violent vs. non-violent，再判 property vs. other
5. 輸出：模型檔、校正器、label encoder 存入 `outputs/models/`

---

## Transfer Learning

### 同國遷移（NYC → Chicago）— `crime_classification_full_Chicage_final_step.py` Cell 19

| Scenario | Precision Macro | 說明 |
|----------|----------------|------|
| Chicago baseline（從頭訓練）| 0.439 | 無遷移 |
| **Zero-shot NYC → Chicago** | **0.614** | 超越本地 +17.5pp |
| Fine-tune 10% | ~ 0.439 | 負遷移 |
| Fine-tune 50% | ~ 0.439 | 負遷移 |
| Teacher-Student (T=3.0) | 略高於 baseline | 效果有限 |

**結論**：同國零樣本遷移有效；加入目標城市資料反而造成負遷移。

### 跨文化遷移（NYC / Chicago → Karachi）

- 兩個來源城市對 Karachi 的 zero-shot 表現**完全相同**（Precision ≈ 0.389）
- 確認障礙來自文化/結構差異，而非城市特異性
- **根本原因**：`hist_*` 特徵在跨城市間 JSD 最高，domain shift 最大

### DANN（域對抗神經網路）— `dann_crime.py` / `dann_v2.py`

**DANN v1 架構**（26-dim → 128 → 64 → 3 類別）：
- Feature Extractor F → Label Predictor C + Domain Classifier D
- 訓練目標：`L = L_class(F,C) - λ·L_domain(F,D)`
- Gradient Reversal Layer 反轉 domain classifier 的梯度

**DANN v2 四項改善**：
1. RF 預訓練初始化 Feature Extractor（取代隨機初始化）
2. RF feature importance 篩選 top-K 特徵（降低 alignment 維度）
3. Ensemble Teacher-Student（RF + LightGBM + XGBoost 三模型平均 soft label）
4. JSD 自適應 λ 調度（`lambda_max = 0.5 × exp(-mean_JSD)`）

已訓練的 DANN 模型存於 `outputs/models/dann_{source}_{target}.pt`。

---

## Model Performance Summary（各城市）

| 城市 | 格子數 | 地圖準確率 | 平均風險 | 最高風險 |
|------|--------|-----------|---------|---------|
| NYC | 3,501 | 71.0% | 2.3 | 51.5 |
| Chicago | 2,740 | 81.9% | 27.2 | 56.9 |
| LA | 4,130 | 68.2% | 11.5 | 20.3 |
| London | 7,381 | 57.9% | 23.3 | 56.3 |
| Philadelphia | 1,495 | 81.5% | 0.9 | 1.6 |
| DC | 633 | 96.5% | 0.0 | 0.0 |
| West Yorkshire | 4,940 | 69.3% | 67.9 | 79.0 |

---

## Outputs

| 路徑 | 內容 |
|------|------|
| `outputs/models/model_{city}_catboost.cbm` | CatBoost 模型 |
| `outputs/models/model_{city}_lgb.pkl` | LightGBM 模型 |
| `outputs/models/cal_platt_{city}.pkl` | Platt 校正器 |
| `outputs/models/cal_iso_{city}.pkl` | Isotonic 校正器 |
| `outputs/models/label_encoder_{city}.pkl` | Label Encoder |
| `outputs/models/grid_risk_{city}.csv` | 各 grid 風險分數（含預測類別、信心度） |
| `outputs/models/ablation_{city}.csv` | 特徵群組 Ablation 結果 |
| `outputs/models/method_comparison_{city}.csv` | 各方法 Precision/F1 比較 |
| `outputs/models/transfer_{src}_{tgt}.csv` | 遷移學習結果（各 Scenario） |
| `outputs/models/dann_results.csv` | DANN 所有 city pair 結果 |
| `outputs/eda/cm_{city}_{model}.png` | 混淆矩陣 |
| `outputs/eda/feature_importance_{city}.png` | 特徵重要性 |
| `outputs/eda/shap_{city}.png` | SHAP summary plot |
| `outputs/eda/ablation_{city}.png` | Ablation study 圖 |
| `outputs/eda/transfer_{src}_{tgt}.png` | Transfer Learning 比較圖 |
| `outputs/eda/reliability_{city}_{model}.png` | 機率校正 Reliability Diagram |
| `outputs/eda/precision_coverage_{city}.png` | Precision-Coverage 曲線 |
| `outputs/maps/crime_map_v3.html` | **主要互動地圖**（最新版，見下方說明） |

---

## Interactive Map — `crime_map_v3.html`

獨立 HTML 檔案，無需後端，直接瀏覽器開啟。所有 7 個城市的 grid 資料靜態嵌入。

### 功能清單

| 功能 | 說明 |
|------|------|
| 城市切換（7 城市） | NYC / Chicago / LA / London / Philadelphia / DC / West Yorkshire |
| 時間段動態播放 | 4 段（深夜/早晨/下午/夜晚），1.8 秒自動循環 |
| 格子點擊詳情 | 座標、真實類別、預測類別(✓/✗)、信心度、事件數、風險分數、機率長條圖 |
| 告警閾值設定 | 滑桿設定風險門檻，超過門檻的格子標記 + 徽章 |
| 路徑風險查詢 | 輸入起終點座標，計算沿途格子平均/最高風險、暴力佔比 |
| 圖層切換 | 預測類別 / 暴力熱力圖 / 告警標記，各自獨立開關 |
| **Top 10 高風險格子** | 依當前城市+時段排序，點擊 → 地圖飛到該格子（zoom 16）+ 詳情面板 |
| **Time Slot Distribution** | 4 個時段的 violent/property/other 比例堆疊條，加權 = `模型機率 × 事件數`，點擊 → 切換地圖時段 |
| **黑/白主題切換** | CartoDB dark_all ↔ light_all 底圖同步切換 |

### 格子顏色規則
- **顏色**：類別（紅=violent / 藍=property / 綠=other）
- **透明度**：信心度（高≥0.80 / 中0.55–0.80 / 低<0.55）

### Grid 資料欄位
```
lat, lon       格子中心座標
ts             時段（0–3）
cnt            事件數
dom            真實主導類別
pred           模型預測類別
conf           信心度（最高類別機率）
tier           high / medium / uncertain
ok             是否預測正確（boolean）
risk           風險分數（0–100）
pv, pp, po     三類別模型機率
gap            第一名與第二名類別的機率差
ent            熵值（犯罪分佈均勻度）
```

---

## Known Issues & Notes

- `requirements.txt` 缺少 `catboost`、`lightgbm`、`scipy`、`joblib`、`torch`，需手動補裝
- London / DC / Philadelphia / West Yorkshire 的 `.ipynb` 版本尚未建立，目前只有 `.py` 版
- Karachi 有三個版本原始資料（`karachi_raw.csv`、`karachi_synthetic_raw.csv`、`karachi_crime_dataset.csv`），前處理腳本有對應的兩個函式分別處理
- DC 原始資料缺少 2022 年（以年份命名的 CSV 中無 2022 版）
- West Yorkshire 原始資料按月份分資料夾，覆蓋 2023-04 至 2026-03（以 data.police.uk 格式）
- `notebook/catboost_info/` 為 CatBoost 訓練 log，非程式碼，可忽略
- `crime_map_v3.html` 約 4.5 MB，絕大部分為嵌入式 JSON 格子資料；編輯時避免直接讀取整個檔案
