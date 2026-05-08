# Spatiotemporal Crime Transfer Learning
# 時空犯罪遷移學習

### Cross-City Crime Hotspot Prediction with Domain Adaptation
### 跨城市犯罪熱點預測與領域自適應

> **EN** — Predicting dominant crime categories across 7 cities using grid-level spatiotemporal features, calibrated ensemble models, domain adversarial transfer learning, and an interactive map dashboard.
>
> **ZH** — 以格子層級時空特徵、校正整合模型、域對抗遷移學習，對 7 個城市進行犯罪主導類別預測，並搭配互動式地圖儀表板。

---

## Overview / 專案概覽

**EN** — Traditional crime prediction models are city-specific and fail where data is scarce. This project explores cross-city transfer learning for crime hotspot classification: can a model trained on data-rich cities (NYC, Chicago) generalize to other cities with minimal retraining?

**ZH** — 傳統犯罪預測模型只能單一城市使用，在資料稀缺處難以應用。本研究探索跨城市遷移學習：以資料豐富的城市（NYC、Chicago）訓練的模型，能否以最少重新訓練的代價推廣至其他城市？

**Key Contributions / 主要貢獻：**
- **EN** Redefined the task from event-level to **grid-level dominant category prediction**, following established literature
- **ZH** 將任務從事件層級重新定義為 **格子層級主導類別預測**，遵循既有文獻框架

- **EN** Discovered and corrected **feature-target leakage** in grid-based splits by switching to temporal splits
- **ZH** 發現並修正格子切分的 **特徵-目標洩漏**，改採時間切分

- **EN** Demonstrated **zero-shot transfer** NYC→Chicago outperforms local baseline (+17.5pp precision)
- **ZH** 證明 **零樣本遷移** NYC→Chicago 超越本地基準（精確率 +17.5pp）

- **EN** Identified **negative transfer** in same-country fine-tuning and cross-cultural failure with SHAP explanation
- **ZH** 量化同國 fine-tuning 的**負遷移**與跨文化遷移失敗，並以 SHAP 解釋根本原因

- **EN** Built **DANN v1/v2** (Domain Adversarial Neural Network) for deep domain adaptation
- **ZH** 實作 **DANN v1/v2**（域對抗神經網路）進行深度領域自適應

- **EN** Built an **interactive map dashboard** (`crime_map_v3.html`) covering all 7 cities
- **ZH** 建立覆蓋全部 7 城市的 **互動地圖儀表板**（`crime_map_v3.html`）

---

## Data / 資料來源

| City / 城市 | Source / 來源 | Records / 筆數 | Period / 時間範圍 | Download / 取得方式 |
|-------------|--------------|---------------|-----------------|-------------------|
| New York City | NYPD Open Data API | ~9,469,817 | 2006–2024 | Auto / 自動 |
| Chicago | Chicago Data Portal API | ~8,144,765 | 2001–2024 | Auto / 自動 |
| Los Angeles | LA Open Data API | ~875,087 | 2020–2024 | Auto / 自動 |
| London | data.police.uk | — | 2018–2024 | Manual / 手動 |
| Philadelphia | OpenDataPhilly | — | 2019–2024 | Manual / 手動 |
| Washington DC | Open Data DC | — | 2019–2024 | Manual / 手動 |
| West Yorkshire | data.police.uk | — | 2023–2026 | Manual / 手動 |
| Karachi | Kaggle Synthetic | 100,000 | 2020–2025 | Manual / 手動 |

---

## Crime Category Mapping / 犯罪類別對應

**EN** — All cities map to 4 unified labels. In practice, `drug` and `public_order` are merged into `other` due to sparse grid-level support, leaving **3 effective classes**.

**ZH** — 所有城市統一對應至 4 大類。因格子層級資料稀疏，`drug` 與 `public_order` 合併為 `other`，實際使用 **3 個有效類別**。

| Label / 標籤 | EN | ZH |
|-------------|----|----|
| `violent` | Assault, robbery, homicide, rape | 攻擊、搶劫、殺人、強姦 |
| `property` | Theft, burglary, motor vehicle theft, fraud | 竊盜、入室盜竊、汽車竊盜、詐欺 |
| `drug` → `other` | Drug possession/trafficking | 毒品持有/販賣 |
| `public_order` → `other` | Disorderly conduct, traffic violations | 妨害公序、交通違規 |

---

## Methodology / 研究方法

### Task Definition / 任務定義

**EN** — Aggregate crime records into **0.01° × 0.01° spatial grids (~1 km²) × 4 time slots** (midnight / morning / afternoon / night). Predict the **dominant crime category** (most frequent) for each grid-time unit. Grid-level framing is intentional — event-level prediction hits a ~0.35 precision ceiling.

**ZH** — 將犯罪紀錄聚合為 **0.01° × 0.01° 空間格子（約 1 km²）× 4 個時段**（深夜 / 早晨 / 下午 / 夜晚），預測每個格子-時間單位的**主導犯罪類別**。格子層級設計有意為之——事件層級精確率上限約 0.35。

### Data Split / 資料切分（Temporal — No Leakage / 時間切分，無洩漏）

```
Train  Earliest ~ 12 months before end  (~74%)  ← hist_* computed here only
Val    Second-to-last 6 months          (~13%)  ← calibration & threshold tuning
Test   Last 6 months                   (~13%)  ← final evaluation only
```

> **EN** `hist_*` features (historical crime composition per grid) must be computed on Train only and joined to Val/Test. Never compute across the full dataset before splitting.
>
> **ZH** `hist_*` 特徵（格子歷史犯罪組成）只能在 Train 期間計算，再 join 至 Val/Test。絕不能在切分前就計算全資料的 hist_*。

### Features / 特徵工程（27 total）

| Group / 群組 | Features / 特徵 | Note / 說明 |
|-------------|----------------|-------------|
| Historical composition / 歷史組成 | `hist_violent`, `hist_property`, `hist_other` | **3 features alone outperform all 26 combined / 單獨 3 個勝過全部 26 個** |
| Spatial lag / 空間延遲 | `lag_violent`, `lag_property`, `lag_other` | Neighbour grid averages / 鄰近格子平均 |
| Relative percentile / 相對百分位 | `violent_pct`, `density_pct`, `entropy_pct`, `dom_gap_pct` | City-wide rank / 全城市分位數 |
| Stability / 穩定性 | `top1_ratio`, `dominance_gap`, `entropy` | Distribution concentration / 分佈集中度 |
| Temporal / 時間 | `time_slot`, `is_weekend`, sin/cos encodings | Cyclic encoding / 週期性編碼 |
| Spatial / 空間 | `lat_bin`, `lon_bin`, `lat_norm`, `lon_norm` | Grid coords / 格網座標 |

### Models & Calibration / 模型與機率校正

**EN**
1. **CatBoost** + **LightGBM** with balanced class weights
2. **Platt Scaling** / **Isotonic Regression** calibration on validation set
3. **Confidence threshold sweep** — global and per-class thresholds for precision-coverage tradeoff
4. **Two-Stage Model** — first classify violent vs. non-violent, then property vs. other

**ZH**
1. **CatBoost** + **LightGBM**，使用 balanced class weights
2. **Platt Scaling** / **Isotonic Regression** 在 Val 集上校正機率
3. **信心閾值掃描** — 全域及類別別閾值，精確率-覆蓋率權衡
4. **兩階段模型** — 先判 violent vs. non-violent，再判 property vs. other

---

## Results / 實驗結果

### Baseline Performance / 基礎模型表現

| City / 城市 | Model / 模型 | Precision Macro | Precision Weighted | F1 Macro | Accuracy |
|------------|-------------|----------------|-------------------|----------|----------|
| NYC | LightGBM | 0.720 | 0.731 | 0.576 | — |
| NYC | CatBoost | 0.573 | 0.708 | 0.587 | 0.69 |
| Chicago | LightGBM | 0.674 | 0.797 | 0.665 | 0.80 |
| Chicago | CatBoost | 0.648 | 0.795 | 0.653 | 0.81 |
| LA | CatBoost | 0.485 | 0.626 | 0.483 | 0.63 |

### Best Results with Calibration + Threshold / 校正 + 閾值最佳結果

| City / 城市 | Method / 方法 | Precision Macro | Coverage / 覆蓋率 |
|------------|--------------|----------------|-----------------|
| NYC | LightGBM Platt + t=0.45 | **0.798** | 99.4% |
| NYC | CatBoost Iso + t=0.75 | **0.826** | 27.2% |
| Chicago | LightGBM (no threshold) | **0.674** | 100% |

### Interactive Map Accuracy / 互動地圖準確率

| City / 城市 | Grids / 格子數 | Map Accuracy / 地圖準確率 | Avg Risk / 平均風險 |
|------------|--------------|--------------------------|-------------------|
| NYC | 3,501 | 71.0% | 2.3 |
| Chicago | 2,740 | 81.9% | 27.2 |
| LA | 4,130 | 68.2% | 11.5 |
| London | 7,381 | 57.9% | 23.3 |
| Philadelphia | 1,495 | 81.5% | 0.9 |
| DC | 633 | 96.5% | 0.0 |
| West Yorkshire | 4,940 | 69.3% | 67.9 |

---

## Transfer Learning / 遷移學習

### Same-Country Transfer / 同國遷移：NYC → Chicago

| Scenario / 情境 | Precision Macro | F1 Macro | Note / 說明 |
|----------------|----------------|----------|-------------|
| Chicago baseline | 0.439 | 0.415 | Trained from scratch / 從頭訓練 |
| **Zero-shot NYC→Chicago** | **0.614** | 0.212 | No target data / 無目標城市資料 |
| Fine-tune 10% | 0.437 | 0.412 | Negative transfer / 負遷移 |
| Fine-tune 20% | 0.438 | 0.414 | Negative transfer / 負遷移 |
| Fine-tune 50% | 0.439 | 0.413 | Negative transfer / 負遷移 |
| Teacher-Student (T=3.0) | 0.422 | 0.424 | Soft labels / 軟標籤 |

> **EN Key finding** — Zero-shot exceeds local baseline by +17.5pp. Fine-tuning causes negative transfer regardless of data fraction.
>
> **ZH 關鍵發現** — 零樣本超越本地基準 +17.5pp；無論加入多少目標資料，Fine-tuning 均造成負遷移。

### Cross-Cultural Transfer / 跨文化遷移：→ Karachi

| Scenario / 情境 | Precision Macro | F1 Macro |
|----------------|----------------|----------|
| Karachi baseline | 0.625 | 0.625 |
| Zero-shot NYC→Karachi | 0.389 | 0.305 |
| Zero-shot Chicago→Karachi | 0.389 | 0.305 |
| Teacher-Student NYC→Karachi | 0.625 | 0.625 |

> **EN Key finding** — Both US cities produce identical zero-shot results (-23.6pp vs baseline), confirming the barrier is cultural/structural, not city-specific. `hist_*` features encode city-specific crime compositions that do not transfer across cultures.
>
> **ZH 關鍵發現** — 兩個美國城市對 Karachi 的零樣本表現完全相同（-23.6pp），確認障礙來自文化/結構差異。`hist_*` 特徵編碼了城市特有的犯罪組成，無法跨文化遷移。

### DANN (Domain Adversarial Neural Network) / 域對抗神經網路

**EN** — Two implementations for deep transfer:

| Version | Architecture | Key Improvements |
|---------|-------------|-----------------|
| `dann_crime.py` v1 | F(128→64) + Predictor C + Domain Classifier D | Gradient Reversal Layer |
| `dann_v2.py` v2 | Same + 4 enhancements | RF pre-training init, RF top-K feature selection, Ensemble Teacher (RF+LGB+XGB), JSD adaptive λ |

**ZH** — 兩個版本支援深度遷移：

| 版本 | 架構 | 主要改善 |
|------|------|----------|
| `dann_crime.py` v1 | F(128→64) + Predictor C + Domain Classifier D | 梯度反轉層 |
| `dann_v2.py` v2 | 同上 + 4 項改善 | RF 預訓練初始化、RF top-K 特徵篩選、Ensemble Teacher（RF+LGB+XGB）、JSD 自適應 λ 調度 |

Training objective / 訓練目標：`L = L_class(F, C) − λ · L_domain(F, D)`

---

## Feature Ablation Study / 特徵消融研究（NYC）

| Feature Group / 特徵群組 | N | Precision Macro |
|------------------------|---|----------------|
| **`hist_*` only / 僅歷史組成** | 3 | **0.649** |
| All features / 全部特徵 | 26 | 0.567 |
| `hist_*` + `lag_*` | 6 | 0.566 |
| Relative features / 相對特徵 | 4 | 0.547 |
| No `hist_*` / 去除歷史組成 | 23 | 0.520 |
| Stability / 穩定性特徵 | 3 | 0.479 |
| Spatial + Temporal / 時空 | 13 | 0.464 |
| `lag_*` only | 3 | 0.433 |
| Spatial only / 僅空間 | 4 | 0.412 |

---

## Interactive Map Dashboard / 互動地圖儀表板

**EN** — `outputs/maps/crime_map_v3.html` is a self-contained HTML file (no backend required). Open directly in any browser.

**ZH** — `outputs/maps/crime_map_v3.html` 為獨立 HTML 檔案，無需後端，直接用瀏覽器開啟。

| Feature / 功能 | EN | ZH |
|---------------|----|----|
| City switch | 7 cities via tab buttons | 7 城市標籤切換 |
| Time animation | 4 slots, auto-play every 1.8s | 4 時段，自動播放每 1.8 秒 |
| Grid click detail | Category, confidence, risk score, probability bars | 類別、信心度、風險分數、機率長條圖 |
| Alert threshold | Slider to highlight high-risk grids | 滑桿設定風險門檻，標記超標格子 |
| Route risk query | Enter start/end coords → risk along path | 輸入起終點 → 沿途風險統計 |
| Layer toggle | Prediction / Heatmap / Alert markers | 預測類別 / 熱力圖 / 告警標記 |
| **Top 10 panel** | Click → fly to grid (zoom 16) + detail panel | 點擊 → 地圖飛至格子 + 詳情面板 |
| **Time slot distribution** | Weighted bar chart (prob × count); click → switch map | 加權堆疊條（機率×事件數）；點擊 → 切換時段 |
| **Dark / Light theme** | Toggle button + map tile sync (CartoDB dark↔light) | 按鈕切換 + 底圖同步（CartoDB 暗↔亮） |

---

## Project Structure / 專案目錄結構

```
predict-crime/
├── src/
│   ├── 01_download.py              # Data download (NYC/Chicago/LA auto; others manual)
│   └── 02_preprocess.py            # Standardize all cities → all_cities.csv
│                                   # 資料下載（NYC/Chicago/LA 自動；其餘手動）/ 清洗合併
├── notebook/
│   ├── crime_classification_full_NYC.ipynb         # Full pipeline + transfer learning
│   ├── crime_classification_full_Chicage&final_step.ipynb
│   ├── crime_classification_full_LA copy.ipynb
│   ├── crime_classification_full_London.py
│   ├── crime_classification_full_Philadelphia.py
│   ├── crime_classification_full_WestYorkshire.py
│   ├── crime_classification_full_DC.py
│   ├── clean_dc.py                 # DC raw data cleaning / DC 清洗腳本
│   ├── clean_philadelphia.py       # Philadelphia raw data cleaning
│   ├── clean_westyorkshire.py      # West Yorkshire raw data cleaning
│   ├── dann_crime.py               # DANN v1
│   └── dann_v2.py                  # DANN v2 (4 improvements / 4 項改善)
├── data/
│   ├── raw/                        # Original CSVs / 原始資料（未納入版本控制）
│   └── processed/                  # Cleaned & merged / 清洗後資料
│       ├── all_cities.csv          # Main training source / 主要訓練來源
│       ├── nyc_clean.csv
│       ├── chicago_clean.csv
│       └── ...
├── outputs/
│   ├── models/                     # Trained models, calibrators, grid risk CSVs
│   │                               # 訓練模型、校正器、格子風險分數
│   ├── eda/                        # Confusion matrices, SHAP, ablation charts
│   │                               # 混淆矩陣、SHAP 圖、消融研究圖
│   └── maps/
│       └── crime_map_v3.html       # Main interactive dashboard / 主要互動儀表板
├── CLAUDE.md                       # Full project guide for Claude Code
├── README.md
└── requirements.txt
```

---

## Setup / 環境安裝

```bash
conda create -n crime-tl python=3.10 -y
conda activate crime-tl
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 \
            catboost lightgbm xgboost==2.0.3 scipy joblib \
            folium==0.16.0 matplotlib==3.9.0 seaborn==0.13.2 \
            tqdm==4.66.4 jupyter torch
```

> **EN** `requirements.txt` is incomplete — it omits `catboost`, `lightgbm`, `scipy`, `joblib`, and `torch`. Use the command above.
>
> **ZH** `requirements.txt` 不完整，缺少 `catboost`、`lightgbm`、`scipy`、`joblib`、`torch`，請使用上方指令安裝。

---

## Running the Pipeline / 執行流程

```bash
# Step 1 — Download / 下載資料
python src/01_download.py
# NYC, Chicago, LA auto-downloaded; others require manual placement
# NYC、Chicago、LA 自動；其餘需手動放置

# Step 2 — Preprocess / 前處理
python src/02_preprocess.py

# Step 2b — Clean new cities / 清洗新城市（DC / Philadelphia / West Yorkshire）
python notebook/clean_dc.py
python notebook/clean_philadelphia.py
python notebook/clean_westyorkshire.py

# Step 3 — Train / 訓練（Jupyter）
jupyter notebook
# Open: notebook/crime_classification_full_NYC.ipynb
# Set CITY = 'NYC' / 'Chicago' / 'LA' in Cell 3

# Step 4 — DANN transfer (optional) / DANN 遷移（可選）
python notebook/dann_crime.py --source NYC --target Chicago
python notebook/dann_v2.py --all --epochs 100
```

---

## Key Findings / 關鍵發現

| # | EN | ZH |
|---|----|----|
| 1 | Grid-level framing achieves 0.65–0.82 precision vs ~0.35 at event level | 格子層級精確率 0.65–0.82，事件層級上限約 0.35 |
| 2 | `hist_*` (3 features) outperforms all 26 features combined | `hist_*` 3 個特徵勝過全部 26 個特徵 |
| 3 | Zero-shot NYC→Chicago beats local baseline by +17.5pp | 零樣本 NYC→Chicago 超越本地基準 +17.5pp |
| 4 | Fine-tuning causes negative transfer (same-country) | Fine-tuning 造成負遷移（即使同國） |
| 5 | Cross-cultural transfer (→Karachi) fails; both US cities produce identical results | 跨文化遷移失敗；兩個美國城市結果完全相同 |
| 6 | `hist_*` are the strongest but least transferable features | `hist_*` 最強但最不可遷移 |
| 7 | Calibration makes confidence thresholds reliable for selective prediction | 機率校正使信心閾值可靠，適合選擇性預測 |

---

## Limitations / 限制

- **EN** Karachi data is synthetic with limited spatial resolution and randomised temporal fields
- **ZH** Karachi 為合成資料，空間解析度有限，時間欄位已隨機化

- **EN** LA data only covers 2020–2024 (short history reduces `hist_*` reliability)
- **ZH** LA 資料只有 2020–2024（短歷史降低 `hist_*` 可靠性）

- **EN** DC and Philadelphia have fewer grids (633 / 1,495) — models may overfit
- **ZH** DC 與 Philadelphia 格子數少（633 / 1,495），模型可能 overfit

- **EN** `drug` and `public_order` classes merged due to sparse support at grid level
- **ZH** `drug` 與 `public_order` 因格子層級稀疏合併為 `other`

---

## Future Work / 未來方向

- **EN** GNN-based spatial encoding to capture non-grid neighbourhood structure
- **ZH** 以 GNN 捕捉非格子的鄰域結構

- **EN** Real-time Streamlit dashboard with live SHAP explanations
- **ZH** 即時 Streamlit 儀表板搭配即時 SHAP 解釋

- **EN** Extend DANN v2 to Philadelphia / DC / West Yorkshire target cities
- **ZH** 將 DANN v2 擴展至 Philadelphia / DC / West Yorkshire 目標城市

- **EN** Multi-task learning: predict crime category + volume tier simultaneously
- **ZH** 多任務學習：同時預測犯罪類別與數量層級
