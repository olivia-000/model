# 系統介紹 — Spatiotemporal Crime Transfer Learning

## 專案目標

本系統旨在解決傳統犯罪預測模型只能單一城市使用的限制。透過**跨城市遷移學習（Cross-City Transfer Learning）**，以資料豐富的城市（NYC、Chicago）訓練模型，並推廣至資料稀缺的城市，實現犯罪熱點分類預測。

---

## 研究問題

> 能否用一個城市訓練的犯罪預測模型，直接預測另一個城市的犯罪熱點類別？

---

## 資料來源

| 城市 | 資料來源 | 筆數 | 時間範圍 |
|------|----------|------|----------|
| New York City | NYPD Open Data API | 約 9,469,817 | 2006–2024 |
| Chicago | Chicago Data Portal API | 約 8,144,765 | 2001–2024 |
| Los Angeles | LA Open Data API | 約 875,087 | 2020–2024 |
| Karachi | Kaggle 合成犯罪資料集 | 100,000 | 2020–2025 |

---

## 犯罪類別定義

所有城市的犯罪紀錄統一對應至 4 大類：

| 類別 | 說明 | 範例 |
|------|------|------|
| `violent` | 暴力犯罪 | 攻擊、搶劫、殺人、強姦 |
| `property` | 財產犯罪 | 竊盜、入室盜竊、汽車竊盜、詐欺 |
| `drug` | 毒品犯罪 | 持有、販賣毒品 |
| `public_order` | 妨害公序 | 擾亂公共安寧、違反交通法規 |

> 實驗中 `drug` 與 `public_order` 因 Grid 層級資料稀疏，合併為 `other`，最終使用 3 個類別。

---

## 主要成果

### 各城市基礎模型表現

| 城市 | 模型 | Precision Macro | Accuracy |
|------|------|----------------|----------|
| NYC | LightGBM | 0.720 | — |
| NYC | CatBoost (校正後) | 0.826 | 0.69 |
| Chicago | LightGBM | 0.674 | 0.80 |
| LA | CatBoost | 0.485 | 0.63 |

### 遷移學習結果

| 實驗 | Precision Macro | 說明 |
|------|----------------|------|
| Chicago 本地訓練（基準） | 0.439 | 從頭訓練 |
| **NYC → Chicago 零樣本** | **0.614** | 超越本地基準 +17.5pp |
| NYC → Chicago Fine-tune 50% | 0.439 | 負遷移 |
| NYC → Karachi 零樣本 | 0.389 | 跨文化失敗 -23.6pp |

---

## 關鍵發現

1. **Grid 層級預測精度遠高於事件層級**（0.65–0.80 vs ~0.35）
2. **同國零樣本遷移有效**：NYC → Chicago 無需任何目標城市資料即超越本地模型
3. **Fine-tuning 造成負遷移**：加入目標城市資料反而降低效能
4. **跨文化遷移失敗**：NYC/Chicago → Karachi 均失敗，且表現完全相同，確認障礙來自文化/結構差異
5. **`hist_*` 3 個特徵勝過全部 26 個特徵**，但也是跨文化遷移失敗的根本原因

---

## 環境需求

- Python 3.10
- 主要套件：`catboost`、`lightgbm`、`scikit-learn`、`pandas`、`numpy`、`scipy`、`folium`、`shap`

### 安裝

```bash
conda create -n crime-tl python=3.10 -y
conda activate crime-tl
pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 \
            catboost lightgbm xgboost==2.0.3 scipy joblib \
            folium==0.16.0 matplotlib==3.9.0 seaborn==0.13.2 \
            tqdm==4.66.4 jupyter
```

### 執行

```bash
python src/01_download.py    # 下載原始資料
python src/02_preprocess.py  # 清洗與合併
jupyter notebook             # 開啟 notebook 進行訓練
```
