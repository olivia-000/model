# Spatiotemporal Crime Transfer Learning
### Cross-City Crime Hotspot Prediction with Domain Adaptation

> Predicting dominant crime categories across NYC, Chicago, Los Angeles, and Karachi using grid-level spatiotemporal features, calibrated ensemble models, and transfer learning — with interactive Folium maps and SHAP explainability.

---

## Overview

Traditional crime prediction models are trained on a single city's data, making them hard to apply where data is scarce. This project explores **cross-city transfer learning** for crime hotspot classification: can a model trained on data-rich cities (NYC, Chicago) generalize to other cities with minimal retraining?

**Key contributions:**
- Redefined the task from event-level classification to **grid-level dominant category prediction** (crime hotspot classification), following established literature
- Discovered and corrected **feature-target leakage** in grid-based splits by switching to temporal splits
- Demonstrated **zero-shot transfer** from NYC to Chicago outperforms the local baseline (precision 0.61 vs 0.44)
- Identified **negative transfer** in cross-cultural transfer (NYC→Karachi), with SHAP and ablation analysis explaining why
- Built an **interactive Folium map** with confidence-tiered predictions overlaid on real map tiles

---

## Results Summary

### Baseline Performance

| City | Model | Precision Macro | Precision Weighted | F1 Macro | Accuracy |
|------|-------|----------------|-------------------|----------|----------|
| NYC | LightGBM | 0.720 | 0.731 | 0.576 | — |
| NYC | CatBoost | 0.573 | 0.708 | 0.587 | 0.69 |
| Chicago | LightGBM (meaningful classes) | 0.674 | 0.797 | 0.665 | 0.80 |
| Chicago | CatBoost (meaningful classes) | 0.648 | 0.795 | 0.653 | 0.81 |
| LA | CatBoost | 0.485 | 0.626 | 0.483 | 0.63 |

### Best Results with Calibration + Threshold

| City | Method | Precision Macro | Coverage |
|------|--------|----------------|----------|
| NYC | LightGBM Platt + t=0.45 | **0.798** | 99.4% |
| NYC | CatBoost Iso + t=0.75 | **0.826** | 27.2% |
| Chicago | LightGBM no threshold | **0.674** | 100% |

---

### Transfer Learning Results

#### NYC → Chicago (Same-country)

| Scenario | Precision Macro | F1 Macro | Note |
|----------|----------------|----------|------|
| Chicago baseline | 0.439 | 0.415 | trained from scratch |
| **Zero-shot NYC→Chicago** | **0.614** | 0.212 | no target data used |
| Fine-tune 10% | 0.437 | 0.412 | negative transfer |
| Fine-tune 20% | 0.438 | 0.414 | negative transfer |
| Fine-tune 50% | 0.439 | 0.413 | negative transfer |
| Teacher-Student (T=3.0) | 0.422 | 0.424 | soft labels |

**Key finding:** Zero-shot transfer exceeds local baseline by +17.5pp. Fine-tuning causes negative transfer.

#### NYC → Karachi & Chicago → Karachi (Cross-cultural)

| Scenario | Precision Macro | F1 Macro |
|----------|----------------|----------|
| Karachi baseline | 0.625 | 0.625 |
| Zero-shot NYC→Karachi | 0.389 | 0.305 |
| Zero-shot Chicago→Karachi | 0.389 | 0.305 |
| Teacher-Student NYC→Karachi | 0.625 | 0.625 |

**Key finding:** Both US cities show identical zero-shot performance on Karachi (-23.6pp vs baseline), confirming the barrier is cultural/structural, not city-specific.

---

### Feature Ablation Study (NYC)

| Feature Group | N Features | Precision Macro |
|--------------|-----------|----------------|
| **hist_* only** | 3 | **0.649** |
| All features | 26 | 0.567 |
| hist_* + lag_* | 6 | 0.566 |
| Relative features only | 4 | 0.547 |
| No hist_* (23 features) | 23 | 0.520 |
| Stability features | 3 | 0.479 |
| Spatial + Temporal | 13 | 0.464 |
| lag_* only | 3 | 0.433 |
| Spatial only | 4 | 0.412 |

**Key finding:** `hist_*` features (3 features) outperform the full 26-feature model.

---

### SHAP Analysis

**Transferable across cities:** `hist_violent`, `hist_property`, `violent_pct`

**City-specific (not transferable):** `lag_property` (high in NYC, near-zero in Chicago), `dominance_gap`

**Implication:** The strongest features encode city-specific crime compositions — explaining why cross-cultural zero-shot transfer fails while same-country transfer succeeds.

---

## Data

| City | Source | Records | Period |
|------|--------|---------|--------|
| New York City | NYPD Open Data | 9,469,817 | 2006–2024 |
| Chicago | Chicago Data Portal | 8,144,765 | 2001–2024 |
| Los Angeles | LA Open Data | 875,087 | 2020–2024 |
| Karachi | Kaggle Synthetic Crime Dataset | 100,000 | 2020–2025 |

---

## Methodology

### Task Definition

**Grid-level dominant category prediction:** Aggregate crime records into 0.01° × 0.01° spatial grids (~1 km²) × 4 time slots. Predict the dominant (most frequent) crime category for each grid-time combination.

### Data Split (Temporal — no leakage)

```
Train:  2006–2021  (~74%)  ← hist_* computed here
Val:    2022       (~13%)  ← early stopping, hyperparameter tuning
Test:   2023–2024  (~13%)  ← final evaluation only
```

### Features (27 total)

| Group | Features |
|-------|----------|
| Spatial | lat_bin, lon_bin, lat_norm, lon_norm |
| Relative percentile | density_pct, violent_pct, entropy_pct, dom_gap_pct |
| Temporal | time_slot, is_weekend, log_count, hour/month/weekday sin+cos |
| Historical composition | hist_violent, hist_property, hist_other |
| Spatial lag | lag_violent, lag_property, lag_other |
| Stability | top1_ratio, dominance_gap, entropy |

### Models & Calibration

- CatBoost + LightGBM with balanced class weights
- Platt Scaling and Isotonic Regression calibration on validation set
- Confidence threshold sweep for precision-coverage tradeoff
- Class-specific thresholds tuned on validation set

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/spatiotemporal-crime-transfer-learning
cd spatiotemporal-crime-transfer-learning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python src/01_download.py   # Download data
python src/02_preprocess.py # Preprocess & merge
# Then open notebooks/crime_classification_full.ipynb
# Set CITY = 'NYC' / 'Chicago' / 'LA' / 'Karachi' in Cell 3
```

---

## Project Structure

```
spatiotemporal-crime-transfer-learning/
├── src/
│   ├── 01_download.py
│   ├── 02_preprocess.py
│   └── remerge.py
├── notebooks/
│   └── crime_classification_full.ipynb
├── outputs/
│   ├── models/      # Models, calibrators, grid_risk CSVs
│   ├── maps/        # map_nyc.html, map_chicago.html
│   └── eda/         # Ablation, SHAP, confusion matrix charts
└── data/
    ├── raw/         # Downloaded CSVs (not committed)
    └── processed/   # Cleaned data
```

---

## Key Findings

1. **Grid-level framing enables high precision** — event-level prediction hits ~0.35 ceiling; grid-level achieves 0.65–0.80+
2. **hist_* features dominate** — 3 historical composition features outperform all 26 features combined
3. **Same-country zero-shot transfer works** — NYC→Chicago zero-shot beats local baseline (+17.5pp)
4. **Fine-tuning causes negative transfer** — adding local data reintroduces city-specific noise
5. **Cross-cultural transfer fails** — NYC/Chicago→Karachi zero-shot underperforms baseline; hist_* features are not transferable across cultural contexts
6. **Calibration matters** — Platt scaling makes confidence thresholds reliable for selective prediction

---

## Limitations

- Karachi data: synthetic dataset with limited spatial resolution and randomized temporal fields
- LA: only 2020–2024 available via current API
- drug and public_order merged into other due to sparse grid-level support

## Future Work

- Add London for non-US comparison
- GNN-based spatial encoding
- Streamlit dashboard with real-time SHAP explanations
- Adversarial domain adaptation for cross-cultural transfer
- Multi-task learning: crime category + volume tier
