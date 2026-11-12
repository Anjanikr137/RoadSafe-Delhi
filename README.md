# RoadSafe Delhi — Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning

---

## Project Overview

**Study Area:** National Capital Territory (NCT) of Delhi  
**Objective:** Estimate relative road risk, identify hazardous road corridors (blackspots), interpret driving risk factors, and prioritize spatial locations for preventive municipal safety interventions.

**Lead Author & Coordinator:** Anjani (Team Leader)  
**Repository State:** Audited, 5 Critical Issues Resolved, Common Dataset Prepared, ANN Trained & Tested, Team Notebooks Pre-Created.

---

## 1. Final Team Allocation

| Team Member | Model Assignment | Core Responsibilities | Target Variable | Status |
|:---|:---|:---|:---|:---|
| **Anjani (Team Leader)** | **Artificial Neural Network (ANN)** | Dataset auditing, raster extraction, feature contract governance, ANN model training, threshold tuning, and team coordination. | `is_crash_segment` (Binary) | **Fully Implemented & Evaluated** |
| **Nakshtr** | **Random Forest** | Non-parametric tree ensemble, data cleaning, Gini feature importance analysis, and maintenance. | `is_crash_segment` (Binary) | Ready for execution |
| **Kaavish** | **Support Vector Machine (SVM)** | Maximum-margin hyperplane classification, kernel geometries, and literature research (3 papers). | `is_crash_segment` (Binary) | Ready for execution |
| **Anushree** | **Linear Regression (Baseline)** | Interpretable continuous exposure rate modeling, feature engineering support, and literature research (5 papers). | `crash_density_per_km` (Continuous) | Ready for execution |
| **Amit** | **Logistic Regression** | Parametric probabilistic log-odds risk estimation, odds-ratio interpretability, and literature research (5 papers). | `is_crash_segment` (Binary) | Ready for execution |

---

## 2. Resolution of the Five Critical Project Audit Issues

| Issue # | Previous Problem | Technical Solution Implemented | Status |
|:---|:---|:---|:---|
| **Issue 1: Synthetic Features** | `elevation_m`, `slope_deg`, and `builtup_pct` were generated via random mathematical approximations (`np.random`). | Mosaicked NASA SRTM 30m DEM tiles (`N28E076`, `N28E077`) to extract real elevation and computed terrain gradient slope. Processed ESA WorldCover 10m GeoTIFF to extract true regional urban built-up percentages. | **RESOLVED** |
| **Issue 2: Low Crash Matching Rate** | Low matching rate (362 / 2,433 within 100m) because matching considered only 16,630 arterial links and ignored residential/service corridors. | Expanded matching to the complete OSM vehicular network (198,529 links). Matches within 100m increased to 829. Diagnosed that 1,162 crashes fall outside the OSM shapefile's bounding box; within the coverage area, the match rate is 65.2%. Preserved unmatched points. | **RESOLVED** |
| **Issue 3: Target Imbalance & Threshold** | The 85th percentile of `crash_density` was 0.0 due to 96% zero-inflation, improperly classifying any road with 1 crash as "high risk". | Statistically formulated clean binary targets (`is_crash_segment` for baseline hazard occurrence; `is_high_risk` for severe/recurrent corridors with $\ge 2$ crashes, fatal/major injuries, or EPDO $\ge 5$), continuous rate (`crash_density_per_km`), and 4-tier ordinal classification (`risk_tier`). Test set is never artificially resampled. | **RESOLVED** |
| **Issue 4: Target Leakage** | Crash outcomes (`crash_count`, `fatal_count`, casualties, `risk_score`, weather ratios) could leak into predictors. | Created a formal `feature_contract.json` enforcing strict quarantine. Asserted zero intersection between approved features and targets/quarantined columns. Fitted preprocessors exclusively on `train.parquet`. | **RESOLVED** |
| **Issue 5: Multicollinearity** | High VIF (>13) caused by synthetic feature formulas that artificially coupled built-up percentage and elevation to distance. | Re-extracted real raster features. Recalculated correlations and VIF: **all approved predictors maintain VIF < 1.46**, demonstrating structural independence. | **RESOLVED** |

---

## 3. Directory Structure

```
RoadSafe Delhi/
├── Datasets/                                # Raw immutable source datasets
│   ├── DEM/                                 # NASA SRTM 30m HGT tiles (N28E076, N28E077)
│   ├── Weather Datasets/                    # Safdarjung Meteostat hourly records 2022–2025
│   ├── ESA_WorldCover_10m_2021...tif        # ESA 10m land cover GeoTIFF
│   ├── NewDelhi.osm.shp.zip                 # Full OpenStreetMap road vector shapefile
│   └── Indian-Road-Accident-Dataset...zip   # Third-party crash dataset
├── data_processed/                          # Cleaned intermediate files
├── data_engineered/                         # Feature-engineered intermediate matrices
├── data_final/                              # APPROVED COMMON DATASETS & CONTRACTS
│   ├── delhi_road_risk_common_dataset.parquet
│   ├── train.parquet                        # 11,502 segments (59.2%)
│   ├── validation.parquet                   # 4,229 segments (21.8%)
│   ├── test.parquet                         # 3,686 segments (19.0%)
│   ├── feature_contract.json                # Formal predictor & target schema
│   ├── target_definition.md                 # Statistically justified target formulas
│   ├── split_summary.csv                    # Spatial block partition sizes
│   ├── data_quality_report.md               # Missing value & physical extent audit
│   └── leakage_audit_report.md              # Zero-leakage verification report
├── notebooks/                               # 11 Fixed End-to-End Pipeline Notebooks
├── models/
│   ├── ann/                                 # Anjani (Team Leader) — Fully Trained ANN
│   │   ├── artifacts/                       # ann_model.keras, scaler, encoder, threshold
│   │   ├── outputs/                         # metrics.json, predictions.csv, plots, report
│   │   ├── notebooks/                       # ANN_Model_Development.ipynb (Executed)
│   │   └── src/                             # preprocessing.py, ann_model.py, train.py, evaluate.py
│   ├── random_forest/                       # Nakshtr — Random Forest Development
│   ├── svm/                                 # Kaavish — Support Vector Machine Development
│   ├── linear_regression/                   # Anushree — Linear Regression Baseline
│   └── logistic_regression/                 # Amit — Logistic Regression Development
├── src/                                     # Core pipeline and validation scripts
├── outputs/                                 # Project-wide visual plots, tables, and reports
├── predict.py                               # Reusable inference script for new road segments
├── requirements.txt                         # Environment dependencies
└── README.md
```

---

## 4. Artificial Neural Network (ANN) Performance

The ANN model was trained on `train.parquet` using class weighting to counteract 4% class prevalence, tuned for optimal decision threshold on `validation.parquet`, and evaluated on the untouched `test.parquet`:

- **Test Accuracy:** 99.48%
- **Test Precision:** 1.0000 (0 false positives)
- **Test Recall:** 83.19% (94 of 113 true crash segments identified)
- **Test F1-Score:** 0.9082
- **Test ROC-AUC:** 0.9785
- **Test PR-AUC:** 0.9289
- **Confusion Matrix:** True Negatives: 3,573 | False Positives: 0 | False Negatives: 19 | True Positives: 94

---

## 5. Quickstart & Model Inference

### A. Run Risk Prediction on Road Records
```bash
python predict.py [input_features.csv] [output_predictions.csv]
```

### B. Run the ANN Development Notebook
Open Jupyter Notebook or VS Code and launch:
```bash
jupyter notebook models/ann/notebooks/ANN_Model_Development.ipynb
```

### C. Instructions for Teammates (Nakshtr, Kaavish, Anushree, Amit)
Each team member has a dedicated folder and pre-created notebook:
1. Navigate to your model folder: `models/<your_model>/`.
2. Review the folder's `README.md` for specific model guidelines.
3. Open your pre-created development notebook: `models/<your_model>/notebooks/<Model>_Model_Development.ipynb`.
4. Run the pre-configured data loader cells that automatically import `train.parquet`, `validation.parquet`, and `test.parquet`.
5. Develop, tune, and evaluate your model using your assigned architecture.
6. Save your trained model checkpoint into `models/<your_model>/artifacts/` and your evaluation metrics into `models/<your_model>/outputs/`.
