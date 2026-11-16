# RoadSafe Delhi — Random Forest Classification Pipeline

**Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
**Assigned Team Member:** Nakshtr  
**Target Variable:** `is_crash_segment` (Binary Hazard Occurrence: 1 = Crash, 0 = Non-Crash)  
**Evaluation Partition:** Untouched Test Partition (`data_final/test.parquet`, 3,686 segments)  
**Decision Threshold:** Frozen Validation Threshold $t^* = 0.516077$  
**Status:** **INDEPENDENTLY VERIFIED & FULLY EXECUTED**

---

## 1. Problem Definition & Objectives

The primary objective of the Random Forest model is to classify OpenStreetMap road segments in the National Capital Territory of Delhi as either historically hazardous (`is_crash_segment = 1`) or baseline non-crash (`is_crash_segment = 0`).

### Methodological Distinction: Historical Classification vs. Future Forecasting
> [!IMPORTANT]
> This model performs **historical spatial hazard association mapping** based on static road geometry, network hierarchy, and surrounding land-use features. It classifies road segments that accumulated historical crashes, serving municipal prioritization for physical safety audits and infrastructure retrofits. It is **not** a real-time or future accident forecasting system (which would require high-frequency dynamic inputs such as live traffic congestion, weather telemetry, and driver behavior).

### Why Random Forest?
1. **Non-Linear Decision Surfaces:** Crash risk involves complex threshold interactions (e.g., steep slope combined with narrow carriageway and high built-up density) that linear models cannot capture without manual feature engineering.
2. **Robustness to Multicollinearity:** Geometric features and road classifications are naturally correlated; ensemble bagging mitigates variance.
3. **Scale Invariance:** Decision tree partitions rely solely on monotonic order statistics, completely eliminating the need for feature standardization or scaling (`StandardScaler` intentionally omitted).
4. **Dual Interpretability:** Combines fast Mean Decrease in Impurity (Gini MDI) with robust out-of-sample validation Permutation Importance.

---

## 2. Common Approved Dataset & Spatial Partitions

The model strictly adheres to the official, zero-leakage spatial cross-validation split of the NCT of Delhi:

- **Canonical Dataset:** `data_final/delhi_road_risk_common_dataset.parquet` (19,417 segments)
- **Spatial Partitions:**
  - **Training Partition:** `data_final/train.parquet` — **11,502 segments** (491 positive, 4.27%)
  - **Validation Partition:** `data_final/validation.parquet` — **4,229 segments** (177 positive, 4.19%)
  - **Test Partition:** `data_final/test.parquet` — **3,686 segments** (113 positive, 3.07%)
- **Zero Spatial Leakage:** The spatial blocks across partitions are mutually disjoint:
  $$\text{Blocks}(\text{Train}) \cap \text{Blocks}(\text{Val}) = \emptyset, \quad \text{Blocks}(\text{Train}) \cap \text{Blocks}(\text{Test}) = \emptyset, \quad \text{Blocks}(\text{Val}) \cap \text{Blocks}(\text{Test}) = \emptyset$$

---

## 3. Approved Feature Contract & Preprocessing

The model operates under strict zero-leakage constraints using **10 raw predictors** mapped into **23 encoded features**:

### A. 9 Numerical Predictors (Scale-Invariant, Raw Inputs Preserved):
1. `length_m`: Road segment length in meters.
2. `est_lanes`: Estimated directional lane count.
3. `is_oneway`: Binary indicator (1 if one-way, 0 otherwise).
4. `is_bridge`: Binary indicator (1 if bridge/flyover, 0 otherwise).
5. `has_maxspeed`: Binary indicator (1 if statutory speed limit posted, 0 otherwise).
6. `dist_to_center_km`: Euclidean distance to Delhi center (Connaught Place) in km.
7. `elevation_m`: Elevation above sea level in meters (SRTM DEM).
8. `slope_deg`: Terrain slope in degrees.
9. `builtup_pct`: Surrounding urban built-up surface percentage (ESA WorldCover 10m).

### B. 1 Categorical Predictor (One-Hot Encoded into 14 Categories):
- `type`: OSM functional road hierarchy (`type_motorway`, `type_primary`, `type_secondary`, `type_tertiary`, `type_residential`, etc.).
- Preprocessing: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted **strictly on `train.parquet`**.

### C. Quarantined Columns (Strictly Forbidden to Prevent Target Leakage):
- Casualty counts: `fatal_count`, `major_count`, `minor_count`, `total_casualties`
- Target/Risk indicators: `crash_count`, `risk_score`, `EPDO_score`, `is_high_risk`, `risk_tier`
- Temporal crash ratios: `night_crash_ratio`, `peak_crash_ratio`, `fog_crash_ratio`
- Spatial IDs & unsupervised components: `osm_id`, `spatial_block`, `pca_comp_1..3`

---

## 4. Hyperparameter Tuning & Model Selection

To address minority class imbalance (~4.02% crash segments), candidate configurations across weighting schemes and tree architectures were benchmarked on the official validation partition:

| Model Configuration | Class Weight | Max Depth | Estimators | Val ROC-AUC | Val PR-AUC | Optimal Thresh | Val F1 | Val Precision | Val Recall | Selection Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1_unweighted_d10** | `None` | 10 | 100 | 0.9528 | 0.8548 | 0.6441 | 0.8833 | 1.0000 | 0.7910 | High precision, lower PR-AUC |
| **M2_balanced_d8** | `balanced` | 8 | 150 | 0.9718 | 0.8872 | 0.7735 | 0.8726 | 1.0000 | 0.7740 | Constrained tree depth |
| **M3_balanced_d10** | `balanced` | 10 | 200 | 0.9730 | 0.8860 | 0.6199 | 0.8773 | 0.9597 | 0.8079 | Strong balanced baseline |
| **M4_balanced_d12** | `balanced` | 12 | 200 | 0.9721 | 0.8856 | 0.5949 | 0.8780 | 0.9536 | 0.8136 | Slight tree variance |
| **M6_balanced_subsample_d10** | `balanced_subsample` | 10 | 200 | **0.9749** | **0.8910** | **0.5161** | **0.8807** | **0.9600** | **0.8136** | **SELECTED (Top PR-AUC & ROC-AUC)** |
| **M7_balanced_subsample_d12** | `balanced_subsample` | 12 | 200 | 0.9733 | 0.8871 | 0.5056 | 0.8834 | 0.9664 | 0.8136 | Competitive alternative |

### Selection Rationale:
`M6_balanced_subsample_d10` was chosen because:
1. **Top PR-AUC:** Highest validation Precision-Recall AUC (**0.8910**), reflecting superior minority-class ranking.
2. **Top ROC-AUC:** Highest overall discriminability (**0.9749**).
3. **Natural Probability Calibration:** `balanced_subsample` recalculates weights dynamically per bootstrap sample, centering the optimal threshold ($t^* = 0.516077$) near standard 0.50.

---

## 5. Frozen Threshold Selection Methodology

Decision threshold optimization was conducted **strictly on validation data**:

$$\tau^* = \arg\max_\tau F_1(\text{Val}; \tau) = 0.516077$$

The model weights and decision threshold were **frozen** prior to test partition evaluation. No test data was evaluated during model selection or threshold tuning.

---

## 6. Test Performance Results

Applying the frozen model and frozen validation threshold to the untouched holdout test partition (`data_final/test.parquet`, 3,686 segments):

| Evaluation Metric | Validation Partition (`validation.parquet`) | Holdout Test Partition (`test.parquet`) | Assessment |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.08%** | **99.16%** | High overall classification fidelity across network |
| **Precision** | **0.9600** (96.0%) | **0.8727** (87.3%) | 87.3% of flagged corridors genuinely had crashes |
| **Recall (Sensitivity)** | **0.8136** (81.4%) | **0.8496** (85.0%) | Captures 85.0% of all historical crash segments |
| **F1-Score** | **0.8807** | **0.8610** | Strong harmonic balance in imbalanced setting |
| **ROC-AUC** | **0.9749** | **0.9780** | Exceptional global ranking discriminability |
| **PR-AUC** | **0.8910** | **0.9162** | Outstanding minority-class ranking precision |

### Test Confusion Matrix (3,686 Segments):
| | Predicted Negative (0) | Predicted Positive (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Non-Crash (0)** | **3,559** (TN) | **14** (FP) | 3,573 |
| **Actual Crash (1)** | **17** (FN) | **96** (TP) | 113 |
| **Total** | 3,576 | 110 | 3,686 |

---

## 7. Feature Importance & Domain Interpretability

Both Gini impurity (Mean Decrease in Impurity) and validation Permutation Importance were computed:

```
Top 5 Gini MDI Features:
1. is_oneway          (0.2261)
2. type_residential   (0.1923)
3. length_m           (0.1640)
4. type_service       (0.0751)
5. type_tertiary      (0.0663)

Top 5 Permutation Importances on Validation (PR-AUC Drop):
1. type_residential   (0.2743)
2. type_tertiary      (0.1116)
3. length_m           (0.0531)
4. elevation_m        (0.0288)
5. type_service       (0.0132)
```

### Domain Insights & Non-Causal Associations:
- **Corridor Hierarchy & One-Way Separation:** Arterial and collector roads (primary/secondary) accumulate vastly higher crash frequencies than residential and service lanes. `is_oneway` and functional classifications serve as strong network dividers.
- **Physical Corridor Length (`length_m`):** Longer segments capture higher spatial exposure envelopes.
- **Critical Caution:** These parameters represent **statistical correlations and split frequencies**, **not direct physical causation**. Installing a median or re-labeling a road does not causally alter risk without addressing traffic volumes and driver speeds.

---

## 8. Directory Structure & Artifacts Inventory

```
models/random_forest/
├── README.md                                          # Comprehensive model documentation
├── artifacts/                                         # Serialized production artifacts
│   ├── random_forest_model.joblib                     # Trained Random Forest model (200 trees, max_depth=10)
│   ├── encoder.joblib                                 # OneHotEncoder fitted strictly on train.parquet
│   ├── threshold.json                                 # Frozen validation threshold metadata (t = 0.516077)
│   ├── feature_list.json                              # Schema of 23 input features
│   └── model_config.json                              # Full hyperparameters and test performance metadata
├── notebooks/                                         # Development notebooks
│   └── Random_Forest_Model_Development.ipynb         # Fully executed notebook (12 code cells, 2 visual plots)
├── outputs/                                           # Evaluation outputs & diagnostic figures
│   ├── metrics.json                                   # Test & validation metrics
│   ├── predictions.csv                                # Test set predictions (3,686 records, order preserved)
│   ├── sample_predictions.csv                         # Top 10 sample predictions for quick review
│   ├── evaluation_report.md                           # Detailed technical markdown evaluation report
│   ├── confusion_matrix.png                           # Heatmap confusion matrix (Threshold = 0.5161)
│   ├── roc_pr_curves.png                              # Combined ROC and Precision-Recall curves
│   └── feature_importance.png                         # Dual MDI and Permutation Importance bar plots
└── src/                                               # Modular Python pipeline
    ├── __init__.py                                    # Module marker
    ├── preprocessing.py                               # RoadSafeRFPreprocessor class
    ├── rf_model.py                                    # Model factory and threshold tuning functions
    ├── train.py                                       # Training, validation benchmark, and artifact serialization
    ├── evaluate.py                                    # Test evaluation and plot generation script
    └── predict.py                                     # Standalone RandomForestRoadRiskPredictor class
```

---

## 9. Reproduction Instructions

### A. Run Modular Pipeline from Command Line:
```bash
# 1. Train model and serialize artifacts
python models/random_forest/src/train.py

# 2. Evaluate model on untouched test partition and generate plots
python models/random_forest/src/evaluate.py

# 3. Test standalone inference module
python models/random_forest/src/predict.py
```

### B. Run Interactive Development Notebook:
Open and run `models/random_forest/notebooks/Random_Forest_Model_Development.ipynb`. All cells execute sequentially using relative paths.

---

## 10. Limitations

1. **Static Feature Set:** Relies exclusively on static infrastructure and terrain data. Does not incorporate dynamic hourly traffic volume, seasonal weather, or driver behavioral patterns.
2. **Class Imbalance Sensitivity:** Although mitigated through `balanced_subsample`, minor shifts in threshold can affect the trade-off between false alarms (14 FP) and missed hazards (17 FN).
3. **Observational Nature:** Feature importances must not be interpreted as causal policy interventions.
