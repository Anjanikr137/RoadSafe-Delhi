# RoadSafe Delhi — Linear Regression (Ridge Baseline) Model Development

**Assigned Team Member:** Anushree  
**Model Name:** Ridge Linear Regression (`sklearn.linear_model.Ridge`)  
**Assigned Target:** `crash_density_per_km` (Continuous Crash Rate per Kilometer)  
**Role:** Parametric Continuous Rate Baseline, Literature Research, Feature Evaluation  
**Status:** **INDEPENDENTLY VERIFIED & AUDITED**

---

## 1. Executive Summary & Objective

In the RoadSafe Delhi multi-model benchmark, while the classification models (ANN, Logistic Regression, Random Forest, SVM) predict the binary spatial occurrence of road hazards (`is_crash_segment`), **Linear Regression** is formulated as the **parametric continuous rate baseline**. It models the continuous crash intensity normalized by segment length:

$$\text{crash\_density\_per\_km} = \frac{\text{crash\_count}}{\text{length\_km}} = \frac{\text{crash\_count}}{\text{length\_m} / 1000}$$

### Why Ridge Regression ($\alpha = 10.0$)?
Ordinary Least Squares (OLS) regression is sensitive to multicollinearity among structural road hierarchy indicators (e.g., motorway vs. primary vs. secondary corridors) and spatial-environmental covariates. L2 regularization (Ridge regression with $\alpha = 10.0$) was selected and validated because:
1. **Shrinkage Stability:** It penalizes large regression coefficients ($\sum \beta_j^2$), stabilizing variance and preventing overfitting to spatial idiosyncrasies in dense corridors.
2. **Singularity Prevention:** It guarantees numerical stability in the inversion of the Gram matrix $(X^T X + \alpha I)^{-1} X^T y$ under high-dimensional one-hot encoded categories.
3. **Generalization Performance:** On the official spatial validation set, Ridge ($\alpha = 10.0$) achieved an $R^2$ of $0.4251$ and an RMSE of $1.3644\text{ crashes/km}$.

---

## 2. Common Approved Dataset & Spatial Partitioning

All training, validation, and evaluation strictly follow the official spatial block cross-validation partitioning scheme established for RoadSafe Delhi:

- **Canonical Dataset:** `data_final/delhi_road_risk_common_dataset.parquet` (19,417 road segments)
- **Spatial Partitions (Zero-Leakage Block Partitioning):**
  - **Training Partition:** `data_final/train.parquet` — **11,502 road segments** (59.2%)
  - **Validation Partition:** `data_final/validation.parquet` — **4,229 road segments** (21.8%)
  - **Test Partition:** `data_final/test.parquet` — **3,686 road segments** (19.0%)
- **Zero Spatial Leakage:** The spatial blocks across Train, Validation, and Test partitions are mutually disjoint:
  $$\text{Blocks}(\text{Train}) \cap \text{Blocks}(\text{Val}) = \emptyset, \quad \text{Blocks}(\text{Train}) \cap \text{Blocks}(\text{Test}) = \emptyset, \quad \text{Blocks}(\text{Val}) \cap \text{Blocks}(\text{Test}) = \emptyset$$

### Target Distribution Across Partitions:
| Partition | Row Count | Positive Segments ($>0$) | Zero Segments ($=0$) | Mean Crash Density (crashes/km) | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 11,502 | 462 (4.02%) | 11,040 (95.98%) | 0.2238 | 1.8340 |
| **Validation** | 4,229 | 170 (4.02%) | 4,059 (95.98%) | 0.2353 | 1.8021 |
| **Test** | 3,686 | 148 (4.02%) | 3,538 (95.98%) | 0.1834 | 1.7346 |

---

## 3. Approved Feature Contract & Preprocessing

The model operates under strict zero-leakage constraints using **10 raw predictors** expanded into **23 encoded features**:

### A. 9 Numerical Predictors (StandardScaler fitted strictly on Train):
1. `length_m`: Segment length in meters.
2. `est_lanes`: Estimated directional lane count.
3. `is_oneway`: Binary indicator (1 if one-way, 0 otherwise).
4. `is_bridge`: Binary indicator (1 if elevated bridge / flyover, 0 otherwise).
5. `has_maxspeed`: Binary indicator (1 if statutory speed limit posted, 0 otherwise).
6. `dist_to_center_km`: Euclidean distance to Delhi center (Connaught Place) in km.
7. `elevation_m`: Elevation above sea level in meters (extracted from SRTM DEM).
8. `slope_deg`: Topographic terrain slope in degrees.
9. `builtup_pct`: Surrounding urban built-up surface percentage (ESA WorldCover 10m).

### B. 1 Categorical Predictor (OneHotEncoder fitted strictly on Train):
- `type`: OSM functional road classification, encoded into 14 dummy indicators:
  - `type_motorway`, `type_motorway_link`, `type_trunk`, `type_trunk_link`
  - `type_primary`, `type_primary_link`, `type_secondary`, `type_secondary_link`
  - `type_tertiary`, `type_tertiary_link`, `type_residential`, `type_living_street`
  - `type_unclassified`, `type_road`

### C. Quarantined Columns (Strictly Excluded to Prevent Target Leakage):
- Casualty counts: `fatal_count`, `major_count`, `minor_count`, `total_casualties`
- Aggregated crash metrics: `crash_count`, `risk_score`, `EPDO_score`
- Spatial identifiers & coordinates: `osm_id`, `spatial_block`, `geometry`, `center_lat`, `center_lon`
- Unsupervised components: `pca_comp_1`, `pca_comp_2`, `pca_comp_3`

---

## 4. Verified Model Performance

The trained Ridge model (`alpha=10.0, random_state=42`) was independently reloaded and evaluated on the official validation and untouched holdout test sets:

| Evaluation Metric | Validation Set (`validation.parquet`) | Holdout Test Set (`test.parquet`) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **0.2448 crashes/km** | **0.215807 crashes/km** | Average absolute residual deviation on unseen road segments |
| **Root Mean Squared Error (RMSE)** | **1.3644 crashes/km** | **1.407025 crashes/km** | Penalizes large residual errors in high-density corridors |
| **Coefficient of Determination ($R^2$)** | **0.4251** | **0.343341** | Explains 34.33% of variance in crash density across unseen blocks |

*Numerical Verification: Confirmed 100% exact match across independent Python evaluation, automated scripts, and original notebook execution.*

---

## 5. Directory Structure & Artifacts Inventory

```
models/linear_regression/
├── README.md                                          # This documentation file
├── artifacts/                                         # Serialized production artifacts
│   ├── linear_regression_model.joblib                 # Trained Ridge model (alpha=10.0, random_state=42)
│   ├── scaler.joblib                                  # StandardScaler fitted strictly on train.parquet
│   ├── encoder.joblib                                 # OneHotEncoder fitted strictly on train.parquet
│   ├── feature_list.json                              # 23 feature schema names
│   └── model_config.json                              # Model hyperparameters & evaluation metadata
├── notebooks/                                         # Interactive development notebooks
│   └── Linear_Regression_Model_Development.ipynb     # Cleaned development notebook (26 cells executed)
├── outputs/                                           # Evaluation outputs & visual diagnostics
│   ├── metrics.json                                   # JSON test metrics (MAE, RMSE, R2)
│   ├── predictions.csv                                # Test set predictions (3,686 segments, 7 columns)
│   ├── sample_predictions.csv                         # Top 10 sample predictions for quick inspection
│   ├── evaluation_report.md                           # Detailed technical evaluation report
│   ├── coefficients.png                               # Standardized regression coefficients plot
│   └── residuals_distribution.png                     # Residual distribution and Q-Q diagnostic plot
└── src/                                               # Modular Python scripts
    ├── __init__.py                                    # Package initialization
    ├── preprocessing.py                               # RoadSafeLinearPreprocessor class
    ├── train.py                                       # Training & validation pipeline script
    ├── evaluate.py                                    # Test evaluation & artifact generation script
    └── predict.py                                     # Standalone inference class (LinearRoadRiskPredictor)
```

---

## 6. Execution Instructions

### A. Running the Interactive Development Notebook
Open the development notebook in VS Code or Jupyter:
```bash
jupyter notebook models/linear_regression/notebooks/Linear_Regression_Model_Development.ipynb
```
*Note: The notebook runs end-to-end using project-relative paths. All machine-specific personal directories have been sanitized.*

### B. Running the Modular Pipeline Scripts
All scripts are executable directly from the repository root:

1. **Evaluate Saved Model on Test Partition:**
   ```bash
   python models/linear_regression/src/evaluate.py
   ```
   Loads `artifacts/linear_regression_model.joblib`, runs inference on `data_final/test.parquet`, and updates `outputs/metrics.json`, `outputs/predictions.csv`, and visual plots.

2. **Run Standalone Inference / Batch Prediction:**
   ```bash
   python models/linear_regression/src/predict.py
   ```
   Initializes `LinearRoadRiskPredictor`, loads artifacts, and outputs predictions on sample test segments.

3. **Train Model from Scratch:**
   ```bash
   python models/linear_regression/src/train.py
   ```
   *Caution: Do not run `train.py` unless intending to retrain and regenerate model weights.*

---

## 7. Domain Research Literature Review Status

- **Assigned Topic:** Linear and Count Regression in Road Safety & Crash Frequency Modeling (5 Papers)
- **Status:** **PENDING** (To be added by Anushree in subsequent documentation update)
- **Planned Literature Focus:**
  1. Parametric rate baselines vs. Poisson / Negative Binomial count models in spatial crash modeling.
  2. Multi-collinearity mitigation using L2 (Ridge) and L1 (Lasso) regularization on geometric road features.
  3. Addressing zero-inflation (96% zero segments) in urban road network safety analyses.
  4. Segment length normalization and exposure metrics ($VMT$ vs. kilometer length).
  5. Empirical interpretability of linear coefficients in transport policy and safety interventions.
