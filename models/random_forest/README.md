# RoadSafe Delhi — Random Forest Model Development

## 1. Assigned Ownership & Allocation
- **Assigned Member:** Nakshtr
- **Assigned Model:** Random Forest
- **Role:** Model development, hyperparameter tuning, literature research, and model-specific reporting.

## 2. Common Approved Dataset Location
All members MUST use the exact approved common dataset and partitions:
- **Common Parquet:** `data_final/delhi_road_risk_common_dataset.parquet`
- **Training Partition:** `data_final/train.parquet` (11,502 segments, 59.2%)
- **Validation Partition:** `data_final/validation.parquet` (4,229 segments, 21.8%)
- **Test Partition:** `data_final/test.parquet` (3,686 segments, 19.0%)
- **Feature Contract:** `data_final/feature_contract.json`

## 3. Target Variable
- **Assigned Target:** `is_crash_segment`
- **Target Description:** Binary hazard occurrence classification (0 vs 1)

## 4. Approved Features (Zero-Leakage Contract)
- **Numerical Features (9):**
  `length_m`, `est_lanes`, `is_oneway`, `is_bridge`, `has_maxspeed`, `dist_to_center_km`, `elevation_m`, `slope_deg`, `builtup_pct`
- **Categorical Feature (1):**
  `type` (road functional class: motorway, trunk, primary, secondary, tertiary, etc.)
- **Quarantined Columns (STRICTLY FORBIDDEN AS FEATURES):**
  `risk_score`, `crash_count`, `fatal_count`, `major_count`, `minor_count`, `total_casualties`, `crash_density_per_km`, `EPDO_score`, `pca_comp_1..3`, coordinates, IDs.

## 5. Dedicated Development Notebook
Open and run your pre-created model development notebook:
`models/random_forest/notebooks/Random_Forest_Model_Development.ipynb`

## 6. Preprocessing Requirements
- Preprocessing objects (`StandardScaler`, `OneHotEncoder`, etc.) MUST be fitted **exclusively on `train.parquet`**.
- Validation and test partitions must be transformed using the fitted training objects. Never fit or tune on test data.

## 7. Expected Outputs & Artifacts
After developing your model, save:
- Model checkpoint: `artifacts/random_forest_model.<keras/joblib/pkl>`
- Preprocessing objects: `artifacts/scaler.joblib`, `artifacts/encoder.joblib`
- Selected decision threshold (if classification): `artifacts/threshold.json`
- Metrics report: `outputs/metrics.json`
- Plots: `outputs/confusion_matrix.png` (or regression plot), `outputs/evaluation_report.md`
