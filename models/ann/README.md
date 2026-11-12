# RoadSafe Delhi — Artificial Neural Network (ANN) Model

## 1. Team Allocation & Ownership
- **Assigned Member:** Anjani (Team Leader)
- **Assigned Model:** Artificial Neural Network (Feedforward Deep Learning Architecture)
- **Role & Responsibilities:** Overall coordination, dataset auditing, raster extraction, feature contract enforcement, ANN model development, training, and evaluation.

## 2. Common Dataset & Partitions
- **Approved Common Dataset:** `data_final/delhi_road_risk_common_dataset.parquet`
- **Spatial Train Partition:** `data_final/train.parquet` (11,502 road segments, 59.2%)
- **Spatial Validation Partition:** `data_final/validation.parquet` (4,229 road segments, 21.8%)
- **Spatial Test Partition:** `data_final/test.parquet` (3,686 road segments, 19.0%)
- **Feature Contract:** `data_final/feature_contract.json`

## 3. Target Variable
- **Primary Binary Target:** `is_crash_segment` (1 if segment has $\ge 1$ crash records, 0 otherwise)
- **Alternative Target:** `is_high_risk` (1 if severe or recurrent corridor, 0 otherwise)
- **Base Rate (Imbalance):** 4.02% positive class across the network. Handled via loss class weighting (`neg_count / pos_count`).

## 4. Approved Features (Zero-Leakage Contract)
- **Numerical Features (9):**
  - `length_m`: Segment length in meters
  - `est_lanes`: Estimated carriageway lanes based on OSM hierarchy
  - `is_oneway`: Binary indicator (1 if one-way traffic, 0 otherwise)
  - `is_bridge`: Binary indicator (1 if elevated bridge or flyover, 0 otherwise)
  - `has_maxspeed`: Binary indicator (1 if posted speed limit exists, 0 otherwise)
  - `dist_to_center_km`: Geodesic distance to Connaught Place in km
  - `elevation_m`: Real terrain elevation in meters (NASA SRTM 30m DEM)
  - `slope_deg`: Real terrain slope gradient in degrees (NASA SRTM DEM)
  - `builtup_pct`: Real urban built-up land cover percentage (ESA WorldCover 10m)
- **Categorical Feature (1):**
  - `type`: Road functional hierarchy (motorway, trunk, primary, secondary, tertiary, etc., One-Hot Encoded)
- **Quarantined Columns (STRICTLY FORBIDDEN AS PREDICTORS):**
  - `risk_score`, `crash_count`, `fatal_count`, `major_count`, `minor_count`, `total_casualties`, `crash_density_per_km`, `EPDO_score`, `pca_comp_1..3`, coordinates, IDs.

## 5. Model Architecture & Preprocessing
- **Preprocessing:** `RoadSafePreprocessor` fits `StandardScaler` and `OneHotEncoder` strictly on `train.parquet`. Validation and test sets are transformed with frozen training parameters.
- **Architecture:**
  - Input Layer: 23 features
  - Hidden Layer 1: 64 Dense neurons, ReLU, Dropout (0.2)
  - Hidden Layer 2: 32 Dense neurons, ReLU, Dropout (0.2)
  - Output Layer: 1 Dense neuron, Sigmoid activation
- **Optimization:** Adam optimizer, Binary Crossentropy loss, Early Stopping on validation PR-AUC, ReduceLROnPlateau.

## 6. Threshold Tuning & Evaluation
- **Decision Threshold:** Tuned strictly on `validation.parquet` to maximize F1-score (`optimal_threshold.json`).
- **Test Set Rules:** Applied frozen threshold unchanged on `test.parquet`.
- **Metrics Reported:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.

## 7. Artifacts & Outputs
- **Model Checkpoint:** `artifacts/ann_model.keras`
- **Fitted Transformers:** `artifacts/scaler.joblib`, `artifacts/encoder.joblib`
- **Metadata:** `artifacts/feature_list.json`, `artifacts/threshold.json`, `artifacts/training_history.json`
- **Outputs:**
  - `outputs/metrics.json`
  - `outputs/predictions.csv`
  - `outputs/confusion_matrix.png`
  - `outputs/roc_pr_curves.png`
  - `outputs/training_history.png`
  - `outputs/evaluation_report.md`

## 8. Inference / Prediction Execution
To score new road segment records:
```bash
python predict.py [input_features.csv] [output_predictions.csv]
```
Or in Python:
```python
from models.ann.src.predict import RoadRiskPredictor
predictor = RoadRiskPredictor()
results = predictor.predict(df_new_segments)
```

