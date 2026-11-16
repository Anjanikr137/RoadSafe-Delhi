# RoadSafe Delhi — Linear Regression (Ridge Baseline) Evaluation Report

**Model:** Ridge Linear Regression (`sklearn.linear_model.Ridge`)  
**Assigned Member:** Anushree  
**Target Variable:** `crash_density_per_km` (Continuous Crash Rate per Kilometer)  
**Evaluation Partition:** Untouched Test Partition (`data_final/test.parquet`, 3,686 segments)  
**Status:** **INDEPENDENTLY VERIFIED & AUDITED**

---

## 1. Model Objective & Target Formulation

Linear Regression serves as the **parametric continuous rate baseline** in the RoadSafe Delhi multi-model benchmark. While classification models (ANN, Logistic Regression, Random Forest, SVM) predict the binary hazard occurrence (`is_crash_segment`), Linear Regression models the continuous intensity of crash frequency normalized by segment length:

$$\text{crash\_density\_per\_km} = \frac{\text{crash\_count}}{\text{length\_km}}$$

### Continuous Target Distribution across Official Spatial Partitions:
- **Training Set (`train.parquet`):** Mean = 0.2238 crashes/km (11,502 segments)
- **Validation Set (`validation.parquet`):** Mean = 0.2353 crashes/km (4,229 segments)
- **Test Set (`test.parquet`):** Mean = 0.1834 crashes/km (3,686 segments)

---

## 2. Approved Features & Zero-Leakage Preprocessing

### Approved Model Inputs (23 Total Encoded Features):
- **Numerical Predictors (9):** `length_m`, `est_lanes`, `is_oneway`, `is_bridge`, `has_maxspeed`, `dist_to_center_km`, `elevation_m`, `slope_deg`, `builtup_pct`.
- **Categorical Predictor (1):** `type` (OSM functional road class, one-hot encoded into 14 categories).
- **Leakage Prevention:** `StandardScaler` and `OneHotEncoder` were fitted **strictly on `train.parquet`**. All casualty totals (`fatal_count`, `major_count`, etc.), risk scores, spatial block IDs, and coordinates were programmatically quarantined.

---

## 3. Model Configuration & Regularization

To stabilize Ordinary Least Squares (OLS) estimation against multi-collinearity among road hierarchy indicators and spatial features, L2 regularization (Ridge) was implemented:
- **Algorithm:** `sklearn.linear_model.Ridge`
- **Regularization Strength ($\alpha$):** `10.0`
- **Random Seed:** `42`
- **Solver:** `auto` (Cholesky/L-BFGS)

---

## 4. Verified Model Performance Metrics

Both validation and holdout test partitions were evaluated using the verified saved model artifacts:

| Evaluation Metric | Validation Holdout (`validation.parquet`) | Untouched Test Partition (`test.parquet`) | Benchmark Assessment |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **0.2448 crashes/km** | **0.215807 crashes/km** | Low average absolute deviation on unseen road segments |
| **Root Mean Squared Error (RMSE)** | **1.3644 crashes/km** | **1.407025 crashes/km** | Captures variance with penalty on large residual outliers |
| **Coefficient of Determination ($R^2$)** | **0.4251** | **0.343341** | Explains 34.33% of spatial crash density variance on holdout |

*Verification Note: 100% exact numerical match confirmed against independent reload of `linear_regression_model.joblib`.*

---

## 5. Domain Interpretability & Feature Associations

Examining the standardized Ridge regression coefficients:
- **Positive Linear Associations:** Wide multi-lane primary corridors (`type_primary`, `est_lanes`), segment length, and high surrounding urban built-up density (`builtup_pct`) correlate with higher continuous crash density per kilometer.
- **Negative Linear Associations:** Residential access roads (`type_residential`), link roads, and peripheral corridors (`dist_to_center_km`) exhibit lower continuous crash rates per kilometer.
- **Methodological Note:** In cross-sectional observational network data, these parameters represent **empirical statistical associations**, not direct causal intervention effects. Unmetered exposure (e.g. traffic volume / AADT) remains an unobserved confounding factor.

---

## 6. Limitations of Linear Continuous Modeling

1. **Extreme Zero-Inflation:** Because 95.98% of road segments have zero recorded crashes, continuous OLS/Ridge models face heavy clustering at zero, leading to right-skewed residual distributions.
2. **Linear Additivity Assumption:** Linear models cannot inherently capture complex nonlinear topographic and environmental interactions without explicit polynomial feature expansion.
3. **Complementary Role:** The Ridge baseline provides an essential linear benchmark that justifies the deployment of nonlinear deep learning (ANN) and ensemble models (Random Forest).
