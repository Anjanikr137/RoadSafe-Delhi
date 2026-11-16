# RoadSafe Delhi — Random Forest Model Evaluation Report

**Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
**Assigned Team Member:** Nakshtr  
**Target Variable:** `is_crash_segment` (Binary Hazard Occurrence: 1 = Crash, 0 = Non-Crash)  
**Evaluation Partition:** Untouched Holdout Test Partition (`data_final/test.parquet`, 3,686 segments)  
**Decision Threshold:** Frozen Validation Threshold $t^* = 0.516077$  
**Status:** **INDEPENDENTLY VERIFIED & AUDITED**

---

## 1. Executive Summary & Model Objective

Random Forest serves as the **non-parametric ensemble tree baseline** in the RoadSafe Delhi multi-model benchmark. Designed to capture complex, non-linear geometric and environmental interactions across urban road corridors, the model handles class imbalance through balanced subsample bootstrap re-weighting (`class_weight='balanced_subsample'`).

### Core Model Strengths:
1. **Non-Linear Interactions:** Trees capture threshold-based boundary conditions (e.g., steep slope combined with narrow multi-lane corridor) without explicit feature interactions.
2. **Scale Invariance:** Splits depend solely on order statistics; no normalization or scaling is applied to raw inputs.
3. **Dual Explainability:** Evaluated through both tree-intrinsic Mean Decrease in Impurity (Gini MDI) and out-of-sample validation Permutation Importance.

---

## 2. Model Configuration & Hyperparameters

Selected via systematic validation grid search:
- **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
- **Number of Estimators (`n_estimators`):** `200`
- **Maximum Depth (`max_depth`):** `10`
- **Minimum Samples per Split (`min_samples_split`):** `5`
- **Minimum Samples per Leaf (`min_samples_leaf`):** `2`
- **Max Features (`max_features`):** `'sqrt'` (5 features per split)
- **Class Weight:** `'balanced_subsample'`
- **Bootstrap:** `True`
- **Random State:** `42`
- **Input Features:** 23 total features (9 numerical + 14 one-hot encoded road functional types).

---

## 3. Threshold Selection Methodology

In heavily imbalanced road safety data (Delhi crash prevalence ~3.07% to 4.27%), standard default thresholds ($t=0.50$) can skew recall. The optimal decision threshold was identified **strictly on the validation partition (`validation.parquet`, 4,229 segments)** by optimizing minority-class F1-score:

Optimal Decision Threshold: **0.516077**

At this frozen threshold:
- **Validation F1-Score:** `0.8807`
- **Validation Precision:** `0.9600`
- **Validation Recall:** `0.8136`
- **Validation PR-AUC:** `0.8910`
- **Validation ROC-AUC:** `0.9749`

---

## 4. Test Performance on Untouched Partition

Applying the frozen model and frozen validation threshold to the untouched test partition (`test.parquet`, 3,686 segments):

| Evaluation Metric | Validation Set (`validation.parquet`) | Holdout Test Set (`test.parquet`) | Assessment |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.08%** | **99.16%** | High overall classification fidelity across network |
| **Precision** | **0.9600** (96.0%) | **0.8727** (87.3%) | Low false alarm rate; 87.3% of flagged roads had crashes |
| **Recall (Sensitivity)** | **0.8136** (81.4%) | **0.8496** (85.0%) | Flags 85.0% of all confirmed historical crash segments |
| **F1-Score** | **0.8807** | **0.8610** | Harmonic balance between precision and recall |
| **ROC-AUC** | **0.9749** | **0.9780** | Exceptional overall ranking discriminability |
| **PR-AUC** | **0.8910** | **0.9162** | Strong performance in minority class detection |

### Confusion Matrix Breakdown (Test Partition, 3,686 Segments):
- **True Negatives (TN):** `3,559` (Correctly identified non-crash segments)
- **False Positives (FP):** `14` (Non-crash segments falsely flagged as hazardous)
- **False Negatives (FN):** `17` (Crash segments missed by the model)
- **True Positives (TP):** `96` (Correctly flagged historical crash segments)

$$\text{Precision} = \frac{96}{96 + 14} = 0.8727, \quad \text{Recall} = \frac{96}{96 + 17} = 0.8496$$

---

## 5. Explainability & Feature Importances

### Top Risk Associations (Gini MDI vs. Permutation Importance):
1. **`is_oneway` & Road Functional Class (`type_residential`, `type_tertiary`):** Strongest discriminators separating high-exposure primary/secondary arterials from low-speed residential accessways.
2. **`length_m` (Segment Length):** Physical corridor length exhibits significant positive importance, reflecting spatial exposure (longer road segments encompass larger exposure envelopes).
3. **`elevation_m` & `builtup_pct`:** Topographical elevation gradients and surrounding urban density provide contextual risk modulation across Delhi's diverse zones.

### Methodological Interpretability Note:
Feature importances describe **statistical correlations and split frequency in the trained ensemble**, **not causal mechanisms**. For instance, corridor length correlates with higher crash occurrence due to greater physical exposure, not because length itself causes driver errors. Unobserved factors such as traffic volume (AADT) remain confounding variables.

---

## 6. Comparison with Multi-Model Benchmarks

| Model Architecture | Target Variable | Test ROC-AUC | Test PR-AUC | Test F1-Score | Key Strength |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest (Nakshtr)** | `is_crash_segment` | **0.9780** | **0.9162** | **0.8610** | Non-parametric, robust to non-linearities, scale-invariant |
| **Artificial Neural Network (Anjani)** | `is_crash_segment` | `0.9782` | `0.8871` | `0.8767` | Deep representation learning, smooth decision boundaries |
| **Logistic Regression (Amit)** | `is_crash_segment` | `0.9577` | `0.8384` | `0.8571` | Parametric log-odds interpretability, baseline linear model |
| **Linear Regression (Anushree)** | `crash_density_per_km` | $R^2=0.3433$ | MAE=0.2158 | RMSE=1.4070 | Continuous crash frequency rate benchmark |

---

## 7. Conclusions & Deployment Readiness

The Random Forest model demonstrates outstanding stability, high minority-class detection fidelity (PR-AUC 0.9162), and balanced precision-recall (F1 = 0.8610). All artifacts are serialized, verified, and ready for integration into the RoadSafe Delhi municipal intervention prioritization tool.
