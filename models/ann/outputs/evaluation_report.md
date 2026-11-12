# RoadSafe Delhi — Artificial Neural Network (ANN) Evaluation Report

**Model Type:** Feedforward Neural Network (Multi-Layer Perceptron)  
**Lead Engineer:** Anjani (Team Leader)  
**Target Variable:** `is_crash_segment`  
**Evaluation Partition:** Untouched Test Partition (`data_final/test.parquet`, 3,686 segments)  
**Execution Date:** 2026-11-13  

---

## 1. Network Architecture & Hyperparameters

- **Input Dimension:** 23 features (approved environmental + infrastructure variables)
- **Hidden Layer 1:** 64 units, ReLU activation, Batch Dropout (p=0.20)
- **Hidden Layer 2:** 32 units, ReLU activation, Batch Dropout (p=0.20)
- **Output Layer:** 1 unit, Sigmoid activation (calibrated crash probability)
- **Optimizer:** Adam (Initial lr=0.001, ReduceLROnPlateau factor=0.5)
- **Loss Function:** Binary Crossentropy with training-class balancing weights
- **Regularization:** Dropout + Validation Early Stopping (patience=10 on val_pr_auc)

---

## 2. Decision Threshold Tuning (Validation Only)

Because road crashes are rare spatial events (3.07% test base rate), an uncalibrated 0.50 threshold fails to optimize detection for public safety prioritization.
- **Tuning Set:** `validation.parquet` (Geographically independent spatial holdout)
- **Optimization Metric:** Maximum F1-Score on Validation Data
- **Selected Threshold:** **0.9994**
- **Validation F1 at Threshold:** **0.8833**

---

## 3. Test Set Performance Metrics (Untouched Partition)

| Metric | Validation Set | Test Set (Final) | Benchmark Assessment |
|:---|:---|:---|:---|
| **Accuracy** | 99.13% | **99.48%** | High overall classification accuracy |
| **Precision** | 1.0000 | **1.0000** | Proportion of predicted hotspots that are actual crash sites |
| **Recall (Sensitivity)** | 0.7910 | **0.8319** | Proportion of actual crash sites successfully flagged |
| **F1-Score** | 0.8833 | **0.9082** | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.9677 | **0.9785** | Outstanding global ranking ability across all operating points |
| **PR-AUC (Average Precision)** | 0.8826 | **0.9289** | Far exceeds no-skill baseline (0.0307) |

### Confusion Matrix Breakdown (Test Partition, N=3,686):
- **True Negatives (TN):** 3,573 segments correctly classified as baseline
- **False Positives (FP):** 0 segments flagged as hazard corridors
- **False Negatives (FN):** 19 crash segments missed
- **True Positives (TP):** 94 crash segments correctly identified

---

## 4. Key Engineering & Methodological Observations

1. **Zero Data Leakage:** Preprocessing (scaling and categorical encoding) was strictly fitted on `train.parquet` only. No test or validation information leaked into parameters.
2. **Geographical Generalization:** The model was tested on geographically disjoint spatial blocks (`Central_South` and `East_North`), confirming true spatial generalization rather than spatial proximity memorization.
3. **Physical Feature Grounding:** Predictions are driven by real physical measurements (NASA SRTM DEM elevation, slope, ESA WorldCover built-up urban density, road carriage lanes) with VIF < 1.46.
