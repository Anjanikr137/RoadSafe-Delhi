# RoadSafe Delhi — Logistic Regression Model Evaluation Report

**Model:** Regularized Logistic Regression (`sklearn.linear_model.LogisticRegression`)  
**Assigned Member:** Amit  
**Target Variable:** `is_crash_segment` (Binary: 1 = crash segment, 0 = baseline)  
**Evaluation Partition:** Untouched Test Partition (`data_final/test.parquet`, 3,686 segments)  
**Execution Date:** 2026-11-15  

---

## 1. Selected Model Configuration & Justification

- **Configuration:** Balanced Class Weight with L2 Regularization (`C=1.0`)
- **Solver:** `lbfgs` with `max_iter=1000`
- **Class Imbalance Strategy:** `class_weight='balanced'` assigns inverse class frequency weighting ($w_0 = 1.0, w_1 = 22.43$) during optimization to penalize false negatives on the rare positive class (4.02% base prevalence).
- **Validation Selection Rationale:** Compared against unweighted baseline and multiple regularization parameters ($C \in [0.01, 0.1, 1.0, 10.0, 100.0]$). Balanced $C=1.0$ achieved high validation ROC-AUC (0.9553) and superior recall without excessive coefficient variance.

---

## 2. Decision Threshold Selection (Validation Only)

Because road crashes are rare spatial events (3.07% base rate), an uncalibrated default threshold (0.50) with balanced weights over-predicts positives.
- **Tuning Set:** `validation.parquet` (Geographically independent spatial holdout)
- **Selection Criterion:** Maximum F1-Score on Validation Data
- **Optimal Threshold Selected:** **0.9924**
- **Validation F1 at Optimal Threshold:** **0.8571**

---

## 3. Test Set Performance Metrics (Untouched Partition)

| Metric | Validation Set | Test Set (Final) | Benchmark Assessment |
|:---|:---|:---|:---|
| **Accuracy** | 98.91% | **99.48%** | High overall classification accuracy |
| **Precision** | 0.9517 | **0.9700** | High precision with minimal false alarms |
| **Recall (Sensitivity)** | 0.7797 | **0.8584** | Successfully identifies 97 of 113 true crash segments |
| **F1-Score** | 0.8571 | **0.9108** | Balanced harmonic mean of precision and recall |
| **ROC-AUC** | 0.9553 | **0.9761** | Excellent global ranking across all decision thresholds |
| **PR-AUC (Average Precision)** | 0.7704 | **0.8894** | Substantially exceeds no-skill baseline (0.0307) |

### Test Confusion Matrix (N = 3,686 segments):
- **True Negatives (TN):** 3,570 segments correctly classified as baseline
- **False Positives (FP):** 3 segments falsely flagged
- **False Negatives (FN):** 16 crash corridors missed
- **True Positives (TP):** 97 crash corridors correctly identified

---

## 4. Model Comparison: ANN vs. Logistic Regression

Both models were evaluated on **exactly the same untouched spatial test partition** (`data_final/test.parquet`, 3,686 segments) with the approved feature contract (23 input features) and identical spatial split boundaries.

| Metric | Artificial Neural Network (ANN) | Logistic Regression | Comparison Insight |
|:---|:---|:---|:---|
| **Accuracy** | 99.48% | 99.48% | Both achieve $\ge 99.4\%$ accuracy on imbalanced network |
| **Precision** | 1.0000 | 0.9700 | ANN achieves 100% precision; Logistic Regression achieves 0.9700 |
| **Recall** | 0.8319 | 0.8584 | Logistic Regression achieves higher sensitivity (0.8584 vs 0.8319) |
| **F1-Score** | 0.9082 | 0.9108 | Both models achieve $\ge 0.90$ F1-score; highly competitive |
| **ROC-AUC** | 0.9785 | 0.9761 | ANN slightly leads (0.9785 vs 0.9761) due to non-linear layers |
| **PR-AUC** | 0.9289 | 0.8894 | ANN leads (0.9289 vs 0.8894) in high-precision recall zone |
| **Training Time** | 70.47s | 0.1760s | Logistic Regression trains $\sim 700\times$ faster |

---

## 5. Domain Interpretability: Odds Ratios

Logistic Regression provides explicit mathematical interpretability via exponentiated coefficients:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^p \beta_j X_j$$

### Top Risk-Increasing Factors (Odds Ratio > 1.0):
```
           feature  coefficient  odds_ratio
  type_residential       6.2396    512.6435
      type_service       4.9011    134.4327
          length_m       3.1103     22.4276
 type_unclassified       2.9876     19.8386
type_living_street       2.7630     15.8468
```

### Top Risk-Decreasing / Baseline Factors:
```
            feature  coefficient  odds_ratio
    type_trunk_link      -2.1049      0.1219
 type_motorway_link      -2.7289      0.0653
 type_tertiary_link      -2.9087      0.0545
  type_primary_link      -3.3949      0.0335
type_secondary_link      -3.4540      0.0316
```

---

## 6. Key Methodological Observations

1. **Zero Data Leakage:** Preprocessing was strictly fitted on `train.parquet` only. All outcome-derived columns and spatial block IDs were programmatically quarantined.
2. **True Spatial Generalization:** Evaluated on holdout spatial blocks (`Central_South` and `East_North`), confirming true geographic generalization.
3. **Complementary Utility:** While the ANN offers higher PR-AUC, Logistic Regression offers ultra-fast training (0.1760s) and closed-form odds-ratio interpretability for municipal policy makers.
