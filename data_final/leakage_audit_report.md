# RoadSafe Delhi — Target Leakage Audit Report

**Audit Status:** PASSED (Zero Leakage Detected)  
**Execution Timestamp:** 2026-11-13  
**Common Dataset Path:** `data_final/delhi_road_risk_common_dataset.parquet`  

---

## 1. Audit Scope & Quarantine Verification

All candidate columns were screened for potential reverse-causality, future-information, or target-derivative leakage.

| Column Name | Category | Audit Result | Action Taken |
|:---|:---|:---|:---|
| `risk_score` | Third-Party Synthetic Label | **FATAL LEAKAGE** | Quarantined and permanently removed from raw crash records. |
| `crash_count` | Crash Aggregation | **DIRECT TARGET** | Isolated to Target domain; strictly forbidden as model predictor. |
| `fatal_count` | Severity Aggregation | **DIRECT TARGET** | Isolated to Target domain; strictly forbidden as model predictor. |
| `major_count` | Severity Aggregation | **DIRECT TARGET** | Isolated to Target domain; strictly forbidden as model predictor. |
| `minor_count` | Severity Aggregation | **DIRECT TARGET** | Isolated to Target domain; strictly forbidden as model predictor. |
| `total_casualties` | Severity Aggregation | **DIRECT TARGET** | Isolated to Target domain; strictly forbidden as model predictor. |
| `crash_density_per_km` | Normalized Outcome | **REGRESSION TARGET** | Dedicated target for Linear Regression; excluded from features. |
| `EPDO_score` | Severity Index | **REGRESSION TARGET** | Dedicated target for severity modeling; excluded from features. |
| `is_crash_segment` | Binary Classification | **PRIMARY TARGET** | Target variable for ANN, RF, SVM, Logistic Regression. |
| `is_high_risk` | Priority Classification | **ALTERNATIVE TARGET** | Priority intervention classification target. |
| `night_crash_ratio` | Crash-derived ratio | **LEAKAGE CONCERN** | Dropped because undefined when `crash_count == 0`. |
| `fog_crash_ratio` | Crash-derived ratio | **LEAKAGE CONCERN** | Dropped because undefined when `crash_count == 0`. |
| `pca_comp_1..3` | Rotated Eigenvectors | **LATENT ARTIFACT** | Excluded from primary feature contract to maintain physical explainability. |

---

## 2. Programmatic Verification Checks

```python
# Programmatic Leakage Assertion (enforced across all models)
contract = json.load(open('data_final/feature_contract.json'))
approved = set(contract['approved_predictors'])
targets = set(contract['target_definitions'].keys())
forbidden = set(contract['quarantined_leakage_columns']).union(targets)

assert len(approved.intersection(forbidden)) == 0, "FATAL: Predictor set contains target or quarantined features!"
```

---

## 3. Spatial Separation & Leakage Isolation

- **Spatial Partitioning:** 3x3 geographic grid blocks.
- **Split Breakdown:**
  - `train`: 11502 segments (59.2%)
  - `val`: 4229 segments (21.8%)
  - `test`: 3686 segments (19.0%)
- **Spatial Separation:** Train, validation, and test blocks share no overlapping geographic grids, preventing spatial autocorrelation leakage (Tobler's First Law).
