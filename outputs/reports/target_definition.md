# RoadSafe Delhi — Target Variable Definitions & Formulation Rationale

**Document Version:** 2.0  
**Dataset Reference:** `data_final/delhi_road_risk_common_dataset.parquet`  
**Total Road Network Segments:** 19,417  

---

## 1. Problem Statement: Extreme Zero-Inflation in Road Safety

In municipal road networks, traffic collisions are rare spatial events:
- Total analyzed vehicular road segments: **19,417**
- Segments with $\ge 1$ matched crash: **781 (4.02%)**
- Segments with 0 crashes: **18,636 (95.98%)**

In the previous project implementation, `is_high_risk` was defined as `(crash_density_per_km >= 85th_percentile)`. Because 96% of segments had zero crashes, the 85th percentile of crash density was identically 0.0, erroneously labeling any road with a single minor incident as "high risk".

---

## 2. Statistically Justified Target Definitions

### A. Binary Classification Target: `is_crash_segment` (Primary Target for ANN, RF, SVM, Logistic Regression)
- **Definition:** `is_crash_segment = 1` if `crash_count > 0` else `0`.
- **Prevalence:** 4.02% positive class (781 positive segments).
- **Domain Utility:** Models the baseline road hazard probability based purely on physical geometry, infrastructure, and environmental factors.

### B. High Priority Intervention Target: `is_high_risk` (Alternative Binary Target)
- **Definition:** `is_high_risk = 1` if segment has $\ge 2$ recurrent crashes, OR $\ge 1$ fatal crash, OR $\ge 1$ major injury crash, OR `EPDO_score >= 5.0`.
- **Prevalence:** 2.71% positive class (527 segments).
- **Domain Utility:** Filters out isolated single minor scrapes to identify corridors demanding urgent capital intervention (blackspots).

### C. Continuous Exposure Target: `crash_density_per_km` (Target for Linear Regression Baseline — Anushree)
- **Mathematical Formula:** 
  $$	ext{crash\_density\_per\_km} = \frac{\text{crash\_count}}{\text{length\_m} / 1000.0}$$
- **Distribution:** Mean = 0.2186, Max = 97.11 crashes/km.
- **Domain Utility:** Normalizes crash frequency by segment exposure length, avoiding segment length bias in regression.

### D. Severity-Weighted Target: `EPDO_score` (Equivalent Property Damage Only Index)
- **Mathematical Formula:**
  $$\text{EPDO\_score} = \frac{6 \times \text{fatal} + 3 \times \text{major} + 1 \times \text{minor}}{\text{length\_m} / 1000.0}$$
- **Distribution:** Mean = 0.5026, Max = 231.06 EPDO/km.

### E. Ordinal Multi-Class Target: `risk_tier`
- **Class 0 (Baseline):** 0 crashes (Segment count: 18,636)
- **Class 1 (Low Risk):** 1 minor crash, EPDO < 5 (Segment count: 254)
- **Class 2 (Moderate Risk):** Major injury or EPDO 5-10 (Segment count: 189)
- **Class 3 (Critical Risk):** Fatal crash, $\ge 2$ crashes, or EPDO $\ge 10$ (Segment count: 338)

---

## 3. Methodological Rule on Class Imbalance

**STRICT DIRECTIVE:** The test dataset must **NEVER** be artificially balanced (via SMOTE, random oversampling, or undersampling). Artificial balancing on test data distorts real-world prior probability distributions and generates illusory performance metrics. Rebalancing techniques (such as focal loss, class weights, or threshold adjustment) must only be applied to the training set.
