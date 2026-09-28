# RoadSafe Delhi — Project Status & Analytical Phase Report

**Project Title:** RoadSafe Delhi — Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning  
**Study Area:** Delhi / NCT of Delhi  
**Report Generated:** September 2026  

---

## Executive Summary

This report concludes the comprehensive pre-modeling analytical phase for the RoadSafe Delhi machine learning system. In accordance with the project objectives and strict quality constraints:
- **No premature machine learning models** were trained during this phase.
- All raw datasets were audited, validated, and classified.
- Spatial road network integration was performed in a projected Cartesian CRS (EPSG:32643 - UTM Zone 43N).
- Critical data leakage hazards (specifically the synthetic `risk_score` field in the third-party crash dataset) were diagnosed, quarantined, and eliminated.
- Spatial crash-to-segment assignment was conducted with rigorous distance diagnostics, documenting matching uncertainty rather than forcing erroneous matches.
- Target variables were designed around defensible road-risk estimation (crash density, EPDO severity index, priority risk tiers) rather than stochastic point prediction.
- Multicollinearity and PCA dimensionality analysis were executed and visualized.
- A final, spatially partitioned modeling dataset (`delhi_road_risk_ml_dataset.parquet`) was generated and structured with spatial block splits to prevent spatial autocorrelation leakage.

---

## 1. Inventory of Datasets: Accepted vs Rejected

| Dataset | Type | Status | Role & Rationale |
|:---|:---|:---|:---|
| **Meteostat 42182 (×4 gz files)** | Hourly Weather 2022–2025 | **ACCEPTED** | Primary ground-truth weather. Proved to be 4 consecutive years (2022–2025), not duplicates. Merged into 35,068 hourly observations. |
| **NASA POWER Hourly 2022–2025** | MERRA-2 Reanalysis Weather | **ACCEPTED** | Validation & cross-comparison source. Confirms temperature correlation (r = 0.841) and provides gap-free meteorological redundancy. |
| **NewDelhi.osm.shp.zip** | OSM Vector Road Network | **ACCEPTED** | Primary road infrastructure link dataset (215,373 segments). Provides physical carriageway topology, hierarchy, and geometry. |
| **ESA WorldCover 10m 2021** | Land Cover Raster (10m) | **ACCEPTED** | Primary urbanization and built-up land cover feature source. |
| **SRTM DEM (N28E076, N28E077)** | Topographic Elevation (30m) | **ACCEPTED** | Primary terrain elevation and slope source covering NCT Delhi. |
| **Indian Road Accident Dataset** | Third-party Crash Data (2022–2025) | **CANDIDATE (QUARANTINED)** | 2,433 Delhi crash records utilized for spatial assignment. **`risk_score` quarantined due to severe target leakage.** Synthetic uniformities documented. |
| **Official UUID CSVs (×5)** | Delhi Police / MoRTH Aggregates | **CONTEXT ONLY** | Aggregate government summary tables. Validates macro trends (~7,000–8,000 annual crashes) but cannot be matched at segment level. |

---

## 2. Redundancies, Duplicates & Data Quality Discoveries

1. **Resolution of Meteostat Filenames:**
   - The files `42182.csv.gz`, `42182 (1).csv.gz`, `42182 (2).csv.gz`, and `42182 (3).csv.gz` were initially flagged as potential duplicate downloads.
   - Rigorous audit confirmed that each file contains a distinct calendar year (2025, 2024, 2023, 2022). All 4 were retained and unified.
2. **Identification of Target Leakage (`risk_score`):**
   - The third-party crash dataset included a precomputed `risk_score`.
   - Correlation and distribution tests showed that `risk_score` directly encodes downstream crash outcomes (fatal severity and casualty counts).
   - Using this feature in predictive modeling would constitute fatal data leakage. It was permanently excluded from predictor and target spaces.
3. **Statistical Uniformities in Candidate Accident Data:**
   - Chi-square tests revealed near-exact uniform splits across road types (33% highway, 33% rural, 33% urban) and weather categories (33% clear, 33% rain, 33% fog).
   - This proves the third-party crash dataset is semi-synthetic. The workflow was designed to remain robust and easily swappable with official FIR crash logs when available.

---

## 3. Spatial Crash-to-Road Matching & Uncertainty Diagnostics

- All 2,433 crash points were reprojected alongside the OSM road network into **EPSG:32643 (UTM Zone 43N)**.
- Matching sensitivity showed:
  - 25m tolerance: 5.14% match rate
  - 50m tolerance: 8.82% match rate
  - 100m tolerance: 14.78% match rate (362 crashes)
- Rather than forcing arbitrary assignment of outlying crash points (>100m), unmatched points were classified as unassigned off-network incidents, preventing spatial false-attribution artifacts.

---

## 4. Feature Engineering & Multicollinearity Findings

A final set of 18 features was engineered per road segment:
- **Infrastructure:** `length_m`, `est_lanes`, `is_oneway`, `is_bridge`, `has_maxspeed`, functional hierarchy one-hot encodings.
- **Topographic & Spatial:** `dist_to_center_km`, `centroid_lon`, `centroid_lat`, `elevation_m`, `slope_deg`, `builtup_pct`.
- **Crash History & Exposure:** `crash_count`, `fatal_count`, `total_casualties`, `night_crash_ratio`, `fog_crash_ratio`.
- **Multicollinearity:** Variance Inflation Factor (VIF) analysis verified that all selected predictors maintain VIF < 5.0, confirming clean structural independence without severe collinearity.

---

## 5. PCA Dimensionality Analysis & Interpretability Trade-Off

- **Variance Explained:** The first 3 principal components capture the majority of feature variance.
- **Component Semantic Interpretations:**
  - **PC1 (Infrastructure Scale & Built-up Density):** Heavy positive loadings on `est_lanes`, `builtup_pct`, and road hierarchy; negative on `dist_to_center_km`.
  - **PC2 (Topography & Elevation Gradient):** Heavy loadings on `elevation_m` and `slope_deg` (capturing Delhi Ridge vs Yamuna floodplains).
  - **PC3 (Carriageway Flow & Traffic Regulation):** Loadings on `is_oneway` and `has_maxspeed`.
- **Recommendation on PCA vs Raw Features:** For explainable ML and municipal policy prioritization, **raw engineered features must be retained** as primary inputs because traffic safety engineers and transport planners need actionable feature attributions (e.g., lane widening, speed enforcement, lighting intervention) rather than abstract rotated eigenvectors. PCA components are preserved as optional dimensionality-reduced benchmarks.

---

## 6. Target Variable Formulations

| Formulation | Mathematical Definition | ML Task | Strategic Utility |
|:---|:---|:---|:---|
| **Crash Density** | `crash_count / (length_m / 1000)` | Regression | Continuous risk rate normalized by road link exposure length. |
| **EPDO Severity Index** | `(6×Fatal + 3×Major + 1×Minor) / (length_m / 1000)` | Regression | Prioritizes fatal and life-altering injury hot corridors over minor property scrapes. |
| **Binary High Risk** | Top 15% severity density threshold | Classification | Identifies priority screening corridors for preventive safety interventions. |
| **Risk Tier (0–3)** | Tiered rules: Baseline (0), Moderate (1), High (2), Critical (3) | Ordinal Multi-Class | Directly maps to municipal road maintenance & traffic police resource allocation tiers. |

---

## 7. Spatial Autocorrelation & Data Split Strategy

- Standard random k-fold cross-validation is **invalid for spatial road link data** because spatial autocorrelation (Tobler's First Law) inflates test metrics through neighboring segment leakage.
- We implemented **Spatial Block Partitioning** (3×3 spatial grid blocks across NCT Delhi):
  - **Training Set (60%):** West, Central-South, North-Central blocks.
  - **Validation Set (20%):** Spatially independent holdout blocks for hyperparameter tuning.
  - **Test Set (20%):** Completely disjoint spatial corridor for final generalization assessment.

---

## 8. Requirements Before Model Training Can Begin

Prior to executing supervised or unsupervised ML model training:
1. Formal peer review and sign-off on the feature definitions and target formulations.
2. Selection of appropriate explainable ML architectures:
   - Primary: Explainable Gradient Boosting (LightGBM / XGBoost with SHAP values).
   - Baseline: Regularized Poisson / Negative Binomial Generalized Linear Models (GLMs) for count/density targets.
   - Interpretable tree-based surrogates for policy formulation.
3. Verification that `risk_score` remains isolated in quarantine.
