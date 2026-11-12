# RoadSafe Delhi — Data Quality & Integrity Report

**Dataset:** `data_final/delhi_road_risk_common_dataset.parquet`  
**Total Records:** 19,417  
**Total Columns:** 25  

---

## 1. Missing Value Statistics

No missing values exist across any approved predictor or target variable:
```
length_m                0
type                    0
est_lanes               0
is_oneway               0
is_bridge               0
has_maxspeed            0
dist_to_center_km       0
elevation_m             0
slope_deg               0
builtup_pct             0
is_crash_segment        0
is_high_risk            0
risk_tier               0
crash_density_per_km    0
EPDO_score              0
```

---

## 2. Feature Extent & Physical Ranges

| Feature | Min | Median | Mean | Max | Data Source |
|:---|:---|:---|:---|:---|:---|
| `length_m` | 0.6 | 123.7 | 490.1 | 205397.6 | OpenStreetMap Vector |
| `elevation_m` | 194.0 | 214.0 | 216.0 | 303.0 | NASA SRTM 30m DEM (HGT) |
| `slope_deg` | 0.00 | 1.28 | 1.70 | 26.36 | SRTM DEM Gradient |
| `builtup_pct` | -0.0 | 76.4 | 70.2 | 100.0 | ESA WorldCover 10m (Class 50) |
| `dist_to_center_km` | 0.01 | 12.48 | 13.40 | 35.43 | Geodesic to Connaught Place |
| `est_lanes` | 1 | 2.0 | 2.24 | 6 | OSM Hierarchy Mapping |

---

## 3. Crash Matching Performance

- **Candidate Crash Records:** 2,433
- **Matches within 25m:** 567 (23.17%)
- **Matches within 50m:** 697 (28.48%)
- **Matches within 100m:** 829 (33.88%)
- **Matches within OSM Bounding Box (<=100m):** 829 / 1,271 (65.23%)
- **Unmatched Preservation:** All 18,636 non-crash segments and all unmatched crash points are preserved in `data_processed/delhi_crashes_unmatched.csv`.
