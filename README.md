# RoadSafe Delhi
## Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning

---

## Project Overview

**Study Area:** Delhi / NCT of Delhi  
**Objective:** Estimate relative road-risk, identify high-risk road locations/segments, understand the factors associated with road risk, and support prioritization of locations for preventive road-safety assessment.

**Research Question:**  
> *"Can machine learning integrate historical road-crash patterns with road-network, spatial, temporal and environmental characteristics to identify high-risk road locations in Delhi and support prioritization for preventive road-safety assessment?"*

---

## Directory Structure

```
RoadSafe Delhi/
├── Datasets/               ← RAW DATA — NEVER MODIFY
├── notebooks/              ← All Jupyter notebooks (analysis lives here)
├── data_processed/         ← Cleaned/processed datasets (copies of raw)
├── data_engineered/        ← Feature-engineered datasets
├── outputs/
│   ├── eda/               ← EDA plots
│   ├── pca/               ← PCA outputs
│   ├── maps/              ← Spatial maps
│   ├── tables/            ← Important tables
│   └── reports/           ← Markdown/text reports
└── README.md
```

---

## Notebook Workflow

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_Dataset_Audit.ipynb` | Complete inventory of all datasets |
| 02 | `02_Accident_Data_EDA.ipynb` | Crash data EDA and quality checks |
| 03 | `03_Weather_EDA.ipynb` | Meteostat + NASA POWER comparison |
| 04 | `04_OSM_Road_Network_EDA.ipynb` | Road network analysis |
| 05 | `05_LandCover_DEM_Analysis.ipynb` | WorldCover raster + SRTM DEM |
| 06 | `06_Spatial_Crash_Road_Matching.ipynb` | Crash-to-segment matching |
| 07 | `07_Feature_Engineering.ipynb` | Road-segment feature construction |
| 08 | `08_Target_Construction.ipynb` | Candidate target variable design |
| 09 | `09_Multicollinearity_Analysis.ipynb` | Correlation + VIF analysis |
| 10 | `10_PCA_Analysis.ipynb` | PCA dimensionality analysis |
| 11 | `11_Final_Dataset_Preparation.ipynb` | Final ML dataset schema |

---

## Data Sources

| Dataset | Type | Status |
|---------|------|--------|
| Meteostat 42182 (×4 gz files) | Hourly weather 2022–2025 | KEEP |
| NASA POWER Hourly 2022–2025 | Reanalysis weather | KEEP (validation) |
| NewDelhi.osm.pbf | OSM road network (PBF) | KEEP |
| NewDelhi.osm.shp.zip | OSM road network (SHP) | KEEP (primary) |
| ESA WorldCover 10m 2021 | Land cover raster | KEEP |
| DEM N28E076.hgt.zip | SRTM elevation tile | KEEP |
| DEM N28E077.hgt.zip | SRTM elevation tile | KEEP |
| Indian Road Accident Dataset (zip) | Third-party crash data 2022–2025 | CANDIDATE |
| UUID CSVs (×5) | Official Delhi aggregate statistics | CONTEXT ONLY |

---

## Key Rules

- **NEVER modify raw datasets** in `Datasets/`
- **DO NOT use `risk_score`** as a target or feature
- **Do NOT train final ML models** until all pre-processing phases are reviewed
- The Indian accident dataset is **third-party data** — NOT official Delhi Police records
- UUID CSVs are **aggregate statistics** — NOT crash-level records

---

## Status

*See individual notebooks for detailed phase status.*
