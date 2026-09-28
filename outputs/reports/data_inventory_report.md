# RoadSafe Delhi — Data Inventory & Audit Report

**Study Area:** National Capital Territory of Delhi  
**Generated On:** September 2026  

## 1. Inventory Summary

Total raw files surveyed: 16  
Total raw storage volume: 181.42 MB  

| File Path | Size (MB) | Data Classification | Decision | Operational Notes |
|:---|:---|:---|:---|:---|
| `b75c6222-4488-4d03-a20c-4f8ac1c2a1e5.csv` | 0.0 | Official Delhi Aggregate Statistics | **CONTEXT ONLY (Reality Check)** | Official aggregate government statistics on Delhi accidents |
| `c5a50842-cba9-4217-9202-bf40d9788789.csv` | 0.0 | Official Delhi Aggregate Statistics | **CONTEXT ONLY (Reality Check)** | Official aggregate government statistics on Delhi accidents |
| `d3b5df7e-b964-4d65-97c3-df4c07a03fd7.csv` | 0.0 | Official Delhi Aggregate Statistics | **CONTEXT ONLY (Reality Check)** | Official aggregate government statistics on Delhi accidents |
| `d62b502e-af7d-4bc3-8784-664e3e0b8353.csv` | 0.0 | Official Delhi Aggregate Statistics | **CONTEXT ONLY (Reality Check)** | Official aggregate government statistics on Delhi accidents |
| `ebf87168-a3d4-44a0-9041-b6a386478cb9.csv` | 0.0 | Official Delhi Aggregate Statistics | **CONTEXT ONLY (Reality Check)** | Official aggregate government statistics on Delhi accidents |
| `ESA_WorldCover_10m_2021_v200_N27E075_Map.tif` | 86.86 | Land Cover Raster (ESA 10m) | **KEEP (Primary Land Cover)** | 10m resolution land cover classification (tile N27E075) |
| `Indian-Road-Accident-Dataset-Analysis-2022-2025-Python---Jupyternotebook-main.zip` | 2.92 | Crash Records 2022-2025 (Third-Party) | **CANDIDATE (Quarantine risk_score)** | 20,000 national crash records; 2,433 in Delhi. risk_score is leakage |
| `NewDelhi.osm.pbf` | 35.43 | OSM Vector Road Network (PBF) | **KEEP (Alternative / Raw Vector)** | OpenStreetMap raw protobuf format |
| `NewDelhi.osm.shp.zip` | 53.02 | OSM Vector Road Network (Shapefile) | **KEEP (Primary Road Links)** | Complete Delhi road network (215,373 line features) |
| `POWER_Point_Hourly_20220101_20251231_028d61N_077d21E_LST.csv` | 1.31 | Hourly Weather (NASA POWER MERRA-2) | **KEEP (Validation / Reanalysis)** | Hourly reanalysis meteorological data 2022-2025 |
| `DEM\N28E076.hgt.zip` | 0.75 | Elevation / Topography (SRTM 30m HGT) | **KEEP (Primary Elevation)** | SRTM tiles covering Delhi (N28E076, N28E077) |
| `DEM\N28E077.hgt.zip` | 0.73 | Elevation / Topography (SRTM 30m HGT) | **KEEP (Primary Elevation)** | SRTM tiles covering Delhi (N28E076, N28E077) |
| `Weather Datasets\42182 (1).csv.gz` | 0.1 | Hourly Weather (Meteostat Safdarjung) | **KEEP (Primary Weather)** | Hourly weather data for NCT Delhi station 42182 |
| `Weather Datasets\42182 (2).csv.gz` | 0.1 | Hourly Weather (Meteostat Safdarjung) | **KEEP (Primary Weather)** | Hourly weather data for NCT Delhi station 42182 |
| `Weather Datasets\42182 (3).csv.gz` | 0.1 | Hourly Weather (Meteostat Safdarjung) | **KEEP (Primary Weather)** | Hourly weather data for NCT Delhi station 42182 |
| `Weather Datasets\42182.csv.gz` | 0.1 | Hourly Weather (Meteostat Safdarjung) | **KEEP (Primary Weather)** | Hourly weather data for NCT Delhi station 42182 |

## 2. Key Audit Findings & Architectural Determinations

1. **Meteostat Files are Multi-Year Records, NOT Duplicates:**
   - Analysis of `42182.csv.gz`, `42182 (1).csv.gz`, `42182 (2).csv.gz`, and `42182 (3).csv.gz` reveals that each file represents exactly one calendar year (2025, 2024, 2023, and 2022 respectively) from Delhi Safdarjung Station (WMO ID 42182).
   - They were successfully concatenated into a contiguous 4-year hourly record (35,068 rows).

2. **Weather Source Evaluation (Meteostat vs NASA POWER):**
   - **Meteostat** is ground-station observational data (Safdarjung Airport station, 28.58°N, 77.20°E). It captures localized fog, temperature dips, and precipitation events.
   - **NASA POWER** is MERRA-2 spatial reanalysis on a 0.5° × 0.625° grid. It provides complete continuous coverage with zero gaps.
   - **Decision:** Use Meteostat as primary ground-truth weather, with NASA POWER as reanalysis cross-validation.

3. **Accident Dataset Status & Leakage Warning:**
   - The crash dataset contains 2,433 Delhi records from 2022 to 2025.
   - **Critical Vulnerability Identified:** The column `risk_score` is a formulaic post-hoc combination of severity and casualties. **Using `risk_score` as a target or predictor represents severe data leakage.** It is strictly quarantined.
   - The dataset shows synthetic statistical uniformities (equal distribution across highway/rural/urban, equal weather distributions), which indicates benchmark/synthetic origin.

4. **Official UUID Datasets (Delhi Police / MoRTH aggregates):**
   - These 5 CSVs are aggregate government summary tables showing actual historical totals (~7,000 to ~8,400 annual accidents, ~2,000 fatal accidents).
   - They serve as macro reality benchmarks for validation, but cannot be joined spatially at the segment level.
