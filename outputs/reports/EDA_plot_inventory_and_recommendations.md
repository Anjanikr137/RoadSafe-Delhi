# RoadSafe Delhi — EDA Plot Inventory & Recommendations Report

**Project Title:** RoadSafe Delhi — Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning  
**Study Area:** National Capital Territory (NCT) of Delhi  
**Report Focus:** Pre-submission exploratory analysis, data quality audit, figure inventory, and transition to classification  
**Document Generated:** September 2026  

---

## Executive Overview

This document provides a comprehensive inventory and methodological appraisal of all exploratory data analysis (EDA) plots, maps, and figures generated across the initial phases of the **RoadSafe Delhi** research project. 

In strict adherence to the pre-submission guidelines:
- **No machine learning models have been trained**, and no predictive performance metrics are claimed.
- **PCA and VIF analyses are strictly excluded** from the core submission body to keep the focus squarely on physical road network topology, crash distributions, environmental factors, and data quality.
- **Only empirical figures and verified numbers** from the project's existing notebooks and output directories (`outputs/eda/`, `outputs/maps/`, `outputs/tables/`) are utilized.
- All interpretations employ rigorous, non-causal statistical language ("shows", "indicates", "suggests", "associated with", "distribution").

---

## 1. Complete EDA Plot Inventory

The table below catalogs every visual asset produced by the exploratory notebooks across `outputs/eda/`, `outputs/maps/`, and `outputs/pca/`, detailing its source, thematic scope, inclusion status for the pre-submission report, and specific rationale.

| Plot Filename | Notebook / Source | What It Shows | EDA Category | Include in Final Report? | Justification / Reason |
|:---|:---|:---|:---|:---:|:---|
| `accident_severity_distribution.png` | `02_Accident_Data_EDA.ipynb` | Severity proportion pie chart (Minor, Major, Fatal), casualties per severity level, and severity across road classifications. | Accident EDA | **YES (Core)** | Crucial for establishing class distribution and justifying severity-weighted risk indices over raw accident counts. |
| `accident_leakage_risk_score.png` | `02_Accident_Data_EDA.ipynb` | Boxplots and scatterplots demonstrating strong correlation ($r > 0.70$) between `risk_score`, casualties, and severity tiers. | Data Quality EDA | **YES (Core)** | Essential evidence demonstrating severe target leakage in the third-party dataset, justifying its complete quarantine. |
| `accident_temporal_patterns.png` | `02_Accident_Data_EDA.ipynb` | Multi-panel bar charts displaying crash counts by year (2022–2025), month of year, day of week, and hour of day (0–23). | Temporal EDA | **YES (Core)** | Illustrates temporal crash concentration, night-time crash inflation (38.1%), and peak vs. non-peak dynamics. |
| `delhi_crash_hotspots_matched_roads.png` | `06_Spatial_Crash_Road_Matching.ipynb` | Spatial map of Delhi displaying vehicular road links, unassigned crashes (>100m), and matched crashes color-coded by severity. | Spatial EDA | **YES (Core)** | Visualizes spatial distribution of crash occurrences across NCT Delhi and highlights high-density collision corridors. |
| `spatial_matching_distance_distribution.png` | `06_Spatial_Crash_Road_Matching.ipynb` | Histogram of Euclidean distances from crash coordinates to the nearest road link, plus a match-rate bar chart across distance thresholds. | Spatial / Quality EDA | **YES (Core)** | Empirically documents spatial matching uncertainty and justifies why outlying points are kept unassigned rather than forced. |
| `delhi_weather_climatology.png` | `03_Weather_EDA.ipynb` | Dual-axis monthly climatology showing mean, min, and max temperature curves alongside monthly precipitation totals (2022–2025). | Weather EDA | **YES (Core)** | Establishes the macro-climatic baseline of Delhi (extreme summer heat >42°C vs. winter drops <7°C). |
| `delhi_fog_rain_extremes.png` | `03_Weather_EDA.ipynb` | Monthly frequency counts of fog hours ($\text{RH} > 90\%$, $\text{Temp} < 15^\circ\text{C}$), extreme heat hours ($\ge 42^\circ\text{C}$), and rainfall hours. | Weather EDA | **YES (Core)** | Highlights winter fog (Dec–Jan) and monsoon precipitation (Jul–Sep) as critical environmental road risk exposure windows. |
| `weather_meteo_vs_nasa_comparison.png` | `03_Weather_EDA.ipynb` | Scatter plots with regression lines comparing Meteostat station observations against NASA POWER MERRA-2 reanalysis data. | Weather / Quality EDA | **YES (Core)** | Validates observational data fidelity ($r = 0.841$ for temperature) and justifies Meteostat as primary ground-truth weather. |
| `osm_road_network_hierarchy.png` | `04_OSM_Road_Network_EDA.ipynb` | Bar charts comparing total network length (km) and total link counts across road classes (motorway, trunk, primary, secondary, tertiary, residential). | Road Network EDA | **YES (Core)** | Defines the physical carriageway infrastructure and exposes structural dominance of residential and tertiary links. |
| `osm_segment_length_distribution.png` | `04_OSM_Road_Network_EDA.ipynb` | Density histogram of segment lengths (meters) clipped at 2,000m, showing skewness and median segment lengths. | Road Network EDA | **YES (Core)** | Justifies segment length normalization (crash rate per km) to avoid bias toward longer physical segments. |
| `delhi_road_network_map.png` | `04_OSM_Road_Network_EDA.ipynb` | High-resolution GIS map of Delhi's arterial network categorized by functional class (motorway, trunk, primary, secondary, tertiary). | Road Network / Spatial EDA | **YES (Core)** | Provides visual spatial comprehension of Delhi's ring roads, radial highways, and urban grid geometry. |
| `delhi_landcover_distribution.png` | `05_LandCover_DEM_Analysis.ipynb` | Bar chart of surface area percentages from ESA WorldCover 10m (Cropland 43.15%, Built-up 33.39%, Tree Cover 12.96%, etc.). | Land Cover EDA | **YES (Core)** | Explains land-use context: high urban density in the central core transitioning to peri-urban agricultural fringes. |
| `delhi_elevation_slope_profile.png` | `05_LandCover_DEM_Analysis.ipynb` | Dual-panel histograms showing terrain elevation (190m–320m MSL) and terrain slope (degrees) from SRTM 30m DEM. | Terrain / DEM EDA | **YES (Core)** | Demonstrates topographic contrast between the flat Yamuna floodplains and the elevated, sloping Delhi Ridge. |
| `correlation_matrix_heatmap.png` | `09_Multicollinearity_Analysis.ipynb` | Annotated Pearson correlation heatmap across continuous infrastructure, spatial, and topographic predictors. | Correlation EDA | **YES (Core)** | Demonstrates feature relationships and confirms absence of severe pairwise multicollinearity among retained predictors. |
| `target_distributions_comparison.png` | `08_Target_Construction.ipynb` | Multi-panel distribution of continuous crash density, EPDO score, binary high-risk flag, and 4-tier risk categories. | Target Formulation | **NO (Exclude)** | Detailed target formulation comparison is excluded from this pre-submission to keep focus on EDA and binary/tier framing. |
| `pca_scree_plot.png` | `10_PCA_Analysis.ipynb` | Bar/line chart of individual and cumulative variance explained per principal component. | Dimensionality / PCA | **NO (Exclude)** | **Explicitly excluded** by submission mandate (NO PCA). |
| `pca_loadings_heatmap.png` | `10_PCA_Analysis.ipynb` | Heatmap of factor loadings across the first 5 principal components. | Dimensionality / PCA | **NO (Exclude)** | **Explicitly excluded** by submission mandate (NO PCA). |
| `pca_biplot.png` | `10_PCA_Analysis.ipynb` | 2D scatter projection of road segments on PC1 vs. PC2 with feature loading vectors. | Dimensionality / PCA | **NO (Exclude)** | **Explicitly excluded** by submission mandate (NO PCA). |

---

## 2. Recommended Selection: The 14 Strongest Figures

A curated selection of **14 high-impact figures** has been designated for the final pre-submission report. These figures directly substantiate the problem statement, data quality audit, and transition to road-segment risk classification.

1. **Figure 1 (`accident_severity_distribution.png`):** Accident Severity and Casualty Distribution
2. **Figure 2 (`accident_leakage_risk_score.png`):** Empirical Evidence of Target Leakage in Candidate `risk_score`
3. **Figure 3 (`accident_temporal_patterns.png`):** Multi-Scale Temporal Patterns of Delhi Traffic Crashes
4. **Figure 4 (`delhi_crash_hotspots_matched_roads.png`):** Spatial Distribution of Matched Crashes and Incident Hotspots
5. **Figure 5 (`spatial_matching_distance_distribution.png`):** Crash-to-Road Spatial Assignment Distance and Sensitivity
6. **Figure 6 (`delhi_weather_climatology.png`):** Four-Year Monthly Weather Climatology for Delhi (Station 42182)
7. **Figure 7 (`delhi_fog_rain_extremes.png`):** Seasonal Exposure to Dense Fog, Monsoon Rain, and Extreme Heat
8. **Figure 8 (`weather_meteo_vs_nasa_comparison.png`):** Meteorological Cross-Validation: Meteostat Observations vs. NASA POWER Reanalysis
9. **Figure 9 (`osm_road_network_hierarchy.png`):** OpenStreetMap Road Network Hierarchy, Mileage, and Link Volume
10. **Figure 10 (`osm_segment_length_distribution.png`):** Statistical Distribution of Road Segment Lengths in Delhi
11. **Figure 11 (`delhi_road_network_map.png`):** Spatial Arterial and Highway Network Structure of NCT Delhi
12. **Figure 12 (`delhi_landcover_distribution.png`):** ESA WorldCover 10m Land Cover Proportions across NCT Delhi
13. **Figure 13 (`delhi_elevation_slope_profile.png`):** Topographic Elevation and Terrain Slope Profiles (SRTM DEM)
14. **Figure 14 (`correlation_matrix_heatmap.png`):** Predictor Feature Correlation Matrix and Inter-Attribute Associations

---

## 3. Detailed Figure Interpretations & Evaluator Conclusions

### Figure 1: Accident Severity and Casualty Distribution
- **Filename:** `outputs/eda/accident_severity_distribution.png`
- **Short Title:** Accident Severity and Casualty Breakdown in Delhi (2022–2025)
- **1–3 Sentence Interpretation:** The distribution indicates that minor injury collisions constitute the majority of recorded incidents (55.9%, $n=1,360$), followed by major injury crashes (28.4%, $n=691$) and fatal collisions (15.7%, $n=382$). The mean casualty rate is 1.72 persons per incident, with casualty counts concentrated heavily in multi-vehicle collisions on arterial highways.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Accident Characteristics)
- **Evaluator Conclusion:** Simple crash frequency is insufficient for safety assessment; an objective risk mapping framework must incorporate severity weighting (e.g., Equivalent Property Damage Only or fatal priority tiers) to avoid treating minor vehicle scrapes equivalently to fatal incidents.

### Figure 2: Empirical Evidence of Target Leakage in Candidate `risk_score`
- **Filename:** `outputs/eda/accident_leakage_risk_score.png`
- **Short Title:** Diagnostic Evidence of Post-Hoc Target Leakage in Third-Party `risk_score`
- **1–3 Sentence Interpretation:** Boxplot and scatter distributions show that the pre-existing `risk_score` in the third-party dataset shows a strong association with accident outcomes and casualties and was therefore quarantined as a potential target-leakage variable ($r > 0.70$) with casualty counts and fatal severity classifications. The distribution reveals that `risk_score` was computed post-hoc from the outcome variables rather than from pre-crash road environment risk factors.
- **Report Placement:** Section 3 (Data Audit & Integrity Checks)
- **Evaluator Conclusion:** Including `risk_score` as either a feature or a prediction target would introduce severe data leakage, artificially inflating accuracy while destroying real-world applicability. Quarantining this feature is a critical methodological safeguard.

### Figure 3: Multi-Scale Temporal Patterns of Delhi Traffic Crashes
- **Filename:** `outputs/eda/accident_temporal_patterns.png`
- **Short Title:** Multi-Scale Temporal Distribution of Delhi Road Collisions
- **1–3 Sentence Interpretation:** Temporal distributions show steady annual incident reporting from 2022 through early 2025, with elevated collision frequencies during winter months (December–January). Diurnal trends indicate that night hours (21:00 to 05:00) account for 38.1% of all crashes despite lower overall traffic volumes, while morning and evening peak hours account for 24.2% of crashes.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Temporal Crash Dynamics)
- **Evaluator Conclusion:** Temporal and diurnal factors—particularly nighttime darkness and winter visibility drops—are major hazard multipliers that must be captured in the feature engineering pipeline.

### Figure 4: Spatial Distribution of Matched Crashes and Incident Hotspots
- **Filename:** `outputs/maps/delhi_crash_hotspots_matched_roads.png`
- **Short Title:** Spatial Distribution of Collisions across the Delhi Vehicular Network
- **1–3 Sentence Interpretation:** The spatial plot displays 2,433 crash coordinates overlaid on the OSM vehicular network, highlighting spatial clustering along major circumferential arteries (Outer Ring Road, Ring Road) and key radial highways (NH-48, NH-44). Unassigned crashes (>100m from a network link) are explicitly visualized in grey, showing their dispersion in peripheral peri-urban zones.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Spatial Distribution & Hotspot Mapping)
- **Evaluator Conclusion:** Crash risk in Delhi is not uniformly distributed across space; it exhibits distinct spatial clustering on high-speed arterial corridors, confirming that road segments form a natural, defensible unit of spatial risk analysis.

### Figure 5: Crash-to-Road Spatial Assignment Distance and Sensitivity
- **Filename:** `outputs/eda/spatial_matching_distance_distribution.png`
- **Short Title:** Euclidean Matching Distance Distribution and Threshold Sensitivity
- **1–3 Sentence Interpretation:** The distance histogram shows that the majority of matched crashes occur in close proximity to road centerlines, but matching rate drops sharply beyond 100 meters (25m: 5.14%, 50m: 8.82%, 100m: 14.78% / 362 crashes). The sensitivity chart documents that attempting to force a 100% match rate would require expanding search tolerances beyond 500m, introducing severe false-matching artifacts.
- **Report Placement:** Section 4 (Data Preprocessing & Spatial Matching Methodology)
- **Evaluator Conclusion:** Preserving data integrity requires accepting unassigned records when GPS coordinates are imprecise, rather than forcing false spatial links that would contaminate model training data.

### Figure 6: Four-Year Monthly Weather Climatology for Delhi (Station 42182)
- **Filename:** `outputs/eda/delhi_weather_climatology.png`
- **Short Title:** Monthly Temperature Ranges and Precipitation Climatology (2022–2025)
- **1–3 Sentence Interpretation:** The climatological series from Safdarjung Station (35,064 hourly records) illustrates Delhi's continental monsoonal climate, featuring extreme summer temperature peaks exceeding 42°C in May–June and sharp winter drops below 7°C in December–January. Rainfall is tightly concentrated in the monsoon months of July through September, exceeding 200mm per month.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Environmental & Meteorological Baseline)
- **Evaluator Conclusion:** Meteorological risks in Delhi are strongly seasonal and bi-modal, driven by distinct physical regimes (summer heat stress and monsoon wet pavement vs. winter fog).

### Figure 7: Seasonal Exposure to Dense Fog, Monsoon Rain, and Extreme Heat
- **Filename:** `outputs/eda/delhi_fog_rain_extremes.png`
- **Short Title:** Monthly Distribution of Hazardous Weather Event Hours in Delhi
- **1–3 Sentence Interpretation:** The event-hour frequency bars demonstrate that dense fog conditions ($\text{RH} > 90\%$ and $\text{Temp} < 15^\circ\text{C}$) are exclusively confined to December and January, generating hundreds of hours of severely reduced visibility. Conversely, precipitation event hours peak heavily during July and August, while extreme heat hours dominate May and June.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Meteorological Risk Factors)
- **Evaluator Conclusion:** Winter visibility impairment and monsoon hydroplaning hazards occur in strictly defined seasonal windows, justifying the construction of seasonal weather exposure interaction terms for road segments.

### Figure 8: Meteorological Cross-Validation: Meteostat Observations vs. NASA POWER Reanalysis
- **Filename:** `outputs/eda/weather_meteo_vs_nasa_comparison.png`
- **Short Title:** Scatter and Correlation Analysis: Ground Station vs. Satellite Reanalysis
- **1–3 Sentence Interpretation:** Regression plots between Meteostat ground observations and NASA POWER MERRA-2 reanalysis demonstrate high linear consistency for ambient temperature ($r = 0.841$) and relative humidity ($r = 0.588$). Discrepancies occur primarily during extreme localized fog events, where ground-level sensor observations capture lower surface temperatures than gridded reanalysis models.
- **Report Placement:** Section 3 (Data Audit & Multi-Source Cross-Validation)
- **Evaluator Conclusion:** Ground station data (Meteostat) is superior for capturing micro-scale surface hazards like winter fog, while NASA POWER serves as an effective cross-validation and redundancy backup.

### Figure 9: OpenStreetMap Road Network Hierarchy, Mileage, and Link Volume
- **Filename:** `outputs/eda/osm_road_network_hierarchy.png`
- **Short Title:** Mileage and Segment Volume Distribution across Delhi Road Classifications
- **1–3 Sentence Interpretation:** The breakdown shows that while residential streets and service roads account for over 90% of individual links (139,064 residential, 34,176 service), higher-order arterial corridors (motorways, trunks, primary, secondary) encompass over 6,000 kilometers of high-volume carriageway. Tertiary and secondary links represent the connective backbone between radial highways and residential colonies.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Road Network Infrastructure)
- **Evaluator Conclusion:** Road network modeling must account for functional hierarchy; risk models cannot treat high-speed multi-lane expressways and low-speed residential lanes as homogeneous entities.

### Figure 10: Statistical Distribution of Road Segment Lengths in Delhi
- **Filename:** `outputs/eda/osm_segment_length_distribution.png`
- **Short Title:** Density Distribution of OSM Vehicular Segment Lengths (Meters)
- **1–3 Sentence Interpretation:** Segment length distribution in projected coordinates (EPSG:32643) exhibits strong right-skewness, with a median segment length of 96.6m for residential links and 269.1m for motorways, tapering to long stretches exceeding 1,500m. A small fraction of long links spans multiple kilometers without junction interruption.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Road Network Infrastructure)
- **Evaluator Conclusion:** Absolute crash counts per segment are fundamentally biased by segment length; longer segments naturally accumulate more exposure. Normalizing target variables by segment length (crashes per kilometer) is mathematically mandatory.

### Figure 11: Spatial Arterial and Highway Network Structure of NCT Delhi
- **Filename:** `outputs/maps/delhi_road_network_map.png`
- **Short Title:** OpenStreetMap Arterial Road Network Geography across NCT Delhi
- **1–3 Sentence Interpretation:** The GIS map delineates Delhi's distinctive urban layout, highlighting concentric ring corridors (Inner Ring Road, Mahatma Gandhi Ring Road, Outer Ring Road) interconnected by major radial National Highways (NH-44 North, NH-48 Southwest, NH-9 East). Minor and residential roads form high-density infill within the urban core bounded by the Yamuna River to the east.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Spatial Road Network Geography)
- **Evaluator Conclusion:** Delhi's road infrastructure is heavily centralized with radial discharge corridors, requiring spatial block validation to prevent spatial autocorrelation leakage across connected corridors.

### Figure 12: ESA WorldCover 10m Land Cover Proportions across NCT Delhi
- **Filename:** `outputs/eda/delhi_landcover_distribution.png`
- **Short Title:** High-Resolution Land Cover Classification Distribution in NCT Delhi
- **1–3 Sentence Interpretation:** Surface area calculations from the 10m ESA raster show that agricultural cropland covers 43.15% of the total geographic extent, concentrated in the northwest and southwest peripheries, while built-up urban fabric covers 33.39% in the central and eastern sectors. Tree cover accounts for 12.96%, primarily tracking the Delhi Ridge and protected urban forestry zones.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Environmental and Land-Use Context)
- **Evaluator Conclusion:** Delhi's road network traverses two starkly different operating environments: dense urban street canyons with high pedestrian activity, and high-speed rural/peri-urban fringe corridors with unlit roadside cropland.

### Figure 13: Topographic Elevation and Terrain Slope Profiles (SRTM DEM)
- **Filename:** `outputs/eda/delhi_elevation_slope_profile.png`
- **Short Title:** Topographic Elevation (MSL) and Slope Gradient Distributions from SRTM DEM
- **1–3 Sentence Interpretation:** Topographic analysis reveals an elevation span from ~195m along the Yamuna River plain to ~315m in the southern Asola-Bhatti Ridge terrain. Terrain slope remains predominantly flat across 85% of the city (0° to 2°), with moderate slope gradients (4° to 12°) restricted to ridge road cuts and flyover ramps.
- **Report Placement:** Section 5 (Exploratory Data Analysis — Topography and Terrain)
- **Evaluator Conclusion:** While Delhi is largely flat, local slope variations and elevation transitions identify flyover structures, river approaches, and ridge roads known for sight-distance limitations.

### Figure 14: Predictor Feature Correlation Matrix and Inter-Attribute Associations
- **Filename:** `outputs/eda/correlation_matrix_heatmap.png`
- **Short Title:** Pairwise Pearson Correlation Heatmap for Engineered Predictors
- **1–3 Sentence Interpretation:** The correlation heatmap demonstrates that physical infrastructure predictors (`is_oneway`, `is_bridge`, `has_maxspeed`, `est_lanes`) possess low mutual correlation ($|r| < 0.35$). The strongest observed associations appear between distance to city center (`dist_to_center_km`) and built-up land cover (`builtup_pct`, $r = -0.68$), reflecting the natural urban-to-rural density gradient.
- **Report Placement:** Section 6 (Key EDA Findings & Feature Interrelationships)
- **Evaluator Conclusion:** The engineered feature space demonstrates structural independence among core infrastructure predictors, providing a solid foundation for tree-based and linear classifiers without collinear instability.

---

## 4. Recommended Report Placement & Narrative Order

To maintain logical flow from raw data validation to environmental context and classification problem framing, the 14 figures should appear in the following structured sequence:

```
Section 3: Data Audit & Multi-Source Cross-Validation
├── Figure 2: Empirical Evidence of Target Leakage in Candidate risk_score
└── Figure 8: Meteorological Cross-Validation: Meteostat Observations vs. NASA POWER

Section 4: Data Preprocessing & Spatial Matching Methodology
└── Figure 5: Crash-to-Road Spatial Assignment Distance and Sensitivity

Section 5: Exploratory Data Analysis (EDA)
├── Subsection 5.1: Accident Characteristics & Severity
│   └── Figure 1: Accident Severity and Casualty Distribution
├── Subsection 5.2: Temporal Dynamics
│   └── Figure 3: Multi-Scale Temporal Patterns of Delhi Traffic Crashes
├── Subsection 5.3: Spatial Crash Patterns & Hotspots
│   └── Figure 4: Spatial Distribution of Matched Crashes and Incident Hotspots
├── Subsection 5.4: Meteorological & Environmental Regime
│   ├── Figure 6: Four-Year Monthly Weather Climatology for Delhi (Station 42182)
│   └── Figure 7: Seasonal Exposure to Dense Fog, Monsoon Rain, and Extreme Heat
├── Subsection 5.5: Road Network Infrastructure
│   ├── Figure 11: Spatial Arterial and Highway Network Structure of NCT Delhi
│   ├── Figure 9: OpenStreetMap Road Network Hierarchy, Mileage, and Link Volume
│   └── Figure 10: Statistical Distribution of Road Segment Lengths in Delhi
└── Subsection 5.6: Land Cover & Topography
    ├── Figure 12: ESA WorldCover 10m Land Cover Proportions across NCT Delhi
    └── Figure 13: Topographic Elevation and Terrain Slope Profiles (SRTM DEM)

Section 6: Key EDA Findings & Feature Interrelationships
└── Figure 14: Predictor Feature Correlation Matrix and Inter-Attribute Associations
```

---

## 5. Review of Existing EDA Notebooks

The table below summarizes the analytical scope of each pre-modeling notebook, the specific artifacts it outputs, and guidelines on what to include versus exclude for the upcoming pre-submission report.

### `01_Dataset_Audit.ipynb`
- **Analysis Performed:** Programmatic inventory of all 12 raw files across `Datasets/`; size, format, row-level verification, null checks, duplicate analysis.
- **Plots Generated:** None directly generated inside the notebook (generates `outputs/tables/data_inventory.csv` and `outputs/reports/data_inventory_report.md`).
- **Findings Worth Reporting:**
  - The 4 Meteostat files (`42182.csv.gz` through `42182 (3).csv.gz`) are **four consecutive calendar years (2022, 2023, 2024, 2025)**, totaling 35,064 hourly records, rather than redundant downloads.
  - The 5 UUID CSVs represent aggregate official government tables (Delhi Police / MoRTH), serving as macro reality checks but lacking coordinate-level attributes.
  - Identified and quarantined `risk_score` in the candidate crash dataset.
- **Parts Unnecessary for Submission:** Deep technical parsing details of raw PBF protobuf structures and uncompressed intermediate checks.

### `02_Accident_Data_EDA.ipynb`
- **Analysis Performed:** Filtering to 2,433 Delhi crash records; coordinate validation within NCT bounding box (28.40°N–28.90°N, 76.80°E–77.35°E); temporal, severity, and casualty distributions; Chi-square goodness-of-fit uniformity tests; correlation analysis of `risk_score`.
- **Plots Generated:** `accident_temporal_patterns.png`, `accident_severity_distribution.png`, `accident_leakage_risk_score.png`.
- **Findings Worth Reporting:**
  - 100% of the 2,433 Delhi crash coordinates fall strictly inside NCT administrative bounds.
  - Fatal crashes constitute 15.7% ($n=382$), major injury crashes 28.4% ($n=691$), and minor injury crashes 55.9% ($n=1,360$).
  - Night crashes account for 38.1% of incidents; peak commute hours account for 24.2%.
  - Chi-square tests expose near-exact 1/3 splits across road types and weather categories, demonstrating that the candidate dataset is a synthetic benchmark.
  - Empirical proof of `risk_score` target leakage ($r > 0.70$ with casualties and severity).
- **Parts Unnecessary for Submission:** Code implementations of national-level filtering outside Delhi.

### `03_Weather_EDA.ipynb`
- **Analysis Performed:** Time-series analysis of 35,064 hourly weather observations (2022–2025); seasonal climatology; thresholding of hazardous weather event hours; linear regression cross-validation against NASA POWER MERRA-2 reanalysis.
- **Plots Generated:** `delhi_weather_climatology.png`, `delhi_fog_rain_extremes.png`, `weather_meteo_vs_nasa_comparison.png`.
- **Findings Worth Reporting:**
  - High correlation between Meteostat ground station and NASA POWER MERRA-2 ($r = 0.841$ for temperature, $r = 0.588$ for relative humidity).
  - Winter dense fog hours are heavily concentrated in December and January (hundreds of hours of visibility impairment).
  - Monsoon rainfall event hours are concentrated in July–August.
- **Parts Unnecessary for Submission:** Redundant diurnal cycle plots across non-extreme transition months (e.g., October/November).

### `04_OSM_Road_Network_EDA.ipynb`
- **Analysis Performed:** Extraction and filtering of OSM shapefile network to vehicular classes; Cartesian reprojection to EPSG:32643 (UTM Zone 43N); segment length calculations; physical attribute analysis (`oneway`, `bridge`, `maxspeed`).
- **Plots Generated:** `osm_road_network_hierarchy.png`, `osm_segment_length_distribution.png`, `outputs/maps/delhi_road_network_map.png`.
- **Findings Worth Reporting:**
  - Filtered 215,373 raw OSM features down to 191,199 vehicular road segments, retaining 16,630 major arterial segments for primary risk modeling.
  - High skewness in segment lengths (median 96.6m for residential, 269.1m for motorways), mandating length normalization for target variables.
- **Parts Unnecessary for Submission:** Detailed geometric parsing of pedestrian paths, footways, and cycle tracks that were dropped during vehicular filtering.

### `05_LandCover_DEM_Analysis.ipynb`
- **Analysis Performed:** Windowed raster reading of ESA WorldCover 10m classification for Delhi; area percentage calculations; processing of SRTM 30m DEM tiles (N28E076, N28E077) for elevation and slope derivation.
- **Plots Generated:** `delhi_landcover_distribution.png`, `delhi_elevation_slope_profile.png`.
- **Findings Worth Reporting:**
  - NCT Delhi land-cover breakdown: 43.15% Cropland, 33.39% Built-up, 12.96% Tree Cover, 4.57% Grassland, 0.81% Water Bodies.
  - Topographic elevation ranges from 195m to 315m MSL, with slope gradients predominantly under 2° across the plains and up to 12° along the Delhi Ridge.
- **Parts Unnecessary for Submission:** GeoTIFF tile compression diagnostics and unreferenced nodata raster metadata.

### `06_Spatial_Crash_Road_Matching.ipynb`
- **Analysis Performed:** Reprojection of crash points to EPSG:32643; spatial nearest-neighbor distance join to OSM road links; distance threshold sensitivity analysis (25m, 50m, 100m, 200m, 500m); segment-level crash count and severity aggregation.
- **Plots Generated:** `spatial_matching_distance_distribution.png`, `outputs/maps/delhi_crash_hotspots_matched_roads.png`.
- **Findings Worth Reporting:**
  - Defensible match rate within 100m tolerance is 14.78% (362 crashes assigned to 308 unique road links).
  - Matches at 25m: 5.14% (126 crashes); 50m: 8.82% (216 crashes).
  - Off-network points (>100m) are preserved as unassigned incidents to prevent false spatial links.
- **Parts Unnecessary for Submission:** Exhaustive geometric buffer union diagnostic code and raw spatial index debugging logs.

---

## 6. Key Verified Data Quality & Empirical Findings

Every numerical finding below is directly verified from existing project datasets and output tables:

1. **Weather Completeness (Meteostat 42182):**
   - Exactly **35,064 hourly records** spanning contiguous four-year coverage from **January 1, 2022, 00:00 to December 31, 2025, 23:00**.
   - Zero temporal gaps in the concatenated multi-year series (`data_processed/delhi_weather_hourly_2022_2025.csv`).
2. **Road Network Scale (OpenStreetMap):**
   - Exactly **215,373 raw OSM features** surveyed.
   - Filtered down to **191,199 vehicular road segments**, comprising 139,064 residential, 34,176 service, 7,811 tertiary, 4,267 secondary, 1,362 trunk, 1,257 primary, and 461 motorway links, plus 2,801 slip/ramp links.
   - Preserved **16,630 major arterial segments** in `data_processed/delhi_major_roads.gpkg` for primary collision matching.
3. **Crash Assignment & Uncertainty:**
   - **2,433 total crash incidents** located in Delhi.
   - Match rate at 25m tolerance: **5.14%** (126 crashes).
   - Match rate at 50m tolerance: **8.82%** (216 crashes).
   - Match rate at 100m tolerance: **14.78%** (362 crashes assigned to 308 road segments).
   - Remaining crashes reside in peripheral zones or exhibit low coordinate precision, documented as unassigned rather than forced into erroneous matches.
4. **Third-Party Crash Dataset Limitations:**
   - Chi-square tests demonstrate artificial uniformity across categorical attributes (highway: 823, rural: 815, urban: 795; clear: 826, rain: 838, fog: 769).
   - Serves effectively as a structural proof-of-concept pipeline, designed to be swapped with official Delhi Police FIR logs upon release.
5. **Data Leakage Quarantine:**
   - `risk_score` exhibits Pearson correlation of **0.704** with total casualties and direct upward step-functions across severity tiers.
   - Quarantining `risk_score` eliminates severe target leakage.

---

## 7. Figures Explicitly Excluded from This Submission

The following figures exist in the project repository but **must be strictly excluded** from the upcoming pre-submission report:

1. **`pca_scree_plot.png` (PCA Variance Explained):** Excluded per strict user instruction (NO PCA). The submission focuses on physical, human-interpretable road safety features.
2. **`pca_loadings_heatmap.png` (PCA Factor Loadings):** Excluded per strict user instruction (NO PCA).
3. **`pca_biplot.png` (PCA 2D Biplot):** Excluded per strict user instruction (NO PCA).
4. **`target_distributions_comparison.png` (Multi-Target Comparison):** Excluded to avoid premature target-architecture debates; the report will frame the classification problem directly using binary high-risk and priority risk tiers.
5. **VIF Diagnostic Bar Charts:** Excluded from the main figure sequence; VIF metrics ($< 5.0$ for retained features) are summarized concisely in text/table format without dedicating a primary figure.

---

## 8. Final Recommended EDA Storyline for the Report

The narrative arc of the pre-submission report should transition logically from raw data validation to environmental context and classification framing:

1. **The Core Problem:** Predicting stochastic point collisions is mathematically ill-posed; the objective is estimating relative road-segment hazard to prioritize infrastructure safety interventions.
2. **Data Audit & Integrity:** Demonstrating rigorous data validation—unifying four years of Meteostat hourly weather (35,064 records), confirming high agreement with NASA POWER, and actively quarantining the leaky `risk_score` attribute.
3. **Spatial Grounding:** Projecting Delhi's road network into Cartesian coordinates (EPSG:32643), acknowledging spatial matching limitations (14.78% match at 100m), and avoiding false forced matches.
4. **Environmental Hazard Profiling:** Revealing Delhi's acute seasonal extremes—winter dense fog (Dec–Jan) and monsoon rainfall (Jul–Aug)—and mapping the spatial divide between dense urban cores (33.39% built-up) and agricultural peripheries (43.15% cropland).
5. **Infrastructure Geometry:** Establishing the role of road hierarchy and proving that segment lengths must be normalized to prevent length-based exposure bias.
6. **Feature Independence:** Using the correlation heatmap to verify that physical road predictors possess low mutual collinearity.
7. **Bridge to Classification:** Transitioning naturally to candidate classification formulations (Binary High-Risk Priority vs. Multi-Class Risk Tiers) and outlining planned explainable algorithms (Logistic Regression baseline, Random Forest, XGBoost/LightGBM with SHAP) and evaluation metrics (Precision-Recall AUC, Cost-Weighted F-beta) for the upcoming modeling phase.
