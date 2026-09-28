# RoadSafe Delhi — Final Analytical Dataset Dictionary

**Dataset File:** `data_engineered/delhi_road_risk_ml_dataset.parquet`  
**Row Grain:** 1 row per OSM Road Segment (Link)  
**Total Records:** 12315  

| Column Name | Type | Domain | Role | Description |
|:---|:---|:---|:---|:---|
| `osm_id` | Int64 | Identifier | Index | Unique OpenStreetMap road link identifier |
| `type` | String | Infrastructure | Metadata | Road functional classification (motorway, trunk, primary, etc.) |
| `length_m` | Float64 | Infrastructure | Predictor | Segment length in meters (projected in EPSG:32643) |
| `est_lanes` | Int32 | Infrastructure | Predictor | Estimated carriage lanes based on OSM hierarchy |
| `is_oneway` | Int32 | Infrastructure | Predictor | 1 if designated one-way traffic, 0 otherwise |
| `is_bridge` | Int32 | Infrastructure | Predictor | 1 if road segment is an elevated structure / flyover |
| `has_maxspeed` | Int32 | Infrastructure | Predictor | 1 if speed limit attribute is explicitly posted |
| `centroid_lon` | Float64 | Spatial | Coordinate | Longitude of road link geographic centroid (WGS84) |
| `centroid_lat` | Float64 | Spatial | Coordinate | Latitude of road link geographic centroid (WGS84) |
| `dist_to_center_km`| Float64 | Spatial | Predictor | Distance to Connaught Place urban core (km) |
| `elevation_m` | Float64 | Topographic | Predictor | Mean elevation above sea level in meters (SRTM DEM) |
| `slope_deg` | Float64 | Topographic | Predictor | Terrain gradient slope in degrees |
| `builtup_pct` | Float64 | Environmental | Predictor | Surrounding built-up land cover percentage (ESA WorldCover 10m) |
| `pca_comp_1` | Float64 | Latent Feature | Predictor | 1st Principal Component (Infrastructure scale & urbanization) |
| `pca_comp_2` | Float64 | Latent Feature | Predictor | 2nd Principal Component (Topography & elevation gradient) |
| `pca_comp_3` | Float64 | Latent Feature | Predictor | 3rd Principal Component (Traffic control & carriageway flow) |
| `crash_count` | Int32 | Crash History | Summary | Total historical crash points assigned within 100m tolerance |
| `fatal_count` | Int32 | Crash Severity | Summary | Total fatal crashes assigned to segment |
| `major_count` | Int32 | Crash Severity | Summary | Total major injury crashes assigned to segment |
| `minor_count` | Int32 | Crash Severity | Summary | Total minor injury crashes assigned to segment |
| `total_casualties` | Int32 | Crash Severity | Summary | Total persons killed or injured on segment |
| `crash_density_per_km` | Float64 | Crash Exposure | Target (Cont) | Normalized crash rate per kilometer of segment length |
| `EPDO_score` | Float64 | Severity Index | Target (EPDO) | Equivalent Property Damage Only index per km |
| `is_high_risk` | Int32 | Classification | Target (Binary)| 1 if segment is high-risk priority candidate, 0 otherwise |
| `risk_tier` | Int32 | Ordinal Tier | Target (Multi) | 0: Baseline, 1: Moderate, 2: High, 3: Critical Priority |
| `spatial_block` | String | Spatial Validation| Partition | Spatial grid block identifier for spatial cross-validation |
| `ml_split` | String | Spatial Validation| Split | Data split tag: `train` (60%), `val` (20%), `test` (20%) |
