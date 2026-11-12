"""
build_presubmission_report.py
Generates the official RoadSafe Delhi Pre-Submission Report in both .docx (Word)
and .md (Markdown) formats with embedded figures, strictly adhering to factual consistency,
non-causal language, and all course submission constraints.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = r"d:\RoadSafe Delhi"
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
EDA_DIR = os.path.join(PROJECT_ROOT, "outputs", "eda")
MAPS_DIR = os.path.join(PROJECT_ROOT, "outputs", "maps")
TABLES_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")

os.makedirs(REPORTS_DIR, exist_ok=True)

print("Building Pre-Submission Report (.docx and .md)...")

doc = Document()

# Set standard 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Configure default font
style_normal = doc.styles['Normal']
font = style_normal.font
font.name = 'Calibri'
font.size = Pt(11)
font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_header(title, level=1):
    p = doc.add_heading(level=level)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(title)
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x29, 0x80, 0xB9)
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x34, 0x49, 0x5E)
    return p

def add_paragraph(text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.add_run(text)
    return p

def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.add_run(text)
    return p

def add_figure(img_rel_path, fig_num, caption, observation, interpretation):
    img_full_path = os.path.join(PROJECT_ROOT, img_rel_path)
    if os.path.exists(img_full_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(img_full_path, width=Inches(5.8))
        
        # Caption
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(4)
        r_cap_lbl = p_cap.add_run(f"Figure {fig_num}: ")
        r_cap_lbl.bold = True
        r_cap_lbl.font.size = Pt(10)
        r_cap_lbl.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        r_cap_txt = p_cap.add_run(caption)
        r_cap_txt.font.size = Pt(10)
        r_cap_txt.font.italic = True
        
        # Observation and Interpretation Callout Box / Text
        p_desc = doc.add_paragraph()
        p_desc.paragraph_format.space_after = Pt(8)
        p_desc.paragraph_format.line_spacing = 1.15
        
        r_obs_tag = p_desc.add_run("Observation: ")
        r_obs_tag.bold = True
        r_obs_tag.font.size = Pt(9.5)
        r_obs_tag.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
        r_obs_txt = p_desc.add_run(observation + " ")
        r_obs_txt.font.size = Pt(9.5)
        
        r_int_tag = p_desc.add_run("Interpretation: ")
        r_int_tag.bold = True
        r_int_tag.font.size = Pt(9.5)
        r_int_tag.font.color.rgb = RGBColor(0x29, 0x80, 0xB9)
        r_int_txt = p_desc.add_run(interpretation)
        r_int_txt.font.size = Pt(9.5)
    else:
        print(f"WARNING: Image not found: {img_full_path}")

# ==============================================================================
# TITLE & COVER INFORMATION
# ==============================================================================
p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(20)
p_title.paragraph_format.space_after = Pt(2)
r_t = p_title.add_run("RoadSafe Delhi")
r_t.font.size = Pt(26)
r_t.bold = True
r_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(12)
r_sub = p_sub.add_run("Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning\nPre-Submission Project Evaluation Report")
r_sub.font.size = Pt(13)
r_sub.font.color.rgb = RGBColor(0x29, 0x80, 0xB9)

# Metadata Block
table_meta = doc.add_table(rows=4, cols=2)
table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_data = [
    ("Study Area:", "National Capital Territory (NCT) of Delhi, India"),
    ("Target Analysis Unit:", "Road Segments / Road Links (OpenStreetMap Carriageway Elements)"),
    ("Current Milestone:", "Data Audit, Preprocessing, Exploratory Analysis, & Classification Framing"),
    ("Methodological Compliance:", "Strict Zero-Leakage Protocol, Non-Causal EDA, Verified Multi-Source Integration")
]
for i, (k, v) in enumerate(meta_data):
    cell_k, cell_v = table_meta.rows[i].cells
    cell_k.text = k
    cell_v.text = v
    cell_k.paragraphs[0].runs[0].bold = True
    cell_k.paragraphs[0].runs[0].font.size = Pt(10)
    cell_v.paragraphs[0].runs[0].font.size = Pt(10)
    set_cell_background(cell_k, "F2F4F7")
    set_cell_background(cell_v, "FFFFFF")

doc.add_paragraph().paragraph_format.space_after = Pt(12)

# ==============================================================================
# SECTION 1: PROBLEM STATEMENT
# ==============================================================================
add_header("1. Problem Statement", level=1)
add_paragraph(
    "Urban road traffic collisions represent one of the most critical public safety and public health crises in metropolitan Delhi. "
    "However, conventional modeling attempts often frame the objective around predicting the exact, point-in-time occurrence of individual accidents. "
    "Stochastic individual collision events are mathematically rare and characterized by extreme temporal sparsity, making point prediction scientifically ill-posed and practically unhelpful for transport planning.",
    bold_prefix="Core Problem Formulation: "
)
add_paragraph(
    "RoadSafe Delhi explicitly rejects point-occurrence prediction. Instead, this research formulates the machine learning task as "
    "road-segment relative risk estimation and safety intervention prioritization. "
    "The central research question guiding the project is:",
    bold_prefix="Research Paradigm: "
)
p_q = doc.add_paragraph()
p_q.paragraph_format.left_indent = Inches(0.4)
p_q.paragraph_format.right_indent = Inches(0.4)
p_q.paragraph_format.space_before = Pt(4)
p_q.paragraph_format.space_after = Pt(8)
r_q = p_q.add_run(
    '"Can machine learning integrate historical road-crash patterns with road-network, spatial, temporal, '
    'and environmental characteristics to identify high-risk road locations in Delhi and support prioritization for preventive road-safety assessment?"'
)
r_q.italic = True
r_q.bold = True
r_q.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

add_paragraph(
    "The analysis unit is established as the Road Segment / Road Link. By estimating segment-level risk rates and categorizing corridors into "
    "actionable risk priority tiers, the resulting models will directly support municipal engineers and traffic enforcement agencies in deploying "
    "preventive interventions (such as speed management, geometric realignment, lighting, and targeted enforcement) prior to future collisions.",
    bold_prefix="Practical Utility: "
)

# ==============================================================================
# SECTION 2: DATASETS AND SOURCES
# ==============================================================================
add_header("2. Datasets and Multi-Source Architecture", level=1)
add_paragraph(
    "To characterize road hazards comprehensively, the project integrates multi-source geospatial, infrastructure, meteorological, and incident data. "
    "All raw datasets are permanently maintained in an unedited, pristine state inside the Datasets/ directory.",
    bold_prefix="Multi-Source Integration: "
)

table_ds = doc.add_table(rows=6, cols=4)
table_ds.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Dataset Source", "Format / Scale", "Verified Volume", "Role in Pipeline"]
for j, h in enumerate(headers):
    c = table_ds.rows[0].cells[j]
    c.text = h
    c.paragraphs[0].runs[0].bold = True
    c.paragraphs[0].runs[0].font.size = Pt(9.5)
    c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    set_cell_background(c, "1B365D")

ds_rows = [
    ("Meteostat Ground Observations", "Hourly CSV (.gz)\nStation 42182 (Safdarjung)", "35,064 records\n(2022–2025)", "Primary ground-truth weather for temperature, humidity, precipitation, and fog identification."),
    ("NASA POWER Reanalysis", "Hourly CSV\nMERRA-2 Satellite Grid", "35,064 records\n(2022–2025)", "Secondary reanalysis source utilized for meteorological cross-validation and redundancy verification."),
    ("OpenStreetMap (OSM) Delhi", "Vector Shapefile (.shp.zip)\nCartesian EPSG:32643", "215,373 raw features\n191,199 vehicular links", "Primary road network infrastructure defining geometry, hierarchy, carriageway width, and segment lengths."),
    ("ESA WorldCover 10m (2021)", "GeoTIFF Raster (10m)\nTile N27E075", "10m resolution\nNCT Delhi coverage", "High-resolution land-cover classification distinguishing urban fabric, tree canopy, cropland, and water."),
    ("SRTM Digital Elevation Model", "HGT Elevation Tiles\nN28E076 & N28E077", "30m resolution\n195m–315m MSL", "Topographic elevation gradients and terrain slope computations across the study area.")
]
for i, row in enumerate(ds_rows):
    for j, val in enumerate(row):
        c = table_ds.rows[i+1].cells[j]
        c.text = val
        c.paragraphs[0].runs[0].font.size = Pt(9)
        bg = "F9FAFC" if i % 2 == 0 else "FFFFFF"
        set_cell_background(c, bg)

add_paragraph(
    "A critical distinction is maintained regarding crash data provenance. Official Delhi government road accident data (represented by five aggregate UUID CSV files) "
    "provides macro-level totals (~7,000–8,000 crashes per year, ~2,000 annual fatalities) but lacks spatial coordinates. "
    "To develop the spatial matching and feature-engineering pipeline, a candidate third-party accident dataset (2,433 records in Delhi) was ingested as a supplementary benchmark. "
    "It is treated strictly as a candidate proof-of-concept pipeline dataset, fully subject to replacement when official FIR crash-level logs are released.",
    bold_prefix="Crash Data Provenance & Scope: "
)

# ==============================================================================
# SECTION 3: DATA AUDIT AND INTEGRITY CHECKS
# ==============================================================================
add_header("3. Data Audit and Integrity Checks", level=1)
add_paragraph(
    "The data audit phase programmatically inspected every raw asset to determine true internal structure rather than trusting file naming conventions. "
    "Three crucial findings emerged from this audit:",
    bold_prefix="Audit Objectives: "
)
add_bullet("The files 42182.csv.gz through 42182 (3).csv.gz were initially suspected to be redundant duplicate downloads. Detailed row-level inspection proved they are four distinct, consecutive single-year records covering 2022, 2023, 2024, and 2025. They were successfully concatenated into a contiguous 35,064-hour series.", bold_prefix="Resolution of Meteostat Multi-Year Records: ")
add_bullet("The candidate crash dataset contains an attribute named 'risk_score'. Detailed correlation analysis revealed a strong association with crash outcomes and casualties (r = 0.704), indicating it was derived post-hoc. Retaining it would introduce fatal data leakage; it was permanently quarantined.", bold_prefix="Proactive Data Leakage Quarantine: ")
add_bullet("Station observations from Safdarjung (WMO 42182) were cross-validated against MERRA-2 gridded satellite reanalysis, confirming high agreement while exposing localized surface-level fog capture.", bold_prefix="Cross-Validation of Weather Sources: ")

add_figure(
    img_rel_path=r"outputs\eda\accident_leakage_risk_score.png",
    fig_num=1,
    caption="Empirical Evidence of Target Leakage in the Candidate 'risk_score' Feature",
    observation="Boxplots and scatter plots show that risk_score values scale directly with casualty counts and step upward deterministically across minor, major, and fatal injury classes (r = 0.704).",
    interpretation="The risk_score variable shows a strong association with accident outcomes and casualties and was therefore quarantined as a potential target-leakage variable to prevent artificial accuracy inflation."
)

add_figure(
    img_rel_path=r"outputs\eda\weather_meteo_vs_nasa_comparison.png",
    fig_num=2,
    caption="Meteorological Cross-Validation: Meteostat Ground Observations vs. NASA POWER Reanalysis",
    observation="Scatter plots and linear regressions show strong temperature correlation (r = 0.841) and moderate relative humidity correlation (r = 0.588) between Safdarjung ground sensors and MERRA-2 satellite grid cells.",
    interpretation="Ground-station observations accurately capture micro-scale surface cooling and localized winter fog that gridded satellite reanalysis smooths out, validating Meteostat as the primary weather record."
)

# ==============================================================================
# SECTION 4: DATA PREPROCESSING
# ==============================================================================
add_header("4. Data Preprocessing and Spatial Matching Methodology", level=1)
add_paragraph(
    "Data preprocessing established geometric rigor and physical consistency across disparate spatial layers:",
    bold_prefix="Methodological Foundation: "
)
add_bullet("Unprojected geographic coordinates (WGS84 / EPSG:4326) produce distorted distance measurements when calculating segment lengths or buffer radiuses. All vector geometries (OSM road links and crash coordinates) were reprojected into EPSG:32643 (UTM Zone 43N), ensuring all spatial computations are performed in true Cartesian meters.", bold_prefix="Cartesian Reprojection: ")
add_bullet("The 215,373 raw OSM features were filtered to remove pedestrian pathways, footways, and cycle tracks, retaining 191,199 vehicular road segments. From these, 16,630 major arterial segments were designated for primary spatial collision assignment.", bold_prefix="Vehicular Filtering: ")
add_bullet("Crash coordinates were mapped to road links using nearest-neighbor spatial joins in UTM coordinates. Distance sensitivity was evaluated across 25m, 50m, 100m, 200m, and 500m thresholds.", bold_prefix="Spatial Assignment Diagnostics: ")

add_figure(
    img_rel_path=r"outputs\eda\spatial_matching_distance_distribution.png",
    fig_num=3,
    caption="Crash-to-Road Spatial Assignment Distance Distribution and Threshold Sensitivity",
    observation="The distance histogram shows that matched crashes concentrate near road centerlines, with match rates of 5.14% at 25m (126 crashes), 8.82% at 50m (216 crashes), and 14.78% at 100m (362 crashes assigned to 308 road links).",
    interpretation="Preserving data integrity requires accepting unassigned records when GPS coordinates are imprecise, rather than forcing distant road assignments (>100m) that would corrupt model training."
)

# ==============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
add_header("5. Exploratory Data Analysis (EDA)", level=1)

add_header("5.1 Accident Characteristics and Severity Distribution", level=2)
add_paragraph(
    "Analysis of the 2,433 Delhi crash records highlights the distribution of injury severity and victim involvement across different road environments.",
    bold_prefix="Accident Profile: "
)

add_figure(
    img_rel_path=r"outputs\eda\accident_severity_distribution.png",
    fig_num=4,
    caption="Crash Severity Proportions and Casualty Breakdown in NCT Delhi",
    observation="The distribution indicates that minor injury collisions constitute 55.9% (n=1,360), major injury collisions 28.4% (n=691), and fatal crashes 15.7% (n=382), with a mean casualty rate of 1.72 persons per incident.",
    interpretation="The severe skew toward property-damage and minor collisions demonstrates that simple crash counts are inadequate; risk modeling must incorporate severity weighting to prioritize fatal and major injury hotspots."
)

add_header("5.2 Temporal Crash Dynamics", level=2)
add_paragraph(
    "Temporal crash occurrences were examined across multi-year, seasonal, weekly, and diurnal scales.",
    bold_prefix="Temporal Trends: "
)

add_figure(
    img_rel_path=r"outputs\eda\accident_temporal_patterns.png",
    fig_num=5,
    caption="Multi-Scale Temporal Patterns of Delhi Collisions (Annual, Monthly, Day-of-Week, Hourly)",
    observation="Collision distributions show steady annual reporting, winter monthly elevations, and pronounced night-time concentration where night hours (21:00–05:00) account for 38.1% of crashes, while commute peak hours account for 24.2%.",
    interpretation="The high concentration of collisions during off-peak night hours suggests that reduced visibility and higher nighttime travel speeds act as substantial hazard multipliers in Delhi."
)

add_header("5.3 Spatial Crash Patterns and Road Hotspots", level=2)
add_paragraph(
    "Spatial overlay of crash coordinates on the arterial road network reveals pronounced spatial clustering.",
    bold_prefix="Spatial Mapping: "
)

add_figure(
    img_rel_path=r"outputs\maps\delhi_crash_hotspots_matched_roads.png",
    fig_num=6,
    caption="Spatial Distribution of Matched Collisions and Hotspots across the Delhi Vehicular Network",
    observation="The spatial map indicates collision clustering along major circumferential arteries (Ring Road, Outer Ring Road) and high-capacity radial corridors (NH-48, NH-44), while unmatched peripheral points remain distributed.",
    interpretation="Crash hazards are concentrated along high-speed multi-lane corridors rather than distributed evenly, verifying that individual road links provide a meaningful spatial unit for safety intervention."
)

add_header("5.4 Meteorological and Environmental Regime", level=2)
add_paragraph(
    "Four years of hourly meteorological observations (35,064 rows) from Safdarjung Station were analyzed to establish Delhi's environmental hazard profile.",
    bold_prefix="Climate Regime: "
)

add_figure(
    img_rel_path=r"outputs\eda\delhi_weather_climatology.png",
    fig_num=7,
    caption="Four-Year Monthly Temperature Climatology and Precipitation Totals (2022–2025)",
    observation="Monthly averages demonstrate Delhi's extreme continental climate, with summer temperatures peaking above 42°C in May–June and winter minimums dropping below 7°C in December–January, coupled with monsoonal rain peaks in July–August.",
    interpretation="Delhi's environmental conditions exhibit acute bi-modal extremes, creating distinct seasonal driving hazards (extreme heat and wet monsoon pavement vs. severe winter cold)."
)

add_figure(
    img_rel_path=r"outputs\eda\delhi_fog_rain_extremes.png",
    fig_num=8,
    caption="Monthly Distribution of Hazardous Weather Event Hours: Dense Fog, Monsoon Rain, and Extreme Heat",
    observation="Event-hour counts show that dense fog conditions (RH > 90% and Temp < 15°C) are strictly confined to December and January, while heavy precipitation hours concentrate in July–August and extreme heat hours dominate May–June.",
    interpretation="Adverse visibility hazards are strictly seasonal, confirming that weather exposure terms must be integrated to capture seasonal crash risk amplification."
)

add_header("5.5 Road Network Infrastructure and Geometry", level=2)
add_paragraph(
    "OpenStreetMap road infrastructure defines the physical carriageway topology of Delhi.",
    bold_prefix="Infrastructure Baseline: "
)

add_figure(
    img_rel_path=r"outputs\maps\delhi_road_network_map.png",
    fig_num=9,
    caption="OpenStreetMap Vehicular and Arterial Road Network Structure of NCT Delhi",
    observation="The GIS network delineates Delhi's concentric ring structures intersected by major national radial highways, with high network density in the urban core transitioning to sparser arterial links in outer sectors.",
    interpretation="The centralized radial geometry channels regional traffic through high-density urban corridors, indicating that spatial block partitioning is necessary during validation to avoid spatial leakage."
)

add_figure(
    img_rel_path=r"outputs\eda\osm_road_network_hierarchy.png",
    fig_num=10,
    caption="OpenStreetMap Road Network Hierarchy: Cumulative Mileage (km) and Segment Volume",
    observation="Bar distributions show that while residential streets dominate numerical segment volume (139,064 links), arterial corridors (motorways, trunks, primary, secondary) encompass over 6,000 kilometers of high-volume carriageway.",
    interpretation="Functional hierarchy directly governs vehicle speeds, traffic volume, and carriageway separation, making road classification a mandatory predictor in risk modeling."
)

add_figure(
    img_rel_path=r"outputs\eda\osm_segment_length_distribution.png",
    fig_num=11,
    caption="Statistical Distribution of Road Segment Lengths in Delhi (EPSG:32643 Meters)",
    observation="The segment length histogram shows severe right-skewness, with a median length of 96.6m for residential links and 269.1m for motorways, extending to long unobstructed spans exceeding 1,500m.",
    interpretation="Longer segments accumulate more physical exposure and higher raw crash counts, proving that normalizing targets by segment length (crash rate per kilometer) is mathematically essential."
)

add_header("5.6 Land Cover and Topographic Context", level=2)
add_paragraph(
    "Environmental setting and terrain elevation provide spatial context for surrounding human activity.",
    bold_prefix="Environmental Context: "
)

add_figure(
    img_rel_path=r"outputs\eda\delhi_landcover_distribution.png",
    fig_num=12,
    caption="ESA WorldCover 10m Surface Area Classification Distribution in NCT Delhi",
    observation="Land cover proportions indicate that agricultural cropland covers 43.15% of the total study area, built-up urban fabric covers 33.39%, and tree cover covers 12.96%, with grassland (4.57%) and water bodies (0.81%) comprising the remainder.",
    interpretation="Delhi's road network spans two starkly contrasting operating environments: dense urban street canyons with pedestrian conflicts, and dark peri-urban agricultural corridors with higher vehicle speeds."
)

add_figure(
    img_rel_path=r"outputs\eda\delhi_elevation_slope_profile.png",
    fig_num=13,
    caption="Topographic Elevation (MSL) and Slope Gradient Distributions from SRTM 30m DEM",
    observation="Histograms show elevation ranging from ~195m along the Yamuna River plain to ~315m on the southern Delhi Ridge, with predominantly low slopes across 85% of the territory and higher local slopes on ridge cuts and flyovers.",
    interpretation="While Delhi is largely flat, local slope variations identify flyover approach ramps and ridge transitions where sight distance and braking dynamics are constrained."
)

# ==============================================================================
# SECTION 6: KEY EDA FINDINGS
# ==============================================================================
add_header("6. Key EDA Findings and Feature Relationships", level=1)
add_paragraph(
    "The exploratory phase revealed several foundational insights that govern the transition to machine learning modeling:",
    bold_prefix="Synthesis of Findings: "
)
add_bullet("Crash occurrence in Delhi is concentrated during nighttime hours (38.1% of crashes between 21:00 and 05:00) and along major arterial ring corridors.", bold_prefix="1. Night and Arterial Hazard Multiplication: ")
add_bullet("Dense fog is confined exclusively to December and January, coinciding with winter collision peaks, while monsoon hydroplaning hazards peak in July and August.", bold_prefix="2. Seasonal Climatological Windows: ")
add_bullet("A significant percentage of crashes occur on higher-order arterial corridors, yet residential streets dominate link counts, requiring structural hierarchy features.", bold_prefix="3. Infrastructure Exposure Skewness: ")
add_bullet("Raw crash counts are heavily confounded by segment length, mandating rate-based target normalization (crashes per kilometer).", bold_prefix="4. Normalization Mandate: ")
add_bullet("Correlation analysis revealed strong pairwise associations between distance to city center and built-up land cover (r ≈ -0.96) and between elevation and slope (r ≈ 0.93). These strong associations will be carefully considered during subsequent feature selection to prevent redundancy.", bold_prefix="5. Inter-Attribute Associations: ")

add_figure(
    img_rel_path=r"outputs\eda\correlation_matrix_heatmap.png",
    fig_num=14,
    caption="Pairwise Pearson Correlation Heatmap for Engineered Road Segment Predictors",
    observation="The heatmap displays correlation coefficients across continuous predictors, showing strong inverse association between dist_to_center_km and builtup_pct (r ≈ -0.96) and strong positive association between elevation_m and slope_deg (r ≈ 0.93), while physical road attributes (is_oneway, is_bridge, has_maxspeed, est_lanes) exhibit low mutual correlation (|r| < 0.35).",
    interpretation="Strong pairwise associations were observed between spatial-density and topographic features and will be considered during subsequent feature selection to maintain model interpretability."
)

# ==============================================================================
# SECTION 7: TRANSITION TO CLASSIFICATION
# ==============================================================================
add_header("7. Transition to Classification", level=1)
add_paragraph(
    "Rather than treating road safety assessment as an unconstrained regression task or an ill-posed point-occurrence prediction, "
    "the project transitions the analytical outputs into supervised classification. Classification directly aligns with how municipal transport departments operate: "
    "allocating finite annual budgets to inspect, audit, and remediate high-risk road corridors.",
    bold_prefix="Methodological Rationale: "
)
add_paragraph(
    "Two complementary classification target formulations have been established at the road-segment level:",
    bold_prefix="Formulated Classification Targets: "
)
add_bullet("A binary indicator flagging segments that fall within the top 15% risk density corridor (or segments accumulating repeated fatal/major collisions). Segments flagged as Class 1 represent priority candidates for immediate engineering inspection.", bold_prefix="Target 1 — Binary High-Risk Screening (is_high_risk): ")
add_bullet("An ordinal 4-class target categorizing road segments into Baseline Risk (Tier 0), Moderate Risk (Tier 1), High Risk (Tier 2), and Critical Intervention Priority (Tier 3). This multi-class structure enables tiered budget allocation for municipal road safety audits.", bold_prefix="Target 2 — Multi-Class Priority Tier (risk_tier): ")

# ==============================================================================
# SECTION 8: PLANNED CLASSIFICATION MODELS
# ==============================================================================
add_header("8. Planned Classification Models", level=1)
add_paragraph(
    "To ensure transparency, prevent black-box decisions, and provide actionable engineering recommendations, "
    "the planned classification suite emphasizes explainability alongside predictive discrimination:",
    bold_prefix="Modeling Architecture: "
)
add_bullet("Serves as the foundational linear benchmark. Regularized logistic regression provides direct odds-ratio interpretability, quantifying the baseline risk multiplier associated with road class, lane count, and environmental exposure.", bold_prefix="1. Elastic-Net Regularized Logistic Regression: ")
add_bullet("An ensemble bagging algorithm that handles non-linear interactions and skewed segment length distributions without overfitting. Generates Gini impurity and permutation feature importance rankings.", bold_prefix="2. Balanced Random Forest Classifier: ")
add_bullet("State-of-the-art gradient boosting frameworks optimized for tabular data. These models will be trained using cost-sensitive weighting and focal loss to address class imbalance between safe and hazardous segments.", bold_prefix="3. LightGBM and XGBoost Classifiers: ")
add_bullet("To support policy justification, all tree-based ensemble models will be integrated with TreeSHAP (SHapley Additive exPlanations). This will output local feature contribution plots for every flagged road segment, explaining exactly which factors (e.g., lack of median, high speed limit, winter fog exposure) drove the high-risk classification.", bold_prefix="4. Post-Hoc SHAP Explainability Engine: ")

# ==============================================================================
# SECTION 9: CLASSIFICATION EVALUATION METRICS
# ==============================================================================
add_header("9. Classification Evaluation Metrics", level=1)
add_paragraph(
    "Because high-risk road segments constitute a minority class, conventional accuracy is fundamentally misleading (a naive classifier predicting all roads as safe would achieve ~85% accuracy while failing completely). "
    "The evaluation framework relies on specialized metrics:",
    bold_prefix="Metric Framework: "
)
add_bullet("Evaluates classifier discrimination across all operational thresholds without being distorted by the large number of true negative (safe) road links.", bold_prefix="Precision-Recall AUC (PR-AUC): ")
add_bullet("In road safety prioritization, a False Negative (failing to identify a genuinely hazardous fatal road segment) carries far higher societal cost than a False Positive (inspecting a segment that turns out safe). F-beta (with beta = 2) explicitly penalizes missed hazard corridors.", bold_prefix="Cost-Weighted F2 Score (Recall-Biased): ")
add_bullet("Standard random k-fold cross-validation causes spatial data leakage due to spatial autocorrelation (neighboring road segments share traffic patterns). The project enforces a 3x3 Spatial Block Partitioning across Delhi (60% Train, 20% Validation, 20% Holdout Test) to verify true geographic generalization.", bold_prefix="Spatial Block Cross-Validation: ")

# ==============================================================================
# SECTION 10: CURRENT PROGRESS
# ==============================================================================
add_header("10. Current Progress Summary", level=1)
add_paragraph(
    "The project has completed all foundational data engineering, spatial integration, and exploratory analysis phases:",
    bold_prefix="Completed Deliverables: "
)
add_bullet("All 12 raw files audited, cataloged, and verified. Meteostat files unified into 35,064 hourly records; NASA POWER cross-validated; candidate risk_score successfully quarantined.", bold_prefix="Phase 1 (Audit & Integrity): ")
add_bullet("Crash distributions, severity profiles, temporal trends, and environmental regimes fully analyzed across Notebooks 02 and 03.", bold_prefix="Phases 2 & 3 (Accident & Weather EDA): ")
add_bullet("OSM network filtered to 191,199 vehicular links and reprojected to EPSG:32643; ESA 10m land cover and SRTM DEM topography extracted across Notebooks 04 and 05.", bold_prefix="Phases 4 & 5 (Road & Environmental GIS): ")
add_bullet("Spatial nearest join executed in UTM coordinates with threshold sensitivity diagnostics; 362 crashes matched within defensible 100m tolerance across 308 road links.", bold_prefix="Phase 6 (Spatial Matching): ")
add_bullet("Engineered 18 segment-level features; formulated binary and ordinal classification targets; established 3x3 spatial block cross-validation partitions; exported final dataset to data_engineered/delhi_road_risk_ml_dataset.parquet.", bold_prefix="Phases 7, 8, & 11 (Dataset Preparation): ")

# ==============================================================================
# SECTION 11: LIMITATIONS
# ==============================================================================
add_header("11. Limitations and Threat Mitigation", level=1)
add_paragraph(
    "Rigorous scientific assessment requires acknowledging current data constraints:",
    bold_prefix="Methodological Candor: "
)
add_bullet("The third-party crash dataset exhibits artificial uniformities across certain categorical features, indicating synthetic benchmark properties. It serves as a proof-of-concept pipeline dataset, designed to be swapped seamlessly with official FIR crash records upon availability.", bold_prefix="1. Crash Data Provenance: ")
add_bullet("Due to GPS coordinate dispersion in candidate crash records, 85.2% of crashes fall outside the strict 100m road centerline tolerance. The project deliberately preserved these as unassigned rather than forcing distant false matches.", bold_prefix="2. Spatial Matching Coverage: ")
add_bullet("Annual average daily traffic (AADT) volume counts are currently unavailable at the individual link level across OpenStreetMap. Road hierarchy classes (motorway, trunk, primary) and carriageway lane counts serve as proxy indicators for traffic volume exposure.", bold_prefix="3. Traffic Volume Exposure Proxy: ")

# ==============================================================================
# SECTION 12: NEXT STEPS
# ==============================================================================
add_header("12. Next Steps", level=1)
add_paragraph(
    "Following faculty review and pre-submission sign-off, the project will immediately commence model training:",
    bold_prefix="Immediate Roadmap: "
)
add_bullet("Train regularized Logistic Regression, Balanced Random Forest, and LightGBM classifiers on the spatial training partitions.", bold_prefix="Step 1 — Baseline & Ensemble Training: ")
add_bullet("Optimize decision thresholds using cost-weighted F2 metrics to maximize recall on fatal and high-severity corridors.", bold_prefix="Step 2 — Cost-Sensitive Optimization: ")
add_bullet("Generate SHAP summary and waterfall plots to provide actionable, explainable intervention recommendations for the top-priority hazardous corridors in Delhi.", bold_prefix="Step 3 — Explainability & Intervention Mapping: ")
add_bullet("Evaluate generalization performance strictly on the unseen holdout spatial blocks to confirm real-world municipal screening utility.", bold_prefix="Step 4 — Spatial Generalization Validation: ")

# Save Word document
docx_path = os.path.join(REPORTS_DIR, "RoadSafe_Delhi_Pre_Submission_Report.docx")
doc.save(docx_path)
print(f"Successfully generated Word report: {docx_path}")

# ==============================================================================
# GENERATE MATCHING MARKDOWN VERSION
# ==============================================================================
md_content = """# RoadSafe Delhi — Pre-Submission Project Evaluation Report
## Explainable Road Risk Mapping and Safety Intervention Prioritization Using Machine Learning

**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Target Analysis Unit:** Road Segments / Road Links (OpenStreetMap Carriageway Elements)  
**Current Milestone:** Data Audit, Preprocessing, Exploratory Analysis, & Classification Framing  
**Methodological Compliance:** Strict Zero-Leakage Protocol, Non-Causal EDA, Verified Multi-Source Integration  
**Document Generated:** September 2026  

---

## 1. Problem Statement

### Core Problem Formulation
Urban road traffic collisions represent one of the most critical public safety and public health crises in metropolitan Delhi. However, conventional modeling attempts often frame the objective around predicting the exact, point-in-time occurrence of individual accidents. Stochastic individual collision events are mathematically rare and characterized by extreme temporal sparsity, making point prediction scientifically ill-posed and practically unhelpful for transport planning.

### Research Paradigm
RoadSafe Delhi explicitly rejects point-occurrence prediction. Instead, this research formulates the machine learning task as **road-segment relative risk estimation and safety intervention prioritization**. The central research question guiding the project is:

> *"Can machine learning integrate historical road-crash patterns with road-network, spatial, temporal, and environmental characteristics to identify high-risk road locations in Delhi and support prioritization for preventive road-safety assessment?"*

### Practical Utility
The analysis unit is established as the **Road Segment / Road Link**. By estimating segment-level risk rates and categorizing corridors into actionable risk priority tiers, the resulting models will directly support municipal engineers and traffic enforcement agencies in deploying preventive interventions (such as speed management, geometric realignment, lighting, and targeted enforcement) prior to future collisions.

---

## 2. Datasets and Multi-Source Architecture

To characterize road hazards comprehensively, the project integrates multi-source geospatial, infrastructure, meteorological, and incident data. All raw datasets are permanently maintained in an unedited, pristine state inside the `Datasets/` directory.

| Dataset Source | Format / Scale | Verified Volume | Role in Pipeline |
|:---|:---|:---|:---|
| **Meteostat Ground Observations** | Hourly CSV (`.gz`)<br>Station 42182 (Safdarjung) | 35,064 records<br>(2022–2025) | Primary ground-truth weather for temperature, humidity, precipitation, and fog identification. |
| **NASA POWER Reanalysis** | Hourly CSV<br>MERRA-2 Satellite Grid | 35,064 records<br>(2022–2025) | Secondary reanalysis source utilized for meteorological cross-validation and redundancy verification. |
| **OpenStreetMap (OSM) Delhi** | Vector Shapefile (`.shp.zip`)<br>Cartesian EPSG:32643 | 215,373 raw features<br>191,199 vehicular links | Primary road network infrastructure defining geometry, hierarchy, carriageway width, and segment lengths. |
| **ESA WorldCover 10m (2021)** | GeoTIFF Raster (10m)<br>Tile N27E075 | 10m resolution<br>NCT Delhi coverage | High-resolution land-cover classification distinguishing urban fabric, tree canopy, cropland, and water. |
| **SRTM Digital Elevation Model** | HGT Elevation Tiles<br>N28E076 & N28E077 | 30m resolution<br>195m–315m MSL | Topographic elevation gradients and terrain slope computations across the study area. |

### Crash Data Provenance & Scope
A critical distinction is maintained regarding crash data provenance. Official Delhi government road accident data (represented by five aggregate UUID CSV files) provides macro-level totals (~7,000–8,000 crashes per year, ~2,000 annual fatalities) but lacks spatial coordinates. To develop the spatial matching and feature-engineering pipeline, a candidate third-party accident dataset (2,433 records in Delhi) was ingested as a supplementary benchmark. It is treated strictly as a candidate proof-of-concept pipeline dataset, fully subject to replacement when official FIR crash-level logs are released.

---

## 3. Data Audit and Integrity Checks

The data audit phase programmatically inspected every raw asset to determine true internal structure rather than trusting file naming conventions:
- **Resolution of Meteostat Multi-Year Records:** The files `42182.csv.gz` through `42182 (3).csv.gz` were initially suspected to be redundant duplicate downloads. Detailed row-level inspection proved they are four distinct, consecutive single-year records covering 2022, 2023, 2024, and 2025. They were successfully concatenated into a contiguous 35,064-hour series.
- **Proactive Data Leakage Quarantine:** The candidate crash dataset contains an attribute named `risk_score`. Detailed correlation analysis revealed a strong association with crash outcomes and casualties ($r = 0.704$), indicating it was derived post-hoc. Retaining it would introduce fatal data leakage; it was permanently quarantined.
- **Cross-Validation of Weather Sources:** Station observations from Safdarjung (WMO 42182) were cross-validated against MERRA-2 gridded satellite reanalysis, confirming high agreement while exposing localized surface-level fog capture.

### Figure 1: Empirical Evidence of Target Leakage in the Candidate 'risk_score' Feature
![Figure 1: Empirical Evidence of Target Leakage](../eda/accident_leakage_risk_score.png)

* **Observation:** Boxplots and scatter plots show that `risk_score` values scale directly with casualty counts and step upward deterministically across minor, major, and fatal injury classes ($r = 0.704$).
* **Interpretation:** The `risk_score` variable shows a strong association with accident outcomes and casualties and was therefore quarantined as a potential target-leakage variable to prevent artificial accuracy inflation.

### Figure 2: Meteorological Cross-Validation: Meteostat Ground Observations vs. NASA POWER Reanalysis
![Figure 2: Meteorological Cross-Validation](../eda/weather_meteo_vs_nasa_comparison.png)

* **Observation:** Scatter plots and linear regressions show strong temperature correlation ($r = 0.841$) and moderate relative humidity correlation ($r = 0.588$) between Safdarjung ground sensors and MERRA-2 satellite grid cells.
* **Interpretation:** Ground-station observations accurately capture micro-scale surface cooling and localized winter fog that gridded satellite reanalysis smooths out, validating Meteostat as the primary weather record.

---

## 4. Data Preprocessing and Spatial Matching Methodology

Data preprocessing established geometric rigor and physical consistency across disparate spatial layers:
- **Cartesian Reprojection:** Unprojected geographic coordinates (WGS84 / EPSG:4326) produce distorted distance measurements when calculating segment lengths or buffer radiuses. All vector geometries (OSM road links and crash coordinates) were reprojected into **EPSG:32643 (UTM Zone 43N)**, ensuring all spatial computations are performed in true Cartesian meters.
- **Vehicular Filtering:** The 215,373 raw OSM features were filtered to remove pedestrian pathways, footways, and cycle tracks, retaining 191,199 vehicular road segments. From these, 16,630 major arterial segments were designated for primary spatial collision assignment.
- **Spatial Assignment Diagnostics:** Crash coordinates were mapped to road links using nearest-neighbor spatial joins in UTM coordinates. Distance sensitivity was evaluated across 25m, 50m, 100m, 200m, and 500m thresholds.

### Figure 3: Crash-to-Road Spatial Assignment Distance Distribution and Threshold Sensitivity
![Figure 3: Spatial Matching Distance Distribution](../eda/spatial_matching_distance_distribution.png)

* **Observation:** The distance histogram shows that matched crashes concentrate near road centerlines, with match rates of 5.14% at 25m (126 crashes), 8.82% at 50m (216 crashes), and 14.78% at 100m (362 crashes assigned to 308 road links).
* **Interpretation:** Preserving data integrity requires accepting unassigned records when GPS coordinates are imprecise, rather than forcing distant road assignments (>100m) that would corrupt model training.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Accident Characteristics and Severity Distribution

### Figure 4: Crash Severity Proportions and Casualty Breakdown in NCT Delhi
![Figure 4: Accident Severity Distribution](../eda/accident_severity_distribution.png)

* **Observation:** The distribution indicates that minor injury collisions constitute 55.9% ($n=1,360$), major injury collisions 28.4% ($n=691$), and fatal crashes 15.7% ($n=382$), with a mean casualty rate of 1.72 persons per incident.
* **Interpretation:** The severe skew toward property-damage and minor collisions demonstrates that simple crash counts are inadequate; risk modeling must incorporate severity weighting to prioritize fatal and major injury hotspots.

### 5.2 Temporal Crash Dynamics

### Figure 5: Multi-Scale Temporal Patterns of Delhi Collisions
![Figure 5: Accident Temporal Patterns](../eda/accident_temporal_patterns.png)

* **Observation:** Collision distributions show steady annual reporting, winter monthly elevations, and pronounced night-time concentration where night hours (21:00–05:00) account for 38.1% of crashes, while commute peak hours account for 24.2%.
* **Interpretation:** The high concentration of collisions during off-peak night hours suggests that reduced visibility and higher nighttime travel speeds act as substantial hazard multipliers in Delhi.

### 5.3 Spatial Crash Patterns and Road Hotspots

### Figure 6: Spatial Distribution of Matched Collisions and Hotspots across the Delhi Vehicular Network
![Figure 6: Delhi Crash Hotspots Map](../maps/delhi_crash_hotspots_matched_roads.png)

* **Observation:** The spatial map indicates collision clustering along major circumferential arteries (Ring Road, Outer Ring Road) and high-capacity radial corridors (NH-48, NH-44), while unmatched peripheral points remain distributed.
* **Interpretation:** Crash hazards are concentrated along high-speed multi-lane corridors rather than distributed evenly, verifying that individual road links provide a meaningful spatial unit for safety intervention.

### 5.4 Meteorological and Environmental Regime

### Figure 7: Four-Year Monthly Temperature Climatology and Precipitation Totals (2022–2025)
![Figure 7: Delhi Weather Climatology](../eda/delhi_weather_climatology.png)

* **Observation:** Monthly averages demonstrate Delhi's extreme continental climate, with summer temperatures peaking above 42°C in May–June and winter minimums dropping below 7°C in December–January, coupled with monsoonal rain peaks in July–August.
* **Interpretation:** Delhi's environmental conditions exhibit acute bi-modal extremes, creating distinct seasonal driving hazards (extreme heat and wet monsoon pavement vs. severe winter cold).

### Figure 8: Monthly Distribution of Hazardous Weather Event Hours
![Figure 8: Delhi Fog, Rain, and Extreme Heat](../eda/delhi_fog_rain_extremes.png)

* **Observation:** Event-hour counts show that dense fog conditions ($\text{RH} > 90\%$ and $\text{Temp} < 15^\circ\text{C}$) are strictly confined to December and January, while heavy precipitation hours concentrate in July–August and extreme heat hours dominate May–June.
* **Interpretation:** Adverse visibility hazards are strictly seasonal, confirming that weather exposure terms must be integrated to capture seasonal crash risk amplification.

### 5.5 Road Network Infrastructure and Geometry

### Figure 9: OpenStreetMap Vehicular and Arterial Road Network Structure of NCT Delhi
![Figure 9: Delhi Road Network Map](../maps/delhi_road_network_map.png)

* **Observation:** The GIS network delineates Delhi's concentric ring structures intersected by major national radial highways, with high network density in the urban core transitioning to sparser arterial links in outer sectors.
* **Interpretation:** The centralized radial geometry channels regional traffic through high-density urban corridors, indicating that spatial block partitioning is necessary during validation to avoid spatial leakage.

### Figure 10: OpenStreetMap Road Network Hierarchy
![Figure 10: OSM Road Hierarchy](../eda/osm_road_network_hierarchy.png)

* **Observation:** Bar distributions show that while residential streets dominate numerical segment volume (139,064 links), arterial corridors (motorways, trunks, primary, secondary) encompass over 6,000 kilometers of high-volume carriageway.
* **Interpretation:** Functional hierarchy directly governs vehicle speeds, traffic volume, and carriageway separation, making road classification a mandatory predictor in risk modeling.

### Figure 11: Statistical Distribution of Road Segment Lengths in Delhi
![Figure 11: OSM Segment Length Distribution](../eda/osm_segment_length_distribution.png)

* **Observation:** The segment length histogram shows severe right-skewness, with a median length of 96.6m for residential links and 269.1m for motorways, extending to long unobstructed spans exceeding 1,500m.
* **Interpretation:** Longer segments accumulate more physical exposure and higher raw crash counts, proving that normalizing targets by segment length (crash rate per kilometer) is mathematically essential.

### 5.6 Land Cover and Topographic Context

### Figure 12: ESA WorldCover 10m Surface Area Classification Distribution in NCT Delhi
![Figure 12: Land Cover Distribution](../eda/delhi_landcover_distribution.png)

* **Observation:** Land cover proportions indicate that agricultural cropland covers 43.15% of the total study area, built-up urban fabric covers 33.39%, and tree cover covers 12.96%, with grassland (4.57%) and water bodies (0.81%) comprising the remainder.
* **Interpretation:** Delhi's road network spans two starkly contrasting operating environments: dense urban street canyons with pedestrian conflicts, and dark peri-urban agricultural corridors with higher vehicle speeds.

### Figure 13: Topographic Elevation (MSL) and Slope Gradient Distributions from SRTM 30m DEM
![Figure 13: Elevation and Slope Profile](../eda/delhi_elevation_slope_profile.png)

* **Observation:** Histograms show elevation ranging from ~195m along the Yamuna River plain to ~315m on the southern Delhi Ridge, with predominantly low slopes across 85% of the territory and higher local slopes on ridge cuts and flyovers.
* **Interpretation:** While Delhi is largely flat, local slope variations identify flyover approach ramps and ridge transitions where sight distance and braking dynamics are constrained.

---

## 6. Key EDA Findings and Feature Relationships

The exploratory phase revealed several foundational insights that govern the transition to machine learning modeling:
1. **Night and Arterial Hazard Multiplication:** Crash occurrence in Delhi is concentrated during nighttime hours (38.1% of crashes between 21:00 and 05:00) and along major arterial ring corridors.
2. **Seasonal Climatological Windows:** Dense fog is confined exclusively to December and January, coinciding with winter collision peaks, while monsoon hydroplaning hazards peak in July and August.
3. **Infrastructure Exposure Skewness:** A significant percentage of crashes occur on higher-order arterial corridors, yet residential streets dominate link counts, requiring structural hierarchy features.
4. **Normalization Mandate:** Raw crash counts are heavily confounded by segment length, mandating rate-based target normalization (crashes per kilometer).
5. **Inter-Attribute Associations:** Correlation analysis revealed strong pairwise associations between distance to city center and built-up land cover ($r \approx -0.96$) and between elevation and slope ($r \approx 0.93$). These strong associations will be carefully considered during subsequent feature selection to prevent redundancy.

### Figure 14: Pairwise Pearson Correlation Heatmap for Engineered Road Segment Predictors
![Figure 14: Correlation Heatmap](../eda/correlation_matrix_heatmap.png)

* **Observation:** The heatmap displays correlation coefficients across continuous predictors, showing strong inverse association between `dist_to_center_km` and `builtup_pct` ($r \approx -0.96$) and strong positive association between `elevation_m` and `slope_deg` ($r \approx 0.93$), while physical road attributes (`is_oneway`, `is_bridge`, `has_maxspeed`, `est_lanes`) exhibit low mutual correlation ($|r| < 0.35$).
* **Interpretation:** Strong pairwise associations were observed between spatial-density and topographic features and will be considered during subsequent feature selection to maintain model interpretability.

---

## 7. Transition to Classification

### Methodological Rationale
Rather than treating road safety assessment as an unconstrained regression task or an ill-posed point-occurrence prediction, the project transitions the analytical outputs into **supervised classification**. Classification directly aligns with how municipal transport departments operate: allocating finite annual budgets to inspect, audit, and remediate high-risk road corridors.

### Formulated Classification Targets
Two complementary classification target formulations have been established at the road-segment level:
- **Target 1 — Binary High-Risk Screening (`is_high_risk`):** A binary indicator flagging segments that fall within the top 15% risk density corridor (or segments accumulating repeated fatal/major collisions). Segments flagged as Class 1 represent priority candidates for immediate engineering inspection.
- **Target 2 — Multi-Class Priority Tier (`risk_tier`):** An ordinal 4-class target categorizing road segments into Baseline Risk (Tier 0), Moderate Risk (Tier 1), High Risk (Tier 2), and Critical Intervention Priority (Tier 3). This multi-class structure enables tiered budget allocation for municipal road safety audits.

---

## 8. Planned Classification Models

To ensure transparency, prevent black-box decisions, and provide actionable engineering recommendations, the planned classification suite emphasizes explainability alongside predictive discrimination:
1. **Elastic-Net Regularized Logistic Regression:** Serves as the foundational linear benchmark. Regularized logistic regression provides direct odds-ratio interpretability, quantifying the baseline risk multiplier associated with road class, lane count, and environmental exposure.
2. **Balanced Random Forest Classifier:** An ensemble bagging algorithm that handles non-linear interactions and skewed segment length distributions without overfitting. Generates Gini impurity and permutation feature importance rankings.
3. **LightGBM and XGBoost Classifiers:** State-of-the-art gradient boosting frameworks optimized for tabular data. These models will be trained using cost-sensitive weighting and focal loss to address class imbalance between safe and hazardous segments.
4. **Post-Hoc SHAP Explainability Engine:** To support policy justification, all tree-based ensemble models will be integrated with TreeSHAP (SHapley Additive exPlanations). This will output local feature contribution plots for every flagged road segment, explaining exactly which factors (e.g., lack of median, high speed limit, winter fog exposure) drove the high-risk classification.

---

## 9. Classification Evaluation Metrics

Because high-risk road segments constitute a minority class, conventional accuracy is fundamentally misleading. The evaluation framework relies on specialized metrics:
- **Precision-Recall AUC (PR-AUC):** Evaluates classifier discrimination across all operational thresholds without being distorted by the large number of true negative (safe) road links.
- **Cost-Weighted $F_2$ Score (Recall-Biased):** In road safety prioritization, a False Negative (failing to identify a genuinely hazardous fatal road segment) carries far higher societal cost than a False Positive (inspecting a segment that turns out safe). $F_\beta$ (with $\beta = 2$) explicitly penalizes missed hazard corridors.
- **Spatial Block Cross-Validation:** Standard random $k$-fold cross-validation causes spatial data leakage due to spatial autocorrelation. The project enforces a 3×3 Spatial Block Partitioning across Delhi (60% Train, 20% Validation, 20% Holdout Test) to verify true geographic generalization.

---

## 10. Current Progress Summary

The project has completed all foundational data engineering, spatial integration, and exploratory analysis phases:
- **Phase 1 (Audit & Integrity):** All 12 raw files audited, cataloged, and verified. Meteostat files unified into 35,064 hourly records; NASA POWER cross-validated; candidate `risk_score` successfully quarantined.
- **Phases 2 & 3 (Accident & Weather EDA):** Crash distributions, severity profiles, temporal trends, and environmental regimes fully analyzed across Notebooks 02 and 03.
- **Phases 4 & 5 (Road & Environmental GIS):** OSM network filtered to 191,199 vehicular links and reprojected to EPSG:32643; ESA 10m land cover and SRTM DEM topography extracted across Notebooks 04 and 05.
- **Phase 6 (Spatial Matching):** Spatial nearest join executed in UTM coordinates with threshold sensitivity diagnostics; 362 crashes matched within defensible 100m tolerance across 308 road links.
- **Phases 7, 8, & 11 (Dataset Preparation):** Engineered 18 segment-level features; formulated binary and ordinal classification targets; established 3×3 spatial block cross-validation partitions; exported final dataset to `data_engineered/delhi_road_risk_ml_dataset.parquet`.

---

## 11. Limitations and Threat Mitigation

Rigorous scientific assessment requires acknowledging current data constraints:
1. **Crash Data Provenance:** The third-party crash dataset exhibits artificial uniformities across certain categorical features, indicating synthetic benchmark properties. It serves as a proof-of-concept pipeline dataset, designed to be swapped seamlessly with official FIR crash records upon availability.
2. **Spatial Matching Coverage:** Due to GPS coordinate dispersion in candidate crash records, 85.2% of crashes fall outside the strict 100m road centerline tolerance. The project deliberately preserved these as unassigned rather than forcing distant false matches.
3. **Traffic Volume Exposure Proxy:** Annual average daily traffic (AADT) volume counts are currently unavailable at the individual link level across OpenStreetMap. Road hierarchy classes (motorway, trunk, primary) and carriageway lane counts serve as proxy indicators for traffic volume exposure.

---

## 12. Next Steps

Following faculty review and pre-submission sign-off, the project will immediately commence model training:
1. **Step 1 — Baseline & Ensemble Training:** Train regularized Logistic Regression, Balanced Random Forest, and LightGBM classifiers on the spatial training partitions.
2. **Step 2 — Cost-Sensitive Optimization:** Optimize decision thresholds using cost-weighted $F_2$ metrics to maximize recall on fatal and high-severity corridors.
3. **Step 3 — Explainability & Intervention Mapping:** Generate SHAP summary and waterfall plots to provide actionable, explainable intervention recommendations for the top-priority hazardous corridors in Delhi.
4. **Step 4 — Spatial Generalization Validation:** Evaluate generalization performance strictly on the unseen holdout spatial blocks to confirm real-world municipal screening utility.
"""

md_path = os.path.join(REPORTS_DIR, "RoadSafe_Delhi_Pre_Submission_Report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"Successfully generated Markdown report: {md_path}")
