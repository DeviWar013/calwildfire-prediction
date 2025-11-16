# ----------------------------------------------------------
# fire_clean.py
# ----------------------------------------------------------
"""
Wildfire Perimeter Cleaning Pipeline (LEGACY — logic preserved)

This script loads the RAW Interagency Fire Perimeter dataset and performs:
1. Reprojection & geometry validation
2. Clipping to California
3. DATE_CUR parsing into DATE_CUR_parsed
4. FIRE_YEAR filtering (>= 1981)
5. Optional visualization
6. Saving cleaned perimeters to GPKG

Outputs
-------
fire_perimeter_ca_cleaned.gpkg
    - Clipped to CA
    - Contains parsed DATE_CUR_parsed
    - Limited to years >= 1981
    - Geometry repaired where needed (make_valid)

Notes
-----
• This is the canonical cleaned perimeter file used by fire_process.py.
• Logic is intentionally preserved exactly as in the original pipeline.
• Only comments, docstring, and formatting have been improved.
"""
from fire_utils import *

lBaseDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------
# File paths
# ----------------------------------------------------------
fire_fp = os.path.join(lBaseDir,"pdata/raw/InterAgencyFirePerimeterHistory_All_Years_View/InterAgencyFirePerimeterHistory_All_Years_View.shp")
ca_fp   = os.path.join(lBaseDir,"pdata/raw/ca_state/CA_State.shp")
out_fp  = os.path.join(lBaseDir,"pdata/processed/fire/fire_perimeter_ca_cleaned.gpkg")
os.makedirs(os.path.dirname(out_fp), exist_ok=True)

# ----------------------------------------------------------
# 1. Clip to California
# ----------------------------------------------------------
fire_ca = clip_fire_to_california(fire_fp, ca_fp)

# ----------------------------------------------------------
# 2. Normalize DATE_CUR
# ----------------------------------------------------------
fire_ca = normalize_fire_dates(fire_ca)

# Verify the DATE_CUR parsing results
verify_date_column(fire_ca)


# ----------------------------------------------------------
# 3. Keep essential columns and clean
# ----------------------------------------------------------
cols_keep = ["FIRE_YEAR", "DATE_CUR_parsed", "AGENCY", "SOURCE", "GIS_ACRES", "geometry"]
fire_ca = fire_ca[cols_keep].copy()
fire_ca = fire_ca[fire_ca.is_valid & fire_ca["FIRE_YEAR"].notna()]

#keep pdata after 1980
fire_ca = filter_fire_years(fire_ca, 1981)

# Quick visualization of a subset
plot_fire_perimeters(fire_ca)

# ----------------------------------------------------------
# 4. Save cleaned output
# ----------------------------------------------------------
fire_ca.to_file(out_fp, driver="GPKG")
print(f"Saved cleaned file to: {out_fp}")

print("Earliest:", fire_ca["DATE_CUR_parsed"].min())
print("Latest:", fire_ca["DATE_CUR_parsed"].max())
print("Unique FIRE_YEARs:", fire_ca["FIRE_YEAR"].nunique())
