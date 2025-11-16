# ----------------------------------------------------------
# fire_process.py
# ----------------------------------------------------------
"""
Wildfire Tiling Pipeline (LEGACY — logic preserved)

This script takes the ALREADY-CLEANED wildfire perimeter dataset and
aggregates burned area into tile–year–month fire metrics for the entire
analysis period.

This is the **final tiling step** in the wildfire module, producing:

Outputs
-------
1) fire_events_tile_year.gpkg
   - Contains ONLY POSITIVE FIRE EVENTS (rows where fire_occurred == 1)
   - Includes spatial geometry for visualization or GIS inspection
   - Negative records DO NOT appear here (legacy behavior)

2) fire_events_tile_year.parquet
   - Contains ALL tile–year–month rows (positive AND negative)
   - Includes fire_index (0–1) and fire_occurred (0/1)
   - Used by the modeling pipeline (training_yearly_data)

Important Notes
---------------
• No geometry is saved for negative fire events.
• Geometry-heavy GPKG format CANNOT store 200M+ polygons; storing only
  positive events ensures success and keeps RAM usage safe.
• The parquet file is the authoritative dataset for modeling.
• DO NOT modify tiling logic or year range unless updating training years.

This script closely preserves the behavior of the original legacy pipeline.
Only documentation and file organization have been modernized.
"""
from fire_utils import *

# ----------------------------------------------------------
# 1. File paths
# ----------------------------------------------------------
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fire_fp = os.path.join(base, "data/processed/fire/fire_perimeter_ca_cleaned.gpkg")
tile_fp = os.path.join(base, "data/processed/tiled_ca/ca_tiles_1km.shp")
out_fp  = os.path.join(base, "data/processed/fire/fire_events_tile_year.gpkg")
out_parquet = os.path.join(base, "data/processed/fire/fire_events_tile_year.parquet")

# ----------------------------------------------------------
# 2. Run tiling
# ----------------------------------------------------------
if os.path.exists(out_fp):
    print(f"Output already exists: {out_fp}")
    print("Skipping tiling step.")
    fire_events = gpd.read_file(out_fp)
else:
    summary = tile_fire_perimeters(fire_fp, tile_fp)
    summary = summary.drop(columns="burned_area_ha")

    # ----------------------------------------------------------
    # 3. Save results
    # ----------------------------------------------------------
    tiles = gpd.read_file(tile_fp).to_crs(3310)
    fire_events = save_fire_events(summary, tiles, out_fp,include_negatives=True)

fire_events.drop(columns=["geometry"]).to_parquet(out_parquet, index=False)
fire_events.to_file(out_fp,driver="GPKG")