## 📘 Execution Guide — Reproducing the Entire Pipeline

This document provides a high-level, minimal-but-complete walkthrough of how to reproduce the full wildfire susceptibility modeling pipeline, starting after all raw datasets have been downloaded (see data.md).
Each module is described in a clean sequence with the correct order of execution, expected outputs, and common pitfalls.

### 1. Tiling (Base Grid — Required by All Modules)

#### Script:
src/tiling/generate_tiles.py
#### Purpose: 
Creates the 1-km California tile grid used as the spatial backbone for all other modules.

#### Output:

data/processed/tiled_ca/ca_tiles_1km.shp

#### Potential issues:
CRS mismatch

### 2. Wildfire Perimeter Processing

#### Scripts:

src/wildfires/fire_clean.py

src/wildfires/fire_process.py

#### Purpose: 
Cleans historic fire perimeters and intersects them with tiles to produce annual fire-per-tile summaries (burned area, fire index, fire occurrence).

#### Output:

data/processed/fire/fire_perimeter_ca_cleaned.gpkg

data/processed/fire/fire_events_tile_year.parquet

data/processed/fire/fire_events_tile_year.gpkg

#### Potential issues:

Missing raw GDB data → re-download via the USFS link in data.md.

Memory pressure occurs only when loading gigantic GPKG into GeoPandas; Parquet is safe.

### 3. Precipitation (PRISM)

#### Script: 
src/precipitation/fetch_process_precip.py

#### Purpose: 
Tiles PRISM precipitation values (monthly) into per-tile tables.

#### Output:
data/processed/precipitation_data/ppt_long.parquet
data/processed/precipitation_data/tile_id_grid.npy

#### Potential issues:
Ensure all .bil or .tif files follow the expected PRISM naming scheme.

Raster CRS must match tile CRS (script handles reprojection).

### 4. Temporal (PRISM)

#### Script: 
src/temporal/temporal_process.py

#### Purpose: 
Tiles PRISM temporal mean values (monthly) into per-tile tables.

#### Output:
data/processed/temporal_data/tmean_long.parquet

#### Notes:
The precipitation module must be finished first because temporal tables use its date ranges.

### 5. Topography

#### Script: 
src/topography/process_topography.py

#### Purpose: 
Tile topography tifs and calculate elevation and slope for each tile.

#### Output:
data/processed/topographic_data/tile_topography_summary.parquet


### 6. ERA5 Wind & Soil Moisture

#### Script: 
src/wind_soil/ 

fetchdata_api.py --> 

| ERA5_clip.py --> 

| ERA5_dataload.py -->

| ERA5_extract.py -->

| tiling_to_parquet.py

#### Purpose: 
Fetch data and tiles ERA5-Land wind speed and soil moisture rasters.

#### Output:

data/processed/era5/tiled/era5_src_tiled.parquet

| era5_swvl1_tiled.parquet

| era5_swvl2_tiled.parquet

| era5_u10_tiled.parquet

| era5_v10_tiled.parquet

| era5_windspeed_tiled.parquet


### 7. NDVI (Vegetation Greenness)

#### Script: 

src/ndvi/fetch_process_ndvi.py -->

src/ndvi/integrate_parquet_NDVI.py


#### Purpose: 
Downloads (or manually ingests) NDVI data, tiles it, and produces monthly summaries.

#### Output:

data/processed/ndvi/ndvi_tile_month.parquet

#### Notes:

Has a built-in fetcher; manual path is acceptable too.


### 8. Landcover (NLCD)

#### Script: 
src/landcover/landcover_process.py

#### Purpose: 
Aggregates NLCD classifications into compact tile-level descriptors (dominant type, urban %, vegetation entropy).

#### Output:

data/processed/landcover/landcover_tile_static.parquet

### 9. Road Density

#### Script: 
src/roaddensity/roaddensity_process.py

#### Purpose: 
Clips TIGER/Line road data to tiles and computes per-tile road length and density.

#### Output:

data/processed/roads/roads_tile_static.parquet

#### Potential issues:
Ensure all road shapefiles are in EPSG:3310; TIGER is usually in EPSG:4269 → auto-reprojected by script.

### 10. Fire-Infrastructure Facility Index

#### Script: 
src/facility/compute_facility.py

#### Purpose: 
Processes the CAL FIRE facility dataset and computes per-tile facility presence and density.

#### Output:

data/processed/facility_index/facility_index_final.parquet

### 11. Join Final Yearly Dataset for Modeling

#### Script: 
src/modeling/joindata_yearly.py

#### Purpose: 
Collects all processed datasets into unified annual training tables for modeling.

#### Output:

data/processed/model_ready/train_year_*.parquet

#### Potential issues:

Missing dataset from earlier modules → join script will warn and fail.

### 12. Train Stacked Ensemble Model

#### Script: 

multicollinearity.py -> optuna_tune.py -> stacked_regression_train.py -> 
final_prediction.py -> performance_metrics.py

#### Output and folder: 

Optuna results saved in src/modeling/optuna_results/stack_reg_xxtime

Models parameters saved in models/stacked_model.joblib

Training metrics saved in output/analysis/performance_outputs

Final prediction for (2013 - 2023) saved as src/modeling/data/predictions/raw_predictions_2013_2023.parquet

Relevant metrics or decade-long prediction saved in src/modeling/data/predictions



### 13. Resource–Risk Discrepancy Analysis

#### Script: 
src/discrepancy/alignment.py -->

src/discrepancy/alignment_visual.py


#### Purpose: 
Compares infrastructure placement with susceptibility distribution using spatial correlation, regression, and mutual information.

Output saved in: output/discrepancy/

#### Output:
Several analytical plots and discrepancy analysis