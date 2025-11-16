"""
==============================================================================
LAND COVER (NLCD) PROCESSING PIPELINE — EXECUTION SCRIPT
==============================================================================

This script runs the full NLCD preprocessing workflow:

1. Clip the raw NLCD land cover raster (GeoTIFF) to California.
2. Compute tile-level land cover statistics:
       - Dominant NLCD cover class
       - Dominant class percentage
       - Urban percentage (based on NLCD urban codes)
3. Save the resulting tile shapefile with added feature columns.
4. Run optional verification plots and summaries.

PRIMARY OUTPUTS:
    <...>/Annual_NLCD_LndCov_2024_California.tif        (clipped raster)
    <...>/ca_tiles_landcover.shp                        (tile-level features.md)

NOTES:
- This file is the orchestrator only.
- All geospatial operations are implemented in `landcover_utils.py`.
- Paths are defined internally; update them before running if directory
  structure changes.

==============================================================================
"""
from landcover_utils import *

#defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lRawCoverPath = os.path.join(BASE_DIR, 'data/raw/landcover_data/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif')
lBoundaryPath = os.path.join(BASE_DIR,'data/raw/ca_state/CA_State.shp')
lOutputPath = os.path.join(BASE_DIR, 'data/raw/landcover_data/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_California.tif')
lTilesPath = os.path.join(BASE_DIR,'data/processed/tiled_ca/ca_tiles_1km.shp')
lCalculatedPath = os.path.join(BASE_DIR,'data/processed/landcover_data/ca_tiles_landcover.shp')

#clip raster and save
clip_raster_to_shape(lRawCoverPath, lBoundaryPath,lOutputPath)

#tiling landcover
landcover_features(lTilesPath,lOutputPath,lCalculatedPath)

#verify clipped landcover
verify_LC_raster(lOutputPath)

#verify tiled shapefile
verify_landcover_tiles(lCalculatedPath,5, False)