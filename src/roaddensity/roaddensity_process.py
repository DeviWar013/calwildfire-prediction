"""
Road Density Processing Pipeline
--------------------------------

This script executes the full workflow for computing road density across the
California 1 km tile grid. It relies on `roaddensity_utils.py` for all helper
functions and geometry operations.

Pipeline Steps:
1. Load the raw TIGER/Line primary+secondary road centerline shapefile.
2. Optionally visualize road geometries for inspection.
3. Load the 1 km California tile grid.
4. Intersect road centerlines with tiles (tile_roads).
5. Compute road density per tile using calc_road_density:
       RoadDens = (total road length in km) / (tile area in km²)
6. Save the final tile-level road density results as a shapefile.

Outputs:
    data/processed/roaddensity_data/tiled_roaddensity.shp
"""

from roaddensity_utils import *
import matplotlib.pyplot as plt
import os
import geopandas as gpd

#read files and verify
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lRoadPath = os.path.join(BASE_DIR, 'data/raw/roaddensity_data/tl_2024_06_prisecroads/tl_2024_06_prisecroads.shp')
lTilesPath = os.path.join(BASE_DIR,'data/processed/tiled_ca/ca_tiles_1km.shp')
check_roads(lRoadPath)

#plot road map
lRoadData = gpd.read_file(lRoadPath)
##plot_roads(lRoadData)

#tiling
lTilesData = gpd.read_file(lTilesPath)
lTiledRoad = tile_roads(lRoadData, lTilesData)
print("Tiled road data:")
print(lTiledRoad.head())
print("Total geometries:", len(lTiledRoad))
print(lTiledRoad.geom_type.value_counts())
print("CRS:", lTiledRoad.crs)

print("Tiled Roads Columns:", lTiledRoad.columns)
print("Tiles Columns:", lTilesData.columns)

#calculate road density per tile
#unit: length of roads (km) per tile (sq km)
road_density = calc_road_density(lTiledRoad, lTilesData)
print(road_density[["tile_id", "RoadDens"]].describe())

#save road density results
out_path = os.path.join(BASE_DIR, "data/processed/roaddensity_data/tiled_roaddensity.shp")
road_density.to_file(out_path)